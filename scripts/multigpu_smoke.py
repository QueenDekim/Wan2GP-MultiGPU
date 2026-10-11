#!/usr/bin/env python3
"""Run lightweight, checkpoint-free MultiGPU dispatch and MMGP adapter tests.

Requires the Wan2GP Python environment and at least two NVIDIA CUDA GPUs:
    venv\Scripts\python scripts\multigpu_smoke.py --require-gpu

This does not validate the numerical behavior of individual Wan2GP models.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.multigpu_offload import AccelerateMultiGPU


class Block(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.linear = nn.Linear(features, features)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(value)) + value


class DummyTransformer(nn.Module):
    def __init__(self, features: int = 192, count: int = 10):
        super().__init__()
        self.preproj = nn.Linear(features, features)
        self.transformer_blocks = nn.ModuleList(
            [Block(features) for _ in range(count)]
        )
        self.head = nn.Linear(features, features)
        self.register_buffer("root_scale", torch.ones(features))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.preproj(value) * self.root_scale
        for block in self.transformer_blocks:
            value = block(value)
        return self.head(value)


class Qwen3VLTextModel(nn.Module):
    """Tiny root-layer Qwen3-VL stand-in, without external checkpoints."""

    def __init__(self, features: int = 192, count: int = 10):
        super().__init__()
        self.embed_tokens = nn.Embedding(256, features)
        self.layers = nn.ModuleList([Block(features) for _ in range(count)])
        self.norm = nn.LayerNorm(features)
        self.rotary_emb = nn.Identity()

    def forward(self, ids: torch.LongTensor) -> torch.Tensor:
        hidden = self.embed_tokens(ids)
        for block in self.layers:
            hidden = block(hidden)
        return self.norm(hidden)


def check_root_qwen_decoder(manager: AccelerateMultiGPU):
    root = Qwen3VLTextModel().eval()
    cpu_reference = copy.deepcopy(root).eval()
    tokens = torch.tensor([[1, 17, 35, 200]], dtype=torch.long)
    with torch.inference_mode():
        expected = cpu_reference(tokens)

    device_map = manager._qwen3vl_text_device_map(root)
    assert device_map and "layers.0" in device_map
    assert "layers.9" in device_map and "embed_tokens" in device_map
    assert "norm" in device_map and "rotary_emb" in device_map
    assert set(map(str, device_map.values())) == {str(d) for d in manager.devices}
    manager._validate_device_map_coverage("qwen3vl-smoke", root, device_map)

    from accelerate import dispatch_model

    with torch.inference_mode():
        root = dispatch_model(
            root, device_map=device_map, main_device=manager.devices[0],
            force_hooks=True, offload_buffers=False,
        )
        actual = root(tokens.to(manager.devices[0])).detach().cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    print("[MultiGPU smoke] Qwen3-VL root decoder GPU-only forward: OK")

class GemmaLike(nn.Module):
    """Toy decoder with a tied embed_tokens/lm_head pair on GPU0."""

    def __init__(self, features=192, count=10):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(256, features)
        self.model.layers = nn.ModuleList([Block(features) for _ in range(count)])
        self.model.norm = nn.LayerNorm(features)
        self.lm_head = nn.Linear(features, 256, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, ids):
        hidden = self.model.embed_tokens(ids)
        for block in self.model.layers:
            hidden = block(hidden)
        return self.lm_head(self.model.norm(hidden))



def check_ltx2_tiled_vae_device_switch():
    """Tiled VAE latent stitching tolerates CUDA output-device changes.

    Video extension / first-last-frame conditioning can call the encoder
    repeatedly. Simulate an MMGP stage transition between tiles and verify
    that the assembled latent and masks stay on the input GPU.
    """
    from models.ltx2.ltx_core.model.video_vae.video_vae import encode_video
    from models.ltx2.ltx_core.model.video_vae.tiling import (
        SpatialTilingConfig,
        TilingConfig,
    )

    class SwitchingVAE(nn.Module):
        latent_channels = 4

        def __init__(self, alternate=False):
            super().__init__()
            self.alternate = alternate
            self.calls = 0

        def forward(self, tile):
            device = "cuda:1" if self.alternate and self.calls % 2 else "cuda:0"
            self.calls += 1
            return torch.ones(
                (
                    tile.shape[0],
                    self.latent_channels,
                    (tile.shape[2] - 1) // 8 + 1,
                    tile.shape[3] // 32,
                    tile.shape[4] // 32,
                ),
                device=device,
                dtype=tile.dtype,
            )

    video = torch.zeros((1, 3, 9, 128, 128), device="cuda:0")
    tiling = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=64,
            tile_overlap_in_pixels=0,
        )
    )
    with torch.inference_mode():
        stable_vae = SwitchingVAE()
        expected = encode_video(video, stable_vae, tiling)
        switching_vae = SwitchingVAE(alternate=True)
        actual = encode_video(video, switching_vae, tiling)

    assert switching_vae.calls > 1, "Test must exercise multiple VAE tiles"
    assert actual.device == video.device
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    # One-frame and disabled-tiling conditioning may take the direct encoder
    # route. The input is on cuda:0 even when the encoder returns on cuda:1.
    direct_vae = SwitchingVAE(alternate=True)
    direct_vae.calls = 1
    with torch.inference_mode():
        direct = encode_video(video, direct_vae, None)
    assert direct.device == video.device
    assert direct.shape == expected.shape
    torch.testing.assert_close(direct, expected, rtol=1e-5, atol=1e-5)
    print("[MultiGPU smoke] LTX2 tiled and direct VAE cross-GPU conditioning: OK")


def check_gguf_direct_cuda_transfer(manager):
    """Exercise LTX2 GGUFSourceTensor root-parameter preplacement on CUDA."""
    from shared.qtypes.gguf import GGUFSourceTensor

    # GGUFSourceTensor carries packed storage and a logical tensor shape.
    # No real checkpoint or dequantization is required for a device test.
    packed_bytes = torch.arange(128, dtype=torch.uint8)
    source = GGUFSourceTensor.wrap(
        packed_bytes, tensor_type="SMOKE_Q4", tensor_shape=(16, 32),
    )
    parameter = nn.Parameter(source, requires_grad=False)
    # nn.Parameter construction drops arbitrary tensor subclass attributes;
    # real GGUF loading stores them on the registered parameter.
    parameter.tensor_type = source.tensor_type
    parameter.tensor_shape = source.tensor_shape
    owner = nn.Module()
    owner.register_parameter("packed_weight", parameter)
    bias = GGUFSourceTensor.wrap(
        torch.arange(32, dtype=torch.uint8),
        tensor_type="SMOKE_BIAS", tensor_shape=(32,),
    )
    bias._gguf_bias_orig_dtype = torch.float16
    bias._gguf_bias_orig_tensor_type = "SMOKE_BIAS_SOURCE"
    owner.register_buffer("packed_bias", bias)

    manager._patch_accelerate_quanto()
    count = manager._preplace_direct_device_map_tensors(
        owner,
        {"packed_weight": manager.devices[1], "packed_bias": manager.devices[0]},
    )
    assert count == 2
    moved = owner.packed_weight
    assert isinstance(moved, GGUFSourceTensor)
    assert moved.device == manager.devices[1]
    assert moved.dtype == torch.uint8
    assert moved.tensor_type == "SMOKE_Q4"
    assert tuple(moved.tensor_shape) == (16, 32)
    torch.testing.assert_close(moved.cpu().as_subclass(torch.Tensor), packed_bytes)
    assert "packed_bias" in owner._buffers
    assert "packed_bias" not in owner._parameters
    moved_bias = owner.packed_bias
    assert isinstance(moved_bias, GGUFSourceTensor)
    assert moved_bias.device == manager.devices[0]
    assert moved_bias.tensor_type == "SMOKE_BIAS"
    assert moved_bias._gguf_bias_orig_dtype == torch.float16
    assert moved_bias._gguf_bias_orig_tensor_type == "SMOKE_BIAS_SOURCE"
    print("[MultiGPU smoke] LTX2 GGUFSourceTensor parameter/buffer CUDA preplacement: OK")

def check_nested_mmgp_restore(manager):
    """Ensure unload restores all MMGP child block tensors, not just root."""
    backing = GemmaLike(features=32, count=4).eval()
    source_root = backing.model.embed_tokens.weight
    source_first = backing.model.layers[0].linear.weight
    source_last = backing.model.layers[-1].linear.bias
    # Simulate Accelerate replacing CPU-backed originals with CUDA/sharded
    # copies. Distinct CPU tensors suffice to test reference restoration.
    original_tie = backing.lm_head.weight
    backing.model.embed_tokens.weight = nn.Parameter(torch.randn_like(source_root))
    backing.lm_head.weight = backing.model.embed_tokens.weight
    backing.model.layers[0].linear.weight = nn.Parameter(torch.randn_like(source_first))
    backing.model.layers[-1].linear.bias = nn.Parameter(torch.randn_like(source_last))
    assert backing.model.layers[0].linear.weight is not source_first

    registry = {
        "nested_restore": [],
        "nested_restore/model.layers.0": [
            (backing.model.layers[0].linear, "weight", source_first, False, None),
        ],
        "nested_restore/model.layers.3": [
            (backing.model.layers[-1].linear, "bias", source_last, False, None),
        ],
    }
    root_blocks = [
        (backing.model.embed_tokens, "weight", source_root, False, None),
        (backing.lm_head, "weight", original_tie, False, (backing.model.embed_tokens, "weight")),
    ]
    manager.offload.models["nested_restore"] = backing
    manager.offload.blocks_of_modules = registry
    manager._suspended_mmgp_blocks["nested_restore"] = root_blocks
    manager._move_model_to_cpu(backing)
    assert backing.model.embed_tokens.weight is source_root
    assert backing.lm_head.weight is source_root
    assert backing.model.layers[0].linear.weight is source_first
    assert backing.model.layers[-1].linear.bias is source_last
    assert registry["nested_restore"] is root_blocks
    print("[MultiGPU smoke] Nested MMGP block / tied weight restoration: OK")
    del manager.offload.models["nested_restore"]
    del manager.offload.blocks_of_modules

def check_minimax_h3_text_conditioning():
    """Verify Qwen3-VL GPU1 embeddings enter H3 GPU0 projection safely."""
    from models.minimax_h3.transformer import MiniMaxH3Model

    class Refiner(nn.Module):
        def forward(self, chunks):
            return torch.tanh(chunks[0])

    class Harness(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_size = 12
            self.condition_proj = nn.Linear(8, 12).to("cuda:0")
            self.token_refiner = Refiner().to("cuda:0")
            self._interrupt = False

    harness = Harness()
    # Model preprocessing occurs before its first MMGP GPU load/dispatch.
    # Mimic the actual lifecycle rather than injecting the marker directly:
    # this caught the dialogue failure missed by the old smoke test.
    fake_offload = SimpleNamespace(
        models={"transformer": harness},
        gpu_load_blocks=lambda *args, **kwargs: None,
        gpu_unload_blocks=lambda *args, **kwargs: None,
        unload_all=lambda *args, **kwargs: None,
        release=lambda *args, **kwargs: None,
    )
    stage_manager = AccelerateMultiGPU(
        fake_offload, ["cuda:0", "cuda:1"], verbose=0,
    )
    assert not hasattr(harness, "_multigpu_primary_device")
    stage_manager.install()
    assert harness._multigpu_primary_device == torch.device("cuda:0")
    assert "transformer" not in stage_manager.dispatched
    text_states = torch.randn(1, 5, 8, device="cuda:1")
    with torch.inference_mode():
        expected = harness.token_refiner(
            [harness.condition_proj(text_states.to("cuda:0")[0])]
        ).unsqueeze(0)
        actual = MiniMaxH3Model.preprocess_text_embeds(harness, text_states)
        assert actual.device == torch.device("cuda:0")
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

        # The projection may still be CPU/mmap-backed after the preceding
        # MMGP stage was unloaded. It loads lazily on the first forward, so
        # weight.device cannot be used to decide where input embeddings go.
        cpu_proj = nn.Linear(8, 12)
        cpu_proj.load_state_dict(harness.condition_proj.cpu().state_dict())
        harness.condition_proj = cpu_proj
        def lazy_gpu_load(module, _args):
            module.to("cuda:0")
        hook = harness.condition_proj.register_forward_pre_hook(lazy_gpu_load)
        try:
            lazy_result = MiniMaxH3Model.preprocess_text_embeds(harness, text_states)
        finally:
            hook.remove()
        assert lazy_result.device == torch.device("cuda:0")
        torch.testing.assert_close(lazy_result, expected, rtol=1e-5, atol=1e-5)

        # Cached/preprocessed 12-wide embeddings must also be on cuda:0.
        cached = torch.randn(1, 5, 12, device="cuda:1")
        passthrough = MiniMaxH3Model.preprocess_text_embeds(harness, cached)
        assert passthrough.device == torch.device("cuda:0")
        torch.testing.assert_close(passthrough, cached.to("cuda:0"), rtol=0, atol=0)

    stage_manager.release()
    assert not hasattr(harness, "_multigpu_primary_device")
    print("[MultiGPU smoke] MiniMax H3 GPU1 Qwen embeddings -> GPU0 refiner before dispatch: OK")


def check_gemma_vram_budget(manager):
    """Catch the 0 GiB secondary-GPU / tied embedding OOM regression."""
    from unittest.mock import patch
    from accelerate import dispatch_model

    root = GemmaLike().eval()
    reference = copy.deepcopy(root).eval()
    inputs = torch.tensor([[1, 17, 35, 200]], dtype=torch.long)
    with torch.inference_mode():
        expected = reference(inputs)

    # Emulate the reported case: GPUs are full before text encoder dispatch.
    with patch.object(
        torch.cuda, "mem_get_info",
        return_value=(256 * 1024**2, 16 * 1024**3),
    ):
        try:
            manager._gemma_device_map(root)
        except RuntimeError as exc:
            assert "GPU-only Gemma placement impossible" in str(exc)
        else:
            raise AssertionError("Near-zero-VRAM Gemma map was not rejected")

    device_map = manager._gemma_device_map(root)
    assert device_map and device_map["lm_head"] == device_map["model.embed_tokens"]
    assert set(map(str, device_map.values())) == {str(d) for d in manager.devices}
    manager._validate_device_map_coverage("gemma-smoke", root, device_map)
    manager._preflight_gpu_device_map("gemma-smoke", root, device_map)
    with torch.inference_mode():
        dispatched = dispatch_model(
            root, device_map=device_map, main_device=manager.devices[0],
            force_hooks=True, offload_buffers=False,
        )
        result = dispatched(inputs.to(manager.devices[0])).detach().cpu()
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    print("[MultiGPU smoke] Gemma tied embeddings + VRAM preflight: OK")

def check_adapter_residency(manager: AccelerateMultiGPU, model: DummyTransformer):
    model._loras_active_adapters = ["lora", "dora", "lokr"]
    model._loras_model_shortcuts = {}
    model._loras_model_data = {}
    for block in model.transformer_blocks:
        module = block.linear
        a = torch.randn(8, module.in_features)
        b = torch.randn(module.out_features, 8)
        model._loras_model_data[module] = {
            "lora": [a, b, None, None, 1.0, {"type": "lora"}],
            "dora": [a, b, None, torch.ones(module.out_features), 1.0, {"type": "dora"}],
            "lokr": [a, b, None, None, 1.0, {"type": "lokr"}],
        }

    manager._load_loras_for_dispatch("smoke", model)
    count = 0
    for module, data in model._loras_model_data.items():
        device = module.weight.device
        for adapter in model._loras_active_adapters:
            parts = data.get(adapter + "_GPU")
            assert parts is not None, f"Missing GPU adapter {adapter}"
            assert all(
                not torch.is_tensor(part) or part.device == device
                for part in parts[:4]
            ), f"Adapter {adapter} not resident alongside {device}"
            count += 1

    manager._unload_dispatched_loras(model)
    for data in model._loras_model_data.values():
        assert not any(key.endswith("_GPU") for key in data)
    print(f"[MultiGPU smoke] {count} LoRA/DoRA/LoKr bindings: OK")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        if args.require_gpu:
            print("[MultiGPU smoke] FAIL: two CUDA GPUs required")
            return 2
        print("[MultiGPU smoke] SKIP: two CUDA GPUs not available")
        return 0

    torch.manual_seed(31337)
    devices = ["cuda:0", "cuda:1"]
    model = DummyTransformer().eval()
    reference = copy.deepcopy(model).eval()
    cpu_input = torch.randn(2, 192)

    with torch.inference_mode():
        expected = reference(cpu_input)

    manager = AccelerateMultiGPU(
        SimpleNamespace(models={"smoke": model}),
        devices,
        fraction=0.90,
        verbose=2,
    )
    device_map = manager._weighted_top_level_pipeline_map(model)
    assert device_map, "No generic transformer block map generated"
    assert all(str(device).startswith("cuda:") for device in device_map.values())
    assert set(map(str, device_map.values())) == set(devices), "GPU not used"
    manager._validate_device_map_coverage("smoke", model, device_map)

    from accelerate import dispatch_model

    with torch.inference_mode():
        model = dispatch_model(
            model,
            device_map=device_map,
            main_device=torch.device(devices[0]),
            force_hooks=True,
            offload_buffers=False,
        )
        result = model(cpu_input.to(devices[0]))
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        actual = result.detach().cpu()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        print(
            "[MultiGPU smoke] Cross-GPU forward matches CPU reference; "
            f"result device: {result.device}"
        )
        check_adapter_residency(manager, model)
        check_root_qwen_decoder(manager)
        check_gemma_vram_budget(manager)
        check_nested_mmgp_restore(manager)
        check_gguf_direct_cuda_transfer(manager)
        check_ltx2_tiled_vae_device_switch()
        check_minimax_h3_text_conditioning()

    print("[MultiGPU smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
