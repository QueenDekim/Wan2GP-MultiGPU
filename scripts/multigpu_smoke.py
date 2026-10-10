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

    print("[MultiGPU smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
