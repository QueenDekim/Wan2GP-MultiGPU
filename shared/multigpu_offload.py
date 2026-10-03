"""Multi-GPU model sharding for WanGP/MMGP using Hugging Face Accelerate.

MMGP remains responsible for discovering/loading models and model switching.
Accelerate owns the residency of an active model:

    GPU#0 -> GPU#1 -> ... -> CPU

Unlike the old cache implementation, this does not copy an evicted block to a
second GPU and then try to restore it later. The complete active model is
dispatched once using Accelerate's device map, so a large model can actually
execute with parameters resident on multiple GPUs.
"""
from __future__ import annotations

import gc
import types

import torch


class AccelerateMultiGPU:
    def __init__(self, offload, devices: list[str], fraction: float = 0.82, verbose: int = 1):
        self.offload = offload
        self.devices = [torch.device(d) for d in devices]
        self.fraction = max(0.50, min(float(fraction), 0.98))
        self.verbose = int(verbose)
        self.dispatched: dict[str, torch.nn.Module] = {}
        self.installed = False

    @property
    def enabled(self):
        return len(self.devices) >= 2 and all(d.type == "cuda" for d in self.devices)

    def _log(self, message):
        if self.verbose >= 1:
            print(f"[MultiGPU] {message}", flush=True)

    def _restore_mmgp_forwards(self, model):
        # MMGP wraps forwards with _mm_forward before profile() returns.
        # Restore the wrapped function before Accelerate installs its own
        # device hooks. This preserves MMGP LoRA wrappers, but removes the
        # residency/offload checks that would otherwise fight Accelerate.
        for module in model.modules():
            previous = getattr(module, "_mm_forward", None)
            if previous is not None:
                module.forward = previous

    def _remove_accelerate_hooks(self, model):
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(model)
        except ImportError:
            pass

    def _move_model_to_cpu(self, model):
        self._remove_accelerate_hooks(model)
        self._restore_mmgp_forwards(model)

        try:
            model.to("cpu")
        except Exception:
            # Some quantized modules do not implement .to(). Restore the
            # original MMGP block tensors instead.
            for model_id, candidate in self.offload.models.items():
                if candidate is model:
                    for parent, name, source, is_buffer, tied in self.offload.blocks_of_modules.get(model_id, []):
                        if tied is not None:
                            setattr(parent, name, getattr(tied[0], tied[1]))
                        else:
                            setattr(parent, name, source)
                    break

    def _max_memory(self):
        max_memory = {}
        for device in self.devices:
            index = device.index
            free, _ = torch.cuda.mem_get_info(index)
            budget = int(free * self.fraction)
            max_memory[index] = max(1, budget)

        # Do not cap CPU RAM here. Accelerate will use the currently available
        # host memory and will only place weights there after GPU capacity is
        # exhausted.
        return max_memory

    def _no_split_classes(self, model):
        classes = list(getattr(model, "_no_split_modules", None) or [])

        # Custom diffusion models often do not expose Transformers'
        # _no_split_modules. Keep common transformer/residual blocks intact.
        common = {
            "BasicAVTransformerBlock",
            "BasicTransformerBlock",
            "TransformerBlock",
            "LTXVTransformerBlock",
            "LTXTransformerBlock",
            "Gemma3DecoderLayer",
            "WanTransformerBlock",
            "Wan2TransformerBlock",
        }
        known = {module.__class__.__name__ for module in model.modules()}
        for name in common:
            if name in known and name not in classes:
                classes.append(name)
        return classes

    def _dispatch(self, model_id):
        model = self.offload.models[model_id]

        if model_id in self.dispatched:
            return self.dispatched[model_id]

        self._restore_mmgp_forwards(model)
        self._remove_accelerate_hooks(model)

        max_memory = self._max_memory()

        from accelerate import dispatch_model
        from accelerate.utils import get_balanced_memory, infer_auto_device_map

        no_split = self._no_split_classes(model)

        balanced_memory = get_balanced_memory(
            model,
            max_memory=max_memory,
            no_split_module_classes=no_split,
            low_zero=False,
        )

        device_map = infer_auto_device_map(
            model,
            max_memory=balanced_memory,
            no_split_module_classes=no_split,
            clean_result=True,
            offload_buffers=True,
        )

        # Make the map visible in the log. This is much more useful than
        # pretending that a block was merely "cached" on GPU#1.
        counts = {}
        for device in device_map.values():
            key = str(device)
            counts[key] = counts.get(key, 0) + 1

        self._log(
            f"{model_id}: Accelerate device map "
            + ", ".join(f"{device}={count}" for device, count in counts.items())
        )

        dispatched = dispatch_model(
            model,
            device_map=device_map,
            main_device=self.devices[0],
            offload_buffers=True,
            force_hooks=True,
        )

        self.dispatched[model_id] = dispatched
        return dispatched

    def load(self, model_id, blocks_name, preload=False):
        # A dispatched model owns all of its internal blocks. MMGP must not
        # subsequently pull individual blocks back to GPU#0.
        if blocks_name is not None:
            return

        model = self._dispatch(model_id)
        self.offload.loaded_blocks[model_id] = None

        gpu_parts = []
        for name, module in model.named_modules():
            if not any(True for _ in module.parameters(recurse=False)):
                continue
            param = next(module.parameters(recurse=False), None)
            if param is not None:
                gpu_parts.append((name or "<root>", str(param.device)))

        devices_used = sorted(set(device for _, device in gpu_parts))
        self._log(f"{model_id}: active on " + ", ".join(devices_used))

    def unload(self, model_id, blocks_name):
        if blocks_name is not None:
            return

        model = self.dispatched.pop(model_id, None)
        if model is None:
            return

        self._log(f"{model_id}: releasing Accelerate dispatch -> RAM")
        self._move_model_to_cpu(model)
        self.offload.loaded_blocks[model_id] = None
        gc.collect()
        torch.cuda.empty_cache()

    def unload_all(self):
        if not self.installed:
            return

        active_ids = list(dict.fromkeys(getattr(self.offload, "active_models_ids", []) or []))
        for model_id in active_ids:
            self.unload(model_id, None)

        self.offload.active_models = []
        self.offload.active_models_ids = []
        gc.collect()
        torch.cuda.empty_cache()

    def ensure_model_loaded(self, obj, model_id):
        if model_id in getattr(obj, "active_models_ids", []):
            return

        # Do not use MMGP's co-tenancy decision here. Two Accelerate-dispatched
        # models can each legitimately occupy both GPUs, so keeping both alive
        # defeats the purpose of the memory budget.
        self.unload_all()

        model = obj.models[model_id]
        obj.active_models.append(model)
        obj.active_models_ids.append(model_id)
        self.load(model_id, None, preload=True)

    def release(self):
        if not self.installed:
            return

        self.unload_all()

        self.offload.gpu_load_blocks = self._original_load
        self.offload.gpu_unload_blocks = self._original_unload
        self.offload.unload_all = self._original_unload_all
        self.offload.release = self._original_release
        self.offload.ensure_model_loaded = self._original_ensure_model_loaded

        self.dispatched.clear()
        self.installed = False

    def install(self):
        if not self.enabled:
            raise ValueError("MultiGPU requires at least two CUDA devices")

        if self.installed:
            return self

        self._original_load = self.offload.gpu_load_blocks
        self._original_unload = self.offload.gpu_unload_blocks
        self._original_unload_all = self.offload.unload_all
        self._original_release = self.offload.release
        self._original_ensure_model_loaded = self.offload.ensure_model_loaded

        self.offload.gpu_load_blocks = types.MethodType(
            lambda obj, model_id, blocks_name, preload=False: self.load(model_id, blocks_name, preload),
            self.offload,
        )
        self.offload.gpu_unload_blocks = types.MethodType(
            lambda obj, model_id, blocks_name: self.unload(model_id, blocks_name),
            self.offload,
        )
        self.offload.unload_all = types.MethodType(
            lambda obj: self.unload_all(),
            self.offload,
        )
        self.offload.ensure_model_loaded = types.MethodType(
            self.ensure_model_loaded,
            self.offload,
        )
        self.offload.release = types.MethodType(
            lambda obj: self.release(),
            self.offload,
        )

        self.installed = True
        self._log(
            "Accelerate multi-GPU: "
            + " -> ".join(f"GPU#{i}" for i in range(len(self.devices)))
            + " -> RAM"
        )
        return self


def attach(offload, device_spec: str, fraction: float = 0.82, verbose: int = 1):
    devices = [part.strip() for part in str(device_spec or "").split(",") if part.strip()]
    normalized = [part if part.startswith("cuda:") else f"cuda:{part}" for part in devices]

    if len(normalized) < 2:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("MultiGPU requires CUDA")

    for device in normalized:
        index = torch.device(device).index
        if index is None or index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"Invalid MultiGPU CUDA device: {device}")

    torch.cuda.set_device(torch.device(normalized[0]))

    manager = AccelerateMultiGPU(
        offload,
        normalized,
        fraction=fraction,
        verbose=verbose,
    )

    # MMGP's async prefetch and residency are incompatible with a static
    # multi-device dispatch. Accelerate now owns model residency.
    if hasattr(offload, "async_transfers"):
        offload.async_transfers = False

    for model_id in getattr(offload, "preloaded_blocks_per_model", {}):
        offload.preloaded_blocks_per_model[model_id] = []

    manager.install()
    offload.multigpu = manager

    for no, device in enumerate(manager.devices):
        idx = manager._index(device) if hasattr(manager, "_index") else device.index
        free, total = torch.cuda.mem_get_info(idx)
        manager._log(
            f"GPU#{no} cuda:{idx}: {free / 1024**3:.2f} GiB free / "
            f"{total / 1024**3:.2f} GiB total"
        )

    return manager
