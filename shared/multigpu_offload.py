"""Tiered Multi-GPU cache for MMGP.

The normal WanGP/MMGP model remains RAM-backed. GPU#0 is the active execution
device. When MMGP evicts a block, this module keeps the GPU copy on GPU#1,
then GPU#2, ... before finally dropping it back to the original RAM tensor.
"""
from __future__ import annotations

import gc
import types
from dataclasses import dataclass

import torch


def _parameter(tensor):
    if torch.is_inference_mode_enabled():
        with torch.inference_mode(False):
            return torch.nn.Parameter(tensor, requires_grad=False)
    return torch.nn.Parameter(tensor, requires_grad=False)


def _buffer(tensor):
    if torch.is_inference_mode_enabled():
        with torch.inference_mode(False):
            return torch.nn.Buffer(tensor)
    return torch.nn.Buffer(tensor)


@dataclass
class _CachedBlock:
    tier: int
    size: int
    tensors: dict[int, torch.Tensor]
    last_used: int


class TieredGPUOffload:
    def __init__(self, offload, devices: list[str], fraction: float = 0.82, verbose: int = 1):
        self.offload = offload
        self.devices = [torch.device(d) for d in devices]
        self.fraction = max(0.05, min(float(fraction), 0.98))
        self.verbose = int(verbose)
        self.cache: dict[str, _CachedBlock] = {}
        self.used = [0] * max(0, len(self.devices) - 1)
        self.clock = 0
        self.installed = False

    @property
    def enabled(self):
        return len(self.devices) >= 2 and all(d.type == "cuda" for d in self.devices)

    def _log(self, message):
        if self.verbose >= 1:
            print(f"[MultiGPU] {message}", flush=True)

    def _key(self, model_id, blocks_name):
        return model_id if blocks_name is None else f"{model_id}/{blocks_name}"

    def _index(self, device):
        return device.index if device.index is not None else torch.cuda.current_device()

    def _capacity(self, tier):
        total = torch.cuda.get_device_properties(self._index(self.devices[tier + 1])).total_memory
        return int(total * self.fraction)

    def _size(self, entry):
        return int(self.offload.blocks_of_modules_sizes.get(entry, 0))

    def _clear_cache_entry(self, entry):
        item = self.cache.pop(entry, None)
        if item is None:
            return
        self.used[item.tier] -= item.size
        item.tensors.clear()

    def _demote(self, entry, target_tier):
        item = self.cache.get(entry)
        if item is None:
            return

        if target_tier >= len(self.used):
            self._clear_cache_entry(entry)
            self._log(f"{entry}: GPU#{item.tier + 1} -> RAM")
            return

        self._ensure_capacity(target_tier, item.size, exclude=entry)
        target = self.devices[target_tier + 1]
        moved = {}
        try:
            with torch.cuda.device(target):
                for key, tensor in item.tensors.items():
                    moved[key] = tensor.to(target, non_blocking=False)
        except Exception:
            for tensor in moved.values():
                del tensor
            raise

        self.used[item.tier] -= item.size
        self.used[target_tier] += item.size
        old_tier = item.tier
        item.tier = target_tier
        item.tensors = moved
        item.last_used = self.clock
        self._log(f"{entry}: GPU#{old_tier + 1} -> GPU#{target_tier + 1}")

    def _ensure_capacity(self, tier, size, exclude=None):
        capacity = self._capacity(tier)
        while self.used[tier] + size > capacity:
            candidates = [
                (name, item.last_used)
                for name, item in self.cache.items()
                if item.tier == tier and name != exclude
            ]
            if not candidates:
                raise RuntimeError(
                    f"GPU#{tier + 1} cache cannot fit {size / 1024**2:.1f} MiB"
                )
            victim = min(candidates, key=lambda x: x[1])[0]
            self._demote(victim, tier + 1)

    def _current_tensors(self, entry):
        tensors = {}
        for parent, name, source, is_buffer, tied in self.offload.blocks_of_modules[entry]:
            if tied is not None:
                continue
            current = getattr(parent, name)
            if torch.is_tensor(current) and current.is_cuda:
                tensors[id(source)] = current
        return tensors

    def _restore_cpu(self, entry):
        model_id = entry.split("/", 1)[0]
        model = self.offload.models[model_id]
        lora_modules = {}
        active = getattr(model, "_loras_active_adapters", None)
        lora_data = getattr(model, "_loras_model_data", None) if active else None

        for parent, name, source, is_buffer, _ in self.offload.blocks_of_modules[entry]:
            q = _buffer(source) if is_buffer else _parameter(source)
            setattr(parent, name, q)
            if lora_data is not None and parent in lora_data:
                lora_modules[parent] = lora_data[parent]

        if active and lora_modules:
            self.offload._move_loras(active, lora_modules, False, model)

    def _install_block(self, entry, tensors):
        target = self.devices[0]
        for parent, name, source, is_buffer, tied in self.offload.blocks_of_modules[entry]:
            if tied is not None:
                tied_value = getattr(tied[0], tied[1])
                setattr(parent, name, tied_value)
                continue

            cached = tensors.get(id(source))
            q = source.to(target, non_blocking=True) if cached is None else cached.to(target, non_blocking=True)
            q = _buffer(q) if is_buffer else _parameter(q)
            setattr(parent, name, q)

        model_id = entry.split("/", 1)[0]
        model = self.offload.models[model_id]
        active = getattr(model, "_loras_active_adapters", None)
        lora_data = getattr(model, "_loras_model_data", None) if active else None
        if active and lora_data is not None:
            modules = {}
            for parent, _, _, _, _ in self.offload.blocks_of_modules[entry]:
                if parent in lora_data:
                    modules[parent] = lora_data[parent]
            if modules:
                self.offload._move_loras(active, modules, True, model)

    def unload(self, model_id, blocks_name, cache=True):
        entry = self._key(model_id, blocks_name)

        if blocks_name is not None and blocks_name == self.offload.loaded_blocks[model_id]:
            self.offload.loaded_blocks[model_id] = None

        if entry not in self.offload.blocks_of_modules:
            return

        current = self._current_tensors(entry)
        if not current:
            return

        if not cache:
            self._restore_cpu(entry)
            return

        size = self._size(entry)
        try:
            self._ensure_capacity(0, size)
            target = self.devices[1]
            moved = {}
            with torch.cuda.device(target):
                for key, tensor in current.items():
                    moved[key] = tensor.to(target, non_blocking=False)
            self._restore_cpu(entry)
            self.cache[entry] = _CachedBlock(0, size, moved, self.clock)
            self.used[0] += size
            self._log(f"{entry}: GPU#0 -> GPU#1")
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            for tensor in current.values():
                del tensor
            self._restore_cpu(entry)
            self._clear_cache_entry(entry)
            gc.collect()
            torch.cuda.empty_cache()
            self._log(f"{entry}: GPU#0 -> RAM")

    def load(self, model_id, blocks_name, preload=False):
        entry = self._key(model_id, blocks_name)

        loaded = self.offload.loaded_blocks[model_id]
        if not preload and loaded is not None and loaded != blocks_name:
            self.unload(model_id, loaded, cache=True)

        item = self.cache.pop(entry, None)
        tensors = {}
        if item is not None:
            self.used[item.tier] -= item.size
            tensors = item.tensors
            self.clock += 1
            self._log(f"{entry}: GPU#{item.tier + 1} -> GPU#0")

        self._install_block(entry, tensors)
        tensors.clear()

        if not preload:
            self.offload.loaded_blocks[model_id] = blocks_name
        self.clock += 1

    def _release_cache(self):
        for item in self.cache.values():
            item.tensors.clear()
        self.cache.clear()
        self.used = [0] * len(self.used)
        gc.collect()
        torch.cuda.empty_cache()

    def unload_all(self):
        # unload_all is expected to return everything to RAM, not populate the
        # intermediate GPU cache. Temporarily restore MMGP's implementation.
        if not self.installed:
            return
        self.offload.gpu_load_blocks = self._original_load
        self.offload.gpu_unload_blocks = self._original_unload
        try:
            self._original_unload_all()
        finally:
            self.offload.gpu_load_blocks = types.MethodType(
                lambda obj, model_id, blocks_name, preload=False: self.load(model_id, blocks_name, preload),
                self.offload,
            )
            self.offload.gpu_unload_blocks = types.MethodType(
                lambda obj, model_id, blocks_name: self.unload(model_id, blocks_name, cache=True),
                self.offload,
            )
            self._release_cache()

    def release(self):
        if not self.installed:
            return
        self.offload.gpu_load_blocks = self._original_load
        self.offload.gpu_unload_blocks = self._original_unload
        self.offload.unload_all = self._original_unload_all
        self.offload.release = self._original_release
        self._release_cache()
        self._original_release()
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

        self.offload.gpu_load_blocks = types.MethodType(
            lambda obj, model_id, blocks_name, preload=False: self.load(model_id, blocks_name, preload),
            self.offload,
        )
        self.offload.gpu_unload_blocks = types.MethodType(
            lambda obj, model_id, blocks_name: self.unload(model_id, blocks_name, cache=True),
            self.offload,
        )
        self.offload.unload_all = types.MethodType(
            lambda obj: self.unload_all(),
            self.offload,
        )
        self.offload.release = types.MethodType(
            lambda obj: self.release(),
            self.offload,
        )
        self.installed = True
        self._log("Tiered offload: " + " -> ".join(f"GPU#{i}" for i in range(len(self.devices))) + " -> RAM")
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
    manager = TieredGPUOffload(offload, normalized, fraction=fraction, verbose=verbose)

    # MMGP preloading keeps extra blocks resident on GPU#0. That conflicts
    # with tiered offload, where secondary GPUs are the persistent cache.
    for model_id in getattr(offload, "preloaded_blocks_per_model", {}):
        offload.preloaded_blocks_per_model[model_id] = []

    manager.install()
    offload.multigpu = manager
    return manager
