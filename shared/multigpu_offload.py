"""Multi-GPU model sharding for WanGP/MMGP using Hugging Face Accelerate.

MMGP remains responsible for discovering/loading models and model switching.
Accelerate owns the residency of an active model:

    GPU#0 -> GPU#1 -> ... -> CPU

The active model is dispatched once using Accelerate's device map, allowing
parameters to reside on multiple GPUs with CPU/RAM as the final tier.
"""
from __future__ import annotations

import gc
import os
import types

import torch


class AccelerateMultiGPU:
    def __init__(self, offload, devices: list[str], fraction: float = 0.92, verbose: int = 1):
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
        # MMGP can wrap the same module more than once. Unwrap every MMGP
        # residency wrapper before Accelerate gets control of the model.
        for module in model.modules():
            seen = set()
            while hasattr(module, "_mm_forward"):
                previous = getattr(module, "_mm_forward", None)
                if previous is None or id(previous) in seen:
                    break
                seen.add(id(previous))
                module.forward = previous
                try:
                    delattr(module, "_mm_forward")
                except AttributeError:
                    break

            if hasattr(module, "_hf_hook"):
                try:
                    delattr(module, "_hf_hook")
                except AttributeError:
                    pass

    def _remove_accelerate_hooks(self, model):
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(model)
        except ImportError:
            pass

    def _move_model_to_cpu(self, model):
        # Accelerate's hook detach path can try to materialize a meta Quanto
        # tensor and fail. Restore MMGP's original CPU tensor references
        # directly instead.
        model_id = None
        for candidate_id, candidate in self.offload.models.items():
            if candidate is model:
                model_id = candidate_id
                break

        self._restore_mmgp_forwards(model)

        if model_id is not None:
            for parent, name, source, is_buffer, tied in self.offload.blocks_of_modules.get(model_id, []):
                try:
                    if tied is not None:
                        setattr(parent, name, getattr(tied[0], tied[1]))
                    else:
                        setattr(parent, name, source)
                except Exception:
                    pass

        for module in model.modules():
            if hasattr(module, "_hf_hook"):
                try:
                    delattr(module, "_hf_hook")
                except AttributeError:
                    pass

        if hasattr(model, "hf_device_map"):
            try:
                delattr(model, "hf_device_map")
            except AttributeError:
                pass

        gc.collect()
        torch.cuda.empty_cache()

    def _available_ram(self):
        try:
            import psutil
            return int(psutil.virtual_memory().available)
        except Exception:
            pass

        if os.name == "posix":
            try:
                page = os.sysconf("SC_PAGE_SIZE")
                pages = os.sysconf("SC_AVPHYS_PAGES")
                return int(page * pages)
            except Exception:
                pass

        # Safe fallback: enough host memory for the final tier on typical
        # WanGP systems. Accelerate will still fail explicitly if the OS
        # cannot allocate it.
        return 32 * 1024**3

    def _max_memory(self):
        # CPU/disk entries are intentionally excluded. RAM remains MMGP's
        # model-switch tier; Accelerate only shards the active model across
        # the configured GPUs.
        max_memory = {}
        for no, device in enumerate(self.devices):
            index = device.index
            free, _ = torch.cuda.mem_get_info(index)
            share = self.fraction
            if no == 0 and len(self.devices) > 1:
                # GPU#0 also receives inputs and activations.
                share *= 0.84
            max_memory[index] = max(1, int(free * share))
        return max_memory

    def _no_split_classes(self, model):
        classes = list(getattr(model, "_no_split_modules", None) or [])

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

    def _patch_accelerate_quanto(self):
        """Make Accelerate dispatch compatible with optimum-quanto weights.

        Accelerate's generic parameter path reconstructs the parameter as
        type(param)(tensor) after moving it. That is valid for nn.Parameter
        and bitsandbytes parameters, but not for optimum-quanto weights.
        Quanto already implements the correct tensor.to(device) operation,
        so keep the quantized wrapper intact.
        """
        try:
            from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor
        except Exception:
            return

        import accelerate.hooks as accelerate_hooks
        import accelerate.utils.modeling as accelerate_modeling

        if getattr(accelerate_hooks, "_wgp_quanto_patch", False):
            return

        original = accelerate_modeling.set_module_tensor_to_device

        def set_module_tensor_to_device_compat(
            module, tensor_name, device, value=None, dtype=None,
            fp16_statistics=None, tied_params_map=None, non_blocking=False,
            clear_cache=True,
        ):
            if "." in tensor_name:
                parts = tensor_name.split(".")
                for part in parts[:-1]:
                    module = getattr(module, part)
                tensor_name = parts[-1]

            old_value = getattr(module, tensor_name)
            if isinstance(old_value, WeightQBytesTensor):
                if value is not None:
                    if not isinstance(value, WeightQBytesTensor):
                        return original(
                            module, tensor_name, device, value=value, dtype=dtype,
                            fp16_statistics=fp16_statistics,
                            tied_params_map=tied_params_map,
                            non_blocking=non_blocking,
                            clear_cache=clear_cache,
                        )
                    new_value = value.to(device, non_blocking=non_blocking)
                else:
                    new_value = old_value.to(device, non_blocking=non_blocking)

                module._parameters[tensor_name] = new_value
                if clear_cache and torch.device(device).type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                return

            return original(
                module, tensor_name, device, value=value, dtype=dtype,
                fp16_statistics=fp16_statistics,
                tied_params_map=tied_params_map,
                non_blocking=non_blocking,
                clear_cache=clear_cache,
            )

        accelerate_modeling.set_module_tensor_to_device = set_module_tensor_to_device_compat
        accelerate_hooks.set_module_tensor_to_device = set_module_tensor_to_device_compat
        try:
            import accelerate.utils as accelerate_utils
            accelerate_utils.set_module_tensor_to_device = set_module_tensor_to_device_compat
        except Exception:
            pass

        accelerate_hooks._wgp_quanto_patch = True
        self._log("Accelerate Quanto compatibility enabled")


    def _gemma_device_map(self, model):
        """Build a contiguous Gemma map with a real CPU/RAM safety tier.

        The previous GPU-only split packed ~16 GiB of Quanto weights into two
        16 GiB cards. That leaves essentially no room for Gemma activations or
        temporary CUDA allocations, so the first embedding/attention operation
        can still OOM even though the static weights fit.

        Accelerate already has the correct mechanism for this: GPU model
        parallelism plus CPU offload. Keep the decoder contiguous and reserve
        substantial VRAM for execution instead of filling both cards.
        """
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
        if layers is None or len(layers) < 2 or len(self.devices) < 2:
            return None

        device_map = {
            "model.embed_tokens": self.devices[0],
            "lm_head": self.devices[0],
        }

        for name, module in backbone.named_children():
            if name != "layers":
                device_map[f"model.{name}"] = self.devices[-1]

        total = len(layers)
        gpu_targets = []
        for no, device in enumerate(self.devices):
            free, total_vram = torch.cuda.mem_get_info(device.index)
            ratio = 0.36 if no == 0 else 0.42
            gpu_targets.append(max(1, int(min(free, total_vram) * ratio)))

        from accelerate.utils import compute_module_sizes
        sizes = compute_module_sizes(model)
        used = [0] * len(self.devices)
        current = 0
        for i in range(total):
            name = f"model.layers.{i}"
            size = int(sizes.get(name, 0))
            if current < len(self.devices) - 1 and used[current] + size > gpu_targets[current]:
                current += 1
            if current == len(self.devices) - 1 and used[current] + size > gpu_targets[current]:
                device_map[name] = "cpu"
            else:
                device_map[name] = self.devices[current]
                used[current] += size

        return device_map

    def _dispatch(self, model_id):
        model = self.offload.models[model_id]

        if model_id in self.dispatched:
            return self.dispatched[model_id]

        self._restore_mmgp_forwards(model)
        self._remove_accelerate_hooks(model)

        # Gemma uses tied input/output embeddings. Accelerate warns about this
        # when inferring a device map and may otherwise put lm_head on disk.
        tie_weights = getattr(model, "tie_weights", None)
        if callable(tie_weights):
            try:
                tie_weights()
            except Exception:
                pass

        max_memory = self._max_memory()

        from accelerate import dispatch_model
        from accelerate.utils import infer_auto_device_map

        no_split = self._no_split_classes(model)

        # Gemma/Quanto gets a contiguous GPU + CPU map with deliberate VRAM
        # headroom. CPU here means system RAM, not disk; MMGP still owns the
        # model's RAM residency between model switches.
        device_map = self._gemma_device_map(model)

        if device_map is not None:
            self._log(f"{model_id}: using contiguous Gemma GPU + RAM layer split")
        else:
            # Generic models still use Accelerate's allocator. fallback_allocation
            # is enabled so it can recover from an unlucky first-fit placement.
            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split,
                clean_result=True,
                offload_buffers=False,
                fallback_allocation=True,
            )

        counts = {}
        for device in device_map.values():
            key = str(device)
            counts[key] = counts.get(key, 0) + 1

        if "disk" in counts or "meta" in counts:
            raise RuntimeError(
                f"Accelerate would require disk/meta offload for {model_id}: {device_map}. "
                "Disk offload is intentionally disabled; use the RAM tier or a smaller "
                "quantized checkpoint."
            )

        self._log(
            f"{model_id}: Accelerate device map "
            + ", ".join(f"{device}={count}" for device, count in counts.items())
        )

        # At verbose=2 expose the important root assignments. Gemma's tied
        # embedding/lm_head can otherwise hide a large GPU#0 allocation.
        if self.verbose >= 2:
            for name in (
                "model.embed_tokens",
                "model.layers.0",
                "model.layers.23",
                "model.layers.24",
                "model.layers.47",
                "lm_head",
                "model.norm",
            ):
                if name in device_map:
                    self._log(f"{model_id}: {name} -> {device_map[name]}")

        self._patch_accelerate_quanto()

        # Keep Accelerate's standard dispatch path. It handles cross-device
        # activation transfers and tied-parameter bookkeeping.
        dispatched = dispatch_model(
            model,
            device_map=device_map,
            main_device=self.devices[0],
            offload_buffers=False,
            force_hooks=True,
        )
        dispatched.hf_device_map = device_map

        self.dispatched[model_id] = dispatched
        return dispatched

    def load(self, model_id, blocks_name, preload=False):
        if blocks_name is not None:
            return

        model = self._dispatch(model_id)
        self.offload.loaded_blocks[model_id] = None

        devices_used = set()
        for module in model.modules():
            param = next(module.parameters(recurse=False), None)
            if param is not None:
                devices_used.add(str(param.device))

        self._log(f"{model_id}: active on " + ", ".join(sorted(devices_used)))

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


def attach(offload, device_spec: str, fraction: float = 0.92, verbose: int = 1):
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

    if hasattr(offload, "async_transfers"):
        offload.async_transfers = False

    for model_id in getattr(offload, "preloaded_blocks_per_model", {}):
        offload.preloaded_blocks_per_model[model_id] = []

    manager.install()
    offload.multigpu = manager

    for no, device in enumerate(manager.devices):
        idx = device.index
        free, total = torch.cuda.mem_get_info(idx)
        manager._log(
            f"GPU#{no} cuda:{idx}: {free / 1024**3:.2f} GiB free / "
            f"{total / 1024**3:.2f} GiB total"
        )

    return manager
