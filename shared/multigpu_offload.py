"""Multi-GPU model sharding for WanGP/MMGP using Hugging Face Accelerate.

MMGP remains responsible for discovering/loading models and model switching.
Accelerate owns the residency of an active model across every configured CUDA
device. System RAM is not a normal execution tier: MMGP only keeps its original
file-backed/CPU references so a dispatched model can be restored without a
second pinned host copy.
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
        # MMGP still owns the model's block registry and hooks.  While an
        # active model is dispatched by Accelerate, its block registry must
        # be empty so MMGP cannot silently pull a 1-2 GiB block back to GPU0.
        self._suspended_mmgp_blocks: dict[str, object] = {}
        self._suspended_mmgp_forwards: dict[str, list[tuple[torch.nn.Module, object]]] = {}
        self.installed = False

    @property
    def enabled(self):
        return len(self.devices) >= 2 and all(d.type == "cuda" for d in self.devices)

    def _log(self, message):
        if self.verbose >= 1:
            print(f"[MultiGPU] {message}", flush=True)

    @staticmethod
    def _mmgp_wrapper_target(forward, module):
        """Return the callable wrapped by one MMGP forward hook, if any.

        MMGP builds hooks with functools.partial + update_wrapper. update_wrapper
        intentionally copies the wrapped callable's __module__/__name__, so
        looking at forward.__module__ is unreliable. The actual MMGP wrapper is
        forward.func, while __wrapped__ (or module._mm_forward) points at the
        previous callable.
        """
        wrapper = getattr(forward, "func", forward)
        wrapper_module = getattr(wrapper, "__module__", "") or ""
        wrapper_name = getattr(wrapper, "__name__", "") or ""
        # Only strip MMGP *offload* wrappers. MMGP also owns LoRA forward
        # wrappers (_mm_lora_*); those must remain active because they apply
        # the adapter math after Accelerate places the base layer.
        is_mmgp = (
            wrapper_name.startswith("check_load_into_GPU_needed")
            or wrapper_name.startswith("check_change_module")
            or wrapper_name.startswith("_mm_wrap_")
        )
        if not is_mmgp:
            return None
        previous = getattr(forward, "__wrapped__", None)
        if previous is None:
            previous = getattr(module, "_mm_forward", None)
        return previous if callable(previous) and previous is not forward else None

    def _suspend_mmgp_forwards(self, model_id, model):
        """Temporarily remove *all* MMGP forward wrappers for Accelerate.

        The wrappers are saved verbatim and restored when the stage unloads.
        This prevents MMGP from performing an independent RAM->GPU block load
        from inside Accelerate's _old_forward.
        """
        if model_id in self._suspended_mmgp_forwards:
            return

        saved = []
        removed = 0
        for module in model.modules():
            current = getattr(module, "forward", None)
            if not callable(current):
                continue

            original_wrapper = current
            changed = False
            seen = set()
            while callable(current) and id(current) not in seen:
                seen.add(id(current))
                previous = self._mmgp_wrapper_target(current, module)
                if previous is None:
                    break
                current = previous
                changed = True
                removed += 1

            if changed:
                saved.append((module, original_wrapper))
                module.forward = current

        self._suspended_mmgp_forwards[model_id] = saved
        if removed:
            self._log(f"{model_id}: suspended {removed} MMGP forward hook(s)")

    def _restore_mmgp_forwards(self, model_id):
        saved = self._suspended_mmgp_forwards.pop(model_id, None)
        if not saved:
            return
        for module, wrapper in saved:
            try:
                module.forward = wrapper
            except Exception:
                pass
        self._log(f"{model_id}: restored {len(saved)} MMGP forward hook(s)")

    def _sanitize_accelerate_old_forwards(self, model_id, model):
        """Ensure Accelerate never calls back into an MMGP wrapper."""
        fixed = 0
        for module in model.modules():
            old_forward = getattr(module, "_old_forward", None)
            if not callable(old_forward):
                continue
            current = old_forward
            seen = set()
            while callable(current) and id(current) not in seen:
                seen.add(id(current))
                previous = self._mmgp_wrapper_target(current, module)
                if previous is None:
                    break
                current = previous
                fixed += 1
            if current is not old_forward:
                module._old_forward = current
        if fixed:
            self._log(f"{model_id}: removed {fixed} nested MMGP hook(s) from Accelerate _old_forward")
        elif self.verbose >= 2:
            self._log(f"{model_id}: Accelerate _old_forward verified clean of MMGP hooks")

    @staticmethod
    def _module_device(module):
        weight = getattr(module, "weight", None)
        device = getattr(weight, "device", None) if weight is not None else None
        if device is not None and torch.device(device).type != "meta":
            return torch.device(device)
        param = next(module.parameters(recurse=False), None)
        if param is not None and param.device.type != "meta":
            return param.device
        return None

    def _load_loras_for_dispatch(self, model_id, model):
        """Place active LoRA tensors beside each sharded base layer."""
        active = list(getattr(model, "_loras_active_adapters", None) or [])
        loras_model_data = getattr(model, "_loras_model_data", None)
        if not active or not loras_model_data:
            return

        shortcuts = getattr(model, "_loras_model_shortcuts", None) or {}
        tensor_cache = {}
        alias_cache = {}
        moved_bytes = 0
        moved_modules = 0

        def move_tensor(item, device):
            nonlocal moved_bytes
            if item is None:
                return None
            if torch.is_tensor(item):
                key = (id(item), str(device))
                moved = tensor_cache.get(key)
                if moved is None:
                    moved = item.to(device, non_blocking=True)
                    tensor_cache[key] = moved
                    try:
                        moved_bytes += moved.numel() * moved.element_size()
                    except Exception:
                        pass
                return moved
            return item

        def resolve_alias(value, adapter, device):
            if not isinstance(value, str):
                return move_tensor(value, device)
            cache_key = (adapter, value, str(device))
            if cache_key in alias_cache:
                return alias_cache[cache_key]
            seen = set()
            while isinstance(value, str):
                if value in seen or "#" not in value:
                    return None
                seen.add(value)
                target, slot = value.rsplit("#", 1)
                target_data = shortcuts.get(target)
                if target_data is None or adapter not in target_data:
                    return None
                value = target_data[adapter][int(slot)]
            resolved = move_tensor(value, device)
            alias_cache[cache_key] = resolved
            return resolved

        for module, adapters in loras_model_data.items():
            device = self._module_device(module)
            if device is None or device.type != "cuda":
                continue
            touched = False
            for adapter in active:
                source = adapters.get(adapter)
                if source is None:
                    continue
                moved = list(source)
                for slot in range(min(4, len(moved))):
                    moved[slot] = resolve_alias(moved[slot], adapter, device)
                adapters[adapter + "_GPU"] = moved
                touched = True
            if touched:
                moved_modules += 1

        if moved_modules:
            self._log(
                f"{model_id}: LoRA GPU residency prepared for {moved_modules} module(s), "
                f"{moved_bytes / 1024**2:.1f} MiB copied across shard devices"
            )

    def _unload_dispatched_loras(self, model):
        loras_model_data = getattr(model, "_loras_model_data", None)
        if not loras_model_data:
            return
        for adapters in loras_model_data.values():
            for key in [key for key in adapters if isinstance(key, str) and key.endswith("_GPU")]:
                del adapters[key]

    def _trim_host_working_set(self, reason=""):
        """Return inactive CPU/MMAP pages to Windows immediately.

        GPU-first dispatch necessarily reads the CPU backing tensors once while
        copying them to CUDA. On Windows those file-backed pages remain in this
        process' working set and can consume essentially all physical RAM even
        though the active model no longer needs them on CPU. EmptyWorkingSet
        evicts those inactive pages without dropping the mmap/backing objects,
        so MMGP can fault them back in later when another stage needs them.
        """
        if os.name != "nt":
            return

        gc.collect()
        before = after = None
        try:
            import psutil
            process = psutil.Process(os.getpid())
            before = process.memory_info().rss
        except Exception:
            process = None

        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            get_current_process = kernel32.GetCurrentProcess
            get_current_process.restype = wintypes.HANDLE
            empty_working_set = psapi.EmptyWorkingSet
            empty_working_set.argtypes = [wintypes.HANDLE]
            empty_working_set.restype = wintypes.BOOL

            handle = get_current_process()
            if not empty_working_set(handle):
                if self.verbose >= 2:
                    self._log(
                        f"Windows working-set trim failed ({ctypes.get_last_error()})"
                    )
                return
        except Exception as exc:
            if self.verbose >= 2:
                self._log(f"Windows working-set trim unavailable: {exc}")
            return

        if process is not None:
            try:
                after = process.memory_info().rss
            except Exception:
                pass

        if self.verbose >= 1:
            suffix = f" after {reason}" if reason else ""
            if before is not None and after is not None:
                self._log(
                    f"Host working set trimmed{suffix}: "
                    f"{before / 1024**3:.2f} -> {after / 1024**3:.2f} GiB"
                )
            else:
                self._log(f"Host working set trimmed{suffix}")

    def _remove_accelerate_hooks(self, model):
        try:
            from accelerate.hooks import remove_hook_from_submodules
            remove_hook_from_submodules(model)
        except ImportError:
            pass

    def _suspend_mmgp_blocks(self, model_id):
        if model_id in self._suspended_mmgp_blocks:
            return
        blocks = getattr(self.offload, "blocks_of_modules", {}).get(model_id)
        if blocks is None:
            return
        self._suspended_mmgp_blocks[model_id] = blocks
        self.offload.blocks_of_modules[model_id] = []
        self._log(f"{model_id}: MMGP block loader suspended; Accelerate owns residency")

    def _restore_mmgp_blocks(self, model_id):
        blocks = self._suspended_mmgp_blocks.pop(model_id, None)
        if blocks is not None:
            self.offload.blocks_of_modules[model_id] = blocks

    def _move_model_to_cpu(self, model):
        # Accelerate's hook detach path can try to materialize a meta Quanto
        # tensor and fail. Restore MMGP's original CPU tensor references
        # directly instead.
        model_id = None
        for candidate_id, candidate in self.offload.models.items():
            if candidate is model:
                model_id = candidate_id
                break

        self._unload_dispatched_loras(model)

        original_blocks = self._suspended_mmgp_blocks.get(model_id) if model_id is not None else None
        if model_id is not None:
            for parent, name, source, is_buffer, tied in (original_blocks or getattr(self.offload, "blocks_of_modules", {}).get(model_id, [])):
                try:
                    if tied is not None:
                        setattr(parent, name, getattr(tied[0], tied[1]))
                    else:
                        setattr(parent, name, source)
                except Exception:
                    pass

        for module in model.modules():
            # Do not call Accelerate's normal detach path here: for Quanto it
            # may materialize a meta tensor on CPU and transiently duplicate a
            # large weight. Restore the original forward callable directly.
            old_forward = getattr(module, "_old_forward", None)
            if old_forward is not None:
                try:
                    module.forward = old_forward
                    delattr(module, "_old_forward")
                except Exception:
                    pass
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

        if model_id is not None:
            self._restore_mmgp_blocks(model_id)
            self._restore_mmgp_forwards(model_id)

        gc.collect()
        torch.cuda.empty_cache()
        self._trim_host_working_set(
            f"{model_id} unload" if model_id is not None else "model unload"
        )

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
        transfer_state = {"calls": 0}

        def maybe_trim_dispatch_working_set(device):
            # Accelerate streams weights from MMGP's CPU/MMAP backing to CUDA.
            # Keep the process working set bounded while that streaming is in
            # progress instead of letting Windows retain every touched page.
            transfer_state["calls"] += 1
            if transfer_state["calls"] % 8:
                return
            try:
                import psutil
                process = psutil.Process(os.getpid())
                rss = process.memory_info().rss
                available = psutil.virtual_memory().available
                total = psutil.virtual_memory().total
                pressure = (
                    rss >= 3 * 1024**3
                    or available <= max(2 * 1024**3, int(total * 0.15))
                )
            except Exception:
                pressure = False

            if not pressure:
                return

            # Ensure any H2D read from the CPU backing completed before those
            # pages are made reclaimable by EmptyWorkingSet.
            try:
                torch.cuda.current_stream(torch.device(device)).synchronize()
            except Exception:
                pass
            self._trim_host_working_set("Accelerate tensor streaming")

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
                if torch.device(device).type == "cuda":
                    maybe_trim_dispatch_working_set(device)
                    if clear_cache:
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
        """GPU-only contiguous Gemma split using every configured CUDA device.

        Keep tied embed_tokens/lm_head together on the primary device and
        distribute decoder layers proportionally to currently available VRAM.
        A single contiguous range per GPU minimizes inter-device activation
        transfers while leaving extra headroom on GPU#0 for embeddings, logits
        and pipeline activations.
        """
        backbone = getattr(model, "model", None)
        layers = getattr(backbone, "layers", None)
        if layers is None or len(layers) < 2 or len(self.devices) < 2:
            return None

        total_layers = len(layers)
        usable = []
        for no, device in enumerate(self.devices):
            free, _ = torch.cuda.mem_get_info(device.index)
            reserve = (2.5 if no == 0 else 0.75) * 1024**3
            usable_bytes = max(256 * 1024**2, int(free * self.fraction) - int(reserve))
            usable.append(usable_bytes)

        total_usable = max(1, sum(usable))
        raw = [total_layers * value / total_usable for value in usable]
        counts = [int(value) for value in raw]
        remaining = total_layers - sum(counts)
        order = sorted(
            range(len(raw)),
            key=lambda i: (raw[i] - counts[i], usable[i]),
            reverse=True,
        )
        for i in order[:remaining]:
            counts[i] += 1

        # Use every configured GPU whenever there are enough decoder layers.
        if total_layers >= len(self.devices):
            empty = [i for i, count in enumerate(counts) if count == 0]
            for empty_i in empty:
                donor = max(range(len(counts)), key=lambda i: counts[i])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[empty_i] += 1

        device_map = {
            "model.embed_tokens": self.devices[0],
            "lm_head": self.devices[0],
        }

        cursor = 0
        for device, count in zip(self.devices, counts):
            for layer_no in range(cursor, min(total_layers, cursor + count)):
                device_map[f"model.layers.{layer_no}"] = device
            cursor += count
        while cursor < total_layers:
            device_map[f"model.layers.{cursor}"] = self.devices[-1]
            cursor += 1

        for child, _ in backbone.named_children():
            if child not in ("layers", "embed_tokens"):
                device_map[f"model.{child}"] = self.devices[-1]

        self._log(
            "Gemma layer split: "
            + ", ".join(f"{device}={count} layers" for device, count in zip(self.devices, counts))
        )
        return device_map

    def _dispatch(self, model_id):
        model = self.offload.models[model_id]

        if model_id in self.dispatched:
            return self.dispatched[model_id]

        self._remove_accelerate_hooks(model)
        self._suspend_mmgp_forwards(model_id, model)

        # Gemma uses tied input/output embeddings. Accelerate warns about this
        # when inferring a device map and may otherwise put lm_head on disk.
        tie_weights = getattr(model, "tie_weights", None)
        if callable(tie_weights):
            try:
                tie_weights()
            except Exception:
                pass

        # Gemma is deliberately GPU-only. Do not give Accelerate a CPU tier:
        # with 16 GiB system RAM, even a temporary CPU copy is undesirable.
        from accelerate import dispatch_model
        from accelerate.utils import infer_auto_device_map

        no_split = self._no_split_classes(model)
        device_map = None
        if getattr(model, "model", None) is not None and getattr(getattr(model, "model", None), "layers", None) is not None:
            # Gemma has tied input/output embeddings. Letting Accelerate infer
            # them independently can place embed_tokens and lm_head on
            # different GPUs, duplicating a very large tied tensor and causing
            # a delayed OOM in the first pre_forward input transfer.
            device_map = self._gemma_device_map(model)

        if device_map is not None:
            self._log(f"{model_id}: using GPU-only weighted contiguous split")
        else:
            # Generic models still use Accelerate's allocator. fallback_allocation
            # is enabled so it can recover from an unlucky first-fit placement.
            device_map = infer_auto_device_map(
                model,
                max_memory=self._max_memory(),
                no_split_module_classes=no_split,
                clean_result=True,
                offload_buffers=False,
                fallback_allocation=True,
            )

        counts = {}
        for device in device_map.values():
            key = str(device)
            counts[key] = counts.get(key, 0) + 1

        if "disk" in counts or "cpu" in counts or "meta" in counts:
            raise RuntimeError(
                f"Accelerate would require host/disk offload for {model_id}: {device_map}. "
                "MultiGPU mode is intentionally GPU-only because the host has limited RAM."
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

        # From this point until unload(), MMGP must not be allowed to perform
        # its own RAM -> CUDA block transfers. The model is already owned by
        # Accelerate and its parameters are placed by the device map.
        self._suspend_mmgp_blocks(model_id)

        # MMGP forward hooks are suspended above. Accelerate must see the
        # original module forwards, never MMGP's RAM->GPU wrappers.

        # Keep Accelerate's standard dispatch path. It handles cross-device
        # activation transfers and tied-parameter bookkeeping.
        main_device = device_map.get("model.embed_tokens", self.devices[0])
        try:
            dispatched = dispatch_model(
                model,
                device_map=device_map,
                main_device=main_device,
                offload_buffers=False,
                force_hooks=True,
            )
            self._sanitize_accelerate_old_forwards(model_id, dispatched)
            self._load_loras_for_dispatch(model_id, dispatched)
            dispatched.hf_device_map = device_map

            # At this point the active stage is fully GPU-resident. The CPU
            # originals are only backing storage, so evict their touched pages
            # now instead of waiting until the stage finishes.
            self._trim_host_working_set(f"{model_id} GPU dispatch")
        except Exception:
            # Never leave MMGP with a disabled block registry if Accelerate
            # fails during dispatch.
            self._restore_mmgp_blocks(model_id)
            self._restore_mmgp_forwards(model_id)
            raise

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

    def unload(self, model_id, blocks_name=None, cache=True):
        if blocks_name is not None:
            return
        model = self.dispatched.pop(model_id, None)
        if model is None:
            return
        self._log(f"{model_id}: stage complete; releasing all dispatched GPU weights")
        self._move_model_to_cpu(model)
        self.offload.loaded_blocks[model_id] = None
        gc.collect()
        for device in self.devices:
            try:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception:
                pass

    def unload_all(self):
        for model_id in list(self.dispatched):
            self.unload(model_id, None)
        self._release_mmgp()
        gc.collect()
        torch.cuda.empty_cache()

    def _release_mmgp(self):
        for model_id in list(self._suspended_mmgp_blocks):
            self._restore_mmgp_blocks(model_id)
        for model_id in list(self._suspended_mmgp_forwards):
            self._restore_mmgp_forwards(model_id)

    def release(self):
        if not self.installed:
            return
        self.unload_all()
        self.offload.gpu_load_blocks = self._original_load
        self.offload.gpu_unload_blocks = self._original_unload
        self.offload.unload_all = self._original_unload_all
        self.offload.release = self._original_release
        self.installed = False
        try:
            delattr(self.offload, "multigpu")
        except AttributeError:
            pass

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
            lambda obj, model_id, blocks_name: self.unload(model_id, blocks_name),
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
        self._log(
            "Accelerate GPU-first multi-GPU: " +
            " + ".join(f"GPU#{i}" for i in range(len(self.devices))) +
            " (RAM is backing/fallback only)"
        )
        for no, device in enumerate(self.devices):
            free, total = torch.cuda.mem_get_info(device.index)
            self._log(
                f"GPU#{no} {device}: {free / 1024**3:.2f} GiB free / "
                f"{total / 1024**3:.2f} GiB total"
            )
        return self


def attach(offload, device_spec: str, fraction: float = 0.92, verbose: int = 1):
    devices = [x.strip() for x in str(device_spec or "").split(",") if x.strip()]
    normalized = [x if x.startswith("cuda:") else f"cuda:{x}" for x in devices]
    if len(normalized) < 2:
        return None
    if not torch.cuda.is_available():
        raise RuntimeError("MultiGPU requires CUDA")
    count = torch.cuda.device_count()
    for name in normalized:
        index = torch.device(name).index
        if index is None or index < 0 or index >= count:
            raise ValueError(f"Invalid MultiGPU CUDA device: {name}")
    torch.cuda.set_device(torch.device(normalized[0]))
    manager = AccelerateMultiGPU(
        offload,
        normalized,
        fraction=fraction,
        verbose=verbose,
    )
    manager.install()
    offload.multigpu = manager
    return manager
