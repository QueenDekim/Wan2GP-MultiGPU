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
        self._jit_lora_hooks: dict[str, list[object]] = {}
        self.installed = False

    @property
    def enabled(self):
        return len(self.devices) >= 2 and all(d.type == "cuda" for d in self.devices)

    def _log(self, message):
        if self.verbose >= 1:
            print(f"[MultiGPU] {message}", flush=True)

    def _log_host_commit(self, label):
        """Track private commit separately from reclaimable working-set pages."""
        if self.verbose < 2:
            return
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            info = proc.memory_info()
            private = getattr(info, "private", None)
            if private is None:
                private = getattr(info, "pagefile", None)
            if private is None:
                private = info.vms
            available = psutil.virtual_memory().available
            self._log(
                f"Host memory [{label}]: private_commit={private / 1024**3:.2f} GiB; "
                f"working_set={info.rss / 1024**3:.2f} GiB; "
                f"system_available={available / 1024**3:.2f} GiB"
            )
        except Exception:
            pass

    @staticmethod
    def _iter_modules_unique(root):
        """Iterate an nn.Module graph once per object, even if cyclic."""
        stack = [root]
        seen = set()
        while stack:
            module = stack.pop()
            if not isinstance(module, torch.nn.Module):
                continue
            ident = id(module)
            if ident in seen:
                continue
            seen.add(ident)
            yield module
            children = getattr(module, "_modules", None)
            if isinstance(children, dict):
                stack.extend(child for child in children.values() if child is not None)

    def _detach_registered_lora_owner_cycles(self, model_id, model):
        """Keep MMGP _lora_owner references without registering them as children."""
        fixed = 0
        for module in list(self._iter_modules_unique(model)):
            modules = getattr(module, "_modules", None)
            if not isinstance(modules, dict):
                continue
            owner = modules.get("_lora_owner")
            if owner is None:
                continue
            try:
                del modules["_lora_owner"]
                object.__setattr__(module, "_lora_owner", owner)
                fixed += 1
            except Exception:
                pass
        if fixed:
            self._log(f"{model_id}: converted {fixed} registered _lora_owner link(s) to non-module references")

    def _detach_registered_module_cycles(self, model_id, model):
        """De-register only back-edges that make the nn.Module graph cyclic.

        Accelerate calls named_parameters(remove_duplicate=False), whose
        named_modules traversal has no cycle protection. Shared modules in a
        normal DAG are preserved; only edges to an ancestor currently being
        visited are converted to plain Python references.
        """
        state = {id(model): 1}  # 1=visiting, 2=finished
        stack = [(model, iter(list(getattr(model, "_modules", {}).items())))]
        fixed = []

        while stack:
            parent, children = stack[-1]
            try:
                name, child = next(children)
            except StopIteration:
                state[id(parent)] = 2
                stack.pop()
                continue

            if not isinstance(child, torch.nn.Module):
                continue

            child_state = state.get(id(child), 0)
            if child_state == 1:
                modules = getattr(parent, "_modules", None)
                if isinstance(modules, dict) and modules.get(name) is child:
                    try:
                        del modules[name]
                        object.__setattr__(parent, name, child)
                        fixed.append(name)
                    except Exception:
                        pass
                continue

            if child_state == 2:
                # Legitimate shared module / DAG edge.
                continue

            state[id(child)] = 1
            stack.append((
                child,
                iter(list(getattr(child, "_modules", {}).items())),
            ))

        if fixed:
            preview = ", ".join(fixed[:6])
            if len(fixed) > 6:
                preview += ", ..."
            self._log(
                f"{model_id}: de-registered {len(fixed)} cyclic module edge(s): {preview}"
            )

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
        for module in self._iter_modules_unique(model):
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
        for module in self._iter_modules_unique(model):
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

    def _profile_activation_reserve_bytes(self):
        """Activation headroom selected from the primary (cuda:0) VRAM class."""
        try:
            _, total = torch.cuda.mem_get_info(self.devices[0].index)
            gib = float(total) / 1024**3
        except Exception:
            gib = 16.0
        if gib < 10.0:
            reserve_gib = 1.25
        elif gib < 14.0:
            reserve_gib = 1.75
        elif gib < 20.0:
            reserve_gib = 2.50
        else:
            reserve_gib = 3.50
        return int(reserve_gib * 1024**3)

    @staticmethod
    def _resolve_lora_source(shortcuts, adapter, value):
        seen = set()
        while isinstance(value, str):
            if value in seen or "#" not in value:
                return None
            seen.add(value)
            target, slot = value.rsplit("#", 1)
            target_data = shortcuts.get(target)
            if target_data is None or adapter not in target_data:
                return None
            try:
                value = target_data[adapter][int(slot)]
            except Exception:
                return None
        return value

    def _lora_bytes_by_device(self, model, modules=None):
        """Estimate exact persistent GPU LoRA bytes using full-residency semantics."""
        active = list(getattr(model, "_loras_active_adapters", None) or [])
        loras_model_data = getattr(model, "_loras_model_data", None)
        if not active or not loras_model_data:
            return {}

        shortcuts = getattr(model, "_loras_model_shortcuts", None) or {}
        allowed = None if modules is None else {id(module) for module in modules}
        seen = set()
        totals = {}

        for module, adapters in loras_model_data.items():
            if allowed is not None and id(module) not in allowed:
                continue
            device = self._module_device(module)
            if device is None or device.type != "cuda":
                continue
            key_device = str(device)
            for adapter in active:
                source = adapters.get(adapter)
                if source is None:
                    continue
                for value in list(source)[:4]:
                    value = self._resolve_lora_source(shortcuts, adapter, value)
                    if not torch.is_tensor(value):
                        continue
                    key = (id(value), key_device)
                    if key in seen:
                        continue
                    seen.add(key)
                    totals[key_device] = totals.get(key_device, 0) + self._tensor_nbytes(value)
        return totals

    def _move_loras_for_modules(self, model, modules, forced_device=None):
        """Materialize active adapters for a selected module subset only."""
        active = list(getattr(model, "_loras_active_adapters", None) or [])
        loras_model_data = getattr(model, "_loras_model_data", None)
        if not active or not loras_model_data or not modules:
            return 0, 0

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
                    with torch.cuda.device(device):
                        moved = item.to(device, non_blocking=True)
                    tensor_cache[key] = moved
                    moved_bytes += self._tensor_nbytes(moved)
                return moved
            return item

        def resolve_alias(value, adapter, device):
            if not isinstance(value, str):
                return move_tensor(value, device)
            cache_key = (adapter, value, str(device))
            if cache_key in alias_cache:
                return alias_cache[cache_key]
            resolved = self._resolve_lora_source(shortcuts, adapter, value)
            resolved = move_tensor(resolved, device)
            alias_cache[cache_key] = resolved
            return resolved

        for module in modules:
            adapters = loras_model_data.get(module)
            if not adapters:
                continue
            device = torch.device(forced_device) if forced_device is not None else self._module_device(module)
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
        return moved_modules, moved_bytes

    @staticmethod
    def _drop_loras_for_modules(model, modules):
        loras_model_data = getattr(model, "_loras_model_data", None)
        if not loras_model_data:
            return
        for module in modules:
            adapters = loras_model_data.get(module)
            if not adapters:
                continue
            for key in [key for key in adapters if isinstance(key, str) and key.endswith("_GPU")]:
                del adapters[key]

    def _install_block_jit_loras(self, model_id, model):
        """Hybrid LoRA residency: fill spare VRAM, JIT only the remainder."""
        active = list(getattr(model, "_loras_active_adapters", None) or [])
        loras_model_data = getattr(model, "_loras_model_data", None)
        stages = self._block_pipeline_stages(model)
        blocks = [block for _, stage_blocks, _ in stages for block in stage_blocks]
        if not active or not loras_model_data or not blocks:
            return False

        full_bytes = self._lora_bytes_by_device(model)
        if not full_bytes:
            return False

        reserve = self._profile_activation_reserve_bytes()
        pressure = False
        diagnostics = []
        for device in self.devices:
            free, _ = torch.cuda.mem_get_info(device.index)
            lora_bytes = int(full_bytes.get(str(device), 0))
            remaining = int(free) - lora_bytes
            diagnostics.append(
                f"{device}: full-LoRA={lora_bytes / 1024**3:.2f} GiB, "
                f"post-LoRA-free={max(0, remaining) / 1024**3:.2f} GiB"
            )
            if lora_bytes and remaining < reserve:
                pressure = True

        if not pressure:
            if self.verbose >= 2:
                self._log(
                    f"{model_id}: full LoRA residency fits activation reserve "
                    f"({reserve / 1024**3:.2f} GiB); " + ", ".join(diagnostics)
                )
            return False

        module_to_group = {}
        groups = []
        for block in blocks:
            group = [
                module
                for module in self._iter_modules_unique(block)
                if module in loras_model_data
            ]
            groups.append(group)
            for module in group:
                module_to_group[id(module)] = True

        outside = [
            module for module in loras_model_data
            if id(module) not in module_to_group
        ]
        outside_modules, outside_bytes = self._move_loras_for_modules(model, outside)

        # Use otherwise-idle VRAM as a LoRA cache. Keep the activation reserve
        # untouched and leave an extra 256 MiB allocator guard per GPU.
        persistent_budget = {}
        for device in self.devices:
            free, _ = torch.cuda.mem_get_info(device.index)
            persistent_budget[str(device)] = max(
                0,
                int(free) - reserve - 256 * 1024**2,
            )

        persistent_by_device = {}
        jit_groups = []
        persistent_group_count = 0
        persistent_estimated_bytes = 0
        max_block_bytes = 0
        grouped_modules = 0

        for block, group in zip(blocks, groups):
            if not group:
                continue
            grouped_modules += len(group)
            device = None
            for lora_module in group:
                device = self._module_device(lora_module)
                if device is not None and device.type == "cuda":
                    break
            if device is None or device.type != "cuda":
                jit_groups.append((block, group))
                continue

            group_by_device = self._lora_bytes_by_device(model, group)
            group_bytes = int(group_by_device.get(str(device), 0))
            max_block_bytes = max(max_block_bytes, group_bytes)

            budget = persistent_budget.get(str(device), 0)
            if group_bytes and group_bytes <= budget:
                persistent_by_device.setdefault(str(device), (device, []))[1].extend(group)
                persistent_budget[str(device)] = budget - group_bytes
                persistent_group_count += 1
                persistent_estimated_bytes += group_bytes
            else:
                jit_groups.append((block, group))

        persistent_moved_bytes = 0
        persistent_modules = 0
        for _, (device, modules) in persistent_by_device.items():
            moved_modules, moved_bytes = self._move_loras_for_modules(
                model,
                modules,
                forced_device=device,
            )
            persistent_modules += moved_modules
            persistent_moved_bytes += moved_bytes

        handles = []
        for block, group in jit_groups:
            def pre_hook(_module, _inputs, group=tuple(group)):
                device = None
                for lora_module in group:
                    device = self._module_device(lora_module)
                    if device is not None and device.type == "cuda":
                        break
                if device is not None and device.type == "cuda":
                    self._move_loras_for_modules(model, group, forced_device=device)

            def post_hook(_module, _inputs, output, group=tuple(group)):
                self._drop_loras_for_modules(model, group)
                return output

            handles.append(block.register_forward_pre_hook(pre_hook))
            try:
                handles.append(block.register_forward_hook(post_hook, always_call=True))
            except TypeError:
                handles.append(block.register_forward_hook(post_hook))

        if handles:
            self._jit_lora_hooks.setdefault(model_id, []).extend(handles)

        # Even if every block fitted persistently, return True: the LoRA
        # residency was already prepared here and full-loader must not run.
        if persistent_group_count or handles or outside_modules:
            self._log(
                f"{model_id}: hybrid LoRA residency: "
                f"{persistent_group_count} block(s) persistent, "
                f"{len(jit_groups)} block(s) JIT; "
                f"persistent ~{persistent_moved_bytes / 1024**2:.1f} MiB, "
                f"JIT peak ~{max_block_bytes / 1024**2:.1f} MiB"
            )
            self._log(
                f"{model_id}: outside-block residency {outside_bytes / 1024**2:.1f} MiB "
                f"({outside_modules} module(s)); activation reserve "
                f"{reserve / 1024**3:.2f} GiB; " + ", ".join(diagnostics)
            )
            return True

        return False

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
        model_id = None
        for candidate_id, candidate in self.offload.models.items():
            if candidate is model:
                model_id = candidate_id
                break

        if model_id is not None:
            for handle in self._jit_lora_hooks.pop(model_id, []):
                try:
                    handle.remove()
                except Exception:
                    pass

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
        before = after = private_after = None
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
                mem_info = process.memory_info()
                after = mem_info.rss
                private_after = getattr(mem_info, "private", None)
            except Exception:
                pass

        if self.verbose >= 1:
            suffix = f" after {reason}" if reason else ""
            if before is not None and after is not None:
                private_suffix = (
                    f", private={private_after / 1024**3:.2f} GiB"
                    if private_after is not None else ""
                )
                self._log(
                    f"Host working set trimmed{suffix}: "
                    f"{before / 1024**3:.2f} -> {after / 1024**3:.2f} GiB"
                    f"{private_suffix}"
                )
            else:
                self._log(f"Host working set trimmed{suffix}")

    def _trim_host_working_set_if_pressure(self, reason="runtime pressure"):
        """Trim reclaimable mmap pages only when host RAM is actually tight."""
        if os.name != "nt":
            return False
        try:
            import psutil
            vm = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            rss = int(process.memory_info().rss)
            threshold = max(3 * 1024**3, int(vm.total * 0.20))
            pressured = int(vm.available) < threshold or rss > max(4 * 1024**3, int(vm.total * 0.30))
        except Exception:
            return False
        if not pressured:
            return False
        self._trim_host_working_set(reason)
        return True

    def _install_pass_end_host_trim(self, model_id, model):
        """Evict cold file-backed pages once per transformer pass under pressure."""
        stages = self._block_pipeline_stages(model)
        blocks = [block for _, stage_blocks, _ in stages for block in stage_blocks]
        if not blocks:
            return
        last_block = blocks[-1]

        def post_hook(_module, _inputs, output):
            self._trim_host_working_set_if_pressure(f"{model_id} denoise pass")
            return output

        try:
            handle = last_block.register_forward_hook(post_hook, always_call=True)
        except TypeError:
            handle = last_block.register_forward_hook(post_hook)
        self._jit_lora_hooks.setdefault(model_id, []).append(handle)

    def _ensure_state_dict_compat(self, model_id, model):
        """Repair every MMGP Quanto state_dict monkey-patch in the module graph.

        MMGP may install _quantize_dirty_hack on nested quantized modules, not
        just on the root model. PyTorch state_dict() recursively calls each
        child with destination/prefix/keep_vars, so every patched module must
        honor the normal nn.Module.state_dict signature.
        """
        import traceback
        from collections import OrderedDict

        repaired = 0

        def make_compat(module, real_state_dict):
            def state_dict_compat(*args, **kwargs):
                real_sd = real_state_dict(*args, **kwargs)

                # Recursive PyTorch / Accelerate calls pass destination,
                # prefix and/or keep_vars. Preserve that contract exactly.
                if args or kwargs:
                    return real_sd

                # MMGP's fake non-quantized key view is only required for the
                # LoRA initialization path that originally motivated the hack.
                fakeit = any(
                    "_lora_" in frame.name
                    for frame in traceback.extract_stack(limit=8)
                )
                if not fakeit:
                    return real_sd

                sd = OrderedDict()
                for key, value in real_sd.items():
                    if key.endswith("._data"):
                        key = key[:-6]
                    sd[key] = value
                return sd

            return state_dict_compat

        for module in self._iter_modules_unique(model):
            if getattr(module, "_wgp_state_dict_compat", False):
                continue
            real_state_dict = getattr(module, "_real_state_dict", None)
            if not callable(real_state_dict):
                continue
            try:
                module.state_dict = make_compat(module, real_state_dict)
                module._wgp_state_dict_compat = True
                repaired += 1
            except Exception:
                pass

        if repaired and self.verbose >= 2:
            self._log(
                f"{model_id}: repaired MMGP state_dict signature on "
                f"{repaired} module(s)"
            )

    def _remove_accelerate_hooks(self, model):
        """Remove Accelerate hooks without recursive child traversal."""
        try:
            from accelerate.hooks import remove_hook_from_module
        except ImportError:
            return

        removed = 0
        for module in self._iter_modules_unique(model):
            if not hasattr(module, "_hf_hook") and not hasattr(module, "_old_forward"):
                continue
            try:
                remove_hook_from_module(module, recurse=False)
                removed += 1
            except Exception:
                old_forward = getattr(module, "_old_forward", None)
                if callable(old_forward):
                    try:
                        module.forward = old_forward
                    except Exception:
                        pass
                    try:
                        delattr(module, "_old_forward")
                    except Exception:
                        pass
                if hasattr(module, "_hf_hook"):
                    try:
                        delattr(module, "_hf_hook")
                    except Exception:
                        pass

        if removed and self.verbose >= 2:
            self._log(f"removed Accelerate/fake HF hooks from {removed} module(s)")

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

        for module in self._iter_modules_unique(model):
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
            "LlamaDecoderLayer",
            "Qwen2DecoderLayer",
            "Qwen2VLDecoderLayer",
            "Qwen2_5_VLDecoderLayer",
            "WanTransformerBlock",
            "Wan2TransformerBlock",
        }
        stages = self._block_pipeline_stages(model)
        for _, blocks, _ in stages:
            if blocks:
                common.add(blocks[0].__class__.__name__)
        known = {module.__class__.__name__ for module in self._iter_modules_unique(model)}
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
            from optimum.quanto.tensor.qtensor import QTensor as QuantoQTensor
        except Exception:
            QuantoQTensor = ()

        try:
            from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor
        except Exception:
            WeightQBytesTensor = ()

        try:
            from shared.qtypes.scaled_fp8 import ScaledFP8WeightTensor
        except Exception:
            ScaledFP8WeightTensor = ()

        if QuantoQTensor == () and WeightQBytesTensor == () and ScaledFP8WeightTensor == ():
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
            special_types = tuple(
                cls for cls in (QuantoQTensor, WeightQBytesTensor, ScaledFP8WeightTensor)
                if isinstance(cls, type)
            )
            if special_types and isinstance(old_value, special_types):
                source = old_value
                if value is not None:
                    if isinstance(value, special_types):
                        source = value
                    else:
                        # GPU-only dispatch normally moves the existing
                        # wrapper in place. If Accelerate supplies a raw value
                        # for a custom quantized wrapper, let the original path
                        # handle it rather than guessing quantization metadata.
                        return original(
                            module, tensor_name, device, value=value, dtype=dtype,
                            fp16_statistics=fp16_statistics,
                            tied_params_map=tied_params_map,
                            non_blocking=non_blocking,
                            clear_cache=clear_cache,
                        )

                target = torch.device(device)
                if target.type == "cuda":
                    with torch.cuda.device(target):
                        new_value = source.to(target, non_blocking=non_blocking)
                else:
                    new_value = source.to(target, non_blocking=non_blocking)

                # Do not reconstruct custom QTensor subclasses through
                # param_cls(new_value): ScaledFP8WeightTensor and Quanto
                # wrappers have metadata-rich constructors. Their .to()
                # implementations preserve the wrapper and quantization data.
                module._parameters[tensor_name] = new_value
                if target.type == "cuda":
                    maybe_trim_dispatch_working_set(target)
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
        self._log("Accelerate quantized-tensor compatibility enabled (Quanto/Wan2GP custom QTensor)")


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
        if layers is None or len(layers) < len(self.devices) or len(self.devices) < 2:
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

        device_map = {}
        # Top-level decoder wrappers can carry extra modalities (vision,
        # projectors, adapters). Cover them without splitting their internals.
        for name, tensor in getattr(model, "_parameters", {}).items():
            if tensor is not None:
                device_map[name] = self.devices[0]
        for name, tensor in getattr(model, "_buffers", {}).items():
            if tensor is not None:
                device_map[name] = self.devices[0]
        for name, module in getattr(model, "_modules", {}).items():
            if module is not None and name != "model":
                device_map[name] = self.devices[0]

        for name, tensor in getattr(backbone, "_parameters", {}).items():
            if tensor is not None:
                device_map[f"model.{name}"] = self.devices[0]
        for name, tensor in getattr(backbone, "_buffers", {}).items():
            if tensor is not None:
                device_map[f"model.{name}"] = self.devices[0]

        cursor = 0
        for device, count in zip(self.devices, counts):
            for layer_no in range(cursor, min(total_layers, cursor + count)):
                device_map[f"model.layers.{layer_no}"] = device
            cursor += count
        while cursor < total_layers:
            device_map[f"model.layers.{cursor}"] = self.devices[-1]
            cursor += 1

        for child, _ in backbone.named_children():
            if child == "layers":
                continue
            device_map[f"model.{child}"] = (
                self.devices[-1] if child in ("norm", "final_layernorm") else self.devices[0]
            )

        self._log(
            f"{type(model).__name__} decoder layer split: "
            + ", ".join(f"{device}={count} layers" for device, count in zip(self.devices, counts))
        )
        return device_map

    @staticmethod
    def _tensor_nbytes(tensor):
        if tensor is None:
            return 0

        # Wan2GP custom quantized wrappers expose their true packed storage
        # explicitly. Summing those subtensors is more accurate than the
        # logical wrapper shape/dtype and works for FP8, NF4, NVFP4, GGUF,
        # W4A8 and Nunchaku tensors.
        getter = getattr(tensor, "get_quantized_subtensors", None)
        if callable(getter):
            try:
                total = 0
                seen = set()
                for _, item in getter():
                    if not torch.is_tensor(item) or id(item) in seen:
                        continue
                    seen.add(id(item))
                    total += int(item.numel()) * int(item.element_size())
                if total:
                    return total
            except Exception:
                pass

        data = getattr(tensor, "_data", None)
        if torch.is_tensor(data):
            total = int(data.numel()) * int(data.element_size())
            scale = getattr(tensor, "_scale", None)
            if torch.is_tensor(scale):
                total += int(scale.numel()) * int(scale.element_size())
            return total

        try:
            return int(tensor.numel()) * int(tensor.element_size())
        except Exception:
            return 0

    def _module_nbytes(self, module):
        """Approximate resident bytes without following external backrefs."""
        total = 0
        seen = set()
        for submodule in self._iter_modules_unique(module):
            for parameter in getattr(submodule, "_parameters", {}).values():
                if parameter is None or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                total += self._tensor_nbytes(parameter)
            for buffer in getattr(submodule, "_buffers", {}).values():
                if buffer is None or id(buffer) in seen:
                    continue
                seen.add(id(buffer))
                total += self._tensor_nbytes(buffer)
        return total

    def _block_pipeline_stages(self, model):
        """Return ordered whole-block stages for known WanGP transformer families."""
        velocity_model = getattr(model, "velocity_model", None)
        if velocity_model is not None and getattr(velocity_model, "transformer_blocks", None) is not None:
            return [("velocity_model.transformer_blocks", velocity_model.transformer_blocks, "LTX2")]

        if (
            getattr(model, "transformer_blocks", None) is not None
            and getattr(model, "patchify_proj", None) is not None
            and getattr(model, "adaln_single", None) is not None
        ):
            return [("transformer_blocks", model.transformer_blocks, "LTX-Video")]

        if (
            getattr(model, "blocks", None) is not None
            and getattr(model, "patch_embedding", None) is not None
            and getattr(model, "head", None) is not None
        ):
            return [("blocks", model.blocks, "Wan")]

        if getattr(model, "double_blocks", None) is not None and getattr(model, "final_layer", None) is not None:
            stages = [("double_blocks", model.double_blocks, "Hunyuan")]
            single = getattr(model, "single_blocks", None)
            if single is not None and len(single):
                stages.append(("single_blocks", single, "Hunyuan"))
            return stages

        if getattr(model, "visual_transformer_blocks", None) is not None and getattr(model, "out_layer", None) is not None:
            stages = []
            text_blocks = getattr(model, "text_transformer_blocks", None)
            if text_blocks is not None and len(text_blocks):
                stages.append(("text_transformer_blocks", text_blocks, "Kandinsky5"))
            stages.append(("visual_transformer_blocks", model.visual_transformer_blocks, "Kandinsky5"))
            return stages

        # MiniMax H3 uses the root Qwen3-VL text decoder without a
        # nested "model" wrapper. It must participate in block-JIT LoRA
        # residency even when its large embedding table exceeds 40% of
        # the parameter storage.
        root_layers = getattr(model, "layers", None)
        if (
            type(model).__name__ == "Qwen3VLTextModel"
            and isinstance(root_layers, (torch.nn.ModuleList, torch.nn.Sequential))
            and len(root_layers) >= len(self.devices)
        ):
            return [("layers", root_layers, "Qwen3-VL") ]
        # Hugging Face decoder wrappers (Gemma, Qwen, Llama and multimodal
        # variants) are mapped by the decoder-specific strategies above, but
        # still expose contiguous decoder layers to the hybrid LoRA cache.
        backbone = getattr(model, "model", None)
        decoder_layers = getattr(backbone, "layers", None)
        if isinstance(decoder_layers, (torch.nn.ModuleList, torch.nn.Sequential)):
            if len(decoder_layers) >= len(self.devices):
                return [("model.layers", decoder_layers, "HF-decoder")]

        language_model = getattr(model, "language_model", None)
        backbone = getattr(language_model, "model", None)
        decoder_layers = getattr(backbone, "layers", None)
        if isinstance(decoder_layers, (torch.nn.ModuleList, torch.nn.Sequential)):
            if len(decoder_layers) >= len(self.devices):
                return [("language_model.model.layers", decoder_layers, "LLaVA-decoder")]

        # Many newer WanGP architectures have a single, sequential
        # transformer backbone rather than the historical Wan/LTX block names.
        # The fallback is intentionally conservative: only recognised root
        # ModuleLists/Sequentials holding most of the model weights qualify.
        # Unknown multi-stage control-flow remains with the generic allocator.
        candidates = (
            "transformer_blocks",
            "blocks",
            "joint_transformer_blocks",
            "single_transformer_blocks",
            "joint_blocks",
            "dit_blocks",
            "layers",
        )
        minimum_blocks = max(4, len(self.devices))
        for name in candidates:
            blocks = getattr(model, name, None)
            if not isinstance(blocks, (torch.nn.ModuleList, torch.nn.Sequential)):
                continue
            if len(blocks) < minimum_blocks:
                continue
            total_bytes = self._module_nbytes(model)
            if total_bytes <= 0:
                continue
            block_bytes = sum(self._module_nbytes(block) for block in blocks)
            if block_bytes < total_bytes * 0.60:
                continue
            return [(name, blocks, f"{type(model).__name__}/{name}")]

        return []

    def _weighted_top_level_pipeline_map(self, model):
        """GPU-only map for Wan/LTX-Video/Hunyuan/Kandinsky block pipelines.

        Only whole transformer blocks are sharded. All preprocessing, direct
        parameters and output modules stay on cuda:0, so model-specific root
        forward code always begins and ends on the primary GPU.
        """
        stages = self._block_pipeline_stages(model)
        # Nested decoder and LTX2 paths have dedicated maps. This builder
        # handles only root ModuleLists so all non-block support layers are
        # guaranteed to be covered on the primary GPU.
        if not stages or any("." in stage_path for stage_path, _, _ in stages):
            return None

        flattened = []
        for path, blocks, _ in stages:
            for index, block in enumerate(blocks):
                flattened.append((f"{path}.{index}", block))
        if len(flattened) < len(self.devices):
            return None

        block_sizes = [max(1, self._module_nbytes(block)) for _, block in flattened]
        total_block_bytes = sum(block_sizes)
        stage_roots = {path.split(".", 1)[0] for path, _, _ in stages}

        primary_support_bytes = 0
        seen = set()
        for value in getattr(model, "_parameters", {}).values():
            if value is not None and id(value) not in seen:
                seen.add(id(value))
                primary_support_bytes += self._tensor_nbytes(value)
        for value in getattr(model, "_buffers", {}).values():
            if value is not None and id(value) not in seen:
                seen.add(id(value))
                primary_support_bytes += self._tensor_nbytes(value)
        for child_name, child in getattr(model, "_modules", {}).items():
            if child is None or child_name in stage_roots:
                continue
            primary_support_bytes += self._module_nbytes(child)

        reserve = self._profile_activation_reserve_bytes()
        usable = []
        free_now = []
        for no, device in enumerate(self.devices):
            free, _ = torch.cuda.mem_get_info(device.index)
            free_now.append(int(free))
            budget = int(free * self.fraction) - reserve
            if no == 0:
                budget -= primary_support_bytes
            usable.append(max(256 * 1024**2, budget))

        if sum(usable) < total_block_bytes:
            self._log(
                f"{stages[0][2]} GPU-only map is tight: "
                f"{total_block_bytes / 1024**3:.2f} GiB block weights vs "
                f"{sum(usable) / 1024**3:.2f} GiB profiled capacity"
            )

        total_usable = max(1, sum(usable))
        cumulative_targets = []
        running = 0
        for value in usable[:-1]:
            running += value
            cumulative_targets.append(total_block_bytes * running / total_usable)

        boundaries = []
        running_bytes = 0
        target_no = 0
        for block_no, size in enumerate(block_sizes):
            if target_no >= len(cumulative_targets):
                break
            before = running_bytes
            after = running_bytes + size
            target = cumulative_targets[target_no]
            blocks_left_after = len(block_sizes) - block_no - 1
            gpus_left = len(self.devices) - target_no - 1
            must_cut = blocks_left_after == gpus_left
            if must_cut or after >= target:
                previous_cut = boundaries[-1] if boundaries else 0
                cut_after = block_no + 1
                if (
                    not must_cut
                    and block_no > previous_cut
                    and abs(target - before) < abs(after - target)
                ):
                    cut_after = block_no
                cut_after = max(previous_cut + 1, min(cut_after, len(block_sizes) - gpus_left))
                boundaries.append(cut_after)
                target_no += 1
            running_bytes = after

        while len(boundaries) < len(self.devices) - 1:
            previous = boundaries[-1] if boundaries else 0
            remaining = len(self.devices) - 1 - len(boundaries)
            boundaries.append(min(len(block_sizes) - remaining, previous + 1))

        ranges = []
        start = 0
        for end in boundaries + [len(block_sizes)]:
            ranges.append((start, end))
            start = end

        device_map = {}
        # Direct root tensors and every non-block child stay on the primary.
        for name, value in getattr(model, "_parameters", {}).items():
            if value is not None:
                device_map[name] = self.devices[0]
        for name, value in getattr(model, "_buffers", {}).items():
            if value is not None:
                device_map[name] = self.devices[0]
        for child_name, child in getattr(model, "_modules", {}).items():
            if child is None or child_name in stage_roots:
                continue
            device_map[child_name] = self.devices[0]

        range_logs = []
        for device, (start, end) in zip(self.devices, ranges):
            bytes_on_device = 0
            names = []
            for index in range(start, end):
                path, _ = flattened[index]
                device_map[path] = device
                bytes_on_device += block_sizes[index]
                names.append(path)
            short_first = names[0] if names else "-"
            short_last = names[-1] if names else "-"
            range_logs.append(
                f"{device}={short_first}..{short_last} "
                f"({bytes_on_device / 1024**3:.2f} GiB)"
            )

        label = stages[0][2]
        self._log(f"{label} weighted contiguous split: " + ", ".join(range_logs))
        if self.verbose >= 2:
            self._log(
                f"{label} primary support weights: "
                f"{primary_support_bytes / 1024**3:.2f} GiB on {self.devices[0]}; "
                f"activation reserve={reserve / 1024**3:.2f} GiB/device"
            )
        return device_map

    def _qwen3vl_text_device_map(self, model):
        """GPU-only whole-layer map for MiniMax H3 Qwen3VLTextModel.

        The Qwen3-VL text backbone exposes 50 decoder layers directly at
        model.layers rather than model.model.layers. Its embedding table is
        large enough that the generic 60%-of-weights block heuristic does
        not reliably recognize this as a sequential decoder. Reserve actual
        VRAM for root embeddings and assign only contiguous whole layers.
        Reject checkpoints that cannot fit; never silently offload to RAM.
        """
        layers = getattr(model, "layers", None)
        if (
            type(model).__name__ != "Qwen3VLTextModel"
            or not isinstance(layers, (torch.nn.ModuleList, torch.nn.Sequential))
            or len(layers) < len(self.devices)
        ):
            return None

        layer_bytes = [max(1, self._module_nbytes(layer)) for layer in layers]
        total_layers = len(layers)
        total_layer_bytes = sum(layer_bytes)
        root_support_bytes = 0
        for name, tensor in getattr(model, "_parameters", {}).items():
            if tensor is not None:
                root_support_bytes += self._tensor_nbytes(tensor)
        for name, tensor in getattr(model, "_buffers", {}).items():
            if tensor is not None:
                root_support_bytes += self._tensor_nbytes(tensor)
        for name, module in getattr(model, "_modules", {}).items():
            if module is not None and name != "layers":
                root_support_bytes += self._module_nbytes(module)

        # Text encoding has much smaller activation peaks than video denoising,
        # but the embedding lookup, rotary arguments and allocator still need
        # breathing room. GPU0 pays the support-module cost explicitly.
        capacities = []
        free_bytes = []
        for index, device in enumerate(self.devices):
            free, _ = torch.cuda.mem_get_info(device.index)
            free_bytes.append(int(free))
            guard = (768 if index == 0 else 512) * 1024**2
            support = root_support_bytes if index == 0 else 0
            capacities.append(max(0, int(free * self.fraction) - guard - support))

        total_capacity = sum(capacities)
        if total_layer_bytes > total_capacity:
            needed = (total_layer_bytes + root_support_bytes) / 1024**3
            accessible = sum(free_bytes) / 1024**3
            raise RuntimeError(
                f"MiniMax H3 Qwen3-VL text encoder needs ~{needed:.2f} GiB "
                f"packed GPU weights, but only {accessible:.2f} GiB VRAM is "
                f"currently free across {len(self.devices)} GPU(s) "
                f"(safe decoder budget {total_capacity / 1024**3:.2f} GiB). "
                "MultiGPU remains GPU-only: select a smaller Text Encoder "
                "checkpoint in MiniMax H3 settings (NVFP4 AWQ, GGUF Q4_K_M, "
                "or GGUF Q2_K), or free VRAM on the GPUs. "
                "Host RAM/disk offload is disabled."
            )

        # Find contiguous layer ranges that all actually fit. A plain
        # proportional split can still OOM due to one large decoder layer.
        prefix = [0]
        for amount in layer_bytes:
            prefix.append(prefix[-1] + amount)
        target = [total_layer_bytes * value / max(1, total_capacity) for value in capacities]
        states = {0: (0.0, ())}
        for device_no, capacity in enumerate(capacities):
            next_states = {}
            remaining_devices = len(self.devices) - device_no - 1
            for start, (score, ranges) in states.items():
                upper = total_layers - remaining_devices
                for end in range(start + 1, upper + 1):
                    weight = prefix[end] - prefix[start]
                    if weight > capacity:
                        break
                    if remaining_devices == 0 and end != total_layers:
                        continue
                    delta = (weight - target[device_no]) / max(1.0, target[device_no])
                    candidate = (score + delta * delta, ranges + ((start, end),))
                    previous = next_states.get(end)
                    if previous is None or candidate[0] < previous[0]:
                        next_states[end] = candidate
            states = next_states
            if not states:
                break
        if total_layers not in states:
            raise RuntimeError(
                "MiniMax H3 Qwen3-VL: no feasible whole-layer GPU split at "
                "the current free VRAM levels. Free GPU memory or choose "
                "GGUF Q4_K_M / Q2_K; host RAM fallback is disabled."
            )

        ranges = states[total_layers][1]
        device_map = {}
        for name, tensor in getattr(model, "_parameters", {}).items():
            if tensor is not None:
                device_map[name] = self.devices[0]
        for name, tensor in getattr(model, "_buffers", {}).items():
            if tensor is not None:
                device_map[name] = self.devices[0]
        for name, module in getattr(model, "_modules", {}).items():
            if module is None or name == "layers":
                continue
            device_map[name] = self.devices[-1] if name == "norm" else self.devices[0]

        distribution = []
        for device, (start, end) in zip(self.devices, ranges):
            for idx in range(start, end):
                device_map[f"layers.{idx}"] = device
            amount = (prefix[end] - prefix[start]) / 1024**3
            distribution.append(f"{device}=layers.{start}..{end-1} ({amount:.2f} GiB)")
        self._log(
            "Qwen3-VL 50-layer GPU-only contiguous split: " + ", ".join(distribution)
        )
        if self.verbose >= 2:
            self._log(
                "Qwen3-VL primary support: "
                f"{root_support_bytes / 1024**3:.2f} GiB; "
                f"decoder weights {total_layer_bytes / 1024**3:.2f} GiB"
            )
        return device_map

    def _llava_decoder_device_map(self, model):
        """Whole-layer split for the custom LLaVA/Llama text encoder."""
        language_model = getattr(model, "language_model", None)
        backbone = getattr(language_model, "model", None) if language_model is not None else None
        layers = getattr(backbone, "layers", None) if backbone is not None else None
        if layers is None or len(layers) < len(self.devices):
            return None

        sizes = [max(1, self._module_nbytes(layer)) for layer in layers]
        total = sum(sizes)
        usable = []
        for no, device in enumerate(self.devices):
            free, _ = torch.cuda.mem_get_info(device.index)
            reserve = (2.5 if no == 0 else 1.0) * 1024**3
            usable.append(max(256 * 1024**2, int(free * self.fraction) - int(reserve)))
        total_usable = max(1, sum(usable))
        targets = []
        running = 0
        for value in usable[:-1]:
            running += value
            targets.append(total * running / total_usable)

        boundaries = []
        acc = 0
        target_no = 0
        for index, size in enumerate(sizes):
            if target_no >= len(targets):
                break
            before, after = acc, acc + size
            remaining_layers = len(sizes) - index - 1
            remaining_gpus = len(self.devices) - target_no - 1
            if remaining_layers == remaining_gpus or after >= targets[target_no]:
                cut = index + 1
                if index > (boundaries[-1] if boundaries else 0) and abs(targets[target_no]-before) < abs(after-targets[target_no]):
                    cut = index
                cut = max((boundaries[-1] if boundaries else 0) + 1, min(cut, len(sizes)-remaining_gpus))
                boundaries.append(cut)
                target_no += 1
            acc = after

        ranges = []
        start = 0
        for end in boundaries + [len(sizes)]:
            ranges.append((start, end))
            start = end

        device_map = {}
        for name, tensor in getattr(model, "_parameters", {}).items():
            if tensor is not None:
                device_map[name] = self.devices[0]
        for name, tensor in getattr(model, "_buffers", {}).items():
            if tensor is not None:
                device_map[name] = self.devices[0]
        for name, tensor in getattr(language_model, "_parameters", {}).items():
            if tensor is not None:
                device_map[f"language_model.{name}"] = self.devices[0]
        for name, tensor in getattr(language_model, "_buffers", {}).items():
            if tensor is not None:
                device_map[f"language_model.{name}"] = self.devices[0]
        for name, tensor in getattr(backbone, "_parameters", {}).items():
            if tensor is not None:
                device_map[f"language_model.model.{name}"] = self.devices[0]
        for name, tensor in getattr(backbone, "_buffers", {}).items():
            if tensor is not None:
                device_map[f"language_model.model.{name}"] = self.devices[0]
        # Vision/projector side stays on primary.
        for child_name, child in getattr(model, "_modules", {}).items():
            if child is not None and child_name != "language_model":
                device_map[child_name] = self.devices[0]
        # Keep embeddings and lm_head together on primary; norm follows last layer.
        for child_name, child in getattr(language_model, "_modules", {}).items():
            if child is not None and child_name != "model":
                device_map[f"language_model.{child_name}"] = self.devices[0]
        for child_name, child in getattr(backbone, "_modules", {}).items():
            if child is None or child_name == "layers":
                continue
            device_map[f"language_model.model.{child_name}"] = (
                self.devices[-1] if child_name == "norm" else self.devices[0]
            )

        counts = []
        for device, (start, end) in zip(self.devices, ranges):
            for index in range(start, end):
                device_map[f"language_model.model.layers.{index}"] = device
            counts.append(end - start)
        self._log(
            "LLaVA/Llama decoder split: "
            + ", ".join(f"{device}={count} layers" for device, count in zip(self.devices, counts))
        )
        return device_map
    def _ltx2_device_map(self, model):
        """GPU-only weighted contiguous split for LTX2 X0Model.

        Accelerate generic inference can fall back to disk for the quantized
        X0Model even when aggregate CUDA VRAM is sufficient. LTX2 has an
        explicit sequential transformer_blocks pipeline, so place those blocks
        in contiguous ranges weighted by each GPU current usable VRAM.
        """
        velocity_model = getattr(model, "velocity_model", None)
        blocks = getattr(velocity_model, "transformer_blocks", None)
        if velocity_model is None or blocks is None or len(blocks) < 2 or len(self.devices) < 2:
            return None

        block_sizes = [max(1, self._module_nbytes(block)) for block in blocks]
        total_block_bytes = sum(block_sizes)

        block_ids = {id(block) for block in blocks}
        output_param_names = {"scale_shift_table", "audio_scale_shift_table"}
        output_child_names = {"norm_out", "proj_out", "audio_norm_out", "audio_proj_out"}
        primary_support_bytes = 0
        terminal_support_bytes = 0
        seen_tensors = set()

        for name, parameter in getattr(velocity_model, "_parameters", {}).items():
            if parameter is None or id(parameter) in seen_tensors:
                continue
            seen_tensors.add(id(parameter))
            size = self._tensor_nbytes(parameter)
            if name in output_param_names:
                terminal_support_bytes += size
            else:
                primary_support_bytes += size
        for name, buffer in getattr(velocity_model, "_buffers", {}).items():
            if buffer is None or id(buffer) in seen_tensors:
                continue
            seen_tensors.add(id(buffer))
            size = self._tensor_nbytes(buffer)
            if name in output_param_names:
                terminal_support_bytes += size
            else:
                primary_support_bytes += size
        for child_name, child in getattr(velocity_model, "_modules", {}).items():
            if child is None or child_name == "transformer_blocks" or id(child) in block_ids:
                continue
            size = self._module_nbytes(child)
            if child_name in output_child_names:
                terminal_support_bytes += size
            else:
                primary_support_bytes += size

        usable = []
        free_now = []
        for no, device in enumerate(self.devices):
            free, _ = torch.cuda.mem_get_info(device.index)
            free_now.append(int(free))
            budget = int(free * self.fraction)
            # Reserve activation/workspace VRAM according to the primary GPU
            # profile (8/12/16/24 GB). The same profile class governs all
            # shards, while the actual free VRAM still weights each GPU.
            budget -= self._profile_activation_reserve_bytes()
            if no == 0:
                budget -= primary_support_bytes
            if no == len(self.devices) - 1:
                budget -= terminal_support_bytes
            usable.append(max(256 * 1024**2, budget))

        if sum(usable) < total_block_bytes:
            self._log(
                "LTX2 GPU-only map is tight: "
                f"{total_block_bytes / 1024**3:.2f} GiB block weights vs "
                f"{sum(usable) / 1024**3:.2f} GiB profiled block capacity; "
                "using all available GPUs without host/disk offload"
            )

        total_usable = max(1, sum(usable))
        cumulative_targets = []
        running_usable = 0
        for value in usable[:-1]:
            running_usable += value
            cumulative_targets.append(total_block_bytes * running_usable / total_usable)

        boundaries = []
        running_bytes = 0
        target_no = 0
        for block_no, size in enumerate(block_sizes):
            if target_no >= len(cumulative_targets):
                break
            target = cumulative_targets[target_no]
            before = running_bytes
            after = running_bytes + size
            blocks_left_after = len(block_sizes) - (block_no + 1)
            gpus_left = len(self.devices) - (target_no + 1)
            must_cut = blocks_left_after == gpus_left
            crossed = after >= target
            if must_cut or crossed:
                cut_after = block_no + 1
                previous_cut = boundaries[-1] if boundaries else 0
                if (
                    not must_cut
                    and block_no > previous_cut
                    and abs(target - before) < abs(after - target)
                ):
                    cut_after = block_no
                min_cut = previous_cut + 1
                max_cut = len(block_sizes) - gpus_left
                cut_after = max(min_cut, min(cut_after, max_cut))
                boundaries.append(cut_after)
                target_no += 1
            running_bytes = after

        while len(boundaries) < len(self.devices) - 1:
            previous = boundaries[-1] if boundaries else 0
            remaining_gpus = len(self.devices) - 1 - len(boundaries)
            boundaries.append(min(len(block_sizes) - remaining_gpus, previous + 1))

        ranges = []
        start = 0
        for end in boundaries + [len(block_sizes)]:
            ranges.append((start, end))
            start = end

        # Never map the whole velocity_model root: Accelerate treats such
        # an entry as place_submodules=True and would temporarily move the
        # entire 19B tree to cuda:0 before child shard hooks are installed.
        # Direct parameters/buffers are listed separately and pre-positioned
        # before dispatch; real child modules receive explicit placements.
        device_map = {}
        last_device = self.devices[-1]

        for name, parameter in getattr(velocity_model, "_parameters", {}).items():
            if parameter is None:
                continue
            device_map[f"velocity_model.{name}"] = (
                last_device if name in output_param_names else self.devices[0]
            )
        for name, buffer in getattr(velocity_model, "_buffers", {}).items():
            if buffer is None:
                continue
            device_map[f"velocity_model.{name}"] = self.devices[0]

        for child_name, child in getattr(velocity_model, "_modules", {}).items():
            if child is None or child_name == "transformer_blocks":
                continue
            device_map[f"velocity_model.{child_name}"] = (
                last_device if child_name in output_child_names else self.devices[0]
            )

        range_logs = []
        for device, (start, end) in zip(self.devices, ranges):
            bytes_on_device = 0
            for block_no in range(start, end):
                device_map[f"velocity_model.transformer_blocks.{block_no}"] = device
                bytes_on_device += block_sizes[block_no]
            range_logs.append(
                f"{device}=blocks {start}-{end - 1} "
                f"({bytes_on_device / 1024**3:.2f} GiB weights)"
            )

        self._log("LTX2 weighted contiguous split: " + ", ".join(range_logs))
        if self.verbose >= 2:
            self._log(
                "LTX2 support weights: "
                f"{primary_support_bytes / 1024**3:.2f} GiB on {self.devices[0]}, "
                f"{terminal_support_bytes / 1024**3:.2f} GiB output heads on {self.devices[-1]}"
            )
            self._log(
                "LTX2 current free VRAM: "
                + ", ".join(
                    f"{device}={free / 1024**3:.2f} GiB"
                    for device, free in zip(self.devices, free_now)
                )
            )
        return device_map
    def _preplace_direct_device_map_tensors(self, model, device_map):
        """Place device-map entries that name direct parameters/buffers.

        Accelerate validates parameter-name entries for coverage but only
        installs execution hooks on actual submodules. LTX2 has a few direct
        velocity_model parameters, so move those tiny tensors explicitly while
        leaving every large child module to its own shard hook.
        """
        try:
            import accelerate.utils.modeling as accelerate_modeling
        except Exception:
            return 0

        try:
            module_names = set(dict(model.named_modules()).keys())
        except Exception:
            module_names = set()

        moved = 0
        for path, device in device_map.items():
            if not path or path in module_names:
                continue

            if "." in path:
                parent_path, tensor_name = path.rsplit(".", 1)
                try:
                    parent = model.get_submodule(parent_path)
                except Exception:
                    continue
            else:
                # Root-level direct parameters (for example LTX-Video's
                # scale_shift_table) are valid device-map coverage keys too.
                parent = model
                tensor_name = path

            is_parameter = tensor_name in getattr(parent, "_parameters", {})
            is_buffer = tensor_name in getattr(parent, "_buffers", {})
            if not is_parameter and not is_buffer:
                continue

            accelerate_modeling.set_module_tensor_to_device(
                parent,
                tensor_name,
                device,
                non_blocking=True,
                clear_cache=False,
            )
            moved += 1

        if moved and self.verbose >= 2:
            self._log(f"pre-positioned {moved} direct parameter/buffer device-map tensor(s)")
        return moved

    def _validate_device_map_coverage(self, model_id, model, device_map):
        """Reject missing tensors and divergent placement of tied parameters.

        A partial explicit map can silently strand root buffers on CPU, which
        then fails during forward or corrupts outputs on less common models.
        Check the complete model graph before any destructive GPU dispatch.
        """
        if not device_map:
            raise RuntimeError(f"{model_id}: empty MultiGPU device map")
        if "" in device_map:
            return  # A whole-model assignment covers every tensor.

        assigned = {}
        missing = []
        tied = {}
        for iterator in (
            model.named_parameters(recurse=True, remove_duplicate=False),
            model.named_buffers(recurse=True, remove_duplicate=False),
        ):
            for name, tensor in iterator:
                matching = name if name in device_map else None
                if matching is None:
                    prefix = name
                    while "." in prefix:
                        prefix = prefix.rsplit(".", 1)[0]
                        if prefix in device_map:
                            matching = prefix
                            break
                if matching is None:
                    missing.append(name)
                    continue
                placement = device_map[matching]
                target = str(torch.device(f"cuda:{placement}" if isinstance(placement, int) else placement))
                assigned[target] = assigned.get(target, 0) + 1

                # Tied parameters must not be pulled into multiple distinct
                # GPU allocations, especially for large LM input/output
                # embeddings. Ignore identity-equal meta buffers only when
                # there is no actual underlying storage to tie.
                if not getattr(tensor, "is_meta", False):
                    key = id(tensor)
                    prev = tied.get(key)
                    if prev is not None and prev[0] != target:
                        raise RuntimeError(
                            f"{model_id}: tied tensor {prev[1]!r} and {name!r} "
                            f"mapped to different devices ({prev[0]} vs {target})"
                        )
                    tied[key] = (target, name)

        if missing:
            preview = ", ".join(missing[:12])
            raise RuntimeError(
                f"{model_id}: incomplete GPU-only device map "
                f"({len(missing)} uncovered parameters/buffers): {preview}"
            )

        if self.verbose >= 2:
            self._log(
                f"{model_id}: device map covers all tensors; "
                + ", ".join(f"{dev}={count}" for dev, count in assigned.items())
            )

    def _dispatch(self, model_id):
        model = self.offload.models[model_id]

        if model_id in self.dispatched:
            return self.dispatched[model_id]

        self._log_host_commit(f"{model_id} before dispatch")
        self._detach_registered_lora_owner_cycles(model_id, model)
        self._detach_registered_module_cycles(model_id, model)
        self._ensure_state_dict_compat(model_id, model)
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
            # Gemma has tied input/output embeddings.
            device_map = self._gemma_device_map(model)
        elif (
            getattr(model, "velocity_model", None) is not None
            and getattr(getattr(model, "velocity_model", None), "transformer_blocks", None) is not None
        ):
            # LTX2 X0Model.
            device_map = self._ltx2_device_map(model)
        elif type(model).__name__ == "Qwen3VLTextModel":
            # MiniMax H3/Qwen3-VL exposes a root ModuleList, not model.layers.
            # The generic allocator otherwise sends the last layers to disk.
            device_map = self._qwen3vl_text_device_map(model)
        elif (
            getattr(getattr(model, "language_model", None), "model", None) is not None
            and getattr(getattr(getattr(model, "language_model", None), "model", None), "layers", None) is not None
        ):
            # Custom Hunyuan LLaVA encoder: never split inside a Llama decoder layer.
            device_map = self._llava_decoder_device_map(model)
        else:
            # Wan, LTX-Video 0.9.x, Hunyuan and Kandinsky5 expose explicit
            # sequential block lists. Shard only whole blocks and keep all
            # root preprocessing/output modules on cuda:0.
            device_map = self._weighted_top_level_pipeline_map(model)

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

        self._validate_device_map_coverage(model_id, model, device_map)

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
                "velocity_model.patchify_proj",
                "velocity_model.transformer_blocks.0",
                "velocity_model.transformer_blocks.23",
                "velocity_model.transformer_blocks.24",
                "velocity_model.transformer_blocks.47",
                "velocity_model.proj_out",
                "language_model.model.embed_tokens",
                "language_model.model.layers.0",
                "language_model.model.layers.15",
                "language_model.model.layers.16",
                "language_model.model.layers.31",
                "language_model.lm_head",
                "transformer_blocks.0",
                "transformer_blocks.23",
                "transformer_blocks.24",
                "transformer_blocks.47",
                "blocks.0",
                "blocks.19",
                "blocks.20",
                "blocks.39",
                "double_blocks.0",
                "double_blocks.19",
                "double_blocks.20",
                "double_blocks.53",
                "single_blocks.0",
                "single_blocks.39",
                "text_transformer_blocks.0",
                "text_transformer_blocks.3",
                "visual_transformer_blocks.0",
                "visual_transformer_blocks.29",
                "visual_transformer_blocks.30",
                "visual_transformer_blocks.59",
                "head",
                "final_layer",
                "out_layer",
            ):
                if name in device_map:
                    self._log(f"{model_id}: {name} -> {device_map[name]}")

        self._patch_accelerate_quanto()
        self._preplace_direct_device_map_tensors(model, device_map)

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
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"The following device_map keys do not match any submodules in the model:.*",
                    category=UserWarning,
                )
                dispatched = dispatch_model(
                    model,
                    device_map=device_map,
                    main_device=main_device,
                    offload_buffers=False,
                    force_hooks=True,
                )
            self._sanitize_accelerate_old_forwards(model_id, dispatched)
            if not self._install_block_jit_loras(model_id, dispatched):
                self._load_loras_for_dispatch(model_id, dispatched)
            self._install_pass_end_host_trim(model_id, dispatched)
            dispatched.hf_device_map = device_map

            # At this point the active stage is fully GPU-resident. The CPU
            # originals are only backing storage, so evict their touched pages
            # now instead of waiting until the stage finishes.
            self._trim_host_working_set(f"{model_id} GPU dispatch")
        except Exception:
            # dispatch_model mutates modules in-place as it progresses. Roll
            # the model all the way back to MMGP's original CPU/MMAP backing
            # instead of only restoring hook registries; otherwise a failed
            # dispatch can leak CUDA tensors or leave partial HF hooks behind.
            try:
                self._move_model_to_cpu(model)
            except Exception:
                self._remove_accelerate_hooks(model)
                self._restore_mmgp_blocks(model_id)
                self._restore_mmgp_forwards(model_id)

            gc.collect()
            for device in self.devices:
                try:
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                        if hasattr(torch.cuda, "ipc_collect"):
                            torch.cuda.ipc_collect()
                except Exception:
                    pass
            raise

        self.dispatched[model_id] = dispatched
        self._log_host_commit(f"{model_id} after dispatch")
        return dispatched

    def load(self, model_id, blocks_name, preload=False):
        if blocks_name is not None:
            return

        try:
            model = self._dispatch(model_id)
        except Exception:
            # Map inference and coverage checks happen before dispatch_model's
            # own rollback handler. Always restore MMGP hooks even when an
            # architecture is rejected before Accelerate touches its weights.
            if model_id not in self.dispatched:
                try:
                    self._move_model_to_cpu(self.offload.models[model_id])
                except Exception:
                    self._restore_mmgp_blocks(model_id)
                    self._restore_mmgp_forwards(model_id)
            raise
        self.offload.loaded_blocks[model_id] = None

        # Keep MMGP's own residency bookkeeping coherent. ensure_model_loaded()
        # uses active_models_ids to decide whether a model switch is required;
        # if we dispatch through Accelerate without updating these lists MMGP
        # will immediately try to unload/reload an already-active model.
        active_ids = getattr(self.offload, "active_models_ids", None)
        active_models = getattr(self.offload, "active_models", None)
        if isinstance(active_ids, list) and model_id not in active_ids:
            active_ids.append(model_id)
        if isinstance(active_models, list) and model not in active_models:
            active_models.append(model)

        return model

    def _mark_inactive(self, model_id, model=None):
        active_ids = getattr(self.offload, "active_models_ids", None)
        active_models = getattr(self.offload, "active_models", None)

        if isinstance(active_ids, list):
            while model_id in active_ids:
                active_ids.remove(model_id)

        if isinstance(active_models, list):
            if model is None:
                model = getattr(self.offload, "models", {}).get(model_id)
            if model is not None:
                while model in active_models:
                    active_models.remove(model)

    def unload(self, model_id, blocks_name=None, cache=True):
        if blocks_name is not None:
            return
        model = self.dispatched.pop(model_id, None)
        if model is None:
            self._mark_inactive(model_id)
            return
        self._log_host_commit(f"{model_id} before unload")
        self._log(f"{model_id}: stage complete; releasing all dispatched GPU weights")
        self._move_model_to_cpu(model)
        self._log_host_commit(f"{model_id} after unload")
        self._mark_inactive(model_id, model)
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
        if self.verbose >= 2:
            parts = []
            for device in self.devices:
                try:
                    free, total = torch.cuda.mem_get_info(device.index)
                    parts.append(f"{device}={free / 1024**3:.2f}/{total / 1024**3:.2f} GiB free")
                except Exception:
                    pass
            if parts:
                self._log(f"{model_id}: post-unload VRAM: " + ", ".join(parts))

    def unload_all(self, keep=None, *args, **kwargs):
        # MMGP 3.8.2 may call unload_all(keep=[...]) to preserve cotenants.
        # Preserve exactly those model ids and release everything else.
        if keep is None:
            keep_ids = set()
        elif isinstance(keep, str):
            keep_ids = {keep}
        else:
            try:
                keep_ids = set(keep)
            except TypeError:
                keep_ids = set()

        for model_id in list(self.dispatched):
            if model_id not in keep_ids:
                self.unload(model_id, None)

        self._release_mmgp(keep=keep_ids)

        # Defensive cleanup: MMGP can retain stale active ids for models which
        # were never dispatched due to a failed switch.
        active_ids = getattr(self.offload, "active_models_ids", None)
        active_models = getattr(self.offload, "active_models", None)
        if isinstance(active_ids, list):
            active_ids[:] = [mid for mid in active_ids if mid in keep_ids]
        if isinstance(active_models, list):
            models = getattr(self.offload, "models", {})
            kept_models = {models[mid] for mid in keep_ids if mid in models}
            active_models[:] = [m for m in active_models if m in kept_models]

        gc.collect()
        for device in self.devices:
            try:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _release_mmgp(self, keep=None):
        keep_ids = set(keep or ())
        for model_id in list(self._suspended_mmgp_blocks):
            if model_id not in keep_ids:
                self._restore_mmgp_blocks(model_id)
        for model_id in list(self._suspended_mmgp_forwards):
            if model_id not in keep_ids:
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
            lambda obj, model_id, blocks_name, preload=False, *a, **kw:
                self.load(model_id, blocks_name, preload),
            self.offload,
        )
        self.offload.gpu_unload_blocks = types.MethodType(
            lambda obj, model_id, blocks_name=None, *a, **kw:
                self.unload(model_id, blocks_name, **{
                    k: v for k, v in kw.items() if k == "cache"
                }),
            self.offload,
        )
        self.offload.unload_all = types.MethodType(
            lambda obj, *a, **kw: self.unload_all(*a, **kw),
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
