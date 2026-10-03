############# WanGP Copyright DeepBeepMeep 2025-2026 #############
import os, sys
os.environ["GRADIO_LANG"] = "en"
# ROCm: without it, SDPA silently falls back to the math kernel on GPUs where AOTriton is still marked experimental
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
p = os.path.dirname(os.path.abspath(__file__))
if p not in sys.path:
    sys.path.insert(0, p)
from shared.native_runtime import preload_preferred_libstdcxx
preload_preferred_libstdcxx()
from shared.default_device import set_default_cuda_device_from_arg; set_default_cuda_device_from_arg("gpu")
# # os.environ.pop("TORCH_LOGS", None)  # make sure no env var is suppressing/overriding
# os.environ["TORCH_LOGS"]= "recompiles"
import torch._logging as tlog
# tlog.set_logs(recompiles=True, guards=True, graph_breaks=True)
# from shared.utils.crash_diagnostics import install_wgp_crash_diagnostics; install_wgp_crash_diagnostics(__file__)
# Ensure plugin-side `import wgp` resolves to this live module instance.
if sys.modules.get("wgp") is not sys.modules.get(__name__):
    sys.modules["wgp"] = sys.modules[__name__]
import asyncio
if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
if sys.platform.startswith("linux") and "NUMBA_THREADING_LAYER" not in os.environ:
    os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
from shared.asyncio_utils import silence_proactor_connection_reset
silence_proactor_connection_reset()

# ── Apple Silicon MPS patch: MUST come before mmgp import ──
import torch
is_mps = sys.platform == 'darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
if is_mps:
    from shared.mps.device_patch import apply_mps_patch
    apply_mps_patch()

import time
import threading
from functools import partial
from shared.utils.download_progress import DownloadCancelled, check_download_cancelled
from shared.utils.download import DownloadError, generation_downloads
from shared.utils.media_control import MediaProcessingAborted, controlled_model_loading, media_abort_requested
from shared.gradio import metadata_events
from shared.utils.config_store import config_lock, read_config, write_config, update_config
import warnings
warnings.filterwarnings('ignore', message='Failed to find.*', module='triton')
warnings.filterwarnings("ignore", message=r"Failed to launch Triton kernels, likely due to missing CUDA toolkit; falling back to a slower .* implementation\.\.\.", category=UserWarning, module=r"whisper\.timing")
from mmgp import offload, safetensors2, profile_type , quant_router
from shared.multigpu_offload import attach as attach_multigpu
try:
    import triton
except ImportError:
    pass
from pathlib import Path
from shared.tinyvae.media import VideoPreview
from datetime import datetime
import gradio as gr
from shared.gradio import downloads as gradio_downloads
from shared.gradio.progress import WangpProgress
from shared.gradio import gradio_event_context_patch, gradio_model_switch_patch, gradio_queue_focus_patch, gradio_startup_patch, video_preview
from gradio.themes.utils.sizes import Size
import random
import json
import copy
import numpy as np
import importlib
from models import model_metadata
from shared import notifications
from shared.utils import notification_sound
from shared.utils.setting_names import unknown_settings_error
from shared.utils.loras_mutipliers import preparse_loras_multipliers, parse_loras_multipliers
from shared.utils.utils import convert_tensor_to_image, convert_video_tensor_to_uint8_chunked, save_image, get_video_info, get_file_creation_date, convert_image_to_video, calculate_new_dimensions, convert_image_to_tensor, calculate_dimensions_and_resize_image, rescale_and_crop, get_video_frame, resize_and_remove_background, rgb_bw_to_rgba_mask, image_editor_layer_to_rgb_mask, to_rgb_tensor, get_resampled_video_transparent, get_video_summary_extras
from shared.utils.utils import calculate_new_dimensions, get_outpainting_dims, get_outpainting_frame_location, get_outpainting_full_area_dimensions, resolve_outpainting_dims
from shared.utils.utils import has_video_file_extension, has_image_file_extension, has_audio_file_extension
from shared.utils.audio_video import extract_audio_tracks, combine_video_with_audio_tracks, combine_and_concatenate_video_with_audio_tracks, cleanup_temp_audio_files, normalize_audio_volumes_to_temp_files, save_video, save_hdr_video, save_image, get_image_format, get_media_duration_and_audio_layouts
from shared.utils.audio_video import append_sliding_window_audio, read_image_metadata, extract_audio_track_to_wav, write_wav_file, save_audio_file, get_audio_codec_extension, create_silent_wav_file
from shared.utils.audio_video import truncate_audio, shift_audio_trim_ranges, trim_audio_ranges, trim_audio_file_ranges, slice_audio_window, resolve_mux_audio_sampling_rate
from shared.utils.audio_metadata import read_audio_metadata, extract_creation_datetime_from_metadata, resolve_audio_creation_datetime
from shared.utils.media_recording import record_file_metadata as shared_record_file_metadata
from shared.utils.settings_bundle import is_wangp_settings_filename
from shared.utils.video_decode import decode_video_frames_ffmpeg, probe_video_stream_metadata
from shared.utils.virtual_media import get_virtual_image, get_virtual_media_entry, get_virtual_media_vsource, media_source_exists, parse_virtual_media_path, replace_virtual_media_source, strip_virtual_media_suffix
from shared.utils.frame_scheduler import build_default_window_plan, build_extension_window, build_frame_scheduler, floor_frame_count, has_slash_commands, normalize_frame_count, prepare_loras_mult_windows
from shared.match_archi import match_nvidia_architecture
from shared.attention import ATTENTION_MODE_AVAILABILITY, get_attention_modes, get_supported_attention_modes, get_default_attention_mode, get_override_attention_modes, get_supported_override_attention_modes
from shared.utils.utils import truncate_for_filesystem, sanitize_file_name, process_images_multithread, get_default_workers, resize_lanczos_frames, expand_or_shrink_mask, prepare_binary_mask_frame
from shared.utils.process_locks import (
    acquire_GPU_ressources,
    acquire_main_GPU_ressources,
    any_GPU_process_running,
    gen_lock,
    register_GPU_resident,
    release_GPU_ressources,
    set_main_generation_running,
    unregister_GPU_resident,
)
from shared.utils.model_unload import model_unload_guard, wait_for_model_unload
from shared.deepy.config import DEEPY_KV_CACHE_QUANTIZATION_DEFAULT, DEEPY_KV_CACHE_QUANTIZATION_KEY, get_deepy_default_runtime_config, set_deepy_runtime_config
from shared.deepy.onboarding import apply_first_launch_deepy_prime_defaults
from shared.remote_llm.config import LLM_CONFIG_KEY, is_remote_engine, normalize_llm_config, resolve_role_engine
from shared.loras_migration import migrate_loras_layout
from shared.lora_paths import resolve_lora_dir
from shared.utils.wgp_config_migration import migrate_extension_defaults
from shared.utils import files_locator as fl 
from shared.gradio.audio_gallery import AudioGallery  
from shared.utils.self_refiner import normalize_self_refiner_plan, ensure_refiner_list, add_refiner_rule, remove_refiner_rule
from shared.deepy import controller as deepy_controller
from shared.deepy.hybrid import HybridService, SharedState, service_for
from shared.utils.gallery_view import gallery_window
from shared.gradio.import_files import ImportFiles
from shared.deepy import filesystem as deepy_filesystem
from shared.deepy import cli as deepy_cli
from shared.deepy import voice as deepy_voice
from shared.deepy import gradio_ui as deepy_gradio_ui
from shared.deepy import session_store as deepy_session_store
from shared import extra_settings
from shared import config_groups as model_config_groups
from shared import resolutions as resolution_utils
import torch
import gc
import traceback
import math 
import typing
import inspect
from shared.utils import prompt_parser
from shared.prompt_enhancer import chaining as prompt_enhancer_chaining
from shared.prompt_enhancer import images as prompt_enhancer_images
from shared.prompt_enhancer import labels as prompt_enhancer_labels
from shared.prompt_enhancer.progress import EnhancementProgress
from shared.prompt_enhancer.config import PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT, PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, normalize_prompt_enhancer_speculative_decoding
import base64
import io
from PIL import Image
import zipfile
import tempfile
import atexit
import shutil
import glob
import cv2
import html
try:
    from gradio_rangeslider import RangeSlider
except ImportError:
    RangeSlider = None
import re
from transformers.utils import logging
logging.set_verbosity_error
from tqdm import tqdm
import requests
from shared.gradio.gallery import AdvancedMediaGallery, get_gradio_file_path
from shared.gradio.hierarchy_selector import HierarchySelector, build_choices_hierarchy
from shared.ffmpeg_setup import download_ffmpeg
from shared.api import apply_video_length_duration, get_api_output_options, store_api_output_artifact
from shared.utils.side_files import process_side_files
from shared.utils.plugins import PluginManager, WAN2GPApplication, SYSTEM_PLUGINS
from shared.llm_engines.nanovllm.vllm_support import resolve_lm_decoder_engine
from shared.gradio import assistant_chat, field_help, finetune_editor, gallery_files, local_file_picker, model_infos, model_output_filter, model_selector_toolbar, ui_studio
from shared.gradio.magic_mask import MagicMaskUI, video_mask_area_visible, video_mask_controls_visible, video_mask_dropdown_visible
from shared import model_dropdowns
from shared import settings_metadata
from postprocessing import audio_processors as audio_processor_api
from postprocessing import temporal_upsamplers as temporal_upsampler_api
from postprocessing import spatial_upsamplers as upsampler_api
from shared.cli_args import parse_wgp_args
from collections import defaultdict

MagicMaskUI.patch_image_editor()

# import torch._dynamo as dynamo
# dynamo.config.recompile_limit = 2000   # default is 256
# dynamo.config.accumulated_recompile_limit = 2000  # or whatever limit you want

STARTUP_LOCK_FILE = "startup.lock"
global_queue_ref = []
AUTOSAVE_FILENAME = "queue.zip"
AUTOSAVE_PATH = AUTOSAVE_FILENAME
AUTOSAVE_ERROR_FILENAME = "error_queue.zip"
AUTOSAVE_TEMPLATE_PATH = AUTOSAVE_FILENAME
CONFIG_FILENAME = "wgp_config.json"
PROMPT_VARS_MAX = 10
target_mmgp_version = "3.8.2"
WanGP_version = "13.141"
settings_version = 2.79
max_source_video_frames = 3000
prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer = None, None, None, None
image_names_list = ["image_start", "image_end", "image_refs"]
CUSTOM_SETTINGS_MAX = 5
CUSTOM_SETTINGS_PER_ROW = 2
CUSTOM_SETTING_DROPDOWN_MAX = 5
CUSTOM_SETTING_TYPES = {"int", "float", "text", "dropdown"}
PHASE_2_TILING_VIDEO_PROMPT_FLAG = "~"
PHASE_2_TILING_GUIDANCE_VALUE = "2~"
lm_decoder_engine = ""
int8_kernels = "auto"
theme_text_size = Size("8.1px", "9px", "10.8px", "12.6px", "14.4px", "19.8px", "23.4px", name="wangp_text_90")
theme_spacing_size = Size("0.9px", "1.8px", "3.6px", "5.4px", "7.2px", "9px", "14.4px", name="wangp_spacing_90")
theme_radius_size = Size("0.9px", "1.8px", "3.6px", "5.4px", "7.2px", "10.8px", "19.8px", name="wangp_radius_90")
app = None
# All media attachment keys for queue save/load
ATTACHMENT_KEYS = ["image_start", "image_end", "image_refs", "image_guide", "image_mask",
                   "video_guide", "video_guide2", "video_guide3", "video_mask", "video_source", "audio_guide", "audio_guide2", "audio_guide3", "audio_source", "replace_voice_sample", "replace_voice_sample2", "custom_guide"]
PRESERVE_MEDIA_ON_SETTINGS_IMPORT = True

from importlib.metadata import version
mmgp_version = version("mmgp")
if mmgp_version != target_mmgp_version:
    print(f"Incorrect version of mmgp ({mmgp_version}), version {target_mmgp_version} is needed. Please upgrade with the command 'pip install -r requirements.txt'")
    exit()
lock = threading.Lock()
current_task_id = None
task_id = 0
unique_id = 0
unique_id_lock = threading.Lock()
offloadobj = enhancer_offloadobj = wan_model = None
loaded_config = ""
reload_needed = True
_HANDLER_MODULES = [
    "shared.qtypes.scaled_fp8",
    "shared.qtypes.nvfp4",
    "shared.qtypes.bnb_nf4",
    "shared.qtypes.nunchaku_int4",
    "shared.qtypes.nunchaku_fp4",
    "shared.qtypes.asym_w4a8_int8",
    "shared.qtypes.int8_convrot",
    "shared.qtypes.gguf",
]
quant_router.unregister_handler(".fp8_quanto_bridge")
for handler in _HANDLER_MODULES:
    quant_router.register_handler(handler)
from shared.qtypes import gguf as gguf_handler
quant_router.register_file_extension("gguf", gguf_handler)
from shared.kernels import int8_backend, kernel_policy


def apply_int8_kernel_setting(selection: str, notify_disabled=False, resolved=None) -> bool:
    global int8_kernels
    enabled = int8_backend.configure(selection, verbose_level, resolved=resolved)
    int8_kernels = selection
    return enabled


def set_wgp_global(variable_name: str, new_value: any) -> str:
    if variable_name not in globals():
        error_msg = f"Plugin tried to modify a non-existent global: '{variable_name}'."
        print(f"ERROR: {error_msg}")
        gr.Warning(error_msg)
        return f"Error: Global variable '{variable_name}' does not exist."

    try:
        globals()[variable_name] = new_value
    except Exception as e:
        error_msg = f"Error while setting global '{variable_name}': {e}"
        print(f"ERROR: {error_msg}")
        return error_msg

def clear_gen_cache():
    if "_cache" in offload.shared_state:
        del offload.shared_state["_cache"]



def release_model():
    global wan_model, offloadobj, reload_needed
    wan_model = None
    clear_gen_cache()
    if "_cache" in offload.shared_state:
        del offload.shared_state["_cache"]
    if offloadobj is not None:
        offloadobj.release()
        offloadobj = None
    offload.flush_torch_caches()
    gc.collect()
    torch.cuda.empty_cache()
    reload_needed = True
def get_unique_id():
    global unique_id  
    with unique_id_lock:
        unique_id += 1
    return str(time.time()+unique_id)

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    elif seconds >= 60:
        return f"{minutes}m {secs:02d}s"
    else:
        return f"{seconds:.1f}s"

def format_generation_time(seconds):
    """Format generation time showing raw seconds with human-readable time in parentheses when over 60s"""
    raw_seconds = f"{int(seconds)}s"
    
    if seconds < 60:
        return raw_seconds
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        human_readable = f"{hours}h {minutes}m {secs}s"
    else:
        human_readable = f"{minutes}m {secs}s"
    
    return f"{raw_seconds} ({human_readable})"

def pil_to_base64_uri(pil_image, format="png", quality=75):
    if pil_image is None:
        return None

    if isinstance(pil_image, str):
        virtual_image = get_virtual_image(pil_image)
        if virtual_image is not None:
            pil_image = virtual_image
        # Check file type and load appropriately
        elif has_video_file_extension(pil_image):
            from shared.utils.utils import get_video_frame
            pil_image = get_video_frame(pil_image, 0)
        elif has_image_file_extension(pil_image):
            pil_image = Image.open(pil_image)
        else:
            # Audio or unknown file type - can't convert to image
            return None

    buffer = io.BytesIO()
    try:
        img_to_save = pil_image
        if format.lower() == 'jpeg' and pil_image.mode == 'RGBA':
            img_to_save = pil_image.convert('RGB')
        elif format.lower() == 'png' and pil_image.mode not in ['RGB', 'RGBA', 'L', 'P']:
             img_to_save = pil_image.convert('RGBA')
        elif pil_image.mode == 'P':
             img_to_save = pil_image.convert('RGBA' if 'transparency' in pil_image.info else 'RGB')
        if format.lower() == 'jpeg':
            img_to_save.save(buffer, format=format, quality=quality)
        else:
            img_to_save.save(buffer, format=format)
        img_bytes = buffer.getvalue()
        encoded_string = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/{format.lower()};base64,{encoded_string}"
    except Exception as e:
        print(f"Error converting PIL to base64: {e}")
        return None


def _open_image_input(image):
    if not isinstance(image, str):
        return image
    virtual_image = get_virtual_image(image)
    return virtual_image if virtual_image is not None else Image.open(strip_virtual_media_suffix(image))

def is_integer(n):
    try:
        float(n)
    except ValueError:
        return False
    else:
        return float(n).is_integer()

def get_state_model_type(state):
    key= "model_type" if state.get("active_form", "add") == "add" else "edit_model_type"
    return state[key]

def compute_sliding_window_no(current_video_length, sliding_window_size, discard_last_frames, reuse_frames):
    left_after_first_window = current_video_length - sliding_window_size + discard_last_frames
    return 1 + math.ceil(left_after_first_window / (sliding_window_size - discard_last_frames - reuse_frames))

def estimate_first_window_overlap_frames(image_start, video_source, keep_frames_video_source, target_fps):
    if image_start is not None:
        return 1
    if video_source is None:
        return 0
    try:
        source_fps, _, _, source_frames = get_video_info(video_source)
        source_frames = int(source_frames / source_fps * target_fps) if source_fps else int(source_frames)
    except Exception:
        return max_source_video_frames
    try:
        keep_frames = max_source_video_frames if len(str(keep_frames_video_source or "")) == 0 else int(keep_frames_video_source)
    except Exception:
        return source_frames
    source_frames = max(0, source_frames + keep_frames) if keep_frames < 0 else min(source_frames, keep_frames)
    return source_frames

def clean_image_list(gradio_list):
    if not isinstance(gradio_list, list): gradio_list = [gradio_list]
    gradio_list = [ tup[0] if isinstance(tup, tuple) else tup for tup in gradio_list ]        

    if any( not isinstance(image, (Image.Image, str))  for image in gradio_list): return None
    if any( isinstance(image, str) and not has_image_file_extension(image) for image in gradio_list): return None
    gradio_list = [convert_image(_open_image_input(img)) for img in gradio_list]
    return gradio_list


def silent_cancel_edit(state):
    gen = get_gen_info(state)
    state["editing_task_id"] = None
    if gen.get("queue_paused_for_edit"):
        gen["queue_paused_for_edit"] = False
    return gr.Tabs(selected="media_gen"), None, gr.update(visible=False)

def cancel_edit(state):
    gen = get_gen_info(state)
    state["editing_task_id"] = None
    if gen.get("queue_paused_for_edit"):
        gen["queue_paused_for_edit"] = False
        gr.Info("Edit cancelled. Resuming queue processing.")
    else:
        gr.Info("Edit cancelled.")
    return gr.Tabs(selected="media_gen"), gr.update(visible=False)

def validate_edit(state):
    state["validate_edit_success"] = 0
    model_type = get_state_model_type(state)

    inputs = state.get("edit_state", None)
    if inputs is None: 
        return
    override_inputs, prompts, image_start, image_end, _validation_error = validate_settings(state, model_type, True, inputs)
    if override_inputs is None: 
        return
    inputs.update(override_inputs) 
    state["edit_state"] = inputs
    state["validate_edit_success"] = 1

def edit_task_in_queue( state ):
    gen = get_gen_info(state)
    queue = gen.get("queue", [])

    editing_task_id = state.get("editing_task_id", None)

    new_inputs = state.pop("edit_state", None)

    if editing_task_id is None or new_inputs is None:
        gr.Warning("No task selected for editing.")
        return None, gr.Tabs(selected="media_gen"), gr.update(visible=False), gr.update()

    if state.get("validate_edit_success", 0) == 0:
        return None, gr.update(), gr.update(), gr.update()


    task_to_edit_index = -1
    with lock:
        task_to_edit_index = next((i for i, task in enumerate(queue) if task['id'] == editing_task_id), -1)

    if task_to_edit_index == -1:
        gr.Warning("Task not found in queue. It might have been processed or deleted.")
        state["editing_task_id"] = None
        gen["queue_paused_for_edit"] = False
        return None, gr.Tabs(selected="media_gen"), gr.update(visible=False), gr.update()
    model_type = get_state_model_type(state)
    new_inputs["model_type"] = model_type 
    new_inputs["state"] = state 
    new_inputs["model_filename"] = get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)

    task_to_edit = queue[task_to_edit_index]
            
    task_to_edit['params'] = new_inputs
    task_to_edit['prompt'] = new_inputs.get('prompt')
    task_to_edit['length'] = new_inputs.get('video_length')
    task_to_edit['steps'] = new_inputs.get('num_inference_steps')
    update_task_thumbnails(task_to_edit, task_to_edit['params'])
    
    gr.Info(f"Task ID {task_to_edit['id']} has been updated successfully.")

    state["editing_task_id"] = None
    if gen.get("queue_paused_for_edit"):
        gr.Info("Resuming queue processing.")
        gen["queue_paused_for_edit"] = False

    return task_to_edit_index -1, gr.Tabs(selected="media_gen"), gr.update(visible=False), update_queue_data(queue)

def remember_spatial_upsampler_parameters(state, parameter_values):
    state["spatial_upsampler_parameters"] = dict(parameter_values or {})


def process_prompt_and_add_tasks(state, current_gallery_tab, model_choice):
    def ret():
        return gr.update(), gr.update()

    gen = get_gen_info(state)

    current_gallery_tab
    gen["last_was_audio"] = current_gallery_tab == 1

    if state.get("validate_success",0) != 1:
        ret()
    
    state["validate_success"] = 0
    model_type = get_state_model_type(state)
    inputs = get_model_settings(state, model_type)

    if model_choice != model_type or inputs ==None:
        raise gr.Error("Webform can not be used as the App has been restarted since the form was displayed. Please refresh the page")
    
    inputs["state"] =  state
    inputs["model_type"] = model_type
    inputs["spatial_upsampler_parameters"] = dict(state.get("spatial_upsampler_parameters", {}))
    inputs.pop("lset_name", None)
    if inputs == None:
        gr.Warning("Internal state error: Could not retrieve inputs for the model.")
        queue = gen.get("queue", [])
        return ret()
    if "mode" not in inputs:
        pass
    mode = inputs["mode"]
    if mode == "edit_audio":
        edit_audio_source = gen.get("edit_audio_source", None)
        edit_overrides = gen.get("edit_overrides", None)
        if edit_audio_source is None or edit_overrides is None:
            gr.Info("You must select an Audio file")
            return ret()
        for prop in ["state", "model_type", "mode"]:
            edit_overrides[prop] = inputs[prop]
        for k,v in inputs.items():
            inputs[k] = None
        inputs.update(edit_overrides)
        del gen["edit_audio_source"], gen["edit_overrides"]
        inputs["audio_source"] = edit_audio_source
        postprocess_audio = inputs.get("postprocess_audio", "") or ""
        postprocess_audio_meta = audio_processor_api.method_metadata(postprocess_audio)
        validation_error = ""
        if not media_source_exists(edit_audio_source):
            validation_error = "Selected audio file is missing"
        elif not has_audio_file_extension(edit_audio_source):
            validation_error = "Post processing is only available with Audio files"
        elif postprocess_audio:
            validation_error = audio_processor_api.validate_method(
                postprocess_audio,
                audio_processor_api.AUDIO_PROCESSOR_TYPE_AUDIO_EDIT,
                voice_sample=inputs.get("replace_voice_sample"),
                voice_sample2=inputs.get("replace_voice_sample2"),
            )
        else:
            validation_error = "You must choose at least one Audio Post Processing Method"
        if validation_error:
            gr.Info(validation_error)
            return ret()
        prompt = [audio_processor_api.format_method_label(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING)]
        inputs["repeat_generation"] = 1
        inputs["prompt"] = ", ".join(prompt)
        add_video_task(**inputs)
        new_prompts_count = gen["prompts_max"] = 1 + gen.get("prompts_max",0)
        state["validate_success"] = 1
        queue= gen.get("queue", [])
        update_global_queue_ref(queue)
        return update_queue_data(queue), gr.update(open=True) if new_prompts_count > 1 else gr.update()
    if mode.startswith("edit_"):
        edit_media_source =gen.get("edit_media_source", None)
        edit_overrides =gen.get("edit_overrides", None)
        if edit_media_source is None or edit_overrides is None:
            gr.Info("You must select a Video or Image file")
            return ret()
        frames_count = 1 if has_image_file_extension(edit_media_source) else get_video_info(edit_media_source)[3]
        if frames_count > max_source_video_frames:
            gr.Info(f"Post processing is not supported on videos longer than {max_source_video_frames} frames. Output Video will be truncated")
            # return
        for prop in ["state", "model_type", "mode"]:
            edit_overrides[prop] = inputs[prop]
        for k,v in inputs.items():
            inputs[k] = None    
        inputs.update(edit_overrides)
        del gen["edit_media_source"], gen["edit_overrides"]
        inputs["video_source"]= edit_media_source
        prompt = []

        repeat_generation = 1
        if mode == "edit_postprocessing":
            video_source = inputs["video_source"]
            source_is_image = has_image_file_extension(video_source)
            source_is_video = has_video_file_extension(video_source)
            temporal_upsampling = temporal_upsampler_api.normalize_temporal_upsampling_value(inputs.get("temporal_upsampling", ""))
            spatial_upsampling = upsampler_api.normalize_upsampling_value(inputs.get("spatial_upsampling", ""))
            inputs["temporal_upsampling"] = temporal_upsampling
            inputs["spatial_upsampling"] = spatial_upsampling
            validation_error = ""
            if not media_source_exists(video_source):
                validation_error = "Selected video or image file is missing"
            elif not (source_is_video or source_is_image):
                validation_error = "Post processing is only available with Videos or Images"
            else:
                validation_error = temporal_upsampler_api.validate_temporal_upsampling(temporal_upsampling, source_is_image=source_is_image)
            if not validation_error:
                validation_error = upsampler_api.validate_postprocessing_spatial_upsampling(spatial_upsampling, 1 if source_is_image else 0)
            if not validation_error:
                from postprocessing.film_grain import is_film_grain_enabled
                if len(temporal_upsampling) == 0 and len(spatial_upsampling) == 0 and not is_film_grain_enabled(inputs.get("film_grain_intensity", 0)):
                    validation_error = "You must choose at least one Post Processing Method"
            if validation_error:
                gr.Info(validation_error)
                return ret()
            if len(spatial_upsampling) > 0: prompt += [upsampler_api.format_upsampling_label(spatial_upsampling)]
            if len(temporal_upsampling) >0: prompt += ["Temporal Upsampling"]
            film_grain_intensity  = inputs.get("film_grain_intensity",0)
            film_grain_saturation  = inputs.get("film_grain_saturation",0.5)        
            # if film_grain_intensity >0: prompt += [f"Film Grain: intensity={film_grain_intensity}, saturation={film_grain_saturation}"]
            if film_grain_intensity >0: prompt += ["Film Grain"]
        elif mode =="edit_remux":
            postprocess_audio = inputs.get("postprocess_audio", "") or ""
            postprocess_audio_meta = audio_processor_api.method_metadata(postprocess_audio)
            video_source = inputs["video_source"]
            validation_error = ""
            if not media_source_exists(video_source):
                validation_error = "Selected video or image file is missing"
            elif not has_video_file_extension(video_source):
                validation_error = "Audio remuxing is only available with Videos"
            elif postprocess_audio and audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"]:
                validation_error = audio_processor_api.validate_method(
                    postprocess_audio,
                    audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK,
                    video_source=video_source,
                    audio_source=inputs.get("audio_source"),
                    media_source_exists=media_source_exists,
                    has_audio_file_extension=has_audio_file_extension,
                )
            elif postprocess_audio and audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT in postprocess_audio_meta["types"]:
                validation_error = audio_processor_api.validate_method(
                    postprocess_audio,
                    audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT,
                    voice_sample=inputs.get("replace_voice_sample"),
                    voice_sample2=inputs.get("replace_voice_sample2"),
                )
            else:
                validation_error = "You must choose at least one Remux Method"
            if validation_error:
                gr.Info(validation_error)
                return ret()
            repeat_generation= inputs.get("repeat_generation",1)
            audio_source = inputs["audio_source"]
            if audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"]:
                prompt += [audio_processor_api.format_method_label(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING)]
                if not postprocess_audio_meta["needs_audio_source"]:
                    audio_source = None
                    inputs["audio_source"] = audio_source
                if not postprocess_audio_meta["supports_repeat"]:
                    repeat_generation = 1
            elif audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT in postprocess_audio_meta["types"]:
                prompt += [audio_processor_api.format_method_label(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING)]
                audio_source = None
                inputs["audio_source"] = audio_source
                repeat_generation = 1
            else:
                gr.Info(f"Unsupported remux method: {postprocess_audio}")
                return ret()
            seed = inputs.get("seed",None)
        inputs["repeat_generation"] = repeat_generation
        if len(prompt) == 0:
            gr.Info("No edit action selected")
            return ret()
        inputs["prompt"] = ", ".join(prompt)
        add_video_task(**inputs)
        new_prompts_count = gen["prompts_max"] = 1 + gen.get("prompts_max",0)
        state["validate_success"] = 1
        queue= gen.get("queue", [])
        return update_queue_data(queue), gr.update(open=True) if new_prompts_count > 1 else gr.update()

    inputs, prompts, image_start, image_end, _validation_error = validate_settings(state, model_type, False, inputs)

    if inputs is None:
        return ret()

    multi_prompts_gen_type = inputs["multi_prompts_gen_type"]
    prompt_history = prompt_parser.parse_prompt_history(inputs["prompt"], "", multi_prompts_gen_type)
    queued_prompts = prompts.copy()
    if prompt_history is not None:
        original_prompts = prompt_enhancer_chaining.originals_for_outputs(prompt_history[0], len(prompts))
        queued_prompts = [prompt_parser.serialize_prompt_blocks_with_prefix([one_prompt], [original_prompts[idx]]) for idx, one_prompt in enumerate(prompts)]
    model_def = get_model_def(model_type)
    alt_prompt_text = str(inputs.get("alt_prompt") or "")
    alt_prompt_has_history = alt_prompt_text.startswith(prompt_parser.PROMPT_UNIT_PREFIX)
    auto_dependent_alt_prompt = server_config.get("enhancer_enabled", 0) > 0 and server_config.get("enhancer_mode", 1) == 0 and prompt_enhancer_chaining.requires_alt_prompt_inheritance(inputs.get("prompt_enhancer", ""))
    queued_alt_prompts = [alt_prompt_text] * len(prompts)
    if model_def.get("alt_prompt_inherits_prompt_paragraphs", False) and (alt_prompt_has_history or auto_dependent_alt_prompt):
        alt_prompt_history = prompt_parser.parse_prompt_history(alt_prompt_text, "", multi_prompts_gen_type)
        alt_prompts = alt_prompt_history[1] if alt_prompt_history is not None else (prompt_parser.split_prompt_units(alt_prompt_text, multi_prompts_gen_type) or [""])
        alt_prompts = prompt_enhancer_chaining.repeat_evenly(alt_prompts, len(prompts))
        if alt_prompt_history is None:
            queued_alt_prompts = alt_prompts
        else:
            original_alt_prompts = prompt_enhancer_chaining.repeat_evenly(alt_prompt_history[0], len(prompts))
            queued_alt_prompts = [prompt_parser.serialize_prompt_blocks_with_prefix([one_prompt], [original_alt_prompts[idx]]) for idx, one_prompt in enumerate(alt_prompts)]

    if "W" not in multi_prompts_gen_type:
        image_slots = image_start if image_start != None and len(image_start) > 0 else image_end
        if image_slots != None and len(image_slots) > 0:
            if inputs["multi_images_gen_type"] == 0:
                new_prompts = []
                new_queued_prompts = []
                new_queued_alt_prompts = []
                new_image_start = []
                new_image_end = []
                for i in range(len(prompts) * len(image_slots)):
                    new_prompts.append(  prompts[ i % len(prompts)] )
                    new_queued_prompts.append(queued_prompts[i % len(queued_prompts)])
                    new_queued_alt_prompts.append(queued_alt_prompts[i % len(queued_alt_prompts)])
                    if image_start != None:
                        new_image_start.append(image_start[i // len(prompts)] )
                    if image_end != None:
                        new_image_end.append(image_end[i // len(prompts)] )
                prompts = new_prompts
                queued_prompts = new_queued_prompts
                queued_alt_prompts = new_queued_alt_prompts
                image_start = new_image_start if image_start != None else None
                image_end = new_image_end if image_end != None else None
            else:
                if len(prompts) >= len(image_slots):
                    if len(prompts) % len(image_slots) != 0:
                        gr.Info("If there are more text prompts than input images the number of text prompts should be dividable by the number of images")
                        return ret()
                    rep = len(prompts) // len(image_slots)
                    new_image_start = []
                    new_image_end = []
                    for i, _ in enumerate(prompts):
                        if image_start != None:
                            new_image_start.append(image_start[i//rep] )
                        if image_end != None:
                            new_image_end.append(image_end[i//rep] )
                    image_start = new_image_start if image_start != None else None
                    image_end = new_image_end if image_end != None else None
                else: 
                    if len(image_slots) % len(prompts)  !=0:
                        gr.Info("If there are more input images than text prompts the number of images should be dividable by the number of text prompts")
                        return ret()
                    rep = len(image_slots) // len(prompts)
                    new_prompts = []
                    new_queued_prompts = []
                    new_queued_alt_prompts = []
                    for i, _ in enumerate(image_slots):
                        new_prompts.append(  prompts[ i//rep] )
                        new_queued_prompts.append(queued_prompts[i//rep])
                        new_queued_alt_prompts.append(queued_alt_prompts[i//rep])
                    prompts = new_prompts
                    queued_prompts = new_queued_prompts
                    queued_alt_prompts = new_queued_alt_prompts
            if image_start == None or len(image_start) == 0:
                image_start = [None] * len(prompts)
            if image_end == None or len(image_end) == 0:
                image_end = [None] * len(prompts)

            for queued_prompt, queued_alt_prompt, start, end in zip(queued_prompts, queued_alt_prompts, image_start, image_end) :
                inputs.update({
                    "prompt" : queued_prompt,
                    "alt_prompt": queued_alt_prompt,
                    "image_start": start,
                    "image_end" : end,
                })
                add_video_task(**inputs)
        else:
            for queued_prompt, queued_alt_prompt in zip(queued_prompts, queued_alt_prompts):
                inputs["prompt"] = queued_prompt
                inputs["alt_prompt"] = queued_alt_prompt
                add_video_task(**inputs)
        new_prompts_count = len(prompts)
    else:
        new_prompts_count = 1
        add_video_task(**inputs)
    new_prompts_count += gen.get("prompts_max",0)
    gen["prompts_max"] = new_prompts_count
    state["validate_success"] = 1
    queue= gen.get("queue", [])
    return update_queue_data(queue), gr.update(open=True) if new_prompts_count > 1 else gr.update()

def get_custom_setting_key(index):
    return f"custom_setting_{index + 1}"

def get_custom_setting_slider_key(index):
    return f"custom_setting_slider_{index + 1}"

def get_custom_setting_dropdown_key(index):
    return f"custom_setting_dropdown_{index + 1}"

def _normalize_custom_setting_type(setting_type):
    parsed_type = str(setting_type or "text").strip().lower()
    return parsed_type if parsed_type in CUSTOM_SETTING_TYPES else "text"

def _normalize_custom_setting_name(name):
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(name or "").strip().lower()).strip("_")
    return normalized

def get_custom_setting_id(setting_def, setting_index):
    explicit_id = setting_def.get("id", None)
    if explicit_id is not None and len(str(explicit_id).strip()) > 0:
        normalized_id = _normalize_custom_setting_name(explicit_id)
        if len(normalized_id) > 0:
            return normalized_id
    for field_name in ("name", "param"):
        normalized_name = _normalize_custom_setting_name(setting_def.get(field_name, ""))
        if len(normalized_name) > 0:
            return normalized_name
    return get_custom_setting_key(setting_index)

def get_custom_setting_dropdown_choices(setting_def):
    if not isinstance(setting_def, dict) or setting_def.get("type") != "dropdown":
        return None
    choices = setting_def.get("choices", [])
    if not isinstance(choices, list):
        return None
    normalized = []
    for choice in choices:
        if isinstance(choice, (list, tuple)) and len(choice) >= 2:
            normalized.append((str(choice[0]), choice[1]))
        else:
            normalized.append((str(choice), choice))
    return normalized if len(normalized) > 0 else None

def get_custom_setting_dropdown_value(raw_value, choices):
    if not choices:
        return raw_value
    values = [value for _, value in choices]
    for value in values:
        if raw_value == value or str(raw_value) == str(value):
            return value
    return values[0]

def get_custom_setting_display_value(setting_def, raw_value):
    choices = get_custom_setting_dropdown_choices(setting_def)
    if choices is not None:
        for label, value in choices:
            if raw_value == value or str(raw_value) == str(value):
                return label
    return raw_value

def get_model_custom_settings(model_def):
    if not isinstance(model_def, dict):
        return []
    custom_settings = model_def.get("custom_settings", [])
    if not isinstance(custom_settings, list):
        return []
    normalized = []
    used_ids = set()
    dropdown_count = 0
    for idx, setting in enumerate(custom_settings[:CUSTOM_SETTINGS_MAX]):
        if not isinstance(setting, dict):
            continue
        one = setting.copy()
        one["label"] = str(one.get("label", f"Custom Setting {idx + 1}"))
        one["name"] = str(one.get("name", f"Custom Setting {idx + 1}"))
        one["type"] = _normalize_custom_setting_type(one.get("type", "text"))
        if one["type"] == "dropdown":
            dropdown_count += 1
            if dropdown_count > CUSTOM_SETTING_DROPDOWN_MAX or get_custom_setting_dropdown_choices(one) is None:
                one["type"] = "text"
        setting_id = get_custom_setting_id(one, idx)
        if setting_id in used_ids:
            setting_id = get_custom_setting_key(idx)
        used_ids.add(setting_id)
        one["id"] = setting_id
        normalized.append(one)
    return normalized

def apply_custom_settings_defaults(model_def, ui_defaults):
    custom_settings = ui_defaults.get("custom_settings")
    for setting_def in get_model_custom_settings(model_def):
        if "default" not in setting_def:
            continue
        if not isinstance(custom_settings, dict):
            custom_settings = {}
            ui_defaults["custom_settings"] = custom_settings
        custom_settings.setdefault(setting_def["id"], copy.deepcopy(setting_def["default"]))

def get_custom_setting_slider_bounds(setting_def):
    if not isinstance(setting_def, dict) or setting_def.get("type") not in {"int", "float"} or not all(key in setting_def for key in ("min", "max", "inc")):
        return None
    try:
        min_value, max_value, step_value = float(setting_def["min"]), float(setting_def["max"]), float(setting_def["inc"])
    except Exception:
        return None
    if max_value < min_value or step_value <= 0:
        return None
    if setting_def.get("type") == "int":
        if not min_value.is_integer() or not max_value.is_integer() or not step_value.is_integer():
            return None
        return int(min_value), int(max_value), int(step_value)
    return min_value, max_value, step_value

def get_custom_setting_slider_value(raw_value, slider_bounds):
    min_value, max_value, _ = slider_bounds
    try:
        value = float(raw_value)
    except Exception:
        value = min_value
    return min(max(value, min_value), max_value)


def end_frames_always_enabled(model_def):
    return bool((model_def or {}).get("end_frames_always_enabled", False))


def end_frames_option_visible(model_def, image_prompt_type):
    if "E" not in (model_def or {}).get("image_prompt_types_allowed", ""):
        return False
    return end_frames_always_enabled(model_def) or any_letters(image_prompt_type or "", "SVL")


def injected_frames_positions_visible(video_prompt_type):
    return "F" in (video_prompt_type or "")


def input_video_strength_visible(model_def, image_prompt_type, video_prompt_type=""):
    input_video_strength = model_def.get("input_video_strength", {})
    input_video_strength_label = input_video_strength.get("label", "").strip()
    return len(input_video_strength_label) > 0 and (
        any_letters(image_prompt_type or "", "SVLE") or injected_frames_positions_visible(video_prompt_type)
    )


def custom_setting_visible(setting_def, video_prompt_type="", audio_prompt_type=""):
    if not isinstance(setting_def, dict): return False
    vflags, aflags = str(setting_def.get("video_prompt_type", "") or ""), str(setting_def.get("audio_prompt_type", "") or "")
    vflags_not, aflags_not = str(setting_def.get("video_prompt_type_not", "") or ""), str(setting_def.get("audio_prompt_type_not", "") or "")
    if vflags_not and any_letters(video_prompt_type or "", vflags_not): return False
    if aflags_not and any_letters(audio_prompt_type or "", aflags_not): return False
    if not (vflags or aflags): return True
    return bool(vflags and any_letters(video_prompt_type or "", vflags)) or bool(aflags and any_letters(audio_prompt_type or "", aflags))


def custom_settings_visibility_trigger_update(state, old_video=None, new_video=None, old_audio=None, new_audio=None):
    settings = [s for s in get_model_custom_settings(get_model_def(get_state_model_type(state))) if isinstance(s, dict) and ("video_prompt_type" in s or "audio_prompt_type" in s or "video_prompt_type_not" in s or "audio_prompt_type_not" in s)]
    if not settings: return gr.update()
    old_video, new_video = (new_video if old_video is None else old_video), (old_video if new_video is None else new_video)
    old_audio, new_audio = (new_audio if old_audio is None else old_audio), (old_audio if new_audio is None else new_audio)
    ids = [s["id"] for s in settings if custom_setting_visible(s, old_video, old_audio) != custom_setting_visible(s, new_video, new_audio)]
    return gr.update() if not ids else f"{time.time()}:{','.join(ids)}"


def refresh_custom_settings_visibility(state, video_prompt_type, audio_prompt_type):
    custom_settings = get_model_custom_settings(get_model_def(get_state_model_type(state)))
    defs = [custom_settings[i] if i < len(custom_settings) else None for i in range(CUSTOM_SETTINGS_MAX)]
    visible = [custom_setting_visible(setting, video_prompt_type, audio_prompt_type) for setting in defs]
    slider_settings = [get_custom_setting_slider_bounds(setting) is not None for setting in defs]
    dropdown_settings = [get_custom_setting_dropdown_choices(setting) is not None for setting in defs]
    text_settings = [setting is not None and not slider_settings[idx] and not dropdown_settings[idx] for idx, setting in enumerate(defs)]
    row_updates = [gr.update(visible=any(visible[row_idx * CUSTOM_SETTINGS_PER_ROW:min((row_idx + 1) * CUSTOM_SETTINGS_PER_ROW, CUSTOM_SETTINGS_MAX)])) for row_idx in range(math.ceil(CUSTOM_SETTINGS_MAX / CUSTOM_SETTINGS_PER_ROW))]
    return row_updates + [gr.update(visible=visible[idx] and text_settings[idx]) for idx in range(CUSTOM_SETTINGS_MAX)] + [gr.update(visible=visible[idx] and slider_settings[idx]) for idx in range(CUSTOM_SETTINGS_MAX)] + [gr.update(visible=visible[idx] and dropdown_settings[idx]) for idx in range(CUSTOM_SETTINGS_MAX)]


def parse_custom_setting_typed_value(raw_value, setting_type, setting_def=None):
    if raw_value is None:
        return None, None
    if isinstance(raw_value, str):
        raw_value = raw_value.strip()
        if len(raw_value) == 0:
            return None, None
    setting_type = _normalize_custom_setting_type(setting_type)
    if setting_type == "dropdown":
        choices = get_custom_setting_dropdown_choices(setting_def)
        if choices is None:
            return str(raw_value).strip(), None
        for _, value in choices:
            if raw_value == value or str(raw_value) == str(value):
                return value, None
        return None, "Expected one of the dropdown choices."
    if setting_type == "int":
        if isinstance(raw_value, bool):
            return None, "Expected an integer value."
        if isinstance(raw_value, int):
            return raw_value, None
        if isinstance(raw_value, float):
            if raw_value.is_integer():
                return int(raw_value), None
            return None, "Expected an integer value."
        try:
            return int(str(raw_value).strip()), None
        except Exception:
            try:
                float_value = float(str(raw_value).strip())
                if float_value.is_integer():
                    return int(float_value), None
            except Exception:
                pass
            return None, "Expected an integer value."
    if setting_type == "float":
        if isinstance(raw_value, bool):
            return None, "Expected a float value."
        try:
            return float(raw_value), None
        except Exception:
            return None, "Expected a float value."
    return str(raw_value).strip(), None

def get_custom_setting_value_from_dict(custom_settings_values, setting_def, setting_index):
    setting_id = setting_def.get("id", get_custom_setting_id(setting_def, setting_index))
    if isinstance(custom_settings_values, dict) and setting_id in custom_settings_values:
        return custom_settings_values.get(setting_id, None)
    return setting_def.get("default", "")

def collect_custom_settings_from_inputs(model_def, inputs, strict=False):
    custom_settings_dict = {}
    existing_custom_settings = inputs.get("custom_settings", None)
    if not isinstance(existing_custom_settings, dict):
        existing_custom_settings = {}
    custom_settings = get_model_custom_settings(model_def)
    for idx, setting_def in enumerate(custom_settings):
        slot_key = get_custom_setting_key(idx)
        slider_key = get_custom_setting_slider_key(idx)
        dropdown_key = get_custom_setting_dropdown_key(idx)
        setting_id = setting_def["id"]
        if get_custom_setting_dropdown_choices(setting_def) is not None:
            raw_value = inputs.get(dropdown_key, None)
        elif get_custom_setting_slider_bounds(setting_def) is not None:
            raw_value = inputs.get(slider_key, None)
        else:
            raw_value = inputs.get(slot_key, None)
        if raw_value is None and setting_id in existing_custom_settings:
            raw_value = existing_custom_settings.get(setting_id, None)
        parsed_value, parse_error = parse_custom_setting_typed_value(raw_value, setting_def.get("type", "text"), setting_def)
        if parse_error is not None:
            if strict:
                return None, f"{setting_def.get('label', slot_key)} {parse_error}"
            if raw_value is not None:
                raw_text = str(raw_value).strip() if isinstance(raw_value, str) else raw_value
                if not (isinstance(raw_text, str) and len(raw_text) == 0):
                    custom_settings_dict[setting_id] = raw_text
            continue
        if parsed_value is not None:
            custom_settings_dict[setting_id] = parsed_value
    return custom_settings_dict if len(custom_settings_dict) > 0 else None, None

def clear_custom_setting_slots(inputs):
    for idx in range(CUSTOM_SETTINGS_MAX):
        inputs.pop(get_custom_setting_key(idx), None)
        inputs.pop(get_custom_setting_slider_key(idx), None)
        inputs.pop(get_custom_setting_dropdown_key(idx), None)

def validate_settings(state, model_type, single_prompt, inputs, silent=False):
    def err(error=""):
        error = str(error or "")
        if len(error) > 0 and not silent:
            gr.Info(error)
        return None, None, None, None, error

    unknown_error = unknown_settings_error(inputs)
    if unknown_error:
        return err(unknown_error)
    model_def = get_model_def(model_type)
    model_handler = get_model_handler(model_type)
    image_outputs = inputs["image_mode"] > 0
    if image_outputs:
        image_batch_size_max = max(1, int(model_def.get("image_batch_size_max", 16)))
        if int(inputs.get("batch_size", 1) or 1) > image_batch_size_max:
            return err(f"This model supports a maximum of {image_batch_size_max} image{'s' if image_batch_size_max > 1 else ''} per generation.")
    is_edit_mode = str(inputs.get("mode", "") or "").startswith("edit_")
    any_steps_skipping = model_def.get("tea_cache", False) or model_def.get("mag_cache", False) or model_def.get("spectrum_cache", False) or model_def.get("first_block_cache", False)
    model_type = get_base_model_type(model_type)

    model_filename = get_model_filename(model_type)  


    if inputs.get("cfg_star_switch", 0) != 0 and inputs.get("apg_switch", 0) != 0:
        return err("Adaptive Progressive Guidance and Classifier Free Guidance Star can not be set at the same time")
    multi_prompts_gen_type = inputs["multi_prompts_gen_type"]
    prompt = inputs["prompt"]
    keep_empty_lines = model_def.get("preserve_empty_prompt_lines", False) or "P" in multi_prompts_gen_type or prompt_parser.PROMPT_UNIT_PREFIX in prompt
    if not model_def.get("skip_prompt_template", False):
        prompt, errors = prompt_parser.process_template(prompt, keep_comments=prompt_parser.PROMPT_UNIT_PREFIX in prompt, keep_empty_lines=keep_empty_lines)
        if len(errors) > 0:
            return err("Error processing prompt template: " + errors)
    prompt = prompt.strip("\n").strip()

    prompts = prompt_parser.split_prompt_units(prompt, multi_prompts_gen_type)
    if len(prompts) == 0:
        return err("Prompt cannot be empty.")
    if single_prompt and multi_prompts_gen_type in {"G", "PG"} and len(prompts) > 1:
        return err(f"multi_prompts_gen_type='{multi_prompts_gen_type}' parses this prompt into {len(prompts)} separate generation requests, but this submission accepts one generation task. Submit separate tasks or use 'FG' if the line breaks belong to one prompt.")
    window_boundary_error = prompt_parser.validate_sliding_window_prompt_boundaries(prompts, multi_prompts_gen_type, inputs.get("image_end"))
    if window_boundary_error:
        return err(window_boundary_error)
    validation_prompts = prompts
    frames_minimum, frames_steps, latent_size = get_model_min_frames_and_step(model_type)
    inputs.pop("frame_scheduler", None)
    frame_scheduler = None
    scheduler_supported = frame_scheduler_supported(model_type, model_def, inputs.get("image_mode", 0), is_edit_mode)
    if not scheduler_supported and has_slash_commands(prompts):
        return err("Prompt slash window commands require a video model with Sliding Window support.")
    if scheduler_supported:
        schedule_fps = get_computed_fps(inputs.get("force_fps", ""), model_type, inputs.get("video_guide"), inputs.get("video_source"))
        frame_scheduler, frame_scheduler_error = build_frame_scheduler(
            prompts,
            total_frames=int(inputs.get("video_length", frames_minimum) or frames_minimum),
            fps=float(schedule_fps),
            window_size=int(inputs.get("sliding_window_size", inputs.get("video_length", frames_minimum)) or frames_minimum),
            default_overlap=int(inputs.get("sliding_window_overlap", 0) or 0),
            minimum=frames_minimum,
            step=frames_steps,
            frame_offset=model_def.get("frames_offset", 1),
            overlap_offset=model_def.get("sliding_window_defaults", {}).get("overlap_offset", 1),
            max_overlap=model_def.get("sliding_window_defaults", {}).get("overlap_max"),
            preserve_exact_output_frames=model_def.get("image_end_frame_position", False),
            output_frame_policy=model_def.get("frame_scheduler_output_policy"),
            supported_model_commands=model_def.get("prompt_slash_commands", []),
            allow_new_shot=image_prompt_types_allow_t2v(model_def, inputs.get("image_mode", 0)),
            first_window_overlap_frames=estimate_first_window_overlap_frames(inputs.get("image_start"), inputs.get("video_source"), inputs.get("keep_frames_video_source", ""), schedule_fps),
            initial_shared_frames=1 if inputs.get("image_start") is not None or inputs.get("video_source") is not None else 0,
            discard_last_frames=int(inputs.get("sliding_window_discard_last_frames", 0) or 0) // latent_size * latent_size,
        )
        if frame_scheduler_error is not None:
            return err(frame_scheduler_error)
        validation_prompts = frame_scheduler["prompts"]
        inputs["frame_scheduler"] = frame_scheduler
    inputs["prompt"] = prompt if prompt.startswith(prompt_parser.PROMPT_UNIT_PREFIX) else prompt_parser.serialize_prompt_units(prompt, prompts, multi_prompts_gen_type)

    parsed_custom_settings, custom_settings_error = collect_custom_settings_from_inputs(model_def, inputs, strict=True)
    if custom_settings_error is not None:
        return err(custom_settings_error)
    inputs["custom_settings"] = parsed_custom_settings
    clear_custom_setting_slots(inputs)
    inputs["guidance_phases"], inputs["video_prompt_type"] = normalize_phase_2_tiling_selection(model_def, inputs["guidance_phases"], inputs["video_prompt_type"])
    extra_settings_error = extra_settings.validate_inputs(inputs, model_def, get_max_frames=get_max_frames)
    if len(extra_settings_error) > 0:
        return err(extra_settings_error)

    if hasattr(model_handler, "validate_generative_prompt"):
        for one_prompt in validation_prompts:
            error = model_handler.validate_generative_prompt(model_type, model_def, inputs, one_prompt)
            if error is not None and len(error) > 0:
                return err(error)

    resolution = inputs["resolution"]
    width, height = resolution.split("x")
    width, height = int(width), int(height)
    image_start = inputs["image_start"]
    image_end = inputs["image_end"]
    image_refs = inputs["image_refs"]
    image_prompt_type = inputs["image_prompt_type"]
    audio_prompt_type = inputs["audio_prompt_type"]
    if image_prompt_type == None: image_prompt_type = ""
    video_prompt_type = inputs["video_prompt_type"]
    if video_prompt_type == None: video_prompt_type = ""
    force_fps = inputs["force_fps"]
    audio_guide = inputs["audio_guide"]
    audio_guide2 = inputs["audio_guide2"]
    audio_guide3 = inputs["audio_guide3"]
    audio_source = inputs["audio_source"]
    replace_voice_method = audio_processor_api.normalize_method(inputs.get("replace_voice_method", "") or "")
    replace_voice_sample = inputs.get("replace_voice_sample", None)
    replace_voice_sample2 = inputs.get("replace_voice_sample2", None)
    video_guide = inputs["video_guide"]
    video_guide2 = inputs["video_guide2"]
    video_guide3 = inputs["video_guide3"]
    image_guide = inputs["image_guide"]
    video_mask = inputs["video_mask"]
    image_mask = inputs["image_mask"]
    custom_guide = inputs["custom_guide"]
    speakers_locations = inputs["speakers_locations"]
    video_source = inputs["video_source"]
    frames_positions = inputs["frames_positions"]
    keep_frames_video_guide= inputs["keep_frames_video_guide"] 
    keep_frames_video_source = inputs["keep_frames_video_source"]
    denoising_strength= inputs["denoising_strength"]     
    masking_strength= inputs["masking_strength"]     
    input_video_strength = inputs.get("input_video_strength", 1.0)
    sliding_window_size = inputs["sliding_window_size"]
    sliding_window_overlap = inputs["sliding_window_overlap"]
    sliding_window_discard_last_frames = inputs["sliding_window_discard_last_frames"]
    video_length = inputs["video_length"]
    num_inference_steps= inputs["num_inference_steps"]
    skip_steps_cache_type= inputs["skip_steps_cache_type"]
    skip_steps_multiplier = inputs["skip_steps_multiplier"]
    postprocess_audio = inputs.get("postprocess_audio", "") or ""
    image_mode = inputs["image_mode"]
    switch_threshold = inputs["switch_threshold"]
    loras_multipliers = inputs["loras_multipliers"]
    activated_loras = inputs["activated_loras"]
    guidance_phases= inputs["guidance_phases"]
    model_switch_phase = inputs["model_switch_phase"]    
    switch_threshold = inputs["switch_threshold"]
    switch_threshold2 = inputs["switch_threshold2"]
    video_guide_outpainting = inputs["video_guide_outpainting"]
    video_guide_outpainting_ratio = inputs.get("video_guide_outpainting_ratio", "")
    spatial_upsampling = upsampler_api.normalize_upsampling_value(inputs["spatial_upsampling"])
    temporal_upsampling = temporal_upsampler_api.normalize_temporal_upsampling_value(inputs.get("temporal_upsampling", ""))
    inputs["spatial_upsampling"] = spatial_upsampling
    inputs["temporal_upsampling"] = temporal_upsampling
    motion_amplitude = inputs["motion_amplitude"]
    self_refiner_setting = inputs["self_refiner_setting"]
    self_refiner_plan = inputs["self_refiner_plan"]
    model_mode = inputs["model_mode"]
    if image_mode == 0 and model_def.get("image_outputs", False): image_mode = 1
    medium = "Videos" if image_mode == 0 else "Images"
    prompt_enhancer_choices, prompt_enhancer_default, prompt_enhancer_def = get_prompt_enhancer_choices(model_def, model_def.get("audio_only", False), image_mode, include_disabled=True, video_prompt_type=inputs["video_prompt_type"], inputs=inputs)
    prompt_enhancer_values = [value for _, value in prompt_enhancer_choices]
    prompt_enhancer_mode = prompt_enhancer_chaining.normalize_choice(str(inputs.get("prompt_enhancer") or ""), prompt_enhancer_values, prompt_enhancer_default)
    inputs["prompt_enhancer"] = build_prompt_enhancer_value(prompt_enhancer_mode, "K" in str(inputs.get("prompt_enhancer") or "") and len(prompt_enhancer_mode) > 0)
    inheritance_error = prompt_enhancer_chaining.validate_alt_prompt_inheritance(prompt_enhancer_mode, len(prompts), model_def.get("alt_prompt_inherits_prompt_paragraphs", False)) if server_config.get("enhancer_enabled", 0) > 0 and server_config.get("enhancer_mode", 1) == 0 else ""
    if inheritance_error:
        return err(inheritance_error)
    alt_prompt_has_history = str(inputs.get("alt_prompt") or "").startswith(prompt_parser.PROMPT_UNIT_PREFIX)
    auto_dependent_alt_prompt = server_config.get("enhancer_enabled", 0) > 0 and server_config.get("enhancer_mode", 1) == 0 and prompt_enhancer_chaining.requires_alt_prompt_inheritance(prompt_enhancer_mode)
    if model_def.get("alt_prompt_inherits_prompt_paragraphs", False) and (alt_prompt_has_history or auto_dependent_alt_prompt):
        alt_prompts = prompt_parser.split_prompt_units(str(inputs.get("alt_prompt") or ""), multi_prompts_gen_type) or [""]
        alt_prompt_count_error = prompt_enhancer_chaining.validate_alt_prompt_count(len(prompts), len(alt_prompts), model_def["prompt_class"], model_def["alt_prompt"]["label"])
        if alt_prompt_count_error:
            return err(alt_prompt_count_error)

    if image_start is not None and not isinstance(image_start, list): image_start = [image_start]
    outpainting_modes = model_def.get("video_guide_outpainting", [])
    if image_mode not in outpainting_modes: 
        video_guide_outpainting = ""
        video_guide_outpainting_ratio = ""

    outpainting_dims = get_outpainting_dims(video_guide_outpainting, video_guide_outpainting_ratio)

    model_modes_visibility = [0,1,2]
    model_mode_choices = model_def.get("model_modes", None)
    if model_mode_choices is not None: model_modes_visibility= model_mode_choices.get("image_modes", model_modes_visibility)
    if model_mode is not None and image_mode not in model_modes_visibility:
        model_mode = None
    guide_custom_choices = get_guide_custom_choices(model_def, image_mode)
    if image_mode > 0 and guide_custom_choices is not None:
        guide_custom_value = filter_letters(video_prompt_type, guide_custom_choices["letters_filter"])
        if len(guide_custom_value) > 0 and guide_custom_value not in {value for _, value in guide_custom_choices["choices"]}:
            video_prompt_type = del_in_sequence(video_prompt_type, guide_custom_choices["letters_filter"])
            inputs["video_prompt_type"] = video_prompt_type
    if server_config.get("fit_canvas", 0) == 2 and outpainting_dims is not None and any_letters(video_prompt_type, "VKF"):
        gr.Info("Output Resolution Cropping will be not used for this Generation as it is not compatible with Video Outpainting")
    if self_refiner_setting != 0:
        from shared.utils.self_refiner import normalize_self_refiner_plan, convert_refiner_list_to_string
        if isinstance(self_refiner_plan, list):
            self_refiner_plan = convert_refiner_list_to_string(self_refiner_plan)
        max_p = model_def.get("self_refiner_max_plans", 1)
        _, error = normalize_self_refiner_plan(self_refiner_plan, max_plans=max_p)
        if len(error):
            return err(error)

    if not model_def.get("motion_amplitude", False): motion_amplitude = 1.
    if spatial_upsampling and upsampler_api.find_upsampler(spatial_upsampling) is None:
        return err(f"Spatial upsampling method '{spatial_upsampling}' is not supported by the current WanGP install")
    temporal_upsampling_error = temporal_upsampler_api.validate_temporal_upsampling(temporal_upsampling, source_is_image=image_mode > 0)
    if temporal_upsampling_error:
        return err(f"Temporal upsampling method '{temporal_upsampling}' is not supported by the current WanGP install" if temporal_upsampler_api.find_temporal_upsampler(temporal_upsampling) is None else temporal_upsampling_error)
    vae_upsampling_error = upsampler_api.validate_model_vae_upsampling(spatial_upsampling, image_mode, model_type, model_def, medium)
    if vae_upsampling_error:
        return err(vae_upsampling_error)
    edit_upsampler = upsampler_api.find_postprocessing_upsampler(spatial_upsampling)
    if edit_upsampler is not None:
        edit_upsampling_error = edit_upsampler.validate_upsampling(spatial_upsampling, image_mode)
        if edit_upsampling_error:
            return err(edit_upsampling_error)

    if len(activated_loras) > 0:
        activated_loras = update_loras_url_cache(get_lora_dir(model_type), activated_loras)
        inputs["activated_loras"] = activated_loras
        error = check_loras_exist(model_type, activated_loras)
        if len(error) > 0:
            return err(error)
    if  model_def.get("lock_guidance_phases", False):
        guidance_phases = model_def.get("guidance_max_phases", 0)
    else:
        guidance_phases = min(guidance_phases, model_def.get("guidance_max_phases", 0))
                          
    lora_multiplier_phases = int(model_def.get("lora_multiplier_phases", guidance_phases) or guidance_phases or 1)
    lora_multiplier_branches = model_def.get("lora_multiplier_branches", None)
    if len(loras_multipliers) > 0:
        _, _, errors =  parse_loras_multipliers(loras_multipliers, len(activated_loras), num_inference_steps, nb_phases= lora_multiplier_phases, lora_multiplier_branches=lora_multiplier_branches)
        if len(errors) > 0: 
            return err(f"Error parsing Loras Multipliers: {errors}")
    loras_mult_error = prepare_loras_mult_windows(frame_scheduler, activated_loras, num_inference_steps, lora_multiplier_phases, lora_multiplier_branches=lora_multiplier_branches)
    if loras_mult_error is not None:
        return err(loras_mult_error)
    if guidance_phases == 3:
        if switch_threshold < switch_threshold2:
            return err(f"Phase 1-2 Switch Noise Level ({switch_threshold}) should be Greater than Phase 2-3 Switch Noise Level ({switch_threshold2}). As a reminder, noise will gradually go down from 1000 to 0.")
    else:
        model_switch_phase = 1
        
    if not any_steps_skipping: skip_steps_cache_type = ""
    supported_cache_types = {""}
    if model_def.get("tea_cache", False): supported_cache_types.add("tea")
    if model_def.get("mag_cache", False): supported_cache_types.add("mag")
    if model_def.get("spectrum_cache", False): supported_cache_types.add("spectrum")
    if model_def.get("first_block_cache", False): supported_cache_types.add("first_block")
    if skip_steps_cache_type not in supported_cache_types:
        return err(f"This model does not support step-skipping type '{skip_steps_cache_type}'.")
    if skip_steps_cache_type == "first_block" and float(skip_steps_multiplier) not in model_def["first_block_cache_thresholds"]:
        return err(f"Unsupported First Block Cache threshold '{skip_steps_multiplier}'.")
    if not model_def.get("lock_inference_steps", False) and model_type in ["ltxv_13B"] and num_inference_steps < 20:
        return err("The minimum number of steps should be 20")
    if skip_steps_cache_type == "mag":
        if num_inference_steps > 50:
            return err("Mag Cache maximum number of steps is 50")
        
    if image_mode > 0:
        audio_prompt_type = ""
        postprocess_audio = ""
        replace_voice_method = ""
        replace_voice_sample = None
        replace_voice_sample2 = None
    postprocess_audio_meta = audio_processor_api.method_metadata(postprocess_audio)
    replace_voice_meta = audio_processor_api.method_metadata(replace_voice_method)

    if postprocess_audio and postprocess_audio != "control" and audio_processor_api.find_processor(postprocess_audio) is None:
        return err(f"Audio processing method '{postprocess_audio}' is not supported by the current WanGP install")
    if replace_voice_method and audio_processor_api.find_processor(replace_voice_method) is None:
        return err(f"Audio processing method '{replace_voice_method}' is not supported by the current WanGP install")

    if "K" in audio_prompt_type and "V" not in video_prompt_type:
        return err("You must enable a Control Video to use the Control Video Audio Track as an audio prompt")

    if (model_def.get("multitalk_class", False) or model_def.get("speaker_locations", False)) and ("B" in audio_prompt_type or "X" in audio_prompt_type) and not model_def.get("one_speaker_only", False):
        from models.wan.multitalk.multitalk import parse_speakers_locations
        speakers_bboxes, error = parse_speakers_locations(speakers_locations)
        if len(error) > 0:
            return err(error)

    if postprocess_audio and postprocess_audio != "control" and audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK not in postprocess_audio_meta["types"]:
        return err(f"{postprocess_audio_meta['label']} is not available as a soundtrack processor")
    if "F" in video_prompt_type:
        if len(frames_positions.strip()) > 0:
            positions = frames_positions.replace(","," ").split(" ")
            for pos_str in positions:
                if not pos_str.upper() in ["L", "X"] and len(pos_str)>0:
                    if not is_integer(pos_str):
                        return err(f"Invalid Frame Position '{pos_str}'")
                    pos = int(pos_str)
                    if pos <1 or pos > max_source_video_frames:
                        return err(f"Invalid Frame Position Value'{pos_str}'")
    else:
        frames_positions = None

    if postprocess_audio and audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"]:
        soundtrack_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK, audio_source=audio_source, media_source_exists=media_source_exists, has_audio_file_extension=has_audio_file_extension)
        if soundtrack_error:
            return err(soundtrack_error)
        if not postprocess_audio_meta["needs_audio_source"]:
            audio_source = None
    else:
        audio_source = None
    if replace_voice_method:
        replace_voice_error = audio_processor_api.validate_method(replace_voice_method, audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT, voice_sample=replace_voice_sample, voice_sample2=replace_voice_sample2)
        if replace_voice_error:
            return err(replace_voice_error)
    else:
        replace_voice_sample = None
        replace_voice_sample2 = None
    if not replace_voice_meta["needs_voice_sample2"]:
        replace_voice_sample2 = None
    if len(filter_letters(image_prompt_type, "VLG")) > 0 and len(keep_frames_video_source) > 0:
        if not is_integer(keep_frames_video_source) or int(keep_frames_video_source) == 0:
            return err("The number of frames to keep must be a non null integer")
    else:
        keep_frames_video_source = ""

    if image_outputs:
        image_prompt_type = image_prompt_type.replace("V", "").replace("L", "")
    custom_guide_def = model_def.get("custom_guide", None)
    if custom_setting_visible(custom_guide_def, video_prompt_type, audio_prompt_type):
        if custom_guide is None and custom_guide_def.get("required", False):
            return err(f"You must provide a {custom_guide_def.get('label', 'Custom Guide')}")
    else:
        custom_guide = None

    if not is_edit_mode:
        if "V" in image_prompt_type:
            if video_source == None:
                return err("You must provide a Source Video file to continue")
        else:
            video_source = None

    if not input_video_strength_visible(model_def, image_prompt_type, video_prompt_type):
        input_video_strength = 1.0

    if "A" in audio_prompt_type:
        if audio_guide == None and not model_def.get("auto_null_audio", False):
            return err("You must provide an Audio Source")
    else:
        audio_guide = None


    if "B" in audio_prompt_type:
        if audio_guide2 == None:
            return err("You must provide a second Audio Source")
    else:
        audio_guide2 = None
    if "D" in audio_prompt_type:
        if audio_guide3 == None:
            return err("You must provide a third Audio Source")
    else:
        audio_guide3 = None
    if not all_letters(audio_prompt_type, "AB"):
        audio_prompt_type = del_in_sequence(audio_prompt_type, "N")
    if model_type in ["vace_multitalk_14B"] and ("B" in audio_prompt_type or "X" in audio_prompt_type):
        if not "I" in video_prompt_type and not not "V" in video_prompt_type:
            gr.Info("To get good results with Multitalk and two people speaking, it is recommended to set a Reference Frame or a Control Video (potentially truncated) that contains the two people one on each side")

    if model_def.get("one_image_ref_needed", False):
        if image_refs is None:
            return err("You must provide an Image Reference")
        if len(image_refs) > 1:
            return err("Only one Image Reference (a person) is supported for the moment by this model")
    if model_def.get("at_least_one_image_ref_needed", False):
        if image_refs is None:
            return err("You must provide at least one Image Reference")
        
    if "I" in video_prompt_type:
        if image_refs == None or len(image_refs) == 0:
            return err("You must provide at least one Reference Image")
        image_refs = clean_image_list(image_refs)
        if image_refs == None :
            return err("A Reference Image should be an Image")
        if model_def.get("one_image_ref_only", False) and (model_def.get("one_image_ref_only_with_background", False) or not any_letters(video_prompt_type, "KF")) and len(image_refs) > 1:
            return err("Only one Reference Image is supported by this model mode")
    else:
        image_refs = None

    if "V" in video_prompt_type:
        if image_outputs:
            if image_guide is None:
                return err("You must provide a Control Image")
        else:
            if video_guide is None:
                return err("You must provide a Control Video")
        if "+" in video_prompt_type and video_guide2 is None:
            return err("You must provide a second Control Video")
        if "*" in video_prompt_type and video_guide3 is None:
            return err("You must provide a third Control Video")
        if "A" in video_prompt_type and not "U" in video_prompt_type:             
            if image_outputs:
                if image_mask is None:
                    return err("You must provide a Image Mask")
            else:
                if video_mask is None:
                    return err("You must provide a Video Mask")
        else:
            video_mask = None
            image_mask = None

        if "G" in video_prompt_type:
                if denoising_strength < 1. and not model_def.get("custom_denoising_strength", False):
                    gr.Info(f"With Denoising Strength {denoising_strength:.1f}, Denoising will start at Step no {int(round(num_inference_steps * (1. - denoising_strength),4))} ")
        else: 
            denoising_strength = 1.0
                    
        if "G" in video_prompt_type or model_def.get("mask_strength_always_enabled", False):
                if "A" in video_prompt_type and "U" not in video_prompt_type and masking_strength < 1.:
                    masking_duration = math.ceil(num_inference_steps * masking_strength)
                    if masking_strength:
                        gr.Info(f"With Masking Strength {masking_strength:.1f}, Masking will last {masking_duration}{' Step' if masking_duration==1 else ' Steps'}")
        else: 
            masking_strength = 1.0
        if len(keep_frames_video_guide) > 0 and model_type in ["ltxv_13B"]:
            return err("Keep Frames for Control Video is not supported with LTX Video")
        _, error = parse_keep_frames_video_guide(keep_frames_video_guide, video_length)
        if len(error) > 0:
            return err(f"Invalid Keep Frames property: {error}")
    else:
        video_guide = None
        video_guide2 = None
        video_guide3 = None
        image_guide = None
        video_mask = None
        image_mask = None
        keep_frames_video_guide = ""
        denoising_strength = 1.0
        masking_strength = 1.0
    
    if image_outputs:
        video_guide = None
        video_guide2 = None
        video_guide3 = None
        video_mask = None
    else:
        image_guide = None
        image_mask = None
    image_prompt_types_allowed = model_def.get("image_prompt_types_allowed", "")
    if "S" in image_prompt_type:
        if "S" not in image_prompt_types_allowed:
            return err("This model doesn't accept a Start Image")
    
        if model_def.get("black_frame", False) and len(image_start or [])==0:
            if "E" in image_prompt_type and len(image_end or []):
                image_end = clean_image_list(image_end)        
                image_start = [Image.new("RGB", image.size, (0, 0, 0, 255)) for image in image_end] 
            else:
                image_start = [Image.new("RGB", (width, height), (0, 0, 0, 255))] 

        if image_start == None or isinstance(image_start, list) and len(image_start) == 0:
            return err("You must provide a Start Image")
        image_start = clean_image_list(image_start)        
        if image_start == None :
            return err("Start Image should be an Image")
        if "W" in multi_prompts_gen_type and len(image_start) > 1:
            return err("Only one Start Image is supported when a multi-prompt Sliding Window mode is selected")
    else:
        image_start = None

    if not end_frames_always_enabled(model_def) and not any_letters(image_prompt_type, "SVL"):
        image_prompt_type = image_prompt_type.replace("E", "")
    if "E" in image_prompt_type:
        if "E" not in image_prompt_types_allowed:
            return err("This model doesn't accept an End Image")
    
        if image_end == None or isinstance(image_end, list) and len(image_end) == 0:
            return err("You must provide an End Image")
        image_end = clean_image_list(image_end)        
        if image_end == None :
            return err("End Image should be an Image")
        if (video_source is not None or "L" in image_prompt_type):
            if "W" not in multi_prompts_gen_type and len(image_end)> 1:
                return err("If you want to Continue a Video, you can use Multiple End Images only when a multi-prompt Sliding Window mode is selected")        
        elif "W" not in multi_prompts_gen_type:
            if len(image_start or []) > 0 and len(image_start or []) != len(image_end or []):
                return err("The number of Start and End Images should be the same unless a multi-prompt Sliding Window mode is selected")    
    else:        
        image_end = None

    if "V" in video_prompt_type and "O" in video_prompt_type:
        if image_start is None and video_source is None and "L" not in video_prompt_type and not all_letters(video_prompt_type, "IK"):
            return err("Aligned Pose transfer requires a Start Image, a Source Video to continue or Background Ref Frame to be used")    
        if "A" in video_prompt_type and any_letters(video_prompt_type, "YWZ"):
            return err("Aligned Pose transfer supports only Inpainting process outside the masked area")    

    if test_any_sliding_window(model_type) and image_mode == 0:
        if frame_scheduler is not None and frame_scheduler["active"]:
            extra = f", which is more than the original number of frames {video_length}" if frame_scheduler["predicted_total_frames"] > video_length else ""
            gr.Info(f"{len(frame_scheduler['windows'])} Sliding Windows will be generated, for an estimated total of {frame_scheduler['predicted_total_frames']} output frames{extra}.")
        elif video_length > sliding_window_size:
            if test_class_t2v(model_type) and not "G" in video_prompt_type :
                return err(f"You have requested to Generate Sliding Windows with a Text to Video model. Unless you use the Video to Video feature this is useless as a t2v model doesn't see past frames and it will generate the same video in each new window.")
            full_video_length = video_length if video_source is None else video_length +  sliding_window_overlap -1
            extra = "" if full_video_length == video_length else f" including {sliding_window_overlap} added for Video Continuation"
            no_windows = compute_sliding_window_no(full_video_length, sliding_window_size, sliding_window_discard_last_frames, sliding_window_overlap)
            gr.Info(f"The Number of Frames to generate ({video_length}{extra}) is greater than the Sliding Window Size ({sliding_window_size}), {no_windows} Windows will be generated")
    if "recam" in model_filename:
        if video_guide == None:
            return err("You must provide a Control Video")
        computed_fps = get_computed_fps(force_fps, model_type , video_guide, video_source )
        frames = get_resampled_video(video_guide, 0, 81, computed_fps)
        if len(frames)<81:
            return err(f"Recammaster Control video should be at least 81 frames once the resampling at {computed_fps} fps has been done")

    if "hunyuan_custom_custom_edit" in model_filename:
        if len(keep_frames_video_guide) > 0: 
            return err("Filtering Frames with this model is not supported")

    if "W" in multi_prompts_gen_type or single_prompt:
        if image_start != None and len(image_start) > 1:
            return err("Only one Start Image can be provided in Edit Mode" if single_prompt else "Only one Start Image must be provided if multiple prompts are used for different windows")

        # if image_end != None and len(image_end) > 1:
        #     gr.Info("Only one End Image must be provided if multiple prompts are used for different windows") 
        #     return

    override_inputs = {
        "image_start": image_start[0] if image_start !=None and len(image_start) > 0 else None,
        "image_end": image_end, #[0] if image_end !=None and len(image_end) > 0 else None,
        "image_refs": image_refs,
        "audio_guide": audio_guide,
        "audio_guide2": audio_guide2,
        "audio_guide3": audio_guide3,
        "audio_source": audio_source,
        "replace_voice_method": replace_voice_method,
        "replace_voice_sample": replace_voice_sample,
        "replace_voice_sample2": replace_voice_sample2,
        "postprocess_audio": postprocess_audio,
        "video_guide": video_guide,
        "video_guide2": video_guide2,
        "video_guide3": video_guide3,
        "image_guide": image_guide,
        "video_mask": video_mask,
        "image_mask": image_mask,
        "custom_guide": custom_guide,
        "video_source": video_source,
        "frames_positions": frames_positions,
        "keep_frames_video_source": keep_frames_video_source,
        "input_video_strength": input_video_strength,
        "keep_frames_video_guide": keep_frames_video_guide,
        "denoising_strength": denoising_strength,
        "masking_strength": masking_strength,
        "image_prompt_type": image_prompt_type,
        "video_prompt_type": video_prompt_type,        
        "audio_prompt_type": audio_prompt_type,
        "guidance_phases": guidance_phases,
        "skip_steps_cache_type": skip_steps_cache_type,
        "model_switch_phase": model_switch_phase,
        "motion_amplitude": motion_amplitude,
        "model_mode": model_mode,
        "video_guide_outpainting": video_guide_outpainting,
        "video_guide_outpainting_ratio": inputs.get("video_guide_outpainting_ratio", ""),
        "custom_settings": inputs.get("custom_settings", None),
        "self_refiner_plan": self_refiner_plan,
        "image_mode": image_mode,
    } 
    inputs.update(override_inputs)
    if hasattr(model_handler, "validate_generative_settings"):
        error = model_handler.validate_generative_settings(model_type, model_def, inputs)
        if error is not None and len(error) > 0:
            return err(error)
    inputs.pop("frame_scheduler", None)
    return inputs, prompts, image_start, image_end, ""


def get_preview_images(inputs):
    inputs_to_query = ["image_start", "video_source", "image_end", "video_guide", "video_guide2", "video_guide3", "image_guide", "video_mask", "image_mask", "image_refs"]
    labels = ["Start Image", "Video Source", "End Image", "Video Guide", "Video Guide 2", "Video Guide 3", "Image Guide", "Video Mask", "Image Mask", "Image Reference"]
    start_image_data = None
    start_image_labels = []
    end_image_data = None
    end_image_labels = []
    for label, name in  zip(labels,inputs_to_query):
        image= inputs.get(name, None)
        if image is not None:
            image= [image] if not isinstance(image, list) else image.copy()
            if start_image_data == None:
                start_image_data = image
                start_image_labels += [label] * len(image)
            else:
                if end_image_data == None:
                    end_image_data = image
                else:
                    end_image_data += image 
                end_image_labels += [label] * len(image)

    if start_image_data != None and len(start_image_data) > 1 and  end_image_data  == None:
        end_image_data = start_image_data [1:]
        end_image_labels = start_image_labels [1:]
        start_image_data = start_image_data [:1] 
        start_image_labels = start_image_labels [:1] 
    return start_image_data, end_image_data, start_image_labels, end_image_labels 

def add_video_task(**inputs):
    global task_id
    state = inputs["state"]
    gen = get_gen_info(state)
    queue = gen["queue"]
    task_id += 1
    current_task_id = task_id

    start_image_data, end_image_data, start_image_labels, end_image_labels = get_preview_images(inputs)
    plugin_data = inputs.pop('plugin_data', {})
    
    queue.append({
        "id": current_task_id,
        "params": inputs.copy(),
        "plugin_data": plugin_data,
        "repeats": inputs.get("repeat_generation",1),
        "length": inputs.get("video_length",0) or 0, 
        "steps": inputs.get("num_inference_steps",0) or 0,
        "prompt": inputs.get("prompt", ""),
        "start_image_labels": start_image_labels,
        "end_image_labels": end_image_labels,
        "start_image_data": start_image_data,
        "end_image_data": end_image_data,
        "start_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in start_image_data] if start_image_data != None else None,
        "end_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in end_image_data] if end_image_data != None else None
    })

def update_task_thumbnails(task,  inputs):
    start_image_data, end_image_data, start_labels, end_labels = get_preview_images(inputs)

    task.update({
        "start_image_labels": start_labels,
        "end_image_labels": end_labels,
        "start_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in start_image_data] if start_image_data != None else None,
        "end_image_data_base64": [pil_to_base64_uri(img, format="jpeg", quality=70) for img in end_image_data] if end_image_data != None else None
    })

def move_task(queue, old_index_str, new_index_str):
    try:
        old_idx = int(old_index_str)
        new_idx = int(new_index_str)
    except (ValueError, IndexError):
        return update_queue_data(queue)

    with lock:
        old_idx += 1
        new_idx += 1

        if not (0 < old_idx < len(queue)):
            return update_queue_data(queue)

        item_to_move = queue.pop(old_idx)
        if old_idx < new_idx:
            new_idx -= 1
        clamped_new_idx = max(1, min(new_idx, len(queue)))
        
        queue.insert(clamped_new_idx, item_to_move)

    return update_queue_data(queue)

def remove_task(queue, task_id_to_remove):
    if not task_id_to_remove:
        return update_queue_data(queue)

    with lock:
        idx_to_del = next((i for i, task in enumerate(queue) if task['id'] == task_id_to_remove), -1)
        
        if idx_to_del != -1:
            if idx_to_del == 0:
                wan_model._interrupt = True
            del queue[idx_to_del]
            
    return update_queue_data(queue)

def update_global_queue_ref(queue):
    global global_queue_ref
    with lock:
        global_queue_ref = queue[:]

def _unwrap_attachment_item(item):
    if isinstance(item, (tuple, list)) and len(item) > 0:
        item = item[0]
    if isinstance(item, dict):
        item = item.get("path") or item.get("name") or item.get("orig_name") or item.get("url") or item
    elif not isinstance(item, (Image.Image, str)):
        item = getattr(item, "path", None) or getattr(item, "name", None) or item
    return item

def _save_queue_to_zip(queue, output):
    """Save queue to ZIP. output can be a filename (str) or BytesIO buffer.
    Returns True on success, False on failure.
    """
    if not queue:
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        queue_manifest = []
        file_paths_in_zip = {}

        for task_index, task in enumerate(queue):
            if task is None or not isinstance(task, dict) or task.get('id') is None:
                continue

            params_copy = task.get('params', {}).copy()
            task_id_s = task.get('id', f"task_{task_index}")

            for key in ATTACHMENT_KEYS:
                value = params_copy.get(key)
                if value is None:
                    continue

                is_originally_list = isinstance(value, list)
                items = value if is_originally_list else [value]

                processed_filenames = []
                for item_index, item in enumerate(items):
                    item = _unwrap_attachment_item(item)
                    if isinstance(item, Image.Image):
                        item_id = id(item)
                        if item_id in file_paths_in_zip:
                            processed_filenames.append(file_paths_in_zip[item_id])
                            continue
                        filename_in_zip = f"task{task_id_s}_{key}_{item_index}.png"
                        save_path = os.path.join(tmpdir, filename_in_zip)
                        try:
                            item.save(save_path, "PNG")
                            processed_filenames.append(filename_in_zip)
                            file_paths_in_zip[item_id] = filename_in_zip
                        except Exception as e:
                            print(f"Error saving attachment {filename_in_zip}: {e}")
                    elif isinstance(item, str):
                        if item in file_paths_in_zip:
                            processed_filenames.append(file_paths_in_zip[item])
                            continue
                        if not os.path.isfile(item):
                            continue
                        _, extension = os.path.splitext(item)
                        filename_in_zip = f"task{task_id_s}_{key}_{item_index}{extension if extension else ''}"
                        save_path = os.path.join(tmpdir, filename_in_zip)
                        try:
                            shutil.copy2(item, save_path)
                            processed_filenames.append(filename_in_zip)
                            file_paths_in_zip[item] = filename_in_zip
                        except Exception as e:
                            print(f"Error copying attachment {item}: {e}")

                if processed_filenames:
                    params_copy[key] = processed_filenames if is_originally_list else processed_filenames[0]

            # Remove runtime-only keys
            for runtime_key in ['state', 'start_image_labels', 'end_image_labels',
                                'start_image_data_base64', 'end_image_data_base64',
                                'start_image_data', 'end_image_data']:
                params_copy.pop(runtime_key, None)

            params_copy['settings_version'] = settings_version
            if _is_edit_task_params(params_copy):
                params_copy.pop("model_type", None)
                params_copy.pop("base_model_type", None)
            else:
                params_copy['base_model_type'] = get_base_model_type(params_copy["model_type"])

            manifest_entry = {"id": task.get('id'), "params": params_copy}
            manifest_entry = {k: v for k, v in manifest_entry.items() if v is not None}
            queue_manifest.append(manifest_entry)

        manifest_path = os.path.join(tmpdir, "queue.json")
        try:
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(queue_manifest, f, indent=4)
        except Exception as e:
            print(f"Error writing queue.json: {e}")
            return False

        try:
            with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(manifest_path, arcname="queue.json")
                for saved_file_rel_path in file_paths_in_zip.values():
                    saved_file_abs_path = os.path.join(tmpdir, saved_file_rel_path)
                    if os.path.exists(saved_file_abs_path):
                        zf.write(saved_file_abs_path, arcname=saved_file_rel_path)
            return True
        except Exception as e:
            print(f"Error creating zip: {e}")
            return False

def save_queue_action(state):
    gen = get_gen_info(state)
    queue = gen.get("queue", [])

    if not queue or len(queue) == 0:
        gr.Info("Queue is empty. Nothing to save.")
        return ""

    queue = queue[:]
    return gradio_downloads.register_download("queue.zip", "application/zip", lambda: gradio_downloads.stream_writer(lambda writer: _save_queue_to_zip(queue, writer)))

def clean_settings(model_type, params):
    saved_settings_version = params.get('settings_version', 0)
    merged = get_factory_settings(model_type)
    merged.update({k: v for k, v in params.items() if v is not None or k not in merged})
    params.clear()
    params.update(merged)
    fix_settings(model_type, params, saved_settings_version)
    for k, v in primary_settings.items():
        params.setdefault(k, v)
    params.setdefault("client_id", "")
    params.setdefault("mode", "")
    for meta_key in ['type', 'base_model_type']:
        params.pop(meta_key, None)


def _attachment_has_path_values(value):
    if isinstance(value, str):
        return len(value.strip()) > 0
    if isinstance(value, (list, tuple)):
        return any(isinstance(item, str) and len(item.strip()) > 0 for item in value)
    return False


def _task_has_path_attachments(params):
    return any(_attachment_has_path_values(params.get(key)) for key in ATTACHMENT_KEYS)


def _load_task_attachments(params, media_base_path, cache_dir=None, log_prefix="[load]"):

    preserve_edit_media_source_path = str(params.get("mode", "") or "").startswith("edit_")
    for key in ATTACHMENT_KEYS:
        value = params.get(key)
        if value is None:
            continue

        is_originally_list = isinstance(value, list)
        filenames = value if is_originally_list else [value]

        loaded_items = []
        for filename in filenames:
            if not isinstance(filename, str) or not filename.strip():
                print(f"{log_prefix} Warning: Invalid filename for key '{key}'. Skipping.")
                continue
            virtual_spec = parse_virtual_media_path(filename)
            if virtual_spec is not None and get_virtual_media_vsource(virtual_spec) is not None:
                loaded_items.append(filename)
                print(f"{log_prefix} Using virtual source: {filename}")
                continue
            source_name = virtual_spec.source_path if virtual_spec is not None else filename

            if os.path.isabs(source_name):
                source_path = source_name
            else:
                source_path = os.path.join(media_base_path, source_name)

            if not os.path.exists(source_path):
                print(f"{log_prefix} Warning: File not found for '{key}': {source_path}")
                continue

            if cache_dir:
                try:
                    final_path = gr.processing_utils.save_file_to_cache(source_path, cache_dir)
                except Exception as e:
                    print(f"{log_prefix} Error copying {filename}: {e}")
                    continue
            else:
                final_path = source_path

            # Load images as PIL, keep videos/audio as paths
            if key == "video_source" and preserve_edit_media_source_path:
                loaded_items.append(replace_virtual_media_source(filename, final_path) if virtual_spec is not None else final_path)
                print(f"{log_prefix} Using path: {final_path}")
            elif has_image_file_extension(final_path):
                try:
                    with Image.open(final_path) as loaded_image:
                        loaded_items.append(loaded_image.copy())
                    print(f"{log_prefix} Loaded image: {final_path}")
                except Exception as e:
                    print(f"{log_prefix} Error loading image {final_path}: {e}")
            else:
                loaded_items.append(replace_virtual_media_source(filename, final_path) if virtual_spec is not None else final_path)
                print(f"{log_prefix} Using path: {final_path}")

        # Update params, preserving list/single structure
        if loaded_items:
            if key == "image_refs" or is_originally_list:
                params[key] = loaded_items
            else:
                params[key] = loaded_items[0]
        else:
            params.pop(key, None)


def _build_runtime_task(task_id_val, params, plugin_data=None):
    """Build a runtime task dict from params."""
    primary_preview, secondary_preview, primary_labels, secondary_labels = get_preview_images(params)

    start_b64 = [pil_to_base64_uri(primary_preview[0], format="jpeg", quality=70)] if isinstance(primary_preview, list) and primary_preview else None
    end_b64 = [pil_to_base64_uri(secondary_preview[0], format="jpeg", quality=70)] if isinstance(secondary_preview, list) and secondary_preview else None

    return {
        "id": task_id_val,
        "params": params,
        "plugin_data": plugin_data or {},
        "repeats": params.get('repeat_generation', 1),
        "length": params.get('video_length'),
        "steps": params.get('num_inference_steps'),
        "prompt": params.get('prompt'),
        "start_image_labels": primary_labels,
        "end_image_labels": secondary_labels,
        "start_image_data": params.get("image_start") or params.get("image_refs"),
        "end_image_data": params.get("image_end"),
        "start_image_data_base64": start_b64,
        "end_image_data_base64": end_b64,
    }


def _is_edit_task_params(params):
    return isinstance(params, dict) and str(params.get("mode", "") or "").startswith("edit_")


STATIC_AUDIO_POSTPROCESS_STATUS = {"control": "Control Audio Remuxing"}
EDIT_TASK_STATUS = {"edit_audio": ("Applying Audio Post Processing", True), "edit_remux": ("Applying Audio Remuxing", True), "edit_postprocessing": ("Applying Media Post Processing", False)}


def get_task_status_text(task):
    params = task.get("params", {}) if isinstance(task, dict) else {}
    prefix, has_audio_action = EDIT_TASK_STATUS.get(params.get("mode", ""), ("Generating...", False))
    method = audio_processor_api.normalize_method(params.get("postprocess_audio") or "")
    status = STATIC_AUDIO_POSTPROCESS_STATUS.get(method) or audio_processor_api.method_metadata(method)["status"] or "Audio Post Processing"
    return f"{prefix} - {status}" if has_audio_action else prefix


def _extract_model_type(params, state, log_prefix="[load]"):
    if _is_edit_task_params(params):
        params.pop("model_type", None)
        params.pop("base_model_type", None)
        return "", None

    base_model_type = params.get('base_model_type', None)
    model_type = original_model_type = params.get('model_type', base_model_type)

    if model_type is not None and get_model_def(model_type) is None:
        model_type = base_model_type

    if model_type is None:
        return None, f"Unknown model type: {original_model_type}" if original_model_type is not None else "Settings must contain 'model_type'"
    params["model_type"] = model_type
    if get_model_def(model_type) is None:
        return None, f"Unknown model type: {original_model_type}"
    return model_type, None


def _parse_task_manifest(manifest, state, media_base_path, cache_dir=None, log_prefix="[load]", verbose_output = True, skip_validate_settings=False):
    global task_id
    newly_loaded_queue = []
    first_error = None

    for task_index, task_data in enumerate(manifest):
        if task_data is None or not isinstance(task_data, dict):
            if first_error is None:
                first_error = f"Invalid task data at index {task_index}"
            print(f"{log_prefix} Skipping invalid task data at index {task_index}")
            continue

        params = task_data.get('params', {})
        task_id_loaded = task_data.get('id', task_id + 1)

        model_type, error = _extract_model_type(params, state, log_prefix)
        if error:
            if first_error is None:
                first_error = error
            print(f"{log_prefix} {error} for task #{task_id_loaded}. Skipping.")
            continue

        params['state'] = state

        if media_base_path is not None or _task_has_path_attachments(params):
            _load_task_attachments(params, media_base_path or os.path.dirname(os.path.abspath(__file__)), cache_dir, log_prefix)

        params, error = validate_task(task_data, state, skip_validate_settings=skip_validate_settings)
        if error:
            if first_error is None:
                first_error = error
            print(f"{log_prefix} {error} for task #{task_id_loaded}. Skipping.")
            continue

        # Build runtime task
        runtime_task = _build_runtime_task(task_id_loaded, params, task_data.get('plugin_data', {}))
        newly_loaded_queue.append(runtime_task)
        if verbose_output:
            task_label = params.get("mode", "") if _is_edit_task_params(params) else model_type
            print(f"{log_prefix} Task {task_index+1}/{len(manifest)} ready, ID: {task_id_loaded}, model: {task_label}")

    # Update global task_id
    if newly_loaded_queue:
        current_max_id = max([t['id'] for t in newly_loaded_queue if 'id' in t] + [0])
        if current_max_id >= task_id:
            task_id = current_max_id + 1

    return newly_loaded_queue, None if len(newly_loaded_queue) > 0 else first_error or "No valid task could be unpacked."


def _parse_queue_zip_tasks(filename, state, task_limit=None, log_prefix="[load_queue]", skip_validate_settings=False):
    """Parse queue ZIP file. Returns (queue_list, error_msg or None, source_task_count)."""
    cache_dir = gr.utils.get_upload_folder()

    try:
        print(f"{log_prefix} Attempting to load queue from: {filename}")
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(filename, 'r') as zf:
                if "queue.json" not in zf.namelist():
                    return None, "queue.json not found in zip file", 0
                print(f"{log_prefix} Extracting to temp directory...")
                zf.extractall(tmpdir)

            manifest_path = os.path.join(tmpdir, "queue.json")
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            source_task_count = len(manifest)
            print(f"{log_prefix} Loaded manifest with {source_task_count} tasks.")
            if task_limit is not None:
                manifest = manifest[:task_limit]

            queue, error = _parse_task_manifest(manifest, state, tmpdir, cache_dir, log_prefix, skip_validate_settings=skip_validate_settings)
            return queue, error, source_task_count

    except Exception as e:
        traceback.print_exc()
        return None, str(e), 0


def _parse_queue_zip(filename, state):
    """Parse queue ZIP file. Returns (queue_list, error_msg or None)."""
    queue, error, _ = _parse_queue_zip_tasks(filename, state)
    return queue, error


def _parse_settings_zip(filename, state, skip_validate_settings=False):
    """Parse a settings ZIP file and extract only the first task."""
    return _parse_queue_zip_tasks(filename, state, task_limit=1, log_prefix="[load_settings]", skip_validate_settings=skip_validate_settings)


def _parse_settings_json(filename, state):
    """Parse a single settings JSON file. Returns (queue_list, error_msg or None).

    Media paths in JSON are filesystem paths (absolute or relative to WanGP folder).
    """
    global task_id

    try:
        print(f"[load_settings] Loading settings from: {filename}")

        with open(filename, 'r', encoding='utf-8') as f:
            params = json.load(f)

        if isinstance(params, list):
            # Accept full queue manifests or a list of settings dicts
            if all(isinstance(item, dict) and "params" in item for item in params):
                manifest = params
            else:
                manifest = []
                for item in params:
                    if not isinstance(item, dict):
                        continue
                    task_id += 1
                    manifest.append({"id": task_id, "params": item, "plugin_data": {}})
        elif isinstance(params, dict):
            # Wrap as single-task manifest
            task_id += 1
            manifest = [{"id": task_id, "params": params, "plugin_data": {}}]
        else:
            return None, "Settings file must contain a JSON object or a list of tasks"

        # Media paths are relative to WanGP folder (no cache needed)
        wgp_folder = os.path.dirname(os.path.abspath(__file__))

        return _parse_task_manifest(manifest, state, wgp_folder, None, "[load_settings]")

    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    except Exception as e:
        traceback.print_exc()
        return None, str(e)


def record_queue_error(state, queue, error, abort= False):
    gen = get_gen_info(state)
    queue_errors = gen.get("queue_errors", None)
    if queue_errors is None:
        gen["queue_errors"] = queue_errors = {}

    for i, task in enumerate(queue):
        params = task["params"]
        client_id= params.get("client_id", "") or ""
        if len(client_id):
            queue_errors[client_id] = (error, abort, i>0)


def _normalize_inline_queue_priority(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {"", "0", "false", "off", "no"}:
            return False
        if value in {"1", "true", "on", "yes"}:
            return True
    return bool(value)


def _pop_runtime_task_priority(task):
    if not isinstance(task, dict):
        return False
    params = task.get("params", None)
    raw_value = params.pop("priority", None) if isinstance(params, dict) else None
    if raw_value is None:
        raw_value = task.pop("priority", None)
    return _normalize_inline_queue_priority(raw_value)


def load_queue_action(filepath, state, evt:gr.EventData):
    """Load queue from ZIP or JSON file (Gradio UI wrapper)."""
    global task_id
    service = service_for(state)
    if service is not None and evt.target is None:
        if service.autoloaded_queue:
            return gr.update()
        service.autoloaded_queue = True
    gen = get_gen_info(state)
    original_queue = gen.get("queue", [])

    # Determine filename (autoload vs user upload)
    delete_autoqueue_file = False
    filename = file_path = None
    verbose_output = True
    newly_loaded_queue = gen.pop("inline_queue", None)
    if newly_loaded_queue is not None:
        verbose_output = False
        inline_queue_source = newly_loaded_queue
        if isinstance(newly_loaded_queue, dict): 
            newly_loaded_queue = [ {"id": 0, "params": newly_loaded_queue}]
        else:
            inline_queue_source = newly_loaded_queue
        try:
            newly_loaded_queue, error = _parse_task_manifest(newly_loaded_queue, state, None, None, "[unpack queue]", verbose_output = verbose_output )
        except Exception as exception:
            traceback.print_exc()
            newly_loaded_queue, error = [], f"Inline queue validation failed: {exception}"
        if error:
            if isinstance(inline_queue_source, dict):
                inline_queue_source = [{"id": 0, "params": inline_queue_source}]
            record_queue_error(state, inline_queue_source or [], error)
            gr.Warning(f"Failed to unpack inline queue: {error[:200]}")
            return update_queue_data(original_queue)

    elif evt.target == None:
        # Autoload only works with empty queue
        if original_queue:
            return
        autoload_path = None
        if Path(AUTOSAVE_PATH).is_file():
            autoload_path = AUTOSAVE_PATH
            delete_autoqueue_file = True
        elif AUTOSAVE_TEMPLATE_PATH != AUTOSAVE_PATH and Path(AUTOSAVE_TEMPLATE_PATH).is_file():
            autoload_path = AUTOSAVE_TEMPLATE_PATH
        else:
            return
        print(f"Autoloading queue from {autoload_path}...")
        filename = autoload_path
    else:
        if not filepath or not hasattr(filepath, 'name') or not Path(filepath.name).is_file():
            print("[load_queue_action] Warning: No valid file selected or file not found.")
            return update_queue_data(original_queue)
        filename = filepath.name

    try:
        # Detect file type and use appropriate parser
        if filename is not None:
            if filename.lower().endswith('.json'):
                newly_loaded_queue, error = _parse_settings_json(filename, state)
                # Safety: clear attachment paths when loading JSON through UI
                # (JSON files contain filesystem paths which could be security-sensitive)
                if newly_loaded_queue:
                    for task in newly_loaded_queue:
                        params = task.get('params', {})
                        for key in ATTACHMENT_KEYS:
                            if key in params:
                                params[key] = None
            else:
                newly_loaded_queue, error = _parse_queue_zip(filename, state)
            if error:
                gr.Warning(f"Failed to load queue: {error[:200]}")
                return update_queue_data(original_queue)

        # Merge with existing queue: renumber task IDs to avoid conflicts
        # IMPORTANT: Modify list in-place to preserve references held by process_tasks
        if original_queue:
            # Find the highest existing task ID
            max_existing_id = max([t.get('id', 0) for t in original_queue] + [0])
            # Renumber newly loaded tasks
            for i, task in enumerate(newly_loaded_queue):
                task['id'] = max_existing_id + 1 + i
            priority_tasks = []
            regular_tasks = []
            for task in newly_loaded_queue:
                if _pop_runtime_task_priority(task):
                    priority_tasks.append(task)
                else:
                    regular_tasks.append(task)
            # Update global task_id counter
            task_id = max_existing_id + len(newly_loaded_queue) + 1
            with lock:
                if priority_tasks:
                    original_queue[1:1] = priority_tasks
                if regular_tasks:
                    original_queue.extend(regular_tasks)
                gen["queue"] = original_queue
            action_msg = f"Merged {len(newly_loaded_queue)} task(s) with existing {len(original_queue) - len(newly_loaded_queue)} task(s)"
            merged_queue = original_queue
        else:
            for task in newly_loaded_queue:
                _pop_runtime_task_priority(task)
            # No existing queue - assign newly loaded queue directly
            merged_queue = newly_loaded_queue
            action_msg = f"Loaded {len(newly_loaded_queue)} task(s)"
            with lock:
                gen["queue"] = merged_queue

        # Update state (Gradio-specific)
        with lock:
            gen["prompts_max"] = len(merged_queue)
        update_global_queue_ref(merged_queue)
        if verbose_output:
            print(f"[load_queue_action] {action_msg}.")
            gr.Info(action_msg)
        return update_queue_data(merged_queue)

    except Exception as e:
        error_message = f"Error during queue load: {e}"
        print(f"[load_queue_action] Caught error: {error_message}")
        traceback.print_exc()
        gr.Warning(f"Failed to load queue: {error_message[:200]}")
        return update_queue_data(original_queue)

    finally:
        if filename and delete_autoqueue_file:
            if os.path.isfile(filename):
                os.remove(filename)
                print(f"Clear Queue: Deleted autosave file '{filename}'.")

        if filepath and hasattr(filepath, 'name') and filepath.name and os.path.exists(filepath.name):
            if tempfile.gettempdir() in os.path.abspath(filepath.name):
                try:
                    os.remove(filepath.name)
                    print(f"[load_queue_action] Removed temporary upload file: {filepath.name}")
                except OSError as e:
                    print(f"[load_queue_action] Info: Could not remove temp file {filepath.name}: {e}")
            else:
                print(f"[load_queue_action] Info: Did not remove non-temporary file: {filepath.name}")


def clear_queue_action(state):
    gen = get_gen_info(state)
    gen["resume"] = True
    queue = gen.get("queue", [])
    aborted_current = False
    cleared_pending = False

    with lock:
        if "in_progress" in gen and gen["in_progress"]:
            print("Clear Queue: Signalling abort for in-progress task.")
            gen["abort"] = True
            gen["extra_orders"] = 0
            if wan_model is not None:
                wan_model._interrupt = True
            aborted_current = True

        if queue:
             if len(queue) > 1 or (len(queue) == 1 and queue[0] is not None and queue[0].get('id') is not None):
                 print(f"Clear Queue: Clearing {len(queue)} tasks from queue.")
                 queue.clear()
                 cleared_pending = True
             else:
                 pass

        if aborted_current or cleared_pending:
            gen["prompts_max"] = 0

    if cleared_pending:
        try:
            if os.path.isfile(AUTOSAVE_PATH):
                os.remove(AUTOSAVE_PATH)
                print(f"Clear Queue: Deleted autosave file '{AUTOSAVE_PATH}'.")
        except OSError as e:
            print(f"Clear Queue: Error deleting autosave file '{AUTOSAVE_PATH}': {e}")
            gr.Warning(f"Could not delete the autosave file '{AUTOSAVE_PATH}'. You may need to remove it manually.")

    if aborted_current and cleared_pending:
        gr.Info("Queue cleared and current generation aborted.")
    elif aborted_current:
        gr.Info("Current generation aborted.")
    elif cleared_pending:
        gr.Info("Queue cleared.")
    else:
        gr.Info("Queue is already empty or only contains the active task (which wasn't aborted now).")

    return update_queue_data([])
def quit_application():
    print("Save and Quit requested...")
    clear_startup_lock()
    autosave_queue()
    import signal
    os.kill(os.getpid(), signal.SIGINT)

def restart_application():
    print("Restart requested...")
    clear_startup_lock()
    autosave_queue()
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(42)

def start_quit_process():
    return 5, gr.update(visible=False), gr.update(visible=True)

def cancel_quit_process():
    return -1, gr.update(visible=True), gr.update(visible=False)

def show_countdown_info_from_state(current_value: int):
    if current_value > 0:
        gr.Info(f"Quitting in {current_value}...")
        return current_value - 1
    return current_value
quitting_app = False
def autosave_queue():
    global quitting_app
    quitting_app = True
    global global_queue_ref
    if not global_queue_ref:
        print("Autosave: Queue is empty, nothing to save.")
        return

    print(f"Autosaving queue ({len(global_queue_ref)} items) to {AUTOSAVE_PATH}...")
    try:
        if _save_queue_to_zip(global_queue_ref, AUTOSAVE_PATH):
            print(f"Queue autosaved successfully to {AUTOSAVE_PATH}")
        else:
            print("Autosave failed.")
    except Exception as e:
        print(f"Error during autosave: {e}")
        traceback.print_exc()

def finalize_generation_with_state(current_state):
     service = service_for(current_state)
     if service is not None:
         return (*service.finalized_updates(), gr.update(), current_state)
     if not isinstance(current_state, dict) or 'gen' not in current_state:
         return (
             gr.update(),
             gr.update(),
             gr.update(),
             gr.update(),
             gr.update(),
             gr.update(),
             gr.update(interactive=True),
             gr.update(interactive=True),
             gr.update(visible=True),
             gr.update(visible=False),
             gr.update(visible=False),
             gr.update(visible=False, value=""),
             gr.update(),
             current_state,
         )

     gallery_tabs_update, current_gallery_tab_update, gallery_update, audio_files_paths_update, audio_file_selected_update, audio_gallery_refresh_trigger_update, abort_btn_update, earlystop_btn_update, gen_btn_update, add_queue_btn_update, current_gen_col_update, gen_info_update = finalize_generation(current_state)
     accordion_update = gr.Accordion(open=False) if len(get_gen_info(current_state).get("queue", [])) <= 1 else gr.update()
     return gallery_tabs_update, current_gallery_tab_update, gallery_update, audio_files_paths_update, audio_file_selected_update, audio_gallery_refresh_trigger_update, abort_btn_update, earlystop_btn_update, gen_btn_update, add_queue_btn_update, current_gen_col_update, gen_info_update, accordion_update, current_state

def generate_queue_html(queue):
    if len(queue) <= 1:
        return "<div style='text-align: center; color: grey; padding: 20px;'>Queue is empty.</div>"

    top_button_html = ""
    bottom_button_html = ""
    
    if len(queue) > 11:
        btn_style = "width: 100%; padding: 8px; margin-bottom: 2px; font-weight: bold; display: flex; justify-content: center; align-items: center;"
        
        top_button_html = f"""
        <div style="margin-bottom: 5px;">
            <button onclick="scrollToQueueTop()" 
                    ondragenter="scrollToQueueTop(); event.preventDefault();" 
                    ondragover="event.preventDefault();"
                    class="gr-button gr-button-secondary" 
                    style="{btn_style}">
                <img src="/gradio_api/file=icons/top.svg" alt="Top" style="width: 1.3em; height: 1.3em; margin-right: 6px;">
                Scroll to Top
            </button>
        </div>
        """
        
        bottom_button_html = f"""
        <div style="margin-top: 5px;">
            <button onclick="scrollToQueueBottom()" 
                    ondragenter="scrollToQueueBottom(); event.preventDefault();" 
                    ondragover="event.preventDefault();"
                    class="gr-button gr-button-secondary" 
                    style="{btn_style.replace('margin-bottom', 'margin-top')}">
                <img src="/gradio_api/file=icons/bottom.svg" alt="Bottom" style="width: 1.3em; height: 1.3em; margin-right: 6px;">
                Scroll to Bottom
            </button>
        </div>
        """

    table_header = """
    <table>
        <thead>
            <tr>
                <th style="width:5%;" class="center-align">Qty</th>
                <th style="width:auto;" class="text-left">Prompt</th>
                <th style="width:7%;" class="center-align">Length</th>
                <th style="width:7%;" class="center-align">Steps</th>
                <th style="width:10%;" class="center-align">Start/Ref</th>
                <th style="width:10%;" class="center-align">End</th>
                <th style="width:4%;" class="center-align" title="Edit"></th>
                <th style="width:4%;" class="center-align" title="Remove"></th>
            </tr>
        </thead>
        <tbody>
    """
    
    table_rows = []
    scheme = server_config.get("queue_color_scheme", "pastel")

    for i, item in enumerate(queue):
        if i == 0:
            continue
        
        row_index = i - 1
        task_id = item['id']
        full_prompt = html.escape(item['prompt'])
        truncated_prompt = (html.escape(item['prompt'][:97]) + '...') if len(item['prompt']) > 100 else full_prompt
        prompt_cell = f'<div class="prompt-cell" title="{full_prompt}">{truncated_prompt}</div>'
        
        start_img_data = item.get('start_image_data_base64') or [None]
        start_img_uri = start_img_data[0]
        start_img_labels = item.get('start_image_labels', [''])
        
        end_img_data = item.get('end_image_data_base64') or [None]
        end_img_uri = end_img_data[0]
        end_img_labels = item.get('end_image_labels', [''])
        
        num_steps = item.get('steps')
        length = item.get('length')
        
        start_img_md = ""
        if start_img_uri:
            start_img_md = f'<div class="hover-image" onclick="showImageModal(\'start_{row_index}\')"><img src="{start_img_uri}" alt="{start_img_labels[0]}" /></div>'
            
        end_img_md = ""
        if end_img_uri:
            end_img_md = f'<div class="hover-image" onclick="showImageModal(\'end_{row_index}\')"><img src="{end_img_uri}" alt="{end_img_labels[0]}" /></div>'

        edit_btn = "" if _is_edit_task_params(item.get("params", {})) else f"""<button onclick="updateAndTrigger('edit_{task_id}')" class="action-button" title="Edit"><img src="/gradio_api/file=icons/edit.svg" style="width: 20px; height: 20px;"></button>"""
        remove_btn = f"""<button onclick="updateAndTrigger('remove_{task_id}')" class="action-button" title="Remove"><img src="/gradio_api/file=icons/remove.svg" style="width: 20px; height: 20px;"></button>"""

        row_class = "draggable-row"
        row_style = ""
        
        if scheme == "pastel":
            hue = (task_id * 137.508 + 22241) % 360
            row_class += " pastel-row"
            row_style = f'--item-hue: {hue:.0f};'
        else:
            row_class += " alternating-grey-row"
            if row_index % 2 == 0:
                row_class += " even-row"
                
        row_html = f"""
        <tr draggable="true" class="{row_class}" data-index="{row_index}" style="{row_style}" title="Drag to reorder">
            <td class="center-align">{item.get('repeats', "1")}</td>
            <td>{prompt_cell}</td>
            <td class="center-align">{length}</td>
            <td class="center-align">{num_steps}</td>
            <td class="center-align">{start_img_md}</td>
            <td class="center-align">{end_img_md}</td>
            <td class="center-align">{edit_btn}</td>
            <td class="center-align">{remove_btn}</td>
        </tr>
        """
        table_rows.append(row_html)
        
    table_footer = "</tbody></table>"
    table_html = table_header + "".join(table_rows) + table_footer
    scrollable_div = f'<div id="queue-scroll-container" style="max-height: 650px; overflow-y: auto;">{table_html}</div>'

    return top_button_html + scrollable_div + bottom_button_html

def update_queue_data(queue):
    update_global_queue_ref(queue)
    html_content = generate_queue_html(queue)
    return gr.HTML(value=html_content)


def create_html_progress_bar(percentage=0.0, text="Idle", is_idle=True):
    bar_class = "progress-bar-custom idle" if is_idle else "progress-bar-custom"
    bar_text_html = f'<div class="progress-bar-text">{text}</div>'

    html = f"""
    <div class="progress-container-custom">
        <div class="{bar_class}" style="width: {percentage:.1f}%;" role="progressbar" aria-valuenow="{percentage:.1f}" aria-valuemin="0" aria-valuemax="100">
           {bar_text_html}
        </div>
    </div>
    """
    return html

def update_generation_status(html_content):
    if(html_content):
        return gr.update(value=html_content)

family_handlers = ["models.wan.wan_handler", "models.wan.ovi_handler", "models.wan.df_handler", "models.hyvideo.hunyuan_handler", "models.ltx_video.ltxv_handler", "models.ltx2.ltx2_handler", "models.ltx2.ltx_audio_tts_handler", "models.longcat.longcat_handler", "models.minimax_h3.minimax_h3_handler", "models.flux.flux_handler", "models.qwen.qwen_handler", "models.ming_image.ming_handler", "models.kandinsky5.kandinsky_handler",  "models.z_image.z_image_handler", "models.hidream.hidream_handler", "models.ideogram4.ideogram4_handler", "models.krea2.krea2_handler", "models.magi_human.magi_human_handler", "models.sensenova_u1.sensenova_u1_handler",  "models.TTS.ace_step_handler", "models.TTS.chatterbox_handler", "models.TTS.qwen3_handler", "models.TTS.heartmula_handler", "models.TTS.kugelaudio_handler", "models.TTS.index_tts2_handler", "models.TTS.stable_audio3_handler", "models.TTS.omnivoice_handler", "models.TTS.minimax_music3.minimax_music3_handler", "models.TTS.auk.auk_handler", "models.TTS.yue2.yue2_handler"]
DEFAULT_LORA_ROOT = "loras" #"models.cosmos3.cosmos3_handler",

def get_lora_root():
    cli_lora_root = getattr(args, "loras", "")
    if isinstance(cli_lora_root, str):
        cli_lora_root = cli_lora_root.strip()
    config_lora_root = None
    if "server_config" in globals():
        config_lora_root = server_config.get("loras_root", DEFAULT_LORA_ROOT)
    lora_root = cli_lora_root or config_lora_root or DEFAULT_LORA_ROOT
    return lora_root

def get_lora_dir(model_type):
    base_model_type = get_base_model_type(model_type)
    if base_model_type is None:
        raise Exception("loras unknown")

    handler = get_model_handler(model_type)
    get_dir = getattr(handler, "get_lora_dir", None)
    if get_dir is None:
        raise Exception("loras unknown")

    try:
        inspect.signature(get_dir).bind(base_model_type)
    except TypeError:
        # Legacy handlers return a resolved path, not a LoRA config key.
        lora_dir = get_dir(base_model_type, args, get_lora_root())
        if lora_dir is None: raise Exception("loras unknown")
        return lora_dir
    lora_key = get_dir(base_model_type)
    if lora_key is None: raise Exception("loras unknown")
    return resolve_lora_dir(lora_key, get_lora_root(), args.lora_config)

attention_modes_installed = get_attention_modes()
attention_modes_supported = get_supported_attention_modes()
override_attention_modes_installed = get_override_attention_modes()
override_attention_modes_supported = get_supported_override_attention_modes()
args = parse_wgp_args(CONFIG_FILENAME)
migrate_loras_layout()

gpu_major, gpu_minor = torch.cuda.get_device_capability(args.gpu if len(args.gpu) > 0 else None)
if  gpu_major < 8:
    print("Switching to FP16 models when possible as GPU architecture doesn't support optimed BF16 Kernels")
    bfloat16_supported = False
else:
    bfloat16_supported = True

args.flow_reverse = True
processing_device = args.gpu
if len(processing_device) == 0:
    processing_device = "mps" if is_mps else "cuda"
# torch.backends.cuda.matmul.allow_fp16_accumulation = True
lock_ui_attention = False
lock_ui_transformer = False
lock_ui_compile = False

force_profile_no = float(args.profile)
verbose_level = int(args.verbose)
check_loras = args.check_loras ==1
ui_perf_debug = bool(args.debug_gen_form or os.getenv("WANGP_DEBUG_UI", "").strip().lower() in {"1", "true", "yes", "on"})
model_dropdowns.MODEL_SELECTOR_DEBUG = ui_perf_debug

with open("models/_settings.json", "r", encoding="utf-8") as f:
    primary_settings = json.load(f)

wgp_root = os.path.abspath(os.getcwd())
deepy_session_store.configure_sessions_root(args.deepy_sessions_dir or os.path.join(wgp_root, deepy_session_store.DEFAULT_SESSIONS_FOLDER))
config_dir = args.config.strip()
server_config_filename = CONFIG_FILENAME
server_config_fallback = server_config_filename
if config_dir:
    config_dir = os.path.abspath(config_dir)
    os.makedirs(config_dir, exist_ok=True)
    server_config_filename = os.path.join(config_dir, CONFIG_FILENAME)
    server_config_fallback = os.path.join(wgp_root, CONFIG_FILENAME)
    AUTOSAVE_PATH = os.path.join(config_dir, AUTOSAVE_FILENAME)
    AUTOSAVE_TEMPLATE_PATH = os.path.join(wgp_root, AUTOSAVE_FILENAME)
else:
    AUTOSAVE_PATH = AUTOSAVE_FILENAME
    AUTOSAVE_TEMPLATE_PATH = AUTOSAVE_FILENAME

if not os.path.isdir("settings"):
    os.mkdir("settings")
if os.path.isfile("t2v_settings.json"):
    for f in glob.glob(os.path.join(".", "*_settings.json*")):
        target_file = os.path.join("settings",  Path(f).parts[-1])
        shutil.move(f, target_file)

config_load_filename = server_config_filename
if config_dir and not Path(server_config_filename).is_file():
    if Path(server_config_fallback).is_file():
        config_load_filename = server_config_fallback

src_move = [ "ltx-2-19b-dev-fp4_diffusion_model.safetensors" ]
tgt_move = [ "ltx-2-19b-dev-nvfp4_diffusion_model.safetensors" ]
for src_name, tgt_name in zip(src_move, tgt_move):
    src = fl.locate_file(src_name, error_if_none=False)
    if src is not None:
        tgt = os.path.join(os.path.dirname(src), tgt_name)
        try:
            if os.path.isfile(tgt):
                os.remove(src)
            else:
                os.replace(src, tgt)
        except:
            pass
    

if not Path(config_load_filename).is_file():
    server_config = {
        "attention_mode" : "auto",  
        "transformer_types": [], 
        "transformer_quantization": "int8",
        "text_encoder_quantization" : "int8",
        "lm_decoder_engine": "",
        "save_path": "outputs",  
        "image_save_path": "outputs",  
        "compile" : "",
        "metadata_type": "metadata",
        "boost" : 1,
        "int8_kernels": "auto",
        "kernel_precision": "fast",
        "clear_file_list" : 5,
        "keep_intermediate_sliding_windows": 1,
        "keep_resolution_on_model_switch": True,
        "enable_4k_resolutions": 0,
        "max_reserved_loras": -1,
        "vae_config": 0,
        "profile" : profile_type.LowRAM_LowVRAM,
        "video_profile": profile_type.LowRAM_LowVRAM,
        "image_profile": profile_type.LowRAM_LowVRAM,
        "audio_profile": 3.5,
        "preload_model_policy": [],
        "UI_theme": "default",
        "floating_generate_button": True,
        "checkpoints_paths": fl.default_checkpoints_paths,
        "loras_root": DEFAULT_LORA_ROOT,
        "save_queue_if_crash": 1,
        "queue_color_scheme": "pastel",
        "process_queues_when_browser_unfocused": 1,
        "multi_prompts_gen_type": prompt_parser.DEFAULT_MULTI_PROMPTS_MODE,
        "model_hierarchy_type": 1,
        upsampler_api.UPSAMPLER_CONFIG_KEY: upsampler_api.default_config_sections(),
        audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY: audio_processor_api.default_config_sections(),
        temporal_upsampler_api.TEMPORAL_UPSAMPLER_CONFIG_KEY: temporal_upsampler_api.default_config_sections(),
        **get_deepy_default_runtime_config(),
        LLM_CONFIG_KEY: normalize_llm_config({}),
        "prompt_enhancer_quantization": "quanto_int8",
        PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY: PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT,
        "prompt_enhancer_temperature": 0.6,
        "prompt_enhancer_top_p": 0.9,
        "prompt_enhancer_randomize_seed": True,
        "audio_save_path": "outputs",
        **notifications.default_config(),
    }
    apply_first_launch_deepy_prime_defaults(server_config)

    write_config(server_config, server_config_filename)
else:
    server_config = read_config(config_load_filename)

server_config.setdefault("prompt_enhancer_quantization", "quanto_int8")
notifications.apply_defaults(server_config)
server_config.setdefault(PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT)
server_config[LLM_CONFIG_KEY] = normalize_llm_config(server_config)
server_config["multi_prompts_gen_type"] = prompt_parser.normalize_multi_prompts_mode(
    server_config.get("multi_prompts_gen_type", prompt_parser.DEFAULT_MULTI_PROMPTS_MODE),
    default=prompt_parser.DEFAULT_MULTI_PROMPTS_MODE,
)
primary_settings["multi_prompts_gen_type"] = server_config["multi_prompts_gen_type"]
server_config.setdefault(gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY, 1)
gradio_queue_focus_patch.BACKGROUND_SCHEDULER_DEFAULT_ENABLED = bool(server_config.get(gradio_queue_focus_patch.FOCUS_QUEUE_SERVER_CONFIG_KEY, 1))
gradio_queue_focus_patch.install()
gradio_model_switch_patch.install(verbose=ui_perf_debug)
gradio_startup_patch.install()
gradio_event_context_patch.install()
deepy_voice.install_gradio_routes(get_service=lambda: _deepy_hybrid, language=args.deepy_voice_language)

checkpoints_paths = server_config.get("checkpoints_paths", None)
if checkpoints_paths is None: checkpoints_paths = server_config["checkpoints_paths"] = fl.default_checkpoints_paths
fl.set_checkpoints_paths(checkpoints_paths)
three_levels_hierarchy = server_config.get("model_hierarchy_type", 1) == 1

if app is None:
    app = WAN2GPApplication()
app.plugin_manager.set_server_config(server_config, server_config_filename)
plugin_model_extensions = app.plugin_manager.discover_plugin_model_extensions(server_config.get("enabled_plugins", []))
model_handler_sources = {path: {"profile_roots": ["profiles"], "plugin_id": ""} for path in family_handlers}
model_definition_sources = [{"defaults_root": "defaults", "plugin_id": ""}]
for extension in plugin_model_extensions:
    model_definition_sources.append({"defaults_root": extension.defaults_root, "plugin_id": extension.plugin_id})
    for handler_path in extension.model_handlers:
        if handler_path not in family_handlers:
            family_handlers.append(handler_path)
        model_handler_sources[handler_path] = {"profile_roots": [extension.profiles_root, "profiles"], "plugin_id": extension.plugin_id}

from shared.utils import offload_registry

upsampler_api.register_spatial_upsamplers(server_config, fl)
audio_processor_api.register_audio_processors(server_config, fl)
temporal_upsampler_api.register_temporal_upsamplers(server_config, fl)
migrate_extension_defaults(server_config, server_config_filename)

def _normalize_profile_defaults(config):
    if "profile" not in config:
        config["profile"] = profile_type.LowRAM_LowVRAM
    base_profile = config.get("profile", profile_type.LowRAM_LowVRAM)
    config.setdefault("video_profile", base_profile)
    config.setdefault("image_profile", base_profile)
    config.setdefault("audio_profile", 3.5)
    return config["video_profile"], config["image_profile"], config["audio_profile"]

def _normalize_output_paths(config):
    if "save_path" not in config:
        config["save_path"] = "outputs"
    if "image_save_path" not in config:
        config["image_save_path"] = config["save_path"]
    if "audio_save_path" not in config:
        config["audio_save_path"] = config["save_path"]

_normalize_profile_defaults(server_config)
_normalize_output_paths(server_config)
lm_decoder_engine = server_config.get("lm_decoder_engine", "")

from preprocessing.matanyone.utils.model_assets import migrate_matanyone_install
migration_note = migrate_matanyone_install(server_config)
if migration_note:
    print(migration_note)

#   Deprecated models
for path in  ["wan2.1_Vace_1.3B_preview_bf16.safetensors", "sky_reels2_diffusion_forcing_1.3B_bf16.safetensors","sky_reels2_diffusion_forcing_720p_14B_bf16.safetensors",
"sky_reels2_diffusion_forcing_720p_14B_quanto_int8.safetensors", "sky_reels2_diffusion_forcing_720p_14B_quanto_fp16_int8.safetensors", "wan2.1_image2video_480p_14B_bf16.safetensors", "wan2.1_image2video_480p_14B_quanto_int8.safetensors",
"wan2.1_image2video_720p_14B_quanto_int8.safetensors", "wan2.1_image2video_720p_14B_quanto_fp16_int8.safetensors", "wan2.1_image2video_720p_14B_bf16.safetensors",
"wan2.1_text2video_14B_bf16.safetensors", "wan2.1_text2video_14B_quanto_int8.safetensors",
"wan2.1_Vace_14B_mbf16.safetensors", "wan2.1_Vace_14B_quanto_mbf16_int8.safetensors", "wan2.1_FLF2V_720p_14B_quanto_int8.safetensors", "wan2.1_FLF2V_720p_14B_bf16.safetensors",  "wan2.1_FLF2V_720p_14B_fp16.safetensors", "wan2.1_Vace_1.3B_mbf16.safetensors", "wan2.1_text2video_1.3B_bf16.safetensors",
"ltxv_0.9.7_13B_dev_bf16.safetensors", "ltx-2-19b-distilled-fp8.safetensors", "ltx-2-19b-dev-fp8.safetensors", "ltx-2-19b-distilled.safetensors", "ltx-2-19b-dev.safetensors"
]:
    if fl.locate_file(path, error_if_none= False) is not None:
        print(f"Removing old version of model '{path}'. A new version of this model will be downloaded next time you use it.")
        os.remove( fl.locate_file(path))

models_def = {}


# only needed for imported old settings files
model_signatures = {"t2v": "text2video_14B", "t2v_1.3B" : "text2video_1.3B",   "fun_inp_1.3B" : "Fun_InP_1.3B",  "fun_inp" :  "Fun_InP_14B", 
                    "i2v" : "image2video_480p", "i2v_720p" : "image2video_720p" , "vace_1.3B" : "Vace_1.3B", "vace_14B": "Vace_14B", "recam_1.3B": "recammaster_1.3B", 
                    "sky_df_1.3B" : "sky_reels2_diffusion_forcing_1.3B", "sky_df_14B" : "sky_reels2_diffusion_forcing_14B", 
                    "sky_df_720p_14B" : "sky_reels2_diffusion_forcing_720p_14B",
                    "phantom_1.3B" : "phantom_1.3B", "phantom_14B" : "phantom_14B", "ltxv_13B" : "ltxv_0.9.7_13B_dev", "ltxv_13B_distilled" : "ltxv_0.9.7_13B_distilled", 
                    "hunyuan" : "hunyuan_video_720", "hunyuan_i2v" : "hunyuan_video_i2v_720", "hunyuan_custom" : "hunyuan_video_custom_720", "hunyuan_custom_audio" : "hunyuan_video_custom_audio", "hunyuan_custom_edit" : "hunyuan_video_custom_edit",
                    "hunyuan_avatar" : "hunyuan_video_avatar"  }


def map_family_handlers(family_handlers):
    base_types_handlers, families_infos, models_eqv_map, models_comp_map = {}, {"unknown": (100, "Unknown")}, {}, {}
    for path in family_handlers:
        handler = importlib.import_module(path).family_handler
        handler_source = model_handler_sources.get(path, {})
        profile_roots = handler_source.get("profile_roots", ["profiles"])
        plugin_id = handler_source.get("plugin_id", "")
        for model_type in handler.query_supported_types():
            if model_type in base_types_handlers:
                prev = base_types_handlers[model_type].__name__
                raise Exception(f"Model type {model_type} supported by {prev} and {handler.__name__}")
            base_types_handlers[model_type] = handler
            model_profile_roots_by_architecture[model_type] = list(profile_roots)
            if plugin_id:
                model_plugin_ids_by_architecture[model_type] = plugin_id
        families_infos.update(handler.query_family_infos())
        eq_map, comp_map = handler.query_family_maps()
        models_eqv_map.update(eq_map); models_comp_map.update(comp_map)
    return base_types_handlers, families_infos, models_eqv_map, models_comp_map

# models_eqv_map: bidirectional compatibility between base model types
# models_comp_map : mono directional compatibility between base model types {"A" : {"B", "C"} } means B & C model types can accept A  model types (but not the other way). Said otherwise B & C are derived data types

model_profile_roots_by_architecture = {}
model_plugin_ids_by_architecture = {}

model_types_handlers, families_infos,  models_eqv_map, models_comp_map = map_family_handlers(family_handlers)

def collect_prompt_helper_assets(asset_name):
    chunks = []
    for handler in dict.fromkeys(model_types_handlers.values()):
        getter = getattr(handler, asset_name, None)
        if getter is not None:
            chunk = getter()
            if chunk:
                chunks.append(str(chunk))
    return "\n".join(chunks)

def _store_model_metadata(model_type, model_def):
    return model_metadata.store_metadata(model_type, model_def, model_types_handlers, families_infos)

def list_model_defs(family=None, base_model_type=None, finetune=None, model_type=None, main_output=None, inputs=None):
    return model_metadata.list_model_defs(models_def, family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs)

def get_model_defs(**filters):
    return list_model_defs(**filters)

def get_base_model_type(model_type):
    model_def = get_model_def(model_type)
    if model_def == None:
        return model_type if model_type in model_types_handlers else None 
        # return model_type
    else:
        return model_def["architecture"]

def get_parent_model_type(model_type):
    base_model_type =  get_base_model_type(model_type)
    if base_model_type is None: return None
    model_def = get_model_def(base_model_type)
    return model_def.get("parent_model_type", base_model_type)
    
def get_model_handler(model_type):
    base_model_type = get_base_model_type(model_type)
    if base_model_type is None:
        raise Exception(f"Unknown model type {model_type}")
    model_handler = model_types_handlers.get(base_model_type, None)
    if model_handler is None:
        raise Exception(f"No model handler found for base model type {base_model_type}")
    return model_handler

def are_model_types_compatible(imported_model_type, current_model_type):
    imported_base_model_type = get_base_model_type(imported_model_type)
    curent_base_model_type = get_base_model_type(current_model_type)

    if imported_base_model_type in models_eqv_map:
        imported_base_model_type = models_eqv_map[imported_base_model_type]

    if curent_base_model_type in models_eqv_map:
        curent_base_model_type = models_eqv_map[curent_base_model_type]

    if imported_base_model_type == curent_base_model_type:
        return True

    comp_list=  models_comp_map.get(imported_base_model_type, None)
    if comp_list == None: return False
    return curent_base_model_type in comp_list 

def get_model_def(model_type):
    return models_def.get(model_type, None )


extra_settings.configure(get_model_def=get_model_def)


def get_model_type(model_filename):
    for model_type, signature in model_signatures.items():
        if signature in model_filename:
            return model_type
    return None
    # raise Exception("Unknown model:" + model_filename)

def get_model_family(model_type, for_ui = False):
    base_model_type = get_base_model_type(model_type)
    if base_model_type is None:
        return "unknown"
    model_def = get_model_def(model_type) or {}
    return model_metadata.get_model_family(base_model_type, model_def, model_types_handlers, families_infos, for_ui=for_ui)

def test_class_i2v(model_type):    
    model_def = get_model_def(model_type)
    return model_def.get("i2v_class", False)

def test_vace_module(model_type):
    model_def = get_model_def(model_type)
    return model_def.get("vace_class", False)

def test_class_t2v(model_type):
    model_def = get_model_def(model_type)
    return model_def.get("t2v_class", False)

def image_prompt_types_allow_t2v(model_def, image_mode=0):
    return int(image_mode or 0) == 0 and "T" in model_def.get("image_prompt_types_allowed", "")

def get_guide_custom_choices(model_def, image_mode=0):
    if int(image_mode or 0) > 0 and model_def.get("guide_custom_choices_image", None) is not None:
        return model_def["guide_custom_choices_image"]
    return model_def.get("guide_custom_choices", None)

def frame_scheduler_supported(model_type, model_def, image_mode=0, is_edit_mode=False):
    return not is_edit_mode and int(image_mode or 0) == 0 and not model_def.get("audio_only", False) and test_any_sliding_window(model_type)

def test_any_sliding_window(model_type):
    model_def = get_model_def(model_type)
    if model_def is None:
        return False
    return model_def.get("sliding_window", False)

def get_model_min_frames_and_step(model_type):
    model_def = get_model_def(model_type)
    frames_minimum = model_def.get("frames_minimum", 5)
    frames_steps = model_def.get("frames_steps", 4)
    latent_size = model_def.get("latent_size", frames_steps)
    return frames_minimum, frames_steps, latent_size 

def get_model_fps(model_type):
    model_def = get_model_def(model_type)
    fps= model_def.get("fps", 16)
    return fps

def get_computed_fps(force_fps, base_model_type , video_guide, video_source ):
    if force_fps == "auto":
        if video_source != None:
            fps,  _, _, _ = get_video_info(video_source)
        elif video_guide != None:
            fps,  _, _, _ = get_video_info(video_guide)
        else:
            fps = get_model_fps(base_model_type)
    elif force_fps == "control" and video_guide != None:
        fps,  _, _, _ = get_video_info(video_guide)
    elif force_fps == "source" and video_source != None:
        fps,  _, _, _ = get_video_info(video_source)
    elif len(force_fps) > 0 and is_integer(force_fps) :
        fps = int(force_fps)
    else:
        fps = get_model_fps(base_model_type)
    return fps

def get_model_name(model_type, description_container = [""]):
    model_def = get_model_def(model_type)
    if model_def == None: 
        return f"Unknown model {model_type}"
    model_name = model_def["name"]
    description = model_def["description"]
    description_container[0] = description
    return model_name

def get_model_record(model_name):
    return f"WanGP v{WanGP_version} by DeepBeepMeep - " +  model_name

def get_model_recursive_prop(model_type, prop = "URLs", sub_prop_name = None, return_list = True, stack = [], model_def = None):
    if model_def is None: model_def = models_def.get(model_type, None)
    if model_def != None: 
        prop_value = model_def.get(prop, None)
        if prop_value == None:
            return []
        if sub_prop_name is not None:
            if sub_prop_name == "_list":
                if not isinstance(prop_value,list) or len(prop_value) != 1:
                    raise Exception(f"Sub property value for property {prop} of model type {model_type} should be a list of size 1")
                prop_value = prop_value[0]
            else:
                if not isinstance(prop_value,dict) and not sub_prop_name in prop_value:
                    raise Exception(f"Invalid sub property value {sub_prop_name} for property {prop} of model type {model_type}")
                prop_value = prop_value[sub_prop_name]
        if isinstance(prop_value, str):
            if len(stack) > 10: raise Exception(f"Circular Reference in Model {prop} dependencies: {stack}")
            return get_model_recursive_prop(prop_value, prop = prop, sub_prop_name =sub_prop_name, stack = stack + [prop_value] )
        else:
            return prop_value
    else:
        if model_type in model_types:
            return [] if return_list else model_type 
        else:
            raise Exception(f"Unknown model type '{model_type}'")


def get_model_config_groups(model_type, model_def=None):
    return [get_model_recursive_prop(model_type, key, return_list=False, model_def=model_def) or {} for key in model_config_groups.CONFIG_KEYS]


def get_model_filename(model_type, quantization ="int8", dtype_policy = "", module_type = None, submodel_no = 1, URLs = None, stack=[], model_def = None):
    if URLs is not None:
        pass
    elif module_type is not None:
        base_model_type = get_base_model_type(model_type) 
        # model_type_handler = model_types_handlers[base_model_type]
        # modules_files = model_type_handler.query_modules_files() if hasattr(model_type_handler, "query_modules_files") else {}
        if isinstance(module_type, list):
            URLs = module_type
        else:
            if "#" not in module_type:
                sub_prop_name = "_list"
            else:
                pos = module_type.rfind("#")
                sub_prop_name =  module_type[pos+1:]
                module_type = module_type[:pos]  
            URLs = get_model_recursive_prop(module_type, "modules", sub_prop_name =sub_prop_name, return_list= False)

        # choices = modules_files.get(module_type, None)
        # if choices == None: raise Exception(f"Invalid Module Id '{module_type}'")
    else:
        key_name = "URLs" if submodel_no  <= 1 else f"URLs{submodel_no}"

        if model_def is None: model_def = models_def.get(model_type, None)
        if model_def == None: return ""
        URLs = model_def.get(key_name, [])
        if isinstance(URLs, str):
            if len(stack) > 10: raise Exception(f"Circular Reference in Model {key_name} dependencies: {stack}")
            return get_model_filename(URLs, quantization=quantization, dtype_policy=dtype_policy, submodel_no = submodel_no, stack = stack + [URLs])

    choices = URLs if isinstance(URLs, list) else [URLs]
    if len(choices) == 0:
        return ""
    if len(quantization) == 0:
        quantization = "bf16"

    dtype = get_transformer_dtype(model_type, dtype_policy)
    if len(choices) <= 1:
        raw_filename = choices[0]
    else:
        quant_tokens = []
        quant_order = []
        if quantization != "bf16":
            if quantization == "int8":
                quant_order =["int8", "fp8"]
            elif quantization == "fp8":
                quant_order =["fp8", "int8"]

        for quant_type in quant_order:
            quant_tokens += quant_router.get_quantization_tokens(quant_type) or []
        sub_choices = []
        for token in quant_tokens:
            sub_choices += [name for name in choices if token in os.path.basename(name).lower()]

        if len(sub_choices) > 0:
            dtype_str = "fp16" if dtype == torch.float16 else "bf16"
            new_sub_choices = [ name for name in sub_choices if dtype_str in os.path.basename(name) or dtype_str.upper() in os.path.basename(name)]
            sub_choices = new_sub_choices if len(new_sub_choices) > 0 else sub_choices
            raw_filename = sub_choices[0]
        else:
            raw_filename = choices[0]

    return raw_filename

def get_transformer_dtype(model_type, transformer_dtype_policy):
    base_model_type = get_base_model_type(model_type)
    model_def = get_model_def(base_model_type)
    dtype = model_def.get("dtype", None)
    if dtype is not None: 
        return torch.float16 if dtype =="fp16" else torch.bfloat16
    model_family =  get_model_family(base_model_type) 
    if not isinstance(transformer_dtype_policy, str):
        return transformer_dtype_policy
    if len(transformer_dtype_policy) == 0:
        if not bfloat16_supported:
            return torch.float16
        else:
            if model_family == "wan"and False:
                return torch.float16
            else: 
                return torch.bfloat16
        return transformer_dtype
    elif transformer_dtype_policy =="fp16":
        return torch.float16
    else:
        return torch.bfloat16

def get_settings_file_name(model_type):
    return  os.path.join(args.settings, model_type + "_settings.json")

def fix_postprocess_audio_settings(ui_defaults, settings_version):
    return audio_processor_api.fix_settings(ui_defaults, settings_version, attachment_has_path_values=_attachment_has_path_values)

def fix_settings(model_type, ui_defaults, min_settings_version = 0):
    if model_type is None: return

    settings_version =  max(min_settings_version, ui_defaults.get("settings_version", 0))
    if settings_version < 2.79:
        if "spatial_upsampler_face_count" in ui_defaults:
            ui_defaults["spatial_upsampler_param"] = ui_defaults.pop("spatial_upsampler_face_count")
        parameters = ui_defaults.get("spatial_upsampler_parameters", {})
        if "spatial_upsampler_face_count" in parameters:
            parameters["spatial_upsampler_param"] = parameters.pop("spatial_upsampler_face_count")
        ui_defaults.setdefault("spatial_upsampler_param2", None)
    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)
    duration_conversion = apply_video_length_duration(ui_defaults, model_def)
    if duration_conversion is not None:
        print(f"WanGP settings converted video_length={duration_conversion['input']!r} to {duration_conversion['frames']} frames at {duration_conversion['fps']:g} fps for {model_type}")

    prompts = ui_defaults.get("prompts", "")
    if len(prompts) > 0:
        ui_defaults["prompt"] = prompts
    image_prompt_type = ui_defaults.get("image_prompt_type", None)
    if image_prompt_type != None :
        if not isinstance(image_prompt_type, str):
            image_prompt_type = "S" if image_prompt_type  == 0 else "SE"
        if settings_version <= 2:
            image_prompt_type = image_prompt_type.replace("G","")
        ui_defaults["image_prompt_type"] = image_prompt_type

    if "alt_prompt" not in ui_defaults:
        ui_defaults["alt_prompt"] = ""

    if "lset_name" in ui_defaults: del ui_defaults["lset_name"]

    if settings_version < 2.54:
        renamed_settings = {
            "slg_switch": "perturbation_switch",
            "slg_layers": "perturbation_layers",
            "slg_start_perc": "perturbation_start_perc",
            "slg_end_perc": "perturbation_end_perc",
        }
        for old_name, new_name in renamed_settings.items():
            if old_name in ui_defaults:
                ui_defaults.setdefault(new_name, ui_defaults[old_name])
                del ui_defaults[old_name]

    if settings_version < 2.55:
        ui_defaults.setdefault("alt_scale", 0.0)

    if settings_version < 2.56:
        legacy_multi_prompts_mode = ui_defaults.get("multi_prompts_gen_type", None)
        if legacy_multi_prompts_mode is None:
            ui_defaults["multi_prompts_gen_type"] = server_config["multi_prompts_gen_type"]
        else:
            ui_defaults["multi_prompts_gen_type"] = prompt_parser.normalize_multi_prompts_mode(legacy_multi_prompts_mode, default=server_config["multi_prompts_gen_type"])
    else:
        ui_defaults["multi_prompts_gen_type"] = prompt_parser.normalize_multi_prompts_mode(ui_defaults.get("multi_prompts_gen_type", server_config["multi_prompts_gen_type"]), default=server_config["multi_prompts_gen_type"])
    if not test_any_sliding_window(model_type):
        ui_defaults["multi_prompts_gen_type"] = ui_defaults["multi_prompts_gen_type"].replace("W", "G")

    if settings_version < 2.60:
        ui_defaults.setdefault("replace_voice_sample", None)

    if settings_version < 2.61:
        prompt_enhancer = str(ui_defaults.get("prompt_enhancer") or "")
        if prompt_enhancer == "I":
            ui_defaults["prompt_enhancer"] = "TI"
        elif prompt_enhancer == "IK":
            ui_defaults["prompt_enhancer"] = "TIK"

    audio_prompt_type = fix_postprocess_audio_settings(ui_defaults, settings_version)
    if settings_version < 2.78:
        if "spatial_upsampling" in ui_defaults:
            ui_defaults["spatial_upsampling"] = upsampler_api.normalize_upsampling_value(ui_defaults["spatial_upsampling"])
        if "temporal_upsampling" in ui_defaults:
            ui_defaults["temporal_upsampling"] = temporal_upsampler_api.normalize_temporal_upsampling_value(ui_defaults["temporal_upsampling"])
    if settings_version < 2.2: 
        if audio_prompt_type == None :
            if any_audio_track(base_model_type):
                audio_prompt_type ="A"
                ui_defaults["audio_prompt_type"] = audio_prompt_type

    if settings_version < 2.35 and any_audio_track(base_model_type): 
        audio_prompt_type = audio_prompt_type or ""
        audio_prompt_type += "V"
        ui_defaults["audio_prompt_type"] = audio_prompt_type

    video_prompt_type = ui_defaults.get("video_prompt_type", "")

    if base_model_type in ["hunyuan"]:
        video_prompt_type = video_prompt_type.replace("I", "")

    if base_model_type in ["flux"] and settings_version < 2.23:
        video_prompt_type = video_prompt_type.replace("K", "").replace("I", "KI")

    remove_background_images_ref = ui_defaults.get("remove_background_images_ref", None)
    if settings_version < 2.22:
        if "I" in video_prompt_type:
            if remove_background_images_ref == 2:
                video_prompt_type = video_prompt_type.replace("I", "KI")
        if remove_background_images_ref != 0:
            remove_background_images_ref = 1
    if base_model_type in ["hunyuan_avatar"]: 
        remove_background_images_ref = 0
        if settings_version < 2.26:
            if not "K" in video_prompt_type: video_prompt_type = video_prompt_type.replace("I", "KI")
    if remove_background_images_ref is not None:
        ui_defaults["remove_background_images_ref"] = remove_background_images_ref

    ui_defaults["video_prompt_type"] = video_prompt_type

    tea_cache_setting = ui_defaults.get("tea_cache_setting", None)
    tea_cache_start_step_perc = ui_defaults.get("tea_cache_start_step_perc", None)

    if tea_cache_setting != None:
        del ui_defaults["tea_cache_setting"]
        if tea_cache_setting > 0:
            ui_defaults["skip_steps_multiplier"] = tea_cache_setting
            ui_defaults["skip_steps_cache_type"] = "tea"
        else:
            ui_defaults["skip_steps_multiplier"] = 1.75
            ui_defaults["skip_steps_cache_type"] = ""

    if tea_cache_start_step_perc != None:
        del ui_defaults["tea_cache_start_step_perc"]
        ui_defaults["skip_steps_start_step_perc"] = tea_cache_start_step_perc

    image_prompt_type = ui_defaults.get("image_prompt_type", "")
    if len(image_prompt_type) > 0:
        image_prompt_types_allowed = model_def.get("image_prompt_types_allowed","")
        image_prompt_type = filter_letters(image_prompt_type, image_prompt_types_allowed)
    ui_defaults["image_prompt_type"] = image_prompt_type

    video_prompt_type = ui_defaults.get("video_prompt_type", "")
    image_ref_choices_list = model_def.get("image_ref_choices", {}).get("choices", [])
    if get_guide_custom_choices(model_def, ui_defaults.get("image_mode", 0)) is None:
        if len(image_ref_choices_list)==0:
            video_prompt_type = del_in_sequence(video_prompt_type, "IK")
        else:
            first_choice = image_ref_choices_list[0][1]
            if "I" in first_choice and not "I" in video_prompt_type: video_prompt_type += "I"
            if len(image_ref_choices_list)==1 and "K" in first_choice and not "K" in video_prompt_type: video_prompt_type += "K"
        ui_defaults["video_prompt_type"] = video_prompt_type

    model_handler = get_model_handler(base_model_type)
    if hasattr(model_handler, "fix_settings"):
        model_handler.fix_settings(base_model_type, settings_version, model_def, ui_defaults)

    ui_defaults["settings_version"] = settings_version

def get_default_prompt(i2v):
    if i2v:
        return "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field."
    return "A large orange octopus is seen resting on the bottom of the ocean floor, blending in with the sandy and rocky terrain. Its tentacles are spread out around its body, and its eyes are closed. The octopus is unaware of a king crab that is crawling towards it from behind a rock, its claws raised and ready to attack. The crab is brown and spiny, with long legs and antennae. The scene is captured from a wide angle, showing the vastness and depth of the ocean. The water is clear and blue, with rays of sunlight filtering through. The shot is sharp and crisp, with a high dynamic range. The octopus and the crab are in focus, while the background is slightly blurred, creating a depth of field effect."


def get_factory_settings(model_type):
    i2v = test_class_i2v(model_type)
    model_def = get_model_def(model_type)
    base_model_type = get_base_model_type(model_type)
    ui_defaults = copy.deepcopy(primary_settings)
    apply_custom_settings_defaults(model_def, ui_defaults)
    ui_defaults.update({
        "settings_version": settings_version,
        "prompt": get_default_prompt(i2v),
        "resolution": "1280x720" if "720" in base_model_type else "832x480",
        "flow_shift": 7.0 if "720" not in base_model_type and i2v else 5.0,
    })
    get_model_handler(model_type).update_default_settings(base_model_type, model_def, ui_defaults)
    model_settings = model_def.get("settings")
    if model_settings is not None:
        ui_defaults.update(copy.deepcopy(model_settings))
    if len(ui_defaults.get("prompt", "")) == 0:
        ui_defaults["prompt"] = get_default_prompt(i2v)
    # needs to implement settings md version for defaults/finetunes
    # fix_settings(model_type, ui_defaults, settings_version)
    return ui_defaults


def get_default_settings(model_type):
    defaults_filename = get_settings_file_name(model_type)
    if not Path(defaults_filename).is_file():
        ui_defaults = get_factory_settings(model_type)
        with open(defaults_filename, "w", encoding="utf-8") as f:
            json.dump(ui_defaults, f, indent=4)
    else:
        with open(defaults_filename, "r", encoding="utf-8") as f:
            ui_defaults = json.load(f)
        fix_settings(model_type, ui_defaults)            
    
    default_seed = args.seed
    if default_seed > -1:
        ui_defaults["seed"] = default_seed
    default_number_frames = args.frames
    if default_number_frames > 0:
        ui_defaults["video_length"] = default_number_frames
    default_number_steps = args.steps
    if default_number_steps > 0:
        ui_defaults["num_inference_steps"] = default_number_steps
    return ui_defaults


def init_model_def(model_type, model_def):
    base_model_type = model_def.get("architecture", None) or get_base_model_type(model_type)
    family_handler = model_types_handlers.get(base_model_type, None)
    if family_handler is None:
        if model_def.get("visible", True):
            print(f"Skipping model type '{model_type}' with unsupported architecture '{base_model_type}'.")
        model_def["visible"] = False
        return model_def
    default_model_def = family_handler.query_model_def(base_model_type, model_def)
    if default_model_def is None: return model_def
    default_model_def.update(model_def)
    default_model_def["_profile_roots"] = model_profile_roots_by_architecture.get(base_model_type, ["profiles"])
    plugin_id = model_plugin_ids_by_architecture.get(base_model_type, "")
    if plugin_id:
        default_model_def["_plugin_id"] = plugin_id
    return _store_model_metadata(model_type, default_model_def)


def refresh_model_defs():
    global models_def, model_types, displayed_model_types
    new_models_def, parse_errors, previous_models_def, old_model_types = {}, [], models_def.copy(), set()
    defaults_paths = set()
    for source in model_definition_sources:
        defaults_root = source["defaults_root"]
        defaults_paths.update(glob.glob(os.path.join(defaults_root, "*.json")))
    models_def_paths = sorted([*defaults_paths, *glob.glob(os.path.join("finetunes", "*.json"))])
    def warn(msg):
        print(msg)
        parse_errors.append(msg)
    def use_previous_model_def(model_type, file_path, error):
        previous_model_def = previous_models_def.get(model_type, None)
        if previous_model_def is None:
            return False
        warn(f"Model Definition File '{file_path}' could not be refreshed; using previous definition for '{model_type}': {str(error)}")
        old_model_types.add(model_type)
        new_models_def[model_type] = previous_model_def
        return True
    for idx, file_path in enumerate(models_def_paths, 1):
        file_start = time.perf_counter()
        model_type = os.path.basename(file_path)[:-5]
        if model_type in old_model_types:
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                json_def = json.load(f)
            except Exception as e:
                if use_previous_model_def(model_type, file_path, e):
                    continue
                elif file_path in defaults_paths:
                    raise Exception(f"Error while parsing Model Definition File '{file_path}': {str(e)}")
                else:
                    warn(f"Finetune Definition File '{file_path}' will be ignored as there was an error in its parsing: {str(e)}")
                    continue
        try:
            model_def = json_def.pop("model")
            model_def["path"] = file_path
            existing_model_def = new_models_def.get(model_type, None)
            if existing_model_def is not None:
                existing_model_def.setdefault("settings", {}).update(json_def)
                existing_model_def.update(model_def)
                _store_model_metadata(model_type, existing_model_def)
            else:
                new_models_def[model_type] = model_def
                model_def = init_model_def(model_type, model_def)
                new_models_def[model_type] = model_def
                model_def["settings"] = json_def
                _store_model_metadata(model_type, model_def)
        except Exception as e:
            if use_previous_model_def(model_type, file_path, e):
                continue
            elif file_path in defaults_paths:
                raise Exception(f"Error while refreshing Model Definition File '{file_path}': {str(e)}")
            else:
                warn(f"Finetune Definition File '{file_path}' will be ignored as there was an error in its refresh: {str(e)}")

    models_def = new_models_def
    model_types = models_def.keys()
    displayed_model_types = [model_type for model_type, model_def in models_def.items() if model_def.get("visible", True)]
    return parse_errors

refresh_model_defs()

transformer_types = server_config.get("transformer_types", [])
new_transformer_types = []
for model_type in transformer_types:
    if get_model_def(model_type) == None:
        print(f"Model '{model_type}' is missing. Either install it in the finetune folder or remove this model from ley 'transformer_types' in {CONFIG_FILENAME}")
    else:
        new_transformer_types.append(model_type)
transformer_types = new_transformer_types
transformer_type = server_config.get("last_model_type", None)
advanced = server_config.get("last_advanced_choice", False)
last_resolution = server_config.get("last_resolution_choice", None)
if args.advanced: advanced = True 

if transformer_type != None and not transformer_type in model_types and not transformer_type in models_def: transformer_type = None
if transformer_type == None:
    transformer_type = transformer_types[0] if len(transformer_types) > 0 else "t2v"

transformer_quantization =server_config.get("transformer_quantization", "int8")

transformer_dtype_policy = server_config.get("transformer_dtype_policy", "")
if args.fp16:
    transformer_dtype_policy = "fp16" 
if args.bf16:
    transformer_dtype_policy = "bf16" 
text_encoder_quantization =server_config.get("text_encoder_quantization", "int8")
attention_mode = server_config["attention_mode"]
if len(args.attention)> 0:
    if args.attention in ["auto", "sdpa", "sage", "sage2", "flash", "xformers"]:
        attention_mode = args.attention
        server_config["attention_mode"] = attention_mode
        lock_ui_attention = True
    else:
        raise Exception(f"Unknown attention mode '{args.attention}'")

default_profile_video = force_profile_no if force_profile_no >= 0 else server_config["video_profile"]
default_profile_image = force_profile_no if force_profile_no >= 0 else server_config["image_profile"]
default_profile_audio = force_profile_no if force_profile_no >= 0 else server_config["audio_profile"]
default_profile = default_profile_video
loaded_profile = force_profile_no = -1
compile = server_config.get("compile", "")
if args.compile:
    compile="transformer"
    lock_ui_compile = True
if is_mps: compile = ""
boost = server_config.get("boost", 1)
int8_kernels = server_config.get("int8_kernels", "auto")
apply_int8_kernel_setting(int8_kernels)
kernel_policy.configure(server_config.get("kernel_precision", "fast"))
vae_config = server_config.get("vae_config", 0)
if len(args.vae_config) > 0:
    vae_config = int(args.vae_config)

reload_needed = False
save_path = server_config.get("save_path", os.path.join(os.getcwd(), "outputs"))
image_save_path = server_config.get("image_save_path", os.path.join(os.getcwd(), "outputs"))
audio_save_path = server_config.get("audio_save_path", save_path)
if not "video_output_codec" in server_config: server_config["video_output_codec"]= "libx264_8"
if not "hdr_video_crf" in server_config: server_config["hdr_video_crf"] = 8
if not "video_container" in server_config: server_config["video_container"]= "mp4"
if not "embed_source_images" in server_config: server_config["embed_source_images"]= False
if not "keep_resolution_on_model_switch" in server_config: server_config["keep_resolution_on_model_switch"]= True
if not "enable_4k_resolutions" in server_config: server_config["enable_4k_resolutions"]= 0
if not "max_reserved_loras" in server_config: server_config["max_reserved_loras"]= -1
if not "image_output_codec" in server_config: server_config["image_output_codec"]= "jpeg_95"
if not "audio_output_codec" in server_config: server_config["audio_output_codec"]= "aac_128"
if not "audio_stand_alone_output_codec" in server_config: server_config["audio_stand_alone_output_codec"]= "wav"
upsampler_api.require_upsampler_by_method("flashvsr").normalize_config()
if "loras_root" not in server_config: server_config["loras_root"] = DEFAULT_LORA_ROOT
if "save_queue_if_crash" not in server_config: server_config["save_queue_if_crash"] = 1
if "keep_intermediate_sliding_windows" not in server_config: server_config["keep_intermediate_sliding_windows"] = 1
if "prompt_enhancer_temperature" not in server_config: server_config["prompt_enhancer_temperature"] = 0.6
if "prompt_enhancer_top_p" not in server_config: server_config["prompt_enhancer_top_p"] = 0.9
if "prompt_enhancer_randomize_seed" not in server_config: server_config["prompt_enhancer_randomize_seed"] = True
set_deepy_runtime_config(server_config, server_config_filename)
if "int8_kernels" not in server_config: server_config["int8_kernels"] = "auto"

preload_model_policy = server_config.get("preload_model_policy", []) 


if args.t2v_14B or args.t2v: 
    transformer_type = "t2v"

if args.i2v_14B or args.i2v: 
    transformer_type = "i2v"

if args.t2v_1_3B:
    transformer_type = "t2v_1.3B"

if args.i2v_1_3B:
    transformer_type = "fun_inp_1.3B"

if args.vace_1_3B: 
    transformer_type = "vace_1.3B"

only_allow_edit_in_advanced = False
lora_preselected_preset = args.lora_preset
lora_preset_model = transformer_type



def save_model(model, model_type, dtype,  config_file,  submodel_no = 1,  is_module = False, filter = None, no_fp16_main_model = True, module_source_no = 1):
    model_def = get_model_def(model_type)
    # To save module and quantized modules
    # 1) set Transformer Model Quantization Type to 16 bits
    # 2) insert in def module_source : path and "model_fp16.safetensors in URLs"
    # 3) Generate (only quantized fp16 will be created)
    # 4) replace in def module_source : path and "model_bf16.safetensors in URLs"
    # 5) Generate (both bf16 and quantized bf16 will be created)
    if model_def == None: return
    if is_module:
        url_key = "modules"
        source_key = "module_source" if module_source_no <=1 else "module_source2"
    else:
        url_key = "URLs" if submodel_no <=1 else "URLs" + str(submodel_no)
        source_key = "source" if submodel_no <=1 else "source2"
    URLs= model_def.get(url_key, None)
    if URLs is None: return
    if isinstance(URLs, str):
        print("Unable to save model for a finetune that references external files")
        return
    from mmgp import offload    
    dtypestr= "bf16" if dtype == torch.bfloat16 else "fp16"
    if no_fp16_main_model: dtypestr = dtypestr.replace("fp16", "bf16")
    model_filename = None
    if is_module:
        if not isinstance(URLs,list) or len(URLs) != 1:
            print("Target Module files are missing")
            return 
        URLs= URLs[0]
    if isinstance(URLs, dict):
        url_dict_key = "URLs" if module_source_no ==1 else "URLs2"
        URLs = URLs[url_dict_key]
    for url in URLs:
        if "quanto" not in url and dtypestr in url:
            model_filename = os.path.basename(url)
            break
    if model_filename is None:
        print(f"No target filename with bf16 or fp16 in its name is mentioned in {url_key}")
        return

    finetune_file = os.path.join(os.path.dirname(model_def["path"]) , model_type + ".json")
    with open(finetune_file, 'r', encoding='utf-8') as reader:
        saved_finetune_def = json.load(reader)

    update_model_def = False
    model_filename_path = os.path.join(fl.get_download_location(), model_filename)
    quanto_dtypestr= "bf16" if dtype == torch.bfloat16 else "fp16"
    if ("m" + dtypestr) in model_filename: 
        dtypestr = "m" + dtypestr 
        quanto_dtypestr = "m" + quanto_dtypestr 
    if fl.locate_file(model_filename, error_if_none= False) is None and (not no_fp16_main_model or dtype == torch.bfloat16):
        offload.save_model(model, model_filename_path, config_file_path=config_file, filter_sd=filter)
        print(f"New model file '{model_filename}' had been created for finetune Id '{model_type}'.")
        del saved_finetune_def["model"][source_key]
        del model_def[source_key]
        print(f"The 'source' entry has been removed in the '{finetune_file}' definition file.")
        update_model_def = True

    if is_module:
        quanto_filename = model_filename.replace(dtypestr, "quanto_" + quanto_dtypestr + "_int8" )
        quanto_filename_path = os.path.join(fl.get_download_location() , quanto_filename)
        if hasattr(model, "_quanto_map"):
            print("unable to generate quantized module, the main model should at full 16 bits before quantization can be done")
        elif fl.locate_file(quanto_filename, error_if_none= False) is None:
            offload.save_model(model, quanto_filename_path, config_file_path=config_file, do_quantize= True, filter_sd=filter)
            print(f"New quantized file '{quanto_filename}' had been created for finetune Id '{model_type}'.")
            if isinstance(model_def[url_key][0],dict): 
                model_def[url_key][0][url_dict_key].append(quanto_filename) 
                saved_finetune_def["model"][url_key][0][url_dict_key].append(quanto_filename)
            else: 
                model_def[url_key][0].append(quanto_filename) 
                saved_finetune_def["model"][url_key][0].append(quanto_filename)
            update_model_def = True
    if update_model_def:
        with open(finetune_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(saved_finetune_def, indent=4))

def save_quantized_model(model, model_type, model_filename, dtype, config_file, submodel_no=1, convrot_layout=None):
    if "quanto" in model_filename or "convrot" in model_filename: return
    source_filename = model_filename
    model_def = get_model_def(model_type)
    if model_def == None: return
    url_key = "URLs" if submodel_no <=1 else "URLs" + str(submodel_no)
    URLs= model_def.get(url_key, None)
    if URLs is None: return
    if isinstance(URLs, str):
        print("Unable to create a quantized model for a finetune that references external files")
        return
    from mmgp import offload
    if dtype == torch.bfloat16:
         model_filename =  model_filename.replace("fp16", "bf16").replace("FP16", "bf16")
    elif dtype == torch.float16:
         model_filename =  model_filename.replace("bf16", "fp16").replace("BF16", "bf16")

    for rep in ["mfp16", "fp16", "mbf16", "bf16"]:
        if "_" + rep in model_filename:
            replacement = "_int8_convrot" if args.convrot else f"_quanto_{rep}_int8"
            model_filename = model_filename.replace("_" + rep, replacement)
            break
    if not any(token in model_filename for token in ("quanto", "convrot")):
        pos = model_filename.rfind(".")
        suffix = "_int8_convrot" if args.convrot else "_quanto_int8"
        model_filename = model_filename[:pos] + suffix + model_filename[pos:]

    model_filename = os.path.basename(model_filename)
    if fl.locate_file(model_filename, error_if_none= False) is not None:
        print(f"There isn't any model to quantize as quantized model '{model_filename}' aready exists")
    else:
        model_filename_path = os.path.join(fl.get_download_location(), model_filename)
        if args.convrot:
            from shared.convert.convrot import save_convrot_model
            source_path = source_filename if os.path.isfile(source_filename) else fl.locate_file(source_filename)
            save_convrot_model(model, source_path, model_filename_path, layout=convrot_layout, verboseLevel=verbose_level)
        else:
            offload.save_model(model, model_filename_path, do_quantize=True, config_file_path=config_file)
        print(f"New quantized file '{model_filename}' had been created for finetune Id '{model_type}'.")
        if not model_filename in URLs:
            URLs.append(model_filename)
            finetune_file = os.path.join(os.path.dirname(model_def["path"]) , model_type + ".json")
            with open(finetune_file, 'r', encoding='utf-8') as reader:
                saved_finetune_def = json.load(reader)
            saved_finetune_def["model"][url_key] = URLs
            with open(finetune_file, "w", encoding="utf-8") as writer:
                writer.write(json.dumps(saved_finetune_def, indent=4))
            print(f"The '{finetune_file}' definition file has been automatically updated with the local path to the new quantized model.")


def get_loras_preprocessor(transformer, model_type):
    preprocessor =  getattr(transformer, "preprocess_loras", None)
    if preprocessor == None:
        return None
    
    def preprocessor_wrapper(sd):
        return preprocessor(model_type, sd)

    return preprocessor_wrapper    


def process_files_def(repoId = None, sourceFolderList = None, fileList = None, targetFolderList = None, gen=None, show_filename=True):
    from shared.utils.download import process_files_def as shared_process_files_def

    return shared_process_files_def(repoId=repoId, sourceFolderList=sourceFolderList, fileList=fileList, targetFolderList=targetFolderList, gen=gen, show_filename=show_filename)

def release_flashvsr_vram():
    upsampler_api.require_upsampler_by_method("flashvsr").release_vram()

def release_pid_vram():
    handler = upsampler_api.find_upsampler_by_method("flux_pid") or upsampler_api.find_upsampler_by_method("flux2_pid")
    if handler is not None:
        handler.release_vram()

def release_coz_vram():
    upsampler_api.require_upsampler_by_method("coz").release_vram()

def release_extension_offloadobjs():
    return offload_registry.release_all()

def download_requested_postprocessing_assets(send_cmd, *, postprocess_audio="", temporal_upsampling="", spatial_upsampling="", replace_voice_method="", gen=None):
    process_files = partial(process_files_def, gen=gen)
    for method in dict.fromkeys([audio_processor_api.normalize_method(postprocess_audio), audio_processor_api.normalize_method(replace_voice_method)]):
        if method and method != "control":
            audio_processor_api.download_for_method(method, process_files, send_cmd=send_cmd)
    temporal_upsampler_api.download_for_value(temporal_upsampling, process_files, send_cmd=send_cmd)
    upsampler_api.download_for_value(spatial_upsampling, process_files, send_cmd=send_cmd)


def download_file(url,filename, gen=None, show_filename=True):
    from shared.utils.download import download_url as shared_download_file

    return shared_download_file(url, filename, gen=gen, show_filename=show_filename)

def query_global_shared_model_files():
    from shared.utils.optional_assets import query_download_defs

    return query_download_defs(server_config)

def download_models(model_filename = None, model_type= None, file_type = 0, submodel_no = 1, force_path = None, model_def = None, gen=None, show_filename=True):
    def computeList(filename):
        if filename == None:
            return []
        pos = filename.rfind("/")
        filename = filename[pos+1:]
        return [filename]        


    if model_filename is None: return

    base_model_type = get_base_model_type(model_type)
    model_def = model_def or get_model_def(model_type)
    
    any_source = ("source2" if submodel_no ==2 else "source") in model_def
    any_module_source = ("module_source2" if submodel_no ==2 else "module_source") in model_def 
    model_type_handler = model_types_handlers[base_model_type]

    if not (any_source and file_type==0 or any_module_source and file_type==1):
        local_model_filename = fl.get_local_model_filename(model_filename, extra_paths= force_path)
        if local_model_filename is None and len(model_filename) > 0:
            local_model_filename = fl.get_smart_download_location(model_filename, force_path= force_path)
            url = model_filename

            if not url.startswith("http"):
                raise Exception(f"Model '{model_filename}' was not found locally and no URL was provided to download it. Please add an URL in the model definition file.")
            try:
                download_file(url, local_model_filename, gen=gen, show_filename=show_filename)
            except (DownloadCancelled, DownloadError):
                raise
            except Exception as e:
                if os.path.isfile(local_model_filename): os.remove(local_model_filename) 
                raise Exception(f"'{url}' is invalid for Model '{model_type}' : {str(e)}'")
            if file_type!=0: return
    lora_dir = get_lora_dir(model_type) 
    for prop, recursive in zip(["preload_URLs", "VAE_URLs"], [True, False]):
        if recursive:
            preload_URLs = get_model_recursive_prop(model_type, prop, return_list= True, model_def=model_def)
        else:
            preload_URLs = model_def.get(prop, [])
            if isinstance(preload_URLs, str): preload_URLs = [preload_URLs]

        for url in preload_URLs:
            filename = fl.get_local_model_filename(url, lora_dir = lora_dir)
            if filename is None: 
                filename = fl.get_download_location(url, lora_dir = lora_dir)
                if not url.startswith("http"):
                    raise Exception(f"{prop} '{filename}' was not found locally and no URL was provided to download it. Please add an URL in the model definition file.")
                try:
                    download_file(url, filename, gen=gen, show_filename=show_filename)
                except (DownloadCancelled, DownloadError):
                    raise
                except Exception as e:
                    if os.path.isfile(filename): os.remove(filename) 
                    raise Exception(f"{prop} '{url}' is invalid: {str(e)}'")

    model_loras = get_model_recursive_prop(model_type, "loras", return_list= True, model_def=model_def)
    for url in model_loras:
        filename = get_lora_local_path(lora_dir, url)
        if not os.path.isfile(filename ): 
            if not url.startswith("http"):
                raise Exception(f"Lora '{filename}' was not found in the Loras Folder and no URL was provided to download it. Please add an URL in the model definition file.")
            try:
                download_file(url, filename, gen=gen, show_filename=show_filename)
            except (DownloadCancelled, DownloadError):
                raise
            except Exception as e:
                if os.path.isfile(filename): os.remove(filename) 
                raise Exception(f"Lora URL '{url}' is invalid: {str(e)}'")
            
    if file_type != 0: return            
    model_files = model_type_handler.query_model_files(computeList, base_model_type, model_def)
    if not isinstance(model_files, list): model_files = [model_files]
    for one_repo in model_files:
        process_files_def(**one_repo, gen=gen, show_filename=show_filename)

offload.default_verboseLevel = verbose_level

loras_url_cache = None
loras_cache_file = "loras_url_cache_v2.json"
def _ensure_loras_url_cache():
    global loras_url_cache
    if loras_url_cache is None:
        if os.path.isfile(loras_cache_file):
            try:
                with open(loras_cache_file, 'r', encoding='utf-8') as f:
                    loras_url_cache = json.load(f)
            except:
                loras_url_cache = {}
        else:
            loras_url_cache = {}


def get_lora_local_path(lora_dir, lora):
    if os.path.isabs(lora): return lora
    if (lora.startswith("http:") or lora.startswith("https:")):
        parts = lora.split("|")
        lora_path = os.path.join(fl.clean_relative_path(parts[1]), os.path.basename(parts[0])) if len(parts) > 1 else os.path.basename(lora)
    else:
        lora_path = lora
    return lora_path if lora_dir is None else os.path.join(lora_dir, lora_path) 

def get_lora_URL(lora_dir, lora):
    if os.path.isabs(lora): return lora
    _ensure_loras_url_cache()
    rel_path = get_lora_local_path(None, lora)
    if lora_dir is None: return rel_path
    url = loras_url_cache.get(lora_dir + "|" +  rel_path, None)         
    if url is None:
        return rel_path
    base = os.path.dirname(rel_path)
    return url if len(base)==0 else url + "|" + base

def check_loras_exist(model_type, loras_choices_files, download = False, send_cmd = None, gen=None):
    _ensure_loras_url_cache()
    lora_dir = get_lora_dir(model_type)
    missing_local_loras = []
    missing_remote_loras = []
    for lora_file in loras_choices_files:
        local_path = get_lora_local_path(lora_dir, lora_file)
        if not os.path.isfile(local_path):
            rel_path = get_lora_local_path(None, lora_file)
            url = loras_url_cache.get(lora_dir + "|" +  rel_path, None)         
            if url is not None:
                if download:
                    if send_cmd is not None:
                        send_cmd("status", f'Downloading Lora {os.path.basename(lora_file)}...')
                    try:
                        download_file(url, local_path, gen=gen)
                    except DownloadCancelled:
                        raise
                    except Exception as e:
                        print(f"Error downloading {url}:{e}")
                        missing_remote_loras.append(lora_file)
            else:
                missing_local_loras.append(lora_file)

    error = ""
    if len(missing_local_loras) > 0:
        error += f"The following Loras files are missing or invalid: {missing_local_loras}."
    if len(missing_remote_loras) > 0:
        error += f"The following Loras files could not be downloaded: {missing_remote_loras}."
    
    return error

def extract_preset(model_type, lset_name, loras):
    loras_choices = []
    loras_choices_files = []
    loras_mult_choices = ""
    prompt =""
    full_prompt =""
    lset_name = sanitize_file_name(lset_name)
    lora_dir = get_lora_dir(model_type)
    if not lset_name.endswith(".lset"):
        lset_name_filename = os.path.join(lora_dir, lset_name + ".lset" ) 
    else:
        lset_name_filename = os.path.join(lora_dir, lset_name ) 
    error = ""
    if not os.path.isfile(lset_name_filename):
        error = f"Preset '{lset_name}' not found "
    else:

        with open(lset_name_filename, "r", encoding="utf-8") as reader:
            text = reader.read()
        lset = json.loads(text)

        loras_choices = lset["loras"]
        loras_mult_choices = lset["loras_mult"]
        prompt = lset.get("prompt", "")
        full_prompt = lset.get("full_prompt", False)
    return loras_choices, loras_mult_choices, prompt, full_prompt, error


def setup_loras(model_type, transformer,  lora_dir, lora_preselected_preset, split_linear_modules_map = None):
    loras =[]
    default_loras_choices = []
    default_loras_multis_str = ""
    loras_presets = []
    default_lora_preset = ""
    default_lora_preset_prompt = ""

    from pathlib import Path
    base_model_type = get_base_model_type(model_type)
    lora_dir = get_lora_dir(base_model_type)
    if lora_dir != None :
        if not os.path.isdir(lora_dir):
            raise Exception("--lora-dir should be a path to a directory that contains Loras")


    if lora_dir != None:
        dir_loras = glob.glob(os.path.join(lora_dir, "**", "*.sft"), recursive=True) + glob.glob(os.path.join(lora_dir, "**", "*.safetensors"), recursive=True)
        dir_loras.sort(key=lambda path: os.path.relpath(path, lora_dir).casefold())
        loras += [element for element in dir_loras if element not in loras ]

        dir_presets_settings = glob.glob( os.path.join(lora_dir , "*.json") ) + glob.glob( os.path.join(lora_dir , "*.zip") )
        dir_presets_settings.sort()
        dir_presets =   glob.glob( os.path.join(lora_dir , "*.lset") )
        dir_presets.sort()
        # loras_presets = [ Path(Path(file_path).parts[-1]).stem for file_path in dir_presets_settings + dir_presets]
        loras_presets = [ Path(file_path).parts[-1] for file_path in dir_presets_settings + dir_presets]

    if transformer !=None:
        loras = offload.load_loras_into_model(transformer, loras,  activate_all_loras=False, check_only= True, preprocess_sd=get_loras_preprocessor(transformer, base_model_type), split_linear_modules_map = split_linear_modules_map) #lora_multiplier,

    if len(loras) > 0:
        loras = [get_lora_local_path(None, os.path.relpath(lora, lora_dir).replace("\\", "/")) if lora_dir is not None else get_lora_local_path(None, lora) for lora in loras]

    if len(lora_preselected_preset) > 0:
        if not os.path.isfile(os.path.join(lora_dir, lora_preselected_preset + ".lset")):
            raise Exception(f"Unknown preset '{lora_preselected_preset}'")
        default_lora_preset = lora_preselected_preset
        default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, _ , error = extract_preset(base_model_type, default_lora_preset, loras)
        if len(error) > 0:
            print(error[:200])
    return loras, loras_presets, default_loras_choices, default_loras_multis_str, default_lora_preset_prompt, default_lora_preset

def get_transformer_model(model, submodel_no = 1):
    if submodel_no > 1:
        model_key = f"model{submodel_no}"
        if not hasattr(model, model_key): return None

    if hasattr(model, "model"):
        if submodel_no > 1:
            return getattr(model, f"model{submodel_no}")
        else:
            return model.model
    elif hasattr(model, "transformer"):
        return model.transformer
    else:
        raise Exception("no transformer found")

def _normalize_output_type(output_type):
    if output_type is None:
        return "video"
    output_type = str(output_type).lower()
    if output_type not in ("video", "image", "audio"):
        return "video"
    return output_type

def get_default_profile(output_type):
    if force_profile_no >= 0:
        return force_profile_no
    output_type = _normalize_output_type(output_type)
    if output_type == "image":
        return default_profile_image
    if output_type == "audio":
        return default_profile_audio
    return default_profile_video

def compute_profile(override_profile, output_type="video"):
    return override_profile if override_profile != -1 else get_default_profile(output_type)

def get_profile_type_for_model(model_type, image_mode=0):
    model_def = get_model_def(model_type)
    if model_def is None: return "video"
    profile_type = model_def.get("profile_type", None)
    if profile_type is not None: return profile_type
    if model_def.get("audio_only", False):
        return "audio"
    if image_mode and image_mode > 0:
        return "image"
    return "video"

def init_pipe(pipe, kwargs, profile):
    preload =int(args.preload)
    if preload == 0:
        preload = server_config.get("preload_in_VRAM", 0)

    kwargs["extraModelsToQuantize"]=  None
    source_budgets = kwargs.get("budgets", None)
    if source_budgets is None:  kwargs["budgets"] = source_budgets = {}
    mmgp_profile = int(profile)
    if mmgp_profile in (2, 4, 5):
        default_transformer_budget = default_transformer2_budget= kwargs.get("budgets", 100) 
        if isinstance(default_transformer_budget, dict):
            default_transformer_budget = default_transformer_budget.get("transformer", 100) 
            default_transformer2_budget = default_transformer2_budget.get("transformer2", 100) 

        budgets = { "transformer" : default_transformer_budget if preload  == 0 else preload, "text_encoder" : 100 if preload  == 0 else preload, "*" : max(1000 if profile==5 else 3000 , preload) }
        if "transformer2" in pipe:
            budgets["transformer2"] = default_transformer2_budget if preload  == 0 else preload
        source_budgets.update(budgets)
    elif mmgp_profile == 3:
        source_budgets.update({ "*" : "70%" })

    if "transformer2" in pipe:
        if profile in [3,4]:
            kwargs["pinnedMemory"] = ["transformer", "transformer2"]
    
    if profile == 4.5:
        kwargs["asyncTransfers"] = False
    elif profile == 3.5:
        kwargs["pinnedMemory"] = False
    if is_mps:
        kwargs["pinnedMemory"] = False
        kwargs["asyncTransfers"] = False

    return mmgp_profile

reset_prompt_enhancer_requested = False
def unload_prompt_enhancer_runtime():
    deepy_controller._unload_prompt_enhancer_runtime(prompt_enhancer_image_caption_model, prompt_enhancer_llm_model)


def reset_prompt_enhancer():
    global reset_prompt_enhancer_requested
    reset_prompt_enhancer_requested = True

def reset_prompt_enhancer_if_requested():
    global reset_prompt_enhancer_requested, prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer, enhancer_offloadobj
    if not reset_prompt_enhancer_requested:
        return
    reset_prompt_enhancer_requested = False
    unload_prompt_enhancer_runtime()
    prompt_enhancer_image_caption_model = None
    prompt_enhancer_image_caption_processor = None
    prompt_enhancer_llm_model = None
    prompt_enhancer_llm_tokenizer = None
    if enhancer_offloadobj is not None:
        enhancer_offloadobj.release()
        enhancer_offloadobj = None

def setup_prompt_enhancer(pipe, kwargs, gen=None):
    global prompt_enhancer_image_caption_model, prompt_enhancer_image_caption_processor, prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer
    model_no = server_config.get("enhancer_enabled", 0) 
    if model_no != 0:
        from shared.prompt_enhancer import load_prompt_enhancer_runtime

        runtime = load_prompt_enhancer_runtime(
            partial(process_files_def, gen=gen),
            enhancer_enabled=model_no,
            lm_decoder_engine=server_config.get("lm_decoder_engine", ""),
            qwen_backend=server_config.get("prompt_enhancer_quantization", "quanto_int8"),
            speculative_decoding=normalize_prompt_enhancer_speculative_decoding(server_config.get(PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT)),
            deepy_kv_cache_quantization=server_config.get(DEEPY_KV_CACHE_QUANTIZATION_KEY, DEEPY_KV_CACHE_QUANTIZATION_DEFAULT),
        )
        prompt_enhancer_image_caption_model = runtime.image_caption_model
        prompt_enhancer_image_caption_processor = runtime.image_caption_processor
        prompt_enhancer_llm_model = runtime.llm_model
        prompt_enhancer_llm_tokenizer = runtime.llm_tokenizer
        pipe.update(runtime.pipe_models)
        if runtime.budgets:
            kwargs.setdefault("budgets", {}).update(runtime.budgets)
        if runtime.co_tenants:
            kwargs.setdefault("coTenantsMap", {}).update(runtime.co_tenants)
    else:
        reset_prompt_enhancer()


def ensure_prompt_enhancer_loaded(override_profile=-1, progress=None, send_cmd=None, gen=None):
    global enhancer_offloadobj

    reset_prompt_enhancer_if_requested()
    check_download_cancelled(gen)
    if enhancer_offloadobj is None:
        from shared.prompt_enhancer import download_prompt_enhancer_assets

        download_prompt_enhancer_assets(
            enhancer_enabled=server_config.get("enhancer_enabled", 0),
            qwen_backend=server_config.get("prompt_enhancer_quantization", "quanto_int8"),
            speculative_decoding=normalize_prompt_enhancer_speculative_decoding(server_config.get(PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT)),
            send_cmd=send_cmd,
            progress=progress,
            gen=gen,
        )
        if progress is not None:
            progress(0, "Please Wait While Loading Prompt Enhancer")
        def loading_progress(phase, completed, total, model_id):
            progress((completed, total), desc=f"Loading - {phase} Prompt Enhancer", unit="items")

        loading_callback = offload.LoadingCallback(lambda: gen is not None and gen.get("abort", False), loading_progress) if progress is not None else None
        with model_unload_guard():
            if enhancer_offloadobj is None:
                try:
                    with offload.loading_context(loading_callback):
                        kwargs = {}
                        pipe = {}
                        setup_prompt_enhancer(pipe, kwargs, gen=gen)
                        profile = compute_profile(override_profile, "video")
                        mmgp_profile = init_pipe(pipe, kwargs, profile)
                        kwargs["pinnedMemory"] = False
                        enhancer_offloadobj = offload.profile(pipe, profile_no=mmgp_profile, loading_callback=loading_callback, **kwargs)
                except (DownloadCancelled, offload.LoadingCancelled, InterruptedError):
                    reset_prompt_enhancer()
                    reset_prompt_enhancer_if_requested()
                    raise

    if prompt_enhancer_llm_model is None or prompt_enhancer_llm_tokenizer is None:
        raise gr.Error("Prompt enhancer text runtime is not available.")
    return prompt_enhancer_llm_model, prompt_enhancer_llm_tokenizer

@controlled_model_loading
def load_models(model_type, override_profile = -1, output_type="video", config_id = None, runtime_model_type=None, track_as_main=True, gen=None, loading_callback=None, **model_kwargs):
    global transformer_type, loaded_profile, loaded_config
    def _load_models_info(message):
        if int(verbose_level) > 0:
            print(message)

    base_model_type = get_base_model_type(model_type)
    model_def = get_model_def(model_type)
    if config_id is not None and len(config_id):
        config_groups = get_model_config_groups(model_type, model_def)
        model_def = model_def.copy()
        for _, _, current_config in model_config_groups.selected_model_configs(config_groups, config_id):
            model_def.update(current_config)
    resolver = getattr(model_types_handlers[base_model_type], "resolve_runtime_model_def", None)
    if resolver is not None:
        model_def = resolver(model_def, dict(server_config, transformer_quantization=transformer_quantization,
            text_encoder_quantization=text_encoder_quantization, mixed_precision=server_config.get("mixed_precision", "0"),
            vae_precision=server_config.get("vae_precision", "16")))
    save_quantized = args.save_quantized and model_def != None
    model_filename = get_model_filename(model_type=model_type, quantization= "" if save_quantized else transformer_quantization, dtype_policy = transformer_dtype_policy, model_def=model_def)
    if "URLs2" in model_def:
        model_filename2 = get_model_filename(model_type=model_type, quantization= "" if save_quantized else transformer_quantization, dtype_policy = transformer_dtype_policy, submodel_no=2, model_def=model_def) # !!!!
    else:
        model_filename2 = None
    modules = get_model_recursive_prop(model_type, "modules", return_list=True, model_def=model_def)
    modules = [get_model_recursive_prop(module, "modules", sub_prop_name  ="_list",  return_list= True) if isinstance(module, str) else module for module in modules ]
    if save_quantized and "quanto" in model_filename:
        save_quantized = False
        print("Need to provide a non quantized model to create a quantized model to be saved") 
    if save_quantized and len(modules) > 0:
        print(f"Unable to create a finetune quantized model as some modules are declared in the finetune definition. If your finetune includes already the module weights you can remove the 'modules' entry and try again. If not you will need also to change temporarly the model 'architecture' to an architecture that wont require the modules part ({modules}) to quantize and then add back the original 'modules' and 'architecture' entries.")
        save_quantized = False
    quantizeTransformer = not save_quantized and model_def !=None and transformer_quantization in ("int8", "fp8") and model_def.get("auto_quantize", False) and not "quanto" in model_filename
    if quantizeTransformer and len(modules) > 0:
        print(f"Autoquantize is not yet supported if some modules are declared")
        quantizeTransformer = False
    model_family = get_model_family(model_type)
    transformer_dtype = get_transformer_dtype(model_type, transformer_dtype_policy)
    if quantizeTransformer or "quanto" in model_filename:
        transformer_dtype = torch.bfloat16 if "bf16" in model_filename or "BF16" in model_filename else transformer_dtype
        transformer_dtype = torch.float16 if "fp16" in model_filename or"FP16" in model_filename else transformer_dtype
    perc_reserved_mem_max = args.perc_reserved_mem_max
    vram_safety_coefficient = args.vram_safety_coefficient 
    model_file_list = [model_filename]
    model_type_list = [model_type]
    source_type_list = [0]
    model_submodel_no_list = [1]
    if model_filename2 != None:
        model_file_list += [model_filename2]
        model_type_list += [model_type]
        source_type_list += [0]
        model_submodel_no_list += [2]
    for module_type in modules:
        if isinstance(module_type,dict):
            URLs1 = module_type.get("URLs", None)
            if URLs1 is None: raise Exception(f"No URLs defined for Module {module_type}")
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, URLs = URLs1))
            URLs2 = module_type.get("URLs2", None)
            if URLs2 is None: raise Exception(f"No URL2s defined for Module {module_type}")
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, URLs = URLs2))
            model_type_list += [model_type] * 2
            source_type_list += [1] * 2
            model_submodel_no_list += [1,2]
        else:
            model_file_list.append(get_model_filename(model_type, transformer_quantization, transformer_dtype, module_type= module_type))
            model_type_list.append(model_type)
            source_type_list.append(True)
            model_submodel_no_list.append(0) 

    local_model_file_list= []
    for filename, file_model_type, file_source_type, submodel_no in zip(model_file_list, model_type_list, source_type_list, model_submodel_no_list):
        if len(filename) == 0: continue 
        download_models(filename, file_model_type, file_source_type, submodel_no, model_def = model_def, gen=gen)
        local_file_name = fl.get_local_model_filename(filename )
        local_model_file_list.append( os.path.basename(filename) if local_file_name is None else local_file_name )
    if len(local_model_file_list) == 0:
        download_models("", model_type, 0, -1, model_def = model_def, gen=gen)

    VAE_dtype = torch.float16 if server_config.get("vae_precision","16") == "16" else torch.float
    mixed_precision_transformer =  server_config.get("mixed_precision","0") == "1"
    if track_as_main:
        transformer_type = None

    for source_type, filename in zip(source_type_list, local_model_file_list):
        if source_type==0:  
            _load_models_info(f"Loading Model '{filename}' ...")
        elif source_type==1:  
            _load_models_info(f"Loading Module '{filename}' ...")


    model_type_handler = model_types_handlers[base_model_type] 
    text_encoder_URLs= get_model_recursive_prop(model_type, "text_encoder_URLs", return_list=True, model_def=model_def)
    if text_encoder_URLs is not None:
        text_encoder_filename = get_model_filename(model_type=model_type, quantization= text_encoder_quantization, dtype_policy = transformer_dtype_policy, URLs=text_encoder_URLs)
    if text_encoder_filename is not None and len(text_encoder_filename):
        text_encoder_folder = model_def.get("text_encoder_folder", None)
        if text_encoder_filename is not None:
            download_models(text_encoder_filename, file_model_type, 2, -1, force_path =text_encoder_folder, model_def = model_def, gen=gen)
            text_encoder_filename =  fl.get_local_model_filename(text_encoder_filename, extra_paths=text_encoder_folder)
            _load_models_info(f"Loading Text Encoder '{text_encoder_filename}' ...")


    check_download_cancelled(gen)
    profile = compute_profile(override_profile, output_type)
    lm_decoder_engine_obtained = resolve_lm_decoder_engine(lm_decoder_engine, model_def.get("lm_engines", []) )
    if lm_decoder_engine_obtained in ("cg", "vllm") and int(profile) not in [ 1, 3]:
        _load_models_info(f"Unable to use LM Engine '{lm_decoder_engine_obtained}' as it requires a Memory Profile such as 1,3 or 3+ that loads entirely the Main Models in VRAM. Switching to Legacy LM Engine...")
        lm_decoder_engine_obtained = "legacy"
    preview_mode = server_config.get("generation_preview", "rgb")
    preview_name = preview_path = preview_decoder = None
    if preview_mode not in ("rgb", "tiny_vae_frames", "tiny_vae_video"):
        raise ValueError("generation_preview must be rgb, tiny_vae_frames or tiny_vae_video")
    if preview_mode != "rgb":
        from shared.tinyvae.decoder import decoder_for, prepare_decoder, load_decoder
        preview_name = decoder_for(base_model_type, model_def)
        if preview_name is not None:
            if not getattr(offload.offload, "supports_cotenant_wildcards", False):
                raise RuntimeError("GPU TinyVAE previews require MMGP with wildcard cotenant support. Please update MMGP.")
            preview_path = prepare_decoder(preview_name, gen=gen)
    loading_model_ids = {filename: "transformer" if i == 0 else f"transformer {i + 1}" for i, filename in enumerate(local_model_file_list)}
    if text_encoder_filename:
        loading_model_ids[text_encoder_filename] = "text_encoder"
    if preview_path is not None:
        loading_model_ids[preview_path] = "tiny_vae"
    with model_unload_guard(), offload.loading_context(loading_callback, loading_model_ids):
        torch.set_default_device('cpu')
        if preview_path is not None:
            preview_decoder = load_decoder(preview_name, preview_path)
        wan_model, pipe = model_type_handler.load_model(
                    local_model_file_list, runtime_model_type or model_type, base_model_type, model_def, quantizeTransformer = quantizeTransformer, text_encoder_quantization = text_encoder_quantization,
                    dtype = transformer_dtype, VAE_dtype = VAE_dtype, mixed_precision_transformer = mixed_precision_transformer, save_quantized = save_quantized, submodel_no_list   = model_submodel_no_list, text_encoder_filename = text_encoder_filename, profile=profile, lm_decoder_engine=lm_decoder_engine_obtained, **model_kwargs )

        kwargs = {}
        if "pipe" in pipe:
            kwargs = pipe
            pipe = kwargs.pop("pipe")
        if "coTenantsMap" not in kwargs: kwargs["coTenantsMap"] = {}
        if preview_decoder is not None:
            pipe["tiny_vae"] = preview_decoder
            kwargs["coTenantsMap"]["tiny_vae"] = "*"
        mmgp_profile = init_pipe(pipe, kwargs, profile)
        loras_transformer = kwargs.pop("loras", [])
        if "transformer" in pipe:
            loras_transformer += ["transformer"]
        if "transformer2" in pipe:
            loras_transformer += ["transformer2"]
        compile_modules = model_def.get("compile", compile) if len(compile) > 0 else False
        custom_compile = len(compile) > 0 and hasattr(wan_model, "custom_compile")
        int8_backend.prepare_compile_cache(compile_modules or custom_compile)
        if custom_compile:
            wan_model.custom_compile(backend= "inductor", mode ="default")
        if compile_modules == False and len(compile):
            _load_models_info("Pytorch compilation is not supported for this Model")
        if args.multigpu:
            # Two 16 GiB GPUs + a 16 GiB host cannot afford MMGP's normal
            # reserved-RAM copy. Profile 4 pins the transformer by default
            # (the run can reserve ~5-6 GiB before generation), so MultiGPU
            # explicitly disables RAM pinning and async prefetching.
            kwargs["pinnedMemory"] = False
            kwargs["pinnedPEFTLora"] = False
            kwargs["asyncTransfers"] = False
            kwargs["partialPinning"] = False
            print("[MultiGPU] Host RAM mode: no pinned model copies / no async RAM prefetch", flush=True)
        offloadobj = offload.profile(pipe, profile_no= mmgp_profile, compile = compile_modules, quantizeTransformer = False, loras = loras_transformer, perc_reserved_mem_max = perc_reserved_mem_max , vram_safety_coefficient = vram_safety_coefficient , convertWeightsFloatTo = transformer_dtype, loading_callback=loading_callback, **kwargs)
    if args.multigpu:
        multigpu_devices = [x.strip() for x in args.multigpu.split(",") if x.strip()]
        primary_multigpu = multigpu_devices[0]
        if not primary_multigpu.startswith("cuda:"):
            primary_multigpu = f"cuda:{primary_multigpu}"
        if args.gpu:
            requested_gpu = args.gpu if args.gpu.startswith("cuda:") else f"cuda:{args.gpu}"
            if requested_gpu != primary_multigpu:
                raise ValueError(f"--gpu ({args.gpu}) must match the first --multigpu device ({primary_multigpu})")
        args.gpu = primary_multigpu
        attach_multigpu(offloadobj, args.multigpu, fraction=args.multigpu_cache_fraction, verbose=int(verbose_level))
    offloadobj.tiny_vae = preview_decoder
    if len(args.gpu) > 0:
        torch.set_default_device(args.gpu)
    if track_as_main:
        transformer_type = model_type
        loaded_profile = profile
        loaded_config = config_id or ""
    return wan_model, offloadobj 

if not "P" in preload_model_policy:
    wan_model, offloadobj, transformer = None, None, None
    reload_needed = True
else:
    wan_model, offloadobj = load_models(
        transformer_type,
        output_type=get_profile_type_for_model(transformer_type, 0),
    )
    if check_loras:
        transformer = get_transformer_model(wan_model)
        if hasattr(wan_model, "get_trans_lora"):
            transformer, _ = wan_model.get_trans_lora()
        setup_loras(transformer_type, transformer,  get_lora_dir(transformer_type), "", None)
        exit()

gen_in_progress = False

def is_generation_in_progress():
    global gen_in_progress
    return gen_in_progress

def get_auto_attention():
    return get_default_attention_mode()

def generate_model_description(model_type):
    description_container = [""]
    model_name = get_model_name(model_type, description_container)
    model_def = get_model_def(model_type) or {}
    description  = description_container[0]
    description = model_infos.render_model_description(description, model_def.get("infos", None), model_type=model_type, model_name=model_name, height=60 if server_config.get('display_stats', 0) == 1 else 40)
    return description


def generate_header(model_type, compile, attention_mode, override_attention=""):
    description = generate_model_description(model_type)
    model_def = get_model_def(model_type) or {}
    full_filename = get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)
    model_filename = os.path.basename(full_filename)
    model_attention = get_overridden_attention(model_type)
    overridden_attention = override_attention or model_attention
    attn_mode = attention_mode if overridden_attention is None else overridden_attention
    header = "<DIV style='align:right;width:100%'><FONT SIZE=2>Attention mode <B>" + (attn_mode if attn_mode!="auto" else "auto/" + get_auto_attention() )
    custom_attention = model_def.get("custom_attention_modes", {}).get(attn_mode)
    if custom_attention is not None:
        custom_status = ATTENTION_MODE_AVAILABILITY.get(attn_mode, {})
        custom_installed = custom_attention.get("installed", custom_status.get("installed", True))
        custom_supported = custom_attention.get("supported", custom_status.get("supported", custom_installed))
    else:
        custom_installed = custom_supported = False
    if attn_mode not in override_attention_modes_installed and not custom_installed:
        header += " -NOT INSTALLED-"
    elif attn_mode not in override_attention_modes_supported and not custom_supported:
        header += " -NOT SUPPORTED-"
    elif not override_attention and model_attention is not None and attention_mode != model_attention:
        header += " -MODEL SPECIFIC-"
    header += "</B>"

    if compile:
        header += ", Pytorch compilation <B>ON</B>"
    if "fp16" in model_filename:
        header += ", Data Type <B>FP16</B>"
    else:
        header += ", Data Type <B>BF16</B>"

    quant_label = quant_router.detect_quantization_label_from_filename(fl.get_local_model_filename(full_filename))
    if quant_label:
        header += f", Quantization <B>{quant_label}</B>"
    header += "</DIV>"

    return description,header

def release_RAM():
    if gen_in_progress:
        gr.Info("Unable to release RAM when a Generation is in Progress")
    else:
        release_model()
        gr.Info("Models stored in RAM have been released")

def get_gen_info(state):
    cache = state.get("gen", None)
    if cache == None:
        cache = dict()
        state["gen"] = cache
    return cache

def build_callback(state, pipe, send_cmd, status, num_inference_steps, preview_meta=None, tiny_preview=None):
    gen = get_gen_info(state)
    gen["num_inference_steps"] = num_inference_steps
    generation_start_time = last_refresh_time = time.time()
    denoising_start_time = None
    estimated_total_time = None
    def callback(step_idx = -1, latent = None, force_refresh = True, read_state = False, override_num_inference_steps = -1, pass_no = -1, preview_meta=preview_meta, denoising_extra ="", progress_unit = None, status_prefix = "", progress_title = None):
        nonlocal denoising_start_time, estimated_total_time, last_refresh_time
        step_completed = step_idx >= 0
        in_pause = False
        with gen_lock:
            process_status = gen.get("process_status", None)
            pause_msg = None
            if isinstance(process_status, str) and process_status.startswith("request:"):
                gen["process_status"] = "process:" + process_status[len("request:"):]
                offloadobj.unload_all()
                pause_msg = gen.get("pause_msg", "Unknown Pause")
                in_pause = True

        if in_pause:
            send_cmd("progress", [0, pause_msg])
            while True:
                time.sleep(0.1)            
                with gen_lock:
                    process_status = gen.get("process_status", None)
                    if isinstance(process_status, str) and process_status.startswith("request:"):
                        gen["process_status"] = "process:" + process_status[len("request:"):]
                        continue
                    if process_status == "process:main": break
            force_refresh = True
        if gen.get("early_stop", False) and not gen.get("early_stop_forwarded", False):
            gen["early_stop_forwarded"] = True
            if hasattr(pipe, "request_early_stop"):
                pipe.request_early_stop()
            elif wan_model is not None and hasattr(wan_model, "request_early_stop"):
                wan_model.request_early_stop()
            elif hasattr(pipe, "_early_stop"):
                pipe._early_stop = True
            elif wan_model is not None and hasattr(wan_model, "_early_stop"):
                wan_model._early_stop = True
        refresh_id =  gen.get("refresh", -1)
        current_time = time.time()
        if force_refresh and (step_idx < 0 or progress_title is not None):
            denoising_start_time = current_time
            estimated_total_time = None
        if force_refresh or step_idx >= 0:
            pass
        elif read_state:
            if current_time - last_refresh_time < 1:
                return
        else:
            refresh_id =  gen.get("refresh", -1)
            if refresh_id < 0:
                return
            UI_refresh = state.get("refresh", 0)
            if UI_refresh >= refresh_id:
                return  
        if override_num_inference_steps > 0 and progress_title is None:
            gen["num_inference_steps"] = override_num_inference_steps
             
        num_inference_steps = gen.get("num_inference_steps", 0)
        if read_state:
            num_inference_steps, progress_unit = gen.get("phase_progress_units", (num_inference_steps, progress_unit))
        else:
            if progress_title is not None:
                num_inference_steps = override_num_inference_steps
            gen["phase_progress_units"] = (num_inference_steps, progress_unit)
        status = status_prefix or gen["progress_status"]
        state["refresh"] = refresh_id
        if read_state:
            phase, state_step_idx = gen["progress_phase"]
            step_idx = state_step_idx if step_idx < 0 else step_idx + 1
        else:
            step_idx += 1         
            if gen.get("abort", False):
                # pipe._interrupt = True
                phase = "Aborting"    
            elif gen.get("early_stop", False):
                phase = "Early Stop in progress"
            elif progress_title is not None:
                phase = progress_title
            elif step_idx  == num_inference_steps:
                phase = "VAE Decoding"    
            else:
                if pass_no <=0:
                    phase = "Denoising"
                elif pass_no == 1:
                    phase = "Denoising First Phase"
                elif pass_no == 2:
                    phase = "Denoising Second Phase"
                elif pass_no == 3:
                    phase = "Denoising Third Phase"
                else:
                    phase = f"Denoising {pass_no}th Phase"

                if len(denoising_extra) > 0: phase += " | " + denoising_extra

            gen["progress_phase"] = (phase, step_idx)
        status_msg = merge_status_context(status, phase)      

        elapsed_time = current_time - generation_start_time
        progress_time = format_time(elapsed_time)
        if step_completed and step_idx > 0 and num_inference_steps > 0 and denoising_start_time is not None:
            denoising_elapsed_time = current_time - denoising_start_time
            estimated_total_time = denoising_start_time - generation_start_time + max(denoising_elapsed_time, denoising_elapsed_time * num_inference_steps / step_idx)
        if estimated_total_time is not None:
            progress_time += f" / {format_time(max(elapsed_time, estimated_total_time))}"
        status_msg = merge_status_context(status, f"{phase} | {progress_time}")
        if step_idx >= 0:
            progress_args = [(step_idx , num_inference_steps) , status_msg  ,  num_inference_steps]
            if progress_unit:
                progress_args.append(progress_unit)
        else:
            progress_args = [0, status_msg]
        
        # progress(*progress_args)
        send_cmd("progress", progress_args)
        last_refresh_time = current_time
        if latent is not None:
            payload = pipe.prepare_preview_payload(latent, preview_meta) if hasattr(pipe, "prepare_preview_payload") else latent
            if tiny_preview is not None:
                lat = payload["latents"] if isinstance(payload, dict) else payload
                if lat is not None:
                    if preview_meta is not None and preview_meta.get("first_latent_only", False):
                        lat = lat[:, :1]
                    tiny_preview.capture(lat, step_idx - 1, num_inference_steps, pass_no)
                return
            if isinstance(payload, dict):
                data = payload.copy()
                lat = data.get("latents")
                if torch.is_tensor(lat):
                    if preview_meta is not None and preview_meta.get("first_latent_only", False) and lat.ndim == 4:
                        lat = lat[:, :1]
                    data["latents"] = lat.to("cpu")
                payload = data
            elif torch.is_tensor(payload):
                if preview_meta is not None and preview_meta.get("first_latent_only", False) and payload.ndim == 4:
                    payload = payload[:, :1]
                payload = payload.to("cpu")
            if payload is not None:
                send_cmd("preview", payload)
            
        # gen["progress_args"] = progress_args
            
    callback.checkpoint = lambda: callback(-1, None, False, True)
    callback.tiny_vae = tiny_preview is not None
    return callback

def pause_generation(state):
    gen = get_gen_info(state)
    process_id = "pause"
    GPU_process_running = any_GPU_process_running(state, process_id, ignore_main= True )
    if GPU_process_running:
        gr.Info("Unable to pause, a PlugIn is using the GPU")
        yield gr.update(), gr.update()
        return
    gen["resume"] = False
    yield gr.Button(interactive= False), gr.update()
    pause_msg = "Generation on Pause, click Resume to Restart Generation"
    acquire_GPU_ressources(state, process_id , "Pause", gr= gr, custom_pause_msg= pause_msg, custom_wait_msg= "Please wait while the Pause Request is being Processed...")      
    gr.Info(pause_msg)
    yield gr.Button(visible= False, interactive= True), gr.Button(visible= True)
    while not gen.get("resume", False):
        time.sleep(0.5)

    release_GPU_ressources(state, process_id )
    gen["resume"] = False
    yield gr.Button(visible= True, interactive= True), gr.Button(visible= False)

def resume_generation(state):
    gen = get_gen_info(state)
    gen["resume"] = True
    if str(gen.get("process_status", "")).startswith(("request:deepy_pause_", "process:deepy_pause_", "request:pause", "process:pause")):
        from shared.deepy.engine import resume_assistant
        session = state.get("assistant_session")
        if session is not None and session.media_tool_active:
            resume_assistant(session)

def abort_generation(state, client_id="", notify = True):
    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    service = service_for(state)
    in_progress = "in_progress" in gen or (service is not None and service.generation_running)
    with lock:
        if len(queue):
            if len(client_id):
                for i, task in enumerate(queue):
                    queue_client_id = task["params"].get("client_id","")
                    if queue_client_id == client_id:
                        if i == 0:
                            if not in_progress:
                                del queue[0]
                                if "prompt_no" in gen: gen["prompt_no"] += 1
                                return gr.update(), gr.HTML(value=generate_queue_html(queue))
                            break
                        del queue[i]
                        if "prompt_no" in gen: gen["prompt_no"] += 1
                        return gr.update(), gr.HTML(value=generate_queue_html(queue))
            elif not in_progress:
                del queue[0]
                gen["prompt_no"] += 1
                return gr.update(), gr.HTML(value=generate_queue_html(queue))

    gen["resume"] = True
    if in_progress:
        if wan_model != None:
            wan_model._interrupt= True
        gen["abort"] = True            
        msg = "Processing Request to abort Current Generation"
        gen["status"] = msg
        if service is not None:
            service._generation_aborting = True
            service._generation_event("status", "Aborting generation…")
        if notify:
            gr.Info(msg)
        return gr.Button(interactive=  False), gr.update()
    else:
        return gr.Button(interactive=  True), gr.update()

def early_stop_generation(state):
    gen = get_gen_info(state)
    gen["resume"] = True
    queue = gen.get("queue", [])
    if not queue:
        return gr.Button(interactive=True)
    model_type = queue[0].get("params", {}).get("model_type")
    model_def = get_model_def(model_type) if model_type else None
    if not model_def or not model_def.get("supports_early_stop", False):
        gr.Info("Early Stop is not supported for this model.")
        return gr.Button(interactive=True)
    with gen_lock:
        if gen.get("early_stop", False):
            return gr.Button(interactive=False)
        gen["early_stop"] = True
        gen["early_stop_forwarded"] = False
    msg = "Early Stop Requested"
    gen["status"] = msg
    gr.Info(msg)
    return gr.Button(interactive=False)

def gallery_update(file_list, selected_index, **kwargs):
    visible, local_index, _ = gallery_window(file_list, selected_index, server_config['clear_file_list'])
    return gr.Gallery(value=visible, selected_index=local_index, **kwargs)


def pack_audio_gallery_state(audio_file_list, selected_index, refresh = True):
    visible, _, offset = gallery_window(audio_file_list, selected_index, server_config['clear_file_list'])
    return [json.dumps({'paths': visible, 'offset': offset}), selected_index, time.time()]

def unpack_audio_list(packed_audio_file_list):
    value = json.loads(packed_audio_file_list)
    return value['paths'] if isinstance(value, dict) else value

def can_extend_sample(params):
    if _is_edit_task_params(params) or params.get("mode") == "edit" or params.get("image_mode", 0) != 0:
        return False
    model_type = params["model_type"]
    base_model_type, model_def = get_base_model_type(model_type), get_model_def(model_type)
    preprocess_all = resolve_model_preprocess_all(model_def, base_model_type=base_model_type, video_prompt_type=params.get("video_prompt_type", ""), image_prompt_type=params.get("image_prompt_type", ""), audio_prompt_type=params.get("audio_prompt_type", ""), custom_settings=params.get("custom_settings", {}), params=params)
    return test_any_sliding_window(base_model_type) and not preprocess_all


def refresh_gallery(state): #, msg
    service = service_for(state)
    if service is not None:
        service.host_changed()
    gen = get_gen_info(state)

    # gen["last_msg"] = msg
    clear_deleted_files(state, False)
    clear_deleted_files(state, True)
    file_list = gen.get("file_list", None)      
    choice = gen.get("selected",0)
    audio_file_list = gen.get("audio_file_list", None)      
    audio_choice = gen.get("audio_selected",-1)

    header_text = gen.get("header_text", "")
    in_progress = "in_progress" in gen
    if gen.get("last_selected", True) and file_list is not None:
        choice = max(len(file_list) - 1,0)  
    if gen.get("audio_last_selected", True) and audio_file_list is not None:
        audio_choice = max(len(audio_file_list) - 1,-1)  
    last_was_audio = gen.get("last_was_audio", False)
    queue = gen.get("queue", [])
    abort_interactive = not gen.get("abort", False)
    early_stop_interactive = not gen.get("early_stop", False)
    early_stop_visible = False

    if gen.pop("refresh_tab", False):
        gen["current_gallery_source"] = "audio" if last_was_audio else "video"
        if last_was_audio: 
            output_tabs = [gr.Tabs(selected= "audio"), 1]
        else:
            output_tabs = [gr.Tabs(selected= "video_images"), 0]
    else:
        output_tabs = [gr.update(), gr.update()]

    if service is not None:
        audio_active = gen["current_gallery_source"] == "audio"
        output_tabs = [gr.Tabs(selected="audio" if audio_active else "video_images"), 1 if audio_active else 0]
        last_was_audio = False  # Synchronize the visual selection even while displaying audio.

    if not in_progress or len(queue) == 0:
        return *output_tabs, gallery_update(file_list, choice), gr.update() if last_was_audio else choice, *pack_audio_gallery_state(audio_file_list, audio_choice), gr.HTML("", visible= False),  gr.Button(visible=True), gr.Button(visible=False), gr.Row(visible=False), gr.update(), update_queue_data(queue), gr.Button(interactive=  abort_interactive), gr.Button(interactive=  early_stop_interactive, visible= early_stop_visible), gr.Button(visible= False), gr.update(visible=False)
    else:
        task = queue[0]
        prompt =  task["prompt"]
        params = task["params"]
        model_type = params.get("model_type", "")
        is_edit_task = _is_edit_task_params(params)
        multi_prompts_gen_type = "FG" if is_edit_task else params["multi_prompts_gen_type"]
        if is_edit_task:
            base_model_type, model_def = None, None
        else:
            base_model_type, model_def = get_base_model_type(model_type), get_model_def(model_type)
        onemorewindow_visible = can_extend_sample(params)
        early_stop_visible = bool(model_def and model_def.get("supports_early_stop", False))
        enhanced = False
        if prompt.startswith(prompt_parser.ENHANCED_PROMPT_PREFIX):
            enhanced = True
            prompt = prompt[len(prompt_parser.ENHANCED_PROMPT_PREFIX):]
        prompt_units = [prompt] if is_edit_task else prompt_parser.split_prompt_units(prompt, multi_prompts_gen_type)
        if multi_prompts_gen_type == "FG" or len(prompt_units) <= 1:
            prompt = html.escape(prompt_units[0] if len(prompt_units) > 0 else prompt).replace("\n", "<BR>")
        else:
            window_no = gen.get("window_no", 1)
            if window_no > len(prompt_units):
                window_no = len(prompt_units)
            window_no -= 1
            escaped_prompts = []
            for idx, prompt_unit in enumerate(prompt_units):
                escaped_prompt = html.escape(prompt_unit).replace("\n", "<BR>")
                if "W" in multi_prompts_gen_type and idx == window_no:
                    escaped_prompt = "<B>" + escaped_prompt + "</B>"
                escaped_prompts.append(escaped_prompt)
            prompt = "<BR><DIV style='height:8px'></DIV>".join(escaped_prompts)
        if is_edit_task:
            summary, prompt = task["prompt"], ""
        else:
            is_image = params["image_mode"] > 0
            audio_only = model_def.get("audio_only", False)
            count = params["batch_size"] if is_image else params["repeat_generation"]
            summary = "New " + ("Audio" if audio_only else "Image" if is_image else "Video")
            if count > 1:
                summary += f" x{count}"
            if audio_only:
                duration_def = model_def.get("duration_slider")
                duration = params["duration_seconds"]
                if duration_def is not None and duration > 0 and duration_def.get("name", duration_def["label"]).lower().startswith("max"):
                    summary += f", Max Duration is {duration:g} s"
            else:
                if not is_image:
                    frames = params["video_length"]
                    fps = get_computed_fps(params["force_fps"], base_model_type, params.get("video_guide"), params.get("video_source"))
                    summary += f", {frames} frames ({round(frames / fps, 1):g}s)"
                summary += f", {params['resolution']}, {params['num_inference_steps']} inference steps"

        details = f'<div class="generation-note">{html.escape(header_text)}</div>' if header_text else ""
        if prompt:
            prompt_label = "Enhanced Prompt" if enhanced else "Prompt"
            details += f'<div class="generation-prompt"><div class="generation-label">{prompt_label}</div><div>{prompt}</div></div>'
        thumbnails = ""
        for side in ("start", "end"):
            images, labels = task.get(f'{side}_image_data_base64'), task.get(f'{side}_image_labels')
            if images and labels:
                for i, (img_uri, img_label) in enumerate(zip(images, labels)):
                    label = html.escape(img_label)
                    thumbnails += f'<button type="button" class="generation-reference" onclick="showImageModal(\'current_{side}_{i}\')" aria-label="View {label}" title="View {label}"><img src="{img_uri}" alt="{label}" /><span>{label}</span></button>'
        if thumbnails:
            thumbnails = f'<td><div class="generation-references" role="group" aria-label="Generation References">{thumbnails}</div></td>'
        table = f'<div class="generation-table-wrap"><table id="PINFO"><tbody><tr><td class="generation-prompt-cell">{details}</td>{thumbnails}</tr></tbody></table></div>' if details or thumbnails else ""
        html_content = f'<div class="wangp-generation-info">{table}</div>'
        html_output = gr.HTML(html_content, visible= True)
        if last_was_audio:
            audio_choice = max(-1, audio_choice)
        else:
            choice = max(0, choice)
                    
        return *output_tabs, gallery_update(file_list, choice), gr.update() if last_was_audio else choice, *pack_audio_gallery_state(audio_file_list, audio_choice), html_output, gr.Button(visible=False), gr.Button(visible=True), gr.Row(visible=True), gr.update(), update_queue_data(queue), gr.Button(interactive=  abort_interactive), gr.Button(interactive=  early_stop_interactive, visible= early_stop_visible), gr.Button(visible= onemorewindow_visible), gr.update(label=summary, visible=True)



def finalize_generation(state):
    service = service_for(state)
    return service.finalized_updates() if service is not None else _finalize_generation(state)


def _finalize_generation(state):
    gen = get_gen_info(state)
    choice = gen.get("selected",0)
    if "in_progress" in gen:
        del gen["in_progress"]
    if gen.get("last_selected", True):
        file_list = gen.get("file_list", [])
        choice = len(file_list) - 1

    audio_file_list = gen.get("audio_file_list", [])
    audio_choice  = gen.get("audio_selected", -1)
    if gen.get("audio_last_selected", True):
        audio_choice = len(audio_file_list) - 1

    gen["extra_orders"] = 0
    last_was_audio = gen.get("last_was_audio", False)
    gen["current_gallery_source"] = "audio" if last_was_audio else "video"
    gallery_tabs = gr.Tabs(selected= "audio" if last_was_audio else "video_images")
    time.sleep(0.2)
    global gen_in_progress
    gen_in_progress = False
    gen["early_stop"] = False
    gen["early_stop_forwarded"] = False
    return gallery_tabs, 1 if last_was_audio else 0, gr.update() if last_was_audio else gallery_update(gen.get("file_list", []), choice),  *pack_audio_gallery_state(audio_file_list, audio_choice), gr.Button(interactive=  True), gr.Button(interactive=  True, visible= False), gr.Button(visible= True), gr.Button(visible= False), gr.Column(visible= False), gr.HTML(visible= False, value="")

def get_default_video_info():
    return "Please Select a Video / Image"    


def get_file_list(state, input_file_list, audio_files = False):
    gen = get_gen_info(state)
    with lock:
        if audio_files:
            file_list_name = "audio_file_list"
            file_settings_name = "audio_file_settings_list"
        else:
            file_list_name = "file_list"
            file_settings_name = "file_settings_list"

        if file_list_name in gen:
            file_list = gen[file_list_name]
            file_settings_list = gen[file_settings_name]
        else:
            file_list = []
            file_settings_list = []
            if input_file_list != None:
                if not isinstance(input_file_list, list): input_file_list = [input_file_list]
                for file_path in input_file_list:
                    file_path = get_gradio_file_path(file_path)
                    if not file_path or not os.path.isfile(file_path):
                        continue
                    file_settings, _, _ = get_settings_from_file(state, file_path, False, False, False)
                    file_list.append(file_path)
                    file_settings_list.append(file_settings)
 
            gen[file_list_name] = file_list 
            gen[file_settings_name] = file_settings_list 
    return file_list, file_settings_list

def set_file_choice(gen, file_list, choice, audio_files = False):
    if len(file_list) > 0: choice = max(choice,0)
    gen["audio_last_selected" if audio_files else "last_selected"] = (choice + 1) >= len(file_list)
    gen["audio_selected" if audio_files else "selected"] = choice
    gen["current_gallery_source"] = "audio" if audio_files else "video"
    gen["selected_video_time"] = None if audio_files or choice < 0 or choice >= len(file_list) or not has_video_file_extension(file_list[choice]) else 0.0
    if _deepy_hybrid is not None and gen is _deepy_hybrid._state["gen"]:
        _deepy_hybrid.host_changed()

def get_selected_late_processing_tabs_visibility(state):
    gen = get_gen_info(state)
    audio_files = gen.get("current_gallery_source", "video") == "audio"
    files = gen.get("audio_file_list" if audio_files else "file_list", [])
    choice = gen.get("audio_selected" if audio_files else "selected", -1 if audio_files else 0)
    if len(files) > 0:
        choice = min(len(files) - 1, max(choice, 0))
    if choice < 0 or choice >= len(files) or not os.path.isfile(files[choice]):
        return False, False, False
    is_audio = has_audio_file_extension(files[choice])
    is_video = has_video_file_extension(files[choice])
    is_image = not (is_audio or is_video)
    return is_audio, is_video or is_image, is_video

def select_audio(state, audio_files_paths, audio_file_selected):
    gen = get_gen_info(state)
    audio_file_list, audio_file_settings_list = get_file_list(state, unpack_audio_list(audio_files_paths))

    if audio_file_selected >= 0:
        choice = audio_file_selected
    else:
        choice = min(len(audio_file_list)-1, gen.get("audio_selected",-1)) if len(audio_file_list) > 0 else -1
    set_file_choice(gen,  audio_file_list, choice, audio_files=True )


video_guide_processes = "OPEDSLCMU"
all_guide_processes = video_guide_processes + "VGBH"

process_map_outside_mask = { "Y" : "depth", "W": "scribble", "X": "inpaint", "Z": "flow"}
process_map_video_guide = { "O": "pose_align", "P": "pose", "D" : "depth", "S": "scribble", "E": "canny", "L": "flow", "C": "gray", "M": "inpaint", "U": "identity"}
all_process_map_video_guide =  { "B": "face", "H" : "bbox"}
all_process_map_video_guide.update(process_map_video_guide)
processes_names = { "pose": "Open Pose", "pose_align": "Aligned Open Pose", "depth": "Depth Mask", "scribble" : "Shapes", "flow" : "Flow Map", "gray" : "Gray Levels", "inpaint" : "Inpaint Mask", "identity": "Identity Mask", "raw" : "Raw Format", "canny" : "Canny Edges", "face": "Face Movements", "bbox": "BBox"}


def resolve_media_creation_date(file_name, configs=None):
    creation_dt = extract_creation_datetime_from_metadata(configs) if isinstance(configs, dict) else None
    if creation_dt is None and has_audio_file_extension(file_name):
        try:
            creation_dt = resolve_audio_creation_datetime(file_name, wangp_metadata=configs if isinstance(configs, dict) else None)
        except Exception:
            creation_dt = None
    if creation_dt is None:
        creation_dt = get_file_creation_date(file_name)
    creation_date = str(creation_dt)
    if "." in creation_date:
        creation_date = creation_date[:creation_date.rfind(".")]
    return creation_date


def is_deepy_display_metadata(configs):
    return isinstance(configs, dict) and str(configs.get("model_type", "") or "").strip() == "Deepy"


def update_video_prompt_type(state, any_video_guide = False, any_video_mask = False, any_background_image_ref = False, process_type = None, default_update = ""):
    letters = default_update
    settings = get_current_model_settings(state)
    video_prompt_type = settings["video_prompt_type"]
    if process_type  is not None:
        video_prompt_type = del_in_sequence(video_prompt_type, video_guide_processes)
        for one_process_type in process_type: 
            for k,v in process_map_video_guide.items():
                if v== one_process_type:
                    letters += k
                    break
    model_type = get_state_model_type(state)
    model_def = get_model_def(model_type)
    guide_preprocessing = model_def.get("guide_preprocessing", None) 
    mask_preprocessing = model_def.get("mask_preprocessing", None) 
    guide_custom_choices = get_guide_custom_choices(model_def, settings.get("image_mode", 0))
    if any_video_guide: letters += "V"
    if any_video_mask: letters += "A"
    if any_background_image_ref: 
        video_prompt_type = del_in_sequence(video_prompt_type, "F")
        letters += "KI"
    validated_letters = ""
    for letter in letters:
        if not guide_preprocessing is None:
            if any(letter in choice for choice in guide_preprocessing["selection"] ):
                validated_letters += letter
                continue
        if not mask_preprocessing is None:
            if any(letter in choice for choice in mask_preprocessing["selection"] ):
                validated_letters += letter
                continue
        if not guide_custom_choices is None:
            if any(letter in choice for label, choice in guide_custom_choices["choices"] ):
                validated_letters += letter
                continue
    video_prompt_type = add_to_sequence(video_prompt_type, letters)
    settings["video_prompt_type"] = video_prompt_type 


def format_media_info(file_name, configs):
    """Shared selected-media details for Gradio and the standalone Deepy app."""
    model_def = None
    is_audio = False
    is_image = False
    is_video = False
    is_deleted = False
    if file_name is not None:
        values = [html.escape(os.path.basename(strip_virtual_media_suffix(file_name)))]
        labels = [ "File Name"]
        misc_values= []
        misc_labels = []
        pp_values= []
        pp_labels = []
        configs_summary = ""
        nb_audio_tracks =  0 

        if not os.path.isfile(file_name):
            is_deleted = True
            configs = None
        elif has_audio_file_extension(file_name):
            is_audio = True        
            width, height = 0, 0
            frames_count = fps = 1
        elif not has_video_file_extension(file_name):
            img = _open_image_input(file_name)
            width, height = img.size
            is_image = True
            frames_count = fps = 1
        else:
            fps, width, height, frames_count = get_video_info(file_name)
            is_video = True

        if is_audio or is_video:
            duration, layouts = get_media_duration_and_audio_layouts(file_name)
            actual_duration = f"{duration:.2f}s" if duration is not None else "Unknown"
            if len(layouts) > 1:
                layouts = [f"Track {index + 1}: {layout}" for index, layout in enumerate(layouts)]
            audio_summary = "; ".join(layouts) or "No audio"
            if is_audio:
                values += [html.escape(f"{actual_duration}, {audio_summary}")]
                labels += ["Duration"]
            nb_audio_tracks = len(layouts) if is_video else 0
            audio_tracks_summary = html.escape(f"{nb_audio_tracks}, {audio_summary}")

        if configs != None:
            # Deepy reference IDs alone do not describe a generation model.
            if configs.get("type") or configs.get("model_type"):
                video_model_name = configs.get("type") or get_model_name(configs["model_type"])
                if "-" in video_model_name: video_model_name = video_model_name[video_model_name.find("-")+2:]
                misc_values += [video_model_name]
                misc_labels += ["Model"]
            metadata_model_def = get_model_def(configs.get("model_type", None))
            config_groups = get_model_config_groups(configs.get("model_type", None), metadata_model_def) if metadata_model_def is not None else [{} for _ in model_config_groups.CONFIG_KEYS]
            configs_summary = model_config_groups.format_config_selection(config_groups, configs.get("config", ""))
            model_modes_def = metadata_model_def.get("model_modes", None) if metadata_model_def is not None else None
            if model_modes_def is not None and "model_mode" in configs and configs.get("image_mode", 0) in model_modes_def.get("image_modes", [0, 1, 2]):
                misc_values += [next((label for label, value in model_modes_def["choices"] if value == configs["model_mode"]), configs["model_mode"])]
                misc_labels += [model_modes_def["label"]]
            video_temporal_upsampling = temporal_upsampler_api.format_temporal_upsampling_label(temporal_upsampler_api.normalize_temporal_upsampling_value(configs.get("temporal_upsampling", "")))
            video_spatial_upsampling = upsampler_api.format_upsampling_label(upsampler_api.normalize_upsampling_value(configs.get("spatial_upsampling", "")))
            video_film_grain_intensity = configs.get("film_grain_intensity", 0)
            video_film_grain_saturation = configs.get("film_grain_saturation", 0.5)
            video_postprocess_audio = audio_processor_api.normalize_method(configs.get("postprocess_audio", "") or "")
            video_postprocess_audio_meta = audio_processor_api.method_metadata(video_postprocess_audio)
            video_postprocess_audio_prompt = configs.get("postprocess_audio_prompt", "")
            video_postprocess_audio_neg_prompt = configs.get("postprocess_audio_neg_prompt", "")
            video_seed = configs.get("seed", -1)
            video_postprocess_audio_seed = configs.get("postprocess_audio_seed", video_seed)
            video_replace_voice_method = audio_processor_api.normalize_method(configs.get("replace_voice_method", "") or "")
            if len(video_spatial_upsampling) > 0:
                video_temporal_upsampling += " " + video_spatial_upsampling
            if len(video_temporal_upsampling) > 0:
                pp_values += [ video_temporal_upsampling ]
                pp_labels += [ "Upsampling" ]
            if video_film_grain_intensity > 0:
                pp_values += [ f"Intensity={video_film_grain_intensity}, Saturation={video_film_grain_saturation}" ]
                pp_labels += [ "Film Grain" ]
            if video_postprocess_audio == "control":
                pp_values += [ "Control Video Audio Track" ]
                pp_labels += [ "Audio Postprocess" ]
            elif video_postprocess_audio:
                audio_summary = audio_processor_api.format_method_label(video_postprocess_audio)
                audio_details = []
                if video_postprocess_audio_meta["needs_prompt"] or video_postprocess_audio_meta["needs_negative_prompt"]:
                    audio_details.append(f'Prompt="{video_postprocess_audio_prompt}", Neg Prompt="{video_postprocess_audio_neg_prompt}", Seed={video_postprocess_audio_seed}')
                pp_values += [", ".join([audio_summary, *audio_details]) if audio_details else audio_summary]
                pp_labels += [ "Audio Postprocess" ]
            if video_replace_voice_method:
                pp_values += [ audio_processor_api.format_method_label(video_replace_voice_method) ]
                pp_labels += [ "Voice Replacement" ]


        if is_deepy_display_metadata(configs):
            values += ["Deepy"]
            labels += ["Made By"]
            video_prompt = html.escape(str(configs.get("prompt", "") or "")[:1024]).replace("\n", "<BR>")
            if len(video_prompt) > 0:
                values += [video_prompt]
                labels += ["Prompt"]
            video_creation_date = "Deleted" if is_deleted else resolve_media_creation_date(file_name, configs)
            if is_image:
                values += [f"{width}x{height}"]
                labels += ["Resolution"]
            elif is_video:
                values += [f"{width}x{height}", f"{frames_count} frames (duration={frames_count/fps:.1f}s, fps={round(fps)})"]
                labels += ["Resolution", "Frames"]
            if nb_audio_tracks > 0:
                values += [audio_tracks_summary]
                labels += ["Nb Audio Tracks"]
            values += [video_creation_date]
            labels += ["Creation Date"]
        elif configs == None or not "seed" in configs:
            values += misc_values
            labels += misc_labels
            
            video_creation_date = "Deleted" if is_deleted else resolve_media_creation_date(file_name, configs)
            if is_audio:
                pass
            elif is_image:
                values += [f"{width}x{height}"]
                labels += ["Resolution"]
            elif is_video:
                values += [f"{width}x{height}",  f"{frames_count} frames (duration={frames_count/fps:.1f}s, fps={round(fps)})"]
                labels += ["Resolution", "Frames"]
                extra_values, extra_labels = get_video_summary_extras(file_name)
                values += extra_values
                labels += extra_labels
            if nb_audio_tracks  > 0:
                values +=[audio_tracks_summary]
                labels +=["Nb Audio Tracks"]

            values += pp_values
            labels += pp_labels

            values +=[video_creation_date]
            labels +=["Creation Date"]
        else: 
            video_prompt_text = str(configs.get("prompt", "") or "")
            enhanced_video_prompt_text = str(configs.get("enhanced_prompt", "") or "")
            video_video_prompt_type = configs.get("video_prompt_type", "")
            video_image_prompt_type = configs.get("image_prompt_type", "")
            video_audio_prompt_type = configs.get("audio_prompt_type", "")
            def check(src, cond):
                pos, neg = cond if isinstance(cond, tuple) else (cond, None)
                if not all_letters(src, pos): return False
                if neg is not None and any_letters(src, neg): return False
                return True
            image_outputs = configs.get("image_mode",0) > 0
            video_model_type =  configs.get("model_type", "t2v")
            model_family = get_model_family(video_model_type)
            model_def = get_model_def(video_model_type)
            multi_prompts_gen_type = prompt_parser.normalize_multi_prompts_mode(configs.get("multi_prompts_gen_type"), "FG")
            prompt_history = prompt_parser.parse_prompt_history(video_prompt_text, enhanced_video_prompt_text, multi_prompts_gen_type)
            if prompt_history is not None:
                original_prompts, enhanced_prompts = prompt_history
                video_prompt_text = prompt_parser.serialize_prompt_units("", original_prompts, multi_prompts_gen_type)
                enhanced_video_prompt_text = prompt_parser.serialize_prompt_units("", enhanced_prompts, multi_prompts_gen_type)
            video_prompt = html.escape(video_prompt_text[:4096]).replace("\n", "<BR>")
            enhanced_video_prompt = html.escape(enhanced_video_prompt_text[:4096]).replace("\n", "<BR>")
            map_video_prompt  = {"V" : "Control Image" if image_outputs else "Control Video", ("VA", "U") : "Mask Image" if image_outputs else "Mask Video", "I" : "Reference Images", "&": "HDR Output"}
            map_image_prompt  = {"V" : "Source Video", "L" : "Last Video", "S" : "Start Image", "E" : "End Image"}
            map_audio_prompt  = {"A" : "Audio Source", "O": "Force Output Audio", "B" : "Audio Source #2", "D" : "Audio Source #3", "K": "Control Video Audio Track", "N": "Normalized Audio Volumes"}
            custom_audio_option_label, custom_audio_option_flag = get_audio_prompt_type_custom_option_def(model_def)
            if len(custom_audio_option_flag) > 0:
                map_audio_prompt[custom_audio_option_flag] = custom_audio_option_label
            audio_prompt_type_sources_def = model_def.get("audio_prompt_type_sources", None)
            if isinstance(audio_prompt_type_sources_def, dict):
                custom_flags = audio_prompt_type_sources_def.get("custom_flags", {})
                if isinstance(custom_flags, dict):
                    for flag, label in custom_flags.items():
                        flag = str(flag or "")
                        if len(flag) == 1 and flag in "0123456789" and isinstance(label, str) and len(label) > 0:
                            map_audio_prompt[flag] = label
            video_other_prompts =  [ v for s,v in map_image_prompt.items() if all_letters(video_image_prompt_type,s)] \
                                 + [ v for s,v in map_video_prompt.items() if check(video_video_prompt_type,s)] \
                                 + [ v for s,v in map_audio_prompt.items() if all_letters(video_audio_prompt_type,s)] 
            custom_guide_def = model_def.get("custom_guide")
            if configs.get("custom_guide_used") and custom_setting_visible(custom_guide_def, video_video_prompt_type, video_audio_prompt_type):
                video_other_prompts.append(html.escape(custom_guide_def.get("name", custom_guide_def.get("label", "Custom Guide"))))
            any_mask = "A" in video_video_prompt_type and not "U" in video_video_prompt_type            
            multiple_submodels = model_def.get("multiple_submodels", False)
            video_other_prompts = ", ".join(video_other_prompts)
            if is_audio:
                video_resolution = None
                video_length_summary = None
                video_length_label = ""
                original_fps = 0
                video_num_inference_steps = configs.get("num_inference_steps", None)
            else:
                video_length = configs.get("video_length", 0)
                original_fps= int(video_length/frames_count*fps)
                video_length_summary = f"{video_length} frames"
                video_window_no = configs.get("window_no", 0)
                if video_window_no > 0: video_length_summary +=f", Window no {video_window_no }" 
                if is_image:
                    video_length_summary = configs.get("batch_size", 1)
                    video_length_label = "Number of Images"
                else:
                    video_length_summary += " ("
                    video_length_label = "Video Length"
                    if video_length != frames_count: video_length_summary += f"real: {frames_count} frames, "
                    video_length_summary += f"{frames_count/fps:.1f}s, {round(fps)} fps)"
                video_resolution = configs.get("resolution", "")
                real_video_resolution = f"{width}x{height}"
                if video_resolution !=  real_video_resolution: video_resolution +=  f" (real: {real_video_resolution})"
                video_num_inference_steps = configs.get("num_inference_steps", 0)

            video_guidance_scale = configs.get("guidance_scale", None)
            video_guidance2_scale = configs.get("guidance2_scale", None)
            video_guidance3_scale = configs.get("guidance3_scale", None)
            video_audio_guidance_scale = configs.get("audio_guidance_scale", None)
            video_alt_guidance_scale = configs.get("alt_guidance_scale", None)
            video_alt_scale = configs.get("alt_scale", None)
            video_temperature = configs.get("temperature", None)
            video_top_p = configs.get("top_p", None)
            video_top_k = configs.get("top_k", None)
            video_switch_threshold = configs.get("switch_threshold", 0)
            video_switch_threshold2 = configs.get("switch_threshold2", 0)
            video_model_switch_phase = configs.get("model_switch_phase", 1)
            video_guidance_phases = configs.get("guidance_phases", 0)
            video_embedded_guidance_scale = configs.get("embedded_guidance_scale", None)
            video_guidance_label = "Guidance"
            visible_phases = model_def.get("visible_phases", video_guidance_phases)
            if model_def.get("embedded_guidance", False):
                video_guidance_scale = video_embedded_guidance_scale
                video_guidance_label = "Embedded Guidance Scale"
            elif video_guidance_phases == 0 or visible_phases ==0:
                video_guidance_scale = None 
            elif video_guidance_phases > 0:
                if video_guidance_phases == 1 and visible_phases >=1:
                    video_guidance_scale = f"{video_guidance_scale}"
                elif video_guidance_phases == 2 and visible_phases >=2:
                    if multiple_submodels:
                        video_guidance_scale = f"{video_guidance_scale} (High Noise), {video_guidance2_scale} (Low Noise) with Switch at Noise Level {video_switch_threshold}"
                    else:
                        video_guidance_scale = f"{video_guidance_scale}, {video_guidance2_scale}" + ("" if video_switch_threshold ==0 else f" with Guidance Switch at Noise Level {video_switch_threshold}")
                elif visible_phases >=3:
                    video_guidance_scale = f"{video_guidance_scale}, {video_guidance2_scale} & {video_guidance3_scale} with Switch at Noise Levels {video_switch_threshold} & {video_switch_threshold2}"
                    if multiple_submodels:
                        video_guidance_scale += f" + Model Switch at {video_switch_threshold if video_model_switch_phase ==1 else video_switch_threshold2}"
            video_phases_label = "Phases"
            video_phases_value = None
            if video_guidance_phases != visible_phases :
                video_phases_value = str(video_guidance_phases)

            if  model_def.get("flow_shift", False): 
                video_flow_shift = configs.get("flow_shift", None)
            else:
                video_flow_shift = None 

            video_video_guide_outpainting = configs.get("video_guide_outpainting", "")
            video_video_guide_outpainting_ratio = configs.get("video_guide_outpainting_ratio", "")
            video_outpainting = ""
            if len(video_video_guide_outpainting) > 0  and not video_video_guide_outpainting.startswith("#") \
                    and (any_letters(video_video_prompt_type, "VFK") ) :
                video_video_guide_outpainting = video_video_guide_outpainting.split(" ")
                video_outpainting = f"Top={video_video_guide_outpainting[0]}%, Bottom={video_video_guide_outpainting[1]}%, Left={video_video_guide_outpainting[2]}%, Right={video_video_guide_outpainting[3]}%" 
            elif len(video_video_guide_outpainting_ratio) > 0 and not video_video_guide_outpainting.startswith("#") and any_letters(video_video_prompt_type, "VFK"):
                video_outpainting = "Top=0%, Bottom=0%, Left=0%, Right=0%"
            if len(video_outpainting) > 0 and len(video_video_guide_outpainting_ratio) > 0:
                video_outpainting += f", Fit {video_video_guide_outpainting_ratio}"
            video_creation_date = resolve_media_creation_date(file_name, configs)
            video_generation_time = format_generation_time(float(configs.get("generation_time", "0")))
            video_activated_loras = configs.get("activated_loras", [])
            video_loras_multipliers = configs.get("loras_multipliers", "")
            video_loras_multipliers =  preparse_loras_multipliers(video_loras_multipliers)
            video_loras_multipliers += [""] * len(video_activated_loras)
            lora_dir = None if video_model_type is None else get_lora_dir(video_model_type)
            video_activated_loras = [ f"<span class='copy-swap' tabindex=0><SPAN class='copy-swap__trunc' >{get_lora_local_path(None, lora)}</span><span class='copy-swap__full'>{get_lora_URL(lora_dir, lora) .split('|')[0]}</span></span>" for lora in video_activated_loras] 
            video_activated_loras = [ f"<TR><TD style='padding-top:0px;padding-left:0px;width:100%;max-width:0'>{lora}</TD><TD style='width:1%;white-space:nowrap;vertical-align:top'>x{str(multiplier).strip() or '1'}</TD></TR>" for lora, multiplier in zip(video_activated_loras, video_loras_multipliers) ]
            video_activated_loras_str = "<TABLE style='border:0px;padding:0px;width:100%;table-layout:fixed'>" + "".join(video_activated_loras) + "</TABLE>" if len(video_activated_loras) > 0 else ""
            video_duration_seconds = configs.get("duration_seconds", 0)
            duration_def = model_def.get("duration_slider", None)
            if duration_def is not None and video_duration_seconds > 0:
                misc_values += [ f"{video_duration_seconds}s"]
                misc_labels += [html.escape(duration_def.get("name", duration_def.get("label", "Max Duration")))]
            prompt_class = model_def.get("prompt_class","Text Prompt")
            values +=  misc_values + [video_prompt]
            labels += misc_labels + [f"Original {prompt_class}" if enhanced_video_prompt else prompt_class]
            video_comments = html.escape(str(configs.get("comments", "") or "")[:4096]).replace("\n", "<BR>")
            if len(video_comments) > 0:
                values += [video_comments]
                labels += ["Comments"]
            alt_prompt_def = model_def.get("alt_prompt", None)
            if alt_prompt_def is not None:
                alt_prompt_label = alt_prompt_def.get("name", alt_prompt_def.get("label")) 
                alt_prompt_text = str(configs.get("alt_prompt", "") or "")
                enhanced_alt_prompt_text = str(configs.get("enhanced_alt_prompt", "") or "")
                alt_prompt_mode = multi_prompts_gen_type if model_def.get("alt_prompt_inherits_prompt_paragraphs", False) else "FG"
                alt_prompt_history = prompt_parser.parse_prompt_history(alt_prompt_text, enhanced_alt_prompt_text, alt_prompt_mode)
                if alt_prompt_history is not None:
                    original_alt_prompts, enhanced_alt_prompts = alt_prompt_history
                    alt_prompt_text = prompt_parser.serialize_prompt_units("", original_alt_prompts, alt_prompt_mode)
                    enhanced_alt_prompt_text = prompt_parser.serialize_prompt_units("", enhanced_alt_prompts, alt_prompt_mode)
                alt_prompt = html.escape(alt_prompt_text[:4096]).replace("\n", "<BR>")
                if len(alt_prompt):
                    values += [alt_prompt]
                    labels += [f"Original {alt_prompt_label}" if enhanced_alt_prompt_text else alt_prompt_label]
                enhanced_alt_prompt = html.escape(enhanced_alt_prompt_text[:4096]).replace("\n", "<BR>")
                if len(enhanced_alt_prompt):
                    values += [enhanced_alt_prompt]
                    labels += [alt_prompt_label]
            extra_info = configs.get("extra_info", None)
            if isinstance(extra_info, dict):
                for extra_label, extra_text in extra_info.items():
                    if extra_text is None:
                        continue
                    extra_text = str(extra_text).strip()
                    if len(extra_text) == 0:
                        continue
                    values += [html.escape(extra_text[:4096]).replace("\n", "<BR>")]
                    labels += [html.escape(str(extra_label))]
            if len(enhanced_video_prompt):
                values += [enhanced_video_prompt]
                labels += [prompt_class]
            if len(video_other_prompts) >0 :
                values += [video_other_prompts]
                labels += ["Other Prompts"]
            def gen_process_list(map):
                video_preprocesses = ""
                for k,v in map.items():
                    if k in video_video_prompt_type:
                        process_name = processes_names[v]
                        video_preprocesses += process_name if len(video_preprocesses) == 0 else ", " + process_name 
                return video_preprocesses 

            video_preprocesses_in = gen_process_list(all_process_map_video_guide) if "V" else ""
            video_preprocesses_out = gen_process_list(process_map_outside_mask) if "V" else ""
            if "N" in video_video_prompt_type:
                alt = video_preprocesses_in
                video_preprocesses_in = video_preprocesses_out
                video_preprocesses_out = alt
            if len(video_preprocesses_in) >0 and "V" in video_video_prompt_type:
                values += [video_preprocesses_in]
                labels += [ "Process Inside Mask" if any_mask else "Preprocessing"]

            if len(video_preprocesses_out) >0 and "V" in video_video_prompt_type:
                values += [video_preprocesses_out]
                labels += [ "Process Outside Mask"]
            video_frames_positions = configs.get("frames_positions", "")
            if "F" in video_video_prompt_type and len(video_frames_positions):
                values += [video_frames_positions]
                labels += [ "Injected Frames"]
            if len(video_outpainting) >0:
                values += [video_outpainting]
                labels += ["Outpainting"]
            if input_video_strength_visible(model_def, video_image_prompt_type, video_video_prompt_type):
                values += [configs.get("input_video_strength",1)]
                labels += [extra_settings.get_summary_label("input_video_strength", model_def, fallback="Input Image Strength")]

            if "G" in video_video_prompt_type and "V" in video_video_prompt_type:
                values += [configs.get("denoising_strength",1)]
                labels += [extra_settings.get_summary_label("denoising_strength", model_def, fallback="Denoising Strength")]
            if ("G" in video_video_prompt_type or model_def.get("mask_strength_always_enabled", False)) and "A" in video_video_prompt_type and "U" not in video_video_prompt_type:
                values += [configs.get("masking_strength",1)]
                labels += [extra_settings.get_summary_label("masking_strength", model_def, fallback="Masking Strength")]

            video_sample_solver = configs.get("sample_solver", "")
            if model_def.get("sample_solvers", None) is not None and len(video_sample_solver) > 0 :
                values += [video_sample_solver]
                labels += ["Sampler Solver"]                                        
            values += [video_resolution, video_length_summary, video_seed, video_phases_value, video_guidance_scale, video_audio_guidance_scale]
            labels += ["Resolution", video_length_label, "Seed", video_phases_label,  video_guidance_label, "Audio Guidance Scale"]
            if is_video:
                extra_values, extra_labels = get_video_summary_extras(file_name)
                values += extra_values
                labels += extra_labels
            video_custom_settings = configs.get("custom_settings", None)
            if isinstance(video_custom_settings, dict):
                custom_settings = get_model_custom_settings(model_def)
                for idx, setting_def in enumerate(custom_settings):
                    if not custom_setting_visible(setting_def, video_video_prompt_type, video_audio_prompt_type):
                        continue
                    setting_id = setting_def.get("id", get_custom_setting_id(setting_def, idx))
                    setting_value = video_custom_settings.get(setting_id, None)
                    if setting_value is None:
                        continue
                    if isinstance(setting_value, str) and len(setting_value.strip()) == 0:
                        continue
                    values += [get_custom_setting_display_value(setting_def, setting_value)]
                    labels += [setting_def.get("name", f"Custom Setting {idx + 1}")]
            if model_def.get("temperature", True) and video_temperature is not None:
                values += [video_temperature]
                labels += ["Temperature"]
            if model_def.get("top_p_slider", False) and video_top_p is not None:
                values += [video_top_p]
                labels += ["Top-p"]
            if model_def.get("top_k_slider", False) and video_top_k is not None:
                values += [video_top_k]
                labels += ["Top-k"]
            if is_audio and model_def.get("pause_between_sentences", False):
                values += [configs.get("pause_seconds", 0.0)]
                labels += ["Pause (s)"]
            alt_guidance_type = model_def.get("alt_guidance", None)
            if alt_guidance_type is not None and video_alt_guidance_scale is not None:
                values += [video_alt_guidance_scale]
                labels += [alt_guidance_type]
            alt_scale_type = model_def.get("alt_scale", None)
            if alt_scale_type is not None and video_alt_scale is not None:
                values += [video_alt_scale]
                labels += [alt_scale_type]
            if model_def.get("flow_shift", False):
                values += [video_flow_shift]
                labels += ["Shift Scale"]
            if model_def.get("inference_steps", True) and video_num_inference_steps is not None:
                values += [video_num_inference_steps]
                labels += ["Num Inference steps"]
            video_negative_prompt = configs.get("negative_prompt", "")
            if len(video_negative_prompt) > 0:
                values += [video_negative_prompt]
                labels += ["Negative Prompt"]        
            video_NAG_scale = configs.get("NAG_scale", None)
            if video_NAG_scale is not None and video_NAG_scale > 1: 
                video_NAG_tau = configs.get("NAG_tau", None)
                video_NAG_alpha = configs.get("NAG_alpha", None)
                values += [f"scale={video_NAG_scale}, tau={video_NAG_tau}, alpha={video_NAG_alpha}"]
                labels += ["NAG"]      
            video_self_refiner_setting = configs.get("self_refiner_setting", 0)
            if video_self_refiner_setting > 0:  
                video_self_refiner_plan = configs.get('self_refiner_plan','')
                if len(video_self_refiner_plan)==0: video_self_refiner_plan ='default'
                values += [f"Norm P{video_self_refiner_setting}, Plan='{video_self_refiner_plan}', Uncertainty={configs.get('self_refiner_f_uncertainty',0.0)}, Certain Percentage='{configs.get('self_refiner_certain_percentage', 0.999)} "]
                # values += [f"Norm P{video_self_refiner_setting}, Plan='{video_self_refiner_plan}'"]
                labels += ["Self Refiner"]      
            video_apg_switch = configs.get("apg_switch", None)
            if video_apg_switch is not None and video_apg_switch != 0: 
                values += ["on"]
                labels += ["APG"]      
            video_motion_amplitude = configs.get("motion_amplitude", 1.)
            if  video_motion_amplitude != 1: 
                values += [video_motion_amplitude]
                labels += ["Motion Amplitude"]
            control_net_weight_name = model_def.get("control_net_weight_name", "")
            control_net_weight = ""
            if len(control_net_weight_name):
                video_control_net_weight = configs.get("control_net_weight", 1)
                if len(filter_letters(video_video_prompt_type, video_guide_processes))> 1:
                    video_control_net_weight2 = configs.get("control_net_weight2", 1)
                    control_net_weight = f"{control_net_weight_name} #1={video_control_net_weight}, {control_net_weight_name} #2={video_control_net_weight2}"
                else:
                    control_net_weight = f"{control_net_weight_name}={video_control_net_weight}"
            control_net_weight_alt_name = model_def.get("control_net_weight_alt_name", "")
            if len(control_net_weight_alt_name) >0:
                if len(control_net_weight): control_net_weight += ", "
                control_net_weight += control_net_weight_alt_name + "=" + str(configs.get("control_net_weight_alt", 1))
            if len(control_net_weight) > 0: 
                values += [control_net_weight]
                labels += ["Control Net Weights"]      

            audio_scale_name = model_def.get("audio_scale_name", "")
            if len(audio_scale_name) > 0 and any_letters(video_audio_prompt_type,"AB"):
                values += [configs.get("audio_scale", 1)]
                labels += [audio_scale_name]

            video_skip_steps_cache_type = configs.get("skip_steps_cache_type", "")
            video_skip_steps_multiplier = configs.get("skip_steps_multiplier", 0)
            video_skip_steps_cache_start_step_perc = configs.get("skip_steps_start_step_perc", 0)
            if len(video_skip_steps_cache_type) > 0:
                video_skip_steps_cache = {"tea": "TeaCache", "mag": "MagCache", "spectrum": "Spectrum", "first_block": "First Block Cache"}.get(video_skip_steps_cache_type, video_skip_steps_cache_type)
                if video_skip_steps_cache_type in ("tea", "mag"):
                    video_skip_steps_cache += f" x{video_skip_steps_multiplier}"
                elif video_skip_steps_cache_type == "first_block":
                    video_skip_steps_cache += f" (threshold {float(video_skip_steps_multiplier):g})"
                if video_skip_steps_cache_start_step_perc >0:  video_skip_steps_cache += f", Start from {video_skip_steps_cache_start_step_perc}%"
                values += [ video_skip_steps_cache ]
                labels += [ "Skip Steps" ]

            values += pp_values
            labels += pp_labels

            if len(video_activated_loras_str) > 0:
                values += [video_activated_loras_str]
                labels += ["LoRAs"] 
            if nb_audio_tracks  > 0:
                values +=[audio_tracks_summary]
                labels +=["Nb Audio Tracks"]
            values += [ video_creation_date, video_generation_time ]
            labels += [ "Creation Date", "Generation Time" ]
        if configs_summary:
            values += [configs_summary]
            labels += ["Configs"]
        labels = [label for value, label in zip(values, labels) if value is not None]
        values = [value for value in values if value is not None]

        table_style = """<STYLE>
            #video_info, #video_info TR, #video_info TD {
            background-color: transparent; 
            color: inherit; 
            padding: 3px 4px;
            border:0px !important;
            font-size:11px;
            }
            </STYLE>
        """
        rows = [f"<TR><TD style='text-align: right;' WIDTH=1% NOWRAP VALIGN=TOP>{label}</TD><TD><B>{value}</B></TD></TR>" for label, value in zip(labels, values)]
        html_content = f"{table_style}<TABLE ID=video_info WIDTH=100%>" + "".join(rows) + "</TABLE>"
    else:
        html_content =  get_default_video_info()
    return html_content, is_image, is_video, is_audio, is_deleted


# Keep event_data required: on Python 3.10, `event_data: gr.EventData = None` becomes Optional[EventData],
# and Gradio 5.29 stops injecting the gallery selection index, breaking the selected media choice.
def select_media(state, current_gallery_tab, input_file_list, file_selected, audio_files_paths, audio_file_selected, source, current_spatial_upsampling, current_spatial_parameters, spatial_help_target_id, event_data: gr.EventData):
    gen = get_gen_info(state)
    shared_gallery = service_for(state) is not None
    late_parameter_count = len(upsampler_api.ui_parameter_definitions(upsampler_api.PARAMETER_UI_LATE_POSTPROCESSING)) * 2 + 2
    if source=="video":
        if current_gallery_tab != 0:
            return [gr.update()] * (16 + late_parameter_count)
        file_list, file_settings_list = get_file_list(state, input_file_list)
        data = event_data._data if event_data is not None else None
        if not shared_gallery and data!=None and isinstance(data, dict):
            choice = data.get("index",0)
        else:
            choice = gen.get("selected", file_selected)
        choice = min(len(file_list)-1, choice)
        if choice < 0 and len(file_list) > 0: choice = 0
        if not shared_gallery:
            set_file_choice(gen, file_list, choice)
        files, settings_list = file_list, file_settings_list
    else:
        if current_gallery_tab != 1:
            return [gr.update()] * (16 + late_parameter_count)
        audio_file_list, audio_file_settings_list = get_file_list(state, unpack_audio_list(audio_files_paths), audio_files= True)
        if not shared_gallery and audio_file_selected >= 0:
            choice = audio_file_selected
        else:
            choice = gen.get("audio_selected",-1)
        choice = min(len(audio_file_list)-1, choice)
        if choice < 0 and len(audio_file_list) > 0: choice = 0
        if not shared_gallery:
            set_file_choice(gen, audio_file_list, choice, audio_files=True)
        files, settings_list = audio_file_list, audio_file_settings_list

    html_content, is_image, is_video, is_audio, is_deleted = format_media_info(files[choice] if files else None, settings_list[choice] if files else None)
    visible= len(files) > 0
    visual_media = is_image or is_video
    post_temporal_update = gr.update(visible=False, **({"value": ""} if is_image else {}))
    post_temporal_method_update = gr.update(visible=is_video, **({} if is_video else {"value": ""}))
    post_temporal_multiplier_update = gr.update() if is_video else gr.update(visible=False)
    if visual_media:
        spatial_state = upsampler_api.late_postprocessing_ui_state(current_spatial_upsampling, image_outputs=is_image, parameter_values=current_spatial_parameters)
        post_spatial_update = gr.update(visible=False, value=spatial_state["value"])
        post_spatial_method_update = gr.update(choices=spatial_state["method_choices"], value=spatial_state["method"], visible=True)
        post_spatial_ratio_update = gr.update(choices=spatial_state["ratio_choices"], value=spatial_state["scale"] if spatial_state["ratio_choices"] else None, visible=bool(spatial_state["method"] and spatial_state["ratio_choices"]))
    else:
        spatial_state = {"method": "", "method_choices": [("None", "")], "parameters": upsampler_api.parameter_ui_state("", upsampler_api.PARAMETER_UI_LATE_POSTPROCESSING, current_spatial_parameters)}
        post_spatial_update = gr.update(visible=False, value="")
        post_spatial_method_update = gr.update(visible=False, value="")
        post_spatial_ratio_update = gr.update(visible=False)
    help_media_profile = upsampler_api.UPSAMPLER_PROFILE_IMAGE if is_image else upsampler_api.UPSAMPLER_PROFILE_VIDEO
    help_title, help_markdown = upsampler_api.spatial_help_popup(spatial_state["method_choices"], media_profile=help_media_profile, field_help=field_help)
    post_spatial_help_update = gr.update(value=field_help.render_marker(spatial_help_target_id, upsampler_api.spatial_help_id(spatial_state["method_choices"], media_profile=help_media_profile), title=help_title, markdown=help_markdown))
    late_parameter_state = spatial_state["parameters"]
    late_parameter_updates = [post_spatial_help_update, *(gr.update(visible=str(parameter["name"]) in late_parameter_state["active"]) for parameter in late_parameter_state["definitions"]),
                              *(gr.update(value=late_parameter_state["values"][str(parameter["name"])], **({"label": parameter["label"], "info": parameter.get("description"), "minimum": parameter.get("minimum", 0), "maximum": parameter.get("maximum", 1), "step": parameter.get("step", 1)} if upsampler_api.parameter_component_type(parameter) == "slider" else {})) for parameter in late_parameter_state["definitions"]), late_parameter_state["values"]]
    return choice if source=="video" else gr.update(), html_content, gr.update(visible=visible and is_video) , gr.update(visible=visible and is_image), gr.update(visible=visible and is_audio), gr.update(visible=visible and is_deleted and source=="video"), gr.update(visible=visible and is_deleted and source=="audio"), gr.update(visible=visible and is_audio), gr.update(visible=visible and (is_video or is_image)) , gr.update(visible=visible and is_video), post_temporal_update, post_temporal_method_update, post_temporal_multiplier_update, post_spatial_update, post_spatial_method_update, post_spatial_ratio_update, *late_parameter_updates

def convert_image(image):

    from PIL import ImageOps
    from typing import cast
    if isinstance(image, str):
        image = _open_image_input(image)
    image = image.convert('RGB')
    return cast(Image, ImageOps.exif_transpose(image))

def get_resampled_video(video_in, start_frame, max_frames, target_fps, bridge='torch', hdr_linear=False):
    if hdr_linear:
        return decode_video_frames_ffmpeg(video_in, start_frame, max_frames, target_fps=target_fps, bridge=bridge, hdr_linear=True)
    return get_resampled_video_transparent(video_in, start_frame, max_frames, target_fps, bridge)

def _virtual_media_has_hdr_flag(value):
    spec = parse_virtual_media_path(value) if isinstance(value, str) else None
    extras = {str(k).strip().lower(): str(v).strip().lower() for k, v in (spec.extras if spec is not None else ())}
    if extras.get("hdr") in {"1", "true", "yes"}:
        return True
    entry = get_virtual_media_entry(value) if isinstance(value, str) else None
    return bool(isinstance(entry, dict) and entry.get("hdr"))

def _video_input_is_hdr(value):
    if _virtual_media_has_hdr_flag(value):
        return True
    if not isinstance(value, str):
        return False
    metadata = probe_video_stream_metadata(value)
    return bool(metadata and (metadata.get("hdr") or metadata.get("needs_tonemap")))

# def get_resampled_video(video_in, start_frame, max_frames, target_fps):
#     from torchvision.io import VideoReader
#     import torch
#     from shared.utils.utils import resample

#     vr = VideoReader(video_in, "video")
#     meta = vr.get_metadata()["video"]

#     fps = round(float(meta["fps"][0]))
#     duration_s = float(meta["duration"][0])
#     num_src_frames = int(round(duration_s * fps))  # robust length estimate

#     if max_frames < 0:
#         max_frames = max(int(num_src_frames / fps * target_fps + max_frames), 0)

#     frame_nos = resample(
#         fps, num_src_frames,
#         max_target_frames_count=max_frames,
#         target_fps=target_fps,
#         start_target_frame=start_frame
#     )
#     if len(frame_nos) == 0:
#         return torch.empty((0,))  # nothing to return

#     target_ts = [i / fps for i in frame_nos]

#     # Read forward once, grabbing frames when we pass each target timestamp
#     frames = []
#     vr.seek(target_ts[0])
#     idx = 0
#     tol = 0.5 / fps  # half-frame tolerance
#     for frame in vr:
#         t = float(frame["pts"])       # seconds
#         if idx < len(target_ts) and t + tol >= target_ts[idx]:
#             frames.append(frame["data"].permute(1,2,0))  # Tensor [H, W, C]
#             idx += 1
#             if idx >= len(target_ts):
#                 break

#     return frames


def get_preprocessor(process_type, inpaint_color, pre_video_guide=None):
    from preprocessing.processors import get_preprocessor as create_preprocessor

    return create_preprocessor(process_type, inpaint_color, pre_video_guide, server_config)


def extract_faces_from_video_with_mask(input_video_path, input_mask_path, max_frames, start_frame, target_fps, size = 512):
    if not input_video_path or max_frames <= 0:
        return None, None
    pad_frames = 0
    if start_frame < 0:
        pad_frames= -start_frame
        max_frames += start_frame
        start_frame = 0

    any_mask = input_mask_path != None
    video = get_resampled_video(input_video_path, start_frame, max_frames, target_fps)
    if len(video) == 0: return None
    frame_height, frame_width, _ = video[0].shape
    num_frames = len(video)
    if any_mask:
        mask_video = get_resampled_video(input_mask_path, start_frame, max_frames, target_fps)
        num_frames = min(num_frames, len(mask_video))
    if num_frames == 0: return None
    video = video[:num_frames]
    if any_mask:
        mask_video = mask_video[:num_frames]

    from preprocessing.face_preprocessor  import FaceProcessor 
    face_processor = FaceProcessor()

    face_list = []
    for frame_idx in range(num_frames):
        frame = video[frame_idx].cpu().numpy() 
        # video[frame_idx] = None
        if any_mask:
            mask = Image.fromarray(mask_video[frame_idx].cpu().numpy()) 
            # mask_video[frame_idx] = None
            if (frame_width, frame_height) != mask.size:
                mask = mask.resize((frame_width, frame_height), resample=Image.Resampling.LANCZOS)
            mask = np.array(mask)
            alpha_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            alpha_mask[mask > 127] = 1
            frame = frame * alpha_mask
        frame = Image.fromarray(frame)
        face = face_processor.process(frame, resize_to=size)
        face_list.append(face)

    face_processor = None
    gc.collect()
    torch.cuda.empty_cache()

    face_tensor= torch.tensor(np.stack(face_list, dtype= np.float32) / 127.5 - 1).permute(-1, 0, 1, 2 ) # t h w c -> c t h w
    if pad_frames > 0:
        face_tensor= torch.cat([face_tensor[:, -1:].expand(-1, pad_frames, -1, -1), face_tensor ], dim=2)
        
    if args.save_masks:
        from preprocessing.dwpose.pose import save_one_video
        saved_faces_frames = [np.array(face) for face in face_list ]
        save_one_video(f"faces.mp4", saved_faces_frames, fps=target_fps, quality=8, macro_block_size=None)
    return face_tensor


def preprocess_video_with_mask(pre_video_guide, input_video_path, input_mask_path, height, width,  max_frames, start_frame=0, fit_canvas = None, fit_crop = False, target_fps = 16, block_size= 16, expand_scale = 2, process_type = "inpaint", process_type2 = None, to_bbox = False, RGB_Mask = False, negate_mask = False, process_outside_mask = None, inpaint_color = 127, outpainting_dims = None, outpainting_ratio = "", proc_no = 1, outpainting_quantize_margins = 0):

    def mask_to_xyxy_box(mask):
        rows, cols = np.where(mask == 255)
        xmin = min(cols)
        xmax = max(cols) + 1
        ymin = min(rows)
        ymax = max(rows) + 1
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, mask.shape[1])
        ymax = min(ymax, mask.shape[0])
        box = [xmin, ymin, xmax, ymax]
        box = [int(x) for x in box]
        return box
    inpaint_color = parse_guide_inpaint_color(inpaint_color)
    inpaint_color = to_rgb_tensor(inpaint_color, device="cpu", dtype=torch.uint8)
    inpaint_color_np = tuple(int(v) for v in inpaint_color.view(-1).tolist())
    pad_frames = 0
    if start_frame < 0:
        pad_frames= -start_frame
        max_frames += start_frame
        start_frame = 0

    if not input_video_path or max_frames <= 0:
        return None, None
    any_mask = input_mask_path != None
    pose_special = "pose" in process_type
    any_identity_mask = False
    if process_type == "identity":
        any_identity_mask = True
        negate_mask = False
        process_outside_mask = None
    if process_type == "pose_align" and any_mask:
        process_outside_mask = None
    preproc = get_preprocessor(process_type, inpaint_color, pre_video_guide=pre_video_guide)
    preproc2 = None
    if process_type2 != None:
        preproc2 = get_preprocessor(process_type2, inpaint_color, pre_video_guide=pre_video_guide) if process_type != process_type2 else preproc
    if process_outside_mask == process_type :
        preproc_outside = preproc
    elif preproc2 != None and process_outside_mask == process_type2 :
        preproc_outside = preproc2
    else:
        preproc_outside = get_preprocessor(process_outside_mask, inpaint_color)
    video = get_resampled_video(input_video_path, start_frame, max_frames, target_fps)
    if any_mask:
        mask_video = get_resampled_video(input_mask_path, start_frame, max_frames, target_fps)

    if len(video) == 0 or any_mask and len(mask_video) == 0:
        return None, None
    if fit_crop and outpainting_dims != None:
        fit_crop = False
        fit_canvas = 0 if fit_canvas is not None else None

    frame_height, frame_width, _ = video[0].shape

    source_frame_height, source_frame_width = frame_height, frame_width
    if outpainting_dims != None:
        if fit_canvas != None:
            frame_height, frame_width = get_outpainting_full_area_dimensions(frame_height, frame_width, outpainting_dims, outpainting_ratio)
        else:
            frame_height, frame_width = height, width

    if fit_canvas != None:
        height, width = calculate_new_dimensions(height, width, frame_height, frame_width, fit_into_canvas = fit_canvas, block_size = block_size)

    if outpainting_dims != None:
        final_height, final_width = height, width
        height, width, margin_top, margin_left = get_outpainting_frame_location(final_height, final_width, outpainting_dims, 1, outpainting_ratio, source_frame_height, source_frame_width, quantize_margins=outpainting_quantize_margins)

    if any_mask:
        num_frames = min(len(video), len(mask_video))
    else:
        num_frames = len(video)

    if any_identity_mask:
        any_mask = True

    proc_list =[]
    proc_list_outside =[]
    proc_mask = []

    # for frame_idx in range(num_frames):
    def prep_prephase(frame_idx):
        frame = Image.fromarray(video[frame_idx].cpu().numpy()) #.asnumpy()
        if fit_crop:
            frame = rescale_and_crop(frame, width, height)
        else:
            frame = frame.resize((width, height), resample=Image.Resampling.LANCZOS) 
        frame = np.array(frame) 
        if any_mask:
            if any_identity_mask:
                mask = np.full( (height, width, 3), 0, dtype= np.uint8)
            else:
                mask = Image.fromarray(mask_video[frame_idx].cpu().numpy()) #.asnumpy()
                if fit_crop:
                    mask = rescale_and_crop(mask, width, height)
                else:
                    mask = mask.resize((width, height), resample=Image.Resampling.LANCZOS) 
                mask = np.array(mask)

            if len(mask.shape) == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 127.5, 255, cv2.THRESH_BINARY)
            original_mask = mask.copy()
            mask = expand_or_shrink_mask(mask, expand_scale)

            if to_bbox and np.sum(mask == 255) > 0 : #or True 
                x0, y0, x1, y1 = mask_to_xyxy_box(mask)
                mask = mask * 0
                mask[y0:y1, x0:x1] = 255
            if negate_mask:
                mask = 255 - mask
                if pose_special:
                    original_mask = 255 - original_mask

        if pose_special and any_mask:            
            target_frame = np.where(original_mask[..., None], frame, 0) 
        else:
            target_frame = frame 

        if any_mask:
            return (target_frame, frame, mask) 
        else:
            return (target_frame, None, None)
    max_workers = get_default_workers()
    proc_lists = process_images_multithread(prep_prephase, [frame_idx for frame_idx in range(num_frames)], "prephase", wrap_in_list= False, max_workers=max_workers, in_place= True)
    proc_list, proc_list_outside, proc_mask = [None] * len(proc_lists), [None] * len(proc_lists), [None] * len(proc_lists)
    for frame_idx, frame_group in enumerate(proc_lists): 
        proc_list[frame_idx], proc_list_outside[frame_idx], proc_mask[frame_idx] = frame_group
    prep_prephase = None
    video = None
    mask_video = None

    if preproc2 != None:
        proc_list2 = process_images_multithread(preproc2, proc_list, process_type2, max_workers=max_workers)
        #### to be finished ...or not
    proc_list = process_images_multithread(preproc, proc_list, process_type, max_workers=max_workers)
    if any_mask:
        proc_list_outside = process_images_multithread(preproc_outside, proc_list_outside, process_outside_mask, max_workers=max_workers)
    else:
        proc_list_outside = proc_mask = len(proc_list) * [None]

    masked_frames = []
    masks = []
    for frame_no, (processed_img, processed_img_outside, mask) in enumerate(zip(proc_list, proc_list_outside, proc_mask)):
        if isinstance(processed_img, (list, tuple)):
            processed_img = np.full((height, width, 3), processed_img, dtype=np.uint8)
        if isinstance(processed_img_outside, (list, tuple)):
            processed_img_outside = np.full((height, width, 3), processed_img_outside, dtype=np.uint8)
        if any_mask :
            if process_type == "pose_align":
                masked_frame = processed_img
                mask = np.full_like(mask, 0)
            else:
                masked_frame = np.where(mask[..., None], processed_img, processed_img_outside)
            if process_outside_mask != None:
                mask = np.full_like(mask, 255)
            mask = torch.from_numpy(mask)
            if RGB_Mask:
                mask =  mask.unsqueeze(-1).repeat(1,1,3)
            if outpainting_dims != None:
                full_frame= torch.full( (final_height, final_width, mask.shape[-1]), 255, dtype= torch.uint8, device= mask.device)
                full_frame[margin_top:margin_top+height, margin_left:margin_left+width] = mask
                mask = full_frame 
            masks.append(mask[:, :, 0:1].clone())
        else:
            masked_frame = processed_img

        if isinstance(masked_frame, (int, float, np.integer)) or (isinstance(masked_frame, (list, tuple)) and len(masked_frame) == 3):
            masked_frame= np.full( (height, width, 3), inpaint_color_np, dtype= np.uint8)

        masked_frame = torch.from_numpy(masked_frame)
        if masked_frame.shape[-1] == 1:
            masked_frame =  masked_frame.repeat(1,1,3).to(torch.uint8)

        if outpainting_dims != None:
            color = inpaint_color.to(masked_frame.device).view(1, 1, 3)
            if masked_frame.shape[-1] == 4:
                color = torch.cat([color, color.new_full((1, 1, 1), 255)], dim=-1)
            full_frame = color.expand(final_height, final_width, masked_frame.shape[-1]).clone()
            full_frame[margin_top:margin_top+height, margin_left:margin_left+width] = masked_frame
            masked_frame = full_frame 

        masked_frames.append(masked_frame)
        proc_list[frame_no] = proc_list_outside[frame_no] = proc_mask[frame_no] = None


    # if args.save_masks:
    #     from preprocessing.dwpose.pose import save_one_video
    #     saved_masked_frames = [mask.cpu().numpy() for mask in masked_frames ]
    #     save_one_video(f"masked_frames{'' if proc_no==1 else str(proc_no)}.mp4", saved_masked_frames, fps=target_fps, quality=8, macro_block_size=None)
    #     if any_mask:
    #         saved_masks = [mask.cpu().numpy() for mask in masks ]
    #         save_one_video("masks.mp4", saved_masks, fps=target_fps, quality=8, macro_block_size=None)
    preproc = None
    preproc_outside = None
    gc.collect()
    torch.cuda.empty_cache()
    if pad_frames > 0:
        masked_frames = masked_frames[0] * pad_frames + masked_frames
        if any_mask: masked_frames = masks[0] * pad_frames + masks
    masked_frames = torch.stack(masked_frames).permute(-1,0,1,2).float().div_(127.5).sub_(1.)
    masks = torch.stack(masks).permute(-1,0,1,2).float().div_(255) if any_mask else None

    return masked_frames, masks

def preprocess_video(height, width, video_in, max_frames, start_frame=0, fit_canvas = None, fit_crop = False, target_fps = 16, block_size = 16, preserve_hdr = False):

    hdr_input = bool(preserve_hdr and _video_input_is_hdr(video_in))
    frames_list = get_resampled_video(video_in, start_frame, max_frames, target_fps, hdr_linear=hdr_input)

    if len(frames_list) == 0:
        return None
    frames_list = list(frames_list)

    if fit_canvas == None or fit_crop:
        new_height = height
        new_width = width
    else:
        frame_height, frame_width, _ = frames_list[0].shape
        if fit_canvas :
            scale1  = min(height / frame_height, width /  frame_width)
            scale2  = min(height / frame_width, width /  frame_height)
            scale = max(scale1, scale2)
        else:
            scale =   ((height * width ) /  (frame_height * frame_width))**(1/2)

        new_height = round(frame_height * scale / block_size) * block_size
        new_width = round(frame_width * scale / block_size) * block_size

    def resize_hdr_frame(frame):
        frame_t = frame.detach().cpu().to(dtype=torch.float32) if torch.is_tensor(frame) else torch.from_numpy(np.asarray(frame, dtype=np.float32))
        if frame_t.ndim == 3 and frame_t.shape[0] in (1, 3, 4) and frame_t.shape[-1] not in (1, 3, 4):
            frame_t = frame_t.permute(1, 2, 0)
        frame_t = frame_t[..., :3].permute(2, 0, 1).unsqueeze(0)
        if fit_crop:
            src_h, src_w = int(frame_t.shape[2]), int(frame_t.shape[3])
            scale = max(new_height / max(1, src_h), new_width / max(1, src_w))
            resize_h = max(1, int(round(src_h * scale)))
            resize_w = max(1, int(round(src_w * scale)))
            frame_t = torch.nn.functional.interpolate(frame_t, size=(resize_h, resize_w), mode="bilinear", align_corners=False)
            top = max(0, (resize_h - new_height) // 2)
            left = max(0, (resize_w - new_width) // 2)
            frame_t = frame_t[:, :, top:top + new_height, left:left + new_width]
        else:
            frame_t = torch.nn.functional.interpolate(frame_t, size=(new_height, new_width), mode="bilinear", align_corners=False)
        return frame_t[0].permute(1, 2, 0).contiguous()

    if hdr_input:
        def resize_frame(frame):
            return resize_hdr_frame(frame)

        frames_list = process_images_multithread(
            resize_frame,
            frames_list,
            "upsample",
            wrap_in_list=False,
            max_workers=get_default_workers(),
            in_place=True,
        )
    else:
        frames_list = resize_lanczos_frames(frames_list, new_height, new_width, crop=fit_crop, max_workers=get_default_workers())

    # from preprocessing.dwpose.pose import save_one_video
    # save_one_video("test.mp4", frames_list, fps=8, quality=8, macro_block_size=None)

    return torch.stack(frames_list) 


def get_reference_video_dimensions(video_in, height, width, max_size, block_size):
    _, source_width, source_height, _ = get_video_info(video_in)
    max_short_edge, max_long_edge = max_size
    scale = min((height * width / (source_height * source_width)) ** 0.5,
                max_short_edge / min(source_height, source_width), max_long_edge / max(source_height, source_width))
    height_cap, width_cap = (max_short_edge, max_long_edge) if source_height <= source_width else (max_long_edge, max_short_edge)
    target_height = min(height_cap // block_size * block_size, max(block_size, round(source_height * scale / block_size) * block_size))
    target_width = min(width_cap // block_size * block_size, max(block_size, round(source_width * scale / block_size) * block_size))
    return target_height, target_width

 
def parse_keep_frames_video_guide(keep_frames, video_length):
        
    def absolute(n):
        if n==0:
            return 0
        elif n < 0:
            return max(0, video_length + n)
        else:
            return min(n-1, video_length-1)
    keep_frames = keep_frames.strip()
    if len(keep_frames) == 0:
        return [True] *video_length, "" 
    frames =[False] *video_length
    error = ""
    sections = keep_frames.split(" ")
    for section in sections:
        section = section.strip()
        if ":" in section:
            parts = section.split(":")
            if not is_integer(parts[0]):
                error =f"Invalid integer {parts[0]}"
                break
            start_range = absolute(int(parts[0]))
            if not is_integer(parts[1]):
                error =f"Invalid integer {parts[1]}"
                break
            end_range = absolute(int(parts[1]))
            for i in range(start_range, end_range + 1):
                frames[i] = True
        else:
            if not is_integer(section) or int(section) == 0:
                error =f"Invalid integer {section}"
                break
            index = absolute(int(section))
            frames[index] = True

    if len(error ) > 0:
        return [], error
    for i in range(len(frames)-1, 0, -1):
        if frames[i]:
            break
    frames= frames[0: i+1]
    return  frames, error


def get_loaded_model_context():
    if wan_model is None or offloadobj is None or transformer_type is None or reload_needed:
        return None
    from postprocessing.model_context import LoadedModelContext
    return LoadedModelContext(model=wan_model, offloadobj=offloadobj, model_type=transformer_type, base_model_type=get_base_model_type(transformer_type), model_family=get_model_family(transformer_type), model_def=get_model_def(transformer_type), profile=loaded_profile, config_id=loaded_config)


def perform_temporal_upsampling(sample, previous_last_frame, temporal_upsampling, fps, abort_callback=None, progress_callback=None):
    wait_for_model_unload()
    return temporal_upsampler_api.temporal_upsample(temporal_upsampling, sample, previous_last_frame, fps, main_offloadobj=offloadobj, loaded_model_context=get_loaded_model_context(), processing_device=processing_device, to_uint8_callback=convert_video_tensor_to_uint8_chunked, process_files=process_files_def, init_pipe=init_pipe, profile=loaded_profile if loaded_profile >= 0 else get_default_profile("video"), abort_callback=abort_callback, progress_callback=progress_callback)


def perform_spatial_upsampling(sample, spatial_upsampling, seed=0, flashvsr_continue_cache=None, return_flashvsr_continue_cache=False, vae_tile_size=None, still_image=False, abort_callback=None, progress_callback=None, fps=24.0, frame_offset=0, prompt="", negative_prompt="", audio_waveform=None, audio_sample_rate=0, source_audio_path=None, reference_images=None, image_refs_relative_size=100.0, spatial_upsampler_prompt="", spatial_upsampler_reference_images=None, spatial_upsampler_param=None, spatial_upsampler_param2=None, spatial_upsampler_parameters=None):
    wait_for_model_unload()
    if upsampler_api.is_vae_upsampling(spatial_upsampling):
        sample = upsampler_api.post_model_process_vae_upsampling(sample, spatial_upsampling)
        return (sample, None) if return_flashvsr_continue_cache else sample
    edit_upsampler = upsampler_api.find_postprocessing_upsampler(spatial_upsampling)
    if edit_upsampler is not None:
        parameter_values = dict(spatial_upsampler_parameters or {})
        parameter_values.setdefault("spatial_upsampler_prompt", spatial_upsampler_prompt)
        parameter_values.setdefault("spatial_upsampler_reference_images", spatial_upsampler_reference_images)
        parameter_values.setdefault("spatial_upsampler_param", spatial_upsampler_param)
        parameter_values.setdefault("spatial_upsampler_param2", spatial_upsampler_param2)
        profile_type = upsampler_api.profile_type_for_handler(edit_upsampler)
        if profile_type == upsampler_api.UPSAMPLER_PROFILE_IMAGE:
            profile = get_default_profile("image")
        elif profile_type == upsampler_api.UPSAMPLER_PROFILE_AUDIO:
            profile = get_default_profile("audio")
        else:
            profile = loaded_profile if loaded_profile >= 0 else get_default_profile("video")
        sample, upsampler_cache = upsampler_api.upscale_postprocessing(edit_upsampler, sample, spatial_upsampling, main_offloadobj=offloadobj, loaded_model_context=get_loaded_model_context(), seed=seed, continue_cache=flashvsr_continue_cache, return_continue_cache=return_flashvsr_continue_cache, vae_tile_size=vae_tile_size, process_files=process_files_def, vae_config=vae_config, init_pipe=init_pipe, profile=profile, still_image=still_image, fps=fps, frame_offset=frame_offset, prompt=prompt, negative_prompt=negative_prompt, audio_waveform=audio_waveform, audio_sample_rate=audio_sample_rate, source_audio_path=source_audio_path, reference_images=reference_images, image_refs_relative_size=image_refs_relative_size, abort_callback=abort_callback, progress_callback=progress_callback, **parameter_values)
        return (sample, upsampler_cache) if return_flashvsr_continue_cache else sample
    raise ValueError(f"No spatial upsampler registered for '{spatial_upsampling}'")


def perform_image_spatial_upsampling(sample, spatial_upsampling, seed=0, vae_tile_size=None, abort_callback=None, progress_callback=None, fps=24.0, prompt="", negative_prompt="", spatial_upsampler_prompt="", spatial_upsampler_reference_images=None, spatial_upsampler_param=None, spatial_upsampler_param2=None, spatial_upsampler_parameters=None):
    edit_upsampler = upsampler_api.find_postprocessing_upsampler(spatial_upsampling)
    if edit_upsampler is None or sample.shape[1] <= 1 or getattr(edit_upsampler, "batch_image_inputs", False):
        return perform_spatial_upsampling(sample, spatial_upsampling, seed=seed, vae_tile_size=vae_tile_size, still_image=True, fps=fps, prompt=prompt, negative_prompt=negative_prompt, spatial_upsampler_prompt=spatial_upsampler_prompt, spatial_upsampler_reference_images=spatial_upsampler_reference_images, spatial_upsampler_param=spatial_upsampler_param, spatial_upsampler_param2=spatial_upsampler_param2, spatial_upsampler_parameters=spatial_upsampler_parameters, abort_callback=abort_callback, progress_callback=progress_callback)
    frames = []
    for frame_no in range(sample.shape[1]):
        if abort_callback is not None and abort_callback():
            return None
        frames.append(perform_spatial_upsampling(sample[:, frame_no:frame_no + 1], spatial_upsampling, seed=seed, vae_tile_size=vae_tile_size, still_image=True, fps=fps, prompt=prompt, negative_prompt=negative_prompt, spatial_upsampler_prompt=spatial_upsampler_prompt, spatial_upsampler_reference_images=spatial_upsampler_reference_images, spatial_upsampler_param=spatial_upsampler_param, spatial_upsampler_param2=spatial_upsampler_param2, spatial_upsampler_parameters=spatial_upsampler_parameters, abort_callback=abort_callback, progress_callback=progress_callback))
    return torch.cat(frames, dim=1)


def any_audio_track(model_type):
    model_def = get_model_def(model_type)
    if not model_def:
        return False
    return ( model_def.get("returns_audio", False) or model_def.get("any_audio_prompt", False) )

def get_available_filename(target_path, video_source, suffix = "", force_extension = None):
    name, extension =  os.path.splitext(os.path.basename(strip_virtual_media_suffix(video_source)))
    if force_extension != None:
        extension = force_extension
    name+= suffix
    full_path= os.path.join(target_path, f"{name}{extension}")
    if not os.path.exists(full_path):
        return full_path
    counter = 2
    while True:
        full_path= os.path.join(target_path, f"{name}({counter}){extension}")
        if not os.path.exists(full_path):
            return full_path
        counter += 1

def set_seed(seed):
    import random
    seed = random.randint(0, 999999999) if seed == None or seed < 0 else seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return seed

def edit_media(
                send_cmd,
                state,
                mode,
                video_source,
                seed,   
                temporal_upsampling,
                spatial_upsampling,
                film_grain_intensity,
                film_grain_saturation,
                postprocess_audio,
                postprocess_audio_prompt,
                postprocess_audio_neg_prompt,
                repeat_generation,
                audio_source,
                replace_voice_method,
                replace_voice_sample,
                replace_voice_sample2,
                prompt="",
                image_refs=None,
                image_refs_relative_size=100.0,
                spatial_upsampler_prompt="",
                spatial_upsampler_reference_images=None,
                spatial_upsampler_param=None, spatial_upsampler_param2=None,
                spatial_upsampler_parameters=None,
                client_id="",
                plugin_data=None,
                **kwargs
                ):

    operation_start_time = time.time()
    gen = get_gen_info(state)
    api_return_video_uint8, api_return_audio = get_api_output_options(plugin_data)
    api_options = plugin_data.get("api", {}) if isinstance(plugin_data, dict) and isinstance(plugin_data.get("api", {}), dict) else {}
    api_suppress_source_audio = bool(api_options.get("suppress_source_audio"))
    upsampler_frame_offset = int(api_options.get("upsampler_frame_offset", 0))
    flashvsr_continue_cache = api_options.get("flashvsr_continue_cache")
    return_flashvsr_continue_cache = bool(api_options.get("return_flashvsr_continue_cache"))

    if gen.get("abort", False): return 
    abort = False
		
		
    postprocess_audio = audio_processor_api.normalize_method(postprocess_audio or "")
    replace_voice_method = audio_processor_api.normalize_method(replace_voice_method or "")
    source_is_image = has_image_file_extension(video_source)
    spatial_upsampler = upsampler_api.find_postprocessing_upsampler(spatial_upsampling)
    spatial_method = spatial_upsampler.split_value(spatial_upsampling)[0] if spatial_upsampler is not None else ""
    source_audio_conditioning = spatial_upsampler is not None and spatial_upsampler.query_upsampler_def().get("source_audio_conditioning", False)
    source_image_conditioning = upsampler_api.method_uses_setting(spatial_method, "reference_images")
    if source_is_image:
        postprocess_audio = ""
        replace_voice_method = ""
    postprocess_audio_meta = audio_processor_api.method_metadata(postprocess_audio)
    soundtrack_method = postprocess_audio if audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"] else ""
    voice_method = postprocess_audio if audio_processor_api.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT in postprocess_audio_meta["types"] else replace_voice_method
    soundtrack_meta = audio_processor_api.method_metadata(soundtrack_method)
    configs, _ , _ = get_settings_from_file(state, video_source, False, False, False)
    configs = configs.copy() if configs is not None else { "type" : get_model_record("Post Processing") }
    reference_images = None
    if not source_is_image and source_image_conditioning:
        reference_images = clean_image_list(spatial_upsampler_reference_images or image_refs or [])
        if not reference_images:
            extract_and_apply_source_images(video_source, configs)
            reference_images = clean_image_list(configs.get("image_refs") or [])
    if len(str(client_id or "").strip()) > 0:
        configs["client_id"] = client_id

    has_already_audio = False
    audio_tracks = []
    audio_metadata = None
    download_requested_postprocessing_assets(
        send_cmd,
        postprocess_audio=postprocess_audio,
        temporal_upsampling=temporal_upsampling if mode == "edit_postprocessing" else "",
        spatial_upsampling=spatial_upsampling if mode == "edit_postprocessing" else "",
        replace_voice_method=voice_method,
        gen=gen,
    )

    temp_audio_tracks = []
    conditioning_audio_path = None
    if not source_is_image and not soundtrack_method and not api_suppress_source_audio:
        audio_tracks, audio_metadata = extract_audio_tracks(video_source, temp_format="wav" if voice_method else None, codec_key=server_config.get("audio_output_codec", "aac_128"))
        temp_audio_tracks = audio_tracks.copy()
        has_already_audio = len(temp_audio_tracks) > 0 
    if not source_is_image and source_audio_conditioning and not api_suppress_source_audio:
        if audio_source:
            conditioning_audio_path = audio_source
        elif voice_method and audio_tracks:
            conditioning_audio_path = audio_tracks[0]
        else:
            conditioning_audio_tracks, _ = extract_audio_tracks(video_source, temp_format="wav")
            temp_audio_tracks += conditioning_audio_tracks
            conditioning_audio_path = conditioning_audio_tracks[0] if conditioning_audio_tracks else None

    source_prompt = str(spatial_upsampler_prompt or "").strip() or (configs.get("prompt") if upsampler_api.method_uses_setting(spatial_method, "prompt") else str(prompt or "").strip() or configs.get("prompt"))
    spatial_upsampling_prompt = upsampler_api.resolve_late_postprocessing_prompt(spatial_upsampling, source_prompt) if mode == "edit_postprocessing" else ""
    if mode == "edit_postprocessing" and spatial_upsampler is not None and hasattr(spatial_upsampler, "validate_runtime_inputs"):
        spatial_upsampler.validate_runtime_inputs(prompt=spatial_upsampling_prompt, reference_images=reference_images, source_audio_path=conditioning_audio_path)
    
    if soundtrack_method and soundtrack_meta["needs_audio_source"] and audio_source is not None:
        audio_tracks = [audio_source]

    with lock:
        file_list = gen["file_list"]
        file_settings_list = gen["file_settings_list"]



    seed = set_seed(seed)

    if source_is_image:
        image = convert_image(_open_image_input(video_source))
        width, height = image.size
        fps, frames_count = 1, 1
    else:
        from shared.utils.utils import get_video_info_details
        media_info = get_video_info_details(video_source)
        fps = float(media_info["fps_float"] or media_info["fps"])
        width, height, frames_count = media_info["display_width"], media_info["display_height"], media_info["frame_count"]
    frames_count = min(frames_count, max_source_video_frames)
    sample = None

    if mode == "edit_postprocessing":
        if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 or film_grain_intensity > 0:
            spatial_status = upsampler_api.method_progress_label(spatial_upsampling)
            send_cmd("progress", [0, get_latest_status(state, spatial_status if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 else "Adding Film Grain")])
            if source_is_image:
                sample = torch.from_numpy(np.array(image).astype(np.uint8)).unsqueeze(0).permute(-1,0,1,2)
            else:
                sample = get_resampled_video(video_source, 0, max_source_video_frames, None)
                sample = sample.permute(-1,0,1,2)
            frames_count = sample.shape[1] 

        output_fps = fps
        def upsampler_progress(phase, current_step=None, total_steps=None):
            phase_text = str(phase)
            gen["progress_phase"] = (phase_text, int(current_step) if current_step is not None else -1)
            status_msg = get_latest_status(state, phase_text)
            if current_step is not None and total_steps is not None and int(total_steps) > 0:
                send_cmd("progress", [(int(current_step), int(total_steps)), status_msg, int(total_steps)])
            else:
                send_cmd("progress", [0, status_msg])
        if len(temporal_upsampling) > 0:
            sample, previous_last_frame, output_fps = perform_temporal_upsampling(sample, None, temporal_upsampling, fps, abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
            if gen.get("abort", False) or sample is None:
                return
            configs["temporal_upsampling"] = temporal_upsampling
            frames_count = sample.shape[1] 


        if len(spatial_upsampling) > 0:
            if source_is_image:
                sample = perform_image_spatial_upsampling(sample, spatial_upsampling, seed=seed, fps=output_fps, prompt=spatial_upsampling_prompt, negative_prompt=str(configs.get("negative_prompt", "")), spatial_upsampler_prompt=spatial_upsampler_prompt, spatial_upsampler_reference_images=spatial_upsampler_reference_images, spatial_upsampler_param=spatial_upsampler_param, spatial_upsampler_param2=spatial_upsampler_param2, spatial_upsampler_parameters=spatial_upsampler_parameters, abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
                flashvsr_continue_cache = None
            else:
                sample = perform_spatial_upsampling(sample, spatial_upsampling, seed=seed, flashvsr_continue_cache=flashvsr_continue_cache, return_flashvsr_continue_cache=return_flashvsr_continue_cache, fps=output_fps, frame_offset=upsampler_frame_offset, prompt=spatial_upsampling_prompt, negative_prompt=str(configs.get("negative_prompt", "")), source_audio_path=conditioning_audio_path, reference_images=reference_images, image_refs_relative_size=image_refs_relative_size, spatial_upsampler_prompt=spatial_upsampler_prompt, spatial_upsampler_reference_images=spatial_upsampler_reference_images, spatial_upsampler_param=spatial_upsampler_param, spatial_upsampler_param2=spatial_upsampler_param2, spatial_upsampler_parameters=spatial_upsampler_parameters, abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
            if return_flashvsr_continue_cache and not source_is_image:
                sample, flashvsr_continue_cache = sample
            if gen.get("abort", False) or sample is None:
                return
            configs["spatial_upsampling"] = spatial_upsampling
            configs.update(spatial_upsampler_prompt=spatial_upsampler_prompt, spatial_upsampler_reference_images=spatial_upsampler_reference_images or [], spatial_upsampler_param=spatial_upsampler_param, spatial_upsampler_param2=spatial_upsampler_param2, spatial_upsampler_parameters=dict(spatial_upsampler_parameters or {}))

        if film_grain_intensity > 0:
            from postprocessing.film_grain import add_film_grain
            sample = add_film_grain(sample, film_grain_intensity, film_grain_saturation) 
            configs["film_grain_intensity"] = film_grain_intensity
            configs["film_grain_saturation"] = film_grain_saturation
    else:
        output_fps = fps

    soundtrack_error = audio_processor_api.validate_method(soundtrack_method, audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK, video_source=None, frames_count=frames_count, fps=output_fps, audio_source=audio_source, media_source_exists=media_source_exists, has_audio_file_extension=has_audio_file_extension) if soundtrack_method else ""
    any_soundtrack_processor = bool(soundtrack_method and not soundtrack_error)
    if soundtrack_error and mode == "edit_remux":
        raise gr.Error(soundtrack_error)
    any_voice_replacement = bool(voice_method and replace_voice_sample is not None and len(audio_tracks) > 0)
    video_container = server_config.get("video_container", "mp4")
    video_extension = f".{video_container}"

    tmp_path = None
    saved_video_duration = None
    any_change = False
    api_video_tensor = None
    if sample != None:
        if source_is_image:
            image_extension = get_image_format(server_config.get("image_output_codec", None), rgba=sample.shape[0] == 4)["ext"]
            image_path = get_available_filename(image_save_path, video_source, "_post", force_extension=image_extension)
            image_paths = []
            for no, img in enumerate(sample.transpose(1,0)):
                img_path = get_available_filename(image_save_path, image_path, "" if no == 0 else f"_{no}")
                image_paths.append(save_image(img, save_file=img_path, quality=server_config.get("image_output_codec", None)))
            video_path = image_paths if len(image_paths) > 1 else image_paths[0]
            print(f"Postprocessed image saved to Path: {video_path}")
            configs["generation_time"] = round(time.time() - operation_start_time)
            record_file_metadata(video_path, configs, True, False, gen)
            if api_return_video_uint8 or api_return_audio or return_flashvsr_continue_cache:
                store_api_output_artifact(gen, client_id, video_path, "image", sample if api_return_video_uint8 else None, None, None, None)
            send_cmd("output")
            clear_status(state)
            return
        video_path = get_available_filename(save_path, video_source, "_tmp", force_extension=video_extension) if any_soundtrack_processor or has_already_audio else get_available_filename(save_path, video_source, "_post", force_extension=video_extension)
        video_path = save_video( tensor=sample[None], save_file=video_path, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1), codec_type= server_config.get("video_output_codec", None), container=server_config.get("video_container", "mp4"))
        saved_video_duration = sample.shape[1] / output_fps
        api_video_tensor = sample if api_return_video_uint8 else None
        if api_video_tensor is None:
            sample = None
            gc.collect()

        if any_soundtrack_processor or has_already_audio: tmp_path = video_path
        any_change = True
    else:
        video_path = video_source

    repeat_no = 0
    extra_generation = 0
    initial_total_windows = 0
    any_change_initial = any_change
    while not gen.get("abort", False): 
        any_change = any_change_initial
        with gen_lock:
            extra_generation += gen.get("extra_orders",0)
            gen["extra_orders"] = 0
        total_generation = repeat_generation + extra_generation
        gen["total_generation"] = total_generation         
        if repeat_no >= total_generation: break
        repeat_no +=1
        repeat_start_time = operation_start_time if repeat_no == 1 else time.time()
        gen["repeat_no"] = repeat_no
        suffix =  "" if "_post" in video_source else "_post"

        if soundtrack_method and soundtrack_meta["needs_audio_source"] and audio_source is not None:
            audio_prompt_type = configs.get("audio_prompt_type", "")
            if not "T" in audio_prompt_type:audio_prompt_type += "T"
            configs["audio_prompt_type"] = audio_prompt_type
            configs["postprocess_audio"] = postprocess_audio
            any_change = True
        elif any_voice_replacement:
            configs["postprocess_audio"] = postprocess_audio
            configs["replace_voice_method"] = voice_method
            any_change = True

        if any_soundtrack_processor:
            def audio_progress(phase, current_step=None, total_steps=None):
                phase_text = f"Audio - {phase}"
                gen["progress_phase"] = (phase_text, int(current_step) if current_step is not None else -1)
                status_msg = get_latest_status(state, phase_text)
                if current_step is not None and total_steps is not None and int(total_steps) > 0:
                    send_cmd("progress", [(int(current_step), int(total_steps)), status_msg, int(total_steps)])
                else:
                    send_cmd("progress", [0, status_msg])
            new_video_path = get_available_filename(save_path, video_source, suffix, force_extension=video_extension)
            generated_audio_path = get_available_filename(save_path, f"tmp_audio_seed{seed}_{repeat_no}.wav")
            generated_audio_path = audio_processor_api.generate_soundtrack(
                soundtrack_method,
                video_path=video_path,
                audio_source=audio_source,
                prompt=postprocess_audio_prompt,
                negative_prompt=postprocess_audio_neg_prompt,
                seed=seed,
                duration=frames_count / output_fps,
                output_path=generated_audio_path,
                send_cmd=send_cmd,
                status_callback=lambda status: send_cmd("progress", [0, get_latest_status(state, status)]),
                verbose_level=verbose_level,
                audio_codec_key=server_config.get("audio_output_codec", "aac_128"),
                process_files=process_files_def,
                init_pipe=init_pipe,
                profile=server_config.get("audio_profile", 4),
                abort_callback=lambda: media_abort_requested(gen),
                progress_callback=audio_progress,
            )
            if gen.get("abort", False) or generated_audio_path is None:
                return
            combine_video_with_audio_tracks(video_path, [generated_audio_path], new_video_path, audio_codec_key=server_config.get("audio_output_codec", "aac_128"), video_duration=saved_video_duration)
            if generated_audio_path is not None and generated_audio_path != audio_source and os.path.isfile(generated_audio_path):
                os.remove(generated_audio_path)
            configs["postprocess_audio"] = postprocess_audio
            configs["postprocess_audio_prompt"] = postprocess_audio_prompt
            configs["postprocess_audio_neg_prompt"] = postprocess_audio_neg_prompt
            configs["postprocess_audio_seed"] = seed
            any_change = True
        elif len(audio_tracks) > 0:
            new_video_path = get_available_filename(save_path, video_source, suffix, force_extension=video_extension)
            if any_voice_replacement:
                replaced_audio_tracks, replace_voice_temp_tracks = audio_processor_api.replace_voice_tracks(
                    voice_method,
                    audio_tracks,
                    voice_sample=replace_voice_sample,
                    output_dir=save_path,
                    prefix=f"tmp_seed{seed}_{repeat_no}",
                    process_files=process_files_def,
                    profile_no=server_config.get("audio_profile", 4),
                    verbose_level=verbose_level,
                    init_pipe=init_pipe,
                    voice_sample2=replace_voice_sample2,
                    status_callback=lambda status: send_cmd("progress", [0, get_latest_status(state, status)]),
                )
                replaced_sample_rate = resolve_mux_audio_sampling_rate(22050, audio_paths=replaced_audio_tracks)
                combine_and_concatenate_video_with_audio_tracks(
                    new_video_path,
                    video_path,
                    [],
                    replaced_audio_tracks,
                    0,
                    replaced_sample_rate,
                    audio_codec_key=server_config.get("audio_output_codec", "aac_128"),
                    verbose=verbose_level >= 2,
                )
                cleanup_temp_audio_files(replace_voice_temp_tracks)
            else:
                combine_video_with_audio_tracks(video_path, audio_tracks, new_video_path, audio_metadata=audio_metadata, audio_codec_key=server_config.get("audio_output_codec", "aac_128"), video_duration=saved_video_duration)
        else:
            new_video_path = video_path

        if any_change:
            configs["generation_time"] = round(time.time() - repeat_start_time)
            if mode == "edit_remux":
                print(f"Remuxed Video saved to Path: "+ new_video_path)
            else:
                print(f"Postprocessed video saved to Path: "+ new_video_path)
            with lock:
                file_list.append(new_video_path)
                file_settings_list.append(configs)

            if configs != None:
                from shared.utils.video_metadata import extract_source_images, save_video_metadata
                embedded_images = None
                temp_images_path = None
                if not bool(api_options.get("suppress_metadata_images")):
                    temp_images_path = get_available_filename(save_path, video_source, force_extension= ".temp")
                    embedded_images = extract_source_images(video_source, temp_images_path)
                save_video_metadata(new_video_path, configs, embedded_images, allow_inplace_update=True, verbose_level=verbose_level)
                if temp_images_path is not None and os.path.isdir(temp_images_path):
                    shutil.rmtree(temp_images_path, ignore_errors= True)
            notifications.record_generation(server_config, gen, new_video_path, configs)
            if api_return_video_uint8 or api_return_audio or return_flashvsr_continue_cache:
                store_api_output_artifact(
                    gen,
                    client_id,
                    new_video_path,
                    "video",
                    api_video_tensor,
                    None,
                    None,
                    output_fps,
                    flashvsr_continue_cache=flashvsr_continue_cache if return_flashvsr_continue_cache else None,
                )
            gen["last_was_audio"] = False
            send_cmd("output")
            seed = set_seed(-1)
    if tmp_path is not None and os.path.isfile(tmp_path):
        os.remove(tmp_path)
    cleanup_temp_audio_files(temp_audio_tracks)
    clear_status(state)


def edit_audio(send_cmd, state, audio_source, postprocess_audio, replace_voice_sample, replace_voice_sample2, client_id="", plugin_data=None):
    operation_start_time = time.time()
    gen = get_gen_info(state)
    api_return_video_uint8, api_return_audio = get_api_output_options(plugin_data)
    if gen.get("abort", False):
        return
    postprocess_audio = audio_processor_api.normalize_method(postprocess_audio or "")
    validation_error = ""
    if not media_source_exists(audio_source):
        validation_error = "Selected audio file is missing"
    elif not has_audio_file_extension(audio_source):
        validation_error = "Post processing is only available with Audio files"
    elif postprocess_audio:
        validation_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_AUDIO_EDIT, voice_sample=replace_voice_sample, voice_sample2=replace_voice_sample2)
    else:
        validation_error = "You must choose at least one Audio Post Processing Method"
    if validation_error:
        raise gr.Error(validation_error)

    configs, _, _ = get_settings_from_file(state, audio_source, False, False, False)
    configs = configs.copy() if configs is not None else {"type": get_model_record("Audio Post Processing")}
    if len(str(client_id or "").strip()) > 0:
        configs["client_id"] = client_id
    os.makedirs(audio_save_path, exist_ok=True)

    download_requested_postprocessing_assets(send_cmd, postprocess_audio=postprocess_audio, gen=gen)
    new_audio_path = audio_processor_api.process_audio_file(
        postprocess_audio,
        audio_source=audio_source,
        voice_sample=replace_voice_sample,
        voice_sample2=replace_voice_sample2,
        output_path=get_available_filename(audio_save_path, audio_source, "_post", ".wav"),
        process_files=process_files_def,
        profile_no=server_config.get("audio_profile", 4),
        verbose_level=verbose_level,
        init_pipe=init_pipe,
        status_callback=lambda status: send_cmd("progress", [0, get_latest_status(state, status)]),
    )
    configs["postprocess_audio"] = postprocess_audio
    configs["audio_postprocess"] = audio_processor_api.format_method_label(postprocess_audio)
    configs["generation_time"] = round(time.time() - operation_start_time)

    print("Postprocessed audio saved to Path: " + new_audio_path)
    record_file_metadata(new_audio_path, configs, False, True, gen)
    if api_return_video_uint8 or api_return_audio:
        artifact_audio = None
        artifact_sampling_rate = None
        if api_return_audio:
            import soundfile as sf
            artifact_audio, artifact_sampling_rate = sf.read(new_audio_path, dtype="float32", always_2d=False)
        store_api_output_artifact(gen, client_id, new_audio_path, "audio", None, artifact_audio, artifact_sampling_rate, None)
    send_cmd("output")
    clear_status(state)


def get_overridden_attention(model_type):
    model_def = get_model_def(model_type)
    override_attention = model_def.get("attention", None)
    if override_attention is None: return None
    gpu_version = gpu_major * 10 + gpu_minor
    attention_list = match_nvidia_architecture(override_attention, gpu_version) 
    if len(attention_list ) == 0: return None
    override_attention = attention_list[0]
    custom = model_def.get("custom_attention_modes", {}).get(override_attention)
    custom_status = ATTENTION_MODE_AVAILABILITY.get(override_attention, {})
    custom_supported = custom and custom.get("supported", custom_status.get("supported", custom.get("installed", custom_status.get("installed", True))))
    if override_attention is not None and override_attention not in attention_modes_supported and not custom_supported: return None
    return override_attention

def get_transformer_loras(model_type):
    model_def = get_model_def(model_type)
    transformer_loras_filenames = get_model_recursive_prop(model_type, "loras", return_list=True)
    lora_dir = get_lora_dir(model_type)
    transformer_loras_multipliers = get_model_recursive_prop(model_type, "loras_multipliers", return_list=True) + [1.] * len(transformer_loras_filenames)
    transformer_loras_multipliers = transformer_loras_multipliers[:len(transformer_loras_filenames)]
    return transformer_loras_filenames, transformer_loras_multipliers

class DynamicClass:
    def __init__(self, **kwargs):
        self._data = {}
        # Preassign default properties from kwargs
        for key, value in kwargs.items():
            self._data[key] = value
    
    def __getattr__(self, name):
        if name in self._data:
            return self._data[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if not hasattr(self, '_data'):
                super().__setattr__('_data', {})
            self._data[name] = value
    
    def assign(self, **kwargs):
        """Assign multiple properties at once"""
        for key, value in kwargs.items():
            self._data[key] = value
        return self  # For method chaining
    
    def update(self, dict):
        """Alias for assign() - more dict-like"""
        return self.assign(**dict)

def process_prompt_enhancer(model_type, model_def, prompt_enhancer, original_prompts,  image_start, original_image_refs, is_image, audio_only, seed, prompt_enhancer_instructions = None, text_encoder_max_tokens = 512, enhancer_kwargs = None, batch_images = False):
    global enhancer_offloadobj
    prompt_enhancer_mode = str(prompt_enhancer or "")
    original_prompts, window_commands = prompt_enhancer_images.strip_window_commands(original_prompts)
    prompt_enhancer_instructions, text_encoder_max_tokens = resolve_prompt_enhancer_settings(
        model_type,
        model_def,
        prompt_enhancer_mode,
        is_image,
        prompt_enhancer_instructions=prompt_enhancer_instructions,
        text_encoder_max_tokens=text_encoder_max_tokens,
        enhancer_kwargs = enhancer_kwargs,
    )
    prompt_enhancer_instructions = prompt_enhancer_chaining.append_combined_input_contract(prompt_enhancer_instructions, prompt_enhancer_mode)
    if "W" in prompt_enhancer_mode:
        from shared.prompt_enhancer.transcription import TRANSCRIPT_INSTRUCTIONS, transcript_inputs

        original_prompts = transcript_inputs(original_prompts, enhancer_kwargs["source_audio_transcript"])
        prompt_enhancer_instructions = f"{prompt_enhancer_instructions}\n\n{TRANSCRIPT_INSTRUCTIONS}"

    from shared.prompt_enhancer.prompt_enhance_utils import generate_cinematic_prompt
    image_contexts = (enhancer_kwargs or {}).get("image_contexts") if prompt_enhancer_images.enabled(server_config, prompt_enhancer_mode) else None
    if image_contexts is not None:
        image_contexts = prompt_enhancer_images.contexts_for_prompts(image_contexts, len(original_prompts))
    duration_contexts = (enhancer_kwargs or {}).get("image_contexts")
    prompt_durations = [context.duration_seconds for context in prompt_enhancer_images.contexts_for_prompts(duration_contexts, len(original_prompts))] if duration_contexts is not None else None
    if image_start is None:
        image_start = (enhancer_kwargs or {}).get("control_image")
    prompt_images, image_contexts = prompt_enhancer_images.select_images(prompt_enhancer_mode, image_start, original_image_refs, batch_images, image_contexts)
    prompt_images = [_open_image_input(img) if isinstance(img, str) else img for img in prompt_images]
    video_duration = (enhancer_kwargs or {}).get("video_duration_seconds")
    if prompt_durations is None and model_def.get("prompt_enhancer_video_duration", False) and not is_image and not audio_only and video_duration is not None:
        if prompt_enhancer_instructions is None:
            from shared.prompt_enhancer.prompt_enhance_utils import T2V_CINEMATIC_PROMPT, IT2V_CINEMATIC_PROMPT
            prompt_enhancer_instructions = IT2V_CINEMATIC_PROMPT if prompt_images else T2V_CINEMATIC_PROMPT
        duration_text = f"{video_duration:.2f}".rstrip("0").rstrip(".")
        prompt_enhancer_instructions += f"\n\nThe video duration is {duration_text} seconds. Keep the described action within this duration."
    has_text_input = any(letter in prompt_enhancer_mode for letter in "TLB")
    if len(original_prompts) == 0 and not has_text_input:
        return None
    else:
        remote_engine = resolve_role_engine(server_config, "prompt_enhancer")
        if is_remote_engine(remote_engine):
            from shared.remote_llm.prompt_enhancer import enhance_prompt as enhance_prompt_remote

            prompts = enhance_prompt_remote(
                remote_engine,
                server_config,
                original_prompts if has_text_input else ["an image"] * len(original_prompts),
                prompt_images,
                instructions=prompt_enhancer_instructions or "",
                max_output_tokens=text_encoder_max_tokens,
                prompt_durations=prompt_durations,
                enhancement_progress=(enhancer_kwargs or {}).get("generation_callbacks", {}).get("enhancement_progress"),
            )
            return prompt_enhancer_images.restore_window_commands(prompts, window_commands)
        import secrets
        enhancer_temperature = server_config.get("prompt_enhancer_temperature", 0.6)
        enhancer_top_p = server_config.get("prompt_enhancer_top_p", 0.9)
        randomize_seed = server_config.get("prompt_enhancer_randomize_seed", True)
        prompt_enhancer_images.warn_florence_images(server_config, prompt_enhancer_mode, image_start, original_image_refs, batch_images, len(original_prompts), (enhancer_kwargs or {}).get("caption_input_info"))
        if randomize_seed:
            enhancer_seed = secrets.randbits(32)
        else:
            enhancer_seed = seed if seed is not None and seed >= 0 else 0
        prompts = generate_cinematic_prompt(
            prompt_enhancer_image_caption_model,
            prompt_enhancer_image_caption_processor,
            prompt_enhancer_llm_model,
            prompt_enhancer_llm_tokenizer,
            original_prompts if has_text_input else ["an image"] * len(original_prompts),
            prompt_images if len(prompt_images) > 0 else None,
            video_prompt = not is_image,
            text_prompt = audio_only,
            max_new_tokens=text_encoder_max_tokens,
            prompt_enhancer_instructions = prompt_enhancer_instructions,
            do_sample = True,
            temperature = enhancer_temperature,
            top_p = enhancer_top_p,
            seed = enhancer_seed,
            image_contexts=image_contexts,
            offload_manager=enhancer_offloadobj,
            prompt_durations=prompt_durations,
            thinking_enabled = "K" in prompt_enhancer_mode,
            generation_callbacks=(enhancer_kwargs or {}).get("generation_callbacks"),
        )
        return prompt_enhancer_images.restore_window_commands(prompts, window_commands)


def resolve_prompt_enhancer_settings(model_type, model_def, prompt_enhancer_mode, is_image, prompt_enhancer_instructions = None, text_encoder_max_tokens = 512, enhancer_kwargs = None):
    prompt_enhancer_mode = str(prompt_enhancer_mode or "")
    if model_def is None or len(model_type) == 0:
        return prompt_enhancer_instructions, int(text_encoder_max_tokens)
    prompt_profile_id = "0"
    prompt_profile_match = re.search(r"\d", prompt_enhancer_mode)
    if prompt_profile_match is not None:
        prompt_profile_id = prompt_profile_match.group(0)
    prompt_profile_suffix = "" if prompt_profile_id == "0" else prompt_profile_id

    model_handler = get_model_handler(model_type)
    if hasattr(model_handler, "get_custom_prompt_enhancer_instructions"):
        ret_prompt_enhancer_instructions, ret_text_encoder_max_tokens =  model_handler.get_custom_prompt_enhancer_instructions(model_type, prompt_enhancer_mode, is_image, enhancer_kwargs)
        if ret_prompt_enhancer_instructions is not None: prompt_enhancer_instructions = ret_prompt_enhancer_instructions 
        if ret_text_encoder_max_tokens is not None: text_encoder_max_tokens = ret_text_encoder_max_tokens

    visual_prompt_prefix = "image" if is_image else "video"
    prompt_instructions_key = f"{visual_prompt_prefix}_prompt_enhancer_instructions{prompt_profile_suffix}"
    prompt_max_tokens_key = f"{visual_prompt_prefix}_prompt_enhancer_max_tokens{prompt_profile_suffix}"
    prompt_enhancer_instructions = model_def.get(prompt_instructions_key, model_def.get(f"{visual_prompt_prefix}_prompt_enhancer_instructions", prompt_enhancer_instructions))
    text_encoder_max_tokens = model_def.get(prompt_max_tokens_key, model_def.get(f"{visual_prompt_prefix}_prompt_enhancer_max_tokens", text_encoder_max_tokens))

    if "I" not in prompt_enhancer_mode:
        prompt_instructions_key = f"text_prompt_enhancer_instructions{prompt_profile_suffix}"
        prompt_max_tokens_key = f"text_prompt_enhancer_max_tokens{prompt_profile_suffix}"
        prompt_enhancer_instructions = model_def.get(prompt_instructions_key, model_def.get("text_prompt_enhancer_instructions", prompt_enhancer_instructions))
        text_encoder_max_tokens = model_def.get(prompt_max_tokens_key, model_def.get("text_prompt_enhancer_max_tokens", text_encoder_max_tokens))
    return prompt_enhancer_instructions, int(text_encoder_max_tokens)

def exec_prompt_enhancer_engine(state, model_type, model_def, prompt_enhancer_modes, original_prompts, image_start, original_image_refs, is_image, audio_only, seed, progress, override_profile, send_cmd = None, tools = None, enhancer_kwargs = None, alt_prompts = None, alt_prompt_inherits_prompt_paragraphs = False):
    global enhancer_offloadobj
    assistant_mode = "A" in prompt_enhancer_modes
    if assistant_mode:
        return _deepy.run_assistant_prompt_turn(state, model_def, prompt_enhancer_modes, original_prompts, seed, override_profile=override_profile, send_cmd=send_cmd, tools=tools)

    if "W" in prompt_enhancer_modes:
        from shared.prompt_enhancer.transcription import transcribe_source_audio

        wait_for_model_unload()
        acquire_GPU_ressources(state, "prompt_enhancer_transcription", "Source Audio Transcription")
        try:
            transcription_gen = get_gen_info(state) if send_cmd is not None else progress.gen
            enhancer_kwargs["source_audio_transcript"] = transcribe_source_audio(enhancer_kwargs["audio_guide"], gen=transcription_gen, report_status=lambda text: progress((0, 1), desc=text, total=1), duration_seconds=enhancer_kwargs["duration_seconds"] if model_def.get("prompt_enhancer_transcription_use_duration", False) else None)
            if enhancer_kwargs["source_audio_transcript"] is None:
                prompt_enhancer_modes = prompt_enhancer_modes.replace("W", "")
        finally:
            release_GPU_ressources(state, "prompt_enhancer_transcription")

    local_runtime = not is_remote_engine(resolve_role_engine(server_config, "prompt_enhancer"))
    if local_runtime:
        wait_for_model_unload()
        acquire_GPU_ressources(state, "prompt_enhancer", "Prompt Enhancer")
        try:
            download_gen = get_gen_info(state) if send_cmd is not None else progress.gen if isinstance(progress, WangpProgress) else None
            ensure_prompt_enhancer_loaded(override_profile=override_profile, progress=progress, send_cmd=send_cmd, gen=download_gen)
        except Exception:
            release_GPU_ressources(state, "prompt_enhancer")
            raise

    seed = set_seed(seed)
    num_prompts = len(original_prompts) 

    if alt_prompts is not None and prompt_enhancer_chaining.uses_routing(prompt_enhancer_modes):
        chain_steps = prompt_enhancer_chaining.parse_mode(prompt_enhancer_modes)
        step_no = 0

        def enhance_step(step, step_inputs):
            nonlocal step_no
            progress((step_no, len(chain_steps)), desc=f"Please Wait While Enhancing {step.output_field.replace('_', ' ').title()}", total=len(chain_steps))
            step_no += 1
            step_images = image_start if image_start is not None and "I" in step.mode else None
            return process_prompt_enhancer(model_type, model_def, step.mode, step_inputs, step_images, original_image_refs, is_image, audio_only, seed, enhancer_kwargs=enhancer_kwargs, batch_images=True)

        try:
            enhanced_result = prompt_enhancer_chaining.run_chain(prompt_enhancer_modes, original_prompts, alt_prompts, alt_prompt_inherits_prompt_paragraphs, enhance_step)
        except Exception as e:
            if local_runtime:
                unload_prompt_enhancer_runtime()
                enhancer_offloadobj.unload_all()
                release_GPU_ressources(state, "prompt_enhancer")
            if isinstance(e, (InterruptedError, DownloadCancelled, offload.LoadingCancelled)):
                raise
            print(traceback.format_exc())
            raise gr.Error(e)
        if local_runtime:
            unload_prompt_enhancer_runtime()
            enhancer_offloadobj.unload_all()
            release_GPU_ressources(state, "prompt_enhancer")
        return enhanced_result

    enhanced_prompts = []
    batch_windows = (enhancer_kwargs or {}).get("image_contexts") is not None and prompt_enhancer_images.enabled(server_config, prompt_enhancer_modes)
    batches = [(original_prompts, image_start)] if batch_windows else [([one_prompt], [one_image] if one_image is not None else None) for one_prompt, one_image in zip(original_prompts, image_start)]
    for i, (batch_prompts, start_images) in enumerate(batches):
        status = 'Enhancing Prompt' + (f' {i+1}/{num_prompts}' if num_prompts > 1 else '')
        progress((i , num_prompts), desc=status, total= num_prompts)

        try:
            enhanced_prompt = process_prompt_enhancer(model_type, model_def, prompt_enhancer_modes, batch_prompts, start_images, original_image_refs, is_image, audio_only, seed, enhancer_kwargs=prompt_enhancer_images.batch_kwargs(enhancer_kwargs, i, batch_windows, num_prompts))
        except Exception as e:
            if local_runtime:
                unload_prompt_enhancer_runtime()
                enhancer_offloadobj.unload_all()
                release_GPU_ressources(state, "prompt_enhancer")
            if isinstance(e, (InterruptedError, DownloadCancelled, offload.LoadingCancelled)):
                raise
            print(traceback.format_exc())
            raise gr.Error(e)
        enhanced_prompts.extend([[value] for value in enhanced_prompt] if batch_windows else [enhanced_prompt])

    if local_runtime:
        unload_prompt_enhancer_runtime()
        enhancer_offloadobj.unload_all()
        release_GPU_ressources(state, "prompt_enhancer")
    return enhanced_prompts

def keep_generated_prompt_newlines(multi_prompts_gen_type):
    multi_prompts_gen_type = str(multi_prompts_gen_type or "")
    return "P" in multi_prompts_gen_type or multi_prompts_gen_type == "FG"

def prompt_enhancer_outputs_multiple_prompts(prompt_enhancer_mode):
    return "M" in str(prompt_enhancer_mode or "")

def normalize_generated_prompt_lines(prompt, multi_prompts_gen_type, multi_prompt_output=False):
    texts, commands = prompt_enhancer_images.strip_window_commands([str(prompt or "")])
    output = _normalize_generated_prompt_lines(texts[0], multi_prompts_gen_type, multi_prompt_output)
    return prompt_enhancer_images.restore_window_commands([output], commands, multi_prompts_gen_type)[0]

def _normalize_generated_prompt_lines(prompt, multi_prompts_gen_type, multi_prompt_output=False):
    prompt = str(prompt or "").replace("\r\n", "\n").replace("\r", "\n")
    if multi_prompt_output:
        prompt_lines = [line.strip() for line in prompt.split("\n") if line.strip()]
        if len(prompt_lines) == 0:
            return ""
        if str(multi_prompts_gen_type or "") == "FG":
            return "\n".join(prompt_lines)
        return ("\n\n" if "P" in str(multi_prompts_gen_type or "") else "\n").join(prompt_lines)
    if "P" in str(multi_prompts_gen_type or ""):
        return "\n".join(line for line in prompt.split("\n") if line.strip()).strip()
    if keep_generated_prompt_newlines(multi_prompts_gen_type):
        return prompt
    return re.sub(r"[\r\n]+", " ", prompt).strip()

def enhance_prompt(state, prompt, alt_prompt, prompt_enhancer, multi_images_gen_type, multi_prompts_gen_type, override_profile, video_prompt_type, image_prompt_type, audio_prompt_type, progress=WangpProgress()):
    model_type = get_state_model_type(state)
    inputs = get_model_settings(state, model_type)
    model_def = get_model_def(model_type)
    original_prompts = inputs["prompt"]

    if not model_def.get("skip_prompt_template", False):
        original_prompts, errors = prompt_parser.process_template(
            original_prompts,
            keep_comments=True,
            keep_empty_lines="P" in multi_prompts_gen_type or prompt_parser.PROMPT_UNIT_PREFIX in original_prompts,
        )
        if len(errors) > 0:
            gr.Info("Error processing prompt template: " + errors)
            return gr.update(), gr.update(), gr.update()
    original_prompts = prompt_parser.split_prompt_units(original_prompts, multi_prompts_gen_type, originals=True)
    routed = prompt_enhancer_chaining.uses_routing(prompt_enhancer)
    alt_prompt_inherits = bool(model_def.get("alt_prompt_inherits_prompt_paragraphs", False))
    alt_prompt_has_history = str(inputs["alt_prompt"] or "").startswith(prompt_parser.PROMPT_UNIT_PREFIX)
    alt_prompt_mode = multi_prompts_gen_type if alt_prompt_inherits and (alt_prompt_has_history or prompt_enhancer_chaining.requires_alt_prompt_inheritance(prompt_enhancer)) else "FG"
    original_alt_prompts = prompt_parser.split_prompt_units(str(inputs["alt_prompt"] or ""), alt_prompt_mode, originals=True) or [""]
    multi_prompt_output = prompt_enhancer_outputs_multiple_prompts(prompt_enhancer)
    if multi_prompt_output:
        original_prompts = original_prompts[:1]
    num_prompts = len(original_prompts) 
    alt_prompt_count_error = prompt_enhancer_chaining.validate_alt_prompt_count(num_prompts, len(original_alt_prompts), model_def["prompt_class"], model_def["alt_prompt"]["label"]) if alt_prompt_inherits and prompt_enhancer_chaining.requires_alt_prompt_inheritance(prompt_enhancer) else ""
    if alt_prompt_count_error:
        gr.Info(alt_prompt_count_error)
        return gr.update(), gr.update(), gr.update()
    image_prompt_type = inputs["image_prompt_type"]
    video_prompt_type = inputs["video_prompt_type"]
    multi_image_enhancer = prompt_enhancer_images.enabled(server_config, prompt_enhancer)
    image_contexts = None
    image_start = inputs["image_start"] if "S" in image_prompt_type else None
    continuation_image = None
    if inputs["image_mode"] == 0 and any_letters(image_prompt_type, "VL"):
        inputs = dict(inputs, video_source=prompt_enhancer_images.resolve_continue_video(inputs["video_source"], image_prompt_type, get_processed_queue(get_gen_info(state))[0], save_path))
        if "I" in prompt_enhancer and not image_start and inputs["video_source"] is not None:
            progress(0, desc="Reading Continuation Start Image")
            fps = get_computed_fps(inputs["force_fps"], get_base_model_type(model_type), inputs["video_guide"], inputs["video_source"])
            continuation_image = prompt_enhancer_images.continuation_start_image(inputs["video_source"], inputs["keep_frames_video_source"], fps, max_source_video_frames)
    if multi_image_enhancer or has_slash_commands(original_prompts) or continuation_image is not None:
        fps = get_computed_fps(inputs["force_fps"], get_base_model_type(model_type), inputs["video_guide"], inputs["video_source"])
        source_frames = estimate_first_window_overlap_frames(image_start, inputs["video_source"] if any_letters(image_prompt_type, "VL") else None, inputs["keep_frames_video_source"], fps)
        original_prompts, image_contexts = prompt_enhancer_images.prepare_manual(original_prompts, inputs, model_def, fps=fps, source_frames=source_frames, open_image=convert_image, multi_prompt_output=multi_prompt_output, continuation_image=continuation_image)
        num_prompts = len(original_prompts)
        if multi_image_enhancer:
            image_start = None
    if image_start is None and not multi_image_enhancer:
        image_start = inputs["image_end"] if "E" in image_prompt_type else None
    if image_start is None or not "I" in prompt_enhancer:
        image_start = [None] * num_prompts
    else:
        image_start = [convert_image(img[0]) for img in image_start]
        if multi_prompt_output:
            image_start = image_start[:1] or [None] * num_prompts
        elif len(image_start) == 1:
            image_start = image_start * num_prompts
        else:
            if multi_images_gen_type !=1:
                gr.Info("On Demand Prompt Enhancer with multiple Start Images requires that option 'Match images and text prompts' is set")
                return gr.update(), gr.update(), gr.update()

            if len(image_start) != num_prompts:
                gr.Info("On Demand Prompt Enhancer supports only mutiple Start Images if their number matches the number of Text Prompts")
                return gr.update(), gr.update(), gr.update()

    if continuation_image is not None and not multi_image_enhancer:
        image_start = [next((image for image, label in zip(context.images, context.labels) if label in ("start image", "end image")), None) for context in image_contexts]
    original_image_refs = inputs["image_refs"] if "I" in video_prompt_type else None
    if original_image_refs is not None:
        original_image_refs = [ convert_image(tup[0]) for tup in original_image_refs ]        
    is_image = inputs["image_mode"] > 0
    seed = inputs["seed"]

    model_def = get_model_def(get_state_model_type(state))
    audio_only = model_def.get("audio_only", False)
    enhancer_kwargs = {"image_prompt_type": image_prompt_type, "video_prompt_type": video_prompt_type, "audio_prompt_type": audio_prompt_type, "audio_guide": inputs["audio_guide"] if "A" in audio_prompt_type else None, "duration_seconds": inputs["duration_seconds"]}
    enhancer_kwargs["control_image"] = prompt_enhancer_images.control_image_input(inputs)
    enhancer_kwargs["caption_input_info"] = prompt_enhancer_images.caption_input_info((inputs["image_start"] if "S" in image_prompt_type else None) or continuation_image, inputs["image_end"] if "E" in image_prompt_type else None, original_image_refs, enhancer_kwargs["control_image"], "K" in video_prompt_type)
    if image_contexts is not None:
        enhancer_kwargs["image_contexts"] = image_contexts
    if model_def.get("prompt_enhancer_video_duration", False) and not is_image and not audio_only:
        frames = min(inputs["video_length"], inputs["sliding_window_size"]) if model_def.get("sliding_window", False) else inputs["video_length"]
        fps = get_computed_fps(inputs["force_fps"], get_base_model_type(model_type), inputs["video_guide"], inputs["video_source"])
        enhancer_kwargs["video_duration_seconds"] = frames / fps
    enhancer_kwargs["generation_callbacks"] = {"stop_requested": lambda: progress.gen["abort"], "enhancement_progress": EnhancementProgress(progress, progress.stream_tokens)}
    enhanced = exec_prompt_enhancer_engine(state, model_type, model_def, prompt_enhancer, original_prompts, image_start, original_image_refs, is_image, audio_only, seed, progress, override_profile, enhancer_kwargs=enhancer_kwargs, alt_prompts=original_alt_prompts if routed else None, alt_prompt_inherits_prompt_paragraphs=alt_prompt_inherits)

    progress.check_cancelled()

    if routed:
        if enhanced.prompt_changed:
            output_prompts = [normalize_generated_prompt_lines(value, multi_prompts_gen_type, multi_prompt_output=multi_prompt_output) for value in enhanced.prompts]
            prompt = prompt_parser.serialize_prompt_blocks_with_prefix(output_prompts, original_prompts)
        if enhanced.alt_prompt_changed:
            output_alt_prompts = [normalize_generated_prompt_lines(value, multi_prompts_gen_type) for value in enhanced.alt_prompts]
            alt_prompt = prompt_parser.serialize_prompt_blocks_with_prefix(output_alt_prompts, prompt_enhancer_chaining.originals_for_outputs(original_alt_prompts, len(output_alt_prompts)))
    else:
        output_prompts = [normalize_generated_prompt_lines(value[0], multi_prompts_gen_type, multi_prompt_output=multi_prompt_output) for value in enhanced if value is not None]
        prompt = prompt_parser.serialize_prompt_blocks_with_prefix(output_prompts, original_prompts)
    if num_prompts > 1:
        gr.Info(f'{num_prompts} Prompts have been Enhanced')
    else:
        gr.Info("Prompt enhancement completed")
    return prompt, alt_prompt, prompt

def parse_guide_inpaint_color(value):
    if isinstance(value, str):
        cleaned = value.strip()
        hex_value = cleaned[1:] if cleaned.startswith("#") else cleaned
        if len(hex_value) == 6 and all(c in "0123456789abcdefABCDEF" for c in hex_value):
            return tuple(int(hex_value[i:i+2], 16) for i in (0, 2, 4))
        if cleaned.lower().startswith("rgb"):
            cleaned = cleaned[3:]
        for ch in "()[]{}":
            cleaned = cleaned.replace(ch, "")
        cleaned = cleaned.replace(",", " ")
        parts = [p for p in cleaned.split() if p]
        if len(parts) == 3:
            try:
                return tuple(max(0, min(255, int(round(float(p))))) for p in parts)
            except ValueError:
                return 127.5
        return 127.5
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(max(0, min(255, int(round(float(p))))) for p in value)
    return value

def resolve_model_preprocess_all(model_def, **kwargs):
    preprocess_all = model_def.get("preprocess_all", False)
    return preprocess_all(**kwargs) if callable(preprocess_all) else preprocess_all

def get_outpainting_quantize_margins(model_def):
    value = model_def.get("outpainting_quantize_margins", 0)
    return int(model_def.get("vae_block_size", 16) if value is True else value or 0)

def custom_preprocess_video_with_mask(model_handler, base_model_type, pre_video_guide, video_guide, video_mask, height, width, max_frames, start_frame, fit_canvas, fit_crop, target_fps, block_size, expand_scale, video_prompt_type, model_def=None, custom_settings=None):
    pad_frames = 0
    if start_frame < 0:
        pad_frames= -start_frame
        max_frames += start_frame
        start_frame = 0

    max_workers = get_default_workers()

    if not video_guide or max_frames <= 0:
        return None, None, None, None
    any_mask = video_mask is not None
    model_def = model_def or {}
    raw_custom_preprocessor_inputs = model_def.get("custom_preprocessor_raw_inputs", False)
    video_guide = get_resampled_video(video_guide, start_frame, max_frames, target_fps)
    if not raw_custom_preprocessor_inputs:
        video_guide = video_guide.permute(-1, 0, 1, 2) / 127.5 - 1.
    pose_mask = None
    if video_mask is not None:
        video_mask = get_resampled_video(video_mask, start_frame, max_frames, target_fps)
        if not raw_custom_preprocessor_inputs:
            video_mask = video_mask.permute(-1, 0, 1, 2).float()[:1] / 255.
            pose_mask = video_mask

    # Mask filtering: resize, binarize, expand mask and keep only masked areas of video guide
    if any_mask and not raw_custom_preprocessor_inputs:
        invert_mask = "N" in video_prompt_type
        tgt_h, tgt_w = video_guide.shape[2], video_guide.shape[3]
        def process_mask(idx):
            return torch.from_numpy(prepare_binary_mask_frame(pose_mask[0, idx] * 255, tgt_h, tgt_w, expand_scale=expand_scale, invert=invert_mask))
        pose_mask = torch.stack(process_images_multithread(process_mask, list(range(pose_mask.shape[1])), "prephase", wrap_in_list=False, max_workers=max_workers, in_place=False)).unsqueeze(0)
        video_guide = video_guide * pose_mask + (-1) * (1-pose_mask)
        video_mask = pose_mask

    guide_frame_count = video_guide.shape[0] if raw_custom_preprocessor_inputs else video_guide.shape[1]
    mask_frame_count = video_mask.shape[0] if raw_custom_preprocessor_inputs and any_mask else video_mask.shape[1] if any_mask else 0
    if guide_frame_count == 0 or any_mask and mask_frame_count == 0:
        return None, None, None, None
    
    video_guide_processed, video_guide_processed2, video_mask_processed, video_mask_processed2  = model_handler.custom_preprocess(base_model_type = base_model_type, pre_video_guide = pre_video_guide, video_guide = video_guide, video_mask = video_mask, pose_mask = pose_mask, height = height, width = width, fit_canvas = fit_canvas , fit_crop = fit_crop, target_fps = target_fps,  block_size = block_size, max_workers = max_workers, expand_scale = expand_scale, video_prompt_type=video_prompt_type, model_def=model_def, custom_settings=custom_settings)

    # if pad_frames > 0:
    #     masked_frames = masked_frames[0] * pad_frames + masked_frames
    #     if any_mask: masked_frames = masks[0] * pad_frames + masks

    return video_guide_processed, video_guide_processed2, video_mask_processed, video_mask_processed2 

def get_output_filepath(file_path, is_image, audio_only):
    if is_image:
        base_path = image_save_path
    elif audio_only:
        base_path = audio_save_path
    else:
        base_path = save_path
    return get_available_filename(base_path, file_path)


def record_file_metadata(video_path, configs, is_image, audio_only, gen, embedded_images=None, replace_last_file=False, notify_generation=True, write_metadata=True, record_notification=True):
    shared_record_file_metadata(video_path, configs, is_image, audio_only, gen, get_processed_queue=get_processed_queue, metadata_choice=server_config.get("metadata_type", "metadata"), embedded_images=embedded_images, replace_last_file=replace_last_file, lock=lock, verbose_level=verbose_level, write_metadata=write_metadata)
    if record_notification:
        notifications.record_generation(server_config, gen, video_path, configs, replace_last=replace_last_file, notify=notify_generation)


@generation_downloads(get_gen_info, lambda: offloadobj)
def generate_media(
    task,
    send_cmd,
    client_id,
    image_mode,
    prompt,
    alt_prompt,
    negative_prompt,    
    resolution,
    video_length,
    duration_seconds,
    pause_seconds,
    batch_size,
    seed,
    force_fps,
    num_inference_steps,
    guidance_scale,
    guidance2_scale,
    guidance3_scale,
    switch_threshold,
    switch_threshold2,
    guidance_phases,
    model_switch_phase,
    alt_guidance_scale,
    alt_scale,
    audio_guidance_scale,
    audio_scale,
    flow_shift,
    sample_solver,
    embedded_guidance_scale,
    repeat_generation,
    multi_prompts_gen_type,
    multi_images_gen_type,
    skip_steps_cache_type,
    skip_steps_multiplier,
    skip_steps_start_step_perc,    
    activated_loras,
    loras_multipliers,
    image_prompt_type,
    image_start,
    image_end,
    model_mode,
    video_source,
    keep_frames_video_source,
    input_video_strength,
    video_prompt_type,
    image_refs,
    frames_positions,
    video_guide,
    video_guide2,
    video_guide3,
    image_guide,
    keep_frames_video_guide,
    denoising_strength,
    masking_strength,     
    video_guide_outpainting,
    video_guide_outpainting_ratio,
    video_mask,
    image_mask,
    control_net_weight,
    control_net_weight2,
    control_net_weight_alt,
    motion_amplitude,
    mask_expand,
    audio_guide,
    audio_guide2,
    audio_guide3,
    custom_guide,
    audio_source,
    replace_voice_method,
    replace_voice_sample,
    replace_voice_sample2,
    audio_prompt_type,
    speakers_locations,
    sliding_window_size,
    sliding_window_overlap,
    sub_parallel_window_size,
    sub_parallel_window_overlap,
    sliding_window_color_correction_strength,
    sliding_window_overlap_noise,
    sliding_window_discard_last_frames,
    sliding_window_trim_first_frames,
    image_refs_relative_size,
    remove_background_images_ref,
    temporal_upsampling,
    spatial_upsampling,
    spatial_upsampler_prompt,
    spatial_upsampler_reference_images,
    spatial_upsampler_param,
    spatial_upsampler_param2,
    film_grain_intensity,
    film_grain_saturation,
    postprocess_audio,
    postprocess_audio_prompt,
    postprocess_audio_neg_prompt,
    RIFLEx_setting,
    NAG_scale,
    NAG_tau,
    NAG_alpha,
    perturbation_switch,
    perturbation_layers,
    perturbation_start_perc,
    perturbation_end_perc,
    apg_switch,
    cfg_star_switch,
    cfg_zero_step,
    prompt_enhancer,
    min_frames_if_references,
    override_profile,
    override_attention,
    attention_sparsity,
    temperature,
    custom_settings,
    top_p,
    top_k,
    self_refiner_setting,
    self_refiner_plan,
    self_refiner_f_uncertainty,
    self_refiner_certain_percentage,
    output_filename,
    config,
    state,
    model_type,
    mode,
    plugin_data=None,
    spatial_upsampler_parameters=None,
):
    wait_for_model_unload()

    def remove_temp_filenames(temp_filenames_list):
        for temp_filename in temp_filenames_list: 
            if temp_filename!= None and os.path.isfile(temp_filename):
                os.remove(temp_filename)

    def set_progress_status(status):
        phase_text = str(status or "").strip()
        if len(phase_text) == 0:
            return
        gen["progress_phase"] = (phase_text, -1)
        send_cmd("progress", [0, get_latest_status(state, phase_text)])

    global wan_model, offloadobj, reload_needed, loaded_config
    gen = get_gen_info(state)
    api_return_video_uint8, api_return_audio = get_api_output_options(plugin_data)
    api_options = plugin_data.get("api", {}) if isinstance(plugin_data, dict) and isinstance(plugin_data.get("api", {}), dict) else {}
    flashvsr_continue_cache = api_options.get("flashvsr_continue_cache")
    return_flashvsr_continue_cache = bool(api_options.get("return_flashvsr_continue_cache"))
    api_return_side_files = bool(api_options.get("return_side_files", api_return_video_uint8 or api_return_audio))
    gen["last_progress_args"] = None
    torch.set_grad_enabled(False) 
    if mode == "edit_audio":
        edit_audio(send_cmd, state, audio_source, postprocess_audio, replace_voice_sample, replace_voice_sample2, client_id=client_id, plugin_data=plugin_data)
        return True
    if mode.startswith("edit_"):
        edit_media(send_cmd, state, mode, video_source, seed, temporal_upsampling, spatial_upsampling, film_grain_intensity, film_grain_saturation, postprocess_audio, postprocess_audio_prompt, postprocess_audio_neg_prompt, repeat_generation, audio_source, replace_voice_method, replace_voice_sample, replace_voice_sample2, prompt=prompt, image_refs=image_refs, image_refs_relative_size=image_refs_relative_size, spatial_upsampler_prompt=spatial_upsampler_prompt, spatial_upsampler_reference_images=spatial_upsampler_reference_images, spatial_upsampler_param=spatial_upsampler_param, spatial_upsampler_param2=spatial_upsampler_param2, spatial_upsampler_parameters=spatial_upsampler_parameters, client_id=client_id, plugin_data=plugin_data)
        return True
    enhancer_mode = server_config.get("enhancer_mode", 1)
    auto_prompt_enhancer_requested = server_config.get("enhancer_enabled", 0) > 0 and enhancer_mode == 0 and prompt_enhancer is not None and len(prompt_enhancer) > 0
    postprocess_audio = audio_processor_api.normalize_method(postprocess_audio or "")
    replace_voice_method = audio_processor_api.normalize_method(replace_voice_method or "")
    postprocess_audio_meta = audio_processor_api.method_metadata(postprocess_audio)
    replace_voice_meta = audio_processor_api.method_metadata(replace_voice_method)
    if not postprocess_audio_meta["needs_audio_source"]:
        audio_source = None
    if not replace_voice_method:
        replace_voice_sample = None
        replace_voice_sample2 = None
    elif not replace_voice_meta["needs_voice_sample2"]:
        replace_voice_sample2 = None

    model_def = get_model_def(model_type) 
    config_groups = get_model_config_groups(model_type, model_def)
    config = model_config_groups.normalize_config_selection(config_groups, config)
    is_image = image_mode > 0
    audio_only = model_def.get("audio_only", False)
    duration_def = model_def.get("duration_slider", None)

    set_video_prompt_type = model_def.get("set_video_prompt_type", None)
    if set_video_prompt_type is not None:
        video_prompt_type = add_to_sequence(video_prompt_type, set_video_prompt_type)
    reference_videos = "-" in video_prompt_type
    preprocess_video_guide2 = model_def.get("preprocess_video_guide2", False)
    if is_image:
        image_batch_size_max = max(1, int(model_def.get("image_batch_size_max", 16)))
        if batch_size > image_batch_size_max:
            raise ValueError(f"This model supports a maximum of {image_batch_size_max} image{'s' if image_batch_size_max > 1 else ''} per generation.")
        if not model_def.get("custom_video_length", False):
            if min_frames_if_references >= 1000:
                video_length = min_frames_if_references - 1000
            else:
                video_length = min_frames_if_references if "I" in video_prompt_type or "V" in video_prompt_type else 1 
    else:
        batch_size = 1
    temp_filenames_list = []

    if image_guide is not None:
        if isinstance(image_guide, str): 
            image_guide = _open_image_input(image_guide)
        if isinstance(image_guide, Image.Image):
            video_guide = image_guide
            image_guide = None

    if image_mask is not None:
        if isinstance(image_mask, str):
            image_mask = _open_image_input(image_mask)
        if isinstance(image_mask, Image.Image):
            video_mask = image_mask
            image_mask = None

    if model_def.get("no_background_removal", False): remove_background_images_ref = 0
    
    base_model_type = get_base_model_type(model_type)
    model_handler = get_model_handler(base_model_type)
    block_size = model_def.get("vae_block_size", 16)
    width, height = resolution.split("x")
    width, height = int(width) // block_size * block_size, int(height) // block_size * block_size

    from preprocessing.processors import prepare_generation_assets
    prepare_generation_assets(model_def, base_model_type, locals(), server_config, process_map_video_guide, process_map_outside_mask)
    download_requested_postprocessing_assets(
        send_cmd,
        postprocess_audio=postprocess_audio if not (is_image or audio_only) else "",
        temporal_upsampling=temporal_upsampling if not (is_image or audio_only) else "",
        spatial_upsampling=spatial_upsampling if not audio_only else "",
        replace_voice_method=replace_voice_method if not (is_image or audio_only) else "",
        gen=gen,
    )

    if "P" in preload_model_policy and not "U" in preload_model_policy:
        while wan_model == None:
            time.sleep(1)
    model_kwargs = {}
    new_model_vae_upsampling = upsampler_api.model_load_vae_upsampling_value(spatial_upsampling, base_model_type, model_def, image_mode)
    old_model_vae_upsampling = None if reload_needed or wan_model is None else upsampler_api.loaded_model_vae_upsampling_value(wan_model)
    reload_needed = reload_needed or old_model_vae_upsampling != new_model_vae_upsampling
    if new_model_vae_upsampling:
        model_kwargs.update(upsampler_api.model_load_kwargs_for_vae_upsampling(spatial_upsampling, base_model_type, model_def, image_mode))
    output_type = get_profile_type_for_model(model_type, image_mode)
    profile = compute_profile(override_profile, output_type)
    if model_type != transformer_type or reload_needed or profile != loaded_profile or config != loaded_config:
        release_model()
        loading_model_name = get_model_name(model_type)
        send_cmd("status", f"Loading Model {loading_model_name}...")
        def loading_progress(phase, completed, total, model_id):
            component = " ".join(word.upper() if word.lower() in ("vae", "clip", "llm", "t5") else word.capitalize() for word in model_id.replace("_", " ").split())
            subtask = f"Pinning Model{' ' + component if component else ''} to Reserved RAM" if phase == "Pinning" else phase + (f" {component} Model" if component else "")
            title = f"Loading {loading_model_name} - {subtask}"
            send_cmd("progress", [(completed, total), title, total, "blocks" if phase == "Pinning" else "phases"])

        try:
            wan_model, offloadobj = load_models(model_type, override_profile, output_type=output_type, config_id=config, gen=gen, loading_callback=offload.LoadingCallback(lambda: gen.get("abort", False), loading_progress), **model_kwargs)
        except offload.LoadingCancelled as error:
            traceback.clear_frames(error.__traceback__)
            gc.collect()
            return True
        send_cmd("status", f"Model {loading_model_name} Loaded")
        send_cmd("refresh_models", get_unique_id())
        reload_needed=  False
    vae_upsampler_handler = upsampler_api.find_vae_upsampler(spatial_upsampling)
    vae_upsampler_session = None
    if vae_upsampler_handler is not None and hasattr(vae_upsampler_handler, "prepare_vae_upsampler"):
        upsampler_name = vae_upsampler_handler.query_upsampler_def()["name"]
        send_cmd("status", f"Preparing {upsampler_name} upsampler...")
        vae_upsampler_session = upsampler_api.prepare_vae_upsampler(vae_upsampler_handler, spatial_upsampling, send_cmd=send_cmd, process_files=process_files_def, init_pipe=init_pipe, profile=compute_profile(override_profile, upsampler_api.profile_type_for_handler(vae_upsampler_handler)), attention_mode=attention_mode, spatial_upsampler_param=spatial_upsampler_param, spatial_upsampler_param2=spatial_upsampler_param2, spatial_upsampler_parameters=spatial_upsampler_parameters)
        send_cmd("status", f"{upsampler_name} upsampler prepared")
    if args.test and auto_prompt_enhancer_requested:
        try:
            ensure_prompt_enhancer_loaded(override_profile=override_profile, send_cmd=send_cmd, gen=gen)
        finally:
            unload_prompt_enhancer_runtime()
            if enhancer_offloadobj is not None:
                enhancer_offloadobj.unload_all()
    if args.test:
        upsampler_api.release_vae_upsampler(vae_upsampler_handler, vae_upsampler_session)
        send_cmd("info", "Test mode: model loaded, skipping generation.")
        return True
    overridden_attention = override_attention if len(override_attention) else get_overridden_attention(model_type)
    # if overridden_attention is not None and overridden_attention !=  attention_mode: print(f"Attention mode has been overriden to {overridden_attention} for model type '{model_type}'")
    attn = overridden_attention if overridden_attention is not None else attention_mode
    if attn == "auto":
        attn = get_auto_attention()
    elif attn not in override_attention_modes_supported and not (
        (custom := model_def.get("custom_attention_modes", {}).get(attn)) and
        custom.get("supported", ATTENTION_MODE_AVAILABILITY.get(attn, {}).get("supported", custom.get("installed", ATTENTION_MODE_AVAILABILITY.get(attn, {}).get("installed", True))))
    ):
        send_cmd("info", f"You have selected attention mode '{attn}'. However it is not installed or supported on your system. You should either install it or switch to the default 'sdpa' attention.")
        send_cmd("exit")
        return True
    elif attn in ATTENTION_MODE_AVAILABILITY and attn not in model_def.get("custom_attention_modes", {}):
        send_cmd("info", f"Attention mode '{attn}' is not implemented for model type '{model_type}'.")
        send_cmd("exit")
        return True
    
    default_image_size = (height, width)

    if perturbation_switch == 0:
        perturbation_layers = None

    offload.shared_state["_attention"] =  attn
    device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
    if  hasattr(wan_model, "vae") and hasattr(wan_model.vae, "get_VAE_tile_size"):
        get_tile_size = wan_model.vae.get_VAE_tile_size
        try:
            sig = inspect.signature(get_tile_size)
        except (TypeError, ValueError):
            sig = None
        if sig is not None and "output_height" in sig.parameters:
            VAE_tile_size = get_tile_size(
                vae_config,
                device_mem_capacity,
                server_config.get("vae_precision", "16") == "32",
                output_height=height,
                output_width=width,
            )
        else:
            VAE_tile_size = get_tile_size(
                vae_config,
                device_mem_capacity,
                server_config.get("vae_precision", "16") == "32",
            )
    else:
        VAE_tile_size = None

    trans = get_transformer_model(wan_model)
    trans2 = get_transformer_model(wan_model, 2)
    audio_sampling_rate = 16000

    prompt_has_history = str(prompt or "").startswith(prompt_parser.PROMPT_UNIT_PREFIX)
    prompts = prompt_parser.split_prompt_units(prompt, multi_prompts_gen_type)
    display_prompts = prompts.copy()
    display_original_prompts = prompt_parser.split_prompt_units(prompt, multi_prompts_gen_type, originals=True) if prompt_has_history else display_prompts.copy()
    alt_prompt_inherits = bool(model_def.get("alt_prompt_inherits_prompt_paragraphs", False))
    alt_prompt_has_history = str(alt_prompt or "").startswith(prompt_parser.PROMPT_UNIT_PREFIX)
    auto_dependent_alt_prompt = auto_prompt_enhancer_requested and prompt_enhancer_chaining.requires_alt_prompt_inheritance(prompt_enhancer)
    alt_prompt_mode = multi_prompts_gen_type if alt_prompt_inherits and (alt_prompt_has_history or auto_dependent_alt_prompt) else "FG"
    alt_prompts = prompt_parser.split_prompt_units(str(alt_prompt or ""), alt_prompt_mode) or [""]
    display_alt_prompts = alt_prompts.copy()
    display_original_alt_prompts = prompt_parser.split_prompt_units(str(alt_prompt or ""), alt_prompt_mode, originals=True) if alt_prompt_has_history else display_alt_prompts.copy()
    frames_minimum, frames_steps, latent_size = get_model_min_frames_and_step(model_type)
    parsed_keep_frames_video_source= max_source_video_frames if len(keep_frames_video_source) ==0 else int(keep_frames_video_source) 
    transformer_loras_filenames, transformer_loras_multipliers  = get_transformer_loras(model_type)
    lora_dir = get_lora_dir(model_type)
    if guidance_phases < 1: guidance_phases = 1
    lora_multiplier_phases = int(model_def.get("lora_multiplier_phases", guidance_phases) or guidance_phases or 1)
    lora_multiplier_branches = model_def.get("lora_multiplier_branches", None)
    if transformer_loras_filenames != None:
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(transformer_loras_multipliers, len(transformer_loras_filenames), num_inference_steps, nb_phases = lora_multiplier_phases, model_switch_phase= model_switch_phase, lora_multiplier_branches=lora_multiplier_branches)
        if len(errors) > 0: raise Exception(f"Error parsing Transformer Loras: {errors}")
        loras_selected = transformer_loras_filenames[:] 

    if hasattr(wan_model, "get_loras_transformer"):
        extra_loras_transformers, extra_loras_multipliers = wan_model.get_loras_transformer(get_model_recursive_prop, **locals())
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(extra_loras_multipliers, len(extra_loras_transformers), num_inference_steps, nb_phases = lora_multiplier_phases, merge_slist= loras_slists, model_switch_phase= model_switch_phase, lora_multiplier_branches=lora_multiplier_branches)
        if len(errors) > 0: raise Exception(f"Error parsing Extra Transformer Loras: {errors}")
        loras_selected += extra_loras_transformers 

    base_loras_slists = loras_slists
    if len(activated_loras) > 0:
        loras_list_mult_choices_nums, loras_slists, errors =  parse_loras_multipliers(loras_multipliers, len(activated_loras), num_inference_steps, nb_phases = lora_multiplier_phases, merge_slist= loras_slists, model_switch_phase= model_switch_phase, lora_multiplier_branches=lora_multiplier_branches)
        if len(errors) > 0: raise Exception(f"Error parsing Loras: {errors}")
        loras_selected += activated_loras

    if hasattr(wan_model, "get_trans_lora"):
        trans_lora, trans2_lora = wan_model.get_trans_lora()
    else:
        trans_lora, trans2_lora = trans, trans2

    if len(loras_selected) > 0:
        loras_selected = update_loras_url_cache(lora_dir, loras_selected)
        errors = check_loras_exist(model_type, loras_selected, True, send_cmd, gen=gen)
        if len(errors) > 0 : raise gr.Error(errors)
        loras_selected = [ get_lora_local_path(lora_dir, lora) for lora in loras_selected]
        pinnedLora = not is_mps and loaded_profile !=5  # and transformer_loras_filenames == None False # # #
        preprocess_target = trans_lora if trans_lora is not None else trans
        split_linear_modules_map = getattr(preprocess_target, "split_linear_modules_map", None)
        offload.load_loras_into_model(
            trans_lora,
            loras_selected,
            loras_list_mult_choices_nums,
            activate_all_loras=True,
            preprocess_sd=get_loras_preprocessor(preprocess_target, base_model_type),
            pinnedLora=pinnedLora,
            maxReservedLoras=server_config.get("max_reserved_loras", -1),
            split_linear_modules_map=split_linear_modules_map,
        )
        errors = trans_lora._loras_errors
        if len(errors) > 0:
            error_files = [msg for _ ,  msg  in errors]
            raise gr.Error("Error while loading Loras: " + ", ".join(error_files))
        if trans2_lora is not None: 
            offload.sync_models_loras(trans_lora, trans2_lora)
        
    seed = None if seed == -1 else seed
    # negative_prompt = "" # not applicable in the inference
    model_filename = get_model_filename(base_model_type)  

    frames_offset = model_def.get("frames_offset", 1)
    video_length = floor_frame_count(video_length, frames_minimum, latent_size, frames_offset)
    if sliding_window_size !=0:
        sliding_window_size = floor_frame_count(sliding_window_size, frames_minimum, latent_size, frames_offset)
    sliding_window_defaults = model_def.get("sliding_window_defaults", {})
    if sliding_window_overlap !=0:
        if sliding_window_defaults.get("overlap_default", 0) != sliding_window_overlap:
            overlap_offset = sliding_window_defaults.get("overlap_offset", 1)
            sliding_window_overlap = sliding_window_overlap // latent_size * latent_size if overlap_offset == 0 else (sliding_window_overlap - overlap_offset) // latent_size * latent_size + overlap_offset
    sub_parallel_window_size = max(0, int(sub_parallel_window_size or 0))
    if sub_parallel_window_size !=0:
        sub_parallel_window_size += 1
    sub_parallel_window_overlap = max(0, int(sub_parallel_window_overlap or 0))
    if sliding_window_discard_last_frames !=0:
        sliding_window_discard_last_frames = sliding_window_discard_last_frames // latent_size * latent_size 

    current_video_length = video_length
    # VAE Tiling
    device_mem_capacity = torch.cuda.get_device_properties(None).total_memory / 1048576
    outpainting_dims = get_outpainting_dims(video_guide_outpainting, video_guide_outpainting_ratio)
    any_outpainting = outpainting_dims is not None
    outpainting_quantize_margins = get_outpainting_quantize_margins(model_def) if any_outpainting else 0
    guide_inpaint_color = model_def.get("guide_inpaint_color", 127.5)
    if image_mode==2:
        guide_inpaint_color = model_def.get("inpaint_color", guide_inpaint_color)
    if callable(guide_inpaint_color):
        guide_inpaint_color = guide_inpaint_color(video_prompt_type, any_outpainting, custom_settings if isinstance(custom_settings, dict) else {})
    guide_inpaint_color = parse_guide_inpaint_color(guide_inpaint_color)
    extract_guide_from_window_start = model_def.get("extract_guide_from_window_start", False) 
    hunyuan_custom = "hunyuan_video_custom" in model_filename
    hunyuan_custom_edit =  hunyuan_custom and "edit" in model_filename
    fantasy = base_model_type in ["fantasy"]
    multitalk = model_def.get("multitalk_class", False)
    fake_start_image = model_def.get("fake_start_image", False) and image_start is not None

    if (multitalk or model_def.get("speaker_locations", False)) and ("B" in audio_prompt_type or "X" in audio_prompt_type):
        from models.wan.multitalk.multitalk import parse_speakers_locations
        speakers_bboxes, error = parse_speakers_locations(speakers_locations)
    else:
        speakers_bboxes = None        
    if "L" in image_prompt_type:
        video_source = prompt_enhancer_images.resolve_continue_video(video_source, image_prompt_type, get_processed_queue(gen)[0], save_path)
    fps = 1 if is_image else get_computed_fps(force_fps, base_model_type , video_guide, video_source )
    gen_state = {}
    frame_scheduler = None
    first_window_available_overlap = estimate_first_window_overlap_frames(None if fake_start_image else image_start, video_source, keep_frames_video_source, fps)
    scheduler_supported = frame_scheduler_supported(model_type, model_def, image_mode)
    if not scheduler_supported and has_slash_commands(prompts):
        raise gr.Error("Prompt slash window commands require a video model with Sliding Window support.")
    if scheduler_supported:
        frame_scheduler, frame_scheduler_error = build_frame_scheduler(
            prompts,
            total_frames=int(video_length or frames_minimum),
            fps=float(fps),
            window_size=int(sliding_window_size or video_length or frames_minimum),
            default_overlap=int(sliding_window_overlap or 0),
            minimum=frames_minimum,
            step=frames_steps,
            frame_offset=frames_offset,
            overlap_offset=model_def.get("sliding_window_defaults", {}).get("overlap_offset", 1),
            max_overlap=model_def.get("sliding_window_defaults", {}).get("overlap_max"),
            preserve_exact_output_frames=model_def.get("image_end_frame_position", False),
            output_frame_policy=model_def.get("frame_scheduler_output_policy"),
            supported_model_commands=model_def.get("prompt_slash_commands", []),
            allow_new_shot=image_prompt_types_allow_t2v(model_def, image_mode),
            first_window_overlap_frames=first_window_available_overlap,
            initial_shared_frames=1 if (not fake_start_image and image_start is not None) or video_source is not None else 0,
            discard_last_frames=sliding_window_discard_last_frames,
        )
        if frame_scheduler_error is not None:
            raise gr.Error(frame_scheduler_error)
    if frame_scheduler is not None and frame_scheduler["active"]:
        loras_mult_error = prepare_loras_mult_windows(frame_scheduler, activated_loras, num_inference_steps, lora_multiplier_phases, base_loras_slists=base_loras_slists, model_switch_phase=model_switch_phase, store_slists=True, lora_multiplier_branches=lora_multiplier_branches)
        if loras_mult_error is not None: raise gr.Error(loras_mult_error)
        prompts = frame_scheduler["prompts"]
        video_length = frame_scheduler["predicted_total_frames"]
        current_video_length = video_length
    control_audio_tracks = source_audio_tracks = source_audio_metadata = []
    if postprocess_audio == "control" and video_guide is not None:
        control_audio_tracks, _  = extract_audio_tracks(video_guide, temp_format="wav")
    if "K" in audio_prompt_type and video_guide is not None:
        extracted_guides = [None, None, None]
        for index, guide in enumerate((video_guide, video_guide2, video_guide3)):
            if guide is None:
                continue
            try:
                if extract_audio_tracks(guide, query_only=True) == 0:
                    print(f"No audio track found in Control Video #{index + 1}: {guide}")
                else:
                    extracted_guides[index] = extract_audio_track_to_wav(guide, get_available_filename(save_path, guide, suffix=f"_control_audio{index + 1}", force_extension=".wav"))
                    temp_filenames_list.append(extracted_guides[index])
            except Exception as e:
                print(f"Unable to extract Audio track from Control Video #{index + 1}: {e}")
        audio_guide, audio_guide2, audio_guide3 = extracted_guides
    if video_source is not None:
        source_audio_tracks, source_audio_metadata = extract_audio_tracks(video_source, temp_format="wav")
        video_fps, _, _, video_frames_count = get_video_info(video_source)
        video_source_duration = video_frames_count / video_fps
    else:
        video_source_duration = 0

    if "A" in audio_prompt_type and audio_guide is None:
        audio_guide = create_silent_wav_file(save_path, video_length / fps, audio_sampling_rate)
        temp_filenames_list.append(audio_guide)


    reset_control_aligment = "T" in video_prompt_type

    original_image_refs = image_refs
    image_refs = None if image_refs is None else ([] + image_refs) # work on a copy as it is going to be modified
    # image_refs = None
    # nb_frames_positions= 0
    # Output Video Ratio Priorities:
    # Source Video or Start Image > Control Video > Image Ref (background or positioned frames only) >  UI Width, Height
    # Image Ref (non background and non positioned frames) are boxed in a white canvas in order to keep their own width/height ratio
    frames_to_inject = []
    any_background_ref  = 0
    custom_frames_injection = model_def.get("custom_frames_injection", False) and "F" in video_prompt_type and image_refs is not None and len(image_refs) > 0
    if "K" in video_prompt_type: 
        any_background_ref = 2 if model_def.get("all_image_refs_are_background_ref", False) or custom_frames_injection else 1
    fit_canvas = server_config.get("fit_canvas", 0)
    fit_crop = fit_canvas == 2
    if fit_crop and outpainting_dims is not None:
        fit_crop = False
        fit_canvas = 0

    joint_pass = boost ==1 #and profile != 1 and profile != 3  
    
    skip_steps_cache = None if len(skip_steps_cache_type) == 0 else DynamicClass(cache_type = skip_steps_cache_type) 

    if skip_steps_cache != None:
        skip_steps_cache.update({     
        "multiplier" : skip_steps_multiplier,
        "start_step":  int(skip_steps_start_step_perc*num_inference_steps/100)
        })
        model_handler.set_cache_parameters(skip_steps_cache_type, base_model_type, model_def, locals(), skip_steps_cache)
        if skip_steps_cache_type == "mag":
            def_mag_ratios = model_def.get("magcache_ratios", None) if model_def != None else None
            if def_mag_ratios is not None: skip_steps_cache.def_mag_ratios = def_mag_ratios
        elif skip_steps_cache_type == "tea":
            def_tea_coefficients = model_def.get("teacache_coefficients", None) if model_def != None else None
            if def_tea_coefficients is not None: skip_steps_cache.coefficients = def_tea_coefficients
        elif skip_steps_cache_type in ("spectrum", "first_block"):
            pass
        else:
            raise Exception(f"unknown cache type {skip_steps_cache_type}")
    trans.cache = skip_steps_cache
    if trans2 is not None: trans2.cache = skip_steps_cache
    face_arc_embeds = None
    src_ref_images = src_ref_masks = None
    output_new_audio_data = None
    output_new_audio_filepath = None
    original_audio_guide = audio_guide
    original_audio_guide2 = audio_guide2
    original_audio_guide3 = audio_guide3
    audio_proj_split = None
    audio_proj_full = None
    audio_scale = audio_scale if model_def.get("audio_scale_name") else None
    audio_context_lens = None
    full_audio_guide_waveform, full_audio_guide_sample_rate = None, 0
    control_video_trim = not model_def.get("control_video_trim_disabled", False) and (model_def.get("control_video_trim", False) or "|" in video_prompt_type)

    if audio_guide != None:
        from preprocessing.extract_vocals import get_vocals
        import librosa
        duration = librosa.get_duration(path=audio_guide)
        combination_type = "add"
        clean_audio_files = "V" in audio_prompt_type
        if audio_guide2 is not None:
            duration2 = librosa.get_duration(path=audio_guide2)
            if clean_audio_files:  # isolate vocals first so normalization measures the voices, not the music
                audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, audio_guide, "_clean", ".wav"))
                audio_guide2 = get_vocals(original_audio_guide2, get_available_filename(save_path, audio_guide2, "_clean2", ".wav"))
                temp_filenames_list += [audio_guide, audio_guide2]
                if audio_guide3 is not None:
                    audio_guide3 = get_vocals(original_audio_guide3, get_available_filename(save_path, audio_guide3, "_clean3", ".wav"))
                    temp_filenames_list.append(audio_guide3)
            if "N" in audio_prompt_type:
                max_total_duration = model_def.get("audio_reference_max_total_duration", None)
                guides = [guide for guide in (audio_guide, audio_guide2, audio_guide3) if guide is not None]
                total_duration = duration + duration2 + (librosa.get_duration(path=audio_guide3) if audio_guide3 is not None else 0)
                normalization_max_duration = float(max_total_duration) / len(guides) if max_total_duration is not None and total_duration > float(max_total_duration) else None
                normalized_guides = normalize_audio_volumes_to_temp_files(guides, output_dir=save_path, prefix="audio_norm_", max_duration_seconds=normalization_max_duration)
                temp_filenames_list += normalized_guides
                audio_guide, audio_guide2, audio_guide3 = (normalized_guides + [None])[:3]
                if normalization_max_duration is not None:
                    duration, duration2 = min(duration, normalization_max_duration), min(duration2, normalization_max_duration)
            if "C" in audio_prompt_type: duration += duration2
            else: duration = min(duration, duration2)
            combination_type = "para" if "P" in audio_prompt_type else "add"
        else:
            if "X" in audio_prompt_type: 
                # dual speaker, voice separation
                from preprocessing.speaker_separator import extract_dual_audio
                combination_type = "para"
                if args.save_speakers:
                    audio_guide, audio_guide2  = "speaker1.wav", "speaker2.wav"
                else:
                    audio_guide, audio_guide2  = get_available_filename(save_path, audio_guide, "_tmp1", ".wav"),  get_available_filename(save_path, audio_guide, "_tmp2", ".wav")
                    temp_filenames_list +=   [audio_guide, audio_guide2]                  
                if clean_audio_files:
                    clean_audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, original_audio_guide, "_clean", ".wav"))
                    temp_filenames_list += [clean_audio_guide]
                extract_dual_audio(clean_audio_guide if clean_audio_files else original_audio_guide, audio_guide, audio_guide2)

            elif clean_audio_files:
                # Single Speaker
                audio_guide = get_vocals(original_audio_guide, get_available_filename(save_path, audio_guide, "_clean", ".wav"))
                temp_filenames_list += [audio_guide]

            output_new_audio_filepath = original_audio_guide
        video_length_limited_by_audio = not model_def.get("video_length_not_limited_by_audio", False) or control_video_trim 
        if "F" in audio_prompt_type:
            full_audio_guide_waveform, full_audio_guide_sample_rate = slice_audio_window(audio_guide, 0, max_source_video_frames, fps, save_path, suffix=f"_full", pad_head=False, pad_tail=False)
        elif video_length_limited_by_audio:
            current_video_length = min(normalize_frame_count(int(fps * duration) + latent_size, frames_minimum, latent_size, frames_offset), current_video_length)
        if fantasy:
            from models.wan.fantasytalking.infer import parse_audio
            # audio_proj_split_full, audio_context_lens_full = parse_audio(audio_guide, num_frames= max_source_video_frames, fps= fps,  padded_frames_for_embeddings= (reuse_frames if reset_control_aligment else 0), device= processing_device  )
            if audio_scale is None:
                audio_scale = 1.0
        elif multitalk:
            from models.wan.multitalk.multitalk import get_full_audio_embeddings
            # pad audio_proj_full if aligned to beginning of window to simulate source window overlap
            min_audio_duration =  current_video_length/fps if reset_control_aligment else video_source_duration + current_video_length/fps
            audio_proj_full, output_new_audio_data = get_full_audio_embeddings(audio_guide1 = audio_guide, audio_guide2= audio_guide2, combination_type= combination_type , num_frames= max_source_video_frames, sr= audio_sampling_rate, fps =fps, padded_frames_for_embeddings = (reuse_frames if reset_control_aligment else 0), min_audio_duration = min_audio_duration) 
            if output_new_audio_data is not None: # not none if modified
                if clean_audio_files: # need to rebuild the sum of audios with original audio
                    _, output_new_audio_data = get_full_audio_embeddings(audio_guide1 = original_audio_guide, audio_guide2= original_audio_guide2, combination_type= combination_type , num_frames= max_source_video_frames, sr= audio_sampling_rate, fps =fps, padded_frames_for_embeddings = (reuse_frames if reset_control_aligment else 0), min_audio_duration = min_audio_duration, return_sum_only= True) 
                output_new_audio_filepath=  None # need to build original speaker track if it changed size (due to padding at the end) or if it has been combined

    if hunyuan_custom_edit and video_guide != None:
        import cv2
        cap = cv2.VideoCapture(video_guide)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_video_length = min(current_video_length, length)

    scheduler_active = frame_scheduler is not None and frame_scheduler["active"]
    scheduled_windows_template = [dict(window) for window in frame_scheduler["windows"]] if scheduler_active else []
    scheduled_windows = [dict(window) for window in scheduled_windows_template]
    any_sliding_window = test_any_sliding_window(model_type)
    default_reuse_frames = min(sliding_window_size - latent_size, sliding_window_overlap) if any_sliding_window else 0
    if scheduler_active:
        sliding_window = True
        reuse_frames = default_reuse_frames
        current_video_length = scheduled_windows[0]["frame_num"]
    elif any_sliding_window:
        sliding_window = current_video_length > sliding_window_size
        reuse_frames = default_reuse_frames
    else:
        sliding_window = False
        sliding_window_size = current_video_length
        reuse_frames = 0
    if scheduler_active:
        default_windows_template = []
    elif any_sliding_window:
        first_window_overlap = min(max(1, default_reuse_frames), first_window_available_overlap) if video_source is not None else min(1, first_window_available_overlap)
        default_windows_template = build_default_window_plan(total_frames=current_video_length, window_size=sliding_window_size, default_overlap=default_reuse_frames, discard_last_frames=sliding_window_discard_last_frames, minimum=frames_minimum, step=frames_steps, frame_offset=frames_offset, overlap_offset=sliding_window_defaults.get("overlap_offset", 1), max_overlap=sliding_window_defaults.get("overlap_max"), first_window_overlap=first_window_overlap, first_window_available_overlap=first_window_available_overlap, initial_shared_frames=1 if first_window_overlap > 0 else 0, preserve_exact_output_frames=model_def.get("image_end_frame_position", False), output_frame_policy=model_def.get("frame_scheduler_output_policy"))
    else:
        default_windows_template = [{"output_frames": current_video_length, "overlap_frames": 0, "discard_last_frames": 0, "trim_last_frames": 0, "frame_num": current_video_length}]
    default_windows = [dict(window) for window in default_windows_template]
    if not scheduler_active:
        sliding_window = len(default_windows) > 1 or default_windows[0]["overlap_frames"] > 0
    seed = set_seed(seed)

    torch.set_grad_enabled(False) 
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(audio_save_path, exist_ok=True)
    gc.collect()
    torch.cuda.empty_cache()
    wan_model._interrupt = False
    abort = False
    if gen.get("abort", False):
        return True
    # gen["abort"] = False
    gen["prompt"] = prompt    
    repeat_no = 0
    extra_generation = 0
    discard_last_frames = sliding_window_discard_last_frames
    default_discard_last_frames = discard_last_frames
    default_requested_frames_to_generate = current_video_length if scheduler_active else sum(window["output_frames"] for window in default_windows)
    nb_frames_positions = 0
    if scheduler_active:
        default_requested_frames_to_generate = frame_scheduler["predicted_total_frames"]
        current_video_length = scheduled_windows[0]["frame_num"]
    else:
        current_video_length = default_windows[0]["frame_num"]

    first_window_video_length = current_video_length
    original_prompts = display_original_prompts.copy()
    original_alt_prompts = display_original_alt_prompts.copy()
    gen["sliding_window"] = sliding_window 
    while not abort: 
        stop_current_sample = False
        with gen_lock:
            extra_generation += gen.get("extra_orders",0)
            gen["extra_orders"] = 0
        total_generation = repeat_generation + extra_generation
        gen["total_generation"] = total_generation     
        gen["header_text"] = ""    
        if repeat_no >= total_generation: break
        repeat_no +=1
        gen["repeat_no"] = repeat_no
        gen_state = {}
        if scheduler_active:
            scheduled_windows = [dict(window) for window in scheduled_windows_template]
        else:
            default_windows = [dict(window) for window in default_windows_template]
        src_video = src_video2 = src_video3 = src_mask = src_mask2 = src_faces = sparse_video_image = full_generated_audio =None
        prefix_video = pre_video_frame = None
        source_video_overlap_frames_count = 0 # number of frames overalapped in source video for first window
        source_video_frames_count = 0  # number of frames to use in source video (processing starts source_video_overlap_frames_count frames before )
        frames_already_processed = []
        frames_already_processed_count = 0
        retained_video_frames = 0  # output timeline at generation FPS, before temporal upsampling
        external_audio_trim_ranges = []
        overlapped_latents = None
        pre_video_guide_is_hdr = False
        context_scale = None
        window_no = 0
        end_frame_no = 0
        extra_windows = 0
        gen_cache = {}
        stop_sample_scheduled = False
        guide_start_frame = 0 # pos of of first control video frame of current window  (reuse_frames later than the first processed frame)
        keep_frames_parsed = [] # aligned to the first control frame of current window (therefore ignore previous reuse_frames)
        pre_video_guide = None # reuse_frames of previous window
        pre_audio_guide, pre_audio_guide_sample_rate = None, 0
        image_size = default_image_size #  default frame dimensions for budget until it is change due to a resize
        sample_fit_canvas = fit_canvas
        current_video_length = first_window_video_length
        gen["total_windows"] = 1
        gen["window_no"] = 1
        input_waveform, input_waveform_sample_rate = None, 0
        num_frames_generated = 0 # num of new frames created (lower than the number of frames really processed due to overlaps and discards)
        requested_frames_to_generate = default_requested_frames_to_generate # num  of num frames to create (if any source window this num includes also the overlapped source window frames)
        cached_video_guide_processed = cached_video_mask_processed = cached_video_guide_processed2 = cached_video_mask_processed2 = cached_video_guide_processed3 = None
        cached_video_video_start_frame = cached_video_video_end_frame = -1
        start_time = time.time()
        prompt_was_enhanced = prompt_has_history
        alt_prompt_was_enhanced = alt_prompt_has_history
        if auto_prompt_enhancer_requested:
            send_cmd("progress", [0, get_latest_status(state, "Enhancing Prompt")])
            enhancer_kwargs = {"image_prompt_type":  image_prompt_type, "video_prompt_type":  video_prompt_type, "audio_prompt_type":  audio_prompt_type}
            enhancer_kwargs["control_image"] = video_guide if is_image and "V" in video_prompt_type else None
            enhancer_kwargs["generation_callbacks"] = {"stop_requested": lambda: gen.get("abort", False), "enhancement_progress": EnhancementProgress(lambda value, desc, total, unit: send_cmd("progress", [value, get_latest_status(state, desc), total, unit]))}
            if model_def.get("prompt_enhancer_video_duration", False) and not is_image and not audio_only:
                frames = min(video_length, sliding_window_size) if model_def.get("sliding_window", False) else video_length
                enhancer_kwargs["video_duration_seconds"] = frames / fps
            multi_prompt_output = prompt_enhancer_outputs_multiple_prompts(prompt_enhancer)
            prompts_to_enhance = original_prompts[:1] if multi_prompt_output else original_prompts
            routed = prompt_enhancer_chaining.uses_routing(prompt_enhancer)
            try:
                continuation_image = None
                enhancer_windows = scheduled_windows if scheduler_active else default_windows
                if "I" in prompt_enhancer and not is_image and not audio_only and image_start is None and video_source is not None and not enhancer_windows[0].get("new_shot", False):
                    send_cmd("progress", [0, get_latest_status(state, "Reading Continuation Start Image")])
                    continuation_image = prompt_enhancer_images.continuation_start_image(video_source, keep_frames_video_source, fps, max_source_video_frames)
                enhancer_start = image_start if image_start is not None else continuation_image
                enhancer_kwargs["caption_input_info"] = prompt_enhancer_images.caption_input_info(enhancer_start, image_end, original_image_refs, enhancer_kwargs["control_image"], "K" in video_prompt_type)
                if "W" in prompt_enhancer:
                    from shared.prompt_enhancer.transcription import transcribe_source_audio

                    offloadobj.unload_all()
                    enhancer_kwargs["source_audio_transcript"] = transcribe_source_audio(audio_guide if "A" in audio_prompt_type else None, gen=gen, report_status=lambda text: send_cmd("progress", [0, get_latest_status(state, text)]), duration_seconds=duration_seconds if model_def.get("prompt_enhancer_transcription_use_duration", False) else None)
                    if enhancer_kwargs["source_audio_transcript"] is None:
                        prompt_enhancer = prompt_enhancer.replace("W", "")
                ensure_prompt_enhancer_loaded(override_profile=override_profile, send_cmd=send_cmd, gen=gen)
                if prompt_enhancer_images.enabled(server_config, prompt_enhancer) or has_slash_commands(original_prompts):
                    original_prompts, enhancer_kwargs["image_contexts"] = prompt_enhancer_images.prepare_auto(original_prompts, None if fake_start_image else image_start, image_end, original_image_refs, windows=scheduled_windows if scheduler_active else default_windows, fps=fps, window_size=sliding_window_size, positions=frames_positions, source_frames=first_window_available_overlap, reset_alignment=reset_control_aligment, model_def=model_def, discard_frames=sliding_window_discard_last_frames, reuse_frames=default_reuse_frames, multi_prompt_output=multi_prompt_output, video=not is_image and not audio_only, video_prompt_type=video_prompt_type, control_image=enhancer_kwargs["control_image"], continuation_image=continuation_image)
                    prompts_to_enhance = original_prompts[:1] if multi_prompt_output else original_prompts
                if routed:
                    enhanced = prompt_enhancer_chaining.run_chain(prompt_enhancer, original_prompts, original_alt_prompts, alt_prompt_inherits, lambda step, values: process_prompt_enhancer(model_type, model_def, step.mode, values, enhancer_start if enhancer_start is not None else image_end, original_image_refs, is_image, audio_only, seed, enhancer_kwargs=enhancer_kwargs, batch_images=True))
                else:
                    enhanced = process_prompt_enhancer(model_type, model_def, prompt_enhancer, prompts_to_enhance, enhancer_start if enhancer_start is not None else image_end, original_image_refs, is_image, audio_only, seed, enhancer_kwargs=enhancer_kwargs)
            finally:
                unload_prompt_enhancer_runtime()
                if enhancer_offloadobj is not None:
                    enhancer_offloadobj.unload_all()
            if enhanced is not None:
                if routed:
                    if enhanced.prompt_changed:
                        prompts = [normalize_generated_prompt_lines(value, multi_prompts_gen_type, multi_prompt_output=multi_prompt_output) for value in enhanced.prompts]
                        task["prompt"] = prompt_parser.ENHANCED_PROMPT_PREFIX + prompt_parser.serialize_prompt_units("", prompts, multi_prompts_gen_type)
                        prompt_was_enhanced = True
                    if enhanced.alt_prompt_changed:
                        alt_prompts = [normalize_generated_prompt_lines(value, multi_prompts_gen_type) for value in enhanced.alt_prompts]
                        task["alt_prompt"] = prompt_parser.serialize_prompt_blocks_with_prefix(alt_prompts, prompt_enhancer_chaining.originals_for_outputs(original_alt_prompts, len(alt_prompts)))
                        alt_prompt_was_enhanced = True
                    print(f"Enhanced prompts: {prompts}; enhanced alt prompts: {alt_prompts}")
                else:
                    enhanced_prompts = enhanced
                    print(f"Enhanced prompts: {enhanced_prompts}" )
                    if multi_prompt_output:
                        enhanced_prompt = normalize_generated_prompt_lines(enhanced_prompts[0], multi_prompts_gen_type, multi_prompt_output=True)
                        enhanced_prompts = prompt_parser.split_prompt_units(enhanced_prompt, multi_prompts_gen_type)
                    else:
                        enhanced_prompts = [normalize_generated_prompt_lines(one_prompt, multi_prompts_gen_type) for one_prompt in enhanced_prompts]
                    # On-the-fly enhancement keeps task prompts clean; originals are saved in metadata.
                    task["prompt"] = prompt_parser.ENHANCED_PROMPT_PREFIX + prompt_parser.serialize_prompt_units("", enhanced_prompts, multi_prompts_gen_type)
                    prompts = enhanced_prompts
                    prompt_was_enhanced = True
                gen["last_was_audio"] = audio_only
                send_cmd("output")
                if scheduler_active:
                    for idx, window in enumerate(scheduled_windows):
                        window["prompt"] = prompt_enhancer_images.strip_window_commands([prompts[idx] if idx < len(prompts) else prompts[-1]])[0][0]
                    frame_scheduler["prompts"] = [window["prompt"] for window in scheduled_windows]
                abort = gen.get("abort", False)

 
        while not abort and not stop_current_sample:
            with gen_lock:
                new_extra_windows = gen.get("extra_windows",0)
                gen["extra_windows"] = 0
            extra_windows += new_extra_windows
            if scheduler_active:
                for _ in range(new_extra_windows):
                    scheduled_windows.append(build_extension_window(scheduled_windows[-1]["prompt"], window_size=sliding_window_size, overlap_frames=default_reuse_frames, discard_last_frames=default_discard_last_frames, minimum=frames_minimum, step=frames_steps, frame_offset=frames_offset, preserve_exact_output_frames=model_def.get("image_end_frame_position", False), output_frame_policy=model_def.get("frame_scheduler_output_policy")))
                    requested_frames_to_generate += scheduled_windows[-1]["output_frames"]
                if window_no >= len(scheduled_windows):
                    break
                frame_window_options = scheduled_windows[window_no]
                prompt, reuse_frames, current_video_length, new_shot, discard_last_frames = frame_window_options["prompt"], frame_window_options["overlap_frames"], frame_window_options["frame_num"], frame_window_options["new_shot"], frame_window_options["discard_last_frames"]
                automatic_trim_last_frames = frame_window_options["trim_last_frames"]
                if "requested_output_frames" in frame_window_options and abs(frame_window_options["requested_output_frames"] - frame_window_options["output_frames"]) > 1:
                    print(f"Requested frame contribution adjusted from {frame_window_options['requested_output_frames']} to {frame_window_options['output_frames']} for model-compatible scheduling (Sliding Window {window_no + 1}).")
                current_loras_slists = frame_window_options.get("loras_slists", loras_slists)
                sliding_window = True
            else:
                frame_window_options, current_loras_slists, new_shot, discard_last_frames = None, loras_slists, False, default_discard_last_frames
                for _ in range(new_extra_windows):
                    default_windows.append(build_extension_window("", window_size=sliding_window_size, overlap_frames=default_reuse_frames, discard_last_frames=default_discard_last_frames, minimum=frames_minimum, step=frames_steps, frame_offset=frames_offset, preserve_exact_output_frames=model_def.get("image_end_frame_position", False), output_frame_policy=model_def.get("frame_scheduler_output_policy")))
                    requested_frames_to_generate += default_windows[-1]["output_frames"]
                if window_no >= len(default_windows):
                    break
                default_window = default_windows[window_no]
                reuse_frames, current_video_length, discard_last_frames = default_window["overlap_frames"], default_window["frame_num"], default_window["discard_last_frames"]
                automatic_trim_last_frames = default_window["trim_last_frames"]
                if "requested_output_frames" in default_window and abs(default_window["requested_output_frames"] - default_window["output_frames"]) > 1:
                    print(f"Requested frame contribution adjusted from {default_window['requested_output_frames']} to {default_window['output_frames']} for model-compatible scheduling (Sliding Window {window_no + 1}).")
                prompt =  prompts[window_no] if window_no < len(prompts) else prompts[-1]
                sliding_window = len(default_windows) > 1 or reuse_frames > 0
            gen["sliding_window"] = sliding_window
            current_alt_prompt = alt_prompts[window_no] if window_no < len(alt_prompts) else alt_prompts[-1]
            if scheduler_active:
                next_overlap_frames = scheduled_windows[window_no + 1]["overlap_frames"] if window_no + 1 < len(scheduled_windows) else default_reuse_frames
            else:
                next_overlap_frames = default_windows[window_no + 1]["overlap_frames"] if window_no + 1 < len(default_windows) else default_reuse_frames

            total_windows = len(scheduled_windows) if scheduler_active else len(default_windows)
            gen["total_windows"] = total_windows
            if window_no >= total_windows:
                break
            window_no += 1
            gen["window_no"] = window_no
            enable_RIFLEx = model_def.get("riflex", False) and image_mode == 0 and (RIFLEx_setting == 0 and current_video_length > (6 * fps + 1) or RIFLEx_setting == 1)
            return_latent_slice = None 
            frames_relative_positions_list = []
            tail_trim_frames = discard_last_frames + automatic_trim_last_frames
            if reuse_frames > 0:
                tail_trim_latents = tail_trim_frames // latent_size
                return_latent_slice = slice(- max(1, (reuse_frames + tail_trim_frames) // latent_size), None if tail_trim_latents == 0 else -tail_trim_latents)
            refresh_preview  = {"image_guide" : image_guide, "image_mask" : image_mask} if image_mode >= 1 else {}
            if new_shot:
                pre_video_guide, pre_audio_guide, pre_audio_guide_sample_rate = None, None, 0

            if hasattr(model_handler, "custom_prompt_preprocess"):
                prompt = model_handler.custom_prompt_preprocess(**locals())
            image_start_tensor = image_end_tensor = None
            if window_no == 1 and (video_source is not None or (image_start is not None and not new_shot)):
                if image_start is not None:
                    image_start_tensor, new_height, new_width = calculate_dimensions_and_resize_image(image_start, height, width, sample_fit_canvas, fit_crop, block_size = block_size)
                    if fit_crop: refresh_preview["image_start"] = image_start_tensor 
                    image_start_tensor = convert_image_to_tensor(image_start_tensor)
                    pre_video_guide =  prefix_video = image_start_tensor.unsqueeze(1)
                else:
                    prefix_video_is_hdr = "&" in video_prompt_type and _video_input_is_hdr(video_source)
                    prefix_video  = preprocess_video(width=width, height=height,video_in=video_source, max_frames= parsed_keep_frames_video_source , start_frame = 0, fit_canvas= sample_fit_canvas, fit_crop = fit_crop, target_fps = fps, block_size = block_size, preserve_hdr=prefix_video_is_hdr )
                    prefix_video  = prefix_video.permute(3, 0, 1, 2)

                    if fit_crop or "L" in image_prompt_type: refresh_preview["video_source"] = convert_tensor_to_image(prefix_video, 0) 

                    new_height, new_width = prefix_video.shape[-2:]
                    source_overlap = 0 if new_shot else min(prefix_video.shape[1], reuse_frames if reuse_frames > 0 else 1)
                    if source_overlap > 0:
                        pre_video_guide = prefix_video[:, -source_overlap:].float()
                        if prefix_video_is_hdr:
                            pre_video_guide_is_hdr = True
                        else:
                            pre_video_guide = pre_video_guide.div_(127.5).sub_(1.) # c, f, h, w
                pre_video_frame = convert_tensor_to_image(prefix_video[:, -1])
                source_video_overlap_frames_count = source_overlap if video_source is not None else pre_video_guide.shape[1]
                source_video_frames_count = prefix_video.shape[1]

                if sample_fit_canvas != None: 
                    image_size = (pre_video_guide if pre_video_guide is not None else prefix_video).shape[-2:]
                    sample_fit_canvas = None
                guide_start_frame =  prefix_video.shape[1]
                if fake_start_image:
                    source_video_overlap_frames_count = source_video_frames_count = guide_start_frame = 0
            if image_end is not None and not (frame_window_options is not None and frame_window_options.get("no_end_image", False)):
                image_end_list=  image_end if isinstance(image_end, list) else [image_end]
                if end_frame_no < len(image_end_list):
                    new_height, new_width = image_size                    
                    image_end_tensor, _, _ = calculate_dimensions_and_resize_image(image_end_list[end_frame_no], new_height, new_width, sample_fit_canvas, fit_crop, block_size = block_size)
                    end_frame_no += 1
                    refresh_preview["image_end"] = image_end_tensor 
                    image_end_tensor = convert_image_to_tensor(image_end_tensor)
                    if sample_fit_canvas != None: 
                        image_size  = image_end_tensor.shape[-2:]
                        sample_fit_canvas = None
                image_end_list= None
            image_end_frame_position = current_video_length - tail_trim_frames - 1 if model_def.get("image_end_frame_position", False) else None  # also locates an injected frame used as the window's end image
            window_start_frame = guide_start_frame - (reuse_frames if window_no > 1 else source_video_overlap_frames_count)
            guide_end_frame = guide_start_frame + current_video_length - (source_video_overlap_frames_count if window_no == 1 else reuse_frames)
            alignment_shift = source_video_frames_count if reset_control_aligment else 0
            aligned_guide_start_frame = guide_start_frame - alignment_shift
            aligned_guide_end_frame = guide_end_frame - alignment_shift
            aligned_window_start_frame = window_start_frame - alignment_shift  
            input_waveform, input_waveform_sample_rate = None, 0
            if full_audio_guide_waveform is not None:
                input_waveform, input_waveform_sample_rate = full_audio_guide_waveform, full_audio_guide_sample_rate
            elif audio_guide is not None and model_def.get("audio_guide_window_slicing", False):
                audio_start_frame = aligned_window_start_frame
                # if reset_control_aligment:
                #     audio_start_frame += source_video_overlap_frames_count
                input_waveform, input_waveform_sample_rate = slice_audio_window(audio_guide, audio_start_frame, current_video_length, fps, save_path, suffix=f"_win{window_no}", pad_tail= video_length_limited_by_audio) 
                if input_waveform.shape[0] == 0: input_waveform, input_waveform_sample_rate = pre_audio_guide, pre_audio_guide_sample_rate
            elif model_def.get("audio_guide_window_slicing", False):
                if pre_audio_guide is not None and pre_audio_guide.shape[0] > 0:
                    input_waveform, input_waveform_sample_rate = pre_audio_guide, pre_audio_guide_sample_rate
                elif window_no == 1 and source_video_overlap_frames_count > 0 and len(source_audio_tracks) > 0:
                    source_audio_start_frame = max(0, source_video_frames_count - source_video_overlap_frames_count)
                    input_waveform, input_waveform_sample_rate = slice_audio_window(source_audio_tracks[0], source_audio_start_frame, source_video_overlap_frames_count, fps, save_path, suffix=f"_source_overlap_win{window_no}", pad_head=False, pad_tail=False)
                    if input_waveform.shape[0] == 0: input_waveform, input_waveform_sample_rate = None, 0
            if fantasy and audio_guide is not None:
                audio_proj_split , audio_context_lens = parse_audio(audio_guide, start_frame = aligned_window_start_frame, num_frames= current_video_length, fps= fps,  device= processing_device  )
            if multitalk:
                from models.wan.multitalk.multitalk import get_window_audio_embeddings
                # special treatment for start frame pos when alignement to first frame requested as otherwise the start frame number will be negative due to overlapped frames (has been previously compensated later with padding)
                audio_proj_split = get_window_audio_embeddings(audio_proj_full, audio_start_idx= aligned_window_start_frame + (source_video_overlap_frames_count if reset_control_aligment else 0 ), clip_length = current_video_length)

            if repeat_no == 1 and window_no == 1 and image_refs is not None and len(image_refs) > 0:
                frames_positions_list = prompt_enhancer_images.resolve_injected_positions(frames_positions, len(image_refs), windows=scheduled_windows if scheduler_active else default_windows, source_frames=source_video_frames_count, source_overlap=source_video_overlap_frames_count, window_size=sliding_window_size, discard_frames=sliding_window_discard_last_frames, reuse_frames=default_reuse_frames, alignment_shift=alignment_shift)
                nb_frames_positions = len(frames_positions_list) 
                if nb_frames_positions > 0:
                    frames_to_inject = [None] * (max(frames_positions_list) + 1)
                    for i, pos in enumerate(frames_positions_list):
                        frames_to_inject[pos] = image_refs[i] 

            video_guide_processed = video_mask_processed = video_guide_processed2 = video_mask_processed2 = video_guide_processed3 = sparse_video_image = None
            skip_video_guide_preprocess = model_def.get("skip_video_guide_preprocess", False)  # True, or video_prompt_type letters that make the model read the guide itself
            skip_video_guide_preprocess = any_letters(video_prompt_type, skip_video_guide_preprocess) if isinstance(skip_video_guide_preprocess, str) else bool(skip_video_guide_preprocess)
            if video_guide is not None and not skip_video_guide_preprocess:
                guide_frames_limit = round(model_def["reference_video_max_frames"] * fps / model_def["fps"]) if reference_videos else max(aligned_guide_end_frame, source_video_frames_count - source_video_overlap_frames_count + requested_frames_to_generate)
                keep_frames_parsed_full, error = parse_keep_frames_video_guide(keep_frames_video_guide, guide_frames_limit)
                if len(error) > 0:
                    raise gr.Error(f"invalid keep frames {keep_frames_video_guide}")
                guide_frames_extract_start = 0 if reference_videos else aligned_window_start_frame if extract_guide_from_window_start else aligned_guide_start_frame
                extra_control_frames = model_def.get("extra_control_frames", 0)
                if extra_control_frames > 0 and aligned_guide_start_frame >= extra_control_frames: guide_frames_extract_start -= extra_control_frames
                        
                keep_frames_parsed = [True] * -guide_frames_extract_start if guide_frames_extract_start  <0 else []
                keep_frames_parsed += keep_frames_parsed_full[max(0, guide_frames_extract_start): guide_frames_limit if reference_videos else aligned_guide_end_frame]
                guide_frames_extract_count = len(keep_frames_parsed)

                process_all = resolve_model_preprocess_all(model_def, base_model_type=base_model_type, video_prompt_type=video_prompt_type, image_prompt_type=image_prompt_type, audio_prompt_type=audio_prompt_type, custom_settings=custom_settings)
                if process_all:
                    guide_slice_to_extract  = guide_frames_extract_count
                    guide_frames_extract_count = (-guide_frames_extract_start if guide_frames_extract_start  <0 else 0) +  len( keep_frames_parsed_full[max(0, guide_frames_extract_start):] )

                # Extract Faces to video
                if "B" in video_prompt_type:
                    send_cmd("progress", [0, get_latest_status(state, "Extracting Face Movements")])
                    src_faces = extract_faces_from_video_with_mask(video_guide, video_mask, max_frames= guide_frames_extract_count, start_frame= guide_frames_extract_start, size= 512, target_fps = fps)
                    if src_faces is not None and src_faces.shape[1] < current_video_length:
                        src_faces = torch.cat([src_faces, torch.full( (3, current_video_length - src_faces.shape[1], 512, 512 ), -1, dtype = src_faces.dtype, device= src_faces.device) ], dim=1)

                # Sparse Video to Video
                sparse_video_image = None
                if "R" in video_prompt_type:
                    sparse_video_image = get_video_frame(video_guide, aligned_guide_start_frame, return_last_if_missing = True, target_fps = fps, return_PIL = True)

                if not process_all or cached_video_video_start_frame < 0:
                    # Generic Video Preprocessing
                    process_outside_mask = process_map_outside_mask.get(filter_letters(video_prompt_type, "YWX"), None)
                    preprocess_type, preprocess_type2 =  "raw", None 
                    for process_num, process_letter in enumerate( filter_letters(video_prompt_type, video_guide_processes)):
                        if process_num == 0:
                            preprocess_type = process_map_video_guide.get(process_letter, "raw")
                        else:
                            preprocess_type2 = process_map_video_guide.get(process_letter, None)
                    if reference_videos and preprocess_type == "identity":
                        preprocess_type = "raw"
                    custom_preprocessor = model_def.get("custom_preprocessor", None) 
                    if custom_preprocessor is not None:
                        status_info = custom_preprocessor
                        send_cmd("progress", [0, get_latest_status(state, status_info)])
                        video_guide_processed, video_guide_processed2, video_mask_processed, video_mask_processed2 =  custom_preprocess_video_with_mask(model_handler, base_model_type, pre_video_guide, video_guide if sparse_video_image is None else sparse_video_image, video_mask, height=image_size[0], width = image_size[1], max_frames= guide_frames_extract_count, start_frame = guide_frames_extract_start, fit_canvas = sample_fit_canvas, fit_crop = fit_crop, target_fps = fps,  block_size = block_size, expand_scale = mask_expand, video_prompt_type= video_prompt_type, model_def=model_def, custom_settings=custom_settings)
                    else:
                        status_info = "Extracting " + processes_names[preprocess_type]
                        extra_process_list = ([] if preprocess_type2==None else [preprocess_type2]) + ([] if process_outside_mask==None or process_outside_mask == preprocess_type else [process_outside_mask])
                        if len(extra_process_list) == 1:
                            status_info += " and " + processes_names[extra_process_list[0]]
                        elif len(extra_process_list) == 2:
                            status_info +=  ", " + processes_names[extra_process_list[0]] + " and " + processes_names[extra_process_list[1]]
                        context_scale = [control_net_weight /2, control_net_weight2 /2] if preprocess_type2 is not None else [control_net_weight]

                        if not (preprocess_type == "identity" and preprocess_type2 is None and video_mask is None):send_cmd("progress", [0, get_latest_status(state, status_info)])
                        inpaint_color = 0 if "pose" in preprocess_type and process_outside_mask == "inpaint" else guide_inpaint_color
                        if "O" in video_prompt_type and pre_video_guide is None and all_letters(video_prompt_type, "IK"):
                            from shared.utils.utils import get_outpainting_full_area_dimensions
                            w, h = image_refs[0].size
                            if outpainting_dims != None:
                                h, w = get_outpainting_full_area_dimensions(h, w, outpainting_dims, video_guide_outpainting_ratio)
                            image_size = calculate_new_dimensions(height, width, h, w, fit_canvas, block_size=block_size)
                            sample_fit_canvas = None 
                            ref_pose_tensor  = resize_and_remove_background(image_refs[nb_frames_positions:nb_frames_positions+1], image_size[1], image_size[0],
                                                                                            False, True, 
                                                                                            fit_into_canvas= model_def.get("fit_into_canvas_image_refs", 1),
                                                                                            block_size=block_size,
                                                                                            outpainting_dims =outpainting_dims,
                                                                                            outpainting_ratio = video_guide_outpainting_ratio,
                                                                                            background_ref_outpainted = model_def.get("background_ref_outpainted", True),
                                                                                            outpainting_quantize_margins = outpainting_quantize_margins,
                                                                                            return_tensor= True)[0][0]
                        else:
                            ref_pose_tensor = pre_video_guide
 
                        guide_source = video_guide if sparse_video_image is None else sparse_video_image
                        guide_height, guide_width = image_size
                        guide_fit_canvas = sample_fit_canvas
                        if reference_videos:
                            guide_height, guide_width = get_reference_video_dimensions(guide_source, *image_size, model_def["reference_video_max_size"], block_size)
                            guide_fit_canvas = None
                        video_guide_processed, video_mask_processed = preprocess_video_with_mask(ref_pose_tensor, guide_source, video_mask, height=guide_height, width=guide_width, max_frames=guide_frames_extract_count, start_frame=guide_frames_extract_start, fit_canvas=guide_fit_canvas, fit_crop=fit_crop, target_fps=fps, process_type=preprocess_type, expand_scale=mask_expand, RGB_Mask=True, negate_mask="N" in video_prompt_type, process_outside_mask=process_outside_mask, outpainting_dims=outpainting_dims, outpainting_ratio=video_guide_outpainting_ratio, proc_no=1, inpaint_color=inpaint_color, block_size=block_size, to_bbox="H" in video_prompt_type, outpainting_quantize_margins=outpainting_quantize_margins)
                        if preprocess_video_guide2 and "+" in video_prompt_type and video_guide2 is not None:
                            guide2_height, guide2_width = image_size
                            guide2_fit_canvas = sample_fit_canvas
                            if reference_videos:
                                guide2_height, guide2_width = get_reference_video_dimensions(video_guide2, *image_size, model_def["reference_video_max_size"], block_size)
                                guide2_fit_canvas = None
                            video_guide_processed2, video_mask_processed2 = preprocess_video_with_mask(None, video_guide2, None, height=guide2_height, width=guide2_width, max_frames=guide_frames_extract_count, start_frame=guide_frames_extract_start, fit_canvas=guide2_fit_canvas, fit_crop=fit_crop, target_fps=fps, process_type=preprocess_type, proc_no=2, block_size=block_size)
                        elif preprocess_type2 != None:
                            video_guide_processed2, video_mask_processed2 = preprocess_video_with_mask(ref_pose_tensor, video_guide, video_mask, height=image_size[0], width = image_size[1], max_frames= guide_frames_extract_count, start_frame = guide_frames_extract_start, fit_canvas = sample_fit_canvas, fit_crop = fit_crop, target_fps = fps,  process_type = preprocess_type2, expand_scale = mask_expand, RGB_Mask = True, negate_mask = "N" in video_prompt_type, process_outside_mask = process_outside_mask, outpainting_dims = outpainting_dims, outpainting_ratio = video_guide_outpainting_ratio, proc_no =2, block_size = block_size, to_bbox = "H" in video_prompt_type, outpainting_quantize_margins = outpainting_quantize_margins)
                        if preprocess_video_guide2 and "*" in video_prompt_type and video_guide3 is not None:
                            guide3_height, guide3_width = get_reference_video_dimensions(video_guide3, *image_size, model_def["reference_video_max_size"], block_size) if reference_videos else image_size
                            video_guide_processed3, _ = preprocess_video_with_mask(None, video_guide3, None, height=guide3_height, width=guide3_width, max_frames=guide_frames_extract_count, start_frame=guide_frames_extract_start, fit_canvas=None if reference_videos else sample_fit_canvas, fit_crop=fit_crop, target_fps=fps, process_type=preprocess_type, proc_no=3, block_size=block_size)

                    if video_guide_processed is not None and sample_fit_canvas is not None and not reference_videos:
                        image_size = video_guide_processed.shape[-2:]
                        sample_fit_canvas = None

                    if process_all:
                        cached_video_guide_processed, cached_video_mask_processed, cached_video_guide_processed2, cached_video_mask_processed2, cached_video_guide_processed3 = video_guide_processed, video_mask_processed, video_guide_processed2, video_mask_processed2, video_guide_processed3
                        cached_video_video_start_frame = guide_frames_extract_start

                if process_all:
                    process_slice = slice(guide_frames_extract_start - cached_video_video_start_frame, guide_frames_extract_start - cached_video_video_start_frame + guide_slice_to_extract  )
                    video_guide_processed = None if cached_video_guide_processed is None else cached_video_guide_processed[:, process_slice] 
                    video_mask_processed =  None if cached_video_mask_processed is None else cached_video_mask_processed[:, process_slice] 
                    video_guide_processed2 =  None if cached_video_guide_processed2 is None else cached_video_guide_processed2[:, process_slice] 
                    video_mask_processed2 = None if cached_video_mask_processed2 is None else cached_video_mask_processed2[:, process_slice] 
                    video_guide_processed3 = None if cached_video_guide_processed3 is None else cached_video_guide_processed3[:, process_slice]
                    
            if window_no == 1 and image_refs is not None and len(image_refs) > 0:
                ignored_image_refs = model_def.get("no_processing_on_last_images_refs", 0)
                if sample_fit_canvas is not None and (nb_frames_positions > 0 or "K" in video_prompt_type) :
                    from shared.utils.utils import get_outpainting_full_area_dimensions
                    w, h = image_refs[0].size
                    if outpainting_dims != None:
                        h, w = get_outpainting_full_area_dimensions(h, w, outpainting_dims, video_guide_outpainting_ratio)
                    image_size = calculate_new_dimensions(height, width, h, w, fit_canvas, block_size=block_size)
                sample_fit_canvas = None
                if repeat_no == 1:
                    if fit_crop:
                        if any_background_ref == 2:
                            end_ref_position = len(image_refs)
                        elif any_background_ref == 1:
                            end_ref_position = nb_frames_positions + 1
                        else:
                            end_ref_position = nb_frames_positions 
                        end_ref_position = min(end_ref_position, max(nb_frames_positions, len(image_refs) - ignored_image_refs))
                        for i, img in enumerate(image_refs[:end_ref_position]):
                            image_refs[i] = rescale_and_crop(img, default_image_size[1], default_image_size[0])
                        refresh_preview["image_refs"] = image_refs

                    if len(image_refs) > nb_frames_positions:
                        src_ref_images = image_refs[nb_frames_positions:]
                        if "Q" in video_prompt_type:
                            from preprocessing.arc.face_encoder import FaceEncoderArcFace, get_landmarks_from_image
                            image_pil = src_ref_images[-1]
                            face_encoder = FaceEncoderArcFace()
                            face_encoder.init_encoder_model(processing_device)
                            face_arc_embeds = face_encoder(image_pil, need_proc=True, landmarks=get_landmarks_from_image(image_pil))
                            face_arc_embeds = face_arc_embeds.squeeze(0).cpu()
                            face_encoder = image_pil = None
                            gc.collect()
                            torch.cuda.empty_cache()

                        if remove_background_images_ref > 0:
                            send_cmd("progress", [0, get_latest_status(state, "Removing Images References Background")])

                        background_removal_color = model_def.get("background_removal_color", [255, 255, 255])
                        if callable(background_removal_color):
                            background_removal_color = background_removal_color(video_prompt_type, custom_settings if isinstance(custom_settings, dict) else {})
                        background_removal_color = parse_guide_inpaint_color(background_removal_color)
                        background_removal_color = list(background_removal_color) if isinstance(background_removal_color, tuple) else background_removal_color
                        src_ref_images, src_ref_masks  = resize_and_remove_background(src_ref_images , image_size[1], image_size[0],
                                                                                        remove_background_images_ref > 0, any_background_ref, 
                                                                                        fit_into_canvas= model_def.get("fit_into_canvas_image_refs", 1),
                                                                                        block_size=block_size,
                                                                                        outpainting_dims =outpainting_dims,
                                                                                        outpainting_ratio = video_guide_outpainting_ratio,
                                                                                        background_ref_outpainted = model_def.get("background_ref_outpainted", True),
                                                                                        outpainting_quantize_margins = outpainting_quantize_margins,
                                                                                        return_tensor= model_def.get("return_image_refs_tensor", False),
                                                                                        ignore_last_refs=ignored_image_refs,
                                                                                        background_removal_color = background_removal_color)
            custom_postprocessor = model_def.get("custom_image_ref_postprocessor", None)
            if window_no == 1 and repeat_no == 1 and custom_postprocessor is not None:
                src_ref_images, src_ref_masks = custom_postprocessor(
                    src_ref_images, src_ref_masks, image_size[1], image_size[0], image_start, image_prompt_type, image_end, video_prompt_type,
                    send_cmd, model_def, custom_settings, image_start_tensor=image_start_tensor, pre_video_frame=pre_video_frame
                )

            frames_to_inject_parsed = frames_to_inject[ window_start_frame if extract_guide_from_window_start else guide_start_frame: guide_end_frame]
            if (video_guide is not None and not skip_video_guide_preprocess) or len(frames_to_inject_parsed) > 0 and not custom_frames_injection or model_def.get("forced_guide_mask_inputs", False):
                any_mask = video_mask is not None or model_def.get("forced_guide_mask_inputs", False)
                any_guide_padding = model_def.get("pad_guide_video", False)
                dont_cat_preguide = extract_guide_from_window_start or model_def.get("dont_cat_preguide", False) or sparse_video_image is not None 
                from shared.utils.utils import prepare_video_guide_and_mask
                src_videos, src_masks = prepare_video_guide_and_mask(   [video_guide_processed] + ([] if video_guide_processed2 is None else [video_guide_processed2]) + ([] if video_guide_processed3 is None else [video_guide_processed3]),
                                                                        [video_mask_processed] + ([] if video_guide_processed2 is None else [video_mask_processed2]) + ([] if video_guide_processed3 is None else [None]),
                                                                        None if dont_cat_preguide or fake_start_image and window_no==1 else pre_video_guide, 
                                                                        image_size, current_video_length, latent_size,
                                                                        any_mask, any_guide_padding, guide_inpaint_color, 
                                                                        keep_frames_parsed, [] if custom_frames_injection else frames_to_inject_parsed , outpainting_dims, video_guide_outpainting_ratio, outpainting_quantize_margins=outpainting_quantize_margins, frame_offset=frames_offset)
                video_guide_processed = video_guide_processed2 = video_guide_processed3 = video_mask_processed = video_mask_processed2 = None
                src_video, src_video2, src_video3 = (src_videos + [None, None])[:3]
                src_mask, src_mask2 = (src_masks + [None])[:2] 
                src_videos = src_masks = None
                if control_video_trim:
                    if src_video is None:
                        abort = True 
                        break
                    elif window_no >1 and src_video.shape[1] <= sliding_window_overlap and not dont_cat_preguide:
                        abort = True 
                        break
                    elif src_video.shape[1] < current_video_length:
                        current_video_length = src_video.shape[1]
                        stop_sample_scheduled = True 
                if src_faces is not None:
                    if src_faces.shape[1] < src_video.shape[1]:
                        src_faces = torch.concat( [src_faces,  src_faces[:, -1:].repeat(1, src_video.shape[1] - src_faces.shape[1], 1,1)], dim =1)
                    else:
                        src_faces = src_faces[:, :src_video.shape[1]]
                if (video_guide is not None and not skip_video_guide_preprocess) or len(frames_to_inject_parsed) > 0:
                    if args.save_masks:
                        if src_video is not None: 
                            save_video( src_video, "masked_frames.mp4", fps)
                            if any_mask: save_video( src_mask, "masks.mp4", fps, value_range=(0, 1))
                        if src_video2 is not None: 
                            save_video( src_video2, "masked_frames2.mp4", fps)
                            if any_mask: save_video( src_mask2, "masks2.mp4", fps, value_range=(0, 1))
                if video_guide is not None and not skip_video_guide_preprocess:
                    preview_frame_no = 0 if extract_guide_from_window_start or model_def.get("dont_cat_preguide", False) or sparse_video_image is not None else (guide_start_frame - window_start_frame) 
                    if src_video is not None:
                        preview_frame_no = min(src_video.shape[1] -1, preview_frame_no)
                        refresh_preview["video_guide"] = convert_tensor_to_image(src_video, preview_frame_no)
                    if src_video2 is not None and not model_def.get("no_guide2_refresh", False):
                        refresh_preview["video_guide"] = [refresh_preview["video_guide"], convert_tensor_to_image(src_video2, preview_frame_no)] 
                    if src_mask is not None and video_mask is not None and not model_def.get("no_mask_refresh", False):                        
                        refresh_preview["video_mask"] = convert_tensor_to_image(src_mask, preview_frame_no, mask_levels = True)

            if src_ref_images is not None or nb_frames_positions:
                if len(frames_to_inject_parsed):
                    if custom_frames_injection:
                        frames_relative_positions_list= [ frame_no + (0 if extract_guide_from_window_start else (aligned_guide_start_frame - aligned_window_start_frame)) for frame_no, frame in enumerate(frames_to_inject_parsed) if frame is not None]
                        frames_to_inject_parsed = new_image_refs = [frame for frame in frames_to_inject_parsed if frame is not None]
                    else:
                        new_image_refs = [convert_tensor_to_image(src_video, frame_no + (0 if extract_guide_from_window_start else (aligned_guide_start_frame - aligned_window_start_frame)) ) for frame_no, inject in enumerate(frames_to_inject_parsed) if inject]
                else:
                    new_image_refs = []
                if src_ref_images is not None:
                    new_image_refs +=  [convert_tensor_to_image(img) if torch.is_tensor(img) else img for img in src_ref_images  ]
                refresh_preview["image_refs"] = new_image_refs
                new_image_refs = None

            if len(refresh_preview) > 0:
                new_inputs= locals()
                new_inputs.update(refresh_preview)
                update_task_thumbnails(task, new_inputs)
                send_cmd("output")

            if window_no ==  1:                
                conditioning_latents_size = ( (source_video_overlap_frames_count-1) // latent_size) + 1 if source_video_overlap_frames_count > 0 else 0
            else:
                conditioning_latents_size = ( (reuse_frames-1) // latent_size) + 1

            status = get_latest_status(state)
            gen["progress_status"] = status
            progress_phase = "Generating Audio" if audio_only else "Encoding Prompt"
            gen["progress_phase"] = (progress_phase , -1 )
            tiny_preview = None
            if offloadobj.tiny_vae is not None:
                from shared.tinyvae.session import PreviewSession
                tiny_preview = PreviewSession(offloadobj.tiny_vae, send_cmd, gen, is_image, video=server_config.get("generation_preview", "rgb") == "tiny_vae_video", duration=floor_frame_count(current_video_length, frames_minimum, latent_size, frames_offset) / fps)
            callback = build_callback(state, trans, send_cmd, status, num_inference_steps, preview_meta={"first_latent_only": not model_def.get("preview_all_images", False)} if is_image else None, tiny_preview=tiny_preview)
            progress_args = [0, merge_status_context(status, progress_phase )]
            send_cmd("progress", progress_args)

            if skip_steps_cache !=  None:
                skip_steps_cache.update({
                "num_steps" : num_inference_steps,                
                "skipped_steps" : 0,
                "previous_residual": None,
                "previous_modulated_input":  None,
                })
            # samples = torch.empty( (1,2)) #for testing
            # if False:
            def set_header_text(txt):
                gen["header_text"] = txt
                send_cmd("output")

            try:
                input_video_for_model = None if new_shot else pre_video_guide
                input_video_is_hdr = pre_video_guide_is_hdr
                prefix_frames_count = 0 if new_shot else source_video_overlap_frames_count if window_no <= 1 else reuse_frames
                prefix_video_for_model = None if model_def.get("joyai_echo", False) or str(base_model_type).startswith("ltx2") else prefix_video
                if new_shot:
                    prefix_video_for_model = None
                if prefix_video_for_model is not None and prefix_video_for_model.dtype == torch.uint8:
                    prefix_video_for_model = prefix_video_for_model.float().div_(127.5).sub_(1.0)
                if window_no <= 1 and video_source is not None and "&" in video_prompt_type and _video_input_is_hdr(video_source):
                    input_video_is_hdr = True
                custom_settings_for_model = custom_settings if isinstance(custom_settings, dict) else {}
                model_outpainting_dims = outpainting_dims
                if outpainting_dims is not None and len((video_guide_outpainting_ratio or "").strip()) > 0:
                    if isinstance(video_guide, Image.Image):
                        control_source_width, control_source_height = video_guide.size
                    elif video_guide is not None:
                        _, control_source_width, control_source_height, _ = get_video_info(video_guide)
                    else:
                        control_source_width = control_source_height = None
                    if control_source_height is not None and control_source_width is not None:
                        model_outpainting_dims = resolve_outpainting_dims(control_source_height, control_source_width, outpainting_dims, video_guide_outpainting_ratio)
                overridden_inputs = None
                if vae_upsampler_handler is not None and vae_upsampler_session is None:
                    vae_upsampler_session = upsampler_api.prepare_vae_upsampler(vae_upsampler_handler, spatial_upsampling, send_cmd=send_cmd, process_files=process_files_def, init_pipe=init_pipe, profile=compute_profile(override_profile, upsampler_api.profile_type_for_handler(vae_upsampler_handler)), attention_mode=attention_mode, spatial_upsampler_param=spatial_upsampler_param, spatial_upsampler_param2=spatial_upsampler_param2, spatial_upsampler_parameters=spatial_upsampler_parameters)
                samples = wan_model.generate(
                    input_prompt = prompt,
                    alt_prompt = current_alt_prompt,
                    image_start = image_start_tensor,  
                    image_end = image_end_tensor,
                    image_end_frame_position=image_end_frame_position,
                    input_frames = src_video,
                    input_frames2 = src_video2,
                    input_frames3 = src_video3,
                    input_ref_images=  src_ref_images,
                    input_ref_masks = src_ref_masks,
                    input_masks = src_mask,
                    input_masks2 = src_mask2,
                    input_video= input_video_for_model,
                    input_faces = src_faces,
                    input_custom = custom_guide,
                    video_guide= video_guide,
                    video_guide2=video_guide2,
                    video_guide3=video_guide3,
                    denoising_strength=denoising_strength,
                    masking_strength=masking_strength,
                    prefix_frames_count = prefix_frames_count,
                    frame_num=floor_frame_count(current_video_length, frames_minimum, latent_size, frames_offset),
                    batch_size = batch_size,
                    height = image_size[0],
                    width = image_size[1],
                    fit_into_canvas = fit_canvas,
                    shift=flow_shift,
                    sample_solver=sample_solver,
                    sampling_steps=num_inference_steps,
                    guide_scale=guidance_scale,
                    guide2_scale = guidance2_scale,
                    guide3_scale = guidance3_scale,
                    switch_threshold = switch_threshold, 
                    switch2_threshold = switch_threshold2,
                    guide_phases= guidance_phases,
                    model_switch_phase = model_switch_phase,
                    embedded_guidance_scale=embedded_guidance_scale,
                    n_prompt=negative_prompt,
                    seed=seed,
                    callback=callback,
                    enable_RIFLEx = enable_RIFLEx,
                    VAE_tile_size = VAE_tile_size,
                    joint_pass = joint_pass,
                    perturbation_switch = perturbation_switch,
                    perturbation_layers = perturbation_layers,
                    perturbation_start = perturbation_start_perc/100,
                    perturbation_end = perturbation_end_perc/100,
                    apg_switch = apg_switch,
                    cfg_star_switch = cfg_star_switch,
                    cfg_zero_step = cfg_zero_step,
                    alt_guide_scale= alt_guidance_scale,
                    audio_cfg_scale= audio_guidance_scale,
                    input_waveform=input_waveform, 
                    input_waveform_sample_rate=input_waveform_sample_rate,
                    audio_guide=audio_guide,
                    audio_guide2=audio_guide2,
                    audio_guide3=audio_guide3,
                    audio_prompt_type=audio_prompt_type,
                    audio_proj= audio_proj_split,
                    audio_scale= audio_scale,
                    audio_context_lens= audio_context_lens,
                    context_scale = context_scale,
                    control_scale_alt = control_net_weight_alt,
                    alt_scale = alt_scale,
                    motion_amplitude = motion_amplitude,
                    model_mode = model_mode,
                    causal_block_size = 5,
                    causal_attention = True,
                    fps = fps,
                    overlapped_latents = overlapped_latents,
                    return_latent_slice= return_latent_slice,
                    overlap_noise = sliding_window_overlap_noise,
                    overlap_size = reuse_frames,
                    sub_parallel_window_size=sub_parallel_window_size,
                    sub_parallel_window_overlap=sub_parallel_window_overlap,
                    color_correction_strength = sliding_window_color_correction_strength,
                    conditioning_latents_size = conditioning_latents_size,
                    input_video_is_hdr=input_video_is_hdr,
                    lora_dir=lora_dir,
                    keep_frames_parsed = keep_frames_parsed,
                    model_filename = model_filename,
                    model_type = base_model_type,
                    loras_slists = current_loras_slists,
                    NAG_scale = NAG_scale,
                    NAG_tau = NAG_tau,
                    NAG_alpha = NAG_alpha,
                    attention_sparsity = attention_sparsity,
                    speakers_bboxes =speakers_bboxes,
                    image_mode =  image_mode,
                    video_prompt_type= video_prompt_type,
                    window_no = window_no, 
                    offloadobj = offloadobj,
                    set_header_text= set_header_text,
                    pre_video_frame = pre_video_frame,
                    prefix_video = prefix_video_for_model,
                    original_input_ref_images = original_image_refs[nb_frames_positions:] if original_image_refs is not None else [],
                    image_refs_relative_size = image_refs_relative_size,
                    outpainting_dims = model_outpainting_dims,
                    face_arc_embeds = face_arc_embeds,
                    custom_settings=custom_settings_for_model,
                    frame_window_options=frame_window_options,
                    gen_state=gen_state,
                    temperature=temperature,
                    window_start_frame_no = window_start_frame,
                    input_video_strength = input_video_strength,
                    self_refiner_setting = self_refiner_setting,
                    self_refiner_plan=self_refiner_plan,
                    self_refiner_f_uncertainty = self_refiner_f_uncertainty,
                    self_refiner_certain_percentage = self_refiner_certain_percentage,
                    duration_seconds=duration_seconds,
                    pause_seconds=pause_seconds,
                    top_p=top_p,
                    top_k=top_k,
                    set_progress_status=set_progress_status,
                    loras_selected=loras_selected,
                    frames_relative_positions_list = frames_relative_positions_list,
                    frames_to_inject = frames_to_inject_parsed,
                    verbose_level=verbose_level,
                    gen_cache=gen_cache,
                    vae_upsampler=vae_upsampler_session,
                    save_masks=args.save_masks,
                )
                if tiny_preview is not None:
                    tiny_preview.close()
                upsampler_api.release_vae_upsampler(vae_upsampler_handler, vae_upsampler_session)
                vae_upsampler_session = None
            except Exception as e:
                if tiny_preview is not None:
                    tiny_preview.close(cancel=True)
                upsampler_api.release_vae_upsampler(vae_upsampler_handler, vae_upsampler_session)
                vae_upsampler_session = None
                if len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0:
                    cleanup_temp_audio_files(control_audio_tracks + source_audio_tracks)
                remove_temp_filenames(temp_filenames_list)
                gen_state = plugin_data = None
                clear_gen_cache()
                offloadobj.unload_all()
                trans.cache = None 
                if trans2 is not None: 
                    trans2.cache = None 
                offload.unload_loras_from_model(trans_lora)
                if trans2_lora is not None: 
                    offload.unload_loras_from_model(trans2_lora)
                skip_steps_cache = None
                # if compile:
                #     cache_size = torch._dynamo.config.cache_size_limit                                      
                #     torch.compiler.reset()
                #     torch._dynamo.config.cache_size_limit = cache_size

                gc.collect()
                torch.cuda.empty_cache()
                if isinstance(e, (DownloadCancelled, DownloadError, MediaProcessingAborted)):
                    raise
                s = str(e)
                keyword_list = {"CUDA out of memory" : "VRAM", "Tried to allocate":"VRAM", "CUDA error: out of memory": "RAM", "CUDA error: too many resources requested": "RAM"}
                crash_type = ""
                for keyword, tp  in keyword_list.items():
                    if keyword in s:
                        crash_type = tp 
                        break
                state["prompt"] = ""
                if crash_type == "VRAM":
                    new_error = "The generation of the video has encountered an error: it is likely that you have unsufficient VRAM and you should therefore reduce the video resolution or its number of frames."
                elif crash_type == "RAM":
                    new_error = "The generation of the video has encountered an error: it is likely that you have unsufficient RAM and / or Reserved RAM allocation should be reduced using 'perc_reserved_mem_max' or using a different Profile."
                else:
                    new_error =  gr.Error(f"The generation of the video has encountered an error, please check your terminal for more information. '{s}'")
                tb = traceback.format_exc().split('\n')[:-1] 
                print('\n'.join(tb))
                send_cmd("error", new_error)
                clear_status(state)
                return False
            src_video = src_video2 = src_video3 = src_mask = src_mask2 = None
            if skip_steps_cache != None :
                skip_steps_cache.previous_residual = None
                skip_steps_cache.previous_modulated_input = None
                print(f"Skipped Steps:{skip_steps_cache.skipped_steps}/{skip_steps_cache.num_steps}" )
            generated_audio = None
            drop_generated_audio = False
            side_files = {}
            post_decode_pre_trim = 0
            output_audio_sampling_rate= audio_sampling_rate
            sample_is_hdr = False
            if samples != None:
                if isinstance(samples, dict):
                    sample_is_hdr = samples.get("hdr", False)
                    overlapped_latents = samples.get("latent_slice", None)
                    side_files = samples.get("side_files", {})
                    generated_audio = samples.get("audio", generated_audio)
                    overridden_inputs = samples.get("overridden_inputs", None)
                    output_audio_sampling_rate = samples.get("audio_sampling_rate", audio_sampling_rate)
                    input_fills_window = input_waveform is not None and input_waveform.shape[0] >= int(round(current_video_length * input_waveform_sample_rate / fps))
                    keep_input_audio = (model_def.get("output_audio_is_input_audio", False) or "S" in audio_prompt_type) and output_new_audio_filepath is not None and "O" not in audio_prompt_type
                    if generated_audio is not None:
                        if keep_input_audio and input_fills_window:
                            drop_generated_audio = True
                        elif input_fills_window:
                            output_new_audio_filepath = None
                    post_decode_pre_trim = samples.get("post_decode_pre_trim", 0) 
                    samples = samples.get("x", None)

                if samples is not None:
                    samples = samples.to("cpu")
  
            clear_gen_cache()
            offloadobj.unload_all()
            gc.collect()
            torch.cuda.empty_cache()

            if samples == None:
                abort = True
                state["prompt"] = ""
                send_cmd("output")  
            else:
                sample = samples.cpu()
                samples = None
                stop_current_sample = stop_sample_scheduled or (not (is_image or audio_only) and sample.shape[1] < current_video_length)
                # if True: # for testing
                #     torch.save(sample, "output.pt")
                # else:
                #     sample =torch.load("output.pt")
                if post_decode_pre_trim > 0 :
                    sample = sample[:, post_decode_pre_trim:]
                if gen.get("extra_windows",0) > 0:
                    sliding_window = True 
                if sliding_window :
                    guide_start_frame += current_video_length
                    if discard_last_frames > 0:
                        sample = sample[: , :-discard_last_frames]
                        guide_start_frame -= discard_last_frames
                        if generated_audio is not None:
                            generated_audio = truncate_audio(generated_audio, 0, discard_last_frames, fps, output_audio_sampling_rate)
                    if automatic_trim_last_frames > 0:
                        frame_label = "frame" if automatic_trim_last_frames == 1 else "frames"
                        print(f"{automatic_trim_last_frames} {frame_label} trimmed to match expected duration (Sliding Window {window_no}).")
                        sample = sample[:, :-automatic_trim_last_frames]
                        guide_start_frame -= automatic_trim_last_frames
                        if generated_audio is not None:
                            generated_audio = truncate_audio(generated_audio, 0, automatic_trim_last_frames, fps, output_audio_sampling_rate)
                    if generated_audio is not None and next_overlap_frames > 0 and not drop_generated_audio:
                        pre_audio_guide = generated_audio[-int(round(next_overlap_frames * output_audio_sampling_rate / fps)):]
                        pre_audio_guide_sample_rate = output_audio_sampling_rate
                    else:
                        pre_audio_guide, pre_audio_guide_sample_rate = None, 0

                    pre_video_guide = sample[:, -next_overlap_frames:].clone() if next_overlap_frames > 0 else None if scheduler_active else sample[:, max_source_video_frames:].clone()
                    pre_video_guide_is_hdr = sample_is_hdr
                    if pre_video_guide is not None and pre_video_guide.dtype == torch.uint8:
                        pre_video_guide =  pre_video_guide.float().div_(127.5).sub_(1.0)
                    window_overlap_frames = source_video_overlap_frames_count if window_no == 1 else reuse_frames
                    trim_first_frames = min(sliding_window_trim_first_frames, max(0, sample.shape[1] - 1)) if window_overlap_frames == 0 else 0
                    if trim_first_frames > 0:
                        audio_trim_start_frame = retained_video_frames + (prefix_video.shape[1] if prefix_video is not None and window_no == 1 else 0)
                        external_audio_trim_ranges.append((audio_trim_start_frame, trim_first_frames))
                        sample = sample[:, trim_first_frames:]
                        if generated_audio is not None:
                            generated_audio = truncate_audio(generated_audio, trim_first_frames, 0, fps, output_audio_sampling_rate)
                if not (audio_only or is_image):
                    if not sample_is_hdr:
                        sample = convert_video_tensor_to_uint8_chunked(sample)

                if prefix_video != None and window_no == 1 :
                    if sample_is_hdr:
                        prefix_was_uint8 = prefix_video.dtype == torch.uint8
                        prefix_video = prefix_video.float()
                        if prefix_was_uint8:
                            prefix_video = prefix_video.div_(255.0)
                        elif not input_video_is_hdr and torch.is_floating_point(prefix_video):
                            prefix_video = prefix_video.add_(1.0).mul_(0.5).clamp_(0.0, 1.0)
                        prefix_video = prefix_video.to(dtype=sample.dtype)
                    elif prefix_video.dtype != sample.dtype:
                        if sample.dtype == torch.uint8:
                            prefix_video = convert_video_tensor_to_uint8_chunked(prefix_video)
                        elif prefix_video.dtype == torch.uint8:
                            prefix_video = prefix_video.float().div_(127.5).sub_(1.0)
                    if prefix_video.shape[1] > 1:
                        # remove sliding window overlapped frames at the beginning of the generation
                        sample = torch.cat([ prefix_video, sample[: , source_video_overlap_frames_count:]], dim = 1)
                    else:
                        # remove source video overlapped frames at the beginning of the generation if there is only a start frame
                        sample = torch.cat([ prefix_video[:, :-source_video_overlap_frames_count], sample], dim = 1)
                    prefix_video = None
                    guide_start_frame -= source_video_overlap_frames_count 
                    if generated_audio is not None:
                        generated_audio = truncate_audio( generated_audio, 0 if video_source is None else source_video_overlap_frames_count, 0, fps, output_audio_sampling_rate,)
                elif sliding_window and window_no > 1 and reuse_frames > 0:
                    # remove sliding window overlapped frames at the beginning of the generation
                    sample = sample[: , reuse_frames:]
                    guide_start_frame -= reuse_frames 
                    if generated_audio is not None:
                        generated_audio = truncate_audio( generated_audio, reuse_frames, 0, fps, output_audio_sampling_rate,)

                num_frames_generated = guide_start_frame - (source_video_frames_count - source_video_overlap_frames_count) 
                if drop_generated_audio: generated_audio = None
                if generated_audio is not None:
                    # generated audio starts after the Source Video frames (its overlap frames were already removed from the sample)
                    source_audio_frames = source_video_frames_count if video_source is not None else 0
                    committed_audio_samples = int(round(max(0, retained_video_frames - source_audio_frames) * output_audio_sampling_rate / fps))
                    if full_generated_audio is None:
                        # a kept soundtrack file is aligned with the start of the Source Video unless control alignment is reset: skip the source part
                        prefix_start_samples = 0 if reset_control_aligment or output_new_audio_data is not None else int(round(source_audio_frames * output_audio_sampling_rate / fps))
                        full_generated_audio = append_sliding_window_audio(output_new_audio_data, output_new_audio_filepath, generated_audio, output_audio_sampling_rate, committed_audio_samples,
                                                                           prefix_start_samples=prefix_start_samples, keep_existing=keep_input_audio and output_new_audio_data is None,
                                                                           trim_ranges=shift_audio_trim_ranges(external_audio_trim_ranges, source_audio_frames), video_fps=fps) if output_new_audio_data is not None or output_new_audio_filepath is not None else generated_audio
                    else:
                        full_generated_audio = np.concatenate([full_generated_audio, generated_audio], axis=0)
                    output_new_audio_data = full_generated_audio
                if not (audio_only or is_image):
                    retained_video_frames += sample.shape[1]


                if len(temporal_upsampling) > 0 or len(spatial_upsampling) > 0 and (not upsampler_api.is_vae_upsampling(spatial_upsampling) or upsampler_api.has_post_model_process_vae_upsampling(spatial_upsampling)):
                    spatial_status = upsampler_api.method_progress_label(spatial_upsampling)
                    send_cmd("progress", [0, merge_status_context(status, spatial_status)])
                
                output_fps  = fps
                def upsampler_progress(phase, current_step=None, total_steps=None):
                    phase_text = str(phase)
                    gen["progress_phase"] = (phase_text, int(current_step) if current_step is not None else -1)
                    status_msg = merge_status_context(status, phase_text)
                    if current_step is not None and total_steps is not None and int(total_steps) > 0:
                        send_cmd("progress", [(int(current_step), int(total_steps)), status_msg, int(total_steps)])
                    else:
                        send_cmd("progress", [0, status_msg])
                if len(temporal_upsampling) > 0:
                    sample, previous_last_frame, output_fps = perform_temporal_upsampling(sample, previous_last_frame if sliding_window and window_no > 1 else None, temporal_upsampling, fps, abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
                    if gen.get("abort", False) or sample is None:
                        abort = True
                        break

                if len(spatial_upsampling) > 0:
                    if is_image:
                        sample = perform_image_spatial_upsampling(sample, spatial_upsampling, seed=seed, vae_tile_size=VAE_tile_size, fps=output_fps, prompt=prompt, negative_prompt=negative_prompt, spatial_upsampler_prompt=spatial_upsampler_prompt, spatial_upsampler_reference_images=spatial_upsampler_reference_images, spatial_upsampler_param=spatial_upsampler_param, spatial_upsampler_param2=spatial_upsampler_param2, spatial_upsampler_parameters=spatial_upsampler_parameters, abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
                        flashvsr_continue_cache = None
                    else:
                        late_upsampler = upsampler_api.find_postprocessing_upsampler(spatial_upsampling)
                        upsampler_uses_audio = late_upsampler is not None and late_upsampler.query_upsampler_def().get("source_audio_conditioning", False)
                        upsampler_audio = full_generated_audio if upsampler_uses_audio and full_generated_audio is not None else input_waveform if upsampler_uses_audio else None
                        upsampler_audio_sample_rate = (output_audio_sampling_rate if full_generated_audio is not None else input_waveform_sample_rate) if upsampler_uses_audio else 0
                        if upsampler_audio is not None and upsampler_audio_sample_rate > 0:
                            required_audio_samples = int(round(sample.shape[1] * upsampler_audio_sample_rate / output_fps))
                            if upsampler_audio.shape[0] < required_audio_samples:
                                upsampler_audio = np.pad(upsampler_audio, ((required_audio_samples - upsampler_audio.shape[0], 0), (0, 0))) if upsampler_audio.ndim == 2 else np.pad(upsampler_audio, (required_audio_samples - upsampler_audio.shape[0], 0))
                            else:
                                upsampler_audio = upsampler_audio[-required_audio_samples:]
                        upsampler_reference_images = original_image_refs[nb_frames_positions:] if original_image_refs is not None else None
                        sample = perform_spatial_upsampling(sample, spatial_upsampling, seed=seed, flashvsr_continue_cache=flashvsr_continue_cache, return_flashvsr_continue_cache=return_flashvsr_continue_cache, vae_tile_size=VAE_tile_size, fps=output_fps, prompt=prompt, negative_prompt=negative_prompt, audio_waveform=upsampler_audio, audio_sample_rate=upsampler_audio_sample_rate, reference_images=upsampler_reference_images, image_refs_relative_size=image_refs_relative_size, spatial_upsampler_prompt=spatial_upsampler_prompt, spatial_upsampler_reference_images=spatial_upsampler_reference_images, spatial_upsampler_param=spatial_upsampler_param, spatial_upsampler_param2=spatial_upsampler_param2, spatial_upsampler_parameters=spatial_upsampler_parameters, abort_callback=lambda: media_abort_requested(gen), progress_callback=upsampler_progress)
                    if return_flashvsr_continue_cache and not is_image:
                        sample, flashvsr_continue_cache = sample
                    if gen.get("abort", False) or sample is None:
                        abort = True
                        break
                if film_grain_intensity> 0:
                    from postprocessing.film_grain import add_film_grain
                    sample = add_film_grain(sample, film_grain_intensity, film_grain_saturation) 
                if audio_only:
                    output_video_frames = None
                    output_frame_count = None
                    any_soundtrack_processor = False
                elif is_image:
                    output_video_frames = sample.detach().cpu().clone().contiguous() if api_return_video_uint8 else None
                    output_frame_count = None
                    any_soundtrack_processor = False
                else:
                    frames_already_processed.append(sample)
                    frames_already_processed_count += sample.shape[1]
                    output_video_frames = frames_already_processed
                    output_frame_count = frames_already_processed_count
                    sample = None
                    soundtrack_error = audio_processor_api.validate_method(postprocess_audio, audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK, frames_count=output_frame_count, fps=fps, audio_source=audio_source, media_source_exists=media_source_exists, has_audio_file_extension=has_audio_file_extension) if audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"] else ""
                    any_soundtrack_processor = bool(postprocess_audio and not soundtrack_error and audio_processor_api.AUDIO_PROCESSOR_TYPE_SOUNDTRACK in postprocess_audio_meta["types"])
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%Hh%Mm%Ss")
                save_prompt = original_prompts[0]
                if audio_only:
                    audio_codec = server_config.get("audio_stand_alone_output_codec", "wav")
                    extension = get_audio_codec_extension(audio_codec)
                    output_dir = audio_save_path
                elif is_image:
                    extension = get_image_format(server_config.get("image_output_codec", None), rgba=sample.shape[0] == 4)["ext"][1:]
                    output_dir = image_save_path
                else:
                    container = server_config.get("video_container", "mp4")
                    extension = container
                    output_dir = save_path
                inputs = get_function_arguments(generate_media, locals())
                if overridden_inputs is not None: inputs.update(overridden_inputs)
                if scheduler_active and output_frame_count is not None:
                    inputs["video_length"] = output_frame_count
                if len(output_filename):
                    from shared.utils.filename_formatter import FilenameFormatter
                    file_name = FilenameFormatter.format_filename(output_filename, inputs)                    
                    file_name = f"{sanitize_file_name(truncate_for_filesystem(os.path.splitext(os.path.basename(file_name))[0])).strip()}.{extension}"
                    file_name = os.path.basename(get_available_filename(output_dir, file_name))
                else:
                    file_name = f"{time_flag}_seed{seed}_{sanitize_file_name(truncate_for_filesystem(save_prompt)).strip()}.{extension}"
                video_path = os.path.join(output_dir, file_name)
                saving_status = f"Saving File {file_name if len(file_name) <= 50 else file_name[:12] + '...' + file_name[-35:]}"

                if not is_image:
                    set_progress_status(saving_status)
                if audio_only:
                    audio_path = os.path.join(output_dir, file_name)
                    audio_path = save_audio_file(audio_path, sample.squeeze(0), output_audio_sampling_rate, audio_codec)
                    video_path = audio_path
                elif is_image:
                    image_path = os.path.join(output_dir, file_name)
                    sample =  sample.transpose(1,0)  #c f h w -> f c h w 
                    new_image_path = []
                    for no, img in enumerate(sample):  
                        img_path = get_available_filename(output_dir, image_path, "" if no == 0 else f"_{no}")
                        img_name = os.path.basename(img_path)
                        set_progress_status(f"Saving File {img_name if len(img_name) <= 50 else img_name[:12] + '...' + img_name[-35:]}")
                        new_image_path.append(save_image(img, save_file = img_path, quality = server_config.get("image_output_codec", None)))

                    video_path= new_image_path
                elif len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0 or output_new_audio_filepath is not None or any_soundtrack_processor or output_new_audio_data is not None or audio_source is not None:
                    video_path = os.path.join(save_path, file_name)
                    save_path_tmp = video_path.rsplit('.', 1)[0] + f"_tmp.{container}"
                    if sample_is_hdr:
                        save_hdr_video(tensor=output_video_frames, save_file=save_path_tmp, fps=output_fps, codec_type=server_config.get("hdr_video_crf", 8), container=container)
                    else:
                        save_video( tensor=output_video_frames, save_file=save_path_tmp, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1), codec_type = server_config.get("video_output_codec", None), container=container)
                    output_new_audio_temp_filepath = None
                    new_audio_tracks_are_pretrimmed = False
                    new_audio_added_from_audio_start = reset_control_aligment or (full_generated_audio is not None and len(control_audio_tracks) == 0) # if not beginning of audio will be skipped
                    source_audio_duration = 0 if video_source is None else source_video_frames_count / fps
                    new_audio_trim_ranges = shift_audio_trim_ranges(external_audio_trim_ranges, source_video_frames_count if new_audio_added_from_audio_start else 0)
                    if any_soundtrack_processor:
                        def audio_progress(phase, current_step=None, total_steps=None):
                            phase_text = f"Audio - {phase}"
                            gen["progress_phase"] = (phase_text, int(current_step) if current_step is not None else -1)
                            status_msg = get_latest_status(state, phase_text)
                            if current_step is not None and total_steps is not None and int(total_steps) > 0:
                                send_cmd("progress", [(int(current_step), int(total_steps)), status_msg, int(total_steps)])
                            else:
                                send_cmd("progress", [0, status_msg])
                        requested_audio_path = get_available_filename(save_path, f"tmp{time_flag}.wav" )
                        output_new_audio_filepath = audio_processor_api.generate_soundtrack(
                            postprocess_audio,
                            video_path=save_path_tmp,
                            audio_source=audio_source,
                            prompt=postprocess_audio_prompt,
                            negative_prompt=postprocess_audio_neg_prompt,
                            seed=seed,
                            duration=output_frame_count / fps,
                            output_path=requested_audio_path,
                            send_cmd=send_cmd,
                            status_callback=lambda status: send_cmd("progress", [0, get_latest_status(state, status)]),
                            verbose_level=verbose_level,
                            audio_codec_key=server_config.get("audio_output_codec", "aac_128"),
                            process_files=process_files_def,
                            init_pipe=init_pipe,
                            profile=server_config.get("audio_profile", 4),
                            abort_callback=lambda: media_abort_requested(gen),
                            progress_callback=audio_progress,
                        )
                        if gen.get("abort", False) or output_new_audio_filepath is None:
                            abort = True
                            break
                        if os.path.abspath(str(output_new_audio_filepath)) == os.path.abspath(str(requested_audio_path)): output_new_audio_temp_filepath = requested_audio_path
                        new_audio_added_from_audio_start = postprocess_audio_meta["needs_audio_source"]
                    elif audio_source is not None:
                        output_new_audio_filepath = audio_source
                        new_audio_added_from_audio_start =  True
                    elif output_new_audio_data is not None and len(control_audio_tracks) == 0:
                        output_new_audio_filepath = output_new_audio_temp_filepath = get_available_filename(save_path, f"tmp{time_flag}.wav" )
                        audio_data_to_write = output_new_audio_data if full_generated_audio is not None or len(new_audio_trim_ranges) == 0 else trim_audio_ranges(output_new_audio_data, new_audio_trim_ranges, fps, output_audio_sampling_rate)
                        new_audio_tracks_are_pretrimmed = full_generated_audio is not None or audio_data_to_write is not output_new_audio_data
                        write_wav_file(output_new_audio_filepath, audio_data_to_write, output_audio_sampling_rate)
                    if output_new_audio_filepath is not None:
                        new_audio_tracks = [output_new_audio_filepath]
                    else:
                        new_audio_tracks = control_audio_tracks
                    trimmed_mux_audio_tracks = []
                    if len(new_audio_trim_ranges) > 0 and len(new_audio_tracks) > 0 and not (any_soundtrack_processor or audio_source is not None or new_audio_tracks_are_pretrimmed):
                        new_audio_tracks = trimmed_mux_audio_tracks = [ trim_audio_file_ranges(path, new_audio_trim_ranges, fps, get_available_filename(save_path, path, suffix=f"_trim{i}", force_extension=".wav")) for i, path in enumerate(new_audio_tracks) ]
                    replace_voice_temp_audio_tracks = []
                    if replace_voice_method and replace_voice_sample is not None and len(new_audio_tracks) > 0:
                        new_audio_tracks, replace_voice_temp_audio_tracks = audio_processor_api.replace_voice_tracks(replace_voice_method, new_audio_tracks, voice_sample=replace_voice_sample, output_dir=save_path, prefix=f"tmp{time_flag}", process_files=process_files_def, profile_no=server_config.get("audio_profile", 4), verbose_level=verbose_level, init_pipe=init_pipe, voice_sample2=replace_voice_sample2, status_callback=lambda status: send_cmd("progress", [0, get_latest_status(state, status)]))
                    if generated_audio is not None: output_new_audio_filepath = None
                    mux_audio_sampling_rate = resolve_mux_audio_sampling_rate(output_audio_sampling_rate, source_audio_metadata, new_audio_tracks)

                    set_progress_status(saving_status)
                    combine_and_concatenate_video_with_audio_tracks(
                        video_path,
                        save_path_tmp,
                        source_audio_tracks,
                        new_audio_tracks,
                        source_audio_duration,
                        mux_audio_sampling_rate,
                        new_audio_from_start=new_audio_added_from_audio_start,
                        source_audio_metadata=source_audio_metadata,
                        audio_codec_key=server_config.get("audio_output_codec", "aac_128"),
                        verbose=verbose_level >= 2,
                    )
                    os.remove(save_path_tmp)
                    if output_new_audio_temp_filepath is not None: os.remove(output_new_audio_temp_filepath)
                    cleanup_temp_audio_files(trimmed_mux_audio_tracks)
                    cleanup_temp_audio_files(replace_voice_temp_audio_tracks)

                else:
                    if sample_is_hdr:
                        video_path = save_hdr_video(tensor=output_video_frames, save_file=video_path, fps=output_fps, codec_type=server_config.get("hdr_video_crf", 8), container=container)
                    else:
                        save_video( tensor=output_video_frames, save_file=video_path, fps=output_fps, nrow=1, normalize=True, value_range=(-1, 1),  codec_type= server_config.get("video_output_codec", None), container= container)

                end_time = time.time()

                inputs.pop("send_cmd")
                inputs.pop("task")
                inputs.pop("mode")
                inputs["model_type"] = model_type
                inputs["model_filename"] = get_model_filename(model_type, transformer_quantization, transformer_dtype_policy)
                if is_image:
                    inputs["image_quality"] = server_config.get("image_output_codec", None)
                else:
                    inputs["video_quality"] = server_config.get("video_output_codec", None)
                if sample_is_hdr:
                    inputs["hdr"] = True
                    inputs["hdr_video_crf"] = server_config.get("hdr_video_crf", 8)
                    inputs["video_quality"] = f"x265_crf_{inputs['hdr_video_crf']}"

                modules = get_model_recursive_prop(model_type, "modules", return_list= True)
                if len(modules) > 0 : inputs["modules"] = modules
                if len(transformer_loras_filenames) > 0:
                    inputs.update({
                    "transformer_loras_filenames" : transformer_loras_filenames,
                    "transformer_loras_multipliers" : transformer_loras_multipliers
                    })
                embedded_images = {img_name: inputs[img_name] for img_name in image_names_list } if server_config.get("embed_source_images", False) else None
                configs = prepare_inputs_dict("metadata", inputs, model_type)
                if replace_voice_method and replace_voice_sample is not None:
                    configs["replace_voice_method"] = replace_voice_method
                if sliding_window: configs["window_no"] = window_no
                configs.pop("prompt_enhancer", None)
                configs.pop("enhanced_prompt", None)
                configs.pop("enhanced_alt_prompt", None)
                configs["prompt"] = prompt_parser.serialize_prompt_blocks_with_prefix(prompts, prompt_enhancer_chaining.originals_for_outputs(original_prompts, len(prompts))) if prompt_was_enhanced else prompt_parser.serialize_prompt_units("", original_prompts, multi_prompts_gen_type)
                configs["alt_prompt"] = prompt_parser.serialize_prompt_blocks_with_prefix(alt_prompts, prompt_enhancer_chaining.originals_for_outputs(original_alt_prompts, len(alt_prompts))) if alt_prompt_was_enhanced else prompt_parser.serialize_prompt_units("", original_alt_prompts, alt_prompt_mode)
                configs["generation_time"] = round(end_time-start_time)
                configs["creation_date"] = datetime.fromtimestamp(end_time).isoformat(timespec="seconds")
                configs["creation_timestamp"] = int(end_time)
                # if sample_is_image: configs["is_image"] = True
                keep_intermediate_windows = server_config.get("keep_intermediate_sliding_windows", 1)
                record_file_metadata(video_path, configs, is_image, audio_only, gen, embedded_images=embedded_images, replace_last_file=sliding_window and window_no > 1 and not keep_intermediate_windows, notify_generation=not sliding_window or keep_intermediate_windows or window_no == total_windows)
                output_side_files = process_side_files(video_path[0] if isinstance(video_path, list) else video_path, side_files, return_files=api_return_side_files)
                if api_return_video_uint8 or api_return_audio or return_flashvsr_continue_cache or (api_return_side_files and output_side_files):
                    media_type = "audio" if audio_only else ("image" if is_image else "video")
                    artifact_audio = output_new_audio_data if api_return_audio else None
                    if artifact_audio is None and api_return_audio and audio_only and sample is not None:
                        artifact_audio = sample.squeeze(0).detach().cpu().float().numpy()
                    artifact_video = output_video_frames if api_return_video_uint8 else None
                    store_api_output_artifact(gen, client_id, video_path, media_type, artifact_video, artifact_audio, output_audio_sampling_rate, output_fps if not audio_only else None, hdr=bool(sample_is_hdr), flashvsr_continue_cache=flashvsr_continue_cache if return_flashvsr_continue_cache else None, side_files=output_side_files if api_return_side_files else None)
                side_files = output_side_files = None
                embedded_images = None
                # Play notification sound for single video
                try:
                    if server_config.get("notification_sound_enabled", 0):
                        volume = server_config.get("notification_sound_volume", 50)
                        notification_sound.notify_video_completion(
                            video_path=video_path, 
                            volume=volume
                        )
                except Exception as e:
                    print(f"Error playing notification sound for individual video: {e}")

                send_cmd("output")

        seed = set_seed(-1)
    gen_state = plugin_data = None
    clear_status(state)
    trans.cache = None
    offload.unload_loras_from_model(trans_lora)
    if not trans2_lora is None:
        offload.unload_loras_from_model(trans2_lora)

    if not trans2 is None:
       trans2.cache = None
 
    if len(control_audio_tracks) > 0 or len(source_audio_tracks) > 0:
        cleanup_temp_audio_files(control_audio_tracks + source_audio_tracks)

    remove_temp_filenames(temp_filenames_list)
    return True

def prepare_generate_media(state):

    if state.get("validate_success",0) != 1:
        return gr.Button(visible= True), gr.Button(visible= False), gr.Column(visible= False), gr.Button(visible=False), gr.Button(visible=False)
    else:
        if not get_gen_info(state).get("in_progress", False):
            clear_status(state)
        params = get_gen_info(state)["queue"][0]["params"]
        extend_visible = can_extend_sample(params)
        early_stop_visible = not _is_edit_task_params(params) and bool(get_model_def(params["model_type"]).get("supports_early_stop", False))
        return gr.Button(visible= False), gr.Button(visible= True), gr.Column(visible= True), gr.Button(visible=extend_visible), gr.Button(visible=early_stop_visible, interactive=True)


def generate_preview(model_type, payload):
    if isinstance(payload, (Image.Image, VideoPreview)):
        return payload
    import einops
    if payload is None:
        return None
    if isinstance(payload, dict):
        meta = {k: v for k, v in payload.items() if k != "latents"}
        latents = payload.get("latents")
    else:
        meta = {}
        latents = payload
    if latents is None:
        return None
    # latents shape should be C, T, H, W (no batch)    
    if not torch.is_tensor(latents):
        return None
    model_handler = get_model_handler(model_type)
    base_model_type = get_base_model_type(model_type)
    custom_preview = getattr(model_handler, "preview_latents", None)
    if callable(custom_preview):
        preview = custom_preview(base_model_type, latents, meta)
        if preview is not None:
            return preview
    if hasattr(model_handler, "get_rgb_factors"):
        latent_rgb_factors, latent_rgb_factors_bias = model_handler.get_rgb_factors(base_model_type )
    else:
        return None
    if latent_rgb_factors is None: return None
    latents = latents.unsqueeze(0) 
    nb_latents = latents.shape[2]
    latents_to_preview = 4
    latents_to_preview = min(nb_latents, latents_to_preview)
    skip_latent =  nb_latents / latents_to_preview
    latent_no = 0
    selected_latents = []
    while latent_no < nb_latents:
        selected_latents.append( latents[:, : , int(latent_no): int(latent_no)+1])
        latent_no += skip_latent 

    latents = torch.cat(selected_latents, dim = 2)
    weight = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)[:, :, None, None, None]
    bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

    images = torch.nn.functional.conv3d(latents, weight, bias=bias, stride=1, padding=0, dilation=1, groups=1)
    images = images.add_(1.0).mul_(127.5)
    images = images.detach().cpu()
    if images.dtype == torch.bfloat16:
        images = images.to(torch.float16)
    images = images.numpy().clip(0, 255).astype(np.uint8)
    images = einops.rearrange(images, 'b c t h w -> (b h) (t w) c')
    h, w, _ = images.shape
    scale = 200 / h
    images= Image.fromarray(images)
    images = images.resize(( int(w*scale),int(h*scale)), resample=Image.Resampling.BILINEAR) 
    return images


def process_tasks(state):
    service = service_for(state)
    yield from service.gradio_generation() if service is not None else _process_tasks(state)


def _process_tasks(state):
    from shared.utils.thread_utils import AsyncStream, async_run_in

    gen = get_gen_info(state)
    queue = gen.get("queue", [])
    progress = None

    if len(queue) == 0:
        gen["status_display"] =  False
        return
    # Gallery limits apply only to rendered views. Workspaces retain every output.
    set_main_generation_running(state, True)
    acquire_main_GPU_ressources(state)

    def release_gen():
        set_main_generation_running(state, False)
        with gen_lock:
            process_status = gen.get("process_status", None)
            if isinstance(process_status, str) and process_status.startswith("request:"):
                gen["process_status"] = "process:" + process_status[len("request:"):]
            else:
                gen["process_status"] = None

    start_time = time.time()
    notification_run = notifications.start_queue(gen, len(queue))

    global gen_in_progress
    gen_in_progress = True
    gen["in_progress"] = True
    gen["preview"] = None
    gen["status"] = get_task_status_text(queue[0])
    gen["header_text"] = ""    

    yield time.time(), time.time(), gr.update()

    com_stream = AsyncStream()
    send_cmd = com_stream.output_queue.push

    def queue_worker_func():
        gen["prompt_no"] = 0
        try:
            while len(queue) > 0:
                paused_for_edit = False
                while gen.get("queue_paused_for_edit", False):
                    if not paused_for_edit:
                        send_cmd("info", "Queue Paused until Current Task Edition is Done")
                        send_cmd("status", "Queue paused for editing...")
                        send_cmd("output", None) 
                        paused_for_edit = True
                    time.sleep(0.5)
                
                if paused_for_edit:
                    send_cmd("status", "Resuming queue processing...")
                    send_cmd("output", None)

                gen["prompt_no"] += 1

                task = None
                with lock:
                    if len(queue) > 0:
                        task = queue[0]
                        gen["api_active_queue_task"] = task

                if task is None:
                    break

                task_id = task["id"] 
                params = task['params']
                send_cmd("status", get_task_status_text(task))
                for key in ["model_filename", "lset_name"]:
                    params.pop(key, None)
                
                try:
                    import inspect
                    model_type = params.get('model_type')
                    if model_type and not _is_edit_task_params(params):
                        default_settings = get_factory_settings(model_type)
                        expected_args = set(inspect.signature(generate_media).parameters.keys())
                        for arg_name in expected_args:
                            if arg_name not in params and arg_name in default_settings:
                                params[arg_name] = default_settings[arg_name]
                    else:
                        expected_args = set(inspect.signature(generate_media).parameters.keys())
                    filtered_params = {k: v for k, v in params.items() if k in expected_args}
                    if _is_edit_task_params(params):
                        filtered_params.setdefault("model_type", "")
                    plugin_data = task.pop('plugin_data', {})
                    success = generate_media(task, send_cmd, plugin_data=plugin_data,  **filtered_params)

                except DownloadCancelled:
                    success = True
                except Exception as e:
                    tb = traceback.format_exc().split('\n')[:-1] 
                    print('\n'.join(tb))
                    send_cmd("error", str(e))
                    return

                with lock:
                    abort = gen.get("abort", False)
                    if abort:
                        record_queue_error(state, [task], "abort", abort=True)
                        gen["abort"] = False
                    gen.pop("api_active_queue_task", None)
                    if success:
                        queue[:] = [item for item in queue if item['id'] != task_id]
                if abort:
                    notification_run.interrupt("Generation aborted by the user.", aborted=True)
                    send_cmd("status", "Video Generation Aborted")
                    send_cmd("output", None)

                clear_status(state)
                if not success:
                    notification_run.interrupt("Generation stopped before the queue completed.")
                    break
                notification_run.task_completed()
                update_global_queue_ref(queue)
                
        except Exception as e:
            traceback.print_exc()
            notification_run.fail(str(e))
            send_cmd("error", f"Queue worker crashed: {e}")
        finally:
            with lock:
                gen.pop("api_active_queue_task", None)
            send_cmd("worker_exit", None)

    async_run_in("generation", queue_worker_func)

    while True:
        cmd, data = com_stream.output_queue.next()
        service = service_for(state)
        if service is not None:
            service.generation_event(cmd, data)
        if cmd == "exit":
            pass
        elif cmd == "worker_exit":
            break
        elif cmd == "info":
            gr.Info(data)
        elif cmd == "error": 
            notification_run.fail(str(data))
            record_queue_error(state, queue, data)
            queue.clear()
            try:
                save_queue_if_crash = server_config.get("save_queue_if_crash", 1)
                if save_queue_if_crash:
                    error_filename = AUTOSAVE_ERROR_FILENAME if save_queue_if_crash == 1 else get_available_filename("", AUTOSAVE_ERROR_FILENAME, f"_{datetime.now():%Y%m%d_%H%M%S}")
                    if _save_queue_to_zip(global_queue_ref, error_filename):
                        print(f"Error Queue autosaved successfully to {error_filename}")
                        from gradio.context import LocalContext
                        if LocalContext.blocks.get() is not None and LocalContext.event_id.get() is not None:
                            gr.Info(f"Error Queue autosaved successfully to {error_filename}")
                    else:
                        print("Autosave Error Queue failed.")
            except Exception as e:
                print(f"Error during autosave: {e}")

            update_global_queue_ref(queue)
            gen["prompts_max"] = 0
            gen["prompt"] = ""
            gen["status_display"] =  False
            notifications.finish_queue(server_config, gen, notification_run, time.time() - start_time)
            release_gen()
            raise gr.Error(data, print_exception= False, duration = 0)
        elif cmd == "status":
            gen["status"] = data
            status_text = str(data or "").strip()
            gen["last_progress_args"] = [0, status_text] if len(status_text) > 0 else None
        elif cmd == "output":
            gen["preview"] = None
            gen["refresh_tab"] = True
            yield time.time(), time.time(), gr.update()
        elif cmd == "progress":
            gen["last_progress_args"] = gen["progress_args"] = data
        elif cmd == "preview":
            current_model_type = "unknown"
            with lock:
                if len(queue) > 0:
                    current_model_type = queue[0]["params"].get("model_type")
            
            try:
                if not isinstance(data, (Image.Image, VideoPreview)):
                    torch.cuda.current_stream().synchronize()
                preview = None if data is None else generate_preview(current_model_type, data) 
                gen["preview"] = preview
                yield time.time(), gr.Text(), gr.update()
            except Exception:
                pass
        elif cmd == "refresh_models":
            yield gr.update(), gr.update(), (data if data is not None else get_unique_id())
        else:
            pass

    gen["prompts_max"] = 0
    gen["prompt"] = ""
    end_time = time.time()
    if gen.get("abort", False):
        notification_run.interrupt("Generation aborted by the user.", aborted=True)
    if notification_run.interrupted:
        if notification_run.aborted:
            record_queue_error(state, queue[:1], "abort", abort=True)
        status = f"Queue processing was interrupted. Total Generation Time: {format_time(end_time-start_time)}"
    else:
        status = f"Total Generation Time: {format_time(end_time-start_time)}"
        try:
            if server_config.get("notification_sound_enabled", 1):
                volume = server_config.get("notification_sound_volume", 50)
                notification_sound.notify_video_completion(volume=volume)
        except Exception as e:
            print(f"Error playing notification sound: {e}")
    notifications.finish_queue(server_config, gen, notification_run, end_time - start_time)
    gen["status"] = status
    gen["status_display"] =  False
    release_gen()


def validate_task(task, state, skip_validate_settings=False):
    """Validate a task's settings. Returns (updated params dict or None, validation error)."""
    params = task.get('params', {})
    if _is_edit_task_params(params):
        inputs = primary_settings.copy()
        inputs.update(params)
        inputs.pop("model_type", None)
        inputs.pop("base_model_type", None)
        fix_postprocess_audio_settings(inputs, params.get("settings_version", 0))
        inputs.setdefault("prompt", "Edit")
        inputs.setdefault("image_mode", 0)
        inputs.setdefault("client_id", "")
        mode = inputs.get("mode", "") or ""
        if mode == "edit_postprocessing":
            video_source = inputs.get("video_source")
            source_is_image = has_image_file_extension(video_source)
            source_is_video = has_video_file_extension(video_source)