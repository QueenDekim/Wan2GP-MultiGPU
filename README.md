# Wan2GP-MultiGPU

-----
<p align="center">
  <b>Wan2GP v13.141 by <a href="https://github.com/deepbeepmeep">DeepBeepMeep</a></b><br>
  <b>MultiGPU by <a href="https://github.com/QueenDekim">QueenDekim</a></b>
</p>

**Wan2GP-MultiGPU** is a GPU-first fork of [Wan2GP / WanGP](https://github.com/deepbeepmeep/Wan2GP), bringing architecture-aware NVIDIA MultiGPU execution to WanGP's open-source video, image, audio, and text-to-speech tools. The original application is developed by **DeepBeepMeep**; the MultiGPU integration and GPU-first memory management are maintained by **QueenDekim**. The fork preserves WanGP's single-GPU workflow when MultiGPU is disabled.

## Highlights

| Modality | Supported models |
| --- | --- |
| **Video** | **Wan 2.1/2.2** and derived models, **MiniMax H3**, **LTX-2/2.3/2.5**, **Hunyuan Video 1/1.5**, **LongCat**, **Kandinsky**, **LTXV**, **MagiHuman** |
| **Image** | **Krea 2**, **Qwen Image**, **Z-Image**, **Flux 1/2** (*Klein*, Chroma), **SenseNova**, **Ideogram 4**, **HiDream** |
| **Audio / TTS** | **Qwen3 TTS**, **MiniMax H3 Voice Clone**, **Ace Step 1/2/XL**, **Omnivoice**, **Index TTS2/2.5**, **KugelAudio**, **HeartMula**, **Chatterbox**, **Minimax Music**, **Stable Audio 3** |

### Run More Models on More Hardware

- **Low VRAM requirements**: run select models with as little as **6 GB of VRAM**.
- **NVIDIA MultiGPU support**: run compatible model blocks across two or more CUDA GPUs with GPU-only Accelerate dispatch, rather than silently falling back to CPU/disk offload.
- **GPU-first memory management**: prioritize VRAM for model weights, activations and adapters; reduce pinned host-memory usage and avoid unnecessary CPU roundtrips.
- **Per-mode memory profiles**: separate MultiGPU video, image and audio profiles for 8, 12, 16 and 24 GB+ primary GPUs, with automatic GPU discovery.
- **Architecture-aware model placement**: contiguous transformer and decoder layer splits for supported families; generic dispatch is retained as an experimental fallback for other architectures.
- **Older Nvidia GPU support**: use GTX 10XX, RTX 20XX, and newer cards.
- **AMD GPU support**: run on RDNA 4, 3, 3.5, and 2 hardware; see the Installation section below.
- **Fast latest-GPU performance**: take advantage of modern GPU acceleration.
- **Full web interface**: generate, manage, and reuse outputs from an easy browser UI.
- **LoRA customization**: adapt each model with LoRAs, reuse LoRAs stored in another App. MultiGPU keeps supported LoRA/DoRA/LoKr data with its owning GPU modules and can use block-level JIT LoRA residency when VRAM headroom is limited.
- **Many quantized checkpoint formats**: use int8, fp8, gguf, NV FP4, and Nunchaku.
- **Architecture-aware downloads**: automatically fetch the model files suited to your hardware.
- **Finetunes**: add your own finetunes / checkpoints or the ones you found on Hugging Face or CivitAI
- **Generation queue**: line up videos, images, and audio jobs, then come back later.
- **Headless mode**: launch batches from the command line for images, videos, and audio.
- **WanGP API**: add generative capabilities to your own apps.

### Built-In Creation Tools

- **Video, image, and audio galleries**: browse generations and reuse them as new inputs.
- **Reusable settings**: extract settings from any generation, create templates, and share them.
- **Per-model prompt enhancer**: improve prompts with model-specific syntax and expectations.
- **Input preparation tools**: use the mask editor, background remover, pose/depth/flow extractors, speaker diarization, and background noise/song remover.
- **Deepy low-VRAM offline agent**: orchestrate generation jobs and tedious tasks such as transcription, video splitting, and color-frame generation while you are away.
- **Temporal and spatial upsampling**: improve outputs with RIFE, FlashVSR, and Lanczos.
- **Audio postprocessing**: generate soundtracks with MMAudio, replace voices with SeedVC, or remux a video with any soundtrack.
- **Ready-to-use plug-ins**: Gallery Browser, Motion Designer, Models/Checkpoints Manager, CivitAI browser and downloader, and more.

**Discord Server to get Help from the WanGP Community and show your Best Gens:** https://discord.gg/g7efUW9jGV

**Follow DeepBeepMeep on Twitter/X to get the Latest News**: https://x.com/deepbeepmeep

**Official WanGP Web Site**: https://wangp.ai/

> [!IMPORTANT]
> **WanGP is free to use locally.** The official project will never ask you to pay a license fee, subscription, or donation to run WanGP on your own computer (see the license for terms).
>
> **Upstream WanGP is maintained by DeepBeepMeep at [deepbeepmeep/Wan2GP](https://github.com/deepbeepmeep/Wan2GP).** This repository is the independently maintained [QueenDekim/Wan2GP-MultiGPU](https://github.com/QueenDekim/Wan2GP-MultiGPU) fork, not the official WanGP distribution. For the upstream project and official services, use its original repository and [wangp.ai](https://wangp.ai/).


## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🖥️ Tiered MultiGPU Offload](#️-tiered-multigpu-offload)
- [🎯 Usage](#-usage)
- [📚 Documentation](#-documentation)
- [🔗 Related Projects](#-related-projects)


## 🚀 Quick Start

### One-click Bat/SH Script Auto-installer:

The 1-click automated scripts for both **Windows (`.bat`)** and **Linux/macOS (`.sh`)** make installation, environment management, and updates as seamless as possible. These scripts will not only install WanGP but also best acceleration kernels (Triton, Sage, Flash, GGuf, Lightx2v, Nunchaku) available for your config.

*👉 **Windows Users:** Double-click the `.bat` files. **Linux Users:** Run the `.sh` files in your terminal.*

#### **1️⃣ Installation (`scripts\install.bat` | `scripts/install.sh`)**

**Choose Installation Type**
- **Auto Install**
- **Manual Install**

**Manual Install**

If you selected Manual Install, you will be guided through:

1. **Choose your package manager**
2. **Name your environment**
3. **Select your Install Mode**

#### 2️⃣ Starting the App (`scripts\run.bat` | `scripts/run.sh`)
Once installed, use this script to launch the application. It runs WAN2GP using your active environment.

*   **⚙️ Customizing Launch Arguments (`args.txt`)**
    *   If you want to pass extra command-line flags to the launcher (like enabling advanced UI features or automatically opening your browser), create an `args.txt` file in your `scripts` folder.
    *   **Example `args.txt`:**
        ```text
        --advanced --open-browser
        ```

#### 3️⃣ Updating & Upgrading (`scripts\update.bat` | `scripts/update.sh`)
Use this script to get the latest updates for WAN2GP and upgrade dependencies.
* **1. Update:** Fetches the latest code from GitHub and updates requirements.
* **2. Upgrade:** Allows you to manually individually upgrade heavy backend components (like PyTorch, Triton, Sage Attention).

Triton recommendations follow both your GPU and selected PyTorch: **3.3.x with PyTorch 2.7**, **3.6.x with PyTorch 2.10** on RTX 30XX or newer, **3.2.x on RTX 20XX**, and **3.7.x with PyTorch 2.13** on AMD. Use **Upgrade** to correct an older Triton installation; **Update** alone does not change it.

#### 4️⃣ Managing Environments (`scripts\manage.bat` | `scripts/manage.sh`)
Use this script to manage and switch between your sandboxed environments safely.

* **Example Scenario 1: Migrating an Existing Setup**
    * If you have a folder named `venv` that works perfectly and want to use it with the new one-click scripts, run `manage.bat` and select **Add Existing Environment**.
    * Copy-paste the folder path (e.g., `C:\WAN2GP\venv`), select type `venv`, then use **Set Active Environment** to make it the default. Now `run.bat` and `update.bat` will target your existing setup.

* **Example Scenario 2: Testing New Configurations**
    * Let's say you have an environment named `env_stable` that works perfectly, but you want to try different components. Run `install.bat`, create a *new* environment called `env_testing`, and select **Manual Selection**. **Autoselect** installs the recommended stack for your GPU.
    * If the testing environment breaks, simply open `manage.bat`, select **Set Active Environment**, and switch back to `env_stable`. You are back up and running instantly.

---

### One-click Installers
- Pinokio installer
Get started instantly with [Pinokio App](https://pinokio.computer/)\
It is recommended to use in Pinokio the Community Scripts *wan2gp* or *wan2gp-amd* by **Morpheus** rather than the official Pinokio install.

- Wan2GP Desktop by GKArtist
[Wan2GP Desktop](https://github.com/GKartist75/Wan2GP-Desktop-Tauri) is a desktop launcher for Wan2GP that installs, updates, and runs it from one window — handling Git, Python, CUDA, and PyTorch setup so you don't have to configure them manually.

### Manual installation: (for RTX20xx - RTX50xx)

```bash
git clone https://github.com/QueenDekim/Wan2GP-MultiGPU.git
cd Wan2GP-MultiGPU
conda create -n wan2gp python=3.11.14
conda activate wan2gp
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

On Windows, install Triton in the same environment, using the command matching your GPU and PyTorch:

| GPU | PyTorch | Command |
| --- | --- | --- |
| RTX 30XX - RTX 50XX | 2.10 (recommended) | `python -m pip install -U "triton-windows>=3.6,<3.7"` |
| RTX 30XX - RTX 50XX | 2.7 / 2.7.1 | `python -m pip install -U "triton-windows>=3.3,<3.4"` |
| RTX 20XX | 2.7 or 2.10 (legacy Triton compatibility exception) | `python -m pip install -U "triton-windows>=3.2,<3.3"` |

Triton 3.2 on RTX 30XX or newer can crash YuE2's optimized INT8 ConvRot kernels because it lacks `triton.language.gather`. Upgrade Triton as above and restart WanGP. RTX 20XX must stay on 3.2 and uses the existing standard INT8 path; compatibility with newer PyTorch is not guaranteed upstream. RTX 50XX users should use PyTorch 2.10 / Triton 3.6 for WanGP's optimized INT8 and NV FP4 support. Linux normally receives matching Triton through PyTorch. See the **[Triton Installation Guide](docs/INSTALLATION.md#triton-installation)** for details and RTX 20XX limitations.

For optimized attention, install **SageAttention 1.0.6 on RTX 20XX** and **SageAttention 2.2.0 on RTX 30XX or newer**. SageAttention 2 requires an Ampere-or-newer GPU; GTX 10XX should use SDPA. See the **[Installation Guide](docs/INSTALLATION.md#sage-attention)** for the platform-specific commands.

### Manual installation: (for GTX 10xx)

```bash
git clone https://github.com/QueenDekim/Wan2GP-MultiGPU.git
cd Wan2GP-MultiGPU
conda create -n wan2gp python=3.10.9
conda activate wan2gp
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/test/cu128
pip install -r requirements.txt
```

#### Run the application:

```bash
python wgp.py
```
If you are low on VRAM, there is a trick to increase the amount of VRAM available (between 1GB and 5GB of VRAM to be gained depending on the GPU): *disable GPU Usage in your Web Browser*.

Run *scripts/start-chrome-no-gpu.bat* or *scripts/start-chrome-no-gpu.sh* to launch Chrome without using your GPU. 

First time using WanGP ? Just check the *Guides* tab, and you will find a selection of recommended models to use.

#### Update the application (stay in the current python / pytorch version):
If using Pinokio use Pinokio to update otherwise:
Get in the directory where WanGP is installed and:
```bash
git pull
conda activate wan2gp
pip install -r requirements.txt
```

#### Upgrade from Python 3.10, Pytorch 2.7.1, Cuda 12.8 to Python 3.11, Pytorch 2.10, Cuda 13/13.1 (for non GTX10xx users)
I recommend renaming first the old conda environment to avoid bad surprises when installing a different config in this old environment.

```bash
conda rename -n wan2gp  old_wan2gp
```

Get in the directory where WanGP is installed and:
```bash
git pull
conda create -n wan2gp python=3.11.9
conda activate wan2gp
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Once you are done you will have to reinstall *Sage Attention*, *Triton*, *Flash Attention*. Check the **[Installation Guide](docs/INSTALLATION.md)** -

if you get some error messages related to git, you may try the following (beware this will overwrite local changes made to the source code of WanGP):
```bash
git fetch origin && git reset --hard origin/main
conda activate wan2gp
pip install -r requirements.txt
```
When you have the confirmation it works well you can then delete the old conda env:
```bash
conda uninstall -n old_wan2gp --all  
```

#### Run headless (batch processing):

Process saved queues without launching the web UI:
```bash
# Process a saved queue
python wgp.py --process my_queue.zip
```
Create your queue in the web UI, save it with "Save Queue", then process it headless. See [CLI Documentation](docs/CLI.md) for details.

## 🐳 Docker:

**For Debian-based systems (Ubuntu, Debian, etc.):**

```bash
./run-docker-cuda-deb.sh
```

This automated script will:

- Detect your GPU model and VRAM automatically
- Select optimal CUDA architecture for your GPU
- Install NVIDIA Docker runtime if needed
- Build a Docker image with all dependencies
- Run WanGP with optimal settings for your hardware

**Docker environment includes:**

- NVIDIA CUDA 12.4.1 with cuDNN support
- PyTorch 2.6.0 with CUDA 12.4 support
- SageAttention compiled for your specific GPU architecture
- Optimized environment variables for performance (TF32, threading, etc.)
- Automatic cache directory mounting for faster subsequent runs
- Current directory mounted in container - all downloaded models, loras, generated videos and files are saved locally

**Supported GPUs:** RTX 40XX, RTX 30XX, RTX 20XX, GTX 16XX, GTX 10XX, Tesla V100, A100, H100, and more.

## 📦 Installation

### Nvidia
For detailed installation instructions for different GPU generations:
- **[Installation Guide](docs/INSTALLATION.md)** - Complete setup instructions for GTX 10XX, RTX 20XX to RTX 50XX
- **[Optional DLSS 5 Upsamplers](docs/DLSS5.md)** - Native-resolution refinement, spatial upsampling, and Frame Generation runtime setup

### AMD
For detailed installation instructions for different GPU generations:
- **[Installation Guide](docs/AMD-INSTALLATION.md)** - Complete setup instructions for RDNA 4, 3, 3.5, and 2

## 🖥️ Tiered MultiGPU Offload

**MultiGPU by [QueenDekim](https://github.com/QueenDekim)** extends the original **Wan2GP v13.141 by [DeepBeepMeep](https://github.com/deepbeepmeep)** with a GPU-first, multi-device execution path. For supported architectures, model blocks are **dispatched to multiple NVIDIA GPUs** with Hugging Face Accelerate. Unlike a simple secondary-GPU LRU cache, those blocks perform their forward computations on their assigned GPUs.

The preferred execution/storage hierarchy is:

`GPU#0 → GPU#1 → ... → GPU#N → RAM (last-resort backing during model switching)`

**RAM is not an extra transformer shard:** active MultiGPU model maps intentionally exclude CPU and disk. The system keeps limited host-backed checkpoint data for reloading/switching models; it does not promise zero RAM usage. If a selected checkpoint cannot fit in usable GPU memory, the loader reports an error instead of transparently performing CPU/disk inference. Some preprocessing, media encoding and model-specific code still require host memory.

**Starting MultiGPU**

On systems with multiple compatible CUDA GPUs the launcher can discover them automatically. To select devices and their order explicitly:

```bash
python wgp.py --multigpu cuda:0,cuda:1
```

Or with three GPUs:

```bash
python wgp.py --multigpu cuda:0,cuda:1,cuda:2 --verbose 2
```

The **first** device is the primary GPU; `--gpu`, when explicitly specified, must match it. Other GPUs may have different VRAM capacities. Actual assignments are weighted using free VRAM, model size and an activation reserve; `--verbose 2` shows the split and host-memory diagnostics.

**MultiGPU memory profiles**

MultiGPU maintains independent **video, image and audio** profiles, separate from the original single-GPU profiles. The automatic choice uses the **primary** GPU's nominal VRAM: **8 GB, 12 GB, 16 GB, or 24 GB+**. You can select a profile in the web UI or override it with `--profile 8`, `--profile 12`, `--profile 16` or `--profile 24`. The CLI option `--multigpu-cache-fraction` is retained for compatibility but, in this GPU-first path, controls the usable VRAM fraction for placement rather than an LRU-only secondary cache:

```bash
python wgp.py --multigpu cuda:0,cuda:1 --profile 16 --multigpu-cache-fraction 0.90
```

**Architecture-specific integrations**

- **LTX2**: X0Model transformer partitioning, sharded Gemma, quantized-weight compatibility, and adaptive LoRA block-JIT residency.
- **Wan 2.1 / 2.2, LTX-Video 0.9.x, Hunyuan Video and Kandinsky 5**: whole-block contiguous distribution with explicit placement of support layers.
- **MiniMax H3 / Qwen3-VL**: dedicated whole-layer GPU-only Qwen3-VL text-encoder mapping with capacity checks.
- **Other models**: generic Accelerate GPU-only placement and conservative sequential-block discovery where compatible; **not all families/checkpoints are validated**.
- **LoRA / DoRA / LoKr**: adapter residency on the assigned devices where supported; MMGP-backed JIT LoRA residency is used when necessary to preserve activation headroom.

**Validation and troubleshooting**

```bash
python scripts/multigpu_smoke.py --require-gpu
python scripts/audit_multigpu.py --out multigpu-audit.json
```

The smoke test requires at least two CUDA GPUs and checks a small synthetic model; it does **not** certify full inference quality or performance for every checkpoint. For large models, monitor CUDA VRAM, Windows Private Bytes and generation throughput. If one model still fails, attach its `--verbose 2` log and the selected model/quantization/LoRA settings.

**Without `--multigpu`**, WanGP retains the standard MMGP single-GPU behavior and profile settings.
## 🎯 Usage

### Basic Usage
- **[Getting Started Guide](docs/GETTING_STARTED.md)** - First steps and basic usage
- **[Models Overview](docs/MODELS.md)** - Available models and their capabilities
- **[Prompts Guide](docs/PROMPTS.md)** - How WanGP interprets prompts, images as prompts, enhancers, and macros

### Advanced Features
- **[Deepy Assistant](docs/DEEPY.md)** - Launch Deepy in Gradio, CLI or standalone Web mode; configure tools, media references, saved sessions and phone access
- **[Authentication, HTTPS, and Reverse Proxies](docs/AUTHENTICATION.md)** - Protect web access, configure proxy hosting with `--public-url`, and set up MCP OAuth
- **[Remote LLMs](docs/REMOTE_LLMS.md)** - Configure Codex, Claude Code, and OpenCode providers for Deepy and Prompt Enhancer
- **[Loras Guide](docs/LORAS.md)** - Using and managing Loras for customization
- **[Finetunes](docs/FINETUNES.md)** - Add manually new models to WanGP
- **[VACE ControlNet](docs/VACE.md)** - Advanced video control and manipulation
- **[Processing Guide](docs/PROCESSING.md)** - Preprocessing, masks, sliding windows, and postprocessing
- **[Command Line Reference](docs/CLI.md)** - All available command line options

## 📚 Documentation

- **[Changelog](docs/CHANGELOG.md)** - Latest updates and version history
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 📚 Video Guides
- Nice Video that explain how to use Vace:\
https://www.youtube.com/watch?v=FMo9oN2EAvE
- Another Vace guide:\
https://www.youtube.com/watch?v=T5jNiEhf9xk

## 🔗 Related Projects

### Other Models for the GPU Poor
- **[HuanyuanVideoGP](https://github.com/deepbeepmeep/HunyuanVideoGP)** - One of the best open source Text to Video generators
- **[Hunyuan3D-2GP](https://github.com/deepbeepmeep/Hunyuan3D-2GP)** - Image to 3D and text to 3D tool
- **[FluxFillGP](https://github.com/deepbeepmeep/FluxFillGP)** - Inpainting/outpainting tools based on Flux
- **[Cosmos1GP](https://github.com/deepbeepmeep/Cosmos1GP)** - Text to world generator and image/video to world
- **[OminiControlGP](https://github.com/deepbeepmeep/OminiControlGP)** - Flux-derived application for object transfer
- **[YuE GP](https://github.com/deepbeepmeep/YuEGP)** - Song generator with instruments and singer's voice

---

<p align="center">
Wan2GP v13.141 by <a href="https://github.com/deepbeepmeep">DeepBeepMeep</a><br>
MultiGPU by <a href="https://github.com/QueenDekim">QueenDekim</a>
</p>
