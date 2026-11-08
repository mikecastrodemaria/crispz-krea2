# crispz-krea2 1.16: a Krea 2 text-to-image studio, local, 1-click

![crispz-krea2](https://raw.githubusercontent.com/mikecastrodemaria/crispz-krea2/main/assets/screenshot.png)

**crispz-krea2** is a Fooocus-style web studio for **Krea 2**, the open-weights
successor of FLUX.1 Krea, wrapped in the same workbench as crispz-studio. Everything
runs on your own GPU, no ComfyUI, no cloud.

- **Text to Image** with styles, wildcards, tag autocomplete, prompt variants, live
  preview and Stop; **Upscale after generate** chains each image through ESRGAN.
- **Models**: single-file checkpoints from CivitAI, LoRAs with hot swap, a CivitAI search
  with previews and trigger words.
- **Job queue and X/Y/Z grids** for overnight batches and side-by-side comparisons.
- **Prompt AI, local**: Describe, Improve and Vision Mix through Ollama; Remove BG, Face
  Swap, CLI and the family protocol v1.
- **What it does not do, said plainly**: diffusers has no Krea 2 img2img pipeline yet,
  so the upscale is a pure ESRGAN enlargement and there is no Vary, no inpaint. For
  those, send the image to another crispz studio (klein, flux3, studio).

Hardware: an NVIDIA GPU; 16 GB of VRAM is comfortable with offload. Check the Krea 2
model card on Hugging Face for its licence and token requirement.

Pinokio launcher: https://github.com/mikecastrodemaria/crispz-krea2.pinokio
Source: https://github.com/mikecastrodemaria/crispz-krea2

## Install (Pinokio)

1. In Pinokio, open **Discover** and paste
   `https://github.com/mikecastrodemaria/crispz-krea2.pinokio`.
2. Click **Install**: the launcher clones the app, creates the venv, installs the
   requirements and PyTorch (CUDA 12.8 on NVIDIA).
3. Click **Start**, then **Open Web UI**. The first render downloads the model.
