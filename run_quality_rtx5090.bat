@echo off
title crispz-qwen-edit - RTX 5090 (local)
cd /d "%~dp0"
echo ============================================
echo  crispz-qwen-edit - RTX 5090 (local 127.0.0.1)
echo ============================================
echo.
REM CUDA optimisations (harmless, BF16)
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set GRADIO_SERVER_PORT=7860
REM A UTF-8 console (avoids the cp1252 crashes on the HF progress bars)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM === LOCAL-ONLY: uses the HF cache ONLY, never RE-downloads Qwen-Image. ===
REM Qwen/Qwen-Image is already cached (46 GB) -> loaded locally, 0 download.
REM Side effect: the Edit tab (Qwen-Image-Edit-2509, not cached) will show an error
REM instead of hoovering up ~20 GB. Comment this line out (REM) to allow the
REM downloads once (then put it back).
set HF_HUB_OFFLINE=1
REM === VRAM (RTX 5090, 32 GB). This env forces the offload whatever the UI/config says.
REM   - With a quantised GGUF transformer (e.g. qwen-image-Q4_K_M.gguf, ~12 GB): 'model'
REM     fits with room to spare AND stays fast (~1 s/step). Recommended (the default below).
REM   - With the FULL bf16 Qwen-Image (~44 GB, no GGUF): 'model' OOMs -> set 'sequential'
REM     (layer by layer, it fits but it is slow). ===
set CZ_OFFLOAD=model
REM Delegates to run.bat (venv detection + ESRGAN_DIR + launch)
call "%~dp0run.bat" %*
