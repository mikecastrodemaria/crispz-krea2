@echo off
rem Rebuilds the conversion cache of the Civitai checkpoints (single-file).
rem Re-runnable at will: whatever is already converted is skipped in a second.
rem Option: rebuild_cache.bat --cpu  (dequantises without touching the GPU)
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe tools\rebuild_convert_cache.py %*
pause
