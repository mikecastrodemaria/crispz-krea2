@echo off
rem Reconstruit le cache de conversion des checkpoints Civitai (single-file).
rem Relancable a volonte: ce qui est deja converti est saute en une seconde.
rem Option: rebuild_cache.bat --cpu  (dequantification sans toucher au GPU)
cd /d "%~dp0"
set PYTHONUTF8=1
.venv\Scripts\python.exe tools\rebuild_convert_cache.py %*
pause
