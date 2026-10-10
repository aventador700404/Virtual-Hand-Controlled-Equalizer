@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title Ambient Conductor

rem  Usage:
rem    double-click run.bat       : pick a song from the songs\ folder
rem    drag an .mp3 onto run.bat  : play that song
rem  The first run installs everything once (PyTorch + Demucs included) and
rem  downloads the models. Later runs skip all of that and start right away.
rem  To force a re-check of the install, delete venv\.ambient_ready

set "PY=venv\Scripts\python.exe"
set "READY=venv\.ambient_ready"

if exist "%READY%" goto :run

rem ---------------- one-time setup ----------------
if not exist "%PY%" (
    echo [setup] Creating virtual environment...
    py -3 -m venv venv 2>nul || python -m venv venv
)
if not exist "%PY%" (
    echo [error] Python not found. Install Python 3.10+ from python.org, then run again.
    goto :fail
)

"%PY%" -c "import cv2, mediapipe, sounddevice, soundfile, scipy" 2>nul
if errorlevel 1 (
    echo [setup] Installing base packages...
    "%PY%" -m pip install opencv-python mediapipe numpy scipy sounddevice soundfile || goto :fail
)

"%PY%" -c "import torch, demucs.api" 2>nul
if errorlevel 1 (
    call :install_splitter || goto :fail
)

echo [setup] Downloading the song-splitting model, about 80 MB...
"%PY%" src\song_prep.py --download-model
if errorlevel 1 echo [setup] Model download failed - it will be retried the first time a song is split.

echo ready> "%READY%"
echo [setup] Done. Next time run.bat starts right away.
echo.

rem ---------------- run ----------------
:run
if not exist "models\hand_landmarker.task" (
    echo [setup] Downloading hand tracking model...
    if not exist models mkdir models
    curl.exe -L -o "models\hand_landmarker.task" "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" || goto :fail
)

if "%~1"=="" (
    "%PY%" src\main.py --choose
) else (
    "%PY%" src\main.py --song "%~1"
)
if errorlevel 1 goto :fail
exit /b 0


:install_splitter
echo.
echo [setup] Installing PyTorch + Demucs for song splitting. First time only, this takes a while.
where nvidia-smi >nul 2>nul
if errorlevel 1 goto :torch_cpu
echo [setup] NVIDIA GPU found - installing the CUDA build of PyTorch, about 3 GB...
for %%C in (cu128 cu126 cu130) do (
    "%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/%%C && goto :torch_done
)
echo [setup] CUDA build failed - falling back to the CPU build.
:torch_cpu
"%PY%" -m pip install torch || exit /b 1
:torch_done
"%PY%" -m pip install "demucs>=4.1" || exit /b 1
exit /b 0


:fail
echo.
echo Something went wrong - scroll up for the message.
pause
exit /b 1
