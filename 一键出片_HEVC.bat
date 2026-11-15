@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title VVERTIGO - one click (HEVC small size)
cd /d "%~dp0"

rem ---------------------------------------------------------------
rem  Same as 一键出片.bat, only the encoder differs: HEVC (H.265).
rem  Encoded with  -c:v hevc_nvenc  ->  same quality, 30-50% smaller
rem  files than the h264 build. Needs a recent GPU (GTX 16xx / RTX
rem  or newer); without NVENC it silently falls back to libx264.
rem
rem  Put videos into the Pending folder (its Chinese name lives in
rem  run.py, so this file stays pure ASCII and cmd's code page can
rem  not mangle it).
rem
rem  Reference face is auto-matched:  3-00.01.56-00.02.38.mp4 -> 3.jpg
rem  If a video has none, run.py prints how to make one.
rem
rem  Usage:
rem    double click                      -> scan the Pending folder
rem    drag a folder onto this bat       -> that folder
rem    one-click-hevc.bat "D:\clips" --max-targets 1
rem ---------------------------------------------------------------

set "PY=py -3.12"
rem  Force UTF-8, otherwise Chinese in the console comes out garbled.
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
set "SRC="
set "EXTRA="

if not "%~1"=="" set SRC=--src "%~1"
if not "%~1"=="" shift
:collect
if not "%~1"=="" (
    set "EXTRA=!EXTRA! %1"
    shift
    goto collect
)

where %PY% >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python 3.12 not found. Edit the PY= line above.
    pause
    exit /b 1
)

echo ============================================================
echo   VVERTIGO  -  one click  (HEVC)
echo ------------------------------------------------------------
echo   engine : ffmpeg internal crop + GPU face
echo   codec  : hevc_nvenc  (H.265, smaller)
echo   output : outputs\
echo   log    : run_log.txt
echo ============================================================
echo.

rem No pipe here on purpose. Piping would replace the exit code with the
rem exit code of whatever sits at the end of the pipe, so a failed run
rem would still report success. run.py writes run_log.txt itself.
%PY% -u run.py %SRC% --face-mode anchor --render-engine ffmpeg --codec hevc_nvenc %EXTRA%
set "RC=%ERRORLEVEL%"

echo.
if not "%RC%"=="0" goto fail

echo ============================================================
echo   DONE.  Results in outputs\<video>\IDn\
echo          .mp4  9:16 vertical cut  (HEVC / hvc1)
echo          .csv  tracking points for AE
echo          .jpg  face of that person
echo ============================================================
pause
exit /b 0

:fail
echo ============================================================
echo   FAILED  rc=%RC%
echo   Full log: run_log.txt
echo ============================================================
pause
exit /b %RC%
