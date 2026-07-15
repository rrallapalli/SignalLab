@echo off
REM ===========================================================
REM  SignalLab - Windows launcher
REM  Double-click this file to start the application.
REM ===========================================================
setlocal
cd /d "%~dp0"
title SignalLab

set "PY="

REM --- Prefer the Python launcher (py), fall back to python ---
where py >nul 2>&1
if not errorlevel 1 set "PY=py -3"

if not defined PY (
    where python >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

if not defined PY goto :nopython

REM --- Make sure it actually runs (Windows ships a Store stub named python.exe) ---
%PY% -c "import sys" >nul 2>&1
if errorlevel 1 goto :nopython

%PY% "%~dp0start.py" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo -----------------------------------------------------------
    echo  SignalLab stopped with an error. The message above explains why.
    echo -----------------------------------------------------------
    pause
)
exit /b %RC%

:nopython
echo.
echo -----------------------------------------------------------
echo  Python was not found on this computer.
echo.
echo  1. Download Python from https://www.python.org/downloads/
echo  2. During installation, TICK the box "Add python.exe to PATH"
echo  3. Restart your computer, then double-click this file again.
echo -----------------------------------------------------------
echo.
pause
exit /b 1
