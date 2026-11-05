@echo off
chcp 437 >nul
cd /d "%~dp0"
set "SIMNIBS_ENV=%CD%"
set "PYTHON_EXE=%SIMNIBS_ENV%\python.exe"
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found
    pause
    exit /b 1
)
set "PYTHONPATH=%SIMNIBS_ENV%\Lib\site-packages"
"%PYTHON_EXE%" -m simnibs.cli.charm %*
