@echo off
chcp 437 >nul
setlocal enabledelayedexpansion

:: ===============================
:: 自动获取当前目录（SIMNIBS_ENV）
:: ===============================
cd /d "%~dp0"
set "SIMNIBS_ENV=%CD%"
set "PYTHON_EXE=%SIMNIBS_ENV%\python.exe"

:: ===============================
:: 检查 Python 是否存在
:: ===============================
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found
    pause
    exit /b 1
)

:: ===============================
:: 正确启动 charm（核心修复）
:: ===============================
set "PYTHONPATH=%SIMNIBS_ENV%\Lib\site-packages"
"%PYTHON_EXE%" -m simnibs.cli.charm %*