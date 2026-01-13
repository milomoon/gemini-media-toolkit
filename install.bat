@echo off
chcp 65001 >nul
cd /d "%~dp0"

set LOG=install.log

echo [%date% %time%] Start >> %LOG%

echo.
echo ========================================
echo   Gemini Media Toolkit
echo ========================================
echo.

echo [1/4] Check Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [Error] Python not found!
    echo Please install Python first
    pause
    exit /b 1
)
echo       OK

echo [2/4] Create venv...
python -m venv venv
if errorlevel 1 (
    echo [Error] Create venv failed
    pause
    exit /b 1
)
echo       OK

echo [3/4] Activate venv...
call venv\Scripts\activate.bat

echo [4/4] Install packages (China mirror)...
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple 2>&1

if errorlevel 1 (
    echo Try tsinghua mirror...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1
)

if errorlevel 1 (
    echo Try official source...
    pip install -r requirements.txt 2>&1
)

if errorlevel 1 (
    echo [Error] Install failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Done! Run start.vbs to launch
echo ========================================
echo.
timeout /t 3 >nul
start "" pythonw main.py
exit
