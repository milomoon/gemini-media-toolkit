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

:: ============================================
:: Find Python (跳过 Windows Store 假 Python)
:: ============================================
set PYTHON_CMD=

:: 1. 尝试 py 命令（推荐，会自动找真实 Python）
py --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON_CMD=py
    goto :python_found
)

:: 2. 使用 where 查找所有 python，跳过 WindowsApps 的假货
for /f "tokens=*" %%i in ('where python 2^>nul') do (
    echo %%i | findstr /i "WindowsApps" >nul
    if errorlevel 1 (
        set PYTHON_CMD=%%i
        goto :python_found
    )
)

:: 3. 检查常见安装路径
if exist "D:\python\python.exe" (
    set PYTHON_CMD=D:\python\python.exe
    goto :python_found
)

if exist "C:\Python312\python.exe" (
    set PYTHON_CMD=C:\Python312\python.exe
    goto :python_found
)

if exist "C:\Python311\python.exe" (
    set PYTHON_CMD=C:\Python311\python.exe
    goto :python_found
)

if exist "C:\Python310\python.exe" (
    set PYTHON_CMD=C:\Python310\python.exe
    goto :python_found
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    goto :python_found
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe
    goto :python_found
)

if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe
    goto :python_found
)

:: Python not found
echo [ERROR] Real Python not found!
echo.
echo Found Windows Store Python stub, but it cannot create venv.
echo.
echo Please install REAL Python 3.8+ from:
echo https://www.python.org/downloads/
echo.
echo IMPORTANT: 
echo   1. Check "Add Python to PATH" during installation
echo   2. DO NOT use Windows Store Python
echo.
pause
start https://www.python.org/downloads/
exit /b 1

:python_found
echo [OK] Found Python: %PYTHON_CMD%
%PYTHON_CMD% --version
echo       OK

echo [2/4] Create venv...
echo This may take 1-2 minutes...

:: Delete corrupted venv if exists
if exist "venv" (
    echo Cleaning up old venv...
    rmdir /s /q venv
)

:: Try to create venv with pip first
%PYTHON_CMD% -m venv venv >nul 2>&1

:: Check if venv was created
if not exist "venv\Scripts\python.exe" (
    echo [WARN] Standard venv creation failed, trying without pip...
    %PYTHON_CMD% -m venv venv --without-pip >nul 2>&1
    
    if not exist "venv\Scripts\python.exe" (
        echo [ERROR] Failed to create virtual environment!
        echo.
        echo Possible solutions:
        echo   1. Run as Administrator
        echo   2. Disable antivirus temporarily
        echo   3. Reinstall Python with "pip" option checked
        echo   4. Make sure you installed REAL Python, not Windows Store version
        echo.
        pause
        exit /b 1
    )
    
    :: Install pip manually
    echo Installing pip manually...
    curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    if exist "get-pip.py" (
        venv\Scripts\python.exe get-pip.py >nul 2>&1
        del get-pip.py
    )
)
echo       OK

echo [3/4] Activate venv...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    echo.
    echo Deleting corrupted venv...
    rmdir /s /q venv
    echo Please run this script again.
    echo.
    pause
    exit /b 1
)
echo       OK

echo [4/4] Install packages...
echo This may take 2-3 minutes, please wait...
echo.

:: Upgrade pip first
python -m pip install --upgrade pip -q >nul 2>&1

:: Try with China mirror first
echo Installing with Aliyun mirror...
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple 2>&1

if errorlevel 1 (
    echo.
    echo [WARN] Aliyun mirror failed, trying Tsinghua mirror...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1
)

if errorlevel 1 (
    echo.
    echo [WARN] Tsinghua mirror failed, trying official PyPI...
    pip install -r requirements.txt 2>&1
)

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies!
    echo.
    echo Please check:
    echo   1. Internet connection is working
    echo   2. Firewall is not blocking pip
    echo   3. Disk space is sufficient
    echo.
    pause
    exit /b 1
)

echo.
echo [OK] Dependencies installed successfully

echo.
echo ========================================
echo   Done! Run start.vbs to launch
echo   Or run START.bat again to restart
echo ========================================
echo.
timeout /t 3 >nul
start "" pythonw main.py
exit
