@echo off
chcp 65001 >nul
cd /d "%~dp0"

set LOG=install.log

echo ============================================ >> %LOG%
echo [%date% %time%] 开始安装 >> %LOG%
echo ============================================ >> %LOG%

echo.
echo ========================================
echo   媒体处理器 - 首次安装
echo ========================================
echo.

echo [1/3] 创建虚拟环境...
echo [%time%] 创建虚拟环境 >> %LOG%
python -m venv venv
if errorlevel 1 (
    echo [错误] 创建虚拟环境失败！请确保已安装 Python
    echo [%time%] 错误: 创建虚拟环境失败 >> %LOG%
    pause
    exit /b 1
)
echo       完成 ✓

echo [2/3] 激活虚拟环境...
call venv\Scripts\activate.bat

echo [3/3] 安装依赖包...
echo [%time%] 开始安装依赖 >> %LOG%
echo.
pip install -r requirements.txt 2>&1
if errorlevel 1 (
    echo.
    echo [错误] 安装依赖失败！
    echo [%time%] 错误: 安装依赖失败 >> %LOG%
    pause
    exit /b 1
)

echo.
echo [%time%] 安装完成 >> %LOG%
echo ========================================
echo   安装完成！
echo ========================================
echo.
echo 已安装的包:
pip list --format=columns
echo.
echo 日志已保存到: %LOG%
echo.
echo 3秒后启动程序...
timeout /t 3 >nul

echo [%time%] 启动程序 >> %LOG%
start "" pythonw main.py
exit
