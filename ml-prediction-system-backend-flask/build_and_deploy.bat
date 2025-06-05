@echo off
echo 正在构建和部署前后端集成系统...
echo.

echo 1. 进入前端项目目录...
cd /d "%~dp0\..\ml-prediction-system"

echo 2. 安装前端依赖...
call npm install

echo 3. 构建前端静态文件...
call npm run build

echo 4. 清理并复制前端文件到后端...
cd /d "%~dp0"
if exist "app\static\frontend" rmdir /s /q "app\static\frontend"
mkdir "app\static\frontend"
xcopy /E /I /Y "..\ml-prediction-system\out\*" "app\static\frontend\"

echo 5. 检查前端文件是否复制成功...
if exist "app\static\frontend\index.html" (
    echo ✓ 前端文件复制成功
) else (
    echo ✗ 前端文件复制失败
    echo 手动复制前端文件到 app\static\frontend\ 目录
    pause
    exit /b 1
)

echo 6. 启动 Flask 服务...
python run.py

pause 