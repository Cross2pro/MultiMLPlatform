@echo off
chcp 65001 >nul
echo ============================================
echo 自动编译和部署脚本
echo ============================================

echo [1/4] 进入前端项目目录...
cd ml-prediction-system
if errorlevel 1 (
    echo 错误：无法进入 ml-prediction-system 目录
    pause
    exit /b 1
)

echo [2/4] 开始编译前端项目...
echo 正在运行 npm run build...
call npm run build
if errorlevel 1 (
    echo 错误：前端项目编译失败
    pause
    exit /b 1
)

echo [3/4] 清空目标目录...
cd ..
set TARGET_DIR=ml-prediction-system-backend-flask\app\static\frontend
if exist "%TARGET_DIR%" (
    echo 正在清空 %TARGET_DIR% 目录...
    rmdir /s /q "%TARGET_DIR%"
    if errorlevel 1 (
        echo 警告：清空目录时出现错误，继续执行...
    )
)

echo [4/4] 复制编译文件到目标目录...
echo 正在复制文件从 ml-prediction-system\out 到 %TARGET_DIR%...
xcopy "ml-prediction-system\out\*" "%TARGET_DIR%\" /E /I /H /Y
if errorlevel 1 (
    echo 错误：文件复制失败
    pause
    exit /b 1
)

echo ============================================
echo 编译和部署完成！
echo 源目录：ml-prediction-system\out
echo 目标目录：%TARGET_DIR%
echo ============================================
pause 