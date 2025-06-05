@echo off
chcp 65001
echo ========================================
echo 彻底解决 Docker 构建网络连接问题
echo ========================================
echo.

echo 步骤1: 配置 Docker 网络设置...
call configure-docker-network.bat

echo.
echo 步骤2: 清理 Docker 缓存...
docker system prune -f
docker builder prune -f

echo.
echo 步骤3: 尝试使用优化的 Dockerfile 构建...
echo 正在使用多重镜像源策略构建后端镜像...

cd ml-prediction-system-backend-flask

REM 尝试主要 Dockerfile
echo 尝试方案1: 使用主要 Dockerfile...
docker build -t ml-backend:latest . 2>build_error.log

if %ERRORLEVEL% NEQ 0 (
    echo 方案1失败，尝试方案2: 使用备选 Dockerfile...
    docker build -f Dockerfile.alternative -t ml-backend:latest . 2>build_error_alt.log
    
    if %ERRORLEVEL% NEQ 0 (
        echo 两种方案都失败，显示错误信息：
        echo ===== 主要构建错误 =====
        type build_error.log
        echo.
        echo ===== 备选构建错误 =====
        type build_error_alt.log
        echo.
        echo 请检查网络连接或尝试使用手机热点。
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo 后端镜像构建成功！
echo ========================================
echo.

cd ..

echo 步骤4: 构建完整应用...
docker-compose up --build -d

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo 🎉 应用部署成功！
    echo ========================================
    echo 前端地址: http://localhost:3000
    echo 后端地址: http://localhost:5000
    echo ========================================
) else (
    echo 应用部署失败，请检查 docker-compose 日志
    docker-compose logs
)

pause 