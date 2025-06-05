#!/bin/bash

echo "正在构建和部署前后端集成系统..."
echo

echo "1. 进入前端项目目录..."
cd "$(dirname "$0")/../ml-prediction-system"

echo "2. 安装前端依赖..."
npm install

echo "3. 构建前端静态文件..."
npm run build

echo "4. 复制前端文件到后端..."
rm -rf "../ml-prediction-system-backend-flask/app/static/frontend"
mkdir -p "../ml-prediction-system-backend-flask/app/static/frontend"
cp -r out/* "../ml-prediction-system-backend-flask/app/static/frontend/"

echo "5. 返回后端目录..."
cd "../ml-prediction-system-backend-flask"

echo "6. 检查前端文件是否复制成功..."
if [ -f "app/static/frontend/index.html" ]; then
    echo "✓ 前端文件复制成功"
else
    echo "✗ 前端文件复制失败"
    echo "请手动复制前端文件到 app/static/frontend/ 目录"
    exit 1
fi

echo "7. 启动 Flask 服务..."
python run.py 