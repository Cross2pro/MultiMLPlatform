#!/bin/bash

# 确保脚本停止在错误处
set -e

echo "开始部署UHPC接缝抗剪承载力预测系统后端..."

# 确保日志目录存在
sudo mkdir -p /var/log/gunicorn
sudo mkdir -p /var/log/supervisor

# 创建Python虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "创建Python虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt

# 创建Supervisor配置目录（如果不存在）
sudo mkdir -p /etc/supervisor/conf.d

# 更新Supervisor配置
echo "配置Supervisor..."
# 替换配置文件中的路径为实际路径
CURRENT_DIR=$(pwd)
sed "s|/path/to/venv|$CURRENT_DIR/venv|g" supervisor.conf | \
sed "s|/path/to/ml-prediction-system-backend-flask|$CURRENT_DIR|g" | \
sudo tee /etc/supervisor/conf.d/ml-prediction.conf > /dev/null

# 重新加载Supervisor配置
echo "重新加载Supervisor配置..."
sudo supervisorctl reread
sudo supervisorctl update

# 启动服务
echo "启动服务..."
sudo supervisorctl start ml-prediction-system

echo "部署完成！后端服务已启动。"
echo "可以使用 'sudo supervisorctl status ml-prediction-system' 检查服务状态"
echo "或使用 'sudo supervisorctl stop ml-prediction-system' 停止服务" 