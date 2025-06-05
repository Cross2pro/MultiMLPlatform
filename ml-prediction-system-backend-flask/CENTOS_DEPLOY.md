# UHPC接缝抗剪承载力预测系统 - CentOS部署指南

本文档提供在CentOS服务器上部署UHPC接缝抗剪承载力预测系统后端的详细步骤。

## 前置条件

确保您的CentOS服务器上已安装以下软件：

- Python 3.6+
- pip
- Git (用于克隆代码仓库)
- Supervisor

```bash
# 安装依赖
sudo yum update
sudo yum install -y python3 python3-pip python3-devel git
sudo yum install -y epel-release
sudo yum install -y supervisor
sudo systemctl enable supervisord
sudo systemctl start supervisord
```

## 部署步骤

### 1. 克隆代码仓库

```bash
git clone [你的代码仓库URL] /opt/ml-prediction-system
cd /opt/ml-prediction-system/ml-prediction-system-backend-flask
```

### 2. 设置部署脚本权限

```bash
chmod +x deploy.sh
```

### 3. 运行部署脚本

```bash
./deploy.sh
```

脚本将自动执行以下操作：
- 创建Python虚拟环境
- 安装所需依赖
- 配置Supervisor
- 启动服务

### 4. 检查服务状态

```bash
sudo supervisorctl status ml-prediction-system
```

## 防火墙配置

如果您的服务器开启了防火墙，请确保允许5000端口通过：

```bash
sudo firewall-cmd --permanent --add-port=5000/tcp
sudo firewall-cmd --reload
```

## 管理命令

### 查看服务状态

```bash
sudo supervisorctl status ml-prediction-system
```

### 停止服务

```bash
sudo supervisorctl stop ml-prediction-system
```

### 启动服务

```bash
sudo supervisorctl start ml-prediction-system
```

### 重启服务

```bash
sudo supervisorctl restart ml-prediction-system
```

### 查看日志

```bash
# 查看Supervisor标准输出日志
tail -f /var/log/supervisor/ml-prediction-stdout.log

# 查看Supervisor错误日志
tail -f /var/log/supervisor/ml-prediction-stderr.log

# 查看Gunicorn访问日志
tail -f /var/log/gunicorn/access.log

# 查看Gunicorn错误日志
tail -f /var/log/gunicorn/error.log
```

## 故障排除

如果服务无法启动，请检查以下可能的原因：

1. **日志目录权限**：确保日志目录具有适当的权限
   ```bash
   sudo chmod -R 755 /var/log/gunicorn
   sudo chmod -R 755 /var/log/supervisor
   ```

2. **Python虚拟环境**：确保虚拟环境正确创建并激活
   ```bash
   source venv/bin/activate
   python -V  # 应显示Python 3.6+
   ```

3. **配置文件路径**：确保supervisor.conf中的路径已正确替换

4. **端口占用**：确保5000端口未被其他服务占用
   ```bash
   sudo netstat -tulpn | grep 5000
   ```

## 性能调优

如需调整性能参数，请编辑`gunicorn_config.py`文件：

1. **工作进程数**：调整`workers`值（默认为CPU核心数*2+1）
2. **工作模式**：可更改`worker_class`为其他模式如"gevent"（需先安装gevent）
3. **超时设置**：调整`timeout`值（默认60秒）
4. **最大请求数**：调整`max_requests`值（默认1000）

修改后，重启服务生效：
```bash
sudo supervisorctl restart ml-prediction-system
``` 