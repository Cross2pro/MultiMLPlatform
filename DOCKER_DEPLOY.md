# Docker 部署指南

## 项目结构

```
.
├── ml-prediction-system/           # Next.js 前端
│   ├── Dockerfile
│   └── .dockerignore
├── ml-prediction-system-backend-flask/  # Flask 后端
│   ├── Dockerfile
│   └── .dockerignore
├── docker-compose.yml             # 开发环境配置
├── docker-compose.prod.yml        # 生产环境配置
├── nginx.conf                     # Nginx 配置
├── deploy.sh                      # 部署脚本
└── DOCKER_DEPLOY.md              # 本文档
```

## 快速开始

### 1. 环境要求

- Docker 20.10+
- Docker Compose 2.0+
- 服务器内存至少 4GB

### 2. 快速部署

```bash
# 克隆项目后，进入项目根目录
cd /path/to/your/project

# 给部署脚本执行权限
chmod +x deploy.sh

# 开发环境部署
./deploy.sh dev

# 生产环境部署
./deploy.sh prod
```

### 3. 访问应用

- **主应用**: http://your-server-ip
- **前端直接访问**: http://your-server-ip:3000
- **后端API**: http://your-server-ip:5000

## 详细部署说明

### 开发环境部署

```bash
# 构建并启动所有服务
docker-compose up -d --build

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 生产环境部署

```bash
# 使用生产配置
docker-compose -f docker-compose.prod.yml up -d --build

# 或使用部署脚本
./deploy.sh prod
```

## 部署脚本使用

```bash
# 基本部署
./deploy.sh                    # 默认开发环境
./deploy.sh dev               # 开发环境
./deploy.sh prod              # 生产环境

# 强制重新构建
./deploy.sh dev --build
./deploy.sh prod --build

# 停止服务
./deploy.sh dev --down
./deploy.sh prod --down

# 查看日志
./deploy.sh dev --logs
./deploy.sh prod --logs
```

## 服务架构

### 服务组件

1. **Frontend (Next.js)**
   - 端口: 3000
   - 容器名: ml-frontend / ml-frontend-prod

2. **Backend (Flask)**
   - 端口: 5000
   - 容器名: ml-backend / ml-backend-prod

3. **Nginx (反向代理)**
   - 端口: 80, 443
   - 容器名: ml-nginx / ml-nginx-prod

### 网络配置

- 所有服务运行在 `ml-network` 网络中
- 前端通过 `/api` 路径访问后端
- Nginx 提供反向代理和负载均衡

## 环境变量配置

### 前端环境变量

```env
NODE_ENV=production
NEXT_PUBLIC_API_URL=/api
```

### 后端环境变量

```env
FLASK_ENV=production
FLASK_DEBUG=False
```

## SSL/HTTPS 配置

### 1. 准备SSL证书

```bash
# 创建SSL证书目录
mkdir ssl

# 将您的SSL证书文件放入ssl目录
# ssl/cert.pem - 证书文件
# ssl/key.pem  - 私钥文件
```

### 2. 修改Nginx配置

取消 `nginx.conf` 中HTTPS配置的注释，并修改域名：

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;  # 修改为您的域名
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    # ... 其他配置
}
```

## 监控和日志

### 查看服务状态

```bash
# 查看所有容器状态
docker-compose ps

# 查看特定服务状态
docker-compose ps backend
```

### 查看日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f nginx
```

### 健康检查

所有服务都配置了健康检查：

- **Backend**: `curl -f http://localhost:5000/health`
- **Frontend**: `curl -f http://localhost:3000`

## 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 检查端口占用
   netstat -tulpn | grep :80
   netstat -tulpn | grep :3000
   netstat -tulpn | grep :5000
   ```

2. **内存不足**
   ```bash
   # 查看内存使用
   docker stats
   
   # 清理未使用的资源
   docker system prune -a
   ```

3. **服务启动失败**
   ```bash
   # 查看详细错误信息
   docker-compose logs [service-name]
   
   # 重新构建特定服务
   docker-compose build [service-name]
   ```

### 性能优化

1. **镜像优化**
   - 使用多阶段构建减小镜像大小
   - 利用 `.dockerignore` 排除不必要文件

2. **资源限制**
   - 生产环境配置了CPU和内存限制
   - 可根据服务器配置调整资源分配

3. **缓存配置**
   - Nginx配置了静态文件缓存
   - 使用CDN可进一步提升性能

## 备份和恢复

### 数据备份

```bash
# 备份日志数据
docker run --rm -v ml-prediction-system_logs_data:/data -v $(pwd):/backup alpine tar czf /backup/logs_backup.tar.gz /data

# 备份整个应用
tar czf app_backup.tar.gz .
```

### 数据恢复

```bash
# 恢复日志数据
docker run --rm -v ml-prediction-system_logs_data:/data -v $(pwd):/backup alpine tar xzf /backup/logs_backup.tar.gz -C /
```

## 扩展部署

### 多实例部署

可以通过修改 `docker-compose.yml` 实现服务的水平扩展：

```yaml
services:
  backend:
    # ... 其他配置
    deploy:
      replicas: 3  # 启动3个后端实例
```

### 集群部署

对于大规模部署，建议使用 Docker Swarm 或 Kubernetes。

## 安全建议

1. **网络安全**
   - 使用防火墙限制不必要的端口访问
   - 配置SSL/TLS加密

2. **容器安全**
   - 使用非root用户运行容器
   - 定期更新基础镜像

3. **数据安全**
   - 定期备份重要数据
   - 使用安全的密钥管理

## 联系支持

如有部署问题，请检查：
1. Docker和Docker Compose版本
2. 服务器资源是否充足
3. 网络连接是否正常
4. 查看详细的错误日志 