#!/bin/bash

# Docker部署脚本
# 使用方法: ./deploy.sh [环境] [选项]
# 环境: dev, prod
# 选项: --build, --down, --logs

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Docker是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
}

# 环境设置
ENVIRONMENT=${1:-dev}
COMPOSE_FILE="docker-compose.yml"

if [ "$ENVIRONMENT" = "prod" ]; then
    COMPOSE_FILE="docker-compose.prod.yml"
fi

print_info "使用环境: $ENVIRONMENT"
print_info "使用配置文件: $COMPOSE_FILE"

# 解析命令行参数
BUILD_FLAG=""
DOWN_FLAG=""
LOGS_FLAG=""

for arg in "$@"; do
    case $arg in
        --build)
            BUILD_FLAG="--build"
            shift
            ;;
        --down)
            DOWN_FLAG="true"
            shift
            ;;
        --logs)
            LOGS_FLAG="true"
            shift
            ;;
    esac
done

# 主要功能函数
build_and_deploy() {
    print_info "开始构建和部署..."
    
    # 停止现有容器
    docker-compose -f $COMPOSE_FILE down
    
    # 清理未使用的镜像（可选）
    print_info "清理未使用的Docker镜像..."
    docker image prune -f
    
    # 构建并启动服务
    print_info "构建并启动服务..."
    docker-compose -f $COMPOSE_FILE up -d $BUILD_FLAG
    
    print_info "等待服务启动..."
    sleep 10
    
    # 检查服务状态
    check_services
}

stop_services() {
    print_info "停止所有服务..."
    docker-compose -f $COMPOSE_FILE down
    print_info "服务已停止"
}

show_logs() {
    print_info "显示服务日志..."
    docker-compose -f $COMPOSE_FILE logs -f
}

check_services() {
    print_info "检查服务状态..."
    
    # 检查后端健康状态
    if curl -f http://localhost:5000/health &> /dev/null; then
        print_info "✅ 后端服务运行正常"
    else
        print_warning "⚠️  后端服务可能未就绪"
    fi
    
    # 检查前端
    if curl -f http://localhost:3000 &> /dev/null; then
        print_info "✅ 前端服务运行正常"
    else
        print_warning "⚠️  前端服务可能未就绪"
    fi
    
    # 检查Nginx
    if curl -f http://localhost &> /dev/null; then
        print_info "✅ Nginx代理运行正常"
    else
        print_warning "⚠️  Nginx代理可能未就绪"
    fi
    
    print_info "服务状态检查完成"
    docker-compose -f $COMPOSE_FILE ps
}

# 主逻辑
check_docker

if [ "$DOWN_FLAG" = "true" ]; then
    stop_services
elif [ "$LOGS_FLAG" = "true" ]; then
    show_logs
else
    build_and_deploy
fi

print_info "部署完成！"
print_info "访问地址:"
print_info "  - 应用主页: http://localhost"
print_info "  - 前端直接访问: http://localhost:3000"
print_info "  - 后端API: http://localhost:5000" 