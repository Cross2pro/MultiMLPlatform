# 使用官方Python镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_RETRIES=10

# 强制替换镜像源（多种方案确保成功）
RUN sed -i 's|http://deb.debian.org|https://mirrors.163.com|g' /etc/apt/sources.list* || true && \
    sed -i 's|http://security.debian.org|https://mirrors.163.com|g' /etc/apt/sources.list* || true && \
    echo "deb https://mirrors.163.com/debian/ bookworm main" > /etc/apt/sources.list && \
    echo "deb https://mirrors.163.com/debian/ bookworm-updates main" >> /etc/apt/sources.list

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    --no-install-recommends && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 配置pip镜像源（多重备选方案）
RUN mkdir -p ~/.pip && \
    echo "[global]" > ~/.pip/pip.conf && \
    echo "timeout = 100" >> ~/.pip/pip.conf && \
    echo "retries = 10" >> ~/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple/" >> ~/.pip/pip.conf && \
    echo "extra-index-url = https://mirrors.aliyun.com/pypi/simple/" >> ~/.pip/pip.conf && \
    echo "                  https://pypi.douban.com/simple/" >> ~/.pip/pip.conf && \
    echo "                  https://mirrors.163.com/pypi/simple/" >> ~/.pip/pip.conf && \
    echo "                  https://pypi.mirrors.ustc.edu.cn/simple/" >> ~/.pip/pip.conf && \
    echo "[install]" >> ~/.pip/pip.conf && \
    echo "trusted-host = pypi.tuna.tsinghua.edu.cn" >> ~/.pip/pip.conf && \
    echo "               mirrors.aliyun.com" >> ~/.pip/pip.conf && \
    echo "               pypi.douban.com" >> ~/.pip/pip.conf && \
    echo "               mirrors.163.com" >> ~/.pip/pip.conf && \
    echo "               pypi.mirrors.ustc.edu.cn" >> ~/.pip/pip.conf

# 复制requirements.txt
COPY requirements.txt .

# 分阶段安装依赖包（避免一次性安装失败）
# 首先安装基础包
RUN pip install --no-cache-dir wheel setuptools pip --upgrade

# 安装核心依赖（分批安装，降低失败风险）
RUN pip install --no-cache-dir flask flask-cors || \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/ flask flask-cors || \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ flask flask-cors || \
    pip install --no-cache-dir -i https://pypi.douban.com/simple/ flask flask-cors

RUN pip install --no-cache-dir numpy pandas || \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/ numpy pandas || \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ numpy pandas || \
    pip install --no-cache-dir -i https://pypi.douban.com/simple/ numpy pandas

RUN pip install --no-cache-dir scikit-learn || \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/ scikit-learn || \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ scikit-learn || \
    pip install --no-cache-dir -i https://pypi.douban.com/simple/ scikit-learn

# 安装torch（特殊处理，因为包比较大）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu || \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/ torch || \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ torch || \
    pip install --no-cache-dir -i https://pypi.douban.com/simple/ torch

RUN pip install --no-cache-dir gunicorn supervisor || \
    pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple/ gunicorn supervisor || \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ gunicorn supervisor || \
    pip install --no-cache-dir -i https://pypi.douban.com/simple/ gunicorn supervisor

# 复制应用代码
COPY . .

# 创建logs目录
RUN mkdir -p logs

# 创建非root用户
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 5000

# 使用gunicorn启动应用
CMD ["gunicorn", "--config", "gunicorn_config.py", "run:app"] 