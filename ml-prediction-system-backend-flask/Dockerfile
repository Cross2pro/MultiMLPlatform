# 使用官方Python镜像
FROM python:3.9

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 复制requirements.txt
COPY requirements.txt .

# 直接使用预编译的wheel包安装Python依赖（避免编译）
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/  -r requirements.txt --default-timeout=1000

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