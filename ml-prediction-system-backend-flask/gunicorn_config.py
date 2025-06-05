import multiprocessing

# Gunicorn配置

# 绑定IP和端口
bind = "0.0.0.0:5000"

# 工作进程数量 - 通常设置为CPU核心数或2-4倍CPU核心数
workers = multiprocessing.cpu_count() * 2 + 1

# 工作模式 - 异步工作模式，提高并发处理能力
worker_class = "sync"

# 超时设置（秒）
timeout = 60

# 最大请求数，超过后worker会重启（避免内存泄漏）
max_requests = 1000
max_requests_jitter = 50

# 日志配置
errorlog = "/var/log/gunicorn/error.log"
accesslog = "/var/log/gunicorn/access.log"
loglevel = "info"

# 守护进程模式 - 设置为False，因为Supervisor会接管守护进程功能
daemon = False

# 进程名称前缀
proc_name = "ml-prediction-system-backend" 