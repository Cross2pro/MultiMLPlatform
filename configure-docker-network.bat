@echo off
echo 正在配置 Docker 网络设置以解决镜像源连接问题...

REM 创建 Docker daemon 配置目录
if not exist "%USERPROFILE%\.docker" mkdir "%USERPROFILE%\.docker"

REM 创建 Docker daemon.json 配置文件
echo { > "%USERPROFILE%\.docker\daemon.json"
echo   "registry-mirrors": [ >> "%USERPROFILE%\.docker\daemon.json"
echo     "https://registry.cn-hangzhou.aliyuncs.com", >> "%USERPROFILE%\.docker\daemon.json"
echo     "https://docker.mirrors.ustc.edu.cn", >> "%USERPROFILE%\.docker\daemon.json"
echo     "https://hub-mirror.c.163.com", >> "%USERPROFILE%\.docker\daemon.json"
echo     "https://mirror.baidubce.com" >> "%USERPROFILE%\.docker\daemon.json"
echo   ], >> "%USERPROFILE%\.docker\daemon.json"
echo   "dns": ["8.8.8.8", "114.114.114.114"], >> "%USERPROFILE%\.docker\daemon.json"
echo   "max-concurrent-downloads": 3, >> "%USERPROFILE%\.docker\daemon.json"
echo   "max-concurrent-uploads": 3, >> "%USERPROFILE%\.docker\daemon.json"
echo   "experimental": false >> "%USERPROFILE%\.docker\daemon.json"
echo } >> "%USERPROFILE%\.docker\daemon.json"

echo Docker 网络配置已更新！
echo 请重启 Docker Desktop 以应用新配置。
echo.
echo 配置文件位置: %USERPROFILE%\.docker\daemon.json
echo.
pause 