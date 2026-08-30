# Docker 容器与 GPU 使用手册

## 容器内看不到 GPU

容器里 `nvidia-smi` 报 `command not found` 或 `torch.cuda.is_available()` 为 False：

1. 启动命令必须带 `--gpus all`（或 `--gpus '"device=2,3"'` 指定卡），漏掉是最常见原因；
2. 宿主机需安装 nvidia-container-toolkit，未安装时 `--gpus` 参数会直接报错；
3. 镜像内的 CUDA runtime 版本不能高于宿主驱动支持上限（驱动向后兼容旧 CUDA）。

## DataLoader 报 bus error / 共享内存不足

容器默认共享内存只有 64MB，PyTorch DataLoader 多进程加载会报 `bus error` 或 `unable to open shared memory`。启动容器时加 `--shm-size=16g`，或将 DataLoader 的 `num_workers` 设为 0 临时绕过（会变慢）。

## 退出容器的残留清理

`docker ps -a` 中大量 Exited 容器会占用磁盘（可写层）。清理规范：

- 只清理自己名下的容器：`docker rm <容器ID>`；
- `docker system prune` 会清掉所有停止的容器和悬空镜像，**影响他人，必须在值班群确认后由管理员执行**；
- 建议启动一次性调试容器时加 `--rm`，退出自动删除。

## 容器内 root 写共享卷的属主问题

容器默认以 root 运行，往挂载的共享目录写文件会产生 root 属主文件，导致其他同学在宿主机上无法修改或删除。规范：启动时加 `-u $(id -u):$(id -g)` 以本人身份运行；已产生的 root 文件由管理员 `chown` 修复。

## 镜像构建与存储位置

镜像不要存放在系统盘（默认 /var/lib/docker），本实验室已把 docker 数据目录迁移到 /data/docker。构建大镜像前先确认 `df -h /data` 剩余空间；公共基础镜像（含常用 CUDA + PyTorch 组合）在私有 registry，优先复用而非重复构建。
