# Jupyter 与远程开发排查手册

## Jupyter kernel 频繁挂掉或重启

- kernel 占用内存超过 cgroup 配额被杀：`dmesg` 里能看到 OOM 记录，减小数据加载量或申请提高配额；
- notebook 里加载了大数据集又反复重跑 cell，内存只增不减：重启 kernel 释放，长期方案是把预处理挪到脚本里；
- kernel 空闲超过 24 小时会被平台自动回收，属正常策略，重开即可。

## Jupyter 占着显存不释放

调试完没关的 kernel 是显存残留的头号来源。notebook 页面关掉不等于 kernel 退出，必须在 Jupyter 首页 Running 标签里 shutdown，或 `jupyter notebook list` 找到进程后退出。值班中发现无输出但占显存的 jupyter_kernel 进程，按占卡冲突流程联系本人。

## 端口转发与访问问题

- 服务器上的 Jupyter 只监听 localhost，本地访问用 SSH 隧道：`ssh -L 8888:localhost:8888 user@server`；
- 打开页面要求 token：看启动日志里的 token，或 `jupyter notebook list` 查看；
- 端口被占用（`Address already in use`）：换 `--port 8899` 或找出旧实例关掉。

## VSCode Remote-SSH 问题

- 连接后一直转圈：多为服务器端 `.vscode-server` 损坏，删除 `~/.vscode-server` 重连自动重装；
- `.vscode-server` 及扩展占满 home 配额：把该目录迁移到 /data 个人区并软链接回来；
- 服务器上残留大量 `vscode-server` 进程占内存：本地 VSCode 关闭窗口不会杀远端进程，累积后 `pkill -u $(whoami) -f vscode-server` 清理自己的。

## 长任务不要放 notebook

notebook 依赖浏览器会话，断连或 kernel 回收会丢运行状态。超过 1 小时的训练一律转成脚本，放 tmux 里跑并输出日志（参见 GPU 使用规范）。notebook 只用于探索和可视化。
