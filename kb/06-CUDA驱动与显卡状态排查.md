# CUDA 驱动与显卡状态排查手册

## driver/library version mismatch

执行 `nvidia-smi` 报 `Failed to initialize NVML: Driver/library version mismatch`，多发生在驱动升级后内核模块未重新加载。处理：

1. 确认是否近期升级过驱动（`dpkg -l | grep nvidia` 或询问管理员）；
2. 彻底解决需要重启服务器，或依次 `rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia` 后重新加载；
3. 重启属于高危操作，值班员只负责收集现象并联系管理员安排重启窗口，不要自行执行。

## nvidia-smi 卡住或响应极慢

`nvidia-smi` 执行超过 10 秒无输出，常见原因是某张卡的驱动状态异常或有进程处于不可中断状态（D 状态）。排查：先 `dmesg | grep -i xid` 查是否有 Xid 报错，再 `ps aux | awk '$8 ~ /D/'` 找 D 状态进程。若确认驱动挂死，记录 Xid 码后升级给管理员。

## Xid 错误码速查

内核日志中的 Xid 是显卡故障的关键线索：

- **Xid 31**：显存页错误，多为应用程序非法访存（用户代码 bug），让用户检查代码，不属于硬件故障；
- **Xid 48**：双比特 ECC 错误，属于硬件问题，该卡应停止调度并走报修流程；
- **Xid 79**：GPU has fallen off the bus，掉卡，多与供电或过热有关；断电重启后若复现即走报修流程，反复掉卡的直接停用报修。

## ECC 错误计数检查

`nvidia-smi -q -d ECC` 可查看每张卡的 ECC 错误计数，分 Volatile（本次开机以来）与 Aggregate（累计）。单卡 Volatile 双比特错误大于 0 就应告警；Aggregate 持续增长说明显存老化，建议标记该卡为观察状态，重要实验避开。

## 驱动与 CUDA 版本对应关系

驱动版本决定可用的最高 CUDA 版本（向后兼容）：如 535 驱动支持 CUDA 12.2 及以下。用户报 `CUDA driver version is insufficient for CUDA runtime version` 时，说明其环境装的 CUDA runtime 高于驱动支持上限，让用户降低 PyTorch/CUDA 版本，而不是要求升级驱动（升级驱动影响全机用户，需管理员评估）。
