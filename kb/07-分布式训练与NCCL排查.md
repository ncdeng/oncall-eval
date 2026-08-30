# 分布式训练与 NCCL 排查手册

## NCCL timeout 排查顺序

多机训练报 `NCCL communication timeout` 时，超时本身通常是表象，真正原因是某个 rank 进程先挂了，其余 rank 在集合通信处等待超时。排查顺序：

1. 对比各 rank 日志的最后输出时间，**先停止输出的那个 rank 所在节点是嫌疑节点**；
2. 到嫌疑节点查 `dmesg`（是否被 OOM 杀掉）、查进程退出码（137 = 被 SIGKILL，多为 OOM 或人为 kill）；
3. 网络问题放最后排查：两节点互 ping、检查带宽，NCCL 因纯网络故障超时的比例远低于进程先挂。

## NCCL 调试环境变量

- `NCCL_DEBUG=INFO`：打印 NCCL 初始化与通信细节，定位在哪一步卡住；
- `NCCL_SOCKET_IFNAME=eth0`：多网卡机器指定通信网卡，避免 NCCL 选错网卡（如选到 docker0 虚拟网卡导致不通）；
- `NCCL_IB_DISABLE=1`：临时禁用 InfiniBand 改走以太网，用于排除 IB 链路问题。

## rendezvous 失败与端口占用

启动即报 `Address already in use`，是 `MASTER_PORT` 被占用：`ss -tlnp | grep <端口>` 找到占用进程，换端口重跑即可。若报 `Connection refused`，检查 MASTER_ADDR 是否写错、主节点进程是否先启动、防火墙是否放行该端口。

## 多机环境一致性检查

多机训练要求各节点：CUDA/NCCL/PyTorch 版本一致、`/etc/hosts` 中主机名解析一致、共享存储挂载路径一致。版本不一致时经常表现为 hang 在初始化阶段而非明确报错。排查口诀：**先单机多卡跑通，再上多机**，单机能跑说明代码没问题，问题在环境或网络。

## 训练 hang 住无报错

所有 rank 都活着但长时间无输出：多为某 rank 数据加载阻塞（共享盘 IO 慢）或集合通信死锁（各 rank 执行了不同的通信次序，常见于条件分支里有 all_reduce）。可用 `py-spy dump --pid <PID>` 看各 rank 卡在哪一行。
