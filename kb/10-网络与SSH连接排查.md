# 网络与 SSH 连接排查手册

## SSH 连不上的排查顺序

1. `ping <服务器IP>`：不通先查本机网络与 VPN 状态（校外访问必须先连校园 VPN）；
2. ping 通但 SSH 超时：`telnet <IP> 22` 测端口，不通可能是防火墙或 sshd 挂了；
3. 端口通但登录无响应：多为服务器负载过高（load 飙升、内存耗尽导致 sshd 无法 fork），此时走带外管理（BMC/IPMI）或联系机房，属于 P0 级问题；
4. 报 `Connection refused`：sshd 未运行或端口改过，升级管理员。

## SSH 登录慢（等 10 秒以上才出提示）

两个经典原因：

- **DNS 反向解析超时**：服务器端 `sshd_config` 的 `UseDNS` 未关。值班员可建议用户临时忍受，根治需管理员改配置；
- **GSSAPI 认证尝试**：客户端连接加 `-o GSSAPIAuthentication=no` 立即生效。

## 断连与会话保活

- 训练任务必须放 `tmux`/`screen`，SSH 断开不影响任务（参见 GPU 使用规范）；
- 频繁断连可在客户端配置 `ServerAliveInterval 60`；
- 校园网夜间闲置回收导致的断连属于正常现象，重连即可。

## VPN 与 Docker 网段冲突

连上 VPN 后反而访问不了服务器：常见于 VPN 分配网段与服务器 docker0 默认网段（172.17.0.0/16）冲突，路由被容器网桥劫持。处理：管理员修改 docker `bip` 配置改用 172.31 网段；用户侧临时方案是断开 VPN 用校内网络直连。

## 下载与镜像源加速

服务器出口带宽有限，大流量下载请错峰：

- pip 换清华源：`pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`；
- conda 配置 tuna 镜像 channels；
- apt 源由管理员统一配置，不要自行修改 `/etc/apt/sources.list`；
- HuggingFace 模型下载走镜像站（详见模型下载与缓存管理手册）。
