# 环境配置常见问题 FAQ

## CUDA 版本不匹配

报错 `CUDA error: no kernel image is available for execution`，或 PyTorch 检测不到 GPU（`torch.cuda.is_available()` 为 False）。多为 PyTorch 编译的 CUDA 版本与驱动不匹配。用 `nvidia-smi` 看驱动支持的最高 CUDA 版本，安装对应的 PyTorch。

## conda 环境冲突

多人共用机器，`conda activate` 后包版本混乱。建议每人建独立环境，不要动 base 环境；`pip install` 前先确认已激活自己的环境。

## 找不到 nvcc / 动态库

报 `libcudart.so not found` 或 `nvcc: command not found`，多为环境变量未配置。检查 `LD_LIBRARY_PATH` 与 `PATH` 是否包含 CUDA 安装路径。

## 权限问题

安装到系统目录报 `Permission denied`，应使用用户级安装（`pip install --user`）或自己的 conda 环境，不要 sudo 装包污染全局。
