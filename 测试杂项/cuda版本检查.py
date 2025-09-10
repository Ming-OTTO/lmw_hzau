#!/usr/bin/env python3
# check_cuda.py
import subprocess
import sys
import os
import shutil

def run_cmd(cmd: str) -> str:
    """执行 shell 命令并返回输出；失败返回空字符串。"""
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return ""

def main():
    print("======== 当前 CUDA / PyTorch 版本检测 ========\n")

    # 1. 全局 CUDA 版本（来自 nvidia-smi）
    nvidia_smi = shutil.which("nvidia-smi") or \
                 (r"C:\Windows\System32\nvidia-smi.exe" if os.name == "nt" else None)
    if nvidia_smi:
        # 从 nvidia-smi 标准输出里抓 “CUDA Version”
        smi_text = run_cmd(f'"{nvidia_smi}"') or ""
        for line in smi_text.splitlines():
            if "CUDA Version" in line:
                # 例如 |    CUDA Version      : 12.2     |
                ver = line.split(":")[-1].strip()
                print(f"驱动级 CUDA 版本 (nvidia-smi) : {ver}")
                break
        else:
            print("nvidia-smi 输出中未找到 CUDA 版本信息")
    else:
        print("nvidia-smi 未找到，请确认 NVIDIA 驱动已正确安装")

    # 2. PyTorch 相关信息（可选）
    print()
    try:
        import torch
        print(f"PyTorch 版本 : {torch.__version__}")
        print(f"PyTorch 自带 CUDA 版本 : {torch.version.cuda or 'CPU-only'}")
        print(f"PyTorch 检测 GPU 可用 : {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch 未安装，跳过 PyTorch 相关检测")

    print("\n======== 检测完成 ========")

if __name__ == "__main__":
    main()