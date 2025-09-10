#!/usr/bin/env python3
"""
项目环境快速配置脚本
"""
import subprocess
import sys
import os

def run_command(cmd):
    """运行命令并处理输出"""
    print(f"运行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
    else:
        print(f"成功: {result.stdout}")
    return result.returncode == 0

def main():
    print("开始配置项目环境...")
    
    # 检查是否在conda环境中
    if 'CONDA_PREFIX' not in os.environ:
        print("请先激活conda环境: conda activate lmw_python_project")
        return
    
    # 安装依赖
    commands = [
        "python -m pip install --upgrade pip",
        "pip install paddlepaddle-gpu==2.6.1",
        "pip install -r requirements.txt",
        "pip install -e root/PaddleSeg"
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            print(f"命令执行失败: {cmd}")
            break
    
    print("\n环境配置完成！")
    print("测试命令:")
    print("python -c 'import paddle; print(paddle.__version__)'")
    print("python -c 'import paddleseg; print(paddleseg.__version__)'")

if __name__ == "__main__":
    main()