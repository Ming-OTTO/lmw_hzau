#!/usr/bin/env python3
"""
环境验证脚本
"""
import sys
import importlib

def check_package(package_name):
    """检查包是否已安装"""
    try:
        importlib.import_module(package_name)
        print(f"✓ {package_name} 已安装")
        return True
    except ImportError:
        print(f"✗ {package_name} 未安装")
        return False

def main():
    print("环境验证报告")
    print("=" * 50)
    
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查核心包
    packages = [
        'numpy', 'cv2', 'matplotlib', 'scipy', 'sklearn', 'pandas',
        'paddle', 'paddleseg', 'PyQt5', 'yaml', 'tqdm', 'PIL'
    ]
    
    missing = []
    for pkg in packages:
        if not check_package(pkg):
            missing.append(pkg)
    
    print("\n" + "=" * 50)
    if missing:
        print(f"缺失的包: {', '.join(missing)}")
        print("请运行: python quick_setup.py")
    else:
        print("所有包都已安装完成！")

if __name__ == "__main__":
    main()