#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试图像拼接脚本的文件名生成功能
"""

import os
import tempfile
import shutil
from root.图像拼接L版 import pinjie

def test_filename_generation():
    """测试文件名生成功能"""
    print("🧪 测试图像拼接文件名生成功能")
    
    # 创建临时测试目录
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建测试目录结构
        test_root = os.path.join(temp_dir, "test_cotton")
        sample1_dir = os.path.join(test_root, "sample001")
        sample2_dir = os.path.join(test_root, "sample002")
        
        os.makedirs(sample1_dir, exist_ok=True)
        os.makedirs(sample2_dir, exist_ok=True)
        
        # 创建简单的测试图像（黑色图像）
        import numpy as np
        import cv2
        
        # 创建两个简单的测试图像
        img1 = np.zeros((100, 4000, 3), dtype=np.uint8)
        img2 = np.zeros((100, 4000, 3), dtype=np.uint8)
        
        # 添加一些区别以便识别
        img1[:, :2000] = 255  # 左半部分白色
        img2[:, 2000:] = 255  # 右半部分白色
        
        # 保存测试图像
        cv2.imwrite(os.path.join(sample1_dir, "img1.jpg"), img1)
        cv2.imwrite(os.path.join(sample1_dir, "img2.jpg"), img2)
        cv2.imwrite(os.path.join(sample2_dir, "img1.jpg"), img1)
        cv2.imwrite(os.path.join(sample2_dir, "img2.jpg"), img2)
        
        # 创建输出目录
        output_dir = os.path.join(temp_dir, "output")
        
        print(f"测试目录: {test_root}")
        print(f"输出目录: {output_dir}")
        
        # 运行拼接函数
        try:
            pinjie(test_root, output_dir)
            
            # 检查生成的文件
            expected_files = [
                os.path.join(output_dir, "sample001", "sample001.png"),
                os.path.join(output_dir, "sample002", "sample002.png")
            ]
            
            print("\n📁 检查生成的文件:")
            for expected_file in expected_files:
                if os.path.exists(expected_file):
                    print(f"✅ 找到: {os.path.basename(expected_file)}")
                    print(f"   路径: {expected_file}")
                else:
                    print(f"❌ 未找到: {expected_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False

if __name__ == "__main__":
    test_filename_generation()
    print("\n✅ 测试完成！")