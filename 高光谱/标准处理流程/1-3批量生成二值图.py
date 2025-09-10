import spectral
import numpy as np
import cv2
import os

# 设置父文件夹路径和输出文件夹路径
parent_dir = r"I:\熊海燕老师\20250308\cameraPath\verifyDir"  # 包含所有子文件夹的父文件夹
output_dir = r"I:\熊海燕老师\20250308\cameraPath\verifyDir\verify_001\20250308091737"  # 输出文件夹
os.makedirs(output_dir, exist_ok=True)

# 遍历父文件夹中的所有子文件夹
for subdir in os.listdir(parent_dir):
    subdir_path = os.path.join(parent_dir, subdir)
    # 确保是文件夹
    if os.path.isdir(subdir_path):
        # 查找子文件夹中的 .hdr 文件
        hdr_files = [f for f in os.listdir(subdir_path) if f.endswith('.hdr')]

        for hdr_file in hdr_files:
            hdr_path = os.path.join(subdir_path, hdr_file)
            file_name = hdr_file.split('.')[0]
            #print(hdr_file)

            # 打开图像文件（延迟加载，不立即读取数据）
            img = spectral.open_image(hdr_path)
            band1 = 8
            band2 = 143

            # 优化1：直接读取所需波段并转换为float32减少内存占用
            band1_data = np.ascontiguousarray(img[:, :, band1].astype(np.float32))
            band2_data = np.ascontiguousarray(img[:, :, band2].astype(np.float32))

            # 优化2：使用预分配数组和条件除法避免后续无效值处理
            selected_bands_division = np.zeros_like(band1_data, dtype=np.float32)
            np.divide(band1_data, band2_data,
                      out=selected_bands_division,
                      where=band2_data != 0)

            # 优化3：使用向量化操作处理剩余无效值
            np.nan_to_num(selected_bands_division, copy=False, nan=0, posinf=0, neginf=0)

            # 优化4：优化归一化计算流程
            min_val = np.min(selected_bands_division)
            max_val = np.max(selected_bands_division)

            if max_val > min_val:  # 避免除零和多余计算
                scale = 255.0 / (max_val - min_val)
                # 使用原地操作减少内存分配
                np.subtract(selected_bands_division, min_val, out=selected_bands_division)
                np.multiply(selected_bands_division, scale, out=selected_bands_division)
                normalized = np.clip(selected_bands_division, 0, 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(selected_bands_division, dtype=np.uint8)

            # 优化5：使用OTSU阈值优化的同时进行二值化
            _, thresholded = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 保存结果
            output_subdir = os.path.join(output_dir, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            ym_name = file_name + '.png'
            output_path = os.path.join(output_subdir, ym_name)
            cv2.imwrite(output_path, thresholded)
            print(f"优化结果已保存至：{output_path}")