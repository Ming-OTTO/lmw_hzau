import spectral
import numpy as np
import cv2
import os

# 设置文件路径
file_path = r"F:\2025cotton3\hyperspectral\TEST3\Specim_data\202507090643\1\20250709064646.hdr"

# 打开图像文件
img = spectral.open_image(file_path)

# 获取图像的形状
rows, cols, bands = img.shape

# 指定要遍历的波段范围
start_band = 0
end_band = bands - 1

# 创建一个目录来保存结果图像（如果不存在）
# output_dir1 = r"J:\yunnan\1207\yunnan1207pot1\yunnanpot1\verify_001\results1"
# os.makedirs(output_dir1, exist_ok=True)
output_dir2 = r"F:\data3"
os.makedirs(output_dir2, exist_ok=True)

# 遍历波段对
for numerator_band in range(start_band, end_band):
    for denominator_band in range(start_band, end_band):
        if numerator_band != denominator_band:
            # 执行安全的除法操作
            with np.errstate(divide='ignore', invalid='ignore'):
                selected_bands_division = np.divide(img[:, :, numerator_band].astype(float),
                                                   img[:, :, denominator_band].astype(float))
                # 处理无穷大和NaN值
                selected_bands_division[~np.isfinite(selected_bands_division)] = 0

            # 归一化到0-255范围
            min_val = np.min(selected_bands_division)
            max_val = np.max(selected_bands_division)
            normalized = ((selected_bands_division - min_val) / (max_val - min_val)) * 255
            normalized = normalized.astype(np.uint8)

            # 自动选择阈值并进行二值化
            _, thresholded = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 反转二值化结果（如果需要）
            thresholded2 = 255 - thresholded

            # 保存结果图像
            # output_path1 = os.path.join(output_dir1, f"band_{numerator_band + 1}_div_band_{denominator_band + 1}.png")
            # cv2.imwrite(output_path1, normalized)
            # print(f"Saved: {output_path1}")
            output_path2 = os.path.join(output_dir2, f"band_{numerator_band+1}_div_band_{denominator_band+1}_OTSU.png")
            cv2.imwrite(output_path2, thresholded)
            print(f"Saved: {output_path2}")