import cv2
import os
import numpy as np

def remove_small_noise(image, min_area):
    # 连通区域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)

    # 获取每个连通区域的面积
    areas = stats[:, cv2.CC_STAT_AREA]

    # 标记为噪声的区域
    noise_labels = np.where(areas < min_area)[0]

    # 将噪声区域设为背景
    for label in noise_labels:
        labels[labels == label] = 0

    # 重新构建二值图像
    result = np.where(labels > 0, 255, 0).astype(np.uint8)

    return result

# 文件夹路径
folder_path = r"G:\FX17-yunnan1\daxi\hps\ym"

# 输出文件夹路径
output_folder_path = r"G:\FX17-yunnan1\daxi\hps\ym_removesmall"

# 最小面积阈值
min_area = 100

# 遍历文件夹中的图像文件
for filename in os.listdir(folder_path):
    # 构建图像文件的完整路径
    file_path = os.path.join(folder_path, filename)

    # 读取图像文件
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # 对图像进行去噪声处理
    result = remove_small_noise(image, min_area)

    # 构建输出文件的完整路径
    output_file_path = os.path.join(output_folder_path, filename)

    # 保存处理后的图像
    cv2.imwrite(output_file_path, result)

    print(f"Processed {filename}")

print("All images processed successfully.")