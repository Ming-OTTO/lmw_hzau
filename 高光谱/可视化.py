import numpy as np
import matplotlib.pyplot as plt

# 读取切割后的数据文件（假设为 cut_data.img）
data = np.fromfile(r'H:\lmw_python_pr\高光谱\output\20250709064646_cut_8_72_500_700.dat', dtype=np.uint16)

# BIL 格式解析：(lines, samples, bands)
lines = 628
samples = 492
bands = 224
data_reshaped = data.reshape(lines, samples, bands)

# 可视化第 1 波段（索引 0）
plt.imshow(data_reshaped[:, :, 152], cmap='gray')
plt.title('Band 0 (BIL 解析)')
plt.show()

# 可视化程序预览的 RGB 组合（假设 default bands 是 186,129,64 → 索引转换）
rgb_bands = [185, 128, 63]  # 注意：Python 索引从 0 开始，ENVI 显示从 1 开始
rgb_data = np.stack([data_reshaped[:, :, rgb_bands[0]],
                     data_reshaped[:, :, rgb_bands[1]],
                     data_reshaped[:, :, rgb_bands[2]]], axis=-1)
rgb_data = (rgb_data / rgb_data.max()) * 255  # 归一化到 0-255
plt.imshow(rgb_data.astype(np.uint8))
plt.title('程序预览 RGB 组合')
plt.show()