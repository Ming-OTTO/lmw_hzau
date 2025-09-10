import numpy as np
import matplotlib.pyplot as plt
from spectral import envi, open_image
import os
import re
from tqdm import tqdm


class HyperspectralProcessor:
    def __init__(self, data_path, dark_path=None, white_path=None, file_format='envi'):
        """
        初始化高光谱数据处理器

        参数:
        data_path: 高光谱数据文件路径
        dark_path: 暗电流校正文件路径
        white_path: 白板校正文件路径
        file_format: 文件格式，可选'envi'或'manual'
        """
        self.data_path = data_path
        self.dark_path = dark_path
        self.white_path = white_path
        self.file_format = file_format

        # 打印调试信息
        print(f"数据路径: {data_path}")
        print(f"暗电流路径: {dark_path}")
        print(f"白板路径: {white_path}")
        print(f"文件格式: {file_format}")

        # 检查文件是否存在
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")

        if dark_path and not os.path.exists(dark_path):
            print(f"警告: 暗电流文件不存在: {dark_path}")

        if white_path and not os.path.exists(white_path):
            print(f"警告: 白板文件不存在: {white_path}")

        # 读取高光谱数据
        self.data = self._load_hyperspectral_data(data_path)
        self.dark = self._load_hyperspectral_data(dark_path) if dark_path else None
        self.white = self._load_hyperspectral_data(white_path) if white_path else None

        # 检查数据是否成功加载
        print(f"数据加载状态: {self.data is not None}")
        if self.data is None:
            raise ValueError("高光谱数据加载失败")

        # 获取波长信息
        self.wavelengths = self._get_wavelengths()

        # 反射率数据初始化为None
        self.reflectance = None

        # 打印数据形状信息
        self._print_data_shapes()

    def _print_data_shapes(self):
        """打印数据形状信息"""
        if self.data:
            data_shape = self.data.load().shape if hasattr(self.data, 'load') else self.data.data.shape
            print(f"数据形状: {data_shape}")

        if self.dark:
            dark_shape = self.dark.load().shape if hasattr(self.dark, 'load') else self.dark.data.shape
            print(f"暗电流形状: {dark_shape}")

        if self.white:
            white_shape = self.white.load().shape if hasattr(self.white, 'load') else self.white.data.shape
            print(f"白板形状: {white_shape}")

    def _load_hyperspectral_data(self, path):
        """加载高光谱数据"""
        if not path or not os.path.exists(path):
            print(f"路径不存在: {path}")
            return None

        try:
            if self.file_format == 'envi':
                return self._load_envi_format(path)
            else:
                return self._load_manual_format(path)
        except Exception as e:
            print(f"无法加载数据 {path}: {e}")
            return None

    def _load_envi_format(self, path):
        """加载ENVI格式数据"""
        try:
            # 尝试作为ENVI格式打开
            if path.endswith('.hdr'):
                return open_image(path)
            elif path.endswith('.dat'):
                # 尝试寻找对应的.hdr文件
                hdr_path = os.path.splitext(path)[0] + '.hdr'
                if os.path.exists(hdr_path):
                    return open_image(hdr_path)
                else:
                    print(f"找不到对应的.hdr文件: {hdr_path}")
                    return None
            else:
                print(f"不支持的文件格式: {path}")
                return None
        except Exception as e:
            print(f"ENVI格式加载失败: {e}")
            # 尝试手动加载
            return self._load_manual_format(path)

    def _load_manual_format(self, path):
        """手动解析数据文件"""
        try:
            if path.endswith('.hdr'):
                # 读取头文件
                with open(path, 'r') as f:
                    hdr_lines = f.readlines()

                # 解析头文件中的关键参数
                metadata = {}
                in_bands_section = False
                wavelengths = []

                for line in hdr_lines:
                    line = line.strip()
                    if not line or line.startswith(';'):
                        continue

                    # 处理等号分隔的键值对
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip().lower()
                        value = value.strip()

                        if key == 'bands':
                            metadata['bands'] = int(value)
                        elif key == 'lines':
                            metadata['lines'] = int(value)
                        elif key == 'samples':
                            metadata['samples'] = int(value)
                        elif key == 'data type':
                            metadata['data type'] = int(value)
                        elif key == 'interleave':
                            metadata['interleave'] = value.lower()
                        elif key == 'byte order':
                            metadata['byte order'] = int(value)
                        elif key == 'wavelength':
                            in_bands_section = True
                            wavelengths = []

                    # 处理波长数据
                    elif in_bands_section:
                        # 提取括号内的所有数值
                        values = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                        if values:
                            wavelengths.extend([float(v) for v in values])

                        # 检查波段部分是否结束
                        if '}' in line:
                            in_bands_section = False
                            metadata['wavelength'] = wavelengths

                # 构建数据路径
                dat_path = os.path.splitext(path)[0] + '.dat'
                if not os.path.exists(dat_path):
                    print(f"找不到数据文件: {dat_path}")
                    return None

                # 根据数据类型确定numpy数据类型
                dtype_map = {
                    1: np.uint8,
                    2: np.int16,
                    3: np.int32,
                    4: np.float32,
                    5: np.float64,
                    12: np.uint16,
                    13: np.uint32,
                    14: np.int64,
                    15: np.uint64
                }
                data_type = metadata.get('data type', 4)  # 默认float32
                dtype = dtype_map.get(data_type, np.float32)

                # 确定字节序
                byte_order = metadata.get('byte order', 0)
                if byte_order == 1:
                    dtype = dtype.newbyteorder('>')  # 大端字节序

                # 获取文件大小（用于进度条）
                file_size = os.path.getsize(dat_path)
                chunk_size = 1024 * 1024  # 1MB块大小

                # 使用进度条读取大数据文件
                print(f"正在读取数据文件: {dat_path}")
                data_chunks = []
                with open(dat_path, 'rb') as f:
                    with tqdm(total=file_size, unit='B', unit_scale=True, desc="读取数据") as pbar:
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            data_chunk = np.frombuffer(chunk, dtype=dtype)
                            data_chunks.append(data_chunk)
                            pbar.update(len(chunk))

                # 合并所有块
                data = np.concatenate(data_chunks)

                # 确定数据维度
                lines = metadata.get('lines', 0)
                samples = metadata.get('samples', 0)
                bands = metadata.get('bands', 0)

                if lines * samples * bands != len(data):
                    print(f"数据维度不匹配: lines={lines}, samples={samples}, bands={bands}, data_len={len(data)}")
                    # 尝试估计维度
                    if lines == 0 and samples > 0 and bands > 0:
                        lines = len(data) // (samples * bands)
                        print(f"估计行数: {lines}")
                        if lines * samples * bands == len(data):
                            metadata['lines'] = lines
                        else:
                            print("无法正确估计数据维度")
                            return None
                    else:
                        return None

                # 根据交错格式重塑数据
                interleave = metadata.get('interleave', 'bsq')

                print("正在重塑数据维度...")
                if interleave == 'bsq':
                    # 波段顺序（Band Sequential）
                    data = data.reshape(bands, lines, samples).transpose(1, 2, 0)
                elif interleave == 'bil':
                    # 波段按行交错（Band Interleaved by Line）
                    data = data.reshape(lines, bands, samples).transpose(0, 2, 1)
                elif interleave == 'bip':
                    # 波段按像素交错（Band Interleaved by Pixel）
                    data = data.reshape(lines, samples, bands)
                else:
                    print(f"不支持的交错格式: {interleave}，默认使用BIP")
                    data = data.reshape(lines, samples, bands)

                # 存储波长信息
                if 'wavelength' in metadata:
                    self.wavelengths = np.array(metadata['wavelength'])

                # 创建一个模拟的spectral.Image对象
                class MockImage:
                    def __init__(self, data, metadata):
                        self.data = data
                        self.metadata = metadata

                    def load(self):
                        return self.data

                return MockImage(data, metadata)

            else:
                print(f"不支持的文件格式: {path}")
                return None
        except Exception as e:
            print(f"手动加载失败: {e}")
            return None

    def _get_wavelengths(self):
        """获取波长信息"""
        if self.data is None:
            return None

        if hasattr(self.data, 'metadata') and 'wavelength' in self.data.metadata:
            return np.array([float(w) for w in self.data.metadata.get('wavelength', [])])

        # 如果波长信息未加载，尝试从手动解析的数据中获取
        if hasattr(self, 'wavelengths') and self.wavelengths is not None:
            return self.wavelengths

        return None

    def _check_dimensions(self, data, reference):
        """检查两个数据数组的空间维度是否匹配"""
        if data is None or reference is None:
            return True

        data_shape = data.shape if isinstance(data, np.ndarray) else data.load().shape
        ref_shape = reference.shape if isinstance(reference, np.ndarray) else reference.load().shape

        # 只检查行数和列数
        return data_shape[0] == ref_shape[0] and data_shape[1] == ref_shape[1]

    def calculate_reflectance(self):
        """计算反射率"""
        if self.data is None:
            raise ValueError("未加载高光谱数据")

        # 将高光谱数据转换为numpy数组
        print("正在加载数据...")
        data_arr = self.data.load()
        data_shape = data_arr.shape

        # 应用暗电流校正
        if self.dark is not None:
            print("正在应用暗电流校正...")
            dark_arr = self.dark.load()

            # 检查维度匹配
            if not self._check_dimensions(dark_arr, data_arr):
                print(f"警告: 暗电流数据维度 ({dark_arr.shape}) 与主数据 ({data_shape}) 不匹配")
                print("尝试调整暗电流数据维度...")

                # 尝试适配暗电流数据
                if dark_arr.shape[0] == 1 and data_shape[0] > 1:
                    # 如果暗电流只有一行，但数据有多行，尝试复制
                    dark_arr = np.repeat(dark_arr, data_shape[0], axis=0)
                    print(f"已将暗电流数据扩展为: {dark_arr.shape}")
                else:
                    # 无法适配，使用平均值
                    dark_mean = np.mean(dark_arr, axis=0, keepdims=True)
                    dark_arr = np.repeat(dark_mean, data_shape[0], axis=0)
                    print(f"已使用暗电流数据的平均值，形状: {dark_arr.shape}")

            # 再次检查维度
            if dark_arr.shape[0:2] != data_arr.shape[0:2]:
                raise ValueError(f"暗电流数据维度 ({dark_arr.shape}) 与主数据 ({data_shape}) 不兼容")

            # 截断或扩展波段数以匹配
            if dark_arr.shape[2] > data_arr.shape[2]:
                dark_arr = dark_arr[:, :, :data_arr.shape[2]]
                print(f"截断暗电流波段数以匹配主数据: {dark_arr.shape}")
            elif dark_arr.shape[2] < data_arr.shape[2]:
                # 创建一个零数组并填充可用的暗电流数据
                new_dark = np.zeros_like(data_arr, dtype=dark_arr.dtype)
                new_dark[:, :, :dark_arr.shape[2]] = dark_arr
                dark_arr = new_dark
                print(f"扩展暗电流波段数以匹配主数据: {dark_arr.shape}")

            data_arr = data_arr - dark_arr

        # 应用白板校正
        if self.white is not None:
            print("正在应用白板校正...")
            white_arr = self.white.load()

            # 检查维度匹配
            if not self._check_dimensions(white_arr, data_arr):
                print(f"警告: 白板数据维度 ({white_arr.shape}) 与主数据 ({data_shape}) 不匹配")
                print("尝试调整白板数据维度...")

                # 尝试适配白板数据
                if white_arr.shape[0] == 1 and data_shape[0] > 1:
                    # 如果白板只有一行，但数据有多行，尝试复制
                    white_arr = np.repeat(white_arr, data_shape[0], axis=0)
                    print(f"已将白板数据扩展为: {white_arr.shape}")
                else:
                    # 无法适配，使用平均值
                    white_mean = np.mean(white_arr, axis=0, keepdims=True)
                    white_arr = np.repeat(white_mean, data_shape[0], axis=0)
                    print(f"已使用白板数据的平均值，形状: {white_arr.shape}")

            # 再次检查维度
            if white_arr.shape[0:2] != data_arr.shape[0:2]:
                raise ValueError(f"白板数据维度 ({white_arr.shape}) 与主数据 ({data_shape}) 不兼容")

            # 截断或扩展波段数以匹配
            if white_arr.shape[2] > data_arr.shape[2]:
                white_arr = white_arr[:, :, :data_arr.shape[2]]
                print(f"截断白板波段数以匹配主数据: {white_arr.shape}")
            elif white_arr.shape[2] < data_arr.shape[2]:
                # 创建一个零数组并填充可用的白板数据
                new_white = np.zeros_like(data_arr, dtype=white_arr.dtype)
                new_white[:, :, :white_arr.shape[2]] = white_arr
                white_arr = new_white
                print(f"扩展白板波段数以匹配主数据: {white_arr.shape}")

            # 应用暗电流校正到白板数据
            if self.dark is not None:
                white_arr = white_arr - dark_arr

            # 使用进度条进行反射率计算
            reflectance = np.zeros_like(data_arr, dtype=np.float32)
            lines, samples, bands = data_arr.shape

            # 按行处理以显示进度
            with tqdm(total=lines, desc="计算反射率") as pbar:
                for i in range(lines):
                    # 避免除以零
                    mask = white_arr[i] > 0
                    reflectance[i, mask] = data_arr[i, mask] / white_arr[i, mask]
                    reflectance[i, ~mask] = 0  # 处理除以零的情况
                    pbar.update(1)
        else:
            # 如果没有白板数据，假设已经是反射率数据或需要其他校正方法
            print("没有提供白板数据，跳过白板校正...")
            reflectance = data_arr.astype(np.float32)

        self.reflectance = reflectance
        return reflectance

    def extract_spectrum(self, row, col):
        """
        提取指定像元的光谱

        参数:
        row: 行索引
        col: 列索引

        返回:
        光谱数据
        """
        if self.reflectance is None:
            self.calculate_reflectance()

        # 确保索引在有效范围内
        if row < 0 or row >= self.reflectance.shape[0] or col < 0 or col >= self.reflectance.shape[1]:
            raise IndexError(f"索引超出范围: row={row}, col={col}, 数据尺寸: {self.reflectance.shape}")

        return self.reflectance[row, col, :]

    def plot_spectrum(self, row, col, ax=None, label=None, title=None):
        """
        绘制指定像元的光谱曲线

        参数:
        row: 行索引
        col: 列索引
        ax: matplotlib轴对象，如果为None则创建新的
        label: 图例标签
        title: 图表标题
        """
        spectrum = self.extract_spectrum(row, col)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(self.wavelengths, spectrum, label=label or f'像元 ({row},{col})')
        ax.set_xlabel('波长 (nm)')
        ax.set_ylabel('反射率')
        ax.set_title(title or '高光谱反射率曲线')
        ax.grid(True, linestyle='--', alpha=0.7)

        if label:
            ax.legend()

        return ax

    def export_reflectance(self, output_path):
        """
        导出反射率数据

        参数:
        output_path: 输出文件路径
        """
        if self.reflectance is None:
            self.calculate_reflectance()

        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 保存反射率数据
        try:
            if self.file_format == 'envi' and hasattr(self.data, 'metadata'):
                # 保存为ENVI格式
                print(f"正在保存反射率数据到: {output_path}")
                envi.save_image(output_path, self.reflectance, dtype=np.float32, force=True,
                                metadata=self.data.metadata)
            else:
                # 保存为numpy格式
                print(f"正在保存反射率数据到: {output_path}.npy")
                np.save(output_path, self.reflectance)
                output_path = output_path + '.npy'

            print(f"反射率数据已成功保存")
        except Exception as e:
            print(f"保存反射率数据失败: {e}")
            # 尝试备用保存方法
            try:
                print(f"尝试备用保存方法...")
                np.save(output_path, self.reflectance)
                print(f"已使用备用格式保存反射率数据至: {output_path}.npy")
            except:
                print("所有保存方法均失败")


# 使用示例
if __name__ == "__main__":
    # 设置中文字体，确保中文正常显示
    plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

    # 请根据实际情况修改以下参数
    # 数据路径
    data_path = r"E:\2025cotton3\hyperspectral\TEST3\Specim_data\202507090643\1\20250709064646.hdr"
    dark_path = r"E:\2025cotton3\hyperspectral\TEST3\Specim_data\202507090643\DarkCurrect\DarkCurrect.hdr"
    white_path = r"E:\2025cotton3\hyperspectral\TEST3\Specim_data\2600弱光白板给1\WhiteBoard\WhiteBoard.hdr"

    # 文件格式，可选'envi'或'manual'
    file_format = 'envi'

    try:
        # 创建处理器实例
        processor = HyperspectralProcessor(data_path, dark_path, white_path, file_format)

        # 计算反射率
        reflectance = processor.calculate_reflectance()
        print(f"反射率数据形状: {reflectance.shape}")

        # 提取并绘制特定像元的光谱
        # 选择感兴趣的像元位置（确保在有效范围内）
        row, col = 100, 100
        processor.plot_spectrum(row, col)

        # 批量绘制多个像元的光谱
        pixels = [(50, 50), (100, 100), (150, 150)]
        fig, ax = plt.subplots(figsize=(10, 6))
        for r, c in pixels:
            processor.plot_spectrum(r, c, ax=ax, label=f'像元 ({r},{c})')

        plt.tight_layout()
        plt.show()

        # 导出反射率数据
        output_path = r"H:\lmw_python_pr\高光谱\output\reflectance"
        processor.export_reflectance(output_path)

    except Exception as e:
        print(f"程序运行出错: {e}")