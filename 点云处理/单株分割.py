import os
import numpy as np
import open3d as o3d
import cv2
from scipy import ndimage
from skimage import measure, morphology
from osgeo import gdal
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List


class TreeSegmentationPipeline:
    def __init__(self, config: Dict = None):
        """初始化树木分割管道，配置参数"""
        self.config = config or {
            'voxel_size': 0.05,
            'radius_outlier_nb_points': 16,
            'radius_outlier_radius': 0.05,
            'height_threshold': 0.5,
            'min_tree_height': 2.0,
            'watershed_mark_distance': 1.0,
            'min_tree_area': 10,
            'rgb_weight': 0.3,
            'hyperspectral_weight': 0.5
        }

    def process(self, point_cloud_path: str,
                rgb_path: Optional[str] = None,
                hyperspectral_path: Optional[str] = None) -> Dict:
        """处理点云数据，结合RGB或高光谱图像进行单株分割"""
        # 加载点云
        pcd = self.load_point_cloud(point_cloud_path)

        # 点云预处理
        pcd_filtered = self.preprocess_point_cloud(pcd)

        # 提取地面点和非地面点
        ground_cloud, non_ground_cloud = self.segment_ground(pcd_filtered)

        # 生成冠层高度模型
        chm = self.create_canopy_height_model(non_ground_cloud)

        # 结合图像数据（如果提供）
        if rgb_path:
            rgb_image = self.load_rgb_image(rgb_path)
            boundary_probability = self.extract_boundaries_from_rgb(rgb_image)
            chm = self.fuse_chm_with_image(chm, boundary_probability, self.config['rgb_weight'])

        if hyperspectral_path:
            hyperspectral_data = self.load_hyperspectral_data(hyperspectral_path)
            boundary_probability = self.extract_boundaries_from_hyperspectral(hyperspectral_data)
            chm = self.fuse_chm_with_image(chm, boundary_probability, self.config['hyperspectral_weight'])

        # 应用分水岭算法进行分割
        labels = self.apply_watershed_segmentation(chm)

        # 后处理和提取单株树木
        tree_instances = self.postprocess_segmentation(labels, chm)

        # 将分割结果映射回点云
        segmented_pcd = self.map_segmentation_to_point_cloud(non_ground_cloud, labels, chm)

        return {
            'canopy_height_model': chm,
            'segmentation_labels': labels,
            'tree_instances': tree_instances,
            'segmented_point_cloud': segmented_pcd
        }

    def load_point_cloud(self, path: str) -> o3d.geometry.PointCloud:
        """加载点云数据"""
        try:
            pcd = o3d.io.read_point_cloud(path)
            print(f"成功加载点云数据: {len(pcd.points)} 个点")
            return pcd
        except Exception as e:
            print(f"加载点云失败: {e}")
            raise

    def preprocess_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """点云预处理：降采样和去除离群点"""
        # 降采样
        pcd_down = pcd.voxel_down_sample(voxel_size=self.config['voxel_size'])

        # 去除离群点
        pcd_filtered, ind = pcd_down.remove_radius_outlier(
            nb_points=self.config['radius_outlier_nb_points'],
            radius=self.config['radius_outlier_radius']
        )

        print(f"预处理后点云: {len(pcd_filtered.points)} 个点")
        return pcd_filtered

    def segment_ground(self, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
        """使用RANSAC算法分割地面点和非地面点"""
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.03,
            ransac_n=3,
            num_iterations=1000
        )

        ground_cloud = pcd.select_by_index(inliers)
        non_ground_cloud = pcd.select_by_index(inliers, invert=True)

        print(f"地面点数量: {len(ground_cloud.points)}, 非地面点数量: {len(non_ground_cloud.points)}")
        return ground_cloud, non_ground_cloud

    def create_canopy_height_model(self, pcd: o3d.geometry.PointCloud,
                                   resolution: float = 0.1) -> np.ndarray:
        """从点云创建冠层高度模型(CHM)"""
        points = np.asarray(pcd.points)

        # 计算边界
        x_min, y_min = np.min(points[:, :2], axis=0) - 1
        x_max, y_max = np.max(points[:, :2], axis=0) + 1

        # 创建网格
        x_grid = np.arange(x_min, x_max, resolution)
        y_grid = np.arange(y_min, y_max, resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)

        # 初始化CHM为零
        chm = np.zeros_like(xx)

        # 对每个网格单元，找到最高点
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                mask = (points[:, 0] >= x_grid[j]) & (points[:, 0] < x_grid[j] + resolution) & \
                       (points[:, 1] >= y_grid[i]) & (points[:, 1] < y_grid[i] + resolution)

                if np.sum(mask) > 0:
                    chm[i, j] = np.max(points[mask, 2])

        # 插值填充空洞
        mask = np.isnan(chm)
        chm[mask] = ndimage.gaussian_filter(chm, sigma=1, mode='nearest')[mask]

        return chm

    def load_rgb_image(self, path: str) -> np.ndarray:
        """加载RGB图像"""
        try:
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            print(f"加载RGB图像失败: {e}")
            raise

    def load_hyperspectral_data(self, path: str) -> np.ndarray:
        """加载高光谱数据"""
        try:
            # 假设输入是.hdr文件
            dataset = gdal.Open(path)
            if dataset is None:
                # 尝试加载.dat文件
                hdr_path = os.path.splitext(path)[0] + '.hdr'
                dataset = gdal.Open(hdr_path)

            if dataset is None:
                raise ValueError("无法加载高光谱数据")

            bands = []
            for i in range(dataset.RasterCount):
                band = dataset.GetRasterBand(i + 1)
                bands.append(band.ReadAsArray())

            hyperspectral_data = np.stack(bands, axis=2)
            return hyperspectral_data
        except Exception as e:
            print(f"加载高光谱数据失败: {e}")
            raise

    def extract_boundaries_from_rgb(self, image: np.ndarray) -> np.ndarray:
        """从RGB图像中提取边界信息"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)

        # 转换为概率图
        boundary_probability = edges / 255.0

        return boundary_probability

    def extract_boundaries_from_hyperspectral(self, data: np.ndarray) -> np.ndarray:
        """从高光谱数据中提取边界信息"""
        # 简单方法：使用前3个波段作为RGB
        if data.shape[2] >= 3:
            rgb = data[:, :, :3].astype(np.uint8)
            return self.extract_boundaries_from_rgb(rgb)

        # 更复杂的方法：计算NDVI并检测边界
        if data.shape[2] > 4:  # 假设至少有红光和近红外波段
            red = data[:, :, 2]  # 第3个波段通常是红光
            nir = data[:, :, 7]  # 第8个波段通常是近红外

            # 计算NDVI
            ndvi = (nir - red) / (nir + red + 1e-6)

            # 高斯滤波后计算梯度
            ndvi_smooth = ndimage.gaussian_filter(ndvi, sigma=1)
            sobel_x = ndimage.sobel(ndvi_smooth, axis=0)
            sobel_y = ndimage.sobel(ndvi_smooth, axis=1)
            gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

            # 归一化
            boundary_probability = gradient_mag / np.max(gradient_mag)
            return boundary_probability

        # 默认情况
        return np.zeros((data.shape[0], data.shape[1]))

    def fuse_chm_with_image(self, chm: np.ndarray, boundary_probability: np.ndarray, weight: float) -> np.ndarray:
        """融合冠层高度模型和图像边界信息"""
        # 调整边界概率图大小以匹配CHM
        if boundary_probability.shape != chm.shape:
            boundary_probability = cv2.resize(boundary_probability, (chm.shape[1], chm.shape[0]))

        # 平滑边界概率图
        boundary_probability = ndimage.gaussian_filter(boundary_probability, sigma=1)

        # 融合：边界区域降低CHM值，促进分割
        chm_fused = chm.copy()
        chm_fused = chm_fused - weight * boundary_probability * chm_fused

        return chm_fused

    def apply_watershed_segmentation(self, chm: np.ndarray) -> np.ndarray:
        """应用分水岭算法进行分割"""
        # 高斯平滑
        chm_smooth = ndimage.gaussian_filter(chm, sigma=1)

        # 找到局部最大值作为种子点
        local_maxi = self._detect_local_maxima(chm_smooth)

        # 标记种子点
        markers = measure.label(local_maxi)

        # 距离变换作为梯度
        distance = ndimage.distance_transform_edt(chm_smooth > self.config['height_threshold'])

        # 应用分水岭算法
        labels = morphology.watershed(-distance, markers, mask=chm_smooth > self.config['min_tree_height'])

        return labels

    def _detect_local_maxima(self, image: np.ndarray) -> np.ndarray:
        """检测局部最大值"""
        # 定义结构元素
        footprint = np.ones((5, 5))

        # 找到局部最大值
        local_max = (ndimage.maximum_filter(image, footprint=footprint) == image)

        # 确保最大值点高于阈值
        local_max &= (image > self.config['height_threshold'])

        return local_max

    def postprocess_segmentation(self, labels: np.ndarray, chm: np.ndarray) -> List[Dict]:
        """分割结果后处理"""
        # 移除小区域
        unique_labels, counts = np.unique(labels, return_counts=True)
        min_area = self.config['min_tree_area']

        tree_instances = []

        for label, count in zip(unique_labels, counts):
            if label == 0:  # 跳过背景
                continue

            if count < min_area:
                labels[labels == label] = 0
                continue

            # 计算树的属性
            mask = labels == label
            tree_height = np.max(chm[mask])
            tree_area = np.sum(mask)

            # 计算重心
            y, x = np.nonzero(mask)
            centroid_x = np.mean(x)
            centroid_y = np.mean(y)

            tree_instances.append({
                'label': label,
                'height': tree_height,
                'area': tree_area,
                'centroid': (centroid_x, centroid_y),
                'mask': mask
            })

        return tree_instances

    def map_segmentation_to_point_cloud(self, pcd: o3d.geometry.PointCloud,
                                        labels: np.ndarray, chm: np.ndarray) -> o3d.geometry.PointCloud:
        """将分割结果映射回点云"""
        points = np.asarray(pcd.points)
        x_min, y_min = np.min(points[:, :2], axis=0) - 1
        resolution = (np.max(points[:, 0]) - np.min(points[:, 0])) / chm.shape[1]

        # 为每个点分配标签
        point_labels = np.zeros(len(points), dtype=np.int32)

        for i, (x, y, z) in enumerate(points):
            col = int((x - x_min) / resolution)
            row = int((y - y_min) / resolution)

            if 0 <= row < labels.shape[0] and 0 <= col < labels.shape[1]:
                point_labels[i] = labels[row, col]

        # 创建彩色点云
        segmented_pcd = pcd.clone()
        colors = np.zeros((len(points), 3))

        # 为每个标签分配随机颜色
        for label in np.unique(point_labels):
            if label == 0:  # 背景
                continue

            mask = point_labels == label
            color = np.random.rand(3)
            colors[mask] = color

        segmented_pcd.colors = o3d.utility.Vector3dVector(colors)

        return segmented_pcd

    def visualize_results(self, results: Dict):
        """可视化分割结果"""
        # 创建一个3x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 显示冠层高度模型
        axes[0, 0].imshow(results['canopy_height_model'], cmap='terrain')
        axes[0, 0].set_title('冠层高度模型')
        axes[0, 0].axis('off')

        # 显示分割标签
        axes[0, 1].imshow(results['segmentation_labels'], cmap='tab20')
        axes[0, 1].set_title('分割结果')
        axes[0, 1].axis('off')

        # 显示几个随机树木的掩码
        tree_masks = np.zeros_like(results['segmentation_labels'])
        random_trees = np.random.choice(len(results['tree_instances']), min(5, len(results['tree_instances'])),
                                        replace=False)

        for i, idx in enumerate(random_trees):
            tree = results['tree_instances'][idx]
            tree_masks[tree['mask']] = i + 1

        axes[1, 0].imshow(tree_masks, cmap='Set1')
        axes[1, 0].set_title('随机选择的树木')
        axes[1, 0].axis('off')

        # 关闭最后一个子图
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

        # 可视化点云结果
        o3d.visualization.draw_geometries([results['segmented_point_cloud']])


if __name__ == "__main__":
    # 示例用法
    pipeline = TreeSegmentationPipeline()

    # 替换为实际文件路径
    point_cloud_path = "path/to/your/point_cloud.ply"
    rgb_path = "path/to/your/rgb_image.jpg"
    hyperspectral_path = "path/to/your/hyperspectral_data.hdr"

    try:
        results = pipeline.process(point_cloud_path, rgb_path, hyperspectral_path)
        pipeline.visualize_results(results)

        # 保存分割结果
        o3d.io.write_point_cloud("segmented_trees.ply", results['segmented_point_cloud'])
        print("分割结果已保存为 segmented_trees.ply")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")