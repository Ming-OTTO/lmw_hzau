import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def image_to_pointcloud(image_path, threshold=50, output_file=None):
    """
    将图像转换为点云数据

    参数:
    image_path (str): 输入图像的路径
    threshold (int): 二值化阈值，范围0-255
    output_file (str): 输出点云文件路径，如果为None则不保存

    返回:
    numpy.ndarray: 点云数据，形状为(n, 2)，其中n是点数，每行为[x, y]坐标
    """
    # 读取图像并转换为灰度图
    try:
        img = Image.open(image_path).convert('L')
    except FileNotFoundError:
        print(f"错误：找不到文件 {image_path}")
        return None
    except Exception as e:
        print(f"错误：打开图像时出错 - {e}")
        return None

    # 将图像转换为numpy数组
    img_array = np.array(img)

    # 二值化处理
    binary = img_array < threshold

    # 获取非零点的坐标
    points = np.column_stack(np.where(binary))

    # 转换坐标系统：图像坐标(y为行，x为列) -> 笛卡尔坐标(x为水平，y为垂直)
    # 同时调整坐标原点到图像中心
    height, width = img_array.shape
    points = np.column_stack([
        points[:, 1] - width / 2,  # x坐标
        -(points[:, 0] - height / 2)  # y坐标，负号是因为图像的y轴向下
    ])

    # 保存点云数据
    if output_file:
        try:
            np.savetxt(output_file, points, fmt='%.2f', delimiter=',')
            print(f"点云数据已保存到 {output_file}")
        except Exception as e:
            print(f"错误：保存点云数据时出错 - {e}")

    return points


def visualize_pointcloud(points, title="点云可视化", output_image=None):
    """
    可视化点云数据

    参数:
    points (numpy.ndarray): 点云数据，形状为(n, 2)
    title (str): 图表标题
    output_image (str): 输出图像路径，如果为None则显示在屏幕上
    """
    if points is None or len(points) == 0:
        print("错误：没有点云数据可显示")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.5)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')  # 保证x和y轴比例相同
    plt.grid(True, linestyle='--', alpha=0.7)

    if output_image:
        try:
            plt.savefig(output_image, dpi=300, bbox_inches='tight')
            print(f"点云可视化已保存到 {output_image}")
        except Exception as e:
            print(f"错误：保存点云可视化时出错 - {e}")
    else:
        plt.show()


if __name__ == "__main__":
    # 配置参数
    input_image = "H:\lmw_python_pr\高光谱\jpg\test3lab1band152rgb.jpg"  # 替换为你的图像路径
    output_cloud = "pointcloud.csv"
    output_visualization = "pointcloud_visualization.png"
    threshold_value = 50  # 二值化阈值，可以根据图像调整

    # 检查输入文件是否存在
    if not os.path.exists(input_image):
        print(f"错误：输入图像文件 '{input_image}' 不存在")
        print("请替换代码中的 'input_image' 变量为有效的图像路径")
    else:
        # 执行转换
        pointcloud = image_to_pointcloud(input_image, threshold_value, output_cloud)

        # 可视化结果
        if pointcloud is not None:
            visualize_pointcloud(pointcloud, "图像转换的点云", output_visualization)