import os
import shutil
import cv2
from tqdm import tqdm
import argparse
from pathlib import Path


def create_output_directory(source_dir, output_base, side):
    """创建输出目录，保持与输入目录相同的子目录结构"""
    relative_path = os.path.relpath(source_dir, os.path.dirname(os.path.dirname(source_dir)))
    output_dir = os.path.join(output_base, relative_path)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def sample_images(input_dir, output_dir, interval, side, verbose=False):
    """按照指定间隔采样图片，并复制到输出目录"""
    # 获取所有图片文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = sorted([
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
           and os.path.splitext(f)[1].lower() in image_extensions
    ])

    if not image_files:
        if verbose:
            print(f"警告: 在目录 {input_dir} 中未找到图片文件")
        return 0, 0

    # 采样图片
    sampled_files = image_files[::interval]

    # 复制采样的图片到输出目录
    copied_count = 0
    for img_file in tqdm(sampled_files, desc=f"处理 {os.path.basename(input_dir)}", unit="张"):
        src_path = os.path.join(input_dir, img_file)
        dst_path = os.path.join(output_dir, img_file)

        # 复制文件
        try:
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        except Exception as e:
            if verbose:
                print(f"错误: 无法复制文件 {src_path}: {e}")

    return len(image_files), copied_count


def process_directory(input_dir, output_base, interval, side, verbose=False):
    """递归处理目录及其子目录中的所有图片"""
    total_images = 0
    total_sampled = 0

    # 为当前目录创建输出目录
    output_dir = create_output_directory(input_dir, output_base, side)

    # 处理当前目录中的图片
    images, sampled = sample_images(input_dir, output_dir, interval, side, verbose)
    total_images += images
    total_sampled += sampled

    # 递归处理子目录
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path):
            sub_images, sub_sampled = process_directory(item_path, output_base, interval, side, verbose)
            total_images += sub_images
            total_sampled += sub_sampled

    return total_images, total_sampled


def main():
    # 直接在代码中设置输入和输出目录
    INPUT_DIR = r"G:\2025cotton4\3DRGB"
    OUTPUT_DIR = r"G:\2025cotton4\3DRGB_sampled"
    SAMPLE_INTERVAL = 2  # 采样间隔
    VERBOSE = True  # 是否显示详细信息

    # 验证输入目录
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录 {INPUT_DIR} 不存在")
        return

    # 创建输出基础目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"开始图片采样...")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"采样间隔: 每 {SAMPLE_INTERVAL} 张图片取1张")
    print("-" * 50)

    total_images = 0
    total_sampled = 0

    # 处理01到06的文件夹
    for i in range(1, 7):
        folder_num = f"{i:02d}"  # 格式化为两位数字，如01, 02等

        # 处理L文件夹
        left_folder = os.path.join(INPUT_DIR, folder_num, f"{folder_num}L")
        if os.path.exists(left_folder):
            print(f"\n开始处理 {left_folder} 及其子目录...")
            side_name = f"{folder_num}L"
            images, sampled = process_directory(left_folder, OUTPUT_DIR, SAMPLE_INTERVAL, side_name, VERBOSE)
            total_images += images
            total_sampled += sampled
            print(f"文件夹 {side_name}: 总共处理了 {images} 张图片，采样 {sampled} 张")

        # 处理R文件夹
        right_folder = os.path.join(INPUT_DIR, folder_num, f"{folder_num}R")
        if os.path.exists(right_folder):
            print(f"\n开始处理 {right_folder} 及其子目录...")
            side_name = f"{folder_num}R"
            images, sampled = process_directory(right_folder, OUTPUT_DIR, SAMPLE_INTERVAL, side_name, VERBOSE)
            total_images += images
            total_sampled += sampled
            print(f"文件夹 {side_name}: 总共处理了 {images} 张图片，采样 {sampled} 张")

    print("-" * 50)
    print(f"采样完成!")
    print(f"总共处理了 {total_images} 张图片")
    print(f"总共采样了 {total_sampled} 张图片")
    print(f"采样图片保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()