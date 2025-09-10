import os
import cv2
import numpy as np


def pinjie(path_img, path_save):
    """
    遍历 path_img 目录树，对每一组 2 张图做指定拼接，
    并在 path_save 下保持与原目录结构完全一致。
    拼接结果命名为父文件夹名称.png，保存在"原目录对应位置"。
    """
    for root, dirs, files in os.walk(path_img):
        if len(files) != 2:
            continue

        # 原文件夹完整相对路径
        rel_dir = os.path.relpath(root, path_img)        # 例如  Data0724COTTON\601
        out_dir = os.path.join(path_save, rel_dir)       # 目标对应目录
        
        # 获取父文件夹名称作为文件名
        parent_folder_name = os.path.basename(root)
        save_filename = f"{parent_folder_name}.png"
        
        # 输出文件完整路径
        save_path = os.path.join(out_dir, save_filename)

        # 如果已存在则跳过
        if os.path.exists(save_path):
            continue

        # 读取两张图
        img1 = cv2.imread(os.path.join(root, files[0]))
        img2 = cv2.imread(os.path.join(root, files[1]))
        if img1 is None or img2 is None:
            print('读取失败，跳过：', root)
            continue

        h, w = img1.shape[:2]

        # —————— 切割位置 ——————
        x1, x2, x3 = 1800, 2850, 3500
        part1 = img1[:, 0:x1]
        part2 = img1[:, x1:x2]
        part3 = img1[:, x2:x3]
        part4 = img1[:, x3:w]

        part5 = img2[:, 0:x1]
        part6 = img2[:, x1:x2]
        part7 = img2[:, x2:x3]
        part8 = img2[:, x3:w]

        # —————— 按顺序拼接 ——————
        final_img = np.hstack([part1, part6, part3, part8])

        # 确保目录存在
        os.makedirs(out_dir, exist_ok=True)

        # 保存
        cv2.imwrite(save_path, final_img,
                    [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        print('已保存:', save_path)


if __name__ == '__main__':
    path_img = r'G:\2025cotton'   # 原始数据根目录
    path_save = r'G:\out_img_pinjie'     # 拼接结果根目录
    pinjie(path_img, path_save)