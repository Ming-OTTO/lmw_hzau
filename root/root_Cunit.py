
import cv2
import numpy as np
import os
import pandas as pd

# 读取xlsx文件
def read_xlsx(file_path):
    try:
        df = pd.read_excel(file_path)
        # 将 DataFrame 转换为列表形式
        data = df.values.tolist()
        # 添加表头
        data.insert(0, df.columns.tolist())
        return data
    except Exception as e:
        print(f"读取文件 {file_path} 时出现错误: {e}")
        return []

# 处理目标列的值
def process_column_value(target_value, is_area):
    try:
        target_value = float(target_value)
        if is_area:
            modified_value = (target_value * 485 * 485) / (2656 * 2656)
        else:
            modified_value = (target_value * 485) / 2656
        return str(modified_value)
    except ValueError:
        print(f"无法将值 '{target_value}' 转换为数字，跳过此值。")
        return target_value

# 对读取的文件进行处理
def deal_data(all_data):
    if not all_data:
        return []
    # 获取表头
    header = all_data[0]
    # 假设你想要的列名是 "target_column"
    area_column_name = ['area', 'convex_area', 'mass_1_A1', 'mass_1_A2', 'mass_2_A1', 'mass_2_A2', 'mass_3_A1', 'mass_3_A2']
    length_column_name = ['length', 'depth', 'width', 'mass_1_L1', 'mass_1_L2', 'mass_2_L1', 'mass_2_L2', 'mass_3_L1', 'mass_3_L2']
    for fheader in header:
        if fheader in area_column_name:
            target_column_index = header.index(fheader)
            for row in all_data[1:]:  # 跳过表头行
                if len(row) > target_column_index:
                    row[target_column_index] = process_column_value(row[target_column_index], is_area=True)
        elif fheader in length_column_name:
            target_column_index = header.index(fheader)
            for row in all_data[1:]:  # 跳过表头行
                if len(row) > target_column_index:
                    row[target_column_index] = process_column_value(row[target_column_index], is_area=False)
    return all_data

# 保存csv文件
def save_csv(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            for row in data:
                row_str = [str(item) for item in row]
                file.write(','.join(row_str) + '\n')
    except Exception as e:
        print(f"保存文件 {file_path} 时出现错误: {e}")

if __name__ == '__main__':
    path_root = r'F:\xiaogenhe\hpy_rice\result'
    path_save = r'F:\xiaogenhe\danwei'
    # 创建保存目录
    if not os.path.exists(path_save):
        os.makedirs(path_save)
    files = tool_house.filepath_get(path_root)
    for file in files:
        if file.lower().endswith('.xlsx'):
            all_data = read_xlsx(file)
        else:
            continue  # 仅处理 .xlsx 文件
        deal_result = deal_data(all_data)
        file_name = os.path.basename(file).replace('.xlsx', '.csv')  # 获取文件名并转换为 .csv
        save_path = os.path.join(path_save, file_name)  # 构建保存路径
        save_csv(save_path, deal_result)
        print(f"处理后的文件已保存至: {save_path}")
