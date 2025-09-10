import os
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def read_data(file_path):
    """读取Excel或CSV文件为DataFrame"""
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        return pd.DataFrame()

    try:
        if file_path.lower().endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        elif file_path.lower().endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            logger.warning(f"不支持的文件格式: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
        return pd.DataFrame()


def process_column_value(value, is_area):
    """处理单列值"""
    try:
        value = float(value)
        return (value * 485 * 485) / (2656 * 2656) if is_area else (value * 485) / 2656
    except (ValueError, TypeError):
        logger.warning(f"无法转换值 '{value}'，保留原始值")
        return value


def process_dataframe(df):
    """处理整个DataFrame"""
    if df.empty:
        logger.warning("收到空DataFrame，跳过处理")
        return df

    # 定义需要处理的列及其类型
    AREA_COLS = ['area', 'convex_area',
                 'mass_1_A1', 'mass_1_A2',
                 'mass_2_A1', 'mass_2_A2',
                 'mass_3_A1', 'mass_3_A2']

    LENGTH_COLS = ['length', 'depth', 'width',
                   'mass_1_L1', 'mass_1_L2',
                   'mass_2_L1', 'mass_2_L2',
                   'mass_3_L1', 'mass_3_L2']

    # 查找数据中实际存在的列名
    actual_area_cols = [col for col in df.columns if col in AREA_COLS]
    actual_length_cols = [col for col in df.columns if col in LENGTH_COLS]

    logger.info(f"找到的面积列: {actual_area_cols}")
    logger.info(f"找到的长度列: {actual_length_cols}")

    # 处理面积列
    for col in actual_area_cols:
        df[col] = df[col].apply(lambda x: process_column_value(x, is_area=True))

    # 处理长度列
    for col in actual_length_cols:
        df[col] = df[col].apply(lambda x: process_column_value(x, is_area=False))

    return df


def convert_files(root_dir, save_dir):
    """转换指定目录中的所有Excel和CSV文件"""
    if not os.path.exists(root_dir):
        logger.error(f"源目录不存在: {root_dir}")
        return 0

    os.makedirs(save_dir, exist_ok=True)
    processed_count = 0

    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.lower().endswith(('.xlsx', '.xls', '.csv')):
                continue

            file_path = os.path.join(foldername, filename)
            logger.info(f"开始处理: {file_path}")

            df = read_data(file_path)
            if df.empty:
                logger.warning(f"文件为空或读取失败，跳过: {file_path}")
                continue

            try:
                # 处理数据
                processed_df = process_dataframe(df)

                # 准备保存路径，添加"_rs"后缀
                base_name = os.path.splitext(filename)[0]
                save_path = os.path.join(save_dir, f"{base_name}_rs.csv")

                # 保存为CSV
                processed_df.to_csv(save_path, index=False, encoding='utf-8')
                logger.info(f"已保存到: {save_path}")
                processed_count += 1
            except Exception as e:
                logger.error(f"处理文件 {filename} 时出错: {e}")

    return processed_count


if __name__ == '__main__':
    # 配置路径 - 请更改为您的实际路径
    ROOT_DIR = r'H:\lmw_python_pr\root\PaddleSeg\cotton_results'
    SAVE_DIR = r'H:\lmw_python_pr\root\PaddleSeg\cotton_results\info_result'

    logger.info(f"开始处理目录: {ROOT_DIR}")
    count = convert_files(ROOT_DIR, SAVE_DIR)
    logger.info(f"处理完成! 共转换 {count} 个文件")