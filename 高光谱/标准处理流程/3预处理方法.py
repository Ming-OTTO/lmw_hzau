import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.signal import savgol_filter, argrelextrema

# 变量定义
data_filepath = "你的数据文件路径.csv"
window_length, poly_order = 11, 3  # 调整成你认为合适的值

# 导入数据
data = pd.read_csv(data_filepath)
spectra = data.iloc[:, 2:].values

# 预处理方法定义

def snv(input_data):
    mean = input_data.mean(axis=1, keepdims=True)
    std = input_data.std(axis=1, keepdims=True)
    return (input_data - mean) / std

def msc(input_data):
    ref = np.mean(input_data, axis=0)
    msc_result = input_data.copy()
    for i in range(input_data.shape[0]):
        fit = np.polyfit(ref, input_data[i, :], 1, full=True)
        msc_result[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]
    return msc_result

def standardize(input_data):
    scaler = StandardScaler().fit(input_data)
    standardized = scaler.transform(input_data)
    return standardized

def sg_filter(input_data, window_length, poly_order):
    sg_result = savgol_filter(input_data, window_length, poly_order)
    return sg_result

def first_derivative(input_data):
    derivative = np.diff(input_data, axis=1)
    return derivative

def second_derivative(input_data):
    derivative2 = np.diff(input_data, n=2, axis=1)
    return derivative2

def normalize(input_data):
    min_max_scaler = MinMaxScaler()
    normalized = min_max_scaler.fit_transform(input_data)
    return normalized

# 预处理方法使用和结果保存

def save_preprocess_resuls(method_name, func, *params):
    processed_data = func(spectra, *params)
    df = pd.concat([data.iloc[:, :2], pd.DataFrame(processed_data, columns=data.columns[2:])], axis=1)
    df.to_csv(f"{method_name}_preprocessed.csv", index=False)

save_preprocess_resuls("snv", snv)
save_preprocess_resuls("msc", msc)
save_preprocess_resuls("standardization", standardize)
save_preprocess_resuls("sg", sg_filter, window_length, poly_order)
save_preprocess_resuls("first_derivative", first_derivative)
save_preprocess_resuls("second_derivative", second_derivative)
save_preprocess_resuls("normalization", normalize)