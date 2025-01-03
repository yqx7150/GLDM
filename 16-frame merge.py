import os
import numpy as np
from scipy.io import loadmat, savemat

# 定义输入目录和输出目录
input_dir = '/home/who/桌面/zkl-gy/时间交错mask训练/train(16帧)/data_xt_train'
output_dir = '/home/who/桌面/zkl-gy/时间交错mask训练/train(16帧)/datatrain_3200_1zhen'
mask_path = '/home/who/桌面/zkl-gy/时间交错mask训练/train(16帧)/mask/UIH_TIS_mask_t_192_192_X800_ACS_16_R_5.42.mat'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 定义k2wgt函数，占位符，需要根据实际情况实现
def k2wgt(X, W):
    result = np.multiply(X, W)
    return result
# def k2wgt(k_space, mask):
#     # 请根据实际情况实现k2wgt函数
#     return k_space * mask


# 获取输入目录中的所有.mat文件
data_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]

# 读取掩码文件
mask = loadmat(mask_path)['mask_t']

# 遍历所有.mat文件
for file_name in data_files:
    # 构建文件路径
    file_path = os.path.join(input_dir, file_name)

    # 读取.mat文件
    cardiac_xt_data = loadmat(file_path)['data_xt']
    k_space_c = np.zeros((192, 192, 16), dtype=np.complex64)

    # 计算K空间数据
    for i in range(16):
        k_space_c[:, :, i] = np.fft.fftshift(np.fft.fft2(cardiac_xt_data[:, :, i]))

    k_sample_data = np.zeros((192, 192, 16), dtype=np.complex64)

    # 应用k2wgt函数
    for i in range(16):
        k_sample_data[:, :, i] = k2wgt(k_space_c[:, :, i], mask[:, :, i])

    k_sample_data_combine = np.zeros((192, 192), dtype=np.complex64)

    # 合并K空间数据
    for i in range(16):
        k_sample_data_combine = k_sample_data[:, :, i] + k_sample_data_combine

    k_space = np.zeros((192, 192, 16), dtype=np.complex64)
    for i in range(16):
        k_space[:, :, i] = k_sample_data[:, :, i] + k_sample_data_combine * (1 - mask[:, :, i])

    # 保存每一帧的K空间数据为单独的.mat文件
    base_filename = os.path.splitext(file_name)[0]
    for i in range(16):
        output_filename = f"{base_filename}_frame_{i + 1}.mat"
        output_filepath = os.path.join(output_dir, output_filename)
        savemat(output_filepath, {'data_kt': k_space[:, :, i]})

print("处理完成。")
