import os
import numpy as np
from scipy.io import loadmat, savemat

# 定义输入目录和输出目录
input_dir = '/home/who/桌面/zkl-gy/时间交错mask训练/train(4帧)/data_xt_train'
output_dir = '/home/who/桌面/zkl-gy/时间交错mask训练/train(4帧)/datatrain_3200_1zhen'
mask_path = '/home/who/桌面/zkl-gy/时间交错mask训练/train(4帧)/mask/UIH2_mask_192_192_16.mat'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 定义k2wgt函数，占位符，需要根据实际情况实现
def k2wgt(X, W):
    result = np.multiply(X, W)
    return result

# 获取输入目录中的所有.mat文件
data_files = [f for f in os.listdir(input_dir) if f.endswith('.mat')]

# 读取掩码文件
mask = loadmat(mask_path)['mask_tt']

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

    k_sample_data_combine0 = np.zeros((192, 192), dtype=np.complex64)
    k_sample_data_combine1 = np.zeros((192, 192), dtype=np.complex64)
    k_sample_data_combine2 = np.zeros((192, 192), dtype=np.complex64)
    k_sample_data_combine3 = np.zeros((192, 192), dtype=np.complex64)

    # 合并K空间数据
    for i in range(0, 4):
        k_sample_data_combine0 = k_sample_data[:, :, i] + k_sample_data_combine0
    for i in range(4, 8):
        k_sample_data_combine1 = k_sample_data[:, :, i] + k_sample_data_combine1
    for i in range(8, 12):
        k_sample_data_combine2 = k_sample_data[:, :, i] + k_sample_data_combine2
    for i in range(12, 16):
        k_sample_data_combine3 = k_sample_data[:, :, i] + k_sample_data_combine3

    k_space0 = np.zeros((192, 192, 4), dtype=np.complex64)
    k_space1 = np.zeros((192, 192, 4), dtype=np.complex64)
    k_space2 = np.zeros((192, 192, 4), dtype=np.complex64)
    k_space3 = np.zeros((192, 192, 4), dtype=np.complex64)
    for i in range(0, 4):
        k_space0[:, :, i] = k_sample_data[:, :, i] + k_sample_data_combine0 * (1 - mask[:, :, i])
    for i in range(4, 8):
        k_space1[:, :, i - 4] = k_sample_data[:, :, i] + k_sample_data_combine1 * (1 - mask[:, :, i])
    for i in range(8, 12):
        k_space2[:, :, i - 8] = k_sample_data[:, :, i] + k_sample_data_combine2 * (1 - mask[:, :, i])
    for i in range(12, 16):
        k_space3[:, :, i - 12] = k_sample_data[:, :, i] + k_sample_data_combine3 * (1 - mask[:, :, i])

    # 保存每一帧的K空间数据为单独的.mat文件
    base_filename = os.path.splitext(file_name)[0]
    for i in range(0, 4):
        output_filename = f"{base_filename}_frame_{i + 1}.mat"
        output_filepath = os.path.join(output_dir, output_filename)
        savemat(output_filepath, {'data_kt': k_space0[:, :, i]})
    for i in range(4, 8):
        output_filename = f"{base_filename}_frame_{i + 1}.mat"
        output_filepath = os.path.join(output_dir, output_filename)
        savemat(output_filepath, {'data_kt': k_space1[:, :, i - 4]})
    for i in range(8, 12):
        output_filename = f"{base_filename}_frame_{i + 1}.mat"
        output_filepath = os.path.join(output_dir, output_filename)
        savemat(output_filepath, {'data_kt': k_space2[:, :, i - 8]})
    for i in range(12, 16):
        output_filename = f"{base_filename}_frame_{i + 1}.mat"
        output_filepath = os.path.join(output_dir, output_filename)
        savemat(output_filepath, {'data_kt': k_space3[:, :, i - 12]})

print("处理完成。")
