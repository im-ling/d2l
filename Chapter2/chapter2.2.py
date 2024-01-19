# 2.2. 数据预处理
# 2.2.1. 读取数据集
import torch
import os
import pandas as pd


def data_write():
    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    data_file = os.path.join('..', 'data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')


def read_file():
    data_file = os.path.join('..', 'data', 'house_tiny.csv')
    data = pd.read_csv(data_file)
    return data


# data_write()
# data = read_file()


# 2.2.2. 处理缺失值
data = read_file()
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# print(inputs)
# print(outputs)
# print(inputs.mean(skipna=True, numeric_only=False))
# inputs = inputs.fillna(inputs.mean())
inputs = inputs.fillna(inputs.select_dtypes(include='number').mean())
# print(type(inputs))
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)


# 2.2.3. 转换为张量格式
X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print(X)
print(Y)
