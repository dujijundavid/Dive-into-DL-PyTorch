"""
torch.Tensor 基础与常用操作
--------------------------
功能说明：
- Tensor 是 PyTorch 的核心数据结构，支持多维数组、自动微分、GPU加速。

原理讲解：
- Tensor 类似于 numpy 的 ndarray，但可用于自动微分和 GPU 运算。
- 支持标量、向量、矩阵、高维张量。
- 常用操作包括创建、索引、切片、变形、数学运算、广播等。

工程意义：
- 所有神经网络的输入、参数、输出都以 Tensor 形式表示。

可运行案例：
"""
import torch

# 1. 创建张量
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # 指定值
z = torch.zeros(2, 3)                       # 全0
o = torch.ones(2, 3)                        # 全1
r = torch.randn(2, 3)                       # 标准正态分布
print("张量x:\n", x)
print("全0张量z:\n", z)
print("全1张量o:\n", o)
print("随机张量r:\n", r)

# 2. 张量属性
print("形状:", x.shape)
print("数据类型:", x.dtype)
print("设备:", x.device)

# 3. 索引与切片
print("第一行:", x[0])
print("第一列:", x[:, 0])

# 4. 变形与拼接
x2 = x.view(4)  # 改变形状
print("变形后:", x2)
cat = torch.cat([x, x], dim=0)  # 按行拼接
print("拼接后:", cat)

# 5. 数学运算与广播
print("加法:", x + 1)
print("乘法:", x * 2)
print("矩阵乘法:", torch.mm(x, x))

# 6. 与 numpy 互操作
import numpy as np
n = np.array([[9, 8], [7, 6]])
t = torch.from_numpy(n)
print("numpy转tensor:", t)
print("tensor转numpy:", x.numpy())

# 7. GPU 支持
if torch.cuda.is_available():
    x_gpu = x.to('cuda')
    print("张量转到GPU:", x_gpu) 