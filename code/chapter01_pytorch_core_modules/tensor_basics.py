"""
torch.Tensor 基础与常用操作
--------------------------
第一性原理思考：
1. 什么是张量？
   - 张量是多维数组的数学抽象
   - 标量是0维张量，向量是1维张量，矩阵是2维张量
   - 在深度学习中，张量是数据的基本表示形式

2. 为什么需要张量？
   - 统一的数据结构：可以表示各种类型的数据（图像、文本、音频等）
   - 高效的数学运算：支持向量化操作，提高计算效率
   - 自动微分支持：为深度学习提供基础

3. 张量的核心特性是什么？
   - 形状（shape）：描述张量的维度结构
   - 数据类型（dtype）：决定存储精度和范围
   - 设备（device）：支持CPU和GPU计算
   - 内存布局：影响计算效率

苏格拉底式提问与验证：
1. 张量的形状如何影响运算？
   - 问题：不同形状的张量如何进行计算？
   - 验证：尝试不同形状的张量运算
   - 结论：广播机制使不同形状的张量可以运算

2. 为什么需要数据类型？
   - 问题：不同数据类型对计算有什么影响？
   - 验证：比较不同数据类型的计算精度和内存使用
   - 结论：数据类型影响计算精度和效率

3. GPU加速的原理是什么？
   - 问题：为什么GPU能加速张量计算？
   - 验证：比较CPU和GPU的计算速度
   - 结论：GPU的并行架构适合张量运算

费曼学习法讲解：
1. 概念解释
   - 用简单的类比解释张量（如：多维表格）
   - 通过实际例子说明张量操作
   - 强调张量在深度学习中的重要性

2. 实例教学
   - 从简单到复杂的张量操作
   - 通过可视化理解张量形状
   - 实践常见张量操作

3. 知识巩固
   - 总结张量的核心概念
   - 提供常见操作的最佳实践
   - 建议进阶学习方向

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
import time

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

# 8. 验证广播机制
print("\n验证广播机制：")
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([1, 2])
print("a:", a)
print("b:", b)
print("a + b (广播):", a + b)

# 9. 验证数据类型影响
print("\n验证数据类型影响：")
x_int = torch.tensor([1, 2, 3], dtype=torch.int32)
x_float = torch.tensor([1, 2, 3], dtype=torch.float32)
print("整数张量:", x_int)
print("浮点张量:", x_float)
print("整数张量内存大小:", x_int.element_size() * x_int.nelement())
print("浮点张量内存大小:", x_float.element_size() * x_float.nelement())

# 10. 验证GPU加速
print("\n验证GPU加速：")
if torch.cuda.is_available():
    # 创建大矩阵
    large_matrix = torch.randn(1000, 1000)
    
    # CPU计算时间
    start_time = time.time()
    cpu_result = torch.mm(large_matrix, large_matrix)
    cpu_time = time.time() - start_time
    
    # GPU计算时间
    gpu_matrix = large_matrix.cuda()
    start_time = time.time()
    gpu_result = torch.mm(gpu_matrix, gpu_matrix)
    gpu_time = time.time() - start_time
    
    print(f"CPU计算时间: {cpu_time:.4f}秒")
    print(f"GPU计算时间: {gpu_time:.4f}秒")
    print(f"加速比: {cpu_time/gpu_time:.2f}倍") 