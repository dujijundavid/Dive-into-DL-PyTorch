"""
nn.Linear（全连接层）核心原理与用法
---------------------------------
第一性原理思考：
1. 什么是线性变换？
   - 线性变换是保持加法和数乘运算的映射
   - 在数学上表示为 y = Ax + b
   - 是神经网络中最基础的变换单元

2. 为什么需要线性层？
   - 实现特征空间的线性映射
   - 作为复杂网络的基础构建块
   - 提供可学习的参数（权重和偏置）

3. 线性层的核心特性是什么？
   - 可学习的参数：权重矩阵和偏置向量
   - 自动微分支持：参数梯度自动计算
   - 批处理能力：支持批量数据输入

苏格拉底式提问与验证：
1. 线性层的参数如何影响输出？
   - 问题：权重和偏置如何改变输入特征？
   - 验证：观察不同参数下的输出变化
   - 结论：权重决定特征变换，偏置决定平移

2. 为什么需要偏置项？
   - 问题：没有偏置项会有什么限制？
   - 验证：比较有无偏置的模型表现
   - 结论：偏置提供模型灵活性

3. 批处理的作用是什么？
   - 问题：为什么需要批处理？
   - 验证：比较单样本和批处理的效率
   - 结论：批处理提高计算效率

费曼学习法讲解：
1. 概念解释
   - 用简单的数学公式解释线性变换
   - 通过几何直观理解权重和偏置
   - 强调线性层在神经网络中的基础地位

2. 实例教学
   - 从简单到复杂的线性变换
   - 通过可视化理解参数作用
   - 实践常见应用场景

3. 知识巩固
   - 总结线性层的核心概念
   - 提供参数初始化的最佳实践
   - 建议进阶学习方向

功能说明：
- nn.Linear 实现 y = xA^T + b 的仿射变换，是神经网络最基础的结构单元。

原理讲解：
- 输入张量 x（形状 [batch, in_features]），权重矩阵 A（[out_features, in_features]），偏置 b（[out_features]）。
- 前向传播自动完成线性变换，参数可自动参与反向传播。

工程意义：
- 用于特征变换、分类、回归等场景，是 MLP、输出层等的基础。

可运行案例：
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# 1. 创建线性层并查看参数
linear = nn.Linear(4, 2)
print("权重 shape:", linear.weight.shape)
print("偏置 shape:", linear.bias.shape)

# 2. 基本前向传播
x = torch.randn(3, 4)
y = linear(x)
print("输出 shape:", y.shape)

# 3. 反向传播示例
loss = y.sum()
loss.backward()
print("权重梯度 shape:", linear.weight.grad.shape)

# 4. 验证参数影响
print("\n验证参数影响：")
# 创建两个不同的线性层
linear1 = nn.Linear(2, 1)
linear2 = nn.Linear(2, 1)

# 使用相同的输入
x = torch.tensor([[1.0, 2.0]])
y1 = linear1(x)
y2 = linear2(x)
print("相同输入在不同参数下的输出：")
print("linear1输出:", y1)
print("linear2输出:", y2)

# 5. 验证偏置作用
print("\n验证偏置作用：")
# 创建有偏置和无偏置的线性层
linear_with_bias = nn.Linear(2, 1)
linear_no_bias = nn.Linear(2, 1, bias=False)

# 使用相同的输入和权重
x = torch.tensor([[1.0, 2.0]])
linear_no_bias.weight.data = linear_with_bias.weight.data
y1 = linear_with_bias(x)
y2 = linear_no_bias(x)
print("有偏置输出:", y1)
print("无偏置输出:", y2)
print("偏置值:", linear_with_bias.bias.item())

# 6. 验证批处理效率
print("\n验证批处理效率：")
import time

# 创建线性层
linear = nn.Linear(100, 50)

# 单样本处理
x_single = torch.randn(1, 100)
start_time = time.time()
for _ in range(1000):
    y = linear(x_single)
single_time = time.time() - start_time

# 批处理
x_batch = torch.randn(1000, 100)
start_time = time.time()
y = linear(x_batch)
batch_time = time.time() - start_time

print(f"单样本处理1000次时间: {single_time:.4f}秒")
print(f"批处理1000个样本时间: {batch_time:.4f}秒")
print(f"加速比: {single_time/batch_time:.2f}倍")

# 7. 可视化线性变换
print("\n可视化线性变换：")
# 创建2D到2D的线性变换
linear = nn.Linear(2, 2)

# 生成网格点
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)
points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)

# 应用线性变换
with torch.no_grad():
    transformed_points = linear(points)

# 绘制原始点和变换后的点
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.scatter(points[:, 0], points[:, 1], alpha=0.5)
plt.title('Original Points')
plt.grid(True)

plt.subplot(122)
plt.scatter(transformed_points[:, 0], transformed_points[:, 1], alpha=0.5)
plt.title('Points After Linear Transformation')
plt.grid(True)
plt.show() 