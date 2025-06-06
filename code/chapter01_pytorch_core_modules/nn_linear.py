"""
nn.Linear（全连接层）核心原理与用法
---------------------------------
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

# 创建一个线性层：输入4维，输出2维
linear = nn.Linear(4, 2)
print("权重 shape:", linear.weight.shape)
print("偏置 shape:", linear.bias.shape)

# 输入一个 batch（3个样本，每个4维特征）
x = torch.randn(3, 4)
y = linear(x)
print("输出 shape:", y.shape)

# 反向传播示例
loss = y.sum()
loss.backward()
print("权重梯度 shape:", linear.weight.grad.shape) 