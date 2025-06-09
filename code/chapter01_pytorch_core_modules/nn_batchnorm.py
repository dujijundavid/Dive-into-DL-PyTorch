"""
nn.BatchNorm（批量归一化）核心原理与用法
-----------------------------------------
第一性原理思考：
1. 什么是归一化？
   - 归一化是将数据转换为标准分布的过程
   - 通过减去均值、除以标准差实现
   - 使数据分布更加稳定和可预测

2. 为什么需要批量归一化？
   - 内部协变量偏移：网络各层输入分布不断变化
   - 梯度问题：缓解梯度消失/爆炸
   - 优化问题：允许使用更大的学习率

3. 批量归一化的核心特性是什么？
   - 可学习的参数：缩放因子和偏移量
   - 统计量：均值和方差
   - 训练/推理模式：不同行为

苏格拉底式提问与验证：
1. 为什么需要可学习的参数？
   - 问题：简单的归一化为什么不够？
   - 验证：比较有无可学习参数的效果
   - 结论：可学习参数增加模型灵活性

2. 训练和推理模式的区别是什么？
   - 问题：为什么需要两种模式？
   - 验证：观察不同模式下的输出
   - 结论：训练用batch统计，推理用全局统计

3. 批量大小如何影响效果？
   - 问题：小批量会带来什么问题？
   - 验证：比较不同批量大小的效果
   - 结论：批量太小时统计不稳定

费曼学习法讲解：
1. 概念解释
   - 用简单的数据分布解释归一化
   - 通过可视化理解归一化效果
   - 强调批量归一化的重要性

2. 实例教学
   - 从简单到复杂的归一化操作
   - 通过实际例子理解参数作用
   - 实践常见应用场景

3. 知识巩固
   - 总结批量归一化的核心概念
   - 提供使用的最佳实践
   - 建议进阶学习方向

功能说明：
- nn.BatchNorm 对每个通道做归一化，缓解梯度消失/爆炸，加速收敛。

原理讲解：
- 对每个 mini-batch，归一化为均值0方差1，并引入可学习缩放/偏移参数。
- 支持1D/2D/3D（如BatchNorm1d/2d/3d），常用于卷积/全连接层后。
- 训练/推理模式下行为不同（训练用batch统计，推理用全局均值方差）。

使用场景：
- 深层网络（CNN/MLP/RNN）训练加速、提升稳定性。

常见bug：
- 输入 shape 不匹配（如BatchNorm2d需[N,C,H,W]）。
- 推理时未切换到eval()，导致归一化不一致。
- 小batch下统计不稳定。

深度学习研究员精华笔记：
- BN 可缓解梯度消失/爆炸，允许更大学习率。
- BN 也有正则化效果，部分场景可替代Dropout。
- 小batch时可用GroupNorm/LayerNorm等替代。

可运行案例：
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# 1. 创建BatchNorm2d层（常用于卷积输出）
bn = nn.BatchNorm2d(6)

# 2. 输入一个 batch（4张6通道10x10图片）
x = torch.randn(4, 6, 10, 10)
out = bn(x)
print("BatchNorm输出 shape:", out.shape)

# 3. 训练/推理模式切换
bn.eval()  # 切换到推理模式
out_eval = bn(x)
print("推理模式输出 shape:", out_eval.shape)

# 4. 输入 shape 不匹配 bug 演示
try:
    bad_x = torch.randn(4, 10, 10, 6)  # 通道维不对
    bn(bad_x)
except RuntimeError as e:
    print("输入 shape 不匹配报错:", e)

# 5. 验证归一化效果
print("\n验证归一化效果：")
# 创建测试数据
test_data = torch.randn(100, 3, 32, 32)
bn_test = nn.BatchNorm2d(3)

# 应用BatchNorm
with torch.no_grad():
    normalized = bn_test(test_data)

# 计算统计量
mean = normalized.mean(dim=(0, 2, 3))
std = normalized.std(dim=(0, 2, 3))
print("归一化后均值:", mean)
print("归一化后方差:", std)

# 6. 验证可学习参数的作用
print("\n验证可学习参数：")
# 创建两个BatchNorm层
bn1 = nn.BatchNorm2d(1, affine=False)  # 无可学习参数
bn2 = nn.BatchNorm2d(1, affine=True)   # 有可学习参数

# 测试数据
test_input = torch.randn(10, 1, 5, 5)

# 应用BatchNorm
with torch.no_grad():
    out1 = bn1(test_input)
    out2 = bn2(test_input)

print("无参数BN输出范围:", out1.min().item(), "到", out1.max().item())
print("有参数BN输出范围:", out2.min().item(), "到", out2.max().item())

# 7. 验证不同批量大小的效果
print("\n验证批量大小影响：")
def test_batch_size(batch_size):
    bn = nn.BatchNorm2d(1)
    x = torch.randn(batch_size, 1, 32, 32)
    with torch.no_grad():
        out = bn(x)
    return out.std().item()

batch_sizes = [1, 2, 4, 8, 16, 32]
stds = [test_batch_size(bs) for bs in batch_sizes]

plt.figure(figsize=(10, 5))
plt.plot(batch_sizes, stds, 'o-')
plt.xlabel('批量大小')
plt.ylabel('输出标准差')
plt.title('批量大小对归一化效果的影响')
plt.grid(True)
plt.show()

# 8. 验证训练和推理模式的区别
print("\n验证训练/推理模式：")
bn = nn.BatchNorm2d(1)
x = torch.randn(10, 1, 5, 5)

# 训练模式
bn.train()
out_train = bn(x)
print("训练模式输出统计量:")
print("均值:", out_train.mean().item())
print("方差:", out_train.var().item())

# 推理模式
bn.eval()
out_eval = bn(x)
print("\n推理模式输出统计量:")
print("均值:", out_eval.mean().item())
print("方差:", out_eval.var().item())

# 9. 可视化归一化效果
print("\n可视化归一化效果：")
# 创建原始数据
original = torch.randn(100, 1, 1, 1)
bn_viz = nn.BatchNorm2d(1)

# 应用BatchNorm
with torch.no_grad():
    normalized = bn_viz(original)

# 绘制直方图
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.hist(original.numpy().flatten(), bins=30)
plt.title('原始数据分布')
plt.subplot(122)
plt.hist(normalized.numpy().flatten(), bins=30)
plt.title('归一化后分布')
plt.show()

# 10. 验证梯度流动
print("\n验证梯度流动：")
# 创建需要梯度的输入
x = torch.randn(10, 1, 5, 5, requires_grad=True)
bn_grad = nn.BatchNorm2d(1)

# 前向传播
y = bn_grad(x)
loss = y.sum()
loss.backward()

print("输入梯度是否存在:", x.grad is not None)
print("输入梯度形状:", x.grad.shape)
print("输入梯度范数:", x.grad.norm().item()) 