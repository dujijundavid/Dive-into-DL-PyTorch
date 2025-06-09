"""
nn.Dropout（随机失活）核心原理与用法
-------------------------------------
第一性原理思考：
1. 什么是过拟合？
   - 模型在训练集上表现很好，但在新数据上表现差
   - 模型过度记忆训练数据，缺乏泛化能力
   - 是机器学习中的常见问题

2. 为什么需要Dropout？
   - 防止过拟合：通过随机丢弃神经元
   - 模型集成：隐式地训练多个子网络
   - 特征依赖：减少神经元之间的共适应

3. Dropout的核心特性是什么？
   - 随机性：按概率随机丢弃神经元
   - 缩放：保持期望值不变
   - 模式切换：训练和推理行为不同

苏格拉底式提问与验证：
1. 为什么需要缩放因子？
   - 问题：不缩放会有什么影响？
   - 验证：比较有无缩放的效果
   - 结论：缩放保持期望值不变

2. Dropout概率如何影响模型？
   - 问题：不同概率值的效果如何？
   - 验证：观察不同概率下的模型表现
   - 结论：概率需要根据任务调整

3. 为什么推理时关闭Dropout？
   - 问题：推理时使用Dropout会怎样？
   - 验证：比较训练和推理模式
   - 结论：推理时需要确定性输出

费曼学习法讲解：
1. 概念解释
   - 用简单的类比解释Dropout
   - 通过可视化理解随机丢弃
   - 强调Dropout的重要性

2. 实例教学
   - 从简单到复杂的Dropout操作
   - 通过实际例子理解参数作用
   - 实践常见应用场景

3. 知识巩固
   - 总结Dropout的核心概念
   - 提供使用的最佳实践
   - 建议进阶学习方向

功能说明：
- nn.Dropout 在训练时随机丢弃部分神经元输出，防止过拟合，提升泛化能力。

原理讲解：
- 按概率p将部分输出置零，其余缩放1/(1-p)，保证期望不变。
- 只在训练模式下生效，推理时自动关闭。

使用场景：
- 深层网络（MLP/CNN/RNN）防止过拟合，常与BatchNorm/L2正则联合使用。

常见bug：
- 推理时未切换到eval()，导致输出不稳定。
- Dropout概率设置过大，模型欠拟合。
- 用于输入层时需谨慎，信息损失大。

深度学习研究员精华笔记：
- Dropout等价于模型集成平均，提升鲁棒性。
- 可用于特征、权重、激活等多种Dropout变体。
- 现代CNN常用BN替代Dropout，但MLP/RNN仍常用。

可运行案例：
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# 1. 创建Dropout层
drop = nn.Dropout(p=0.5)

# 2. 输入一个 batch（4x10）
x = torch.ones(4, 10)
out = drop(x)
print("训练模式下Dropout输出:", out)

# 3. 切换到推理模式
drop.eval()
out_eval = drop(x)
print("推理模式下Dropout输出:", out_eval)

# 4. Dropout概率过大 bug 演示
drop2 = nn.Dropout(p=0.99)
out2 = drop2(x)
print("高概率Dropout输出:", out2)

# 5. 验证期望值保持不变
print("\n验证期望值：")
# 创建测试数据
test_data = torch.ones(1000, 10)
drop_test = nn.Dropout(p=0.5)

# 多次运行Dropout
with torch.no_grad():
    outputs = []
    for _ in range(100):
        out = drop_test(test_data)
        outputs.append(out.mean().item())

print("原始数据均值:", test_data.mean().item())
print("Dropout后平均均值:", np.mean(outputs))
print("理论缩放因子:", 1/(1-0.5))

# 6. 验证不同概率的效果
print("\n验证不同概率：")
probabilities = [0.1, 0.3, 0.5, 0.7, 0.9]
test_input = torch.ones(100, 10)

plt.figure(figsize=(10, 5))
for p in probabilities:
    drop = nn.Dropout(p=p)
    with torch.no_grad():
        out = drop(test_input)
    plt.hist(out.mean(dim=1).numpy(), bins=20, alpha=0.5, label=f'p={p}')

plt.xlabel('输出均值')
plt.ylabel('频率')
plt.title('不同Dropout概率的分布')
plt.legend()
plt.grid(True)
plt.show()

# 7. 验证训练和推理模式的区别
print("\n验证训练/推理模式：")
drop = nn.Dropout(p=0.5)
x = torch.ones(10, 10)

# 训练模式
drop.train()
out_train = drop(x)
print("训练模式输出统计量:")
print("均值:", out_train.mean().item())
print("方差:", out_train.var().item())

# 推理模式
drop.eval()
out_eval = drop(x)
print("\n推理模式输出统计量:")
print("均值:", out_eval.mean().item())
print("方差:", out_eval.var().item())

# 8. 验证特征依赖减少
print("\n验证特征依赖：")
# 创建相关特征
x = torch.randn(100, 2)
x[:, 1] = x[:, 0] + 0.1 * torch.randn(100)  # 第二个特征与第一个高度相关

# 应用Dropout
drop = nn.Dropout(p=0.5)
with torch.no_grad():
    x_drop = drop(x)

# 计算相关系数
corr_before = torch.corrcoef(x.t())[0, 1]
corr_after = torch.corrcoef(x_drop.t())[0, 1]

print("原始特征相关系数:", corr_before.item())
print("Dropout后相关系数:", corr_after.item())

# 9. 可视化Dropout效果
print("\n可视化Dropout效果：")
# 创建测试数据
test_data = torch.randn(100, 2)
drop_viz = nn.Dropout(p=0.5)

# 应用Dropout
with torch.no_grad():
    dropped = drop_viz(test_data)

# 绘制散点图
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.scatter(test_data[:, 0], test_data[:, 1])
plt.title('原始数据')
plt.subplot(122)
plt.scatter(dropped[:, 0], dropped[:, 1])
plt.title('Dropout后数据')
plt.show()

# 10. 验证梯度流动
print("\n验证梯度流动：")
# 创建需要梯度的输入
x = torch.randn(10, 10, requires_grad=True)
drop_grad = nn.Dropout(p=0.5)

# 前向传播
y = drop_grad(x)
loss = y.sum()
loss.backward()

print("输入梯度是否存在:", x.grad is not None)
print("输入梯度形状:", x.grad.shape)
print("输入梯度范数:", x.grad.norm().item()) 