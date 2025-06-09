"""
nn.Embedding（嵌入层）核心原理与用法
-------------------------------------
第一性原理思考：
1. 什么是嵌入？
   - 将离散的符号映射到连续的向量空间
   - 捕捉符号之间的语义关系
   - 实现从稀疏到稠密的转换

2. 为什么需要嵌入层？
   - 处理离散特征：将类别特征转换为数值
   - 降维：将高维稀疏向量转换为低维稠密向量
   - 语义表示：学习符号的分布式表示

3. 嵌入层的核心特性是什么？
   - 查表操作：高效的向量查找
   - 可学习参数：通过反向传播优化
   - 共享参数：同一符号在不同位置共享表示

苏格拉底式提问与验证：
1. 为什么需要可学习的嵌入？
   - 问题：固定嵌入和可学习嵌入的区别？
   - 验证：比较不同初始化方式的效果
   - 结论：可学习嵌入能适应具体任务

2. 嵌入维度如何影响模型？
   - 问题：维度大小对表示能力的影响？
   - 验证：观察不同维度的效果
   - 结论：维度需要平衡表达能力和计算效率

3. 如何处理未知符号？
   - 问题：遇到词表外的符号怎么办？
   - 验证：测试未知符号的处理
   - 结论：需要合理的未知符号处理策略

费曼学习法讲解：
1. 概念解释
   - 用简单的类比解释嵌入
   - 通过可视化理解向量空间
   - 强调嵌入在深度学习中的重要性

2. 实例教学
   - 从简单到复杂的嵌入操作
   - 通过实际例子理解参数作用
   - 实践常见应用场景

3. 知识巩固
   - 总结嵌入层的核心概念
   - 提供使用的最佳实践
   - 建议进阶学习方向

功能说明：
- nn.Embedding 实现离散ID到稠密向量的映射，是NLP、推荐系统等的基础模块。

原理讲解：
- 输入为整数索引（如词ID），输出为对应的嵌入向量。
- 嵌入矩阵参数可学习，支持大规模词表。

使用场景：
- 词向量、用户/物品嵌入、分类特征编码等。

常见bug：
- 输入必须是Long类型（int64），否则报错。
- 索引超出词表范围会报错。
- 嵌入层参数未参与优化（未传给优化器）。

深度学习研究员精华笔记：
- 嵌入层本质是查表操作，效率高。
- 可用预训练词向量初始化 embedding.weight。
- 嵌入层可用于多种离散特征，不限于文本。

可运行案例：
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# 1. 创建嵌入层：词表大小100，嵌入维度16
embed = nn.Embedding(num_embeddings=100, embedding_dim=16)

# 2. 输入一组词ID（batch=4, seq_len=5）
ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.long)
vecs = embed(ids)
print("嵌入输出 shape:", vecs.shape)

# 3. 输入类型错误 bug 演示
try:
    bad_ids = torch.tensor([[1.0, 2.0]])  # float类型
    embed(bad_ids)
except RuntimeError as e:
    print("输入类型错误报错:", e)

# 4. 词表越界 bug 演示
try:
    out_of_range = torch.tensor([[101]], dtype=torch.long)
    embed(out_of_range)
except IndexError as e:
    print("词表越界报错:", e)

# 5. 验证嵌入初始化
print("\n验证嵌入初始化：")
# 创建两个嵌入层
embed1 = nn.Embedding(10, 3)  # 默认初始化
embed2 = nn.Embedding(10, 3)  # 自定义初始化
embed2.weight.data = torch.randn(10, 3)  # 随机初始化

# 测试输入
test_ids = torch.tensor([0, 1, 2], dtype=torch.long)
out1 = embed1(test_ids)
out2 = embed2(test_ids)

print("默认初始化输出:", out1)
print("自定义初始化输出:", out2)

# 6. 验证不同维度的效果
print("\n验证不同维度：")
dimensions = [2, 4, 8, 16]
test_ids = torch.tensor([0, 1, 2], dtype=torch.long)

for dim in dimensions:
    embed = nn.Embedding(10, dim)
    out = embed(test_ids)
    print(f"维度 {dim} 输出形状:", out.shape)
    print(f"向量范数:", out.norm(dim=1))

# 7. 可视化嵌入空间
print("\n可视化嵌入空间：")
# 创建嵌入层
embed_viz = nn.Embedding(20, 2)
# 生成一些有语义关系的ID
ids = torch.arange(20, dtype=torch.long)
# 获取嵌入向量
with torch.no_grad():
    vectors = embed_viz(ids)

# 绘制散点图
plt.figure(figsize=(10, 8))
plt.scatter(vectors[:, 0], vectors[:, 1])
for i in range(20):
    plt.annotate(str(i), (vectors[i, 0], vectors[i, 1]))
plt.title('嵌入空间可视化')
plt.grid(True)
plt.show()

# 8. 验证梯度流动
print("\n验证梯度流动：")
# 创建需要梯度的输入
x = torch.tensor([0, 1, 2], dtype=torch.long)
embed_grad = nn.Embedding(10, 3)

# 前向传播
y = embed_grad(x)
loss = y.sum()
loss.backward()

print("嵌入层梯度是否存在:", embed_grad.weight.grad is not None)
print("嵌入层梯度形状:", embed_grad.weight.grad.shape)
print("嵌入层梯度范数:", embed_grad.weight.grad.norm().item())

# 9. 验证预训练嵌入
print("\n验证预训练嵌入：")
# 创建预训练向量
pretrained = torch.randn(10, 3)
embed_pretrained = nn.Embedding(10, 3)
embed_pretrained.weight.data = pretrained

# 测试输入
test_ids = torch.tensor([0, 1, 2], dtype=torch.long)
out_pretrained = embed_pretrained(test_ids)
print("预训练嵌入输出:", out_pretrained)

# 10. 验证嵌入更新
print("\n验证嵌入更新：")
# 创建嵌入层
embed_update = nn.Embedding(10, 3)
original = embed_update.weight.data.clone()

# 模拟训练过程
optimizer = torch.optim.SGD(embed_update.parameters(), lr=0.1)
for _ in range(10):
    x = torch.tensor([0, 1, 2], dtype=torch.long)
    y = embed_update(x)
    loss = y.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("原始嵌入范数:", original.norm())
print("更新后嵌入范数:", embed_update.weight.data.norm())
print("嵌入变化量:", (embed_update.weight.data - original).norm()) 