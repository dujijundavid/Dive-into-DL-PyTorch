"""
nn.Embedding（嵌入层）核心原理与用法
-------------------------------------
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