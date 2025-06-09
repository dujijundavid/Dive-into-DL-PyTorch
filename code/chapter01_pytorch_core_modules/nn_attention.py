"""
注意力机制核心原理与用法
---------------------
第一性原理思考：
1. 什么是注意力机制？
   - 注意力机制允许模型动态地关注输入的不同部分
   - 通过计算查询、键、值之间的相似性来分配注意力权重
   - 是现代深度学习（特别是Transformer）的核心组件

2. 为什么需要注意力机制？
   - 解决长序列依赖问题：传统RNN难以处理长距离依赖
   - 并行化计算：相比RNN，注意力机制可以并行计算
   - 可解释性：注意力权重提供了模型决策的直观解释
   - 选择性聚焦：动态地选择相关信息

3. 注意力机制的核心特性是什么？
   - 查询-键-值结构：Q、K、V三元组是注意力的基础
   - 权重归一化：使用softmax确保权重和为1
   - 多头机制：多个注意力头捕获不同类型的关系
   - 缩放机制：防止softmax梯度消失

苏格拉底式提问与验证：
1. 注意力机制如何工作？
   - 问题：Q、K、V分别代表什么？
   - 验证：实现基础的注意力计算过程
   - 结论：注意力是加权平均，权重由Q和K的相似性决定

2. 为什么需要多头注意力？
   - 问题：单头注意力有什么局限性？
   - 验证：比较单头和多头注意力的表现
   - 结论：多头注意力捕获不同类型的关系

3. 位置编码的作用是什么？
   - 问题：注意力机制如何处理序列位置信息？
   - 验证：有无位置编码的效果对比
   - 结论：位置编码为无序的注意力机制注入位置信息

费曼学习法讲解：
1. 概念解释
   - 用搜索引擎类比解释注意力机制
   - 通过图书馆检索理解Q、K、V的作用
   - 强调注意力在现代AI中的革命性意义

2. 实例教学
   - 从简单到复杂的注意力实现
   - 通过可视化理解注意力权重
   - 实践不同类型的注意力机制

3. 知识巩固
   - 总结各种注意力变体的特点
   - 提供注意力机制设计的指导原则
   - 建议进阶学习方向

功能说明：
- 注意力机制是现代深度学习的核心，特别是在NLP和CV领域。

原理讲解：
- 通过计算查询与键的相似性，对值进行加权聚合。
- 多头注意力并行计算多个注意力子空间的信息。
- 自注意力允许序列内部元素之间的交互。

工程意义：
- 是Transformer、BERT、GPT等先进模型的基础组件。

可运行案例：
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt

# 1. 基础注意力机制
print("1. 基础注意力机制：")

def basic_attention(query, key, value, mask=None):
    """
    基础注意力机制实现
    Args:
        query: [batch_size, seq_len_q, d_k]
        key: [batch_size, seq_len_k, d_k]  
        value: [batch_size, seq_len_v, d_v] (seq_len_v == seq_len_k)
        mask: [batch_size, seq_len_q, seq_len_k]
    """
    d_k = query.size(-1)
    
    # 计算注意力分数: Q * K^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码（如果提供）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 计算注意力权重
    attention_weights = F.softmax(scores, dim=-1)
    
    # 应用注意力权重到值
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights

# 测试基础注意力
batch_size, seq_len, d_model = 2, 5, 64
query = torch.randn(batch_size, seq_len, d_model)
key = torch.randn(batch_size, seq_len, d_model)
value = torch.randn(batch_size, seq_len, d_model)

output, weights = basic_attention(query, key, value)
print(f"输入形状: {query.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")
print(f"注意力权重示例:\n{weights[0]}")

# 2. 缩放点积注意力
print("\n2. 缩放点积注意力：")

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        attention, log_attention = self.get_attention_weights(q, k, mask)
        output = torch.matmul(self.dropout(attention), v)
        return output, attention
    
    def get_attention_weights(self, q, k, mask=None):
        attention = torch.matmul(q / self.temperature, k.transpose(-2, -1))
        
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        
        log_attention = F.log_softmax(attention, dim=-1)
        attention = F.softmax(attention, dim=-1)
        
        return attention, log_attention

# 测试缩放点积注意力
temperature = math.sqrt(d_model)
attention_layer = ScaledDotProductAttention(temperature)
output, weights = attention_layer(query, key, value)
print(f"缩放点积注意力输出形状: {output.shape}")

# 3. 多头注意力机制
print("\n3. 多头注意力机制：")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(temperature=math.sqrt(self.d_k))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性变换并重塑为多头
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 应用注意力
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        output, attention = self.attention(q, k, v, mask=mask)
        
        # 3. 连接多头输出
        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 4. 最终线性变换
        output = self.w_o(output)
        
        return output, attention

# 测试多头注意力
num_heads = 8
multi_head_attention = MultiHeadAttention(d_model, num_heads)
output, weights = multi_head_attention(query, key, value)
print(f"多头注意力输出形状: {output.shape}")
print(f"多头注意力权重形状: {weights.shape}")

# 4. 自注意力机制
print("\n4. 自注意力机制：")

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
    
    def forward(self, x, mask=None):
        # 自注意力：Q=K=V=x
        return self.multihead_attention(x, x, x, mask)

# 测试自注意力
self_attention = SelfAttention(d_model, num_heads)
self_output, self_weights = self_attention(query)
print(f"自注意力输出形状: {self_output.shape}")

# 5. 位置编码
print("\n5. 位置编码：")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

# 测试位置编码
pos_encoding = PositionalEncoding(d_model)
input_embeddings = torch.randn(seq_len, batch_size, d_model)
encoded_input = pos_encoding(input_embeddings)
print(f"位置编码前: {input_embeddings.shape}")
print(f"位置编码后: {encoded_input.shape}")

# 可视化位置编码
pe_example = PositionalEncoding(128, max_len=100)
pos_data = pe_example.pe[:100, 0, :].numpy()

plt.figure(figsize=(12, 8))
plt.imshow(pos_data.T, aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Positional Encoding Visualization')
plt.xlabel('Position')
plt.ylabel('Encoding Dimension')
# plt.show()  # Uncomment to display image
print("Positional encoding visualization is ready (uncomment plt.show() to view graphics)")

# 6. 交叉注意力机制
print("\n6. 交叉注意力机制：")

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
    
    def forward(self, query, context, mask=None):
        # 交叉注意力：Q来自query，K和V来自context
        return self.multihead_attention(query, context, context, mask)

# 测试交叉注意力
cross_attention = CrossAttention(d_model, num_heads)
context = torch.randn(batch_size, 8, d_model)  # 不同长度的上下文
cross_output, cross_weights = cross_attention(query, context)
print(f"交叉注意力输出形状: {cross_output.shape}")
print(f"交叉注意力权重形状: {cross_weights.shape}")

# 7. 因果注意力（掩码注意力）
print("\n7. 因果注意力（掩码注意力）：")

def create_causal_mask(seq_len):
    """创建因果掩码，防止模型看到未来信息"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(d_model, num_heads)
    
    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        causal_mask = create_causal_mask(seq_len).unsqueeze(0).expand(batch_size, -1, -1)
        return self.multihead_attention(x, x, x, mask=causal_mask)

# 测试因果注意力
causal_attention = CausalSelfAttention(d_model, num_heads)
causal_output, causal_weights = causal_attention(query)
print(f"因果注意力输出形状: {causal_output.shape}")
print(f"因果掩码形状: {create_causal_mask(seq_len).shape}")

# 可视化因果掩码
causal_mask_vis = create_causal_mask(10)
plt.figure(figsize=(8, 6))
plt.imshow(causal_mask_vis.numpy(), cmap='Blues')
plt.title('Causal Attention Mask')
plt.xlabel('Key Position')
plt.ylabel('Query Position')
# plt.show()  # Uncomment to display image
print("Causal mask visualization is ready")

# 8. 相对位置编码
print("\n8. 相对位置编码：")

class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model, max_relative_position=20):
        super().__init__()
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        
        # 创建相对位置嵌入
        self.relative_positions_embeddings = nn.Embedding(
            2 * max_relative_position + 1, d_model
        )
    
    def forward(self, seq_len):
        # 创建相对位置矩阵
        range_vec = torch.arange(seq_len)
        range_mat = range_vec.unsqueeze(0).repeat(seq_len, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # 截断到最大相对位置
        distance_mat_clipped = torch.clamp(
            distance_mat, -self.max_relative_position, self.max_relative_position
        )
        
        # 转换为正索引
        final_mat = distance_mat_clipped + self.max_relative_position
        
        # 获取嵌入
        embeddings = self.relative_positions_embeddings(final_mat)
        
        return embeddings

# 测试相对位置编码
rel_pos_encoding = RelativePositionEncoding(d_model)
rel_pos_embeddings = rel_pos_encoding(seq_len)
print(f"相对位置编码形状: {rel_pos_embeddings.shape}")

# 9. 注意力可视化
print("\n9. Attention Visualization:")

def visualize_attention_weights(attention_weights, input_tokens=None, layer_idx=0, head_idx=0):
    """可视化注意力权重"""
    # 选择特定层和头的注意力权重
    weights = attention_weights[0, head_idx].detach().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(weights, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    
    if input_tokens:
        plt.xticks(range(len(input_tokens)), input_tokens, rotation=45)
        plt.yticks(range(len(input_tokens)), input_tokens)
    
    plt.title(f'Attention Weight Heatmap (Layer {layer_idx}, Head {head_idx})')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    # plt.tight_layout()
    # plt.show()

# 生成示例注意力权重进行可视化
example_tokens = ['I', 'love', 'machine', 'learning', 'very', 'much']
example_input = torch.randn(1, len(example_tokens), d_model)
_, example_weights = multi_head_attention(example_input, example_input, example_input)

print("Attention weight visualization is ready")
# visualize_attention_weights(example_weights, example_tokens)  # Uncomment to display image

# 10. Transformer编码器层
print("\n10. Transformer编码器层：")

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attention_output, attention_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attention_weights

# 测试Transformer编码器层
d_ff = 256
encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
encoder_output, encoder_weights = encoder_layer(query)
print(f"Transformer编码器输出形状: {encoder_output.shape}")

# 11. 高效注意力变体
print("\n11. 高效注意力变体：")

class LinearAttention(nn.Module):
    """线性注意力：O(n)复杂度而非O(n²)"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 使用特征映射函数（这里简化为ReLU）
        q = F.relu(q) + 1e-6
        k = F.relu(k) + 1e-6
        
        # 线性注意力计算
        kv = torch.matmul(k.transpose(-2, -1), v)  # [batch, heads, d_k, d_k]
        qkv = torch.matmul(q, kv)  # [batch, heads, seq_len, d_k]
        
        # 归一化
        k_sum = k.sum(dim=-2, keepdim=True)  # [batch, heads, 1, d_k]
        q_k_sum = torch.matmul(q, k_sum.transpose(-2, -1))  # [batch, heads, seq_len, 1]
        output = qkv / (q_k_sum + 1e-6)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(output)

# 测试线性注意力
linear_attention = LinearAttention(d_model, num_heads)
linear_output = linear_attention(query)
print(f"线性注意力输出形状: {linear_output.shape}")

# 12. 注意力机制的应用示例
print("\n12. 注意力机制的应用示例：")

class SimpleTransformer(nn.Module):
    """简化的Transformer模型"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len=1000):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, mask=None):
        # 嵌入和位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        
        attention_weights = []
        
        # 通过编码器层
        for layer in self.layers:
            x, weights = layer(x, mask)
            attention_weights.append(weights)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output, attention_weights

# 测试简化Transformer
vocab_size = 1000
num_layers = 2
simple_transformer = SimpleTransformer(vocab_size, d_model, num_heads, num_layers)

# 创建输入序列（词汇索引）
input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
output, all_attention_weights = simple_transformer(input_ids)

print(f"Transformer输入形状: {input_ids.shape}")
print(f"Transformer输出形状: {output.shape}")
print(f"注意力权重层数: {len(all_attention_weights)}")

# 13. 注意力机制的性能分析
print("\n13. 注意力机制的性能分析：")

def analyze_attention_complexity():
    """分析不同注意力机制的计算复杂度"""
    seq_lengths = [128, 256, 512, 1024, 2048]
    
    print("序列长度对注意力机制的影响:")
    print("序列长度\t标准注意力复杂度\t线性注意力复杂度")
    
    for seq_len in seq_lengths:
        standard_complexity = seq_len ** 2 * d_model
        linear_complexity = seq_len * d_model ** 2
        
        print(f"{seq_len}\t\t{standard_complexity:,}\t\t{linear_complexity:,}")

analyze_attention_complexity()

# 14. 注意力机制选择指南
print("\n14. 注意力机制选择指南：")

def attention_selection_guide():
    """注意力机制选择指南"""
    guide = {
        '标准应用': {
            '自注意力': '序列建模、文本理解、图像patch处理',
            '交叉注意力': '机器翻译、问答系统、图像描述',
            '多头注意力': '大多数Transformer应用的标准选择'
        },
        '特殊需求': {
            '因果注意力': '语言生成、自回归任务',
            '稀疏注意力': '长序列处理、计算资源受限',
            '线性注意力': '超长序列、实时应用'
        },
        '优化变体': {
            '局部注意力': '处理局部依赖关系',
            '相对位置注意力': '需要位置敏感的任务',
            '可学习位置注意力': '复杂的位置关系建模'
        }
    }
    
    print("注意力机制选择指南:")
    for category, mechanisms in guide.items():
        print(f"\n{category}:")
        for mechanism, usage in mechanisms.items():
            print(f"  {mechanism}: {usage}")

attention_selection_guide()

print("\n注意力机制使用要点总结:")
print("1. 理解Q、K、V的作用：Query查询、Key键、Value值")
print("2. 合理设置头数：一般8-16个头效果较好")
print("3. 注意序列长度：标准注意力复杂度为O(n²)")
print("4. 使用适当的掩码：因果掩码、填充掩码等")
print("5. 考虑位置编码：绝对位置或相对位置编码")
print("6. 选择合适的缩放因子：通常使用sqrt(d_k)")
print("7. 注意梯度流动：使用残差连接和层归一化")
print("8. 可视化注意力权重：帮助理解模型行为") 