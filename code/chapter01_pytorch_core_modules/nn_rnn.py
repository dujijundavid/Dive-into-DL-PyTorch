"""
nn.RNN/LSTM/GRU（循环神经网络）核心原理与用法
-----------------------------------------------
第一性原理思考：
1. 什么是序列建模？
   - 处理具有时序依赖关系的数据
   - 需要考虑历史信息的影响
   - 输出可能依赖于整个输入序列

2. 为什么需要循环神经网络？
   - 处理变长序列：输入长度可以不同
   - 捕捉长期依赖：考虑历史信息
   - 参数共享：减少模型参数量

3. RNN的核心特性是什么？
   - 循环连接：信息在时间维度上传递
   - 隐藏状态：存储历史信息
   - 参数共享：不同时间步共享参数

苏格拉底式提问与验证：
1. 为什么需要LSTM和GRU？
   - 问题：RNN的局限性是什么？
   - 验证：比较不同模型的长序列表现
   - 结论：门控机制解决梯度问题

2. 隐藏状态的作用是什么？
   - 问题：如何存储和更新历史信息？
   - 验证：观察隐藏状态的变化
   - 结论：隐藏状态编码历史信息

3. 双向RNN的优势是什么？
   - 问题：为什么需要双向处理？
   - 验证：比较单向和双向的效果
   - 结论：双向可以捕捉双向依赖

费曼学习法讲解：
1. 概念解释
   - 用简单的循环结构解释RNN
   - 通过可视化理解信息流动
   - 强调RNN在序列建模中的重要性

2. 实例教学
   - 从简单到复杂的序列处理
   - 通过实际例子理解参数作用
   - 实践常见应用场景

3. 知识巩固
   - 总结RNN的核心概念
   - 提供使用的最佳实践
   - 建议进阶学习方向

功能说明：
- nn.RNN/LSTM/GRU 用于序列建模，能捕捉时序依赖，是NLP、时序预测等任务的核心模块。

原理讲解：
- RNN 通过隐藏状态递归传递信息，LSTM/GRU 通过门控机制缓解梯度消失/爆炸。
- 输入 shape: [seq_len, batch, input_size]，输出 shape: [seq_len, batch, hidden_size]
- 支持多层、双向、批量处理。

使用场景：
- 文本分类、序列标注、机器翻译、时间序列预测等。

常见bug：
- 输入 shape 不匹配（需 [seq_len, batch, input_size]）。
- 忘记初始化隐藏状态或未用 detach() 截断梯度。
- LSTM/GRU 输出是 (output, (h_n, c_n))，RNN 是 (output, h_n)。

深度学习研究员精华笔记：
- LSTM/GRU 在长序列建模上优于RNN，梯度更稳定。
- 可用 pack_padded_sequence 支持变长序列。
- 现代NLP多用Transformer，但RNN仍适合轻量场景。

可运行案例：
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# 1. 创建RNN层
rnn = nn.RNN(input_size=10, hidden_size=16, num_layers=1)
lstm = nn.LSTM(input_size=10, hidden_size=16, num_layers=1)
gru = nn.GRU(input_size=10, hidden_size=16, num_layers=1)

# 2. 输入一个序列（seq_len=5, batch=3, input_size=10）
x = torch.randn(5, 3, 10)
output, h_n = rnn(x)
print("RNN输出 shape:", output.shape, h_n.shape)
output, (h_n, c_n) = lstm(x)
print("LSTM输出 shape:", output.shape, h_n.shape, c_n.shape)
output, h_n = gru(x)
print("GRU输出 shape:", output.shape, h_n.shape)

# 3. 多层/双向RNN
rnn2 = nn.RNN(10, 16, num_layers=2, bidirectional=True)
out2, h2 = rnn2(x)
print("多层双向RNN输出 shape:", out2.shape)

# 4. 输入 shape 不匹配 bug 演示
try:
    bad_x = torch.randn(3, 5, 10)  # batch和seq_len顺序错了
    rnn(bad_x)
except RuntimeError as e:
    print("输入 shape 不匹配报错:", e)

# 5. 验证长序列依赖
print("\n验证长序列依赖：")
# 创建简单的序列数据
seq_len = 20
x = torch.zeros(seq_len, 1, 1)
x[0] = 1.0  # 在序列开始处设置一个脉冲

# 创建RNN和LSTM
rnn_long = nn.RNN(1, 1, num_layers=1)
lstm_long = nn.LSTM(1, 1, num_layers=1)

# 前向传播
with torch.no_grad():
    out_rnn, _ = rnn_long(x)
    out_lstm, _ = lstm_long(x)

# 绘制结果
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(out_rnn.squeeze().numpy())
plt.title('RNN Long Sequence Response')
plt.subplot(122)
plt.plot(out_lstm.squeeze().numpy())
plt.title('LSTM Long Sequence Response')
plt.show()

# 6. 验证隐藏状态
print("\n验证隐藏状态：")
# 创建序列数据
x = torch.randn(5, 1, 1)
rnn_hidden = nn.RNN(1, 1, num_layers=1)

# 记录隐藏状态
hidden_states = []
h = torch.zeros(1, 1, 1)  # 初始隐藏状态

# 逐步前向传播
with torch.no_grad():
    for t in range(5):
        out, h = rnn_hidden(x[t:t+1], h)
        hidden_states.append(h.squeeze().numpy())

# 绘制隐藏状态变化
plt.figure(figsize=(10, 5))
plt.plot(hidden_states)
plt.title('Hidden State Changes Over Time')
plt.xlabel('Time Step')
plt.ylabel('Hidden State Value')
plt.grid(True)
plt.show()

# 7. 验证双向RNN
print("\n验证双向RNN：")
# 创建序列数据
x = torch.randn(5, 1, 1)
rnn_bi = nn.RNN(1, 1, num_layers=1, bidirectional=True)

# 前向传播
with torch.no_grad():
    out, _ = rnn_bi(x)

# 分离前向和后向输出
forward_out = out[:, :, :1]
backward_out = out[:, :, 1:]

# 绘制结果
plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(forward_out.squeeze().numpy())
plt.title('Forward RNN Output')
plt.subplot(122)
plt.plot(backward_out.squeeze().numpy())
plt.title('Backward RNN Output')
plt.show()

# 8. 验证梯度流动
print("\n验证梯度流动：")
# 创建需要梯度的输入
x = torch.randn(5, 1, 1, requires_grad=True)
rnn_grad = nn.RNN(1, 1, num_layers=1)

# 前向传播
y, _ = rnn_grad(x)
loss = y.sum()
loss.backward()

print("输入梯度是否存在:", x.grad is not None)
print("输入梯度形状:", x.grad.shape)
print("输入梯度范数:", x.grad.norm().item())

# 9. 验证多层RNN
print("\n验证多层RNN：")
# 创建多层RNN
rnn_multi = nn.RNN(1, 1, num_layers=3)
x = torch.randn(5, 1, 1)

# 前向传播
with torch.no_grad():
    out, h = rnn_multi(x)

print("多层RNN输出形状:", out.shape)
print("多层RNN隐藏状态形状:", h.shape)

# 10. 验证序列长度影响
print("\n验证序列长度影响：")
# 测试不同序列长度
lengths = [5, 10, 20, 50]
rnn_len = nn.RNN(1, 1, num_layers=1)

results = []
for length in lengths:
    x = torch.randn(length, 1, 1)
    with torch.no_grad():
        out, _ = rnn_len(x)
    results.append(out[-1].item())

plt.figure(figsize=(10, 5))
plt.plot(lengths, results, 'o-')
plt.xlabel('Sequence Length')
plt.ylabel('Final Output Value')
plt.title('Impact of Sequence Length on Output')
plt.grid(True)
plt.show() 