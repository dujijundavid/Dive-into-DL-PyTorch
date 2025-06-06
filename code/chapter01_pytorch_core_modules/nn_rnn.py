"""
nn.RNN/LSTM/GRU（循环神经网络）核心原理与用法
-----------------------------------------------
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