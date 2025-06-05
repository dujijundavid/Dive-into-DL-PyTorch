"""
PyTorch 循环神经网络（RNN/GRU/LSTM）核心模块示例
--------------------------------------------------
本文件梳理了RNN、GRU、LSTM的原理、结构、调用方式和对比，适合初学者深入理解和快速上手。

【工作流程】
1. 数据预处理与采样
2. 选择RNN结构（RNN/GRU/LSTM）
3. 构建模型与训练
4. 文本生成与评估

【对比说明】
- RNN：结构简单，易受梯度消失影响，适合短依赖。
- GRU：门控机制简化，参数更少，训练快，适合中等依赖。
- LSTM：多门控结构，捕捉长依赖，效果最好但参数多。
"""

import torch
from torch import nn

# 1. 基础RNN单元原理演示
X, W_xh = torch.randn(3, 1), torch.randn(1, 4)
H, W_hh = torch.randn(3, 4), torch.randn(4, 4)
# 方式一：分别线性变换后相加
out1 = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
# 方式二：拼接后一次线性变换
out2 = torch.matmul(torch.cat((X, H), dim=1), torch.cat((W_xh, W_hh), dim=0))
assert torch.allclose(out1, out2)

# 2. RNN模型（PyTorch实现）
class SimpleRNN(nn.Module):
    """
    简单RNN模型，适合短序列建模。
    """
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()
        self.rnn = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)
        self.fc = nn.Linear(num_hiddens, vocab_size)
    def forward(self, X, state=None):
        Y, state = self.rnn(X, state)
        out = self.fc(Y)
        return out, state

# 调用案例
vocab_size, num_hiddens = 100, 16
X = torch.rand(5, 2, vocab_size)  # (seq_len, batch, input_size)
model = SimpleRNN(vocab_size, num_hiddens)
out, state = model(X)
print("SimpleRNN输出形状:", out.shape)

# 3. GRU模型
class SimpleGRU(nn.Module):
    """
    GRU模型，门控机制缓解梯度消失，适合中等依赖。
    """
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()
        self.gru = nn.GRU(input_size=vocab_size, hidden_size=num_hiddens)
        self.fc = nn.Linear(num_hiddens, vocab_size)
    def forward(self, X, state=None):
        Y, state = self.gru(X, state)
        out = self.fc(Y)
        return out, state

# 调用案例
gru_model = SimpleGRU(vocab_size, num_hiddens)
out, state = gru_model(X)
print("GRU输出形状:", out.shape)

# 4. LSTM模型
class SimpleLSTM(nn.Module):
    """
    LSTM模型，多门控结构，适合长依赖序列。
    """
    def __init__(self, vocab_size, num_hiddens):
        super().__init__()
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=num_hiddens)
        self.fc = nn.Linear(num_hiddens, vocab_size)
    def forward(self, X, state=None):
        Y, state = self.lstm(X, state)
        out = self.fc(Y)
        return out, state

# 调用案例
lstm_model = SimpleLSTM(vocab_size, num_hiddens)
out, state = lstm_model(X)
print("LSTM输出形状:", out.shape)

"""
【原理对比与总结】
- RNN：每步仅依赖前一隐藏状态，易梯度消失。
- GRU：引入更新门/重置门，信息流动更灵活，训练快。
- LSTM：引入输入门/遗忘门/输出门和记忆单元，能长期记忆。
- PyTorch高阶API极大简化了模型搭建和训练流程。

【典型应用场景】
- 文本生成、语言建模、机器翻译、语音识别、时间序列预测等。
""" 