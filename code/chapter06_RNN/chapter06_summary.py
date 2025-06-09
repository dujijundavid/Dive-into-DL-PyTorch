"""
PyTorch 循环神经网络（RNN/GRU/LSTM）核心技术与深度理解
-------------------------------------------------
【文件说明】
本文件系统梳理了循环神经网络的核心技术，包括：
- 基础RNN：理解序列建模的基本原理
- GRU（门控循环单元）：简化的门控机制
- LSTM（长短期记忆网络）：复杂的门控结构
- 序列数据处理与文本生成应用
- RNN变体的原理对比与选择策略

【第一性原理思考】
1. 为什么需要循环神经网络？
   - 传统前馈网络无法处理变长序列
   - 序列数据具有时间依赖性，需要记忆机制
   - 参数共享使得模型可以处理任意长度序列

2. 梯度消失问题的本质是什么？
   - 反向传播中梯度指数级衰减
   - 长期依赖难以学习
   - 导致网络只能记住短期信息

3. 门控机制如何解决问题？
   - 选择性记忆：决定保留什么信息
   - 选择性遗忘：决定丢弃什么信息
   - 信息流控制：调节信息传播

【苏格拉底式提问与验证】
1. GRU和LSTM哪个更好？
   - 问题：更多参数是否总是带来更好性能？
   - 验证：在不同任务上比较两者性能
   - 结论：取决于任务复杂度和数据量

2. 双向RNN为什么有效？
   - 问题：未来信息如何帮助理解当前？
   - 验证：通过序列标注任务验证
   - 结论：上下文信息提升理解质量

【费曼学习法讲解】
1. 概念解释
   - 用人类记忆类比RNN的工作机制
   - 用门卫类比门控单元的功能
   - 强调序列建模在实际中的重要性

2. 实例教学
   - 从简单序列预测开始
   - 扩展到复杂的语言建模
   - 通过可视化展示信息流动

【设计意义与工程价值】
- RNN是序列建模的基础，理解其原理对掌握现代序列模型至关重要
- 门控机制的思想影响了后续的注意力机制
- 在语音识别、机器翻译、文本生成等领域有广泛应用

可运行案例：
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math

# 1. 基础RNN原理与实现
# -------------------
# 原理说明：
# RNN通过隐藏状态在时间步间传递信息，实现序列建模
# h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)

print("1. 基础RNN原理与实现")
print("=" * 50)

class BasicRNNDemo:
    """基础RNN演示"""
    
    def rnn_computation_equivalence(self):
        """RNN计算等价性演示"""
        print("RNN计算的两种等价方式：")
        
        # 方式一：分别计算后相加
        X = torch.randn(3, 1)      # 输入
        H = torch.randn(3, 4)      # 隐藏状态
        W_xh = torch.randn(1, 4)   # 输入到隐藏的权重
        W_hh = torch.randn(4, 4)   # 隐藏到隐藏的权重
        
        output1 = torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
        
        # 方式二：拼接后一次计算
        X_H = torch.cat((X, H), dim=1)
        W = torch.cat((W_xh, W_hh), dim=0)
        output2 = torch.matmul(X_H, W)
        
        print(f"方式一输出: {output1.shape}")
        print(f"方式二输出: {output2.shape}")
        print(f"计算结果相等: {torch.allclose(output1, output2)}")
        
        # 这种等价性说明RNN本质上是线性变换的组合
    
    def vanilla_rnn_implementation(self):
        """原始RNN从头实现："""
        print("\n原始RNN从头实现：")
        
        class VanillaRNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.hidden_size = hidden_size
                
                # 参数初始化
                self.W_xh = nn.Parameter(torch.randn(input_size, hidden_size) * 0.1)
                self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
                self.b_h = nn.Parameter(torch.zeros(hidden_size))
                self.W_ho = nn.Parameter(torch.randn(hidden_size, output_size) * 0.1)
                self.b_o = nn.Parameter(torch.zeros(output_size))
            
            def forward(self, inputs, hidden=None):
                batch_size, seq_len, _ = inputs.size()
                
                if hidden is None:
                    hidden = torch.zeros(batch_size, self.hidden_size)
                
                outputs = []
                for t in range(seq_len):
                    x_t = inputs[:, t, :]
                    hidden = torch.tanh(x_t @ self.W_xh + hidden @ self.W_hh + self.b_h)
                    output = hidden @ self.W_ho + self.b_o
                    outputs.append(output.unsqueeze(1))
                
                return torch.cat(outputs, dim=1), hidden
        
        # 测试自实现的RNN
        input_size, hidden_size, output_size = 10, 20, 5
        seq_len, batch_size = 7, 3
        
        rnn = VanillaRNN(input_size, hidden_size, output_size)
        inputs = torch.randn(batch_size, seq_len, input_size)
        
        outputs, final_hidden = rnn(inputs)
        print(f"输出形状: {outputs.shape}")
        print(f"最终隐藏状态形状: {final_hidden.shape}")
        
        return rnn

# 运行基础RNN演示
basic_rnn_demo = BasicRNNDemo()
basic_rnn_demo.rnn_computation_equivalence()
custom_rnn = basic_rnn_demo.vanilla_rnn_implementation()

# 2. PyTorch内置RNN模块
# --------------------
# 原理说明：
# PyTorch提供了高效的RNN实现，支持多层、双向等配置

print("\n\n2. PyTorch内置RNN模块")
print("=" * 50)

class PyTorchRNNDemo:
    """PyTorch RNN模块演示"""
    
    def simple_rnn_model(self):
        """简单RNN模型"""
        print("简单RNN模型：")
        
        class SimpleRNN(nn.Module):
            def __init__(self, vocab_size, hidden_size, output_size, num_layers=1):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x, hidden=None):
                embedded = self.embedding(x)
                output, hidden = self.rnn(embedded, hidden)
                output = self.dropout(output)
                output = self.fc(output)
                return output, hidden
            
            def init_hidden(self, batch_size, device):
                return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        
        # 创建模型
        vocab_size, hidden_size, output_size = 1000, 128, 1000
        model = SimpleRNN(vocab_size, hidden_size, output_size, num_layers=2)
        
        # 测试前向传播
        batch_size, seq_len = 4, 10
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        hidden = model.init_hidden(batch_size, input_ids.device)
        
        output, final_hidden = model(input_ids, hidden)
        print(f"输入形状: {input_ids.shape}")
        print(f"输出形状: {output.shape}")
        print(f"最终隐藏状态形状: {final_hidden.shape}")
        
        return model
    
    def bidirectional_rnn(self):
        """双向RNN演示"""
        print("\n双向RNN演示：")
        
        class BiRNN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
            
            def forward(self, x):
                output, _ = self.rnn(x)
                output = self.fc(output)
                return output
        
        input_size, hidden_size, output_size = 50, 64, 10
        model = BiRNN(input_size, hidden_size, output_size)
        
        batch_size, seq_len = 3, 15
        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)
        
        print(f"双向RNN输出形状: {output.shape}")
        print("双向RNN能同时利用过去和未来的信息")

# 运行PyTorch RNN演示
pytorch_rnn_demo = PyTorchRNNDemo()
simple_model = pytorch_rnn_demo.simple_rnn_model()
pytorch_rnn_demo.bidirectional_rnn()

# 3. GRU（门控循环单元）
# ---------------------
# 原理说明：
# GRU通过重置门和更新门控制信息流动，简化了LSTM的结构

print("\n\n3. GRU（门控循环单元）")
print("=" * 50)

class GRUDemo:
    """GRU演示"""
    
    def gru_theory(self):
        """GRU理论解释"""
        print("GRU理论与公式：")
        print("1. 重置门: r_t = σ(W_r * [h_{t-1}, x_t])")
        print("2. 更新门: z_t = σ(W_z * [h_{t-1}, x_t])")
        print("3. 候选状态: h̃_t = tanh(W_h * [r_t ⊙ h_{t-1}, x_t])")
        print("4. 新状态: h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t")
        print("\n关键思想：")
        print("- 重置门控制多少过去信息用于计算候选状态")
        print("- 更新门控制新信息和旧信息的混合比例")
    
    def gru_implementation(self):
        """GRU实现与应用"""
        print("\nGRU实现与应用：")
        
        class GRUModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, 
                                 batch_first=True, dropout=0.1 if num_layers > 1 else 0)
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x, hidden=None):
                embedded = self.embedding(x)
                output, hidden = self.gru(embedded, hidden)
                output = self.dropout(output)
                output = self.fc(output)
                return output, hidden
        
        # 创建GRU模型
        vocab_size = 1000
        model = GRUModel(vocab_size, embedding_dim=128, hidden_size=256, 
                        output_size=vocab_size, num_layers=2)
        
        # 测试
        batch_size, seq_len = 4, 20
        input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
        output, hidden = model(input_seq)
        
        print(f"GRU模型参数量: {sum(p.numel() for p in model.parameters())}")
        print(f"输出形状: {output.shape}")
        
        return model
    
    def gru_vs_rnn_comparison(self):
        """GRU与RNN性能对比"""
        print("\nGRU与RNN性能对比：")
        
        # 创建简单的序列预测任务
        def create_sequence_data(seq_len=50, num_sequences=1000):
            """创建简单的序列预测数据"""
            sequences = []
            targets = []
            
            for _ in range(num_sequences):
                # 生成简单的正弦波序列
                start = np.random.random() * 2 * np.pi
                seq = np.sin(np.linspace(start, start + 4*np.pi, seq_len + 1))
                sequences.append(seq[:-1])
                targets.append(seq[1:])
            
            return torch.FloatTensor(sequences), torch.FloatTensor(targets)
        
        # 生成数据
        X, y = create_sequence_data(seq_len=30, num_sequences=500)
        X = X.unsqueeze(-1)  # 添加特征维度
        y = y.unsqueeze(-1)
        
        # 创建模型
        rnn_model = nn.RNN(1, 32, batch_first=True)
        gru_model = nn.GRU(1, 32, batch_first=True)
        
        # 简单训练比较
        def train_model(model, X, y, epochs=10):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            losses = []
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                output, _ = model(X)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            return losses[-1]
        
        rnn_loss = train_model(rnn_model, X, y)
        gru_loss = train_model(gru_model, X, y)
        
        print(f"RNN最终损失: {rnn_loss:.6f}")
        print(f"GRU最终损失: {gru_loss:.6f}")
        print("GRU通常在序列建模任务中表现更好")

# 运行GRU演示
gru_demo = GRUDemo()
gru_demo.gru_theory()
gru_model = gru_demo.gru_implementation()
gru_demo.gru_vs_rnn_comparison()

# 4. LSTM（长短期记忆网络）
# ------------------------
# 原理说明：
# LSTM通过遗忘门、输入门、输出门和细胞状态实现长期记忆

print("\n\n4. LSTM（长短期记忆网络）")
print("=" * 50)

class LSTMDemo:
    """LSTM演示"""
    
    def lstm_theory(self):
        """LSTM理论解释"""
        print("LSTM理论与公式：")
        print("1. 遗忘门: f_t = σ(W_f * [h_{t-1}, x_t] + b_f)")
        print("2. 输入门: i_t = σ(W_i * [h_{t-1}, x_t] + b_i)")
        print("3. 候选值: C̃_t = tanh(W_C * [h_{t-1}, x_t] + b_C)")
        print("4. 细胞状态: C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t")
        print("5. 输出门: o_t = σ(W_o * [h_{t-1}, x_t] + b_o)")
        print("6. 隐藏状态: h_t = o_t ⊙ tanh(C_t)")
        print("\n关键思想：")
        print("- 细胞状态作为信息高速公路，减少梯度消失")
        print("- 三个门控制信息的流入、流出和遗忘")
        print("- 比GRU更复杂，但理论上能处理更长的依赖")
    
    def lstm_implementation(self):
        """LSTM实现与应用"""
        print("\nLSTM实现与应用：")
        
        class LSTMModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                                   batch_first=True, dropout=0.1 if num_layers > 1 else 0)
                self.fc = nn.Linear(hidden_size, output_size)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x, hidden=None):
                embedded = self.embedding(x)
                output, hidden = self.lstm(embedded, hidden)
                output = self.dropout(output)
                output = self.fc(output)
                return output, hidden
        
        # 创建LSTM模型
        vocab_size = 1000
        model = LSTMModel(vocab_size, embedding_dim=128, hidden_size=256, 
                         output_size=vocab_size, num_layers=2)
        
        # 测试
        batch_size, seq_len = 4, 20
        input_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
        output, (hidden, cell) = model(input_seq)
        
        print(f"LSTM模型参数量: {sum(p.numel() for p in model.parameters())}")
        print(f"输出形状: {output.shape}")
        print(f"隐藏状态形状: {hidden.shape}")
        print(f"细胞状态形状: {cell.shape}")
        
        return model
    
    def lstm_memory_analysis(self):
        """LSTM记忆能力分析"""
        print("\nLSTM记忆能力分析：")
        
        # 创建长序列记忆任务
        def create_memory_task(seq_len=100, batch_size=32):
            """创建记忆任务：记住序列开始的信息"""
            sequences = torch.randint(1, 10, (batch_size, seq_len))
            # 目标是预测序列第一个元素
            targets = sequences[:, 0]
            # 将第一个元素后的大部分位置设为0（噪声）
            sequences[:, 10:90] = 0
            return sequences, targets
        
        # 生成数据
        X, y = create_memory_task(seq_len=100, batch_size=128)
        
        # 创建简单的LSTM模型
        class MemoryLSTM(nn.Module):
            def __init__(self, vocab_size=10, hidden_size=64):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, vocab_size)
            
            def forward(self, x):
                embedded = self.embedding(x)
                output, _ = self.lstm(embedded)
                # 只使用最后一个时间步的输出
                output = self.fc(output[:, -1, :])
                return output
        
        model = MemoryLSTM()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练几个epoch
        for epoch in range(20):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                accuracy = (output.argmax(1) == y).float().mean()
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={accuracy:.4f}")
        
        print("LSTM能够学习长期依赖关系")

# 运行LSTM演示
lstm_demo = LSTMDemo()
lstm_demo.lstm_theory()
lstm_model = lstm_demo.lstm_implementation()
lstm_demo.lstm_memory_analysis()

# 5. RNN变体对比与选择
# -------------------
# 原理说明：
# 不同RNN变体有各自的优缺点，需要根据任务选择

print("\n\n5. RNN变体对比与选择")
print("=" * 50)

class RNNComparisonDemo:
    """RNN变体对比演示"""
    
    def architecture_comparison(self):
        """架构对比"""
        print("RNN变体架构对比：")
        
        print("\n1. 基础RNN:")
        print("   - 参数最少，计算最快")
        print("   - 容易梯度消失，难以学习长依赖")
        print("   - 适用于短序列或简单任务")
        
        print("\n2. GRU:")
        print("   - 参数适中，计算效率高")
        print("   - 有效缓解梯度消失问题")
        print("   - 在多数任务上表现良好")
        
        print("\n3. LSTM:")
        print("   - 参数最多，计算相对慢")
        print("   - 最强的长期记忆能力")
        print("   - 在复杂序列任务中表现最佳")
    
    def performance_benchmark(self):
        """性能基准测试"""
        print("\n性能基准测试：")
        
        # 创建统一的测试框架
        def create_test_model(model_type, input_size=50, hidden_size=128, output_size=10):
            if model_type == 'RNN':
                return nn.RNN(input_size, hidden_size, batch_first=True)
            elif model_type == 'GRU':
                return nn.GRU(input_size, hidden_size, batch_first=True)
            elif model_type == 'LSTM':
                return nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # 测试不同模型的参数量
        input_size, hidden_size, output_size = 50, 128, 10
        models = {}
        
        for model_type in ['RNN', 'GRU', 'LSTM']:
            model = create_test_model(model_type, input_size, hidden_size, output_size)
            param_count = sum(p.numel() for p in model.parameters())
            models[model_type] = {'model': model, 'params': param_count}
        
        print("参数量对比：")
        for name, info in models.items():
            print(f"{name}: {info['params']:,} 参数")
        
        # 计算时间对比
        import time
        batch_size, seq_len = 32, 100
        x = torch.randn(batch_size, seq_len, input_size)
        
        print("\n前向传播时间对比：")
        for name, info in models.items():
            model = info['model']
            model.eval()
            
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    output, _ = model(x)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 100 * 1000  # ms
            print(f"{name}: {avg_time:.2f}ms")
    
    def selection_guidelines(self):
        """选择指南"""
        print("\n选择指南：")
        
        print("\n1. 根据序列长度：")
        print("   - 短序列（<20）：RNN")
        print("   - 中等序列（20-100）：GRU")
        print("   - 长序列（>100）：LSTM")
        
        print("\n2. 根据计算资源：")
        print("   - 资源受限：RNN > GRU > LSTM")
        print("   - 资源充足：LSTM > GRU > RNN")
        
        print("\n3. 根据任务类型：")
        print("   - 语言建模：LSTM")
        print("   - 情感分析：GRU")
        print("   - 简单分类：RNN")
        
        print("\n4. 根据训练数据：")
        print("   - 小数据集：GRU（防过拟合）")
        print("   - 大数据集：LSTM（充分利用容量）")

# 运行RNN对比演示
comparison_demo = RNNComparisonDemo()
comparison_demo.architecture_comparison()
comparison_demo.performance_benchmark()
comparison_demo.selection_guidelines()

# 6. 实际应用案例
# --------------
# 原理说明：
# 通过文本生成任务展示RNN在实际问题中的应用

print("\n\n6. 实际应用案例")
print("=" * 50)

class RNNApplicationDemo:
    """RNN应用案例演示"""
    
    def text_generation_model(self):
        """文本生成模型"""
        print("文本生成模型：")
        
        class TextGenerator(nn.Module):
            def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=2):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                                   batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, vocab_size)
                self.hidden_size = hidden_size
                self.num_layers = num_layers
            
            def forward(self, x, hidden=None):
                embedded = self.embedding(x)
                output, hidden = self.lstm(embedded, hidden)
                output = self.fc(output)
                return output, hidden
            
            def generate(self, start_tokens, max_length=50, temperature=1.0):
                """生成文本序列"""
                self.eval()
                with torch.no_grad():
                    current_tokens = start_tokens.clone()
                    generated = start_tokens.tolist()
                    hidden = None
                    
                    for _ in range(max_length):
                        output, hidden = self(current_tokens.unsqueeze(0), hidden)
                        
                        # 应用温度缩放
                        output = output[0, -1, :] / temperature
                        probs = F.softmax(output, dim=-1)
                        
                        # 采样下一个token
                        next_token = torch.multinomial(probs, 1)
                        generated.append(next_token.item())
                        
                        current_tokens = torch.cat([current_tokens, next_token])
                        current_tokens = current_tokens[-50:]  # 保持窗口大小
                
                return generated
        
        # 创建模型
        vocab_size = 1000
        model = TextGenerator(vocab_size)
        
        print(f"文本生成模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 演示生成过程
        start_tokens = torch.randint(0, vocab_size, (10,))
        generated_sequence = model.generate(start_tokens, max_length=20)
        
        print(f"起始序列: {start_tokens.tolist()}")
        print(f"生成序列: {generated_sequence}")
        
        return model
    
    def sequence_classification(self):
        """序列分类任务"""
        print("\n序列分类任务：")
        
        class SequenceClassifier(nn.Module):
            def __init__(self, vocab_size, embedding_dim=100, hidden_size=128, num_classes=3):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True, bidirectional=True)
                self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x):
                embedded = self.embedding(x)
                output, (hidden, _) = self.lstm(embedded)
                
                # 使用最后一个时间步的输出（双向）
                final_output = output[:, -1, :]
                final_output = self.dropout(final_output)
                output = self.fc(final_output)
                
                return output
        
        # 创建分类模型
        vocab_size, num_classes = 5000, 3
        classifier = SequenceClassifier(vocab_size, num_classes=num_classes)
        
        # 模拟训练数据
        batch_size, seq_len = 16, 50
        X = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, num_classes, (batch_size,))
        
        # 前向传播
        output = classifier(X)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        
        print(f"分类模型输出形状: {output.shape}")
        print(f"损失: {loss.item():.4f}")
        
        return classifier
    
    def training_best_practices(self):
        """训练最佳实践"""
        print("\n训练最佳实践：")
        
        print("1. 数据预处理：")
        print("   - 序列填充或截断到固定长度")
        print("   - 构建词汇表，处理OOV词")
        print("   - 适当的数据增强")
        
        print("\n2. 模型配置：")
        print("   - 合理的嵌入维度（100-300）")
        print("   - 隐藏层大小（128-512）")
        print("   - 适当的dropout（0.1-0.5）")
        
        print("\n3. 训练技巧：")
        print("   - 梯度裁剪防止梯度爆炸")
        print("   - 学习率调度")
        print("   - 早停和模型保存")
        
        print("\n4. 评估指标：")
        print("   - 困惑度（语言模型）")
        print("   - 准确率/F1（分类任务）")
        print("   - BLEU分数（生成任务）")

# 运行应用案例演示
app_demo = RNNApplicationDemo()
text_generator = app_demo.text_generation_model()
classifier = app_demo.sequence_classification()
app_demo.training_best_practices()

print("\n" + "=" * 50)
print("RNN技术总结：")
print("1. RNN是序列建模的基础，理解其原理是掌握序列模型的关键")
print("2. 梯度消失是RNN的核心问题，门控机制是有效解决方案")
print("3. GRU简化了LSTM的结构，在多数任务中表现良好")
print("4. LSTM具有最强的长期记忆能力，适合复杂序列任务")
print("5. 双向RNN能利用完整的上下文信息，提升理解质量")
print("6. 选择RNN变体需要考虑序列长度、计算资源和任务复杂度")
print("7. 现代应用中，RNN常与注意力机制结合使用")
print("8. 虽然Transformer在很多任务中超越了RNN，但RNN仍在特定场景中有价值") 