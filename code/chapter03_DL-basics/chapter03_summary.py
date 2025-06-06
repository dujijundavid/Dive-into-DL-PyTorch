"""
PyTorch 深度学习基础模块（线性回归、向量化、MLP等）核心示例
--------------------------------------------------
【文件说明】
本文件系统梳理了深度学习入门阶段的核心知识点，包括：
- 向量化与广播机制
- 线性回归与多层感知机（MLP）模型
- 欠拟合/过拟合分析与正则化
- Dropout、L2正则等常用防过拟合手段
- 标准训练与推理流程

【设计意义与原理说明】
- 深度学习的本质是通过优化算法自动调整参数，使模型能够从数据中学习规律。每一步工程实现都对应着数学原理和优化目标。
- 向量化与广播机制：通过批量处理和自动对齐张量形状，极大提升了计算效率，使得大规模数据训练成为可能。这背后依赖于线性代数的高效实现（如BLAS库、GPU并行）。
- 线性回归与MLP：线性回归是最基础的监督学习模型，MLP通过引入非线性激活（如ReLU）和多层结构，能够拟合更复杂的函数关系。
- 欠拟合/过拟合与正则化：正则化（如L2、Dropout）通过限制模型复杂度或增加噪声，防止模型仅记忆训练集，从而提升泛化能力。
- 标准训练与推理流程：包括前向传播、损失计算、反向传播、参数更新等步骤，每一步都对应着优化理论中的梯度下降法。
- 工程实现与数学原理的结合，是深度学习高效、可扩展的关键。

【适用场景】
- 深度学习课程实验、项目原型、面试准备、知识查漏补缺
"""
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# 1. 向量化与广播机制
# -------------------
# 原理说明：
# 向量化操作是指用张量批量处理数据，避免显式for循环。这样可以充分利用底层的线性代数库（如BLAS）和GPU的并行计算能力，极大提升运算速度。
# 广播机制允许不同形状的张量自动对齐，简化代码并提升灵活性。
a = torch.ones(1000)  # 创建1000维全1向量
b = torch.ones(1000)  # 创建另一个同形状向量
c = a + b  # 推荐：向量加法，速度远快于for循环

# 广播机制的意义？
# 让不同形状的张量自动对齐，简化代码，提升灵活性。
x = torch.ones(3, 4)  # 3行4列
y = torch.ones(4)     # 1维长度为4
result = x + y        # 自动广播：y扩展为(3,4)后相加

# 2. 线性回归模型
# -------------------
# 原理说明：
# 线性回归通过最小化预测值与真实值的均方误差，学习输入特征与输出之间的线性关系。
# 这是神经网络的基础，也是理解更复杂模型的起点。
class LinearRegression(nn.Module):
    """简单线性回归模型"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)  # 单层线性变换
    def forward(self, x):
        return self.linear(x)

# 调用示例：输入2维，输出1维
model = LinearRegression(2, 1)
X = torch.randn(10, 2)  # 10个样本，每个2特征
y_pred = model(X)       # 前向推理

# 3. 多层感知机（MLP）
# -------------------
# 原理说明：
# MLP（多层感知机）通过堆叠多层线性变换和非线性激活（如ReLU），能够拟合复杂的非线性函数。
# 非线性激活的引入打破了线性模型的表达瓶颈，使神经网络具备"通用逼近"能力。
class MLP(nn.Module):
    """两层感知机，含ReLU激活"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # 第一层线性变换
            nn.ReLU(),                      # 非线性激活，提升表达能力
            nn.Linear(hidden_dim, out_dim)  # 输出层
        )
    def forward(self, x):
        return self.net(x)

mlp = MLP(784, 256, 10)         # 典型MNIST结构
X = torch.randn(64, 784)        # 64个样本，784特征（28x28图像）
out = mlp(X)                    # 前向推理

# 4. 欠拟合与过拟合分析
# -------------------
# 原理说明：
# 欠拟合：模型容量不足，无法捕捉数据规律，表现为训练/测试误差都高。
# 过拟合：模型容量过大，记住了训练集细节，泛化能力差，表现为测试误差高。
# 工程意义：需平衡模型复杂度与数据量，合理用正则化手段抑制过拟合。

# 5. 权重衰减（L2正则化）
# -------------------
# 原理说明：
# L2正则化通过在损失函数中加入参数平方和的惩罚项，抑制权重过大，防止模型对训练数据过度拟合。
# 其数学本质是让参数分布更平滑，提升泛化能力。
optimizer = torch.optim.SGD(mlp.parameters(), lr=0.1, weight_decay=1e-4)  # weight_decay即L2正则系数

# 6. Dropout正则化
# -------------------
# 原理说明：
# Dropout在训练时随机丢弃部分神经元输出，打破神经元间的"共适应"关系，等价于对模型集成平均，提升鲁棒性。
# 推理时自动关闭Dropout，保证输出稳定。
# 工程意义：与L2正则可叠加使用，进一步提升泛化能力。

dropout_mlp = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 10)
)

# 7. 设备管理与批量处理
# -------------------
# 原理说明：
# 现代深度学习模型参数量大、数据量大，需用GPU加速。PyTorch通过.to(device)接口灵活切换设备。
# 批量处理（mini-batch）能提升训练稳定性和效率。

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp = mlp.to(device)
X = X.to(device)

# 8. 调试与张量属性
# -------------------
# 原理说明：
# 理解张量的shape、dtype、device等属性，有助于排查数据流和设备迁移中的常见bug。
print('张量形状:', X.shape)      # (64, 784)
print('数据类型:', X.dtype)      # float32
print('所在设备:', X.device)     # cpu或cuda

# 9. 可运行的训练与预测流程示例
# -------------------
# 原理说明：
# 训练流程包括：前向传播（计算输出）、损失计算、反向传播（计算梯度）、参数更新。
# 这种流程对应着梯度下降法的数学原理，是深度学习模型学习的核心机制。
# 通过DataLoader实现批量训练，提升效率和泛化能力。
# 训练与推理模式的切换（train/eval）确保如Dropout、BatchNorm等层在不同阶段行为正确。

def accuracy(y_hat, y):
    """计算准确率"""
    preds = y_hat.argmax(dim=1)
    return (preds == y).float().mean().item()

# 生成模拟数据（如MNIST子集规模）
num_samples = 1000
num_features = 784
num_classes = 10
X_train = torch.randn(num_samples, num_features)
y_train = torch.randint(0, num_classes, (num_samples,))
X_test = torch.randn(200, num_features)
y_test = torch.randint(0, num_classes, (200,))

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)
train_iter = DataLoader(train_ds, batch_size=64, shuffle=True)
test_iter = DataLoader(test_ds, batch_size=64)

# 重新初始化模型和优化器，防止与前面变量冲突
demo_mlp = MLP(num_features, 256, num_classes)
optimizer = torch.optim.SGD(demo_mlp.parameters(), lr=0.1, weight_decay=1e-4)
loss_fn = nn.CrossEntropyLoss()

def train_and_eval(model, train_iter, test_iter, loss_fn, optimizer, num_epochs=5):
    """
    标准训练与评估流程，含loss和准确率打印
    原理说明：
    - optimizer.zero_grad(): 梯度清零，防止梯度累加。
    - model(X): 前向传播，计算输出。
    - loss_fn(y_hat, y): 计算损失，衡量预测与真实的差距。
    - loss.backward(): 反向传播，自动计算所有参数的梯度。
    - optimizer.step(): 根据梯度更新参数，实现梯度下降。
    - model.train()/model.eval(): 切换训练/推理模式，确保如Dropout等层行为正确。
    这些步骤共同实现了深度学习的核心优化过程。
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss, total_acc, n = 0, 0, 0
        for X, y in train_iter:
            optimizer.zero_grad()           # 梯度清零，防止梯度累加，确保每个batch独立更新
            y_hat = model(X)                # 前向传播，获得模型输出
            loss = loss_fn(y_hat, y)        # 计算损失，衡量预测与真实标签的差距
            loss.backward()                 # 反向传播，自动计算参数梯度
            optimizer.step()                # 参数更新，执行梯度下降
            total_loss += loss.item() * y.size(0)
            total_acc += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.size(0)
        print(f"epoch {epoch+1}, loss {total_loss/n:.4f}, train acc {total_acc/n:.3f}")
    # 测试集评估
    model.eval()  # 切换到推理模式，关闭Dropout等
    with torch.no_grad():
        total_acc, n = 0, 0
        for X, y in test_iter:
            y_hat = model(X)
            total_acc += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.size(0)
        print(f"test acc {total_acc/n:.3f}")
    model.train()  # 恢复训练模式

# 运行训练与评估
demo_mlp = demo_mlp.to(device)
train_and_eval(demo_mlp, train_iter, test_iter, loss_fn, optimizer, num_epochs=3)

# 训练后模型预测示例
# 原理说明：
# 训练完成后，模型可以对新样本进行预测。通过softmax输出最大概率的类别作为最终预测结果。
sample_X = torch.randn(5, num_features).to(device)
with torch.no_grad():
    pred = demo_mlp(sample_X)
    print("预测类别:", pred.argmax(dim=1).cpu().numpy()) 