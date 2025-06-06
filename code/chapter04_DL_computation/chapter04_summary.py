"""
PyTorch 深度学习计算模块（模型构建、参数管理、GPU加速等）核心示例
--------------------------------------------------
【文件说明】
本文件系统梳理了深度学习工程中常见的模型构建、参数初始化、自定义层、模型保存/加载、设备管理等关键环节。

【设计意义与原理说明】
- 深度学习的工程实现不仅仅是堆叠层，更重要的是理解每一步背后的机制和动机。
- 模型构建：通过继承nn.Module，实现灵活的网络结构定义。
- 参数初始化：合理初始化有助于加速收敛、避免梯度消失/爆炸。
- 自定义层：可扩展性是PyTorch的重要特性，便于实现创新结构。
- 模型保存/加载：支持断点续训、模型复用和部署。
- 设备管理：GPU加速是大规模深度学习的基础。

【适用场景】
- 深度学习课程实验、项目原型、模型调优、工程部署
"""
import torch
from torch import nn

# 1. 构建模型
# -------------------
# 原理说明：
# 通过继承nn.Module，用户可以灵活定义前向传播逻辑（forward），实现任意复杂的神经网络结构。
# 这种面向对象的设计便于模块化、复用和扩展。
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)  # 隐藏层，线性变换
        self.act = nn.ReLU()               # 非线性激活，提升模型表达能力
        self.output = nn.Linear(256, 10)   # 输出层
    def forward(self, x):
        # 前向传播：输入x依次经过隐藏层、激活、输出层
        return self.output(self.act(self.hidden(x)))

# 2. 参数初始化
# -------------------
# 原理说明：
# 合理的参数初始化有助于模型更快收敛，避免梯度消失/爆炸。
# 常用方法有正态分布初始化、常数初始化等。
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)  # 权重正态分布初始化
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)              # 偏置初始化为0

net = MLP()
net.apply(init_weights)  # 递归应用初始化函数到每一层

# 3. 自定义层
# -------------------
# 原理说明：
# 通过继承nn.Module，可以自定义任意前向逻辑，极大提升模型创新能力。
# 例如：实现一个"中心化"层，将输入减去均值，常用于归一化或特征处理。
class CenteredLayer(nn.Module):
    def forward(self, x):
        # 返回去均值后的张量
        return x - x.mean()

# 4. 保存与加载
# -------------------
# 原理说明：
# torch.save和load_state_dict支持模型参数的持久化，便于断点续训、迁移学习和部署。
# 只保存参数（state_dict），而不是整个模型对象，更灵活、兼容性好。
torch.save(net.state_dict(), 'mlp_params.pt')  # 保存参数到本地文件
net2 = MLP()                                   # 新建同结构模型
net2.load_state_dict(torch.load('mlp_params.pt'))  # 加载参数

# 5. GPU 加速
# -------------------
# 原理说明：
# 现代深度学习模型参数量大、数据量大，需用GPU加速。PyTorch通过.to(device)接口灵活切换设备。
# 只有模型和数据都迁移到同一设备，才能高效计算。
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net2.to(device)  # 将模型迁移到GPU（如可用）
x = torch.rand(2, 784).to(device)  # 随机生成2个样本，迁移到同一设备
y = net2(x)  # 前向推理，输出为(2, 10)
# y即为每个样本的10类得分，可进一步用softmax转为概率 