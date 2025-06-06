"""
nn.Dropout（随机失活）核心原理与用法
-------------------------------------
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