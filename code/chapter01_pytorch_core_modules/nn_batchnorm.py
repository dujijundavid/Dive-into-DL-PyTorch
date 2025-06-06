"""
nn.BatchNorm（批量归一化）核心原理与用法
-----------------------------------------
功能说明：
- nn.BatchNorm 对每个通道做归一化，缓解梯度消失/爆炸，加速收敛。

原理讲解：
- 对每个 mini-batch，归一化为均值0方差1，并引入可学习缩放/偏移参数。
- 支持1D/2D/3D（如BatchNorm1d/2d/3d），常用于卷积/全连接层后。
- 训练/推理模式下行为不同（训练用batch统计，推理用全局均值方差）。

使用场景：
- 深层网络（CNN/MLP/RNN）训练加速、提升稳定性。

常见bug：
- 输入 shape 不匹配（如BatchNorm2d需[N,C,H,W]）。
- 推理时未切换到eval()，导致归一化不一致。
- 小batch下统计不稳定。

深度学习研究员精华笔记：
- BN 可缓解梯度消失/爆炸，允许更大学习率。
- BN 也有正则化效果，部分场景可替代Dropout。
- 小batch时可用GroupNorm/LayerNorm等替代。

可运行案例：
"""
import torch
from torch import nn

# 1. 创建BatchNorm2d层（常用于卷积输出）
bn = nn.BatchNorm2d(6)

# 2. 输入一个 batch（4张6通道10x10图片）
x = torch.randn(4, 6, 10, 10)
out = bn(x)
print("BatchNorm输出 shape:", out.shape)

# 3. 训练/推理模式切换
bn.eval()  # 切换到推理模式
out_eval = bn(x)
print("推理模式输出 shape:", out_eval.shape)

# 4. 输入 shape 不匹配 bug 演示
try:
    bad_x = torch.randn(4, 10, 10, 6)  # 通道维不对
    bn(bad_x)
except RuntimeError as e:
    print("输入 shape 不匹配报错:", e) 