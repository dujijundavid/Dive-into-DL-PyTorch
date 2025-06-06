"""
nn.Conv2d（二维卷积层）核心原理与用法
-------------------------------------
功能说明：
- nn.Conv2d 实现二维卷积操作，是图像、时空信号等任务的核心模块。

原理讲解：
- 输入 shape: [batch, in_channels, H, W]，输出 shape: [batch, out_channels, H_out, W_out]
- 卷积核参数自动学习，支持多通道、步幅、填充、分组等。
- 权重共享、局部感受野，参数量远小于全连接层。

使用场景：
- 图像分类、目标检测、分割、特征提取、时序建模等。

常见bug：
- 输入 shape 不匹配（如缺少 batch 维或通道数不符）。
- 卷积核大小、步幅、填充设置不当导致输出尺寸为0或负数。
- 忘记 .view/.reshape 展平特征用于全连接层。

深度学习研究员精华笔记：
- 卷积层的参数量与感受野、通道数、核大小密切相关。
- 合理设计步幅/填充可控制特征图尺寸，影响下游网络结构。
- 卷积可用于非图像任务（如一维信号、文本等）。

可运行案例：
"""
import torch
from torch import nn

# 1. 创建卷积层：输入3通道，输出8通道，3x3卷积核
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)

# 2. 输入一个 batch（4张3通道32x32图片）
x = torch.randn(4, 3, 32, 32)
out = conv(x)
print("输出 shape:", out.shape)  # [4, 8, 32, 32]

# 3. 查看参数量
print("权重 shape:", conv.weight.shape)
print("偏置 shape:", conv.bias.shape)

# 4. 卷积+激活+池化常见组合
model = nn.Sequential(
    nn.Conv2d(3, 8, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2, 2)
)
out2 = model(x)
print("卷积+激活+池化输出 shape:", out2.shape)

# 5. 展平特征用于全连接层
flat = out2.view(out2.size(0), -1)
print("展平后 shape:", flat.shape)

# 6. 输入 shape 不匹配 bug 演示
try:
    bad_x = torch.randn(4, 1, 32, 32)  # 通道数不符
    conv(bad_x)
except RuntimeError as e:
    print("输入 shape 不匹配报错:", e) 