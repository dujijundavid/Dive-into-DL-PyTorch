"""
nn.Conv2d（二维卷积层）核心原理与用法
-------------------------------------
第一性原理思考：
1. 什么是卷积？
   - 卷积是一种特殊的线性运算，用于提取局部特征
   - 通过滑动窗口和权重共享实现特征提取
   - 是图像处理中的基础操作

2. 为什么需要卷积层？
   - 参数共享：大大减少参数量
   - 局部连接：模拟生物视觉系统的感受野
   - 平移不变性：对输入的位置变化具有鲁棒性

3. 卷积层的核心特性是什么？
   - 卷积核：可学习的特征提取器
   - 步幅：控制特征图下采样的程度
   - 填充：控制输出特征图的大小
   - 多通道：提取不同层次的特征

苏格拉底式提问与验证：
1. 卷积核大小如何影响特征提取？
   - 问题：不同大小的卷积核有什么优缺点？
   - 验证：比较不同核大小的效果
   - 结论：大核提取大范围特征，小核提取细节特征

2. 为什么需要多通道？
   - 问题：单通道和多通道卷积的区别？
   - 验证：观察不同通道数的输出
   - 结论：多通道可以提取更丰富的特征

3. 步幅和填充的作用是什么？
   - 问题：如何控制特征图的大小？
   - 验证：尝试不同的步幅和填充设置
   - 结论：步幅控制下采样，填充保持特征图大小

费曼学习法讲解：
1. 概念解释
   - 用简单的滑动窗口解释卷积
   - 通过可视化理解卷积操作
   - 强调卷积在深度学习中的重要性

2. 实例教学
   - 从简单到复杂的卷积操作
   - 通过实际例子理解参数作用
   - 实践常见应用场景

3. 知识巩固
   - 总结卷积层的核心概念
   - 提供参数设置的最佳实践
   - 建议进阶学习方向

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
import matplotlib.pyplot as plt
import numpy as np

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

# 7. 验证不同卷积核大小的效果
print("\n验证不同卷积核大小：")
# 创建不同核大小的卷积层
conv3x3 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
conv5x5 = nn.Conv2d(1, 1, kernel_size=5, padding=2)

# 创建测试图像（简单的边缘）
test_img = torch.zeros(1, 1, 32, 32)
test_img[:, :, 15:17, :] = 1.0

# 应用卷积
with torch.no_grad():
    out3x3 = conv3x3(test_img)
    out5x5 = conv5x5(test_img)

# 可视化结果
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(test_img[0, 0].numpy())
plt.title('Original Image')
plt.subplot(132)
plt.imshow(out3x3[0, 0].numpy())
plt.title('3x3 Convolution Result')
plt.subplot(133)
plt.imshow(out5x5[0, 0].numpy())
plt.title('5x5 Convolution Result')
plt.show()

# 8. 验证步幅和填充的影响
print("\n验证步幅和填充：")
# 创建不同步幅和填充的卷积层
conv_stride1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
conv_stride2 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

# 测试输入
test_input = torch.randn(1, 1, 32, 32)

# 应用卷积
with torch.no_grad():
    out_stride1 = conv_stride1(test_input)
    out_stride2 = conv_stride2(test_input)

print(f"输入shape: {test_input.shape}")
print(f"stride=1输出shape: {out_stride1.shape}")
print(f"stride=2输出shape: {out_stride2.shape}")

# 9. 验证多通道卷积
print("\n验证多通道卷积：")
# 创建多通道卷积层
conv_multi = nn.Conv2d(3, 2, kernel_size=3, padding=1)

# 创建RGB测试图像
test_rgb = torch.randn(1, 3, 32, 32)

# 应用卷积
with torch.no_grad():
    out_multi = conv_multi(test_rgb)

print(f"RGB输入shape: {test_rgb.shape}")
print(f"多通道输出shape: {out_multi.shape}")

# 10. 验证参数量计算
print("\n验证参数量计算：")
def count_parameters(conv_layer):
    kernel_params = conv_layer.in_channels * conv_layer.out_channels * \
                   conv_layer.kernel_size[0] * conv_layer.kernel_size[1]
    bias_params = conv_layer.out_channels
    return kernel_params + bias_params

conv1 = nn.Conv2d(3, 64, kernel_size=3)
conv2 = nn.Conv2d(64, 128, kernel_size=3)

print(f"conv1参数量: {count_parameters(conv1)}")
print(f"conv2参数量: {count_parameters(conv2)}")
print(f"总参数量: {count_parameters(conv1) + count_parameters(conv2)}") 