"""
PyTorch 卷积神经网络核心模块示例
--------------------------------
本文件梳理了 CNN 设计的典型模块及其原理、调用方式和应用场景，适合初学者和工程实践者深入理解和快速上手。

【典型工作流程】
1. 构建基础卷积/池化层，提取空间特征
2. 设计深层网络结构（如 LeNet、AlexNet、VGG、GoogLeNet、DenseNet、ResNet）
3. 加入归一化、激活、残差等机制提升性能
4. 训练与评估模型

【应用场景】
- 图像分类、目标检测、特征提取、迁移学习等
"""

import torch
from torch import nn

# 1. 基础卷积层
class SimpleConv(nn.Module):
    """
    基础二维卷积 + 激活 + 池化
    原理：卷积层用于提取局部空间特征，池化层降低特征图尺寸，提升不变性。
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, kernel_size=5)  # 5x5卷积核，1输入通道，6输出通道
        self.pool = nn.MaxPool2d(2, 2)              # 2x2最大池化
    def forward(self, x):
        # 先卷积提特征，再激活，再池化降采样
        return self.pool(torch.relu(self.conv(x)))

# 调用案例
x = torch.randn(1, 1, 28, 28)
model = SimpleConv()
out = model(x)
print("SimpleConv输出形状:", out.shape)  # [1, 6, 12, 12]

# 2. 多通道卷积
conv2d = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, stride=1)
# 原理：多输入/输出通道用于处理彩色图像和丰富特征表达

# 调用案例
x = torch.randn(4, 3, 32, 32)
out = conv2d(x)
print("多通道卷积输出形状:", out.shape)  # [4, 8, 32, 32]

# 3. 池化层
pool = nn.MaxPool2d(2, 2)
# 原理：池化降低特征图尺寸，减少参数，提升平移不变性

# 调用案例
x = torch.randn(2, 8, 16, 16)
out = pool(x)
print("池化输出形状:", out.shape)  # [2, 8, 8, 8]

# 4. LeNet 结构
class LeNet(nn.Module):
    """
    LeNet: 经典的卷积神经网络结构
    原理：通过卷积-池化-全连接的堆叠，实现端到端的图像分类
    适用场景：入门级图像分类任务，结构简单，易于理解。
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), nn.Sigmoid(), nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.Sigmoid(), nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120), nn.Sigmoid(),
            nn.Linear(120, 84), nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)

# 调用案例
x = torch.randn(8, 1, 28, 28)
net = LeNet()
out = net(x)
print("LeNet输出形状:", out.shape)  # [8, 10]

# 5. AlexNet 结构
class AlexNet(nn.Module):
    """
    AlexNet: 深层大规模卷积神经网络
    原理：更深的网络结构、更大的卷积核和通道数，使用ReLU和Dropout提升性能。
    适用场景：大规模图像分类任务，特征表达能力强。
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256*1*1, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)

# 调用案例
x = torch.randn(4, 1, 224, 224)
alexnet = AlexNet()
out = alexnet(x)
print("AlexNet输出形状:", out.shape)  # [4, 10]

# 6. VGG 结构
class VGGBlock(nn.Module):
    """
    VGG 块：多个相同参数的卷积层堆叠，提升特征提取能力。
    """
    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_channels = out_channels
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class VGG(nn.Module):
    """
    VGG: 重复卷积块+全连接层
    原理：通过堆叠多个VGGBlock，提升网络深度和特征表达能力。
    适用场景：深层网络设计，迁移学习。
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            VGGBlock(1, 64, 2), nn.MaxPool2d(2, 2),
            VGGBlock(64, 128, 2), nn.MaxPool2d(2, 2),
            VGGBlock(128, 256, 3), nn.MaxPool2d(2, 2),
            VGGBlock(256, 512, 3), nn.MaxPool2d(2, 2),
            VGGBlock(512, 512, 3), nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)

# 调用案例
x = torch.randn(2, 1, 32, 32)
vgg = VGG()
out = vgg(x)
print("VGG输出形状:", out.shape)  # [2, 10]

# 7. GoogLeNet (Inception) 结构
class Inception(nn.Module):
    """
    Inception 块：多分支并行卷积，融合多尺度特征。
    """
    def __init__(self, in_channels, c1, c2, c3, c4):
        super().__init__()
        # 1x1卷积
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 1x1卷积后接3x3卷积
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 1x1卷积后接5x5卷积
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 3x3最大池化后接1x1卷积
        self.p4_1 = nn.MaxPool2d(3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)
    def forward(self, x):
        p1 = torch.relu(self.p1_1(x))
        p2 = torch.relu(self.p2_2(torch.relu(self.p2_1(x))))
        p3 = torch.relu(self.p3_2(torch.relu(self.p3_1(x))))
        p4 = torch.relu(self.p4_2(self.p4_1(x)))
        return torch.cat([p1, p2, p3, p4], dim=1)

class GoogLeNet(nn.Module):
    """
    GoogLeNet: 多分支Inception结构，提升多尺度特征融合能力。
    适用场景：复杂图像分类，多尺度特征提取。
    """
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(), nn.MaxPool2d(3, 2, padding=1)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1), nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(3, 2, padding=1)
        )
        self.b3 = nn.Sequential(
            Inception(192, 64, (96, 128), (16, 32), 32),
            Inception(256, 128, (128, 192), (32, 96), 64), nn.MaxPool2d(3, 2, padding=1)
        )
        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128), nn.MaxPool2d(3, 2, padding=1)
        )
        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (192, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.fc = nn.Linear(1024, 10)
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return self.fc(x)

# 调用案例
x = torch.randn(2, 1, 96, 96)
googlenet = GoogLeNet()
out = googlenet(x)
print("GoogLeNet输出形状:", out.shape)  # [2, 10]

# 8. DenseNet 结构
class DenseBlock(nn.Module):
    """
    DenseBlock: 所有层特征拼接，提升特征复用和梯度流动。
    """
    def __init__(self, num_convs, in_channels, growth_rate):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers.append(self._conv_block(in_channels + i * growth_rate, growth_rate))
        self.net = nn.Sequential(*layers)
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        for layer in self.net:
            y = layer(x)
            x = torch.cat([x, y], dim=1)
        return x

class TransitionBlock(nn.Module):
    """
    过渡层：控制特征图尺寸和通道数，防止特征爆炸。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(2, 2)
        )
    def forward(self, x):
        return self.block(x)

class DenseNet(nn.Module):
    """
    DenseNet: 多个DenseBlock和过渡层堆叠，提升特征复用和梯度流动。
    适用场景：高效深层网络设计。
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, padding=1),
            DenseBlock(2, 64, 32), TransitionBlock(128, 64),
            DenseBlock(2, 64, 32), TransitionBlock(128, 64),
            DenseBlock(2, 64, 32), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.fc = nn.Linear(192, 10)
    def forward(self, x):
        x = self.net(x)
        return self.fc(x)

# 调用案例
x = torch.randn(2, 1, 64, 64)
densenet = DenseNet()
out = densenet(x)
print("DenseNet输出形状:", out.shape)  # [2, 10]

# 9. 批量归一化
bn = nn.BatchNorm2d(6)
# 原理：对每个通道做归一化，缓解梯度消失/爆炸，加速收敛

# 调用案例
x = torch.randn(4, 6, 10, 10)
out = bn(x)
print("BatchNorm输出形状:", out.shape)  # [4, 6, 10, 10]

# 10. 残差块
class Residual(nn.Module):
    """
    残差块（ResNet核心单元）
    原理：恒等映射+卷积，便于梯度流动，解决深层网络退化
    """
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
    def forward(self, X):
        Y = torch.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return torch.relu(Y + X)

# 调用案例
x = torch.randn(2, 3, 16, 16)
block = Residual(3, 3)
out = block(x)
print("残差块输出形状:", out.shape)  # [2, 3, 16, 16]

"""
【总结】
- 每个模块都可独立测试和组合，适合搭建各类 CNN 网络。
- 推荐工作流程：基础卷积/池化 → 组合深层结构（如LeNet/AlexNet/VGG/GoogLeNet/DenseNet/ResNet）→ 加入BatchNorm/残差块 → 训练与评估。
- 设计背后的核心思想：空间特征提取、参数高效、梯度流动、归一化提升训练稳定性、多尺度特征融合、特征复用。
""" 