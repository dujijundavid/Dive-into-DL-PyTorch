# PyTorch 卷积神经网络模块学习笔记

---

## 5.1_conv-layer.ipynb

### 1. 功能说明
介绍二维卷积层的基本原理，包括二维互相关运算、卷积层的实现、边缘检测和通过数据学习卷积核。

### 2. 核心逻辑
- **二维互相关**：实现了基本的二维相关操作。
- **自定义卷积层**：通过继承 `nn.Module` 实现自定义卷积层。
- **边缘检测**：用卷积核检测图像边缘。
- **数据驱动学习卷积核**：通过梯度下降学习卷积核参数。

### 3. 应用场景
- 图像特征提取、边缘检测、基础卷积神经网络搭建。

### 4. 调用关系
- 独立使用，作为后续卷积网络的基础。

---

## 5.2_padding-and-strides.ipynb

### 1. 功能说明
讲解卷积层中的填充（padding）和步幅（stride）对输出形状的影响。

### 2. 核心逻辑
- **填充**：通过设置 padding 保持输出尺寸或控制输出大小。
- **步幅**：通过 stride 控制卷积窗口的移动步长，影响输出尺寸。

### 3. 应用场景
- 控制特征图尺寸，适配不同网络结构。

### 4. 调用关系
- 独立使用，常与卷积层搭配。

---

## 5.3_channels.ipynb

### 1. 功能说明
介绍多输入通道和多输出通道的卷积实现，以及 $1\times 1$ 卷积。

### 2. 核心逻辑
- **多输入通道**：对每个输入通道分别卷积后相加。
- **多输出通道**：每个输出通道有独立卷积核。
- **$1\times 1$ 卷积**：用于通道变换和特征压缩。

### 3. 应用场景
- 彩色图像处理、深层网络设计、特征压缩。

### 4. 调用关系
- 独立使用，常作为网络结构的基础。

---

## 5.4_pooling.ipynb

### 1. 功能说明
介绍池化层（最大池化、平均池化）的原理和实现。

### 2. 核心逻辑
- **最大池化/平均池化**：降低特征图尺寸，增强特征不变性。
- **多通道池化**：对每个通道分别池化。

### 3. 应用场景
- 降低特征图尺寸，防止过拟合，提升模型泛化能力。

### 4. 调用关系
- 独立使用，常与卷积层交替使用。

---

## 5.5_lenet.ipynb

### 1. 功能说明
实现经典卷积神经网络 LeNet，并在 Fashion-MNIST 上训练和评估。

### 2. 核心逻辑
- **LeNet 结构**：两层卷积+三层全连接。
- **训练与评估**：标准训练流程，准确率评估。

### 3. 应用场景
- 入门级图像分类任务，基础网络结构学习。

### 4. 调用关系
- 独立使用，作为后续复杂网络的基础。

---

## 5.6_alexnet.ipynb

### 1. 功能说明
实现深度卷积神经网络 AlexNet，并在 Fashion-MNIST 上训练和评估。

### 2. 核心逻辑
- **AlexNet 结构**：多层卷积+大规模全连接+Dropout。
- **训练与评估**：标准流程。

### 3. 应用场景
- 大规模图像分类，深层网络设计。

### 4. 调用关系
- 独立使用，作为深度网络的典型代表。

---

## 5.7_vgg.ipynb

### 1. 功能说明
实现 VGG 网络结构，强调重复卷积块的设计思想。

### 2. 核心逻辑
- **VGG 块**：多个相同参数的卷积层堆叠。
- **全连接层**：大容量分类器。

### 3. 应用场景
- 深层网络设计，特征提取能力提升。

### 4. 调用关系
- 独立使用，适合迁移学习和特征提取。

---

## 5.8_nin.ipynb

### 1. 功能说明
实现网络中的网络（NiN）结构，强调 $1\times 1$ 卷积的灵活性。

### 2. 核心逻辑
- **NiN 块**：卷积后接多个 $1\times 1$ 卷积。
- **全局平均池化**：替代全连接层。

### 3. 应用场景
- 参数高效的深层网络设计。

### 4. 调用关系
- 独立使用，适合轻量级网络设计。

---

## 5.9_googlenet.ipynb

### 1. 功能说明
实现 GoogLeNet（Inception）结构，突出多分支并行卷积。

### 2. 核心逻辑
- **Inception 块**：多种卷积核并行，特征融合。
- **全局平均池化**：输出分类结果。

### 3. 应用场景
- 多尺度特征提取，复杂图像分类。

### 4. 调用关系
- 独立使用，适合多分支网络设计。

---

## 5.10_batch-norm.ipynb

### 1. 功能说明
介绍批量归一化（BatchNorm）的原理、手动实现和 PyTorch 内置实现。

### 2. 核心逻辑
- **手动实现**：理解归一化原理。
- **内置实现**：简化代码，提升训练稳定性。

### 3. 应用场景
- 加速收敛，提升深层网络训练效果。

### 4. 调用关系
- 可嵌入任意网络结构中。

---

## 5.11_resnet.ipynb

### 1. 功能说明
实现残差网络（ResNet），解决深层网络退化问题。

### 2. 核心逻辑
- **残差块**：恒等映射+卷积，便于梯度传播。
- **ResNet 结构**：多层残差块堆叠。

### 3. 应用场景
- 极深网络设计，提升模型表现。

### 4. 调用关系
- 独立使用，适合复杂任务。

---

## 5.12_densenet.ipynb

### 1. 功能说明
实现稠密连接网络（DenseNet），提升特征复用和梯度流动。

### 2. 核心逻辑
- **稠密块**：所有层特征拼接，信息充分流动。
- **过渡层**：控制特征图尺寸和通道数。

### 3. 应用场景
- 高效特征复用，极深网络设计。

### 4. 调用关系
- 独立使用，适合高效深层网络。

---

## 高层次总结

本章系统梳理了卷积神经网络的基础与进阶内容，从基本卷积、池化、通道扩展，到主流深度网络（LeNet、AlexNet、VGG、NiN、GoogLeNet、ResNet、DenseNet）及批量归一化，帮助初学者全面掌握 CNN 设计与实现的核心方法。各模块既可独立学习，也可串联应用于实际项目开发。

---

## Python 代码示例（结构化调用关系）

```python
import torch
from torch import nn

# 1. 基础卷积层
class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        return self.pool(torch.relu(self.conv(x)))

# 2. 多通道卷积
conv2d = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, stride=1)

# 3. 池化层
pool = nn.MaxPool2d(2, 2)

# 4. LeNet 结构
class LeNet(nn.Module):
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

# 5. 批量归一化
bn = nn.BatchNorm2d(6)

# 6. 残差块
class Residual(nn.Module):
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
``` 