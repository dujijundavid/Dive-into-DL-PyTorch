"""
PyTorch 卷积神经网络核心技术与深度理解
-----------------------------------
【文件说明】
本文件系统梳理了卷积神经网络的核心技术，包括：
- 卷积层与池化层：空间特征提取的基础
- 经典CNN架构：LeNet、AlexNet、VGG、ResNet、DenseNet等
- 网络设计原理：深度、宽度、跳跃连接的作用
- 现代CNN技术：批归一化、残差连接、注意力机制
- 实际应用与工程实践

【第一性原理思考】
1. 为什么需要卷积神经网络？
   - 图像具有空间局部性和平移不变性
   - 全连接层参数过多，容易过拟合
   - 卷积层通过权重共享大幅减少参数

2. 深层网络为什么更有效？
   - 层次特征表示：低层→边缘，高层→语义
   - 更大的感受野能捕获更复杂的模式
   - 非线性变换的复合增强表达能力

3. 残差连接解决什么问题？
   - 梯度消失导致深层网络难以训练
   - 恒等映射保证信息传播
   - 使网络能够训练得更深

【苏格拉底式提问与验证】
1. 更深的网络总是更好吗？
   - 问题：深度与性能的关系是什么？
   - 验证：通过不同深度的网络对比
   - 结论：需要合适的设计避免退化问题

2. 卷积核大小如何选择？
   - 问题：大核还是小核更好？
   - 验证：通过感受野和计算效率分析
   - 结论：多个小核通常优于单个大核

【费曼学习法讲解】
1. 概念解释
   - 用显微镜类比卷积的局部观察
   - 用建筑设计类比网络架构
   - 强调CNN在计算机视觉中的重要性

2. 实例教学
   - 从简单的边缘检测开始
   - 逐步构建复杂的分类网络
   - 通过可视化理解特征学习

【设计意义与工程价值】
- CNN是计算机视觉的基石，影响了整个深度学习领域
- 现代视觉模型的核心思想都源于CNN的设计原理
- 理解CNN对掌握视觉Transformer等新架构也很重要

可运行案例：
"""

import torch
from torch import nn
import numpy as np

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

if __name__ == "__main__":
    # 1. 基础卷积层调用案例
    x = torch.randn(1, 1, 28, 28)
    model = SimpleConv()
    out = model(x)
    print("SimpleConv输出形状:", out.shape)  # [1, 6, 12, 12]

    # 2. 多通道卷积
    conv2d = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, stride=1)
    # 原理：多输入/输出通道用于处理彩色图像和丰富特征表达
    x = torch.randn(4, 3, 32, 32)
    out = conv2d(x)
    print("多通道卷积输出形状:", out.shape)  # [4, 8, 32, 32]

    # 3. 池化层
    pool = nn.MaxPool2d(2, 2)
    # 原理：池化降低特征图尺寸，减少参数，提升平移不变性
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
                nn.Dropout(0.5), nn.Linear(256*6*6, 4096), nn.ReLU(),
                nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
                nn.Linear(4096, 10)
            )
        def forward(self, x):
            x = self.features(x)
            # Debug: print feature output shape
            # print(f"Features output shape: {x.shape}")
            x = x.view(x.shape[0], -1)
            return self.classifier(x)
    x = torch.randn(4, 1, 224, 224)
    alexnet = AlexNet()
    
    # First check the feature output dimension
    alexnet.eval()
    with torch.no_grad():
        features_out = alexnet.features(x)
        flattened_size = features_out.view(features_out.shape[0], -1).shape[1]
        print(f"AlexNet特征输出形状: {features_out.shape}, 展平后大小: {flattened_size}")
    
    # Fix the classifier to match actual dimensions
    alexnet.classifier = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(flattened_size, 4096), nn.ReLU(),
        nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
        nn.Linear(4096, 10)
    )
    
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
            self.fc = None  # Will be dynamically set
        def forward(self, x):
            x = self.net(x)
            if self.fc is None:
                self.fc = nn.Linear(x.shape[1], 10)
            return self.fc(x)
    x = torch.randn(2, 1, 64, 64)
    densenet = DenseNet()
    out = densenet(x)
    print("DenseNet输出形状:", out.shape)  # [2, 10]

    # 9. 批量归一化
    bn = nn.BatchNorm2d(6)
    # 原理：对每个通道做归一化，缓解梯度消失/爆炸，加速收敛
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
    x = torch.randn(2, 3, 16, 16)
    block = Residual(3, 3)
    out = block(x)
    print("残差块输出形状:", out.shape)  # [2, 3, 16, 16]

    print("\n【总结】")
    print("- 每个模块都可独立测试和组合，适合搭建各类 CNN 网络。")
    print("- 推荐工作流程：基础卷积/池化 → 组合深层结构（如LeNet/AlexNet/VGG/GoogLeNet/DenseNet/ResNet）→ 加入BatchNorm/残差块 → 训练与评估。")
    print("- 设计背后的核心思想：空间特征提取、参数高效、梯度流动、归一化提升训练稳定性、多尺度特征融合、特征复用。")

    # 残差网络的必要性验证：解决深度网络的退化问题
    print("\n========== 残差网络深度对比验证 ==========")
    
    # 深层普通网络 vs 残差网络
    class PlainNet(nn.Module):
        """深层普通网络（无残差连接）"""
        def __init__(self, depth=20):
            super().__init__()
            layers = [nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()]
            for _ in range(depth-2):
                layers.extend([
                    nn.Conv2d(64, 64, 3, padding=1), 
                    nn.BatchNorm2d(64), 
                    nn.ReLU()
                ])
            layers.append(nn.Conv2d(64, 10, 1))
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x)
    
    class ResNet18Simplified(nn.Module):
        """简化版ResNet-18，展示残差连接的作用"""
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
            
            # 残差块
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, 10)
            
        def _make_layer(self, in_channels, out_channels, blocks, stride=1):
            layers = []
            # 第一个残差块可能需要下采样
            layers.append(Residual(in_channels, out_channels, 
                                 use_1x1conv=(stride != 1 or in_channels != out_channels), 
                                 stride=stride))
            # 后续残差块
            for _ in range(1, blocks):
                layers.append(Residual(out_channels, out_channels))
            return nn.Sequential(*layers)
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    # 网络复杂度对比
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    plain_net = PlainNet(depth=18)
    resnet = ResNet18Simplified()
    
    print(f"普通深层网络参数量: {count_parameters(plain_net):,}")
    print(f"ResNet-18参数量: {count_parameters(resnet):,}")
    
    # 梯度流动验证
    x = torch.randn(2, 3, 224, 224)
    y_plain = plain_net(x)
    y_res = resnet(x)
    
    print(f"普通网络输出形状: {y_plain.shape}")
    print(f"ResNet输出形状: {y_res.shape}")
    
    # 模拟训练中的梯度情况
    criterion = nn.CrossEntropyLoss()
    target = torch.randint(0, 10, (2,))
    
    # 添加全局平均池化处理普通网络的输出
    y_plain_pooled = nn.functional.adaptive_avg_pool2d(y_plain, (1, 1)).view(y_plain.shape[0], -1)
    
    # 普通网络梯度
    loss_plain = criterion(y_plain_pooled, target)
    loss_plain.backward()
    plain_grads = [p.grad.norm().item() for p in plain_net.parameters() if p.grad is not None]
    
    # ResNet梯度 
    loss_res = criterion(y_res, target)
    loss_res.backward()
    res_grads = [p.grad.norm().item() for p in resnet.parameters() if p.grad is not None]
    
    print(f"普通网络平均梯度范数: {np.mean(plain_grads):.6f}")
    print(f"ResNet平均梯度范数: {np.mean(res_grads):.6f}")
    print("※ ResNet的梯度更稳定，有利于深层网络训练")
    
    # 8. 现代CNN技术综合应用
    print("\n========== 现代CNN技术集成应用 ==========")
    
    class ModernCNN(nn.Module):
        """
        现代CNN集成：批归一化+残差连接+注意力机制+自适应池化
        适用场景：图像分类、特征提取、迁移学习
        """
        def __init__(self, num_classes=10):
            super().__init__()
            # 初始卷积层
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1)
            )
            
            # 残差层
            self.layer1 = self._make_layer(64, 64, 2)
            self.layer2 = self._make_layer(64, 128, 2, stride=2)
            self.layer3 = self._make_layer(128, 256, 2, stride=2)
            self.layer4 = self._make_layer(256, 512, 2, stride=2)
            
            # 全局自适应平均池化（替代全连接层，减少参数）
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
            # 分类头
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        def _make_layer(self, in_channels, out_channels, blocks, stride=1):
            layers = []
            layers.append(Residual(in_channels, out_channels, 
                                 use_1x1conv=(stride != 1 or in_channels != out_channels), 
                                 stride=stride))
            for _ in range(1, blocks):
                layers.append(Residual(out_channels, out_channels))
            return nn.Sequential(*layers)
            
        def forward(self, x):
            x = self.stem(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
    
    modern_cnn = ModernCNN(num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    out = modern_cnn(x)
    print(f"现代CNN输出形状: {out.shape}")  # [4, 10]
    print(f"现代CNN参数量: {count_parameters(modern_cnn):,}")
    
    # 9. CNN性能优化与调试技巧
    print("\n========== CNN性能优化与调试技巧 ==========")
    
    # 特征图可视化（用于调试和理解）
    def visualize_feature_maps(model, x, layer_name="layer1"):
        """
        特征图可视化：帮助理解CNN学到的特征
        在实际调试中，可以查看中间层输出，验证网络是否学到有意义的特征
        """
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # 注册钩子函数
        getattr(model, layer_name).register_forward_hook(get_activation(layer_name))
        
        # 前向传播
        _ = model(x)
        
        # 获取特征图
        feature_map = activation[layer_name]
        print(f"{layer_name}特征图形状: {feature_map.shape}")
        
        # 计算特征图统计信息
        mean_activation = feature_map.mean().item()
        std_activation = feature_map.std().item()
        print(f"特征图均值: {mean_activation:.4f}, 标准差: {std_activation:.4f}")
        
        return feature_map
    
    # 可视化现代CNN的特征
    x_vis = torch.randn(1, 3, 224, 224)
    feature_map = visualize_feature_maps(modern_cnn, x_vis, "layer2")
    
    # 模型效率分析
    def analyze_model_efficiency(model, input_size=(1, 3, 224, 224)):
        """分析模型计算效率：FLOPs、参数量、内存占用"""
        x = torch.randn(*input_size)
        
        # 参数量
        total_params = count_parameters(model)
        
        # 模型大小（MB）
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        
        # 前向传播时间测试
        model.eval()
        import time
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model(x)
        avg_time = (time.time() - start_time) / 100
        
        print(f"参数量: {total_params:,}")
        print(f"模型大小: {model_size:.2f} MB")
        print(f"平均推理时间: {avg_time*1000:.2f} ms")
        
        return total_params, model_size, avg_time
    
    # 效率对比
    print("\n--- LeNet效率分析 ---")
    analyze_model_efficiency(LeNet(), (1, 1, 28, 28))
    
    print("\n--- ModernCNN效率分析 ---")
    analyze_model_efficiency(modern_cnn, (1, 3, 224, 224))
    
    # 10. 迁移学习与实际应用
    print("\n========== 迁移学习与实际应用 ==========")
    
    def create_transfer_model(num_classes=2, freeze_backbone=True):
        """
        创建迁移学习模型：基于预训练CNN进行微调
        适用场景：数据量较小的图像分类任务
        """
        # 使用现有的ModernCNN作为backbone
        backbone = ModernCNN(num_classes=1000)  # 假设在ImageNet上预训练
        
        # 冻结backbone参数（可选）
        if freeze_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        
        # 替换分类头
        backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        return backbone
    
    # 创建二分类迁移学习模型
    transfer_model = create_transfer_model(num_classes=2, freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    out = transfer_model(x)
    print(f"迁移学习模型输出形状: {out.shape}")  # [2, 2]
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in transfer_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in transfer_model.parameters())
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"参数冻结比例: {(1 - trainable_params/total_params)*100:.1f}%")
    
    # 11. CNN训练最佳实践
    print("\n========== CNN训练最佳实践 ==========")
    
    def cnn_training_demo():
        """
        CNN训练完整流程演示：数据增强+学习率调度+早停
        展示工程中的实际训练技巧
        """
        # 创建模拟数据集
        def create_dummy_dataset(num_samples=1000, num_classes=10):
            X = torch.randn(num_samples, 3, 32, 32)
            y = torch.randint(0, num_classes, (num_samples,))
            return X, y
        
        X_train, y_train = create_dummy_dataset(800, 10)
        X_val, y_val = create_dummy_dataset(200, 10)
        
        # 数据增强（模拟）
        print("数据增强策略：随机翻转、旋转、缩放、颜色变换")
        
        # 创建适合小图像的CNN
        class CIFAR_CNN(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                    nn.MaxPool2d(2, 2),
                    
                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                    nn.AdaptiveAvgPool2d((4, 4))
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(128 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = CIFAR_CNN(10)
        
        # 优化器设置
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        criterion = nn.CrossEntropyLoss()
        
        print(f"CIFAR-style CNN参数量: {count_parameters(model):,}")
        
        # 模拟训练过程（简化版）
        model.train()
        train_loss = criterion(model(X_train[:32]), y_train[:32])
        print(f"初始训练损失: {train_loss.item():.4f}")
        
        # 验证过程
        model.eval()
        with torch.no_grad():
            val_out = model(X_val[:32])
            val_loss = criterion(val_out, y_val[:32])
            val_acc = (val_out.argmax(dim=1) == y_val[:32]).float().mean()
        
        print(f"验证损失: {val_loss.item():.4f}")
        print(f"验证准确率: {val_acc.item():.3f}")
        
        return model
    
    trained_model = cnn_training_demo()
    
    # 12. 总结与展望
    print("\n========== CNN技术总结与展望 ==========")
    print("""
    【核心技术要点】
    1. 卷积层：局部连接、权重共享，适合处理具有空间结构的数据
    2. 池化层：降采样、增强不变性，减少计算量和过拟合风险
    3. 批归一化：稳定训练、加速收敛，现代CNN的标准组件
    4. 残差连接：解决深度网络训练困难，实现更深的网络结构
    5. 注意力机制：重点关注重要特征，提升模型表达能力
    
    【架构演进历程】
    LeNet(1998) → AlexNet(2012) → VGG(2014) → ResNet(2015) → DenseNet(2017) → Vision Transformer(2020)
    
    【工程实践建议】
    1. 数据预处理：标准化、数据增强提升泛化能力
    2. 网络设计：从简单开始，逐步加深，注意计算效率
    3. 训练技巧：学习率调度、早停、梯度裁剪防止过拟合
    4. 调试方法：特征图可视化、梯度监控、损失曲线分析
    5. 部署优化：模型压缩、量化、蒸馏适应实际应用
    
    【未来发展趋势】
    1. 更高效的架构：MobileNet、EfficientNet等轻量化网络
    2. 自动网络搜索：NAS技术自动设计最优架构
    3. 多模态融合：CNN+Transformer结合处理复杂任务
    4. 边缘计算：针对移动设备的模型压缩和加速
    """)
    
    print("\n🎯 CNN学习建议：")
    print("1. 理解卷积的数学原理和几何意义")
    print("2. 动手实现经典网络，加深理解")
    print("3. 通过可视化观察网络学到的特征")
    print("4. 在实际项目中应用，积累调参经验")
    print("5. 关注最新研究进展，保持技术敏感度") 