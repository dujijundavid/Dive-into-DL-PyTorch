"""
nn.Sequential（顺序容器）核心原理与用法
------------------------------------
第一性原理思考：
1. 什么是顺序容器？
   - 顺序容器是将多个神经网络层按顺序组合的工具
   - 数据按照定义的顺序依次通过每个层
   - 简化了复杂网络的构建和管理

2. 为什么需要顺序容器？
   - 简化网络定义：避免手动调用每个层
   - 提高代码可读性：网络结构一目了然
   - 便于参数管理：统一管理所有层的参数

3. 顺序容器的核心特性是什么？
   - 自动前向传播：数据自动在层间传递
   - 参数统一管理：可以获取所有子模块的参数
   - 模块化设计：可以轻松添加、删除、修改层

苏格拉底式提问与验证：
1. 顺序容器如何简化网络构建？
   - 问题：与手动调用层相比有什么优势？
   - 验证：比较Sequential和手动调用的代码复杂度
   - 结论：Sequential使代码更简洁、可读性更强

2. 如何在Sequential中添加非标准操作？
   - 问题：如何在Sequential中使用自定义函数？
   - 验证：尝试添加激活函数、归一化等操作
   - 结论：可以使用Lambda层或自定义模块

3. Sequential的参数管理机制是什么？
   - 问题：如何访问和修改Sequential中的参数？
   - 验证：查看参数、梯度和优化器的使用
   - 结论：Sequential提供统一的参数接口

费曼学习法讲解：
1. 概念解释
   - 用管道类比解释Sequential的工作原理
   - 通过层叠结构理解数据流向
   - 强调Sequential在网络设计中的重要性

2. 实例教学
   - 从简单到复杂的Sequential构建
   - 通过可视化理解网络结构
   - 实践常见网络架构构建

3. 知识巩固
   - 总结Sequential的优势和局限
   - 提供网络设计的最佳实践
   - 建议进阶学习方向

功能说明：
- nn.Sequential 是一个有序的容器，用于按照顺序组织神经网络层。

原理讲解：
- 将多个nn.Module按顺序排列，前一层的输出自动作为后一层的输入。
- 支持按索引访问、添加、删除层。
- 所有参数自动注册到容器中，支持统一的参数管理。

工程意义：
- 简化深度网络的构建，提高代码可读性和维护性。
- 是构建复杂网络架构的基础工具。

可运行案例：
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

# 1. 基本Sequential构建
print("1. 基本Sequential构建：")
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
print("模型结构：")
print(model)

# 测试前向传播
x = torch.randn(5, 10)
y = model(x)
print("输入形状:", x.shape)
print("输出形状:", y.shape)

# 2. 使用OrderedDict构建
print("\n2. 使用OrderedDict构建：")
from collections import OrderedDict
model = nn.Sequential(OrderedDict([
    ('linear1', nn.Linear(10, 20)),
    ('relu1', nn.ReLU()),
    ('linear2', nn.Linear(20, 10)),
    ('relu2', nn.ReLU()),
    ('output', nn.Linear(10, 1))
]))
print("命名模型结构：")
print(model)

# 3. 添加和删除层
print("\n3. 添加和删除层：")
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU()
)
print("初始模型：")
print(model)

# 添加层
model.add_module('linear2', nn.Linear(20, 10))
model.add_module('output', nn.Linear(10, 1))
print("添加层后：")
print(model)

# 删除最后一层
del model[-1]
print("删除最后一层后：")
print(model)

# 4. 访问和修改层
print("\n4. 访问和修改层：")
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 访问特定层
print("第一层:", model[0])
print("第一层权重形状:", model[0].weight.shape)

# 修改层的参数
with torch.no_grad():
    model[0].weight.fill_(0.1)
    model[0].bias.fill_(0.0)
print("修改后的第一层权重:", model[0].weight[0])

# 5. 参数管理
print("\n5. 参数管理：")
print("总参数数量:", sum(p.numel() for p in model.parameters()))
print("可训练参数数量:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# 查看所有参数
for name, param in model.named_parameters():
    print(f"参数名: {name}, 形状: {param.shape}")

# 6. 与手动调用的对比
print("\n6. 与手动调用的对比：")

# Sequential方式
class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# 手动调用方式
class ManualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(20, 10)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(10, 1)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

seq_model = SequentialModel()
manual_model = ManualModel()

# 比较代码复杂度和可读性
print("Sequential模型参数数:", sum(p.numel() for p in seq_model.parameters()))
print("Manual模型参数数:", sum(p.numel() for p in manual_model.parameters()))

# 7. 复杂网络构建示例
print("\n7. 复杂网络构建示例：")

# CNN示例
cnn = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * 8 * 8, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)

print("CNN模型：")
print(cnn)

# 测试CNN
x = torch.randn(2, 3, 32, 32)
y = cnn(x)
print("CNN输入形状:", x.shape)
print("CNN输出形状:", y.shape)

# 8. 使用Lambda层添加自定义操作
print("\n8. 使用Lambda层添加自定义操作：")

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.Lambda(lambda x: x * 2),  # 自定义操作
    nn.ReLU(),
    nn.Linear(20, 1)
)

x = torch.randn(3, 10)
y = model(x)
print("带Lambda层的输出形状:", y.shape)

# 9. 实际训练示例
print("\n9. 实际训练示例：")

# 创建一个简单的分类模型
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 2)
)

# 生成模拟数据
X = torch.randn(100, 2)
y = torch.randint(0, 2, (100,))

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 10. Sequential的局限性和解决方案
print("\n10. Sequential的局限性和解决方案：")

# 局限性：无法处理复杂的数据流（如残差连接）
# 解决方案：使用自定义Module

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        return x + self.layers(x)  # 残差连接

# 可以在Sequential中使用自定义模块
model = nn.Sequential(
    nn.Linear(10, 20),
    ResidualBlock(20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

print("包含自定义模块的Sequential:")
print(model)

x = torch.randn(3, 10)
y = model(x)
print("输出形状:", y.shape) 