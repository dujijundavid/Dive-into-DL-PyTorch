"""
torch.optim（优化器）核心原理与用法
---------------------------------
第一性原理思考：
1. 什么是优化器？
   - 优化器是用于更新神经网络参数的算法
   - 通过最小化损失函数来改进模型性能
   - 是梯度下降算法的具体实现

2. 为什么需要优化器？
   - 自动化参数更新过程：避免手动计算更新步骤
   - 实现各种优化算法：SGD, Adam, RMSprop等
   - 管理学习率和动量：提供学习率调度功能

3. 优化器的核心特性是什么？
   - 参数组管理：可以为不同参数设置不同的学习率
   - 状态管理：保存优化过程中的中间状态
   - 学习率调度：支持动态调整学习率

苏格拉底式提问与验证：
1. 不同优化器有什么区别？
   - 问题：SGD和Adam的优化行为有何不同？
   - 验证：比较不同优化器的收敛速度和稳定性
   - 结论：每种优化器适用于不同的场景

2. 学习率如何影响训练？
   - 问题：学习率过大或过小会怎样？
   - 验证：观察不同学习率下的训练曲线
   - 结论：学习率需要仔细调节

3. 为什么需要学习率调度？
   - 问题：固定学习率的局限性是什么？
   - 验证：比较固定和动态学习率的效果
   - 结论：动态调整学习率可以提高训练效果

费曼学习法讲解：
1. 概念解释
   - 用山谷和球的类比解释优化过程
   - 通过几何直观理解不同优化算法
   - 强调优化器在深度学习中的核心地位

2. 实例教学
   - 从简单到复杂的优化场景
   - 通过可视化理解优化轨迹
   - 实践不同优化器的使用方法

3. 知识巩固
   - 总结各种优化器的特点和适用场景
   - 提供超参数调节的最佳实践
   - 建议进阶学习方向

功能说明：
- torch.optim 提供各种优化算法，用于更新模型参数以最小化损失函数。

原理讲解：
- 基于计算得到的梯度，使用不同的更新规则调整参数。
- 支持SGD、Adam、RMSprop等多种优化算法。
- 提供学习率调度、权重衰减、动量等功能。

工程意义：
- 是深度学习训练过程的核心组件，直接影响模型收敛速度和最终性能。

可运行案例：
"""
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

# 1. 基本优化器使用
print("1. 基本优化器使用：")

# 创建简单模型
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)

# 创建优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 模拟训练步骤
x = torch.randn(10, 2)
y = torch.randn(10, 1)
criterion = nn.MSELoss()

for epoch in range(5):
    # 前向传播
    pred = model(x)
    loss = criterion(pred, y)
    
    # 反向传播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 2. 不同优化器比较
print("\n2. 不同优化器比较：")

def create_model():
    return nn.Sequential(
        nn.Linear(2, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

def train_model(model, optimizer, epochs=100):
    x = torch.randn(100, 2)
    y = x[:, 0:1] * 2 + x[:, 1:2] * 3 + torch.randn(100, 1) * 0.1  # 线性关系
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        pred = model(x)
        loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return losses

# 测试不同优化器
optimizers = {
    'SGD': optim.SGD,
    'Adam': optim.Adam,
    'RMSprop': optim.RMSprop,
    'AdaGrad': optim.Adagrad
}

results = {}
for name, opt_class in optimizers.items():
    model = create_model()
    optimizer = opt_class(model.parameters(), lr=0.01)
    losses = train_model(model, optimizer)
    results[name] = losses
    print(f"{name} 最终损失: {losses[-1]:.6f}")

# 3. 学习率影响
print("\n3. 学习率影响：")

learning_rates = [0.001, 0.01, 0.1, 1.0]
lr_results = {}

for lr in learning_rates:
    model = create_model()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = train_model(model, optimizer, epochs=50)
    lr_results[f'lr_{lr}'] = losses
    print(f"学习率 {lr}: 最终损失 {losses[-1]:.6f}")

# 4. 参数组（不同参数不同学习率）
print("\n4. 参数组（不同参数不同学习率）：")

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Linear(10, 20)
        self.classifier = nn.Linear(20, 2)
    
    def forward(self, x):
        x = torch.relu(self.feature_extractor(x))
        x = self.classifier(x)
        return x

model = CustomModel()

# 为不同部分设置不同学习率
optimizer = optim.SGD([
    {'params': model.feature_extractor.parameters(), 'lr': 0.001},
    {'params': model.classifier.parameters(), 'lr': 0.01}
], lr=0.01)  # 默认学习率

print("参数组信息:")
for i, group in enumerate(optimizer.param_groups):
    print(f"组 {i}: 学习率 {group['lr']}, 参数数量 {len(group['params'])}")

# 5. 学习率调度器
print("\n5. 学习率调度器：")

model = create_model()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 不同类型的学习率调度器
schedulers = {
    'StepLR': StepLR(optimizer, step_size=10, gamma=0.9),
    'ExponentialLR': ExponentialLR(optimizer, gamma=0.95),
    'CosineAnnealingLR': CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)
}

# 记录学习率变化
for name, scheduler in schedulers.items():
    lrs = []
    temp_optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    if name == 'StepLR':
        temp_scheduler = StepLR(temp_optimizer, step_size=10, gamma=0.9)
    elif name == 'ExponentialLR':
        temp_scheduler = ExponentialLR(temp_optimizer, gamma=0.95)
    else:
        temp_scheduler = CosineAnnealingLR(temp_optimizer, T_max=20, eta_min=0.001)
    
    for epoch in range(50):
        lrs.append(temp_optimizer.param_groups[0]['lr'])
        temp_scheduler.step()
    
    print(f"{name}: 初始lr={lrs[0]:.4f}, 最终lr={lrs[-1]:.4f}")

# 6. 权重衰减（L2正则化）
print("\n6. 权重衰减（L2正则化）：")

# 不带权重衰减
model1 = create_model()
optimizer1 = optim.Adam(model1.parameters(), lr=0.01)

# 带权重衰减
model2 = create_model()
optimizer2 = optim.Adam(model2.parameters(), lr=0.01, weight_decay=0.01)

def get_weight_norm(model):
    total_norm = 0
    for param in model.parameters():
        total_norm += param.norm().item() ** 2
    return total_norm ** 0.5

print("训练前权重范数:")
print(f"无权重衰减: {get_weight_norm(model1):.4f}")
print(f"有权重衰减: {get_weight_norm(model2):.4f}")

# 简单训练
x = torch.randn(50, 2)
y = torch.randn(50, 1)
criterion = nn.MSELoss()

for _ in range(100):
    # 模型1
    pred1 = model1(x)
    loss1 = criterion(pred1, y)
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()
    
    # 模型2
    pred2 = model2(x)
    loss2 = criterion(pred2, y)
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()

print("训练后权重范数:")
print(f"无权重衰减: {get_weight_norm(model1):.4f}")
print(f"有权重衰减: {get_weight_norm(model2):.4f}")

# 7. 梯度裁剪
print("\n7. 梯度裁剪：")

model = nn.Sequential(
    nn.Linear(10, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

optimizer = optim.SGD(model.parameters(), lr=0.01)

# 模拟梯度爆炸情况
x = torch.randn(32, 10) * 10  # 大输入值
y = torch.randn(32, 1)
criterion = nn.MSELoss()

print("训练前后的梯度范数:")
for epoch in range(5):
    pred = model(x)
    loss = criterion(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    # 计算梯度范数
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    
    print(f"Epoch {epoch}: 梯度范数 {total_norm:.4f}")
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 裁剪后的梯度范数
    clipped_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            clipped_norm += param.grad.norm().item() ** 2
    clipped_norm = clipped_norm ** 0.5
    print(f"       裁剪后梯度范数 {clipped_norm:.4f}")
    
    optimizer.step()

# 8. 自定义优化器
print("\n8. 自定义优化器：")

class SimpleOptimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr
    
    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()
    
    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    param -= self.lr * param.grad

# 测试自定义优化器
model = nn.Linear(2, 1)
custom_optimizer = SimpleOptimizer(model.parameters(), lr=0.01)

x = torch.randn(10, 2)
y = torch.randn(10, 1)
criterion = nn.MSELoss()

print("使用自定义优化器训练:")
for epoch in range(5):
    pred = model(x)
    loss = criterion(pred, y)
    
    custom_optimizer.zero_grad()
    loss.backward()
    custom_optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 9. 优化器状态保存和加载
print("\n9. 优化器状态保存和加载：")

model = create_model()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练几步
x = torch.randn(20, 2)
y = torch.randn(20, 1)
criterion = nn.MSELoss()

for _ in range(10):
    pred = model(x)
    loss = criterion(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("训练10步后的损失:", loss.item())

# 保存状态
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()
}, 'checkpoint.pth')

# 创建新模型和优化器
new_model = create_model()
new_optimizer = optim.Adam(new_model.parameters(), lr=0.01)

# 加载状态
checkpoint = torch.load('checkpoint.pth')
new_model.load_state_dict(checkpoint['model_state_dict'])
new_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

print("加载检查点后，优化器状态已恢复")

# 10. 实际训练循环示例
print("\n10. 实际训练循环示例：")

def train_epoch(model, optimizer, data_loader, criterion):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # 梯度裁剪（可选）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def validate_epoch(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)

# 模拟数据加载器
class SimpleDataLoader:
    def __init__(self, x, y, batch_size=32):
        self.x = x
        self.y = y
        self.batch_size = batch_size
    
    def __iter__(self):
        indices = torch.randperm(len(self.x))
        for i in range(0, len(self.x), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            yield self.x[batch_indices], self.y[batch_indices]
    
    def __len__(self):
        return (len(self.x) + self.batch_size - 1) // self.batch_size

# 创建数据
x_train = torch.randn(1000, 2)
y_train = torch.randn(1000, 1)
x_val = torch.randn(200, 2)
y_val = torch.randn(200, 1)

train_loader = SimpleDataLoader(x_train, y_train)
val_loader = SimpleDataLoader(x_val, y_val)

# 创建模型和优化器
model = create_model()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
criterion = nn.MSELoss()

print("完整训练循环:")
for epoch in range(10):
    train_loss = train_epoch(model, optimizer, train_loader, criterion)
    val_loss = validate_epoch(model, val_loader, criterion)
    scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

# 清理临时文件
import os
if os.path.exists('checkpoint.pth'):
    os.remove('checkpoint.pth') 