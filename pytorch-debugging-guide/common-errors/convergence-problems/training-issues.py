"""
PyTorch 模型收敛问题分析
=======================

常见收敛问题：
1. 损失不下降 - 学习率、数据问题
2. 损失振荡 - 学习率过大
3. 过拟合 - 需要正则化
4. 欠拟合 - 模型容量不足
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 创建简单的回归数据
torch.manual_seed(42)
X = torch.randn(100, 10)
y = torch.sum(X[:, :3], dim=1, keepdim=True) + torch.randn(100, 1) * 0.1

class SimpleNet(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def demonstrate_convergence_problems():
    """演示常见收敛问题"""
    print("=== 模型收敛问题演示 ===\n")
    
    problems = {
        "学习率过大": 1.0,
        "学习率过小": 1e-6,
        "合适学习率": 0.01
    }
    
    results = {}
    
    for name, lr in problems.items():
        print(f"🔍 测试: {name} (lr={lr})")
        
        model = SimpleNet()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        losses = []
        
        for epoch in range(50):
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
        
        results[name] = losses
        print(f"  最终损失: {losses[-1]:.6f}\n")
    
    return results

def improved_training_demo():
    """改进的训练策略"""
    print("=== 改进的训练策略 ===\n")
    
    class ImprovedNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(10, 64),
                nn.BatchNorm1d(64),  # 批标准化
                nn.ReLU(),
                nn.Dropout(0.2),     # 防过拟合
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
        
        def forward(self, x):
            return self.net(x)
    
    model = ImprovedNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)  # 学习率衰减
    criterion = nn.MSELoss()
    
    losses = []
    
    print("🚀 使用改进策略:")
    print("  - BatchNorm + Dropout")
    print("  - Adam优化器")
    print("  - 学习率调度器\n")
    
    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        losses.append(loss.item())
        
        if epoch % 20 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:2d}: Loss = {loss.item():.6f}, LR = {lr:.6f}")
    
    print(f"最终损失: {losses[-1]:.6f}")
    return losses

if __name__ == "__main__":
    print("🎯 PyTorch 模型收敛问题分析")
    print("=" * 40)
    
    # 问题演示
    problem_results = demonstrate_convergence_problems()
    
    # 改进方案
    improved_losses = improved_training_demo()
    
    print("\n✅ 收敛问题解决要点:")
    print("1. 🎛️ 合适的学习率 (0.001-0.01)")
    print("2. 🏗️ 适当的模型复杂度")
    print("3. 🔀 数据标准化")
    print("4. 🎯 正则化技术")
    print("5. 📈 学习率调度")
    print("6. 🔍 梯度监控") 