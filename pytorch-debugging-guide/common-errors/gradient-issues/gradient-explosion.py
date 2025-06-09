"""
梯度爆炸问题分析与解决方案
=========================

问题现象：
- 损失值快速增长到 inf
- 梯度范数急剧增大
- 模型参数更新过于剧烈

第一性原理分析：
- 高学习率 × 大数据量级 × 大目标值 = 梯度爆炸
- 链式法则：∂L/∂θ = (∂L/∂output) × (∂output/∂θ)
- 当 ∂L/∂output 很大时，整个梯度会被放大
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List, Tuple

# 设置随机种子确保结果可复现
torch.manual_seed(0)

# ================== 有问题的代码 ==================

class SimpleMLP(nn.Module):
    """简单的多层感知机"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.fc(x)

def problematic_training():
    """展示梯度爆炸的问题代码"""
    print("=== 梯度爆炸示例 ===")
    
    model = SimpleMLP()
    optimizer = optim.SGD(model.parameters(), lr=1.0)  # 🚨 学习率过大
    criterion = nn.MSELoss()

    # 🚨 输入和目标值量级过大
    x = torch.randn(64, 100) * 100      # 输入放大100倍
    target = torch.randn(64, 1) * 1000  # 目标放大1000倍

    losses = []
    grad_norms = []
    
    for i in range(20):  # 减少迭代次数避免溢出
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        
        # 计算梯度范数
        total_grad_norm = sum(p.grad.data.norm(2).item() for p in model.parameters())
        
        print(f"Step {i:2d}, Loss: {loss.item():10.4f}, Grad Norm: {total_grad_norm:8.2f}")
        
        losses.append(loss.item())
        grad_norms.append(total_grad_norm)
        
        # 检查是否已经爆炸
        if loss.item() > 1e6 or total_grad_norm > 1e6:
            print("⚠️  梯度爆炸检测到！停止训练")
            break
            
        optimizer.step()
    
    return losses, grad_norms

# ================== 修复方案 ==================

def diagnose_gradients(model):
    """诊断每层的梯度情况"""
    print("\n=== 梯度诊断 ===")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            param_norm = param.data.norm(2).item()
            print(f"{name:20s}: grad_norm={grad_norm:8.4f}, param_norm={param_norm:8.4f}")

def fixed_training():
    """修复后的训练代码"""
    print("\n=== 修复后的训练 ===")
    
    model = SimpleMLP()
    
    # 修复1: 降低学习率
    optimizer = optim.SGD(model.parameters(), lr=0.001)  # ✅ 合适的学习率
    criterion = nn.MSELoss()

    # 修复2: 数据标准化
    x = torch.randn(64, 100)      # ✅ 标准化输入
    target = torch.randn(64, 1)   # ✅ 标准化目标
    
    # 修复3: 梯度裁剪
    max_grad_norm = 1.0
    
    losses = []
    grad_norms = []
    
    for i in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        
        # 修复3: 应用梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        total_grad_norm = sum(p.grad.data.norm(2).item() for p in model.parameters())
        
        if i % 10 == 0:
            print(f"Step {i:2d}, Loss: {loss.item():8.4f}, Grad Norm: {total_grad_norm:6.4f}")
        
        losses.append(loss.item())
        grad_norms.append(total_grad_norm)
        
        optimizer.step()
    
    return losses, grad_norms

def advanced_fixed_training():
    """高级修复方案：自适应学习率 + 批标准化"""
    print("\n=== 高级修复方案 ===")
    
    # 修复4: 改进网络架构
    class ImprovedMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(100, 256),
                nn.BatchNorm1d(256),  # ✅ 批标准化
                nn.ReLU(),
                nn.Dropout(0.2),      # ✅ 正则化
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1)
            )
        
        def forward(self, x):
            return self.fc(x)
    
    model = ImprovedMLP()
    
    # 修复5: 使用自适应优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # ✅ Adam优化器
    criterion = nn.MSELoss()
    
    # 修复6: 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # 原始数据（故意保持大量级测试修复效果）
    x = torch.randn(64, 100) * 10   # 适中的放大
    target = torch.randn(64, 1) * 10
    
    losses = []
    grad_norms = []
    
    for i in range(100):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        total_grad_norm = sum(p.grad.data.norm(2).item() for p in model.parameters())
        
        if i % 10 == 0:
            print(f"Step {i:2d}, Loss: {loss.item():8.4f}, Grad Norm: {total_grad_norm:6.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        losses.append(loss.item())
        grad_norms.append(total_grad_norm)
        
        optimizer.step()
        scheduler.step(loss)  # 根据损失调整学习率
    
    return losses, grad_norms

def plot_comparison(problem_losses, problem_grads, fixed_losses, fixed_grads, advanced_losses, advanced_grads):
    """可视化比较结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失对比
    ax1.plot(problem_losses[:len(problem_losses)], 'r-', label='问题代码', linewidth=2)
    ax1.plot(fixed_losses, 'g-', label='基础修复', linewidth=2)
    ax1.plot(advanced_losses, 'b-', label='高级修复', linewidth=2)
    ax1.set_xlabel('训练步数')
    ax1.set_ylabel('损失值')
    ax1.set_title('损失值对比')
    ax1.legend()
    ax1.set_yscale('log')
    
    # 梯度范数对比
    ax2.plot(problem_grads[:len(problem_grads)], 'r-', label='问题代码', linewidth=2)
    ax2.plot(fixed_grads, 'g-', label='基础修复', linewidth=2)
    ax2.plot(advanced_grads, 'b-', label='高级修复', linewidth=2)
    ax2.set_xlabel('训练步数')
    ax2.set_ylabel('梯度范数')
    ax2.set_title('梯度范数对比')
    ax2.legend()
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('gradient_explosion_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数：演示梯度爆炸问题和解决方案"""
    print("🔬 PyTorch 梯度爆炸分析与解决方案")
    print("=" * 50)
    
    # 1. 展示问题
    problem_losses, problem_grads = problematic_training()
    
    # 2. 基础修复
    fixed_losses, fixed_grads = fixed_training()
    
    # 3. 高级修复
    advanced_losses, advanced_grads = advanced_fixed_training()
    
    # 4. 可视化对比
    plot_comparison(problem_losses, problem_grads, fixed_losses, fixed_grads, advanced_losses, advanced_grads)
    
    print("\n✅ 总结：")
    print("1. 问题根源：高学习率 + 大数据量级 + 不合适的网络架构")
    print("2. 基础解决：降低学习率 + 数据标准化 + 梯度裁剪")
    print("3. 高级解决：改进架构 + 自适应优化器 + 学习率调度")
    print("4. 关键监控：梯度范数、损失值变化趋势")

if __name__ == "__main__":
    main() 