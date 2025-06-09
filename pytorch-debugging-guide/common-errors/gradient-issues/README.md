# 梯度相关问题 📈

> 深入理解 PyTorch 中的梯度机制，从根本上解决梯度爆炸、梯度消失和梯度异常问题

## 📋 问题清单

| 问题类型 | 文件 | 现象 | 关键词 |
|---------|------|------|--------|
| 梯度爆炸 | `gradient-explosion.py` | 损失快速增大，梯度范数极大 | `loss -> inf`, `grad norm >> 1` |
| 梯度消失 | `gradient-vanishing.py` | 损失不降，梯度接近0 | `grad norm << 1e-6` |
| 梯度为 None | `gradient-none.py` | 反向传播失败 | `grad is None` |
| 梯度不更新 | `gradient-not-updating.py` | 参数值不变 | `param unchanged` |

## 🔬 第一性原理分析

### 梯度的本质
```python
# 梯度 = 损失函数对参数的偏导数
∂L/∂θ = lim(h→0) [L(θ+h) - L(θ)] / h
```

### 反向传播链式法则
```python
# 复合函数求导
∂L/∂θ₁ = (∂L/∂z₂) × (∂z₂/∂z₁) × (∂z₁/∂θ₁)
```

当链式相乘的项过大或过小时，会导致梯度爆炸或消失。

## 🚨 常见错误模式

### 1. 数据规模问题
- **输入数据量级过大** → 激活值过大 → 梯度爆炸
- **目标值量级过大** → 损失值过大 → 梯度爆炸

### 2. 学习率问题
- **学习率过大** → 参数更新步长过大 → 梯度爆炸
- **学习率过小** → 收敛极慢，看似梯度消失

### 3. 网络架构问题
- **网络过深** → 梯度在传播中衰减 → 梯度消失
- **激活函数选择不当** → sigmoid 饱和区域 → 梯度消失

## 🛠️ 系统性解决方案

### 诊断工具
```python
def diagnose_gradients(model):
    """梯度诊断工具函数"""
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            grad_norms.append((name, grad_norm))
            print(f"{name}: {grad_norm:.6f}")
    return grad_norms
```

### 修复策略
1. **数据预处理**：标准化输入和目标
2. **梯度裁剪**：`torch.nn.utils.clip_grad_norm_()`
3. **学习率调整**：使用学习率调度器
4. **网络设计**：残差连接、批标准化

## 📚 参考资料

- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)
- [On the difficulty of training recurrent neural networks](http://proceedings.mlr.press/v28/pascanu13.html) 