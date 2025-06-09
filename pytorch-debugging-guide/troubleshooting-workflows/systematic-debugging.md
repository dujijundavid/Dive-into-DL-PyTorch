# PyTorch 系统化调试流程 🔧

> 结构化的调试方法论，帮你快速定位和解决PyTorch问题

## 🎯 调试方法论

### 第一性原理调试法
```
现象 → 机制 → 根因 → 解决方案 → 验证
```

1. **观察现象**：准确描述问题表现
2. **理解机制**：分析涉及的PyTorch机制
3. **定位根因**：找到问题的本质原因
4. **设计方案**：基于根因设计解决方案
5. **验证效果**：确认方案有效性

## 📋 标准调试Checklist

### 🔍 初步诊断 (5分钟)

- [ ] 复制完整的错误堆栈信息
- [ ] 确认 PyTorch/CUDA/Python 版本
- [ ] 检查最近的代码变更
- [ ] 尝试最小化复现代码
- [ ] 搜索错误关键词

### 🎛️ 环境检查 (2分钟)

```python
# 运行环境诊断脚本
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
```

### 📊 数据验证 (10分钟)

- [ ] 检查输入数据形状和类型
- [ ] 验证数据范围和分布
- [ ] 确认标签格式正确性
- [ ] 测试数据加载流程
- [ ] 检查数据预处理逻辑

### 🏗️ 模型架构检查 (15分钟)

- [ ] 验证层之间的维度匹配
- [ ] 检查激活函数选择
- [ ] 确认损失函数适配性
- [ ] 验证优化器配置
- [ ] 检查设备一致性 (CPU/GPU)

## 🚨 错误类型决策树

```
PyTorch错误
├── 有报错信息
│   ├── RuntimeError → 检查张量操作兼容性
│   ├── CUDA错误 → 检查GPU环境配置
│   ├── size/shape → 分析张量形状变换
│   └── grad相关 → 诊断梯度流
└── 无报错信息
    └── 性能/收敛问题 → 分析训练曲线
```

## 🛠️ 分类调试流程

### 📐 维度错误调试流程

1. **打印所有相关张量形状**
2. **检查操作兼容性**
3. **逐步调试网络形状变化**
4. **验证修复方案**

### 📈 梯度问题调试流程

1. **检查梯度是否存在**
2. **分析梯度数值范围**
3. **诊断异常情况**
4. **应用修复策略**

### 🎯 收敛问题调试流程

1. **损失趋势分析**
2. **收敛速度评估**
3. **超参数调优**
4. **架构优化**

## 🤖 LLM交互优化流程

### 准备阶段
1. **收集完整信息**
   - 错误堆栈 (完整)
   - 相关代码 (最小化)
   - 环境信息 (版本)
   - 数据信息 (形状、类型)

2. **使用结构化模板**
   - 选择对应的debug-templates/中的模板
   - 填入具体信息

### 交互阶段
1. **分步骤提问**
   - 先问诊断 → 再问解决方案
   - 从简单到复杂
   - 逐步深入原理

2. **提供具体上下文**
   - 粘贴实际的数值
   - 说明尝试过的方法
   - 明确期望的解释深度

## 📚 调试工具包

### 通用诊断函数

```python
def debug_model_info(model):
    """打印模型调试信息"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_params:,}")
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            print(f"{name}: grad_norm={grad_norm:.6f}")
```

### 快速修复建议

```python
def quick_fix_suggestions(error_message):
    """基于错误信息的快速修复建议"""
    if "CUDA out of memory" in error_message:
        print("💡 建议: 减小batch_size或清理GPU缓存")
    elif "size mismatch" in error_message:
        print("💡 建议: 检查张量形状和矩阵乘法维度")
```

## 🎓 最佳实践

### 预防性调试
1. 使用类型注解和assertion检查
2. 从简单模型开始逐步增加复杂度
3. 及时打印中间结果形状

### 响应式调试
1. 保持系统化分析思路
2. 记录完整调试过程
3. 验证修复的根本性

---
*系统化的调试能力是深度学习工程师的核心竞争力* 🚀 