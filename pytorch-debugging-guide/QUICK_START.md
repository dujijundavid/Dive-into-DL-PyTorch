# PyTorch 调试指南 - 快速开始 ⚡

> 3分钟快速上手，解决90%的常见PyTorch问题

## 🚀 紧急求助

### 遇到错误？立即查看：

| 错误信息关键词 | 快速定位 | 文件路径 |
|-------------|---------|---------|
| `RuntimeError: mat1 and mat2 shapes` | 矩阵乘法维度问题 | `common-errors/dimension-mismatch/` |
| `CUDA out of memory` | GPU内存不足 | `common-errors/memory-issues/` |
| `loss -> inf` 或 `NaN` | 梯度爆炸 | `common-errors/gradient-issues/gradient-explosion.py` |
| `loss not decreasing` | 收敛问题 | `common-errors/convergence-problems/` |
| `Expected input batch_size` | 批量大小不匹配 | `common-errors/dimension-mismatch/` |

## 💬 与LLM对话模板

### 🔧 通用错误分析（复制使用）

```
我在运行以下 PyTorch 代码时出现了错误，请帮我解释这个错误并给出修复建议：

**错误堆栈信息：**
```
[在这里粘贴完整的错误信息]
```

**问题代码：**
```python
[在这里粘贴出问题的代码片段]
```

**运行环境：**
- PyTorch 版本：
- Python 版本：
- CUDA 版本（如使用）：

**请帮我：**
1. 判断这是 API 用法错误、数据维度问题，还是梯度相关问题
2. 如果可能，给我一个修改后的可运行版本
3. 给我一个解释，使我能从原理理解这个错误

你是一个深入理解 PyTorch 架构的调试专家，解释要具体、分步骤。
```

### 🎯 专门问题模板

**梯度问题：** 使用 `debug-templates/error-analysis-template.md` 中的"梯度问题专用模板"

**维度错误：** 使用 `debug-templates/error-analysis-template.md` 中的"维度错误专用模板"

**性能问题：** 使用 `debug-templates/error-analysis-template.md` 中的"性能问题专用模板"

## 🛠️ 5分钟调试检查清单

### Step 1: 基础检查 (30秒)
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

### Step 2: 数据检查 (1分钟)
```python
# 检查输入数据
print(f"输入形状: {your_input.shape}")
print(f"输入类型: {your_input.dtype}")
print(f"输入设备: {your_input.device}")

# 检查目标数据
print(f"目标形状: {your_target.shape}")
print(f"目标类型: {your_target.dtype}")
```

### Step 3: 模型检查 (1分钟)
```python
# 检查模型参数
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数总数: {total_params:,}")

# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.6f}")
```

### Step 4: 运行小测试 (2分钟)
```python
# 使用小批量测试
small_input = your_input[:2]  # 只取2个样本
try:
    output = model(small_input)
    print(f"小测试成功！输出形状: {output.shape}")
except Exception as e:
    print(f"小测试失败: {e}")
```

## 📋 最常见的5个问题及快速修复

### 1. 梯度爆炸
**现象：** 损失突然变成 `inf` 或 `NaN`
**快速修复：**
```python
# 添加梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2. 维度不匹配
**现象：** `RuntimeError: mat1 and mat2 shapes cannot be multiplied`
**快速修复：**
```python
# 检查并调整维度
print(f"张量A形状: {tensor_a.shape}")
print(f"张量B形状: {tensor_b.shape}")
# 使用 .view() 或 .reshape() 调整
```

### 3. CUDA内存不足
**现象：** `CUDA out of memory`
**快速修复：**
```python
# 减小批量大小
batch_size = batch_size // 2

# 清理GPU缓存
torch.cuda.empty_cache()
```

### 4. 学习率问题
**现象：** 损失不下降或振荡
**快速修复：**
```python
# 降低学习率
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 从0.01降到0.001
```

### 5. 设备不匹配
**现象：** `Expected all tensors to be on the same device`
**快速修复：**
```python
# 确保所有张量在同一设备
model = model.to(device)
input_tensor = input_tensor.to(device)
target_tensor = target_tensor.to(device)
```

## 🎯 运行示例代码

### 测试梯度爆炸问题
```bash
cd pytorch-debugging-guide/common-errors/gradient-issues/
python gradient-explosion.py
```

### 测试维度问题
```bash
cd pytorch-debugging-guide/common-errors/dimension-mismatch/
python tensor-shapes.py
```

### 测试收敛问题
```bash
cd pytorch-debugging-guide/common-errors/convergence-problems/
python training-issues.py
```

## 🤖 与Cursor Copilot互动技巧

1. **选中问题代码** 再提问，让Copilot有上下文
2. **使用具体描述**："我的CNN第3层输出维度不对" 比 "维度有问题" 更好
3. **分步提问**：先问诊断，再问修复，最后问原理
4. **提供完整信息**：错误堆栈 + 相关代码 + 环境信息

## 📞 需要更多帮助？

- 📁 查看完整分类：`common-errors/` 目录
- 📝 使用模板：`debug-templates/` 目录  
- 🔄 系统化流程：`troubleshooting-workflows/` 目录
- 📖 阅读主文档：`README.md`

---

**记住：** 90%的PyTorch问题都是维度不匹配、梯度问题或环境配置。先检查这三个方面！ ��

*快速解决问题，深入理解原理* ⚡ 