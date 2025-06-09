"""
nn.Module（自定义模块）核心原理与用法
----------------------------------
第一性原理思考：
1. 什么是自定义模块？
   - nn.Module是所有神经网络模块的基类
   - 提供了参数管理、前向传播、状态管理等核心功能
   - 允许用户定义复杂的网络结构和计算逻辑

2. 为什么需要自定义模块？
   - 实现复杂的网络架构：如ResNet、Transformer等
   - 封装可复用的计算单元：提高代码复用性
   - 精确控制前向传播过程：实现特殊的计算逻辑

3. 自定义模块的核心特性是什么？
   - 参数自动注册：子模块的参数自动管理
   - 训练/评估模式：支持不同模式下的行为
   - 设备管理：支持CPU/GPU之间的转换

苏格拉底式提问与验证：
1. 如何正确实现自定义模块？
   - 问题：继承nn.Module需要实现哪些方法？
   - 验证：比较正确和错误的实现方式
   - 结论：必须调用super().__init__()和实现forward方法

2. 参数是如何自动注册的？
   - 问题：为什么子模块的参数会自动被发现？
   - 验证：观察参数注册的过程和机制
   - 结论：nn.Module通过__setattr__自动注册参数

3. 训练和评估模式有什么区别？
   - 问题：train()和eval()模式影响哪些操作？
   - 验证：比较不同模式下Dropout、BatchNorm的行为
   - 结论：某些层在不同模式下行为不同

费曼学习法讲解：
1. 概念解释
   - 用面向对象编程类比解释nn.Module
   - 通过继承关系理解模块层次结构
   - 强调自定义模块在深度学习中的重要性

2. 实例教学
   - 从简单到复杂的自定义模块
   - 通过实际案例理解设计原则
   - 实践常见网络组件的实现

3. 知识巩固
   - 总结自定义模块的设计模式
   - 提供代码组织的最佳实践
   - 建议进阶学习方向

功能说明：
- nn.Module 是所有神经网络模块的基类，提供参数管理、前向传播等核心功能。

原理讲解：
- 通过继承nn.Module，自动获得参数注册、梯度计算、设备管理等功能。
- 必须实现forward方法定义前向传播逻辑。
- 支持嵌套模块，形成层次化的网络结构。

工程意义：
- 是构建复杂网络架构的基础，提供了模块化、可复用的设计方式。

可运行案例：
"""
import torch
from torch import nn
import torch.nn.functional as F
import math

# 1. 最简单的自定义模块
print("1. 最简单的自定义模块：")

class SimpleLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()  # 必须调用父类初始化
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

# 测试简单模块
simple_model = SimpleLinear(10, 5)
x = torch.randn(3, 10)
y = simple_model(x)
print("输入形状:", x.shape)
print("输出形状:", y.shape)
print("模型参数数量:", sum(p.numel() for p in simple_model.parameters()))

# 2. 复杂的自定义模块 - MLP
print("\n2. 复杂的自定义模块 - MLP：")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建层列表
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

# 测试MLP
mlp = MLP(10, [64, 32, 16], 2)
print("MLP结构:")
print(mlp)
print("参数数量:", mlp.get_num_params())

# 3. 残差块实现
print("\n3. 残差块实现：")

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Pre-norm residual connection
        residual = x
        x = self.norm1(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.norm2(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x + residual  # 残差连接

# 测试残差块
res_block = ResidualBlock(64)
x = torch.randn(5, 64)
y = res_block(x)
print("残差块输入形状:", x.shape)
print("残差块输出形状:", y.shape)

# 4. 注意力机制实现
print("\n4. 注意力机制实现：")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.W_o(context)
        return output, attention_weights

# 测试注意力机制
attention = MultiHeadAttention(512, 8)
x = torch.randn(2, 10, 512)  # (batch, seq_len, d_model)
output, weights = attention(x, x, x)
print("注意力输入形状:", x.shape)
print("注意力输出形状:", output.shape)
print("注意力权重形状:", weights.shape)

# 5. 参数初始化和管理
print("\n5. 参数初始化和管理：")

class CustomModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        
        # 自定义初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
    def get_parameter_info(self):
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
            print(f"{name:20s} | Shape: {str(param.shape):15s} | Trainable: {param.requires_grad}")
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

# 测试参数管理
model = CustomModel(10, 64, 2)
model.get_parameter_info()

# 6. 训练和评估模式
print("\n6. 训练和评估模式：")

class DropoutBatchNormModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

# 测试不同模式
model = DropoutBatchNormModel(10, 64, 2)
x = torch.randn(32, 10)

# 训练模式
model.train()
print("训练模式:")
print("BatchNorm running_mean:", model.batch_norm.running_mean[:5])
y_train = model(x)
print("训练模式输出形状:", y_train.shape)

# 评估模式
model.eval()
print("\n评估模式:")
with torch.no_grad():
    y_eval = model(x)
print("评估模式输出形状:", y_eval.shape)

# 7. 模块状态保存和加载
print("\n7. 模块状态保存和加载：")

# 保存模型
model_path = "temp_model.pth"
torch.save(model.state_dict(), model_path)
print("模型已保存")

# 加载模型
new_model = DropoutBatchNormModel(10, 64, 2)
new_model.load_state_dict(torch.load(model_path))
print("模型已加载")

# 验证加载的模型
new_model.eval()
with torch.no_grad():
    y_new = new_model(x)
print("加载模型的输出形状:", y_new.shape)

# 8. 设备管理
print("\n8. 设备管理：")

model = CustomModel(10, 64, 2)
print("初始设备:", next(model.parameters()).device)

# 如果有GPU，移动到GPU
if torch.cuda.is_available():
    model = model.cuda()
    print("移动到GPU后:", next(model.parameters()).device)
    
    # 输入也需要移动到同一设备
    x = torch.randn(5, 10).cuda()
    y = model(x)
    print("GPU输出形状:", y.shape)
    
    # 移回CPU
    model = model.cpu()
    print("移回CPU后:", next(model.parameters()).device)

# 9. 子模块访问和修改
print("\n9. 子模块访问和修改：")

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.classifier = nn.Linear(32, 2)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features)
        return output

model = ComplexModel()

# 访问子模块
print("所有子模块:")
for name, module in model.named_modules():
    print(f"  {name}: {type(module).__name__}")

# 修改子模块
print("\n冻结特征提取器:")
for param in model.feature_extractor.parameters():
    param.requires_grad = False

print("可训练参数:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.shape}")

# 10. 钩子函数（Hooks）
print("\n10. 钩子函数（Hooks）：")

class HookModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 1)
        self.activations = {}
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    
    def register_hooks(self):
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        self.linear1.register_forward_hook(hook_fn('linear1'))
        self.linear2.register_forward_hook(hook_fn('linear2'))

model = HookModel()
model.register_hooks()

x = torch.randn(3, 10)
y = model(x)

print("捕获的激活:")
for name, activation in model.activations.items():
    print(f"  {name}: {activation.shape}")

# 清理临时文件
import os
if os.path.exists(model_path):
    os.remove(model_path) 