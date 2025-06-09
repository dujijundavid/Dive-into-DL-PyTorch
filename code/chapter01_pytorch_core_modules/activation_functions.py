"""
激活函数核心原理与用法
-------------------
第一性原理思考：
1. 什么是激活函数？
   - 激活函数为神经网络引入非线性变换
   - 决定神经元的输出是否被"激活"
   - 是深度网络能够学习复杂模式的关键

2. 为什么需要激活函数？
   - 引入非线性：没有激活函数的深度网络等价于线性模型
   - 控制信息流：决定哪些信息应该传递到下一层
   - 梯度传播：影响反向传播的梯度流动

3. 激活函数的核心特性是什么？
   - 非线性：能够处理复杂的数据关系
   - 可微性：支持梯度下降优化
   - 计算效率：影响训练和推理速度
   - 梯度特性：影响深度网络的训练稳定性

苏格拉底式提问与验证：
1. 不同激活函数有什么区别？
   - 问题：ReLU和Sigmoid在什么情况下表现不同？
   - 验证：比较不同激活函数的输出特性和梯度
   - 结论：每种激活函数都有其优势和局限性

2. 激活函数如何影响梯度传播？
   - 问题：为什么深度网络会出现梯度消失问题？
   - 验证：观察不同激活函数的梯度传播特性
   - 结论：激活函数的选择直接影响训练效果

3. 如何选择合适的激活函数？
   - 问题：什么情况下应该使用哪种激活函数？
   - 验证：在不同任务上测试不同激活函数
   - 结论：激活函数的选择需要考虑任务特性

费曼学习法讲解：
1. 概念解释
   - 用开关类比解释激活函数的作用
   - 通过函数图形理解不同激活函数的特性
   - 强调激活函数在深度学习中的基础地位

2. 实例教学
   - 从简单到复杂的激活函数应用
   - 通过可视化理解激活函数的行为
   - 实践不同场景下的激活函数选择

3. 知识巩固
   - 总结各种激活函数的优缺点
   - 提供激活函数选择的指导原则
   - 建议进阶学习方向

功能说明：
- 激活函数为神经网络引入非线性，是深度学习的核心组件。

原理讲解：
- 通过非线性变换，使神经网络能够学习复杂的函数映射。
- 不同的激活函数有不同的数学性质和适用场景。
- 激活函数的选择影响网络的表达能力和训练稳定性。

工程意义：
- 是神经网络设计的关键决策，直接影响模型性能和训练效果。

可运行案例：
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 1. 基本激活函数介绍
print("1. 基本激活函数介绍：")

# 创建测试数据
x = torch.linspace(-5, 5, 100)

# 常用激活函数
activations = {
    'ReLU': F.relu,
    'Sigmoid': torch.sigmoid,
    'Tanh': torch.tanh,
    'LeakyReLU': lambda x: F.leaky_relu(x, 0.1),
    'ELU': lambda x: F.elu(x, alpha=1.0),
    'GELU': F.gelu,
    'Swish': lambda x: x * torch.sigmoid(x)
}

print("激活函数在x=0处的输出：")
for name, func in activations.items():
    output = func(torch.tensor(0.0))
    print(f"{name}: {output.item():.4f}")

# 2. 激活函数的数学性质
print("\n2. 激活函数的数学性质：")

def analyze_activation_properties(func, name, x_range=(-3, 3)):
    x = torch.linspace(x_range[0], x_range[1], 1000, requires_grad=True)
    y = func(x)
    
    # 计算梯度
    gradient = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    
    # 分析性质
    range_min, range_max = y.min().item(), y.max().item()
    is_monotonic = torch.all(gradient >= 0).item() or torch.all(gradient <= 0).item()
    zero_centered = abs(y.mean().item()) < 0.1
    
    print(f"\n{name}激活函数分析：")
    print(f"  输出范围: [{range_min:.3f}, {range_max:.3f}]")
    print(f"  是否单调: {is_monotonic}")
    print(f"  是否零中心: {zero_centered}")
    print(f"  在x=0处的梯度: {gradient[len(gradient)//2].item():.4f}")
    
    return x.detach(), y.detach(), gradient.detach()

# 分析几个关键激活函数
key_activations = {
    'ReLU': F.relu,
    'Sigmoid': torch.sigmoid,
    'Tanh': torch.tanh,
    'GELU': F.gelu
}

analysis_results = {}
for name, func in key_activations.items():
    x_vals, y_vals, grad_vals = analyze_activation_properties(func, name)
    analysis_results[name] = (x_vals, y_vals, grad_vals)

# 3. 激活函数的梯度问题
print("\n3. 激活函数的梯度问题：")

def demonstrate_gradient_flow():
    # 创建深度网络模拟梯度传播
    depth = 10
    input_size = 100
    
    # 不同激活函数的网络
    networks = {
        'Sigmoid': [nn.Linear(input_size, input_size), nn.Sigmoid()] * depth,
        'ReLU': [nn.Linear(input_size, input_size), nn.ReLU()] * depth,
        'LeakyReLU': [nn.Linear(input_size, input_size), nn.LeakyReLU(0.1)] * depth,
    }
    
    results = {}
    
    for name, layers in networks.items():
        model = nn.Sequential(*layers[:-1])  # 去掉最后一个激活函数
        
        # 前向传播
        x = torch.randn(1, input_size, requires_grad=True)
        y = model(x)
        loss = y.sum()
        
        # 反向传播
        loss.backward()
        
        # 计算梯度范数
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
        
        results[name] = {
            'mean_grad_norm': np.mean(grad_norms),
            'std_grad_norm': np.std(grad_norms),
            'min_grad_norm': np.min(grad_norms),
            'max_grad_norm': np.max(grad_norms)
        }
        
        print(f"{name}网络梯度统计:")
        print(f"  平均梯度范数: {results[name]['mean_grad_norm']:.6f}")
        print(f"  梯度范数标准差: {results[name]['std_grad_norm']:.6f}")
    
    return results

gradient_results = demonstrate_gradient_flow()

# 4. 现代激活函数
print("\n4. 现代激活函数：")

class Swish(nn.Module):
    def __init__(self, beta=1.0):
        super(Swish, self).__init__()
        self.beta = beta
    
    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
                                        (x + 0.044715 * torch.pow(x, 3))))

# 测试现代激活函数
modern_activations = {
    'Swish': Swish(),
    'Mish': Mish(),
    'GELU': GELU(),
    'Official_GELU': nn.GELU()
}

x_test = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print("现代激活函数在关键点的输出：")
for name, func in modern_activations.items():
    outputs = func(x_test)
    print(f"{name}: {outputs.detach().numpy()}")

# 5. 激活函数的适用场景
print("\n5. 激活函数的适用场景：")

def test_activation_performance():
    # 创建简单的分类任务
    from torch.utils.data import TensorDataset, DataLoader
    
    # 生成数据
    X = torch.randn(1000, 20)
    y = (X.sum(dim=1) > 0).long()
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 测试不同激活函数的模型
    activations_to_test = {
        'ReLU': nn.ReLU(),
        'Sigmoid': nn.Sigmoid(),
        'Tanh': nn.Tanh(),
        'LeakyReLU': nn.LeakyReLU(0.1),
        'ELU': nn.ELU(),
        'GELU': nn.GELU()
    }
    
    results = {}
    
    for act_name, activation in activations_to_test.items():
        # 创建模型
        model = nn.Sequential(
            nn.Linear(20, 64),
            activation,
            nn.Linear(64, 32),
            activation,
            nn.Linear(32, 2)
        )
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练
        model.train()
        total_loss = 0
        for epoch in range(10):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss
        
        # 评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        results[act_name] = {
            'final_loss': total_loss / 10,
            'accuracy': accuracy
        }
        
        print(f"{act_name}: 损失 {results[act_name]['final_loss']:.4f}, "
              f"准确率 {results[act_name]['accuracy']:.2f}%")
    
    return results

performance_results = test_activation_performance()

# 6. 激活函数的可视化
print("\n6. Activation Function Visualization:")

def plot_activation_functions():
    x = torch.linspace(-3, 3, 1000)
    
    # 选择要可视化的激活函数
    functions_to_plot = {
        'ReLU': F.relu,
        'Sigmoid': torch.sigmoid,
        'Tanh': torch.tanh,
        'LeakyReLU': lambda x: F.leaky_relu(x, 0.1),
        'ELU': lambda x: F.elu(x, alpha=1.0),
        'GELU': F.gelu
    }
    
    plt.figure(figsize=(15, 10))
    
    # 绘制激活函数
    for i, (name, func) in enumerate(functions_to_plot.items(), 1):
        plt.subplot(2, 3, i)
        y = func(x)
        plt.plot(x.numpy(), y.numpy(), linewidth=2)
        plt.title(f'{name} Activation')
        plt.grid(True)
        plt.xlabel('Input')
        plt.ylabel('Output')
    
    plt.tight_layout()
    plt.show()

# Note: Uncomment the following line to run the visualization
# plot_activation_functions()
print("Activation function visualization code is ready (uncomment plot_activation_functions() to view graphics)")

# 7. 自定义激活函数
print("\n7. 自定义激活函数：")

class ParametricReLU(nn.Module):
    def __init__(self, num_parameters=1, init=0.25):
        super(ParametricReLU, self).__init__()
        self.num_parameters = num_parameters
        self.weight = nn.Parameter(torch.Tensor(num_parameters).fill_(init))
    
    def forward(self, x):
        return F.prelu(x, self.weight)

class AdaptiveActivation(nn.Module):
    def __init__(self, in_features):
        super(AdaptiveActivation, self).__init__()
        self.alpha = nn.Parameter(torch.ones(in_features))
        self.beta = nn.Parameter(torch.zeros(in_features))
    
    def forward(self, x):
        return self.alpha * F.relu(x) + self.beta * F.tanh(x)

# 测试自定义激活函数
custom_activations = {
    'PReLU': ParametricReLU(1, 0.1),
    'Adaptive': AdaptiveActivation(10)
}

x_custom = torch.randn(5, 10)
print("自定义激活函数测试：")
for name, func in custom_activations.items():
    if name == 'PReLU':
        output = func(x_custom[:, 0:1])  # PReLU 只需要一维
        print(f"{name} 输出形状: {output.shape}")
    else:
        output = func(x_custom)
        print(f"{name} 输出形状: {output.shape}")

# 8. 激活函数的计算效率
print("\n8. 激活函数的计算效率：")

def benchmark_activations():
    import time
    
    # 准备大数据
    x = torch.randn(1000, 1000)
    if torch.cuda.is_available():
        x = x.cuda()
    
    activations = {
        'ReLU': F.relu,
        'Sigmoid': torch.sigmoid,
        'Tanh': torch.tanh,
        'GELU': F.gelu,
        'ELU': F.elu,
        'Swish': lambda x: x * torch.sigmoid(x)
    }
    
    print("激活函数计算时间对比 (1000x1000 张量, 1000次迭代):")
    
    for name, func in activations.items():
        # 预热
        for _ in range(10):
            _ = func(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(1000):
            _ = func(x)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 1000 * 1000  # 转换为毫秒
        print(f"{name}: {avg_time:.4f} ms")

benchmark_activations()

# 9. 激活函数的死神经元问题
print("\n9. 激活函数的死神经元问题：")

def demonstrate_dying_relu():
    # 创建一个简单网络，演示ReLU死神经元问题
    model = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # 使用较大的学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    # 训练数据
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    dead_neurons_history = []
    
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        # 检查死神经元
        with torch.no_grad():
            activations = model[1](model[0](X))  # 第一层ReLU的输出
            dead_neurons = (activations <= 0).all(dim=0).sum().item()
            dead_neurons_history.append(dead_neurons)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: 死神经元数量 {dead_neurons}/100")
    
    return dead_neurons_history

dying_relu_history = demonstrate_dying_relu()

# 10. 激活函数选择指南
print("\n10. 激活函数选择指南：")

def activation_selection_guide():
    guide = {
        '隐藏层默认选择': {
            'ReLU': '简单有效，计算快速，但可能有死神经元问题',
            'LeakyReLU': '解决ReLU死神经元问题',
            'ELU': '平滑版本的ReLU，避免死神经元',
            'GELU': 'Transformer等现代架构的首选'
        },
        '输出层选择': {
            'Linear': '回归任务',
            'Sigmoid': '二分类任务',
            'Softmax': '多分类任务',
            'Tanh': '输出范围需要在(-1,1)的任务'
        },
        '特殊场景': {
            'Swish/SiLU': '深度网络和复杂任务',
            'Mish': '需要平滑梯度的场景',
            'PReLU': '需要学习负斜率的场景'
        }
    }
    
    print("激活函数选择指南：")
    for category, options in guide.items():
        print(f"\n{category}：")
        for func, description in options.items():
            print(f"  {func}: {description}")

activation_selection_guide()

# 11. 激活函数的梯度可视化
print("\n11. Activation Function Gradient Visualization:")

def visualize_gradients():
    x = torch.linspace(-3, 3, 1000, requires_grad=True)
    
    activations = {
        'ReLU': F.relu,
        'Sigmoid': torch.sigmoid,
        'Tanh': torch.tanh,
        'ELU': F.elu,
        'GELU': F.gelu
    }
    
    print("激活函数在关键点的梯度：")
    test_points = [-2, -1, 0, 1, 2]
    
    for name, func in activations.items():
        gradients = []
        for point in test_points:
            x_point = torch.tensor(float(point), requires_grad=True)
            y = func(x_point)
            grad = torch.autograd.grad(y, x_point)[0]
            gradients.append(grad.item())
        
        print(f"{name}: {[f'{g:.3f}' for g in gradients]}")

visualize_gradients()

# 12. 实际应用建议
print("\n12. 实际应用建议：")

class ActivationRecommendationSystem:
    @staticmethod
    def recommend(task_type, network_depth, data_characteristics):
        recommendations = {
            'shallow_network': {
                'regression': 'ReLU或Tanh',
                'classification': 'ReLU + Softmax输出',
                'binary_classification': 'ReLU + Sigmoid输出'
            },
            'deep_network': {
                'general': 'ReLU、ELU或GELU',
                'very_deep': 'ELU、GELU或ResNet结构',
                'transformer': 'GELU或Swish'
            },
            'special_cases': {
                'gan': 'LeakyReLU (生成器和判别器)',
                'rnn': 'Tanh (隐藏状态), Sigmoid (门控)',
                'autoencoder': 'ReLU (编码器), Sigmoid/Tanh (解码器)'
            }
        }
        
        if network_depth <= 3:
            category = 'shallow_network'
        else:
            category = 'deep_network'
        
        if task_type in recommendations[category]:
            return recommendations[category][task_type]
        elif task_type in recommendations['special_cases']:
            return recommendations['special_cases'][task_type]
        else:
            return recommendations[category].get('general', 'ReLU (默认选择)')

# 使用推荐系统
recommender = ActivationRecommendationSystem()
print("激活函数推荐示例：")
print("深度回归网络:", recommender.recommend('regression', 10, 'normal'))
print("浅层分类网络:", recommender.recommend('classification', 2, 'normal'))
print("GAN网络:", recommender.recommend('gan', 5, 'adversarial'))
print("Transformer:", recommender.recommend('transformer', 12, 'nlp'))

print("\n激活函数使用要点总结：")
print("1. 隐藏层首选ReLU系列（ReLU、LeakyReLU、ELU）")
print("2. 深度网络考虑使用GELU或ELU避免梯度问题")
print("3. 输出层根据任务选择（回归用线性，分类用Softmax/Sigmoid）")
print("4. 注意激活函数的计算效率和内存使用")
print("5. 对于特殊架构使用特定的激活函数（如Transformer用GELU）")
print("6. 可以尝试组合多种激活函数或使用可学习的激活函数") 