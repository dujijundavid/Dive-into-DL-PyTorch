"""
PyTorch 优化算法核心技术与深度理解
---------------------------------
【文件说明】
本文件系统梳理了深度学习中的核心优化算法，包括：
- 梯度下降法族：SGD、Mini-batch SGD
- 动量方法：Momentum、Nesterov Accelerated Gradient
- 自适应学习率方法：AdaGrad、RMSprop、Adam、AdaDelta
- 优化算法的数学原理与工程实现
- 不同优化器在实际问题中的选择策略

【第一性原理思考】
1. 为什么需要优化算法？
   - 深度学习本质是参数优化问题：min f(θ)
   - 损失函数通常非凸、高维、噪声大
   - 需要高效算法在合理时间内找到较好解

2. 不同优化算法解决什么问题？
   - SGD：基础梯度下降，简单但可能陷入局部最优
   - Momentum：利用历史梯度信息，加速收敛
   - AdaGrad/RMSprop：自适应调整学习率，处理稀疏梯度
   - Adam：结合动量和自适应学习率的优势

3. 优化算法的核心权衡是什么？
   - 收敛速度 vs 计算复杂度
   - 泛化能力 vs 训练精度
   - 超参数敏感性 vs 算法鲁棒性

【苏格拉底式提问与验证】
1. 为什么mini-batch比全批次和单样本更好？
   - 问题：计算效率和梯度质量如何平衡？
   - 验证：通过不同batch size的收敛曲线对比
   - 结论：mini-batch兼顾效率和梯度质量

2. 自适应学习率真的总是更好吗？
   - 问题：Adam为什么有时不如SGD？
   - 验证：在不同任务上比较优化器性能
   - 结论：没有万能的优化器，需要根据任务选择

【费曼学习法讲解】
1. 概念解释
   - 用登山类比梯度下降过程
   - 用滑雪类比动量方法
   - 用自适应车速类比自适应学习率

2. 实例教学
   - 从简单二次函数开始理解
   - 扩展到复杂神经网络
   - 通过可视化展示优化轨迹

【设计意义与工程价值】
- 优化算法是深度学习训练的核心引擎
- 选择合适的优化器可以显著提升训练效果
- 理解优化原理有助于调试训练问题

可运行案例：
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, TensorDataset

# 1. 基础梯度下降算法
# ------------------
# 原理说明：
# 梯度下降是所有优化算法的基础，通过负梯度方向更新参数
# θ = θ - η∇f(θ)，其中η是学习率

print("1. 基础梯度下降算法")
print("=" * 50)

class GradientDescentDemo:
    """梯度下降演示"""
    
    def __init__(self):
        self.history = {'loss': [], 'params': []}
    
    def objective_function(self, x, y):
        """目标函数：简单的二次函数"""
        return (x - 2)**2 + (y - 1)**2
    
    def gradient(self, x, y):
        """解析梯度"""
        grad_x = 2 * (x - 2)
        grad_y = 2 * (y - 1)
        return grad_x, grad_y
    
    def vanilla_gd(self, lr=0.1, max_iter=100):
        """原始梯度下降"""
        print("原始梯度下降：")
        
        # 初始化参数
        x, y = torch.tensor(0.0, requires_grad=True), torch.tensor(0.0, requires_grad=True)
        optimizer = optim.SGD([x, y], lr=lr)
        
        print(f"初始位置: x={x:.4f}, y={y:.4f}")
        
        for i in range(max_iter):
            optimizer.zero_grad()
            
            # 计算损失
            loss = (x - 2)**2 + (y - 1)**2
            
            # 反向传播
            loss.backward()
            
            # 参数更新
            optimizer.step()
            
            if i % 20 == 0:
                print(f"迭代 {i}: loss={loss.item():.6f}, x={x.item():.4f}, y={y.item():.4f}")
        
        print(f"最终位置: x={x:.4f}, y={y:.4f}")
        return x.item(), y.item()
    
    def sgd_vs_batch_gd(self):
        """SGD vs 批量梯度下降对比"""
        print("\nSGD vs 批量梯度下降对比：")
        
        # 生成噪声数据来模拟SGD效果
        n_samples = 100
        x_data = torch.randn(n_samples, 2)
        y_data = 3 * x_data[:, 0] + 2 * x_data[:, 1] + torch.randn(n_samples) * 0.1
        
        # 线性回归模型
        model_sgd = nn.Linear(2, 1)
        model_batch = nn.Linear(2, 1)
        
        # 初始化为相同参数
        with torch.no_grad():
            model_batch.weight.copy_(model_sgd.weight)
            model_batch.bias.copy_(model_sgd.bias)
        
        optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.01)
        optimizer_batch = optim.SGD(model_batch.parameters(), lr=0.01)
        
        dataset = TensorDataset(x_data, y_data)
        dataloader_sgd = DataLoader(dataset, batch_size=1, shuffle=True)
        dataloader_batch = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        
        # SGD训练
        model_sgd.train()
        for epoch in range(10):
            for batch_x, batch_y in dataloader_sgd:
                optimizer_sgd.zero_grad()
                pred = model_sgd(batch_x).squeeze()
                loss = ((pred - batch_y) ** 2).mean()
                loss.backward()
                optimizer_sgd.step()
        
        # 批量GD训练
        model_batch.train()
        for epoch in range(10):
            for batch_x, batch_y in dataloader_batch:
                optimizer_batch.zero_grad()
                pred = model_batch(batch_x).squeeze()
                loss = ((pred - batch_y) ** 2).mean()
                loss.backward()
                optimizer_batch.step()
        
        print("训练完成，参数对比：")
        print(f"SGD参数: w={model_sgd.weight.data}, b={model_sgd.bias.data}")
        print(f"Batch GD参数: w={model_batch.weight.data}, b={model_batch.bias.data}")

# 演示梯度下降
gd_demo = GradientDescentDemo()
gd_demo.vanilla_gd()
gd_demo.sgd_vs_batch_gd()

# 2. 动量方法
# ----------
# 原理说明：
# 动量方法通过累积历史梯度信息来加速收敛，减少震荡
# v = βv + (1-β)∇f(θ), θ = θ - ηv

print("\n\n2. 动量方法")
print("=" * 50)

class MomentumDemo:
    """动量方法演示"""
    
    def momentum_visualization(self):
        """动量方法可视化对比"""
        print("动量方法对比：")
        
        # 创建一个椭圆形损失函数（条件数较大）
        def loss_func(x, y):
            return 0.5 * (x**2 + 10 * y**2)
        
        # 不同优化器设置
        x1, y1 = torch.tensor(4.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)
        x2, y2 = torch.tensor(4.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)
        x3, y3 = torch.tensor(4.0, requires_grad=True), torch.tensor(2.0, requires_grad=True)
        
        optimizer_sgd = optim.SGD([x1, y1], lr=0.01)
        optimizer_momentum = optim.SGD([x2, y2], lr=0.01, momentum=0.9)
        optimizer_nesterov = optim.SGD([x3, y3], lr=0.01, momentum=0.9, nesterov=True)
        
        print(f"初始位置: x=4.0, y=2.0")
        
        losses_sgd, losses_momentum, losses_nesterov = [], [], []
        
        for i in range(100):
            # SGD
            optimizer_sgd.zero_grad()
            loss_sgd = loss_func(x1, y1)
            loss_sgd.backward()
            optimizer_sgd.step()
            losses_sgd.append(loss_sgd.item())
            
            # Momentum
            optimizer_momentum.zero_grad()
            loss_momentum = loss_func(x2, y2)
            loss_momentum.backward()
            optimizer_momentum.step()
            losses_momentum.append(loss_momentum.item())
            
            # Nesterov
            optimizer_nesterov.zero_grad()
            loss_nesterov = loss_func(x3, y3)
            loss_nesterov.backward()
            optimizer_nesterov.step()
            losses_nesterov.append(loss_nesterov.item())
            
            if i % 20 == 0:
                print(f"迭代 {i}:")
                print(f"  SGD: loss={losses_sgd[-1]:.6f}, x={x1.item():.4f}, y={y1.item():.4f}")
                print(f"  Momentum: loss={losses_momentum[-1]:.6f}, x={x2.item():.4f}, y={y2.item():.4f}")
                print(f"  Nesterov: loss={losses_nesterov[-1]:.6f}, x={x3.item():.4f}, y={y3.item():.4f}")
        
        print("\n收敛性对比：")
        print(f"SGD最终损失: {losses_sgd[-1]:.6f}")
        print(f"Momentum最终损失: {losses_momentum[-1]:.6f}")
        print(f"Nesterov最终损失: {losses_nesterov[-1]:.6f}")
    
    def momentum_theory(self):
        """动量方法理论解释"""
        print("\n动量方法理论：")
        print("1. 标准动量：v_t = β*v_{t-1} + (1-β)*g_t")
        print("2. Nesterov动量：计算未来位置的梯度")
        print("3. 动量系数β通常取0.9，表示90%的历史信息保留")
        print("4. 动量方法可以穿越局部最优点，加速收敛")

# 演示动量方法
momentum_demo = MomentumDemo()
momentum_demo.momentum_visualization()
momentum_demo.momentum_theory()

# 3. 自适应学习率方法
# ------------------
# 原理说明：
# 自适应方法根据历史梯度自动调整每个参数的学习率
# 主要包括AdaGrad、RMSprop、Adam等算法

print("\n\n3. 自适应学习率方法")
print("=" * 50)

class AdaptiveOptimizersDemo:
    """自适应优化器演示"""
    
    def __init__(self):
        self.optimizers_performance = {}
    
    def create_model_and_data(self):
        """创建模型和数据"""
        # 简单的多层感知机
        model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 生成回归数据
        n_samples = 1000
        X = torch.randn(n_samples, 10)
        y = torch.sum(X[:, :5], dim=1, keepdim=True) + torch.randn(n_samples, 1) * 0.1
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        return model, dataloader
    
    def compare_optimizers(self):
        """比较不同优化器"""
        print("自适应优化器性能对比：")
        
        # 准备优化器配置
        optimizer_configs = {
            'SGD': {'lr': 0.01},
            'SGD+Momentum': {'lr': 0.01, 'momentum': 0.9},
            'AdaGrad': {'lr': 0.1},
            'RMSprop': {'lr': 0.01},
            'Adam': {'lr': 0.001},
            'AdaDelta': {}
        }
        
        results = {}
        
        for opt_name, opt_params in optimizer_configs.items():
            print(f"\n训练使用 {opt_name}:")
            
            # 创建新的模型和数据
            model, dataloader = self.create_model_and_data()
            
            # 创建优化器
            if opt_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), **opt_params)
            elif opt_name == 'SGD+Momentum':
                optimizer = optim.SGD(model.parameters(), **opt_params)
            elif opt_name == 'AdaGrad':
                optimizer = optim.Adagrad(model.parameters(), **opt_params)
            elif opt_name == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), **opt_params)
            elif opt_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), **opt_params)
            elif opt_name == 'AdaDelta':
                optimizer = optim.Adadelta(model.parameters(), **opt_params)
            
            # 训练
            model.train()
            losses = []
            
            for epoch in range(20):
                epoch_loss = 0
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    pred = model(batch_x)
                    loss = nn.MSELoss()(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                losses.append(avg_loss)
                
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch}: {avg_loss:.6f}")
            
            results[opt_name] = losses[-1]
        
        # 输出最终结果对比
        print("\n最终损失对比：")
        for opt_name, final_loss in results.items():
            print(f"{opt_name}: {final_loss:.6f}")
    
    def optimizer_theory_explanation(self):
        """优化器理论解释"""
        print("\n优化器理论解释：")
        
        print("\n1. AdaGrad:")
        print("   - 累积历史梯度平方：G_t = G_{t-1} + g_t²")
        print("   - 自适应学习率：θ = θ - η/√(G_t + ε) * g_t")
        print("   - 优点：稀疏梯度友好")
        print("   - 缺点：学习率单调递减")
        
        print("\n2. RMSprop:")
        print("   - 指数移动平均：G_t = β*G_{t-1} + (1-β)*g_t²")
        print("   - 解决AdaGrad学习率过度衰减问题")
        print("   - 适合非稳态目标函数")
        
        print("\n3. Adam:")
        print("   - 结合动量和RMSprop的优势")
        print("   - 一阶矩估计：m_t = β₁*m_{t-1} + (1-β₁)*g_t")
        print("   - 二阶矩估计：v_t = β₂*v_{t-1} + (1-β₂)*g_t²")
        print("   - 偏差修正：m̂_t = m_t/(1-β₁ᵗ), v̂_t = v_t/(1-β₂ᵗ)")
        print("   - 参数更新：θ = θ - η*m̂_t/(√v̂_t + ε)")
        
        print("\n4. AdaDelta:")
        print("   - 不需要设置学习率")
        print("   - 使用参数更新的历史信息")
        print("   - 适合长时间训练")

# 演示自适应优化器
adaptive_demo = AdaptiveOptimizersDemo()
adaptive_demo.compare_optimizers()
adaptive_demo.optimizer_theory_explanation()

# 4. 优化器选择指南
# ----------------
# 原理说明：
# 不同优化器适用于不同场景，需要根据具体问题选择

print("\n\n4. 优化器选择指南")
print("=" * 50)

class OptimizerSelectionGuide:
    """优化器选择指南"""
    
    def selection_criteria(self):
        """选择标准"""
        print("优化器选择标准：")
        
        print("\n1. 根据数据特性：")
        print("   - 稠密数据：SGD、Adam")
        print("   - 稀疏数据：AdaGrad、Adam")
        print("   - 大规模数据：SGD、Adam")
        
        print("\n2. 根据任务类型：")
        print("   - 计算机视觉：SGD + Momentum、Adam")
        print("   - 自然语言处理：Adam、AdamW")
        print("   - 强化学习：Adam、RMSprop")
        
        print("\n3. 根据计算资源：")
        print("   - 内存受限：SGD")
        print("   - 计算充足：Adam")
        
        print("\n4. 根据调参经验：")
        print("   - 新手：Adam（鲁棒性好）")
        print("   - 专家：SGD（可调性强）")
    
    def hyperparameter_tuning(self):
        """超参数调优建议"""
        print("\n超参数调优建议：")
        
        print("\n1. 学习率调优：")
        print("   - 从较大值开始（如0.1），逐步减小")
        print("   - 使用学习率调度器")
        print("   - 观察损失曲线，避免震荡或停滞")
        
        print("\n2. 批次大小：")
        print("   - 小批次：泛化好，但训练慢")
        print("   - 大批次：训练快，但可能过拟合")
        print("   - 常用：32, 64, 128, 256")
        
        print("\n3. 动量参数：")
        print("   - 一般设置为0.9")
        print("   - 可以随训练进行调整")
        
        print("\n4. Adam参数：")
        print("   - β₁ = 0.9, β₂ = 0.999（默认值通常有效）")
        print("   - ε = 1e-8（防止除零）")
    
    def debugging_training(self):
        """训练调试技巧"""
        print("\n训练调试技巧：")
        
        print("\n1. 常见问题诊断：")
        print("   - 损失不下降：学习率过小、梯度消失")
        print("   - 损失震荡：学习率过大、批次太小")
        print("   - 过拟合：正则化、dropout、数据增强")
        print("   - 欠拟合：增加模型容量、降低正则化")
        
        print("\n2. 监控指标：")
        print("   - 损失曲线")
        print("   - 梯度范数")
        print("   - 参数更新幅度")
        print("   - 验证集性能")

# 演示优化器选择指南
guide = OptimizerSelectionGuide()
guide.selection_criteria()
guide.hyperparameter_tuning()
guide.debugging_training()

# 5. 高级优化技巧
# --------------
# 原理说明：
# 现代深度学习中的高级优化技巧

print("\n\n5. 高级优化技巧")
print("=" * 50)

class AdvancedOptimizationTricks:
    """高级优化技巧"""
    
    def learning_rate_scheduling(self):
        """学习率调度"""
        print("学习率调度策略：")
        
        # 创建简单模型
        model = nn.Linear(10, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        # 不同调度器
        schedulers = {
            'StepLR': optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5),
            'ExponentialLR': optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95),
            'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20),
        }
        
        print("\n各种学习率调度器：")
        for name, scheduler in schedulers.items():
            print(f"{name}: 用于不同的衰减策略")
        
        # 演示学习率变化
        initial_lr = 0.1
        lrs = []
        for epoch in range(30):
            current_lr = initial_lr * (0.95 ** epoch)  # 指数衰减示例
            lrs.append(current_lr)
        
        print(f"\n学习率衰减示例（前10个epoch）:")
        for i in range(min(10, len(lrs))):
            print(f"Epoch {i}: lr={lrs[i]:.6f}")
    
    def gradient_clipping(self):
        """梯度裁剪"""
        print("\n梯度裁剪技术：")
        
        model = nn.LSTM(input_size=10, hidden_size=20, num_layers=2)
        optimizer = optim.Adam(model.parameters())
        
        # 模拟训练步骤
        x = torch.randn(5, 3, 10)  # (seq_len, batch, input_size)
        y = torch.randn(5, 3, 20)
        
        optimizer.zero_grad()
        output, _ = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # 梯度裁剪
        max_norm = 1.0
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        print(f"梯度范数: {total_norm:.6f}")
        print(f"裁剪后最大范数: {max_norm}")
        print("梯度裁剪防止梯度爆炸，特别在RNN中重要")
        
        optimizer.step()
    
    def warm_restart(self):
        """热重启技术"""
        print("\n热重启技术：")
        print("1. 周期性重启学习率")
        print("2. 帮助跳出局部最优")
        print("3. 在训练后期特别有效")
        print("4. SGDR (Stochastic Gradient Descent with Warm Restarts)")
        
        model = nn.Linear(10, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        
        print("\n模拟热重启学习率变化:")
        for epoch in range(30):
            current_lr = optimizer.param_groups[0]['lr']
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: lr={current_lr:.6f}")
            scheduler.step()

# 演示高级优化技巧
advanced_tricks = AdvancedOptimizationTricks()
advanced_tricks.learning_rate_scheduling()
advanced_tricks.gradient_clipping()
advanced_tricks.warm_restart()

# 6. 实际应用案例
# --------------
# 原理说明：
# 在实际深度学习项目中应用优化技巧

print("\n\n6. 实际应用案例")
print("=" * 50)

class OptimizationCaseStudy:
    """优化算法案例研究"""
    
    def neural_network_training(self):
        """神经网络训练案例"""
        print("神经网络训练完整案例：")
        
        # 创建更复杂的模型
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dims, output_dim):
                super().__init__()
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, output_dim))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        # 生成分类数据
        n_samples = 2000
        n_features = 20
        n_classes = 5
        
        X = torch.randn(n_samples, n_features)
        y = torch.randint(0, n_classes, (n_samples,))
        
        # 分割数据
        train_size = int(0.8 * n_samples)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # 创建模型
        model = MLP(n_features, [128, 64, 32], n_classes)
        
        # 优化器选择和配置
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        criterion = nn.CrossEntropyLoss()
        
        print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
        print("开始训练...")
        
        # 训练循环
        model.train()
        for epoch in range(20):
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            
            # 学习率调度
            scheduler.step(avg_loss)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # 测试评估
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            _, test_predicted = test_outputs.max(1)
            test_accuracy = 100. * test_predicted.eq(y_test).sum().item() / y_test.size(0)
            print(f"测试准确率: {test_accuracy:.2f}%")
    
    def optimization_best_practices(self):
        """优化最佳实践总结"""
        print("\n优化最佳实践总结：")
        
        print("\n1. 选择策略：")
        print("   - 快速原型：Adam + 默认参数")
        print("   - 生产模型：SGD + Momentum + 精细调参")
        print("   - 大模型：AdamW + 学习率调度")
        
        print("\n2. 调参顺序：")
        print("   - 首先调整学习率")
        print("   - 然后调整批次大小")
        print("   - 最后调整其他超参数")
        
        print("\n3. 监控指标：")
        print("   - 训练/验证损失")
        print("   - 梯度范数")
        print("   - 参数更新比例")
        print("   - 学习率变化")
        
        print("\n4. 常见陷阱：")
        print("   - 学习率过大导致不收敛")
        print("   - 批次过小导致训练不稳定")
        print("   - 过度依赖默认参数")
        print("   - 忽视验证集性能")

# 演示实际应用案例
case_study = OptimizationCaseStudy()
case_study.neural_network_training()
case_study.optimization_best_practices()

print("\n" + "=" * 50)
print("优化算法总结：")
print("1. SGD是基础，理解其原理对掌握其他优化器至关重要")
print("2. Momentum方法通过历史信息加速收敛，减少震荡")
print("3. 自适应方法（Adam等）自动调整学习率，使用方便")
print("4. 没有万能的优化器，需要根据具体问题选择")
print("5. 超参数调优是一门艺术，需要经验和实验")
print("6. 现代训练技巧（学习率调度、梯度裁剪等）同样重要")
print("7. 监控训练过程，及时发现和解决问题是关键")
