"""
PyTorch 深度学习基础核心技术与深度理解
-----------------------------------
【文件说明】
本文件系统梳理了深度学习的基础核心技术，包括：
- 线性回归与梯度下降：机器学习的起点
- 多层感知机与非线性变换：深度学习的基石
- 正则化技术：防止过拟合的核心方法
- 现代训练技巧：批量处理、优化器选择、学习率调度
- 实际工程实践：模型调试、性能分析、部署优化

【第一性原理思考】
1. 为什么需要深度学习？
   - 传统机器学习需要手工特征工程
   - 深度网络能自动学习层次化特征表示
   - 非线性变换的复合实现复杂函数逼近

2. 梯度下降为什么有效？
   - 损失函数的梯度指向最陡增长方向
   - 负梯度方向是局部最优下降路径
   - 通过迭代逼近全局或局部最优解

3. 为什么需要正则化？
   - 有限数据容易导致过拟合
   - 正则化增加先验约束，提升泛化
   - 平衡模型复杂度与拟合能力

【苏格拉底式提问与验证】
1. 更深的网络总是更好吗？
   - 问题：深度与性能的关系是什么？
   - 验证：通过不同深度的MLP对比
   - 结论：需要合适的数据量和正则化

2. 激活函数的选择有何影响？
   - 问题：ReLU为什么比Sigmoid更流行？
   - 验证：通过梯度流动和训练速度对比
   - 结论：ReLU解决梯度消失，训练更稳定

【费曼学习法讲解】
1. 概念解释
   - 用堆积木类比神经网络的层次结构
   - 用学习过程类比梯度下降优化
   - 强调深度学习在各领域的重要应用

2. 实例教学
   - 从简单的线性回归开始
   - 逐步构建多层感知机
   - 通过可视化理解特征学习过程

【设计意义与工程价值】
- 深度学习是现代AI的基础，影响了计算机视觉、自然语言处理等领域
- 理解基础原理对掌握更复杂模型至关重要
- 工程实践技巧直接影响模型性能和部署效果

可运行案例：
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import time

if __name__ == "__main__":
    print("========== PyTorch深度学习基础核心技术演示 ==========")
    
    # 1. 第一性原理：从线性回归理解梯度下降
    print("\n========== 第一性原理：梯度下降本质 ==========")
    
    def gradient_descent_visualization():
        """
        可视化梯度下降过程，理解优化的本质
        展示参数更新轨迹和损失变化
        """
        # 创建简单的二次函数作为损失函数
        def loss_function(w):
            return (w - 2)**2 + 1
        
        def loss_gradient(w):
            return 2 * (w - 2)
        
        # 梯度下降迭代
        w_history = []
        loss_history = []
        
        w = torch.tensor(-2.0, requires_grad=True)  # 初始参数
        lr = 0.1  # 学习率
        
        print("梯度下降迭代过程：")
        for i in range(20):
            loss = loss_function(w)
            loss_history.append(loss.item())
            w_history.append(w.item())
            
            if i % 5 == 0:
                print(f"迭代{i}: w={w.item():.3f}, loss={loss.item():.3f}")
            
            # 计算梯度并更新参数
            loss.backward()
            with torch.no_grad():
                w -= lr * w.grad
                w.grad.zero_()
        
        print(f"最终结果: w={w.item():.3f}, 理论最优值: w=2.0")
        return w_history, loss_history
    
    w_hist, loss_hist = gradient_descent_visualization()
    
    # 2. 线性回归：深度学习的起点
    print("\n========== 线性回归：从理论到实践 ==========")
    
    class LinearRegressionDetailed(nn.Module):
        """
        详细的线性回归实现，包含参数初始化和前向传播逻辑
        原理：通过最小化均方误差学习输入输出的线性关系
        """
        def __init__(self, in_features, out_features):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)
            # 自定义初始化
            nn.init.normal_(self.linear.weight, mean=0, std=0.01)
            nn.init.constant_(self.linear.bias, 0)
            
        def forward(self, x):
            return self.linear(x)
    
    def linear_regression_demo():
        """线性回归完整演示：数据生成→模型训练→结果分析"""
        # 生成模拟数据
        torch.manual_seed(42)
        n_samples = 1000
        n_features = 2
        
        # 真实参数
        true_w = torch.tensor([[2.0], [-3.5]])
        true_b = 4.2
        
        # 生成数据：y = Xw + b + noise
        X = torch.randn(n_samples, n_features)
        noise = torch.randn(n_samples, 1) * 0.1
        y = X @ true_w + true_b + noise
        
        print(f"数据形状: X={X.shape}, y={y.shape}")
        print(f"真实参数: w={true_w.flatten().tolist()}, b={true_b}")
        
        # 创建模型
        model = LinearRegressionDetailed(n_features, 1)
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # 训练过程
        losses = []
        for epoch in range(1000):
            # 前向传播
            y_pred = model(X)
            loss = criterion(y_pred, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
        
        # 结果分析
        learned_w = model.linear.weight.data
        learned_b = model.linear.bias.item()
        
        print(f"学习到的参数: w={learned_w.flatten().tolist()}, b={learned_b:.3f}")
        print(f"参数误差: w_error={torch.norm(learned_w.flatten() - true_w.flatten()).item():.6f}")
        
        return model, losses
    
    lr_model, lr_losses = linear_regression_demo()
    
    # 3. 苏格拉底式提问：激活函数的重要性
    print("\n========== 苏格拉底式提问：为什么需要激活函数？ ==========")
    
    def activation_function_comparison():
        """
        对比不同激活函数的特性和影响
        证明非线性激活函数的必要性
        """
        # 创建测试数据
        x = torch.linspace(-5, 5, 1000)
        
        # 不同激活函数
        activations = {
            'ReLU': torch.relu,
            'Sigmoid': torch.sigmoid,
            'Tanh': torch.tanh,
            'LeakyReLU': lambda x: nn.functional.leaky_relu(x, 0.1),
            'ELU': nn.functional.elu
        }
        
        print("激活函数特性分析：")
        for name, func in activations.items():
            y = func(x)
            
            # 计算梯度
            x_grad = x.clone().requires_grad_()
            y_grad = func(x_grad)
            y_grad.sum().backward()
            gradient_norm = x_grad.grad.norm().item()
            
            print(f"{name}: 输出范围=[{y.min().item():.3f}, {y.max().item():.3f}], "
                  f"梯度范数={gradient_norm:.3f}")
        
        # 网络深度对激活函数的影响
        def test_deep_network_activation(activation_func, depth=10):
            """测试深层网络中激活函数的表现"""
            layers = []
            for i in range(depth):
                layers.extend([
                    nn.Linear(100, 100),
                    nn.BatchNorm1d(100) if i > 0 else nn.Identity(),
                    activation_func()
                ])
            
            network = nn.Sequential(*layers, nn.Linear(100, 1))
            
            # 测试梯度流动
            x = torch.randn(32, 100)
            y = network(x)
            loss = y.sum()
            loss.backward()
            
            # 统计梯度信息
            grad_stats = []
            for param in network.parameters():
                if param.grad is not None:
                    grad_stats.append(param.grad.norm().item())
            
            return np.mean(grad_stats), np.std(grad_stats)
        
        print("\n深层网络中激活函数的梯度流动：")
        activation_classes = {
            'ReLU': nn.ReLU,
            'Sigmoid': nn.Sigmoid,
            'Tanh': nn.Tanh
        }
        
        for name, act_class in activation_classes.items():
            mean_grad, std_grad = test_deep_network_activation(act_class)
            print(f"{name}: 平均梯度范数={mean_grad:.6f}, 标准差={std_grad:.6f}")
    
    activation_function_comparison()
    
    # 4. 多层感知机：深度学习的基石
    print("\n========== 多层感知机：深度学习基石 ==========")
    
    class MLPDetailed(nn.Module):
        """
        详细的多层感知机实现，包含多种配置选项
        展示网络设计的关键考虑因素
        """
        def __init__(self, input_dim, hidden_dims, output_dim, 
                     activation='relu', dropout_rate=0.0, batch_norm=False):
            super().__init__()
            
            self.layers = nn.ModuleList()
            prev_dim = input_dim
            
            # 构建隐藏层
            for hidden_dim in hidden_dims:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(hidden_dim))
                
                # 激活函数
                if activation == 'relu':
                    self.layers.append(nn.ReLU())
                elif activation == 'tanh':
                    self.layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    self.layers.append(nn.Sigmoid())
                
                if dropout_rate > 0:
                    self.layers.append(nn.Dropout(dropout_rate))
                
                prev_dim = hidden_dim
            
            # 输出层
            self.layers.append(nn.Linear(prev_dim, output_dim))
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    def mlp_architecture_comparison():
        """
        对比不同MLP架构的性能
        验证网络深度、宽度、正则化的影响
        """
        # 生成分类数据
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=5000, n_features=20, n_classes=3, 
                                 n_informative=15, random_state=42)
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # 数据划分
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # 不同架构配置
        architectures = {
            'Shallow': [64],
            'Deep_Narrow': [32, 32, 32, 32],
            'Deep_Wide': [128, 128, 64],
            'Deep_BN': [128, 128, 64],  # 带批归一化
            'Deep_Dropout': [128, 128, 64]  # 带Dropout
        }
        
        results = {}
        
        for arch_name, hidden_dims in architectures.items():
            print(f"\n训练架构: {arch_name}")
            
            # 创建模型
            batch_norm = 'BN' in arch_name
            dropout_rate = 0.3 if 'Dropout' in arch_name else 0.0
            
            model = MLPDetailed(20, hidden_dims, 3, 
                              batch_norm=batch_norm, 
                              dropout_rate=dropout_rate)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            # 训练
            model.train()
            train_losses = []
            
            for epoch in range(200):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # 测试
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test).item()
                test_acc = (test_outputs.argmax(dim=1) == y_test).float().mean().item()
            
            # 计算参数数量
            param_count = sum(p.numel() for p in model.parameters())
            
            results[arch_name] = {
                'test_loss': test_loss,
                'test_acc': test_acc,
                'param_count': param_count,
                'final_train_loss': train_losses[-1]
            }
            
            print(f"参数数量: {param_count}")
            print(f"最终训练损失: {train_losses[-1]:.4f}")
            print(f"测试损失: {test_loss:.4f}")
            print(f"测试准确率: {test_acc:.3f}")
        
        return results
    
    arch_results = mlp_architecture_comparison()
    
    # 5. 正则化技术深度解析
    print("\n========== 正则化技术：防止过拟合的艺术 ==========")
    
    def regularization_comprehensive_demo():
        """
        综合演示各种正则化技术的效果
        包括L1/L2正则化、Dropout、早停等
        """
        # 创建容易过拟合的小数据集
        torch.manual_seed(123)
        n_samples = 200
        n_features = 50
        
        X = torch.randn(n_samples, n_features)
        # 只有前10个特征有用，其余为噪声
        true_w = torch.zeros(n_features, 1)
        true_w[:10] = torch.randn(10, 1)
        y = X @ true_w + 0.1 * torch.randn(n_samples, 1)
        
        # 划分数据
        train_size = 100
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # 不同正则化方法
        configs = {
            'No_Reg': {'weight_decay': 0, 'dropout': 0},
            'L2_Reg': {'weight_decay': 0.01, 'dropout': 0},
            'Dropout': {'weight_decay': 0, 'dropout': 0.5},
            'L2_Dropout': {'weight_decay': 0.01, 'dropout': 0.3}
        }
        
        results = {}
        
        for config_name, config in configs.items():
            print(f"\n正则化配置: {config_name}")
            
            # 创建模型
            model = nn.Sequential(
                nn.Linear(n_features, 100),
                nn.ReLU(),
                nn.Dropout(config['dropout']),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Dropout(config['dropout']),
                nn.Linear(50, 1)
            )
            
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), 
                                       lr=0.001, 
                                       weight_decay=config['weight_decay'])
            
            # 训练和验证损失记录
            train_losses = []
            val_losses = []
            
            for epoch in range(500):
                # 训练
                model.train()
                optimizer.zero_grad()
                train_pred = model(X_train)
                train_loss = criterion(train_pred, y_train)
                train_loss.backward()
                optimizer.step()
                
                # 验证
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)
                    val_loss = criterion(val_pred, y_val)
                
                train_losses.append(train_loss.item())
                val_losses.append(val_loss.item())
            
            # 分析过拟合程度
            overfitting_degree = val_losses[-1] / train_losses[-1]
            
            results[config_name] = {
                'train_loss': train_losses[-1],
                'val_loss': val_losses[-1],
                'overfitting_degree': overfitting_degree
            }
            
            print(f"最终训练损失: {train_losses[-1]:.6f}")
            print(f"最终验证损失: {val_losses[-1]:.6f}")
            print(f"过拟合程度: {overfitting_degree:.3f}")
        
        return results
    
    reg_results = regularization_comprehensive_demo()
    
    # 6. 费曼学习法：通过可视化理解深度学习
    print("\n========== 费曼学习法：可视化理解深度学习 ==========")
    
    def visualize_neural_network_learning():
        """
        可视化神经网络的学习过程
        通过简单的2D分类问题展示决策边界的变化
        """
        # 生成螺旋数据
        def make_spiral_data(n_points=100):
            np.random.seed(0)
            N = n_points  # 每类点数
            D = 2  # 维度
            K = 3  # 类别数
            X = np.zeros((N*K, D))
            y = np.zeros(N*K, dtype='uint8')
            
            for j in range(K):
                ix = range(N*j, N*(j+1))
                r = np.linspace(0.0, 1, N)  # 半径
                t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # 角度
                X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
                y[ix] = j
            
            return torch.FloatTensor(X), torch.LongTensor(y)
        
        X, y = make_spiral_data(200)
        
        # 创建简单的神经网络
        class SpiralClassifier(nn.Module):
            def __init__(self, hidden_size=50):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(2, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, 3)
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = SpiralClassifier(100)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 训练过程记录
        epochs = [0, 50, 100, 200, 500]
        training_snapshots = {}
        
        print("训练螺旋分类器，记录决策边界变化：")
        
        for epoch in range(501):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if epoch in epochs:
                model.eval()
                with torch.no_grad():
                    train_acc = (outputs.argmax(dim=1) == y).float().mean().item()
                    training_snapshots[epoch] = {
                        'loss': loss.item(),
                        'accuracy': train_acc,
                        'model_state': model.state_dict().copy()
                    }
                    print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={train_acc:.3f}")
                model.train()
        
        return model, X, y, training_snapshots
    
    spiral_model, spiral_X, spiral_y, snapshots = visualize_neural_network_learning()
    
    # 7. 现代训练技巧综合应用
    print("\n========== 现代训练技巧：工程最佳实践 ==========")
    
    def modern_training_pipeline():
        """
        现代深度学习训练管道
        包含数据加载、模型定义、训练循环、验证、保存等完整流程
        """
        # 1. 数据准备
        print("1. 数据准备与预处理")
        
        # 生成大规模数据集
        n_samples = 10000
        n_features = 100
        n_classes = 10
        
        # 特征标准化的重要性
        X_raw = torch.randn(n_samples, n_features) * 10 + 5
        y = torch.randint(0, n_classes, (n_samples,))
        
        # 标准化
        X_mean = X_raw.mean(dim=0)
        X_std = X_raw.std(dim=0)
        X = (X_raw - X_mean) / (X_std + 1e-8)
        
        print(f"原始数据范围: [{X_raw.min().item():.2f}, {X_raw.max().item():.2f}]")
        print(f"标准化后范围: [{X.min().item():.2f}, {X.max().item():.2f}]")
        
        # 数据集划分
        indices = torch.randperm(n_samples)
        train_idx = indices[:8000]
        val_idx = indices[8000:9000]
        test_idx = indices[9000:]
        
        train_dataset = TensorDataset(X[train_idx], y[train_idx])
        val_dataset = TensorDataset(X[val_idx], y[val_idx])
        test_dataset = TensorDataset(X[test_idx], y[test_idx])
        
        # 数据加载器
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        # 2. 模型定义
        print("\n2. 现代神经网络架构设计")
        
        class ModernMLP(nn.Module):
            """现代MLP设计：批归一化+残差连接+自适应激活"""
            def __init__(self, input_dim, hidden_dims, output_dim):
                super().__init__()
                
                self.input_layer = nn.Sequential(
                    nn.Linear(input_dim, hidden_dims[0]),
                    nn.BatchNorm1d(hidden_dims[0]),
                    nn.ReLU()
                )
                
                self.hidden_layers = nn.ModuleList()
                for i in range(len(hidden_dims) - 1):
                    layer = nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                        nn.BatchNorm1d(hidden_dims[i+1]),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
                    self.hidden_layers.append(layer)
                
                self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
                
            def forward(self, x):
                x = self.input_layer(x)
                
                for layer in self.hidden_layers:
                    residual = x
                    x = layer(x)
                    # 简单残差连接（维度匹配时）
                    if x.shape == residual.shape:
                        x = x + residual
                
                x = self.output_layer(x)
                return x
        
        model = ModernMLP(n_features, [256, 128, 64], n_classes)
        
        # 3. 优化器和调度器
        print("\n3. 优化策略配置")
        
        optimizer = torch.optim.AdamW(model.parameters(), 
                                    lr=0.001, 
                                    weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-6
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # 4. 训练循环
        print("\n4. 训练过程监控")
        
        def train_epoch(model, train_loader, optimizer, criterion):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            return total_loss / len(train_loader), correct / total
        
        def validate_epoch(model, val_loader, criterion):
            model.eval()
            total_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    total_loss += loss.item()
                    predicted = outputs.argmax(dim=1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            return total_loss / len(val_loader), correct / total
        
        # 训练历史记录
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        # 早停机制
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print("开始训练...")
        start_time = time.time()
        
        for epoch in range(100):
            # 训练
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
            
            # 验证
            val_loss, val_acc = validate_epoch(model, val_loader, criterion)
            
            # 学习率调度
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
                      f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}, LR={current_lr:.6f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"早停触发，在第 {epoch} 轮停止训练")
                break
        
        training_time = time.time() - start_time
        print(f"训练完成，总用时: {training_time:.2f}秒")
        
        # 5. 最终测试
        print("\n5. 最终模型评估")
        
        # 加载最佳模型
        model.load_state_dict(torch.load('best_model.pth'))
        test_loss, test_acc = validate_epoch(model, test_loader, criterion)
        
        print(f"测试集性能: Loss={test_loss:.4f}, Accuracy={test_acc:.3f}")
        
        return model, history
    
    final_model, training_history = modern_training_pipeline()
    
    # 8. 模型分析与诊断
    print("\n========== 模型分析与诊断工具 ==========")
    
    def model_diagnostic_tools(model, history):
        """
        深度学习模型诊断工具
        分析训练过程、参数分布、梯度流动等
        """
        print("1. 训练曲线分析")
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        
        print(f"最终训练损失: {final_train_loss:.4f}")
        print(f"最终验证损失: {final_val_loss:.4f}")
        print(f"过拟合程度: {final_val_loss/final_train_loss:.3f}")
        print(f"准确率差距: {final_train_acc - final_val_acc:.3f}")
        
        # 判断训练状态
        if final_val_loss/final_train_loss > 1.5:
            print("诊断: 存在过拟合，建议增加正则化")
        elif final_train_acc < 0.7:
            print("诊断: 存在欠拟合，建议增加模型复杂度")
        else:
            print("诊断: 模型训练状态良好")
        
        print("\n2. 参数分布分析")
        for name, param in model.named_parameters():
            if param.requires_grad and 'weight' in name:
                weight_std = param.data.std().item()
                weight_mean = param.data.mean().item()
                print(f"{name}: 均值={weight_mean:.6f}, 标准差={weight_std:.6f}")
        
        print("\n3. 模型复杂度统计")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        # 计算模型大小
        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        print(f"模型大小: {model_size:.2f} MB")
        
    model_diagnostic_tools(final_model, training_history)
    
    # 9. 深度学习调试技巧
    print("\n========== 深度学习调试技巧 ==========")
    
    def debugging_techniques():
        """
        深度学习常见问题的调试方法
        提供系统性的问题诊断和解决方案
        """
        print("常见问题诊断清单：")
        
        debugging_checklist = {
            "损失不下降": [
                "检查学习率是否过大或过小",
                "验证数据预处理是否正确",
                "确认标签格式与损失函数匹配",
                "检查模型架构合理性"
            ],
            "过拟合严重": [
                "增加正则化（L2、Dropout）",
                "减少模型复杂度",
                "增加训练数据",
                "使用数据增强"
            ],
            "训练速度慢": [
                "增大批量大小",
                "使用更高效的优化器",
                "检查数据加载瓶颈",
                "考虑混合精度训练"
            ],
            "梯度消失/爆炸": [
                "使用批归一化",
                "调整初始化方法",
                "使用梯度裁剪",
                "考虑残差连接"
            ]
        }
        
        for problem, solutions in debugging_checklist.items():
            print(f"\n{problem}:")
            for i, solution in enumerate(solutions, 1):
                print(f"  {i}. {solution}")
        
        # 简单的损失函数验证
        print("\n实用调试工具示例：")
        
        def sanity_check_model(model, sample_input, expected_output_shape):
            """模型基本功能检查"""
            try:
                model.eval()
                with torch.no_grad():
                    output = model(sample_input)
                    
                print(f"✓ 前向传播成功")
                print(f"✓ 输出形状: {output.shape} (期望: {expected_output_shape})")
                
                if output.shape == expected_output_shape:
                    print("✓ 输出形状匹配")
                else:
                    print("✗ 输出形状不匹配")
                
                # 检查数值稳定性
                if torch.isnan(output).any():
                    print("✗ 输出包含NaN")
                elif torch.isinf(output).any():
                    print("✗ 输出包含Inf")
                else:
                    print("✓ 数值稳定")
                    
            except Exception as e:
                print(f"✗ 前向传播失败: {e}")
        
        # 测试当前模型
        sample_input = torch.randn(1, 100)
        sanity_check_model(final_model, sample_input, (1, 10))
    
    debugging_techniques()
    
    # 10. 总结与展望
    print("\n========== 深度学习基础总结与展望 ==========")
    print("""
    【核心技术要点】
    1. 梯度下降：深度学习优化的基础，理解参数更新机制
    2. 非线性激活：使网络具备复杂函数逼近能力
    3. 正则化技术：防止过拟合，提升模型泛化能力
    4. 批量处理：提升训练效率和稳定性
    5. 现代训练技巧：批归一化、残差连接、学习率调度
    
    【理论到实践的桥梁】
    - 数学原理：理解梯度、反向传播、损失函数的数学基础
    - 工程实现：掌握PyTorch的模块化设计和最佳实践
    - 调试技能：系统性的问题诊断和解决能力
    
    【未来学习路径】
    1. 专业领域：CNN(计算机视觉)、RNN(序列建模)、Transformer(注意力机制)
    2. 高级技术：生成模型、强化学习、元学习
    3. 工程优化：模型压缩、量化、分布式训练
    4. 部署应用：边缘计算、实时推理、A/B测试
    """)
    
    print("\n🎯 深度学习基础掌握建议：")
    print("1. 理论基础：掌握线性代数、概率论、优化理论")
    print("2. 编程实践：熟练使用PyTorch进行模型实现")
    print("3. 问题解决：培养系统性的调试和优化能力")
    print("4. 持续学习：跟踪最新研究进展和技术发展")
    print("5. 项目经验：在实际问题中应用和验证所学知识")

# 运行训练后模型预测示例
# 原理说明：
# 训练完成后，模型可以对新样本进行预测。通过softmax输出最大概率的类别作为最终预测结果。 