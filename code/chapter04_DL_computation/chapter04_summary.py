"""
PyTorch 深度学习计算核心技术与深度理解
-----------------------------------
【文件说明】
本文件系统梳理了深度学习计算的核心技术，包括：
- 模型构建与参数管理：面向对象的网络设计
- 自定义层与模块：扩展PyTorch的表达能力
- 参数初始化策略：影响训练稳定性的关键
- 模型保存与加载：支持断点续训和部署
- 设备管理与GPU加速：大规模训练的基础

【第一性原理思考】
1. 为什么需要模块化设计？
   - 复杂网络需要层次化组织
   - 模块化便于复用和维护
   - 面向对象的抽象降低开发复杂度

2. 参数初始化为什么重要？
   - 初始化影响梯度流动和收敛速度
   - 不当初始化导致梯度消失或爆炸
   - 合理初始化是训练成功的前提

3. GPU加速的本质是什么？
   - 深度学习计算具有高度并行性
   - GPU的大规模并行架构匹配计算需求
   - 内存带宽和计算吞吐量的协同优化

【苏格拉底式提问与验证】
1. 更复杂的模型总是更好吗？
   - 问题：模型复杂度与性能的关系？
   - 验证：通过参数量和性能的权衡分析
   - 结论：需要平衡表达能力和计算效率

2. 如何选择合适的初始化方法？
   - 问题：不同初始化对训练的影响？
   - 验证：通过对比实验观察收敛行为
   - 结论：根据激活函数和网络深度选择

【费曼学习法讲解】
1. 概念解释
   - 用搭积木类比模块化设计
   - 用种子发芽类比参数初始化
   - 强调计算优化在实际应用中的重要性

2. 实例教学
   - 从简单的线性层开始
   - 逐步构建复杂的网络结构
   - 通过性能测试理解优化效果

【设计意义与工程价值】
- 深度学习计算是AI系统的核心，直接影响训练效率和模型性能
- 理解计算原理对优化大规模模型至关重要
- 工程实践技巧是模型成功部署的关键

可运行案例：
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time
import os

if __name__ == "__main__":
    print("========== PyTorch深度学习计算核心技术演示 ==========")
    
    # 1. 第一性原理：模块化设计的必要性
    print("\n========== 第一性原理：模块化设计的力量 ==========")
    
    def demonstrate_modularity():
        """
        展示模块化设计的优势
        对比单体设计与模块化设计的差异
        """
        # 单体设计：难以维护和扩展
        class MonolithicMLP(nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                # 所有层都写在一起，难以复用
                self.layer1 = nn.Linear(input_dim, 256)
                self.layer2 = nn.Linear(256, 128)
                self.layer3 = nn.Linear(128, 64)
                self.layer4 = nn.Linear(64, output_dim)
                
            def forward(self, x):
                x = F.relu(self.layer1(x))
                x = F.relu(self.layer2(x))
                x = F.relu(self.layer3(x))
                x = self.layer4(x)
                return x
        
        # 模块化设计：可复用、可扩展
        class MLPBlock(nn.Module):
            """可复用的MLP块"""
            def __init__(self, input_dim, output_dim, activation=True, dropout_rate=0.0):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)
                self.activation = activation
                self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
                
            def forward(self, x):
                x = self.linear(x)
                if self.activation:
                    x = F.relu(x)
                if self.dropout:
                    x = self.dropout(x)
                return x
        
        class ModularMLP(nn.Module):
            """模块化设计的MLP"""
            def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.0):
                super().__init__()
                self.blocks = nn.ModuleList()
                
                # 构建隐藏层
                prev_dim = input_dim
                for hidden_dim in hidden_dims:
                    self.blocks.append(MLPBlock(prev_dim, hidden_dim, 
                                              activation=True, dropout_rate=dropout_rate))
                    prev_dim = hidden_dim
                
                # 输出层
                self.blocks.append(MLPBlock(prev_dim, output_dim, 
                                          activation=False, dropout_rate=0))
            
            def forward(self, x):
                for block in self.blocks:
                    x = block(x)
                return x
        
        # 性能和灵活性对比
        input_dim, output_dim = 784, 10
        
        # 单体模型
        monolithic = MonolithicMLP(input_dim, output_dim)
        monolithic_params = sum(p.numel() for p in monolithic.parameters())
        
        # 模块化模型（相同结构）
        modular = ModularMLP(input_dim, [256, 128, 64], output_dim)
        modular_params = sum(p.numel() for p in modular.parameters())
        
        print(f"单体模型参数量: {monolithic_params:,}")
        print(f"模块化模型参数量: {modular_params:,}")
        print(f"参数量是否相同: {monolithic_params == modular_params}")
        
        # 模块化的优势：可以轻松修改结构
        flexible_model = ModularMLP(input_dim, [512, 256, 128, 64, 32], output_dim, dropout_rate=0.1)
        flexible_params = sum(p.numel() for p in flexible_model.parameters())
        print(f"灵活模型参数量: {flexible_params:,}")
        
        # 测试前向传播
        x = torch.randn(32, input_dim)
        out1 = monolithic(x)
        out2 = modular(x)
        out3 = flexible_model(x)
        
        print(f"单体模型输出形状: {out1.shape}")
        print(f"模块化模型输出形状: {out2.shape}")
        print(f"灵活模型输出形状: {out3.shape}")
        
        return monolithic, modular, flexible_model
    
    mono_model, mod_model, flex_model = demonstrate_modularity()
    
    # 2. 深度解析：参数初始化的科学与艺术
    print("\n========== 深度解析：参数初始化策略 ==========")
    
    def parameter_initialization_analysis():
        """
        系统分析不同初始化方法的影响
        理解初始化与激活函数、网络深度的关系
        """
        # 不同初始化方法
        def create_network_with_init(init_method, depth=5):
            layers = []
            for i in range(depth):
                layer = nn.Linear(100, 100)
                
                # 应用不同的初始化方法
                if init_method == 'zero':
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
                elif init_method == 'normal':
                    nn.init.normal_(layer.weight, mean=0, std=1.0)
                    nn.init.constant_(layer.bias, 0)
                elif init_method == 'small_normal':
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)
                elif init_method == 'xavier':
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
                elif init_method == 'kaiming':
                    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0)
                
                layers.extend([layer, nn.ReLU()])
            
            # 输出层
            output_layer = nn.Linear(100, 10)
            nn.init.xavier_uniform_(output_layer.weight)
            layers.append(output_layer)
            
            return nn.Sequential(*layers)
        
        # 测试不同初始化方法
        init_methods = ['zero', 'normal', 'small_normal', 'xavier', 'kaiming']
        results = {}
        
        print("初始化方法对比分析：")
        
        for method in init_methods:
            network = create_network_with_init(method)
            
            # 分析参数分布
            weights = []
            for param in network.parameters():
                if param.dim() > 1:  # 只看权重，不看偏置
                    weights.extend(param.data.flatten().tolist())
            
            weights = np.array(weights)
            weight_mean = weights.mean()
            weight_std = weights.std()
            weight_range = weights.max() - weights.min()
            
            # 测试梯度流动
            x = torch.randn(32, 100)
            y = network(x)
            loss = y.sum()
            loss.backward()
            
            # 计算梯度统计
            gradients = []
            for param in network.parameters():
                if param.grad is not None:
                    gradients.extend(param.grad.data.flatten().tolist())
            
            gradients = np.array(gradients)
            grad_mean = gradients.mean()
            grad_std = gradients.std()
            
            results[method] = {
                'weight_mean': weight_mean,
                'weight_std': weight_std,
                'weight_range': weight_range,
                'grad_mean': grad_mean,
                'grad_std': grad_std
            }
            
            print(f"{method:12}: 权重标准差={weight_std:.6f}, 梯度标准差={grad_std:.6f}")
        
        # 分析哪种初始化最好
        print("\n初始化方法分析：")
        print("- zero: 权重全为0，导致对称性问题，梯度消失")
        print("- normal: 标准正态分布，深层网络容易梯度爆炸")
        print("- small_normal: 小标准差，有助于稳定训练")
        print("- xavier: 考虑输入输出维度，适合tanh/sigmoid")
        print("- kaiming: 针对ReLU优化，深层网络首选")
        
        return results
    
    init_results = parameter_initialization_analysis()
    
    # 3. 苏格拉底式提问：自定义层的设计哲学
    print("\n========== 苏格拉底式提问：如何设计自定义层？ ==========")
    
    def custom_layer_design_philosophy():
        """
        探讨自定义层设计的关键考虑因素
        通过实例展示不同设计选择的影响
        """
        # 问题1：自定义层应该封装什么？
        print("问题1：自定义层应该封装什么？")
        
        # 方案A：只封装计算逻辑
        class SimpleAttention(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim
                
            def forward(self, query, key, value):
                # 简单的点积注意力
                scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.dim)
                weights = F.softmax(scores, dim=-1)
                output = torch.matmul(weights, value)
                return output, weights
        
        # 方案B：封装完整的可学习模块
        class LearnableAttention(nn.Module):
            def __init__(self, dim, head_dim=None):
                super().__init__()
                self.dim = dim
                self.head_dim = head_dim or dim
                
                # 可学习的投影层
                self.query_proj = nn.Linear(dim, self.head_dim)
                self.key_proj = nn.Linear(dim, self.head_dim)
                self.value_proj = nn.Linear(dim, self.head_dim)
                self.output_proj = nn.Linear(self.head_dim, dim)
                
            def forward(self, x):
                batch_size, seq_len, _ = x.shape
                
                # 投影到查询、键、值
                query = self.query_proj(x)
                key = self.key_proj(x)
                value = self.value_proj(x)
                
                # 计算注意力
                scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(self.head_dim)
                weights = F.softmax(scores, dim=-1)
                attended = torch.matmul(weights, value)
                
                # 输出投影
                output = self.output_proj(attended)
                return output, weights
        
        # 测试两种设计
        batch_size, seq_len, dim = 4, 10, 64
        x = torch.randn(batch_size, seq_len, dim)
        
        simple_attn = SimpleAttention(dim)
        learnable_attn = LearnableAttention(dim)
        
        # 简单注意力需要手动准备查询、键、值
        out1, weights1 = simple_attn(x, x, x)
        
        # 可学习注意力内部处理
        out2, weights2 = learnable_attn(x)
        
        print(f"简单注意力输出形状: {out1.shape}")
        print(f"可学习注意力输出形状: {out2.shape}")
        print(f"可学习注意力参数量: {sum(p.numel() for p in learnable_attn.parameters())}")
        
        # 问题2：如何处理可变输入？
        print("\n问题2：如何处理可变输入？")
        
        class AdaptiveLayer(nn.Module):
            """自适应不同输入尺寸的层"""
            def __init__(self, output_dim):
                super().__init__()
                self.output_dim = output_dim
                self.projections = nn.ModuleDict()
                
            def forward(self, x):
                input_dim = x.size(-1)
                
                # 动态创建投影层
                if str(input_dim) not in self.projections:
                    self.projections[str(input_dim)] = nn.Linear(input_dim, self.output_dim)
                
                projection = self.projections[str(input_dim)]
                return projection(x)
        
        adaptive_layer = AdaptiveLayer(32)
        
        # 测试不同输入尺寸
        x1 = torch.randn(4, 64)  # 64维输入
        x2 = torch.randn(4, 128)  # 128维输入
        
        out1 = adaptive_layer(x1)
        out2 = adaptive_layer(x2)
        
        print(f"64维输入→32维输出: {x1.shape} → {out1.shape}")
        print(f"128维输入→32维输出: {x2.shape} → {out2.shape}")
        print(f"动态创建的投影层数量: {len(adaptive_layer.projections)}")
        
        return simple_attn, learnable_attn, adaptive_layer
    
    simple_att, learn_att, adapt_layer = custom_layer_design_philosophy()
    
    # 4. 费曼学习法：通过类比理解GPU加速
    print("\n========== 费曼学习法：GPU加速的本质 ==========")
    
    def gpu_acceleration_explained():
        """
        通过类比和实验理解GPU加速的原理
        对比CPU和GPU的计算特性
        """
        print("GPU加速类比：")
        print("- CPU像一个聪明的工程师，能处理复杂逻辑，但只有几个人")
        print("- GPU像一个庞大的工厂，有成千上万个简单工人，擅长重复性工作")
        print("- 深度学习的矩阵运算正好匹配GPU的并行特性")
        
        # 检查GPU可用性
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n当前设备: {device}")
        
        if torch.cuda.is_available():
            print(f"GPU型号: {torch.cuda.get_device_name()}")
            print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # 性能对比实验
        def benchmark_computation(size=2048, device='cpu'):
            """测试不同设备的计算性能"""
            # 创建大型矩阵
            a = torch.randn(size, size, device=device)
            b = torch.randn(size, size, device=device)
            
            # 预热（第一次运行可能较慢）
            _ = torch.matmul(a, b)
            
            # 计时矩阵乘法
            start_time = time.time()
            for _ in range(10):
                result = torch.matmul(a, b)
            
            # 等待GPU计算完成
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 10
            
            # 计算吞吐量
            ops = 2 * size**3  # 矩阵乘法的运算次数
            throughput = ops / avg_time / 1e9  # GFLOPS
            
            return avg_time, throughput
        
        # CPU性能测试
        print("\n性能测试：")
        cpu_time, cpu_throughput = benchmark_computation(1024, torch.device('cpu'))
        print(f"CPU (1024x1024): {cpu_time:.4f}秒, {cpu_throughput:.2f} GFLOPS")
        
        # GPU性能测试（如果可用）
        if torch.cuda.is_available():
            gpu_time, gpu_throughput = benchmark_computation(1024, device)
            speedup = cpu_time / gpu_time
            print(f"GPU (1024x1024): {gpu_time:.4f}秒, {gpu_throughput:.2f} GFLOPS")
            print(f"加速比: {speedup:.2f}x")
            
            # 测试更大的矩阵（GPU优势更明显）
            large_gpu_time, large_gpu_throughput = benchmark_computation(4096, device)
            print(f"GPU (4096x4096): {large_gpu_time:.4f}秒, {large_gpu_throughput:.2f} GFLOPS")
        
        # 内存管理演示
        print("\n内存管理演示：")
        
        def demonstrate_memory_management():
            """演示GPU内存管理的重要性"""
            if not torch.cuda.is_available():
                print("GPU不可用，跳过内存管理演示")
                return
            
            # 清空GPU缓存
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            print(f"初始GPU内存使用: {initial_memory / 1e6:.1f} MB")
            
            # 创建大型张量
            large_tensors = []
            for i in range(5):
                tensor = torch.randn(1000, 1000, device='cuda')
                large_tensors.append(tensor)
                current_memory = torch.cuda.memory_allocated()
                print(f"创建张量{i+1}: {current_memory / 1e6:.1f} MB")
            
            # 手动释放内存
            del large_tensors
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            print(f"释放后GPU内存: {final_memory / 1e6:.1f} MB")
        
        demonstrate_memory_management()
        
        return cpu_time, gpu_time if torch.cuda.is_available() else None
    
    cpu_perf, gpu_perf = gpu_acceleration_explained()
    
    # 5. 模型保存与加载：持久化的艺术
    print("\n========== 模型保存与加载：持久化策略 ==========")
    
    def model_persistence_strategies():
        """
        演示不同的模型保存和加载策略
        探讨各种方法的优缺点和适用场景
        """
        # 创建一个示例模型
        class ExampleModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                )
                self.classifier = nn.Linear(hidden_dim, output_dim)
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.output_dim = output_dim
                
            def forward(self, x):
                features = self.backbone(x)
                output = self.classifier(features)
                return output
        
        # 创建并训练模型
        model = ExampleModel(784, 256, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 模拟训练
        x = torch.randn(32, 784)
        y = torch.randint(0, 10, (32,))
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        print(f"训练完成，最终损失: {loss.item():.4f}")
        
        # 策略1：只保存模型参数（推荐）
        print("\n策略1：保存模型参数（state_dict）")
        torch.save(model.state_dict(), 'model_params.pth')
        
        # 加载参数到新模型
        new_model = ExampleModel(784, 256, 10)
        new_model.load_state_dict(torch.load('model_params.pth'))
        
        # 验证加载成功
        new_output = new_model(x)
        params_match = torch.allclose(model(x), new_output)
        print(f"参数加载验证: {'成功' if params_match else '失败'}")
        
        # 策略2：保存完整模型（不推荐，但有时必要）
        print("\n策略2：保存完整模型")
        try:
            torch.save(model, 'complete_model.pth')
            loaded_model = torch.load('complete_model.pth')
            
            complete_output = loaded_model(x)
            complete_match = torch.allclose(model(x), complete_output)
            print(f"完整模型加载验证: {'成功' if complete_match else '失败'}")
        except Exception as e:
            print(f"完整模型保存失败（这是正常的，因为模型类在函数内部定义）: {type(e).__name__}")
            print("推荐使用 state_dict 方式保存模型参数")
        
        # 策略3：保存训练状态（断点续训）
        print("\n策略3：保存训练状态")
        checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'model_config': {
                'input_dim': 784,
                'hidden_dim': 256,
                'output_dim': 10
            }
        }
        torch.save(checkpoint, 'checkpoint.pth')
        
        # 从检查点恢复训练
        checkpoint = torch.load('checkpoint.pth')
        resume_model = ExampleModel(**checkpoint['model_config'])
        resume_optimizer = torch.optim.Adam(resume_model.parameters(), lr=0.001)
        
        resume_model.load_state_dict(checkpoint['model_state_dict'])
        resume_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        
        print(f"从第{start_epoch}轮恢复训练")
        
        # 策略4：模型版本管理
        print("\n策略4：模型版本管理")
        
        class ModelManager:
            """模型版本管理器"""
            def __init__(self, base_path='models'):
                self.base_path = base_path
                os.makedirs(base_path, exist_ok=True)
            
            def save_model(self, model, name, version, metadata=None):
                """保存带版本信息的模型"""
                model_data = {
                    'state_dict': model.state_dict(),
                    'model_class': model.__class__.__name__,
                    'version': version,
                    'metadata': metadata or {}
                }
                
                filename = f"{name}_v{version}.pth"
                filepath = os.path.join(self.base_path, filename)
                torch.save(model_data, filepath)
                print(f"模型已保存: {filepath}")
                
            def load_model(self, model_class, name, version):
                """加载指定版本的模型"""
                filename = f"{name}_v{version}.pth"
                filepath = os.path.join(self.base_path, filename)
                
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"模型文件不存在: {filepath}")
                
                model_data = torch.load(filepath)
                print(f"加载模型: {filepath}, 版本: {model_data['version']}")
                return model_data
        
        # 使用模型管理器
        manager = ModelManager()
        metadata = {
            'accuracy': 0.95,
            'training_time': '10min',
            'dataset': 'MNIST'
        }
        manager.save_model(model, 'mnist_classifier', '1.0', metadata)
        
        # 清理临时文件
        for temp_file in ['model_params.pth', 'complete_model.pth', 'checkpoint.pth']:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return model, manager
    
    saved_model, model_mgr = model_persistence_strategies()
    
    # 6. 高级计算模式：混合精度与分布式训练
    print("\n========== 高级计算模式：现代训练技术 ==========")
    
    def advanced_computation_modes():
        """
        演示混合精度训练和分布式计算的概念
        展示现代深度学习训练的前沿技术
        """
        print("1. 混合精度训练（Automatic Mixed Precision）")
        
        # 创建示例模型
        model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100)
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            
            # 传统FP32训练
            optimizer = torch.optim.Adam(model.parameters())
            x = torch.randn(64, 1000, device='cuda')
            y = torch.randn(64, 100, device='cuda')
            
            # 测试FP32性能
            start_time = time.time()
            for _ in range(100):
                optimizer.zero_grad()
                output = model(x)
                loss = F.mse_loss(output, y)
                loss.backward()
                optimizer.step()
            
            torch.cuda.synchronize()
            fp32_time = time.time() - start_time
            
            print(f"FP32训练时间: {fp32_time:.3f}秒")
            
            # 混合精度训练（需要较新的GPU）
            try:
                from torch.cuda.amp import autocast, GradScaler
                
                model = model.half()  # 转换为FP16
                optimizer = torch.optim.Adam(model.parameters())
                scaler = GradScaler()
                
                start_time = time.time()
                for _ in range(100):
                    optimizer.zero_grad()
                    
                    with autocast():
                        output = model(x.half())
                        loss = F.mse_loss(output, y.half())
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                torch.cuda.synchronize()
                fp16_time = time.time() - start_time
                speedup = fp32_time / fp16_time
                
                print(f"FP16训练时间: {fp16_time:.3f}秒")
                print(f"混合精度加速比: {speedup:.2f}x")
                
            except ImportError:
                print("当前PyTorch版本不支持自动混合精度")
        
        else:
            print("GPU不可用，跳过混合精度演示")
        
        print("\n2. 分布式训练概念")
        print("分布式训练的核心思想：")
        print("- 数据并行：将数据分割到多个设备")
        print("- 模型并行：将模型分割到多个设备")
        print("- 梯度同步：确保所有设备的参数一致")
        
        # 模拟数据并行的概念
        def simulate_data_parallel():
            """模拟数据并行训练的过程"""
            print("\n数据并行模拟：")
            
            # 假设有4个GPU
            num_gpus = 4
            total_batch_size = 128
            per_gpu_batch_size = total_batch_size // num_gpus
            
            print(f"总批量大小: {total_batch_size}")
            print(f"每GPU批量大小: {per_gpu_batch_size}")
            
            # 模拟梯度聚合
            gradients = []
            for gpu_id in range(num_gpus):
                # 每个GPU计算本地梯度
                local_grad = torch.randn(100, 10)  # 模拟梯度
                gradients.append(local_grad)
                print(f"GPU {gpu_id} 梯度范数: {local_grad.norm().item():.4f}")
            
            # 平均梯度
            avg_gradient = torch.stack(gradients).mean(dim=0)
            print(f"平均梯度范数: {avg_gradient.norm().item():.4f}")
            
            return avg_gradient
        
        avg_grad = simulate_data_parallel()
        
        print("\n3. 计算图优化")
        print("PyTorch的动态图 vs 静态图优化：")
        
        # 动态图示例
        def dynamic_computation():
            x = torch.randn(10, requires_grad=True)
            y = x ** 2
            
            # 根据条件动态改变计算图
            if x.sum() > 0:
                y = y + torch.sin(x)
            else:
                y = y + torch.cos(x)
            
            return y.sum()
        
        # TorchScript优化
        def static_computation(x):
            y = x ** 2
            return (y + torch.sin(x)).sum()
        
        # 转换为TorchScript
        try:
            scripted_func = torch.jit.script(static_computation)
            x = torch.randn(1000)
            
            # 性能对比
            start_time = time.time()
            for _ in range(1000):
                result1 = dynamic_computation()
            dynamic_time = time.time() - start_time
            
            start_time = time.time()
            for _ in range(1000):
                result2 = scripted_func(x)
            scripted_time = time.time() - start_time
            
            print(f"动态图执行时间: {dynamic_time:.4f}秒")
            print(f"TorchScript时间: {scripted_time:.4f}秒")
            print(f"TorchScript加速比: {dynamic_time/scripted_time:.2f}x")
            
        except Exception as e:
            print(f"TorchScript优化失败: {e}")
    
    advanced_computation_modes()
    
    # 7. 计算效率优化实战
    print("\n========== 计算效率优化实战 ==========")
    
    def computation_optimization_practices():
        """
        实战演示各种计算效率优化技巧
        提供可直接应用的优化方法
        """
        print("1. 内存优化技巧")
        
        # 技巧1：就地操作
        def compare_inplace_operations():
            x = torch.randn(1000, 1000)
            
            # 非就地操作
            start_time = time.time()
            for _ in range(100):
                y = torch.relu(x)
                y = y + 1
                y = y * 2
            regular_time = time.time() - start_time
            
            # 就地操作
            start_time = time.time()
            for _ in range(100):
                x_copy = x.clone()
                x_copy.relu_()
                x_copy.add_(1)
                x_copy.mul_(2)
            inplace_time = time.time() - start_time
            
            print(f"常规操作时间: {regular_time:.4f}秒")
            print(f"就地操作时间: {inplace_time:.4f}秒")
            print(f"就地操作加速比: {regular_time/inplace_time:.2f}x")
        
        compare_inplace_operations()
        
        # 技巧2：批量操作优化
        print("\n2. 批量操作优化")
        
        def optimize_batch_operations():
            # 低效：逐个处理
            data = [torch.randn(100) for _ in range(1000)]
            
            start_time = time.time()
            results = []
            for item in data:
                result = torch.softmax(item, dim=0)
                results.append(result)
            sequential_time = time.time() - start_time
            
            # 高效：批量处理
            batch_data = torch.stack(data)
            
            start_time = time.time()
            batch_result = torch.softmax(batch_data, dim=1)
            batch_time = time.time() - start_time
            
            print(f"逐个处理时间: {sequential_time:.4f}秒")
            print(f"批量处理时间: {batch_time:.4f}秒")
            print(f"批量处理加速比: {sequential_time/batch_time:.2f}x")
        
        optimize_batch_operations()
        
        # 技巧3：数据类型优化
        print("\n3. 数据类型优化")
        
        def optimize_data_types():
            size = (1000, 1000)
            
            # FP64
            x_fp64 = torch.randn(size, dtype=torch.float64)
            fp64_memory = x_fp64.element_size() * x_fp64.numel()
            
            # FP32
            x_fp32 = torch.randn(size, dtype=torch.float32)
            fp32_memory = x_fp32.element_size() * x_fp32.numel()
            
            # FP16
            x_fp16 = torch.randn(size, dtype=torch.float16)
            fp16_memory = x_fp16.element_size() * x_fp16.numel()
            
            print(f"FP64内存占用: {fp64_memory / 1e6:.1f} MB")
            print(f"FP32内存占用: {fp32_memory / 1e6:.1f} MB")
            print(f"FP16内存占用: {fp16_memory / 1e6:.1f} MB")
            print(f"FP16相对FP32节省: {(1 - fp16_memory/fp32_memory)*100:.1f}%")
        
        optimize_data_types()
        
        # 技巧4：计算图优化
        print("\n4. 计算图优化")
        
        def optimize_computation_graph():
            # 低效：重复计算
            def inefficient_forward(x):
                y1 = torch.matmul(x, x.t())
                y2 = torch.matmul(x, x.t())  # 重复计算
                return y1 + y2
            
            # 高效：复用计算结果
            def efficient_forward(x):
                y = torch.matmul(x, x.t())
                return y + y  # 复用结果
            
            x = torch.randn(500, 500)
            
            # 性能对比
            start_time = time.time()
            for _ in range(100):
                result1 = inefficient_forward(x)
            inefficient_time = time.time() - start_time
            
            start_time = time.time()
            for _ in range(100):
                result2 = efficient_forward(x)
            efficient_time = time.time() - start_time
            
            print(f"低效计算时间: {inefficient_time:.4f}秒")
            print(f"优化计算时间: {efficient_time:.4f}秒")
            print(f"优化后加速比: {inefficient_time/efficient_time:.2f}x")
        
        optimize_computation_graph()
    
    computation_optimization_practices()
    
    # 8. 总结与最佳实践
    print("\n========== 深度学习计算总结与最佳实践 ==========")
    print("""
    【核心技术要点】
    1. 模块化设计：面向对象的网络构建，提升代码复用性和可维护性
    2. 参数初始化：根据激活函数和网络深度选择合适的初始化策略
    3. 自定义层：封装复用逻辑，扩展PyTorch的表达能力
    4. 设备管理：合理利用GPU加速，注意内存管理
    5. 模型持久化：选择合适的保存策略，支持断点续训和版本管理
    
    【性能优化策略】
    1. 计算优化：批量处理、就地操作、避免重复计算
    2. 内存优化：合适的数据类型、梯度累积、内存复用
    3. 并行化：数据并行、模型并行、混合精度训练
    4. 图优化：TorchScript、算子融合、编译优化
    
    【工程最佳实践】
    1. 代码组织：模块化设计、配置管理、日志记录
    2. 调试技巧：梯度检查、数值稳定性、性能分析
    3. 部署优化：模型量化、动态图转静态图、推理优化
    4. 团队协作：代码规范、版本控制、文档管理
    
    【未来发展趋势】
    1. 编译器优化：图级别优化、算子生成、硬件适配
    2. 硬件加速：专用芯片、内存层次优化、通信优化
    3. 分布式计算：大规模训练、联邦学习、边缘计算
    4. 自动化工具：超参数优化、架构搜索、性能调优
    """)
    
    print("\n🎯 深度学习计算掌握建议：")
    print("1. 理论基础：掌握计算图、自动微分、并行计算原理")
    print("2. 实践技能：熟练使用PyTorch进行模型开发和优化")
    print("3. 性能意识：关注计算效率、内存使用、训练速度")
    print("4. 工程素养：代码质量、可维护性、可扩展性")
    print("5. 持续学习：跟踪最新技术发展和优化方法") 