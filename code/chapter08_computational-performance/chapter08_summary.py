"""
PyTorch 计算性能优化核心技术与实践
---------------------------------
【文件说明】
本文件系统梳理了深度学习计算性能优化的核心技术，包括：
- 混合编程（Hybridize）：结合命令式与符号式编程的优势
- 自动并行化（Auto Parallelism）：智能分配计算资源
- 多GPU训练：数据并行与模型并行策略
- 内存优化与计算图优化：提升训练效率

【第一性原理思考】
1. 为什么需要性能优化？
   - 深度学习模型参数量大（百万到千亿级）
   - 训练数据量大（TB级别），计算密集
   - 现代硬件并行能力强，需充分利用

2. 性能瓶颈在哪里？
   - 内存带宽限制：GPU显存访问速度是瓶颈
   - 计算资源利用率：单卡利用率常低于50%
   - 通信开销：多卡训练时设备间通信耗时

3. 优化策略的本质是什么？
   - 减少内存访问：通过融合操作、内存复用
   - 提高并行度：多设备协同计算
   - 优化计算图：消除冗余操作，预编译优化

【苏格拉底式提问与验证】
1. 混合编程如何平衡灵活性与性能？
   - 问题：命令式编程灵活但慢，符号式编程快但难调试
   - 验证：通过实际测试比较不同编程模式的性能
   - 结论：混合编程在开发期保持灵活，部署期获得性能

2. 多GPU训练的通信开销如何优化？
   - 问题：梯度同步是否成为性能瓶颈？
   - 验证：测量单卡vs多卡的加速比
   - 结论：需要平衡计算与通信，选择合适的并行策略

【费曼学习法讲解】
1. 概念解释
   - 用生产流水线类比计算图优化
   - 用团队协作类比多GPU并行
   - 强调性能优化在生产环境中的重要性

2. 实例教学
   - 从简单模型开始展示优化效果
   - 通过性能对比理解优化价值
   - 实践大规模训练的最佳实践

【设计意义与工程价值】
- 性能优化不仅关乎训练时间，更影响实验迭代速度和模型部署效果
- 掌握这些技术是从研究到工程落地的关键能力
- 在资源受限的环境下，性能优化决定了项目的可行性

可运行案例：
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import numpy as np

# 1. 性能基准测试工具
# -------------------
# 原理说明：
# 性能测试是优化的基础，需要准确测量时间、内存使用等指标
# 通过对比测试验证优化效果的有效性

class PerformanceBenchmark:
    """性能基准测试工具类"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def time_function(self, func, *args, warmup=3, repeat=10, **kwargs):
        """测量函数执行时间"""
        # 预热GPU
        for _ in range(warmup):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            func(*args, **kwargs)
            
        if self.device == 'cuda':
            torch.cuda.synchronize()
            
        start_time = time.time()
        for _ in range(repeat):
            result = func(*args, **kwargs)
            if self.device == 'cuda':
                torch.cuda.synchronize()
        end_time = time.time()
        
        avg_time = (end_time - start_time) / repeat
        return avg_time, result
    
    def memory_usage(self):
        """获取GPU内存使用情况"""
        if self.device == 'cuda':
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'reserved': torch.cuda.memory_reserved() / 1024**2,    # MB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**2
            }
        return {'cpu_memory': 'N/A'}

# 2. 混合编程优化（Hybridize）
# ---------------------------
# 原理说明：
# 混合编程结合了命令式编程的灵活性和符号式编程的性能优势
# 通过JIT编译将动态图转换为静态图，获得更好的优化机会

print("2. 混合编程优化示例：")

class HybridModel(nn.Module):
    """支持混合编程的模型"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# 创建模型和测试数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HybridModel(784, 256, 10).to(device)
x = torch.randn(1000, 784).to(device)

benchmark = PerformanceBenchmark(device)

# 测试常规推理性能
def regular_inference():
    with torch.no_grad():
        return model(x)

# 测试JIT编译后的性能
model_jit = torch.jit.script(model)

def jit_inference():
    with torch.no_grad():
        return model_jit(x)

print("性能对比测试：")
regular_time, _ = benchmark.time_function(regular_inference)
jit_time, _ = benchmark.time_function(jit_inference)

print(f"常规推理时间: {regular_time*1000:.2f}ms")
print(f"JIT推理时间: {jit_time*1000:.2f}ms")
print(f"加速比: {regular_time/jit_time:.2f}x")

# 3. 内存优化技术
# ---------------
# 原理说明：
# 内存优化通过减少内存分配、复用内存等手段降低内存使用
# 包括梯度累积、混合精度训练等技术

print("\n3. 内存优化技术：")

class MemoryEfficientTraining:
    """内存高效训练工具"""
    
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    def gradient_accumulation_step(self, data_loader, accumulation_steps=4):
        """梯度累积训练"""
        self.model.train()
        total_loss = 0
        
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(self.device), y.to(self.device)
            
            # 使用混合精度训练
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x)
                    loss = F.cross_entropy(outputs, y) / accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(x)
                loss = F.cross_entropy(outputs, y) / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item()
            
            if i >= 10:  # 仅演示几个batch
                break
                
        return total_loss

# 演示内存优化训练
model = HybridModel(784, 256, 10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 创建模拟数据
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

memory_trainer = MemoryEfficientTraining(model, optimizer, device)

print("内存使用情况（训练前）:")
print(benchmark.memory_usage())

loss = memory_trainer.gradient_accumulation_step(dataloader)
print(f"训练损失: {loss:.4f}")

print("内存使用情况（训练后）:")
print(benchmark.memory_usage())

# 4. 数据并行训练
# ---------------
# 原理说明：
# 数据并行将不同的数据batch分配到不同GPU上同时训练
# 每个GPU计算梯度后进行全局梯度同步

print("\n4. 数据并行训练：")

class DataParallelTrainer:
    """数据并行训练器"""
    
    def __init__(self, model, device_ids=None):
        if device_ids is None and torch.cuda.device_count() > 1:
            device_ids = list(range(torch.cuda.device_count()))
        
        self.device_ids = device_ids
        self.device = f'cuda:{device_ids[0]}' if device_ids else 'cpu'
        
        if device_ids and len(device_ids) > 1:
            self.model = nn.DataParallel(model, device_ids=device_ids)
            print(f"使用数据并行，GPU数量: {len(device_ids)}")
        else:
            self.model = model
            print("使用单GPU训练")
            
        self.model = self.model.to(self.device)
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for i, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i >= 5:  # 仅演示几个batch
                break
                
        return total_loss / min(len(dataloader), 5)

# 创建并测试数据并行训练
if torch.cuda.device_count() > 1:
    model_parallel = HybridModel(784, 256, 10)
    parallel_trainer = DataParallelTrainer(model_parallel)
    
    optimizer = torch.optim.Adam(parallel_trainer.model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 测试训练时间
    start_time = time.time()
    avg_loss = parallel_trainer.train_epoch(dataloader, optimizer, criterion)
    end_time = time.time()
    
    print(f"数据并行训练 - 平均损失: {avg_loss:.4f}")
    print(f"训练时间: {end_time - start_time:.2f}s")
else:
    print("仅有一个GPU，跳过数据并行演示")

# 5. 分布式训练基础
# -----------------
# 原理说明：
# 分布式训练通过多个进程、多台机器协同训练
# 使用NCCL等通信库实现高效的梯度同步

def setup_distributed(rank, world_size):
    """设置分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", 
                           rank=rank, world_size=world_size)

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()

def distributed_training_worker(rank, world_size, model_class, train_data):
    """分布式训练工作进程"""
    try:
        setup_distributed(rank, world_size)
        
        # 设置设备
        device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型并包装为DDP
        model = model_class(784, 256, 10).to(device)
        ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
        
        # 创建数据加载器（每个进程加载不同数据）
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_data, num_replicas=world_size, rank=rank
        )
        dataloader = DataLoader(train_data, batch_size=32, sampler=sampler)
        
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # 训练一个epoch
        ddp_model.train()
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = ddp_model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            if i >= 3:  # 仅演示几个batch
                break
                
        print(f"Rank {rank} 训练完成")
        
    except Exception as e:
        print(f"Rank {rank} 训练出错: {e}")
    finally:
        cleanup_distributed()

# 6. 模型并行示例
# ---------------
# 原理说明：
# 模型并行将模型的不同部分放置在不同GPU上
# 适用于单个GPU装不下的大模型

print("\n6. 模型并行示例：")

class ModelParallelNet(nn.Module):
    """模型并行网络"""
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # 假设有两个GPU可用
        self.device1 = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device2 = 'cuda:1' if torch.cuda.device_count() > 1 else self.device1
        
        # 将模型前半部分放在第一个设备
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ).to(self.device1)
        
        # 将模型后半部分放在第二个设备
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        ).to(self.device2)
    
    def forward(self, x):
        # 在第一个设备上计算
        x = x.to(self.device1)
        x = self.layer1(x)
        
        # 移动到第二个设备继续计算
        x = x.to(self.device2)
        x = self.layer2(x)
        
        return x

if torch.cuda.device_count() > 1:
    print(f"使用模型并行，GPU数量: {torch.cuda.device_count()}")
    model_parallel = ModelParallelNet(784, 512, 10)
    
    # 测试前向传播
    x = torch.randn(64, 784)
    start_time = time.time()
    output = model_parallel(x)
    end_time = time.time()
    
    print(f"模型并行推理时间: {end_time - start_time:.4f}s")
    print(f"输出形状: {output.shape}")
else:
    print("需要多个GPU才能演示模型并行")

# 7. 性能优化最佳实践
# -------------------
print("\n7. 性能优化最佳实践：")

class OptimizedTrainingPipeline:
    """优化的训练管道"""
    
    def __init__(self, model, device, use_amp=True, use_compile=True):
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        
        # 尝试使用torch.compile (PyTorch 2.0+)
        if use_compile and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(model.to(device))
                print("使用torch.compile优化")
            except:
                self.model = model.to(device)
                print("torch.compile不可用，使用原始模型")
        else:
            self.model = model.to(device)
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            print("启用混合精度训练")
    
    def optimized_step(self, x, y, optimizer, criterion):
        """优化的训练步骤"""
        x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        
        if self.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(x)
                loss = criterion(outputs, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            outputs = self.model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        return loss.item()

# 创建优化的训练管道
optimized_model = HybridModel(784, 256, 10)
pipeline = OptimizedTrainingPipeline(optimized_model, device)

optimizer = torch.optim.Adam(pipeline.model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 测试优化效果
print("\n测试优化效果：")
x_test = torch.randn(100, 784)
y_test = torch.randint(0, 10, (100,))

start_time = time.time()
for i in range(10):
    optimizer.zero_grad()
    loss = pipeline.optimized_step(x_test, y_test, optimizer, criterion)
end_time = time.time()

print(f"优化训练10步耗时: {end_time - start_time:.4f}s")
print(f"平均每步耗时: {(end_time - start_time) / 10 * 1000:.2f}ms")

# 8. 性能分析工具
# ---------------
print("\n8. 性能分析工具：")

def profile_model_performance(model, input_data, device):
    """分析模型性能"""
    model = model.to(device)
    input_data = input_data.to(device)
    
    # PyTorch内置的profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA if device.type == 'cuda' else torch.profiler.ProfilerActivity.CPU,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                output = model(input_data)
    
    # 打印性能统计
    print("CPU时间统计:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    if device.type == 'cuda':
        print("\nCUDA时间统计:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 分析模型性能
test_model = HybridModel(784, 256, 10)
test_input = torch.randn(32, 784)

print("模型性能分析：")
profile_model_performance(test_model, test_input, device)

print("\n性能优化总结：")
print("1. 使用JIT编译和torch.compile可获得显著加速")
print("2. 混合精度训练可减少内存使用并提升速度")
print("3. 数据并行适合大batch训练，模型并行适合大模型")
print("4. 分布式训练可充分利用多机多卡资源")
print("5. 内存优化技术可训练更大模型")
print("6. 性能分析工具帮助定位瓶颈，指导优化方向") 