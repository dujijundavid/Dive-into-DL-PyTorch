"""
torch.device（设备管理）核心原理与用法
------------------------------------
第一性原理思考：
1. 什么是设备管理？
   - 设备管理是指在不同计算设备（CPU、GPU）之间进行数据和计算的调度
   - 包括张量的设备分配、模型的设备转移、计算的设备选择
   - 是实现GPU加速计算的基础

2. 为什么需要设备管理？
   - 性能优化：GPU并行计算能力远超CPU
   - 内存管理：GPU和CPU具有独立的内存空间
   - 资源调度：有效利用多GPU资源
   - 兼容性：确保代码在不同硬件环境下运行

3. 设备管理的核心特性是什么？
   - 自动设备检测：自动识别可用的计算设备
   - 张量设备转移：在不同设备间移动数据
   - 模型设备部署：将模型部署到指定设备
   - 混合精度计算：优化GPU内存使用

苏格拉底式提问与验证：
1. GPU如何加速深度学习计算？
   - 问题：GPU相比CPU的优势在哪里？
   - 验证：比较相同操作在CPU和GPU上的执行时间
   - 结论：GPU的并行架构适合大规模矩阵运算

2. 为什么需要显式的设备管理？
   - 问题：不进行设备管理会有什么问题？
   - 验证：尝试在不同设备上进行计算
   - 结论：设备不匹配会导致计算错误

3. 如何优化设备间的数据传输？
   - 问题：数据传输的瓶颈在哪里？
   - 验证：测量不同数据传输方式的效率
   - 结论：减少不必要的设备间传输可以提高性能

费曼学习法讲解：
1. 概念解释
   - 用工厂流水线类比解释设备管理
   - 通过实际例子理解CPU和GPU的区别
   - 强调设备管理在深度学习中的重要性

2. 实例教学
   - 从基础到进阶的设备管理操作
   - 通过性能对比理解GPU加速原理
   - 实践多GPU训练和推理

3. 知识巩固
   - 总结设备管理的最佳实践
   - 提供性能优化的具体建议
   - 建议进阶学习方向

功能说明：
- torch.device 管理计算设备，支持CPU、CUDA GPU等不同硬件的使用。

原理讲解：
- 通过device对象指定张量和模型的计算设备。
- 支持自动设备检测和手动设备指定。
- 提供设备间数据传输和同步功能。

工程意义：
- 是GPU加速深度学习的基础，直接影响训练和推理的性能。

可运行案例：
"""
import torch
import torch.nn as nn
import time
import numpy as np

# 1. 设备检测和基本信息
print("1. 设备检测和基本信息：")

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA是否可用: {cuda_available}")

if cuda_available:
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("CUDA不可用，将使用CPU进行计算")

# 获取默认设备
default_device = torch.device('cuda' if cuda_available else 'cpu')
print(f"默认设备: {default_device}")

# 2. 设备对象的创建和使用
print("\n2. 设备对象的创建和使用：")

# 创建不同的设备对象
cpu_device = torch.device('cpu')
print(f"CPU设备: {cpu_device}")

if cuda_available:
    gpu_device = torch.device('cuda')
    gpu_device_0 = torch.device('cuda:0')
    print(f"GPU设备: {gpu_device}")
    print(f"指定GPU设备: {gpu_device_0}")

# 3. 张量的设备分配
print("\n3. 张量的设备分配：")

# 在CPU上创建张量
cpu_tensor = torch.randn(3, 4)
print(f"CPU张量设备: {cpu_tensor.device}")

# 直接在GPU上创建张量
if cuda_available:
    gpu_tensor = torch.randn(3, 4, device='cuda')
    print(f"GPU张量设备: {gpu_tensor.device}")
    
    # 使用设备对象创建张量
    gpu_tensor2 = torch.randn(3, 4, device=gpu_device)
    print(f"GPU张量2设备: {gpu_tensor2.device}")

# 4. 张量的设备转移
print("\n4. 张量的设备转移：")

# 创建CPU张量
x = torch.randn(100, 100)
print(f"原始张量设备: {x.device}")

if cuda_available:
    # 转移到GPU
    x_gpu = x.to('cuda')
    print(f"转移到GPU: {x_gpu.device}")
    
    # 转移到CPU
    x_cpu = x_gpu.to('cpu')
    print(f"转移回CPU: {x_cpu.device}")
    
    # 使用cuda()和cpu()方法
    x_gpu2 = x.cuda()
    x_cpu2 = x_gpu2.cpu()
    print(f"使用cuda()方法: {x_gpu2.device}")
    print(f"使用cpu()方法: {x_cpu2.device}")

# 5. 模型的设备部署
print("\n5. 模型的设备部署：")

# 创建简单模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(10, 50)
        self.linear2 = nn.Linear(50, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

model = SimpleModel()
print(f"模型参数设备: {next(model.parameters()).device}")

if cuda_available:
    # 将模型移动到GPU
    model_gpu = model.to('cuda')
    print(f"GPU模型参数设备: {next(model_gpu.parameters()).device}")
    
    # 使用cuda()方法
    model_gpu2 = SimpleModel().cuda()
    print(f"GPU模型2参数设备: {next(model_gpu2.parameters()).device}")

# 6. 设备兼容性检查
print("\n6. 设备兼容性检查：")

def check_device_compatibility(tensor, model):
    tensor_device = tensor.device
    model_device = next(model.parameters()).device
    
    print(f"张量设备: {tensor_device}")
    print(f"模型设备: {model_device}")
    
    if tensor_device == model_device:
        print("设备兼容 ✓")
        return True
    else:
        print("设备不兼容 ✗")
        return False

# 测试兼容性
x = torch.randn(5, 10)
model = SimpleModel()

print("CPU上的兼容性:")
check_device_compatibility(x, model)

if cuda_available:
    print("\nGPU上的兼容性:")
    x_gpu = x.cuda()
    model_gpu = model.cuda()
    check_device_compatibility(x_gpu, model_gpu)
    
    print("\n设备不匹配的情况:")
    check_device_compatibility(x, model_gpu)

# 7. 性能对比：CPU vs GPU
print("\n7. 性能对比：CPU vs GPU：")

def benchmark_computation(device, size=1000, iterations=100):
    """基准测试函数"""
    # 创建数据
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 预热
    for _ in range(10):
        c = torch.mm(a, b)
    
    # 同步GPU（如果适用）
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # 计时
    start_time = time.time()
    for _ in range(iterations):
        c = torch.mm(a, b)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    return (end_time - start_time) / iterations

# CPU基准测试
cpu_time = benchmark_computation(torch.device('cpu'), size=500, iterations=10)
print(f"CPU计算时间 (平均): {cpu_time:.6f} 秒")

if cuda_available:
    # GPU基准测试
    gpu_time = benchmark_computation(torch.device('cuda'), size=500, iterations=10)
    print(f"GPU计算时间 (平均): {gpu_time:.6f} 秒")
    print(f"GPU加速比: {cpu_time / gpu_time:.2f}倍")

# 8. 内存管理
print("\n8. 内存管理：")

if cuda_available:
    # 查看GPU内存使用
    print(f"GPU内存使用情况:")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # 创建大张量
    large_tensor = torch.randn(1000, 1000, device='cuda')
    print(f"创建大张量后:")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # 删除张量
    del large_tensor
    print(f"删除张量后:")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # 清空缓存
    torch.cuda.empty_cache()
    print(f"清空缓存后:")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

# 9. 多GPU使用
print("\n9. 多GPU使用：")

if cuda_available and torch.cuda.device_count() > 1:
    print(f"检测到 {torch.cuda.device_count()} 个GPU")
    
    # 在不同GPU上创建张量
    tensor_gpu0 = torch.randn(100, 100, device='cuda:0')
    tensor_gpu1 = torch.randn(100, 100, device='cuda:1')
    
    print(f"张量1设备: {tensor_gpu0.device}")
    print(f"张量2设备: {tensor_gpu1.device}")
    
    # GPU间数据传输
    tensor_gpu1_to_gpu0 = tensor_gpu1.to('cuda:0')
    print(f"传输后设备: {tensor_gpu1_to_gpu0.device}")
    
    # 使用DataParallel
    if torch.cuda.device_count() > 1:
        model = SimpleModel()
        model_parallel = nn.DataParallel(model)
        model_parallel = model_parallel.cuda()
        print(f"DataParallel模型设备: {next(model_parallel.parameters()).device}")
        
        # 测试并行推理
        x = torch.randn(32, 10, device='cuda')
        y = model_parallel(x)
        print(f"并行推理输出形状: {y.shape}")
        
elif cuda_available:
    print("只有一个GPU可用")
else:
    print("没有GPU可用")

# 10. 实用工具函数
print("\n10. 实用工具函数：")

def get_device(prefer_gpu=True):
    """获取推荐的计算设备"""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(obj, device):
    """将对象（张量、模型等）移动到指定设备"""
    if hasattr(obj, 'to'):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return [to_device(item, device) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}
    else:
        return obj

def print_device_info(obj, name="Object"):
    """打印对象的设备信息"""
    if hasattr(obj, 'device'):
        print(f"{name}设备: {obj.device}")
    elif hasattr(obj, 'parameters'):
        device = next(obj.parameters()).device
        print(f"{name}设备: {device}")
    else:
        print(f"{name}没有设备属性")

# 测试工具函数
device = get_device()
print(f"推荐设备: {device}")

# 创建测试对象
test_tensor = torch.randn(5, 5)
test_model = SimpleModel()
test_list = [torch.randn(3, 3), torch.randn(2, 2)]

print("\n移动前的设备信息:")
print_device_info(test_tensor, "张量")
print_device_info(test_model, "模型")
print_device_info(test_list[0], "列表中的张量")

# 移动到设备
test_tensor = to_device(test_tensor, device)
test_model = to_device(test_model, device)
test_list = to_device(test_list, device)

print("\n移动后的设备信息:")
print_device_info(test_tensor, "张量")
print_device_info(test_model, "模型")
print_device_info(test_list[0], "列表中的张量")

print("\n设备管理最佳实践总结:")
print("1. 总是检查CUDA是否可用")
print("2. 确保模型和数据在同一设备上")
print("3. 使用non_blocking=True优化数据传输")
print("4. 定期清理GPU内存")
print("5. 使用混合精度训练节省内存")
print("6. 考虑使用多GPU进行加速") 