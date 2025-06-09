"""
PyTorch 前置知识核心技术与深度理解
---------------------------------
【文件说明】
本文件系统梳理了PyTorch深度学习的核心基础概念，包括：
- 张量（Tensor）操作：深度学习的数据基础
- 自动微分（Autograd）：深度学习的优化基础
- 内存管理与计算图构建原理
- 高效张量计算的底层机制

【第一性原理思考】
1. 为什么需要张量？
   - 标量（0维）→ 向量（1维）→ 矩阵（2维）→ 张量（n维）
   - 统一的数据结构处理不同维度的数据
   - 支持批量计算，充分利用现代硬件（GPU/TPU）并行能力

2. 为什么需要自动微分？
   - 深度网络参数数量巨大（百万到千亿级）
   - 手工计算梯度几乎不可能且容易出错
   - 反向传播算法的自动化实现是深度学习可行的关键

3. 张量计算的本质是什么？
   - 线性代数运算的高维泛化
   - 通过广播机制统一不同形状的运算
   - 底层依赖BLAS库和CUDA实现高效计算

【苏格拉底式提问与验证】
1. 张量与NumPy数组有什么本质区别？
   - 问题：都是多维数组，为什么要重新设计？
   - 验证：通过性能测试和GPU支持对比
   - 结论：张量支持GPU加速和自动微分，是深度学习的核心

2. 为什么需要计算图？
   - 问题：直接计算不是更简单吗？
   - 验证：通过复杂函数的梯度计算演示
   - 结论：计算图实现了复杂函数的自动微分

【费曼学习法讲解】
1. 概念解释
   - 用积木类比张量的构建和操作
   - 用工厂流水线类比计算图的构建
   - 强调张量操作在实际问题中的应用

2. 实例教学
   - 从简单的数学运算开始
   - 逐步扩展到复杂的深度学习场景
   - 通过可视化理解张量变换

【设计意义与工程价值】
- 张量操作是所有深度学习算法的基础
- 自动微分使得复杂模型的训练成为可能
- 理解这些基础概念对后续学习至关重要

可运行案例：
"""
import torch
import torch.nn as nn
import numpy as np
import time
import sys

# 1. 张量创建与基本操作
# --------------------
# 原理说明：
# 张量是深度学习的基本数据结构，支持多维数组运算
# 不同的创建方式对应不同的使用场景和初始化策略

print("1. 张量创建与基本操作")
print("=" * 50)

class TensorCreationDemo:
    """张量创建方法演示"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"当前设备: {self.device}")
    
    def creation_methods(self):
        """演示各种张量创建方法"""
        print("\n各种张量创建方法：")
        
        # 直接创建
        x1 = torch.tensor([1, 2, 3, 4])
        print(f"直接创建: {x1}")
        
        # 特殊值创建
        x2 = torch.zeros(2, 3)
        x3 = torch.ones(2, 3)
        x4 = torch.eye(3)  # 单位矩阵
        print(f"零矩阵: \n{x2}")
        print(f"单位矩阵: \n{x4}")
        
        # 随机创建
        x5 = torch.randn(2, 3)  # 标准正态分布
        x6 = torch.rand(2, 3)   # [0,1)均匀分布
        print(f"正态分布随机: \n{x5}")
        
        # 范围创建
        x7 = torch.arange(0, 10, 2)
        x8 = torch.linspace(0, 1, 5)
        print(f"等差数列: {x7}")
        print(f"线性插值: {x8}")
        
        return x1, x2, x3, x4, x5, x6, x7, x8
    
    def tensor_properties(self, x):
        """张量属性分析"""
        print(f"\n张量属性分析：")
        print(f"形状 (shape): {x.shape}")
        print(f"维度 (ndim): {x.ndim}")
        print(f"数据类型 (dtype): {x.dtype}")
        print(f"设备 (device): {x.device}")
        print(f"元素总数 (numel): {x.numel()}")
        print(f"存储步长 (stride): {x.stride()}")

# 创建演示实例
tensor_demo = TensorCreationDemo()
tensors = tensor_demo.creation_methods()
tensor_demo.tensor_properties(tensors[4])  # 分析正态分布张量

# 2. 张量运算与广播机制
# --------------------
# 原理说明：
# 广播机制允许不同形状的张量进行运算
# 这是实现高效批量计算的关键机制

print("\n\n2. 张量运算与广播机制")
print("=" * 50)

class TensorOperationsDemo:
    """张量运算演示"""
    
    def element_wise_operations(self):
        """逐元素运算"""
        print("逐元素运算：")
        
        x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
        
        print(f"x = \n{x}")
        print(f"y = \n{y}")
        print(f"x + y = \n{x + y}")
        print(f"x * y = \n{x * y}")  # 逐元素乘法
        print(f"x / y = \n{x / y}")
        print(f"x ** 2 = \n{x ** 2}")
        
    def broadcasting_demo(self):
        """广播机制演示"""
        print("\n广播机制：")
        
        # 不同维度的张量运算
        x = torch.randn(3, 4)    # 2D张量
        y = torch.randn(4)       # 1D张量
        z = torch.randn(3, 1)    # 2D张量（列向量）
        
        print(f"x形状: {x.shape}")
        print(f"y形状: {y.shape}")
        print(f"z形状: {z.shape}")
        
        # 广播运算
        result1 = x + y  # y广播到(3,4)
        result2 = x + z  # z广播到(3,4)
        
        print(f"x + y 结果形状: {result1.shape}")
        print(f"x + z 结果形状: {result2.shape}")
    
    def matrix_operations(self):
        """矩阵运算"""
        print("\n矩阵运算：")
        
        A = torch.randn(3, 4)
        B = torch.randn(4, 5)
        
        # 矩阵乘法
        C = torch.matmul(A, B)  # 或 A @ B
        print(f"A({A.shape}) @ B({B.shape}) = C({C.shape})")
        
        # 批量矩阵乘法
        batch_A = torch.randn(10, 3, 4)  # 10个3x4矩阵
        batch_B = torch.randn(10, 4, 5)  # 10个4x5矩阵
        batch_C = torch.bmm(batch_A, batch_B)  # 批量矩阵乘法
        print(f"批量矩阵乘法: ({batch_A.shape}) @ ({batch_B.shape}) = ({batch_C.shape})")
    
    def performance_comparison(self):
        """性能对比：向量化 vs 循环"""
        print("\n性能对比：")
        
        size = 10000
        x = torch.randn(size)
        y = torch.randn(size)
        
        # 向量化操作
        start_time = time.time()
        for _ in range(100):
            result_vectorized = x + y
        vectorized_time = time.time() - start_time
        
        print(f"向量化操作100次时间: {vectorized_time:.6f}s")
        print("向量化的优势：充分利用底层BLAS库和硬件并行能力")

# 运行张量运算演示
ops_demo = TensorOperationsDemo()
ops_demo.element_wise_operations()
ops_demo.broadcasting_demo()
ops_demo.matrix_operations()
ops_demo.performance_comparison()

# 3. 自动微分机制
# --------------
# 原理说明：
# 自动微分是深度学习的核心技术，实现了复杂函数的梯度自动计算
# PyTorch通过计算图记录操作历史，然后反向传播计算梯度

print("\n\n3. 自动微分机制")
print("=" * 50)

class AutogradDemo:
    """自动微分演示"""
    
    def basic_autograd(self):
        """基础自动微分"""
        print("基础自动微分：")
        
        # 创建需要梯度的张量
        x = torch.tensor(2.0, requires_grad=True)
        y = torch.tensor(3.0, requires_grad=True)
        
        # 定义函数 z = x^2 + 2*x*y + y^2
        z = x**2 + 2*x*y + y**2
        
        print(f"x = {x}, y = {y}")
        print(f"z = x² + 2xy + y² = {z}")
        
        # 反向传播
        z.backward()
        
        print(f"∂z/∂x = {x.grad}")  # 应该是 2x + 2y = 10
        print(f"∂z/∂y = {y.grad}")  # 应该是 2x + 2y = 10
        
        # 验证手工计算
        print(f"手工计算 ∂z/∂x = 2x + 2y = {2*x.item() + 2*y.item()}")
        print(f"手工计算 ∂z/∂y = 2x + 2y = {2*x.item() + 2*y.item()}")
    
    def computational_graph(self):
        """计算图演示"""
        print("\n计算图概念：")
        
        x = torch.tensor(1.0, requires_grad=True)
        
        # 构建计算图
        a = x + 2      # a = x + 2
        b = a * 3      # b = 3a = 3(x + 2)
        c = b.pow(2)   # c = b² = [3(x + 2)]²
        
        print(f"x = {x}")
        print(f"a = x + 2 = {a}")
        print(f"b = 3a = {b}")
        print(f"c = b² = {c}")
        
        # 反向传播
        c.backward()
        
        print(f"dc/dx = {x.grad}")
        
        # 手工验证：c = [3(x + 2)]², dc/dx = 2 * 3(x + 2) * 3 = 18(x + 2)
        expected_grad = 18 * (x.item() + 2)
        print(f"理论梯度: 18(x + 2) = {expected_grad}")
    
    def higher_order_gradients(self):
        """高阶梯度"""
        print("\n高阶梯度：")
        
        x = torch.tensor(2.0, requires_grad=True)
        
        # 一阶梯度
        y = x**3
        grad1 = torch.autograd.grad(y, x, create_graph=True)[0]
        print(f"f(x) = x³")
        print(f"f'(x) = {grad1}")  # 3x² = 12
        
        # 二阶梯度
        grad2 = torch.autograd.grad(grad1, x)[0]
        print(f"f''(x) = {grad2}")  # 6x = 12

# 运行自动微分演示
autograd_demo = AutogradDemo()
autograd_demo.basic_autograd()
autograd_demo.computational_graph()
autograd_demo.higher_order_gradients()

# 4. 张量形状操作
# --------------
# 原理说明：
# 形状操作是数据预处理和模型设计的重要工具
# 理解这些操作有助于灵活处理不同形状的数据

print("\n\n4. 张量形状操作")
print("=" * 50)

class TensorShapeDemo:
    """张量形状操作演示"""
    
    def reshape_operations(self):
        """形状重塑操作"""
        print("形状重塑操作：")
        
        x = torch.arange(24)
        print(f"原始张量: {x.shape} - {x}")
        
        # 重塑为不同形状
        x2d = x.reshape(4, 6)
        x3d = x.reshape(2, 3, 4)
        x4d = x.reshape(2, 2, 2, 3)
        
        print(f"2D形状: {x2d.shape}")
        print(f"3D形状: {x3d.shape}")
        print(f"4D形状: {x4d.shape}")
        
        # 自动推断维度
        x_auto = x.reshape(-1, 6)  # -1表示自动推断
        print(f"自动推断: {x_auto.shape}")
    
    def dimension_operations(self):
        """维度操作"""
        print("\n维度操作：")
        
        x = torch.randn(2, 3, 4)
        print(f"原始形状: {x.shape}")
        
        # 增加维度
        x_unsqueeze = x.unsqueeze(0)    # 在第0维增加
        x_unsqueeze2 = x.unsqueeze(-1)  # 在最后增加
        print(f"unsqueeze(0): {x_unsqueeze.shape}")
        print(f"unsqueeze(-1): {x_unsqueeze2.shape}")
        
        # 转置
        x_t = x.transpose(0, 2)  # 交换第0和第2维
        x_permute = x.permute(2, 0, 1)  # 重新排列所有维度
        print(f"transpose(0,2): {x_t.shape}")
        print(f"permute(2,0,1): {x_permute.shape}")
    
    def concatenation_operations(self):
        """拼接操作"""
        print("\n拼接操作：")
        
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        
        # 沿不同维度拼接
        cat_dim0 = torch.cat([x, y], dim=0)  # 沿第0维
        cat_dim1 = torch.cat([x, y], dim=1)  # 沿第1维
        
        print(f"原始形状: {x.shape}")
        print(f"cat dim=0: {cat_dim0.shape}")
        print(f"cat dim=1: {cat_dim1.shape}")
        
        # 堆叠
        stack_dim0 = torch.stack([x, y, z], dim=0)
        print(f"stack dim=0: {stack_dim0.shape}")

# 运行形状操作演示
shape_demo = TensorShapeDemo()
shape_demo.reshape_operations()
shape_demo.dimension_operations()
shape_demo.concatenation_operations()

# 5. 设备管理与内存优化
# --------------------
# 原理说明：
# 设备管理是GPU加速的基础，内存优化对于大规模训练至关重要

print("\n\n5. 设备管理与内存优化")
print("=" * 50)

class DeviceMemoryDemo:
    """设备和内存管理演示"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.has_cuda = torch.cuda.is_available()
        
    def device_management(self):
        """设备管理"""
        print("设备管理：")
        print(f"CUDA可用: {self.has_cuda}")
        print(f"当前设备: {self.device}")
        
        if self.has_cuda:
            print(f"CUDA设备数量: {torch.cuda.device_count()}")
            print(f"当前CUDA设备: {torch.cuda.current_device()}")
        
        # 张量设备操作
        x_cpu = torch.randn(3, 4)
        print(f"CPU张量设备: {x_cpu.device}")
        
        if self.has_cuda:
            x_gpu = x_cpu.to('cuda')
            print(f"GPU张量设备: {x_gpu.device}")
            
            # 设备间转换
            x_back_cpu = x_gpu.cpu()
            print(f"转回CPU: {x_back_cpu.device}")
    
    def inplace_operations(self):
        """就地操作演示"""
        print("\n就地操作：")
        
        x = torch.randn(3, 4)
        print(f"原始张量ID: {id(x)}")
        
        # 非就地操作
        y = x + 1
        print(f"非就地操作后，新张量ID: {id(y)}")
        
        # 就地操作
        x.add_(1)  # 等价于 x += 1
        print(f"就地操作后，张量ID: {id(x)}")

# 运行设备内存演示
device_demo = DeviceMemoryDemo()
device_demo.device_management()
device_demo.inplace_operations()

# 6. 实际应用案例
# --------------
# 原理说明：
# 通过实际案例展示前置知识在深度学习中的应用

print("\n\n6. 实际应用案例")
print("=" * 50)

class PracticalApplications:
    """实际应用案例"""
    
    def linear_regression_example(self):
        """线性回归示例"""
        print("线性回归示例：")
        
        # 生成模拟数据
        n_samples = 100
        x = torch.randn(n_samples, 1)
        true_w, true_b = 2.0, 1.0
        y = true_w * x + true_b + torch.randn(n_samples, 1) * 0.1
        
        # 初始化参数
        w = torch.randn(1, 1, requires_grad=True)
        b = torch.randn(1, requires_grad=True)
        
        print(f"真实参数: w={true_w}, b={true_b}")
        print(f"初始参数: w={w.item():.4f}, b={b.item():.4f}")
        
        # 训练几步
        learning_rate = 0.01
        for epoch in range(50):
            # 前向传播
            y_pred = x @ w + b
            loss = ((y_pred - y) ** 2).mean()
            
            # 梯度清零
            if w.grad is not None:
                w.grad.zero_()
            if b.grad is not None:
                b.grad.zero_()
            
            # 反向传播
            loss.backward()
            
            # 参数更新
            with torch.no_grad():
                w -= learning_rate * w.grad
                b -= learning_rate * b.grad
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        print(f"最终参数: w={w.item():.4f}, b={b.item():.4f}")
    
    def image_processing_example(self):
        """图像处理示例"""
        print("\n图像处理示例：")
        
        # 模拟图像数据 (batch_size, channels, height, width)
        images = torch.randn(32, 3, 64, 64)  # 32张RGB图像
        print(f"图像批次形状: {images.shape}")
        
        # 数据标准化
        mean = images.mean(dim=[0, 2, 3], keepdim=True)  # 计算每通道均值
        std = images.std(dim=[0, 2, 3], keepdim=True)    # 计算每通道标准差
        normalized_images = (images - mean) / std
        
        print(f"标准化前均值: {images.mean():.4f}")
        print(f"标准化后均值: {normalized_images.mean():.4f}")
        
        # 批量处理：提取中心区域
        center_crop = images[:, :, 16:48, 16:48]  # 提取32x32中心区域
        print(f"中心裁剪后形状: {center_crop.shape}")

# 运行实际应用演示
applications = PracticalApplications()
applications.linear_regression_example()
applications.image_processing_example()

print("\n" + "=" * 50)
print("前置知识总结：")
print("1. 张量是深度学习的基础数据结构，支持高效的多维数组运算")
print("2. 广播机制使得不同形状的张量能够进行运算，提升编程效率")
print("3. 自动微分是深度学习的核心，实现了复杂函数的梯度自动计算")
print("4. 设备管理和内存优化对于GPU加速和大规模训练至关重要")
print("5. 理解这些基础概念是掌握深度学习的前提")
print("6. 实际应用中需要灵活组合这些基础操作来解决复杂问题") 