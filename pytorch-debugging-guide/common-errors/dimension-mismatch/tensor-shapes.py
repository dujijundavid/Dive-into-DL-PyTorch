"""
PyTorch 维度不匹配问题分析与解决方案
====================================

维度不匹配是 PyTorch 中最常见的错误之一。
本文档提供系统性的分析和解决方案。

常见错误类型：
1. RuntimeError: mat1 and mat2 shapes cannot be multiplied
2. RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)
3. RuntimeError: Expected input batch_size (X) to match target batch_size (Y)
4. RuntimeError: Dimension out of range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("🔍 PyTorch 维度不匹配问题全面分析")
print("=" * 50)

# 演示常见维度错误
print("=== 常见维度错误演示 ===\n")

# 错误1: 矩阵乘法维度不匹配
print("1. 矩阵乘法维度不匹配")
try:
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)  # 错误形状
    result = torch.mm(a, b)
except RuntimeError as e:
    print(f"❌ 错误: {e}")
    b_correct = torch.randn(4, 5)
    result = torch.mm(a, b_correct)
    print(f"✅ 修复后形状: {result.shape}\n")

# 错误2: 神经网络输入维度不匹配
print("2. 全连接层输入维度不匹配")
try:
    fc = nn.Linear(10, 5)
    x = torch.randn(32, 8)  # 错误特征数
    output = fc(x)
except RuntimeError as e:
    print(f"❌ 错误: 期望10个特征，得到8个")
    x_correct = torch.randn(32, 10)
    output = fc(x_correct)
    print(f"✅ 修复后形状: {output.shape}\n")

print("📚 总结 - 维度问题解决策略:")
print("1. 🔍 仔细检查错误信息中的形状提示")
print("2. 📐 理解每个操作的形状要求")
print("3. 🧮 掌握广播规则")
print("4. 🛠️ 使用 .shape, .size() 调试形状")

def demonstrate_broadcasting_issues():
    """演示广播机制相关的维度问题"""
    print("=== 广播机制相关问题 ===\n")
    
    # 广播规则回顾
    print("📚 PyTorch 广播规则:")
    print("1. 从最后一个维度开始比较")
    print("2. 维度大小必须相等，或其中一个为1，或其中一个不存在")
    print("3. 缺失的维度会在前面补1\n")
    
    # 正确的广播示例
    print("✅ 正确的广播示例:")
    a = torch.randn(3, 1, 4)  # shape: [3, 1, 4]
    b = torch.randn(2, 4)     # shape: [2, 4] -> 广播为 [1, 2, 4]
    result = a + b            # 结果: [3, 2, 4]
    print(f"a.shape={a.shape}, b.shape={b.shape}")
    print(f"广播后结果: {result.shape}\n")
    
    # 错误的广播示例
    print("❌ 错误的广播示例:")
    try:
        a = torch.randn(3, 4)     # shape: [3, 4]
        b = torch.randn(5, 4)     # shape: [5, 4]
        result = a + b            # 这会报错，因为 3 != 5
    except RuntimeError as e:
        print(f"错误: {e}")
        print("原因: 第一个维度 3 != 5，且都不为1，无法广播")
        
        # 修复方法1: 使用 unsqueeze 增加维度
        a_expanded = a.unsqueeze(0)  # [1, 3, 4]
        b_expanded = b.unsqueeze(1)  # [5, 1, 4]
        result = a_expanded + b_expanded  # [5, 3, 4]
        print(f"✅ 修复方法1: 增加维度后广播")
        print(f"   result.shape={result.shape}\n")

def demonstrate_reshape_pitfalls():
    """演示 reshape 操作的常见陷阱"""
    print("=== Reshape 操作陷阱 ===\n")
    
    # 陷阱1: 总元素数不匹配
    print("1. 总元素数不匹配")
    try:
        x = torch.randn(3, 4, 5)  # 总元素数: 3*4*5 = 60
        x_reshaped = x.reshape(3, 4, 6)  # 期望总元素数: 3*4*6 = 72
    except RuntimeError as e:
        print(f"❌ 错误: {e}")
        print(f"   原因: 原始元素数={3*4*5}, 目标元素数={3*4*6}")
        
        # 正确的做法
        x_reshaped = x.reshape(3, 20)  # 3*20 = 60 ✓
        print(f"✅ 修复: x.shape={x.shape} -> x_reshaped.shape={x_reshaped.shape}\n")
    
    # 陷阱2: -1 的使用
    print("2. -1 参数的正确使用")
    x = torch.randn(2, 3, 4, 5)
    print(f"原始形状: {x.shape}")
    
    # 正确使用 -1
    x1 = x.reshape(-1, 5)      # 自动计算第一维: 2*3*4 = 24
    x2 = x.reshape(2, -1)      # 自动计算第二维: 3*4*5 = 60
    x3 = x.reshape(2, 3, -1)   # 自动计算第三维: 4*5 = 20
    
    print(f"x.reshape(-1, 5): {x1.shape}")
    print(f"x.reshape(2, -1): {x2.shape}")
    print(f"x.reshape(2, 3, -1): {x3.shape}")
    
    # 错误使用 -1（多个 -1）
    try:
        x_wrong = x.reshape(-1, -1)  # 不能有多个 -1
    except RuntimeError as e:
        print(f"❌ 错误: 不能在 reshape 中使用多个 -1")
        print(f"   {str(e)}\n")

def demonstrate_cnn_dimension_issues():
    """演示 CNN 中常见的维度问题"""
    print("=== CNN 维度问题 ===\n")
    
    # 错误1: 卷积层输入通道数不匹配
    print("1. 卷积层输入通道数不匹配")
    try:
        conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        x = torch.randn(32, 1, 28, 28)  # [batch, channels, height, width]
        # 错误：输入是1通道，但conv层期望3通道
        output = conv(x)
    except RuntimeError as e:
        print(f"❌ 错误: {e}")
        print(f"   原因: conv层期望3通道输入，实际得到1通道")
        
        # 修复方法1: 调整输入
        x_correct = torch.randn(32, 3, 28, 28)
        output = conv(x_correct)
        print(f"✅ 修复1: 调整输入通道数")
        print(f"   x_correct.shape={x_correct.shape} -> output.shape={output.shape}")
        
        # 修复方法2: 调整网络
        conv_correct = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        output2 = conv_correct(x)
        print(f"✅ 修复2: 调整网络输入通道数")
        print(f"   x.shape={x.shape} -> output2.shape={output2.shape}\n")
    
    # 错误2: 池化后尺寸计算错误
    print("2. 池化操作尺寸计算")
    x = torch.randn(1, 1, 7, 7)  # 较小的特征图
    
    # 这可能会出问题：池化核太大
    try:
        pool = nn.MaxPool2d(kernel_size=8)  # 8x8 池化核
        output = pool(x)  # 7x7 输入无法被 8x8 池化
    except RuntimeError as e:
        print(f"❌ 潜在问题: 池化核(8x8)大于输入尺寸(7x7)")
        
        # 计算池化后尺寸的公式
        print("📐 池化尺寸计算公式:")
        print("output_size = floor((input_size - kernel_size) / stride) + 1")
        
        # 正确的池化设置
        pool_correct = nn.MaxPool2d(kernel_size=2, stride=2)
        output = pool_correct(x)
        input_h, input_w = 7, 7
        kernel_size = 2
        stride = 2
        expected_h = (input_h - kernel_size) // stride + 1
        expected_w = (input_w - kernel_size) // stride + 1
        
        print(f"✅ 修复: 使用合适的池化参数")
        print(f"   输入: {x.shape}")
        print(f"   计算: ({input_h}-{kernel_size})/{stride}+1 = {expected_h}")
        print(f"   输出: {output.shape}\n")

def dimension_debugging_toolkit():
    """维度调试工具包"""
    print("=== 维度调试工具包 ===\n")
    
    def analyze_tensor_shape(tensor, name="tensor"):
        """分析张量形状的工具函数"""
        print(f"📊 {name} 分析:")
        print(f"   形状: {tensor.shape}")
        print(f"   维度数: {tensor.ndim}")
        print(f"   元素总数: {tensor.numel()}")
        print(f"   数据类型: {tensor.dtype}")
        print(f"   设备: {tensor.device}")
        return tensor.shape
    
    def check_operation_compatibility(tensor_a, tensor_b, operation="elementwise"):
        """检查两个张量操作兼容性"""
        print(f"🔍 {operation} 操作兼容性检查:")
        print(f"   张量A形状: {tensor_a.shape}")
        print(f"   张量B形状: {tensor_b.shape}")
        
        if operation == "matmul":
            if tensor_a.shape[-1] == tensor_b.shape[-2]:
                expected_shape = list(tensor_a.shape[:-1]) + [tensor_b.shape[-1]]
                print(f"   ✅ 兼容！期望输出形状: {expected_shape}")
                return True
            else:
                print(f"   ❌ 不兼容！A的最后一维({tensor_a.shape[-1]}) != B的倒数第二维({tensor_b.shape[-2]})")
                return False
        elif operation == "elementwise":
            try:
                # 尝试广播
                result_shape = torch.broadcast_shapes(tensor_a.shape, tensor_b.shape)
                print(f"   ✅ 兼容！广播后形状: {result_shape}")
                return True
            except RuntimeError:
                print(f"   ❌ 不兼容！无法广播")
                return False
    
    # 示例使用
    print("工具函数使用示例:")
    a = torch.randn(3, 4, 5)
    b = torch.randn(5, 2)
    
    analyze_tensor_shape(a, "张量A")
    analyze_tensor_shape(b, "张量B")
    check_operation_compatibility(a, b, "matmul")
    
    c = torch.randn(3, 1, 5)
    check_operation_compatibility(a, c, "elementwise")

def main():
    """主函数：演示所有维度相关问题"""
    print("🔍 PyTorch 维度不匹配问题全面分析")
    print("=" * 50)
    
    demonstrate_common_dimension_errors()
    demonstrate_broadcasting_issues()
    demonstrate_reshape_pitfalls()
    demonstrate_cnn_dimension_issues()
    dimension_debugging_toolkit()
    
    print("\n📚 总结 - 维度问题解决策略:")
    print("1. 🔍 仔细检查错误信息中的形状提示")
    print("2. 📐 理解每个操作的形状要求")
    print("3. 🧮 掌握广播规则")
    print("4. 🛠️ 使用 .shape, .size() 调试形状")
    print("5. 📊 可视化数据流和形状变化")
    print("6. 🔧 灵活使用 reshape, view, squeeze, unsqueeze")

if __name__ == "__main__":
    main() 