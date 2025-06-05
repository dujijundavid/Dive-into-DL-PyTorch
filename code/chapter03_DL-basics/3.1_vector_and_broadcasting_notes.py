# 3.1 向量操作与广播机制（PyTorch）
# =====================================
# 本脚本详细介绍了PyTorch中的向量化操作、广播机制、优化建议与调试技巧。

import torch
from time import time

# ## 1. 向量化操作的重要性
# 在深度学习中，优先使用向量化操作而不是循环，可以极大提升效率。

a = torch.ones(1000)
b = torch.ones(1000)

# 方法1：循环逐元素相加
start = time()
c = torch.zeros(1000)
for i in range(1000):
    c[i] = a[i] + b[i]
print(f'循环操作耗时: {time() - start:.6f} 秒')

# 方法2：向量化操作
start = time()
d = a + b
print(f'向量化操作耗时: {time() - start:.6f} 秒')

# 结论：向量化操作更快，推荐优先使用。

# 为什么向量化操作更快？
# 1. 利用CPU/GPU并行计算能力
# 2. 减少Python解释器开销
# 3. 利用底层BLAS库
# 4. 减少内存访问次数

# ## 2. 广播机制详解
# 广播机制允许不同形状的张量进行运算。

# 示例1：向量与标量
x = torch.ones(3)
y = 10
print('向量与标量相加:', x + y)

# 示例2：不同形状的矩阵
x = torch.ones(3, 4)
y = torch.ones(4)
print('x的形状:', x.shape)
print('y的形状:', y.shape)
print('广播后的形状:', (x + y).shape)

# 示例3：维度扩展
m = torch.ones(2, 3)
n = torch.ones(3)
print('\nm的形状:', m.shape)
print('n的形状:', n.shape)
print('广播后的形状:', (m + n).shape)

# 广播机制的工作原理：
# 1. 维度对齐，从右向左比较
# 2. 维度不同则扩展
# 3. 某一维为1时可广播

# ## 3. 广播机制最佳实践
# 1. 显式扩展维度（unsqueeze/expand）
# 2. 避免隐式广播，复杂运算时显式指定维度
# 3. 注意内存使用，广播可能创建临时张量

# 1. 使用unsqueeze显式扩展维度
x = torch.ones(3)
y = torch.ones(3, 1)
print('使用unsqueeze后的形状:', x.unsqueeze(1).shape)

# 2. 使用expand而不是repeat（更节省内存）
a = torch.ones(3)
b = a.expand(3, 3)  # 不会分配新内存
print('使用expand后的形状:', b.shape)

# 3. 使用view或reshape改变形状
c = torch.ones(3)
d = c.view(3, 1)
print('使用view后的形状:', d.shape)

# ## 4. 常见向量操作优化
# 1. 内存优化
#    - 使用inplace操作（如add_）减少内存使用
#    - 使用contiguous()确保内存连续
#    - 使用clone()创建新副本

# inplace操作
x = torch.ones(3)
y = torch.ones(3)
x.add_(y)  # 原地操作
print('inplace操作后的x:', x)

# 内存连续
a = torch.ones(3, 4).transpose(0, 1)
print('转置后是否连续:', a.is_contiguous())
a = a.contiguous()
print('contiguous后是否连续:', a.is_contiguous())

# 创建副本
b = torch.ones(3)
c = b.clone()
print('b和c是否共享内存:', b.storage().data_ptr() == c.storage().data_ptr())

# 2. 性能优化
#    - 使用适当的设备（CPU/GPU）
#    - 批量处理数据
#    - 避免频繁的CPU-GPU数据传输

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.ones(3).to(device)
print('张量所在设备:', x.device)

batch_size = 32
x = torch.randn(batch_size, 3, 224, 224)
print('批量数据的形状:', x.shape)

if torch.cuda.is_available():
    x = x.cuda()
    y = x * 2
    z = y + 1
    result = z.cpu()

# ## 5. 调试技巧
# 1. 检查张量形状：shape/size()
# 2. 检查数据类型：dtype
# 3. 检查设备：device
# 4. 检查内存布局：is_contiguous()
# 5. 检查梯度：requires_grad/grad

x = torch.ones(3, requires_grad=True)
y = x * 2
z = y.sum()
z.backward()

print('张量形状:', x.shape)
print('数据类型:', x.dtype)
print('所在设备:', x.device)
print('是否连续:', x.is_contiguous())
print('是否需要梯度:', x.requires_grad)
print('梯度值:', x.grad) 