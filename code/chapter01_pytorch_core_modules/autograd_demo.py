"""
torch.autograd 自动微分与梯度机制
----------------------------------
第一性原理思考：
1. 什么是梯度？
   - 梯度是函数在某点的最大变化率方向
   - 在机器学习中，梯度告诉我们如何调整参数以最小化损失函数
   - 本质上是多变量函数的偏导数向量

2. 为什么需要自动微分？
   - 手动计算复杂函数的导数容易出错
   - 神经网络可能有数百万个参数，手动计算不现实
   - 自动微分可以精确计算任意复杂函数的导数

3. 自动微分的基本原理是什么？
   - 链式法则：复合函数的导数等于各层导数的乘积
   - 计算图：将计算过程表示为有向图
   - 反向传播：从输出到输入逐层计算梯度

苏格拉底式提问与验证：
1. 为什么需要 requires_grad=True？
   - 问题：如果不设置 requires_grad=True 会怎样？
   - 验证：运行下面的代码
   - 结论：只有设置 requires_grad=True 的张量才会参与梯度计算

2. 梯度是如何累积的？
   - 问题：多次调用 backward() 会发生什么？
   - 验证：观察梯度值的变化
   - 结论：梯度会累加，需要手动清零

3. 为什么 inplace 操作会破坏 autograd？
   - 问题：inplace 操作如何影响计算图？
   - 验证：尝试 inplace 操作并观察错误
   - 结论：inplace 操作会破坏计算图的连接性

费曼学习法讲解：
1. 概念解释
   - 用最简单的语言解释自动微分
   - 通过具体例子说明计算过程
   - 避免使用专业术语，除非必要

2. 实例教学
   - 从简单到复杂逐步构建理解
   - 通过代码验证每一步的理解
   - 鼓励读者自己尝试和实验

3. 知识巩固
   - 总结关键概念
   - 提供常见陷阱和解决方案
   - 建议进一步学习的方向

功能说明：
- autograd 是 PyTorch 的自动求导引擎，支持动态图计算和反向传播。

原理讲解：
- Tensor 默认 requires_grad=False，设置为 True 后会追踪所有操作，构建计算图。
- loss.backward() 自动计算所有叶子节点（参数）的梯度。
- 支持链式法则、多次反向传播、动态图灵活性。

使用场景：
- 神经网络训练、梯度检查、可微分编程。

常见bug：
- 忘记设置 requires_grad=True，导致参数不参与反向传播。
- 多次 backward 未清零梯度，导致梯度累加。
- 叶子节点 inplace 操作会破坏 autograd。

深度学习研究员精华笔记：
- 动态计算图让调试和创新更灵活，便于实现复杂结构（如条件分支、循环等）。
- 反向传播本质是链式法则的自动实现。
- 任何可微分操作都能自动求导，极大降低实现新模型的门槛。

可运行案例：
"""
import torch

# 1. 创建可求导张量
a = torch.tensor([2.0, 3.0], requires_grad=True)
b = torch.tensor([6.0, 4.0], requires_grad=True)
c = a * b + b
print("c:", c)

# 2. 构建标量 loss 并反向传播
loss = c.sum()
loss.backward()
print("a 的梯度:", a.grad)
print("b 的梯度:", b.grad)

# 3. 多次 backward 需清零梯度
loss2 = (a * b).sum()
a.grad.zero_(); b.grad.zero_()
loss2.backward()
print("再次反向传播后的 a.grad:", a.grad)

# 4. 叶子节点 inplace 操作示例（会报错）
try:
    a += 1  # inplace 会破坏 autograd
except RuntimeError as e:
    print("inplace 操作报错:", e)

# 5. 动态计算图优势
for i in range(3):
    x = torch.tensor(float(i), requires_grad=True)
    y = x * x if x.item() > 1 else x + 1
    y.backward()
    print(f"x={x.item()}, grad={x.grad}")

# 6. 验证 requires_grad 的影响
print("\n验证 requires_grad 的影响：")
x1 = torch.tensor([1.0], requires_grad=True)
x2 = torch.tensor([1.0], requires_grad=False)
y1 = x1 * 2
y2 = x2 * 2
try:
    y1.backward()
    print("x1 的梯度:", x1.grad)
    y2.backward()
except RuntimeError as e:
    print("x2 无法计算梯度:", e)

# 7. 梯度累积验证
print("\n梯度累积验证：")
x = torch.tensor([1.0], requires_grad=True)
for i in range(3):
    y = x * 2
    y.backward()
    print(f"第 {i+1} 次反向传播后 x 的梯度:", x.grad)
    if i < 2:  # 不清零梯度
        print("不清零梯度，梯度会累加")
    else:
        x.grad.zero_()
        print("清零梯度后")

# 8. 链式法则验证
print("\n链式法则验证：")
x = torch.tensor([2.0], requires_grad=True)
y = x * x  # y = x^2
z = y * x  # z = x^3
z.backward()
print(f"x = {x.item()}")
print(f"dy/dx = 2x = {2*x.item()}")  # 理论值
print(f"dz/dx = 3x^2 = {3*x.item()**2}")  # 理论值
print(f"实际计算的梯度:", x.grad)  # 实际值 