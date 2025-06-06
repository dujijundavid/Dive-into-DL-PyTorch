"""
torch.autograd 自动微分与梯度机制
----------------------------------
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