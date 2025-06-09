"""
torch.nn 损失函数核心原理与用法
--------------------------------
第一性原理思考：
1. 什么是损失函数？
   - 损失函数量化了模型预测与真实标签之间的差距
   - 是优化算法的目标函数，指导模型参数更新方向
   - 不同任务需要不同的损失函数设计

2. 为什么需要损失函数？
   - 提供量化的优化目标：将模型性能转化为数值
   - 引导梯度下降：损失函数的梯度决定参数更新方向
   - 任务适配：不同任务特性需要特定的损失设计

3. 损失函数的核心特性是什么？
   - 可微性：支持梯度计算和反向传播
   - 单调性：损失越小表示性能越好
   - 任务相关性：与具体任务的评价指标相关

苏格拉底式提问与验证：
1. 不同损失函数适用于什么场景？
   - 问题：回归和分类为什么需要不同的损失函数？
   - 验证：比较不同损失函数在相同任务上的表现
   - 结论：损失函数的选择直接影响模型性能

2. 损失函数如何影响训练过程？
   - 问题：损失函数的形状如何影响优化？
   - 验证：观察不同损失函数的梯度特性
   - 结论：损失函数的性质决定了优化的难易程度

3. 如何处理类别不平衡问题？
   - 问题：数据不平衡时如何调整损失函数？
   - 验证：比较加权损失和普通损失的效果
   - 结论：加权损失可以缓解类别不平衡问题

费曼学习法讲解：
1. 概念解释
   - 用导航类比解释损失函数的作用
   - 通过几何直观理解不同损失函数的特性
   - 强调损失函数在深度学习中的核心地位

2. 实例教学
   - 从简单到复杂的损失函数应用
   - 通过可视化理解损失函数的性质
   - 实践不同任务的损失函数选择

3. 知识巩固
   - 总结各种损失函数的适用场景
   - 提供损失函数选择的指导原则
   - 建议进阶学习方向

功能说明：
- PyTorch提供丰富的损失函数，适用于分类、回归、排序等不同任务。

原理讲解：
- 损失函数计算预测值与真实值的差异，生成标量损失值。
- 支持元素级别、批次级别的损失计算和聚合。
- 可以自定义损失函数以适应特定任务需求。

工程意义：
- 是深度学习优化的核心，直接影响模型的收敛速度和最终性能。

可运行案例：
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# 1. 回归损失函数
print("1. 回归损失函数：")

# 生成回归数据
y_true = torch.randn(100, 1)
y_pred = y_true + torch.randn(100, 1) * 0.5  # 添加噪声

# MSE损失 (均方误差)
mse_loss = nn.MSELoss()
mse_value = mse_loss(y_pred, y_true)
print(f"MSE损失: {mse_value.item():.6f}")

# MAE损失 (平均绝对误差)
mae_loss = nn.L1Loss()
mae_value = mae_loss(y_pred, y_true)
print(f"MAE损失: {mae_value.item():.6f}")

# Smooth L1损失 (Huber损失)
smooth_l1_loss = nn.SmoothL1Loss()
smooth_l1_value = smooth_l1_loss(y_pred, y_true)
print(f"Smooth L1损失: {smooth_l1_value.item():.6f}")

# 2. 分类损失函数
print("\n2. 分类损失函数：")

# 生成分类数据
num_classes = 5
batch_size = 32
logits = torch.randn(batch_size, num_classes)  # 模型输出（未归一化）
targets = torch.randint(0, num_classes, (batch_size,))  # 真实标签

# 交叉熵损失
ce_loss = nn.CrossEntropyLoss()
ce_value = ce_loss(logits, targets)
print(f"交叉熵损失: {ce_value.item():.6f}")

# 负对数似然损失
log_softmax = F.log_softmax(logits, dim=1)
nll_loss = nn.NLLLoss()
nll_value = nll_loss(log_softmax, targets)
print(f"负对数似然损失: {nll_value.item():.6f}")

# 多标签分类的二元交叉熵损失
targets_multilabel = torch.randint(0, 2, (batch_size, num_classes)).float()
sigmoid_outputs = torch.sigmoid(logits)
bce_loss = nn.BCELoss()
bce_value = bce_loss(sigmoid_outputs, targets_multilabel)
print(f"二元交叉熵损失: {bce_value.item():.6f}")

# 带logits的二元交叉熵损失（数值更稳定）
bce_with_logits_loss = nn.BCEWithLogitsLoss()
bce_logits_value = bce_with_logits_loss(logits, targets_multilabel)
print(f"带logits的二元交叉熵损失: {bce_logits_value.item():.6f}")

# 3. 损失函数的可视化
print("\n3. Loss Function Visualization:")

def plot_loss_functions():
    # 生成数据范围
    x = torch.linspace(-3, 3, 100)
    target = torch.zeros_like(x)  # 目标值为0
    
    # 计算不同损失函数的值
    mse_values = F.mse_loss(x, target, reduction='none')
    mae_values = F.l1_loss(x, target, reduction='none')
    smooth_l1_values = F.smooth_l1_loss(x, target, reduction='none')
    
    # 绘制损失函数
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131)
    plt.plot(x.numpy(), mse_values.numpy(), label='MSE')
    plt.title('MSE Loss')
    plt.xlabel('Prediction - Target')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(132)
    plt.plot(x.numpy(), mae_values.numpy(), label='MAE', color='orange')
    plt.title('MAE Loss')
    plt.xlabel('Prediction - Target')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(133)
    plt.plot(x.numpy(), smooth_l1_values.numpy(), label='Smooth L1', color='green')
    plt.title('Smooth L1 Loss')
    plt.xlabel('Prediction - Target')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Note: Uncomment the following line to run the visualization
# plot_loss_functions()
print("Loss function visualization code is ready (uncomment plot_loss_functions() to view graphics)")

# 4. 类别不平衡的处理
print("\n4. 类别不平衡的处理：")

# 创建不平衡数据集
# 类别0: 80个样本, 类别1: 15个样本, 类别2: 5个样本
imbalanced_targets = torch.cat([
    torch.zeros(80, dtype=torch.long),
    torch.ones(15, dtype=torch.long),
    torch.full((5,), 2, dtype=torch.long)
])
imbalanced_logits = torch.randn(100, 3)

print("类别分布:", torch.bincount(imbalanced_targets))

# 普通交叉熵损失
normal_ce = nn.CrossEntropyLoss()
normal_loss = normal_ce(imbalanced_logits, imbalanced_targets)
print(f"普通交叉熵损失: {normal_loss.item():.6f}")

# 加权交叉熵损失
class_counts = torch.bincount(imbalanced_targets)
class_weights = 1.0 / class_counts.float()
class_weights = class_weights / class_weights.sum() * len(class_weights)  # 归一化
weighted_ce = nn.CrossEntropyLoss(weight=class_weights)
weighted_loss = weighted_ce(imbalanced_logits, imbalanced_targets)
print(f"加权交叉熵损失: {weighted_loss.item():.6f}")
print(f"类别权重: {class_weights}")

# 5. 自定义损失函数
print("\n5. 自定义损失函数：")

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 测试Focal Loss
focal_loss = FocalLoss(alpha=1, gamma=2)
focal_value = focal_loss(imbalanced_logits, imbalanced_targets)
print(f"Focal Loss: {focal_value.item():.6f}")

# 6. 损失函数的reduction参数
print("\n6. 损失函数的reduction参数：")

# 准备数据
predictions = torch.randn(10, 3)
targets = torch.randint(0, 3, (10,))

# 不同的reduction方式
loss_none = nn.CrossEntropyLoss(reduction='none')
loss_mean = nn.CrossEntropyLoss(reduction='mean')
loss_sum = nn.CrossEntropyLoss(reduction='sum')

loss_values_none = loss_none(predictions, targets)
loss_value_mean = loss_mean(predictions, targets)
loss_value_sum = loss_sum(predictions, targets)

print(f"reduction='none': {loss_values_none.shape}, 前3个值: {loss_values_none[:3]}")
print(f"reduction='mean': {loss_value_mean.item():.6f}")
print(f"reduction='sum': {loss_value_sum.item():.6f}")
print(f"手动验证mean: {loss_values_none.mean().item():.6f}")
print(f"手动验证sum: {loss_values_none.sum().item():.6f}")

# 7. 排序损失函数
print("\n7. 排序损失函数：")

# 准备排序数据
input1 = torch.randn(10, requires_grad=True)
input2 = torch.randn(10, requires_grad=True)
target = torch.ones(10)  # input1应该比input2大

# Margin Ranking Loss
margin_ranking_loss = nn.MarginRankingLoss(margin=1.0)
ranking_loss = margin_ranking_loss(input1, input2, target)
print(f"Margin Ranking Loss: {ranking_loss.item():.6f}")

# Hinge Embedding Loss
hinge_target = torch.randint(0, 2, (10,)) * 2 - 1  # -1 or 1
hinge_loss = nn.HingeEmbeddingLoss()
hinge_value = hinge_loss(input1, hinge_target.float())
print(f"Hinge Embedding Loss: {hinge_value.item():.6f}")

# 8. 序列到序列损失
print("\n8. 序列到序列损失：")

# CTC Loss for sequence-to-sequence tasks
# 注意：这里只演示CTC Loss的基本用法
T = 50      # Input sequence length
C = 20      # Number of classes (including blank)
N = 16      # Batch size
S = 30      # Target sequence length of longest target in batch
S_min = 10  # Minimum target length

# 创建模拟数据
input_lengths = torch.full((N,), T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

# 注意：实际使用时需要确保输入和目标的格式正确
log_probs = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
targets = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)

ctc_loss = nn.CTCLoss()
try:
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    print(f"CTC Loss: {loss.item():.6f}")
except Exception as e:
    print(f"CTC Loss示例出错 (这是正常的，因为随机数据): {e}")

# 9. 损失函数组合
print("\n9. 损失函数组合：")

class CombinedLoss(nn.Module):
    def __init__(self, loss_weights=None):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.weights = loss_weights or [1.0, 1.0]
    
    def forward(self, predictions, targets):
        mse = self.mse_loss(predictions, targets)
        l1 = self.l1_loss(predictions, targets)
        return self.weights[0] * mse + self.weights[1] * l1

# 测试组合损失
combined_loss = CombinedLoss(loss_weights=[0.7, 0.3])
pred = torch.randn(20, 1)
target = torch.randn(20, 1)
combined_value = combined_loss(pred, target)
print(f"组合损失: {combined_value.item():.6f}")

# 10. 损失函数在训练中的应用
print("\n10. 损失函数在训练中的应用：")

# 创建简单模型
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.linear(x)

# 准备数据
X = torch.randn(100, 10)
y = torch.randint(0, 3, (100,))

# 创建模型
model = SimpleClassifier(10, 3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 不同损失函数的训练对比
loss_functions = {
    'CrossEntropy': nn.CrossEntropyLoss(),
    'Focal': FocalLoss(alpha=1, gamma=2),
}

print("训练对比:")
for name, criterion in loss_functions.items():
    # 重新初始化模型
    model = SimpleClassifier(10, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    losses = []
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    print(f"{name}: 初始损失 {losses[0]:.4f}, 最终损失 {losses[-1]:.4f}")

# 11. 损失函数的梯度分析
print("\n11. 损失函数的梯度分析：")

def analyze_loss_gradients():
    # 创建需要梯度的预测值
    predictions = torch.tensor([0.0], requires_grad=True)
    target = torch.tensor([1.0])
    
    # 计算不同损失函数的梯度
    losses = {
        'MSE': nn.MSELoss(),
        'MAE': nn.L1Loss(),
        'Smooth L1': nn.SmoothL1Loss()
    }
    
    print("在预测值=0, 目标值=1时的梯度:")
    for name, loss_fn in losses.items():
        predictions.grad = None  # 清零梯度
        loss = loss_fn(predictions, target)
        loss.backward()
        print(f"{name}: 损失值 {loss.item():.6f}, 梯度 {predictions.grad.item():.6f}")

analyze_loss_gradients()

# 12. 实际应用建议
print("\n12. 实际应用建议：")

def loss_function_selector(task_type, data_characteristics):
    """损失函数选择器"""
    recommendations = {
        'regression': {
            'normal': 'MSELoss - 适用于高斯噪声',
            'outliers': 'L1Loss或SmoothL1Loss - 对异常值鲁棒',
            'mixed': 'HuberLoss - 结合MSE和MAE的优点'
        },
        'classification': {
            'balanced': 'CrossEntropyLoss - 标准选择',
            'imbalanced': 'WeightedCrossEntropyLoss或FocalLoss',
            'multilabel': 'BCEWithLogitsLoss'
        },
        'sequence': {
            'seq2seq': 'CrossEntropyLoss with masking',
            'ctc': 'CTCLoss - 用于序列对齐',
            'ranking': 'MarginRankingLoss'
        }
    }
    
    if task_type in recommendations:
        if data_characteristics in recommendations[task_type]:
            return recommendations[task_type][data_characteristics]
        else:
            return f"对于{task_type}任务，建议查看: {list(recommendations[task_type].keys())}"
    else:
        return "未知任务类型"

# 使用示例
print("损失函数选择建议:")
print("回归任务(正常数据):", loss_function_selector('regression', 'normal'))
print("分类任务(不平衡数据):", loss_function_selector('classification', 'imbalanced'))
print("序列任务(序列到序列):", loss_function_selector('sequence', 'seq2seq'))

print("\n损失函数使用要点:")
print("1. 根据任务类型选择合适的损失函数")
print("2. 考虑数据特性（平衡性、噪声等）")
print("3. 注意数值稳定性（如使用WithLogits版本）")
print("4. 合理设置reduction参数")
print("5. 对于特殊需求，考虑自定义损失函数")
print("6. 在训练过程中监控损失值的变化") 