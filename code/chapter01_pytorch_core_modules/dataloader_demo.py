"""
torch.utils.data.DataLoader（数据加载器）核心原理与用法
--------------------------------------------------
第一性原理思考：
1. 什么是数据加载器？
   - DataLoader是PyTorch中处理数据批次的工具
   - 将数据集分批、打乱、并行加载
   - 提供高效的数据迭代接口

2. 为什么需要数据加载器？
   - 内存管理：支持大于内存的数据集
   - 批处理：提高GPU利用率和训练效率
   - 数据变换：集成数据预处理和增强
   - 多进程：支持并行数据加载

3. 数据加载器的核心特性是什么？
   - 自动批处理：将样本组织成批次
   - 数据打乱：随机化训练顺序
   - 并行加载：多进程数据预处理
   - 内存优化：懒加载和缓存管理

苏格拉底式提问与验证：
1. 批大小如何影响训练？
   - 问题：大批次和小批次有什么区别？
   - 验证：比较不同批大小的训练效果
   - 结论：批大小影响收敛速度和内存使用

2. 数据打乱为什么重要？
   - 问题：不打乱数据会有什么问题？
   - 验证：比较打乱和不打乱的训练效果
   - 结论：数据打乱避免训练偏差

3. 多进程加载的优势是什么？
   - 问题：并行加载如何提高效率？
   - 验证：比较单进程和多进程的加载速度
   - 结论：多进程可以减少数据加载的瓶颈

费曼学习法讲解：
1. 概念解释
   - 用传送带类比解释DataLoader的工作原理
   - 通过流水线理解数据处理过程
   - 强调DataLoader在训练流程中的重要性

2. 实例教学
   - 从简单到复杂的数据加载场景
   - 通过实际例子理解各种参数设置
   - 实践数据预处理和增强技术

3. 知识巩固
   - 总结DataLoader的最佳实践
   - 提供性能优化的建议
   - 建议进阶学习方向

功能说明：
- DataLoader 提供了对Dataset的封装，支持批处理、打乱、并行加载等功能。

原理讲解：
- 基于Dataset接口，自动处理数据的批次化和迭代。
- 支持多进程数据加载，可以并行进行数据预处理。
- 提供灵活的采样策略和数据变换机制。

工程意义：
- 是深度学习训练流程的基础设施，直接影响训练效率和数据处理能力。

可运行案例：
"""
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from torch.utils.data import random_split, WeightedRandomSampler
import numpy as np
import time
from PIL import Image
import os

# 1. 基本DataLoader使用
print("1. 基本DataLoader使用：")

# 创建简单数据集
x = torch.randn(1000, 10)
y = torch.randint(0, 2, (1000,))
dataset = TensorDataset(x, y)

# 创建DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"数据集大小: {len(dataset)}")
print(f"批次数量: {len(dataloader)}")

# 迭代数据
for batch_idx, (data, target) in enumerate(dataloader):
    print(f"批次 {batch_idx}: 数据形状 {data.shape}, 标签形状 {target.shape}")
    if batch_idx >= 2:  # 只显示前3个批次
        break

# 2. 自定义Dataset
print("\n2. 自定义Dataset：")

class CustomDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.data = torch.randn(size, 5)
        self.labels = torch.randint(0, 3, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx],
            'index': idx
        }
        return sample

# 使用自定义数据集
custom_dataset = CustomDataset(500)
custom_dataloader = DataLoader(custom_dataset, batch_size=16, shuffle=True)

print(f"自定义数据集大小: {len(custom_dataset)}")
for batch_idx, batch in enumerate(custom_dataloader):
    print(f"批次 {batch_idx}: 数据 {batch['data'].shape}, "
          f"標签 {batch['label'].shape}, 索引 {batch['index'].shape}")
    if batch_idx >= 1:
        break

# 3. 数据变换和预处理
print("\n3. 数据变换和预处理：")

class TransformDataset(Dataset):
    def __init__(self, size=1000, transform=None):
        self.size = size
        self.data = torch.randn(size, 3, 32, 32)  # 模拟图像数据
        self.labels = torch.randint(0, 10, (size,))
        self.transform = transform
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 定义变换函数
def normalize_transform(x):
    return (x - x.mean()) / (x.std() + 1e-8)

def noise_transform(x):
    return x + torch.randn_like(x) * 0.1

# 创建带变换的数据集
transform_dataset = TransformDataset(200, transform=normalize_transform)
transform_dataloader = DataLoader(transform_dataset, batch_size=8, shuffle=True)

print("变换前后的数据统计:")
original_batch = next(iter(DataLoader(TransformDataset(200), batch_size=8)))
transformed_batch = next(iter(transform_dataloader))

print(f"原始数据 - 均值: {original_batch[0].mean():.4f}, 标准差: {original_batch[0].std():.4f}")
print(f"变换后 - 均值: {transformed_batch[0].mean():.4f}, 标准差: {transformed_batch[0].std():.4f}")

# 4. 数据集分割
print("\n4. 数据集分割：")

# 创建大数据集
full_dataset = TensorDataset(torch.randn(1000, 20), torch.randint(0, 5, (1000,)))

# 分割数据集
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"训练批次数: {len(train_loader)}")
print(f"验证批次数: {len(val_loader)}")

# 5. 不平衡数据集的采样
print("\n5. 不平衡数据集的采样：")

# 创建不平衡数据集
n_samples = 1000
# 类别0: 800个样本, 类别1: 200个样本
labels = torch.cat([torch.zeros(800), torch.ones(200)]).long()
data = torch.randn(n_samples, 10)
imbalanced_dataset = TensorDataset(data, labels)

# 计算类别权重
class_counts = torch.bincount(labels)
print(f"类别分布: {class_counts}")

# 创建加权采样器
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

# 使用加权采样的DataLoader
balanced_loader = DataLoader(imbalanced_dataset, batch_size=64, sampler=sampler)

# 验证采样效果
sampled_labels = []
for batch_idx, (_, batch_labels) in enumerate(balanced_loader):
    sampled_labels.extend(batch_labels.tolist())
    if batch_idx >= 10:  # 只检查前几个批次
        break

sampled_counts = torch.bincount(torch.tensor(sampled_labels))
print(f"采样后的类别分布: {sampled_counts}")

# 6. 多进程数据加载
print("\n6. 多进程数据加载：")

class SlowDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 模拟耗时的数据处理
        time.sleep(0.001)  # 1ms延迟
        return torch.randn(10), torch.randint(0, 2, (1,))

slow_dataset = SlowDataset(100)

# 单进程加载
start_time = time.time()
single_loader = DataLoader(slow_dataset, batch_size=10, num_workers=0)
for batch in single_loader:
    pass
single_time = time.time() - start_time

# 多进程加载
start_time = time.time()
multi_loader = DataLoader(slow_dataset, batch_size=10, num_workers=2)
for batch in multi_loader:
    pass
multi_time = time.time() - start_time

print(f"单进程加载时间: {single_time:.4f}秒")
print(f"多进程加载时间: {multi_time:.4f}秒")
print(f"加速比: {single_time/multi_time:.2f}倍")

# 7. 批处理整理函数（collate_fn）
print("\n7. 批处理整理函数（collate_fn）：")

class VariableLengthDataset(Dataset):
    def __init__(self, size=100):
        self.size = size
        # 创建不同长度的序列
        self.data = [torch.randn(np.random.randint(5, 20)) for _ in range(size)]
        self.labels = torch.randint(0, 3, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def custom_collate_fn(batch):
    # 分离数据和标签
    sequences, labels = zip(*batch)
    
    # 填充序列到相同长度
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = torch.zeros(len(sequences), max_len)
    
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    
    labels = torch.stack(labels)
    
    return padded_sequences, labels

variable_dataset = VariableLengthDataset(50)
variable_loader = DataLoader(variable_dataset, batch_size=8, 
                           collate_fn=custom_collate_fn, shuffle=True)

print("变长序列批处理:")
for batch_idx, (data, labels) in enumerate(variable_loader):
    print(f"批次 {batch_idx}: 数据形状 {data.shape}, 标签形状 {labels.shape}")
    if batch_idx >= 1:
        break

# 8. 数据加载器的内存和性能优化
print("\n8. 数据加载器的内存和性能优化：")

# 使用pin_memory优化GPU传输
gpu_available = torch.cuda.is_available()
if gpu_available:
    print("GPU可用，测试pin_memory优化")
    
    dataset = TensorDataset(torch.randn(1000, 100), torch.randint(0, 10, (1000,)))
    
    # 不使用pin_memory
    normal_loader = DataLoader(dataset, batch_size=64, pin_memory=False)
    
    # 使用pin_memory
    pinned_loader = DataLoader(dataset, batch_size=64, pin_memory=True)
    
    device = torch.device('cuda')
    
    # 测试传输速度
    start_time = time.time()
    for data, target in normal_loader:
        data = data.to(device, non_blocking=False)
        target = target.to(device, non_blocking=False)
    normal_time = time.time() - start_time
    
    start_time = time.time()
    for data, target in pinned_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
    pinned_time = time.time() - start_time
    
    print(f"普通传输时间: {normal_time:.4f}秒")
    print(f"Pin memory传输时间: {pinned_time:.4f}秒")
else:
    print("GPU不可用，跳过pin_memory测试")

# 9. 数据加载器的错误处理和调试
print("\n9. 数据加载器的错误处理和调试：")

class ErrorProneDataset(Dataset):
    def __init__(self, size=100, error_rate=0.1):
        self.size = size
        self.error_rate = error_rate
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 随机引发错误
        if np.random.random() < self.error_rate:
            raise ValueError(f"模拟错误在索引 {idx}")
        
        return torch.randn(5), torch.randint(0, 2, (1,))

# 使用try-except处理数据加载错误
error_dataset = ErrorProneDataset(100, error_rate=0.05)
error_loader = DataLoader(error_dataset, batch_size=10, num_workers=0)

success_count = 0
error_count = 0

for batch_idx, batch in enumerate(error_loader):
    try:
        data, target = batch
        success_count += 1
        if batch_idx >= 5:  # 只测试前几个批次
            break
    except Exception as e:
        error_count += 1
        print(f"批次 {batch_idx} 发生错误: {e}")

print(f"成功加载批次: {success_count}")
print(f"错误批次: {error_count}")

# 10. 实际使用示例：完整的训练数据流水线
print("\n10. 实际使用示例：完整的训练数据流水线：")

class TrainingDataset(Dataset):
    def __init__(self, size=1000, input_dim=20, num_classes=5):
        self.size = size
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 生成合成数据
        self.data = torch.randn(size, input_dim)
        # 添加一些结构，使问题更有意义
        self.labels = (self.data.sum(dim=1) > 0).long() * torch.randint(0, num_classes, (size,))
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def create_data_loaders(dataset, train_ratio=0.8, batch_size=32):
    # 分割数据集
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2 if torch.get_num_threads() > 1 else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2 if torch.get_num_threads() > 1 else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

# 创建完整的数据流水线
full_dataset = TrainingDataset(2000, input_dim=50, num_classes=10)
train_loader, val_loader = create_data_loaders(full_dataset)

print("完整数据流水线信息:")
print(f"总数据集大小: {len(full_dataset)}")
print(f"训练集批次数: {len(train_loader)}")
print(f"验证集批次数: {len(val_loader)}")

# 模拟训练循环
def simulate_training_epoch(dataloader, epoch_name):
    total_samples = 0
    total_batches = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        total_samples += data.size(0)
        total_batches += 1
        
        # 这里会是实际的模型前向传播和反向传播
        # loss = model(data, target)
        # loss.backward()
        # optimizer.step()
        
        if batch_idx == 0:  # 显示第一个批次的信息
            print(f"{epoch_name} - 批次 {batch_idx}: "
                  f"数据形状 {data.shape}, 标签形状 {target.shape}")
    
    print(f"{epoch_name} - 总样本数: {total_samples}, 总批次数: {total_batches}")

print("\n模拟训练过程:")
simulate_training_epoch(train_loader, "训练集")
simulate_training_epoch(val_loader, "验证集")

# 数据加载器属性查看
print("\n数据加载器属性:")
print(f"训练加载器 - batch_size: {train_loader.batch_size}")
print(f"训练加载器 - shuffle: {train_loader.dataset}")
print(f"训练加载器 - num_workers: {train_loader.num_workers}")
print(f"训练加载器 - pin_memory: {train_loader.pin_memory}") 