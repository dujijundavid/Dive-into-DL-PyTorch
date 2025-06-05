# PyTorch 深度学习基础模块学习笔记

---

## 4.1_model-construction.ipynb

### 1. 功能说明
本文件介绍了如何在 PyTorch 中构建神经网络模型，包括自定义模型、使用内置容器（如 `Sequential`、`ModuleList`、`ModuleDict`）以及如何组合复杂模型。

### 2. 核心逻辑
- **自定义模型**：通过继承 `nn.Module`，实现 `__init__` 和 `forward` 方法，定义网络结构和前向传播。
- **Sequential 容器**：顺序堆叠多个层，简化模型定义。
- **ModuleList/ModuleDict**：用于存储多个子模块，便于动态添加和访问。
- **复杂模型嵌套**：支持模型嵌套与参数共享，展示了灵活的模型组合方式。

### 3. 应用场景
- 需要自定义网络结构时（如多层感知机、嵌套模型）。
- 需要灵活管理和组合多个子模块时。
- 适用于所有深度学习项目的模型搭建阶段。

### 4. 调用关系
- 主要为独立使用，作为模型定义的基础模块。
- 其中的自定义类可被其他训练、推理脚本导入和复用。

---

## 4.2_parameters.ipynb

### 1. 功能说明
本文件讲解了如何访问、初始化和共享模型参数，包括参数的读取、梯度查看、自定义初始化方法和参数共享机制。

### 2. 核心逻辑
- **参数访问**：通过 `named_parameters()`、`parameters()` 获取模型各层参数。
- **参数初始化**：使用 `torch.nn.init` 提供的多种初始化方法，或自定义初始化函数。
- **自定义参数**：通过 `nn.Parameter` 显式声明可训练参数。
- **参数共享**：同一个层对象可被多次使用，实现参数共享。

### 3. 应用场景
- 需要自定义参数初始化策略时（如特殊分布、稀疏初始化）。
- 需要实现参数共享（如权重共享的神经网络结构）。
- 训练调优和模型分析阶段。

### 4. 调用关系
- 可独立使用，也常作为模型训练前的参数准备环节。
- 参数共享机制对后续模型训练有直接影响。

---

## 4.4_custom-layer.ipynb

### 1. 功能说明
本文件介绍了如何自定义 PyTorch 层，包括不含参数的层（如中心化层）和含参数的层（如自定义全连接层）。

### 2. 核心逻辑
- **无参数层**：继承 `nn.Module`，只需实现 `forward`，如对输入做中心化处理。
- **有参数层**：通过 `nn.ParameterList` 或 `nn.ParameterDict` 管理多个参数，实现灵活的参数结构。
- **层的组合**：自定义层可与标准层组合，构建更复杂的网络。

### 3. 应用场景
- 需要实现特殊功能的网络层（如归一化、特定变换）。
- 需要灵活管理参数结构的自定义层。
- 适用于模型创新和论文复现。

### 4. 调用关系
- 主要为独立定义，作为自定义层可被其他模型引用。

---

## 4.5_read-write.ipynb

### 1. 功能说明
本文件讲解了如何保存和加载 PyTorch 的张量和模型，包括 `Tensor`、模型参数（`state_dict`）和优化器状态的读写。

### 2. 核心逻辑
- **Tensor 读写**：使用 `torch.save` 和 `torch.load` 保存/加载张量、列表、字典等。
- **模型参数保存与加载**：通过 `state_dict` 保存模型参数，支持断点续训和模型迁移。
- **优化器状态保存**：保存优化器的状态字典，便于恢复训练。

### 3. 应用场景
- 训练过程中的模型断点保存与恢复。
- 模型迁移、部署和推理。
- 结果复现和实验管理。

### 4. 调用关系
- 独立使用，常与训练、推理脚本配合。

---

## 4.6_use-gpu.ipynb

### 1. 功能说明
本文件介绍了如何在 PyTorch 中使用 GPU 进行加速计算，包括设备管理、Tensor 和模型的 GPU 操作。

### 2. 核心逻辑
- **设备查询与管理**：检测 GPU 是否可用、数量、名称等。
- **Tensor 的 GPU 运算**：将张量移动到 GPU，进行加速计算。
- **模型的 GPU 运算**：将模型参数迁移到 GPU，支持多卡训练。

### 3. 应用场景
- 需要加速大规模神经网络训练和推理时。
- 多 GPU 训练、分布式训练场景。
- 适用于所有深度学习项目的性能优化阶段。

### 4. 调用关系
- 独立使用，作为训练和推理脚本的硬件加速基础。

---

## 高层次总结

本目录下的各个文件协同构成了 PyTorch 深度学习项目的基础模块：

- **模型构建（4.1）**：教会你如何灵活定义和组合神经网络结构。
- **参数管理（4.2）**：帮助你理解和控制模型参数的访问、初始化与共享。
- **自定义层（4.4）**：为模型创新和特殊需求提供扩展能力。
- **模型保存与加载（4.5）**：保障训练过程的可恢复性和模型的可迁移性。
- **GPU 加速（4.6）**：为大规模计算提供高效的硬件支持。

这些模块既可独立学习，也可串联使用，帮助初学者系统掌握 PyTorch 的核心用法，为后续的深度学习项目开发打下坚实基础。

---

## Python 代码示例（结构化调用关系）

```python
import torch
from torch import nn

# 1. 构建模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)
    def forward(self, x):
        return self.output(self.act(self.hidden(x)))

# 2. 参数初始化
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

net = MLP()
net.apply(init_weights)

# 3. 自定义层
class CenteredLayer(nn.Module):
    def forward(self, x):
        return x - x.mean()

# 4. 保存与加载
torch.save(net.state_dict(), 'mlp_params.pt')
net2 = MLP()
net2.load_state_dict(torch.load('mlp_params.pt'))

# 5. GPU 加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net2.to(device)
x = torch.rand(2, 784).to(device)
y = net2(x)
``` 