# Chapter 03 深度学习基础：代码结构与学习笔记

## 目录概览

| 文件名 | 用途简述 |
| ------ | -------- |
| 3.1_vector_and_broadcasting_notes.py | 向量操作与广播机制的PyTorch基础与优化 |
| 3.1_linear-regression.ipynb | 线性回归基础与向量操作演示 |
| 3.2_linear-regression-scratch.ipynb | 线性回归的从零实现 |
| 3.3_linear-regression-pytorch.ipynb | 线性回归的PyTorch简洁实现 |
| 3.5_fashion-mnist.ipynb | Fashion-MNIST数据集加载与可视化 |
| 3.6_softmax-regression-scratch.ipynb | softmax回归的从零实现 |
| 3.7_softmax-regression-pytorch.ipynb | softmax回归的PyTorch简洁实现 |
| 3.8_mlp.ipynb | 多层感知机（MLP）原理与实现 |
| 3.9_mlp-scratch.ipynb | MLP的从零实现 |
| 3.10_mlp-pytorch.ipynb | MLP的PyTorch简洁实现 |
| 3.11_underfit-overfit.ipynb | 欠拟合与过拟合实验 |
| 3.12_weight-decay.ipynb | 权重衰减（L2正则化）实验 |
| 3.13_dropout.ipynb | Dropout正则化实验 |
| 3.16_kaggle-house-price.ipynb | Kaggle房价预测实战 |
| submission.csv | Kaggle房价预测的提交样例 |

---

## 详细分析

### 1. 3.1_vector_and_broadcasting_notes.py
- **功能**：讲解PyTorch向量化操作、广播机制、内存与性能优化、调试技巧。
- **核心逻辑**：通过代码对比循环与向量化效率，演示广播机制原理与最佳实践，介绍inplace操作、内存连续性、设备选择等。
- **应用场景**：理解和优化PyTorch张量操作，提升代码效率和可读性。
- **调用关系**：独立脚本，适合基础学习和查阅。

### 2. 3.1_linear-regression.ipynb
- **功能**：线性回归基础，向量操作与广播机制演示。
- **核心逻辑**：对比循环与向量加法效率，展示广播机制。
- **应用场景**：初学者理解线性代数在深度学习中的作用。
- **调用关系**：为后续线性回归实现做理论铺垫。

### 3. 3.2_linear-regression-scratch.ipynb
- **功能**：线性回归的从零实现（不依赖高阶API）。
- **核心逻辑**：
  1. 生成合成数据
  2. 小批量数据读取
  3. 初始化参数
  4. 定义模型、损失函数、优化算法
  5. 训练循环（前向、反向、参数更新）
- **应用场景**：理解深度学习底层实现原理。
- **调用关系**：为后续PyTorch实现做铺垫。

### 4. 3.3_linear-regression-pytorch.ipynb
- **功能**：线性回归的PyTorch简洁实现。
- **核心逻辑**：
  1. 使用`nn.Module`或`nn.Sequential`定义模型
  2. 使用`DataLoader`加载数据
  3. 使用`optim`优化器和`nn.MSELoss`
  4. 训练循环
- **应用场景**：实际项目中快速实现线性回归。
- **调用关系**：对比手写实现与框架实现的差异。

### 5. 3.5_fashion-mnist.ipynb
- **功能**：Fashion-MNIST数据集加载、可视化、标签映射。
- **核心逻辑**：
  1. 使用`torchvision.datasets`下载数据
  2. 数据可视化与标签解释
- **应用场景**：图像分类任务的数据准备与探索。
- **调用关系**：为后续分类模型提供数据基础。

### 6. 3.6_softmax-regression-scratch.ipynb
- **功能**：softmax回归的从零实现。
- **核心逻辑**：
  1. 参数初始化
  2. softmax函数实现
  3. 交叉熵损失
  4. 训练与评估
- **应用场景**：多分类问题的基础建模。
- **调用关系**：为后续softmax回归PyTorch实现做铺垫。

### 7. 3.7_softmax-regression-pytorch.ipynb
- **功能**：softmax回归的PyTorch简洁实现。
- **核心逻辑**：
  1. 使用`nn.Sequential`定义模型
  2. 使用`nn.CrossEntropyLoss`
  3. 使用`optim.SGD`
  4. 训练与评估
- **应用场景**：实际多分类任务的快速实现。
- **调用关系**：对比手写与框架实现。

### 8. 3.8_mlp.ipynb
- **功能**：多层感知机（MLP）原理、激活函数、训练流程。
- **核心逻辑**：
  1. 激活函数（ReLU、sigmoid等）可视化
  2. MLP结构与前向传播
  3. 训练与评估
- **应用场景**：理解深度神经网络的基本结构。
- **调用关系**：为MLP的手写和PyTorch实现做理论基础。

### 9. 3.9_mlp-scratch.ipynb
- **功能**：MLP的从零实现。
- **核心逻辑**：
  1. 参数初始化
  2. 手写ReLU激活
  3. 前向传播
  4. 训练循环
- **应用场景**：深入理解MLP的底层实现。
- **调用关系**：对比PyTorch实现。

### 10. 3.10_mlp-pytorch.ipynb
- **功能**：MLP的PyTorch简洁实现。
- **核心逻辑**：
  1. 使用`nn.Sequential`搭建MLP
  2. 初始化参数
  3. 训练与评估
- **应用场景**：实际项目中快速搭建MLP。
- **调用关系**：对比手写与框架实现。

### 11. 3.11_underfit-overfit.ipynb
- **功能**：模型选择、欠拟合与过拟合实验。
- **核心逻辑**：
  1. 多项式拟合实验
  2. 训练/测试损失曲线可视化
  3. 欠拟合/过拟合现象分析
- **应用场景**：模型复杂度调优、正则化需求分析。
- **调用关系**：为后续正则化实验做铺垫。

### 12. 3.12_weight-decay.ipynb
- **功能**：权重衰减（L2正则化）实验。
- **核心逻辑**：
  1. L2范数惩罚项实现
  2. 对比有无正则化的训练效果
- **应用场景**：防止过拟合，提高泛化能力。
- **调用关系**：与欠拟合/过拟合实验结合。

### 13. 3.13_dropout.ipynb
- **功能**：Dropout正则化实验。
- **核心逻辑**：
  1. Dropout原理与实现
  2. 对比有无Dropout的训练效果
- **应用场景**：深度网络防止过拟合。
- **调用关系**：与MLP、正则化实验结合。

### 14. 3.16_kaggle-house-price.ipynb
- **功能**：Kaggle房价预测实战。
- **核心逻辑**：
  1. 数据读取与预处理
  2. 特征工程
  3. 模型训练与预测
  4. 结果提交
- **应用场景**：实际数据科学竞赛流程。
- **调用关系**：综合应用前面所有知识点。

### 15. submission.csv
- **功能**：Kaggle房价预测的结果提交样例。
- **应用场景**：Kaggle竞赛结果上传。

---

## 高层次总结：组件协同工作

本章代码以"从零实现"与"PyTorch简洁实现"双线并行，帮助初学者既理解底层原理，又掌握高效开发。数据加载、模型定义、训练循环、评估与正则化等模块环环相扣，逐步引导你完成从基础理论到实际项目的完整流程。每个实验都围绕"理论-实现-对比-优化"展开，强调可复用模式（如`nn.Sequential`、`DataLoader`、自定义训练循环）和最佳实践（如正则化、批量处理、调试技巧），为后续更复杂的深度学习任务打下坚实基础。

---

如需对某个文件或主题深入讲解，请随时告知！ 