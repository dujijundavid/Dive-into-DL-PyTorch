# PyTorch 调试指南 🔧

> 通过第一性原理分析 PyTorch 常见问题，提供可复现的最小代码片段和 LLM 友好的调试模板

## 📁 文件夹结构

```
pytorch-debugging-guide/
├── README.md                          # 本文件
├── common-errors/                     # 常见错误分类
│   ├── gradient-issues/              # 梯度相关问题
│   ├── dimension-mismatch/           # 维度不匹配
│   ├── convergence-problems/         # 收敛问题
│   ├── memory-issues/               # 内存相关
│   ├── data-loading/                # 数据加载问题
│   └── model-architecture/          # 模型架构问题
├── debug-templates/                  # LLM 调试模板
│   ├── error-analysis-template.md   # 错误分析模板
│   ├── performance-debug-template.md # 性能调试模板
│   └── model-debug-template.md      # 模型调试模板
├── minimal-reproducible-examples/   # 最小可复现示例
└── troubleshooting-workflows/       # 系统化调试流程

## 🎯 使用方法

### 1. 快速定位问题类型
- 看到 NaN/Inf → `gradient-issues/`
- 维度错误信息 → `dimension-mismatch/`
- 模型不收敛 → `convergence-problems/`
- CUDA OOM → `memory-issues/`

### 2. 使用调试模板
复制 `debug-templates/` 中的相应模板，填入你的错误信息，直接与 LLM 对话。

### 3. 参考最小示例
每个问题都提供可直接运行的最小代码片段，方便快速复现和理解。

## 🧠 第一性原理方法论

我们坚持从 **根本原理** 出发分析问题：
1. **现象** → **机制** → **解决方案**
2. 不只是 "怎么修"，更要理解 "为什么错"
3. 培养系统性调试思维，而非零散的经验积累

## 🤖 LLM 友好设计

- 每个错误都有标准化的描述模板
- 提供完整的上下文信息格式
- 设计渐进式提问策略，逐步深入问题本质

## 🚀 快速开始

1. 遇到错误时，先查看错误信息关键词
2. 找到对应的 `common-errors/` 子目录
3. 阅读问题分析和解决方案
4. 使用 `debug-templates/` 与 LLM 交互
5. 参考 `minimal-reproducible-examples/` 验证理解

---
*构建系统化的 PyTorch 调试能力，让每个错误都成为学习的机会* 