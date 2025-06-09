# PyTorch 自然语言处理（NLP）模块学习笔记

---

## 10.3_word2vec-pytorch.ipynb

### 1. 功能说明
介绍Word2Vec词向量的PyTorch实现，包括Skip-gram和CBOW两种结构。

### 2. 核心逻辑与原理细节
- **Skip-gram**：用中心词预测上下文。
- **CBOW**：用上下文预测中心词。
- **对比分析**：Skip-gram适合小数据，CBOW适合大数据。
- **实现要点**：
  ```python
  class SkipGram(nn.Module):
      def __init__(self, vocab_size, embedding_dim):
          super().__init__()
          self.embedding = nn.Embedding(vocab_size, embedding_dim)
          self.linear = nn.Linear(embedding_dim, vocab_size)
  ```

### 3. 应用场景
- 词向量训练、文本特征提取。
- 预训练语言模型的基础组件。

### 4. 调用关系
- 词向量模块常作为下游NLP任务的输入。
- 可与其他预训练方法(如BERT)集成。

---

## 10.6_similarity-analogy.ipynb

### 1. 功能说明
讲解如何利用预训练词向量进行近义词检索和类比推理。

### 2. 核心逻辑与原理细节
- **余弦相似度**：衡量词向量间的语义接近。
- **KNN检索**：查找最相似词。
- **类比推理**：利用向量运算实现"king-queen=man-woman"类比。
- **实现要点**：
  ```python
  def cosine_similarity(x, y):
      return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))
  
  def find_analogies(word1, word2, word3, embedding):
      v1, v2, v3 = map(lambda w: embedding[w], [word1, word2, word3])
      target = v2 - v1 + v3
      similarities = torch.matmul(embedding.weight, target) / torch.norm(target)
  ```

### 3. 应用场景
- 语义检索、词类比、NLP下游任务。
- 词向量质量评估与可视化。

### 4. 调用关系
- 词向量分析与可视化常用工具。
- 可用于词向量模型的评估流程。

---

## 10.7_sentiment-analysis-rnn.ipynb

### 1. 功能说明
使用RNN(LSTM/GRU)实现文本情感分析。

### 2. 核心逻辑与原理细节
- **双向LSTM**：捕捉双向上下文信息。
- **注意力机制**：关注重要词语。
- **实现要点**：
  ```python
  class SentimentRNN(nn.Module):
      def __init__(self, vocab_size, embedding_dim, hidden_dim):
          super().__init__()
          self.embedding = nn.Embedding(vocab_size, embedding_dim)
          self.lstm = nn.LSTM(embedding_dim, hidden_dim, 
                             bidirectional=True, batch_first=True)
          self.fc = nn.Linear(hidden_dim * 2, 2)  # 二分类
  ```

### 3. 应用场景
- 评论情感分析
- 社交媒体情绪识别
- 客户反馈分析

### 4. 调用关系
- 可与Word2Vec等词向量模型集成
- 常用于文本分类任务的基准模型

---

## 10.8_sentiment-analysis-cnn.ipynb

### 1. 功能说明
使用CNN实现文本情感分析，展示CNN在NLP中的应用。

### 2. 核心逻辑与原理细节
- **多尺度卷积**：捕捉不同长度的语言模式。
- **最大池化**：提取最显著特征。
- **实现要点**：
  ```python
  class TextCNN(nn.Module):
      def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes):
          super().__init__()
          self.embedding = nn.Embedding(vocab_size, embedding_dim)
          self.convs = nn.ModuleList([
              nn.Conv2d(1, n_filters, (k, embedding_dim)) 
              for k in filter_sizes
          ])
  ```

### 3. 应用场景
- 短文本分类
- 实时情感分析
- 高效文本处理

### 4. 调用关系
- 可替代RNN用于文本分类
- 适合并行计算加速

---

## 10.12_machine-translation.ipynb

### 1. 功能说明
实现基于Seq2Seq的神经机器翻译模型。

### 2. 核心逻辑与原理细节
- **编码器-解码器架构**：将源语言编码为向量，再解码为目标语言。
- **注意力机制**：动态关注源句子的相关部分。
- **实现要点**：
  ```python
  class Seq2SeqTranslator(nn.Module):
      def __init__(self, src_vocab_size, tgt_vocab_size, hidden_dim):
          super().__init__()
          self.encoder = Encoder(src_vocab_size, hidden_dim)
          self.decoder = Decoder(tgt_vocab_size, hidden_dim)
          self.attention = Attention(hidden_dim)
  ```

### 3. 应用场景
- 机器翻译
- 文本摘要
- 对话生成

### 4. 调用关系
- 可与预训练模型集成
- 常作为Transformer前身的基准模型

---

## 高层次总结
本章系统介绍了NLP的核心任务与技术，从基础的词向量表示到高级的机器翻译，涵盖了：
1. 词向量训练与应用(Word2Vec)
2. 语义分析与词向量评估
3. 文本分类的两种主流方法(RNN和CNN)
4. 序列到序列学习(机器翻译)

每个模块都提供了PyTorch实现示例，并讨论了实际应用场景。这些基础组件为构建更复杂的NLP系统奠定了基础。

## 实践建议
1. 词向量选择：
   - 小数据集优先使用Skip-gram
   - 大数据集可考虑CBOW
   - 考虑使用预训练词向量(如Word2Vec, GloVe)

2. 模型选择：
   - 需要捕捉长距离依赖 → 选择RNN/LSTM
   - 重视计算效率 → 选择CNN
   - 有充足计算资源 → 考虑预训练模型(如BERT)

3. 训练技巧：
   - 使用适当的词表大小和词向量维度
   - 注意梯度裁剪防止梯度爆炸
   - 采用学习率衰减策略
   - 使用批量归一化或层归一化提升训练稳定性 