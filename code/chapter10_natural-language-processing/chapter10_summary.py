"""
PyTorch 自然语言处理核心技术与实践
-------------------------------
【文件说明】
本文件系统梳理了自然语言处理深度学习的核心技术，包括：
- 词向量表示（Word2Vec）：将文本转换为数值向量
- 词向量应用（相似度与类比）：语义理解的基础
- 循环神经网络文本分析（RNN/LSTM）：序列建模的经典方法
- 卷积神经网络文本分析（CNN）：高效的文本分类方法
- 序列到序列学习（Seq2Seq）：机器翻译等生成任务

【第一性原理思考】
1. 为什么需要将文本转换为向量？
   - 计算机只能处理数值，不能直接理解文本
   - 向量表示能够编码语义信息，相似词语有相似向量
   - 向量运算可以捕捉词语间的语义关系

2. 语言模型的本质是什么？
   - 学习语言的概率分布P(w1, w2, ..., wn)
   - 预测下一个词：P(wt|w1, ..., wt-1)
   - 理解语言的组合性和上下文依赖性

3. 序列建模为什么重要？
   - 语言具有天然的序列性质
   - 词语的含义依赖于上下文
   - 序列信息是理解语义的关键

【苏格拉底式提问与验证】
1. 词向量如何编码语义信息？
   - 问题：为什么相似词的向量相似？
   - 验证：通过分布式假设验证词向量质量
   - 结论：上下文相似的词语语义相似

2. RNN与CNN在NLP中的优劣？
   - 问题：哪种架构更适合文本处理？
   - 验证：比较在不同任务上的性能
   - 结论：RNN善于序列建模，CNN善于局部特征提取

3. 注意力机制解决了什么问题？
   - 问题：序列模型的信息瓶颈在哪里？
   - 验证：比较有无注意力的翻译质量
   - 结论：注意力缓解了长序列的信息丢失

【费曼学习法讲解】
1. 概念解释
   - 用字典查词类比词向量查找
   - 用翻译过程类比序列到序列学习
   - 强调NLP技术在实际应用中的价值

2. 实例教学
   - 从简单的词袋模型开始
   - 逐步扩展到复杂的序列模型
   - 通过实际文本处理理解算法

【设计意义与工程价值】
- NLP是人工智能最具挑战性的领域之一
- 语言理解是通用人工智能的关键能力
- 掌握这些基础技术是理解现代语言模型（如GPT、BERT）的前提

可运行案例：
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import re
import random
from sklearn.manifold import TSNE

# 设置设备和随机种子
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
print(f"使用设备: {device}")

# 1. 文本预处理工具
# -----------------
# 原理说明：
# 文本预处理是NLP任务的第一步，包括分词、构建词汇表、文本清理等
# 好的预处理直接影响模型性能

print("1. 文本预处理工具：")

class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, min_freq=2, max_vocab_size=10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self.vocab_size = 0
        
    def tokenize(self, text):
        """简单的分词"""
        # 转小写，去除标点，分词
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def build_vocab(self, texts):
        """构建词汇表"""
        # 统计词频
        for text in texts:
            tokens = self.tokenize(text)
            self.word_freq.update(tokens)
        
        # 筛选高频词
        vocab_words = [word for word, freq in self.word_freq.most_common() 
                      if freq >= self.min_freq][:self.max_vocab_size]
        
        # 添加特殊token
        special_tokens = ['<UNK>', '<PAD>', '<SOS>', '<EOS>']
        vocab_words = special_tokens + vocab_words
        
        # 构建双向映射
        self.word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(vocab_words)
        
        print(f"词汇表大小: {self.vocab_size}")
        print(f"最高频词: {list(self.word_freq.most_common(10))}")
        
    def text_to_indices(self, text, max_length=None):
        """将文本转换为索引序列"""
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(token, self.word2idx['<UNK>']) 
                  for token in tokens]
        
        if max_length:
            if len(indices) < max_length:
                indices.extend([self.word2idx['<PAD>']] * (max_length - len(indices)))
            else:
                indices = indices[:max_length]
                
        return indices
    
    def indices_to_text(self, indices):
        """将索引序列转换为文本"""
        words = [self.idx2word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(word for word in words if word not in ['<PAD>', '<SOS>', '<EOS>'])

# 创建示例文本数据
sample_texts = [
    "The cat sat on the mat",
    "A dog ran in the park", 
    "The quick brown fox jumps over the lazy dog",
    "Natural language processing is fascinating",
    "Deep learning models can understand text",
    "Machine learning algorithms learn from data",
    "Neural networks are powerful computational models",
    "Text classification is a common NLP task"
]

# 演示文本预处理
preprocessor = TextPreprocessor(min_freq=1, max_vocab_size=50)
preprocessor.build_vocab(sample_texts)

# 测试文本转换
test_text = "The cat is running in the park"
indices = preprocessor.text_to_indices(test_text, max_length=10)
reconstructed = preprocessor.indices_to_text(indices)
print(f"原文: {test_text}")
print(f"索引: {indices}")
print(f"重构: {reconstructed}")

# 2. Word2Vec实现
# ---------------
# 原理说明：
# Word2Vec通过预测上下文学习词向量表示
# Skip-gram：用中心词预测上下文
# CBOW：用上下文预测中心词

print("\n2. Word2Vec实现：")

class SkipGramDataset(Dataset):
    """Skip-gram训练数据集"""
    
    def __init__(self, texts, preprocessor, window_size=2):
        self.preprocessor = preprocessor
        self.window_size = window_size
        self.pairs = self._create_pairs(texts)
    
    def _create_pairs(self, texts):
        """创建(中心词, 上下文词)对"""
        pairs = []
        for text in texts:
            indices = self.preprocessor.text_to_indices(text)
            for i, center in enumerate(indices):
                # 获取窗口内的上下文词
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if j != i:  # 排除中心词自己
                        pairs.append((center, indices[j]))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        center, context = self.pairs[idx]
        return torch.tensor(center), torch.tensor(context)

class SkipGramModel(nn.Module):
    """Skip-gram模型"""
    
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 中心词嵌入和上下文词嵌入
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.uniform_(self.center_embeddings.weight, -0.5/self.embedding_dim, 
                        0.5/self.embedding_dim)
        nn.init.uniform_(self.context_embeddings.weight, -0.5/self.embedding_dim, 
                        0.5/self.embedding_dim)
    
    def forward(self, center_words, context_words):
        """前向传播"""
        center_embeds = self.center_embeddings(center_words)  # [batch_size, embed_dim]
        context_embeds = self.context_embeddings(context_words)  # [batch_size, embed_dim]
        
        # 计算相似度得分
        scores = torch.sum(center_embeds * context_embeds, dim=1)
        return scores
    
    def get_word_embedding(self, word_idx):
        """获取词向量"""
        return self.center_embeddings.weight[word_idx]

# 创建Word2Vec训练数据
skipgram_dataset = SkipGramDataset(sample_texts, preprocessor, window_size=2)
skipgram_loader = DataLoader(skipgram_dataset, batch_size=32, shuffle=True)

print(f"Skip-gram训练对数量: {len(skipgram_dataset)}")

# 创建并训练Skip-gram模型
embedding_dim = 50
skipgram_model = SkipGramModel(preprocessor.vocab_size, embedding_dim).to(device)

# 负采样损失函数
class NegativeSamplingLoss(nn.Module):
    def __init__(self, vocab_size, num_negative=5):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_negative = num_negative
    
    def forward(self, pos_scores, center_words):
        # 正样本损失
        pos_loss = -F.logsigmoid(pos_scores).mean()
        
        # 负采样
        batch_size = center_words.size(0)
        neg_words = torch.randint(0, self.vocab_size, 
                                 (batch_size, self.num_negative)).to(center_words.device)
        
        # 计算负样本得分
        center_embeds = skipgram_model.center_embeddings(center_words).unsqueeze(1)
        neg_embeds = skipgram_model.context_embeddings(neg_words)
        neg_scores = torch.sum(center_embeds * neg_embeds, dim=2)
        
        # 负样本损失
        neg_loss = -F.logsigmoid(-neg_scores).mean()
        
        return pos_loss + neg_loss

# 训练Skip-gram模型
optimizer = torch.optim.Adam(skipgram_model.parameters(), lr=0.01)
criterion = NegativeSamplingLoss(preprocessor.vocab_size)

print("训练Skip-gram模型:")
skipgram_model.train()
for epoch in range(10):
    total_loss = 0
    for batch_idx, (center, context) in enumerate(skipgram_loader):
        center, context = center.to(device), context.to(device)
        
        optimizer.zero_grad()
        scores = skipgram_model(center, context)
        loss = criterion(scores, center)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx >= 5:  # 限制训练步数用于演示
            break
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss/(batch_idx+1):.4f}")

# 3. 词向量相似度与类比
# ---------------------
print("\n3. 词向量相似度与类比：")

class WordVectorAnalyzer:
    """词向量分析器"""
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.model.eval()
    
    def cosine_similarity(self, word1, word2):
        """计算余弦相似度"""
        if word1 not in self.preprocessor.word2idx or word2 not in self.preprocessor.word2idx:
            return 0.0
            
        idx1 = self.preprocessor.word2idx[word1]
        idx2 = self.preprocessor.word2idx[word2]
        
        vec1 = self.model.get_word_embedding(idx1)
        vec2 = self.model.get_word_embedding(idx2)
        
        similarity = F.cosine_similarity(vec1, vec2, dim=0)
        return similarity.item()
    
    def find_similar_words(self, word, top_k=5):
        """找到最相似的词"""
        if word not in self.preprocessor.word2idx:
            return []
        
        word_idx = self.preprocessor.word2idx[word]
        word_vec = self.model.get_word_embedding(word_idx)
        
        similarities = []
        for idx, other_word in self.preprocessor.idx2word.items():
            if idx != word_idx and other_word not in ['<UNK>', '<PAD>', '<SOS>', '<EOS>']:
                other_vec = self.model.get_word_embedding(idx)
                sim = F.cosine_similarity(word_vec, other_vec, dim=0).item()
                similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def word_analogy(self, word1, word2, word3, top_k=3):
        """词类比：word1 - word2 + word3 = ?"""
        words = [word1, word2, word3]
        if not all(word in self.preprocessor.word2idx for word in words):
            return []
        
        indices = [self.preprocessor.word2idx[word] for word in words]
        vecs = [self.model.get_word_embedding(idx) for idx in indices]
        
        # 计算类比向量
        analogy_vec = vecs[0] - vecs[1] + vecs[2]
        
        # 寻找最相似的词
        similarities = []
        for idx, word in self.preprocessor.idx2word.items():
            if (word not in words and 
                word not in ['<UNK>', '<PAD>', '<SOS>', '<EOS>']):
                word_vec = self.model.get_word_embedding(idx)
                sim = F.cosine_similarity(analogy_vec, word_vec, dim=0).item()
                similarities.append((word, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# 分析词向量
analyzer = WordVectorAnalyzer(skipgram_model, preprocessor)

# 测试词相似度
test_pairs = [('cat', 'dog'), ('learning', 'models'), ('the', 'a')]
print("词语相似度:")
for word1, word2 in test_pairs:
    similarity = analyzer.cosine_similarity(word1, word2)
    print(f"{word1} vs {word2}: {similarity:.4f}")

# 测试相似词查找
print(f"\n与'learning'最相似的词:")
similar_words = analyzer.find_similar_words('learning', top_k=3)
for word, sim in similar_words:
    print(f"{word}: {sim:.4f}")

# 4. 基于RNN的情感分析
# --------------------
# 原理说明：
# RNN能够处理变长序列，捕捉文本的序列信息
# LSTM解决了RNN的梯度消失问题
# 双向LSTM能够同时利用前向和后向信息

print("\n4. 基于RNN的情感分析：")

class SentimentDataset(Dataset):
    """情感分析数据集"""
    
    def __init__(self, texts, labels, preprocessor, max_length=50):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        indices = self.preprocessor.text_to_indices(text, self.max_length)
        
        return torch.tensor(indices), torch.tensor(label, dtype=torch.long)

class SentimentRNN(nn.Module):
    """基于RNN的情感分析模型"""
    
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes=2, 
                 num_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, bidirectional=bidirectional, 
                           dropout=dropout if num_layers > 1 else 0)
        
        # 计算LSTM输出维度
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
    def forward(self, x):
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # LSTM处理
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一个时间步的输出
        if self.lstm.bidirectional:
            # 拼接前向和后向的最后隐状态
            output = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            output = hidden[-1]
        
        # 分类
        output = self.dropout(output)
        output = self.fc(output)
        
        return output

# 创建模拟情感分析数据
positive_texts = [
    "This movie is amazing and wonderful",
    "I love this product it works great",
    "Fantastic experience highly recommend",
    "Excellent quality very satisfied",
    "Great service and fast delivery"
]

negative_texts = [
    "This movie is terrible and boring", 
    "I hate this product it doesnt work",
    "Awful experience would not recommend",
    "Poor quality very disappointed", 
    "Bad service and slow delivery"
]

# 构建情感数据集
sentiment_texts = positive_texts + negative_texts
sentiment_labels = [1] * len(positive_texts) + [0] * len(negative_texts)

# 为情感分析构建词汇表
sentiment_preprocessor = TextPreprocessor(min_freq=1)
sentiment_preprocessor.build_vocab(sentiment_texts)

# 创建数据集和数据加载器
sentiment_dataset = SentimentDataset(sentiment_texts, sentiment_labels, 
                                   sentiment_preprocessor, max_length=20)
sentiment_loader = DataLoader(sentiment_dataset, batch_size=4, shuffle=True)

# 创建RNN模型
rnn_model = SentimentRNN(
    vocab_size=sentiment_preprocessor.vocab_size,
    embed_dim=32,
    hidden_dim=64,
    num_classes=2,
    bidirectional=True
).to(device)

print(f"RNN模型参数数: {sum(p.numel() for p in rnn_model.parameters())}")

# 训练RNN模型
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

print("训练RNN情感分析模型:")
rnn_model.train()
for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0
    
    for texts, labels in sentiment_loader:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = rnn_model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    
    if epoch % 2 == 0:
        accuracy = 100. * correct / total
        print(f"Epoch {epoch}: Loss = {total_loss/len(sentiment_loader):.4f}, "
              f"Accuracy = {accuracy:.2f}%")

# 5. 基于CNN的文本分类
# --------------------
# 原理说明：
# CNN通过卷积操作捕捉局部特征
# 多尺度卷积核可以捕捉不同长度的n-gram特征
# 最大池化提取最显著的特征

print("\n5. 基于CNN的文本分类：")

class TextCNN(nn.Module):
    """基于CNN的文本分类模型"""
    
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, 
                 num_classes=2, dropout=0.5):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        
        # 多尺度卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) 
            for k in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        # 词嵌入 [batch_size, seq_len, embed_dim]
        embedded = self.embedding(x)
        
        # 增加通道维度用于卷积 [batch_size, 1, seq_len, embed_dim]
        embedded = embedded.unsqueeze(1)
        
        # 多尺度卷积和池化
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # [batch_size, num_filters, new_seq_len, 1]
            pooled = F.max_pool2d(conv_out, (conv_out.size(2), 1))  # [batch_size, num_filters, 1, 1]
            conv_outputs.append(pooled.squeeze(-1).squeeze(-1))  # [batch_size, num_filters]
        
        # 拼接不同尺度的特征
        if len(conv_outputs) > 1:
            concat_output = torch.cat(conv_outputs, dim=1)
        else:
            concat_output = conv_outputs[0]
        
        # 分类
        output = self.dropout(concat_output)
        output = self.fc(output)
        
        return output

# 创建CNN模型
cnn_model = TextCNN(
    vocab_size=sentiment_preprocessor.vocab_size,
    embed_dim=32,
    num_filters=64,
    filter_sizes=[2, 3, 4],  # 2-gram, 3-gram, 4-gram
    num_classes=2
).to(device)

print(f"CNN模型参数数: {sum(p.numel() for p in cnn_model.parameters())}")

# 训练CNN模型
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

print("训练CNN文本分类模型:")
cnn_model.train()
for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0
    
    for texts, labels in sentiment_loader:
        texts, labels = texts.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = cnn_model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    
    if epoch % 2 == 0:
        accuracy = 100. * correct / total
        print(f"Epoch {epoch}: Loss = {total_loss/len(sentiment_loader):.4f}, "
              f"Accuracy = {accuracy:.2f}%")

# 6. 序列到序列学习（Seq2Seq）
# ----------------------------
# 原理说明：
# Seq2Seq由编码器和解码器组成
# 编码器将输入序列编码为固定长度的向量
# 解码器根据编码向量生成输出序列

print("\n6. 序列到序列学习：")

class Encoder(nn.Module):
    """编码器"""
    
    def __init__(self, input_dim, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

class Decoder(nn.Module):
    """解码器"""
    
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden, cell):
        # x: [batch_size, 1]
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output)
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """序列到序列模型"""
    
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # 存储解码器输出
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 编码
        hidden, cell = self.encoder(src)
        
        # 解码器的第一个输入是<SOS>
        decoder_input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t] = output.squeeze(1)
            
            # Teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            decoder_input = trg[:, t].unsqueeze(1) if use_teacher_forcing else output.argmax(2)
        
        return outputs

# 创建简单的翻译数据（英文到数字）
translation_data = [
    ("one two three", "1 2 3"),
    ("three one two", "3 1 2"), 
    ("two three one", "2 3 1"),
    ("one three two", "1 3 2"),
    ("two one three", "2 1 3"),
    ("three two one", "3 2 1")
]

# 为翻译任务创建词汇表
src_texts = [pair[0] for pair in translation_data]
trg_texts = [pair[1] for pair in translation_data]

src_preprocessor = TextPreprocessor(min_freq=1)
src_preprocessor.build_vocab(src_texts)

trg_preprocessor = TextPreprocessor(min_freq=1)  
trg_preprocessor.build_vocab(trg_texts)

print(f"源语言词汇表大小: {src_preprocessor.vocab_size}")
print(f"目标语言词汇表大小: {trg_preprocessor.vocab_size}")

# 创建Seq2Seq模型
encoder = Encoder(src_preprocessor.vocab_size, 32, 64)
decoder = Decoder(trg_preprocessor.vocab_size, 32, 64)
seq2seq_model = Seq2Seq(encoder, decoder, device).to(device)

print(f"Seq2Seq模型参数数: {sum(p.numel() for p in seq2seq_model.parameters())}")

# 7. 注意力机制
# -------------
# 原理说明：
# 注意力机制允许模型在解码时关注输入序列的不同部分
# 缓解了序列到序列模型的信息瓶颈问题

print("\n7. 注意力机制：")

class Attention(nn.Module):
    """注意力机制"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # 重复解码器隐状态
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # 计算注意力能量
        energy = torch.tanh(self.attn(torch.cat([decoder_hidden, encoder_outputs], dim=2)))
        attention_weights = self.v(energy).squeeze(2)
        
        # softmax归一化
        attention_weights = F.softmax(attention_weights, dim=1)
        
        return attention_weights

class AttentionDecoder(nn.Module):
    """带注意力的解码器"""
    
    def __init__(self, output_dim, embed_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(output_dim, embed_dim)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x)
        
        # 计算注意力权重
        attention_weights = self.attention(hidden[-1], encoder_outputs)
        
        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        
        # 拼接嵌入和上下文
        lstm_input = torch.cat([embedded, context], dim=2)
        
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction = self.fc(output)
        
        return prediction, hidden, cell, attention_weights

# 8. 模型评估与可视化
# -------------------
print("\n8. 模型评估与可视化：")

class NLPModelEvaluator:
    """NLP模型评估器"""
    
    def __init__(self, model, preprocessor, device):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
    
    def evaluate_classification(self, test_texts, test_labels):
        """评估分类模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for text, label in zip(test_texts, test_labels):
                indices = torch.tensor(
                    self.preprocessor.text_to_indices(text, max_length=20)
                ).unsqueeze(0).to(self.device)
                
                output = self.model(indices)
                pred = output.argmax(dim=1).item()
                
                if pred == label:
                    correct += 1
                total += 1
        
        accuracy = 100. * correct / total
        print(f"测试准确率: {accuracy:.2f}%")
        return accuracy
    
    def predict_sentiment(self, text):
        """预测文本情感"""
        self.model.eval()
        with torch.no_grad():
            indices = torch.tensor(
                self.preprocessor.text_to_indices(text, max_length=20)
            ).unsqueeze(0).to(self.device)
            
            output = self.model(indices)
            probabilities = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probabilities[0, pred_class].item()
            
            sentiment = "Positive" if pred_class == 1 else "Negative"
            return sentiment, confidence

# 评估情感分析模型
test_texts = [
    "This is an excellent product I love it",
    "This is terrible I hate it completely",
    "The movie was okay not great but watchable"
]
test_labels = [1, 0, 0]  # 正面，负面，负面

print("评估RNN模型:")
rnn_evaluator = NLPModelEvaluator(rnn_model, sentiment_preprocessor, device)
rnn_evaluator.evaluate_classification(test_texts, test_labels)

print("\n评估CNN模型:")
cnn_evaluator = NLPModelEvaluator(cnn_model, sentiment_preprocessor, device)  
cnn_evaluator.evaluate_classification(test_texts, test_labels)

# 测试情感预测
print("\n情感预测示例:")
test_sentence = "This product is amazing and works perfectly"
sentiment, confidence = rnn_evaluator.predict_sentiment(test_sentence)
print(f"文本: '{test_sentence}'")
print(f"预测情感: {sentiment} (置信度: {confidence:.4f})")

print("\nNLP技术总结：")
print("1. 词向量是NLP的基础，将文本转换为数值表示")
print("2. Word2Vec通过上下文学习语义相似的词向量")
print("3. RNN善于处理序列数据，LSTM解决长依赖问题")
print("4. CNN能高效捕捉局部特征，适合文本分类")
print("5. Seq2Seq模型实现序列到序列的映射")
print("6. 注意力机制缓解信息瓶颈，提升长序列性能")
print("7. 不同架构适合不同任务，需要根据需求选择")

print(f"\n所有演示完成！使用设备: {device}") 