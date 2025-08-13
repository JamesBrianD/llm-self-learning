# 编码器-解码器架构

## 🎯 本节目标

深入理解原始Transformer的编码器-解码器架构，掌握三种注意力机制的区别和作用。

## 📝 知识总结

### 整体架构概览

原始Transformer采用**编码器-解码器(Encoder-Decoder)**架构，专门用于序列到序列的转换任务。

```
输入序列 → 编码器 → 编码表示 → 解码器 → 输出序列
```

#### 核心组件
1. **编码器(Encoder)**: 将输入序列编码为高层语义表示
2. **解码器(Decoder)**: 基于编码表示自回归生成输出序列
3. **三种注意力机制**: 自注意力、掩码自注意力、交叉注意力

### 编码器 (Encoder) 详解

#### 结构组成
- **N层相同的层**(原论文N=6)
- 每层包含两个子层：
  1. 多头自注意力机制
  2. 前馈神经网络
- 每个子层使用残差连接和层归一化

#### 数学表示
```
# 编码器层的计算过程
def encoder_layer(x):
    # 多头自注意力
    attn_output = multi_head_attention(x, x, x)
    x = layer_norm(x + attn_output)  # 残差连接 + 层归一化
    
    # 前馈网络
    ffn_output = feed_forward(x)
    x = layer_norm(x + ffn_output)   # 残差连接 + 层归一化
    
    return x
```

#### 核心特征

**1. 并行处理**
- 可以同时处理整个输入序列
- 每个位置都能看到所有其他位置
- 训练效率高，无序列计算依赖

**2. 双向上下文**
- 自注意力机制允许每个位置关注整个序列
- 能够捕获全局的上下文信息
- 适合理解类任务

**3. 语义提升**
- 将低阶词向量转换为高阶语义表示
- 多层堆叠逐步抽象语义信息
- 最终输出包含丰富的上下文信息

### 解码器 (Decoder) 详解

#### 结构组成
- **N层相同的层**(原论文N=6)
- 每层包含三个子层：
  1. 掩码多头自注意力
  2. 编码器-解码器交叉注意力
  3. 前馈神经网络
- 每个子层使用残差连接和层归一化

#### 数学表示
```
# 解码器层的计算过程
def decoder_layer(x, encoder_output):
    # 1. 掩码自注意力
    masked_attn = masked_multi_head_attention(x, x, x)
    x = layer_norm(x + masked_attn)
    
    # 2. 交叉注意力
    cross_attn = multi_head_attention(
        query=x, 
        key=encoder_output, 
        value=encoder_output
    )
    x = layer_norm(x + cross_attn)
    
    # 3. 前馈网络
    ffn_output = feed_forward(x)
    x = layer_norm(x + ffn_output)
    
    return x
```

#### 核心特征

**1. 自回归生成**
- 逐步生成输出序列
- 当前位置只能看到之前的位置
- 使用掩码机制防止信息泄露

**2. 双输入机制**
- 输入1：解码器之前的输出(自回归)
- 输入2：编码器的输出表示(交叉注意力)
- 结合自身历史和源序列信息

### 三种注意力机制详解

#### 1. 编码器自注意力 (Encoder Self-Attention)

**作用**: 让编码器的每个位置关注输入序列的所有位置

```python
# 编码器自注意力
Q = K = V = encoder_input  # 都来自输入序列
attention_output = Attention(Q, K, V)
```

**特点**:
- 无掩码限制，可以看到全序列
- 建立输入序列内部的依赖关系
- 捕获长距离依赖

#### 2. 解码器掩码自注意力 (Masked Self-Attention)

**作用**: 让解码器的每个位置只关注之前的位置

```python
# 掩码自注意力
Q = K = V = decoder_input  # 都来自解码器输入
mask = create_causal_mask(seq_len)  # 下三角掩码
attention_output = Attention(Q, K, V, mask=mask)
```

**掩码机制**:
```
位置:  0  1  2  3
掩码: [1  0  0  0]  # 位置0只能看自己
      [1  1  0  0]  # 位置1能看0,1
      [1  1  1  0]  # 位置2能看0,1,2
      [1  1  1  1]  # 位置3能看0,1,2,3
```

#### 3. 编码器-解码器交叉注意力 (Cross-Attention)

**作用**: 让解码器关注编码器的输出，实现序列对齐

```python
# 交叉注意力
Q = decoder_hidden        # 查询来自解码器
K = V = encoder_output    # 键值来自编码器
attention_output = Attention(Q, K, V)
```

**工作原理**:
- Query：解码器想要什么信息
- Key：编码器有什么信息
- Value：编码器提供的具体信息
- 实现源序列和目标序列的对齐

### 架构优势与应用

#### 优势特点

**1. 并行训练**
- 编码器可以并行处理整个输入
- 解码器在训练时也可以并行(Teacher Forcing)
- 相比RNN训练速度大幅提升

**2. 长距离依赖**
- 注意力机制直接连接任意两个位置
- 避免了RNN的梯度传播问题
- 更好地捕获长距离依赖关系

**3. 可解释性**
- 注意力权重提供模型决策的可视化
- 可以看到模型关注的输入部分
- 便于分析和调试

#### 典型应用

**1. 机器翻译**
- 原始Transformer的设计目标
- 编码器理解源语言，解码器生成目标语言
- 通过交叉注意力实现语言对齐

**2. 文本摘要**
- 编码器理解原文，解码器生成摘要
- 交叉注意力选择重要信息
- 控制摘要长度和内容

**3. 对话系统**
- 编码器理解用户输入
- 解码器生成回复
- 维持对话上下文

### 现代发展趋势

#### 架构演进

**Encoder-Only**:
- 代表: BERT, RoBERTa
- 擅长: 理解任务(分类、阅读理解)
- 特点: 双向注意力，并行训练

**Decoder-Only**:
- 代表: GPT, LLaMA, ChatGPT
- 擅长: 生成任务(对话、写作)
- 特点: 因果注意力，统一范式

**为什么Decoder-Only成为主流？**
1. **任务统一**: 所有任务都可以表述为生成问题
2. **扩展性好**: 更容易扩展到大规模
3. **涌现能力**: 大规模后展现强大的few-shot能力
4. **工程简化**: 架构更简单，易于优化

## 💬 面试问题解答

### Q1: 编码器和解码器的主要区别是什么？

**核心区别**:

| 维度 | 编码器 | 解码器 |
|------|--------|--------|
| **注意力类型** | 双向自注意力 | 单向掩码自注意力 + 交叉注意力 |
| **处理方式** | 并行处理 | 自回归生成 |
| **输入来源** | 原始输入序列 | 前一步输出 + 编码器输出 |
| **主要功能** | 理解和编码 | 生成和解码 |

### Q2: 交叉注意力的工作原理是什么？

**工作机制**:
1. **Query来自解码器**: 表示"我想要什么信息"
2. **Key/Value来自编码器**: 表示"可以提供什么信息"
3. **注意力计算**: 计算解码器对编码器每个位置的关注度
4. **信息融合**: 根据注意力权重聚合编码器信息

**数学过程**:
$$CrossAttention = Attention(Q_{decoder}, K_{encoder}, V_{encoder})$$

### Q3: 为什么解码器需要掩码自注意力？

**核心原因**: 防止信息泄露

**详细解释**:
1. **训练时**: 使用Teacher Forcing，模型能看到完整目标序列
2. **推理时**: 只能看到已生成的部分
3. **一致性要求**: 训练和推理的可见信息必须一致
4. **掩码作用**: 在训练时人为限制可见范围

**代码示例**:
```python
def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.masked_fill(mask == 0, float('-inf'))
```

### Q4: 为什么现在更流行Decoder-Only架构？

**主要原因**:

1. **统一性**: 
   - 所有任务都可以转化为生成任务
   - 分类 → 生成类别标签
   - 问答 → 生成答案

2. **扩展性**:
   - 架构简单，易于扩大规模
   - 训练更稳定，参数利用率高

3. **涌现能力**:
   - 大规模训练后展现强大的zero/few-shot能力
   - 指令跟随、上下文学习等能力

4. **工程优势**:
   - 实现简单，优化成熟
   - 推理效率高(KV Cache等技术)

## 💻 代码实现

### 完整Encoder-Decoder实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderDecoderTransformer(nn.Module):
    """完整的Encoder-Decoder Transformer实现"""
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, 
                 num_heads=8, num_layers=6, d_ff=2048, max_seq_len=1000):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 编码器和解码器
        self.encoder = Encoder(d_model, num_heads, num_layers, d_ff)
        self.decoder = Decoder(d_model, num_heads, num_layers, d_ff)
        
        # 输出投影层
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 源序列编码
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        
        # 目标序列编码
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        # 编码器
        encoder_output = self.encoder(src_emb, src_mask)
        
        # 解码器
        decoder_output = self.decoder(tgt_emb, encoder_output, tgt_mask, src_mask)
        
        # 输出投影
        output = self.output_projection(decoder_output)
        
        return output

class Encoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class EncoderLayer(nn.Module):
    """编码器层"""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class Decoder(nn.Module):
    """Transformer解码器"""
    
    def __init__(self, d_model, num_heads, num_layers, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, src_mask)
        return x

class DecoderLayer(nn.Module):
    """解码器层"""
    
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # 掩码自注意力子层
        masked_attn = self.masked_self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(masked_attn))
        
        # 交叉注意力子层
        cross_attn = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn))
        
        # 前馈网络子层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

def create_causal_mask(seq_len):
    """创建因果掩码(下三角矩阵)"""
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.masked_fill(mask == 0, float('-inf'))

def create_padding_mask(seq, pad_token=0):
    """创建填充掩码"""
    return (seq != pad_token).unsqueeze(1).unsqueeze(2)

# 使用示例
def demo_encoder_decoder():
    """演示编码器-解码器的使用"""
    
    # 模型参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    seq_len = 20
    batch_size = 2
    
    # 创建模型
    model = EncoderDecoderTransformer(
        src_vocab_size, tgt_vocab_size, d_model
    )
    
    # 模拟数据
    src = torch.randint(1, src_vocab_size, (batch_size, seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, seq_len))
    
    # 创建掩码
    tgt_mask = create_causal_mask(seq_len)
    src_mask = create_padding_mask(src)
    
    # 前向传播
    output = model(src, tgt, src_mask, tgt_mask)
    
    print(f"输入形状: {src.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

if __name__ == "__main__":
    demo_encoder_decoder()
```

## ✅ 学习检验

- [ ] 理解编码器-解码器架构的整体设计
- [ ] 掌握三种注意力机制的区别和作用
- [ ] 理解掩码机制的必要性和实现
- [ ] 能够实现完整的Encoder-Decoder模型
- [ ] 理解为什么现代模型偏向Decoder-Only

## 🔗 相关链接

- [上一节：前馈神经网络](ffn.md)
- [下一节：Attention升级技术](../attention-advanced/index.md)
- [返回：Transformer基础概览](index.md)