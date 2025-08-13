# 语言模型架构

## 🎯 本节目标

理解不同Transformer架构的设计原理，掌握GPT与BERT的核心差异，解释为什么大模型偏爱Decoder-Only架构。

## 📖 阅读材料

### 核心概念文章
参考Attention机制页面的阅读材料，重点关注：
- Encoder-Only vs Decoder-Only架构对比
- GPT系列模型的演进
- BERT的双向编码特性

## 📝 知识总结

### 三大架构类型

| 架构类型 | 代表模型 | 特点 | 应用场景 |
|---------|---------|------|---------|
| **Encoder-Only** | BERT, RoBERTa | 双向注意力，适合理解任务 | 文本分类、阅读理解、情感分析 |
| **Decoder-Only** | GPT, LLaMA, ChatGPT | 因果注意力，适合生成任务 | 文本生成、对话、代码生成 |
| **Encoder-Decoder** | T5, BART | 编码-解码结构 | 翻译、摘要、问答 |

### Attention Mask对比

**BERT (双向注意力):**
```
Token:  [CLS] I    love  AI   [SEP]
Mask:   ✓     ✓    ✓     ✓    ✓
每个token都可以看到所有其他token
```

**GPT (因果注意力):**
```
Token:  I    love  AI    very  much
Mask:   ✓    
        ✓    ✓     
        ✓    ✓     ✓
        ✓    ✓     ✓     ✓
        ✓    ✓     ✓     ✓     ✓
每个token只能看到它之前的token（包括自己）
```

## 💬 面试问题解答

### Q1: Encoder-Only和Decoder-Only架构是什么？GPT和BERT分别是什么架构？

**简洁回答：**
- **Encoder-Only**: 使用双向注意力，可以同时看到前后文本，BERT就是这种架构
- **Decoder-Only**: 使用因果注意力，只能看到当前位置之前的文本，GPT系列都是这种架构

### Q1.1: 架构区别是什么？

**核心区别：**

1. **注意力机制**
   - Encoder-Only: 双向注意力，无掩码限制
   - Decoder-Only: 因果注意力，使用下三角掩码

2. **训练目标**
   - BERT: 掩码语言模型(MLM) + 下一句预测(NSP)
   - GPT: 自回归语言模型，预测下一个token

3. **位置编码**
   - BERT: 绝对位置编码
   - GPT: 绝对位置编码（早期）→ 相对位置编码（现代）

### Q1.2: 用途差异？

**BERT擅长理解任务：**
- 文本分类
- 情感分析  
- 阅读理解
- 实体识别

**GPT擅长生成任务：**
- 文本续写
- 对话生成
- 代码生成
- 创意写作

### Q1.3: 优劣比较

| 方面 | BERT优势 | GPT优势 |
|------|----------|---------|
| **理解能力** | 双向上下文，理解更深入 | 适合序列生成，逻辑连贯 |
| **训练效率** | 并行训练所有位置 | 自回归训练，简单稳定 |
| **推理速度** | 并行推理 | 需要逐步生成 |
| **应用灵活性** | 需要针对下游任务微调 | Zero-shot能力强 |

### Q2: 为什么主流大模型都用Decoder-Only架构？

**主要原因：**

1. **统一的生成范式**
   - 所有任务都可以转化为文本生成
   - 无需针对不同任务设计特殊架构

2. **更强的涌现能力**
   - 大规模预训练后展现出强大的few-shot学习能力
   - 指令跟随能力天然适合对话场景

3. **扩展性更好**
   - 架构简单，易于扩大模型规模
   - 训练稳定性更好

4. **应用场景更广**
   - 从对话到代码生成，一个模型搞定
   - 商业价值更高

### Q3: 简单讲解Qwen/LLaMA模型架构

**LLaMA架构特点：**

```
输入 → Token Embedding + 位置编码
     ↓
   N × Decoder Layer:
     ├── RMSNorm
     ├── Multi-Head Attention (RoPE位置编码)
     ├── 残差连接
     ├── RMSNorm  
     ├── SwiGLU FFN
     └── 残差连接
     ↓
   RMSNorm → 输出层 → 预测下一个token
```

**关键技术改进：**
- **RMSNorm**: 替代LayerNorm，计算更高效
- **RoPE位置编码**: 相对位置编码，支持更长序列
- **SwiGLU激活**: 替代ReLU，效果更好
- **Pre-Norm**: 归一化前置，训练更稳定

## 💻 代码实现

### 架构对比示例

```python
# BERT风格的双向注意力
class BidirectionalAttention(nn.Module):
    def forward(self, x):
        # 可以看到全部序列
        attn_mask = torch.ones(seq_len, seq_len)  # 全1矩阵
        return self.attention(x, mask=attn_mask)

# GPT风格的因果注意力  
class CausalAttention(nn.Module):
    def forward(self, x):
        # 只能看到当前位置之前
        seq_len = x.size(1)
        attn_mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角矩阵
        return self.attention(x, mask=attn_mask)
```

## ✅ 学习检验

- [ ] 能画出BERT和GPT的架构对比图
- [ ] 理解为什么GPT需要因果掩码
- [ ] 能解释主流大模型选择Decoder-Only的原因
- [ ] 掌握LLaMA/Qwen的关键技术改进点

## 🔗 相关链接

- [上一节：Attention机制](attention.md)
- [下一节：Attention升级技术](../attention-advanced/index.md)
- [返回：Transformer基础概览](index.md)