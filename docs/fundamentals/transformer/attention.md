# Attention机制

## 🎯 本节目标

深入理解Self-Attention机制的数学原理和计算过程，掌握面试中的高频问题。

## 📖 阅读材料

### 必读文章
1. [《Attention is All You Need》浅读（简介+代码）](https://kexue.fm/archives/4765/comment-page-1#comments) - 科学空间
2. [GPT与BERT差别深入解析](https://www.zhihu.com/tardis/zm/art/607605399?source_id=1003) - 知乎
3. [深入浅出理解transformer](https://www.zhihu.com/question/471328838/answer/51545078376) - 知乎  
4. [Transformer模型详解（图解最完整版）](https://zhuanlan.zhihu.com/p/338817680) - 知乎
5. [Transformer位置编码（基础）](https://zhuanlan.zhihu.com/p/631363482) - 知乎

### 原论文
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - 经典必读

## 🎬 视频材料

> **学习建议：** 倍速观看，重点理解概念而非细节

1. [GPT，GPT-2，GPT-3 论文精读](https://www.bilibili.com/video/BV1AF411b7xQ?spm_id_from=333.788.videopod.sections&vd_source=a5dd5ad68f590473458dad8b99349b27) - 哔哩哔哩
2. [Llama 3.1论文精读](https://www.bilibili.com/video/BV1Q4421Z7Tj/?spm_id_from=333.1387.search.video_card.click&vd_source=a5dd5ad68f590473458dad8b99349b27) - 哔哩哔哩

## 📝 知识总结

### Self-Attention计算公式

```math
Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V
```

**核心组件:**
- **Query (Q)**: 查询向量，决定当前位置关注什么
- **Key (K)**: 键向量，被查询的内容
- **Value (V)**: 值向量，实际传递的信息
- **缩放因子**: $\sqrt{d_k}$，防止softmax梯度消失

### Multi-Head Attention

将输入投影到多个不同的子空间，并行计算多个注意力头：

```math
MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
```

其中 $head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$

## 💬 面试问题解答

### Q1: Attention计算公式是什么？

**标准回答：**
Self-Attention的核心公式是 `Attention(Q,K,V) = softmax(QK^T/√d_k)V`。

**详细解释：**
1. 先计算Query和Key的点积得到注意力分数
2. 除以√d_k进行缩放
3. 通过softmax归一化得到注意力权重
4. 最后与Value相乘得到输出

### Q1.1: 为什么要除以根号d_k？

**核心原因：** 防止softmax函数进入饱和区导致梯度消失。

**技术解释：**
- 当d_k较大时，QK^T的方差会很大
- 大的数值经过softmax后梯度接近0
- 除以√d_k可以控制方差为1，保持梯度稳定

### Q1.2: Softmax的作用是什么？

**主要作用：**
1. **归一化**: 确保注意力权重之和为1
2. **突出重点**: 通过指数函数放大重要特征
3. **可微分**: 保证反向传播可以正常进行

## 💻 代码实现

### 练习1: 实现Self-Attention机制
**平台**: [Deep-ML Self-Attention](https://www.deep-ml.com/problems/53)

### 练习2: 实现Multi-Head Attention  
**平台**: [Deep-ML Multi-Head Attention](https://www.deep-ml.com/problems/94)

### 代码模板

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # TODO: 实现注意力计算
        pass
        
    def forward(self, x, mask=None):
        # TODO: 实现完整的前向传播
        pass
```

## ✅ 学习检验

完成以下检验才算掌握本节：

- [ ] 能画出Self-Attention的计算流程图
- [ ] 可以手算简单的Attention权重
- [ ] 完成Deep-ML平台的两个编程练习
- [ ] 面试问题能用自己的话流利回答

## 🔗 相关链接

- [下一节：语言模型架构](language-models.md)
- [返回：Transformer基础概览](index.md)