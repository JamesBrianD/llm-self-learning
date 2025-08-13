# 位置编码

## 🎯 本节目标

理解位置编码在Transformer中的作用，掌握从绝对位置编码到RoPE的技术演进，能够推导RoPE的数学原理。

## 📖 阅读材料

### 核心技术文章
1. [Sinusoidal位置编码追根溯源](https://kexue.fm/archives/8231) - 科学空间
2. [博采众长的旋转式位置编码](https://kexue.fm/archives/8265) - 科学空间
3. [让研究人员绞尽脑汁的Transformer位置编码](https://kexue.fm/archives/8130) - 科学空间

## 📝 知识总结

### 为什么需要位置编码？

**核心问题**: Transformer的Self-Attention机制是置换不变的（permutation invariant），无法区分token的顺序。

```python
# 没有位置信息时，这两个序列是等价的
sequence1 = ["我", "爱", "北京"]
sequence2 = ["爱", "北京", "我"]
# Self-Attention会给出相同的结果！
```

**解决方案**: 在输入中注入位置信息，让模型能够理解token之间的相对或绝对位置关系。

### 位置编码分类

```
位置编码
├── 绝对位置编码 (APE)
│   ├── 可训练位置编码 (Learned PE)
│   └── 固定位置编码 (Sinusoidal PE)
└── 相对位置编码 (RPE)
    ├── 经典相对位置编码
    ├── 旋转位置编码 (RoPE)
    └── 其他变体 (ALiBi等)
```

### 绝对位置编码 vs 相对位置编码

| 维度 | 绝对位置编码 | 相对位置编码 |
|------|-------------|-------------|
| **编码对象** | token的绝对位置 | token之间的相对距离 |
| **操作位置** | 输入层添加位置向量 | 注意力层修改计算方式 |
| **实现复杂度** | 简单 | 相对复杂 |
| **长度外推** | 较差 | 较好 |
| **性能表现** | 短序列足够 | 长序列更优 |

### 核心技术详解

#### 1. Sinusoidal位置编码 (原始Transformer)

**数学公式**:
```math
PE(pos, 2i) = sin(pos / 10000^{2i/d_{model}})
PE(pos, 2i+1) = cos(pos / 10000^{2i/d_{model}})
```

**核心特点**:
- 使用正弦余弦函数生成位置编码
- 不同维度使用不同的频率
- 固定编码，不需要训练
- 理论上支持任意长度序列

**优势**:
- 计算简单，不占用参数
- 具有一定的外推能力
- 相对位置有一定的规律性

**缺点**:
- 位置信息在深层可能衰减
- 对相对位置的建模不够直接

#### 2. 可训练位置编码

**实现方式**:
```python
# 为每个位置学习一个向量
position_embeddings = nn.Embedding(max_seq_len, d_model)
pos_emb = position_embeddings(position_ids)
input_emb = token_emb + pos_emb
```

**特点**:
- 每个位置对应一个可学习的向量
- 通过训练优化位置表示
- 在训练长度范围内效果通常更好

#### 3. RoPE (旋转位置编码)

**核心思想**: 通过旋转变换将位置信息编码到查询和键向量中，使得注意力分数自然地依赖于相对位置。

**数学推导**:

**步骤1**: 将特征分为pairs，每对特征看作2D平面的坐标
```math
x = [x_1, x_2, x_3, x_4, ...] → [(x_1, x_2), (x_3, x_4), ...]
```

**步骤2**: 对每一对特征应用旋转矩阵
```math
\begin{pmatrix}
x_{m}^{(1)} \\
x_{m}^{(2)}
\end{pmatrix}
→
\begin{pmatrix}
\cos(m\theta) & -\sin(m\theta) \\
\sin(m\theta) & \cos(m\theta)
\end{pmatrix}
\begin{pmatrix}
x_{m}^{(1)} \\
x_{m}^{(2)}
\end{pmatrix}
```

**步骤3**: 旋转后的向量
```math
\begin{pmatrix}
x_{m}^{(1)} \cos(m\theta) - x_{m}^{(2)} \sin(m\theta) \\
x_{m}^{(2)} \cos(m\theta) + x_{m}^{(1)} \sin(m\theta)
\end{pmatrix}
```

**核心性质**: 相对位置依赖
```math
\langle RoPE(q_m), RoPE(k_n) \rangle = \langle q_m, k_n \rangle \cos((m-n)\theta) + \text{其他项}
```

注意力分数只依赖于相对距离 (m-n)！

## 💬 面试问题解答

### Q1: 什么是绝对位置编码，相对位置编码？

**绝对位置编码 (APE)**:
- **定义**: 为每个token的绝对位置分配一个位置向量
- **实现**: 在输入层将位置向量加到token embedding上
- **特点**: 简单直接，每个位置有固定的编码

**相对位置编码 (RPE)**:
- **定义**: 在计算注意力时考虑token之间的相对距离
- **实现**: 修改注意力计算公式，加入相对位置偏置
- **特点**: 更符合直觉，外推能力更强

**技术细节对比**:
```python
# 绝对位置编码
input_emb = token_emb + position_emb[pos]

# 相对位置编码  
attention_score = QK^T + relative_position_bias[i-j]
```

### Q2: 推导RoPE的数学原理

**推导步骤**:

**目标**: 设计一个函数f，使得：
```math
\langle f(q, m), f(k, n) \rangle = g(q, k, m-n)
```
即注意力分数只依赖相对位置 m-n。

**解决方案**: 复数域的旋转变换

**步骤1**: 将实数向量映射到复数
```math
q_{1} + i q_{2} → q_{complex}
```

**步骤2**: 应用复数旋转
```math
f(q, m) = q_{complex} \cdot e^{im\theta} = q_{complex} \cdot (\cos(m\theta) + i\sin(m\theta))
```

**步骤3**: 验证相对位置性质
```math
\langle f(q,m), f(k,n) \rangle^* = \langle q \cdot e^{im\theta}, k \cdot e^{in\theta} \rangle
= \langle q, k \rangle \cdot e^{i(m-n)\theta}
```

只依赖于 (m-n)！

**步骤4**: 转换回实数域
```math
\begin{pmatrix}
q_1 \cos(m\theta) - q_2 \sin(m\theta) \\
q_1 \sin(m\theta) + q_2 \cos(m\theta)
\end{pmatrix}
```

**关键洞察**: 通过旋转变换，相对位置信息自然地编码在了向量的几何关系中。

### Q3: RoPE相比传统位置编码的优势？

**核心优势**:

1. **自然的相对位置依赖**
   - 注意力分数直接依赖相对距离
   - 无需额外的相对位置偏置项

2. **优秀的外推能力**
   - 训练时的相对位置模式可以泛化到更长序列
   - 理论上支持无限长度外推

3. **计算高效**
   - 无需存储位置嵌入表
   - 旋转操作可以高效实现

4. **理论优雅**
   - 数学基础扎实
   - 基于复数旋转的几何直觉

**实验验证**:
- 在多个NLP任务上超越传统位置编码
- 长序列任务上表现特别突出
- 已被多个大模型采用(LLaMA、PaLM等)

### Q4: RoPE在实际实现中有什么技巧？

**实现优化**:

1. **频率选择**
   ```python
   # 不同维度使用不同频率
   theta = 10000 ** (-2 * torch.arange(0, dim, 2) / dim)
   ```

2. **预计算旋转矩阵**
   ```python
   # 避免重复计算sin/cos
   cos_cached = torch.cos(position * theta)
   sin_cached = torch.sin(position * theta)
   ```

3. **向量化实现**
   ```python
   # 同时处理所有位置和维度
   q_rot = q * cos_cached - q_shifted * sin_cached
   ```

## 💻 代码实现

### RoPE完整实现

```python
import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    """RoPE (Rotary Position Embedding) 实现"""
    
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 计算旋转频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len):
        """预计算并缓存旋转矩阵"""
        # 生成位置序列
        position = torch.arange(seq_len).float()
        
        # 计算角度: position * inv_freq
        freqs = torch.outer(position, self.inv_freq)  # [seq_len, dim//2]
        
        # 拼接，形成完整的频率矩阵
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim]
        
        # 计算cos和sin
        cos_cached = emb.cos()
        sin_cached = emb.sin()
        
        self.register_buffer('cos_cached', cos_cached)
        self.register_buffer('sin_cached', sin_cached)
    
    def rotate_half(self, x):
        """将输入的后半部分取负号并移到前面"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q, k, seq_len=None):
        """
        对查询和键向量应用RoPE
        
        Args:
            q: 查询矩阵 [batch, heads, seq_len, dim]
            k: 键矩阵 [batch, heads, seq_len, dim]
            seq_len: 序列长度
        """
        if seq_len is None:
            seq_len = q.shape[-2]
        
        # 如果序列长度超出缓存，重新构建
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        
        # 获取对应长度的cos和sin
        cos = self.cos_cached[:seq_len]  # [seq_len, dim]
        sin = self.sin_cached[:seq_len]  # [seq_len, dim]
        
        # 应用旋转变换
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        
        return q_rot, k_rot

class MultiHeadAttentionWithRoPE(nn.Module):
    """带RoPE的多头注意力"""
    
    def __init__(self, d_model, num_heads, max_seq_len=2048):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # RoPE只应用到部分维度（通常是前半部分）
        self.rope = RotaryPositionalEmbedding(
            dim=self.head_dim, 
            max_seq_len=max_seq_len
        )
    
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # 计算Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置以符合注意力计算的维度要求
        Q = Q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 应用RoPE
        Q, K = self.rope(Q, K, seq_len)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax归一化
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 计算输出
        out = torch.matmul(attn_weights, V)
        
        # 重塑并合并多头
        out = out.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.W_o(out)

# 不同位置编码的对比测试
def compare_position_encodings():
    """对比不同位置编码的效果"""
    
    d_model, seq_len = 512, 64
    batch_size, num_heads = 2, 8
    
    print("=== 位置编码对比测试 ===")
    
    # 测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 1. 无位置编码的注意力
    attn_no_pos = MultiHeadAttentionWithRoPE(d_model, num_heads)
    # 临时移除RoPE
    attn_no_pos.rope = lambda q, k, seq_len: (q, k)
    out_no_pos = attn_no_pos(x)
    
    # 2. 带RoPE的注意力
    attn_with_rope = MultiHeadAttentionWithRoPE(d_model, num_heads)
    out_with_rope = attn_with_rope(x)
    
    print(f"无位置编码输出标准差: {out_no_pos.std():.4f}")
    print(f"RoPE位置编码输出标准差: {out_with_rope.std():.4f}")
    
    # 3. 测试外推能力
    print("\n=== 外推能力测试 ===")
    
    # 短序列训练
    short_len = 32
    x_short = torch.randn(1, short_len, d_model)
    
    # 长序列推理
    long_len = 128
    x_long = torch.randn(1, long_len, d_model)
    
    try:
        out_short = attn_with_rope(x_short)
        out_long = attn_with_rope(x_long)
        print(f"短序列({short_len})处理成功")
        print(f"长序列({long_len})处理成功 - RoPE支持外推")
    except Exception as e:
        print(f"外推失败: {e}")

# 手动验证RoPE的相对位置性质
def verify_rope_property():
    """验证RoPE的相对位置依赖性质"""
    
    print("=== 验证RoPE相对位置性质 ===")
    
    dim = 64
    rope = RotaryPositionalEmbedding(dim, max_seq_len=10)
    
    # 创建两个位置的查询和键
    q = torch.randn(1, 1, 1, dim)  # 位置0的查询
    k = torch.randn(1, 1, 1, dim)  # 位置0的键
    
    # 在不同相对距离下测试
    distances = [1, 2, 3]
    
    for dist in distances:
        # 计算位置(0, dist)的相对注意力
        q_pos0, k_pos_dist = rope(q, k, seq_len=dist+1)
        score1 = torch.matmul(q_pos0[:,:,0:1], k_pos_dist[:,:,dist:dist+1].transpose(-2,-1))
        
        # 计算位置(1, 1+dist)的相对注意力  
        q_pos1, k_pos1_dist = rope(q, k, seq_len=dist+2)
        score2 = torch.matmul(q_pos1[:,:,1:2], k_pos1_dist[:,:,1+dist:2+dist].transpose(-2,-1))
        
        print(f"相对距离{dist}: 分数差异 = {abs(score1.item() - score2.item()):.6f}")

if __name__ == "__main__":
    compare_position_encodings()
    print()
    verify_rope_property()
```

### Sinusoidal位置编码实现

```python
class SinusoidalPositionalEncoding(nn.Module):
    """原始Transformer的正弦位置编码"""
    
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
```

## ✅ 学习检验

- [ ] 理解绝对位置编码和相对位置编码的区别
- [ ] 能推导RoPE的数学原理
- [ ] 掌握RoPE的实现细节和优化技巧
- [ ] 完成位置编码的代码实现和效果验证

## 🔗 相关链接

- [上一节：归一化技术](normalization.md)
- [下一节：LLM升级技术](../llm-advanced/index.md)
- [返回：Attention升级概览](index.md)