# 多头注意力变体

## 🎯 本节目标

理解从MHA到MLA的技术演进，掌握不同注意力机制的优化原理和应用场景。

## 📖 阅读材料

### 核心技术文章
1. [Transformer的Attention及其各种变体](https://lengm.cn/post/20250226_attention/) - 详细对比分析
2. [缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA](https://spaces.ac.cn/archives/10091) - 科学空间深度解析

## 📝 知识总结

### 技术演进路径

```
MHA (标准多头) → MQA (共享KV) → GQA (分组共享) → MLA (潜在空间)
```

### 各变体详细对比

| 变体 | KV Cache需求 | 计算复杂度 | 性能表现 | 主要应用 |
|------|-------------|-----------|----------|----------|
| **MHA** | O(h×d×L) | O(H x N^2 x D) | 基准性能 | 标准Transformer |
| **MQA** | O(d×L) | O(N^2×D) | 轻微下降 | 资源受限场景 |
| **GQA** | O(g×d×L) | O(G×N^2×D) | 平衡优秀 | 主流大模型 |
| **MLA** | 最优化 | O(R×N^2×D) | 接近MHA | 长上下文 |

> 其中：h=头数，d=维度，L=序列长度，g=组数

### 核心技术细节

#### 1. MHA (Multi-Head Attention)
```python
# 每个头都有独立的Q、K、V
for i in range(num_heads):
    Q_i = input @ W_Q_i  # 每个头独立的查询矩阵
    K_i = input @ W_K_i  # 每个头独立的键矩阵  
    V_i = input @ W_V_i  # 每个头独立的值矩阵
    head_i = attention(Q_i, K_i, V_i)
```

#### 2. MQA (Multi-Query Attention)
```python
# 所有头共享K、V，只有Q独立
K_shared = input @ W_K  # 共享的键矩阵
V_shared = input @ W_V  # 共享的值矩阵

for i in range(num_heads):
    Q_i = input @ W_Q_i  # 每个头独立的查询矩阵
    head_i = attention(Q_i, K_shared, V_shared)
```

#### 3. GQA (Grouped-Query Attention)
```python
# 分组共享：每组内共享K、V
num_groups = num_heads // group_size

for g in range(num_groups):
    K_g = input @ W_K_g  # 组共享的键矩阵
    V_g = input @ W_V_g  # 组共享的值矩阵
    
    for i in range(group_size):
        Q_i = input @ W_Q_i
        head_i = attention(Q_i, K_g, V_g)
```

## 💬 面试问题解答

### Q1: MHA、MQA、GQA、MLA都是什么？

**简洁回答：**
这是Transformer注意力机制的四个演进阶段，主要优化KV Cache的存储需求：

- **MHA**: 标准多头注意力，每个头独立QKV
- **MQA**: 多查询注意力，所有头共享KV
- **GQA**: 分组查询注意力，分组内共享KV  
- **MLA**: 多头潜在注意力，通过低秩分解优化

**技术细节：**

**MHA问题**: KV Cache随头数线性增长，内存开销大
```
内存需求 = 头数 × 维度 × 序列长度
```

**MQA解决方案**: 共享KV矩阵，内存需求降低h倍
```python
# 从 h×(d_k + d_v) 降低到 (d_k + d_v)
```

**GQA平衡方案**: 分组共享，兼顾性能和效率
```python
# 内存需求 = 组数 × 维度 × 序列长度  
# 其中：组数 = 头数 / 每组头数
```

**MLA终极优化**: 潜在空间投影，最小化KV Cache

## 🔬 MLA技术深度解析

### MLA核心创新

**Multi-head Latent Attention (MLA)** 是DeepSeek团队提出的革命性注意力机制，通过三大创新显著降低KV Cache内存需求：

#### 1. 低秩KV联合压缩

**核心思想**: 将高维的Key和Value矩阵联合压缩到低维潜在空间

```python
# 传统方式：每个头独立存储KV
traditional_kv_cache = num_heads × head_dim × seq_len × 2  # K和V

# MLA方式：压缩后的潜在向量
mla_kv_cache = compressed_dim × seq_len + rope_dim × seq_len
```

**压缩过程**:
$$c_t^{KV} = x_t W^{DKV}$$

其中 $W^{DKV} \in \mathbb{R}^{d \times d_c}$，$d_c \ll h \cdot d_h$

#### 2. RoPE解耦机制

**问题**: 位置编码与压缩机制的冲突
- 传统RoPE需要在原始QK空间中应用
- 压缩破坏了位置信息的正确传递

**解决方案**: 将Query和Key分为两部分
- **语义部分** ($q^C, k^C$): 携带主要语义信息，可以压缩
- **位置部分** ($q^R, k^R$): 携带位置信息，保持原维度

```python
def mla_with_rope_decoupling(x, position):
    # 1. 生成潜在向量
    c_kv = x @ W_down_kv  # 压缩
    
    # 2. Query分离
    q = x @ W_q
    q_c, q_r = q[:, :d_c], q[:, d_c:]  # 语义 + 位置
    
    # 3. Key分离和恢复
    k_c = c_kv @ W_up_k   # 从潜在空间恢复语义Key
    k_r = x @ W_k_r       # 直接生成位置Key
    
    # 4. 分别应用RoPE
    q_r = apply_rope(q_r, position)
    k_r = apply_rope(k_r, position)
    
    # 5. 组合计算
    q_combined = concat([q_c, q_r])
    k_combined = concat([k_c, k_r])
    v = c_kv @ W_up_v
    
    return attention(q_combined, k_combined, v)
```

#### 3. 权重吸收优化

**目标**: 减少推理时的矩阵乘法操作

**技术**: 利用矩阵乘法结合律，预先合并权重矩阵

```python
# 原始计算：两次矩阵乘法
c_kv = x @ W_down_kv
k = c_kv @ W_up_k

# 权重吸收：合并为一次乘法
W_combined = W_down_kv @ W_up_k
k = x @ W_combined
```

### 内存效率对比

| 方法 | KV Cache大小 | 压缩比 |
|------|-------------|--------|
| **MHA** | $2 h \cdot d_h \cdot L$ | 1.0× (基准) |
| **MQA** | $2 d_h \cdot L$ | $h$× |
| **GQA** | $2 g \cdot d_h \cdot L$ | $h/g$× |
| **MLA** | $(d_c + d_h^R) \cdot L$ | ~10-20× |

**具体例子** (LLaMA-7B规模):
- 原始MHA: 32头 × 128维 × 2 = 8192维/token
- MLA压缩: 512维 + 128维 = 640维/token
- **压缩比**: 12.8×

### 性能保持机制

尽管大幅压缩，MLA通过巧妙设计保持了接近MHA的性能：

#### 1. 表达能力保持
- 低秩假设：大部分注意力模式可以用低秩矩阵近似
- 关键信息保留：位置信息通过解耦机制完整保留
- 渐进恢复：多层堆叠逐步恢复完整信息

#### 2. 训练稳定性
```python
# 残差连接确保训练稳定
def mla_block(x):
    # MLA注意力
    attn_out = mla_attention(x)
    x = x + attn_out  # 残差连接
    
    # FFN
    ffn_out = feed_forward(x)
    x = x + ffn_out   # 残差连接
    
    return x
```

#### 3. 位置敏感性
- RoPE解耦确保位置信息不丢失
- 位置编码维度可以根据任务需求调整
- 长序列外推能力得到保持

### Q2: 为什么需要这些优化？

**核心动机：**

1. **内存瓶颈**
   - 长序列推理时KV Cache占用大量显存
   - 限制了模型的部署和扩展能力

2. **推理速度**
   - 减少内存访问，提高计算效率
   - 支持更大的batch size

3. **成本考虑**
   - 降低硬件要求
   - 提高服务并发能力

### Q3: 各变体的优缺点对比？

| 维度 | MHA | MQA | GQA | MLA |
|------|-----|-----|-----|-----|
| **性能** | 🟢 基准最好 | 🟡 轻微下降 | 🟢 接近MHA | 🟢 超越MHA |
| **内存** | 🔴 需求最高 | 🟢 显著降低 | 🟡 适中 | 🟢 最优 |
| **速度** | 🟡 标准 | 🟢 最快 | 🟢 较快 | 🟢 优秀 |
| **实现** | 🟢 简单 | 🟢 简单 | 🟡 中等 | 🔴 复杂 |

### Q4: 如何选择合适的注意力机制？

**选择策略：**

```python
if 资源充足 and 追求最佳性能:
    选择 MHA
elif 资源严重受限 and 可接受性能损失:
    选择 MQA  
elif 需要平衡性能和效率:
    选择 GQA  # 主流选择
elif 长上下文 and 内存敏感:
    选择 MLA
```

**实际考虑因素：**
- 硬件内存限制
- 序列长度需求  
- 延迟要求
- 开发复杂度

## 💻 代码实现

### 练习1: 实现Softmax函数
**平台**: [Deep-ML Softmax](https://www.deep-ml.com/problems/23)

### 练习2: MHA到GQA的适配

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super().__init__()
        assert num_heads % num_groups == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.d_k = d_model // num_heads
        
        # Q矩阵：每个头独立
        self.W_q = nn.Linear(d_model, d_model)
        
        # K,V矩阵：按组共享
        self.W_k = nn.Linear(d_model, num_groups * self.d_k)
        self.W_v = nn.Linear(d_model, num_groups * self.d_k)
        
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # 生成Q：每个头独立
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # 生成K,V：按组共享
        K = self.W_k(x).view(batch_size, seq_len, self.num_groups, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_groups, self.d_k)
        
        # 重复K,V以匹配Q的头数
        K = K.repeat_interleave(self.heads_per_group, dim=2)
        V = V.repeat_interleave(self.heads_per_group, dim=2)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        
        # 合并多头输出
        out = out.view(batch_size, seq_len, d_model)
        return self.W_o(out)
```

### 练习3: KV Cache实现预览

```python
class KVCache:
    def __init__(self, max_seq_len, num_heads, d_k):
        self.max_seq_len = max_seq_len
        self.cache_k = torch.zeros(max_seq_len, num_heads, d_k)
        self.cache_v = torch.zeros(max_seq_len, num_heads, d_k) 
        self.current_len = 0
    
    def update(self, new_k, new_v):
        """更新缓存并返回完整的K,V"""
        seq_len = new_k.size(0)
        
        # 存储新的K,V
        self.cache_k[self.current_len:self.current_len+seq_len] = new_k
        self.cache_v[self.current_len:self.current_len+seq_len] = new_v
        
        self.current_len += seq_len
        
        # 返回到目前为止的完整K,V
        return (self.cache_k[:self.current_len], 
                self.cache_v[:self.current_len])
```

## ✅ 学习检验

- [ ] 能解释各变体的核心区别
- [ ] 理解KV Cache优化的原理
- [ ] 完成GQA代码实现
- [ ] 能根据场景选择合适的注意力机制

## 🔗 相关链接

- [下一节：KV Cache技术](kv-cache.md)
- [返回：Attention升级概览](index.md)