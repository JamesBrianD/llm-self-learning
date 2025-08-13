# KV Cache技术

## 🎯 本节目标

深入理解KV Cache的工作原理，掌握这个让大模型推理提速数倍的关键技术。

## 📝 知识总结

### KV Cache是什么？

**定义**: KV Cache是一种推理优化技术，通过缓存之前计算过的Key和Value矩阵，避免重复计算，大幅提升生成速度。

### 为什么需要KV Cache？

**问题背景**: 自回归生成过程中的重复计算

```python
# 生成"我爱北京天安门"的过程
Step 1: 输入["我"]        → 预测"爱" 
Step 2: 输入["我","爱"]    → 预测"北"
Step 3: 输入["我","爱","北"] → 预测"京"
...
```

**重复计算问题**:
- 每一步都要重新计算所有previous tokens的K,V矩阵
- 计算复杂度: O(n²)，其中n是序列长度
- 大量重复计算导致推理速度慢

### KV Cache工作原理

#### 1. 传统方式 (无缓存)

```python
# 每次都重新计算全部K,V
def generate_token_naive(tokens):
    # 重新计算所有token的K,V - 非常低效！
    K = compute_K(tokens)  # 包含所有历史token
    V = compute_V(tokens)  # 包含所有历史token
    Q = compute_Q(tokens[-1])  # 只需要最后一个token的Q
    
    attention_output = attention(Q, K, V)
    return next_token
```

#### 2. KV Cache优化方式

```python
# 只计算新token的K,V，复用历史缓存
def generate_token_with_cache(new_token, kv_cache):
    # 只计算新token的K,V
    new_K = compute_K(new_token)  
    new_V = compute_V(new_token)
    
    # 更新缓存
    kv_cache.append(new_K, new_V)
    
    # 使用完整的K,V (历史+新增)
    Q = compute_Q(new_token)
    attention_output = attention(Q, kv_cache.K, kv_cache.V)
    
    return next_token
```

### 加速效果分析

**时间复杂度对比**:

| 生成步骤 | 无Cache | 有Cache | 加速比 |
|----------|---------|---------|--------|
| 第1步 | O(1) | O(1) | 1x |
| 第2步 | O(4) | O(1) | 4x |
| 第3步 | O(9) | O(1) | 9x |
| 第n步 | O(n²) | O(1) | n²x |

**内存使用**:
- 空间换时间的策略
- 需要存储: `seq_len × num_heads × head_dim × 2` (K和V)
- 长序列时内存需求显著增加

## 💬 面试问题解答

### Q1: KV Cache是什么，为什么KV Cache能加速模型推理？

**核心答案**: 
KV Cache是缓存注意力机制中Key和Value矩阵的技术，通过避免重复计算历史token的K,V来加速推理。

**详细解释**:

1. **问题根源**: 
   - 自回归生成每步都需要完整的attention计算
   - 历史token的K,V矩阵在每步中保持不变
   - 重复计算造成O(n²)的时间复杂度

2. **解决方案**:
   - 缓存已计算的K,V矩阵
   - 新token只需计算自己的K,V并追加到缓存
   - 将时间复杂度从O(n²)降低到O(1)

3. **加速原理**:
   ```
   传统方式: 每步计算完整序列的K,V
   缓存方式: 只计算新增token的K,V
   ```

### Q2: KV Cache的内存开销如何？

**内存需求计算**:
```python
memory_per_token = num_layers × num_heads × head_dim × 2 × dtype_size
total_memory = memory_per_token × max_seq_length
```

**具体例子** (LLaMA-7B):
```
参数: 32层, 32头, 128维度, FP16
每个token: 32 × 32 × 128 × 2 × 2 bytes = 524KB
2048长度: 524KB × 2048 ≈ 1GB
```

**内存优化策略**:
- 使用更低精度(FP16/INT8)
- 分层缓存，只保留最近的token
- 滑动窗口，丢弃过旧的缓存

### Q3: KV Cache在不同注意力变体中的表现？

**各变体的KV Cache需求**:

| 注意力类型 | KV Cache大小 | 说明 |
|-----------|-------------|------|
| **MHA** | `h × d × L` | 每个头独立存储K,V |
| **MQA** | `d × L` | 所有头共享K,V，内存减少h倍 |
| **GQA** | `g × d × L` | 分组共享，内存需求在MHA和MQA之间 |
| **MLA** | 最小 | 通过低秩分解进一步压缩 |

> h=头数, d=维度, L=序列长度, g=组数

## 💻 代码实现

### 完整KV Cache演示

```python
import torch
import torch.nn as nn

class KVCache:
    """简化的KV Cache实现"""
    
    def __init__(self, max_seq_len, num_heads, head_dim, device='cpu'):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # 预分配缓存空间
        self.cache_k = torch.zeros(
            max_seq_len, num_heads, head_dim, 
            device=device, dtype=torch.float16
        )
        self.cache_v = torch.zeros(
            max_seq_len, num_heads, head_dim,
            device=device, dtype=torch.float16
        )
        
        self.current_length = 0
    
    def update_cache(self, new_k, new_v):
        """更新缓存并返回完整的K,V"""
        batch_size, seq_len, num_heads, head_dim = new_k.shape
        
        # 检查是否超出缓存容量
        if self.current_length + seq_len > self.max_seq_len:
            raise ValueError("Sequence length exceeds cache capacity")
        
        # 更新缓存
        end_pos = self.current_length + seq_len
        self.cache_k[self.current_length:end_pos] = new_k[0]  # 假设batch_size=1
        self.cache_v[self.current_length:end_pos] = new_v[0]
        
        self.current_length = end_pos
        
        # 返回到目前为止的完整K,V
        return (
            self.cache_k[:self.current_length].unsqueeze(0),  # 添加batch维度
            self.cache_v[:self.current_length].unsqueeze(0)
        )
    
    def clear(self):
        """清空缓存"""
        self.current_length = 0
    
    def get_cache_info(self):
        """获取缓存状态信息"""
        return {
            'current_length': self.current_length,
            'capacity': self.max_seq_len,
            'usage_ratio': self.current_length / self.max_seq_len,
            'memory_mb': self.cache_k.numel() * 2 * 2 / 1024 / 1024  # FP16
        }


class AttentionWithKVCache(nn.Module):
    """带KV Cache的注意力层"""
    
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.kv_cache = None
    
    def setup_cache(self, max_seq_len, device):
        """初始化KV Cache"""
        self.kv_cache = KVCache(max_seq_len, self.num_heads, self.head_dim, device)
    
    def forward(self, x, use_cache=False):
        batch_size, seq_len, d_model = x.shape
        
        # 计算Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        if use_cache and self.kv_cache is not None:
            # 使用缓存模式：更新缓存并获取完整的K,V
            K, V = self.kv_cache.update_cache(K, V)
        
        # 计算注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        
        # 合并多头输出
        out = out.view(batch_size, seq_len, d_model)
        return self.W_o(out)


# 使用示例
def demo_kv_cache():
    """KV Cache使用演示"""
    
    # 初始化模型
    attention = AttentionWithKVCache(d_model=512, num_heads=8)
    attention.setup_cache(max_seq_len=1024, device='cpu')
    
    print("=== KV Cache演示 ===")
    
    # 模拟生成过程
    vocab_size = 1000
    sequence = []
    
    for step in range(5):
        if step == 0:
            # 第一步：输入完整的prompt
            current_input = torch.randint(0, vocab_size, (1, 3, 512))  # 3个token的prompt
            print(f"Step {step}: 输入prompt (3 tokens)")
        else:
            # 后续步骤：只输入新生成的token
            current_input = torch.randint(0, vocab_size, (1, 1, 512))  # 1个新token
            print(f"Step {step}: 输入新token (1 token)")
        
        # 前向传播（使用缓存）
        output = attention(current_input, use_cache=True)
        
        # 显示缓存状态
        cache_info = attention.kv_cache.get_cache_info()
        print(f"  缓存长度: {cache_info['current_length']}")
        print(f"  内存使用: {cache_info['memory_mb']:.2f} MB")
        print()

if __name__ == "__main__":
    demo_kv_cache()
```

### 性能对比测试

```python
import time

def benchmark_with_without_cache():
    """对比有无KV Cache的性能"""
    
    d_model, num_heads = 768, 12
    max_seq_len = 512
    
    # 初始化模型
    attention_with_cache = AttentionWithKVCache(d_model, num_heads)
    attention_with_cache.setup_cache(max_seq_len, 'cpu')
    
    attention_without_cache = AttentionWithKVCache(d_model, num_heads)
    
    # 模拟序列生成
    prompt_len = 50
    generate_len = 100
    
    print("=== 性能对比测试 ===")
    
    # 测试无缓存版本
    start_time = time.time()
    sequence_input = torch.randn(1, prompt_len, d_model)
    
    for i in range(generate_len):
        # 每次都输入完整序列（无缓存）
        full_input = torch.randn(1, prompt_len + i + 1, d_model)
        _ = attention_without_cache(full_input, use_cache=False)
    
    no_cache_time = time.time() - start_time
    print(f"无缓存生成时间: {no_cache_time:.3f}秒")
    
    # 测试有缓存版本  
    start_time = time.time()
    
    # 处理prompt
    _ = attention_with_cache(sequence_input, use_cache=True)
    
    # 逐步生成
    for i in range(generate_len):
        # 每次只输入新token（有缓存）
        new_token = torch.randn(1, 1, d_model)
        _ = attention_with_cache(new_token, use_cache=True)
    
    with_cache_time = time.time() - start_time
    print(f"有缓存生成时间: {with_cache_time:.3f}秒")
    
    speedup = no_cache_time / with_cache_time
    print(f"加速倍数: {speedup:.1f}x")

if __name__ == "__main__":
    benchmark_with_without_cache()
```

## ✅ 学习检验

- [ ] 理解KV Cache的工作原理和加速机制
- [ ] 能计算KV Cache的内存需求
- [ ] 完成KV Cache演示代码的编写和测试
- [ ] 理解不同注意力变体对KV Cache的影响

## 🔗 相关链接

- [上一节：多头注意力变体](mha-variants.md)
- [下一节：归一化技术](normalization.md)
- [返回：Attention升级概览](index.md)