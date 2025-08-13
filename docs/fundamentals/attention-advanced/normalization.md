# 归一化技术

## 🎯 本节目标

掌握深度学习中三大归一化技术的原理和应用场景，理解Pre-Norm与Post-Norm在Transformer中的影响。

## 📖 阅读材料

### 核心技术文章
1. [大模型中常见的3种Norm](https://zhuanlan.zhihu.com/p/648987575) - 知乎
2. [为什么当前主流的大模型都使用RMS-Norm？](https://zhuanlan.zhihu.com/p/12392406696) - 知乎  
3. [为什么Pre Norm的效果不如Post Norm？](https://spaces.ac.cn/archives/9009) - 科学空间

### 选读深入材料
- [BN究竟起了什么作用？](https://spaces.ac.cn/archives/6992) - 科学空间

## 📝 知识总结

### 三大归一化技术对比

| 技术 | 归一化维度 | 适用场景 | 主要优势 | 计算成本 |
|------|----------|----------|----------|----------|
| **BatchNorm** | 跨样本特征维度 | CNN、大batch | 训练加速、防过拟合 | 中等 |
| **LayerNorm** | 单样本所有特征 | RNN、Transformer | 不依赖batch大小 | 较高 |
| **RMSNorm** | 单样本RMS归一化 | 大型语言模型 | 计算高效、效果相当 | 最低 |

### 数学公式详解

#### 1. Batch Normalization
```math
BN(x) = γ × \frac{x - μ_B}{\sqrt{σ_B^2 + ε}} + β
```

**核心特点**:
- μ_B, σ_B: 在batch维度计算均值和方差
- 训练时使用当前batch统计量，推理时使用移动平均
- 需要γ(缩放)和β(偏移)可学习参数

**问题**:
- 依赖batch大小，小batch效果差
- 训练和推理不一致
- 在序列模型中效果不佳

#### 2. Layer Normalization  
```math
LN(x) = γ × \frac{x - μ_L}{\sqrt{σ_L^2 + ε}} + β
```

**核心特点**:
- μ_L, σ_L: 在特征维度计算均值和方差
- 每个样本独立归一化，不依赖其他样本
- 训练和推理一致

**优势**:
- 适合变长序列
- 不受batch大小影响
- Transformer的标准选择

#### 3. RMS Normalization
```math
RMSNorm(x) = γ × \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + ε}}
```

**核心特点**:
- 只计算RMS，不减去均值
- 简化了LayerNorm的计算
- 只需要γ参数，无需β

**优势**:
- 计算成本更低
- 在大模型中效果不输LayerNorm
- 内存友好

### Pre-Norm vs Post-Norm

#### Post-Norm (原始Transformer)
```
Input → Attention → Add → LayerNorm → FFN → Add → LayerNorm → Output
```

**特点**:
- 归一化在残差连接之后
- 需要学习率warmup才能稳定训练
- 浅层模型(≤6层)效果更好
- 梯度传播可能不稳定

#### Pre-Norm (现代主流)
```  
Input → LayerNorm → Attention → Add → LayerNorm → FFN → Add → Output
```

**特点**:
- 归一化在残差连接之前
- 训练更稳定，无需warmup
- 深层模型训练更容易
- 现代大模型的标准选择

## 💬 面试问题解答

### Q1: Batch Norm和Layer Norm的区别？

**核心区别**:

1. **归一化维度不同**:
   - BatchNorm: 在batch维度归一化，每个特征独立
   - LayerNorm: 在特征维度归一化，每个样本独立

2. **应用场景**:
   - BatchNorm: CNN、视觉任务、大batch训练
   - LayerNorm: NLP、序列模型、小batch或变长序列

3. **依赖性**:
   - BatchNorm: 依赖batch大小和其他样本
   - LayerNorm: 只依赖当前样本，更稳定

**技术细节**:
```python
# BatchNorm: 在batch维度计算统计量
batch_mean = x.mean(dim=0)  # [features]
batch_var = x.var(dim=0)    # [features]

# LayerNorm: 在特征维度计算统计量  
layer_mean = x.mean(dim=-1, keepdim=True)  # [batch, 1]
layer_var = x.var(dim=-1, keepdim=True)    # [batch, 1]
```

### Q2: 为什么现在用RMSNorm？

**主要原因**:

1. **计算效率**:
   - 省略了均值计算，减少了约15%的计算量
   - 内存访问更少，对GPU更友好

2. **效果相当**:
   - 大量实验证明RMSNorm效果不输LayerNorm
   - 在大模型中表现甚至更好

3. **简化实现**:
   - 不需要β参数，减少了参数量
   - 数值稳定性更好

**技术原理**:
```python
# LayerNorm需要计算均值和方差
mean = x.mean(dim=-1, keepdim=True)
var = x.var(dim=-1, keepdim=True)  
ln_out = gamma * (x - mean) / sqrt(var + eps) + beta

# RMSNorm只需要计算RMS
rms = sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
rms_out = gamma * x / rms
```

### Q3: Pre-Norm和Post-Norm的位置区别？

**架构对比**:

**Post-Norm (原始)**:
```
x → Attention → (+) → LayerNorm → FFN → (+) → LayerNorm
    ↑_______________|              ↑_______|
```

**Pre-Norm (现代)**:
```
x → LayerNorm → Attention → (+) → LayerNorm → FFN → (+)
    ↑________________________|              ↑______|
```

**训练稳定性差异**:

| 方面 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| **学习率warmup** | 必需 | 可选 |
| **深层训练** | 容易失败 | 稳定 |
| **梯度传播** | 可能不稳定 | 更平滑 |
| **收敛速度** | 较慢 | 较快 |
| **最终性能** | 浅层更好 | 深层更优 |

### Q4: 为什么Pre-Norm训练更稳定？

**梯度传播分析**:

1. **Post-Norm问题**:
   - 梯度需要经过LayerNorm的反向传播
   - LayerNorm的导数可能放大或缩小梯度
   - 深层网络容易出现梯度爆炸/消失

2. **Pre-Norm优势**:
   - 提供了更强的恒等路径(identity path)
   - 梯度可以更直接地反向传播
   - 在深层网络中梯度的数量级为√L (L为层数)

**数学直觉**:
```
Post-Norm: 梯度需要穿过LayerNorm
∇L/∂x = ∇L/∂norm × ∂norm/∂x  (不稳定)

Pre-Norm: 恒等路径更强  
∇L/∂x = ∇L/∂residual + ∇L/∂processed  (更稳定)
```

## 💻 代码实现

### 三种Norm的PyTorch实现

```python
import torch
import torch.nn as nn
import math

class BatchNorm1d(nn.Module):
    """自实现BatchNorm"""
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        
        # 移动平均统计量
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x):
        # x shape: [batch_size, num_features]
        if self.training:
            # 计算当前batch的统计量
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # 更新移动平均
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # 使用当前batch统计量归一化
            mean, var = batch_mean, batch_var
        else:
            # 推理时使用移动平均
            mean, var = self.running_mean, self.running_var
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class LayerNorm(nn.Module):
    """自实现LayerNorm"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x):
        # 在最后一个维度计算统计量
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # 归一化
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

class RMSNorm(nn.Module):
    """自实现RMSNorm"""
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x):
        # 计算RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # RMS归一化
        x_norm = x / rms
        return self.gamma * x_norm

# 使用示例和性能对比
def compare_normalization():
    """对比三种归一化的计算成本"""
    
    batch_size, seq_len, d_model = 32, 512, 768
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 初始化三种norm
    bn = BatchNorm1d(d_model)
    ln = LayerNorm(d_model)
    rms = RMSNorm(d_model)
    
    print("=== 归一化技术对比 ===")
    
    # 测试LayerNorm
    import time
    start_time = time.time()
    for _ in range(1000):
        _ = ln(x)
    ln_time = time.time() - start_time
    print(f"LayerNorm耗时: {ln_time:.4f}秒")
    
    # 测试RMSNorm
    start_time = time.time()
    for _ in range(1000):
        _ = rms(x)
    rms_time = time.time() - start_time
    print(f"RMSNorm耗时: {rms_time:.4f}秒")
    
    speedup = ln_time / rms_time
    print(f"RMSNorm加速倍数: {speedup:.2f}x")
    
    # 参数量对比
    ln_params = sum(p.numel() for p in ln.parameters())
    rms_params = sum(p.numel() for p in rms.parameters())
    print(f"LayerNorm参数量: {ln_params}")
    print(f"RMSNorm参数量: {rms_params}")
    print(f"参数减少: {(ln_params - rms_params) / ln_params * 100:.1f}%")

# Pre-Norm vs Post-Norm实现
class PostNormBlock(nn.Module):
    """Post-Norm Transformer Block"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Post-Norm: Attention → Add → Norm
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN → Add → Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class PreNormBlock(nn.Module):
    """Pre-Norm Transformer Block"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Pre-Norm: Norm → Attention → Add
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + attn_out
        
        # Norm → FFN → Add
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + ffn_out
        return x

if __name__ == "__main__":
    compare_normalization()
```

## ✅ 学习检验

- [ ] 理解三种归一化的数学原理和计算方式
- [ ] 能解释为什么Transformer选择LayerNorm而非BatchNorm
- [ ] 掌握Pre-Norm相比Post-Norm的训练优势
- [ ] 完成归一化技术的代码实现和性能对比

## 🔗 相关链接

- [上一节：KV Cache技术](kv-cache.md)
- [下一节：位置编码](positional-encoding.md)
- [返回：Attention升级概览](index.md)