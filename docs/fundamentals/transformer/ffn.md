# 前馈神经网络 (FFN)

## 🎯 本节目标

深入理解Transformer中前馈神经网络的作用机制，掌握不同激活函数的演进和知识存储原理。

## 📝 知识总结

### FFN的基本结构

**前馈神经网络(Feed-Forward Network)**是Transformer中除注意力机制外的另一个核心组件，位于每个Transformer层中。

#### 数学表示
$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

其中：
- $W_1$: 第一层权重矩阵 (升维)
- $W_2$: 第二层权重矩阵 (降维)  
- $b_1, b_2$: 偏置向量
- $\max(0, ·)$: ReLU激活函数

#### 维度变化
```
输入: [batch_size, seq_len, d_model]
  ↓ W1 (线性层1)
中间: [batch_size, seq_len, d_ff]    # d_ff = 4 * d_model
  ↓ 激活函数
中间: [batch_size, seq_len, d_ff]    
  ↓ W2 (线性层2)  
输出: [batch_size, seq_len, d_model]
```

### FFN的核心功能

#### 1. 语义信息提取
- **逐位置处理**: 对序列中每个位置独立进行非线性变换
- **特征映射**: 将注意力输出映射到更高维的特征空间
- **模式识别**: 捕获复杂的语义模式和特征组合

#### 2. 知识存储机制
FFN被认为是Transformer的"记忆库"：

**分布式存储**:
- 不同的神经元专门存储不同类型的知识
- 通过权重矩阵编码语言模式和世界知识
- 类似于键值存储，输入作为"键"，激活模式作为"值"

**知识电路**:
- FFN中的特定神经元激活路径形成"知识电路"
- 这些电路编码特定的语义关系和事实知识
- 多层FFN协同工作，构建复杂的知识表示

#### 3. 表达能力增强
- **非线性变换**: 激活函数引入非线性，增强模型表达能力
- **维度扩展**: 中间层的高维度提供更丰富的表示空间
- **特征交互**: 促进不同特征维度之间的交互

### 激活函数的演进

#### 1. ReLU (早期Transformer)
$$\text{ReLU}(x) = \max(0, x)$$

**特点**:
- 简单高效，计算量小
- 解决梯度消失问题
- 但存在"死神经元"问题

#### 2. GELU (GPT等模型)
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}[1 + \text{erf}(\frac{x}{\sqrt{2}})]$$

**特点**:
- 更平滑的激活函数
- 在负值区域有非零梯度
- 性能通常优于ReLU

#### 3. SwiGLU (现代大模型)
$$\text{SwiGLU}(x) = \text{Swish}(W_1 x) \odot (W_2 x)$$
$$\text{Swish}(x) = x \cdot \sigma(x)$$

**特点**:
- 门控机制，更好的特征选择
- 需要额外参数但性能提升明显
- LLaMA、PaLM等现代模型的标准选择

### FFN的独特特性

#### 1. 位置无关处理
```python
# FFN对每个位置独立处理
for position in sequence:
    hidden = ffn_layer1(input[position])
    hidden = activation(hidden)
    output[position] = ffn_layer2(hidden)
```

#### 2. 与注意力机制的互补
| 机制 | 注意力 | FFN |
|------|--------|-----|
| **功能** | 序列内信息整合 | 位置内特征提取 |
| **依赖** | 全序列 | 单个位置 |
| **作用** | 建模关系 | 存储知识 |
| **计算** | 序列长度相关 | 序列长度无关 |

### 现代FFN优化技术

#### 1. Mixture of Experts (MoE)
- 将FFN替换为多个专家网络
- 通过路由机制动态选择专家
- 在保持计算量的同时大幅增加参数

#### 2. Memory Layers
- 引入外部记忆机制
- 缓存和检索相关知识
- 提高长序列处理能力

#### 3. KAN (Kolmogorov-Arnold Networks)
- 替代传统的线性层结构
- 使用可学习的激活函数
- 理论上更强的表达能力

## 💬 面试问题解答

### Q1: FFN在Transformer中起什么作用？

**核心作用**:
1. **知识存储**: 作为模型的"记忆库"，存储语言模式和世界知识
2. **特征提取**: 对每个位置进行非线性特征变换
3. **表达增强**: 通过高维映射增强模型表达能力
4. **与注意力互补**: 提供位置内的深度处理

**技术细节**:
- 逐位置独立处理，与注意力的序列建模形成互补
- 通过升维-激活-降维的过程增强特征表示
- 参数量通常占Transformer模型总参数的2/3

### Q2: 为什么FFN要先升维再降维？

**设计原理**:
1. **表示空间扩展**: 升维提供更丰富的特征表示空间
2. **非线性建模**: 高维空间中更容易拟合复杂函数
3. **特征交互**: 更多维度允许更复杂的特征组合
4. **信息瓶颈**: 最终降维起到信息筛选的作用

**数学直觉**:
```
d_model → d_ff → d_model
512 → 2048 → 512
```
中间的高维空间提供了更强的非线性建模能力。

### Q3: 不同激活函数对模型性能有什么影响？

**性能对比**:

| 激活函数 | 优势 | 劣势 | 适用场景 |
|----------|------|------|----------|
| **ReLU** | 计算简单，训练快 | 死神经元问题 | 早期模型 |
| **GELU** | 平滑，性能好 | 计算稍复杂 | 中等规模模型 |
| **SwiGLU** | 性能最佳，门控机制 | 参数量增加 | 现代大模型 |

**选择策略**:
- 计算资源充足：选择SwiGLU
- 平衡性能和效率：选择GELU
- 极度关注速度：选择ReLU

### Q4: FFN如何存储和检索知识？

**存储机制**:
1. **分布式表示**: 知识分布在不同神经元的权重中
2. **激活模式**: 特定输入触发特定的神经元组合
3. **层次结构**: 不同层的FFN存储不同抽象层次的知识

**检索过程**:
```python
# 简化的知识检索过程
input_features = attention_output  # 查询"键"
activated_neurons = ffn_layer1(input_features)  # 激活相关神经元
knowledge_pattern = activation_function(activated_neurons)  # 知识模式
output_knowledge = ffn_layer2(knowledge_pattern)  # 检索"值"
```

## 💻 代码实现

### 标准FFN实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """标准Transformer FFN实现"""
    
    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        
        # 第一层：升维 + 激活
        hidden = self.activation(self.linear1(x))
        hidden = self.dropout(hidden)
        
        # 第二层：降维
        output = self.linear2(hidden)
        
        return output

class SwiGLU(nn.Module):
    """SwiGLU激活函数的FFN实现"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # SwiGLU需要两个线性层用于门控
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False) 
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # SwiGLU: swish(W1*x) ⊙ (W2*x)
        swish_gate = F.silu(self.w1(x))  # Swish activation
        linear_part = self.w2(x)
        
        # 门控机制
        gated = swish_gate * linear_part
        gated = self.dropout(gated)
        
        # 输出投影
        output = self.w3(gated)
        
        return output

# 性能对比示例
def compare_ffn_activations():
    """对比不同激活函数的FFN性能"""
    
    d_model, d_ff = 512, 2048
    batch_size, seq_len = 32, 128
    
    # 测试数据
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 不同FFN实现
    ffn_relu = FeedForward(d_model, d_ff, activation='relu')
    ffn_gelu = FeedForward(d_model, d_ff, activation='gelu')
    ffn_swiglu = SwiGLU(d_model, d_ff)
    
    print("=== FFN激活函数对比 ===")
    
    # 参数量对比
    relu_params = sum(p.numel() for p in ffn_relu.parameters())
    gelu_params = sum(p.numel() for p in ffn_gelu.parameters())
    swiglu_params = sum(p.numel() for p in ffn_swiglu.parameters())
    
    print(f"ReLU FFN参数量: {relu_params:,}")
    print(f"GELU FFN参数量: {gelu_params:,}")
    print(f"SwiGLU FFN参数量: {swiglu_params:,}")
    
    # 计算时间对比
    import time
    
    with torch.no_grad():
        # ReLU
        start = time.time()
        for _ in range(100):
            _ = ffn_relu(x)
        relu_time = time.time() - start
        
        # GELU
        start = time.time()
        for _ in range(100):
            _ = ffn_gelu(x)
        gelu_time = time.time() - start
        
        # SwiGLU
        start = time.time()
        for _ in range(100):
            _ = ffn_swiglu(x)
        swiglu_time = time.time() - start
    
    print(f"ReLU推理时间: {relu_time:.4f}秒")
    print(f"GELU推理时间: {gelu_time:.4f}秒")
    print(f"SwiGLU推理时间: {swiglu_time:.4f}秒")

# 知识存储可视化
class KnowledgeAnalyzer:
    """分析FFN中的知识存储模式"""
    
    def __init__(self, ffn_model):
        self.ffn = ffn_model
    
    def analyze_neuron_activation(self, inputs, texts):
        """分析不同输入对神经元的激活模式"""
        
        activations = []
        with torch.no_grad():
            for input_tensor in inputs:
                # 获取第一层的激活
                hidden = torch.relu(self.ffn.linear1(input_tensor))
                activations.append(hidden.mean(dim=1))  # 平均池化
        
        # 分析激活模式
        activations = torch.stack(activations)
        
        # 找出最活跃的神经元
        neuron_activity = activations.mean(dim=0)
        top_neurons = torch.topk(neuron_activity, k=10).indices
        
        print("最活跃的神经元索引:", top_neurons.tolist())
        
        # 分析不同输入的激活相似性
        similarity_matrix = torch.cosine_similarity(
            activations.unsqueeze(1), 
            activations.unsqueeze(0), 
            dim=2
        )
        
        return {
            'activations': activations,
            'top_neurons': top_neurons,
            'similarity_matrix': similarity_matrix
        }

if __name__ == "__main__":
    compare_ffn_activations()
```

## ✅ 学习检验

- [ ] 理解FFN的基本结构和数学原理
- [ ] 掌握不同激活函数的特点和适用场景
- [ ] 理解FFN的知识存储机制
- [ ] 能实现和对比不同的FFN变体
- [ ] 理解FFN与注意力机制的互补关系

## 🔗 相关链接

- [下一节：编码器-解码器架构](encoder-decoder.md)
- [返回：Transformer基础概览](index.md)