# 第2节：Attention升级技术

## 🎯 学习目标

掌握现代大模型中的注意力机制优化技术，理解推理加速和训练稳定性的工程解决方案。

**重点面试问题预览：**
- MHA/MQA/GQA/MLA的区别和优势
- KV Cache的工作原理和加速效果
- LayerNorm vs RMSNorm的选择
- RoPE位置编码的数学原理

## 📅 学习计划

**建议学习时间：3天**

- **Day 1**: 注意力变体深度学习 (MHA→MQA→GQA→MLA)
- **Day 2**: KV Cache技术 + 归一化技术详解
- **Day 3**: 位置编码升级 + 综合技术对比分析

## 📚 学习路径

### 1. [多头注意力变体](mha-variants.md)
- MHA → MQA → GQA → MLA演进
- 注意力头数优化策略
- 计算复杂度分析

### 2. [KV Cache技术](kv-cache.md)
- 推理加速原理
- 内存优化策略
- 实现细节和代码示例

### 3. [归一化技术](normalization.md)
- BatchNorm vs LayerNorm vs RMSNorm
- Pre-Norm vs Post-Norm
- 训练稳定性分析

### 4. [位置编码](positional-encoding.md)
- 绝对位置 vs 相对位置编码
- RoPE旋转位置编码推导
- 长序列处理能力

## 📖 核心阅读材料

### 必读技术文章
1. [Transformer的Attention及其各种变体](https://lengm.cn/post/20250226_attention/) - 冷眸博客
2. [缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA](https://spaces.ac.cn/archives/10091) - 科学空间
3. [大模型中常见的3种Norm](https://zhuanlan.zhihu.com/p/648987575) - 知乎
4. [为什么当前主流的大模型都使用RMS-Norm？](https://zhuanlan.zhihu.com/p/12392406696) - 知乎
5. [为什么Pre Norm的效果不如Post Norm？](https://spaces.ac.cn/archives/9009) - 科学空间
6. [Sinusoidal位置编码追根溯源](https://kexue.fm/archives/8231) - 科学空间
7. [博采众长的旋转式位置编码](https://kexue.fm/archives/8265) - 科学空间

### 选读深入材料
- [BN究竟起了什么作用？](https://spaces.ac.cn/archives/6992) - 科学空间

## ✅ 学习检验标准

完成以下项目才算掌握本节：

1. **技术对比**: 清晰区分MHA/MQA/GQA/MLA的优缺点和适用场景
2. **代码实现**: 完成GQA适配和KV Cache演示代码
3. **原理理解**: 能从数学角度解释RoPE和RMSNorm的工作原理
4. **面试准备**: 能解释每种技术选择背后的工程trade-off

## 💡 学习提示

这一节技术含量很高，是现代大模型的核心优化技术，建议：
- **循序渐进**: 先理解基础概念，再深入数学推导
- **重点关注**: 每种技术的motivation和解决的具体问题
- **对比学习**: 通过技术对比加深理解各自的trade-off
- **实践验证**: 通过代码实现加深对原理的理解
- **面试导向**: 重点掌握技术选择的工程考量

## 🚀 开始学习

选择感兴趣的技术模块深入学习，每个都是现代大模型的核心技术！