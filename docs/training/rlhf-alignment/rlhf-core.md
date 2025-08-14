# RLHF核心技术

## 🎯 学习目标

深入理解强化学习人类反馈(RLHF)的核心原理，掌握PPO算法在LLM对齐中的应用，了解整个RLHF训练流程。

**重点面试问题预览：**
- RLHF的三个训练阶段分别是什么？
- 为什么选择PPO算法而不是其他强化学习算法？
- KL散度在RLHF中起什么作用？
- 如何解决奖励黑客(reward hacking)问题？

## 🏗️ RLHF技术架构

### 核心概念
RLHF(Reinforcement Learning from Human Feedback)是一种通过人类反馈来训练语言模型的方法，目标是让模型输出更符合人类偏好。

```
RLHF训练流程
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Stage 1: SFT   │───▶│Stage 2: RM训练  │───▶│ Stage 3: PPO优化│
│ 监督微调        │    │ 奖励模型训练    │    │ 强化学习优化    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
   ┌──────────┐           ┌──────────────┐        ┌─────────────┐
   │指令数据集│           │人类偏好数据集│        │PPO算法优化  │
   └──────────┘           └──────────────┘        └─────────────┘
```

## 📊 三阶段训练流程详解

### Stage 1: 监督微调(SFT)
**目标**: 让预训练模型学会遵循指令

```python
# SFT训练示例代码
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# SFT配置
sft_config = SFTConfig(
    output_dir="./sft_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    max_seq_length=512,
)

# 训练器
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=sft_config,
    train_dataset=instruction_dataset,
    formatting_func=format_instruction,
)

trainer.train()
```

**关键要点**：
- 使用高质量的指令-回答对数据
- 通常需要几万到几十万个样本
- 学习率一般设置为2e-5到5e-5

### Stage 2: 奖励模型训练(RM)
**目标**: 训练一个能够评估回答质量的奖励模型

```python
# 奖励模型训练代码
from transformers import AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig

# 加载SFT模型作为backbone
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./sft_model", 
    num_labels=1
)

# 奖励模型配置
reward_config = RewardConfig(
    output_dir="./reward_model",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
    max_length=512,
)

# 训练器
reward_trainer = RewardTrainer(
    model=reward_model,
    tokenizer=tokenizer,
    args=reward_config,
    train_dataset=preference_dataset,
)

reward_trainer.train()
```

**数据格式**：
```json
{
  "prompt": "解释量子计算的基本原理",
  "chosen": "量子计算利用量子力学原理...(高质量回答)",
  "rejected": "量子计算就是很快的计算...(低质量回答)"
}
```

**关键技术点**：
- 使用对比学习训练奖励模型
- 数据质量比数量更重要
- 需要处理标注者之间的分歧

### Stage 3: PPO强化学习优化
**目标**: 使用奖励模型指导策略模型优化

```python
# PPO训练代码
from trl import PPOTrainer, PPOConfig
import torch

# PPO配置
ppo_config = PPOConfig(
    model_name="./sft_model",
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=4,
    gradient_accumulation_steps=16,
    ppo_epochs=4,
    max_grad_norm=1.0,
    kl_penalty="kl",
    init_kl_coeff=0.2,
)

# 创建PPO训练器
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=policy_model,
    ref_model=reference_model,
    tokenizer=tokenizer,
    reward_model=reward_model,
)

# PPO训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        # 生成回答
        query_tensors = batch["input_ids"]
        response_tensors = ppo_trainer.generate(
            query_tensors, 
            max_new_tokens=128,
            temperature=0.7
        )
        
        # 计算奖励
        rewards = []
        for query, response in zip(query_tensors, response_tensors):
            reward = reward_model(
                torch.cat([query, response])
            ).logits.squeeze()
            rewards.append(reward)
        
        # PPO更新
        stats = ppo_trainer.step(
            query_tensors, 
            response_tensors, 
            rewards
        )
        
        print(f"Reward mean: {torch.tensor(rewards).mean():.2f}")
```

## 🎮 PPO算法核心原理

### 为什么选择PPO？

1. **稳定性**: 避免策略更新过大导致的性能崩塌
2. **效率**: 相比TRPO更简单，计算开销更小
3. **可调节性**: 通过clip参数控制更新幅度

### PPO目标函数

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比率
- $\hat{A}_t$ 是优势函数估计
- $\epsilon$ 是clip参数(通常设为0.2)

### KL散度约束

为了防止策略偏离参考模型过远，添加KL惩罚：

$$L_{total} = L_{PPO} - \beta \cdot KL(\pi_\theta || \pi_{ref})$$

```python
# KL散度计算示例
def compute_kl_penalty(logprobs_new, logprobs_old, kl_coeff):
    """计算KL散度惩罚"""
    kl = logprobs_old - logprobs_new
    kl_penalty = -kl_coeff * kl
    return kl_penalty

# 在训练中应用
total_loss = ppo_loss + kl_penalty
```

## ⚠️ RLHF中的关键挑战

### 1. 奖励黑客(Reward Hacking)
**问题**: 模型学会利用奖励模型的漏洞获得高分

**解决方案**：
- 使用KL散度约束限制偏移
- 定期更新奖励模型
- 使用多个奖励模型的ensemble

```python
# 防止奖励黑客的代码示例
def compute_reward_with_kl_penalty(response, query, reward_model, ref_model, kl_coeff=0.1):
    # 计算基础奖励
    reward = reward_model(response).logits.squeeze()
    
    # 计算KL惩罚
    current_logprobs = model(torch.cat([query, response])).log_softmax(dim=-1)
    ref_logprobs = ref_model(torch.cat([query, response])).log_softmax(dim=-1)
    kl_penalty = kl_coeff * (current_logprobs - ref_logprobs).sum()
    
    return reward - kl_penalty
```

### 2. 训练不稳定
**问题**: PPO训练容易发散或陷入局部最优

**解决方案**：
- 适当的学习率调度
- 梯度裁剪
- 早停策略

### 3. 计算资源需求
**问题**: 需要同时维护多个大模型

**资源需求分析**：
```
70B模型的RLHF训练需求：
├── Policy Model (训练中): ~280GB显存
├── Reference Model (推理): ~140GB显存  
├── Reward Model (推理): ~140GB显存
├── Value Model (训练中): ~280GB显存
└── 总计: ~840GB显存 (约需A100 80G × 12张)
```

## 📈 性能评估指标

### 1. 奖励模型准确率
```python
def evaluate_reward_model(reward_model, test_dataset):
    """评估奖励模型在偏好预测上的准确率"""
    correct = 0
    total = 0
    
    for batch in test_dataset:
        chosen_rewards = reward_model(batch["chosen"]).logits.squeeze()
        rejected_rewards = reward_model(batch["rejected"]).logits.squeeze()
        
        # 统计奖励模型是否正确预测人类偏好
        correct += (chosen_rewards > rejected_rewards).sum().item()
        total += len(batch["chosen"])
    
    accuracy = correct / total
    return accuracy
```

### 2. KL散度监控
```python
def monitor_kl_divergence(policy_model, reference_model, eval_prompts):
    """监控策略模型与参考模型的KL散度"""
    kl_divs = []
    
    for prompt in eval_prompts:
        policy_probs = F.softmax(policy_model(prompt).logits, dim=-1)
        ref_probs = F.softmax(reference_model(prompt).logits, dim=-1)
        
        kl_div = F.kl_div(policy_probs.log(), ref_probs, reduction='sum')
        kl_divs.append(kl_div.item())
    
    return np.mean(kl_divs)
```

### 3. 人类评估指标
- **有用性(Helpfulness)**: 回答是否解决了用户问题
- **无害性(Harmlessness)**: 回答是否避免有害内容
- **诚实性(Honesty)**: 回答是否准确可信

## 🔧 实战经验与最佳实践

### 1. 超参数调优建议

```python
# 推荐的RLHF超参数
RLHF_CONFIG = {
    # SFT阶段
    "sft_learning_rate": 2e-5,
    "sft_epochs": 3,
    "sft_batch_size": 4,
    
    # 奖励模型阶段  
    "rm_learning_rate": 1e-5,
    "rm_epochs": 1,
    "rm_batch_size": 4,
    
    # PPO阶段
    "ppo_learning_rate": 1.41e-5,
    "ppo_batch_size": 64,
    "ppo_epochs": 4,
    "init_kl_coeff": 0.2,
    "clip_range": 0.2,
}
```

### 2. 训练监控要点
- 奖励分数的变化趋势
- KL散度是否在合理范围内
- 生成质量的人工评估
- 训练损失的收敛情况

### 3. 常见问题排查
```python
def diagnose_rlhf_training(stats):
    """诊断RLHF训练中的常见问题"""
    
    # 检查奖励黑客
    if stats['reward_mean'] > stats['reward_threshold']:
        print("警告：可能存在奖励黑客问题")
        print(f"建议：增加KL系数从 {stats['kl_coeff']} 到 {stats['kl_coeff'] * 1.5}")
    
    # 检查训练发散
    if stats['kl_div'] > 10.0:
        print("警告：KL散度过大，训练可能发散")
        print("建议：降低学习率或增加KL系数")
    
    # 检查训练停滞
    if stats['reward_std'] < 0.1:
        print("警告：奖励分布过于集中，可能陷入局部最优")
        print("建议：增加探索性或调整temperature")
```

## 🎯 面试问答总结

### Q1: RLHF的三个训练阶段分别是什么？
**A**: 
1. **SFT(监督微调)**: 让预训练模型学会遵循指令
2. **RM(奖励模型训练)**: 训练一个评估回答质量的模型
3. **PPO(强化学习优化)**: 使用奖励模型指导策略优化

### Q2: 为什么选择PPO算法？
**A**: 
- **稳定性**: 通过clip机制避免策略更新过大
- **简单性**: 相比TRPO更容易实现和调优
- **有效性**: 在LLM对齐任务上表现良好

### Q3: KL散度在RLHF中起什么作用？
**A**: 
- **约束作用**: 防止策略模型偏离参考模型过远
- **稳定训练**: 避免奖励黑客和训练发散
- **保持能力**: 确保模型不会失去原有的语言能力

### Q4: 如何解决奖励黑客问题？
**A**:
- 使用KL散度惩罚限制模型偏移
- 定期更新和验证奖励模型
- 使用多样化的评估方法和人工检查

## 🚀 学习建议

1. **理论先行**: 先理解强化学习基础概念
2. **代码实践**: 跑通完整的RLHF流程
3. **参数调优**: 了解各个超参数的作用
4. **问题诊断**: 学会识别和解决常见问题

这一技术是现代LLM对齐的核心，掌握好了对面试和实际工作都很有帮助！