# 监督微调技术

## 🎯 学习目标

全面掌握监督微调(Supervised Fine-Tuning, SFT)技术，理解指令微调的核心原理，学会构建高质量的指令数据集和实现有效的微调流程。

**重点面试问题预览：**
- SFT在整个训练流程中的作用是什么？
- 指令微调与传统微调有什么区别？
- 如何构建高质量的指令数据集？
- SFT训练中常见的问题和解决方案？

## 🏗️ SFT核心概念

### 定义与作用

监督微调(SFT)是使用**指令-回答对**数据训练模型遵循指令和产生期望输出的过程，是连接预训练模型与实际应用的关键桥梁。

```
SFT在训练流程中的位置
┌─────────────────────────────────────────────────────────────────┐
│                    完整LLM训练流程                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  预训练 → 后预训练 → 监督微调(SFT) → 强化学习对齐                 │
│                        ▲                                        │
│                   关键转换点                                      │
│                                                                 │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐  │
│  │   语言建模       │──▶│   指令遵循       │──▶│   人类偏好       │  │
│  │ 下一词预测       │   │ 任务完成        │   │ 价值对齐        │  │
│  │ 无监督学习       │   │ 监督学习        │   │ 强化学习        │  │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘  │
│         ▲                      ▲                      ▲          │
│    通用语言能力              指令遵循能力              对齐能力     │
└─────────────────────────────────────────────────────────────────┘
```

### 核心价值

1. **能力转换**: 从语言建模能力转换为任务执行能力
2. **格式规范**: 教会模型标准的输入输出格式
3. **指令遵循**: 培养遵循用户指令的基础能力
4. **多任务统一**: 用统一格式处理多种不同任务

## 📊 指令数据构建

### 高质量指令数据的特征

```python
class InstructionDataBuilder:
    """指令数据构建器"""
    
    def __init__(self):
        self.quality_criteria = {
            "多样性": "涵盖不同领域、任务类型、难度层级",
            "准确性": "回答准确、事实正确、逻辑清晰",
            "完整性": "回答完整、结构化、满足指令要求",
            "一致性": "格式统一、风格一致、标准规范"
        }
    
    def instruction_template_design(self):
        """指令模板设计"""
        
        templates = {
            "基础问答模板": {
                "格式": "### Instruction:\n{instruction}\n\n### Response:\n{response}",
                "适用": "简单问答、知识查询类任务",
                "示例": {
                    "instruction": "解释什么是机器学习",
                    "response": "机器学习是一种人工智能技术，通过算法让计算机从数据中学习规律..."
                }
            },
            
            "多轮对话模板": {
                "格式": "### Conversation:\n{conversation_history}\n\n### Human:\n{human_input}\n\n### Assistant:\n{assistant_response}",
                "适用": "对话系统、聊天机器人",
                "示例": {
                    "conversation_history": "之前的对话历史",
                    "human_input": "用户当前输入",
                    "assistant_response": "助手回应"
                }
            },
            
            "任务导向模板": {
                "格式": "### Task:\n{task_description}\n\n### Input:\n{input_data}\n\n### Output:\n{expected_output}",
                "适用": "结构化任务、数据处理",
                "示例": {
                    "task_description": "将以下文本翻译成英文",
                    "input_data": "你好，世界",
                    "expected_output": "Hello, world"
                }
            },
            
            "思维链模板": {
                "格式": "### Problem:\n{problem}\n\n### Solution:\n{reasoning_steps}\n\n### Answer:\n{final_answer}",
                "适用": "推理任务、数学问题、逻辑分析",
                "示例": {
                    "problem": "计算 (3+5) × 2 = ?",
                    "reasoning_steps": "首先计算括号内: 3+5=8\n然后乘以2: 8×2=16",
                    "final_answer": "16"
                }
            }
        }
        
        return templates
    
    def data_diversity_strategy(self):
        """数据多样性策略"""
        
        diversity_dimensions = {
            "任务类型多样性": {
                "文本生成": ["创意写作", "摘要生成", "续写补全"],
                "信息抽取": ["实体识别", "关系抽取", "关键词提取"],
                "文本分析": ["情感分析", "主题分类", "意图理解"],
                "推理问答": ["常识推理", "数学计算", "逻辑推理"],
                "代码相关": ["代码生成", "代码解释", "Bug修复"],
                "创意任务": ["头脑风暴", "故事创作", "诗歌创作"]
            },
            
            "领域覆盖多样性": {
                "科技领域": "计算机、人工智能、生物技术",
                "商业领域": "金融、管理、营销",
                "教育领域": "数学、物理、化学、历史",
                "生活领域": "健康、美食、旅游",
                "文化领域": "文学、艺术、哲学"
            },
            
            "难度层级多样性": {
                "简单": "基础概念、简单问答、直接查询",
                "中等": "分析解释、多步推理、综合应用",
                "困难": "复杂推理、创新思考、专业深度"
            },
            
            "语言风格多样性": {
                "正式风格": "学术论文、商务报告、官方文档",
                "非正式风格": "日常对话、社交媒体、个人博客",
                "专业风格": "技术文档、医学报告、法律条文",
                "创意风格": "文学创作、广告文案、艺术评论"
            }
        }
        
        return diversity_dimensions
    
    def generate_instruction_data(self, seed_topics, num_per_topic=50):
        """生成指令数据"""
        
        instruction_data = []
        templates = self.instruction_template_design()
        
        # 指令生成提示模板
        generation_prompts = {
            "问答类": "基于主题'{topic}'，生成一个需要详细解释的问题：",
            "任务类": "基于主题'{topic}'，设计一个具体的任务指令：",
            "推理类": "基于主题'{topic}'，创建一个需要逻辑推理的问题：",
            "创意类": "基于主题'{topic}'，设计一个创意生成任务："
        }
        
        for topic in seed_topics:
            for prompt_type, prompt_template in generation_prompts.items():
                for i in range(num_per_topic // len(generation_prompts)):
                    
                    # 生成指令
                    instruction_prompt = prompt_template.format(topic=topic)
                    instruction = self.generate_with_llm(instruction_prompt)
                    
                    # 生成回答
                    response_prompt = f"请详细回答以下问题：\n{instruction}"
                    response = self.generate_with_llm(response_prompt)
                    
                    # 质量检查
                    if self.quality_check(instruction, response):
                        instruction_data.append({
                            "instruction": instruction,
                            "response": response,
                            "topic": topic,
                            "type": prompt_type,
                            "quality_score": self.calculate_quality_score(instruction, response)
                        })
        
        return instruction_data
    
    def quality_check(self, instruction, response):
        """质量检查"""
        
        # 基础长度检查
        if len(instruction.split()) < 5 or len(response.split()) < 10:
            return False
        
        # 相关性检查
        relevance_score = self.calculate_relevance(instruction, response)
        if relevance_score < 0.7:
            return False
        
        # 完整性检查
        completeness_score = self.calculate_completeness(response)
        if completeness_score < 0.6:
            return False
        
        # 安全性检查
        if self.contains_harmful_content(instruction + " " + response):
            return False
        
        return True
    
    def create_balanced_dataset(self, instruction_data, target_size=10000):
        """创建平衡数据集"""
        
        # 按类型和主题分组
        grouped_data = {}
        for item in instruction_data:
            key = f"{item['type']}_{item['topic']}"
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(item)
        
        # 计算每组目标数量
        num_groups = len(grouped_data)
        per_group_target = target_size // num_groups
        
        # 从每组采样
        balanced_data = []
        for group, items in grouped_data.items():
            # 按质量分数排序，选择高质量样本
            sorted_items = sorted(items, key=lambda x: x['quality_score'], reverse=True)
            selected = sorted_items[:min(per_group_target, len(sorted_items))]
            balanced_data.extend(selected)
        
        return balanced_data

# 使用示例
builder = InstructionDataBuilder()
templates = builder.instruction_template_design()
diversity = builder.data_diversity_strategy()

print("指令模板:", templates)
print("多样性策略:", diversity)
```

### 数据增强技术

```python
class InstructionDataAugmentation:
    """指令数据增强技术"""
    
    def __init__(self, base_llm):
        self.base_llm = base_llm
        
    def paraphrase_augmentation(self, instruction, response):
        """改写增强"""
        
        # 指令改写
        paraphrase_prompts = [
            f"请用不同的方式表达以下指令，保持含义不变：\n{instruction}",
            f"将以下指令改写得更加正式：\n{instruction}",
            f"将以下指令改写得更加简洁：\n{instruction}"
        ]
        
        augmented_pairs = []
        
        for prompt in paraphrase_prompts:
            new_instruction = self.base_llm.generate(prompt)
            
            # 检查改写质量
            if self.is_valid_paraphrase(instruction, new_instruction):
                # 为新指令生成对应回答
                new_response = self.base_llm.generate(
                    f"请回答：{new_instruction}"
                )
                
                augmented_pairs.append({
                    "instruction": new_instruction,
                    "response": new_response,
                    "augmentation_type": "paraphrase",
                    "original_instruction": instruction
                })
        
        return augmented_pairs
    
    def difficulty_augmentation(self, instruction, response):
        """难度调节增强"""
        
        # 简化版本
        simplify_prompt = f"""
        将以下指令简化，使其更容易理解：
        
        原指令：{instruction}
        
        简化指令：
        """
        
        simple_instruction = self.base_llm.generate(simplify_prompt)
        simple_response = self.base_llm.generate(f"请简单回答：{simple_instruction}")
        
        # 复杂版本  
        complicate_prompt = f"""
        将以下指令扩展得更加具体和复杂：
        
        原指令：{instruction}
        
        扩展指令：
        """
        
        complex_instruction = self.base_llm.generate(complicate_prompt)
        complex_response = self.base_llm.generate(f"请详细回答：{complex_instruction}")
        
        return [
            {
                "instruction": simple_instruction,
                "response": simple_response,
                "augmentation_type": "simplification",
                "difficulty_level": "easy"
            },
            {
                "instruction": complex_instruction, 
                "response": complex_response,
                "augmentation_type": "complication",
                "difficulty_level": "hard"
            }
        ]
    
    def format_augmentation(self, instruction, response):
        """格式变换增强"""
        
        format_variations = []
        
        # 问答格式
        qa_format = {
            "instruction": f"问题：{instruction}",
            "response": f"回答：{response}",
            "format_type": "qa_format"
        }
        
        # 对话格式
        dialog_format = {
            "instruction": f"用户：{instruction}",
            "response": f"助手：{response}",
            "format_type": "dialog_format"
        }
        
        # 结构化格式
        structured_format = {
            "instruction": f"任务：{instruction}\n要求：请提供详细回答",
            "response": f"解答：\n{response}",
            "format_type": "structured_format"
        }
        
        format_variations.extend([qa_format, dialog_format, structured_format])
        
        return format_variations
    
    def context_augmentation(self, instruction, response):
        """上下文增强"""
        
        # 添加背景信息
        context_enhanced = []
        
        contexts = [
            "在学术研究环境中，",
            "在日常生活场景下，", 
            "在商业应用中，",
            "在教育教学中，"
        ]
        
        for context in contexts:
            contextualized_instruction = context + instruction.lower()
            
            # 生成适应上下文的回答
            context_response = self.base_llm.generate(
                f"在{context[:-1]}的背景下，请回答：{instruction}"
            )
            
            context_enhanced.append({
                "instruction": contextualized_instruction,
                "response": context_response,
                "augmentation_type": "context_enhancement",
                "context": context.strip("，")
            })
        
        return context_enhanced

# 数据增强使用示例
def augment_instruction_dataset(original_data, target_multiplier=3):
    """数据增强流程"""
    
    augmenter = InstructionDataAugmentation(base_llm)
    augmented_data = []
    
    for item in original_data:
        instruction = item["instruction"]
        response = item["response"]
        
        # 原始数据
        augmented_data.append(item)
        
        # 改写增强
        paraphrases = augmenter.paraphrase_augmentation(instruction, response)
        augmented_data.extend(paraphrases)
        
        # 难度增强
        difficulty_variants = augmenter.difficulty_augmentation(instruction, response)
        augmented_data.extend(difficulty_variants)
        
        # 格式增强
        format_variants = augmenter.format_augmentation(instruction, response)
        augmented_data.extend(format_variants)
        
        # 上下文增强
        context_variants = augmenter.context_augmentation(instruction, response)
        augmented_data.extend(context_variants)
        
        # 控制增强倍数
        if len(augmented_data) >= len(original_data) * target_multiplier:
            break
    
    return augmented_data[:len(original_data) * target_multiplier]
```

## 🚀 SFT训练实现

### 完整训练流程

```python
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import torch

class SupervisedFineTuner:
    """监督微调训练器"""
    
    def __init__(self, model_name, max_seq_length=2048):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        
        # 初始化tokenizer和model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            load_in_8bit=True,  # 8bit量化节省显存
            trust_remote_code=True
        )
        
        # 启用梯度检查点节省显存
        self.model.gradient_checkpointing_enable()
    
    def format_instruction_data(self, instruction_data, template_type="alpaca"):
        """格式化指令数据"""
        
        formatted_data = []
        
        templates = {
            "alpaca": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}",
            
            "vicuna": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n\nUSER: {instruction}\nASSISTANT: {response}",
            
            "chinese": "以下是一个描述任务的指令。请编写一个适当完成请求的回答。\n\n### 指令：\n{instruction}\n\n### 回答：\n{response}",
            
            "simple": "### Question:\n{instruction}\n\n### Answer:\n{response}"
        }
        
        template = templates.get(template_type, templates["alpaca"])
        
        for item in instruction_data:
            formatted_text = template.format(
                instruction=item["instruction"],
                response=item["response"]
            )
            
            formatted_data.append({
                "text": formatted_text,
                "instruction": item["instruction"],
                "response": item["response"]
            })
        
        return formatted_data
    
    def tokenize_function(self, examples):
        """Tokenization函数"""
        
        # Tokenize完整文本
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.max_seq_length,
            return_tensors=None
        )
        
        # 为指令微调设置labels
        # 只对response部分计算loss，instruction部分mask掉
        input_ids = tokenized["input_ids"]
        labels = []
        
        for i, text in enumerate(examples["text"]):
            # 找到response开始位置
            if "### Response:" in text:
                response_start = text.find("### Response:") + len("### Response:")
            elif "ASSISTANT:" in text:
                response_start = text.find("ASSISTANT:") + len("ASSISTANT:")
            elif "### 回答：" in text:
                response_start = text.find("### 回答：") + len("### 回答：")
            else:
                response_start = len(text) // 2  # 默认从中间开始
            
            # Tokenize到response开始位置的文本
            prefix_tokens = self.tokenizer(
                text[:response_start],
                truncation=True,
                padding=False,
                max_length=self.max_seq_length,
                return_tensors=None
            )["input_ids"]
            
            # 创建labels：instruction部分为-100，response部分为正常token
            label = [-100] * len(prefix_tokens) + input_ids[i][len(prefix_tokens):]
            
            # 确保长度一致
            if len(label) > len(input_ids[i]):
                label = label[:len(input_ids[i])]
            elif len(label) < len(input_ids[i]):
                label.extend(input_ids[i][len(label):])
            
            labels.append(label)
        
        tokenized["labels"] = labels
        return tokenized
    
    def create_data_collator(self):
        """创建数据整理器"""
        
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # 不使用掩码语言模型
            pad_to_multiple_of=8,  # 为了tensor core优化
        )
    
    def train(self, train_data, eval_data=None, output_dir="./sft_output", 
              epochs=3, batch_size=4, learning_rate=2e-4):
        """执行SFT训练"""
        
        # 格式化数据
        formatted_train = self.format_instruction_data(train_data)
        train_dataset = Dataset.from_list(formatted_train)
        
        # Tokenize数据
        tokenized_train = train_dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        
        # 处理验证数据
        eval_dataset = None
        if eval_data:
            formatted_eval = self.format_instruction_data(eval_data)
            eval_dataset = Dataset.from_list(formatted_eval)
            tokenized_eval = eval_dataset.map(
                self.tokenize_function,
                batched=True,
                remove_columns=eval_dataset.column_names
            )
            eval_dataset = tokenized_eval
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            
            # 基础训练设置
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=8,  # 有效batch size = batch_size * 8
            
            # 优化器设置
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            
            # 精度和内存优化
            bf16=True,
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            
            # 评估和保存
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_loss" if eval_dataset else None,
            
            # 日志设置
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            report_to="tensorboard",
            
            # 其他优化
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
            remove_unused_columns=False,
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=eval_dataset,
            data_collator=self.create_data_collator(),
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        print("开始SFT训练...")
        train_result = trainer.train()
        
        # 保存模型
        trainer.save_model()
        trainer.save_state()
        
        print(f"训练完成！模型保存在: {output_dir}")
        print(f"训练统计: {train_result.metrics}")
        
        return trainer, train_result

# 多任务SFT训练
class MultiTaskSFTTrainer(SupervisedFineTuner):
    """多任务SFT训练器"""
    
    def __init__(self, model_name, task_weights=None):
        super().__init__(model_name)
        self.task_weights = task_weights or {}
    
    def prepare_multitask_data(self, task_datasets):
        """准备多任务数据"""
        
        all_data = []
        task_info = {}
        
        for task_name, task_data in task_datasets.items():
            # 计算任务权重
            weight = self.task_weights.get(task_name, 1.0)
            
            # 根据权重采样数据
            sample_size = int(len(task_data) * weight)
            sampled_data = np.random.choice(task_data, sample_size, replace=False)
            
            # 添加任务标识
            for item in sampled_data:
                item_with_task = item.copy()
                item_with_task["task"] = task_name
                all_data.append(item_with_task)
            
            task_info[task_name] = {
                "original_size": len(task_data),
                "sampled_size": sample_size,
                "weight": weight
            }
        
        # 随机打乱
        np.random.shuffle(all_data)
        
        print("多任务数据统计:")
        for task, info in task_info.items():
            print(f"  {task}: {info['sampled_size']} 样本 (权重: {info['weight']})")
        
        return all_data, task_info
    
    def task_aware_formatting(self, instruction_data):
        """任务感知的格式化"""
        
        task_templates = {
            "qa": "Question: {instruction}\nAnswer: {response}",
            "summarization": "Summarize the following text:\n{instruction}\n\nSummary: {response}",
            "translation": "Translate to English:\n{instruction}\n\nTranslation: {response}",
            "classification": "Classify the following text:\n{instruction}\n\nCategory: {response}",
            "generation": "Complete the following:\n{instruction}\n\nCompletion: {response}"
        }
        
        formatted_data = []
        
        for item in instruction_data:
            task = item.get("task", "general")
            template = task_templates.get(task, task_templates["qa"])
            
            formatted_text = template.format(
                instruction=item["instruction"],
                response=item["response"]
            )
            
            formatted_data.append({
                "text": formatted_text,
                "task": task,
                "instruction": item["instruction"],
                "response": item["response"]
            })
        
        return formatted_data

# 使用示例
def run_sft_training():
    """运行SFT训练示例"""
    
    # 准备训练数据
    train_data = [
        {
            "instruction": "解释什么是深度学习",
            "response": "深度学习是机器学习的一个分支，它使用多层神经网络来学习数据的复杂模式..."
        },
        {
            "instruction": "写一个Python函数计算斐波那契数列",
            "response": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        },
        # ... 更多训练数据
    ]
    
    # 创建训练器
    trainer = SupervisedFineTuner("microsoft/DialoGPT-medium")
    
    # 执行训练
    trained_model, results = trainer.train(
        train_data=train_data,
        eval_data=train_data[:100],  # 使用部分数据作为验证集
        output_dir="./my_sft_model",
        epochs=3,
        batch_size=2,
        learning_rate=2e-4
    )
    
    return trained_model, results

# 运行训练
# model, results = run_sft_training()
```

## 📊 SFT效果评估

### 多维度评估框架

```python
class SFTEvaluator:
    """SFT效果评估器"""
    
    def __init__(self, base_model, sft_model, tokenizer):
        self.base_model = base_model
        self.sft_model = sft_model
        self.tokenizer = tokenizer
    
    def instruction_following_evaluation(self, test_instructions):
        """指令遵循能力评估"""
        
        results = {
            "base_model": [],
            "sft_model": [],
            "improvement_scores": []
        }
        
        for instruction in test_instructions:
            # 基础模型响应
            base_response = self.generate_response(self.base_model, instruction)
            base_score = self.evaluate_instruction_following(instruction, base_response)
            
            # SFT模型响应
            sft_response = self.generate_response(self.sft_model, instruction)
            sft_score = self.evaluate_instruction_following(instruction, sft_response)
            
            # 记录结果
            results["base_model"].append({
                "instruction": instruction,
                "response": base_response,
                "score": base_score
            })
            
            results["sft_model"].append({
                "instruction": instruction,
                "response": sft_response,
                "score": sft_score
            })
            
            results["improvement_scores"].append(sft_score - base_score)
        
        # 计算总体指标
        avg_improvement = np.mean(results["improvement_scores"])
        improvement_rate = np.mean([s > 0 for s in results["improvement_scores"]])
        
        return {
            "average_improvement": avg_improvement,
            "improvement_rate": improvement_rate,
            "detailed_results": results
        }
    
    def evaluate_instruction_following(self, instruction, response):
        """评估指令遵循质量"""
        
        score = 0.0
        
        # 1. 格式检查 (0-0.3分)
        format_score = self.check_response_format(response)
        score += format_score * 0.3
        
        # 2. 相关性检查 (0-0.4分)
        relevance_score = self.check_relevance(instruction, response)
        score += relevance_score * 0.4
        
        # 3. 完整性检查 (0-0.2分)
        completeness_score = self.check_completeness(instruction, response)
        score += completeness_score * 0.2
        
        # 4. 质量检查 (0-0.1分)
        quality_score = self.check_response_quality(response)
        score += quality_score * 0.1
        
        return score
    
    def task_specific_evaluation(self, task_test_suites):
        """任务特定评估"""
        
        task_results = {}
        
        for task_name, test_suite in task_test_suites.items():
            print(f"评估任务: {task_name}")
            
            task_scores = {
                "base_model": [],
                "sft_model": []
            }
            
            for test_case in test_suite:
                instruction = test_case["instruction"]
                expected = test_case.get("expected", "")
                
                # 生成回答
                base_response = self.generate_response(self.base_model, instruction)
                sft_response = self.generate_response(self.sft_model, instruction)
                
                # 任务特定评估
                base_score = self.task_specific_score(
                    task_name, instruction, base_response, expected
                )
                sft_score = self.task_specific_score(
                    task_name, instruction, sft_response, expected
                )
                
                task_scores["base_model"].append(base_score)
                task_scores["sft_model"].append(sft_score)
            
            # 计算任务平均分
            task_results[task_name] = {
                "base_avg": np.mean(task_scores["base_model"]),
                "sft_avg": np.mean(task_scores["sft_model"]),
                "improvement": np.mean(task_scores["sft_model"]) - np.mean(task_scores["base_model"])
            }
        
        return task_results
    
    def output_format_analysis(self, test_instructions):
        """输出格式分析"""
        
        format_metrics = {
            "结构化程度": [],
            "长度适中性": [],
            "语言流畅性": [],
            "专业性": []
        }
        
        for instruction in test_instructions:
            sft_response = self.generate_response(self.sft_model, instruction)
            
            # 分析各个维度
            format_metrics["结构化程度"].append(
                self.analyze_structure(sft_response)
            )
            format_metrics["长度适中性"].append(
                self.analyze_length_appropriateness(instruction, sft_response)
            )
            format_metrics["语言流畅性"].append(
                self.analyze_fluency(sft_response)
            )
            format_metrics["专业性"].append(
                self.analyze_professionalism(sft_response)
            )
        
        # 计算平均分
        avg_metrics = {
            metric: np.mean(scores) 
            for metric, scores in format_metrics.items()
        }
        
        return avg_metrics
    
    def generate_evaluation_report(self, instruction_eval, task_eval, format_eval):
        """生成评估报告"""
        
        report = {
            "SFT训练效果总结": {
                "指令遵循改善": f"{instruction_eval['average_improvement']:.3f}分 ({instruction_eval['improvement_rate']*100:.1f}%样本改善)",
                "任务能力提升": {
                    task: f"{results['improvement']:.3f}分" 
                    for task, results in task_eval.items()
                },
                "输出格式质量": {
                    metric: f"{score:.3f}分" 
                    for metric, score in format_eval.items()
                }
            },
            
            "关键发现": {
                "最佳改善任务": max(task_eval.items(), key=lambda x: x[1]['improvement'])[0],
                "需要关注的任务": [
                    task for task, results in task_eval.items() 
                    if results['improvement'] < 0.1
                ],
                "格式化质量": "良好" if np.mean(list(format_eval.values())) > 0.7 else "需要改进"
            },
            
            "改进建议": self.generate_improvement_suggestions(
                instruction_eval, task_eval, format_eval
            )
        }
        
        return report

# 评估使用示例
def evaluate_sft_model(base_model_path, sft_model_path, tokenizer):
    """评估SFT模型效果"""
    
    # 加载模型
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
    sft_model = AutoModelForCausalLM.from_pretrained(sft_model_path)
    
    # 创建评估器
    evaluator = SFTEvaluator(base_model, sft_model, tokenizer)
    
    # 准备测试数据
    test_instructions = [
        "解释量子计算的基本原理",
        "写一个排序算法的Python实现",
        "分析这段代码的时间复杂度",
        # ... 更多测试指令
    ]
    
    task_test_suites = {
        "问答": [
            {"instruction": "什么是机器学习？", "expected": "详细解释"},
            {"instruction": "人工智能的发展历史", "expected": "历史介绍"}
        ],
        "代码生成": [
            {"instruction": "写一个快速排序函数", "expected": "Python代码"},
            {"instruction": "实现二叉树遍历", "expected": "算法实现"}
        ]
    }
    
    # 执行评估
    instruction_results = evaluator.instruction_following_evaluation(test_instructions)
    task_results = evaluator.task_specific_evaluation(task_test_suites)
    format_results = evaluator.output_format_analysis(test_instructions)
    
    # 生成报告
    evaluation_report = evaluator.generate_evaluation_report(
        instruction_results, task_results, format_results
    )
    
    print("SFT评估报告:")
    print(json.dumps(evaluation_report, indent=2, ensure_ascii=False))
    
    return evaluation_report
```

## 🎯 面试问答总结

### Q1: SFT在整个训练流程中的作用是什么？
**A**: SFT是关键的能力转换阶段：
- **功能转换**: 从语言建模转为任务执行
- **格式规范**: 教会模型标准输入输出格式
- **指令遵循**: 建立基础的指令理解和执行能力
- **后续基础**: 为RLHF等后续对齐训练提供基础

### Q2: 指令微调与传统微调有什么区别？
**A**:
- **数据格式**: 指令微调使用指令-回答对，传统微调使用标注数据
- **训练目标**: 指令微调训练通用指令遵循，传统微调训练特定任务
- **泛化能力**: 指令微调具备零样本泛化，传统微调局限于训练任务
- **应用范围**: 指令微调适用于对话助手，传统微调适用于专门任务

### Q3: 如何构建高质量的指令数据集？
**A**: 四个关键维度：
- **多样性**: 任务类型、领域、难度、风格的全面覆盖
- **准确性**: 回答准确、事实正确、逻辑清晰
- **完整性**: 回答完整、结构化、满足指令要求
- **一致性**: 格式统一、风格一致、标准规范

### Q4: SFT训练中常见的问题和解决方案？
**A**:
- **过拟合**: 使用验证集早停、增加数据多样性、适当正则化
- **格式不规范**: 统一数据模板、标准化预处理、质量检查
- **能力不均衡**: 平衡采样、任务权重、多轮训练
- **计算资源不足**: 量化训练、梯度累积、模型并行

## 🚀 学习建议

1. **理解核心**: 深入理解SFT在整个训练流程中的关键作用
2. **数据为王**: 重点掌握高质量指令数据的构建方法
3. **实践验证**: 在实际数据上完整跑通SFT训练流程
4. **效果评估**: 建立多维度的SFT效果评估体系

SFT是连接预训练与应用的关键桥梁，是现代LLM不可或缺的核心技术！