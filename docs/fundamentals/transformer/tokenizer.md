# Tokenizer技术

## 🎯 本节目标

深入理解Tokenizer的工作原理，掌握不同tokenization算法的特点和应用场景，为大模型的文本处理奠定基础。

## 📝 技术原理解析

### Tokenizer基本概念

**Tokenizer**是将原始文本转换为模型可处理的数值表示的关键组件，是连接人类语言和机器学习模型的桥梁。

#### 核心功能
```python
# Tokenizer的基本流程
text = "Hello, world! This is a test."
     ↓ 1. 文本规范化(Normalization)
normalized = "hello world this is a test"
     ↓ 2. 预分词(Pre-tokenization)  
pre_tokens = ["hello", "world", "this", "is", "a", "test"]
     ↓ 3. 模型处理(Model Processing)
tokens = ["hello", "wor", "##ld", "this", "is", "a", "test"]
     ↓ 4. 后处理(Post-processing)
final_tokens = ["[CLS]", "hello", "wor", "##ld", "this", "is", "a", "test", "[SEP]"]
     ↓ 5. 数值映射
token_ids = [101, 7592, 24829, 2094, 2023, 2003, 1037, 3231, 102]
```

### 三大Tokenization范式

#### 1. 词级别分词 (Word-level)

**原理**: 将文本按完整单词分割

```python
class WordLevelTokenizer:
    def __init__(self, vocab_path):
        self.word_to_id = self.load_vocab(vocab_path)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.unk_token = "[UNK]"
    
    def tokenize(self, text):
        words = text.lower().split()
        tokens = []
        for word in words:
            if word in self.word_to_id:
                tokens.append(word)
            else:
                tokens.append(self.unk_token)  # 未知词处理
        return tokens
    
    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.word_to_id.get(token, self.word_to_id[self.unk_token]) 
                for token in tokens]
```

**优势**:
- 保持语义完整性
- 符合人类认知习惯
- 实现简单直观

**劣势**:
- 词汇表庞大(通常>100k)
- 未登录词(OOV)问题严重
- 无法处理形态变化丰富的语言

#### 2. 字符级别分词 (Character-level)

**原理**: 将文本分解为单个字符

```python
class CharacterLevelTokenizer:
    def __init__(self):
        # 基本字符集
        self.chars = list("abcdefghijklmnopqrstuvwxyz0123456789 .,!?'")
        self.char_to_id = {char: i for i, char in enumerate(self.chars)}
        self.id_to_char = {i: char for i, char in enumerate(self.chars)}
    
    def tokenize(self, text):
        return list(text.lower())
    
    def encode(self, text):
        chars = self.tokenize(text)
        return [self.char_to_id.get(char, 0) for char in chars]  # 0 for unknown
    
    def decode(self, token_ids):
        chars = [self.id_to_char.get(id, "") for id in token_ids]
        return "".join(chars)
```

**优势**:
- 词汇表极小(通常<100)
- 无OOV问题
- 对拼写错误鲁棒

**劣势**:
- 序列长度大幅增加
- 丢失词边界信息
- 语义理解困难

#### 3. 子词级别分词 (Subword-level)

**核心思想**: 在词汇表大小和语义保持之间找到平衡

## 🔬 主流Subword算法详解

### 1. BPE (Byte Pair Encoding)

**算法原理**: 迭代合并最频繁的字符对

#### 训练过程
```python
class BPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = []
    
    def train(self, corpus, vocab_size):
        # 1. 初始化：将所有字符作为基础词汇
        word_freqs = self.get_word_frequencies(corpus)
        
        # 将每个词分解为字符
        vocab = set()
        for word in word_freqs:
            for char in word:
                vocab.add(char)
        
        vocab = list(vocab)
        
        # 2. 迭代合并最频繁的字符对
        while len(vocab) < vocab_size:
            # 统计所有字符对的频率
            pairs = self.get_all_pairs(word_freqs)
            
            if not pairs:
                break
            
            # 找到最频繁的字符对
            best_pair = max(pairs, key=pairs.get)
            
            # 合并这个字符对
            vocab.append(''.join(best_pair))
            self.merges.append(best_pair)
            
            # 更新词频字典
            word_freqs = self.merge_vocab(best_pair, word_freqs)
        
        self.vocab = {token: i for i, token in enumerate(vocab)}
    
    def get_all_pairs(self, word_freqs):
        """获取所有相邻字符对及其频率"""
        pairs = {}
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs
    
    def merge_vocab(self, pair, word_freqs):
        """合并指定字符对"""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word in word_freqs:
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = word_freqs[word]
        
        return new_word_freqs
    
    def tokenize(self, text):
        """使用训练好的BPE进行分词"""
        words = text.split()
        result = []
        
        for word in words:
            # 将词分解为字符
            word_tokens = list(word)
            
            # 应用所有学到的合并规则
            for merge in self.merges:
                word_tokens = self.apply_merge(word_tokens, merge)
            
            result.extend(word_tokens)
        
        return result
    
    def apply_merge(self, tokens, merge):
        """应用单个合并规则"""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (i < len(tokens) - 1 and 
                tokens[i] == merge[0] and 
                tokens[i + 1] == merge[1]):
                # 找到匹配的对，合并
                new_tokens.append(''.join(merge))
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

# 使用示例
def demo_bpe():
    corpus = ["low lower newest widest", "low lower newest widest"] * 1000
    
    bpe = BPETokenizer()
    bpe.train(corpus, vocab_size=1000)
    
    # 测试分词
    text = "lowest"
    tokens = bpe.tokenize(text)
    print(f"'{text}' -> {tokens}")
    # 可能输出: ['low', 'est'] 或类似的子词组合

demo_bpe()
```

**BPE特点**:
- 数据驱动，无需语言学知识
- 能处理未见过的词
- GPT系列模型的标准选择

### 2. WordPiece

**核心改进**: 基于语言模型似然度选择合并

```python
class WordPieceTokenizer:
    def __init__(self):
        self.vocab = {}
        self.unk_token = "[UNK]"
    
    def train(self, corpus, vocab_size):
        # 1. 初始化基础词汇(字符 + 特殊token)
        base_vocab = set()
        for text in corpus:
            for char in text:
                base_vocab.add(char)
        
        base_vocab.update(["[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        vocab = list(base_vocab)
        
        # 2. 迭代添加最优子词
        while len(vocab) < vocab_size:
            best_subword = self.find_best_subword(corpus, vocab)
            if best_subword is None:
                break
            vocab.append(best_subword)
        
        self.vocab = {token: i for i, token in enumerate(vocab)}
    
    def find_best_subword(self, corpus, current_vocab):
        """找到能最大化语言模型似然度的子词"""
        candidates = self.generate_candidates(corpus, current_vocab)
        
        best_score = float('-inf')
        best_subword = None
        
        for candidate in candidates:
            # 计算添加这个子词后的语言模型得分
            score = self.calculate_lm_score(corpus, current_vocab + [candidate])
            
            if score > best_score:
                best_score = score
                best_subword = candidate
        
        return best_subword
    
    def generate_candidates(self, corpus, vocab):
        """生成候选子词"""
        candidates = set()
        
        # 基于现有词汇生成候选
        for text in corpus:
            tokens = self.basic_tokenize(text)
            for token in tokens:
                for i in range(len(token)):
                    for j in range(i + 1, len(token) + 1):
                        subword = token[i:j]
                        if len(subword) > 1 and subword not in vocab:
                            candidates.add(subword)
        
        return list(candidates)
    
    def tokenize(self, text):
        """WordPiece分词算法"""
        tokens = []
        
        for word in text.split():
            # 贪心最长匹配
            start = 0
            sub_tokens = []
            
            while start < len(word):
                end = len(word)
                cur_substr = None
                
                # 从最长子串开始尝试
                while start < end:
                    substr = word[start:end]
                    if start > 0:
                        substr = "##" + substr  # WordPiece前缀
                    
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                
                if cur_substr is None:
                    # 无法分词，标记为未知
                    sub_tokens.append(self.unk_token)
                    break
                
                sub_tokens.append(cur_substr)
                start = end
            
            tokens.extend(sub_tokens)
        
        return tokens

# BERT使用的WordPiece示例
def demo_wordpiece():
    text = "unaffable"
    # WordPiece可能分词为: ["un", "##aff", "##able"]
    
    wp = WordPieceTokenizer()
    # 假设已训练
    wp.vocab = {"un": 1, "##aff": 2, "##able": 3, "[UNK]": 0}
    
    tokens = wp.tokenize(text)
    print(f"'{text}' -> {tokens}")

demo_wordpiece()
```

**WordPiece优势**:
- 基于语言模型优化，理论更严格
- BERT等模型的标准选择
- 能更好地保持语义相关性

### 3. Unigram Language Model

**核心思想**: 从大词汇表开始，逐步移除不重要的token

```python
class UnigramTokenizer:
    def __init__(self):
        self.vocab = {}
        self.token_probs = {}
    
    def train(self, corpus, vocab_size):
        # 1. 初始化大词汇表（包含所有可能的子串）
        initial_vocab = self.build_initial_vocab(corpus)
        
        # 2. 初始化token概率
        self.token_probs = self.initialize_probabilities(initial_vocab, corpus)
        
        # 3. 迭代减少词汇表
        current_vocab = list(initial_vocab)
        
        while len(current_vocab) > vocab_size:
            # 计算移除每个token的损失
            losses = {}
            for token in current_vocab:
                if self.is_removable(token):  # 保护特殊token
                    losses[token] = self.calculate_removal_loss(token, corpus)
            
            # 移除损失最小的token
            if losses:
                token_to_remove = min(losses, key=losses.get)
                current_vocab.remove(token_to_remove)
                del self.token_probs[token_to_remove]
                
                # 重新计算概率
                self.update_probabilities(current_vocab, corpus)
        
        self.vocab = {token: i for i, token in enumerate(current_vocab)}
    
    def calculate_removal_loss(self, token, corpus):
        """计算移除token的似然度损失"""
        total_loss = 0
        
        for text in corpus:
            # 计算有token时的最优分词似然度
            with_token = self.get_best_segmentation(text, include_token=token)
            
            # 计算无token时的最优分词似然度  
            without_token = self.get_best_segmentation(text, exclude_token=token)
            
            # 损失 = 无token似然度 - 有token似然度
            loss = without_token['log_prob'] - with_token['log_prob']
            total_loss += loss
        
        return total_loss
    
    def get_best_segmentation(self, text, include_token=None, exclude_token=None):
        """使用动态规划找最优分词"""
        n = len(text)
        
        # dp[i] = (最优对数似然度, 分词方案)
        dp = [(-float('inf'), [])] * (n + 1)
        dp[0] = (0.0, [])
        
        for i in range(n + 1):
            if dp[i][0] == -float('inf'):
                continue
                
            for j in range(i + 1, n + 1):
                token = text[i:j]
                
                # 检查token是否可用
                if exclude_token and token == exclude_token:
                    continue
                if include_token and token not in self.token_probs and token != include_token:
                    continue
                if token not in self.token_probs:
                    continue
                
                # 计算新的似然度
                token_prob = self.token_probs.get(token, 1e-10)
                new_prob = dp[i][0] + math.log(token_prob)
                
                if new_prob > dp[j][0]:
                    dp[j] = (new_prob, dp[i][1] + [token])
        
        return {'log_prob': dp[n][0], 'tokens': dp[n][1]}
    
    def tokenize(self, text):
        """使用训练好的Unigram模型分词"""
        result = self.get_best_segmentation(text)
        return result['tokens']

# SentencePiece使用的Unigram示例
def demo_unigram():
    corpus = ["hello world", "hello universe", "hi world"]
    
    unigram = UnigramTokenizer()
    unigram.train(corpus, vocab_size=20)
    
    text = "hello world"
    tokens = unigram.tokenize(text)
    print(f"'{text}' -> {tokens}")

demo_unigram()
```

### 4. 字节级BPE (Byte-level BPE)

**创新点**: 在UTF-8字节级别进行BPE

```python
class ByteLevelBPE:
    def __init__(self):
        # UTF-8字节到可打印字符的映射
        self.byte_encoder = self.bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    
    def bytes_to_unicode(self):
        """创建字节到Unicode字符的映射"""
        bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
        cs = bs[:]
        n = 0
        for b in range(2**8):
            if b not in bs:
                bs.append(b)
                cs.append(2**8+n)
                n += 1
        cs = [chr(n) for n in cs]
        return dict(zip(bs, cs))
    
    def encode_text_to_bytes(self, text):
        """将文本编码为字节级表示"""
        byte_encoded = []
        for char in text:
            # 获取UTF-8字节
            utf8_bytes = char.encode('utf-8')
            for byte in utf8_bytes:
                byte_encoded.append(self.byte_encoder[byte])
        return ''.join(byte_encoded)
    
    def decode_bytes_to_text(self, byte_string):
        """将字节级表示解码回文本"""
        bytes_list = []
        for char in byte_string:
            bytes_list.append(self.byte_decoder[char])
        
        byte_array = bytes(bytes_list)
        return byte_array.decode('utf-8', errors='replace')
    
    def train(self, corpus, vocab_size):
        """在字节级别训练BPE"""
        # 1. 将所有文本转换为字节级表示
        byte_corpus = []
        for text in corpus:
            byte_text = self.encode_text_to_bytes(text)
            byte_corpus.append(byte_text)
        
        # 2. 在字节级别应用标准BPE
        self.bpe = BPETokenizer()
        self.bpe.train(byte_corpus, vocab_size)
    
    def tokenize(self, text):
        """字节级BPE分词"""
        # 转换为字节级表示
        byte_text = self.encode_text_to_bytes(text)
        
        # 应用BPE
        byte_tokens = self.bpe.tokenize(byte_text)
        
        return byte_tokens

# GPT-2使用的Byte-level BPE
def demo_byte_bpe():
    text = "Hello, 世界!"  # 包含中文
    
    bbpe = ByteLevelBPE()
    byte_text = bbpe.encode_text_to_bytes(text)
    print(f"字节级编码: {byte_text}")
    
    decoded = bbpe.decode_bytes_to_text(byte_text)
    print(f"解码结果: {decoded}")

demo_byte_bpe()
```

**字节级BPE优势**:
- 通用性强，支持所有语言
- 词汇表紧凑
- GPT-2/3/4的标准选择

## 💬 现代Tokenizer发展趋势

### 1. 大概念模型 (Large Concept Models)

**突破性思路**: 超越token级别，直接处理概念级别

```python
class ConceptLevelTokenizer:
    """概念级别的tokenizer (理论模型)"""
    
    def __init__(self, concept_encoder):
        self.concept_encoder = concept_encoder  # 预训练的概念编码器
    
    def encode_to_concepts(self, text):
        """将文本直接编码为概念向量"""
        # 使用句子级别的编码器
        sentences = self.split_to_sentences(text)
        
        concept_vectors = []
        for sentence in sentences:
            # 将句子编码为概念向量而非token序列
            concept_vec = self.concept_encoder.encode(sentence)
            concept_vectors.append(concept_vec)
        
        return concept_vectors
    
    def decode_from_concepts(self, concept_vectors):
        """从概念向量解码回文本"""
        sentences = []
        for concept_vec in concept_vectors:
            sentence = self.concept_encoder.decode(concept_vec)
            sentences.append(sentence)
        
        return ' '.join(sentences)
```

### 2. 动态上下文分词

**核心思想**: 根据上下文动态调整分词策略

```python
class ContextAwareTokenizer:
    """上下文感知的动态分词器"""
    
    def __init__(self, base_tokenizer, context_model):
        self.base_tokenizer = base_tokenizer
        self.context_model = context_model
    
    def tokenize_with_context(self, text, context=""):
        """基于上下文的动态分词"""
        
        # 1. 分析上下文确定分词策略
        context_features = self.analyze_context(context)
        
        # 2. 根据上下文选择分词粒度
        if context_features['domain'] == 'technical':
            # 技术文本：更细粒度分词
            granularity = 'fine'
        elif context_features['domain'] == 'casual':
            # 日常对话：较粗粒度分词
            granularity = 'coarse'
        else:
            granularity = 'medium'
        
        # 3. 应用动态分词
        tokens = self.adaptive_tokenize(text, granularity)
        
        return tokens
    
    def adaptive_tokenize(self, text, granularity):
        """自适应分词"""
        if granularity == 'fine':
            # 更多子词分割
            return self.base_tokenizer.tokenize(text, merge_threshold=0.3)
        elif granularity == 'coarse':
            # 更少子词分割
            return self.base_tokenizer.tokenize(text, merge_threshold=0.8)
        else:
            # 标准分词
            return self.base_tokenizer.tokenize(text)
```

### 3. 多模态Tokenizer

**扩展思路**: 统一处理文本、图像、音频等多种模态

```python
class MultimodalTokenizer:
    """多模态统一tokenizer"""
    
    def __init__(self):
        self.text_tokenizer = BPETokenizer()
        self.image_tokenizer = ImagePatchTokenizer()
        self.audio_tokenizer = AudioSegmentTokenizer()
        
        # 统一词汇表
        self.unified_vocab = self.build_unified_vocab()
    
    def tokenize_multimodal(self, inputs):
        """多模态输入的统一分词"""
        unified_tokens = []
        
        for modality, data in inputs.items():
            if modality == 'text':
                tokens = self.text_tokenizer.tokenize(data)
                # 添加模态标识
                unified_tokens.extend([f"<text>{token}" for token in tokens])
                
            elif modality == 'image':
                patches = self.image_tokenizer.tokenize(data)
                unified_tokens.extend([f"<image>{patch}" for patch in patches])
                
            elif modality == 'audio':
                segments = self.audio_tokenizer.tokenize(data)
                unified_tokens.extend([f"<audio>{segment}" for segment in segments])
        
        return unified_tokens
```

## 💬 面试问题解答

### Q1: BPE、WordPiece、Unigram的核心区别是什么？

**简洁对比**:

| 维度 | BPE | WordPiece | Unigram |
|------|-----|-----------|---------|
| **核心策略** | 频率驱动合并 | 似然度驱动合并 | 概率驱动剪枝 |
| **训练方向** | 从字符向上构建 | 从字符向上构建 | 从完整词汇向下剪枝 |
| **选择标准** | 字符对频率 | 语言模型似然度 | Token移除损失 |
| **代表模型** | GPT系列 | BERT系列 | T5、mT5 |

### Q2: 为什么现代大模型多采用Byte-level BPE？

**核心优势**:

1. **通用性**: 支持所有语言和字符，无需特殊预处理
2. **紧凑性**: 基础词汇表只需256个字节
3. **鲁棒性**: 能处理任何UTF-8编码的文本
4. **一致性**: 训练和推理时的处理完全一致

**技术细节**:
```python
# 传统BPE可能遇到的问题
text = "café"  # 包含重音字符
# 可能导致编码不一致或OOV问题

# Byte-level BPE的解决方案
bytes_representation = text.encode('utf-8')  # [99, 97, 102, 195, 169]
# 每个字节都有固定的映射，保证一致性
```

### Q3: Tokenizer的词汇表大小如何选择？

**权衡考虑**:

**词汇表过小**:
- 序列变长，计算成本增加
- 语义信息丢失
- 长距离依赖建模困难

**词汇表过大**:
- 嵌入层参数激增
- 稀有token训练不充分
- 推理时内存占用大

**经验法则**:
```python
# 常见配置
model_configs = {
    'GPT-2': {'vocab_size': 50257, 'algorithm': 'Byte-level BPE'},
    'BERT': {'vocab_size': 30522, 'algorithm': 'WordPiece'},
    'T5': {'vocab_size': 32128, 'algorithm': 'SentencePiece Unigram'},
    'LLaMA': {'vocab_size': 32000, 'algorithm': 'SentencePiece BPE'}
}

# 选择原则：
# 1. 英文：20k-50k较为合适
# 2. 多语言：50k-100k
# 3. 代码：特殊考虑，可能需要更大词汇表
```

### Q4: 如何评估Tokenizer的好坏？

**评估维度**:

1. **压缩效率**: `compression_ratio = original_chars / num_tokens`
2. **词汇覆盖率**: 测试集中UNK token的比例
3. **语义保持度**: 重要词汇是否被合理分割
4. **fertility**: 平均每个词被分成多少个token

```python
def evaluate_tokenizer(tokenizer, test_corpus):
    """tokenizer评估函数"""
    
    total_chars = sum(len(text) for text in test_corpus)
    total_tokens = sum(len(tokenizer.tokenize(text)) for text in test_corpus)
    
    # 压缩比
    compression_ratio = total_chars / total_tokens
    
    # UNK比例
    unk_count = sum(text.count('[UNK]') for text in 
                   [' '.join(tokenizer.tokenize(text)) for text in test_corpus])
    unk_ratio = unk_count / total_tokens
    
    # Fertility (词汇分割度)
    word_tokens = []
    for text in test_corpus:
        words = text.split()
        for word in words:
            tokens = tokenizer.tokenize(word)
            word_tokens.append(len(tokens))
    
    fertility = sum(word_tokens) / len(word_tokens)
    
    return {
        'compression_ratio': compression_ratio,
        'unk_ratio': unk_ratio,
        'fertility': fertility
    }
```

## 💻 完整实现示例

```python
# 现代Tokenizer的完整实现示例
from transformers import AutoTokenizer

class ModernTokenizerExample:
    """现代tokenizer使用示例"""
    
    def __init__(self, model_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def demonstrate_tokenization(self):
        """演示各种tokenization场景"""
        
        test_cases = [
            "Hello, world!",
            "artificial intelligence",
            "未来的人工智能",
            "café résumé naïve",
            "COVID-19 vaccination",
            "user@example.com",
            "print('Hello, World!')"
        ]
        
        print("=== Tokenization演示 ===")
        for text in test_cases:
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(token_ids)
            
            print(f"\n原文: {text}")
            print(f"Tokens: {tokens}")
            print(f"Token IDs: {token_ids}")
            print(f"解码: {decoded}")
            print(f"压缩比: {len(text)/len(tokens):.2f}")
    
    def analyze_special_tokens(self):
        """分析特殊token的处理"""
        
        special_cases = [
            "Mr. Smith went to the U.S.A.",  # 缩写
            "She said, \"Hello!\"",          # 引号
            "Visit https://example.com",      # URL
            "Temperature: 25.6°C",           # 符号和数字
            "   extra    spaces   ",         # 多余空格
        ]
        
        print("\n=== 特殊情况处理 ===")
        for text in special_cases:
            tokens = self.tokenizer.tokenize(text)
            print(f"'{text}' -> {tokens}")
    
    def compare_tokenizers(self):
        """对比不同tokenizer"""
        
        models = ['bert-base-uncased', 'gpt2', 'facebook/bart-base']
        test_text = "The quick brown fox jumps over the lazy dog."
        
        print("\n=== Tokenizer对比 ===")
        for model_name in models:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                tokens = tokenizer.tokenize(test_text)
                
                print(f"\n{model_name}:")
                print(f"Vocab size: {tokenizer.vocab_size}")
                print(f"Tokens: {tokens}")
                print(f"Token count: {len(tokens)}")
                
            except Exception as e:
                print(f"Error loading {model_name}: {e}")

# 运行演示
if __name__ == "__main__":
    demo = ModernTokenizerExample()
    demo.demonstrate_tokenization()
    demo.analyze_special_tokens()
    demo.compare_tokenizers()
```

## ✅ 学习检验

- [ ] 理解Tokenizer的基本工作流程
- [ ] 掌握BPE、WordPiece、Unigram的核心算法
- [ ] 了解Byte-level BPE的优势
- [ ] 能分析不同tokenizer的适用场景
- [ ] 理解tokenizer对模型性能的影响

## 🔗 相关链接

- [上一节：语言模型架构](language-models.md)
- [下一节：Attention升级技术](../attention-advanced/index.md)
- [返回：Transformer基础概览](index.md)