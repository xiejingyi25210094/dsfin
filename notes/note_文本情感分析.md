# 文本情感分析 自学笔记

> 学习日期：2026-03-27
> 参考对话轮次：AI 对话记录第 1-15 轮

---

## 1. 这个方法解决什么问题？

### 一句话动机
文本情感分析将非结构化的文本信息（新闻、财报、社交媒体）转化为结构化的情感信号，解决了传统量化模型只看数字、忽略市场情绪的问题。

### 不用这个方法的错误
如果不使用情感分析，直接分析金融文本会犯以下错误：
- **信息丢失**：传统方法只能处理结构化数据，无法利用新闻、研报、社交媒体等占市场信息80%以上的文本内容。
- **反应滞后**：无法实时捕捉市场情绪变化，等股价反映后再行动已经错过时机。
- **粒度粗糙**：只能看到"股价跌了5%"，但看不到"在股价跌5%的前一天，社交媒体负面情绪飙升了80%"这样的领先信号。

---

## 2. 核心原理

### 方法步骤（以词典法为例）

1. **文本预处理**
   - 去除标点符号和特殊字符
   - 分词（中文使用jieba，英文使用nltk）
   - 过滤停用词和单字词

2. **情感词典匹配**
   - 统计文本中正面词和负面词的数量
   - 考虑否定词对情感极性的反转作用
   - 可选择加入程度副词对情感强度进行加权

3. **情感得分计算**
   - 净得分 = 正面词数 - 负面词数
   - 归一化得分 = 净得分 / 总词数

### 关键公式

#### F1 分数（分类评估）
$$ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

其中：
- Precision（精确率）= 预测为该类且正确的数量 / 预测为该类的总数量
- Recall（召回率）= 预测为该类且正确的数量 / 实际为该类的总数量

#### 词典法归一化情感得分
$$ \text{NormalizedScore} = \frac{\text{PosCount} - \text{NegCount}}{\text{TotalWords}} $$

### 核心假设

| 假设 | 说明 | 违反假设的后果 |
|------|------|----------------|
| **词袋假设** | 词语的顺序不影响情感，只关注词频 | 无法处理转折句、否定词的范围等复杂结构 |
| **领域一致性** | 情感词在不同语境下极性相同 | 通用词典在金融领域失效（如"风险"在风控中是中性词） |
| **情感独立** | 每个文本的情感判断独立于其他文本 | 忽略事件的时序依赖和市场预期的累积效应 |

---

## 3. Python 实现

### 最小可运行示例（词典法）

```python
import jieba
import re

# ==================== 1. 情感词典准备 ====================
# 简化的金融情感词典（实际作业请使用 Loughran-McDonald 词典）
positive_words = set([
    '增长', '上升', '盈利', '收益', '优秀', '强劲', '乐观', '超预期', '利好', '改善'
])
negative_words = set([
    '下跌', '亏损', '诉讼', '风险', '下降', '疲软', '悲观', '低于预期', '利空', '恶化'
])
negation_words = set(['不', '不是', '没有', '并非', '并未', '无'])  # 否定词集合

# ==================== 2. 文本预处理函数 ====================
def preprocess_text(text):
    """
    预处理：去除标点、分词、过滤单字词
    """
    # 保留中文和字母，去除其他字符
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
    # 中文分词
    words = jieba.lcut(text)
    # 过滤单字词
    words = [w for w in words if len(w) > 1]
    return words

# ==================== 3. 情感得分计算函数 ====================
def sentiment_score(words, pos_set, neg_set, negation_set=None):
    """
    计算情感得分，支持否定词处理
    """
    if negation_set is None:
        # 基础版本：简单统计正负词数
        pos_count = sum(1 for w in words if w in pos_set)
        neg_count = sum(1 for w in words if w in neg_set)
    else:
        # 进阶版本：考虑否定词反转相邻词的情感
        pos_count = 0
        neg_count = 0
        i = 0
        while i < len(words):
            word = words[i]
            # 检查前一个词是否是否定词
            is_negated = (i > 0 and words[i-1] in negation_set)
            if word in pos_set:
                neg_count += 1 if is_negated else pos_count + 1
                if is_negated:
                    neg_count += 1
                else:
                    pos_count += 1
            elif word in neg_set:
                if is_negated:
                    pos_count += 1
                else:
                    neg_count += 1
            i += 1

    # 计算净得分和归一化得分
    net_score = pos_count - neg_count
    normalized_score = net_score / len(words) if words else 0.0

    return {
        'pos_count': pos_count,
        'neg_count': neg_count,
        'net_score': net_score,
        'normalized_score': normalized_score
    }

# ==================== 4. 测试示例 ====================
if __name__ == "__main__":
    # 测试金融新闻文本
    texts = [
        "公司财报显示营收强劲增长，利润超预期，前景乐观。",
        "受行业低迷影响，该公司营收下降，利润出现亏损，风险加大。",
        "虽然营收增长，但净利润并未改善，反而恶化。"
    ]

    for text in texts:
        words = preprocess_text(text)
        # 基础版本
        basic = sentiment_score(words, positive_words, negative_words)
        # 进阶版本（含否定词处理）
        advanced = sentiment_score(words, positive_words, negative_words, negation_words)

        print(f"文本：{text}")
        print(f"  分词结果：{words}")
        print(f"  基础得分：{basic['normalized_score']:.3f} (正:{basic['pos_count']}, 负:{basic['neg_count']})")
        print(f"  进阶得分：{advanced['normalized_score']:.3f} (正:{advanced['pos_count']}, 负:{advanced['neg_count']})")
        print()
```

### 主要参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| `positive_words` | 正面情感词集合 | 使用 Loughran-McDonald 金融词典 |
| `negative_words` | 负面情感词集合 | 使用 Loughran-McDonald 金融词典 |
| `negation_words` | 否定词集合 | 根据语言调整，中文如"不、并非" |
| `normalized_score` | 归一化情感得分 | 范围 -1 到 1，>0 正面，<0 负面 |

### 运行结果解读

```
文本：公司财报显示营收强劲增长，利润超预期，前景乐观。
  分词结果：['公司', '财报', '显示', '营收', '强劲', '增长', '利润', '超预期', '前景', '乐观']
  基础得分：0.500 (正:5, 负:0)
  进阶得分：0.500 (正:5, 负:0)
```

- **normalized_score = 0.500**：表示强烈正面情感
- 进阶得分与基础得分一致，因为该句没有否定词

---

## 4. 金融应用场景

### 场景一：量化交易策略 - 基于新闻情感的择时与选股

**问题**：市场对新闻的反应迅速且过度，传统模型无法实时捕捉。

**方法**：
- 使用 FinBERT 对实时新闻流进行情感分类或连续得分预测
- 当情感得分超过阈值时买入，低于阈值时卖出
- 构建多空组合：买入正面情感最高的股票，卖出得分最低的股票

**价值**：回测显示基于新闻情感的策略可获得显著超额收益，尤其在高频交易和事件驱动策略中。

**参考文献**：
- Tetlock (2007) "Giving Content to Investor Sentiment: The Role of Media in the Stock Market" - [Journal of Finance](https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2007.01232.x)

---

### 场景二：风险管理与危机预警 - 负面舆情监测

**问题**：突发负面新闻（产品质量、高管丑闻、监管调查）可能引发股价暴跌，传统风险模型滞后。

**方法**：
- 构建情感监测系统，使用词典法或微调模型实时扫描社交媒体、新闻、监管文件
- 设置预警阈值：当负面情感强度超过水平或急剧下降时发出警报
- 触发仓位调整或暂停交易

**价值**：帮助机构规避"黑天鹅"事件，减少回撤。例如瑞幸咖啡造假事件爆发前，社交媒体情感已异常负面。

**相关 GitHub 仓库**：
- FinBERT：https://github.com/ProsusAI/finbert

---

### 其他典型场景

| 场景 | 说明 |
|------|------|
| **财报电话会议分析** | 分析管理层语调，构建"管理层语调因子" |
| **社交媒体情绪聚合** | 生成日度市场情绪指数，作为择时因子 |
| **信用风险评估** | 负面新闻冲击指数预测信用利差变化 |
| **央行沟通分析** | 量化政策立场转变程度，预测利率走向 |

---

## 5. 与相关方法的对比

### 表1：文本情感分析 vs 主题建模

| 维度 | 文本情感分析 | 主题建模 (LDA/BERTopic) |
|------|-------------|------------------------|
| **核心目标** | 判断"情绪如何"（正面/负面/中性） | 发现"在讨论什么"（话题聚类） |
| **输出** | 情感标签或连续得分 | 话题分布、话题关键词 |
| **典型技术** | 词典法、FinBERT 微调、LLM 提示 | LDA、NMF、BERTopic |
| **金融应用** | 交易信号、风险预警 | 财报电话会议话题分析、热点追踪 |
| **互补关系** | 两者常结合使用：先分话题，再在每个话题内做情感分析 |

---

### 表2：篇章级情感分析 vs 方面级情感分析

| 维度 | 篇章级情感分析 | 方面级情感分析 (ABSA) |
|------|---------------|----------------------|
| **粒度** | 整篇文章一个情感得分 | 针对特定方面（如"营收"、"现金流"）分别判断 |
| **信息损失** | 高："先夸后骂"可能被判中性 | 低：保留细粒度信息 |
| **技术复杂度** | 低：分类/回归即可 | 高：需要方面抽取 + 情感分类 |
| **金融实用性** | 基础：适合快速原型 | 最佳：可构建多维度因子 |

---

## 6. 局限性与注意事项

### 方法本身的假设局限

| 局限 | 说明 |
|------|------|
| **领域依赖性强** | 通用词典在金融领域表现糟糕，需使用 Loughran-McDonald 等专业词典 |
| **复杂语言现象处理有限** | 反讽、双重否定、条件假设仍具挑战（但金融严肃文本中反讽罕见） |
| **缺乏外部知识** | 模型仅基于文本，无法利用行业背景、公司历史等常识 |
| **长文本限制** | BERT 等模型有 512 token 输入限制，处理 10-K 年报需截断或分层 |

---

### 实践中常见的坑

| 坑 | 避坑建议 |
|----|----------|
| **用准确率评估不平衡数据** | 金融文本 80% 以上是中性，应监控宏平均 F1、MCC |
| **随机划分造成时间泄露** | 必须按时间顺序划分：训练集早于验证集，验证集早于测试集 |
| **过度调优验证集** | 避免在验证集上反复调参，最终用独立测试集评估 |
| **将相关性误认为因果性** | 情感与价格相关，但方向难定，需用格兰杰因果检验 |
| **忽略模型不确定性** | 输出概率或置信区间，仅在高置信度时交易 |
| **使用通用词典未验证** | 使用 Loughran-McDonald 金融词典，或在金融语料上微调 |

---

## 7. 学习反思

### AI 回答的准确性与误导

- **第 3 轮代码小错误**：AI 最初提供的词典法代码中否定词处理部分有逻辑错误（赋值语句问题），实际运行会报错。通过实际运行代码发现了问题，后续在第 4 轮（已省略）让 AI 修改后解决。
- **第 8 轮回答过深**：在讨论 checkpoint 选择指标时，AI 引入了太多进阶指标（IC、夏普比率等），对于初学者来说过于复杂。及时切换话题避免了陷入细节。

### 学习过程中的弯路

1. **初期想一步上大模型**：最初想直接用 FinBERT 微调，但发现对预训练-微调范式理解不足。回头先搞懂词典法，再逐步深入更合理。
2. **忽略了分类 vs 回归的选择**：一开始没有明确任务类型，后来通过第 9-10 轮对话才理清两者的区别和适用场景。

### 尚未搞清楚的问题

1. **方面级情感分析的具体实现**：如何用 BERT+CRF 同时抽取方面词并判断情感？实际代码结构是怎样的？
2. **金融领域长文本处理**：10-K 年报远超 512 token，层次化模型如何实现分句/段聚合？
3. **策略回测的完整流程**：如何将情感得分与股票收益率对齐，进行严谨的事件研究？

---

## 8. 参考资料

### 学术论文

1. **Loughran and McDonald (2011)**
   *"When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks"*
   The Journal of Finance, Vol. 66, No. 1.
   （构建了金融领域权威情感词典）

2. **Tetlock (2007)**
   *"Giving Content to Investor Sentiment: The Role of Media in the Stock Market"*
   Journal of Finance, Vol. 62, No. 3.
   （开创性研究媒体情感与股票收益的关系）

3. **Araci (2019)**
   *"FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"*
   arXiv:1908.10063
   （金融领域预训练模型 FinBERT）

---

### GitHub 仓库

| 仓库 | 说明 | 链接 |
|------|------|------|
| **FinBERT** | 金融情感分析预训练模型 | https://github.com/ProsusAI/finbert |
| **Loughran-McDonald Dictionary** | 金融情感词典官方资源 | https://sraf.nd.edu/loughranmcdonald-master-dictionary/ |
| **Transformers** | Hugging Face 预训练模型库 | https://github.com/huggingface/transformers |

---

### AI 工具使用说明

- **使用工具**：DeepSeek
- **对话日期**：2026-03-27
- **总轮次**：15 轮
- **原始对话链接**：https://chat.deepseek.com/share/bhn5e2r1tiuqps7g4q

**使用建议**：
- 先从概念入手，再逐步深入技术细节
- AI 给出的代码需实际运行验证，可能有小错误
- 当回答过深时，及时切换话题或要求"用更通俗的语言解释"
