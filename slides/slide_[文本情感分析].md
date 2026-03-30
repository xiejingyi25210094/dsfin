---
marp: true
theme: gaia
paginate: true
math: katex
style: |
  /* 全局字体缩小 */
  section {
    font-size: 24px;   
  }
  /* 表格字体可稍作调整，保持相对大小 */
  table {
    font-size: 0.75em;
    width: 100%;
  }
  th {
    background-color: #e8e8e8;
  }
  td {
    border: 1px solid #ddd;
  }
---
<!-- _class: cover -->
<style scoped>
section.cover {
  text-align: center;
}
section.cover h1 {
  margin-bottom: 0.5em;  /* 标题下方间距 */
}
section.cover p, section.cover h2 {
  margin-top: 0.2em;
  margin-bottom: 0.2em;
}
</style>

<br>

# 文本情感分析在金融中的应用
<br>

## 金融数据分析与建模 · 小组作业
<br>
<br>
<br>
<br>

**小组成员**：谢婧怡、梁柏麟、黄彦琦、郭虹辰、黄丽蓉
**日期**：2026 年 3 月
**中山大学岭南学院**

---
<style scoped>
section ul {
  font-size: 1.2em;  /* 比默认稍大 */
}
</style>

## 目录

#### 1. **动机与问题**
#### 2. **方法原理**
#### 3. **Python实现**
#### 4. **金融应用场景**
#### 5. **方法对比**
#### 6. **局限性与注意事项**
#### 7. **学习反思**
#### 8. **参考资料**

---

## 动机：这个方法解决什么问题？
<br>

#### 动机：文本情感分析将非结构化的文本信息（新闻、财报、社交媒体）转化为结构化的情感信号，解决了传统量化模型只看数字、忽略市场情绪的问题。
<br>

#### 不用文本情感分析会出现的错误
- **信息丢失**：只看结构化数据（价格、财务指标），忽略占市场信息 **80%** 以上的文本内容
- **反应滞后**：无法实时捕捉市场情绪变化
- **粒度粗糙**：只能看到"股价跌了5%"，但看不到**"在股价跌5%的前一天，社交媒体负面情绪已飙升80%"**

---

## 核心原理

### 方法步骤（以词典法为例）

**Step 1: 文本预处理**
- 去除标点符号和特殊字符
- 中文分词（jieba）/ 英文分词（nltk）
- 过滤停用词和单字词

**Step 2: 情感词典匹配**
- 统计正面词（增长、盈利）和负面词（下跌、亏损）
- 考虑否定词对情感极性的反转作用
- 可选择加入程度副词对情感强度进行加权

**Step 3: 情感得分计算**
   - 净得分 = 正面词数 - 负面词数
   - 归一化得分 = 净得分 / 总词数
---
<style scoped>
.katex {
  font-size: 0.85em;   /* 缩小公式字体，可调整数值 */
}
</style>
### 关键公式

**F1 分数（分类评估）**
$$ F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall} $$

其中：
- Precision（精确率）= 预测为该类且正确的数量 / 预测为该类的总数量
- Recall（召回率）= 预测为该类且正确的数量 / 实际为该类的总数量

**词典法归一化情感得分**
$$ \text{NormalizedScore} = \frac{\text{PosCount} - \text{NegCount}}{\text{TotalWords}} $$

### 核心假设

| 假设 | 说明 | 违反假设的后果 |
|------|------|----------------|
| **词袋假设** | 词语的顺序不影响情感，只关注词频 | 无法处理转折句、否定词的范围等复杂结构 |
| **领域一致性** | 情感词在不同语境下极性相同 | 通用词典在金融领域失效（如"风险"在风控中是中性词） |
| **情感独立** | 每个文本的情感判断独立于其他文本 | 忽略事件的时序依赖和市场预期的累积效应 |

---

## Python实现：词典法核心代码

```python
import jieba
import re

# 金融情感词典（简化示例）
positive_words = set(['增长', '盈利', '强劲', '超预期'])
negative_words = set(['下跌', '亏损', '风险', '疲软'])
negation_words = set(['不', '并非', '并未', '没有'])

def preprocess_text(text):
    text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
    words = jieba.lcut(text)
    return [w for w in words if len(w) > 1]

def sentiment_score(words):
    pos = sum(1 for w in words if w in positive_words)
    neg = sum(1 for w in words if w in negative_words)
    # 否定词处理：检查前一个词是否是否定词...
    return (pos - neg) / len(words)
关键参数：使用 negation_words 处理"并未改善"这类否定结构
  ```

---
<style scoped>
section {
  font-size: 26px;  /* 调回你满意的较大值，例如 26px */
}
</style>
## 金融应用场景
**场景1：量化交易策略 - 基于新闻情感的择时与选股**
问题：市场对新闻的反应迅速且过度，传统模型无法实时捕捉。
方法：使用 FinBERT 对实时新闻流进行情感分类或连续得分预测；当情感得分超过阈值时买入，低于阈值时卖出
价值：回测显示基于新闻情感的策略可获得显著超额收益，尤其在高频交易和事件驱动策略中。

**场景2：风险管理与危机预警 - 负面舆情监测**
问题：突发负面新闻（高管丑闻、监管调查）引发股价暴跌，传统风险模型滞后。
方法：构建情感监测系统，使用词典法或微调模型实时扫描社交媒体、新闻、监管文件；设置预警阈值，负面情感强度超过水平或急剧下降时发出警报；触发仓位调整或暂停交易
价值：帮助机构规避"黑天鹅"事件，减少回撤。例如瑞幸咖啡造假事件爆发前，社交媒体情感已异常负面。

---

## 与相关方法的对比
### 文本情感分析 vs 主题建模

| 维度 | 文本情感分析 | 主题建模 (LDA/BERTopic) |
|------|-------------|------------------------|
| **核心目标** | 判断"情绪如何"（正面/负面/中性） | 发现"在讨论什么"（话题聚类） |
| **输出** | 情感标签或连续得分 | 话题分布、话题关键词 |
| **典型技术** | 词典法、FinBERT 微调、LLM 提示 | LDA、NMF、BERTopic |
| **金融应用** | 交易信号、风险预警 | 财报电话会议话题分析、热点追踪 |

**互补关系**：两者常结合使用：先分话题，再在每个话题内做情感分析

### 篇章级情感分析 vs 方面级情感分析

| 维度 | 篇章级情感分析 | 方面级情感分析 (ABSA) |
|------|---------------|----------------------|
| **粒度** | 整篇文章一个情感得分 | 针对特定方面（如"营收"、"现金流"）分别判断 |
| **信息损失** | 高："先夸后骂"可能被判中性 | 低：保留细粒度信息 |
| **技术复杂度** | 低：分类/回归即可 | 高：需要方面抽取 + 情感分类 |
| **金融实用性** | 基础：适合快速原型 | 最佳：可构建多维度因子 |

---
<style scoped>
section {
  font-size: 26px;  /* 调回你满意的较大值，例如 26px */
}
</style>
## 局限性与注意事项

| 方法本身的假设局限 | 说明 |
|------|------|
| **领域依赖性强** | 通用词典在金融领域表现糟糕，需使用 Loughran-McDonald 等专业词典 |
| **复杂语言现象处理有限** | 反讽、双重否定、条件假设仍具挑战（但金融严肃文本中反讽罕见） |
| **缺乏外部知识** | 模型仅基于文本，无法利用行业背景、公司历史等常识 |
| **长文本限制** | BERT 等模型有 512 token 输入限制，处理 10-K 年报需截断或分层 |

<br> 

| 实践中常见问题 | 避坑建议 |
|----|----------|
| **用准确率评估不平衡数据** | 金融文本 80% 以上是中性，应监控宏平均 F1、MCC |
| **随机划分造成时间泄露** | 必须按时间顺序划分：训练集早于验证集，验证集早于测试集 |
| **过度调优验证集** | 避免在验证集上反复调参，最终用独立测试集评估 |
| **将相关性误认为因果性** | 情感与价格相关，但方向难定，需用格兰杰因果检验 |
| **忽略模型不确定性** | 输出概率或置信区间，仅在高置信度时交易 |
| **使用通用词典未验证** | 使用 Loughran-McDonald 金融词典，或在金融语料上微调 |
---

## 学习反思：AI哪里不准确？走了哪些弯路？
1. **AI回答不准确之处**
第3轮代码错误：AI最初提供的否定词处理代码存在逻辑错误（赋值语句neg_count += 1位置错误），实际运行时报错。通过调试发现后，第4轮修正。
第8轮过度深入：讨论checkpoint选择时，AI引入过多进阶指标（IC、夏普比率），对初学者过于复杂，及时切换话题。
<br> 

2. **学习过程中的弯路**
初期想一步上大模型：最初想直接用FinBERT微调，但对预训练-微调范式理解不足。纠正：回头先搞懂词典法，再逐步深入。
忽略了分类vs回归的选择：一开始未明确任务类型，后来通过第9-10轮对话才理清两者区别。
<br> 

3. **尚未搞清楚的问题**
方面级情感分析的具体实现：如何用BERT+CRF同时抽取方面词并判断情感？
金融领域长文本处理：10-K年报远超512 token，层次化模型如何实现？

---
<style scoped>
section p, section ul, section table {
  font-size: 0.65em;
}
</style>

## 参考资料
#### 学术论文

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

#### GitHub 仓库

| 仓库 | 说明 | 链接 |
|------|------|------|
| **FinBERT** | 金融情感分析预训练模型 | https://github.com/ProsusAI/finbert |
| **Loughran-McDonald Dictionary** | 金融情感词典官方资源 | https://sraf.nd.edu/loughranmcdonald-master-dictionary/ |
| **Transformers** | Hugging Face 预训练模型库 | https://github.com/huggingface/transformers |

