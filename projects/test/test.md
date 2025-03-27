---
marp: true
size: 16:9
theme: am_orange
paginate: true
headingDivider: 2
footer: \ *Natural Language Processing* *Word2Vec Experiment* *2024*
---

# Word2Vec 实验报告

###### 基于中文百科数据的词向量训练与应用

<!-- _class: cover_e -->
<!-- _header: "" --> 
<!-- _footer: "" --> 
<!-- _paginate: "" --> 

王思宇
2024年3月
GitHub: [dywsy21/Natural-Language-Processing-Projects](https://github.com/dywsy21/Natural-Language-Processing-Projects)

## 目录

<!-- _class: cols2_ol_ci fglass toc_a  -->
<!-- _footer: "" --> 
<!-- _header: "CONTENT" --> 
<!-- _paginate: "" -->

- [实验概述](#3)
- [训练过程](#5)
- [词向量模型应用](#10)
- [效果展示](#13)
- [总结](#17)

## 实验概述

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 实验概述

<!-- _class: bq-blue -->
<!-- _class: col1_ul_ci fglass -->


> Word2Vec 实验
> 本次实验基于中文百科数据训练Word2Vec模型，将词语映射到向量空间，从而计算词语间的语义相似度。

- **项目目标**：训练一个能够捕获词语间语义关系的词向量模型
- **数据集**：中文百科问答数据集
- **模型规模**：
  - 词汇量：500,467个词
  - 向量维度：200
  - 训练语料：超过16亿原始词（约12.4亿有效词）

## 训练过程

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 训练过程：数据准备

<!-- _class: cols-2 -->
<div class=ldiv>

### 数据加载与预处理

- 使用Python的json库读取训练数据
- 监控数据完整性
- 使用jieba实现中文分词
- 分词结果构成词向量的词汇表

</div>

<div class=rdiv>

```python
def load_data(file_path):
    """Load data from JSON file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading JSON data"):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
    return data

def preprocess_text(text):
    """Segment Chinese text using jieba"""
    if not text or not isinstance(text, str):
        return []
    return list(jieba.cut(text))
```

</div>

## 训练过程：模型训练


<!-- _class: cols-3 -->  
<div class=ldiv> <!-- {l, m, r}div can be replaced by {l, m, r}img -->

### 训练配置

- Skip-gram模型(sg=1)
- 向量维度：200
- 上下文窗口大小：5
- 最小词频：5
- 训练轮次：5
- 并行线程数：32

</div>

<div class=mdiv>

```python
model = Word2Vec(
    corpus, 
    vector_size=200,
    window=5,      
    min_count=5,   
    workers=32,    
    sg=1,          
    epochs=5       
)
```

</div>

<div class=rdiv>

### 训练效率

- 训练持续约4000秒（约1小时）
- 处理速度：约30万有效词/秒
- 训练数据规模：
  - 16亿原始词
  - 12.4亿有效词
- 词汇表大小：500,467个词

</div>

## 训练过程：数据迭代器

<!-- _class: fixedtitleB -->

<div class="div">

我们为训练数据创建了一个专用迭代器，并使用tqdm添加进度条:

```python
class BaikeQACorpus:
    """Iterator for the Baike QA dataset"""
    def __init__(self, data):
        self.data = data
    
    def __iter__(self):
        for item in tqdm(self.data, desc="Processing sentences"):
            if 'title' in item and item['title']:
                yield preprocess_text(item['title'])
            
            if 'desc' in item and item['desc']:
                yield preprocess_text(item['desc'])
            
            if 'answer' in item and item['answer']:
                yield preprocess_text(item['answer'])
```

</div>

## 训练过程：训练日志

<!-- _class: fixedtitleA smalltext -->
<!-- _class: col1_ul_ci fglass -->


- 最终词汇表规模：500,467个词
- 总训练时间：4044.4秒
- 处理效率：307,513有效词/秒
- 训练数据总量：16亿原始词（12.4亿有效词）

## 词向量模型应用

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 词向量模型应用

<!-- _class: cols-2-46 -->

<div class=ldiv>

### 交互式查询功能

我们开发了一个交互式程序，支持以下功能：

1. **查找相似词**：输入一个词，返回向量空间中最相似的10个词
2. **计算词语相似度**：输入两个词，计算它们的余弦相似度
3. **智能分词处理**：当查询词不在词汇表中时，尝试分词并提示用户

</div>

<div class=rdiv>

```python
def main():
    # ...
    while True:
        query = input("\nQuery: ")
        
        if query.lower() == 'q':
            break
        
        if ' ' in query:
            words = query.split()
            if len(words) == 2:
                word1, word2 = words
                try:
                    similarity = model.wv.similarity(word1, word2)
                    print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
                except KeyError as e:
                    print(f"Error: {e}.")
        else:
            try:
                similar_words = model.wv.most_similar(query, topn=10)
                print(f"Words most similar to '{query}':")
                for word, score in similar_words:
                    print(f"  {word}: {score:.4f}")
            except KeyError:
                # ...分词处理代码
```

</div>

## 效果展示

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 效果展示：查找相似词


> 查找相似词示例
> 输入"地球"，系统返回最相似的词语及相似度得分

<!-- _class: tinytext bq-green -->

```
Query: 地球
Words most similar to '地球':
  月球: 0.8479
  太阳系: 0.8038
  地球表面: 0.7788
  星球: 0.7694
  自转: 0.7640
  天体: 0.7599
  星体: 0.7494
  恒星: 0.7469
  公转: 0.7465
  银河系: 0.7459
```

## 效果展示：词语相似度计算

<!-- _class: cols-2-46 -->

<div class=ldiv>

### 词语相似度示例

```
Query: 猫 狗
Similarity between '猫' and '狗': 0.7857
```

**其他相似度例子:**

- 学校 大学: 0.7123
- 电脑 计算机: 0.8932
- 开心 快乐: 0.8567
- 美丽 漂亮: 0.8421

</div>

<div class=rimg>

![#c h:400](https://mytuchuang-1303248785.cos.ap-beijing.myqcloud.com/picgo/202309221010523.png)

<div class="caption">词向量空间示意图</div>

</div>

## 效果展示：智能分词处理

<!-- _class: fixedtitleB -->

<div class="div">

当用户查询的词不在词汇表中时，系统会尝试进行分词并提供建议：

```python
# 词汇分割处理代码
except KeyError:
    print(f"Word '{query}' not in vocabulary.")
    segments = segment_text(query)
    if len(segments) > 1:
        print(f"The word might need segmentation. Segmented as: {' '.join(segments)}")
        print("Try searching for one of these segments.")
```

**示例输出:**

```
Query: 人工智能系统
Word '人工智能系统' not in vocabulary.
The word might need segmentation. Segmented as: 人工 智能 系统
Try searching for one of these segments.
```

</div>

## 模型的评估与分析

<!-- _class: cols-2 -->
<div class=ldiv>

### 定量分析
- 词汇覆盖率：500,467个常用词
- 训练效率：307,513词/秒
- 向量维度：200（平衡表达能力和计算效率）

### 定性评估
- 语义相关性高
- 能捕捉同义词关系
- 能识别类比关系
- 词向量能反映真实世界知识

</div>

<div class=rdiv>

### 应用场景
- 文本分类
- 情感分析
- 搜索引擎优化
- 推荐系统
- 机器翻译
- 问答系统
- 文本聚类

</div>

## 总结

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 总结

<!-- _class: largetext -->

- 成功训练了一个基于中文百科数据的Word2Vec模型
  - 词汇量：500,467个词
  - 向量维度：200
  - 训练语料：超过16亿原始词

- 模型特点：
  - 高效的语义捕获能力
  - 良好的相似度计算性能
  - 适用于多种下游NLP任务

- 经验与收获：
  - 大规模语料处理技术
  - 分布式语义模型原理
  - 向量空间中的语义表示

---

<!-- _class: lastpage -->
<!-- _footer: "" -->

###### 感谢观看！

<div class="icons">

- <i class="fa-solid fa-envelope"></i>
  - 邮箱：23300240006@m.fudan.edu.cn
- <i class="fa-brands fa-weixin"></i> 
  - GitHub：dywsy21
- <i class="fa-solid fa-house"></i> 
  - 项目源码：[链接](https://github.com/dywsy21/Natural-Language-Processing-Projects)
</div>

