# 结巴分词

[jieba](https://github.com/fxsjy/jieba)

“结巴”中文分词：做最好的 Python 中文分词组件

## 特点

- 支持三种分词模式：

  - 精确模式，试图将句子最精确地切开，适合文本分析；
  - 全模式，把句子中所有的可以成词的词语都扫描出来, 速度非常快，但是不能解决歧义；
  - 搜索引擎模式，在精确模式的基础上，对长词再次切分，提高召回率，适合用于搜索引擎分词。
  - 支持繁体分词

- 支持自定义词典
- MIT 授权协议

## 算法

- 基于前缀词典实现高效的词图扫描，生成句子中汉字所有可能成词情况所构成的**有向无环图** (DAG)
- 采用了动态规划查找最大概率路径, 找出基于词频的最大切分组合
- 对于未登录词，采用了基于汉字成词能力的 **HMM** 模型，使用了 **Viterbi** 算法