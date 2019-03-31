什么是 BERT？
=============

BERT
是一种预训练语言表示的方法，这意味着我们在大型文本语料库(如维基百科)上训练通用的\ ``语言理解``\ 模型，然后将该模型用于我们关心的下游
NLP 任务(如问题)接听)。 BERT 优于以前的方法，因为它是第一个用于预训练
NLP 的\ *unsupervised*\ ，\ *deeply bidirectional*\ 系统。

*Unsupervised*\ 意味着 BERT
仅使用纯文本语料库进行训练，这很重要，因为大量的纯文本数据可以在网络上以多种语言公开获得。

预先训练的表示也可以是\ *context-free*\ 或\ *contextual*\ ，上下文表示可以进一步是\ *unidirectional*\ 或\ *bidirectional*\ 。

无上下文模型，如\ `word2vec <https://www.tensorflow.org/tutorials/representation/word2vec>`__\ 或\ `GloVe <https://nlp.stanford.edu/projects/glove/>`__\ 生成单个\ ``单词嵌入``\ 词汇表中每个单词的表示，所以\ ``bank``\ 在\ ``bank deposit``\ 和\ ``river bank``\ 中具有相同的表示。
相反，上下文模型生成基于句子中其他单词的每个单词的表示。

BERT 建立在最近培训上下文表示的基础上 -
包括\ `半监督序列学习 <https://arxiv.org/abs/1511.01432>`__\ ，\ `Generative
Pre-Training <https：//blog.openai。%20com%20/%20language-unsupervised%20/>`__\ ，\ `ELMo <https://allennlp.org/elmo>`__\ 和\ `ULMFit <http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit。%20html>`__
- 但至关重要的是这些模型都是\ *unidirectional*\ 或\ *shallowly
bidirectional*\ 。 这意味着每个单词仅使用左侧(或右侧)的单词进行语境化。
例如，在\ ``我做了银行存款``\ 这句话中，\ ``银行``\ 的单向表示只是基于\ ``我做了``\ 而不是\ ``存款``\ 。
之前的一些工作确实结合了来自单独的左上下文和右上下文模型的表示，但仅以\ ``浅层``\ 方式。
BERT 代表\ ``银行``\ 使用它的左右上下文 - ``我做了...存款`` -
从深度神经网络的最底部开始，所以它是\ *deeply bidirectional*\ 。

BERT 使用一种简单的方法：我们屏蔽输入中
15％的单词，通过深度双向\ `Transformer <https://arxiv.org/abs/1706.03762>`__\ 编码器运行整个序列，然后仅预测蒙面的话。

F 或者示例：

::

   Input: the man went to the [MASK1] .
   he bought a [MASK2] of milk.
   Labels: [MASK1] = store; [MASK2] = gallon

为了学习句子之间的关系，我们还训练一个简单的任务，可以从任何单语语料库中生成：给出两个句子\ ``A``\ 和\ ``B``\ ，\ ``B``\ 是\ ``A``\ 之后的实际下一个句子，或者只是语料库中的随机句子？

::

   Sentence A: the man went to the store .
   Sentence B: he bought a gallon of milk .
   Label: IsNextSentence

::

   Sentence A: the man went to the store .
   Sentence B: penguins are flightless .
   Label: NotNextSentence

然后，我们在大型语料库(维基百科+\ `BookCorpus <http://yknzhu.wixsite.com/mbweb>`__)上训练大型模型(12
层到 24 层变换器)很长一段时间(1M 更新步骤)，那就是 BERT。

使用 BERT 有两个阶段：\ *Pre-training*\ 和\ *fine-tuning*\ 。

**预训练**\ 相当昂贵(4 到 16 个云 TPU 为 4
天)，但是每种语言都是一次性程序(目前的型号仅限英语，但多语言型号将在不久的将来发布)
。 我们正在发布一些预先培训的模型，这些模型是在谷歌预先培训过的。 大多数
NLP 研究人员永远不需要从头开始训练他们自己的模型。

**微调**\ 价格便宜。 本文中的所有结果可以在单个云 TPU 上最多 1
小时复制，或者在 GPU 上几小时复制，从完全相同的预训练模型开始。
例如，SQUAD 可以在单个 Cloud TPU 上训练大约 30 分钟，以获得 91.0％的 Dev
F1 得分，这是单系统最先进的。

BERT 的另一个重要方面是它可以非常容易地适应许多类型的 NLP 任务。
在本文中，我们展示了句子级别(例如，SST-2)，句子对级别(例如，MultiNLI)，词级别的最新结果。
