BERT 大规模预训练语言模型
=========================

BERT是谷歌发布的基于双向Transformer的大规模预训练语言模型，该预训练模型能高效抽取文本信息并应用于各种NLP任务，并刷新了11
项NLP 任务的当前最优性能记录。
BERT的全称是基于Transformer的双向编码器表征，其中“双向”表示模型在处理某一个词时，它能同时利用前面的词和后面的词两部分信息。

\*\* 新的 2019 年 2 月 7 日：TfHub 模块 \*\*

BERT 已上传至\ `TensorFlow Hub <https://tfhub.dev>`__. 有关如何使用 TF
Hub
模块的示例，请参阅\ ``run_classifier_with_tfhub.py``\ ，或者在\ `Colab <https://colab.sandbox.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb>`__\ 上的浏览器中运行示例.

\*\* 新的 2018 年 11 月 23 日：未规范化的多语言模式+泰语+蒙古语 \*\*

我们上传了一个新的多语言模型，它不会对输入执行任何规范化（没有下限，重音剥离或
Unicode 规范化），还包括泰语和蒙古语。

**建议使用此版本开发多语言模型，尤其是使用非拉丁字母的语言。**

这不需要任何代码更改，可以在此处下载：

-  ```BERT-Base,多语言套装`` <https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip>`__:
   104 种语言，12 层，768 隐藏，12 头，110M 参数

\*\* 新的 2018 年 11 月 15 日：SOTA SQuAD 2.0 系统 \*\*

我们发布了代码更改，以重现我们的 83％F1 SQuAD 2.0
系统，目前排行榜上第一名是 3％。 有关详细信息，请参阅自述文件的 SQuAD
2.0 部分。

\*\* 新的 2018 年 11 月 5 日：第三方 PyTorch 和 Chainer 版本的 BERT 可用
\*\*

来自 HuggingFace 的 NLP 研究人员制作了\ `PyTorch 版本的
BERT <https://github.com/huggingface/pytorch-pretrained-BERT>`__\ ，它与我们预先训练好的检查点兼容，并能够重现我们的结果。
Sosuke Kobayashi 也提供了\ `Biner 版本的
BERT <https://github.com/soskek/bert-chainer>`__\ （谢谢！）我们没有参与
PyTorch 实现的创建或维护，所以请将任何问题都指向该存储库的作者。

\*\* 新的 2018 年 11 月 3 日：提供多种语言和中文模式 \*\*

我们提供了两种新的 BERT 型号:

-  ```BERT-基地，多语种`` <https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip>`__\ **\ （不推荐，使用\ ``Multilingual Cased``\ 代替）**\ ：102
   种语言，12 层，768 隐藏，12 头，110M 参数
-  ```BERT-Base，中文`` <https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip>`__:
   中文简体和繁体，12 层，768 隐藏，12 头，110M 参数

我们对中文使用基于字符的标记化，对所有其他语言使用 WordPiece 标记化。
两种模型都应该开箱即用，不需要任何代码更改。
我们确实在\ ``tokenization.py``\ 中更新了\ ``BasicTokenizer``\ 的实现以支持中文字符
但是，我们没有更改标记化 API。

有关更多信息，请参阅\ `多语言自述文件 <https://github.com/google-research/bert/blob/master/multilingual.md>`__.

\*\* 结束新信息 \*\*

介绍
----

**BERT**\ ，或\ **B**\ idirectional **E**\ ncoder **R**\ epresentations
from **T**\ ransformers
的表示，是一种预训练语言表示的新方法，它获得了最先进的结果可用于各种自然语言处理（NLP）任务。

我们的学术论文详细描述了 BERT，并提供了许多任务的完整结果:
https://arxiv.org/abs/1810.04805.

为了给出几个数字，这里是\ `SQuAD
v1.1 <https://rajpurkar.github.io/SQuAD-explorer/>`__
问题回答任务的结果:

+-----------------------------------+----------+----------+
| SQuAD v1.1排行榜（2018年10月8日） | Test EM  | Test F1  |
+===================================+==========+==========+
| 1st Place Ensemble - BERT         | **87.4** | **93.2** |
+-----------------------------------+----------+----------+
| 2nd Place Ensemble - nlnet        | 86.0     | 91.7     |
+-----------------------------------+----------+----------+
| 1st Place Single Model - BERT     | **85.1** | **91.8** |
+-----------------------------------+----------+----------+
| 2nd Place Single Model - nlnet    | 83.5     | 90.1     |
+-----------------------------------+----------+----------+

以及几种自然语言推理任务:

+-------------------------+----------+--------------+----------+
| System                  | MultiNLI | Question NLI | SWAG     |
+=========================+==========+==============+==========+
| BERT                    | **86.7** | **91.1**     | **86.3** |
+-------------------------+----------+--------------+----------+
| OpenAI GPT (Prev. SOTA) | 82.2     | 88.1         | 75.0     |
+-------------------------+----------+--------------+----------+

还有许多其他任务。

而且，这些结果都是在几乎没有任务特定的神经网络架构设计的情况下获得的。

如果您已经知道 BERT
是什么并且您只是想要开始，您可以在几分钟内\ `下载预先训练的模型 <#pre-trained-models>`__\ 和\ `运行最先进的微调 <#fine-tuning-with-bert>`__\ 。

该存储库中发布了什么？
----------------------

我们发布以下内容:

-  BERT模型架构的TensorFlow代码（主要是标准的\ `Transformer <https://arxiv.org/abs/1706.03762>`__\ 架构）.
-  来自纸张的\ ``BERT-Base``\ 和\ ``BERT-Large``\ 的小写和套装版本的预先训练的检查点。
-  TensorFlow代码用于按钮复制本文最重要的微调实验，包括SQuAD，MultiNLI和MRPC。

此存储库中的所有代码都与CPU，GPU和云TPU一起开箱即用。

放弃
----

这不是 Google 的官方产品。

联系信息
--------

有关使用 BERT 的帮助或问题，请提交 GitHub 问题。

有关 BERT 的个人通信，请联系 Jacob
Devlin（\ ``jacobdevlin@google.com``\ ），Ming-Wei
Chang（\ ``mingweichang@google.com``\ ）或 Kenton
Lee（\ ``kentonl@google.com``\ ）。
