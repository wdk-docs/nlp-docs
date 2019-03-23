# 斯坦福-StanfordNLP

适用于多种人类语言的 Python NLP 库 <https://github.com/stanfordnlp/stanfordnlp>

[![Travis Status](https://travis-ci.com/stanfordnlp/stanfordnlp.svg?token=RPNzRzNDQRoq2x3J2juj&branch=master)](https://travis-ci.com/stanfordnlp/stanfordnlp)
[![PyPI version](https://img.shields.io/pypi/v/stanfordnlp.svg?colorB=blue)](https://pypi.org/project/stanfordnlp/)

斯坦福 NLP 集团的官方 Python NLP 库。 它包含用于运行 CoNLL 2018 共享任务的最新完全神经管道以及访问 Java Stanford CoreNLP 服务器的软件包。 有关详细信息，请访问我们的[官方网站](https://stanfordnlp.github.io/stanfordnlp/).

## 参考

如果您使用我们的神经管道，包括标记器，多字令牌扩展模型，解释器，POS /形态特征标记器或您研究中的依赖解析器，请引用我们的 CoNLL 2018 共享任务[系统描述文件](https://nlp.stanford.edu/pubs/qi2018universal.pdf):

```bibtex
@inproceedings{qi2018universal,
 address = {Brussels, Belgium},
 author = {Qi, Peng  and  Dozat, Timothy  and  Zhang, Yuhao  and  Manning, Christopher D.},
 booktitle = {Proceedings of the {CoNLL} 2018 Shared Task: Multilingual Parsing from Raw Text to Universal Dependencies},
 month = {October},
 pages = {160--170},
 publisher = {Association for Computational Linguistics},
 title = {Universal Dependency Parsing from Scratch},
 url = {https://nlp.stanford.edu/pubs/qi2018universal.pdf},
 year = {2018}
}
```

PyTorch 在这个存储库中实现神经管道的原因是[Peng Qi](http://qipeng.me)和[Yuhao Zhang](http://yuhao.im)，在[Tim Dozat](https://web.stanford.edu/~tdozat/)的帮助下和[Jason Bolton](mailto:jebolton@stanford.edu)。

此版本与斯坦福大学的 CoNLL 2018 共享任务系统不同。
标记器，变形器，形态特征和多字词系统是共享任务代码的清理版本，但在竞争中我们使用[Tensorflow 版本](https://github.com/tdozat/Parser-v3) )[Tim Dozat](https://web.stanford.edu/~tdozat/)的标记器和解析器，已在 PyTorch 中大致复制(尽管与原始版本有一些偏差)。

如果您使用 CoreNLP 服务器，请引用 CoreNLP 软件包和相应模块[如此处](https://stanfordnlp.github.io/CoreNLP/#citing-stanford-corenlp-in-papers)(“引用斯坦福大学” CoreNLP 在论文中“)。
CoreNLP 客户端主要由[Arun Chaganty](http://arun.chagantys.org/)撰写，[Jason Bolton](mailto:jebolton@stanford.edu)率先将这两个项目合并在一起。

## 问题和使用问答

请使用以下渠道提问和发布报告。

| 目的               | 渠道                                                                      |
| ------------------ | ------------------------------------------------------------------------- |
| 用法问答           | [Google Group](https://groups.google.com/forum/#!forum/stanfordnlp)       |
| 错误报告和功能请求 | [GitHub Issue Tracker](https://github.com/stanfordnlp/stanfordnlp/issues) |

## 安装

StanfordNLP 支持 Python 3.6 或更高版本。我们强烈建议您从 PyPI 安装 StanfordNLP。 如果你已经安装了[pip](https://pip.pypa.io/en/stable/installing/), 简单地跑

```bash
pip install stanfordnlp
```

这也应该有助于解决 StanfordNLP 的所有依赖关系，例如[PyTorch](https://pytorch.org/) 1.0.0 或更高版本。

或者，您也可以从这个 git 存储库的源代码安装，这将为您提供更大的灵活性，可以在 StanfordNLP 之上进行开发并培训您自己的模型。
对于此选项，请运行

```bash
git clone git@github.com:stanfordnlp/stanfordnlp.git
cd stanfordnlp
pip install -e .
```

## 运行 StanfordNLP

### 神经管道入门

要运行您的第一个 StanfordNLP 管道，只需在 Python 交互式解释器中执行这些步骤即可:

```python
import stanfordnlp
stanfordnlp.download('en')   # This downloads the English models for the neural pipeline
nlp = stanfordnlp.Pipeline() # This sets up a default neural pipeline in English
doc = nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
doc.sentences[0].print_dependencies()
```

最后一个命令将打印出输入字符串中第一个句子中的单词(或“Document”，因为它在 StanfordNLP 中表示)，以及在该句子的 Universal Dependencies 解析中管理它的单词的索引(它的“头”)，以及单词之间的依赖关系。
输出应该是这样的:

```
('Barack', '4', 'nsubj:pass')
('Obama', '1', 'flat')
('was', '4', 'aux:pass')
('born', '0', 'root')
('in', '6', 'case')
('Hawaii', '4', 'obl')
('.', '4', 'punct')
```

**Note:** 如果你遇到像'OSError:[Errno 22]无效的参数`这样的问题，你很可能会受到[已知 Python 问题](https://bugs.python.org/issue24658)的影响，我们建议使用 Python 3.6.8 或更高版本以及 Python 3.7.2 或更高版本。

我们还提供了一个多语言[演示脚本](https://github.com/stanfordnlp/stanfordnlp/blob/master/demo/pipeline_demo.py)，演示了如何使用除英语之外的其他语言的 StanfordNLP，例如中文(传统)

```bash
python demo/pipeline_demo.py -l zh
```

有关详细信息，请参阅[我们的入门指南](https://stanfordnlp.github.io/stanfordnlp/installation_download.html#getting-started)。

### 访问 Java Stanford CoreNLP 服务器

除了神经管道，该项目还包括一个官方包装器，用于使用 Python 代码访问 Java Stanford CoreNLP 服务器。

有一些初始设置步骤。

- Download [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and models for the language you wish to use.
- 将模型罐放入分发文件夹中
- 告诉 Stanford CoreNLP 所在的 python 代码: `export CORENLP_HOME=/path/to/stanford-corenlp-full-2018-10-05`

We provide another [demo script](https://github.com/stanfordnlp/stanfordnlp/blob/master/demo/corenlp.py) that shows how one can use the CoreNLP client and extract various annotations from it.

### 神经管道的训练模型

我们目前为 CoNLL 2018 共享任务中的所有树库提供模型。 你可以找到下载和使用这些模型的说明[这里](https://stanfordnlp.github.io/stanfordnlp/installation_download.html#models-for-human-languages).

### 批量管理以最大化管道速度

要最大限度地提高速度性能，必须在批量文档上运行管道。
一次在一个句子上运行 for 循环将非常缓慢。
此时最好的方法是将文档连接在一起，每个文档用空行分隔(即两个换行符`\n\n\n`)。
标记器将空行识别为句子中断。
我们正在积极致力于改进多文档处理。

## 训练自己的神经管道

该库中的所有神经模块，包括标记器，多字令牌(MWT)扩展器，POS /形态特征标记器，变形器和依赖解析器，都可以使用您自己的[CoNLL-U](https://universaldependencies.org/format.html)格式数据进行训练。
目前，我们不支持通过`Pipeline`界面进行模型培训。
因此，要训练自己的模型，您需要克隆此 git 存储库并从源设置。

有关如何培训和评估您自己的模型的详细分步指导，请访问我们的[培训文档](https://stanfordnlp.github.io/stanfordnlp/training.html).

## 执照

StanfordNLP 是在 Apache License 2.0 版下发布的。
有关详细信息，请参阅[LICENSE](https://github.com/stanfordnlp/stanfordnlp/blob/master/LICENSE)文件。
