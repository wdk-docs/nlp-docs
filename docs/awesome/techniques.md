# 技术

## 文字嵌入

Text embeddings allow deep learning to be effective on smaller datasets. These are often first inputs to a deep learning archiectures and most popular way of transfer learning in NLP. Embeddings are simply vectors or a more generically, real valued representations of strings. Word embeddings are considered a great starting point for most deep NLP tasks.

The most popular names in word embeddings are word2vec by Google (Mikolov) and GloVe by Stanford (Pennington, Socher and Manning). fastText seems to be a fairly popular for multi-lingual sub-word embeddings.

### 单词嵌入

| Embedding | Paper                                                                                                                                                                                                                                                                                                 | Organisation | gensim- Training Support         | Blogs                                                                                                                                                                                                                                  |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| word2vec  | [Official Implementation](https://code.google.com/archive/p/word2vec/), T.Mikolove et al. 2013. Distributed Representations of Words and Phrases and their Compositionality. [pdf](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) | Google       | Yes :heavy_check_mark:           | Visual explanations by colah at [Deep Learning, NLP, and Representations](http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/); gensim's [Making Sense of word2vec](https://rare-technologies.com/making-sense-of-word2vec) |
| GloVe     | Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf](https://nlp.stanford.edu/pubs/glove.pdf)                                                                                                                                   | Stanford     | No :negative_squared_cross_mark: | [Morning Paper on GloVe](https://blog.acolyer.org/2016/04/22/glove-global-vectors-for-word-representation/) by acoyler                                                                                                                 |
| fastText  | [Official Implementation](https://github.com/facebookresearch/fastText), T. Mikolov et al. 2017. Enriching Word Vectors with Subword Information. [pdf](https://arxiv.org/abs/1607.04606)                                                                                                             | Facebook     | Yes :heavy_check_mark:           | [Fasttext: Under the Hood](https://towardsdatascience.com/fasttext-under-the-hood-11efc57b2b3)                                                                                                                                         |

Notes for Beginners:

- Thumb Rule: **fastText >> GloVe > word2vec**
- You can find [pre-trained fasttext Vectors](https://fasttext.cc/docs/en/pretrained-vectors.html) in several languages
- If you are interested in the logic and intuition behind word2vec and GloVe: [The Amazing Power of Word Vectors](https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/) and introduce the topics well
- [arXiv: Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759), and [arXiv: FastText.zip: Compressing text classification models](https://arxiv.org/abs/1612.03651) were released as part of fasttext

### 基于句子和语言模型的词嵌入

- _ElMo_ from [Deep Contextualized Word Represenations](https://arxiv.org/abs/1802.05365) - [PyTorch implmentation](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) - [TF Implementation](https://github.com/allenai/bilm-tf)
- _ULimFit_ aka [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) by Jeremy Howard and Sebastian Ruder
- _InferSent_ from [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data](https://arxiv.org/abs/1705.02364) by facebook
- _CoVe_ from [Learned in Translation: Contextualized Word Vectors](https://arxiv.org/abs/1708.00107)
- _Pargraph vectors_ from [Distributed Representations of Sentences and Documents](https://cs.stanford.edu/~quocle/paragraph_vector.pdf). See [doc2vec tutorial at gensim](https://rare-technologies.com/doc2vec-tutorial/)
- [sense2vec](https://arxiv.org/abs/1511.06388) - on word sense disambiguation
- [Skip Thought Vectors](https://arxiv.org/abs/1506.06726) - word representation method
- [Adaptive skip-gram](https://arxiv.org/abs/1502.07257) - similar approach, with adaptive properties
- [Sequence to Sequence Learning](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) - word vectors for machine translation

## 问答和知识提取

- [DrQA: Open Domain Question Answering](https://github.com/facebookresearch/DrQA) by facebook on Wikipedia data
- DocQA: [Simple and Effective Multi-Paragraph Reading Comprehension](https://github.com/allenai/document-qa) by AllenAI
- [Markov Logic Networks for Natural Language Question Answering](https://arxiv.org/pdf/1507.03045v1.pdf)
- [Template-Based Information Extraction without the Templates](https://www.usna.edu/Users/cs/nchamber/pubs/acl2011-chambers-templates.pdf)
- [Relation extraction with matrix factorization and universal schemas](https://www.anthology.aclweb.org/N/N13/N13-1008.pdf)
- [Privee: An Architecture for Automatically Analyzing Web Privacy Policies](https://www.sebastianzimmeck.de/zimmeckAndBellovin2014Privee.pdf)
- [Teaching Machines to Read and Comprehend](https://arxiv.org/abs/1506.03340) - DeepMind paper
- [Relation Extraction with Matrix Factorization and Universal Schemas](https://www.riedelcastro.org//publications/papers/riedel13relation.pdf)
- [Towards a Formal Distributional Semantics: Simulating Logical Calculi with Tensors](https://www.aclweb.org/anthology/S13-1001)
- [Presentation slides for MLN tutorial](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/mln-summary-20150918.ppt)
- [Presentation slides for QA applications of MLNs](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/Markov%20Logic%20Networks%20for%20Natural%20Language%20Question%20Answering.pdf)
- [Presentation slides](https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/poon-paper.pdf)
