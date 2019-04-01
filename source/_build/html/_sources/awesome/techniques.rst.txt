技术
====

文字嵌入
--------

Text embeddings allow deep learning to be effective on smaller datasets.
These are often first inputs to a deep learning archiectures and most
popular way of transfer learning in NLP. Embeddings are simply vectors
or a more generically, real valued representations of strings. Word
embeddings are considered a great starting point for most deep NLP
tasks.

The most popular names in word embeddings are word2vec by Google
(Mikolov) and GloVe by Stanford (Pennington, Socher and Manning).
fastText seems to be a fairly popular for multi-lingual sub-word
embeddings.

单词嵌入
~~~~~~~~

+---+-----------------------------------+---+---+---------------------------+
| E | Paper                             | O | g | Blogs                     |
| m |                                   | r | e |                           |
| b |                                   | g | n |                           |
| e |                                   | a | s |                           |
| d |                                   | n | i |                           |
| d |                                   | i | m |                           |
| i |                                   | s | - |                           |
| n |                                   | a | T |                           |
| g |                                   | t | r |                           |
|   |                                   | i | a |                           |
|   |                                   | o | i |                           |
|   |                                   | n | n |                           |
|   |                                   |   | i |                           |
|   |                                   |   | n |                           |
|   |                                   |   | g |                           |
|   |                                   |   | S |                           |
|   |                                   |   | u |                           |
|   |                                   |   | p |                           |
|   |                                   |   | p |                           |
|   |                                   |   | o |                           |
|   |                                   |   | r |                           |
|   |                                   |   | t |                           |
+===+===================================+===+===+===========================+
| w | `Official                         | G | Y | Visual explanations by    |
| o | Implementation <https://code.goog | o | e | colah at `Deep Learning,  |
| r | le.com/archive/p/word2vec/>`__,   | o | s | NLP, and                  |
| d | T.Mikolove et al. 2013.           | g | : | Representations <http://c |
| 2 | Distributed Representations of    | l | h | olah.github.io/posts/2014 |
| v | Words and Phrases and their       | e | e | -07-NLP-RNNs-Representati |
| e | Compositionality.                 |   | a | ons/>`__;                 |
| c | `pdf <https://papers.nips.cc/pape |   | v | gensim’s `Making Sense of |
|   | r/5021-distributed-representation |   | y | word2vec <https://rare-te |
|   | s-of-words-and-phrases-and-their- |   | _ | chnologies.com/making-sen |
|   | compositionality.pdf>`__          |   | c | se-of-word2vec>`__        |
|   |                                   |   | h |                           |
|   |                                   |   | e |                           |
|   |                                   |   | c |                           |
|   |                                   |   | k |                           |
|   |                                   |   | _ |                           |
|   |                                   |   | m |                           |
|   |                                   |   | a |                           |
|   |                                   |   | r |                           |
|   |                                   |   | k |                           |
|   |                                   |   | : |                           |
+---+-----------------------------------+---+---+---------------------------+
| G | Jeffrey Pennington, Richard       | S | N | `Morning Paper on         |
| l | Socher, and Christopher D.        | t | o | GloVe <https://blog.acoly |
| o | Manning. 2014. GloVe: Global      | a | : | er.org/2016/04/22/glove-g |
| V | Vectors for Word Representation.  | n | n | lobal-vectors-for-word-re |
| e | `pdf <https://nlp.stanford.edu/pu | f | e | presentation/>`__         |
|   | bs/glove.pdf>`__                  | o | g | by acoyler                |
|   |                                   | r | a |                           |
|   |                                   | d | t |                           |
|   |                                   |   | i |                           |
|   |                                   |   | v |                           |
|   |                                   |   | e |                           |
|   |                                   |   | _ |                           |
|   |                                   |   | s |                           |
|   |                                   |   | q |                           |
|   |                                   |   | u |                           |
|   |                                   |   | a |                           |
|   |                                   |   | r |                           |
|   |                                   |   | e |                           |
|   |                                   |   | d |                           |
|   |                                   |   | _ |                           |
|   |                                   |   | c |                           |
|   |                                   |   | r |                           |
|   |                                   |   | o |                           |
|   |                                   |   | s |                           |
|   |                                   |   | s |                           |
|   |                                   |   | _ |                           |
|   |                                   |   | m |                           |
|   |                                   |   | a |                           |
|   |                                   |   | r |                           |
|   |                                   |   | k |                           |
|   |                                   |   | : |                           |
+---+-----------------------------------+---+---+---------------------------+
| f | `Official                         | F | Y | `Fasttext: Under the      |
| a | Implementation <https://github.co | a | e | Hood <https://towardsdata |
| s | m/facebookresearch/fastText>`__,  | c | s | science.com/fasttext-unde |
| t | T. Mikolov et al. 2017. Enriching | e | : | r-the-hood-11efc57b2b3>`_ |
| T | Word Vectors with Subword         | b | h | _                         |
| e | Information.                      | o | e |                           |
| x | `pdf <https://arxiv.org/abs/1607. | o | a |                           |
| t | 04606>`__                         | k | v |                           |
|   |                                   |   | y |                           |
|   |                                   |   | _ |                           |
|   |                                   |   | c |                           |
|   |                                   |   | h |                           |
|   |                                   |   | e |                           |
|   |                                   |   | c |                           |
|   |                                   |   | k |                           |
|   |                                   |   | _ |                           |
|   |                                   |   | m |                           |
|   |                                   |   | a |                           |
|   |                                   |   | r |                           |
|   |                                   |   | k |                           |
|   |                                   |   | : |                           |
+---+-----------------------------------+---+---+---------------------------+

Notes for Beginners:

-  Thumb Rule: **fastText >> GloVe > word2vec**
-  You can find `pre-trained fasttext
   Vectors <https://fasttext.cc/docs/en/pretrained-vectors.html>`__ in
   several languages
-  If you are interested in the logic and intuition behind word2vec and
   GloVe: `The Amazing Power of Word
   Vectors <https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/>`__
   and introduce the topics well
-  `arXiv: Bag of Tricks for Efficient Text
   Classification <https://arxiv.org/abs/1607.01759>`__, and `arXiv:
   FastText.zip: Compressing text classification
   models <https://arxiv.org/abs/1612.03651>`__ were released as part of
   fasttext

基于句子和语言模型的词嵌入
~~~~~~~~~~~~~~~~~~~~~~~~~~

-  *ElMo* from `Deep Contextualized Word
   Represenations <https://arxiv.org/abs/1802.05365>`__ - `PyTorch
   implmentation <https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md>`__
   - `TF Implementation <https://github.com/allenai/bilm-tf>`__
-  *ULimFit* aka `Universal Language Model Fine-tuning for Text
   Classification <https://arxiv.org/abs/1801.06146>`__ by Jeremy Howard
   and Sebastian Ruder
-  *InferSent* from `Supervised Learning of Universal Sentence
   Representations from Natural Language Inference
   Data <https://arxiv.org/abs/1705.02364>`__ by facebook
-  *CoVe* from `Learned in Translation: Contextualized Word
   Vectors <https://arxiv.org/abs/1708.00107>`__
-  *Pargraph vectors* from `Distributed Representations of Sentences and
   Documents <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`__.
   See `doc2vec tutorial at
   gensim <https://rare-technologies.com/doc2vec-tutorial/>`__
-  `sense2vec <https://arxiv.org/abs/1511.06388>`__ - on word sense
   disambiguation
-  `Skip Thought Vectors <https://arxiv.org/abs/1506.06726>`__ - word
   representation method
-  `Adaptive skip-gram <https://arxiv.org/abs/1502.07257>`__ - similar
   approach, with adaptive properties
-  `Sequence to Sequence
   Learning <https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf>`__
   - word vectors for machine translation

问答和知识提取
--------------

-  `DrQA: Open Domain Question
   Answering <https://github.com/facebookresearch/DrQA>`__ by facebook
   on Wikipedia data
-  DocQA: `Simple and Effective Multi-Paragraph Reading
   Comprehension <https://github.com/allenai/document-qa>`__ by AllenAI
-  `Markov Logic Networks for Natural Language Question
   Answering <https://arxiv.org/pdf/1507.03045v1.pdf>`__
-  `Template-Based Information Extraction without the
   Templates <https://www.usna.edu/Users/cs/nchamber/pubs/acl2011-chambers-templates.pdf>`__
-  `Relation extraction with matrix factorization and universal
   schemas <https://www.anthology.aclweb.org/N/N13/N13-1008.pdf>`__
-  `Privee: An Architecture for Automatically Analyzing Web Privacy
   Policies <https://www.sebastianzimmeck.de/zimmeckAndBellovin2014Privee.pdf>`__
-  `Teaching Machines to Read and
   Comprehend <https://arxiv.org/abs/1506.03340>`__ - DeepMind paper
-  `Relation Extraction with Matrix Factorization and Universal
   Schemas <https://www.riedelcastro.org//publications/papers/riedel13relation.pdf>`__
-  `Towards a Formal Distributional Semantics: Simulating Logical
   Calculi with Tensors <https://www.aclweb.org/anthology/S13-1001>`__
-  `Presentation slides for MLN
   tutorial <https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/mln-summary-20150918.ppt>`__
-  `Presentation slides for QA applications of
   MLNs <https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/Markov%20Logic%20Networks%20for%20Natural%20Language%20Question%20Answering.pdf>`__
-  `Presentation
   slides <https://github.com/clulab/nlp-reading-group/blob/master/fall-2015-resources/poon-paper.pdf>`__
