常用分词工具包
==============

+-----------------------+----------------+-----------------+-------------------+----------+
|         名称          |       星       |      更新       |       活跃        |   费用   |
+=======================+================+=================+===================+==========+
| jieba-结巴分词        | stars-jieba    | commit-jieba    | activity-jieba    | 免费     |
+-----------------------+----------------+-----------------+-------------------+----------+
| HanLP-汉语言处理包    | stars-hanlp    | commit-hanlp    | activity-hanlp    | 免费     |
+-----------------------+----------------+-----------------+-------------------+----------+
| SnowNLP-中文的类库    | stars-snownlp  | commit-snownlp  | activity-snownlp  | 免费     |
+-----------------------+----------------+-----------------+-------------------+----------+
| pkuseg-北大多领域分词 | stars-pkuseg   | commit-pkuseg   | activity-pkuseg   | 免费     |
+-----------------------+----------------+-----------------+-------------------+----------+
| NLPIR-中科院汉语分词  | stars-nlpir    | commit-nlpir    | activity-nlpir    | 付费     |
+-----------------------+----------------+-----------------+-------------------+----------+
| FoolNLTK-中文处理     | stars-foolnltk | commit-foolnltk | activity-foolnltk | 免费     |
+-----------------------+----------------+-----------------+-------------------+----------+
| pyltp-哈工大语言云    | stars-pyltp    | commit-pyltp    | activity-pyltp    | 商用付费 |
+-----------------------+----------------+-----------------+-------------------+----------+
| THULAC-清华词法分析   | stars-thulac   | commit-thulac   | activity-thulac   | 商用付费 |
+-----------------------+----------------+-----------------+-------------------+----------+
| Jiagu-甲骨NLP         | stars-jiagu    | commit-jiagu    | activity-jiagu    | 免费     |
+-----------------------+----------------+-----------------+-------------------+----------+

+-----------------------+------+------+------+------+------+------+------+
|         名称          | 分词 | 词性 | 实体 | 情感 | 提炼 | 并行 | 词典 |
+=======================+======+======+======+======+======+======+======+
| jieba 结巴分词        | √    | √    | ×    | ×    | √    | √    | √    |
+-----------------------+------+------+------+------+------+------+------+
| pkuseg-北大多领域分词 | √    | ×    | ×    | ×    | ×    | √    | √    |
+-----------------------+------+------+------+------+------+------+------+
| FoolNLTK              | √    | √    | √    | ×    | ×    | √    | √    |
+-----------------------+------+------+------+------+------+------+------+
| HanLP-汉语言处理包    | √    | √    | √    | √    | √    | ×    | √    |
+-----------------------+------+------+------+------+------+------+------+

.. | stars-jieba       | image:: https://img.shields.io/github/stars/fxsjy/jieba.svg?style=social                                                 |
.. | commit-jieba      | image:: https://img.shields.io/github/last-commit/fxsjy/jieba.svg?label=%E6%9B%B4%E6%96%B0&style=social                  |
.. | activity-jieba    | image:: https://img.shields.io/github/commit-activity/m/fxsjy/jieba.svg?label=%E6%B4%BB%E8%B7%83&style=social            |
.. | stars-hanlp       | image:: https://img.shields.io/github/stars/hankcs/HanLP.svg?style=social                                                |
.. | commit-hanlp      | image:: https://img.shields.io/github/last-commit/hankcs/HanLP.svg?style=social&label=%E6%9B%B4%E6%96%B0                 |
.. | activity-hanlp    | image:: https://img.shields.io/github/commit-activity/m/hankcs/hanlp.svg?label=%E6%B4%BB%E8%B7%83&style=social           |
.. | stars-snownlp     | image:: https://img.shields.io/github/stars/isnowfy/snownlp.svg?style=social                                             |
.. | commit-snownlp    | image:: https://img.shields.io/github/last-commit/isnowfy/snownlp.svg?style=social&label=%E6%9B%B4%E6%96%B0              |
.. | activity-snownlp  | image:: https://img.shields.io/github/commit-activity/m/isnowfy/snownlp.svg?label=%E6%B4%BB%E8%B7%83&style=social        |
.. | stars-pkuseg      | image:: https://img.shields.io/github/stars/lancopku/pkuseg-python.svg?style=social                                      |
.. | commit-pkuseg     | image:: https://img.shields.io/github/last-commit/lancopku/pkuseg-python.svg?style=social&label=%E6%9B%B4%E6%96%B0       |
.. | activity-pkuseg   | image:: https://img.shields.io/github/commit-activity/m/lancopku/pkuseg-python.svg?label=%E6%B4%BB%E8%B7%83&style=social |
.. | stars-nlpir       | image:: https://img.shields.io/github/stars/NLPIR-team/NLPIR.svg?style=social                                            |
.. | commit-nlpir      | image:: https://img.shields.io/github/last-commit/NLPIR-team/NLPIR.svg?style=social&label=%E6%9B%B4%E6%96%B0             |
.. | activity-nlpir    | image:: https://img.shields.io/github/commit-activity/m/NLPIR-team/nlpir.svg?label=%E6%B4%BB%E8%B7%83&style=social       |
.. | stars-foolnltk    | image:: https://img.shields.io/github/stars/rockyzhengwu/FoolNLTK.svg?style=social                                       |
.. | commit-foolnltk   | image:: https://img.shields.io/github/last-commit/rockyzhengwu/FoolNLTK.svg?style=social&label=%E6%9B%B4%E6%96%B0        |
.. | activity-foolnltk | image:: https://img.shields.io/github/commit-activity/m/rockyzhengwu/foolnltk.svg?label=%E6%B4%BB%E8%B7%83&style=social  |
.. | stars-pyltp       | image:: https://img.shields.io/github/stars/HIT-SCIR/pyltp.svg?style=social                                              |
.. | commit-pyltp      | image:: https://img.shields.io/github/last-commit/HIT-SCIR/pyltp.svg?style=social&label=%E6%9B%B4%E6%96%B0               |
.. | activity-pyltp    | image:: https://img.shields.io/github/commit-activity/m/HIT-SCIR/pyltp.svg?label=%E6%B4%BB%E8%B7%83&style=social         |
.. | stars-thulac      | image:: https://img.shields.io/github/stars/thunlp/THULAC.svg?style=social                                               |
.. | commit-thulac     | image:: https://img.shields.io/github/last-commit/thunlp/THULAC.svg?style=social&label=%E6%9B%B4%E6%96%B0                |
.. | activity-thulac   | image:: https://img.shields.io/github/commit-activity/m/thunlp/thulac.svg?label=%E6%B4%BB%E8%B7%83&style=social          |
.. | stars-jiagu       | image:: https://img.shields.io/github/stars/ownthink/Jiagu.svg?style=social                                              |
.. | commit-jiagu      | image:: https://img.shields.io/github/last-commit/ownthink/Jiagu.svg?style=social&label=%E6%9B%B4%E6%96%B0               |
.. | activity-jiagu    | image:: https://img.shields.io/github/commit-activity/m/ownthink/jiagu.svg?label=%E6%B4%BB%E8%B7%83&style=social         |


## 菜单
---------

.. toctree::
   :maxdepth: 2
   :glob:

   FoolNLTK/*
   JieBa/*
   *
