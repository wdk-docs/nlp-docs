模型
====

目前有两种多语言模型可供选择。
我们不打算发布更多的单语言模型，但我们将来可能会发布这两种版本的“BERT-Large”版本:

-  ```BERT-Base, 多语套装(新推荐)`` <https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip>`__:
   104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
-  ```BERT-Base, 多语言Uncased(Orig，不推荐)`` <https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip>`__:
   102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
-  ```BERT-Base, 中文`` <https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip>`__:
   Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads,
   110M parameters

**“多语言套装(新)”模型还修复了许多语言中的规范化问题，因此建议使用非拉丁字母表的语言(对于大多数使用拉丁字母的语言，它通常更好)。
使用此模型时，请确保将\ ``--do_lower_case = false``\ 传递给\ ``run_pretraining.py``\ 和其他脚本。**

请参阅多语言模型支持的\ `语言列表 <＃list-of-languages>`__\ 。
多语言模型确实包含中文(和英文)，但如果您的微调数据仅限中文，则中文模型可能会产生更好的结果。

结果
----

为了评估这些系统，我们使用\ `XNLI
数据集 <https://github.com/facebookresearch/XNLI>`__\ 数据集，这是\ `MultiNLI <https://www.nyu.edu/projects/bowman/multinli/>`__\ 的一个版本其中开发和测试集已经(由人类)翻译成
15 种语言。 请注意，训练集是\ *机器*\ 翻译(我们使用的是 XNLI
提供的翻译，而不是 Google NMT)。 为清楚起见，我们仅报告以下 6 种语言:

.. raw:: html

   <!-- mdformat off(没有表格) -->

+---------+---------+---------+---------+---------+---------+---------+
| 系统    | 英语    | 中文    | 西班牙语 | 德语   | 阿拉伯  | 乌尔都语 |
+=========+=========+=========+=========+=========+=========+=========+
| XNLI    | 73.7    | 67.0    | 68.8    | 66.5    | 65.8    | 56.6    |
| Baselin |         |         |         |         |         |         |
| e       |         |         |         |         |         |         |
| -       |         |         |         |         |         |         |
| Transla |         |         |         |         |         |         |
| te      |         |         |         |         |         |         |
| Train   |         |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| XNLI    | 73.7    | 68.3    | 70.7    | 68.7    | 66.8    | 59.3    |
| Baselin |         |         |         |         |         |         |
| e       |         |         |         |         |         |         |
| -       |         |         |         |         |         |         |
| Transla |         |         |         |         |         |         |
| te      |         |         |         |         |         |         |
| Test    |         |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| BERT -  | **81.9* | **76.6* | **77.8* | **75.9* | **70.7* | 61.6    |
| Transla | *       | *       | *       | *       | *       |         |
| te      |         |         |         |         |         |         |
| Train   |         |         |         |         |         |         |
| Cased   |         |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| BERT -  | 81.4    | 74.2    | 77.3    | 75.2    | 70.5    | 61.7    |
| Transla |         |         |         |         |         |         |
| te      |         |         |         |         |         |         |
| Train   |         |         |         |         |         |         |
| Uncased |         |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| BERT -  | 81.4    | 70.1    | 74.9    | 74.4    | 70.4    | **62.1* |
| Transla |         |         |         |         |         | *       |
| te      |         |         |         |         |         |         |
| Test    |         |         |         |         |         |         |
| Uncased |         |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+
| BERT -  | 81.4    | 63.8    | 74.3    | 70.5    | 62.1    | 58.3    |
| Zero    |         |         |         |         |         |         |
| Shot    |         |         |         |         |         |         |
| Uncased |         |         |         |         |         |         |
+---------+---------+---------+---------+---------+---------+---------+

.. raw:: html

   <!-- mdformat on -->

The first two rows are baselines from the XNLI paper and the last three
rows are our results with BERT.

**Translate Train** means that the MultiNLI training set was machine
translated from English into the foreign language. So training and
evaluation were both done in the foreign language. Unfortunately,
training was done on machine-translated data, so it is impossible to
quantify how much of the lower accuracy (compared to English) is due to
the quality of the machine translation vs. the quality of the
pre-trained model.

**Translate Test** means that the XNLI test set was machine translated
from the foreign language into English. So training and evaluation were
both done on English. However, test evaluation was done on
machine-translated English, so the accuracy depends on the quality of
the machine translation system.

**Zero Shot** means that the Multilingual BERT system was fine-tuned on
English MultiNLI, and then evaluated on the foreign language XNLI test.
In this case, machine translation was not involved at all in either the
pre-training or fine-tuning.

Note that the English result is worse than the 84.2 MultiNLI baseline
because this training used Multilingual BERT rather than English-only
BERT. This implies that for high-resource languages, the Multilingual
model is somewhat worse than a single-language model. However, it is not
feasible for us to train and maintain dozens of single-language model.
Therefore, if your goal is to maximize performance with a language other
than English or Chinese, you might find it beneficial to run
pre-training for additional steps starting from our Multilingual model
on data from your language of interest.

Here is a comparison of training Chinese models with the Multilingual
``BERT-Base`` and Chinese-only ``BERT-Base``:

+-------------------------+---------+
| System                  | Chinese |
+=========================+=========+
| XNLI Baseline           | 67.0    |
+-------------------------+---------+
| BERT Multilingual Model | 74.2    |
+-------------------------+---------+
| BERT Chinese-only Model | 77.2    |
+-------------------------+---------+

Similar to English, the single-language model does 3% better than the
Multilingual model.

微调示例
--------

The multilingual model does **not** require any special consideration or
API changes. We did update the implementation of ``BasicTokenizer`` in
``tokenization.py`` to support Chinese character tokenization, so please
update if you forked it. However, we did not change the tokenization
API.

To test the new models, we did modify ``run_classifier.py`` to add
support for the `XNLI
dataset <https://github.com/facebookresearch/XNLI>`__. This is a
15-language version of MultiNLI where the dev/test sets have been
human-translated, and the training set has been machine-translated.

To run the fine-tuning code, please download the `XNLI dev/test
set <https://s3.amazonaws.com/xnli/XNLI-1.0.zip>`__ and the `XNLI
machine-translated training
set <https://s3.amazonaws.com/xnli/XNLI-MT-1.0.zip>`__ and then unpack
both .zip files into some directory ``$XNLI_DIR``.

To run fine-tuning on XNLI. The language is hard-coded into
``run_classifier.py`` (Chinese by default), so please modify
``XnliProcessor`` if you want to run on another language.

This is a large dataset, so this will training will take a few hours on
a GPU (or about 30 minutes on a Cloud TPU). To run an experiment quickly
for debugging, just set ``num_train_epochs`` to a small value like
``0.1``.

.. code:: shell

   export BERT_BASE_DIR=/path/to/bert/chinese_L-12_H-768_A-12 # or multilingual_L-12_H-768_A-12
   export XNLI_DIR=/path/to/xnli

   python run_classifier.py \
     --task_name=XNLI \
     --do_train=true \
     --do_eval=true \
     --data_dir=$XNLI_DIR \
     --vocab_file=$BERT_BASE_DIR/vocab.txt \
     --bert_config_file=$BERT_BASE_DIR/bert_config.json \
     --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
     --max_seq_length=128 \
     --train_batch_size=32 \
     --learning_rate=5e-5 \
     --num_train_epochs=2.0 \
     --output_dir=/tmp/xnli_output/

With the Chinese-only model, the results should look something like
this:

::

    ***** Eval results *****
   eval_accuracy = 0.774116
   eval_loss = 0.83554
   global_step = 24543
   loss = 0.74603

细节
----

数据源和采样
~~~~~~~~~~~~

The languages chosen were the `top 100 languages with the largest
Wikipedias <https://meta.wikimedia.org/wiki/List_of_Wikipedias>`__. The
entire Wikipedia dump for each language (excluding user and talk pages)
was taken as the training data for each language

However, the size of the Wikipedia for a given language varies greatly,
and therefore low-resource languages may be “under-represented” in terms
of the neural network model (under the assumption that languages are
“competing” for limited model capacity to some extent).

However, the size of a Wikipedia also correlates with the number of
speakers of a language, and we also don’t want to overfit the model by
performing thousands of epochs over a tiny Wikipedia for a particular
language.

To balance these two factors, we performed exponentially smoothed
weighting of the data during pre-training data creation (and WordPiece
vocab creation). In other words, let’s say that the probability of a
language is *P(L)*, e.g., *P(English) = 0.21* means that after
concatenating all of the Wikipedias together, 21% of our data is
English. We exponentiate each probability by some factor *S* and then
re-normalize, and sample from that distribution. In our case we use
*S=0.7*. So, high-resource languages like English will be under-sampled,
and low-resource languages like Icelandic will be over-sampled. E.g., in
the original distribution English would be sampled 1000x more than
Icelandic, but after smoothing it’s only sampled 100x more.

符号化
~~~~~~

For tokenization, we use a 110k shared WordPiece vocabulary. The word
counts are weighted the same way as the data, so low-resource languages
are upweighted by some factor. We intentionally do *not* use any marker
to denote the input language (so that zero-shot training can work).

Because Chinese (and Japanese Kanji and Korean Hanja) does not have
whitespace characters, we add spaces around every character in the `CJK
Unicode
range <https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)>`__
before applying WordPiece. This means that Chinese is effectively
character-tokenized. Note that the CJK Unicode block only includes
Chinese-origin characters and does *not* include Hangul Korean or
Katakana/Hiragana Japanese, which are tokenized with
whitespace+WordPiece like all other languages.

For all other languages, we apply the `same recipe as
English <https://github.com/google-research/bert#tokenization>`__: (a)
lower casing+accent removal, (b) punctuation splitting, (c) whitespace
tokenization. We understand that accent markers have substantial meaning
in some languages, but felt that the benefits of reducing the effective
vocabulary make up for this. Generally the strong contextual models of
BERT should make up for any ambiguity introduced by stripping accent
markers.

语言清单
~~~~~~~~

多语言模型支持以下语言。 选择这些语言是因为它们是具有最大维基百科的前
100 种语言:

-  Afrikaans
-  Albanian
-  Arabic
-  Aragonese
-  Armenian
-  Asturian
-  Azerbaijani
-  Bashkir
-  Basque
-  Bavarian
-  Belarusian
-  Bengali
-  Bishnupriya Manipuri
-  Bosnian
-  Breton
-  Bulgarian
-  Burmese
-  Catalan
-  Cebuano
-  Chechen
-  Chinese (Simplified)
-  Chinese (Traditional)
-  Chuvash
-  Croatian
-  Czech
-  Danish
-  Dutch
-  English
-  Estonian
-  Finnish
-  French
-  Galician
-  Georgian
-  German
-  Greek
-  Gujarati
-  Haitian
-  Hebrew
-  Hindi
-  Hungarian
-  Icelandic
-  Ido
-  Indonesian
-  Irish
-  Italian
-  Japanese
-  Javanese
-  Kannada
-  Kazakh
-  Kirghiz
-  Korean
-  Latin
-  Latvian
-  Lithuanian
-  Lombard
-  Low Saxon
-  Luxembourgish
-  Macedonian
-  Malagasy
-  Malay
-  Malayalam
-  Marathi
-  Minangkabau
-  Nepali
-  Newar
-  Norwegian (Bokmal)
-  Norwegian (Nynorsk)
-  Occitan
-  Persian (Farsi)
-  Piedmontese
-  Polish
-  Portuguese
-  Punjabi
-  Romanian
-  Russian
-  Scots
-  Serbian
-  Serbo-Croatian
-  Sicilian
-  Slovak
-  Slovenian
-  South Azerbaijani
-  Spanish
-  Sundanese
-  Swahili
-  Swedish
-  Tagalog
-  Tajik
-  Tamil
-  Tatar
-  Telugu
-  Turkish
-  Ukrainian
-  Urdu
-  Uzbek
-  Vietnamese
-  Volapük
-  Waray-Waray
-  Welsh
-  West Frisian
-  Western Punjabi
-  Yoruba

**多语言套装(新)**\ 版本还包含\ **泰国**\ 和\ **蒙古语**\ ，这些都未包含在原始版本中。
