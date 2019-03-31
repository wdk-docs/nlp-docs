使用 BERT 进行预训练
====================

我们正在发布代码，在任意文本语料库上做“蒙面 LM”和“下一句话预测”。
请注意，这不是用于论文的确切代码（原始代码是用 C
++编写的，并且有一些额外的复杂性），但是此代码确实生成了本文所述的预训练数据。

Here’s how to run the data generation. The input is a plain text file,
with one sentence per line. (It is important that these be actual
sentences for the “next sentence prediction” task). Documents are
delimited by empty lines. The output is a set of ``tf.train.Example``\ s
serialized into ``TFRecord`` file format.

You can perform sentence segmentation with an off-the-shelf NLP toolkit
such as `spaCy <https://spacy.io/>`__. The
``create_pretraining_data.py`` script will concatenate segments until
they reach the maximum sequence length to minimize computational waste
from padding (see the script for more details). However, you may want to
intentionally add a slight amount of noise to your input data (e.g.,
randomly truncate 2% of input segments) to make it more robust to
non-sentential input during fine-tuning.

This script stores all of the examples for the entire input file in
memory, so for large data files you should shard the input file and call
the script multiple times. (You can pass in a file glob to
``run_pretraining.py``, e.g., ``tf_examples.tf_record*``.)

The ``max_predictions_per_seq`` is the maximum number of masked LM
predictions per sequence. You should set this to around
``max_seq_length`` \* ``masked_lm_prob`` (the script doesn’t do that
automatically because the exact value needs to be passed to both
scripts).

.. code:: shell

   python create_pretraining_data.py \
     --input_file=./sample_text.txt \
     --output_file=/tmp/tf_examples.tfrecord \
     --vocab_file=$BERT_BASE_DIR/vocab.txt \
     --do_lower_case=True \
     --max_seq_length=128 \
     --max_predictions_per_seq=20 \
     --masked_lm_prob=0.15 \
     --random_seed=12345 \
     --dupe_factor=5

Here’s how to run the pre-training. Do not include ``init_checkpoint``
if you are pre-training from scratch. The model configuration (including
vocab size) is specified in ``bert_config_file``. This demo code only
pre-trains for a small number of steps (20), but in practice you will
probably want to set ``num_train_steps`` to 10000 steps or more. The
``max_seq_length`` and ``max_predictions_per_seq`` parameters passed to
``run_pretraining.py`` must be the same as
``create_pretraining_data.py``.

.. code:: shell

   python run_pretraining.py \
     --input_file=/tmp/tf_examples.tfrecord \
     --output_dir=/tmp/pretraining_output \
     --do_train=True \
     --do_eval=True \
     --bert_config_file=$BERT_BASE_DIR/bert_config.json \
     --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
     --train_batch_size=32 \
     --max_seq_length=128 \
     --max_predictions_per_seq=20 \
     --num_train_steps=20 \
     --num_warmup_steps=10 \
     --learning_rate=2e-5

This will produce an output like this:

::

   ***** Eval results *****
     global_step = 20
     loss = 0.0979674
     masked_lm_accuracy = 0.985479
     masked_lm_loss = 0.0979328
     next_sentence_accuracy = 1.0
     next_sentence_loss = 3.45724e-05

Note that since our ``sample_text.txt`` file is very small, this example
training will overfit that data in only a few steps and produce
unrealistically high accuracy numbers.

预训练提示和警告
----------------

-  **If using your own vocabulary, make sure to change ``vocab_size`` in
   ``bert_config.json``. If you use a larger vocabulary without changing
   this, you will likely get NaNs when training on GPU or TPU due to
   unchecked out-of-bounds access.**
-  If your task has a large domain-specific corpus available (e.g.,
   “movie reviews” or “scientific papers”), it will likely be beneficial
   to run additional steps of pre-training on your corpus, starting from
   the BERT checkpoint.
-  The learning rate we used in the paper was 1e-4. However, if you are
   doing additional steps of pre-training starting from an existing BERT
   checkpoint, you should use a smaller learning rate (e.g., 2e-5).
-  Current BERT models are English-only, but we do plan to release a
   multilingual model which has been pre-trained on a lot of languages
   in the near future (hopefully by the end of November 2018).
-  Longer sequences are disproportionately expensive because attention
   is quadratic to the sequence length. In other words, a batch of 64
   sequences of length 512 is much more expensive than a batch of 256
   sequences of length 128. The fully-connected/convolutional cost is
   the same, but the attention cost is far greater for the 512-length
   sequences. Therefore, one good recipe is to pre-train for, say,
   90,000 steps with a sequence length of 128 and then for 10,000
   additional steps with a sequence length of 512. The very long
   sequences are mostly needed to learn positional embeddings, which can
   be learned fairly quickly. Note that this does require generating the
   data twice with different values of ``max_seq_length``.
-  If you are pre-training from scratch, be prepared that pre-training
   is computationally expensive, especially on GPUs. If you are
   pre-training from scratch, our recommended recipe is to pre-train a
   ``BERT-Base`` on a single `preemptible Cloud TPU
   v2 <https://cloud.google.com/tpu/docs/pricing>`__, which takes about
   2 weeks at a cost of about $500 USD (based on the pricing in October
   2018). You will have to scale down the batch size when only training
   on a single Cloud TPU, compared to what was used in the paper. It is
   recommended to use the largest batch size that fits into TPU memory.

预训练数据
----------

We will **not** be able to release the pre-processed datasets used in
the paper. For Wikipedia, the recommended pre-processing is to download
`the latest
dump <https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2>`__,
extract the text with
```WikiExtractor.py`` <https://github.com/attardi/wikiextractor>`__, and
then apply any necessary cleanup to convert it into plain text.

Unfortunately the researchers who collected the
`BookCorpus <http://yknzhu.wixsite.com/mbweb>`__ no longer have it
available for public download. The `Project Guttenberg
Dataset <https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html>`__
is a somewhat smaller (200M word) collection of older books that are
public domain.

`Common Crawl <http://commoncrawl.org/>`__ is another very large
collection of text, but you will likely have to do substantial
pre-processing and cleanup to extract a usable corpus for pre-training
BERT.

学习一个新的 WordPiece 词汇表
-----------------------------

This repository does not include code for *learning* a new WordPiece
vocabulary. The reason is that the code used in the paper was
implemented in C++ with dependencies on Google’s internal libraries. For
English, it is almost always better to just start with our vocabulary
and pre-trained models. For learning vocabularies of other languages,
there are a number of open source options available. However, keep in
mind that these are not compatible with our ``tokenization.py`` library:

-  `Google’s SentencePiece
   library <https://github.com/google/sentencepiece>`__

-  `tensor2tensor’s WordPiece generation
   script <https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder_build_subword.py>`__

-  `Rico Sennrich’s Byte Pair Encoding
   library <https://github.com/rsennrich/subword-nmt>`__
