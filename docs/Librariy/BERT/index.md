# BERT

** 新的 2019 年 2 月 7 日：TfHub 模块 **

BERT 已上传至[TensorFlow Hub](https://tfhub.dev).
有关如何使用 TF Hub 模块的示例，请参阅`run_classifier_with_tfhub.py`，或者在[Colab](https://colab.sandbox.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb)上的浏览器中运行示例.

** 新的 2018 年 11 月 23 日：未规范化的多语言模式+泰语+蒙古语 **

我们上传了一个新的多语言模型，它不会对输入执行任何规范化（没有下限，重音剥离或 Unicode 规范化），还包括泰语和蒙古语。

**建议使用此版本开发多语言模型，尤其是使用非拉丁字母的语言。**

这不需要任何代码更改，可以在此处下载：

- **[`BERT-Base,多语言套装`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)**: 104 种语言，12 层，768 隐藏，12 头，110M 参数

** 新的 2018 年 11 月 15 日：SOTA SQuAD 2.0 系统 **

我们发布了代码更改，以重现我们的 83％F1 SQuAD 2.0 系统，目前排行榜上第一名是 3％。
有关详细信息，请参阅自述文件的 SQuAD 2.0 部分。

** 新的 2018 年 11 月 5 日：第三方 PyTorch 和 Chainer 版本的 BERT 可用 **

来自 HuggingFace 的 NLP 研究人员制作了[PyTorch 版本的 BERT](https://github.com/huggingface/pytorch-pretrained-BERT)，它与我们预先训练好的检查点兼容，并能够重现我们的结果。
Sosuke Kobayashi 也提供了[Biner 版本的 BERT](https://github.com/soskek/bert-chainer)（谢谢！）我们没有参与 PyTorch 实现的创建或维护，所以请将任何问题都指向该存储库的作者。

** 新的 2018 年 11 月 3 日：提供多种语言和中文模式 **

我们提供了两种新的 BERT 型号:

- **[`BERT-基地，多语种`](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip)（不推荐，使用`Multilingual Cased`代替）**：102 种语言，12 层，768 隐藏，12 头，110M 参数
- **[`BERT-Base，中文`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**: 中文简体和繁体，12 层，768 隐藏，12 头，110M 参数

我们对中文使用基于字符的标记化，对所有其他语言使用 WordPiece 标记化。
两种模型都应该开箱即用，不需要任何代码更改。
我们确实在`tokenization.py`中更新了`BasicTokenizer`的实现以支持中文字符
但是，我们没有更改标记化 API。

有关更多信息，请参阅[多语言自述文件](https://github.com/google-research/bert/blob/master/multilingual.md).

** 结束新信息 **

## 介绍

**BERT**，或**B**idirectional **E**ncoder **R**epresentations from **T**ransformers 的表示，是一种预训练语言表示的新方法，它获得了最先进的结果可用于各种自然语言处理（NLP）任务。

我们的学术论文详细描述了 BERT，并提供了许多任务的完整结果: [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805).

为了给出几个数字，这里是[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) 问题回答任务的结果:

| SQuAD v1.1 Leaderboard (Oct 8th 2018) | Test EM  | Test F1  |
| ------------------------------------- | :------: | :------: |
| 1st Place Ensemble - BERT             | **87.4** | **93.2** |
| 2nd Place Ensemble - nlnet            |   86.0   |   91.7   |
| 1st Place Single Model - BERT         | **85.1** | **91.8** |
| 2nd Place Single Model - nlnet        |   83.5   |   90.1   |

以及几种自然语言推理任务:

| System                  | MultiNLI | Question NLI |   SWAG   |
| ----------------------- | :------: | :----------: | :------: |
| BERT                    | **86.7** |   **91.1**   | **86.3** |
| OpenAI GPT (Prev. SOTA) |   82.2   |     88.1     |   75.0   |

还有许多其他任务。

而且，这些结果都是在几乎没有任务特定的神经网络架构设计的情况下获得的。

如果您已经知道 BERT 是什么并且您只是想要开始，您可以在几分钟内[下载预先训练的模型](#pre-trained-models)和[运行最先进的微调](#fine-tuning-with-bert)。

## 什么是 BERT？

BERT 是一种预训练语言表示的方法，这意味着我们在大型文本语料库（如维基百科）上训练通用的“语言理解”模型，然后将该模型用于我们关心的下游 NLP 任务（如问题）接听）。
BERT 优于以前的方法，因为它是第一个用于预训练 NLP 的*unsupervised*，*deeply bidirectional*系统。

*Unsupervised*意味着 BERT 仅使用纯文本语料库进行训练，这很重要，因为大量的纯文本数据可以在网络上以多种语言公开获得。

Pre-trained representations can also either be _context-free_ or _contextual_, and contextual representations can further be _unidirectional_ or _bidirectional_.

Context-free models such as [word2vec](https://www.tensorflow.org/tutorials/representation/word2vec) or
[GloVe](https://nlp.stanford.edu/projects/glove/) generate a single "word embedding" representation for each word in the vocabulary, so `bank` would have the same representation in `bank deposit` and `river bank`.
Contextual models instead generate a representation of each word that is based on the other words in the sentence.

BERT was built upon recent work in pre-training contextual representations — including [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432), [Generative Pre-Training](https://blog.openai.com/language-unsupervised/), [ELMo](https://allennlp.org/elmo), and [ULMFit](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html) — but crucially these models are all _unidirectional_ or _shallowly bidirectional_.
This means that each word is only contextualized using the words to its left (or right).
For example, in the sentence `I made a bank deposit` the unidirectional representation of `bank` is only based on `I made a` but not `deposit`.
Some previous work does combine the representations from separate left-context and right-context models, but only in a "shallow" manner.
BERT represents "bank" using both its left and right context — `I made a ... deposit` — starting from the very bottom of a deep neural network, so it is _deeply bidirectional_.

BERT uses a simple approach for this: We mask out 15% of the words in the input, run the entire sequence through a deep bidirectional [Transformer](https://arxiv.org/abs/1706.03762) encoder, and then predict only the masked words.

For example:

```
Input: the man went to the [MASK1] .
he bought a [MASK2] of milk.
Labels: [MASK1] = store; [MASK2] = gallon
```

In order to learn relationships between sentences, we also train on a simple
task which can be generated from any monolingual corpus: Given two sentences `A`
and `B`, is `B` the actual next sentence that comes after `A`, or just a random
sentence from the corpus?

```
Sentence A: the man went to the store .
Sentence B: he bought a gallon of milk .
Label: IsNextSentence
```

```
Sentence A: the man went to the store .
Sentence B: penguins are flightless .
Label: NotNextSentence
```

We then train a large model (12-layer to 24-layer Transformer) on a large corpus (Wikipedia + [BookCorpus](http://yknzhu.wixsite.com/mbweb)) for a long time (1M update steps), and that's BERT.

Using BERT has two stages: _Pre-training_ and _fine-tuning_.

**Pre-training** is fairly expensive (four days on 4 to 16 Cloud TPUs), but is a one-time procedure for each language (current models are English-only, but multilingual models will be released in the near future).
We are releasing a number of pre-trained models from the paper which were pre-trained at Google.
Most NLP researchers will never need to pre-train their own model from scratch.

**Fine-tuning** is inexpensive.
All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU,starting from the exact same pre-trained model.
SQuAD, for example, can be trained in around 30 minutes on a single Cloud TPU to achieve a Dev F1 score of
91.0%, which is the single system state-of-the-art.

The other important aspect of BERT is that it can be adapted to many types of NLP tasks very easily.
In the paper, we demonstrate state-of-the-art results on sentence-level (e.g., SST-2), sentence-pair-level (e.g., MultiNLI), word-level
(e.g., NER), and span-level (e.g., SQuAD) tasks with almost no task-specific modifications.

## 该存储库中发布了什么？

We are releasing the following:

- TensorFlow code for the BERT model architecture (which is mostly a standard [Transformer](https://arxiv.org/abs/1706.03762) architecture).
- Pre-trained checkpoints for both the lowercase and cased version of `BERT-Base` and `BERT-Large` from the paper.
- TensorFlow code for push-button replication of the most important fine-tuning experiments from the paper, including SQuAD, MultiNLI, and MRPC.

All of the code in this repository works out-of-the-box with CPU, GPU, and Cloud TPU.

## 预先训练的模型

We are releasing the `BERT-Base` and `BERT-Large` models from the paper.
`Uncased` means that the text has been lowercased before WordPiece tokenization, e.g., `John Smith` becomes `john smith`.
The `Uncased` model also strips out any accent markers.
`Cased` means that the true case and accent markers are preserved.
Typically, the `Uncased` model is better unless you know that case information is important for your task (e.g., Named Entity Recognition or Part-of-Speech tagging).

These models are all released under the same license as the source code (Apache 2.0).

For information about the Multilingual and Chinese model, see the [Multilingual README](https://github.com/google-research/bert/blob/master/multilingual.md).

**When using a cased model, make sure to pass `--do_lower=False` to the training scripts.
(Or pass `do_lower_case=False` directly to `FullTokenizer` if you're using your own script.)**

The links to the models are here (right-click, 'Save link as...' on the name):

- **[`BERT-Base, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)**: 12-layer, 768-hidden, 12-heads, 110M parameters
- **[`BERT-Large, Uncased`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip)**: 24-layer, 1024-hidden, 16-heads, 340M parameters
- **[`BERT-Base, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)**: 12-layer, 768-hidden, 12-heads , 110M parameters
- **[`BERT-Large, Cased`](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-24_H-1024_A-16.zip)**: 24-layer, 1024-hidden, 16-heads, 340M parameters
- **[`BERT-Base, Multilingual Cased (New, recommended)`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)**: 104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
- **[`BERT-Base, Multilingual Uncased (Orig, not recommended)`](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip) (Not recommended, use `Multilingual Cased` instead)**: 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters
- **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters

Each .zip file contains three items:

- A TensorFlow checkpoint (`bert_model.ckpt`) containing the pre-trained weights (which is actually 3 files).
- A vocab file (`vocab.txt`) to map WordPiece to word id.
- A config file (`bert_config.json`) which specifies the hyperparameters of the model.

## 使用 BERT 进行微调

**Important**: All results on the paper were fine-tuned on a single Cloud TPU,which has 64GB of RAM.
It is currently not possible to re-produce most of the `BERT-Large` results on the paper using a GPU with 12GB - 16GB of RAM, because the maximum batch size that can fit in memory is too small.
We are working on adding code to this repository which allows for much larger effective batch size
on the GPU.
See the section on [out-of-memory issues](#out-of-memory-issues) for more details.

This code was tested with TensorFlow 1.11.0.
It was tested with Python2 and Python3 (but more thoroughly with Python2, since this is what's used internally
in Google).

The fine-tuning examples which use `BERT-Base` should be able to run on a GPU that has at least 12GB of RAM using the hyperparameters given.

### 使用云 TPU 进行微调

Most of the examples below assumes that you will be running training/evaluation on your local machine, using a GPU like a Titan X or GTX 1080.

However, if you have access to a Cloud TPU that you want to train on, just add the following flags to `run_classifier.py` or `run_squad.py`:

```
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

Please see the [Google Cloud TPU tutorial](https://cloud.google.com/tpu/docs/tutorials/mnist)
for how to use Cloud TPUs.
Alternatively, you can use the Google Colab notebook "[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)".

On Cloud TPUs, the pretrained model and the output directory will need to be on Google Cloud Storage.
For example, if you have a bucket named `some_bucket`, you might use the following flags instead:

```
  --output_dir=gs://some_bucket/my_output_dir/
```

The unzipped pre-trained model files can also be found in the Google Cloud Storage folder `gs://bert_models/2018_10_18`.

For example:

```
export BERT_BASE_DIR=gs://bert_models/2018_10_18/uncased_L-12_H-768_A-12
```

### 句子（和句子对）分类任务

Before running this example you must download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory `$GLUE_DIR`.

Next, download the `BERT-Base` checkpoint and unzip it to some directory `$BERT_BASE_DIR`.

This example code fine-tunes `BERT-Base` on the Microsoft Research Paraphrase Corpus (MRPC) corpus, which only contains 3,600 examples and can fine-tune in a few minutes on most GPUs.

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue

python run_classifier.py \
  --task_name=MRPC \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=/tmp/mrpc_output/
```

You should see output like this:

```
***** Eval results *****
  eval_accuracy = 0.845588
  eval_loss = 0.505248
  global_step = 343
  loss = 0.505248
```

This means that the Dev set accuracy was 84.55%.
Small sets like MRPC have a high variance in the Dev set accuracy, even when starting from the same
pre-training checkpoint.
If you re-run multiple times (making sure to point to different `output_dir`), you should see results between 84% and 88%.

A few other pre-trained models are implemented off-the-shelf in `run_classifier.py`, so it should be straightforward to follow those examples to use BERT for any single-sentence or sentence-pair classification task.

Note: You might see a message `Running train on CPU`.
This really just means that it's running on something other than a Cloud TPU, which includes a GPU.

#### 从分类器预测

Once you have trained your classifier you can use it in inference mode by using the --do_predict=true command.
You need to have a file named test.tsv in the input folder.
Output will be created in file called test_results.tsv in the output folder.
Each line will contain output for each sample, columns are the class probabilities.

```shell
export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
export GLUE_DIR=/path/to/glue
export TRAINED_CLASSIFIER=/path/to/fine/tuned/classifier

python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MRPC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/tmp/mrpc_output/
```

### SQuAD 1.1

The Stanford Question Answering Dataset (SQuAD) is a popular question answering benchmark dataset.
BERT (at the time of the release) obtains state-of-the-art results on SQuAD with almost no task-specific network architecture modifications or data augmentation.
However, it does require semi-complex data pre-processing and post-processing to deal with (a) the variable-length nature of SQuAD context paragraphs, and (b) the character-level answer annotations which are used for SQuAD training.
This processing is implemented and documented in `run_squad.py`.

To run on SQuAD, you will first need to download the dataset.
The [SQuAD website](https://rajpurkar.github.io/SQuAD-explorer/) does not seem to link to the v1.1 datasets any longer, but the necessary files can be found here:

- [train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json)
- [dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json)
- [evaluate-v1.1.py](https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py)

Download these to some directory `$SQUAD_DIR`.

The state-of-the-art SQuAD results from the paper currently cannot be reproduced on a 12GB-16GB GPU due to memory constraints (in fact, even batch size 1 does not seem to fit on a 12GB GPU using `BERT-Large`).
However, a reasonably strong `BERT-Base` model can be trained on the GPU with these hyperparameters:

```shell
python run_squad.py \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=12 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=/tmp/squad_base/
```

The dev set predictions will be saved into a file called `predictions.json` in the `output_dir`:

```shell
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ./squad/predictions.json
```

Which should produce an output like this:

```shell
{"f1": 88.41249612335034, "exact_match": 81.2488174077578}
```

You should see a result similar to the 88.5% reported in the paper for `BERT-Base`.

If you have access to a Cloud TPU, you can train with `BERT-Large`.
Here is a set of hyperparameters (slightly different than the paper) which consistently obtain around 90.5%-91.0% F1 single-system trained only on SQuAD:

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v1.1.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v1.1.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME
```

For example, one random run with these parameters produces the following Dev scores:

```shell
{"f1": 90.87081895814865, "exact_match": 84.38978240302744}
```

If you fine-tune for one epoch on [TriviaQA](http://nlp.cs.washington.edu/triviaqa/) before this the results will be even better, but you will need to convert TriviaQA into the SQuAD json format.

### SQuAD 2.0

This model is also implemented and documented in `run_squad.py`.

To run on SQuAD 2.0, you will first need to download the dataset.
The necessary files can be found here:

- [train-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json)
- [dev-v2.0.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json)
- [evaluate-v2.0.py](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/)

Download these to some directory `$SQUAD_DIR`.

On Cloud TPU you can run with BERT-Large as follows:

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=True \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True
```

We assume you have copied everything from the output directory to a local directory called ./squad/.
The initial dev set predictions will be at ./squad/predictions.json and the differences between the score of no answer ("") and the best non-null answer for each question will be in the file ./squad/null_odds.json

Run this script to tune a threshold for predicting null versus non-null answers:

python $SQUAD_DIR/evaluate-v2.0.py $SQUAD_DIR/dev-v2.0.json ./squad/predictions.json --na-prob-file ./squad/null_odds.json

Assume the script outputs "best_f1_thresh" THRESH.
(Typical values are between -1.0 and -5.0).
You can now re-run the model to generate predictions with the derived threshold or alternatively you can extract the appropriate answers from ./squad/nbest_predictions.json.

```shell
python run_squad.py \
  --vocab_file=$BERT_LARGE_DIR/vocab.txt \
  --bert_config_file=$BERT_LARGE_DIR/bert_config.json \
  --init_checkpoint=$BERT_LARGE_DIR/bert_model.ckpt \
  --do_train=False \
  --train_file=$SQUAD_DIR/train-v2.0.json \
  --do_predict=True \
  --predict_file=$SQUAD_DIR/dev-v2.0.json \
  --train_batch_size=24 \
  --learning_rate=3e-5 \
  --num_train_epochs=2.0 \
  --max_seq_length=384 \
  --doc_stride=128 \
  --output_dir=gs://some_bucket/squad_large/ \
  --use_tpu=True \
  --tpu_name=$TPU_NAME \
  --version_2_with_negative=True \
  --null_score_diff_threshold=$THRESH
```

### 内存不足的问题

All experiments in the paper were fine-tuned on a Cloud TPU, which has 64GB of device RAM.
Therefore, when using a GPU with 12GB - 16GB of RAM, you are likely to encounter out-of-memory issues if you use the same hyperparameters described in the paper.

The factors that affect memory usage are:

- **`max_seq_length`**: The released models were trained with sequence lengths up to 512, but you can fine-tune with a shorter max sequence length to save substantial memory.
  This is controlled by the `max_seq_length` flag in our example code.

- **`train_batch_size`**: The memory usage is also directly proportional to the batch size.

- **Model type, `BERT-Base` vs. `BERT-Large`**: The `BERT-Large` model requires significantly more memory than `BERT-Base`.

- **Optimizer**: The default optimizer for BERT is Adam, which requires a lot of extra memory to store the `m` and `v` vectors.
  Switching to a more memory efficient optimizer can reduce memory usage, but can also affect the
  results.
  We have not experimented with other optimizers for fine-tuning.

Using the default training scripts (`run_classifier.py` and `run_squad.py`), we benchmarked the maximum batch size on single Titan X GPU (12GB RAM) with TensorFlow 1.11.0:

| System       | Seq Length | Max Batch Size |
| ------------ | ---------- | -------------- |
| `BERT-Base`  | 64         | 64             |
| ...          | 128        | 32             |
| ...          | 256        | 16             |
| ...          | 320        | 14             |
| ...          | 384        | 12             |
| ...          | 512        | 6              |
| `BERT-Large` | 64         | 12             |
| ...          | 128        | 6              |
| ...          | 256        | 2              |
| ...          | 320        | 1              |
| ...          | 384        | 0              |
| ...          | 512        | 0              |

Unfortunately, these max batch sizes for `BERT-Large` are so small that they will actually harm the model accuracy, regardless of the learning rate used.
We are working on adding code to this repository which will allow much larger effective batch sizes to be used on the GPU.
The code will be based on one (or both) of the following techniques:

- **梯度积累**: The samples in a minibatch are typically independent with respect to gradient computation (excluding batch normalization, which is not used here).
  This means that the gradients of multiple smaller minibatches can be accumulated before performing the weight
  update, and this will be exactly equivalent to a single larger update.

- [**梯度检查点**](https://github.com/openai/gradient-checkpointing):
  The major use of GPU/TPU memory during DNN training is caching the intermediate activations in the forward pass that are necessary for efficient computation in the backward pass.
  "Gradient checkpointing" trades memory for compute time by re-computing the activations in an intelligent way.

**However, this is not implemented in the current release.**

## 使用 BERT 提取固定的特征向量 (像 ELMo)

In certain cases, rather than fine-tuning the entire pre-trained model end-to-end, it can be beneficial to obtained _pre-trained contextual embeddings_, which are fixed contextual representations of each input token
generated from the hidden layers of the pre-trained model.
This should also mitigate most of the out-of-memory issues.

As an example, we include the script `extract_features.py` which can be used like this:

```shell
# Sentence A and Sentence B are separated by the ||| delimiter for sentence
# pair tasks like question answering and entailment.
# For single sentence inputs, put one sentence per line and DON'T use the
# delimiter.
echo 'Who was Jim Henson ? ||| Jim Henson was a puppeteer' > /tmp/input.txt

python extract_features.py \
  --input_file=/tmp/input.txt \
  --output_file=/tmp/output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8
```

This will create a JSON file (one line per line of input) containing the BERT activations from each Transformer layer specified by `layers` (-1 is the final hidden layer of the Transformer, etc.)

Note that this script will produce very large output files (by default, around 15kb for every input token).

If you need to maintain alignment between the original and tokenized words (for projecting training labels), see the [Tokenization](#tokenization) section below.

**Note:** You may see a message like `Could not find trained model in model_dir: /tmp/tmpuB5g5c, running initialization to predict.` This message is expected, it just means that we are using the `init_from_checkpoint()` API rather than the saved model API.
If you don't specify a checkpoint or specify an invalid checkpoint, this script will complain.

## 符号化

For sentence-level tasks (or sentence-pair) tasks, tokenization is very simple.
Just follow the example code in `run_classifier.py` and `extract_features.py`.
The basic procedure for sentence-level tasks is:

1. Instantiate an instance of `tokenizer = tokenization.FullTokenizer`
2. Tokenize the raw text with `tokens = tokenizer.tokenize(raw_text)`.
3. Truncate to the maximum sequence length.(You can use up to 512, but you probably want to use shorter if possible for memory and speed reasons.)
4. Add the `[CLS]` and `[SEP]` tokens in the right place.

Word-level and span-level tasks (e.g., SQuAD and NER) are more complex, since you need to maintain alignment between your input text and output text so that you can project your training labels.
SQuAD is a particularly complex example because the input labels are _character_-based, and SQuAD paragraphs are often longer than our maximum sequence length.
See the code in `run_squad.py` to show how we handle this.

Before we describe the general recipe for handling word-level tasks, it's important to understand what exactly our tokenizer is doing.
It has three main steps:

1. **Text normalization**: Convert all whitespace characters to spaces, and (for the `Uncased` model) lowercase the input and strip out accent markers. E.g., `John Johanson's, → john johanson's,`.
2. **Punctuation splitting**: Split _all_ punctuation characters on both sides (i.e., add whitespace around all punctuation characters). Punctuation characters are defined as (a) Anything with a `P*` Unicode class, (b) any
   non-letter/number/space ASCII character (e.g., characters like `$` which are technically not punctuation). E.g., `john johanson's, → john johanson ' s ,`
3. **WordPiece tokenization**: Apply whitespace tokenization to the output of the above procedure, and apply
   [WordPiece](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder.py) tokenization to each token separately.
   (Our implementation is directly based on the one from `tensor2tensor`, which is linked).
   E.g., `john johanson ' s , → john johan ##son ' s ,`

The advantage of this scheme is that it is "compatible" with most existing English tokenizers.
For example, imagine that you have a part-of-speech tagging task which looks like this:

```
Input:  John Johanson 's   house
Labels: NNP  NNP      POS NN
```

The tokenized output will look like this:

```
Tokens: john johan ##son ' s house
```

Crucially, this would be the same output as if the raw text were `John Johanson's house` (with no space before the `'s`).

If you have a pre-tokenized representation with word-level annotations, you can simply tokenize each input word independently, and deterministically maintain an original-to-tokenized alignment:

```python
### Input
orig_tokens = ["John", "Johanson", "'s",  "house"]
labels      = ["NNP",  "NNP",      "POS", "NN"]

### Output
bert_tokens = []

# Token map will be an int -> int mapping between the `orig_tokens` index and
# the `bert_tokens` index.
orig_to_tok_map = []

tokenizer = tokenization.FullTokenizer(
    vocab_file=vocab_file, do_lower_case=True)

bert_tokens.append("[CLS]")
for orig_token in orig_tokens:
  orig_to_tok_map.append(len(bert_tokens))
  bert_tokens.extend(tokenizer.tokenize(orig_token))
bert_tokens.append("[SEP]")

# bert_tokens == ["[CLS]", "john", "johan", "##son", "'", "s", "house", "[SEP]"]
# orig_to_tok_map == [1, 2, 4, 6]
```

Now `orig_to_tok_map` can be used to project `labels` to the tokenized representation.

There are common English tokenization schemes which will cause a slight mismatch between how BERT was pre-trained.
For example, if your input tokenization splits off contractions like `do n't`, this will cause a mismatch.
If it is possible to do so, you should pre-process your data to convert these back to raw-looking text, but if it's not possible, this mismatch is likely not a big deal.

## 使用 BERT 进行预训练

我们正在发布代码，在任意文本语料库上做“蒙面 LM”和“下一句话预测”。
请注意，这不是用于论文的确切代码（原始代码是用 C ++编写的，并且有一些额外的复杂性），但是此代码确实生成了本文所述的预训练数据。

Here's how to run the data generation.
The input is a plain text file, with one sentence per line.
(It is important that these be actual sentences for the "next sentence prediction" task).
Documents are delimited by empty lines.
The output is a set of `tf.train.Example`s serialized into `TFRecord` file format.

You can perform sentence segmentation with an off-the-shelf NLP toolkit such as [spaCy](https://spacy.io/).
The `create_pretraining_data.py` script will concatenate segments until they reach the maximum sequence length to minimize computational waste from padding (see the script for more details).
However, you may want to intentionally add a slight amount of noise to your input data (e.g., randomly truncate 2% of input segments) to make it more robust to non-sentential input during fine-tuning.

This script stores all of the examples for the entire input file in memory, so for large data files you should shard the input file and call the script multiple times.
(You can pass in a file glob to `run_pretraining.py`, e.g., `tf_examples.tf_record*`.)

The `max_predictions_per_seq` is the maximum number of masked LM predictions per sequence.
You should set this to around `max_seq_length` \* `masked_lm_prob` (the script doesn't do that automatically because the exact value needs to be passed to both scripts).

```shell
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
```

Here's how to run the pre-training.
Do not include `init_checkpoint` if you are pre-training from scratch.
The model configuration (including vocab size) is specified in `bert_config_file`.
This demo code only pre-trains for a small number of steps (20), but in practice you will probably want to set `num_train_steps` to 10000 steps or more.
The `max_seq_length` and `max_predictions_per_seq` parameters passed to `run_pretraining.py` must be the same as `create_pretraining_data.py`.

```shell
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
```

This will produce an output like this:

```
***** Eval results *****
  global_step = 20
  loss = 0.0979674
  masked_lm_accuracy = 0.985479
  masked_lm_loss = 0.0979328
  next_sentence_accuracy = 1.0
  next_sentence_loss = 3.45724e-05
```

Note that since our `sample_text.txt` file is very small, this example training will overfit that data in only a few steps and produce unrealistically high accuracy numbers.

### 预训练提示和警告

- **If using your own vocabulary, make sure to change `vocab_size` in `bert_config.json`.
  If you use a larger vocabulary without changing this, you will likely get NaNs when training on GPU or TPU due to unchecked out-of-bounds access.**
- If your task has a large domain-specific corpus available (e.g., "movie reviews" or "scientific papers"), it will likely be beneficial to run additional steps of pre-training on your corpus, starting from the BERT checkpoint.
- The learning rate we used in the paper was 1e-4.
  However, if you are doing additional steps of pre-training starting from an existing BERT checkpoint, you should use a smaller learning rate (e.g., 2e-5).
- Current BERT models are English-only, but we do plan to release a multilingual model which has been pre-trained on a lot of languages in the near future (hopefully by the end of November 2018).
- Longer sequences are disproportionately expensive because attention is quadratic to the sequence length.
  In other words, a batch of 64 sequences of length 512 is much more expensive than a batch of 256 sequences of
  length 128.
  The fully-connected/convolutional cost is the same, but the attention cost is far greater for the 512-length sequences.
  Therefore, one good recipe is to pre-train for, say, 90,000 steps with a sequence length of 128 and then for 10,000 additional steps with a sequence length of 512.
  The very long sequences are mostly needed to learn positional embeddings, which can be learned fairly quickly.
  Note that this does require generating the data twice with different values of `max_seq_length`.
- If you are pre-training from scratch, be prepared that pre-training is computationally expensive, especially on GPUs.
  If you are pre-training from scratch, our recommended recipe is to pre-train a `BERT-Base` on a single
  [preemptible Cloud TPU v2](https://cloud.google.com/tpu/docs/pricing), which takes about 2 weeks at a cost of about \$500 USD (based on the pricing in October 2018).
  You will have to scale down the batch size when only training on a single Cloud TPU, compared to what was used in the paper.
  It is recommended to use the largest batch size that fits into TPU memory.

### 预训练数据

We will **not** be able to release the pre-processed datasets used in the paper.
For Wikipedia, the recommended pre-processing is to download
[the latest dump](https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2), extract the text with [`WikiExtractor.py`](https://github.com/attardi/wikiextractor), and then apply any necessary cleanup to convert it into plain text.

Unfortunately the researchers who collected the [BookCorpus](http://yknzhu.wixsite.com/mbweb) no longer have it available for public download.
The [Project Guttenberg Dataset](https://web.eecs.umich.edu/~lahiri/gutenberg_dataset.html) is a somewhat smaller (200M word) collection of older books that are public domain.

[Common Crawl](http://commoncrawl.org/) is another very large collection of text, but you will likely have to do substantial pre-processing and cleanup to extract a usable corpus for pre-training BERT.

### 学习一个新的 WordPiece 词汇表

This repository does not include code for _learning_ a new WordPiece vocabulary.
The reason is that the code used in the paper was implemented in C++ with dependencies on Google's internal libraries.
For English, it is almost always better to just start with our vocabulary and pre-trained models.
For learning vocabularies of other languages, there are a number of open source options available.
However, keep in mind that these are not compatible with our `tokenization.py` library:

- [Google's SentencePiece library](https://github.com/google/sentencepiece)

- [tensor2tensor's WordPiece generation script](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/text_encoder_build_subword.py)

- [Rico Sennrich's Byte Pair Encoding library](https://github.com/rsennrich/subword-nmt)

## 在 Colab 中使用 BERT

If you want to use BERT with [Colab](https://colab.research.google.com), you can get started with the notebook
"[BERT FineTuning with Cloud TPUs](https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/bert_finetuning_with_cloud_tpus.ipynb)".
**At the time of this writing (October 31st, 2018), Colab users can access a Cloud TPU completely for free.** Note: One per user, availability limited, requires a Google Cloud Platform account with storage (although storage may be purchased with free credit for signing up with GCP), and this capability may not longer be available in the future.
Click on the BERT Colab that was just linked for more information.

## 常问问题

### 此代码是否与 Cloud TPU 兼容？ GPU 怎么样？

是的，此存储库中的所有代码都与 CPU，GPU 和云 TPU 一起开箱即用。
但是，GPU 培训仅适用于单 GPU。

### 我得到了内存错误，出了什么问题？

有关更多信息，请参阅[内存不足问题](#内存不足问题)部分。

### 有 PyTorch 版本吗？

没有正式的 PyTorch 实现。
然而，HuggingFace 的 NLP 研究人员制作了[PyTorch 版本的 BERT](https://github.com/huggingface/pytorch-pretrained-BERT)，它与我们预先训练好的检查点兼容，并能够重现我们的结果。
我们没有参与 PyTorch 实现的创建或维护，因此请向该存储库的作者提出任何问题。

### 有 Chainer 版本吗？

没有正式的 Chainer 实施。
However, Sosuke Kobayashi made a [Chainer version of BERT available](https://github.com/soskek/bert-chainer) which is compatible with our pre-trained checkpoints and is able to reproduce our results.
We were not involved in the creation or maintenance of the Chainer implementation so please direct any questions towards the authors of that repository.

### 是否会发布其他语言的模型？

是的，我们计划在不久的将来发布多语言 BERT 模型。
我们无法确定将包含哪些语言，但它可能是一个单一的模型，其中包括具有大小维基百科的大多数语言。

### 是否会发布比“BERT-Large”更大的模型？

到目前为止，我们还没有尝试过比“BERT-Large”更大的训练。
如果我们能够获得重大改进，我们可能会发布更大的模型。

### 该库发布的许可证是什么？

所有代码 _and_ 模型都是在 Apache 2.0 许可下发布的。
有关更多信息，请参阅`LICENSE`文件。

### 我怎么引用 BERT？

现在，引用[Arxiv 论文](https://arxiv.org/abs/1810.04805):

```
@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}
```

如果我们将论文提交给会议或期刊，我们将更新 BibTeX。

## 放弃

这不是 Google 的官方产品。

## 联系信息

有关使用 BERT 的帮助或问题，请提交 GitHub 问题。

有关 BERT 的个人通信，请联系 Jacob Devlin（`jacobdevlin@google.com`），Ming-Wei Chang（`mingweichang@google.com`）或 Kenton Lee（`kentonl@google.com`）。
