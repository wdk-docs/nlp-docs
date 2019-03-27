# 使用 BERT 进行微调

!!! important ""

    All results on the paper were fine-tuned on a single Cloud TPU,which has 64GB of RAM.
    It is currently not possible to re-produce most of the `BERT-Large` results on the paper using a GPU with 12GB - 16GB of RAM, because the maximum batch size that can fit in memory is too small.
    We are working on adding code to this repository which allows for much larger effective batch size
    on the GPU.
    See the section on [out-of-memory issues](#out-of-memory-issues) for more details.

This code was tested with TensorFlow 1.11.0.
It was tested with Python2 and Python3 (but more thoroughly with Python2, since this is what's used internally
in Google).

The fine-tuning examples which use `BERT-Base` should be able to run on a GPU that has at least 12GB of RAM using the hyperparameters given.

## 使用云 TPU 进行微调

Most of the examples below assumes that you will be running training/evaluation on your local machine, using a GPU like a Titan X or GTX 1080.

However, if you have access to a Cloud TPU that you want to train on, just add the following flags to `run_classifier.py` or `run_squad.py`:

```sh
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

## 句子（和句子对）分类任务

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

### 从分类器预测

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

## SQuAD 1.1

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

## SQuAD 2.0

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

## 内存不足的问题

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