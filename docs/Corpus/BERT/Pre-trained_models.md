# 预先训练的模型

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
