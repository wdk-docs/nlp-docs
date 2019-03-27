# 使用 BERT 提取固定的特征向量 (像 ELMo)

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
