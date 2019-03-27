# 常问问题

??? faq "此代码是否与 Cloud TPU 兼容？ GPU 怎么样？"

    是的，此存储库中的所有代码都与 CPU，GPU 和云 TPU 一起开箱即用。
    但是，GPU 培训仅适用于单 GPU。

??? faq "我得到了内存错误，出了什么问题？"

    有关更多信息，请参阅[内存不足问题](#内存不足问题)部分。

??? faq "有 PyTorch 版本吗？"

    没有正式的 PyTorch 实现。
    然而，HuggingFace 的 NLP 研究人员制作了[PyTorch 版本的 BERT](https://github.com/huggingface/pytorch-pretrained-BERT)，它与我们预先训练好的检查点兼容，并能够重现我们的结果。
    我们没有参与 PyTorch 实现的创建或维护，因此请向该存储库的作者提出任何问题。

??? faq "有 Chainer 版本吗？"

    没有正式的 Chainer 实施。
    然而，Sosuke Kobayashi制作了[Biner版本的BERT](https://github.com/soskek/bert-chainer) ，它与我们预先训练好的检查点兼容，并能够重现我们的结果。
    我们没有参与Chainer实施的创建或维护，因此请向该存储库的作者提出任何问题。

??? faq "是否会发布其他语言的模型？"

    是的，我们计划在不久的将来发布多语言 BERT 模型。
    我们无法确定将包含哪些语言，但它可能是一个单一的模型，其中包括具有大小维基百科的大多数语言。

??? faq "是否会发布比“BERT-Large”更大的模型？"

    到目前为止，我们还没有尝试过比“BERT-Large”更大的训练。
    如果我们能够获得重大改进，我们可能会发布更大的模型。

??? faq "该库发布的许可证是什么？"

    所有代码_和_模型都是在 `Apache 2.0` 许可下发布的。
    有关更多信息，请参阅`LICENSE`文件。

??? faq "我怎么引用 BERT？"

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
