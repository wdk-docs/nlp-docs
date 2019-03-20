# 北大-pkuseg

一个多领域中文分词工具包

[pkuseg](https://github.com/lancopku/pkuseg-python) 简单易用，支持细分领域分词，有效提升了分词准确度。

## 主要亮点

pkuseg 具有如下几个特点：

1. 多领域分词。不同于以往的通用中文分词工具，此工具包同时致力于为不同领域的数据提供个性化的预训练模型。根据待分词文本的领域特点，用户可以自由地选择不同的模型。 我们目前支持了新闻领域，网络领域，医药领域，旅游领域，以及混合领域的分词预训练模型。在使用中，如果用户明确待分词的领域，可加载对应的模型进行分词。如果用户无法确定具体领域，推荐使用在混合领域上训练的通用模型。各领域分词样例可参考 [**example.txt**](https://github.com/lancopku/pkuseg-python/blob/master/example.txt)。
2. 更高的分词准确率。相比于其他的分词工具包，当使用相同的训练数据和测试数据，pkuseg 可以取得更高的分词准确率。
3. 支持用户自训练模型。支持用户使用全新的标注数据进行训练。
4. 支持词性标注。

## 编译和安装

- 目前**仅支持 python3**
- 新版本发布：2019-1-23

      - 修改了词典处理方法，扩充了词典，分词效果有提升
      - **效率进行了优化，测试速度较之前版本提升 9 倍左右**
      - 增加了在大规模混合数据集训练的通用模型，并将其设为默认使用模型

- 新版本发布：2019-1-30

      - 支持 fine-tune 训练（从预加载的模型继续训练），支持设定训练轮数

- 新版本发布：2019-2-20

      - **支持词性标注，增加了医疗、旅游细领域模型**

- **为了获得好的效果和速度，强烈建议大家通过 pip install 更新到目前的最新版本**

安装

1.  通过 PyPI 安装(自带模型文件)：

    ```sh
    pip3 install pkuseg
    # 之后通过import pkuseg来引用
    # 建议更新到最新版本以获得更好的开箱体验：
    pip3 install -U pkuseg
    ```

2.  如果 PyPI 官方源下载速度不理想，建议使用镜像源，比如：

    ```sh
    #初次安装：
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pkuseg
    #更新：
    pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pkuseg
    ```

3.  如果不使用 pip 安装方式，选择从 GitHub 下载，可运行以下命令安装：

    ```sh
    python setup.py build_ext -i
    ```

    GitHub 的代码并不包括预训练模型，因此需要用户自行下载或训练模型，预训练模型可详见[release](https://github.com/lancopku/pkuseg-python/releases)。使用时需设定"model_name"为模型文件。

注意：**安装方式 1 和 2 目前仅支持 linux(ubuntu)、mac、windows 64 位的 python3 版本**。如果非以上系统，请使用安装方式 3 进行本地编译安装。

## 使用方式

### 代码示例

以下代码示例适用于 python 交互式环境。

代码示例 1：使用默认配置进行分词（**如果用户无法确定分词领域，推荐使用默认模型分词**）

```python3
import pkuseg

seg = pkuseg.pkuseg()           # 以默认配置加载模型
text = seg.cut('我爱北京天安门')  # 进行分词
print(text)
```

代码示例 2：细领域分词（**如果用户明确分词领域，推荐使用细领域模型分词**）

```python3
import pkuseg

seg = pkuseg.pkuseg(model_name='medicine')  # 程序会自动下载所对应的细领域模型
text = seg.cut('我爱北京天安门')              # 进行分词
print(text)
```

代码示例 3：分词同时进行词性标注，各词性标签的详细含义可参考 [tags.txt](https://github.com/lancopku/pkuseg-python/blob/master/tags.txt)

```python3
import pkuseg

seg = pkuseg.pkuseg(postag=True)  # 开启词性标注功能
text = seg.cut('我爱北京天安门')    # 进行分词和词性标注
print(text)
```

代码示例 4：对文件分词

```python3
import pkuseg

# 对input.txt的文件分词输出到output.txt中
# 开20个进程
pkuseg.test('input.txt', 'output.txt', nthread=20)
```

代码示例 5：额外使用用户自定义词典

```python3
import pkuseg

seg = pkuseg.pkuseg(user_dict='my_dict.txt')  # 给定用户词典为当前目录下的"my_dict.txt"
text = seg.cut('我爱北京天安门')                # 进行分词
print(text)
```

代码示例 6：使用自训练模型分词（以 CTB8 模型为例）

```python3
import pkuseg

seg = pkuseg.pkuseg(model_name='./ctb8')  # 假设用户已经下载好了ctb8的模型并放在了'./ctb8'目录下，通过设置model_name加载该模型
text = seg.cut('我爱北京天安门')            # 进行分词
print(text)
```

代码示例 7：训练新模型 （模型随机初始化）

```python3
import pkuseg

# 训练文件为'msr_training.utf8'
# 测试文件为'msr_test_gold.utf8'
# 训练好的模型存到'./models'目录下
# 训练模式下会保存最后一轮模型作为最终模型
# 目前仅支持utf-8编码，训练集和测试集要求所有单词以单个或多个空格分开
pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', './models')
```

代码示例 8：fine-tune 训练（从预加载的模型继续训练）

```python3
import pkuseg

# 训练文件为'train.txt'
# 测试文件为'test.txt'
# 加载'./pretrained'目录下的模型，训练好的模型保存在'./models'，训练10轮
pkuseg.train('train.txt', 'test.txt', './models', train_iter=10, init_model='./pretrained')
```

### 参数说明

模型配置

```
pkuseg.pkuseg(model_name = "default", user_dict = "default", postag = False)
	model_name		模型路径。
			        "default"，默认参数，表示使用我们预训练好的混合领域模型(仅对pip下载的用户)。
				"news", 使用新闻领域模型。
				"web", 使用网络领域模型。
				"medicine", 使用医药领域模型。
				"tourism", 使用旅游领域模型。
			        model_path, 从用户指定路径加载模型。
	user_dict		设置用户词典。
				"default", 默认参数，使用我们提供的词典。
				None, 不使用词典。
				dict_path, 在使用默认词典的同时会额外使用用户自定义词典，可以填自己的用户词典的路径，词典格式为一行一个词。
	postag		        是否进行词性分析。
				False, 默认参数，只进行分词，不进行词性标注。
				True, 会在分词的同时进行词性标注。
```

对文件进行分词

```
pkuseg.test(readFile, outputFile, model_name = "default", user_dict = "default", postag = False, nthread = 10)
	readFile		输入文件路径。
	outputFile		输出文件路径。
	model_name		模型路径。同pkuseg.pkuseg
	user_dict		设置用户词典。同pkuseg.pkuseg
	postag			设置是否开启词性分析功能。同pkuseg.pkuseg
	nthread			测试时开的进程数。
```

模型训练

```
pkuseg.train(trainFile, testFile, savedir, train_iter = 20, init_model = None)
	trainFile		训练文件路径。
	testFile		测试文件路径。
	savedir			训练模型的保存路径。
	train_iter		训练轮数。
	init_model		初始化模型，默认为None表示使用默认初始化，用户可以填自己想要初始化的模型的路径如init_model='./models/'。
```

### 多进程分词

当将以上代码示例置于文件中运行时，如涉及多进程功能，请务必使用`if __name__ == '__main__'`保护全局语句，如：
mp.py 文件

```python3
import pkuseg

if __name__ == '__main__':
    pkuseg.test('input.txt', 'output.txt', nthread=20)
    pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', './models', nthread=20)
```

运行

```
python3 mp.py
```

详见[无法使用多进程分词和训练功能，提示 RuntimeError 和 BrokenPipeError](https://github.com/lancopku/pkuseg-python/wiki#3-无法使用多进程分词和训练功能提示runtimeerror和brokenpipeerror)。

**在 Windows 平台上，请当文件足够大时再使用多进程分词功能**，详见[关于多进程速度问题](https://github.com/lancopku/pkuseg-python/wiki#9-关于多进程速度问题)。

## 预训练模型

从 pip 安装的用户在使用细领域分词功能时，只需要设置 model_name 字段为对应的领域即可，会自动下载对应的细领域模型。

从 github 下载的用户则需要自己下载对应的预训练模型，并设置 model_name 字段为预训练模型路径。预训练模型可以在[release](https://github.com/lancopku/pkuseg-python/releases)部分下载。以下是对预训练模型的说明：

- **news**: 在 MSRA（新闻语料）上训练的模型。

- **web**: 在微博（网络文本语料）上训练的模型。

- **medicine**: 在医药领域上训练的模型。

- **tourism**: 在旅游领域上训练的模型。

- **mixed**: 混合数据集训练的通用模型。随 pip 包附带的是此模型。

欢迎更多用户可以分享自己训练好的细分领域模型。

## 各类分词工具包的性能对比

我们选择 jieba、THULAC 等国内代表分词工具包与 pkuseg 做性能比较。

考虑到 jieba 分词和 THULAC 工具包等并没有提供细领域的预训练模型，为了便于比较，我们重新使用它们提供的训练接口在细领域的数据集上进行训练，用训练得到的模型进行中文分词。

我们选择 Linux 作为测试环境，在新闻数据(MSRA)、混合型文本(CTB8)、网络文本(WEIBO)数据上对不同工具包进行了准确率测试。我们使用了第二届国际汉语分词评测比赛提供的分词评价脚本。其中 MSRA 与 WEIBO 使用标准训练集测试集划分，CTB8 采用随机划分。对于不同的分词工具包，训练测试数据的划分都是一致的；**即所有的分词工具包都在相同的训练集上训练，在相同的测试集上测试**。对于所有数据集，pkuseg 使用了不使用词典的训练和测试接口。以下是 pkuseg 训练和测试代码示例:

```py
pkuseg.train('msr_training.utf8', 'msr_test_gold.utf8', './models')
pkuseg.test('msr_test.raw', 'output.txt', user_dict=None)
```

### 细领域训练及测试结果

以下是在不同数据集上的对比结果：

| MSRA   | Precision | Recall |   F-score |
| :----- | --------: | -----: | --------: |
| jieba  |     87.01 |  89.88 |     88.42 |
| THULAC |     95.60 |  95.91 |     95.71 |
| pkuseg |     96.94 |  96.81 | **96.88** |

| WEIBO  | Precision | Recall |   F-score |
| :----- | --------: | -----: | --------: |
| jieba  |     87.79 |  87.54 |     87.66 |
| THULAC |     93.40 |  92.40 |     92.87 |
| pkuseg |     93.78 |  94.65 | **94.21** |

### 默认模型在不同领域的测试效果

考虑到很多用户在尝试分词工具的时候，大多数时候会使用工具包自带模型测试。为了直接对比“初始”性能，我们也比较了各个工具包的默认模型在不同领域的测试效果。请注意，这样的比较只是为了说明默认情况下的效果，并不一定是公平的。

| Default | MSRA  | CTB8  |  PKU  | WEIBO | All Average |
| ------- | :---: | :---: | :---: | :---: | :---------: |
| jieba   | 81.45 | 79.58 | 81.83 | 83.56 |    81.61    |
| THULAC  | 85.55 | 87.84 | 92.29 | 86.65 |    88.08    |
| pkuseg  | 87.29 | 91.77 | 92.68 | 93.43 |  **91.29**  |

其中，`All Average`显示的是在所有测试集上 F-score 的平均。

更多详细比较可参见[和现有工具包的比较](https://github.com/lancopku/pkuseg-python/wiki/%E5%92%8C%E7%8E%B0%E6%9C%89%E5%B7%A5%E5%85%B7%E5%8C%85%E7%9A%84%E6%AF%94%E8%BE%83)。

## 版本历史

- v0.0.11(2019-01-09)
  - 修订默认配置：CTB8 作为默认模型，不使用词典
- v0.0.14(2019-01-23)
  - 修改了词典处理方法，扩充了词典，分词效果有提升
  - 效率进行了优化，测试速度较之前版本提升 9 倍左右
  - 增加了在大规模混合数据集训练的通用模型，并将其设为默认使用模型
- v0.0.15(2019-01-30)
  - 支持 fine-tune 训练（从预加载的模型继续训练），支持设定训练轮数
- v0.0.18(2019-02-20)
  - 支持词性标注，增加了医疗、旅游两个细领域模型

## 开源协议

1. 本代码采用 MIT 许可证。
2. 欢迎对该工具包提出任何宝贵意见和建议，请发邮件至 jingjingxu@pku.edu.cn。

## 相关论文

该代码包主要基于以下科研论文，如使用了本工具，请引用以下论文：

- Xu Sun, Houfeng Wang, Wenjie Li. [Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection](http://www.aclweb.org/anthology/P12-1027). ACL. 253–262. 2012

```
@inproceedings{DBLP:conf/acl/SunWL12,
author = {Xu Sun and Houfeng Wang and Wenjie Li},
title = {Fast Online Training with Frequency-Adaptive Learning Rates for Chinese Word Segmentation and New Word Detection},
booktitle = {The 50th Annual Meeting of the Association for Computational Linguistics, Proceedings of the Conference, July 8-14, 2012, Jeju Island, Korea- Volume 1: Long Papers},
pages = {253--262},
year = {2012}}
```

## 常见问题及解答

1. [为什么要发布 pkuseg？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#1-为什么要发布pkuseg)
2. [pkuseg 使用了哪些技术？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#2-pkuseg使用了哪些技术)
3. [无法使用多进程分词和训练功能，提示 RuntimeError 和 BrokenPipeError。](https://github.com/lancopku/pkuseg-python/wiki/FAQ#3-无法使用多进程分词和训练功能提示runtimeerror和brokenpipeerror)
4. [是如何跟其它工具包在细领域数据上进行比较的？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#4-是如何跟其它工具包在细领域数据上进行比较的)
5. [在黑盒测试集上进行比较的话，效果如何？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#5-在黑盒测试集上进行比较的话效果如何)
6. [如果我不了解待分词语料的所属领域呢？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#6-如果我不了解待分词语料的所属领域呢)
7. [如何看待在一些特定样例上的分词结果？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#7-如何看待在一些特定样例上的分词结果)
8. [关于运行速度问题？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#8-关于运行速度问题)
9. [关于多进程速度问题？](https://github.com/lancopku/pkuseg-python/wiki/FAQ#9-关于多进程速度问题)

## 作者

Ruixuan Luo （罗睿轩）, Jingjing Xu（许晶晶）, Xuancheng Ren（任宣丞）, Yi Zhang（张艺）, Bingzhen Wei（位冰镇）， Xu Sun （孙栩）
