清华-THULAC
===========

   `THULAC <https://github.com/thunlp/THULAC-Python>`__\ 一个高效的中文词法分析工具包

目录

-  软件简介
-  在线演示
-  编译和安装
-  使用方式
-  与代表性分词软件的性能对比
-  词性标记集
-  THULAC 的不同配置
-  获取链接
-  注意事项
-  历史
-  开源协议
-  相关论文
-  作者
-  常见问题
-  致谢

软件简介
--------

THULAC（THU Lexical Analyzer for
Chinese）由清华大学自然语言处理与社会人文计算实验室研制推出的一套中文词法分析工具包，具有中文分词和词性标注功能。THULAC
具有如下几个特点：

能力强。利用我们集成的目前世界上规模最大的人工分词和词性标注中文语料库（约含
5800 万字）训练而成，模型标注能力强大。

准确率高。该工具包在标准数据集 Chinese Treebank（CTB5）上分词的 F1
值可达 97.3％，词性标注的 F1 值可达到
92.9％，与该数据集上最好方法效果相当。

速度较快。同时进行分词和词性标注速度为 300KB/s，每秒可处理约 15
万字。只进行分词速度可达到 1.3MB/s。

在线演示
--------

THULAC 在线演示平台 thulac.thunlp.org/demo

编译和安装
----------

C++版

在当前路径下运行 make 会在当前目录下得到 thulac 和 train_c （thulac
需要模型的支持，需要将下载的模型放到当前目录下） java 版

可直接按照分词程序命令格式运行可执行的 jar 包 自行编译需要安装 Gradle,
然后在项目根目录执行 gradle build, 生成文件在 build/libs 下 （thulac
需要模型的支持，需要将下载的模型放到当前目录下） python 版（兼容
python2.x 和 python3.x）

源代码下载

将 thulac 文件放到目录下，通过 import thulac 来引用 thulac
需要模型的支持，需要将下载的模型放到 thulac 目录下。 pip 下载

sudo pip install thulac 通过 import thulac 来引用

使用方式
--------

1.分词和词性标注程序 1.1.命令格式 C++版(接口调用参见 1.5)

./thulac [-t2s][-seg_only] [-deli delimeter][-user userword.txt]
从命令行输入输出 ./thulac [-t2s][-seg_only] [-deli delimeter][-user
userword.txt] outputfile 利用重定向从文本文件输入输出（注意均为 UTF8
文本） java 版

java -jar THULAC_lite_java_run.jar [-t2s][-seg_only] [-deli
delimeter][-user userword.txt] 从命令行输入输出 java -jar
THULAC_lite_java_run.jar [-t2s][-seg_only] [-deli delimeter][-user
userword.txt] -input input_file -output output_file
从文本文件输入输出（注意均为 UTF8 文本） python 版（兼容 python2.x 和
python3.x）

通过 python 程序 import thulac，新建 thulac.thulac(args)类，其中 args
为程序的参数。之后可以通过调用 thulac.cut()进行单句分词。

具体接口参数可查看 python 版接口参数 代码示例

代码示例 1 import thulac

thu1 = thulac.thulac() #默认模式 text = thu1.cut(“我爱北京天安门”,
text=True) #进行一句话分词 print(text) 代码示例 2 thu1 =
thulac.thulac(seg\ *only=True) #只进行分词，不进行词性标注
thu1.cut_f(“input.txt”, “output.txt”) #对 input.txt
文件内容进行分词，输出到 output.txt 1.2.通用参数（C++版、Java 版） -t2s
将句子从繁体转化为简体 -seg_only 只进行分词，不进行词性标注 -deli
delimeter 设置词与词性间的分隔符，默认为下划线* -filter
使用过滤器去除一些没有意义的词语，例如“可以”。 -user userword.txt
设置用户词典，用户词典中的词会被打上 uw 标签。词典中每一个词一行，UTF8
编码(python 版暂无) -model_dir dir 设置模型文件所在文件夹，默认为
models/ 1.3.Java 版特有的参数 -input input_file
设置从文件读入，默认为命令行输入 -output output_file
设置输出到文件中，默认为命令行输出 1.4.python 版接口参数
thulac(user_dict=None, model_path=None, T2S=False, seg_only=False,
filt=False)初始化程序，进行自定义设置

user_dict 设置用户词典，用户词典中的词会被打上 uw
标签。词典中每一个词一行，UTF8 编码 T2S 默认 False,
是否将句子从繁体转化为简体 seg_only 默认 False,
时候只进行分词，不进行词性标注 filt 默认 False,
是否使用过滤器去除一些没有意义的词语，例如“可以”。 model_path
设置模型文件所在文件夹，默认为 models/ cut(文本, text=False)
对一句话进行分词

text 默认为 False, 是否返回文本，不返回文本则返回一个二维数组([[word,
tag]..]),tag_only 模式下 tag 为空字符。 cut_f(输入文件, 输出文件)
对文件进行分词

run() 命令行交互式分词(屏幕输入、屏幕输出)

1.5.C++版接口参数(需 include “include/thulac.h”) 首先需要实例化 THULAC
类，然后可以调用以下接口：

int init(const char\* model_path = NULL, const char\* user\ *path =
NULL, int just_seg = 0, int t2s = 0, int ufilter = 0, char separator =
’*\ ’); 初始化程序，进行自定义设置 user\ *path
设置用户词典，用户词典中的词会被打上 uw 标签。词典中每一个词一行，UTF8
编码 t2s 默认 False, 是否将句子从繁体转化为简体 just_seg 默认 False,
时候只进行分词，不进行词性标注 ufilter 默认 False,
是否使用过滤器去除一些没有意义的词语，例如“可以”。 model_path
设置模型文件所在文件夹，默认为 models/ separator 默认为‘*\ ’,
设置词与词性之间的分隔符 1.5.分词和词性标注模型的使用 THULAC
需要分词和词性标注模型的支持，用户可以在下载列表中下载 THULAC 模型
Models_v1.zip，并放到 THULAC 的根目录即可，或者使用参数-model_dir dir
指定模型的位置。

2.模型训练程序 THULAC 工具包提供模型训练程序 train_c，用户可以使用
train_c 训练获得 THULAC 的所需的模型。

2.1.命令格式 ./train_c [-s separator][-b bigram_threshold] [-i
iteration] training_filename model_filename 使用 training_filename
为训练集，训练出来的模型名字为 model_filename 2.2.参数意义 -s
设置词与词性间的分隔符，默认为斜线/ -b 设置二字串的阈值，默认为 1 -i
设置训练迭代的轮数，默认为 15 2.3.训练集格式
我们使用默认的分隔符（斜线/）作为例子，训练集内容应为

::

   我/r 爱/vm 北京/ns 天安门/ns

类似的已经进行词性标注的句子。

若要训练出只分词的模型，使用默认的分隔符（斜线/）作为例子，训练集内容应为

::

   我/ 爱/ 北京/ 天安门/

类似的句子。

2.4.使用训练出的模型

将训练出来的模型覆盖原来 models
中的对应模型，之后执行分词程序即可使用训练出来的模型。

与代表性分词软件的性能对比
--------------------------

我们选择 LTP-3.2.0 、ICTCLAS(2015 版)
、jieba(C++版)等国内具代表性的分词软件与 THULAC 做性能比较。我们选择
Windows 作为测试环境，根据第二届国际汉语分词测评（The Second
International Chinese Word Segmentation
Bakeoff)发布的国际中文分词测评标准，对不同软件进行了速度和准确率测试。

在第二届国际汉语分词测评中，共有四家单位提供的测试语料（Academia
Sinica、 City University 、Peking University 、Microsoft Research）,
在评测提供的资源 icwb2-data
中包含了来自这四家单位的训练集（training）、测试集（testing）,
以及根据各自分词标准而提供的相应测试集的标准答案（icwb2-data/scripts/gold）．在
icwb2-data/scripts 目录下含有对分词进行自动评分的 perl 脚本 score。

我们在统一测试环境下，对上述流行分词软件和 THULAC
进行了测试，使用的模型为各分词软件自带模型。THULAC
使用的是随软件提供的简单模型 Model_1。评测环境为 Intel Core i5 2.4 GHz
评测结果如下：

msr_test（560KB）

Algorithm Time Precision Recall F-Measure LTP-3.2.0 3.21s 0.867 0.896
0.881 ICTCLAS(2015 版) 0.55s 0.869 0.914 0.891 jieba(C++版) 0.26s 0.814
0.809 0.811 THULAC_lite 0.62s 0.877 0.899 0.888 pku_test（510KB）

Algorithm Time Precision Recall F-Measure LTP-3.2.0 3.83s 0.960 0.947
0.953 ICTCLAS(2015 版) 0.53s 0.939 0.944 0.941 jieba(C++版) 0.23s 0.850
0.784 0.816 THULAC_lite 0.51s 0.944 0.908 0.926

除了以上在标准测试集上的评测，我们也对各个分词工具在大数据上的速度进行了评测，结果如下：

CNKI_journal.txt（51 MB）

Algorithm Time Speed LTP-3.2.0 348.624s 149.80KB/s ICTCLAS(2015 版)
106.461s 490.59KB/s jieba(C++版) 22.558s 2314.89KB/s THULAC_lite 42.625s
1221.05KB/s

词性标记集
----------

通用标记集（适用于所有版本）

-  n/名词 np/人名 ns/地名 ni/机构名 nz/其它专名
-  m/数词 q/量词 mq/数量词 t/时间词 f/方位词 s/处所词
-  v/动词 a/形容词 d/副词 h/前接成分 k/后接成分 i/习语
-  j/简称 r/代词 c/连词 p/介词 u/助词 y/语气助词
-  e/叹词 o/拟声词 g/语素 w/标点 x/其它

特殊标记集（适用于 lite_v1_2 版） 为了方便在分词和词性标注后的过滤，在
v1_2 版本，我们增加了两种词性，如果需要可以下载使用。 vm/能愿动词
vd/趋向动词

THULAC 的不同配置
-----------------

我们随 THULAC 源代码附带了简单的分词模型
Model_1，仅支持分词功能。该模型由人民日报分词语料库训练得到。

我们随 THULAC 源代码附带了分词和词性标注联合模型
Model_2，支持同时分词和词性标注功能。该模型由人民日报分词和词性标注语料库训练得到。

我们还提供更复杂、完善和精确的分词和词性标注联合模型 Model_3
和分词词表。该模型是由多语料联合训练训练得到（语料包括来自多文体的标注文本和人民日报标注文本等）。由于模型较大，如有机构或个人需要，请填写“资源申请表.doc”，并发送至
thunlp@gmail.com ，通过审核后我们会将相关资源发送给联系人。

获取链接
--------

THULAC
工具包分成两个部分组成。第一部分为算法源代码部分，可以通过网站上下载或者从
github 获取最新基础版代码，无需注册；第二部分为算法模型部分，THULAC
需要分词和词性标注模型的支持，可以从 2.算法模型注册后获得。

算法源代码 lite 版

-  Source Version Description Size Date Download
-  THULAC_lite lite 版 THULAC_lite 分词源代码(C++版) 799KB 2017-04-11
   download
-  THULAC_lite 分词源代码(python 版) 44KB 2017-04-11
-  THULAC_lite 分词源代码(java 版) 588KB 2017-01-13
-  THULAC_lite 分词 java 版可执行的 jar 包 55KB 2017-04-11
-  THULAC 模型，包括分词模型和词性标注模型（lite 版） 58.2MB 2016-01-10
-  v1_2 THULAC_lite_v1_2 分词源代码(C++版) 799KB 2017-04-11 download
-  THULAC_lite_v1_2 分词源代码(java 版) 588KB 2017-01-13
-  THULAC_lite_v1_2 分词 java 版可执行的 jar 包 55KB 2017-04-11
-  THULAC 模型，包括分词模型和词性标注模型（v1_2） 58.3MB 2016-07-10
   2.算法源代码 lite 版(github)
-  Source Description Link
-  THULAC_lite_C++ THULAC_lite 分词源代码(C++版) link
-  THULAC_lite_Python THULAC_lite 分词源代码(python 版) link
-  THULAC_lite_Java THULAC_lite 分词源代码(java 版) link
-  THULAC_lite.So THULAC_lite 分词源代码(So 版) link 3.算法模型
-  Source Description Size Date Download
-  THULAC_lite_Model THULAC 模型，包括分词模型和词性标注模型（lite 版）
   58.2MB 2016-01-10 download
-  THULAC_pro_c++_v1.zip THULAC
   模型，包括更复杂完善的分词和词性标注模型以及分词词表 162MB 2016-01-10
   download

注意事项
--------

该工具目前仅处理 UTF8
编码中文文本，之后会逐渐增加支持其他编码的功能，敬请期待。

历史
----

更新时间 更新内容

-  2017-01-17 在 pip 上发布 THULAC 分词 python 版本。
-  2016-10-10 增加 THULAC 分词 so 版本。
-  2016-03-31 增加 THULAC 分词 python 版本。
-  2016-01-20 增加 THULAC 分词 Java 版本。
-  2016-01-10 开源 THULAC 分词工具 C++版本。

开源协议
--------

THULAC 面向国内外大学、研究所、企业以及个人用于研究目的免费开放源代码。
如有机构或个人拟将 THULAC 用于商业目的，请发邮件至 thunlp@gmail.com
洽谈技术许可协议。 欢迎对该工具包提出任何宝贵意见和建议。请发邮件至
thunlp@gmail.com。 如果您在 THULAC
基础上发表论文或取得科研成果，请您在发表论文和申报成果时声明“使用了清华大学
THULAC”，并按如下格式引用：

中文： 孙茂松, 陈新雄, 张开旭, 郭志芃, 刘知远.
THULAC：一个高效的中文词法分析工具包. 2016.

英文： Maosong Sun, Xinxiong Chen, Kaixu Zhang, Zhipeng Guo, Zhiyuan
Liu. THULAC: An Efficient Lexical Analyzer for Chinese. 2016.

相关论文
--------

Zhongguo Li, Maosong Sun. Punctuation as Implicit Annotations for
Chinese Word Segmentation. Computational Linguistics, vol. 35, no. 4,
pp. 505-512, 2009.

作者

Maosong Sun （孙茂松，导师）, Xinxiong Chen（陈新雄，博士生）, Kaixu
Zhang (张开旭，硕士生）, Zhipeng Guo（郭志芃，本科生）, Junhua Ma
（马骏骅，访问学生）, Zhiyuan Liu（刘知远，助理教授）.

常见问题
--------

1. THULAC 工具包提供的模型是如何得到的？

   THULAC 工具包随包附带的分词模型 Model_1 以及分词和词性标注模型
   Model_2
   是由人民日报语料库训练得到的。这份语料库中包含已标注的字数约为一千二百万字。

   同时，我们还提供更复杂、完善和精确的分词和词性标注联合模型 Model_3
   和分词词表。该模型是由多语料联合训练训练得到（语料包括来自多文体的标注文本和人民日报标注文本等）。这份语料包含已标注的字数约为五千八百万字。由于模型较大，如有机构或个人需要，请填写“资源申请表.doc”，并发送至
   thunlp@gmail.com ，通过审核后我们会将相关资源发送给联系人。

2. 能否提供工具包所带的模型的训练原始语料（如人民日报语料库）？

   THULAC
   工具包中所带的模型的训练原始语料都是需要授权获得的。如果需要原始分词语料，如人民日报语料库，请联系北京大学计算语言学研究所授权获取。

致谢
----

感谢清华大学的本科生潘星宇对 THULAC-Python 工具的支持和帮助。
感谢付超群带领的看盘宝团队对 THULAC.so 工具的支持和帮助。
使用者如有任何问题、建议和意见，欢迎发邮件至 thunlp@gmail.com 。

版权所有：清华大学自然语言处理与社会人文计算实验室

Copyright：Natural Language Processing and Computational Social Science
Lab, Tsinghua University
