MMSeg 分词算法简述 博客分类： java

MMSeg 只是实现了 Chih-Hao Tsai 的 MMSEG
算法，这是一个来源于网络的分词算法。我照抄了算法开始的部分：

MMSEG: A Word Identification System for Mandarin Chinese Text Based on
Two Variants of the Maximum Matching Algorithm

Published: 1996-04-29 Updated: 1998-03-06 Document updated: 2000-03-12
License: Free for noncommercial use Copyright 1996-2006 Chih-Hao Tsai
(Email: hao520 at yahoo.com )

您可以在 Chih-Hao Tsai’s Technology Page 找到算法的原文。

我将依据自己的理解来简述 MMSeg 分词算法的基本原理，如有错误请不吝赐教。

首先来理解一下 chunk，它是 MMSeg 分词算法中一个关键的概念。Chunk
中包含依据上下文分出的一组词和相关的属性，包括长度(Length)、平均长度(Average
Length)、标准差的平方(Variance)和自由语素度(Degree Of Morphemic
Freedom)。我在下面列出了这 4 个属性的计算方法：

属性 含义 代码位置 长度(Length) chuck 中各个词的长度之和
org.solol.mmseg.internal.Chunk.getLength() 平均长度(Average Length)
长度(Length)/词数 org.solol.mmseg.internal.Chunk.getAverageLength()
标准差的平方(Variance) 同数学中的定义
org.solol.mmseg.internal.Chunk.getVariance() 自由语素度(Degree Of
Morphemic Freedom) 各单字词词频的对数之和
org.solol.mmseg.internal.Chunk.getDegreeOfMorphemicFreedom()
注意：表中的含义列可能有些模糊，最好参照 MMSeg
的源代码进行理解，代码所在的函数已经给出了。

Chunk 中的 4 个属性采用 Lazy
的方式来计算，即只有在需要该属性的值时才进行计算，而且只计算一次。

其次来理解一下规则(Rule)，它是 MMSeg
分词算法中的又一个关键的概念。实际上我们可以将规则理解为一个过滤器(Filter)，过滤掉不符合要求的
chunk。MMSeg 分词算法中涉及了 4 个规则：

规则 1：取最大匹配的 chunk (Rule 1: Maximum matching) 规则
2：取平均词长最大的 chunk (Rule 2: Largest average word length) 规则
3：取词长标准差最小的 chunk (Rule 3: Smallest variance of word lengths)
规则 4：取单字词自由语素度之和最大的 chunk (Rule 4: Largest sum of
degree of morphemic freedom of one-character words) 这 4 个规则分别位于
org.solol.mmseg.internal.MMRule.java、org.solol.mmseg.internal.LAWLRule.java、org.solol.mmseg.internal.SVWLRule.java
和 org.solol.mmseg.internal.LSDMFOCWRule.java4
个源文件中。之所以这样来处理是因为我们可以方便的增加规则和修改应用规则的顺序。

这 4 个规则符合汉语成词的基本习惯。

再次来理解一下两种匹配方式，简单最大匹配(Simple maximum
matching)和复杂最大匹配(Complex maximum matching)。

简单最大匹配仅仅使用了规则 1。

复杂最大匹配先使用规则 1 来过滤 chunks，如果过滤后的结果多于或等于
2，则使用规则 2 继续过滤，否则终止过滤过程。如果使用规则 2
得到的过滤结果多于或等于 2，则使用规则 3
继续过滤，否则终止过滤过程。如果使用规则 3 得到的过滤结果多于或等于
2，则使用规则 4 继续过滤，否则终止过滤过程。如果使用规则 4
得到的过滤结果多于或等于 2，则抛出一个表示歧义的异常，否则终止过滤过程。

最后通过一个例句–研究生命起源来简述一下复杂最大匹配的分词过程。MMSeg
分词算法会得到 7 个 chunk，分别为：

编号 chunk 长度 0 研\ *究*\ 生 3 1 研\ *究*\ 生命 4 2 研究\ *生*\ 命 4 3
研究\ *生命*\ 起 5 4 研究\ *生命*\ 起源 6 5 研究生\ *命*\ 起 5 6
研究生\ *命*\ 起源 6 使用规则 1 过滤后得到 2 个 chunk，如下：

编号 chunk 长度 4 研究\ *生命*\ 起源 6 6 研究生\ *命*\ 起源 6
计算平均长度后为：

编号 chunk 长度 平均长度 4 研究\ *生命*\ 起源 6 2 6 研究生\ *命*\ 起源 6
2 使用规则 2 过滤后得到 2 个 chunk，如下：

编号 chunk 长度 平均长度 4 研究\ *生命*\ 起源 6 2 6 研究生\ *命*\ 起源 6
2 计算标准差的平方后为：

编号 chunk 长度 平均长度 标准差的平方 4 研究\ *生命*\ 起源 6 2 0 6
研究生\ *命*\ 起源 6 2 4/9 使用规则 3 过滤后得到 1 个 chunk，如下：

编号 chunk 长度 平均长度 标准差的平方 4 研究\ *生命*\ 起源 6 2 0
匹配过程终止。最终取“研究”成词，以相同的方法继续处理“生命起源”。

分词效果:

Simple ->研究生*命*\ 起源\ *Complex->研究*\ 生命\ *起源* Simple
->研究生*教育* Complex->研究生\ *教育*

注意：Simple 表示简单最大匹配的分词效果，Complex
表示复杂最大匹配的分词效果。

Update:这是一篇老文，之前放在http://dobestdeveloper.blogspot.com上，现在移到这里方便大家查阅。

http://m.blog.csdn.net/blog/HHyatt/6202826
