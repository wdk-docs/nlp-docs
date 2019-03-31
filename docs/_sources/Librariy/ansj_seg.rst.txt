Ansj 中文分词
=============

`Ansj <https://github.com/NLPchina/ansj_seg>`__

|1.X Build Status| |Gitter|

#####使用帮助 : `3.x 版本及之前 <http://nlpchina.github.io/ansj_seg/>`__
, `5.x 版本及之后 <https://github.com/NLPchina/ansj_seg/wiki>`__
在线测试地址 : http://demo.nlpcn.org

摘要
''''

   这是一个基于 n-Gram+CRF+HMM 的中文分词的 java 实现.

..

   分词速度达到每秒钟大约 200 万字左右（mac air 下测试），准确率能达到
   96%以上

   目前实现了.中文分词. 中文姓名识别 .
   用户自定义词典,关键字提取，自动摘要，关键字标记等功能

..

   可以应用到自然语言处理等方面,适用于对分词效果要求高的各种项目.

下载 jar
''''''''

-  访问
   `http://maven.nlpcn.org/org/ansj/ <https://oss.sonatype.org/content/repositories/releases/org/ansj/ansj_seg/>`__
   最好下载最新版 ansj_seg/

   -  同时下载\ `nlp-lang.jar <https://oss.sonatype.org/content/repositories/releases/org/nlpcn/nlp-lang/>`__
      需要和 ansj_seg 配套..配套关系可以看 jar 包中的 maven
      依赖,一般最新的 ansj 配最新的 nlp-lang 不会有错。

-  导入到 eclipse ，开始你的程序吧

maven
'''''

::


           <dependency>
               <groupId>org.ansj</groupId>
               <artifactId>ansj_seg</artifactId>
               <version>5.1.1</version>
           </dependency>

调用 demo
'''''''''

如果你第一次下载只想测试测试效果可以调用这个简易接口

.. raw:: html

   <pre><code>
    String str = "欢迎使用ansj_seg,(ansj中文分词)在这里如果你遇到什么问题都可以联系我.我一定尽我所能.帮助大家.ansj_seg更快,更准,更自由!" ;
    System.out.println(ToAnalysis.parse(str));

    ﻿欢迎/v,使用/v,ansj/en,_,seg/en,,,(,ansj/en,中文/nz,分词/n,),在/p,这里/r,如果/c,你/r,遇到/v,什么/r,问题/n,都/d,可以/v,联系/v,我/r,./m,我/r,一定/d,尽我所能/l,./m,帮助/v,大家/r,./m,ansj/en,_,seg/en,更快/d,,,更/d,准/a,,,更/d,自由/a,!
   </code></pre>

Join Us
'''''''

心思了很久，不管有没有人帮忙把。我写上来，如果你有兴趣，有热情可以联系我。

-  补充文档,增加调用实例和说明
-  增加一些规则性
   Recognition，举例\ `身份证号码识别 <https://github.com/NLPchina/ansj_seg/blob/master/src/main/java/org/ansj/recognition/impl/IDCardRecognition.java>`__\ ，目前未完成的有
   ``时间识别``\ ，\ ``IP地址识别``\ ，\ ``邮箱识别``,\ ``网址识别``\ ，\ ``词性识别``\ 等…
-  提供更加优化的 CRF 模型。替换 ansj 的默认模型。
-  补充测试用例，n 多地方测试不完全。如果你有兴趣可以帮忙啦！
-  重构人名识别模型。增加机构名识别等模型。
-  增加句法文法分析
-  实现 lstm 的分词方式
-  拾遗补漏…

.. |1.X Build Status| image:: https://travis-ci.org/NLPchina/ansj_seg.svg?branch=master
   :target: https://travis-ci.org/NLPchina/ansj_seg
.. |Gitter| image:: https://badges.gitter.im/NLPchina/ansj_seg.svg
   :target: https://gitter.im/NLPchina/ansj_seg?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge
