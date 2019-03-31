sego
====

   `sego <https://github.com/huichen/sego>`__\ Go 中文分词

词典用双数组 trie（Double-Array Trie）实现，
分词器算法为基于词频的最短路径加动态规划。

支持普通和搜索引擎两种分词模式，支持用户词典、词性标注，可运行JSON RPC
服务。

分词速度单线程9MB/s，goroutines 并发42MB/s（8 核 Macbook Pro）。

安装/更新
---------

::

   go get -u github.com/huichen/sego

使用
----

.. code:: go

   package main

   import (
       "fmt"
       "github.com/huichen/sego"
   )

   func main() {
       // 载入词典
       var segmenter sego.Segmenter
       segmenter.LoadDictionary("github.com/huichen/sego/data/dictionary.txt")

       // 分词
       text := []byte("中华人民共和国中央人民政府")
       segments := segmenter.Segment(text)

       // 处理分词结果
       // 支持普通模式和搜索模式两种分词，见代码中SegmentsToString函数的注释。
       fmt.Println(sego.SegmentsToString(segments, false))
   }
