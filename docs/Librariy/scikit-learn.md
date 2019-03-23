# scikit-learn

scikit-learn 是一个用于机器学习的 Python 模块，建立在 SciPy 之上，并根据 3-Clause BSD 许可证进行分发。

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed.
See the [About us](http://scikit-learn.org/dev/about.html#authors) page for a list of core contributors.

It is currently maintained by a team of volunteers.

Website: http://scikit-learn.org

## 安装

依赖

```sh

scikit-learn requires:

- Python (>= 3.5)
- NumPy (>= 1.11.0)
- SciPy (>= 0.17.0)

**Scikit-learn 0.20 was the last version to support Python2.7.**
Scikit-learn 0.21 and later require Python 3.5 or newer.

For running the examples Matplotlib >= 1.5.1 is required.
A few examples
require scikit-image >= 0.12.3, a few examples require pandas >= 0.18.0
and a few example require joblib >= 0.11.

scikit-learn also uses CBLAS, the C interface to the Basic Linear Algebra
Subprograms library.
scikit-learn comes with a reference implementation, but
the system CBLAS will be detected by the build system and used if present.
CBLAS exists in many implementations; see `Linear algebra libraries
<http://scikit-learn.org/stable/modules/computing#linear-algebra-libraries>`_
for known issues.

User installation
```

如果您已经安装了 numpy 和 scipy，安装 scikit-learn 的最简单方法是使用`pip` :

    pip install -U scikit-learn

或者`conda`:

    conda install scikit-learn

该文档包含更详细的[安装说明](http://scikit-learn.org/stable/install.html).

## 更新日志

See the [changelog](http://scikit-learn.org/dev/whats_new.html) for a history of notable changes to scikit-learn.

## 发展

We welcome new contributors of all experience levels.
The scikit-learn community goals are to be helpful, welcoming, and effective.
The [Development Guide](http://scikit-learn.org/stable/developers/index.html) has detailed information about contributing code, documentation, tests, and more.
We've included some basic information in this README.

重要链接

```

- Official source code repo: https://github.com/scikit-learn/scikit-learn
- Download releases: https://pypi.org/project/scikit-learn/
- Issue tracker: https://github.com/scikit-learn/scikit-learn/issues

源代码
~~~~~~~~~~~

You can check the latest sources with the command:

    git clone https://github.com/scikit-learn/scikit-learn.git

Setting up a development environment

```

Quick tutorial on how to go about setting up your environment to
contribute to scikit-learn: https://github.com/scikit-learn/scikit-learn/blob/master/CONTRIBUTING.md

测试

```

After installation, you can launch the test suite from outside the
source directory (you will need to have ``pytest`` >= 3.3.0 installed):

    pytest sklearn

See the web page http://scikit-learn.org/dev/developers/advanced_installation.html#testing
for more information.

    Random number generation can be controlled during testing by setting
    the ``SKLEARN_SEED`` environment variable.

Submitting a Pull Request
```

Before opening a Pull Request, have a look at the
full Contributing page to make sure your code complies
with our guidelines: http://scikit-learn.org/stable/developers/index.html

## 项目历史

The project was started in 2007 by David Cournapeau as a Google Summer of Code project, and since then many volunteers have contributed.
See the [About us](http://scikit-learn.org/dev/about.html#authors) page for a list of core contributors.

The project is currently maintained by a team of volunteers.

**Note**: `scikit-learn` was previously referred to as `scikits.learn`.

## 帮助和支持

文档

```

- HTML documentation (stable release): http://scikit-learn.org
- HTML documentation (development version): http://scikit-learn.org/dev/
- FAQ: http://scikit-learn.org/stable/faq.html

通讯
```

- Mailing list: https://mail.python.org/mailman/listinfo/scikit-learn
- IRC channel: `#scikit-learn` at `webchat.freenode.net`
- Stack Overflow: https://stackoverflow.com/questions/tagged/scikit-learn
- Website: http://scikit-learn.org

引文

```

If you use scikit-learn in a scientific publication, we would appreciate citations: http://scikit-learn.org/stable/about.html#citing-scikit-learn
```
