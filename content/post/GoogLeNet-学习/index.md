+++
date = '2025-12-21T13:40:43+08:00'
title = 'GoogLeNet学习'

+++

# GoogLeNet学习

## 概论

在2014年的ImageNet挑战赛(ILSVRC14)上，GoogLeNetz和VGGNet成为了当年的双雄。GoogLeNet获得了图片分类大赛的第一名，VGGNet紧随其后。这两种模型的共同特点是网络深度更深。VGGNet;是基于LeNetz和AlexNet的一些框架结构，GoogLeNet则采用了更加大胆的网络结构。尽管GoogLeNet仅有22层，它的尺寸比AlexNet和VGGNet要小得多。

GoogLeNet的参数数量为500万个，而AlexNet的参数数量是GoogLeNet的12倍，VGGNet的参数数量又是AlexNet的3倍。因此，当内存或计算资源有限时，GoogLeNet是更好的选择。从模型结果来看，GoogLeNet的性能表现更加优越。GoogLeNet是谷歌公司研究出来的深度网络结构，命名为GoogLeNet而不是GoogleNet的原因据说是为了向LeNet致敬。

## 研发动机

一般来说，提升网络性能最直接的办法就是增加网络深度和宽度，或者输入数据的大小；但这种方式存在以下问题：

- 参数太多：如果训练数据集有限，参数太多很容易产生过拟合。
- 难以应用：网络越大、参数越多，计算复杂度越大，导致应用问题。
- 难以优化模型：网络越深，越容易出现梯度弥散问题（梯度越往后越容易消失）难以进行模型优化。

## 原理学习

### Inception结构

![7](/images/7.png)

------

**待完成学习......**
