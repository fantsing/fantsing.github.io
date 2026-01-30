+++
date = '2026-1-15T13:52:23+08:00'
title = 'CS336-大模型概述与tokenization'

+++

# Overview and Tokenization

## Overview

一个很直接的问题：研究者与底层的技术正在越来越脱节。八年前，学者会补充并训练他们自己的模型，六年前还会下载一个模型再 fine-tune，而现在学者们只会向闭源模型（GPT-4/Claude/Gemini）提问。当然这是一个比较夸张的说法，但现在确实随着模型能力的提升，很多基础的工作不再需要研究者去深究了。Full understanding of this technology is necessary for fundamental research.

通过这个课程能学到:

Mechanics: how things work (什么是 Transformer, 模型如何并行使用 GPU)
Mindset: 最大化利用硬件的显存，scaling laws！
Intuitions: 什么样的 data 和 modeling decisions 会有更好的结果
还特别强调了该课程最核心的理念：maximize efficiency! 也就是在给定你的数据和特定 GPU 的情况下，什么是能够训练出来的最好的模型。

正式课程从模型的整体发展历史开始，讲了下不同关键时间点模型的变化情况，其中在讲到开源的时候，第一次看到了对开源的不同的定义，感觉还挺有意思的，他把开源分了两种类型，一种叫 Open-weight models（比如 DeepSeek），另一种叫 Open-source models（比如 OLMo），前者公开模型权重，架构细节，但只提供部分训练细节，没有数据细节；而后者基本全部都公开，但也没有提供训练过程的失败案例。

## Tokenization 

LLM 需要把概率分布置于 token 序列（通常用整数索引表示），所以我们需要将初始文本的字符串 encode 成 token，又需要将 token decode 回字符串，这时候就需要有 Tokenizer 来完成这些事情。

视频中的分词器网站如下：[Tiktokenizer](https://tiktokenizer.vercel.app/?encoder=gpt2)

然后讲了几个 Tokenization 的不同方法，并介绍了目前主流的大家都在用的。

Character-based tokenization：每个 character 被转换成一个 code point（整数），这样对每种语言的处理都很方便，但问题是 vocabulary 太大，有些 character 不常用，这样转换效率不高；
Byte-based tokenization：转换成字节序列，可以把表征范围限制在 0-255，问题是有些 character 会被转换为 \xf0\x9f\x8c\x8d，序列太长，而 transformer 的上下文长度是有限的；
Word-based tokenization：将字符串分割成单词，然后将这些分割的结果映射成整数，但问题词表可能变得巨大，包含很多不常见的词汇，voc abulary 的大小不确定；
Byte Pair Encoding (BPE)。在 BPE 之前普遍用 word-base，之后从 GPT-2 开始使用 BPE。基本思想是在原始的文本上训练 tokenizer，自动确定 vocabulary，也就是常见的 vocabulary 用单一的 token 表示，很少出现的 vocabulary 用多个 token 表示。

































