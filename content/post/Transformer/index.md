+++
date = '2025-10-15T13:52:23+08:00'
title = 'Transformer'

+++

基础知识学习了这篇博客

[CV攻城狮入门VIT(vision transformer)之旅——近年超火的Transformer你再不了解就晚了！ - 掘金](https://juejin.cn/post/7152002993204756487)

论文参考：Attention Is All You Need

[1706.03762](https://arxiv.org/pdf/1706.03762)

然后读Transformer底层代码

环境导入

```python
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

import utils_transformer as utils
import math, copy, time

import matplotlib.pyplot as plt
print("PyTorch Version: ",torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
num_gpu = torch.cuda.device_count()
print('Number of GPUs Available:', num_gpu)

# # Default directory "runs"
# writer = SummaryWriter()
```

HyperParameters设置

````python
batch_size = 2
sequence_length = 6
hidden_size = 16
attention_heads = 8
````

Embeddings 将输入 token 和输出 token 转换为维度为 d 的向量

```python
class Embeddings(nn.Module):  # 定义一个名为Embeddings的类，继承自nn.Module
    def __init__(self, d_model_hidden_size, vocab_size):  # 初始化函数，接收隐藏层大小和词汇表大小作为参数
        super(Embeddings, self).__init__()  # 调用父类的初始化方法
        # vocab_size: 词汇表中的元素数量
        # d_model_hidden_size: 隐藏层的大小
        self.lut = nn.Embedding(vocab_size, d_model_hidden_size)  # 创建一个嵌入层，将词汇表大小和隐藏层大小作为参数
        self.d_model = d_model_hidden_size  # 将隐藏层大小存储在实例变量d_model中

    def forward(self, x):  # 定义前向传播函数，接收输入x
        # 查找嵌入向量，将输入的单词索引转换为对应的嵌入向量
        # 输出形状为 (batch_size, sequence_length, d_model_hidden_size)
        return self.lut(x) * math.sqrt(self.d_model)  # 返回嵌入向量，并乘以隐藏层大小的平方根，以进行缩放
```

Attenion

```python
# 实现注意力机制（缩放点积注意力）
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)  # 获取query的最后一个维度的大小，通常是嵌入向量的维度
    # 计算注意力得分，通过query和key的转置相乘，然后除以d_k的平方根进行缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:  # 如果提供了掩码
        # 使用掩码填充得分为负无穷大，以避免在softmax中被考虑
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # 对得分应用softmax，沿着最后一个维度进行归一化
    p_attn = F.softmax(scores, dim=-1)
    
    if dropout is not None:  # 如果提供了dropout层
        p_attn = dropout(p_attn)  # 对注意力分布应用dropout
    
    # 计算注意力结果，通过注意力分布p_attn和value相乘
    attention_result = torch.matmul(p_attn, value)
    
    # 返回注意力结果和注意力分布
    return attention_result, p_attn
```

MultiHeaded Attention

```python
class MultiHeadedAttention(nn.Module):  # 定义一个名为MultiHeadedAttention的类，继承自nn.Module
    def __init__(self, h, d_model, dropout=0.1):  # 初始化函数，接收头数h、模型维度d_model和dropout比率作为参数
        "Take in model size and number of heads."  # 简短说明：接收模型大小和头的数量
        super(MultiHeadedAttention, self).__init__()  # 调用父类的初始化方法
        assert d_model % h == 0  # 确保模型维度可以被头数整除
        # 我们假设d_v总是等于d_k
        self.d_k = d_model // h  # 计算每个头的维度
        self.h = h  # 存储头的数量
        self.linears = utils.clones(nn.Linear(d_model, d_model), 4)  # 创建四个线性变换层
        self.attn = None  # 初始化注意力权重
        self.dropout = nn.Dropout(p=dropout)  # 创建dropout层，设置丢弃率
        
    def forward(self, query, key, value, mask=None):  # 定义前向传播函数，接收query、key、value和可选的mask作为参数
        
        if mask is not None:  # 如果提供了掩码
            # 对所有的头应用相同的掩码
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # 获取batch的大小
        
        # 1) 在批处理中执行所有线性变换，从d_model => h x d_k
        # query、key和value的原始形状是 [nbatches, seq_len, d_model]
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
                             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) 对所有投影向量应用注意力机制
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        
        # 3) 使用view将结果“连接”起来，并应用最终的线性变换
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)  # 返回最后的线性变换结果
```















