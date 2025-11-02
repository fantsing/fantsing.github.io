+++
date = '2025-10-03T13:40:43+08:00'
title = 'VGG学习'

+++

# VGG学习

## 概论

2014年，牛津大学计算机视觉组(Visual Geometry Group)和Google DeepMind公司的研究员Karen Simonyan和Andrew Zisserman研发出了新的深度卷积神经网络：VGGNet,,并在ILSVRC2014比赛分类项目中取得了第二名的好成绩(第一名是同年提出的GoogLeNet模型)，同时在定位项目中获得第一名。VGGNet模型通过探索卷积神经网络的深度与性能之间的关系，成功构建了16~19层深的卷积神经网络，并证明了**增加网络深度可以在一定程度上提高网络性能，大幅降低错误率**。此外，VGGNet具有很强的拓展性和泛化性，适用于其它类型的图像数据。至今，VGGNet仍然被广泛应用于图像特征提取。VGGNet可以看成是加深版本的AlexNet,都是由卷积层、全连接层两大部分构成。

论文链接：[[1409.1556\] Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

## 原理学习

### 网络退化

![5](images/5.png)

### 残差连接

- 结构简洁，VGG网络中，所有卷积层的卷积核大小，步长和填充都相同，并且通过使用最大池化对卷积层进行分层。所有隐藏层的激活单元都采用RLU函数。由于其极简且清晰的结构，VGGNet至今仍广泛用于图像特征提取。
- 小卷积核，VGGNet中所有的卷积层都使用了小卷积核(3×3)。这种设计有两个优点：一方面，可以大幅减少参数量；另一方面，节省下来的参数可以用于堆叠更多的卷积层，进一步增加了网络的深度和非线性映射能力，从而提高了网络的表达和特征提取能力。

VGG模型中指出两个3×3的卷积堆叠获得的感受野大小，相当于一个5×5的卷积：而3个5×5卷积的堆叠获取到的感受野相当于一个7×7的卷积。这样可以增加非线性映射，也能很好地减少参数(例如5×5的参数为25个，而2个3×3的参数为18)。

- 小池化核、相比AlexNet的3×3的池化核，VGGNet全部采用2×2的池化核。
- 通道数多、VGGNet第一层的通道数为64，后面每层都进行了翻倍，最多达到512个通道。相比较于AlexNet和ZFNet最多得到的通道数是256，VGGNet翻倍的通道数使得更多的信息可以被卷积操作提取出来。
- 层数更深、特征图更多、网络中，卷积层专注于扩大特征图的通道数、池化层专注于缩小特征图的宽和高，使得模型架构上更深更宽的同时，控制了计算量的增加规模。

### 模型结构

![6](images/6.png)

容易部署到硬件，轻量化！

## 代码实现

也是同上一篇AlexNet的参考代码一致，链接：[GitHub - Arwin-Yu/Deep-Learning-Image-Classification-Models-Based-CNN-or-Attention: This project organizes classic images classification neural networks based on convolution or attention, and writes training and inference python scripts](https://github.com/Arwin-Yu/Deep-Learning-Image-Classification-Models-Based-CNN-or-Attention)

主要研究“vggnet.py”文件

```python
import torch.nn as nn
import torch

# 官方提供的预训练权重
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}
```

### 类的定义

```python
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features  # 特征提取部分（卷积层和池化层）
        self.classifier = nn.Sequential(  # 分类部分（全连接层）
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()  # 权重初始化
```

- `features`：这是网络的特征提取部分，由卷积层和池化层组成
- `classifier`：这是分类器部分，由全连接层组成
- `num_classes`：分类的类别数，默认1000（对应ImageNet数据集）

### 前向传播方法

```python
def forward(self, x):
    # N x 3 x 224 x 224  输入图像：批量大小x通道数x高度x宽度
    x = self.features(x)  # 通过特征提取部分
    # N x 512 x 7 x 7     特征提取后的输出
    x = torch.flatten(x, start_dim=1)  # 展平操作，从第1维度开始
    # N x 512*7*7        展平后的向量
    x = self.classifier(x)  # 通过分类器
    return x
```

- 前向传播定义了数据在网络中的流动路径
- `torch.flatten`将卷积输出的多维特征转换为一维向量，以便输入全连接层

### 权重初始化方法

```python
def _initialize_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Conv2d):  # 卷积层权重初始化
            nn.init.xavier_uniform_(m.weight)  # Xavier均匀分布初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # 偏置初始化为0
        elif isinstance(m, nn.Linear):  # 全连接层权重初始化
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
```

- 初始化权重对网络训练非常重要
- 这里使用Xavier初始化方法，适合ReLU激活函数

### 生成特征提取层

```python
def make_features(cfg: list):
    layers = []
    in_channels = 3  # 输入图像是RGB三通道
    for v in cfg:
        if v == "M":  # 如果遇到"M"，添加池化层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:  # 否则添加卷积层和激活函数
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v  # 更新输入通道数为当前输出通道数
    return nn.Sequential(*layers)  # 将所有层按顺序包装
```

- 这个函数根据配置列表`cfg`生成特征提取部分
- 遍历配置列表，遇到"M"就添加池化层，否则添加卷积层+ReLU激活函数
- `kernel_size=3, padding=1`保证卷积后特征图大小不变

### 网络配置参数

```python
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],   
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
```

- 这些配置定义了不同深度的VGG网络
- 数字表示卷积层的输出通道数
- "M"表示最大池化层
- 数字的个数对应卷积层的数量，比如VGG16有16个卷积层和全连接层

### 模型创建函数

```python
def vgg11(num_classes): 
    cfg = cfgs["vgg11"]
    model = VGG(make_features(cfg), num_classes=num_classes)
    return model

# vgg13、vgg16、vgg19函数结构类似
```

- 这些函数提供了便捷的方式创建不同深度的VGG模型
- 使用时只需调用对应的函数，如`model = vgg16(10)`创建一个用于10分类的VGG16模型

### 总结

1. 全部使用3×3的小卷积核，多个小卷积核堆叠相当于一个大卷积核的感受野
2. 采用"卷积层堆叠+池化层"的结构，逐步减小特征图尺寸，增加通道数
3. 网络深度是其重要特征，从11层到19层不等
4. 全连接层部分结构固定，都是两个4096维的隐藏层
