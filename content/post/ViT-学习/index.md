+++
date = '2025-11-16T13:40:43+08:00'
title = 'ViT-学习'

+++

## 心得

首先，跟着李沐的论文带读分享，仔细读了Transformer，Vision transformer 和 Swin Transformer 三篇具有代表性的论文，感触颇多。

Transformer：

论文：[1706.03762](https://arxiv.org/pdf/1706.03762)

视频：[Transformer论文逐段精读【论文精读】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1pu411o7BE?spm_id_from=333.788.videopod.sections&vd_source=1c8d92c874bbc8513666bfff2e44cbcc)

Vision transformer：

论文：[2010.11929](https://arxiv.org/pdf/2010.11929)

视频：[ViT论文逐段精读【论文精读】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV15P4y137jb/?spm_id_from=333.1391.0.0&vd_source=1c8d92c874bbc8513666bfff2e44cbcc)

Swin transformer：

论文：[2103.14030](https://arxiv.org/pdf/2103.14030)

视频：[Swin Transformer论文精读【论文精读】_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV13L4y1475U?spm_id_from=333.788.videopod.sections&vd_source=1c8d92c874bbc8513666bfff2e44cbcc)

最近也仔细思考了下，感觉不能走马观花地把代码复现了就行，于是对照着b站上的【前钰】大佬分享的视频，手撕ViT的底层代码，做了详细的标注并且跟着视频手推。这里可能会反复思考几天，因为基础比较薄弱。近期也是在手推nn-UNet的底层代码，双管齐下，勤于思考吧，加油！

视频链接如下：[从零搭建Vit，手撕attention注意力机制_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Xwc6eoEBa/)

代码如下：

```python
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict

def drop_path(x, drop_prob: float = 0., training: bool = False):
    '''
    每个样本的随机深度Drop Path,应用于残差块的主路径时。
    注意：这与在 EfficientNet 等创建的 DropConnect 实现相同，但与原始 EfficientNet 论文中的 PatchDropout 不同。
    参数：
        x(张量)：输入张量
        drop_prob(浮点数)：元素置零的概率
        training(布尔值)：是否处于训练模式，若为训练模式则应用随机深度
    返回：
        如果不在训练模式或丢弃概率为0,则返回输入张量x本身
        否则，应用随机深度后的输出张量
    '''
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, e.g. 2D Conv, 3D Conv, etc.
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    '''
     随机深度丢弃模块(在残差块的主路径上使用)
     这是一个Pytorch模块，用于在训练期间对输入张量应用随机深度丢弃。
    '''
    def _init_(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    '''
     前向传播函数,应用随机深度丢弃
     参数：
         x(张量)：输入张量
     返回:
         经过drop_path处理后的张量
    '''
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training) 

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768,norm_layer=None):
        # img size 图像大小  patch size 每个patch的大小
        super().__init__()
        img_size = (img_size, img_size)  # 将输入的图像大小变为二维元组
        patch_size = (patch_size, patch_size)  
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])  # 计算网格大小
        self.num_patches = self.grid_size[0]*self.grid_size[1]  # 14*14=196 总的patch数量

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)  # B,3,224,224 -> B,768,14,14使用卷积实现patch的切分和线性映射
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity() # 若有layer norm则使用，若无则保持不变

    def forward(self, x):
        B, C, H, W = x.shape    # 获取输入张量的形状
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"输入图像大小{H}x{W}与模型期望大小{self.img_size[0]}x{self.img_size[1]}不匹配"
        # B,3,224,224 -> B,768,14,14 -> B,768,196 -> B,196,768
        x = self.proj(x).flatten(2).transpose(1, 2)  
        x = self.norm(x) # 归一化
        return x
    
class Attention(nn.Module):
    def __init__(self, 
                 dim,   # 输入的token维度，768
                 num_heads=8,   # 注意力头数为 8
                 qkv_bias=False,    # 生成QKV的时候是否使用偏置
                 qk_scale=None,   # qk的缩放比例,如果None则使用1/sqrt(head_dim)
                 atte_drop_ration=0.,   # 注意力丢弃率
                 proj_drop_ration=0.):    # 投影丢弃率
        super().__init__()
        self.num_heads = num_heads  # 注意力头数
        head_dim = dim // num_heads  # 每个注意力头的维度
        self.scale = qk_scale or head_dim ** -0.5  # 缩放比例
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)  # 通过全连接层生成QKV，为了并行运算提高效率。生成QKV的线性层
        self.att_drop = nn.Dropout(atte_drop_ration)  # 注意力丢弃
        self.proj_drop = nn.Dropout(proj_drop_ration)  # 投影丢弃
        # 将每个head得到的输出进行concat拼接，然后通过线性变换映射回原本的嵌入dim
        self.proj = nn.Linear(dim, dim)  # 最后的线性投影

    def forward(self, x):
        B, N, C = x.shape  # B:batch size N:num_patches+1 C:embed_dim  这个1为clstoken
        # B N 3*C -> B,N,3,num_heads,C//self.num_heads
        # B,N,3,num_heads,C//self.num_heads-> 3,B,num_heads,N,C//self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #方便之后做运算
        # 用切片拿到QKV,形状B,num_heads,N,C//self.num_heads
        q,k,v = qkv[0], qkv[1], qkv[2]  # 各自的形状为B,num_heads,N,C//num_heads
        # 计算qk的点积，并进行缩放，得到注意力分数
        # Q: [B,num_heads,N,C//self.num_heads]
        # k.transpose(-2,-1)  K: [B,num_heads,N,C//self.num_heads] -> [B,num_heads,C//self.num_heads,N]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 计算注意力得分矩阵  B,num_heads,N,N
        attn = attn.softmax(dim=-1)  # (使每行的和为1) 对最后一个维度进行softmax，得到注意力权重
        x = (attn @ v)  # 将注意力权重与V相乘，得到加权后的值  B,num_heads,N,C//self.num_heads
        # 注意力权重对v进行加权求和
        # attn @ v: B,num_heads,N,C//self.num_heads
        # transpose: B,N,num_heads,C//self.num_heads
        # reshape: B,N,C 将最后两个维度进行拼接，合并多个头输出，回到总的嵌入维度
        x = x.transpose(1, 2).reshape(B, N, C)  # 将多个头的输出拼接起来  B,N,C
        # 通过线性变换映射回原本的嵌入dim
        x = self.proj(x)  # 线性投影
        x = self.proj_drop(x)  # 投影丢弃

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop_ration=0.):
        # in_features 输入特征维度 hidden_features 隐藏层特征维度 通常为in_feature的4倍 out_features 输出特征维度 通常与输入维度相等
        super().__init__()
        # or的语法，如果没有传入hidden_features和out_features，则分别赋值为in_features
        out_features = out_features or in_features  # 输出特征维度
        hidden_features = hidden_features or in_features  # 隐藏层特征维度
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二个全连接层
        self.drop = nn.Dropout(drop_ration)  # 丢弃层

    def forward(self, x):
        x = self.fc1(x)  # 输入通过第一个全连接层
        x = self.act(x)  # 激活函数
        x = self.drop(x)  # 丢弃
        x = self.fc2(x)  # 输入通过第二个全连接层
        x = self.drop(x)  # 丢弃
        return x
    
class Block(nn.Module):
    def __init__(self, 
                 dim,   #每个token的维度 768
                 num_heads,     #注意力头数 12
                 mlp_ratio=4,   #mlp中隐藏层维度与输入维度的比例，4倍
                 qkv_bias=False,        
                 qkv_scale=None,        # qkv的缩放比例
                 drop_ratio=0.,         # 多头子注意力机制最后的linear后使用的dropout
                 attn_drop_ratio=0.,     # 注意力丢弃率，生成qkv后的dropout
                 drop_path_ratio=0.,     # 随机深度丢弃率,会用在encoder的残差连接上
                 act_layer=nn.GELU,     # 激活函数
                 norm_layer=nn.LayerNorm,): # 归一化层
        super(Block,self).__init__()
        self.norm1 = norm_layer(dim)  # encoder block的第一个layer norm
        # 多头自注意力机制进行实例化
        self.attn = Attention(   
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_scale=qkv_scale,
            atte_drop_ration=attn_drop_ratio,
            proj_drop_ration=drop_ratio
        )
        # 如果drop_path_ratio大于0，则使用DropPath，否则使用恒等映射(不做改变)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()  # 随机深度丢弃
        self.norm2 = norm_layer(dim)  # encoder block的第二个layer norm
        # MLP部分
        mlp_hidden_dim = int(dim * mlp_ratio)  # mlp中隐藏层的维度
        # 定义MLP层，传入dim = mlp的输入维度
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop_ration=drop_ratio
        )   

    def forward(self, x):
        # 残差连接 + 多头自注意力机制
        x = x + self.drop_path(self.attn(self.norm1(x)))  
        # 残差连接 + MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))  
        return x
    
class VisonTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_c=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12, 
                 num_heads=12, 
                 mlp_ratio=4.0,
                 qkv_bias=True, 
                 qkv_scale=None,
                 reprentation_size=None,
                 distilled=False,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 embed_layer=PatchEmbedding,
                 norm_layer=nn.LayerNorm,
                 act_layer=None):
        super(VisonTransformer,self).__init__()
        self.num_classes = num_classes  # 分类类别数
        self.num_features = self.embed_dim = embed_dim  # 特征维度
        self.num_tokens = 2 if distilled else 1  # token数量，是否使用蒸馏token
        # 设置一个较小的参数，防止除0
        norm_layer = norm_layer or partial(nn.LayerNorm,eps = 1e-6)  # 归一化层
        act_layer = act_layer or nn.GELU  # 激活函数层
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_c=in_c,
            embed_dim=embed_dim,
            norm_layer=None
        )  # patch嵌入层
        num_patches = self.patch_embed.num_patches  # 计算patch数量
        # 分类token和蒸馏token
        # 使用nn.Parameter定义可学习的参数,初始化为0,第一个维度为1表示batch size维度,第二个维度为token数量,第三个维度为嵌入维度
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 分类token
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None  # 蒸馏token
        # 位置编码,pos_embed 大小与concat拼接后的大小一致，197，768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  
        self.pos_drop = nn.Dropout(p=drop_ratio)  # 位置编码丢弃
        # 创建Transformer编码器块
        # 根据传入的drop_path_ratio线性变化生成一个丢弃率列表
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # 随机深度丢弃率线性变化
        # 使用nn.Sequential将多个Block模块串联起来，形成一个深度为depth的Transformer编码器
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qkv_scale=qkv_scale,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                drop_path_ratio=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])  
        self.norm = norm_layer(embed_dim)  # 最后的归一化层

        '''
         表示层
         参数：
             reprentation_size(整数或None)：表示层的维度大小。如果为None，则不使用表示层。
             distilled(布尔值)：是否使用蒸馏token。
             
         返回：
             如果reprentation_size不为None且不使用蒸馏token，则创建一个
             表示层(pre_logits)，包含一个线性层和Tanh激活函数。
             否则，表示层为恒等映射。
        '''

        if reprentation_size and not distilled:
            self.has_logits = True # 是否有logits层
            self.num_features = reprentation_size # 特征维度变为表示层的维度
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, reprentation_size)),
                ('act', nn.Tanh())
            ])) # 表示层
        else:
            self.has_logits = False 
            self.pre_logits = nn.Identity() # 恒等映射

        # 分类头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # 蒸馏头
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head_dist = None
        # 初始化参数
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # 位置编码初始化 
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)  # 蒸馏token初始化
        
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # 分类token初始化
        self.apply(_init_vit_weights)  # 权重初始化

    def forward_features(self, x):
        # 提取特征 B C H W -> B num_patches embed_dim
        x = self.patch_embed(x)  # patch嵌入
        # 扩展分类token和蒸馏token的维度以匹配batch size
        # 1,1,768->B,1,768
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)  # 分类token扩展
        # 如果存在，则拼接dist_token和cls_token
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)  # B 197 768 在维度1上拼接
        # 否则只拼接cls_token和输入的patch特征x
        else:
            dist_token = self.dist_token.expand(x.shape[0], -1, -1)  # 蒸馏token扩展
            x = torch.cat((cls_tokens, dist_token, x), dim=1)  # 拼接分类token、蒸馏token和patch嵌入
       
        x = x + self.pos_embed  # 添加位置编码
        x = self.pos_drop(x)  # 位置编码丢弃

        x = self.blocks(x)  # Transformer编码器块
        x = self.norm(x)  # 最后的归一化层

        if self.dist_token is None:     #dist_token不存在,提取cls_token对应输出
            return self.pre_logits(x[:, 0])  
        else:
            return x[:, 0], x[:, 1]  # 返回分类token和蒸馏token的表示

    def forward(self, x):
        x = self.forward_features(x)
        # 前向传播
        if self.head_dist is not None:  
            x, x_dist = self.head(x[0],self.head_dist(x[1]))  
            # 如果使训练模式且不是脚本模式
            if self.training and not torch.jit.is_scripting():
                return x, x_dist
        else:  # 使用蒸馏token
            x = self.head(x)
        return x
    

    def _init_vit_weights(m):
        # 初始化ViT的权重
        # 判断模块m是否使nn.linear
        if isinstance(m, nn.Linear):   
            nn.init.trunc_normal_(m.weight, std=0.01)  # 截断正态分布初始化
            if m.bias is not None:  #如果线性层存在偏置
                nn.init.zeros_(m.bias)  # 偏置初始化为0

        elif isinstance(m, nn.Conv2d):  # 判断模块m是否使nn.LayerNorm
            nn.init.kaiming_normal_(m.weight, mode='fan_out')  # 对卷积层的权重做一个初始化，Kaiming正态分布初始化
            if m.bias is not None:
                nn.init.zeros_(m.bias)  # 偏置初始化为0
        
        elif isinstance(m, nn.LayerNorm):  # 判断模块m是否使nn.LayerNorm
            nn.init.zeros_(m.bias)  # 偏置初始化为0
            nn.init.ones_(m.weight)  # 权重初始化为1

    def vit_base_patch16_224(num_classes:int=1000, pretrained=False):
        # 创建一个ViT-Base模型，patch大小为16，输入图像大小为224x224
        model = VisonTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            reprentation_size=None,
            num_classes=num_classes,
        )
        return model
```

