+++
date = '2025-11-28T13:40:43+08:00'
title = 'nnUNet代码底层学习-训练'

+++



## run_training.py

找到 `run_training_entry()` 路由进入函数，print 一下配置，如下图所示。

![44](images/44.png)

路由进`run_training()` 函数，找到处理折数参数的代码。

```python
if fold != 'all':
    try:
        fold = int(fold)
    except ValueError as e:
        print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
        raise e
```

`all` 在这里的意思是实验所设定的总折数，可以选择跑满。

问题：怎么理解交叉验证折数 fold 呢？

答：交叉验证中的折数，本质是将完整的数据集切分成的**互不重叠、大小相近的子集数量 / 单个子集的标识**：一方面，总折数（K） 是全局规则，决定把数据集分成 K 份，交叉验证的核心逻辑就是依次用其中 1 份做验证集、剩余 K-1 份做训练集，跑完 K 轮完成一次完整验证；另一方面，代码中单独定义的 `fold` 是单次执行的选择器，它既可以是 `all`（代表跑满 K 轮验证），也可以是具体数字，代表只针对某一个子集（某一折）执行训练 + 验证。

`continue_training and only_run_validation`，继续训练还是仅验证，路由进函数 `maybe_load_checkpoint` ，这里要区分是不是使用了 nnUNet 的网络框架，如果是自己提供预训练权重，则满足 `pretrained_weights_file is not None`。

```python
if not only_run_validation:
    nnunet_trainer.run_training()
```

路由进函数`run_training` 位于 `nnUNetTrainer.py` 文件

## nnUNetTrainer.py

```python
self.on_train_start()  #初始化超参数、网络结构、损失函数、加载数据
```

路由进`on_train_start`函数，找到`initialize`函数，路由进去，发现其功能是初始化网络结构、优化器参数和损失函数。找到了核心网络结构代码部分如下：

```python
self.network = self.build_network_architecture(
    self.configuration_manager.network_arch_class_name,
    self.configuration_manager.network_arch_init_kwargs,
    self.configuration_manager.network_arch_init_kwargs_req_import,
    self.num_input_channels,
    self.label_manager.num_segmentation_heads,
    self.enable_deep_supervision
).to(self.device)
```

路由进 `build_network_architecture`函数，发现核心代码藏在`get_network_from_plans`函数：

```python
return get_network_from_plans(
    architecture_class_name,
    arch_init_kwargs,
    arch_init_kwargs_req_import,
    num_input_channels,
    num_output_channels,
    allow_init=True,
    deep_supervision=enable_deep_supervision)
```

进一步路由进去`get_network_from_plans`发现里面是直接调用了nnUNet封装好的网络

```python
import dynamic_network_architectures
```

但是这个并不能进一步修改网络结构，所以我在[MIC-DKFZ/dynamic-network-architectures](https://github.com/MIC-DKFZ/dynamic-network-architectures)找到了网络包，下载后可以看到具体的网络结构，这里我们暂时按默认初始化。

```python
# 定义优化器和学习率
self.optimizer, self.lr_scheduler = self.configure_optimizers()

# 定义损失函数
self.loss = self._build_loss()
```

路由进`_build_loss`函数，发现有两个损失函数，一个是`DC_and_BCE_loss`，一个是`DC_and_CE_loss`，具体的等后面研究损失函数时再进行分析。

```python
# 加载数据
self.dataloader_train, self.dataloader_val = self.get_dataloaders()
```

这部分内容比较重要，划分训练集和验证集。路由进`get_dataloaders`函数

```python
(
    rotation_for_DA,
    do_dummy_2d_data_aug,
    initial_patch_size,
    mirror_axes,
) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
```

这块不建议改，除非有很深刻理解。

```python
# training pipeline
tr_transforms = self.get_training_transforms(
    patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
    use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
    is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
    regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
    ignore_label=self.label_manager.ignore_label)

# validation pipeline
val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                is_cascaded=self.is_cascaded,
                                                foreground_labels=self.label_manager.foreground_labels,
                                                regions=self.label_manager.foreground_regions if
                                                self.label_manager.has_regions else None,
                                                ignore_label=self.label_manager.ignore_label)
```

这块是训练集和验证集数据扩增的方式。

问题：`self.on_train_epoch_start()`和`on_validation_epoch_start()`在哪里不一样，自行查找资料

答：整体执行流程如下

```plaintext
1.触发 on_train_epoch_start()：
   → 模型切train() → 更新LR → 打印epoch和LR日志
执行训练迭代：遍历训练集，前向+反向+参数更新（开启梯度）
2.触发 on_validation_epoch_start()：
   → 模型切eval()
执行验证迭代：遍历验证集，仅前向计算（关闭梯度）
验证结束：记录ACC/Dice等指标，回到下一轮训练
```

比较重要的代码块

```python
for epoch in range(self.current_epoch, self.num_epochs):
    self.on_epoch_start()
    self.on_train_epoch_start()
    train_outputs = []
    for batch_id in range(self.num_iterations_per_epoch):
        train_outputs.append(self.train_step(next(self.dataloader_train)))
    self.on_train_epoch_end(train_outputs)
    with torch.no_grad():
        self.on_validation_epoch_start()
        val_outputs = []
        for batch_id in range(self.num_val_iterations_per_epoch):
            val_outputs.append(self.validation_step(next(self.dataloader_val)))
        self.on_validation_epoch_end(val_outputs)
    self.on_epoch_end()
```

路由进`train_step`函数

```python
output = self.network(data)
    # del data
    l = self.loss(output, target)
```

这部分是前向传播，输入图像，输出预测结果，然后计算预测值和真实值的差距。

```python
if self.grad_scaler is not None:
    self.grad_scaler.scale(l).backward()
    self.grad_scaler.unscale_(self.optimizer)
    torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
    self.grad_scaler.step(self.optimizer)
    self.grad_scaler.update()
```

这部分就是很重要的反向传播过程了，会有几个问题，作出解答。

问题：为什么要 Scale ？

答：FP16 能表示的最小正数大约是 $6 \times 10^{-5}$。如果梯度非常小（比如 $10^{-6}$），在 FP16 下就会变成 0，模型就学不到东西了。在反向传播前，把 Loss 放大很多倍，这样梯度也会跟着放大。

问题：为什么要 Unscale？

答：刚才我们把梯度放大了，现在要更新参数前，得把它还原回去， unscale 在这里出现的主要目的是为了下一步的梯度裁剪。要裁剪梯度，必须知道梯度的真实大小，而不是放大后的大小。

问题：clip_grad_norm的作用？

作用： 如果计算出的梯度向量太长（模长超过 12），就强行把它缩短到 12。 医学图像数据波动大，有时会导致 Loss 剧烈震荡，产生巨大的梯度把模型权重变成 NaN。12 是 nnU-Net 作者经过大量实验得出的经验值。

问题：scaler.step 和 scaler.update的作用？

step：既然梯度在第一步被放大了，scaler.step 会在内部自动把梯度除回去（unscale），然后应用到权重上。

update：根据这一轮是否发生了梯度溢出（Infinity/NaN），动态调整下一轮的放大倍数。如果这轮没事，下轮尝试放大更多倍；如果溢出了，下轮就缩小倍数。

路由进 `validation_step`函数

```python
if self.enable_deep_supervision:
        output = output[0]
        target = target[0]
```

训练时，我们用了多尺度的输出（output）来辅助训练。而此处验证， 我们只关心全分辨率的那张图（通常是第 0 个）。这里直接取 [0]，丢弃其他的。

```python
if self.label_manager.has_regions:
        predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
```

基于区域的分割 (Region-based)，这些区域在空间上是重叠的。

```python
else:
        # 找到概率最大的类别
        output_seg = output.argmax(1)[:, None]
        # 准备一个全 0 的 One-Hot 容器
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        # 把预测结果填进去
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg
```

普通多分类，概率最大的标签就是1，其他赋0，这步有 one-hot

```python
if self.label_manager.has_ignore_label:
    if not self.label_manager.has_regions:
        mask = (target != self.label_manager.ignore_label).float()
        # CAREFUL that you don't rely on target after this line!
        target[target == self.label_manager.ignore_label] = 0
    else:
        if target.dtype == torch.bool:
            mask = ~target[:, -1:]
        else:
            mask = 1 - target[:, -1:]
        # CAREFUL that you don't rely on target after this line!
        target = target[:, :-1]
else:
    mask = None
```

数据集不完整的时候，只标个未知的情况下，直接剔除。

```python
tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

tp_hard = tp.detach().cpu().numpy()
fp_hard = fp.detach().cpu().numpy()
fn_hard = fn.detach().cpu().numpy()
```

TP (True Positive): 预测对了的前景像素数。

FP (False Positive): 把背景错认成前景的像素数（误报）。

FN (False Negative): 把前景漏掉的像素数（漏报）。

但是需要注意，这里还没有计算出 Dice 分数，真正计算是在`on_validation_epoch_end`函数里。

