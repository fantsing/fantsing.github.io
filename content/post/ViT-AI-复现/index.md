+++
date = '2025-11-26T13:40:43+08:00'
title = 'ViT-AI-复现'

+++

论文出处：

[[2505.12732\] Terrain-aware Deep Learning for Wind Energy Applications: From Kilometer-scale Forecasts to Fine Wind Fields](https://arxiv.org/abs/2505.12732)

使用 Gemini ultra 和一些大佬的博客，研究了下怎么从零搭建一个 vit 项目。

在 autodl 上租用了 A100-PCIE-40GB * 1卡 进行训练。

```
FuXi-CFD-Reproduction/
├── checkpoints/               # 模型 checkpoint 存储目录
├── configs/                   # 配置文件目录
│   └── config.yaml            # 项目配置文件
├── data/                      # 数据存储目录
│   ├── processed/             # 处理后的数据
│   └── raw/                   # 原始数据
├── logs/                      # 日志文件目录
├── scripts/                   # 脚本文件目录
│   ├── checkpoints/
│   ├── evaluate.py            # 评估脚本
│   ├── generate_dummy_data... # 生成虚拟数据脚本
│   ├── inference.py           # 推理脚本
│   ├── inspect_npz.py         # NPZ文件检查脚本
│   ├── train.py               # 训练脚本
│   └── visualize_results.py   # 结果可视化脚本
├── src/                       # 核心代码目录
│   ├── __init__.py
│   ├── dataset.py             # 数据集处理代码
│   ├── loss.py                # 损失函数定义
│   ├── model.py               # 模型结构定义
│   └── utils.py               # 工具函数
├── vis_results/               # 可视化结果目录
│   ├── vis_profiles.png       # 剖面可视化结果
│   └── vis_slices.png         # 切片可视化结果
├── README.md                  # 项目说明文档
├── requirements.txt           # 依赖包清单
└── test.py                    # 测试文件
```

## requirements.txt

```
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0

einops>=0.7.0

scipy>=1.10.0
matplotlib>=3.7.0

tensorboard>=2.14.0

pandas>=2.0.0
```

## config.yaml

```yaml
# ==============================================================================
# FuXi-CFD 复现项目配置文件
# 硬件环境: NVIDIA A100-PCIE-40GB x 1
# ==============================================================================

project:
  name: "FuXi-CFD-Reproduction"
  version: "v1.0"
  description: "Downscaling coarse wind forecasts to fine 3D wind fields using ViT"
  seed: 42

# ------------------------------------------------------------------------------
# 数据集配置 (Data Configuration)
# ------------------------------------------------------------------------------
data:
  # 已自动填入你的绝对路径
  data_dir: "/root/autodl-tmp/FuXi-CFD-Reproduction/data/raw/data"
  
  train_ratio: 0.8                 # 训练集比例
  val_ratio: 0.1                   # 验证集比例
  test_ratio: 0.1                  # 测试集比例
  num_workers: 8                   # DataLoader 线程数

  # 物理参数设置
  input_resolution_coarse: 9       # 粗网格输入尺寸 (9x9 grid)
  input_resolution_fine: 300       # 细网格输入尺寸 (300x300 grid)
  vertical_levels: 27              # 垂直层数
  
  # 输入通道定义
  in_channels: 4                   # Elevation, Roughness, u_coarse, v_coarse
  out_channels: 4                  # u_fine, v_fine, w_fine, k_fine

  # 数据归一化统计量 (可选)
  normalization:
    use_zscore: false              # 暂时关闭

# ------------------------------------------------------------------------------
# 模型架构配置 (Model Architecture)
# ------------------------------------------------------------------------------
model:
  name: "FuXi-ViT"
  img_size: 300
  patch_size: 15
  embed_dim: 768
  depth: 12
  num_heads: 12
  mlp_ratio: 4.0
  drop_rate: 0.0
  attn_drop_rate: 0.0
  
  decoder_dim: 512
  
# ------------------------------------------------------------------------------
# 训练超参数 (Training Hyperparameters)
# ------------------------------------------------------------------------------
train:
  # 显存优化设置
  batch_size: 8                    
  epochs: 300
  grad_clip: 1.0
  
  optimizer:
    name: "AdamW"
    lr: 1.0e-5
    weight_decay: 0.05
    betas: [0.9, 0.999]

  scheduler:
    name: "CosineAnnealing"
    warmup_epochs: 5
    min_lr: 1.0e-6

  # 混合精度训练 (A100 必开)
  amp:
    enabled: true
    dtype: "bfloat16"              

# ------------------------------------------------------------------------------
# 损失函数配置 (Loss Function)
# ------------------------------------------------------------------------------
loss:
  alpha_spatial: 1.0
  alpha_frequency: 0.1
  charbonnier_eps: 1.0e-6

# ------------------------------------------------------------------------------
# 日志与检查点 (Logging & Checkpoints)
# ------------------------------------------------------------------------------
logging:
  save_dir: "./checkpoints"
  log_dir: "./logs"
  save_freq: 10
  eval_freq: 1
  keep_checkpoint_max: 5
```

此处先拿师兄给的10份小数据集，跑了一下整体项目，看看能不能跑通 ，训练300个epoch后，进行可视化。

![42](images/42.png)

![43](images/43.png)

发现已经具备一定的学习能力，并且模型在训练的过程中 loss 一直在下降，只是由于数据集过小的原因导致了训练结果还有点不美观。后续拿到一定规模的数据集后，应该可以成功复现。
