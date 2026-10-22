+++
date = '2025-11-26T13:40:43+08:00'
title = 'ViT-AI-复现'
slug = 'cerebral-hemodynamics-neural-operators'
categories = ["科学机器学习"]
tags = ["神经算子", "论文复现"]
description = "使用 GNOT 与 VANO 复现稀疏数据下的脑血流动力学重建。"

+++

研究对象为Kaczmarek等发表于Computers in Biology and Medicine的论文：[Reconstructing cerebral hemodynamics from sparse data using Neural Operator Transformers](https://www.sciencedirect.com/science/article/pii/S0010482525008431)

本次在CFFF平台上使用作者公开GNOT和VANO权重，完成36,090条血管样本的正向评估和171个仿真网络的无噪声稀疏反演。反演平均相对L₂为3.303318%，病例间总体标准差为2.211696个百分点，与原文3.32±2.21%接近。核心无噪声整体精度已基本复现。

![image-20261022104403450](/images/image-20261022104403450.png)
