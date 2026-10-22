+++
date = '2026-02-10T13:40:43+08:00'
title = 'FNO-Task'
categories = ["科学机器学习"]
tags = ["FNO", "PDE"]
description = "浅水波方程的无量纲化推导与相关数学背景。"

+++

# 浅水波方程的无量纲化推导

## 引言与背景

### 涉及的数学领域

属于流体动力学和偏微分方程领域，具体处理的是浅水波方程（Shallow Water Equations, SWE）。浅水波方程是描述具有自由表面的不可压缩流体运动的一组方程，广泛应用于海洋学、气象学、河流动力学等。

### 方程概述
浅水波方程基于质量守恒和动量守恒原理，通常形式为（二维情形）：

1. **连续性方程**（质量守恒）：
\[
\frac{\partial h}{\partial t} + \frac{\partial (uh)}{\partial x} + \frac{\partial (vh)}{\partial y} = 0
\]

2. **x方向动量方程**：
\[
\frac{\partial (uh)}{\partial t} + \frac{\partial}{\partial x}\left(u^2h + \frac{1}{2}gh^2\right) + \frac{\partial (uvh)}{\partial y} = 0
\]

3. **y方向动量方程**：
\[
\frac{\partial (vh)}{\partial t} + \frac{\partial}{\partial y}\left(v^2h + \frac{1}{2}gh^2\right) + \frac{\partial (uvh)}{\partial x} = 0
\]

其中：
- \(h(x,y,t)\)：水深（自由表面高度）
- \(u(x,y,t), v(x,y,t)\)：流速在x、y方向的分量
- \(g\)：重力加速度
- \(t\)：时间
- \(x, y\)：空间坐标

### 推导目标
本推导的目的是对浅水波方程进行**标准化**和**无量纲化**处理，以便：
1. 消除物理量的量纲，便于理论分析
2. 识别控制流动的关键无量纲参数
3. 便于数值计算中的尺度控制

---

## 标准化处理

### 标准化方法介绍

(1) Z-Score 标准化（标准差标准化）
\[
x_{\text{standardized}} = \frac{x - \mu}{\sigma}
\]
其中：
- \(\mu = \bar{x}\) 为样本均值
- \(\sigma = \sqrt{\mathrm{Var}(x)}\) 为样本标准差

**作用**：将数据变换为均值为0、标准差为1的分布，适用于数据近似正态分布的情况。

(2) Min-Max 标准化（极差标准化）
\[
x_{\text{standardized}} = \frac{x - \min(x)}{\max(x) - \min(x)}
\]
**作用**：将数据线性映射到 \([0, 1]\) 区间，保留原始数据分布形状。

(3) 微分的线性性质

若 \(a, b\) 为常数，\(x\) 为变量，则：
\[
\partial (ax + b) = a \partial x
\]
**说明**：常数加法不影响微分，常数乘法会线性缩放微分。

---

## 浅水波方程的改写

### 引入平均水深假设
假设水深可以分解为**平均水深** \(H\) 与**波动部分** \(h\)：
\[
h_{\text{raw}} = h + H
\]
其中：
- \(H = \bar{h} = 100\)（示例值，表示平均水深）
- \(\bar{u} = \bar{v} = 0\)（假设平均流速为零）

### 代入原方程
将 \(h_{\text{raw}} = h + H\) 代入浅水波方程，得到改写后的方程组（式1）：
\[
\begin{aligned}
&\frac{\partial h}{\partial t} + H\frac{\partial u}{\partial x} + \frac{\partial (uh)}{\partial x} + H\frac{\partial v}{\partial y} + \frac{\partial (vh)}{\partial y} = 0 \\
&\frac{\partial (uh)}{\partial t} + \frac{\partial}{\partial x}\left(u^2h + \frac{1}{2}gh^2\right) + \frac{\partial (uvh)}{\partial y} \\
&\quad + H\left( \frac{\partial u}{\partial t} + \frac{\partial u^2}{\partial x} + g\frac{\partial h}{\partial x} + \frac{\partial (uv)}{\partial y} \right) = 0 \\
&\frac{\partial (vh)}{\partial t} + \frac{\partial}{\partial y}\left(v^2h + \frac{1}{2}gh^2\right) + \frac{\partial (uvh)}{\partial x} \\
&\quad + H\left( \frac{\partial v}{\partial t} + \frac{\partial v^2}{\partial y} + g\frac{\partial h}{\partial y} + \frac{\partial (uv)}{\partial x} \right) = 0
\end{aligned}
\]

**改写目的**：
- 将波动部分 \(h\) 与平均部分 \(H\) 分离
- 便于后续进行无量纲化处理
- 在数值模拟中常采用此类分解以提高计算稳定性

---

## 无量纲化处理

### 特征量的定义
为进行无量纲化，需定义**特征尺度**：

1. **特征长度** \([L]\)：
\[
[L] = \sqrt{\mathrm{Var}(h)}
\]
表示水深波动的标准差，反映水深变化的典型幅度。

2. **特征速度** \([V]\)：
\[
[V] = \frac{\sqrt{\mathrm{Var}(u)} + \sqrt{\mathrm{Var}(v)}}{2}
\]
表示流速波动的平均标准差，反映流速变化的典型幅度。

3. **特征时间** \(T\) 和 **特征空间长度** \(D\)：
- \(T\)：时间尺度（如波浪周期）
- \(D\)：空间尺度（如波长）

### 无量纲变量的定义
定义无量纲变量（用上标 \(*\) 表示）（式2）：
\[
\begin{aligned}
h &= h^* [L] + H \\
u &= u^* [V] \\
v &= v^* [V] \\
t &= t^* T \\
x &= x^* D \\
y &= y^* D
\end{aligned}
\]
**解释**：
- 每个物理量 = 无量纲数 × 特征量 + 常数项
- 这样处理后，\(h^*, u^*, v^*, t^*, x^*, y^*\) 均为无量纲量

### 代入方程进行无量纲化
将上述变换代入式(1)，详细推导如下：

以连续性方程为例（第一行）：
\[
\frac{\partial h}{\partial t} + H\frac{\partial u}{\partial x} + \frac{\partial (uh)}{\partial x} + H\frac{\partial v}{\partial y} + \frac{\partial (vh)}{\partial y} = 0
\]

逐项替换：

1. \(\frac{\partial h}{\partial t} = \frac{\partial (h^*[L] + H)}{\partial (t^* T)} = \frac{[L]}{T} \frac{\partial h^*}{\partial t^*}\)

2. \(H\frac{\partial u}{\partial x} = H \frac{\partial (u^*[V])}{\partial (x^* D)} = \frac{H[V]}{D} \frac{\partial u^*}{\partial x^*}\)

3. \(\frac{\partial (uh)}{\partial x} = \frac{\partial (u^*[V] \cdot (h^*[L] + H))}{\partial (x^* D)} = \frac{[V][L]}{D} \frac{\partial (u^* h^*)}{\partial x^*} + \frac{[V]H}{D} \frac{\partial u^*}{\partial x^*}\)

4. \(y\) 方向类似。

合并后得到无量纲化的连续性方程（式(3)第一行）：
\[
\frac{[L]}{T} \frac{\partial h^*}{\partial t^*} + \frac{[V][L]}{D} \left( \frac{\partial (u^* h^*)}{\partial x^*} + \frac{\partial (v^* h^*)}{\partial y^*} \right) + \frac{H[V]}{D} \left( \frac{\partial u^*}{\partial x^*} + \frac{\partial v^*}{\partial y^*} \right) = 0
\]

类似地，动量方程也进行相同处理，得到式(3)的第二、三行。

---

## 推导的关键点与技巧

### 分离平均量与波动量的技巧
将水深写为 \(h_{\text{raw}} = h + H\) 的优点：
- 物理上：\(H\) 代表背景水深，\(h\) 代表波动或扰动
- 数学上：便于线性化处理，特别当 \(h \ll H\) 时可简化方程

### 无量纲化的核心思想
1. **消除量纲**：使方程不依赖于具体单位系统
2. **揭示无量纲参数**：如弗劳德数（Froude number）、雷诺数（Reynolds number）等
3. **尺度分析**：通过比较各项的量级，判断哪些项可以忽略

### 微分链式法则的应用
在变量替换时，注意：
\[
\frac{\partial}{\partial t} = \frac{\partial t^*}{\partial t} \frac{\partial}{\partial t^*} = \frac{1}{T} \frac{\partial}{\partial t^*}
\]
对空间导数同理。

---

## 数值示例

### 假设参数值
- \(H = 100 \, \text{m}\)
- \([L] = 10 \, \text{m}\)（水深波动幅度）
- \([V] = 2 \, \text{m/s}\)（流速波动幅度）
- \(T = 100 \, \text{s}\)（时间尺度）
- \(D = 1000 \, \text{m}\)（空间尺度）
- \(g = 9.8 \, \text{m/s}^2\)

### 计算无量纲方程中的系数
以连续性方程为例：
- \(\frac{[L]}{T} = 0.1\)
- \(\frac{[V][L]}{D} = 0.02\)
- \(\frac{H[V]}{D} = 0.2\)

因此无量纲连续性方程为：
\[
0.1 \frac{\partial h^*}{\partial t^*} + 0.02 \left( \frac{\partial (u^* h^*)}{\partial x^*} + \frac{\partial (v^* h^*)}{\partial y^*} \right) + 0.2 \left( \frac{\partial u^*}{\partial x^*} + \frac{\partial v^*}{\partial y^*} \right) = 0
\]

### 无量纲参数的出现
在动量方程中，会出现如 \(\frac{gT^2}{D}\)（类似弗劳德数平方的倒数）等无量纲组合，这些参数决定了流动的性质。

---

## 结论与应用

### 推导结论
1. 通过标准化和无量纲化，浅水波方程被改写为只含无量纲变量和若干无量纲系数的形式
2. 方程中的系数由特征尺度 \([L], [V], T, D\) 和物理常数 \(g\) 组合而成
3. 无量纲化后的方程更适用于理论分析和数值模拟

### 实际应用意义
1. **数值模拟**：无量纲方程计算更稳定，便于网格划分和时间步长选择
2. **模型试验**：在实验室中可通过相似准则（保持无量纲数相同）模拟实际流动
3. **参数分析**：通过无量纲参数研究流动特性（如是否考虑非线性、色散性等）

---

## 总结与扩展

### 核心思想总结
本推导的核心是**尺度分析与无量纲化**，通过：
1. 分离平均量与波动量
2. 定义合适的特征尺度
3. 将方程转化为无量纲形式
4. 识别控制流动的关键无量纲参数

### 与其他理论的联系
1. **摄动理论**：当 \(h \ll H\) 时，可进一步线性化
2. **相似理论**：无量纲方程满足相似准则，是模型试验的基础
3. **数值方法**：无量纲化是CFD（计算流体力学）中预处理的重要步骤

### 后续研究与扩展建议
1. **进一步简化**：若某些项量级很小，可忽略得到简化模型
2. **非线性处理**：本推导保留非线性项，可研究非线性效应
3. **三维扩展**：类似方法可用于三维浅水方程或全Navier-Stokes方程
4. **带底地形**：考虑实际海底地形变化，引入底高程项

### 注意事项与常见误区
1. **特征尺度选择不唯一**：不同选择会导致不同的无量纲方程形式
2. **平均水深 \(H\) 的假设**：实际中 \(H\) 可能随空间变化
3. **无量纲参数的物理意义**：需明确每个无量纲数的物理含义
4. **数值计算中的尺度**：无量纲化后，计算网格和时间步长也需相应调整

------
