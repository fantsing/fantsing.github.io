+++
date = '2025-11-22T13:40:43+08:00'
title = 'nnUNet代码底层学习-预处理'

+++

问题1：spacing从（0.5，0.5，0.5）-> (1，1，1)发生了什么变化？

图像被降采样，分辨率降低 ，体素变大了一倍。

公式为：新像素个数 × 新的间距  =  原来的间距 × 原来的像素个数

问题2：当前是怎么做crop的？要修改crop应该怎么做

用 create_nonzero_mask 非零的位置加入掩码，用 crop_to_nonzero 裁剪数据，只保留包含非零区域的 bounding box。目的是去掉大量的黑色背景。最后通过np.get_bbox函数获取分割图像的前景区域，并将所有非零值赋0，为0的值赋-1。

修改crop的方法：有时候想给目标周围多留几个像素，就需要扩大边界，支持 margin 是一种修改，还有 target_shape 支持裁剪后固定的某个形状，可以 padding 可以截断。

问题3：使用的哪个归一化函数?

直接运行程序，print 打印显示使用的 'ZScoreNormalization' 归一化函数 

## default_resampling.py

路径：nnunetv2/preprocessing/resampling/default_resampling.py

- get_do_separate_z：判断是否需要处理 z 轴，具体为 max(spacing) ÷ min(spacing) ，再和 anisotropy_threshold 比较，如果前者大，则单独处理 z 轴；如果后者大，则不单独处理。这里提到一个“各向异性”，就是xyz的差距是否过大。

```python
def get_do_separate_z(
        spacing: Union[Tuple[float, ...], List[float], np.ndarray],
        anisotropy_threshold=ANISO_THRESHOLD):
    # 计算“最大像素间距 ÷ 最小像素间距”，如果大于阈值，就单独处理Z轴
    do_separate_z = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate_z
```

- get_lowres_axis：找到低分辨率的轴

```python
def get_lowres_axis(
        new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]):
    # 找到哪个轴是各向异性的（那个方向就是精度最低的）
    axis = np.where(max(new_spacing) / np.array(new_spacing) == 1)[0]  # find which axis is anisotropic
    return axis
```

- compute_new_shape：比较实用的函数，根据新旧像素间距，计算数据缩放后的新尺寸。其中 old_shape 是数据原来的像素尺寸；old_spacing 是数据原来的像素间距；new_spacing 是想要的新像素间距（；new_shape 是计算后的数据新像素尺寸。

```python
def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    # 检查1：旧 spacing 方向数，必须和旧 shape 方向数一样
    assert len(old_spacing) == len(old_shape)
    # 检查2：新 shape 方向数，必须和旧 shape 方向数一样
    assert len(old_shape) == len(new_spacing)
    # 公式：计算每个方向的新像素数，最后转成整数
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape
```

举个例子：

原来的尺寸 `old_shape = (128, 128, 64)`（X=128像素，Y=128像素，Z=64像素）；
原来的间距 `old_spacing = (2, 2, 4)`（X/Y轴每像素2毫米，Z轴每像素4毫米）；
想要的新间距 `new_spacing = (1, 1, 2)`（X/Y轴每像素1毫米，Z轴每像素2毫米）。

公式为：实际物理大小 = 像素个数 × 像素间距；缩放后，实际物理大小不能变，所以：新像素个数 × 新的间距  =  原来的间距 × 原来的像素个数 。然后利用这个公式逐个方向计算（结果取整数），返回 `new_shape` 的值。

- determine_do_sep_z_and_axis：决定要不要单独处理低精度轴？以及具体处理哪个轴？

```python
def determine_do_sep_z_and_axis(
        force_separate_z: bool, # 手动强制执行“是否单独处理z轴”
        current_spacing,    #数据当前的像素间距
        new_spacing,        #数据想要改成的新像素间距
        separate_z_anisotropy_threshold: float = ANISO_THRESHOLD) -> Tuple[bool, Union[int, None]]:
    # 判断是否强制
    if force_separate_z is not None:
        do_separate_z = force_separate_z    # 直接用指定的
        # 如果指定要处理
        if force_separate_z:
            # 找现在数据的低精度轴
            axis = get_lowres_axis(current_spacing)
        else:
            axis = None
    else:
        # 情况1：current_spacing 是各向异性，则找现在的低精度轴
        if get_do_separate_z(current_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(current_spacing)
        # 情况2：current_spacing 没问题，但 new_spacing 是各向异性，则找新间距的低精度轴
        elif get_do_separate_z(new_spacing, separate_z_anisotropy_threshold):
            do_separate_z = True
            axis = get_lowres_axis(new_spacing)
        # 情况3：现在和新间距都没问题，则不处理
        else:
            do_separate_z = False
            axis = None

    if axis is not None:    # 如果找到了要处理的轴
        if len(axis) == 3:  # 3个轴都是低精度
            do_separate_z = False   # 不处理
            axis = None
        elif len(axis) == 2:    # 2个轴是低精度
            do_separate_z = False   # 不处理
            axis = None
        else:               # 只有1个轴是低精度
            axis = axis[0]  # 取这个轴的编号
    return do_separate_z, axis
```

- resample_data_or_seg_to_spacing：把数据（或分割标签）按照新旧像素间距，精准缩放成新的尺寸，还会根据之前判断的 “是否单独处理低精度轴” 来选择最优的缩放方式，保证缩放后数据不失真、物理大小不变。

```python
def resample_data_or_seg_to_spacing(data: np.ndarray,   #要缩放的原始数据
                                    current_spacing: Union[Tuple[float, ...], List[float], np.ndarray], #数据当前的像素间距
                                    new_spacing: Union[Tuple[float, ...], List[float], np.ndarray], #想要的新像素间距
                                    is_seg: bool = False,   #是不是分割标签的开关
                                    order: int = 3, #普通数据的缩放算法等级（默认3，代表用三次插值，缩放后细节保留得好，适合影像数据）
                                    order_z: int = 0,  #单独处理 Z 轴时的缩放算法等级（默认0，代表最近邻插值，适合低精度轴，避免失真）
                                    force_separate_z: Union[bool, None] = False,    #强制处理z轴
                                    separate_z_anisotropy_threshold: float = ANISO_THRESHOLD): #判断各向异性的阈值
    # 是否单独处理？处理哪个轴？
    do_separate_z, axis = determine_do_sep_z_and_axis(force_separate_z, current_spacing, new_spacing,
                                                      separate_z_anisotropy_threshold)
    # 检查数据格式是否满足4维
    if data is not None:
        assert data.ndim == 4, "data must be c x y z"
    # 计算数据的新尺寸
    shape = np.array(data.shape)    # 先拿到原始数据的尺寸
    new_shape = compute_new_shape(shape[1:], current_spacing, new_spacing)
    # 执行缩放
    data_reshaped = resample_data_or_seg(data, new_shape, is_seg, axis, order, do_separate_z, order_z=order_z)
    return data_reshaped
```

- resample_data_or_seg：按新尺寸，用合适的算法，真正把数据（或分割标签）缩放到位，支持单独处理低精度轴，保证缩放后数据不失真。

  参数具体如下：

  `data`：要缩放的原始数据；

  `new_shape`：要缩放成的新尺寸（比如 `256×256×128`）；

  `is_seg`：是不是分割标签（`True`=是标签，`False`=普通影像）；

  `axis`：要单独处理的低精度轴（比如 `2`=Z轴，`None`=不单独处理）；

  `order`：普通缩放的算法等级（默认3，三次插值，保细节）；

  `do_separate_z`：要不要单独处理低精度轴（`True`=要，`False`=不要）；

  `order_z`：单独处理轴时的算法（默认0，最近邻插值，保精度）；

  `dtype_out`：输出数据的格式（默认和原始数据一样）。



## cropping.py

```python
def create_nonzero_mask(data):
    """
    :param data:形状为 (C, X, Y, Z) 或 (C, X, Y) 的多通道医学图像 C = 通道数
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), 
    # 用第 0 个通道初始化掩码：非零的位置标记为 True
    nonzero_mask = data[0] != 0
    # 遍历剩下的通道，将它们非零的位置加入掩码
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
        # |= 的意思是：mask = mask OR (data[c] != 0)
    '''
    返回值:
        一个布尔值数组，shape 为 (X, Y, Z)        
        True 代表该位置至少有一个通道不是0（有内容）
        False 代表所有通道都是0（背景）
    '''
    # 填洞 binary_fill_holes，应该是为了保证实心
    return binary_fill_holes(nonzero_mask)
```



## default_normalization_schemes.py

路径：nnunetv2/preprocessing/normalization/default_normalization_schemes.py

```python
class ImageNormalization(ABC)...
class ZScoreNormalization(ImageNormalization)...
class CTNormalization(ImageNormalization)...
class NoNormalization(ImageNormalization)...
class RescaleTo01Normalization(ImageNormalization)...
class RGBTo01Normalization(ImageNormalization)...
```

![41](images/41.png)

这部分就是归一化函数的类定义，具体情况具体分析
