## 1. 初始化W

```py
W = np.random.randn(*shape) * (2 / reduce(lambda x, y: x * y, shape[1:]) ** 0.5)
```

### reduce() 函数语法：

```
reduce(function, iterable[, initializer])
```

**参数**

- function -- 函数，有两个参数
- iterable -- 可迭代对象
- initializer -- 可选，初始参数

**返回值**

返回函数计算结果。



## 2. 填充

```
np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
```

```
np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
```



### np.pad()

**1）语法结构**

> pad(array, pad_width, mode, **kwargs)

**2）参数解释**

> array —— 表示需要填充的数组；
>
> pad_width —— 表示每个轴（axis）边缘需要填充的数值数目。
> 		参数输入方式为：（(before_1, after_1), … (before_N, after_N)），其中(before_1, after_1)表示第1轴两边缘分别填充before_1个和after_1个数值。取值为：{sequence, array_like, int}
>
> mode —— 表示填充的方式（取值：str字符串或用户提供的函数）,总共有11种填充模式

**3) **

**填充方式**

> ‘constant’——表示连续填充相同的值，每个轴可以分别指定填充值，constant_values=（x, y）时前面用x填充，后面用y填充，缺省值填充0
>
> ‘edge’——表示用边缘值填充
>
> ‘linear_ramp’——表示用边缘递减的方式填充
>
> ‘maximum’——表示最大值填充
>
> ‘mean’——表示均值填充
>
> ‘median’——表示中位数填充
>
> ‘minimum’——表示最小值填充
>
> ‘reflect’——表示对称填充
>
> ‘symmetric’——表示对称填充
>
> ‘wrap’——表示用原数组后面的值填充前面，前面的值填充后面


函数后面的（**kwargs）


在 numpy 版本 1.12.0 之后，`einsum`加入了 `optimize` 参数，用来优化 `contraction` 操作，对于 `contraction` 运算部分，操作的数组包含三个或三个以上，`optimize` 参数设置能提高计算效率，减小内存占比