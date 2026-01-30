+++
date = '2025-12-21T13:40:43+08:00'
title = 'LeetCode'

+++

## 预备知识

### Number 数字

int：1

float：1.1

complex：4+3j

Bool：True/False

运算：+（加） -（减） ×（乘） /（除） //（整除） %（取余）

取绝对值：abs(-1)=1

最大最小：max(1,2,3)=3，min(1,2,3)=1

pow(x,y) = $x^y$

sqrt(x) = $\sqrt{x}$

### List 列表

a = [1,2,3,4,5]

查找：a[0]:1      a[1]:2

增加：a.append(6)，a=[1,2,3,4,5,6]

更新：a[0]=9，a=[9,2,3,4,5,6]

删除：a.pop(0)，a=[2,3,4,5,6]，pop会返回删除的值，a.pop()，默认删除最后一个

------

常用函数：a=[2,3,4,5]

len(a):4

max(a):5

min(a):2

range(0,10) = range(10) : 0,1,2,...,9

a.reverse()，a=[5,4,3,2]

a.clear()，a:[]

------

迭代/遍历 List

```python
for x in a:
	print(x)
```

```python
for index in range(len(a):
 	print(a[index])   
```

List comrehension

```python
a = [1,2,3,4,5]
b = [i*i for i in a]
b = [1,4,9,16,25]
```

if condition

```python
a = [1,2,3,4,5]
b = [i*i if i<3 else i for i in a]
b = [1,4,3,4,5]
```

### Tuple 元组

List: [1,1.5,"abc"] 增删改查

Tuple:(1,1.5,"abc") 只有查

Tuple 查找：

a=(1,1.5,"abc")

a[0]=1,a[1]=1.5,a[2]="abc"

------

str/char

常用函数：

len(),max(),min()

in 操作

slice 切片操作，可以用于list/tuple

a=[1,2,3,4,5]

b=(1,2,3,4,5)

a[0:3]:0,1,2 [1,2,3]

b[0:3]:0,1,2 (1,2,3)

start:0,end:3,左闭右开

最后一位元素，可以用a[-1]表示

a[2:-1],取不到最后一个元素

a[2: ]:取到最后一个元素

### Set 集合

 a={1,1.5,"abc"},无序不重复

"abc" in a -> True

增：a.add(1) -> {1,1.5,"abc"}

​	a.add(2) -> {1,2,1.5,"abc"}

改：a.update(a) -> {1,2,1.5,"abc"}

a.update({4,5}) -> {1,2,1.5,"abc",4,5}

删：a.remove(2) = {1,1.5,"abc",4,5}

len/max/min

------

= | & ^

a = {1,2,3}   b = {2,5,9}

a - b 

a | b 或

a & b 与

a ^ b 不同时有的

### Dictionary 字典

dict = {"name":"learning","age":18}

key:value,利用key找value

查：

dict["name"]:learning

dict["age"]:18

增：

dict["platform"]="youtube"

dict: name age platform

改：

dict["platform"]="bilibili"

删：

dict.pop("platform")

------

遍历

```python
dict = {"name":"staywithlearning","age":18}
# "name" in dict -> True

for key in dict:
    print(key)
   
for value in dict.values():
	print(value)
    
for k,v in dict.items():
    print(k,v)
```

### String 字符串

s = "hello world"

s[0]:h, s[-1]:d

s[:]:"hello world"

s[0:4]:"hell"

len(s),max(s),min(s)

s.count("h"):1

s.isupper():False 判断是否都为大写字母

s.islower():True 判断是否都为小写字母

s.isdigit():False

s.lower()->改为全小写

s.upper()->改为全大写

s.strip()->去掉空格

s.lstrip()->去掉左边空格

s.rstrip()->去掉右边空格

s.swapcase()->把大写变为小写，吧小写变为大写

replace(old,new). s.repalce("l","E")

split(), s.split(" ")，返回list，['hello','world']

### Array 数组

读非常快，查询、插入、删除非常慢

### Linkedlist 链表

数组 VS 链表

数组访问非常快，搜索、插入和删除很慢

链表插入和删除非常快，访问，搜索都很慢

### Hash Table 哈希表

```python
#常见的hash table：dict
score = {
    "张三": 90,
    "李四": 85,
    "王五": 92
}
#查
print(score["张三"])   # 90
#增
score["赵六"] = 88
#改
score["李四"] = 95
#删
del score["王五"]

```

Python 的 dict 和 set 底层是哈希表。
哈希表通过 hash(key) 直接算出存放位置，所以查找特别快。
只有“不可变”的东西，才能当哈希表的 key。

### Queue 队列

特点：FIFO（先入先出）

双端队列：两边都可进出

List，deque

```python
from collections import deque
	q = deque([1,2,3])  
```

append()，尾部增加

popleft()，头部删除

上面一起使用，底下一起使用

appendlef()，头部增加

pop()，尾部删除

### Stack 栈

特点：LIFO（后入先出）

append()

pop()

len(),max(),min()

### Heap 堆

```python
from heapq import heapify,heappush,heappop

a = [1,2,3]
heapify(a)  # 小堆
heappush(a,4)	# 增加
heappop(a) # a = [2,3,4]
nlargest(2,a) # [3,4]
nsmallest(2,a) # [2,3]
```

大堆乘-1，后按小堆算

### Tree 树

高，深，层

二叉树，普通二叉树，满二叉树（除了叶子节点，每个节点都有左右两个子节点），完全二叉树（从上至下，从左至右编号，编号的节点与满二叉树节点的位置相同）。

二叉树的遍历

前序遍历，根左右

中序遍历，左根右

后序遍历，左右根





## 1. 两数之和

哈希表解法

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}
        for i, num in enumerate(nums):
            com = target - num
            if com in seen:
                return [seen[com],i]
            else:
                seen[num]=i
        return []
```

暴力解法

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                if nums[i] + nums[j] == target:
                    return [i, j]
        return []
```



## 49. 字母异位词分组

```python
class Solution:
    def groupAnagrams(self,strs:List[str]) -> List[List[str]]:
        box = {}
        for s in strs:
            sorted_key = "".join(sorted(s))
            if sorted_key not in box:
                box[sorted_key] = []
            box[sorted_key].append(s)
        return list(box.values())
```

