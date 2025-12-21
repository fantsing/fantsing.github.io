+++
date = '2025-12-21T13:40:43+08:00'
title = 'LeetCode-Day1'

+++

## 两数之和

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

