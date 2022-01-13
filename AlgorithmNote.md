## 第一章 排序与查找

### 1. 时间复杂度

**概念：**用来评估算法运行效率的一个式子

- 时间复杂度是用来估计算法运行时间的一个式子（单位）；

- 一般来说，时间复杂度高的算法比复杂度低的算法慢；

- 常见的时间复杂度（按效率排序）：

  ​		$O(1) < O(logn) < O(n) < O(nlogn) < O(n^2) < O(n^2logn) < O(n^3)$

- 复杂问题的时间复杂度：

​									$O(n!), O(2^n), O(n^n)$



#### 1.1 快速判断算法复杂度（适用于绝大多数简单情况）：

- 确定问题规模；
- 循环减半过程 ——> $O(nlogn)$;
- k层关于n的循环 ——> $O(n^k)$;

#### 1.2 复杂情况： 根据算法执行过程判断



### 2. 空间复杂度

**概念：**用来评估算法内存占用大小的式子

空间复杂度的表示方式与时间复杂度完全一样

 - 算法使用了几个变量：$O(1)$
 - 算法使用了长度为n的一维列表：$O(n)$
 - 算法使用了m行n列的二位列表：$O(mn)$

“空间换时间”



### 3. 递归

递归的两个特点：

- 调用自身；
- 结束条件；

```python
'''
递归实例：汉诺塔问题
上面n-1个圆盘作为一个整体，下面1个圆盘作为一个整体；
过程： 	
		- 把上面n-1个圆盘从A经过C移动到B
		- 把第n个圆盘从A移动到C
		- 把n-1个圆盘从B经过A移动到C
'''
def hanoi(n, a, b, c):      # n个圆盘从a经过b移动到c
    count = 0
    if n > 0:
        hanoi(n-1, a, c, b)
        print('moving from %s to %s' % (a, c))
        hanoi(n-1, b, a, c)

hanoi(3, 'A', 'B', 'C')
```



### 4. 列表查找

**查找：**在一些数据元素中，通过一定的方法找出与给定关键字相同的数据元素的过程

列表查找（线性表查找）：从列表中查找指定元素

- 输入：列表、待查找元素；
- 输出：元素下标（未找到元素时一般返回None或-1）

内置列表查找函数：**index()**——线性查找

#### 4.1 顺序查找（Linear Search）

**概念：**也叫线性查找，从列表第一个元素开始，顺序进行搜索，直到找到元素或搜索到列表最后一个元素为止；

```python
def linearSearch(nums, target):
    for index, num in enumerate(nums):
        if num == target:
            return index
        
    return None

# 时间复杂度：O(n)
# 空间复杂度：O(n)
```

#### 4.2 二分查找（Binary Search）

**概念：**又叫折半查找，从有序列表的初始候选区nums[0：n]开始，通过对待查找的值与候选区中间值的比较，可以使候选区减少一半；

**前提条件：**有序列表

```python
def BinarySearch(nums, target):
    left, right = 0, len(nums)-1
    while left <= right:
        mid = (right-left) // 2 + left
        if nums[mid] == target:
            return mid
        if nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
            
    return None
     
# 时间复杂度：O(logn)
```



### 5. 列表排序

**排序：** 将一组“无序”的记录序列调整为“有序”的记录序列

列表排序：将无序列表变为有序列表

- 输入：列表
- 输出：有序列表

升序与降序

内置排序函数：**sort()**  （基于归并排序）

#### 5.1常见排序算法：

##### 5.1.1 排序Low B三人组：时间复杂度均为$O(n^2)$, 均为原地排序

- 冒泡排序；
- 选择排序；
- 插入排序

##### 5.1.2 排序NB三人组：

- 快速排序；
- 堆排序；
- 归并排序；

##### 5.1.3 其他排序

- 希尔排序；
- 计数排序；
- 基数排序；

#### 5.2 冒泡排序（Bubble Sort)

- 列表每两个相邻的数，如果前面比后面大，则交换这两个数；
- 一趟排序完成后，则无序区减少一个数，有序区增加一个数；
- 代码关键点：趟、无序区范围

```python
def BubbleSort(nums):
    for i in range(len(nums)-1):
        for j in range(len(nums)-i-1):
            if nums[j] > nums[j+1]:			# 默认升序，若降序则改为nums[j] < nums[j+1]
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums
# 时间复杂度：O(n^2)

# 改进：若一趟中没有发生交换，则表明已经排序完成
def BubbleSort(nums):
    for i in range(len(nums)-1):
        exchange = False
        for j in range(len(nums)-i-1):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
                exchange = True
        if not exchange:
            return
    return nums
# 时间复杂度：O(n^2)
```

#### 5.3 选择排序（Select Sort）

- 一趟排序记录最小的数，放到第一个位置；
- 再一趟排序记录列表无序区最小的数，放到第二个位置；
- ......
- 算法关键点：有序区和无序区、无序区最小数的位置

```python
def select_sort(nums):
    for i in range(len(nums)-1):
        exchange = False
        for j in range(i+1, len(nums)):
            if nums[i] > nums[j]:
                nums[i], nums[j] = nums[j], nums[i]
                exchange = True
        if not exchange: 
            return
    return nums
# 时间复杂度：O(n^2)
```

#### 5.4 插入排序（Insert Sort）

- 初始时有序区只有一个数；
- 每次从无序区拿一个数，插入到有序区的正确位置；

```python
def insert_sort(nums):
    for i in range(1, len(nums)):
        tmp = nums[i]
        j = i-1
        while j >= 0 and nums[j] > tmp:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = tmp

    return nums
    
   # 时间复杂度：O(n^2)
```

#### 5.5 快速排序（Quick Sort）

- 快速排序：快
- 快速排序思路：
  - 取一个元素p（第一个元素），使元素p归位；
  - 列表被p分成两部分，左边都比p小，右边都比p大；
  - 递归完成排序
- 快速排序的效率：
  - 时间复杂度：O(nlogn)
- 快速排序的问题：
  - 最坏情况：每次只改变一个数的位置会导致时间复杂度变回到$O(n^2)$
  - 递归

```python
def partition(nums, left, right):
    tmp = nums[left]
    while left < right:

        while left < right and nums[right] >= tmp:      # 从右边找比tmp小的值
            right -= 1                                  # 向左走一步
        nums[left] = nums[right]                        # 把右边的值写到左边的位置上

        while left < right and nums[left] <= tmp:       # 从左边找比tmp大的值
            left += 1                                   # 向右走一步
        nums[right] = nums[left]                        # 把左边的值写到右边的位置上

    nums[left] = tmp                                    # tmp归为
    return left

def quick_sort(nums, left, right):
    if left < right:
        mid = partition(nums, left, right)
        quick_sort(nums, left, mid-1)
        quick_sort(nums, mid+1, right)

    return nums
```

#### 5.6 堆排序

##### 5.6.1 堆排序前传：树与二叉树

 - 树是一种数据结构，比如：目录结构
 - 树是一种可以递归定义的数据结构
 - 树是由n个节点组成的集合：
   - 如果n=0，那这是一棵空树；
   - 如果n>0，那存在1个节点作为树的根节点，其他节点可以分为m个集合，每个集合本身又是一棵树
 - 一些概念：
   - 根节点，叶子节点
   - 树的深度（高度）
   - 节点的度：节点分叉的数目
   - 树的度：整个树所有节点的度中最大的值
   - 孩子节点/父节点
   - 子树
 - 二叉树：度不超过2的树
   - 每个节点最多有两个孩子节点
   - 两个孩子节点被区分为左孩子节点和右孩子节点
 - 满二叉树：一个二叉树，如果每一层的节点数都达到最大值，那么这个二叉树是满二叉树
 - 完全二叉树：叶子节点只能出现在最下层和次下层，并且最下面一层的节点都集中在该层最左边的若干位置的二叉树

<img src="E:\Pycharm\LeetCode\figs\1.PNG" style="zoom:80%;" />

##### 5.6.2 堆排序前传：二叉树的存储方式

- 二叉树的存储方式（表示方式）

  - 链式存储方式

  - **顺序存储方式**

    - 父节点和左孩子节点的编号下标有什么关系：

      - 0-1，1-3， 2-5，3-7
      - i - 2i+1

    - 父节点和右孩子节点的编号下标有什么关系：

      - 0-2，1-4，2-6，3-8，4-10
      - i - 2i+2

      <img src="E:\Pycharm\LeetCode\figs\2.PNG" style="zoom:80%;" />

##### 5.6.3 堆排序前传：堆和堆的向下调整

- 堆：一种特殊的完全二叉树

  - 大根堆：一棵完全二叉树，满足任一节点都比其孩子节点大
  - 小根堆：一棵完全二叉树，满足任一节点都比其孩子节点小

  <img src="E:\Pycharm\LeetCode\figs\3.PNG" style="zoom:80%;" />

- 堆的向下调整性质：

  - 假设根节点的左右子树都是堆，但根节点不满足堆的性质
  - 可以通过一次向下的调整来将其变成一个堆

##### 5.6.4 堆排序的过程

- 1. 建立堆；
  2. 得到堆顶元素，为最大元素；
  3. 去掉堆顶，将堆最后一个元素放到堆顶，此时可通过一次调整重新使堆有序；
  4. 堆顶元素为第二大元素；
  5. 重复步骤3，直到堆变空；

##### 5.6.5 向下调整的实现

```python
def sift(nums, low, high):
    '''

    :param nums:    列表
    :param low:     堆的根节点位置
    :param high:    堆的最后一个元素位置
    :return:
    '''

    i = low                 # i 指向根节点
    j = 2 * i + 1           # j 一开始指向i的左子节点
    tmp = nums[low]         # 保存根节点的值

    while j <= high:        # 没有超出范围的情况下：
        if j+1 <= high and nums[j+1] > nums[j]:     # 比较左右子节点的值
            j = j + 1                               # 则指向右子节点作比较
        if tmp < nums[j]:               # 如果tmp比此时指向的子节点的值小，发生向下调整
            nums[j] = tmp
            i = j
            j = 2 * i + 1
        else:                           # 如果tmp可以直接放在当前i指向的位置，直接退出
            break

    nums[i] = tmp                       # 将tmp赋值给当前i指向的位置
```

##### 5.6.6 堆排序的实现

```python
def sift(nums, low, high):
    '''

    :param nums:    列表
    :param low:     堆的根节点位置
    :param high:    堆的最后一个元素位置
    :return:
    '''

    i = low                 # i 指向根节点
    j = 2 * i + 1           # j 一开始指向左子节点
    tmp = nums[low]         # 保存根节点的值

    while j <= high:        # 没有超出范围的情况下：
        if j+1 <= high and nums[j+1] > nums[j]:     # 如果右子节点的值比左子节点的大：
            j = j + 1                               # 则指向右子节点作比较
        if nums[j] > tmp:               # 如果tmp比此时指向的子节点的值小，发生向下调整
            nums[i] = nums[j]
            i = j
            j = 2 * i + 1
        else:                           # 如果tmp可以直接放在当前i指向的位置，直接退出
            break

    nums[i] = tmp                       # 将tmp赋值给当前i指向的位置

def heap_sort(nums):
    n = len(nums)
    # 建堆
    for i in range((n-2)//2, -1, -1):		# 从第一个非子结点开始逆序遍历
        sift(nums, i, n-1)					# 向下调整完成堆的建立

    # 排序
    for i in range(n-1, -1, -1):
        nums[0], nums[i] = nums[i], nums[0]		# 将最后一个元素放到根结点处开始进行向下调整
        sift(nums, 0, i-1)

    return nums
```

- 时间复杂度：$O(nlogn)$

##### 5.6.7 堆排序的内置模块

- python内置模块：**heapq**
- 常用函数：
  - heapyify(x)
  - heappush(heap, item)
  - heappop(heap)

##### 5.6.8 堆排序——topk问题

- 现在有n个数，设计算法得到前k大的数。（k<n）
- 解决思路：
  - 先排序后切片：$O(nlogn)$
  - 排序LowB三人组： $O(kn)$
  - 堆排序：$O(nlogk)$
- 堆排序解决topk问题思路：
  - 取列表前k个元素建立一个小根堆，堆顶就是目前第k大的数；
  - 依次向后遍历原列表，对于列表中的元素，如果小于堆顶，则忽略该元素；如果大于堆顶，则将堆顶更换为该元素，并且对堆进行一次调整；
  - 遍历列表所有元素后，倒序弹出堆顶；

```python
def sift(nums, low, high):
    i = low
    j = 2*i+1
    tmp = nums[low]
    while j <= high:
        if j+1 <= high and nums[j] > nums[j+1]:     # 注意这里是小根堆
            j = j+1
        if tmp > nums[j]:
            nums[i] = nums[j]
            i = j
            j = 2*i+1
        else:
            break

    nums[i] = tmp

def topk(nums, k):
    # 将前k个数拿出来
    heap = nums[0:k]

    # 对前k个数建小根堆
    for i in range((k-2)//2, -1, -1):
        sift(heap, i, k-1)

    # 对列表中的剩余元素和小根堆进行比较
    for i in range(k, len(nums)-1):
        if nums[i] > heap[0]:
            heap[0] = nums[i]
            sift(heap, 0, k-1)      # 小根堆进行向下调整

    # 对遍历之后的小根堆排序
    for i in range(k-1, -1, -1):
        heap[0], heap[i] = heap[i], heap[0]
        sift(heap, 0, i-1)

    return heap
```

#### 5.7 归并排序（Merge Sort）

- 假设现在的列表分为两段有序，如何将其合并成为一个有序列表？
- 这种操作称为一次归并
- 使用归并：
  - 分解：将列表越分越小，直至分成一个元素；
  - 终止条件：一个元素是有序的；
  - 合并：将两个有序列表归并，列表越来越大；

```python
def merge(nums, low, mid, high):        # 归并函数
    tmp = []
    i, j = low, mid+1                   # i, j 分别指向两部分的第一个元素
    while i <= mid and j <= high:       # 左右两边均不为空的情况下，依次比较元素大小
        if nums[i] < nums[j]:
            tmp.append(nums[i])
            i += 1
        else:
            tmp.append(nums[j])
            j += 1
    while i <= mid:                     # 检查左右连边剩余哪一部分不为空，添加到tmp后面
        tmp.append(nums[i])
        i += 1
    while j <= high:
        tmp.append(nums[j])
        j += 1

    nums[low:high+1] = tmp

def merge_sort(nums, low, high):
    if low < high:
        mid = (high-low) // 2 + low
        merge_sort(nums, low, mid)      # 左半部分归并排序
        merge_sort(nums, mid+1, high)   # 右半部分归并排序
        merge(nums, low, mid, high)     # 整体归并成一个整体

    return nums
```

- 时间复杂度：$O(nlogn)$
- 空间复杂度：$O(n)$

#### 5.8 NB三人组总结

- 三种排序算法的时间复杂度都是$O(nlogn)$
- 一般情况下，就运行时间而言：
  - 快速排序 < 归并排序 < 堆排序
- 三种排序算法的缺点：
  - 快速排序：极端情况下排序效率低；
  - 归并排序：需要额外的内存开销；
  - 堆排序：在快的排序算法中相对较慢；

<img src="E:\Pycharm\LeetCode\figs\4.PNG" style="zoom:80%;" />

- 挨个移动交换位置的都稳定，跳跃交换位置的不稳定；

#### 5.9 希尔排序（Shell Sort）

- 希尔排序（Shell Sort）是一种分组插入排序算法；
- 首先取一个整数$d_1=n/2$，将元素分为$d_1$个组，每组相邻量元素之间距离为$d_1$，在各组内进行直接插入排序；
- 取第二个整数$d_2=d_1/2$，重复上述分组排序过程，直到$d_i$=1，即所有元素在同一组内进行直接插入排序；
- 希尔排序每趟并不使某些元素有序，而是使整体数据越来越接近有序；最有一趟排序使得所有数据有序；

```python
def shell_sort(nums):
    def insert_sort_gap(nums, gap):			# 技巧：插入排序中所有的1换成gap即可
        for i in range(gap, len(nums)):
            tmp = nums[i]
            j = i - gap
            while j >= 0 and nums[j] > tmp:
                nums[j+gap] = nums[j]
                j -= gap

            nums[j+gap] = tmp

    d = len(nums) // 2
    while d >= 1:
        insert_sort_gap(nums, d)
        d //= 2

    return nums
```

- 时间复杂度：比较复杂，与选取的gap序列有关；本例中时间复杂度在$O(N)$与$O(N^2)$之间；

#### 5.10 计数排序（Count Sort）

- 对列表进行排序，已知列表中的数范围都在0到100之间。设计时间复杂度为$O(n)$的算法；

```python
def count_sort(nums, max_count=100):
    count = [0 for _ in range(max_count+1)]
    for val in nums:
        count[val] += 1

    nums.clear()
    for index, val in enumerate(count):
        for i in range(val):
            nums.append(index)

    return nums
```

- 时间复杂度：$O(n)$
- 空间复杂度：$O(n)$
- 局限性：
  - 需要知道列表元素的最大范围；
  - 需要额外开辟空间

#### 5.11 桶排序（Buckets Sort）

- 在计数排序中，如果元素的范围比较大（比如在1到1亿之间），如何改造算法？
- 桶排序（Bucket Sort）：首先将元素分在不同的桶中，再对每个桶中的元素排序；

```python
def buckets_sort(nums, n=100, max_num=10000):
    buckets = [[] for _ in range(n)]
    for var in nums:    
        i = min(n-1, var//(max_num//n))             # 将var放入i号桶
        buckets[i].append(var)
        for j in range(len(buckets[i])-1, 0, -1):   # 对i号桶中的元素进行冒泡排序
            if buckets[i][j] < buckets[i][j-1]:
                buckets[i][j], buckets[i][j-1] = buckets[i][j-1], buckets[i][j]
            else:
                break

    buckets_num = []
    for buc in buckets:
        buckets_num.extend(buc)                     # 将所有桶中的元素拼接起来

    return buckets_num
```

- 桶排序的表现取决于数据的分布，也就是需要对不同的数据排序时采用不同的分桶策略；
- 平均情况时间复杂度：$O(n+k)$，n表示列表长度，k表示桶的平均长度，比较复杂
- 最坏情况时间复杂度：$O(n^2k)$
- 空间复杂度$O(nk)$

#### 5.12 基数排序（Radix Sort）

- 多关键字排序：假如现在有一个员工表，要求按照薪资排序，年龄相同的员工按照年龄排序；此处第一关键字是薪资，第二关键字是年龄；

```python
def radix_sort(nums):
    max_num = max(nums)
    it = 0                                      # 需要分it次桶
    while 10**it <= max_num:
        buckets = [[] for _ in range(10)]
        for var in nums:
            ind = var // 10**it % 10            # 将var加入到第index号桶
            buckets[ind].append(var)

        nums.clear()                            # 分桶结束
        for buc in buckets:
            nums.extend(buc)                    # 将分完后的桶内元素依次输出

        it += 1
    return nums
```

- 时间复杂度：$O(kn)$，k表示分桶的次数，就程序中的it
- 空间复杂度：$O(k+n)$
- k表示数字位数

### 6. 查找排序相关面试题

（1）给两个字符串s和t，判断t是否为s的重新排列后组成的单词；	对应`leetcode-242.有效地字母异位词`

- s = 'anagram', t = 'nagaram',   return True
- s = 'rat', t = 'car',   return False

```python
def isAnagram(s, t):
    return sorted(list(s)) == sorted(list(t))
# 方法2
def isAnagram(s, t):
    dic_s, dic_t = {}, {}
    for ch in s:
        dic_s[ch] = dic_s.get(ch, 0) + 1

    for ch in t:
        dic_t[ch] = dic_t.get(ch, 0) + 1

    return dic_s == dic_t
# 方法3
def isAnagram(s, t):
    if len(s) != len(t):
            return False
    dic_s = [0] * 26
    dic_t = [0] * 26

    n = len(s)
    for i in range(n):
        dic_s[ord(s[i])-ord('a')] += 1
        dic_t[ord(t[i])-ord('a')] += 1

    return dic_s == dic_t
```

（2）给定一个m*n的二维列表，查找一个数是否存在。列表有下列特性：	对应`leetcode-74.搜索二维矩阵`

- 每一行的列表从左到右已经排好序；
- 每一行第一个数比上一行最后一个数大；

```python
# 方法1
def searchMatrix(matrix, target):
    for line in matrix:
            if target in line:
                return True

    return False
# 方法2
def searchMatrix(matrix, target):
    m, n = len(matrix), len(matrix[0])
    for i in range(m):
        if matrix[i][0] <= target <= matrix[i][-1]:
            left, right = 0, n-1
            while left <= right:
                mid = (right-left) // 2 + left
                if matrix[i][mid] == target:
                    return True
                if matrix[i][mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1

        else:
            continue
    return False
```

（3）给定一个列表和一个整数，设计算法找到两个数的下标，使得两个数之和为给定的整数，保证肯定仅有一个结果。

对应`leetcode-1.两数之和`

- 例如，列表[1, 2, 5, 4]与目标整数3，1+2=3，结果为（0，1）

```python
# 方法1
def twoSum(nums, target):
	hashtable = {}
    for i, num in enumerate(nums):
        if target-num in hashtable:
            return[i, hashtable[target-num]]
        hashtable[nums[i]] = i

    return []
# 方法2
def twoSum(nums, target):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[i] + nums[j] == target:
                return sorted([i, j])
```



## 第二章 数据结构

### 1. 数据结构

####  1.1 概念

- 数据结构是指相互之间存在着一种或多种关系的数据元素的集合和该集合中数据元素之间的关系组成；
- 简单来说，数据机构就是设计数据以何种方式组织并存储在计算机中；
- 比如：列表、集合与字典等都是一种数据结构；
- N.Wirth：“程序=数据结构+算法”

#### 1.2 数据结构的分类

- 数据结构按照其逻辑结构可以分为线性结构、树结构、图结构；
  - 线性结构：数据结构中的元素存在一对一的相互关系；
  - 树结构：数据结构中的元素存在一对多的相互关系；
  - 图结构：数据结构中的元素存在多对多的相互关系；

#### 1.3 列表

- 列表（其他语言称数组）是一种基本数据结构；
- 关于列表的问题：
  - 列表中的元素是如何存储的？**在内存空间中连续存储元素对应的地址**
  - 列表的基本操作：按下标查找 $O(1)$、插入元素 $O(N)$、删除元素 $O(N)$......
  - 这些操作的时间复杂度是多少？
- 数组与列表有两点不同：
  - 数组元素类型要相同；
  - 数组长度固定；

#### 1.4 栈

- 栈（stack）是一个数据集合，可以理解为只能在一端进行插入或删除操作的列表；
- 栈的特点：后进先出LIFO（last-in，first-out）
- 栈的概念：栈顶、栈底
- 栈的基本操作：
  - 进栈（压栈）：push
  - 出栈：pop
  - 取栈顶：gettop
- 使用一般的列表结构即可实现栈：
  - 进栈：stack.append()
  - 出栈：stack.pop()
  - 取栈顶：stcak[-1]
- 栈的应用——括号匹配问题      对应`leetcode-20.有效的括号`
  - 给定一个字符串，其中包含小括号、中括号、大括号，求该字符串中的括号是否匹配

```python
def isValid(self, s: str) -> bool:
        if len(s) % 2 == 1:		# 如果长度为奇数，则一定不匹配
            return False
        pairs = {')': '(',
                ']': '[',
                '}': '{'}

        stack = []
        for ch in s:
            if ch in pairs:		# ch为右括号
                if not stack or pairs[ch] != stack[-1]:		# stack为空或左右括号不匹配
                    return False
                stack.pop()
            else:				# ch为左括号
                stack.append(ch)

        return not stack
```

#### 1.5 队列

- 队列（Queue）是一个数据集合，仅允许在列表的一端进行插入，另一端进行删除；
- 进行插入的一端称为队尾（rear），插入动作称为进队或入队；
- 进行删除的一端称为队头（front），删除动作称为出队；
- 队列的性质：先进先出（FIFO，first-in，first-out）
- 队列的实现方式——环形队列：
  - 环形队列：当队尾指针front == Maxsize+1时，再前进一个位置就自动到0；
  - 队首指针前进1：front = (front+1) % Maxsize；
  - 队尾指针前进1：rear = (rear+1) % Maxsize；
  - 队空条件：rear == front；
  - 队满条件：（rear+1） % Maxsize == front；
- 双向队列：两端都支持进队和出队操作
  - 基本操作：队首进队、队首出队、队尾进队、队尾出队；
- 队列的内置模块：
  - 使用方法：from collections import deque
  - 创建队列：queue = deque()
  - 进队：queue.append()
  - 出队：queue.popleft()
  - 双向队列队首进队：queue.appendleft()
  - 双向队列队尾出队：queue.pop()

#### 1.6 栈和队列的应用——迷宫问题

**题目：**给一个二维列表，表示迷宫（0表示通道，1表示围墙），给出算法，求一条走出迷宫的路径。

- 方法一：回溯法：
  - 思路：从一个节点开始，任意找下一个能走的点，当找不到能走的点时，退回上一个点寻找是否有其他方向的点；
  - 使用栈存储当前路径
- 方法二：
  - 思路：从一个节点开始，寻找所有接下来能继续走的点，继续不断寻找，直到找到出口；
  - 使用队列存储当前正在考虑的节点

#### 1.7 链表

**概念：**链表是由一系列节点组成的元素集合。每个节点包含两部分，数据域item和指向下一个节点的指针next。通过节点之间的相互连接，最终串联成一个链表。

```python
class Node(object):
   def __init__(self, item):
       self.item = item
       self.next = None
    
a = Node(1)
b = Node(2)
c = Node(3)

a.next = b
b.next = c

print(a.next.next.item)
```

##### 1.7.1 链表的创建和遍历

**创建链表：**

- 头插法：在头节点的位置插入元素
- 尾插法：在尾节点的位置插入元素

```python
def create_linklist_head(nums):		# 头插法
    head = Node(nums[0])
    for element in nums[1:]:
        node = Node(element)
        node.next = head
        head = node

    return head

def create_linklist_tail(nums):		# 尾插法
    head = Node(nums[0])
    tail = head
    for element in nums[1:]:
        node = Node(element)
        tail.next = node
        tail = node

    return head

def print_linklist(head):			# 遍历
    while head:
        print(head.item, end=',')
        head = head.next
```

##### 1.7.2 链表节点的插入

<img src="E:\Pycharm\LeetCode\figs\5.PNG" style="zoom:80%;" />

```python
p.next = curNode.next
curNode.next = p
```

##### 1.7.3 链表节点的删除

```python
curNode.next = p.next
del p
```

##### 1.7.4 双链表

- 双链表的每个节点有两个指针：一个指向后一个节点，另一个指向前一个结点

```python
class Node(object):
	def __init__(self, item):
		self.item = item
		self.next = None
		self.prior = None
```

- 双链表节点的插入：

<img src="E:\Pycharm\LeetCode\figs\6.PNG" style="zoom:80%;" />

```
p.next = curNode.next
curNode.next.prior = p
p.prior = curNode
curNode.next = p
```

- 双链表节点的删除：

```python
curNode.next = p.next
p.next.prior = curNode
del p
```

##### 1.7.5 链表总结

- 复杂度分析：
  - 顺序表（列表/数组）与链表：
    - 按元素查找：$O(N)；O(1)$
    - 按下标查找：$O(1)；O(N)$
    - 在某元素后插入：$O(N)；O(1)$
    - 删除某元素：$O(N)；O(1)$

- 链表在插入和删除的操作上明显快于顺序表；
- 链表的内存可以更灵活的分配；
  - 试利用链表重新实现栈和队列；
- 链表这种链式存储的数据结构对树和图的结构有很大的启发性；

#### 1.8 哈希表

**概念：**哈希表是一个通过哈希函数来计算数据存储位置的数据结构，通常支持以下操作：

- insert（key, value）插入键值对(key, value)
- get（key）如果存在键为key的键值对则返回其value，否则返回空值；
- delete（key）：删除键为key的键值对

**直接寻址表：**当关键字的全域U比较小时，直接寻址是一种简单而有效的方法。

<img src="E:\Pycharm\LeetCode\figs\7.PNG" style="zoom:80%;" />

- 直接寻址技术的缺点：
  - 当域U很大时，需要消耗大量内存，很不实际；
  - 如果域U很大而实际出现的key很少，则大量空间被浪费；
  - 无法处理关键字不是数字的情况；

**哈希：**

- 直接寻址表：key为k的元素放到k位置上

- 改进直接寻址表：哈希（Hashing）

  - 构建大小为m的寻址表T；
  - key为k的元素放到h(k)位置上；
  - h(k)是一个函数，将其域U映射到表T[0, 1, ..., m-1]

- 哈希表（Hash Table，又称散列表），是一种线性表的存储结构。哈希表由一个直接寻址表和一个哈希函数组成。哈希函数h(k)将元素关键字k作为自变量，返回元素的存储下标。

- 哈希冲突：由于哈希表的大小是有限的，而要存储的值的总数量是无限的，因此对于任何哈希函数，都会出现两个不同元素映射到同一个位置上的情况，这种情况叫做哈希冲突。

  - 解决方式：

    - 开放寻址法：如果哈希函数返回的位置已经有值，则可以向后探查新的位置来存储这个值；
      - 线性探查：如果位置 i 被占用，则探查 i+1, i+2......
      - 二次探查：如果位置 i 被占用，则探查 $i+1^2, i-1^2, i+2^2, i-2^2,......$
      - 二度哈希：有n个哈希函数，当使用第一个哈希函数h1发生冲突时，则尝试使用h2, h3......
    - 拉链法：哈希表每个位置都连接一个链表，当冲突发生时，冲突的元素将被加到该位置链表的最后。

    <img src="E:\Pycharm\LeetCode\figs\8.PNG" style="zoom:80%;" />

- 常见的哈希函数：
  - 除法哈希法：h(k) = k % m
  - 乘法哈希法：h(k) = floor(m*(A*key%1))
  - 全域哈希法：$h_{a,b}(k) = ((a*key+b) mod \ p) mod \ m\;\ a,b=1,2,...,p-1   $

**哈希表的应用：**

- 集合与字典都是通过哈希表来实现的。

  ```python
  a = {'name': 'Alex',
      'age': 18,
      'gender': 'Man'}
  ```

- 使用哈希表存储字典，通过哈希函数将字典的键映射为下标。假设h('name') = 3, h('age') = 1, h('gender') = 4，则哈希表存储为[None, 18, None, 'Alex', 'Man']

- 如果发生哈希冲突，则通过拉链法或开发寻址法解决；

- **md5算法：**MD5（Message-Digest Algorithm 5）曾经时密码学中常用的哈希函数，可以把任意长度的数据映射为128位的哈希值，其曾经包含如下特征：

  - 1. 同样的消息，其MD5值必定相同；
    2. 可以快速计算出任意给定消息的MD5值；
    3. 除非暴力的枚举所有可能的消息，否则不可能从哈希值反推出消息本身；
    4. 两条消息之间即使只有微小的差别，其对应的MD5值也应该是完全不同、完全不相关的；
    5. 不能在有意义的时间内人工的构造两个不同的消息，使其具有相同的MD5值；
  - 应用举例：**文件的哈希值**
    - 算出文件的哈希值，若两个文件的哈希值相同，则可以认为这两个文件时相同的，因此：
      - 1. 用户可以利用它来验证下载的文件是否完整；
        2. 云存储服务商可以利用它来判断用户要上传的文件是否已经存在于服务器上，从而实现秒传的功能，同时避免存储过多相同的文件副本；

#### 1.9 树

**概念：**树是一种数据结构，比如：目录结构

树是一种可以递归定义的数据结构；

树是由n个节点组成的集合：

（1）如果n = 0，拿着是一颗空树；

（2）如果n > 0，那存在一个节点作为树的根节点，其他节点可以分为m个集合，每个集合本身又是一棵树；

##### 1.9.1 树的实例：模拟文件系统

```python
class Node:
    def __init__(self, name, type='dir'):
        self.name = name
        self.type = type    # 'dir' or 'file'
        self.children = []
        self.parent = None
        # 链式存储

    def __repr__(self):
        return self.name


class FileSystemTree:
    def __init__(self):
        self.root = Node('/', type='dir')
        self.now = self.root

    def mkdir(self, name):
        # name以'/'结尾
        if name[-1] != '/':
            name += '/'

        node = Node(name)
        self.now.children.append(node)
        node.parent = self.now

    def ls(self):
        return self.now.children

    def cd(self, name):
        if name[-1] != '/':
            name += '/'
        if name == '../':
            self.now = self.now.parent
            return
        for child in self.now.children:
            if child.name == name:
                self.now = child
                return

        raise ValueError('Invalid dir')
```

##### 1.9.2 二叉树

**二叉树的链式存储：**将二叉树的节点定义为一个对象，节点之间通过类似链表的链接方式来连接

**节点定义：**

```python
class BiTreeNode:
	def __init__(self, data):
		self.data = data
		self.lchild = None
		self.rchild = None
```

**二叉树的遍历：**

- 前序遍历：EACBDGF
- 中序遍历：ABCDEGF
- 后序遍历：BDCAFGE
- 层次遍历：EAGCFBD

<img src="E:\Pycharm\LeetCode\figs\9.PNG" style="zoom:80%;" />

```python
from collections import deque

class BiTreeNode:
    def __init__(self, data):
        self.data = data
        self.lchild = None
        self.rchild = None

# 前序遍历
def pre_order(root):
    if root:
        print(root.data, end=',')
        pre_order(root.lchild)
        pre_order(root.rchild)

# 中序遍历
def in_order(root):
    if root:
        in_order(root.lchild)
        print(root.data, end=',')
        in_order(root.rchild)

# 后序遍历
def post_order(root):
    if root:
        post_order(root.lchild)
        post_order(root.rchild)
        print(root.data, end=',')

# 层次遍历
def level_order(root):
    que = deque()
    que.append(root)
    while que:
        node = que.popleft()
        print(node.data, end=',')
        if node.lchild:
            que.append(node.lchild)
        if node.rchild:
            que.append(node.rchild)
```

#####  1.9.3 二叉搜索树

**概念：**二叉搜索树是一颗二叉树且满足性质：设x是二叉树的一个节点，如果y是x左子树的一个节点，那么 y.key <= x.key；如果y是x的右子树的一个节点，那么 y.key >= x.key。

二叉搜索树的操作：**查询、插入、删除**

<img src="E:\Pycharm\LeetCode\figs\10.PNG" style="zoom:80%;" />

**删除操作：**

- 1.如果删除的节点是叶子节点：直接删除
- 2.如果要删除的节点只有一个孩子：将此节点的父亲与孩子连接，然后删除该节点；
- 3.如果要删除的节点有两个孩子：将其右子树的最小节点（该节点最多有一个右孩子）删除，并替换当前节点；

##### 1.9.4 二叉搜索树的效率

1. 平均情况下，二叉树进行搜索的时间复杂度是$O(lgn)$。
2. 最坏情况下，二叉搜索树可能会非常偏斜。
3. 解决方案：
   1. 随机化插入
   2. AVL树

#### 1.10 AVL树

**概念：**AVL树是一棵自平衡的二叉搜索树。

**AVL树的性质：**

1. 根的左右子树的高度之差的绝对值不能超过1；
2. 根的左右子树都是平衡二叉树；

<img src="E:\Pycharm\LeetCode\figs\11.PNG" style="zoom:80%;" />

##### 1.10.1 AVL树的插入

1. 插入一个节点可能会破坏AVL树的平衡，可以通过**旋转**操作来进行修正。

2. 插入一个节点后，只有从插入节点到根节点的路径上的节点的平衡可能被改变，我们需要找出第一个破坏了平衡条件的节点，称之为K，K的两棵子树的高度差2。

3. 不平衡的出现可能有4种情况：

   1. 不平衡是由于对K的右孩子的右子树插入导致的：**左旋**
   2. 不平衡是由于对K的左孩子的左子树插入导致的：**右旋**
   3. 不平衡是由于对K的右孩子的左子树插入导致的：**右旋-左旋**
   4. 不平衡是由于对K的左孩子的右子树插入导致的：**左旋-右旋**

   *右右-左旋；左左-右旋；左右-左右旋；右左-右左旋*

#### 1.11 B树

**概念：**BTree是一棵自平衡的多路搜索树，常用于数据库的索引。



## 第三章 算法进阶

### 1. 贪心算法

**概念：**贪心算法（贪婪算法）是指，在对问题求解时，总是做出在当前看来是最好的选择。也就是说，不从整体最优上加以考虑，他所做出的是在某种意义上的局部最优解。

贪心算法并不能保证会得到最优解，但是在某些问题上贪心算法的解就是最优解，要会判断一个问题能否用贪心算法来计算。

#### 1.1 找零问题：

**题目：**假设商店老板需要找零n元钱，钱币的面额有：100，50，20，5，1元，如何找零使得所需钱币的数量最少？

```python
t = [100, 50, 20, 5, 1]

def change(t, n):
    m = [0 for _ in range(len(t))]
    for i, money in enumerate(t):
        m[i] = n // money
        n %= money

    return m, n
```

#### 1.2 背包问题

**题目：**一个小偷在某个商店发现有n个商品，第 i 个商品价值$v_i$元，重$w_i$千克。他希望拿走的价值尽量高，但他的背包最多只能容纳W千克的东西。他应该拿走哪些商品？

- **0-1背包**：对于一个商品，小偷要么把它完整拿走，要么留下。不能之拿走一部分，或者把一个商品拿走多次。（商品为金条）
- **分数背包**：对于一个商品，小偷可以拿走其中任意一部分。（商品为金砂）

```python
goods = [(60, 10), (100, 20), (120, 30)]     # (价格, 重量)
goods.sort(key=lambda x: x[0]/x[1], reverse=True)       # 单位重量的价格

def fractional_backbag(goods, w):
    m = [0 for _ in range(len(goods))]                  # 每种商品带走的量
    total_value = 0
    for i, (prize, weight) in enumerate(goods):
        if w >= weight:
            m[i] = 1
            w -= weight
            total_value += prize
        else:
            m[i] = w / weight
            w = 0
            total_value += m[i] * prize
            
    return m, total_value
```

#### 1.3 拼接最大数字问题

**题目：**有n个非负整数，将其按照字符串拼接的方式拼接为一个整数，如何拼接可以使得得到的整数最大？

例如：32，94，128，1286，6，71可以拼接的最大整数为：94716321286128

```python
from functools import cmp_to_key

li = [32, 94, 128, 1286, 6, 71]

def xy_cmp(x, y):
    if x+y < y+x:
        return 1
    elif x+y > y+x:
        return -1
    else:
        return 0
    
def number_join(li):
    li = list(map(str, li))

    li.sort(key=cmp_to_key(xy_cmp))
    return ''.join(li)
```

#### 1.4 活动选择问题

**题目：**假设有n个活动，这些活动要占用同一片场地，而场地在某时刻只能供一个活动使用。每个活动都有一个开始时间$s_i$和结束时间$f_i$（题目中时间以整数表示），表示活动在[$s_i$, $f_i$）区间占用场地。

问：安排哪些活动能够使该场地举办的活动的个数最多？

- 贪心结论：最先结束的活动一定是最优解的一部分；

```python
activities = [(1, 4), (3, 5), (0, 6), (5, 7), (3, 9), (5, 9), (6, 10), (8, 11), (8, 12), (2, 14), (12, 16)]
# 保证活动按照结束时间排好序的
activities.sort(key=lambda x: x[1])

def activities_selection(a):
    res = [a[0]]
    for i in range(1, len(a)):
        if a[i][0] >= res[-1][1]:       # 当前活动的开始时间和上一个活动的结束时间不冲突
            res.append(a[i])
            
    return res
```

### 2. 动态规划

**递归效率低的原因：**子问题的重复计算

**动态规划的思想：**递推式 + 重复子问题

#### 2.1 钢条切个问题

**题目：**某公司出售钢条，出售价格与钢条长度之间的关系如下表：

<img src="E:\Pycharm\LeetCode\figs\12.PNG" style="zoom:80%;" />

问题：现有一段长度为n的钢条和上面的价格表，求切割钢条方案，使得总收益最大。

- 设长度为n的钢条切割后最优收益值为$r_n$，可以得到递推式：

$$
r_n = max(p_n, r_1+r_{n-1}, r_2+r{n-2}, ...,r_{n-1}+r_1)
$$

- 第一个参数$p_n$表示不切割
- 其他n-1个参数分别表示另外n-1种不同切割方案，对方案 i=1, 2, ..., n-1
  - 将钢条切割长度为 i 和 n-i 两段
  - 方案 i 的收益为切割两段的最优收益之和
- 考察所有的 i，选择其中收益最大的方案

**最优子结构**

- 可以将求解规模为n的原问题，划分为规模更小的子问题：完成一次切割后，可以将产生的两段钢条堪称两个独立的钢条切割问题；
- 组合两个子问题的最优解，并在所有可能的两段切割方案中选取组合收益最大的，构成原问题的最优解；
- 钢条切割满足**最优子结构**：问题的最优解由相关子问题的最优解组合而成，这些子问题可以独立求解；

钢条切割问题还存在更简单的递归求解方法：

- 从钢条的左边切割下长度为 i 的一段，只对右边剩下的一段继续进行切割，左边的不再切割
- 递推式简化为：

$$
r_n = max_{1<=i<=n}(p_i+r_{n-i})
$$

- 不做切割的发难就可以描述为：左边一段长度为n，收益为$p_n$，剩余一段长度为0，收益为$r_0=0$;

**方法一：自顶向下递归的实现：**

```python
p = [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30]

def cut_rod_recurision_1(p, n):
    if n == 0:
        return 0
    else:
        res = p[n]
        for i in range(1, n):
            res = max(res, (cut_rod_recurision_1(p, i) + cut_rod_recurision_1(p, n-i)))

        return res

def cut_rod_recurision_2(p, n):
    if n == 0:
        return 0
    else:
        res = 0
        for i in range(1, n+1):
            res = max(res, p[i] + cut_rod_recurision_2(p, n-i))

        return res
```

- 时间复杂度：$O(2^n)$

- 递归算法由于重复求解相同子问题，效率极低；
- 动态规划的思想：
  - 每个子问题只求解一次，保存求解结果；
  - 之后需要此问题时，只需查找保存的结果；

**方法二：自底向上的方法：**

```python
def cut_rod_dp(p, n):
    r = [0]

    for i in range(1, n+1):
        res = 0
        for j in range(1, i+1):
            res = max(res, p[j]+r[i-j])
        r.append(res)

    return r[n]
```

- 时间复杂度：$O(n^2)$

**重构解：**如何修改动态规划算法，使其不仅输出最优解，还输出最优切割方案？

- 对每个子问题，保存切割一次时左边切下的长度

```python
def cut_rod_extend(p, n):
    r, s = [0], [0]

    for i in range(1, n+1):
        res_r, res_s = 0, 0
        for j in range(1, i+1):
            if p[j] + r[i-j] > res_r:
                res_r = p[j] + r[i-j]
                res_s = j
        r.append(res_r)
        s.append(res_s)

    return r[n], s

def cut_rod_solution(p, n):
    r, s = cut_rod_extend(p, n)
    ans = []
    while n > 0:
        ans.append(s[n])
        n -= s[n]
    return ans
```

**动态规划问题关键特征**

- 什么问题可以使用动态规划方法？
  - 最优子结构：
    - 原问题的最优解中涉及多少个子问题
    - 在确定最优解使用哪些子问题时，需要考虑多少种选择
  - 重叠子问题

#### 2.2 最长公共子序列

- 一个序列的子序列是在该序列中删去若干元素后得到的序列。
  - 如：“ABCD”和“BDF”都是序列“ABCDEFG”的子序列
- 最长公共子序列（LCS）问题：给定两个序列X和Y，求X和Y长度最大的公共子序列。
  - 例如：X="ABBCBDE"  Y="DBBCDB"  LCS(X, Y)="BBCD"
- 应用场景：字符串相似度比对

**定理：**(LCS的最优子结构) 令$X=<x_1, x_2, ..., x_m>$和$Y=<y_1, y_2, ..., y_n>$为两个序列，$Z=<z_1, z_2, ..., z_k>$为X和Y的任意LCS。

- 1. 如果$x_m=y_n$，则$z_k=x_m=y_n$且$Z_{k-1}$是$X_{m-1}$和$Y_{n-1}$的一个LCS；
  2. 如果$x_m!=y_n$，那么$z_k!=x_m$意味着Z是$X_{m-1}$和Y的一个LCS；
  3. 如果$x_m!=y_n$，那么$z_k!=$意味着Z是X和$Y_{n-1}$的一个LCS；

<img src="E:\Pycharm\LeetCode\figs\13.PNG" style="zoom:80%;" />



- 例如：求a="ABCBDAB"与b="BDCABA"的LCS：
  - 由于最后一位B!=A：
    - 因此LCS(a，b)应该来源于LCS(a[:-1], b)与LCS(a, b[:-1])中更大的那一个

```python
def lcs_length(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(1, m+1):
        for j in range(1, n+1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
```



#### 2.3 欧几里得算法

如何计算两个数的最大公约数？

- 欧几里得：**辗转相除法（欧几里得法）**
- 《九章算术》：更相减损术

**欧几里得算法**：gcd(a, b) = gcd(b, a mod b)

- 例如：gcd(60, 21) = gcd(21, 18) = ged(18, 3) = gcd(3, 0) = 3

```python
def gcd1(a, b):
    if b == 0:
        return a
    else:
        return gcd1(b, a % b)

def gcd2(a, b):
    while b > 0:
        r = a % b
        a = b
        b = r
    return a
```

**应用：**实现分数计算

- 利用欧几里得算法实现一个分数类，支持分数的四则运算；

#### 2.4 RSA算法

**密码与加密**

- 传统密码：加密算法是秘密的
- 现代密码系统：加密算法是公开的，密钥是秘密的
  - 对称加密
  - 非对称加密
- RSA非对称加密系统：
  - 公钥：用来加密，是公开的
  - 私钥：用来解密，是私有的

<img src="E:\Pycharm\LeetCode\figs\14.PNG" style="zoom:80%;" />

**RSA加密算法过程**

1. 随机选取两个质数p和q
2. 计算n=pq
3. 选取一个与$\phi(n)$互质的小奇数e，$\phi(n)=(p-1)(q-1)$
4. 对模$\phi(n)$，计算e的乘法逆元d，即满足$(e*d)\mod\ \phi(n) = 1$
5. 公钥(e, n)     私钥(d, n)

- 加密过程：$c = m^e\ mod\ n$
- 解密过程：$m = c^d\ mod\ n$















