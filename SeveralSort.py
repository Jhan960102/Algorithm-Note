# 1. 冒泡排序：相邻元素作比较，前面的数大于后面的数，互换
def bubble_sort(nums):
    for i in range(len(nums)):
        exchange = False
        for j in range(1, len(nums)-i):
            if nums[j] < nums[j-1]:
                nums[j], nums[j-1] = nums[j-1], nums[j]
                exchange = True

        if not exchange:
            return nums

    return nums

# 2. 选择排序：遍历将无序区最小的数放到有序区中
def select_sort(nums):
    for i in range(len(nums)):
        for j in range(i+1, len(nums)):
            if nums[j] < nums[i]:
                nums[i], nums[j] = nums[j], nums[i]

    return nums

# 3. 插入排序：将第k个数插入到它的位置
def insert_sort(nums):
    for i in range(1, len(nums)):
        tmp = nums[i]
        j = i - 1
        while j >= 0 and nums[j] > tmp:
            nums[j+1] = nums[j]
            j -= 1
        nums[j+1] = tmp

    return nums

# 4. 快速排序：从第0个数开始，从右边遍历和它比较，再从左边遍历比较，往复
def quick_sort(nums, left, right):

    def partition(nums, left, right):
        tmp = nums[left]
        while left < right:
            while left < right and nums[right] >= tmp:
                right -= 1
            nums[left] = nums[right]
            while left < right and nums[left] < tmp:
                left += 1
            nums[right] = nums[left]

        nums[left] = tmp

        return left

    if left < right:
        mid = partition(nums, left, right)
        quick_sort(nums, left, mid-1)
        quick_sort(nums, mid+1, right)

    return nums

# 5. 堆排序：1.建堆；2.排序
def heap_sort(nums):
    def sift(nums, low, high):
        i = low
        j = 2*i+1
        tmp = nums[low]

        while j <= high:
            if j+1 <= high and nums[j+1] > nums[j]:
                j = j + 1
            if tmp < nums[j]:
                nums[i] = nums[j]
                i = j
                j = 2*i+1
            else:
                break
        nums[i] = tmp

    n = len(nums)
    for i in range((n-2)//2, -1, -1):
        sift(nums, i, n-1)

    for i in range(n-1, -1, -1):
        nums[i], nums[0] = nums[0], nums[i]
        sift(nums, 0, i-1)

    return nums

# 堆排序应用：topk问题
def topk(nums, k):
    def sift(nums, low, high):
        i = low
        j = 2*i+1
        tmp = nums[low]
        while j <= high:
            if j+1 <= high and nums[j+1] < nums[j]:
                j = j+1
            if tmp > nums[j]:
                nums[i] = nums[j]
                i = j
                j = 2*i+1
            else:
                break

        nums[i] = tmp

    n = len(nums)
    litmp = nums[0:k]
    for i in range((k-2)//2, -1, -1):
        sift(litmp, i, k-1)

    for i in range(k+1, n-1):
        if nums[i] > litmp[0]:
            litmp[0] = nums[i]
            sift(litmp, 0, k-1)

    for i in range(k-1, -1, -1):
        litmp[0], litmp[i] = litmp[i], litmp[0]
        sift(litmp, 0, i-1)

    return litmp

# 6. 归并排序：分为左右两边的有序序列，再归并成一个数列
def merge_sort(nums, low, high):

    def merge(nums, low, mid, high):
        tmp = []
        i, j = low, mid + 1
        while i <= mid and j <= high:
            if nums[i] < nums[j]:
                tmp.append(nums[i])
                i += 1
            else:
                tmp.append(nums[j])
                j += 1
        while i <= mid:
            tmp.append(nums[i])
            i += 1
        while j <= high:
            tmp.append(nums[j])
            j += 1

        nums[low:high+1] = tmp

    if low < high:
        mid = (high-low) // 2 + low
        merge_sort(nums, low, mid)
        merge_sort(nums, mid+1, high)
        merge(nums, low, mid, high)

    return nums

# 希尔排序
def shell_sort(nums):
    def insert_gap(nums, gap):
        for i in range(gap, len(nums)):
            tmp = nums[i]
            j = i - gap
            while j >= 0 and nums[j] > tmp:
                nums[j+gap] = nums[j]
                j -= gap
            nums[j+gap] = tmp

    d = len(nums)//2
    while d >= 1:
        insert_gap(nums, d)
        d //= 2

    return nums

# 计数排序
def count_sort(nums, max_count=100):
    count = [0 for _ in range(max_count+1)]
    for val in nums:
        count[val] += 1

    nums.clear()
    for index, val in enumerate(count):
        for i in range(val):
            nums.append(index)

    return nums

# 桶排序
def bucket_sort(nums, n=100, max_num=10000):
    buckets = [[] for _ in range(n)]
    for var in nums:
        i = min(n-1, var//(max_num//n))
        buckets[i].append(var)

        for j in range(len(buckets[i])-1, 0, -1):
            if buckets[i][j-1] > buckets[i][j]:
                buckets[i][j-1], buckets[i][j] = buckets[i][j], buckets[i][j-1]
            else:
                break

    nums.clear()
    for buc in buckets:
        nums.extend(buc)

    return nums

# 基数排序
def radix_sort(nums):
    max_num = max(nums)
    it = 0
    while 10**it <= max_num:
        buckets = [[] for _ in range(10)]
        for var in nums:
            ind = var // 10**it % 10
            buckets[ind].append(var)

        nums.clear()
        for buc in buckets:
            nums.extend(buc)

        it += 1
    return nums





import random
nums = [random.randint(0, 100) for i in range(15)]
print(nums)

print(radix_sort(nums))
