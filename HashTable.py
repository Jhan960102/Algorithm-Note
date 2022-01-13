# -*- encoding:utf-8 -*-
# @Author: jhan
# @FileName: HashTable.py
# @Date: 2022/1/5 11:41

class LinkList:
    class Node:
        def __init__(self, item=None):
            self.item = item
            self.next = None

    class LinkListIterator:
        def __init__(self, node):
            self.node = node

        def __next__(self):
            if self.node:
                cur_node = self.node
                self.node = cur_node.next
                return cur_node.item
            else:
                raise StopIteration

        def __iter__(self):
            return self

    def __init__(self, iterable=None):
        self.head = None
        self.tail = None
        if iterable:
            self.extend(iterable)

    def append(self, obj):
        s = LinkList.Node(obj)
        if not self.head:
            self.head = s
            self.tail = s
        else:
            self.tail.next = s
            self.tail = s

    def extend(self, iterable):
        for obj in iterable:
            self.append(obj)

    def find(self, obj):
        for n in self:
            if n == obj:
                return True
            else:
                return False

    def __iter__(self):
        return self.LinkListIterator(self.head)

    def __repr__(self):
        return '<<' + ','.join(map(str, self)) + '>>'


class HashTable:
    def __init__(self, size=101):
        self.size = size
        self.T = [LinkList() for _ in range(self.size)]

    def h(self, k):
        return k % self.size

    def insert(self, k):
        i = self.h(k)
        if self.find(k):
            print('Duplicated insert!')
        else:
            self.T[i].append(k)

    def find(self, k):
        i = self.h(k)
        return self.T[i].find(k)


import random

if __name__ == '__main__':
    nums = [random.randint(0, 100) for _ in range(15)]
    print(nums)
    ht = HashTable()
    ht.insert(0)
    ht.insert(1)
    ht.insert(3)
    ht.insert(102)

    print(','.join(map(str, ht.T)))