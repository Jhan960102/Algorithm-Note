# -*- encoding:utf-8 -*-
# @Author: jhan
# @FileName: BiTree.py
# @Date: 2022/1/7 15:29

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




a = BiTreeNode('A')
b = BiTreeNode('B')
c = BiTreeNode('C')
d = BiTreeNode('D')
e = BiTreeNode('E')
f = BiTreeNode('F')
g = BiTreeNode('G')

e.lchild = a
e.rchild = g
a.rchild = c
c.lchild = b
c.rchild = d
g.rchild = f

root = e

pre_order(root)
print('\n')
in_order(root)
print('\n')
post_order(root)
print('\n')
level_order(root)