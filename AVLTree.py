# -*- encoding:utf-8 -*-
# @Author: jhan
# @FileName: AVLTree.py
# @Date: 2022/1/9 15:59

from bst import BiTreeNode, BST

class AVLNode(BiTreeNode):
    def __init__(self, data):
        BiTreeNode.__init__(self, data)
        self.bf = 0


class AVLTree(BST):
    def __init__(self, li=None):
        BST.__init__(self, li)

    def rotate_left(self, p, c):
        s2 = c.lchild
        p.rchild = s2
        if s2:
            s2.parent = p

        c.lchild = p
        p.parent = c

        p.bf = 0
        c.bf = 0

        return c

    def rotate_right(self, p, c):
        s2 = c.lchild
        p.lchild = s2
        if s2:
            s2.parent = p

        c.rchild = p
        p.parent = c

        p.bf = 0
        c.bf = 0

        return c

    def rotate_right_left(self, p, c):
        g = c.lchild

        s3 = g.rchild
        c.lchild = s3
        if s3:
            s3.parent = c
        g.rchild = c
        c.parent = g

        s2 = g.lchild
        p.rchild = s2
        if s2:
            s2.parnet = p
        g.lchild = p
        p.parent = g

        if g.bf > 0:
            p.bf = -1
            c.bf = 0
        else:
            p.bf = 0
            c.bf = 1
        g.bf = 0

        return g

    def rotate_left_right(self, p, c):
        g = c.rchild

        s2 = g.lchild
        c.rchild = s2
        if s2:
            s2.parent = c
        g.lchild = c
        c.parent = g

        s3 = g.rchild
        p.lchild = s3
        if s3:
            s3.parent = p
        g.rchild = p
        p.parent = g

        if g.bf < 0:
            c.bf = 0
            p.bf = 1
        else:
            p.bf = 0
            c.bf = -1
        g.bf = 0

        return g


    def insert_no_rec(self, val):
        # 1. 插入
        p = self.root
        if not p:  # 空树
            self.root = BiTreeNode(val)
            return
        while True:
            if val < p.data:
                if p.lchild:
                    p = p.lchild
                else:  # 左子树不存在
                    p.lchild = BiTreeNode(val)
                    p.lchild.parent = p
                    node = p.lchild         # node 存储的是插入的节点
                    break
            elif val > p.data:
                if p.rchild:
                    p = p.rchild
                else:
                    p.rchild = BiTreeNode(val)
                    p.rchild.parent = p
                    node = p.rchild
                    break
            else:           # val == p.data
                return

        # 2.更新 blance factor
        while node.parent:          # 保证node.parent不空
            if node.parent.lchild == node:      # 传递是从左子树来的，左子树更沉
                # 更新node.parent的blance factor
                if node.parent.bf < 0:          # node.parent.bf=-1, 更新为-2
                    g = node.parent.parent      # 用于连接旋转后的子树
                    if node.bf > 0:
                        n = self.rotate_left_right(node.parent, node)
                    else:
                        n = self.rotate_right(node.parent, node)

                elif node.parent.bf > 0:
                    node.parent.bf = 0
                    break
                else:
                    node.parent.bf = -1
                    node = node.parent
                    continue

            else:                               # 传递是从右子树来的，右子树更沉
                if node.parent.bf > 0:
                    g = node.parent.parent
                    if node.bf < 0:
                        n = self.rotate_right_left(node.parent, node)
                    else:
                        n = self.rotate_left(node.parent, node)
                elif node.parent.bf < 0:
                    node.parent.bf = 0
                    break
                else:
                    node.parent.bf = 1
                    node = node.parent
                    continue

            # 连接旋转后的子树
            n.parent = g
            if g:
                if node.parent == g.lchild:
                    g.lchild = n
                else:
                    g.rchild = n
                break
            else:
                self.root = n
                break
