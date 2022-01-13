# -*- encoding:utf-8 -*-
# @Author: jhan
# @FileName: ParseTree.py
# @Date: 2022/1/10 10:16

# 二叉树的构建
import operator


class BinaryTree:
    def __init__(self, rootObj):
        self.key = rootObj
        self.lchild = None
        self.rchild = None

    def insertLeft(self, newNode):
        if self.lchild == None:
            self.lchild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.lchild = self.lchild
            self.lchild = t

    def insertRight(self, newNode):
        if self.rchild == None:
            self.rchild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rchild = self.rchild
            self.rchild = t

    def getLeftChild(self):
        return self.lchild

    def getRightChild(self):
        return self.rchild

    def setRootVal(self, obj):
        self.key = obj

    def getRootVal(self):
        return self.key

# 栈的实现
class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peak(self):
        return self.items[len(self.items)-1]

    def size(self):
        return len(self.items)

# 解析树构造器
def buildParseTree(fpexp):
    fplist = fpexp.split()
    pStack = Stack()
    eTree = BinaryTree('')
    pStack.push(eTree)
    currentTree = eTree

    for i in fplist:
        if i == '(':                                    # 情况1：左括号
            currentTree.insertLeft('')                  # 新建一个左子节点
            pStack.push(currentTree)                    # 将新建左子节点的根节点存入栈中
            currentTree = currentTree.getLeftChild()    # 定位到新建的左子节点处

        elif i not in '+-*/)':                          # 情况2：数字
            currentTree.setRootVal(eval(i))             # 将当前节点设置为该数字
            parent = pStack.pop()                       # 取出当前节点的父节点
            currentTree = parent                        # 定位到父节点

        elif i in '+-*/':                               # 情况3：运算符
            currentTree.setRootVal(i)                   # 将运算符赋值给当前节点
            currentTree.insertRight('')                 # 创建右子节点
            pStack.push(currentTree)                    # 新建的右子节点存入栈中，用于赋值运算数
            currentTree = currentTree.getRightChild()   # 定位到新建的右子节点，进行赋值

        elif i == ')':                                  # 情况4：右括号
            currentTree = pStack.pop()                  # 返回到父节点处，表明该部分已赋值结束

        else:
            raise ValueError('Unknow Operatpe:' + i)

    return eTree

def evaluate(parseTree):                # 传入父节点
    opers = {                           # 定义字典用于存储运算符对应的函数
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv
    }

    leftC = parseTree.getLeftChild()        # 父节点的左子节点
    rightC = parseTree.getRightChild()      # 父节点的右子节点

    if leftC and rightC:                                # 若左右子节点都存在
        fn = opers[parseTree.getRootVal()]              # 父节点为运算符
        return fn(evaluate(leftC), evaluate(rightC))    # 递归左右子节点
    else:
        return parseTree.getRootVal()                   # 左右子节点均不存在，直接返回父节点


# 前序遍历
def preorder(tree):
    if tree:
        print(tree.getRootVal())
        preorder(tree.getLeftChild())
        preorder(tree.getRightChild())

# 后序遍历
def postorder(tree):
    if tree:
        postorder(tree.getLeftChild())
        postorder(tree.getRightChild())
        print(tree.getRootVal())

# 中序遍历
def inorder(tree):
    if tree:
        inorder(tree.getLeftChild())
        print(tree.getRootVal())
        inorder(tree.getRightChild())


