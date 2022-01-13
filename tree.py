# -*- encoding:utf-8 -*-
# @Author: jhan
# @FileName: tree.py
# @Date: 2022/1/7 14:50

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


tree = FileSystemTree()
tree.mkdir('var/')
tree.mkdir('bin/')
tree.mkdir('usr/')

tree.cd('bin/')
tree.mkdir('python/')
tree.cd('../')

print(tree.ls())