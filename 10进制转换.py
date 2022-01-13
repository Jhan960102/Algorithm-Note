# -*- encoding:utf-8 -*-
# @Author: jhan
# @FileName: 10进制转换.py
# @Date: 2022/1/4 17:27

def baseConverter(decNumber, base):
    digits = '0123456789ABCDEF'

    stack = []

    while decNumber > 0:
        x = decNumber % base
        stack.append(x)
        decNumber //= base

    binStr = ''
    while stack:
        binStr += str(digits[stack.pop()])

    return binStr

def toStr(n, base):
    digits = '0123456789ABCDEF'
    if n < base:
        return digits[n]
    else:
        return toStr(n//base, base) + digits[n%base]

num = 233
print(toStr(num, 16))
print(baseConverter(num, 16))