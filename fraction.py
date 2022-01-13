# -*- encoding:utf-8 -*-
# @Author: jhan
# @FileName: fraction.py
# @Date: 2022/1/12 15:31


class Fraction:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        x = self.gcd(a, b)
        self.a /= x
        self.b /= x

    def gcd(self, a, b):
        while b > 0:
            r = a % b
            a = b
            b = r
        return a

    def zgs(self, a, b):
        x = self.gcd(a, b)
        return a * b / x

    def __add__(self, other):
        a = self.a
        b = self.b
        c = other.a
        d = other.b
        deno = self.zgs(b, d)
        mole = a * (deno / b) + c * (deno / d)

        return Fraction(mole, deno)

    def __sub__(self, other):
        a = self.a
        b = self.b
        c = other.a
        d = other.b
        deno = self.zgs(b, d)
        mole = a * (deno / b) - c * (deno / d)

        return Fraction(mole, deno)

    def __mul__(self, other):
        a = self.a
        b = self.b
        c = other.a
        d = other.b
        deno = b * d
        mole = a * c
        # x = self.gcd(mole, deno)

        return Fraction(mole, deno)

    def __divmod__(self, other):
        a = self.a
        b = self.b
        c = other.a
        d = other.b
        deno = b * c
        mole = a * d

        return Fraction(mole, deno)


    def __str__(self):
        return '%d/%d' % (self.a, self.b)



a = Fraction(2, 5)
b = Fraction(5, 6)
print(a // b)