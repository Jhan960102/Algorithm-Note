# -*- encoding:utf-8 -*-
# @Author: jhan
# @FileName: cutRod.py
# @Date: 2022/1/12 10:51

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

def cut_rod_dp(p, n):
    r = [0]

    for i in range(1, n+1):
        res = 0
        for j in range(1, i+1):
            res = max(res, p[j]+r[i-j])
        r.append(res)

    return r[n]

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

a = 'ABCBDAB'
b = 'BDCABA'

print(gcd2(12, 16))