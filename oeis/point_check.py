# -*- coding: utf-8 -*-
# OEIS a(50) 点验证脚本 —— 特意写得每一行都可读，不依赖任何第三方库。
#
# 要验证的断言（A185091 的定义：对奇数 N，最小的非合数 q 使得
# N = 2p + q 且 p 也是非合数；非合数 = 1 或素数）：
#   1) N = 8 567 125 268 699（A002091 已发布的最后一项 a(49)）处最小 q = 8821；
#   2) N = 12 279 230 664 247（我们声称的新项 a(50)）处最小 q = 8969。
# "区间内没有更早突破 8821 的 N"由 verify2 的区间扫描另行验证。

# 确定性 Miller-Rabin：对 n < 3.3×10^24，用下面这组底数检验即为确定性判定
# （文献：Sorenson & Webster 2015）。我们的 N 只有 44 位二进制，远在范围内。
BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    for p in BASES:
        if n % p == 0:
            return n == p
    # 把 n-1 写成 d * 2^s（d 为奇数）
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in BASES:
        x = pow(a, d, n)          # a^d mod n
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False          # 该底数判定 n 为合数
    return True

def is_noncomposite(n):
    return n == 1 or is_prime(n)

def minimal_q(N):
    """A185091：最小非合数 q，使 (N - q)/2 为非合数。逐个尝试 q = 1, 3, 5, ..."""
    q = 1
    while True:
        if is_noncomposite(q) and is_noncomposite((N - q) // 2):
            return q
        q += 2

for N, claimed in [(8567125268699, 8821), (12279230664247, 8969)]:
    got = minimal_q(N)
    p = (N - got) // 2
    status = "OK" if got == claimed else "MISMATCH!"
    print(f"N = {N}: minimal q = {got} (claimed {claimed}) [{status}]")
    print(f"   check: 2*{p} + {got} = {2*p + got}, p is prime: {is_prime(p)}")
