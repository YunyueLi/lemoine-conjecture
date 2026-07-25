# -*- coding: utf-8 -*-
# Lemoine 表示数统计：r(n) = #{(p,q): n = p + 2q, p、q 均为素数}，n 为奇数。
# 方法：FFT 卷积（素数指示函数 × 2·素数指示函数），一次算出 [0, N] 全部 r(n)。
# 输出：
#   figures/comet.png    —— Lemoine 彗星图（按 n 的 3、5 整除性分带着色）
#   figures/hl_ratio.png —— r(n) / (K(n)·I(n)) 与理论常数 2C2 的比较
#   data/hl_fit.csv      —— 拟合采样点数据
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import log

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "PingFang SC",
                                   "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

N = 10**7

# ---------- 素数筛 ----------
is_prime = np.ones(N + 1, dtype=bool)
is_prime[:2] = False
for p in range(2, int(N**0.5) + 1):
    if is_prime[p]:
        is_prime[p * p :: p] = False

# ---------- FFT 卷积求 r(n) ----------
A = is_prime.astype(np.float64)               # p 的指示函数
B = np.zeros(N + 1, dtype=np.float64)         # 2q 的指示函数
q_idx = np.nonzero(is_prime[: N // 2 + 1])[0]
B[2 * q_idx] = 1.0
L = 1 << 25                                   # >= 2N+1
r = np.fft.irfft(np.fft.rfft(A, L) * np.fft.rfft(B, L), L)[: N + 1]
r = np.rint(r).astype(np.int64)
del A, B

# 与暴力计数核对若干小值
def brute(n):
    return sum(1 for q in q_idx[2 * q_idx < n - 1] if is_prime[n - 2 * q])
for n in (15, 99, 1001, 12345, 100001):
    assert r[n] == brute(n), (n, r[n], brute(n))
print("sanity check passed; example r(15) =", r[15])

# ---------- 奇异级数因子 K(n) = prod_{l|n, l>2} (l-1)/(l-2) ----------
def K_of(n):
    k, m = 1.0, n
    while m % 2 == 0:
        m //= 2
    l = 3
    while l * l <= m:
        if m % l == 0:
            k *= (l - 1) / (l - 2)
            while m % l == 0:
                m //= l
        l += 2
    if m > 1:
        k *= (m - 1) / (m - 2)
    return k

# ---------- 图 1：Lemoine 彗星 ----------
LIGHT = dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e")
CAT = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]  # 固定顺序的分类色

lo = N - 200000
ns = np.arange(lo + 1, N + 1, 2)
sub = ns[::7]                                  # 抽稀；步长 14 与 3、5 互素，避免剩余类混叠
cls3, cls5 = (sub % 3 == 0), (sub % 5 == 0)
groups = [
    ("no factor 3 or 5",   ~cls3 & ~cls5, CAT[2]),
    ("3 | n only",       cls3 & ~cls5,  CAT[0]),
    ("5 | n only",       ~cls3 & cls5,  CAT[1]),
    ("15 | n",         cls3 & cls5,   CAT[3]),
]
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
fig.patch.set_facecolor(LIGHT["surface"]); ax.set_facecolor(LIGHT["surface"])
for label, m, c in groups:
    ax.scatter(sub[m], r[sub[m]], s=2.0, c=c, label=label, linewidths=0, alpha=0.8)
ax.set_xlabel("n (odd)", color=LIGHT["text"])
ax.set_ylabel("r(n): representations n = p + 2q", color=LIGHT["text"])
ax.set_title(f"Lemoine comet: n in [{lo:,}, {N:,}]", color=LIGHT["text"])
leg = ax.legend(markerscale=8, frameon=False, loc="upper left", fontsize=9)
for t in leg.get_texts():
    t.set_color(LIGHT["text"])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=LIGHT["muted"])
ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/comet.png")
plt.close(fig)

# ---------- 图 2：Hardy–Littlewood 常数检验 ----------
# 启发式：r(n) ≈ 2·C2 · K(n) · I(n)，I(n) = Σ_{q=3}^{(n-3)/2} 1/(ln q · ln(n-2q))
C2 = 0.6601618158468696
samples = np.unique((np.logspace(4, 7, 60)).astype(np.int64) | 1)
samples = samples[samples <= N - 2]
rows = []
for n in samples:
    qv = np.arange(3.0, (n - 3) // 2 + 1)
    I = float(np.sum(1.0 / (np.log(qv) * np.log(n - 2 * qv))))
    rows.append((int(n), int(r[n]), K_of(int(n)), I, r[n] / (K_of(int(n)) * I)))
rows = np.array(rows)
np.savetxt("data/hl_fit.csv", rows, delimiter=",",
           header="n,r,K,I,ratio", comments="", fmt="%.10g")

fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
fig.patch.set_facecolor(LIGHT["surface"]); ax.set_facecolor(LIGHT["surface"])
ax.axhline(2 * C2, color=CAT[2], linewidth=2, label="prediction 2C2 = 1.3203")
ax.scatter(rows[:, 0], rows[:, 4], s=14, c=CAT[0], linewidths=0,
           label="observed r(n) / (K(n)·I(n))")
ax.set_xscale("log")
ax.set_xlabel("n", color=LIGHT["text"])
ax.set_ylabel("normalized ratio", color=LIGHT["text"])
ax.set_title("Numerical test of the Hardy–Littlewood asymptotic", color=LIGHT["text"])
leg = ax.legend(frameon=False, fontsize=9)
for t in leg.get_texts():
    t.set_color(LIGHT["text"])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=LIGHT["muted"])
ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/hl_ratio.png")
plt.close(fig)

tail = rows[rows[:, 0] > 10**6]
print(f"ratio over n>1e6: mean={tail[:,4].mean():.4f}, std={tail[:,4].std():.4f}, "
      f"target 2C2={2*C2:.4f}")
print(f"r(n) computed for all odd n <= {N:,}; max r = {r[1::2].max()} at n = {1 + 2*int(np.argmax(r[1::2]))}")
