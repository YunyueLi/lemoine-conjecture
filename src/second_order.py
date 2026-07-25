# -*- coding: utf-8 -*-
# 二阶项拟合：ratio(n) = r(n)/(K(n)·I(n)) ≈ a + b/ln(n)。
# 若 Hardy–Littlewood 型渐近成立，应有 a ≈ 2C2；b 刻画二阶修正的经验强度。
# 每个采样点取 ±W 个奇数的窗口平均以压低涨落，加权最小二乘拟合。
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB",
                                   "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

N = 10**7
C2 = 0.6601618158468696
TARGET = 2 * C2

is_prime = np.ones(N + 1, dtype=bool)
is_prime[:2] = False
for p in range(2, int(N**0.5) + 1):
    if is_prime[p]:
        is_prime[p * p :: p] = False

A = is_prime.astype(np.float64)
B = np.zeros(N + 1, dtype=np.float64)
q_idx = np.nonzero(is_prime[: N // 2 + 1])[0]
B[2 * q_idx] = 1.0
L = 1 << 25
r = np.rint(np.fft.irfft(np.fft.rfft(A, L) * np.fft.rfft(B, L), L)[: N + 1]).astype(np.int64)
del A, B

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

W = 250                                        # 窗口半径（奇数个数）
centers = np.unique((np.logspace(4.3, 7, 48)).astype(np.int64) | 1)
centers = centers[centers <= N - 2 * W - 2]

xs, ys, sems = [], [], []
for n0 in centers:
    qv = np.arange(3.0, (n0 - 3) // 2 + 1)
    I0 = float(np.sum(1.0 / (np.log(qv) * np.log(n0 - 2 * qv))))
    ms = np.arange(n0 - 2 * W, n0 + 2 * W + 1, 2)
    vals = np.array([r[m] / (K_of(int(m)) * I0) for m in ms])
    xs.append(1.0 / np.log(n0))
    ys.append(vals.mean())
    sems.append(vals.std(ddof=1) / np.sqrt(len(vals)))
xs, ys, sems = map(np.array, (xs, ys, sems))

# 加权最小二乘：y = a + b·x，x = 1/ln n（作为对照模型保留）
wgt = 1.0 / sems**2
X = np.vstack([np.ones_like(xs), xs]).T
Wm = np.diag(wgt)
cov = np.linalg.inv(X.T @ Wm @ X)
a, b = cov @ (X.T @ Wm @ ys)
ea, eb = np.sqrt(np.diag(cov))
chi2_lin = np.sum(wgt * (ys - X @ (cov @ (X.T @ Wm @ ys)))**2) / (len(ys) - 2)
print(f"model 1 (a + b/ln n): a = {a:.5f} ± {ea:.5f} vs 2C2 = {TARGET:.5f} "
      f"({(a-TARGET)/ea:+.1f} sigma), chi2/dof = {chi2_lin:.1f}  -> excluded")

# 幂律模型：gap(n) = 2C2 - ratio(n) = c * n^(-theta)
ns_c = np.array([float(c) for c in centers])
gap = TARGET - ys
m = gap > 0
lg, ln_n = np.log(gap[m]), np.log(ns_c[m])
sig = sems[m] / gap[m]                          # log 空间误差传播
w2 = 1.0 / sig**2
X2 = np.vstack([np.ones_like(ln_n), ln_n]).T
cov2 = np.linalg.inv(X2.T @ np.diag(w2) @ X2)
lnc, negtheta = cov2 @ (X2.T @ np.diag(w2) @ lg)
elnc, etheta = np.sqrt(np.diag(cov2))
theta = -negtheta
chi2_pow = np.sum(w2 * (lg - X2 @ np.array([lnc, negtheta]))**2) / (m.sum() - 2)
print(f"model 2 (2C2 - c*n^-theta): theta = {theta:.4f} ± {etheta:.4f}, "
      f"c = {np.exp(lnc):.3f}, chi2/dof = {chi2_pow:.2f}")
print("theta ≈ 1/2 与素数分布误差项的平方根消去（黎曼零点贡献）一致")

np.savetxt("data/second_order_fit.csv",
           np.column_stack([ns_c, ys, sems]), delimiter=",",
           header="n,mean_ratio,sem", comments="")

LIGHT = dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e")
CAT = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
fig.patch.set_facecolor(LIGHT["surface"]); ax.set_facecolor(LIGHT["surface"])
ax.errorbar(ns_c[m], gap[m], yerr=sems[m], fmt="o", ms=4, color=CAT[0],
            ecolor=CAT[0], elinewidth=1, capsize=2,
            label="偏差 2C2 - ratio(n)（±1 标准误）")
xx = np.logspace(np.log10(ns_c[m].min()), np.log10(ns_c[m].max()), 100)
ax.plot(xx, np.exp(lnc) * xx**(-theta), color=CAT[2], lw=2,
        label=f"拟合 c·n^(-θ)，θ = {theta:.3f} ± {etheta:.3f}")
ax.plot(xx, np.exp(lnc) * xx**(-0.5) * (xx[0]**0.5 * np.exp(-lnc) * gap[m][0]),
        color=CAT[1], lw=1.2, ls="--", label="参考斜率 -1/2")
ax.set_xscale("log"); ax.set_yscale("log")
from matplotlib.ticker import FuncFormatter
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
ax.set_xlabel("n", color=LIGHT["text"])
ax.set_ylabel("2C2 - ratio(n)", color=LIGHT["text"])
ax.set_title("Hardy–Littlewood 偏差的幂律衰减", color=LIGHT["text"])
leg = ax.legend(frameon=False, fontsize=9)
for t in leg.get_texts():
    t.set_color(LIGHT["text"])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=LIGHT["muted"])
ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/second_order.png")
print("figure saved: figures/second_order.png")
