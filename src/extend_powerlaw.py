# -*- coding: utf-8 -*-
# 幂律检验延伸到 10^8：在若干对数间隔窗口上精确计算 r(n)（窗口卷积法，
# 避免全量 FFT 的内存开销），与 10^7 以内的 48 个点合并重拟合 theta。
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "PingFang SC",
                                   "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

NMAX = 10**8
C2 = 0.6601618158468696
TARGET = 2 * C2
W = 250                                          # 窗口半径（奇数个数）

print("sieving to 1e8 ...")
is_prime = np.ones(NMAX + 1, dtype=bool)
is_prime[:2] = False
for p in range(2, int(NMAX**0.5) + 1):
    if is_prime[p]:
        is_prime[p * p :: p] = False

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

centers = [int(10**e) | 1 for e in (7.2, 7.4, 7.6, 7.8, 8.0)]
centers[-1] = NMAX - 2 * W - 1                   # 顶端窗口贴着 1e8
rows = []
for n0 in centers:
    ns = np.arange(n0 - 2 * W, n0 + 2 * W + 1, 2, dtype=np.int64)
    qs = np.nonzero(is_prime[: (int(ns.max()) - 3) // 2 + 1])[0]
    qs = qs[qs >= 2]
    counts = np.zeros(len(ns), dtype=np.int64)
    CH = 200000
    for i in range(0, len(qs), CH):                # 分块 gather，控制峰值内存
        idx = ns[:, None] - 2 * qs[None, i : i + CH]
        m = idx >= 2
        counts += np.where(m, is_prime[np.where(m, idx, 2)], False).sum(axis=1)
    qv = np.arange(3.0, (n0 - 3) // 2 + 1)
    I0 = float(np.sum(1.0 / (np.log(qv) * np.log(n0 - 2 * qv))))
    vals = np.array([c / (K_of(int(n)) * I0) for c, n in zip(counts, ns)])
    rows.append((n0, vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))))
    print(f"window @ {n0:.3g}: ratio = {rows[-1][1]:.5f} ± {rows[-1][2]:.5f}, "
          f"gap = {TARGET - rows[-1][1]:+.5f}")

old = np.loadtxt("data/second_order_fit.csv", delimiter=",", skiprows=1)
allpts = np.vstack([old, np.array(rows)])
np.savetxt("data/second_order_fit_1e8.csv", allpts, delimiter=",",
           header="n,mean_ratio,sem", comments="")

n_, y_, s_ = allpts[:, 0], allpts[:, 1], allpts[:, 2]
gap = TARGET - y_
m = gap > 0
lg, ln_n = np.log(gap[m]), np.log(n_[m])
w2 = (gap[m] / s_[m]) ** 2
X2 = np.vstack([np.ones_like(ln_n), ln_n]).T
cov2 = np.linalg.inv(X2.T @ np.diag(w2) @ X2)
lnc, negtheta = cov2 @ (X2.T @ np.diag(w2) @ lg)
etheta = np.sqrt(np.diag(cov2))[1]
theta = -negtheta
chi2 = np.sum(w2 * (lg - X2 @ np.array([lnc, negtheta])) ** 2) / (m.sum() - 2)
print(f"combined fit (n up to 1e8): theta = {theta:.4f} ± {etheta:.4f}, "
      f"c = {np.exp(lnc):.3f}, chi2/dof = {chi2:.2f}, points = {m.sum()}")

LIGHT = dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e")
CAT = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
fig.patch.set_facecolor(LIGHT["surface"]); ax.set_facecolor(LIGHT["surface"])
ax.errorbar(n_[m][:-5], gap[m][:-5], yerr=s_[m][:-5], fmt="o", ms=4,
            color=CAT[0], ecolor=CAT[0], elinewidth=1, capsize=2,
            label="window means (n <= 1e7, full FFT counts)")
ax.errorbar(n_[m][-5:], gap[m][-5:], yerr=s_[m][-5:], fmt="s", ms=5,
            color=CAT[3], ecolor=CAT[3], elinewidth=1, capsize=2,
            label="windowed exact counts (1e7 - 1e8)")
xx = np.logspace(np.log10(n_[m].min()), np.log10(n_[m].max()), 100)
ax.plot(xx, np.exp(lnc) * xx ** (-theta), color=CAT[2], lw=2,
        label=f"fit c·n^(-θ), θ = {theta:.3f} ± {etheta:.3f}")
ax.plot(xx, gap[m][0] * (xx / n_[m][0]) ** (-0.5), color=CAT[1], lw=1.2,
        ls="--", label="reference slope -1/2")
ax.set_xscale("log"); ax.set_yscale("log")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
ax.set_xlabel("n", color=LIGHT["text"])
ax.set_ylabel("2C2 - ratio(n)", color=LIGHT["text"])
ax.set_title("Power-law decay of the Hardy–Littlewood deficit (to 1e8)", color=LIGHT["text"])
leg = ax.legend(frameon=False, fontsize=9)
for t in leg.get_texts():
    t.set_color(LIGHT["text"])
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(colors=LIGHT["muted"])
ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/second_order_1e8.png")
print("figure saved: figures/second_order_1e8.png")
