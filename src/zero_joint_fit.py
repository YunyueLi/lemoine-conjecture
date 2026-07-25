# -*- coding: utf-8 -*-
# 零点联合拟合：残差 = Σ_k [a_k cos(γ_k L) + b_k sin(γ_k L)]，k=1..13（零点频率固定）。
# 输出各零点振幅 A_k、联合显著性，并与显式公式的 1/γ 权重预期比较；
# 零假设校准：用 500 组随机频率集合重复拟合，得 Δχ² 的经验分布。
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "PingFang SC",
                                   "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
rng = np.random.default_rng(20260724)

TARGET = 2 * 0.6601618158468696
ZEROS = np.array([14.1347, 21.0220, 25.0109, 30.4249, 32.9351, 37.5862,
                  40.9187, 43.3271, 48.0052, 49.7738, 52.9703, 56.4462, 59.3470])

d = np.loadtxt("data/zero_scan_points.csv", delimiter=",", skiprows=1)
n_, y_, s_ = d[:, 0], d[:, 1], d[:, 2]
gap = TARGET - y_
m = gap > 0
L, g, s = np.log(n_[m]), gap[m], s_[m]

w = (g / s) ** 2
X = np.vstack([np.ones_like(L), L]).T
beta = np.linalg.solve(X.T @ np.diag(w) @ X, X.T @ np.diag(w) @ np.log(g))
model = np.exp(X @ beta)
res, sr = (g - model) / model, s / model
chi2_0 = float(np.sum((res / sr) ** 2))
npts = m.sum()

def joint_fit(freqs):
    cols = [np.ones_like(L), L]                 # 同时重拟合慢变部分，避免吸收偏置
    for gam in freqs:
        cols += [np.cos(gam * L), np.sin(gam * L)]
    Bm = np.vstack(cols).T
    Wv = 1 / sr**2
    coef, *_ = np.linalg.lstsq(Bm * np.sqrt(Wv)[:, None],
                               res * np.sqrt(Wv), rcond=None)
    chi2 = float(np.sum(((res - Bm @ coef) / sr) ** 2))
    return coef, chi2

coef, chi2_joint = joint_fit(ZEROS)
dchi2 = chi2_0 - chi2_joint
k = 2 * len(ZEROS) + 2
print(f"points={npts}, chi2_0={chi2_0:.1f}")
print(f"joint 13-zero fit: chi2={chi2_joint:.1f}, dchi2={dchi2:.1f}, "
      f"params={k}, chi2/dof after = {chi2_joint/(npts-k):.2f}")

amps, errs = [], []
# 振幅误差：用协方差估计
cols = [np.ones_like(L), L]
for gam in ZEROS:
    cols += [np.cos(gam * L), np.sin(gam * L)]
Bm = np.vstack(cols).T
cov = np.linalg.inv(Bm.T @ (Bm / sr[:, None] ** 2))
for j in range(len(ZEROS)):
    a, b = coef[2 + 2 * j], coef[3 + 2 * j]
    va, vb = cov[2 + 2 * j, 2 + 2 * j], cov[3 + 2 * j, 3 + 2 * j]
    A = np.hypot(a, b)
    eA = np.sqrt((a * a * va + b * b * vb) / (A * A)) if A > 0 else np.sqrt(va)
    amps.append(A); errs.append(eA)
    print(f"  gamma_{j+1:<2d} = {ZEROS[j]:7.4f}   A = {A:.5f} ± {eA:.5f}   "
          f"A*gamma = {A*ZEROS[j]:.4f}")
amps, errs = np.array(amps), np.array(errs)

# 零假设校准：随机 13 频率集合（同范围，避开零点 ±1）
null = []
for _ in range(500):
    f = rng.uniform(10, 60, 13)
    while np.min(np.abs(f[:, None] - ZEROS[None, :])) < 1.0:
        f = rng.uniform(10, 60, 13)
    _, c2 = joint_fit(np.sort(f))
    null.append(chi2_0 - c2)
null = np.array(null)
p_emp = float((null >= dchi2).mean())
print(f"null (500 random 13-freq sets): mean dchi2 = {null.mean():.1f}, "
      f"max = {null.max():.1f}; empirical p(zero-set) = {p_emp:.3f}")

LIGHT = dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e")
CAT = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
fig.patch.set_facecolor(LIGHT["surface"]); ax.set_facecolor(LIGHT["surface"])
ax.errorbar(ZEROS, amps, yerr=errs, fmt="o", ms=5, color=CAT[0],
            ecolor=CAT[0], elinewidth=1, capsize=2, label="fitted amplitude A_k")
gg = np.linspace(12, 62, 100)
scale = np.median(amps * ZEROS)
ax.plot(gg, scale / gg, color=CAT[2], lw=1.5, ls="--",
        label="explicit-formula weighting, 1/γ (median-normalized)")
ax.set_xlabel("γ_k", color=LIGHT["text"])
ax.set_ylabel("amplitude A_k (relative-residual units)", color=LIGHT["text"])
ax.set_title("Fitted amplitudes vs the 1/γ expectation", color=LIGHT["text"])
leg = ax.legend(frameon=False, fontsize=9)
for t in leg.get_texts():
    t.set_color(LIGHT["text"])
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=LIGHT["muted"])
ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/zero_amplitudes.png")
print("figure saved: figures/zero_amplitudes.png")
