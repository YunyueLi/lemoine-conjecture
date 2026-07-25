# -*- coding: utf-8 -*-
# 零点振荡检测：加密窗口采样（10^4.7–10^7 共 220 个中心 + 10^7–10^8 五个大窗口），
# 对幂律残差做加权周期图，检验峰位是否落在黎曼零点 γ_k 上。
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "PingFang SC",
                                   "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"

N = 10**7
C2 = 0.6601618158468696
TARGET = 2 * C2
W = 250

is_prime = np.ones(N + 1, dtype=bool)
is_prime[:2] = False
for p in range(2, int(N**0.5) + 1):
    if is_prime[p]:
        is_prime[p * p :: p] = False
A = is_prime.astype(np.float64)
B = np.zeros(N + 1, dtype=np.float64)
q_idx = np.nonzero(is_prime[: N // 2 + 1])[0]
B[2 * q_idx] = 1.0
r = np.rint(np.fft.irfft(np.fft.rfft(A, 1 << 25) * np.fft.rfft(B, 1 << 25),
                         1 << 25)[: N + 1]).astype(np.int64)
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

centers = np.unique((np.logspace(4.7, 7, 220)).astype(np.int64) | 1)
centers = centers[centers <= N - 2 * W - 2]
rows = []
for n0 in centers:
    qv = np.arange(3.0, (n0 - 3) // 2 + 1)
    I0 = float(np.sum(1.0 / (np.log(qv) * np.log(n0 - 2 * qv))))
    ms = np.arange(n0 - 2 * W, n0 + 2 * W + 1, 2)
    vals = np.array([r[m] / (K_of(int(m)) * I0) for m in ms])
    rows.append((float(n0), vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))))
dense = np.array(rows)
big = np.loadtxt("data/second_order_fit_1e8.csv", delimiter=",", skiprows=1)[-5:]
allpts = np.vstack([dense, big])
np.savetxt("data/zero_scan_points.csv", allpts, delimiter=",",
           header="n,mean_ratio,sem", comments="")

n_, y_, s_ = allpts[:, 0], allpts[:, 1], allpts[:, 2]
gap = TARGET - y_
m = gap > 0
L, g, s = np.log(n_[m]), gap[m], s_[m]

w = (g / s) ** 2
X = np.vstack([np.ones_like(L), L]).T
beta = np.linalg.solve(X.T @ np.diag(w) @ X, X.T @ np.diag(w) @ np.log(g))
model = np.exp(X @ beta)
res, sr = (g - model) / model, s / model
chi2_0 = np.sum((res / sr) ** 2)
print(f"points={m.sum()}, theta={-beta[1]:.4f}, chi2_0={chi2_0:.1f} (dof={m.sum()-2})")

gammas = np.linspace(3, 60, 1141)
power = np.empty_like(gammas)
Wm = 1 / sr**2
for i, gam in enumerate(gammas):
    Bm = np.vstack([np.cos(gam * L), np.sin(gam * L)]).T
    ab = np.linalg.solve(Bm.T @ (Bm * Wm[:, None]), Bm.T @ (res * Wm))
    power[i] = chi2_0 - np.sum(((res - Bm @ ab) / sr) ** 2)
ZEROS = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351, 37.5862, 40.9187,
         43.3271, 48.0052, 49.7738, 52.9703, 56.4462, 59.3470]
best = gammas[np.argmax(power)]
print(f"peak: gamma={best:.2f}, dchi2={power.max():.1f}")
for z in ZEROS:
    print(f"  power at gamma={z:7.4f}: {power[np.argmin(abs(gammas - z))]:6.1f}")

LIGHT = dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e")
CAT = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
fig.patch.set_facecolor(LIGHT["surface"]); ax.set_facecolor(LIGHT["surface"])
ax.plot(gammas, power, color=CAT[0], lw=1.5, label="weighted periodogram (Δχ²)")
for j, z in enumerate(ZEROS):
    ax.axvline(z, color=CAT[2], lw=1, ls="--", alpha=0.7,
               label="Riemann zeros γ_k" if j == 0 else None)
ax.set_xlabel("γ (frequency of cos(γ·ln n))", color=LIGHT["text"])
ax.set_ylabel("Δχ²", color=LIGHT["text"])
ax.set_title("Periodogram of power-law residuals vs Riemann zero ordinates", color=LIGHT["text"])
leg = ax.legend(frameon=False, fontsize=9)
for t in leg.get_texts():
    t.set_color(LIGHT["text"])
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=LIGHT["muted"])
ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/zero_periodogram.png")
print("figure saved: figures/zero_periodogram.png")
