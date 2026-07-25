# -*- coding: utf-8 -*-
# 零点检测稳健性检查：在不同窗口宽度 W 和不同 I(n) 离散化下重跑周期图，
# 确认零点处的功率峰不随分析选择漂移。
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
ZEROS = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351, 37.5862, 40.9187, 43.3271]

is_prime = np.ones(N + 1, dtype=bool)
is_prime[:2] = False
for p in range(2, int(N**0.5) + 1):
    if is_prime[p]:
        is_prime[p * p :: p] = False
A = is_prime.astype(np.float64)
B = np.zeros(N + 1, dtype=np.float64)
B[2 * np.nonzero(is_prime[: N // 2 + 1])[0]] = 1.0
r = np.rint(np.fft.irfft(np.fft.rfft(A, 1 << 25) * np.fft.rfft(B, 1 << 25),
                         1 << 25)[: N + 1]).astype(np.int64)
del A, B
print("r(n) ready")

Kcache = {}
def K_of(n):
    v = Kcache.get(n)
    if v is not None:
        return v
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
    Kcache[n] = k
    return k

centers = np.unique((np.logspace(4.7, 7, 220)).astype(np.int64) | 1)
centers = centers[centers <= N - 2 * 500 - 2]      # 留出最大窗口余量

def I0_sum(n0):
    qv = np.arange(3.0, (n0 - 3) // 2 + 1)
    return float(np.sum(1.0 / (np.log(qv) * np.log(n0 - 2 * qv))))

def I0_mid(n0):                                     # 中点积分离散化
    a, b = 2.5, (n0 - 3) // 2 + 0.5
    t = np.linspace(a, b, 200001)
    h = (b - a) / 200000
    f = 1.0 / (np.log(t) * np.log(n0 - 2 * t))
    return float(h * (f.sum() - 0.5 * (f[0] + f[-1])))

I_sum = np.array([I0_sum(c) for c in centers]); print("I_sum ready")
I_mid = np.array([I0_mid(c) for c in centers]); print("I_mid ready")

def periodogram(W, Ivals, label):
    ys, ss = [], []
    for j, n0 in enumerate(centers):
        ms = np.arange(n0 - 2 * W, n0 + 2 * W + 1, 2)
        vals = np.array([r[m] / (K_of(int(m)) * Ivals[j]) for m in ms])
        ys.append(vals.mean()); ss.append(vals.std(ddof=1) / np.sqrt(len(vals)))
    y, s = np.array(ys), np.array(ss)
    gap = TARGET - y
    m = gap > 0
    L, g, sg = np.log(centers[m].astype(float)), gap[m], s[m]
    w = (g / sg) ** 2
    X = np.vstack([np.ones_like(L), L]).T
    beta = np.linalg.solve(X.T @ np.diag(w) @ X, X.T @ np.diag(w) @ np.log(g))
    model = np.exp(X @ beta)
    res, sr = (g - model) / model, sg / model
    chi2_0 = np.sum((res / sr) ** 2)
    gammas = np.linspace(3, 60, 1141)
    power = np.empty_like(gammas)
    for i, gam in enumerate(gammas):
        Bm = np.vstack([np.cos(gam * L), np.sin(gam * L)]).T
        ab = np.linalg.solve(Bm.T @ (Bm / sr[:, None] ** 2), Bm.T @ (res / sr**2))
        power[i] = chi2_0 - np.sum(((res - Bm @ ab) / sr) ** 2)
    peak = gammas[np.argmax(power)]
    at_zeros = [power[np.argmin(abs(gammas - z))] for z in ZEROS]
    print(f"{label:14s} theta={-beta[1]:.4f} peak_gamma={peak:6.2f} "
          f"dchi2={power.max():5.1f} | " +
          " ".join(f"g{k+1}:{v:5.1f}" for k, v in enumerate(at_zeros)))
    return gammas, power

configs = [(250, I_sum, "W=250, sum (baseline)"), (100, I_sum, "W=100, sum"),
           (500, I_sum, "W=500, sum"), (250, I_mid, "W=250, midpoint integral")]
LIGHT = dict(surface="#fcfcfb", text="#0b0b0b", muted="#52514e")
CAT = ["#2a78d6", "#008300", "#e87ba4", "#eda100"]
fig, ax = plt.subplots(figsize=(9.5, 5.5), dpi=150)
fig.patch.set_facecolor(LIGHT["surface"]); ax.set_facecolor(LIGHT["surface"])
for (W, Iv, lab), c in zip(configs, CAT):
    gam, pw = periodogram(W, Iv, lab)
    ax.plot(gam, pw, color=c, lw=1.3, label=lab, alpha=0.9)
for j, z in enumerate(ZEROS):
    ax.axvline(z, color="#52514e", lw=0.8, ls="--", alpha=0.5,
               label="Riemann zeros γ_k" if j == 0 else None)
ax.set_xlabel("γ", color=LIGHT["text"])
ax.set_ylabel("Δχ²", color=LIGHT["text"])
ax.set_title("Periodogram robustness: window width and I(n) discretization", color=LIGHT["text"])
leg = ax.legend(frameon=False, fontsize=9)
for t in leg.get_texts():
    t.set_color(LIGHT["text"])
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=LIGHT["muted"])
ax.grid(axis="y", color="#e6e5e1", linewidth=0.6)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("figures/zero_robustness.png")
print("figure saved: figures/zero_robustness.png")
