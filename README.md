# Lemoine's Conjecture: Exhaustive Verification and Representation Statistics

Code and data accompanying the paper *Exhaustive Verification, Minimal-Prime
Records, and Representation Statistics for Lemoine's Conjecture* (draft in
`paper/`).

Lemoine's conjecture (1895): every odd integer $n > 5$ can be written as
$n = p + 2q$ with $p, q$ prime.

## Results

- **Exhaustive verification for all odd $n \le 10^{12}$: no counterexample**
  (55 minutes on 8 consumer cores; a run to $2\times10^{13}$ is in progress).
- Record sequences of the minimal prime $q$ for both decompositions
  $n = p + 2q$ and $n = 2p + q$; the noncomposite variant reproduces the
  OEIS [A002091](https://oeis.org/A002091) b-file exactly on
  $9 \le n \le 10^{12}$ (41 terms), cross-validating both computations.
- Exact representation counts $r(n)$ for all odd $n \le 10^7$ (FFT
  convolution) plus windowed exact counts to $10^8$, confirming the
  Hardy–Littlewood-type asymptotic $r(n) \approx 2C_2 K(n) I(n)$.
- The deficit $2C_2 - r(n)/(K(n)I(n))$ decays as $c\,n^{-\theta}$ with
  $\theta = 0.519 \pm 0.007$ — square-root cancellation.
- A weighted periodogram of the power-law residuals detects the low-lying
  **Riemann zeros** ($\gamma_1,\gamma_2,\gamma_3,\gamma_4,\gamma_6,\dots$);
  a joint 13-zero fit gives $\Delta\chi^2 = 258$ against a random-frequency
  null whose maximum over 500 trials is 124.5.

## Layout

| Path | Contents |
|---|---|
| `src/verify.cpp` | simple odd-bitmap verifier (used for cross-validation) |
| `src/verify2.cpp` | segmented, multithreaded verifier; computes both decompositions and all record/histogram outputs |
| `src/repstats.py` | exact $r(n)$ via FFT; comet plot; Hardy–Littlewood ratio |
| `src/extend_powerlaw.py` | windowed exact counts to $10^8$; power-law fit |
| `src/second_order.py` | power-law vs $1/\log n$ model comparison |
| `src/zero_scan.py` | dense window sampling; periodogram vs Riemann zeros |
| `src/zero_robustness.py` | periodogram under 4 analysis configurations |
| `src/zero_joint_fit.py` | joint 13-zero fit; random-frequency null calibration |
| `data/` | record sequences, $q_{\min}$ histograms, fit points, run summaries |
| `figures/` | all paper figures (PNG) |
| `paper/main.tex` | paper draft (compiles with `tectonic`) |

## Reproduce

```sh
clang++ -std=c++17 -O2 -march=native -pthread -o src/verify2 src/verify2.cpp
./src/verify2 1000000000000 8          # exhaustive verification to 1e12

python3 -m venv .venv && ./.venv/bin/pip install numpy matplotlib
./.venv/bin/python src/repstats.py     # r(n), comet, HL ratio
./.venv/bin/python src/extend_powerlaw.py
./.venv/bin/python src/zero_scan.py
./.venv/bin/python src/zero_robustness.py
./.venv/bin/python src/zero_joint_fit.py
```

Project notes in Chinese: `PLAN.zh.md`.

## License

MIT
