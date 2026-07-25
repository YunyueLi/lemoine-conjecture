# Lemoine 猜想的数值研究（论文项目）

**目标**：发表一篇实验数学论文。目标期刊（按优先级）：
*Integers: Electronic J. of Combinatorial Number Theory* → *Journal of Integer Sequences* → *Experimental Mathematics*。

**猜想**：每个大于 5 的奇数 n 都可写成 n = p + 2q，p、q 为素数（Lemoine, 1895；又称 Levy 猜想）。

## 文献格局（2026-07-24 核查）

- 正式可引用的穷尽验证只到 **10⁹**（Corbitt，经 MathWorld 转引）；一篇 2019 年博客称 10¹⁰。
- OEIS A002091 注释（Pfoertner, 2011）隐含核查到 **10¹³**，但无正式文献。
- IJMTT 2026 年一篇论文只做 600–1500 位随机奇数的**抽查**（Miller–Rabin），不构成穷尽验证，
  且该刊为低质量期刊；论文中可引用并与穷尽验证明确区分。
- Agama–Gensel 声称的证明（arXiv:1709.05335）已被确认有漏洞，猜想仍开放。
- **结论**：把验证状态用一篇可复现论文统一并推到 10¹²–10¹³⁺，同时给出两侧记录序列，是真实空白。

## 已有结果（全部可复现）

| 内容 | 状态 | 产物 |
|---|---|---|
| **穷尽验证到 10¹²：零反例**（3314 秒 / 8 线程） | 完成 | `data/verify2_1e12_summary.txt`、`data/run1e12/` |
| 穷尽验证到 2×10¹³（冲 OEIS 记录，ETA ~16h，caffeinate+nohup） | 进行中 | `data/verify2_2e13_*.{txt,log}` |
| A 侧记录（n=p+2q 最小 q）：最大 3079 @ n=853 036 242 229（≤10¹²，共 34 项） | 完成 | `data/run1e12/v2_records_A.csv` |
| B 侧记录（n=2p+q 最小素数 q）：最大 7129 @ n=470 243 683 103（共 43 项） | 完成 | `data/run1e12/v2_records_B_prime.csv` |
| **B' 侧与 OEIS A002091 b-file 在 10¹² 内 41 项逐项吻合（完整交叉验证）** | 完成 | `data/run1e12/v2_records_B_noncomposite.csv` |
| r(n) 精确计数（FFT，全部奇数 n ≤ 10⁷）+ 窗口卷积延伸到 10⁸ | 完成 | `src/repstats.py`、`src/extend_powerlaw.py` |
| Lemoine 彗星图（分带比例 1 : 4/3 : 2 : 8/3 与奇异级数一致） | 完成 | `figures/comet.png` |
| HL 偏差幂律：θ = 0.519 ± 0.007（近 4 个量级） | 完成 | `figures/second_order_1e8.png` |
| **周期图在幂律残差中探测到黎曼零点（γ₆ 峰 Δχ²=55.5；13 零点总功率 219.5 vs 本底 26）** | 完成 | `figures/zero_periodogram.png`、`src/zero_scan.py` |
| 论文草稿（英文，tectonic 编译通过，含零点检测一节） | 持续更新 | `paper/main.tex` → `paper/main.pdf` |

## 待办（按顺序）

- [ ] 等 2×10¹³ 跑完（约 16 小时）：若无反例，超越 Pfoertner 的 10¹³ 非正式记录，
      并可向 OEIS A002091 提交 (10¹³, 2×10¹³] 段的新项；
- [x] 零点检测稳健性（4 配置峰位锁定零点，θ∈[0.517,0.525]）→ `figures/zero_robustness.png`
- [x] 13 零点联合拟合：Δχ²=258，500 组随机频率零假设最大仅 124.5（p<0.002）→ `figures/zero_amplitudes.png`、`src/zero_joint_fit.py`
- [ ] 论文剩余 TODO：奇异级数推导化简核对、Cramér 模型对比、补齐参考文献；
- [ ] 建 GitHub 仓库开源代码与数据；
- [ ] 英文润色 → arXiv 预印本 → 投 Integers。

## 复现

```
clang++ -std=c++17 -O2 -march=native -pthread -o src/verify2 src/verify2.cpp
nohup ./src/verify2 1000000000000 8 > data/verify2_1e12_summary.txt 2> data/verify2_1e12_progress.log &
./.venv/bin/python src/repstats.py      # 表示数统计与彗星图
./.venv/bin/python src/second_order.py  # 幂律偏差分析
cd paper && tectonic main.tex           # 编译论文
```

注意：长时计算务必用 nohup 脱离会话启动，否则会话退出会杀掉后台任务（已发生两次）。
