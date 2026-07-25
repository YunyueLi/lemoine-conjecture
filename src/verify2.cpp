// Lemoine 猜想分段筛多线程验证程序（第二版）。
// 对每个奇数 n 同时计算三种最小分解：
//   A 侧：n = p + 2q，最小的 q（q 为素数）
//   B 侧：n = 2p + q，最小的素数 q
//   B'侧：n = 2p + q，最小的非合数 q（允许 q = 1，用于与 OEIS A002091 交叉验证）
// 输出（data/ 下）：三个记录序列 CSV、A/B 侧最小值直方图 CSV。
// 用法：./verify2 NMAX [threads]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <chrono>
using namespace std;

static const uint64_t SEG = 1ULL << 28;      // 每段覆盖的自然数个数
static const uint32_t QMAX = 200000;         // 候选 q 上限（远大于已知记录）

struct Rec { uint64_t n; uint32_t q; };

int main(int argc, char** argv) {
    uint64_t NMAX = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 1000000000000ULL;
    unsigned nthreads = (argc > 2) ? (unsigned)atoi(argv[2]) : 8;
    auto t0 = chrono::steady_clock::now();

    // 基础素数：到 max(sqrt(NMAX), QMAX)，后者保证候选 q 列表完整
    uint64_t R = 1; while ((R + 1) * (R + 1) <= NMAX) R++;
    uint64_t L = max(R, (uint64_t)QMAX);
    vector<uint8_t> comp(L + 1, 0);
    vector<uint32_t> base;
    for (uint64_t i = 2; i <= L; i++) {
        if (!comp[i]) {
            base.push_back((uint32_t)i);
            for (uint64_t j = i * i; j <= L; j += i) comp[j] = 1;
        }
    }
    // 候选 q 列表（窗口回看余量按 QMAX 设计；筛段时只用 <= sqrt(hi) 的素数）
    vector<uint32_t> qs;
    for (uint32_t v : base) { if (v >= QMAX) break; qs.push_back(v); }
    fprintf(stderr, "base primes: %zu (R=%" PRIu64 "), q candidates: %zu\n",
            base.size(), R, qs.size());

    atomic<uint64_t> next_seg{0}, fail_count{0};
    atomic<uint64_t> segs_done{0};
    uint64_t total_segs = (NMAX + SEG - 1) / SEG;
    mutex merge_mtx;
    vector<Rec> candA, candB, candBn;                    // 各线程的候选记录（合并后再全局筛）
    vector<uint64_t> histA(qs.size(), 0), histB(qs.size(), 0);

    auto worker = [&]() {
        // 窗口 1：覆盖 [wlo1, hi] 的奇数位图（供 A 侧查 n-2q）
        // 窗口 2：覆盖 [wlo2, hi/2] 的全数位图（供 B 侧查 (n-q)/2）
        vector<uint64_t> w1((SEG / 2 + 2 * QMAX) / 64 + 2), w2((SEG / 2 + QMAX) / 64 + 2);
        vector<Rec> rA, rB, rBn;
        vector<uint64_t> hA(qs.size(), 0), hB(qs.size(), 0);
        uint32_t maxA = 0, maxB = 0, maxBn = 0;

        for (;;) {
            uint64_t s = next_seg.fetch_add(1);
            uint64_t lo = s * SEG + 1;                    // 奇数起点
            if (lo > NMAX) break;
            uint64_t hi = min(lo + SEG - 1, NMAX);

            // ---- 窗口 1：奇数位图，idx i 代表 wlo1 + 2i ----
            uint64_t wlo1 = (lo > 2ULL * QMAX + 3) ? lo - 2ULL * QMAX : 3;
            if (!(wlo1 & 1)) wlo1++;
            uint64_t m1 = (hi - wlo1) / 2 + 1;
            fill(w1.begin(), w1.end(), ~0ULL);
            for (uint32_t p : base) {
                if (p == 2) continue;
                uint64_t pp = (uint64_t)p * p;
                if (pp > hi) break;
                uint64_t st = max(pp, ((wlo1 + p - 1) / p) * (uint64_t)p);
                if (!(st & 1)) st += p;                   // 只标奇倍数
                for (uint64_t j = st; j <= hi; j += 2ULL * p)
                    w1[(j - wlo1) / 2 >> 6] &= ~(1ULL << (((j - wlo1) / 2) & 63));
            }
            auto prime1 = [&](uint64_t x) -> bool {       // x 为奇数且 >= wlo1
                if (x < wlo1) return false;
                if (x == 1) return false;
                uint64_t i = (x - wlo1) / 2;
                return (w1[i >> 6] >> (i & 63)) & 1;
            };

            // ---- 窗口 2：全数位图，idx i 代表 wlo2 + i ----
            uint64_t hi2 = (hi - 1) / 2;
            uint64_t wlo2 = (lo > QMAX + 4) ? (lo - QMAX) / 2 : 2;
            uint64_t m2 = hi2 - wlo2 + 1;
            fill(w2.begin(), w2.end(), ~0ULL);
            for (uint32_t p : base) {
                uint64_t pp = (uint64_t)p * p;
                if (pp > hi2) break;
                uint64_t st = max(pp, ((wlo2 + p - 1) / p) * (uint64_t)p);
                for (uint64_t j = st; j <= hi2; j += p)
                    w2[(j - wlo2) >> 6] &= ~(1ULL << ((j - wlo2) & 63));
            }
            auto prime2 = [&](uint64_t x) -> bool {       // x >= wlo2 且 <= hi2
                if (x < 2 || x < wlo2) return false;
                uint64_t i = x - wlo2;
                return (w2[i >> 6] >> (i & 63)) & 1;
            };
            (void)m1; (void)m2;

            // ---- 主循环 ----
            for (uint64_t n = (lo < 7 ? 7 : lo) | 1; n <= hi; n += 2) {
                // A 侧：n = p + 2q，最小 q
                bool ok = false;
                for (size_t i = 0; i < qs.size(); i++) {
                    uint64_t twoq = 2ULL * qs[i];
                    if (twoq + 3 > n) break;
                    if (prime1(n - twoq)) {
                        hA[i]++;
                        if (qs[i] > maxA) { maxA = qs[i]; rA.push_back({n, qs[i]}); }
                        ok = true; break;
                    }
                }
                if (!ok) { fail_count++; printf("FAIL A side n=%" PRIu64 "\n", n); }

                // B' 侧：q = 1 的情形（n = 2p + 1）
                bool bn_done = false;
                if (prime2((n - 1) / 2)) {
                    if (1 > maxBn) { maxBn = 1; rBn.push_back({n, 1}); }
                    bn_done = true;
                }
                // B 侧：n = 2p + q，最小素数 q（q 为奇素数，跳过 qs[0]=2）
                ok = false;
                for (size_t i = 1; i < qs.size(); i++) {
                    if ((uint64_t)qs[i] + 4 > n) break;
                    if (prime2((n - qs[i]) / 2)) {
                        hB[i]++;
                        if (qs[i] > maxB) { maxB = qs[i]; rB.push_back({n, qs[i]}); }
                        if (!bn_done && qs[i] > maxBn) { maxBn = qs[i]; rBn.push_back({n, qs[i]}); }
                        ok = true; break;
                    }
                }
                if (!ok && n > 8) { fail_count++; printf("FAIL B side n=%" PRIu64 "\n", n); }
            }
            uint64_t done = segs_done.fetch_add(1) + 1;
            if (done % 64 == 0)
                fprintf(stderr, "progress %" PRIu64 "/%" PRIu64 " segs (%.0fs)\n", done, total_segs,
                        chrono::duration<double>(chrono::steady_clock::now() - t0).count());
        }
        lock_guard<mutex> g(merge_mtx);
        candA.insert(candA.end(), rA.begin(), rA.end());
        candB.insert(candB.end(), rB.begin(), rB.end());
        candBn.insert(candBn.end(), rBn.begin(), rBn.end());
        for (size_t i = 0; i < qs.size(); i++) { histA[i] += hA[i]; histB[i] += hB[i]; }
    };

    vector<thread> pool;
    for (unsigned i = 0; i < nthreads; i++) pool.emplace_back(worker);
    for (auto& t : pool) t.join();

    // 合并候选记录：按 n 排序后线性扫描取真全局记录
    auto finalize = [](vector<Rec>& v, const char* path) {
        sort(v.begin(), v.end(), [](const Rec& a, const Rec& b) { return a.n < b.n; });
        FILE* f = fopen(path, "w");
        fprintf(f, "n,q_min\n");
        uint32_t mx = 0;
        for (const Rec& r : v)
            if (r.q > mx) { mx = r.q; fprintf(f, "%" PRIu64 ",%u\n", r.n, r.q); }
        fclose(f);
        return mx;
    };
    uint32_t mA = finalize(candA, "data/v2_records_A.csv");
    uint32_t mB = finalize(candB, "data/v2_records_B_prime.csv");
    uint32_t mBn = finalize(candBn, "data/v2_records_B_noncomposite.csv");
    auto dump_hist = [&](vector<uint64_t>& h, const char* path) {
        FILE* f = fopen(path, "w");
        fprintf(f, "q,count\n");
        for (size_t i = 0; i < qs.size(); i++)
            if (h[i]) fprintf(f, "%u,%" PRIu64 "\n", qs[i], h[i]);
        fclose(f);
    };
    dump_hist(histA, "data/v2_qmin_hist_A.csv");
    dump_hist(histB, "data/v2_qmin_hist_B.csv");

    printf("NMAX=%" PRIu64 " threads=%u fails=%" PRIu64
           " maxA=%u maxB=%u maxBn=%u total %.0fs\n",
           NMAX, nthreads, fail_count.load(), mA, mB, mBn,
           chrono::duration<double>(chrono::steady_clock::now() - t0).count());
    return 0;
}
