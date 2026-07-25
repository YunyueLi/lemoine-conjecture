// Lemoine 猜想验证程序：每个奇数 n > 5 均可写成 n = p + 2q（p、q 为素数）。
// 策略：对每个奇数 n，按 q 从小到大搜索首个使 n - 2q 为素数的 q。
// 输出：
//   - 最小 q 的新记录（n, q_min）→ data/verify_records.csv
//   - 最小 q 的取值分布直方图 → data/qmin_hist.csv
// 用法：./verify N   （默认 N = 1e9）
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>
#include <vector>
#include <chrono>
using namespace std;

int main(int argc, char** argv) {
    uint64_t N = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 1000000000ULL;
    auto t0 = chrono::steady_clock::now();

    // 奇数位图筛：bit i 代表 2i+1
    uint64_t M = (N - 1) / 2 + 1;
    vector<uint64_t> bits((M + 63) / 64, ~0ULL);
    auto clearbit = [&](uint64_t i) { bits[i >> 6] &= ~(1ULL << (i & 63)); };
    auto testbit  = [&](uint64_t i) -> bool { return (bits[i >> 6] >> (i & 63)) & 1; };
    clearbit(0); // 1 不是素数
    for (uint64_t p = 3; p * p <= N; p += 2) {
        if (testbit((p - 1) / 2))
            for (uint64_t j = p * p; j <= N; j += 2 * p) clearbit((j - 1) / 2);
    }
    auto isprime = [&](uint64_t x) -> bool {
        if (x == 2) return true;
        if (x < 2 || !(x & 1)) return false;
        return testbit((x - 1) / 2);
    };
    fprintf(stderr, "sieve done: %.1fs\n",
            chrono::duration<double>(chrono::steady_clock::now() - t0).count());

    // 候选 q 列表（含 q=2）；上限取充分大，超出即报告失败
    vector<uint32_t> qs;
    for (uint32_t p = 2; p < 2000000; p++) if (isprime(p)) qs.push_back(p);

    vector<uint64_t> hist(qs.size(), 0); // 最小 q 的分布（按下标）
    FILE* frec = fopen("data/verify_records.csv", "w");
    fprintf(frec, "n,q_min\n");
    uint32_t record_q = 0;
    uint64_t checked = 0;

    for (uint64_t n = 7; n <= N; n += 2) {
        bool found = false;
        for (size_t i = 0; i < qs.size(); i++) {
            uint64_t twoq = 2ULL * qs[i];
            if (twoq + 2 > n) break;
            if (isprime(n - twoq)) {
                hist[i]++;
                if (qs[i] > record_q) {
                    record_q = qs[i];
                    fprintf(frec, "%" PRIu64 ",%u\n", n, qs[i]);
                    fflush(frec);
                }
                found = true;
                break;
            }
        }
        if (!found) {
            printf("COUNTEREXAMPLE CANDIDATE: n=%" PRIu64 "\n", n);
            fflush(stdout);
        }
        if (++checked % 100000000 == 0)
            fprintf(stderr, "progress: n=%" PRIu64 " (%.1fs)\n", n,
                    chrono::duration<double>(chrono::steady_clock::now() - t0).count());
    }
    fclose(frec);

    FILE* fh = fopen("data/qmin_hist.csv", "w");
    fprintf(fh, "q,count\n");
    for (size_t i = 0; i < qs.size(); i++)
        if (hist[i]) fprintf(fh, "%u,%" PRIu64 "\n", qs[i], hist[i]);
    fclose(fh);

    printf("verified all odd n in (5, %" PRIu64 "]: %" PRIu64 " numbers, max q_min = %u, total %.1fs\n",
           N, checked, record_q,
           chrono::duration<double>(chrono::steady_clock::now() - t0).count());
    return 0;
}
