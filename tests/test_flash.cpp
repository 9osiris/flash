#include "../src/attn.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace flash;

static int passed = 0;
static int failed = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        passed++;
    } else {
        failed++;
        printf("FAIL: %s\n", name);
    }
}

static bool close(float a, float b, float tol) {
    return std::fabs(a - b) <= tol;
}

// random q/k/v, compare flash against naive at several shapes
static void compare_random(int n, int d, bool causal, int bq, int bk,
                           const char* name) {
    Rng rng(42 + n * 7 + d);
    std::vector<float> q(n * d), k(n * d), v(n * d);
    std::vector<float> o1(n * d), o2(n * d);
    for (int i = 0; i < n * d; i++) {
        q[i] = rng.uniform(-1.0f, 1.0f);
        k[i] = rng.uniform(-1.0f, 1.0f);
        v[i] = rng.uniform(-1.0f, 1.0f);
    }
    naive_attention(q.data(), k.data(), v.data(), n, d, causal, 0.0f,
                    o1.data());
    flash_attention(q.data(), k.data(), v.data(), n, d, causal, 0.0f,
                    bq, bk, o2.data());
    check(max_abs_diff(o1.data(), o2.data(), n, d) < 1e-5f, name);
}

int main() {
    // single row: softmax of one score is 1, output is just the v row
    {
        float q[4] = {1, 2, 3, 4};
        float k[4] = {5, 6, 7, 8};
        float v[4] = {9, 10, 11, 12};
        float o[4];
        naive_attention(q, k, v, 1, 4, true, 0.0f, o);
        check(close(o[0], 9, 1e-6) && close(o[1], 10, 1e-6) &&
                  close(o[2], 11, 1e-6) && close(o[3], 12, 1e-6),
              "n=1 naive output is the v row");
        flash_attention(q, k, v, 1, 4, true, 0.0f, 32, 32, o);
        check(close(o[0], 9, 1e-6) && close(o[3], 12, 1e-6),
              "n=1 flash output is the v row");
    }

    // hand case: q=k=identity rows, v distinct, no mask
    // row 0 attends mostly to itself, scores are 1/sqrt(2) vs 0
    {
        float q[4] = {1, 0, 0, 1};
        float k[4] = {1, 0, 0, 1};
        float v[4] = {10, 0, 0, 20};
        float o[4];
        naive_attention(q, k, v, 2, 2, false, 0.0f, o);
        float s = std::exp(1.0f / std::sqrt(2.0f));
        float w = s / (s + 1.0f);
        check(close(o[0], w * 10, 1e-5) && close(o[3], w * 20, 1e-5),
              "hand case naive matches the softmax by hand");
        flash_attention(q, k, v, 2, 2, false, 0.0f, 32, 32, o);
        check(close(o[0], w * 10, 1e-5) && close(o[3], w * 20, 1e-5),
              "hand case flash matches the softmax by hand");
    }

    // flash matches naive across shapes, masks, and block sizes
    compare_random(2, 2, false, 32, 32, "flash matches naive 2x2");
    compare_random(3, 5, true, 32, 32, "flash matches naive 3x5 causal");
    compare_random(5, 3, false, 2, 2, "flash matches naive tiny blocks 2x2");
    compare_random(17, 8, true, 4, 4, "flash matches naive 17x8 blocks 4");
    compare_random(33, 16, false, 16, 8, "flash matches naive 33x16 mixed blocks");
    compare_random(64, 64, true, 32, 32, "flash matches naive 64x64 causal");
    compare_random(129, 32, true, 16, 16, "flash matches naive 129x32 causal");
    compare_random(128, 64, false, 64, 64, "flash matches naive 128x64 big blocks");
    compare_random(7, 1, true, 32, 32, "flash matches naive d=1 causal");
    compare_random(16, 128, false, 8, 8, "flash matches naive d=128");

    // causal: row i must not see keys past i
    {
        const int n = 8, d = 4;
        Rng rng(7);
        std::vector<float> q(n * d), k(n * d), v(n * d);
        std::vector<float> o1(n * d), o2(n * d);
        for (int i = 0; i < n * d; i++) {
            q[i] = rng.uniform(-1, 1);
            k[i] = rng.uniform(-1, 1);
            v[i] = rng.uniform(-1, 1);
        }
        flash_attention(q.data(), k.data(), v.data(), n, d, true, 0.0f,
                        4, 4, o1.data());
        // scramble every key/value row past row 2
        for (int i = 3 * d; i < n * d; i++) {
            k[i] = 100.0f;
            v[i] = -100.0f;
        }
        flash_attention(q.data(), k.data(), v.data(), n, d, true, 0.0f,
                        4, 4, o2.data());
        bool same = true;
        for (int i = 0; i < 3 * d; i++) {
            if (!close(o1[i], o2[i], 1e-6f)) same = false;
        }
        check(same, "causal rows ignore later keys");
        bool changed = false;
        for (int i = 3 * d; i < n * d; i++) {
            if (!close(o1[i], o2[i], 1e-6f)) changed = true;
        }
        check(changed, "causal later rows do see the scramble");
    }

    // rows are convex combos of v rows, so each output stays in range
    {
        const int n = 16, d = 8;
        Rng rng(11);
        std::vector<float> q(n * d), k(n * d), v(n * d), o(n * d);
        for (int i = 0; i < n * d; i++) {
            q[i] = rng.uniform(-2, 2);
            k[i] = rng.uniform(-2, 2);
            v[i] = rng.uniform(-3, 3);
        }
        flash_attention(q.data(), k.data(), v.data(), n, d, false, 0.0f,
                        8, 8, o.data());
        bool in_range = true;
        for (int t = 0; t < d; t++) {
            float lo = v[t], hi = v[t];
            for (int j = 0; j < n; j++) {
                lo = std::fmin(lo, v[j * d + t]);
                hi = std::fmax(hi, v[j * d + t]);
            }
            for (int i = 0; i < n; i++) {
                if (o[i * d + t] < lo - 1e-4f || o[i * d + t] > hi + 1e-4f) {
                    in_range = false;
                }
            }
        }
        check(in_range, "flash outputs stay inside the v range");
    }

    // huge scores: exp would overflow without the running max
    {
        const int n = 32, d = 16;
        Rng rng(21);
        std::vector<float> q(n * d), k(n * d), v(n * d);
        std::vector<float> o1(n * d), o2(n * d);
        for (int i = 0; i < n * d; i++) {
            q[i] = rng.uniform(-100, 100);
            k[i] = rng.uniform(-100, 100);
            v[i] = rng.uniform(-1, 1);
        }
        naive_attention(q.data(), k.data(), v.data(), n, d, true, 0.0f,
                        o1.data());
        flash_attention(q.data(), k.data(), v.data(), n, d, true, 0.0f,
                        8, 8, o2.data());
        bool finite = true;
        for (int i = 0; i < n * d; i++) {
            if (!std::isfinite(o2[i])) finite = false;
        }
        check(finite, "flash stays finite on huge scores");
        check(max_abs_diff(o1.data(), o2.data(), n, d) < 1e-3f,
              "flash matches naive on huge scores");
    }

    // determinism: same input, same output, twice
    {
        const int n = 24, d = 12;
        Rng rng(31);
        std::vector<float> q(n * d), k(n * d), v(n * d);
        std::vector<float> o1(n * d), o2(n * d);
        for (int i = 0; i < n * d; i++) {
            q[i] = rng.uniform(-1, 1);
            k[i] = rng.uniform(-1, 1);
            v[i] = rng.uniform(-1, 1);
        }
        flash_attention(q.data(), k.data(), v.data(), n, d, true, 0.0f,
                        16, 16, o1.data());
        flash_attention(q.data(), k.data(), v.data(), n, d, true, 0.0f,
                        16, 16, o2.data());
        check(max_abs_diff(o1.data(), o2.data(), n, d) == 0.0f,
              "flash is deterministic");
    }

    // multi-head matches per-head naive, flash and naive paths
    {
        const int heads = 4, n = 16, d = 8;
        Rng rng(55);
        std::vector<float> q(heads * n * d), k(heads * n * d),
            v(heads * n * d);
        std::vector<float> o_flash(heads * n * d), o_naive(heads * n * d);
        for (size_t i = 0; i < q.size(); i++) {
            q[i] = rng.uniform(-1, 1);
            k[i] = rng.uniform(-1, 1);
            v[i] = rng.uniform(-1, 1);
        }
        multi_head_attention(q.data(), k.data(), v.data(), heads, n, d,
                             true, 0.0f, true, o_flash.data());
        multi_head_attention(q.data(), k.data(), v.data(), heads, n, d,
                             true, 0.0f, false, o_naive.data());
        check(max_abs_diff(o_flash.data(), o_naive.data(), heads * n, d) <
                  1e-5f,
              "multi-head flash matches multi-head naive");
        // heads are independent: scrambling head 3 must not move head 0
        for (size_t i = 3 * n * d; i < q.size(); i++) {
            k[i] = 50.0f;
            v[i] = 50.0f;
        }
        std::vector<float> o_again(heads * n * d);
        multi_head_attention(q.data(), k.data(), v.data(), heads, n, d,
                             true, 0.0f, true, o_again.data());
        bool head0_same = true;
        for (int i = 0; i < n * d; i++) {
            if (!close(o_flash[i], o_again[i], 1e-6f)) head0_same = false;
        }
        check(head0_same, "heads are independent of each other");
    }

    // empty input returns without touching anything
    {
        float o[1] = {7.0f};
        naive_attention(nullptr, nullptr, nullptr, 0, 4, false, 0.0f, o);
        flash_attention(nullptr, nullptr, nullptr, 0, 4, false, 0.0f, 32, 32,
                        o);
        check(o[0] == 7.0f, "n=0 leaves output alone");
    }

    // explicit scale is honored, not silently replaced
    {
        const int n = 8, d = 4;
        Rng rng(77);
        std::vector<float> q(n * d), k(n * d), v(n * d);
        std::vector<float> o1(n * d), o2(n * d), o3(n * d);
        for (int i = 0; i < n * d; i++) {
            q[i] = rng.uniform(-1, 1);
            k[i] = rng.uniform(-1, 1);
            v[i] = rng.uniform(-1, 1);
        }
        naive_attention(q.data(), k.data(), v.data(), n, d, false, 1.0f,
                        o1.data());
        naive_attention(q.data(), k.data(), v.data(), n, d, false, 0.5f,
                        o2.data());
        flash_attention(q.data(), k.data(), v.data(), n, d, false, 1.0f,
                        8, 8, o3.data());
        check(max_abs_diff(o1.data(), o3.data(), n, d) < 1e-5f,
              "flash honors explicit scale");
        check(max_abs_diff(o1.data(), o2.data(), n, d) > 1e-3f,
              "different scales give different outputs");
    }

    // max_abs_diff itself
    {
        float a[3] = {1, 2, 3};
        float b[3] = {1, 2, 5};
        check(close(max_abs_diff(a, b, 3, 1), 2.0f, 1e-6f),
              "max_abs_diff finds the worst element");
    }

    printf("%d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
