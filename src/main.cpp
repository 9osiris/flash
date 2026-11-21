#include "attn.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace flash;

static const char* kVersion = "1.0.0";

static void usage() {
    printf("flash %s - flash attention from scratch\n", kVersion);
    printf("\n");
    printf("  flash demo                 tiny 4x4 example, prints both outputs\n");
    printf("  flash verify [n] [d]       flash vs naive, prints max abs diff\n");
    printf("  flash bench [n] [d]        time naive vs flash, prints ms and speedup\n");
    printf("  flash version              print version\n");
    printf("\n");
    printf("  verify and bench default to n=512 d=64, causal on.\n");
    printf("  pass --no-causal after the numbers to turn the mask off.\n");
}

static void print_mat(const char* name, const float* m, int n, int d) {
    printf("%s:\n", name);
    for (int i = 0; i < n; i++) {
        printf("  ");
        for (int t = 0; t < d; t++) {
            printf("%8.4f ", m[i * d + t]);
        }
        printf("\n");
    }
}

static int demo() {
    const int n = 4, d = 4;
    // small fixed matrices, easy to eyeball
    float q[16] = {1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  1, 1, 0, 0};
    float k[16] = {1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1};
    float v[16] = {1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12,  13, 14, 15, 16};
    float o_naive[16], o_flash[16];
    naive_attention(q, k, v, n, d, true, 0.0f, o_naive);
    flash_attention(q, k, v, n, d, true, 0.0f, 32, 32, o_flash);
    print_mat("naive (causal)", o_naive, n, d);
    print_mat("flash (causal)", o_flash, n, d);
    printf("max abs diff: %.6f\n", max_abs_diff(o_naive, o_flash, n, d));
    return 0;
}

static int verify(int n, int d, bool causal) {
    Rng rng(1234);
    std::vector<float> q(n * d), k(n * d), v(n * d);
    std::vector<float> o_naive(n * d), o_flash(n * d);
    for (int i = 0; i < n * d; i++) {
        q[i] = rng.uniform(-1.0f, 1.0f);
        k[i] = rng.uniform(-1.0f, 1.0f);
        v[i] = rng.uniform(-1.0f, 1.0f);
    }
    naive_attention(q.data(), k.data(), v.data(), n, d, causal, 0.0f,
                    o_naive.data());
    flash_attention(q.data(), k.data(), v.data(), n, d, causal, 0.0f,
                    32, 32, o_flash.data());
    float diff = max_abs_diff(o_naive.data(), o_flash.data(), n, d);
    printf("n=%d d=%d causal=%s max abs diff: %.8f\n", n, d,
           causal ? "yes" : "no", diff);
    if (diff > 1e-5f) {
        printf("mismatch, flash does not match naive\n");
        return 1;
    }
    printf("match\n");
    return 0;
}

static double time_it(void (*fn)(const float*, const float*, const float*,
                                  int, int, bool, float, float*),
                      const float* q, const float* k, const float* v,
                      int n, int d, float* o) {
    auto start = std::chrono::steady_clock::now();
    fn(q, k, v, n, d, true, 0.0f, o);
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

static int bench(int n, int d) {
    Rng rng(99);
    std::vector<float> q(n * d), k(n * d), v(n * d), o(n * d);
    for (int i = 0; i < n * d; i++) {
        q[i] = rng.uniform(-1.0f, 1.0f);
        k[i] = rng.uniform(-1.0f, 1.0f);
        v[i] = rng.uniform(-1.0f, 1.0f);
    }
    // warmup so the timing isnt cold cache noise
    naive_attention(q.data(), k.data(), v.data(), n, d, true, 0.0f, o.data());
    flash_attention(q.data(), k.data(), v.data(), n, d, true, 0.0f, 32, 32,
                    o.data());
    double t_naive = time_it(naive_attention, q.data(), k.data(), v.data(),
                             n, d, o.data());
    // flash takes block sizes, wrap it to the same shape
    auto flash_wrap = [](const float* qq, const float* kk, const float* vv,
                         int nn, int dd, bool cc, float ss, float* oo) {
        flash_attention(qq, kk, vv, nn, dd, cc, ss, 32, 32, oo);
    };
    double t_flash = time_it(flash_wrap, q.data(), k.data(), v.data(), n, d,
                             o.data());
    printf("n=%d d=%d causal, single head, float32\n", n, d);
    printf("naive: %.2f ms\n", t_naive);
    printf("flash: %.2f ms\n", t_flash);
    printf("speedup: %.2fx\n", t_naive / t_flash);
    printf("note: this box is one thread and no simd, the win here is\n");
    printf("memory traffic, not flops. on a gpu the gap is much bigger.\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        usage();
        return 0;
    }
    if (strcmp(argv[1], "demo") == 0) return demo();
    if (strcmp(argv[1], "version") == 0) {
        printf("flash %s\n", kVersion);
        return 0;
    }
    if (strcmp(argv[1], "verify") == 0 || strcmp(argv[1], "bench") == 0) {
        int n = 512, d = 64;
        bool causal = true;
        int seen = 0;
        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--no-causal") == 0) {
                causal = false;
            } else if (atoi(argv[i]) > 0 && seen == 0) {
                n = atoi(argv[i]);
                seen++;
            } else if (atoi(argv[i]) > 0 && seen == 1) {
                d = atoi(argv[i]);
                seen++;
            }
        }
        if (strcmp(argv[1], "verify") == 0) return verify(n, d, causal);
        return bench(n, d);
    }
    printf("unknown command: %s\n\n", argv[1]);
    usage();
    return 1;
}
