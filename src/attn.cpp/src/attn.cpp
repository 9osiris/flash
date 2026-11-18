#include "attn.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace flash {

namespace {

// one row of scores, dot product of q row i against every k row
void score_row(const float* q, const float* k, int n, int d, int i,
               float scale, float* s) {
    for (int j = 0; j < n; j++) {
        float dot = 0.0f;
        for (int t = 0; t < d; t++) {
            dot += q[i * d + t] * k[j * d + t];
        }
        s[j] = dot * scale;
    }
}

}  // namespace

void naive_attention(const float* q, const float* k, const float* v,
                     int n, int d, bool causal, float scale, float* o) {
    if (n <= 0 || d <= 0) return;
    if (scale <= 0.0f) scale = 1.0f / std::sqrt((float)d);
    std::vector<float> s(n);
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; i++) {
        score_row(q, k, n, d, i, scale, s.data());
        // stable softmax, max first so big scores dont overflow exp
        float m = neg_inf;
        for (int j = 0; j < n; j++) {
            if (causal && j > i) continue;
            m = std::max(m, s[j]);
        }
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            if (causal && j > i) {
                s[j] = 0.0f;
            } else {
                s[j] = std::exp(s[j] - m);
                sum += s[j];
            }
        }
        for (int t = 0; t < d; t++) {
            float acc = 0.0f;
            for (int j = 0; j < n; j++) {
                acc += s[j] * v[j * d + t];
            }
            o[i * d + t] = acc / sum;
        }
    }
}

void flash_attention(const float* q, const float* k, const float* v,
                     int n, int d, bool causal, float scale,
                     int block_q, int block_k, float* o) {
    if (n <= 0 || d <= 0) return;
    if (scale <= 0.0f) scale = 1.0f / std::sqrt((float)d);
    if (block_q <= 0) block_q = 32;
    if (block_k <= 0) block_k = 32;
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // scratch, sized to one q block by one k block
    std::vector<float> s((size_t)block_q * block_k);
    std::vector<float> m(block_q);
    std::vector<float> l(block_q);

    for (int qb = 0; qb < n; qb += block_q) {
        int bq = std::min(block_q, n - qb);
        for (int i = 0; i < bq; i++) {
            m[i] = neg_inf;
            l[i] = 0.0f;
            for (int t = 0; t < d; t++) o[(qb + i) * d + t] = 0.0f;
        }
        for (int kb = 0; kb < n; kb += block_k) {
            int bk = std::min(block_k, n - kb);
            // scores for this tile, masked positions stay -inf
            for (int i = 0; i < bq; i++) {
                for (int j = 0; j < bk; j++) {
                    int gi = qb + i;
                    int gj = kb + j;
                    if (causal && gj > gi) {
                        s[i * bk + j] = neg_inf;
                        continue;
                    }
                    float dot = 0.0f;
                    for (int t = 0; t < d; t++) {
                        dot += q[gi * d + t] * k[gj * d + t];
                    }
                    s[i * bk + j] = dot * scale;
                }
            }
            // online softmax: fold this tile into the running state
            for (int i = 0; i < bq; i++) {
                float tile_max = neg_inf;
                for (int j = 0; j < bk; j++) {
                    tile_max = std::max(tile_max, s[i * bk + j]);
                }
                float m_new = std::max(m[i], tile_max);
                // rescale everything seen so far to the new max
                float alpha = (m[i] == neg_inf) ? 0.0f : std::exp(m[i] - m_new);
                float row_sum = 0.0f;
                for (int j = 0; j < bk; j++) {
                    float p = (s[i * bk + j] == neg_inf)
                                  ? 0.0f
                                  : std::exp(s[i * bk + j] - m_new);
                    s[i * bk + j] = p;
                    row_sum += p;
                }
                l[i] = alpha * l[i] + row_sum;
                for (int t = 0; t < d; t++) {
                    float acc = 0.0f;
                    for (int j = 0; j < bk; j++) {
                        acc += s[i * bk + j] * v[(kb + j) * d + t];
                    }
                    o[(qb + i) * d + t] =
                        alpha * o[(qb + i) * d + t] + acc;
                }
                m[i] = m_new;
            }
        }
        // divide out the running sum, rows are done
        for (int i = 0; i < bq; i++) {
            for (int t = 0; t < d; t++) {
                o[(qb + i) * d + t] /= l[i];
            }
        }
    }
}

void multi_head_attention(const float* q, const float* k, const float* v,
                          int heads, int n, int d_head, bool causal,
                          float scale, bool use_flash, float* o) {
    if (heads <= 0 || n <= 0 || d_head <= 0) return;
    for (int h = 0; h < heads; h++) {
        size_t off = (size_t)h * n * d_head;
        if (use_flash) {
            flash_attention(q + off, k + off, v + off, n, d_head, causal,
                            scale, 32, 32, o + off);
        } else {
            naive_attention(q + off, k + off, v + off, n, d_head, causal,
                            scale, o + off);
        }
    }
}

float max_abs_diff(const float* a, const float* b, int n, int d) {
    float worst = 0.0f;
    for (int i = 0; i < n * d; i++) {
        worst = std::max(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

float Rng::uniform(float lo, float hi) {
    // xorshift64, good enough for test data
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    float u = (float)(state >> 11) * (1.0f / 9007199254740992.0f);
    return lo + u * (hi - lo);
}

}  // namespace flash
