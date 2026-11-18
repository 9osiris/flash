#pragma once

#include <cstddef>
#include <vector>

namespace flash {

// all matrices are row major, n rows of d floats
// q, k, v are n*d floats each, o is filled with n*d floats
// scale <= 0 means use 1/sqrt(d)
// causal masks out j > i, the standard decoder setup

void naive_attention(const float* q, const float* k, const float* v,
                     int n, int d, bool causal, float scale, float* o);

// same result as naive, but never materializes the n*n score matrix.
// tiles over k/v in blocks and rescales with a running max, the
// online softmax trick. block sizes are just tuning knobs.
void flash_attention(const float* q, const float* k, const float* v,
                     int n, int d, bool causal, float scale,
                     int block_q, int block_k, float* o);

// per head attention over h heads. each head owns n rows of d_head
// floats, laid out head after head: head h starts at h*n*d_head.
void multi_head_attention(const float* q, const float* k, const float* v,
                          int heads, int n, int d_head, bool causal,
                          float scale, bool use_flash, float* o);

// max absolute difference between two n*d matrices
float max_abs_diff(const float* a, const float* b, int n, int d);

// tiny deterministic rng so tests and bench are reproducible
struct Rng {
    unsigned long long state;
    explicit Rng(unsigned long long seed) : state(seed ? seed : 1) {}
    float uniform(float lo, float hi);
};

}  // namespace flash
