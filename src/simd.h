#pragma once

// the two inner loops, vectorized when the compiler is told to.
// build with -mavx2 -mfma (make simd) for the avx2 path, otherwise
// these fall back to plain scalar loops and everything still works.
// no runtime cpuid dance, the build flags decide.

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace flash {
namespace simd {

// true when this binary was built with avx2 on
inline bool simd_available() {
#if defined(__AVX2__)
    return true;
#else
    return false;
#endif
}

// dot product of two d-float rows
inline float dot(const float* a, const float* b, int d) {
#if defined(__AVX2__)
    int i = 0;
    __m256 acc = _mm256_setzero_ps();
    for (; i + 8 <= d; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#if defined(__FMA__)
        acc = _mm256_fmadd_ps(va, vb, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
#endif
    }
    // fold the 8 lanes down to one float
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float out = _mm_cvtss_f32(s);
    for (; i < d; i++) out += a[i] * b[i];
    return out;
#else
    float out = 0.0f;
    for (int i = 0; i < d; i++) out += a[i] * b[i];
    return out;
#endif
}

// y += alpha * x, elementwise over d floats
inline void axpy(float alpha, const float* x, float* y, int d) {
#if defined(__AVX2__)
    int i = 0;
    __m256 av = _mm256_set1_ps(alpha);
    for (; i + 8 <= d; i += 8) {
        __m256 yv = _mm256_loadu_ps(y + i);
#if defined(__FMA__)
        yv = _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i), yv);
#else
        yv = _mm256_add_ps(yv, _mm256_mul_ps(av, _mm256_loadu_ps(x + i)));
#endif
        _mm256_storeu_ps(y + i, yv);
    }
    for (; i < d; i++) y[i] += alpha * x[i];
#else
    for (int i = 0; i < d; i++) y[i] += alpha * x[i];
#endif
}

}  // namespace simd
}  // namespace flash
