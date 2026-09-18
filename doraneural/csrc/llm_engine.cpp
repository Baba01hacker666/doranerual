#include "llm_engine.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif
#include <immintrin.h>

namespace {

inline float fast_silu(float x) {
    if (x < -6.0f) return 0.0f;
    if (x > 6.0f) return x;
    return x / (1.0f + expf(-x));
}
inline float silu(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return 0.0f;
    return x / (1.0f + expf(-x));
}
inline float silu_deriv(float x) {
    if (x > 88.0f) return 1.0f;
    if (x < -88.0f) return 0.0f;
    float s = 1.0f / (1.0f + expf(-x));
    return s + x * s * (1.0f - s);
}

#if defined(__AVX512F__)
inline float reduce_add_ps512(__m512 v) {
    __m256 low = _mm512_castps512_ps256(v);
    __m256 high = _mm512_extractf32x8_ps(v, 1);
    __m256 sum256 = _mm256_add_ps(low, high);
    __m128 low128 = _mm256_castps256_ps128(sum256);
    __m128 high128 = _mm256_extractf128_ps(sum256, 1);
    __m128 sum128 = _mm_add_ps(low128, high128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
}
inline int reduce_add_epi32_512(__m512i v) {
    __m256i low = _mm512_castsi512_si256(v);
    __m256i high = _mm512_extracti64x4_epi64(v, 1);
    __m256i sum256 = _mm256_add_epi32(low, high);
    __m128i low128 = _mm256_castsi256_si128(sum256);
    __m128i high128 = _mm256_extracti128_si256(sum256, 1);
    __m128i sum128 = _mm_add_epi32(low128, high128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    sum128 = _mm_hadd_epi32(sum128, sum128);
    return _mm_cvtsi128_si32(sum128);
}
#endif

inline uint16_t f32_to_f16_scalar(float f) {
    uint32_t x; std::memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;
    uint16_t h = 0;
    if (exp == 255) h = (sign << 15) | 0x7C00 | (mant ? 0x200 : 0);
    else if (exp > 142) h = (sign << 15) | 0x7C00;
    else if (exp < 113) h = (sign << 15);
    else {
        int new_exp = (int)exp - 127 + 15;
        uint32_t new_mant = mant >> 13;
        uint32_t round_bit = (mant >> 12) & 1;
        uint32_t rest = mant & 0xFFF;
        if (round_bit && (rest > 0 || (new_mant & 1))) new_mant++;
        if (new_mant == 0x400) { new_mant = 0; new_exp++; if (new_exp >= 31) { h = (sign << 15) | 0x7C00; return h; } }
        h = (sign << 15) | (new_exp << 10) | (new_mant & 0x3FF);
    }
    return h;
}
inline void convert_f32_to_f16(const float* src, uint16_t* dst, size_t count) {
    size_t i = 0;
#if defined(__F16C__) && defined(__AVX2__)
    for (; i + 7 < count; i += 8) {
        __m256 v = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128((__m128i*)(dst + i), h);
    }
#endif
    for (; i < count; i++) dst[i] = f32_to_f16_scalar(src[i]);
}
inline void convert_f32_to_i8(const float* src, int8_t* dst, float* scales, int32_t* sums, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* row_src = src + (size_t)r * cols;
        int8_t* row_dst = dst + (size_t)r * cols;
        float max_abs = 0.0f;
        for (int c = 0; c < cols; c++) { float v = std::abs(row_src[c]); if (v > max_abs) max_abs = v; }
        float scale = max_abs / 127.0f; if (scale < 1e-8f) scale = 1e-8f;
        scales[r] = scale; float inv = 1.0f / scale;
        int32_t sum = 0;
        for (int c = 0; c < cols; c++) { int q = (int)std::round(row_src[c] * inv); if (q > 127) q = 127; if (q < -127) q = -127; row_dst[c] = (int8_t)q; sum += q; }
        if (sums) sums[r] = sum;
    }
}

inline float dot_product_simd(const float* __restrict__ a, const float* __restrict__ b, int n) {
#if defined(__AVX512F__)
    int j = 0; __m512 vsum = _mm512_setzero_ps();
    for (; j + 15 < n; j += 16) { __m512 va = _mm512_loadu_ps(a + j); __m512 vb = _mm512_loadu_ps(b + j); vsum = _mm512_fmadd_ps(va, vb, vsum); }
    float acc = reduce_add_ps512(vsum);
    if (j + 7 < n) { __m256 va = _mm256_loadu_ps(a + j); __m256 vb = _mm256_loadu_ps(b + j); __m256 vs = _mm256_mul_ps(va, vb); __m128 low = _mm256_castps256_ps128(vs); __m128 high = _mm256_extractf128_ps(vs, 1); __m128 s = _mm_add_ps(low, high); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s); acc += _mm_cvtss_f32(s); j += 8; }
    for (; j < n; j++) acc += a[j] * b[j]; return acc;
#elif defined(__AVX2__)
    int j = 0; __m256 sum0 = _mm256_setzero_ps(); __m256 sum1 = _mm256_setzero_ps();
    for (; j + 15 < n; j += 16) { __m256 va0 = _mm256_loadu_ps(a + j); __m256 vb0 = _mm256_loadu_ps(b + j); sum0 = _mm256_fmadd_ps(va0, vb0, sum0); __m256 va1 = _mm256_loadu_ps(a + j + 8); __m256 vb1 = _mm256_loadu_ps(b + j + 8); sum1 = _mm256_fmadd_ps(va1, vb1, sum1); }
    __m256 vsum = _mm256_add_ps(sum0, sum1); __m128 vlow = _mm256_castps256_ps128(vsum); __m128 vhigh = _mm256_extractf128_ps(vsum, 1); __m128 vs = _mm_add_ps(vlow, vhigh); vs = _mm_hadd_ps(vs, vs); vs = _mm_hadd_ps(vs, vs); float acc = _mm_cvtss_f32(vs);
    for (; j < n; j++) acc += a[j] * b[j]; return acc;
#else
    float acc = 0.0f; for (int j = 0; j < n; j++) acc += a[j] * b[j]; return acc;
#endif
}

void rmsnorm_forward(float* __restrict__ out, const float* __restrict__ x, const float* __restrict__ weight, int size, float eps = 1e-5f) {
    float sum_sq = 0.0f;
#if defined(__AVX512F__)
    int i = 0; __m512 vsum = _mm512_setzero_ps();
    for (; i + 15 < size; i += 16) { __m512 vx = _mm512_loadu_ps(x + i); vsum = _mm512_fmadd_ps(vx, vx, vsum); }
    sum_sq = reduce_add_ps512(vsum);
    if (i + 7 < size) { __m256 vx = _mm256_loadu_ps(x + i); __m256 vs = _mm256_mul_ps(vx, vx); __m128 low = _mm256_castps256_ps128(vs); __m128 high = _mm256_extractf128_ps(vs, 1); __m128 s = _mm_add_ps(low, high); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s); sum_sq += _mm_cvtss_f32(s); i += 8; }
    for (; i < size; i++) sum_sq += x[i] * x[i];
#elif defined(__AVX2__)
    int i = 0; __m256 vsum0 = _mm256_setzero_ps(); __m256 vsum1 = _mm256_setzero_ps();
    for (; i + 15 < size; i += 16) { __m256 v0 = _mm256_loadu_ps(x + i); __m256 v1 = _mm256_loadu_ps(x + i + 8); vsum0 = _mm256_fmadd_ps(v0, v0, vsum0); vsum1 = _mm256_fmadd_ps(v1, v1, vsum1); }
    __m256 vsum = _mm256_add_ps(vsum0, vsum1); __m128 low = _mm256_castps256_ps128(vsum); __m128 high = _mm256_extractf128_ps(vsum, 1); __m128 s = _mm_add_ps(low, high); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s); sum_sq = _mm_cvtss_f32(s);
    for (; i < size; i++) sum_sq += x[i] * x[i];
#else
    for (int i = 0; i < size; i++) sum_sq += x[i] * x[i];
#endif
    float inv_rms = 1.0f / std::sqrt(sum_sq / size + eps);
#if defined(__AVX512F__)
    __m512 vinv = _mm512_set1_ps(inv_rms); i = 0;
    for (; i + 15 < size; i += 16) { __m512 vx = _mm512_loadu_ps(x + i); __m512 vw = _mm512_loadu_ps(weight + i); __m512 vo = _mm512_mul_ps(_mm512_mul_ps(vx, vinv), vw); _mm512_storeu_ps(out + i, vo); }
    if (i + 7 < size) { __m256 vinv256 = _mm256_set1_ps(inv_rms); __m256 vx = _mm256_loadu_ps(x + i); __m256 vw = _mm256_loadu_ps(weight + i); __m256 vo = _mm256_mul_ps(_mm256_mul_ps(vx, vinv256), vw); _mm256_storeu_ps(out + i, vo); i += 8; }
    for (; i < size; i++) out[i] = x[i] * inv_rms * weight[i];
#elif defined(__AVX2__)
    __m256 vinv = _mm256_set1_ps(inv_rms); int i2 = 0;
    for (; i2 + 7 < size; i2 += 8) { __m256 vx = _mm256_loadu_ps(x + i2); __m256 vw = _mm256_loadu_ps(weight + i2); __m256 vo = _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw); _mm256_storeu_ps(out + i2, vo); }
    for (; i2 < size; i2++) out[i2] = x[i2] * inv_rms * weight[i2];
#else
    for (int i = 0; i < size; i++) out[i] = x[i] * inv_rms * weight[i];
#endif
}

void rmsnorm_backward(float* dx, float* dweight, const float* dout, const float* x, const float* weight, int size, float eps = 1e-5f) {
    float sum_sq = 0.0f; for (int i = 0; i < size; i++) sum_sq += x[i] * x[i];
    float mean_sq = sum_sq / size + eps; float inv_rms = 1.0f / std::sqrt(mean_sq); float sum_dout_x = 0.0f;
    for (int i = 0; i < size; i++) { dweight[i] += dout[i] * x[i] * inv_rms; sum_dout_x += dout[i] * weight[i] * x[i]; }
    float factor = sum_dout_x / (size * mean_sq);
    for (int i = 0; i < size; i++) dx[i] = (dout[i] * weight[i] * inv_rms) - (x[i] * inv_rms * factor);
}

void matmul_forward_fp16(float* __restrict__ y, const float* __restrict__ x, const uint16_t* __restrict__ W, int n, int d) {
    if (d <= 0 || n <= 0) return;
    if ((size_t)d * n < 16384) {
        for (int i = 0; i < d; i++) {
            const uint16_t* row = W + (size_t)i * n; float acc = 0.0f;
#if defined(__F16C__) && defined(__AVX2__)
            int j = 0; __m256 vsum = _mm256_setzero_ps();
            for (; j + 7 < n; j += 8) { __m128i h = _mm_loadu_si128((const __m128i*)(row + j)); __m256 w = _mm256_cvtph_ps(h); __m256 xv = _mm256_loadu_ps(x + j); vsum = _mm256_fmadd_ps(w, xv, vsum); }
            __m128 low = _mm256_castps256_ps128(vsum); __m128 high = _mm256_extractf128_ps(vsum, 1); __m128 s = _mm_add_ps(low, high); s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s); acc = _mm_cvtss_f32(s);
            for (; j < n; j++) { uint16_t h = row[j]; uint32_t sign = (h >> 15) & 0x1; uint32_t exp = (h >> 10) & 0x1F; uint32_t mant = h & 0x3FF; float ff=0.0f; if(exp==0){ if(mant){ float m=mant/1024.0f; float e=std::pow(2.0f,-14.0f); ff=m*e; if(sign) ff=-ff; } } else if(exp==31){ ff=mant? std::numeric_limits<float>::quiet_NaN(): std::numeric_limits<float>::infinity(); if(sign) ff=-ff; } else { float e=std::pow(2.0f,(int)exp-15); float m=1.0f+mant/1024.0f; ff=e*m; if(sign) ff=-ff; } acc+=ff*x[j]; }
#else
            for (int j = 0; j < n; j++) { uint16_t h=row[j]; uint32_t sign=(h>>15)&0x1; uint32_t exp=(h>>10)&0x1F; uint32_t mant=h&0x3FF; float ff=0.0f; if(exp==0){ if(mant){ float m=mant/1024.0f; float e=std::pow(2.0f,-14.0f); ff=m*e; if(sign) ff=-ff; } } else if(exp==31){ ff=mant? std::numeric_limits<float>::quiet_NaN(): std::numeric_limits<float>::infinity(); if(sign) ff=-ff; } else { float e=std::pow(2.0f,(int)exp-15); float m=1.0f+mant/1024.0f; ff=e*m; if(sign) ff=-ff; } acc+=ff*x[j]; }
#endif
            y[i]=acc;
        }
        return;
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; i += 4) {
        int rows = std::min(4, d - i);
        const uint16_t* r0 = W + (size_t)(i + 0) * n;
        const uint16_t* r1 = rows > 1 ? W + (size_t)(i + 1) * n : nullptr;
        const uint16_t* r2 = rows > 2 ? W + (size_t)(i + 2) * n : nullptr;
        const uint16_t* r3 = rows > 3 ? W + (size_t)(i + 3) * n : nullptr;
#if defined(__F16C__) && defined(__AVX2__)
        __m256 vsum0 = _mm256_setzero_ps(); __m256 vsum1 = _mm256_setzero_ps(); __m256 vsum2 = _mm256_setzero_ps(); __m256 vsum3 = _mm256_setzero_ps();
        int j = 0;
        for (; j + 7 < n; j += 8) {
            __m256 xv = _mm256_loadu_ps(x + j);
            if (rows > 0) { __m128i h = _mm_loadu_si128((const __m128i*)(r0 + j)); __m256 w = _mm256_cvtph_ps(h); vsum0 = _mm256_fmadd_ps(w, xv, vsum0); }
            if (rows > 1) { __m128i h = _mm_loadu_si128((const __m128i*)(r1 + j)); __m256 w = _mm256_cvtph_ps(h); vsum1 = _mm256_fmadd_ps(w, xv, vsum1); }
            if (rows > 2) { __m128i h = _mm_loadu_si128((const __m128i*)(r2 + j)); __m256 w = _mm256_cvtph_ps(h); vsum2 = _mm256_fmadd_ps(w, xv, vsum2); }
            if (rows > 3) { __m128i h = _mm_loadu_si128((const __m128i*)(r3 + j)); __m256 w = _mm256_cvtph_ps(h); vsum3 = _mm256_fmadd_ps(w, xv, vsum3); }
        }
        auto hsum = [](__m256 v){ __m128 low=_mm256_castps256_ps128(v); __m128 high=_mm256_extractf128_ps(v,1); __m128 s=_mm_add_ps(low,high); s=_mm_hadd_ps(s,s); s=_mm_hadd_ps(s,s); return _mm_cvtss_f32(s); };
        float acc0=hsum(vsum0); float acc1=rows>1?hsum(vsum1):0.0f; float acc2=rows>2?hsum(vsum2):0.0f; float acc3=rows>3?hsum(vsum3):0.0f;
        for (; j < n; j++) { auto f16_to_f32=[](uint16_t h)->float{ uint32_t sign=(h>>15)&0x1; uint32_t exp=(h>>10)&0x1F; uint32_t mant=h&0x3FF; float ff=0.0f; if(exp==0){ if(mant){ float m=mant/1024.0f; float e=std::pow(2.0f,-14.0f); ff=m*e; if(sign) ff=-ff; } } else if(exp==31){ ff=mant? std::numeric_limits<float>::quiet_NaN(): std::numeric_limits<float>::infinity(); if(sign) ff=-ff; } else { float e=std::pow(2.0f,(int)exp-15); float m=1.0f+mant/1024.0f; ff=e*m; if(sign) ff=-ff; } return ff; }; float xj=x[j]; acc0+=f16_to_f32(r0[j])*xj; if(rows>1) acc1+=f16_to_f32(r1[j])*xj; if(rows>2) acc2+=f16_to_f32(r2[j])*xj; if(rows>3) acc3+=f16_to_f32(r3[j])*xj; }
        y[i]=acc0; if(rows>1) y[i+1]=acc1; if(rows>2) y[i+2]=acc2; if(rows>3) y[i+3]=acc3;
#else
        float acc0=0,acc1=0,acc2=0,acc3=0;
        for(int j=0;j<n;j++){ auto f16_to_f32=[](uint16_t h)->float{ uint32_t sign=(h>>15)&0x1; uint32_t exp=(h>>10)&0x1F; uint32_t mant=h&0x3FF; float ff=0.0f; if(exp==0){ if(mant){ float m=mant/1024.0f; float e=std::pow(2.0f,-14.0f); ff=m*e; if(sign) ff=-ff; } } else if(exp==31){ ff=mant? std::numeric_limits<float>::quiet_NaN(): std::numeric_limits<float>::infinity(); if(sign) ff=-ff; } else { float e=std::pow(2.0f,(int)exp-15); float m=1.0f+mant/1024.0f; ff=e*m; if(sign) ff=-ff; } return ff; }; float xj=x[j]; acc0+=f16_to_f32(r0[j])*xj; if(rows>1) acc1+=f16_to_f32(r1[j])*xj; if(rows>2) acc2+=f16_to_f32(r2[j])*xj; if(rows>3) acc3+=f16_to_f32(r3[j])*xj; }
        y[i]=acc0; if(rows>1) y[i+1]=acc1; if(rows>2) y[i+2]=acc2; if(rows>3) y[i+3]=acc3;
#endif
    }
}

inline void matmul_int8_core_4rows(float* y, const float* x, const int8_t* r0, const int8_t* r1, const int8_t* r2, const int8_t* r3, float s0, float s1, float s2, float s3, int rows, int n) {
#if defined(__AVX512BW__) && defined(__AVX512F__)
    __m512 vsum0 = _mm512_setzero_ps(); __m512 vsum1 = _mm512_setzero_ps(); __m512 vsum2 = _mm512_setzero_ps(); __m512 vsum3 = _mm512_setzero_ps();
    int j = 0;
    for (; j + 15 < n; j += 16) {
        __m512 xv = _mm512_loadu_ps(x + j);
        if (rows > 0) { __m128i b = _mm_loadu_si128((const __m128i*)(r0 + j)); __m512i i32 = _mm512_cvtepi8_epi32(b); __m512 w = _mm512_cvtepi32_ps(i32); vsum0 = _mm512_fmadd_ps(w, xv, vsum0); }
        if (rows > 1) { __m128i b = _mm_loadu_si128((const __m128i*)(r1 + j)); __m512i i32 = _mm512_cvtepi8_epi32(b); __m512 w = _mm512_cvtepi32_ps(i32); vsum1 = _mm512_fmadd_ps(w, xv, vsum1); }
        if (rows > 2) { __m128i b = _mm_loadu_si128((const __m128i*)(r2 + j)); __m512i i32 = _mm512_cvtepi8_epi32(b); __m512 w = _mm512_cvtepi32_ps(i32); vsum2 = _mm512_fmadd_ps(w, xv, vsum2); }
        if (rows > 3) { __m128i b = _mm_loadu_si128((const __m128i*)(r3 + j)); __m512i i32 = _mm512_cvtepi8_epi32(b); __m512 w = _mm512_cvtepi32_ps(i32); vsum3 = _mm512_fmadd_ps(w, xv, vsum3); }
    }
    float acc0 = reduce_add_ps512(vsum0) * s0; float acc1 = rows > 1 ? reduce_add_ps512(vsum1) * s1 : 0.0f; float acc2 = rows > 2 ? reduce_add_ps512(vsum2) * s2 : 0.0f; float acc3 = rows > 3 ? reduce_add_ps512(vsum3) * s3 : 0.0f;
    for (; j < n; j++) { float xj=x[j]; acc0+= (float)r0[j]*xj*s0; if(rows>1) acc1+= (float)r1[j]*xj*s1; if(rows>2) acc2+= (float)r2[j]*xj*s2; if(rows>3) acc3+= (float)r3[j]*xj*s3; }
    y[0]=acc0; if(rows>1) y[1]=acc1; if(rows>2) y[2]=acc2; if(rows>3) y[3]=acc3;
#elif defined(__AVX2__)
    __m256 vsum0 = _mm256_setzero_ps(); __m256 vsum1 = _mm256_setzero_ps(); __m256 vsum2 = _mm256_setzero_ps(); __m256 vsum3 = _mm256_setzero_ps();
    int j = 0;
    for (; j + 7 < n; j += 8) {
        __m256 xv = _mm256_loadu_ps(x + j);
        if (rows > 0) { __m128i b8 = _mm_loadl_epi64((const __m128i*)(r0 + j)); __m128i b16 = _mm_cvtepi8_epi16(b8); __m256i b32 = _mm256_cvtepi16_epi32(b16); __m256 w = _mm256_cvtepi32_ps(b32); vsum0 = _mm256_fmadd_ps(w, xv, vsum0); }
        if (rows > 1) { __m128i b8 = _mm_loadl_epi64((const __m128i*)(r1 + j)); __m128i b16 = _mm_cvtepi8_epi16(b8); __m256i b32 = _mm256_cvtepi16_epi32(b16); __m256 w = _mm256_cvtepi32_ps(b32); vsum1 = _mm256_fmadd_ps(w, xv, vsum1); }
        if (rows > 2) { __m128i b8 = _mm_loadl_epi64((const __m128i*)(r2 + j)); __m128i b16 = _mm_cvtepi8_epi16(b8); __m256i b32 = _mm256_cvtepi16_epi32(b16); __m256 w = _mm256_cvtepi32_ps(b32); vsum2 = _mm256_fmadd_ps(w, xv, vsum2); }
        if (rows > 3) { __m128i b8 = _mm_loadl_epi64((const __m128i*)(r3 + j)); __m128i b16 = _mm_cvtepi8_epi16(b8); __m256i b32 = _mm256_cvtepi16_epi32(b16); __m256 w = _mm256_cvtepi32_ps(b32); vsum3 = _mm256_fmadd_ps(w, xv, vsum3); }
    }
    auto hsum = [](__m256 v){ __m128 low=_mm256_castps256_ps128(v); __m128 high=_mm256_extractf128_ps(v,1); __m128 s=_mm_add_ps(low,high); s=_mm_hadd_ps(s,s); s=_mm_hadd_ps(s,s); return _mm_cvtss_f32(s); };
    float acc0 = hsum(vsum0) * s0; float acc1 = rows>1? hsum(vsum1)*s1:0.0f; float acc2 = rows>2? hsum(vsum2)*s2:0.0f; float acc3 = rows>3? hsum(vsum3)*s3:0.0f;
    for (; j < n; j++) { float xj=x[j]; acc0+= (float)r0[j]*xj*s0; if(rows>1) acc1+= (float)r1[j]*xj*s1; if(rows>2) acc2+= (float)r2[j]*xj*s2; if(rows>3) acc3+= (float)r3[j]*xj*s3; }
    y[0]=acc0; if(rows>1) y[1]=acc1; if(rows>2) y[2]=acc2; if(rows>3) y[3]=acc3;
#else
    float acc0=0,acc1=0,acc2=0,acc3=0;
    for(int j=0;j<n;j++){ float xj=x[j]; acc0+= (float)r0[j]*xj; if(rows>1) acc1+= (float)r1[j]*xj; if(rows>2) acc2+= (float)r2[j]*xj; if(rows>3) acc3+= (float)r3[j]*xj; }
    y[0]=acc0*s0; if(rows>1) y[1]=acc1*s1; if(rows>2) y[2]=acc2*s2; if(rows>3) y[3]=acc3*s3;
#endif
}

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
inline float quantize_i8_row(const float* x, int n, int8_t* q, uint8_t* qu) {
    float max_abs = 0.0f;
    for (int i=0;i<n;i++){ float v=std::abs(x[i]); if(v>max_abs) max_abs=v; }
    float scale = max_abs / 127.0f;
    if (scale < 1e-8f) scale = 1e-8f;
    float inv = 1.0f/scale;
    for (int i=0;i<n;i++){
        int qi = (int)std::round(x[i]*inv);
        if (qi>127) qi=127;
        if (qi<-127) qi=-127;
        q[i]=(int8_t)qi;
        qu[i]=(uint8_t)(qi+128);
    }
    return scale;
}
inline int32_t dot_u8_i8_vnni_64(const uint8_t* a, const int8_t* b, int n) {
    __m512i acc = _mm512_setzero_si512();
    int j=0;
    for (; j+63 < n; j+=64){
        __m512i av = _mm512_loadu_si512((const __m512i*)(a+j));
        __m512i bv = _mm512_loadu_si512((const __m512i*)(b+j));
        acc = _mm512_dpbusd_epi32(acc, av, bv);
    }
    int sum = reduce_add_epi32_512(acc);
    for (; j < n; j++) sum += (int32_t)a[j] * (int32_t)b[j];
    return sum;
}
#endif

void matmul_forward_int8(float* __restrict__ y, const float* __restrict__ x, const int8_t* __restrict__ W, const float* __restrict__ scales, int n, int d) {
    if (d <= 0 || n <= 0) return;
    if ((size_t)d * n < 16384) {
        for (int i = 0; i < d; i++) { const int8_t* row = W + (size_t)i * n; float acc = 0.0f; for (int j = 0; j < n; j++) acc += (float)row[j] * x[j]; y[i] = acc * scales[i]; }
        return;
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; i += 4) {
        int rows = std::min(4, d - i);
        const int8_t* r0 = W + (size_t)(i + 0) * n;
        const int8_t* r1 = rows > 1 ? W + (size_t)(i + 1) * n : nullptr;
        const int8_t* r2 = rows > 2 ? W + (size_t)(i + 2) * n : nullptr;
        const int8_t* r3 = rows > 3 ? W + (size_t)(i + 3) * n : nullptr;
        float s0 = scales[i + 0]; float s1 = rows > 1 ? scales[i + 1] : 0.0f; float s2 = rows > 2 ? scales[i + 2] : 0.0f; float s3 = rows > 3 ? scales[i + 3] : 0.0f;
        float tmp[4]={0,0,0,0};
        matmul_int8_core_4rows(tmp, x, r0, r1, r2, r3, s0, s1, s2, s3, rows, n);
        y[i+0]=tmp[0]; if(rows>1) y[i+1]=tmp[1]; if(rows>2) y[i+2]=tmp[2]; if(rows>3) y[i+3]=tmp[3];
    }
}

void matmul_forward_int8_vnni(float* __restrict__ y, const float* __restrict__ x, const int8_t* __restrict__ W, const float* __restrict__ scales, const int32_t* __restrict__ sums, int n, int d) {
    if (d <= 0 || n <= 0) return;
    if ((size_t)d * n < 16384) {
        for (int i = 0; i < d; i++) { const int8_t* row = W + (size_t)i * n; float acc = 0.0f; for (int j = 0; j < n; j++) acc += (float)row[j] * x[j]; y[i] = acc * scales[i]; }
        return;
    }
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
    const int MAXN = 8192;
    alignas(64) int8_t qx_buf[MAXN];
    alignas(64) uint8_t qx_u_buf[MAXN];
    int8_t* qx = qx_buf;
    uint8_t* qx_u = qx_u_buf;
    std::vector<int8_t> qx_heap;
    std::vector<uint8_t> qx_u_heap;
    if (n > MAXN) {
        qx_heap.resize(n);
        qx_u_heap.resize(n);
        qx = qx_heap.data();
        qx_u = qx_u_heap.data();
    }
    float scale_x = quantize_i8_row(x, n, qx, qx_u);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; i += 4) {
        int rows = std::min(4, d - i);
        const int8_t* r0 = W + (size_t)(i + 0) * n;
        const int8_t* r1 = rows > 1 ? W + (size_t)(i + 1) * n : nullptr;
        const int8_t* r2 = rows > 2 ? W + (size_t)(i + 2) * n : nullptr;
        const int8_t* r3 = rows > 3 ? W + (size_t)(i + 3) * n : nullptr;
        float s0 = scales[i + 0]; float s1 = rows > 1 ? scales[i + 1] : 0.0f; float s2 = rows > 2 ? scales[i + 2] : 0.0f; float s3 = rows > 3 ? scales[i + 3] : 0.0f;
        int32_t sum0 = sums ? sums[i+0] : 0; int32_t sum1 = rows>1 && sums ? sums[i+1] : 0; int32_t sum2 = rows>2 && sums ? sums[i+2] : 0; int32_t sum3 = rows>3 && sums ? sums[i+3] : 0;

        int32_t dot_u0 = dot_u8_i8_vnni_64(qx_u, r0, n);
        int32_t dot0 = dot_u0 - 128*sum0;
        float out0 = (float)dot0 * scale_x * s0;
        float out1=0,out2=0,out3=0;
        if (rows>1){ int32_t dot_u1 = dot_u8_i8_vnni_64(qx_u, r1, n); int32_t dot1 = dot_u1 - 128*sum1; out1 = (float)dot1 * scale_x * s1; }
        if (rows>2){ int32_t dot_u2 = dot_u8_i8_vnni_64(qx_u, r2, n); int32_t dot2 = dot_u2 - 128*sum2; out2 = (float)dot2 * scale_x * s2; }
        if (rows>3){ int32_t dot_u3 = dot_u8_i8_vnni_64(qx_u, r3, n); int32_t dot3 = dot_u3 - 128*sum3; out3 = (float)dot3 * scale_x * s3; }
        y[i+0]=out0; if(rows>1) y[i+1]=out1; if(rows>2) y[i+2]=out2; if(rows>3) y[i+3]=out3;
    }
#else
    matmul_forward_int8(y, x, W, scales, n, d);
#endif
}

void matmul_forward(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ W, int n, int d) {
    if (d <= 0 || n <= 0) return;
    if ((size_t)d * n < 16384) { for (int i = 0; i < d; i++) y[i] = dot_product_simd(W + (size_t)i * n, x, n); return; }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; i += 4) {
        int rows = std::min(4, d - i);
        const float* r0 = W + (size_t)(i + 0) * n;
        const float* r1 = rows > 1 ? W + (size_t)(i + 1) * n : nullptr;
        const float* r2 = rows > 2 ? W + (size_t)(i + 2) * n : nullptr;
        const float* r3 = rows > 3 ? W + (size_t)(i + 3) * n : nullptr;
#if defined(__AVX512F__)
        __m512 vsum0 = _mm512_setzero_ps(); __m512 vsum1 = _mm512_setzero_ps(); __m512 vsum2 = _mm512_setzero_ps(); __m512 vsum3 = _mm512_setzero_ps();
        int j = 0;
        for (; j + 15 < n; j += 16) {
            __m512 vx = _mm512_loadu_ps(x + j);
            if (rows > 0) { __m512 vw = _mm512_loadu_ps(r0 + j); vsum0 = _mm512_fmadd_ps(vw, vx, vsum0); }
            if (rows > 1) { __m512 vw = _mm512_loadu_ps(r1 + j); vsum1 = _mm512_fmadd_ps(vw, vx, vsum1); }
            if (rows > 2) { __m512 vw = _mm512_loadu_ps(r2 + j); vsum2 = _mm512_fmadd_ps(vw, vx, vsum2); }
            if (rows > 3) { __m512 vw = _mm512_loadu_ps(r3 + j); vsum3 = _mm512_fmadd_ps(vw, vx, vsum3); }
        }
        float acc0 = reduce_add_ps512(vsum0); float acc1 = rows > 1 ? reduce_add_ps512(vsum1) : 0.0f; float acc2 = rows > 2 ? reduce_add_ps512(vsum2) : 0.0f; float acc3 = rows > 3 ? reduce_add_ps512(vsum3) : 0.0f;
        for (; j < n; j++) { float xj = x[j]; acc0 += r0[j] * xj; if (rows > 1) acc1 += r1[j] * xj; if (rows > 2) acc2 += r2[j] * xj; if (rows > 3) acc3 += r3[j] * xj; }
        y[i] = acc0; if (rows > 1) y[i+1] = acc1; if (rows > 2) y[i+2] = acc2; if (rows > 3) y[i+3] = acc3;
#elif defined(__AVX2__)
        __m256 vsum0_a = _mm256_setzero_ps(), vsum0_b = _mm256_setzero_ps();
        __m256 vsum1_a = _mm256_setzero_ps(), vsum1_b = _mm256_setzero_ps();
        __m256 vsum2_a = _mm256_setzero_ps(), vsum2_b = _mm256_setzero_ps();
        __m256 vsum3_a = _mm256_setzero_ps(), vsum3_b = _mm256_setzero_ps();
        int j = 0;
        for (; j + 15 < n; j += 16) {
            __m256 vx_a = _mm256_loadu_ps(x + j); __m256 vx_b = _mm256_loadu_ps(x + j + 8);
            if (rows > 0) { __m256 vw_a = _mm256_loadu_ps(r0 + j); __m256 vw_b = _mm256_loadu_ps(r0 + j + 8); vsum0_a = _mm256_fmadd_ps(vw_a, vx_a, vsum0_a); vsum0_b = _mm256_fmadd_ps(vw_b, vx_b, vsum0_b); }
            if (rows > 1) { __m256 vw_a = _mm256_loadu_ps(r1 + j); __m256 vw_b = _mm256_loadu_ps(r1 + j + 8); vsum1_a = _mm256_fmadd_ps(vw_a, vx_a, vsum1_a); vsum1_b = _mm256_fmadd_ps(vw_b, vx_b, vsum1_b); }
            if (rows > 2) { __m256 vw_a = _mm256_loadu_ps(r2 + j); __m256 vw_b = _mm256_loadu_ps(r2 + j + 8); vsum2_a = _mm256_fmadd_ps(vw_a, vx_a, vsum2_a); vsum2_b = _mm256_fmadd_ps(vw_b, vx_b, vsum2_b); }
            if (rows > 3) { __m256 vw_a = _mm256_loadu_ps(r3 + j); __m256 vw_b = _mm256_loadu_ps(r3 + j + 8); vsum3_a = _mm256_fmadd_ps(vw_a, vx_a, vsum3_a); vsum3_b = _mm256_fmadd_ps(vw_b, vx_b, vsum3_b); }
        }
        auto hsum256 = [](__m256 a, __m256 b){ __m256 s=_mm256_add_ps(a,b); __m128 low=_mm256_castps256_ps128(s); __m128 high=_mm256_extractf128_ps(s,1); __m128 ss=_mm_add_ps(low,high); ss=_mm_hadd_ps(ss,ss); ss=_mm_hadd_ps(ss,ss); return _mm_cvtss_f32(ss); };
        float acc0 = hsum256(vsum0_a, vsum0_b); float acc1 = rows > 1 ? hsum256(vsum1_a, vsum1_b) : 0.0f; float acc2 = rows > 2 ? hsum256(vsum2_a, vsum2_b) : 0.0f; float acc3 = rows > 3 ? hsum256(vsum3_a, vsum3_b) : 0.0f;
        for (; j < n; j++) { float xj = x[j]; acc0 += r0[j] * xj; if (rows > 1) acc1 += r1[j] * xj; if (rows > 2) acc2 += r2[j] * xj; if (rows > 3) acc3 += r3[j] * xj; }
        y[i] = acc0; if (rows > 1) y[i+1] = acc1; if (rows > 2) y[i+2] = acc2; if (rows > 3) y[i+3] = acc3;
#else
        float acc0=0,acc1=0,acc2=0,acc3=0;
        for(int j=0;j<n;j++){ float xj=x[j]; acc0+=r0[j]*xj; if(rows>1) acc1+=r1[j]*xj; if(rows>2) acc2+=r2[j]*xj; if(rows>3) acc3+=r3[j]*xj; }
        y[i]=acc0; if(rows>1) y[i+1]=acc1; if(rows>2) y[i+2]=acc2; if(rows>3) y[i+3]=acc3;
#endif
    }
}

void matmul_backward(float* dx, float* dW, const float* dy, const float* x, const float* W, int n, int d) {
    if (dx) {
#ifdef _OPENMP
        if ((size_t)d * n >= 32768 && n >= 16) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < n; j++) { float acc=0.0f; for(int i=0;i<d;i++) acc+=W[(size_t)i*n+j]*dy[i]; dx[j]=acc; }
        } else
#endif
        { for(int j=0;j<n;j++){ float acc=0.0f; for(int i=0;i<d;i++) acc+=W[(size_t)i*n+j]*dy[i]; dx[j]=acc; } }
    }
    if (dW) {
#ifdef _OPENMP
        if ((size_t)d * n >= 32768 && d >= 16) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < d; i++) {
                float dy_i=dy[i]; float* row=dW+(size_t)i*n;
#if defined(__AVX512F__)
                __m512 vdy=_mm512_set1_ps(dy_i); int j=0; for(; j+15<n; j+=16){ __m512 vr=_mm512_loadu_ps(row+j); __m512 vx=_mm512_loadu_ps(x+j); vr=_mm512_fmadd_ps(vdy,vx,vr); _mm512_storeu_ps(row+j,vr); } for(;j<n;j++) row[j]+=dy_i*x[j];
#elif defined(__AVX2__)
                __m256 vdy=_mm256_set1_ps(dy_i); int j=0; for(; j+7<n; j+=8){ __m256 vr=_mm256_loadu_ps(row+j); __m256 vx=_mm256_loadu_ps(x+j); vr=_mm256_fmadd_ps(vdy,vx,vr); _mm256_storeu_ps(row+j,vr); } for(;j<n;j++) row[j]+=dy_i*x[j];
#else
                for(int j=0;j<n;j++) row[j]+=dy_i*x[j];
#endif
            }
        } else
#endif
        { for(int i=0;i<d;i++){ float dy_i=dy[i]; float* row=dW+(size_t)i*n; for(int j=0;j<n;j++) row[j]+=dy_i*x[j]; } }
    }
}

void softmax(float* x, int size) {
    if(size<=0) return; float max_val=x[0]; for(int i=1;i<size;i++) if(x[i]>max_val) max_val=x[i];
    float sum=0.0f; for(int j=0;j<size;j++){ x[j]=expf(x[j]-max_val); sum+=x[j]; }
    float inv_sum=1.0f/(sum>0.0f?sum:1e-5f);
#if defined(__AVX512F__)
    __m512 vinv=_mm512_set1_ps(inv_sum); int j=0; for(; j+15<size; j+=16){ __m512 vx=_mm512_loadu_ps(x+j); vx=_mm512_mul_ps(vx,vinv); _mm512_storeu_ps(x+j,vx); } if(j+7<size){ __m256 vinv256=_mm256_set1_ps(inv_sum); __m256 vx=_mm256_loadu_ps(x+j); vx=_mm256_mul_ps(vx,vinv256); _mm256_storeu_ps(x+j,vx); j+=8; } for(;j<size;j++) x[j]*=inv_sum;
#elif defined(__AVX2__)
    __m256 vinv=_mm256_set1_ps(inv_sum); int j=0; for(; j+7<size; j+=8){ __m256 vx=_mm256_loadu_ps(x+j); vx=_mm256_mul_ps(vx,vinv); _mm256_storeu_ps(x+j,vx); } for(;j<size;j++) x[j]*=inv_sum;
#else
    for(int j=0;j<size;j++) x[j]*=inv_sum;
#endif
}

} // namespace

struct FullTrainWorkspace {
    int seq=0,layers=0,dim=0,kv_dim=0,heads=0,hidden=0;
    std::vector<float> states,norm_att,q,k,v,scores,attn_out,attn_state,norm_ffn,gate,up,swiglu,final_norm,d_final;
    std::vector<float> d_states,d_norm_att,d_q,d_k,d_v,d_norm_ffn,d_gate,d_up,d_swiglu,d_attn_out,d_attn_state;
    void resize(int ns,int nl,int nd,int nkv,int nh,int nhid){
        if(seq==ns&&layers==nl&&dim==nd&&kv_dim==nkv&&heads==nh&&hidden==nhid) return;
        seq=ns; layers=nl; dim=nd; kv_dim=nkv; heads=nh; hidden=nhid;
        size_t ld=(size_t)layers*seq*dim, lkv=(size_t)layers*seq*kv_dim, lh=(size_t)layers*seq*hidden, ls=(size_t)layers*seq*heads*seq;
        states.assign((size_t)(layers+1)*seq*dim,0.0f); norm_att.assign(ld,0.0f); q.assign(ld,0.0f); k.assign(lkv,0.0f); v.assign(lkv,0.0f);
        scores.assign(ls,0.0f); attn_out.assign(ld,0.0f); attn_state.assign(ld,0.0f); norm_ffn.assign(ld,0.0f);
        gate.assign(lh,0.0f); up.assign(lh,0.0f); swiglu.assign(lh,0.0f); final_norm.assign((size_t)seq*dim,0.0f); d_final.assign((size_t)seq*dim,0.0f);
        d_states.assign(states.size(),0.0f); d_norm_att.assign(ld,0.0f); d_q.assign(ld,0.0f); d_k.assign(lkv,0.0f); d_v.assign(lkv,0.0f);
        d_norm_ffn.assign(ld,0.0f); d_gate.assign(lh,0.0f); d_up.assign(lh,0.0f); d_swiglu.assign(lh,0.0f); d_attn_out.assign(ld,0.0f); d_attn_state.assign(ld,0.0f);
    }
};

struct LlamaCppEngine {
    LlamaCppConfig config; LlamaCppWeights weights;
    std::vector<float> key_cache,val_cache,cos_cache,sin_cache;
    std::vector<float> x,xb,q,k,v,att,attn_out,hb1,hb3,hb,logits;
    std::vector<float> sample_probs; std::vector<std::pair<float,int>> sample_candidates;
    std::vector<float> train_step_logits,train_dlogits,train_probs,train_grad_embedding;
    FullTrainWorkspace full_workspace;
    std::vector<float> grad_buffer,m_buffer,v_buffer,m_cls,v_cls;
    std::vector<uint16_t> wq_fp16,wk_fp16,wv_fp16,wo_fp16,w1_fp16,w2_fp16,w3_fp16,wcls_fp16;
    std::vector<int8_t> wq_i8,wk_i8,wv_i8,wo_i8,w1_i8,w2_i8,w3_i8,wcls_i8;
    std::vector<float> wq_s,wk_s,wv_s,wo_s,w1_s,w2_s,w3_s,wcls_s;
    std::vector<int32_t> wq_sum,wk_sum,wv_sum,wo_sum,w1_sum,w2_sum,w3_sum,wcls_sum;
    bool use_fp16; bool use_i8; bool use_vnni;
    int adam_step; std::mt19937 rng;
    LlamaCppEngine(const LlamaCppConfig* cfg, LlamaCppWeights* w) : config(*cfg), weights(*w), use_fp16(false), use_i8(false), use_vnni(false), adam_step(0), rng(42) {
        int head_size=config.dim/config.n_heads; int half=head_size/2; int kv_dim=(config.dim*config.n_kv_heads)/config.n_heads;
        key_cache.resize((size_t)config.n_layers*config.seq_len*kv_dim,0.0f); val_cache.resize((size_t)config.n_layers*config.seq_len*kv_dim,0.0f);
        x.resize(config.dim,0.0f); xb.resize(config.dim,0.0f); q.resize(config.dim,0.0f); k.resize(kv_dim,0.0f); v.resize(kv_dim,0.0f);
        att.resize((size_t)config.n_heads*config.seq_len,0.0f); attn_out.resize(config.dim,0.0f);
        hb1.resize(config.hidden_dim,0.0f); hb3.resize(config.hidden_dim,0.0f); hb.resize(config.hidden_dim,0.0f);
        logits.resize(config.vocab_size,0.0f); sample_probs.resize(config.vocab_size,0.0f); sample_candidates.resize(config.vocab_size);
        train_step_logits.resize(config.vocab_size,0.0f); train_dlogits.resize(config.vocab_size,0.0f); train_probs.resize(config.vocab_size,0.0f); train_grad_embedding.resize(config.dim,0.0f);
        cos_cache.resize((size_t)config.seq_len*half,0.0f); sin_cache.resize((size_t)config.seq_len*half,0.0f);
        for(int p_idx=0;p_idx<config.seq_len;p_idx++) for(int i=0;i<half;i++){ float freq=1.0f/std::pow(10000.0f,(2.0f*i)/(float)head_size); float val=p_idx*freq; cos_cache[(size_t)p_idx*half+i]=std::cos(val); sin_cache[(size_t)p_idx*half+i]=std::sin(val); }
        size_t total_weights=calculate_total_parameters(); grad_buffer.resize(total_weights,0.0f); m_buffer.resize(total_weights,0.0f); v_buffer.resize(total_weights,0.0f);
        try{
            size_t wq_elems=(size_t)config.n_layers*config.dim*config.dim;
            size_t wkv_elems=(size_t)config.n_layers*kv_dim*config.dim;
            size_t wo_elems=wq_elems;
            size_t w1_elems=(size_t)config.n_layers*config.hidden_dim*config.dim;
            size_t w2_elems=(size_t)config.n_layers*config.dim*config.hidden_dim;
            size_t w3_elems=w1_elems;
            size_t wcls_elems=(size_t)config.vocab_size*config.dim;
            wq_fp16.resize(wq_elems); wk_fp16.resize(wkv_elems); wv_fp16.resize(wkv_elems); wo_fp16.resize(wo_elems); w1_fp16.resize(w1_elems); w2_fp16.resize(w2_elems); w3_fp16.resize(w3_elems); wcls_fp16.resize(wcls_elems);
            convert_f32_to_f16(weights.wq,wq_fp16.data(),wq_elems); convert_f32_to_f16(weights.wk,wk_fp16.data(),wkv_elems); convert_f32_to_f16(weights.wv,wv_fp16.data(),wkv_elems);
            convert_f32_to_f16(weights.wo,wo_fp16.data(),wo_elems); convert_f32_to_f16(weights.w1,w1_fp16.data(),w1_elems); convert_f32_to_f16(weights.w2,w2_fp16.data(),w2_elems); convert_f32_to_f16(weights.w3,w3_fp16.data(),w3_elems);
            if(weights.wcls) convert_f32_to_f16(weights.wcls,wcls_fp16.data(),wcls_elems); else convert_f32_to_f16(weights.token_embedding_table,wcls_fp16.data(),wcls_elems);
            use_fp16=true;
            wq_i8.resize(wq_elems); wk_i8.resize(wkv_elems); wv_i8.resize(wkv_elems); wo_i8.resize(wo_elems); w1_i8.resize(w1_elems); w2_i8.resize(w2_elems); w3_i8.resize(w3_elems); wcls_i8.resize(wcls_elems);
            wq_s.resize((size_t)config.n_layers*config.dim); wk_s.resize((size_t)config.n_layers*kv_dim); wv_s.resize((size_t)config.n_layers*kv_dim); wo_s.resize((size_t)config.n_layers*config.dim);
            w1_s.resize((size_t)config.n_layers*config.hidden_dim); w2_s.resize((size_t)config.n_layers*config.dim); w3_s.resize((size_t)config.n_layers*config.hidden_dim); wcls_s.resize(config.vocab_size);
            wq_sum.resize((size_t)config.n_layers*config.dim); wk_sum.resize((size_t)config.n_layers*kv_dim); wv_sum.resize((size_t)config.n_layers*kv_dim); wo_sum.resize((size_t)config.n_layers*config.dim);
            w1_sum.resize((size_t)config.n_layers*config.hidden_dim); w2_sum.resize((size_t)config.n_layers*config.dim); w3_sum.resize((size_t)config.n_layers*config.hidden_dim); wcls_sum.resize(config.vocab_size);
            for(int l=0;l<config.n_layers;l++){
                convert_f32_to_i8(weights.wq + (size_t)l*config.dim*config.dim, wq_i8.data() + (size_t)l*config.dim*config.dim, wq_s.data() + (size_t)l*config.dim, wq_sum.data() + (size_t)l*config.dim, config.dim, config.dim);
                convert_f32_to_i8(weights.wk + (size_t)l*kv_dim*config.dim, wk_i8.data() + (size_t)l*kv_dim*config.dim, wk_s.data() + (size_t)l*kv_dim, wk_sum.data() + (size_t)l*kv_dim, kv_dim, config.dim);
                convert_f32_to_i8(weights.wv + (size_t)l*kv_dim*config.dim, wv_i8.data() + (size_t)l*kv_dim*config.dim, wv_s.data() + (size_t)l*kv_dim, wv_sum.data() + (size_t)l*kv_dim, kv_dim, config.dim);
                convert_f32_to_i8(weights.wo + (size_t)l*config.dim*config.dim, wo_i8.data() + (size_t)l*config.dim*config.dim, wo_s.data() + (size_t)l*config.dim, wo_sum.data() + (size_t)l*config.dim, config.dim, config.dim);
                convert_f32_to_i8(weights.w1 + (size_t)l*config.hidden_dim*config.dim, w1_i8.data() + (size_t)l*config.hidden_dim*config.dim, w1_s.data() + (size_t)l*config.hidden_dim, w1_sum.data() + (size_t)l*config.hidden_dim, config.hidden_dim, config.dim);
                convert_f32_to_i8(weights.w2 + (size_t)l*config.dim*config.hidden_dim, w2_i8.data() + (size_t)l*config.dim*config.hidden_dim, w2_s.data() + (size_t)l*config.dim, w2_sum.data() + (size_t)l*config.dim, config.dim, config.hidden_dim);
                convert_f32_to_i8(weights.w3 + (size_t)l*config.hidden_dim*config.dim, w3_i8.data() + (size_t)l*config.hidden_dim*config.dim, w3_s.data() + (size_t)l*config.hidden_dim, w3_sum.data() + (size_t)l*config.hidden_dim, config.hidden_dim, config.dim);
            }
            if(weights.wcls) convert_f32_to_i8(weights.wcls, wcls_i8.data(), wcls_s.data(), wcls_sum.data(), config.vocab_size, config.dim);
            else convert_f32_to_i8(weights.token_embedding_table, wcls_i8.data(), wcls_s.data(), wcls_sum.data(), config.vocab_size, config.dim);
            use_i8=true;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
            use_vnni=true;
#else
            use_vnni=false;
#endif
        }catch(...){ use_fp16=false; use_i8=false; use_vnni=false; }
    }
    size_t calculate_total_parameters() const {
        size_t kv_dim=(config.dim*config.n_kv_heads)/config.n_heads; size_t total=0;
        total+=(size_t)config.vocab_size*config.dim; total+=(size_t)config.n_layers*config.dim; total+=(size_t)config.n_layers*config.dim*config.dim;
        total+=(size_t)config.n_layers*kv_dim*config.dim; total+=(size_t)config.n_layers*kv_dim*config.dim; total+=(size_t)config.n_layers*config.dim*config.dim;
        total+=(size_t)config.n_layers*config.dim; total+=(size_t)config.n_layers*config.hidden_dim*config.dim; total+=(size_t)config.n_layers*config.dim*config.hidden_dim;
        total+=(size_t)config.n_layers*config.hidden_dim*config.dim; total+=(size_t)config.dim;
        if(!weights.shared_classifier && weights.wcls!=weights.token_embedding_table) total+=(size_t)config.vocab_size*config.dim;
        return total;
    }
    void reset_kv_cache(){ std::fill(key_cache.begin(),key_cache.end(),0.0f); std::fill(val_cache.begin(),val_cache.end(),0.0f); }
};

extern "C" {

LlamaCppEngine* llama_create(const LlamaCppConfig* config, LlamaCppWeights* weights){ if(!config||!weights) return nullptr; return new LlamaCppEngine(config,weights); }
void llama_free(LlamaCppEngine* engine){ if(engine) delete engine; }
void llama_reset_cache(LlamaCppEngine* engine){ if(engine) engine->reset_kv_cache(); }
int llama_get_threads(){
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
void llama_set_threads(int num_threads){
#ifdef _OPENMP
    if(num_threads>0) omp_set_num_threads(num_threads);
#endif
}

void llama_forward(LlamaCppEngine* engine, int token, int pos, float* out_logits){
    if(!engine||token<0||token>=engine->config.vocab_size) return;
    if(pos>=engine->config.seq_len) return;
    const LlamaCppConfig& p=engine->config; const LlamaCppWeights& w=engine->weights;
    int head_size=p.dim/p.n_heads; int half=head_size/2; int kv_dim=(p.dim*p.n_kv_heads)/p.n_heads; int kv_mul=p.n_heads/p.n_kv_heads;
    const float* emb_row=w.token_embedding_table + (size_t)token * p.dim;
    std::memcpy(engine->x.data(), emb_row, (size_t)p.dim*sizeof(float));
    const float* cos_ptr=engine->cos_cache.data() + (size_t)pos*half;
    const float* sin_ptr=engine->sin_cache.data() + (size_t)pos*half;
    float inv_sqrt_head=1.0f/std::sqrt((float)head_size);

    for(int l=0;l<p.n_layers;l++){
        rmsnorm_forward(engine->xb.data(), engine->x.data(), w.rms_att_weight + (size_t)l * p.dim, p.dim);

        if(engine->use_i8){
            if(engine->use_vnni){
                matmul_forward_int8_vnni(engine->q.data(), engine->xb.data(), engine->wq_i8.data() + (size_t)l * p.dim * p.dim, engine->wq_s.data() + (size_t)l * p.dim, engine->wq_sum.data() + (size_t)l * p.dim, p.dim, p.dim);
                matmul_forward_int8_vnni(engine->k.data(), engine->xb.data(), engine->wk_i8.data() + (size_t)l * kv_dim * p.dim, engine->wk_s.data() + (size_t)l * kv_dim, engine->wk_sum.data() + (size_t)l * kv_dim, p.dim, kv_dim);
                matmul_forward_int8_vnni(engine->v.data(), engine->xb.data(), engine->wv_i8.data() + (size_t)l * kv_dim * p.dim, engine->wv_s.data() + (size_t)l * kv_dim, engine->wv_sum.data() + (size_t)l * kv_dim, p.dim, kv_dim);
            }else{
                matmul_forward_int8(engine->q.data(), engine->xb.data(), engine->wq_i8.data() + (size_t)l * p.dim * p.dim, engine->wq_s.data() + (size_t)l * p.dim, p.dim, p.dim);
                matmul_forward_int8(engine->k.data(), engine->xb.data(), engine->wk_i8.data() + (size_t)l * kv_dim * p.dim, engine->wk_s.data() + (size_t)l * kv_dim, p.dim, kv_dim);
                matmul_forward_int8(engine->v.data(), engine->xb.data(), engine->wv_i8.data() + (size_t)l * kv_dim * p.dim, engine->wv_s.data() + (size_t)l * kv_dim, p.dim, kv_dim);
            }
        }else if(engine->use_fp16){
            matmul_forward_fp16(engine->q.data(), engine->xb.data(), engine->wq_fp16.data() + (size_t)l * p.dim * p.dim, p.dim, p.dim);
            matmul_forward_fp16(engine->k.data(), engine->xb.data(), engine->wk_fp16.data() + (size_t)l * kv_dim * p.dim, p.dim, kv_dim);
            matmul_forward_fp16(engine->v.data(), engine->xb.data(), engine->wv_fp16.data() + (size_t)l * kv_dim * p.dim, p.dim, kv_dim);
        }else{
            matmul_forward(engine->q.data(), engine->xb.data(), w.wq + (size_t)l * p.dim * p.dim, p.dim, p.dim);
            matmul_forward(engine->k.data(), engine->xb.data(), w.wk + (size_t)l * kv_dim * p.dim, p.dim, kv_dim);
            matmul_forward(engine->v.data(), engine->xb.data(), w.wv + (size_t)l * kv_dim * p.dim, p.dim, kv_dim);
        }

        if(p.rope_type==1){
            for(int h=0;h<p.n_heads;h++){ float* qh=engine->q.data()+(size_t)h*head_size; for(int i=0;i<half;i++){ float fcr=cos_ptr[i]; float fci=sin_ptr[i]; float q0=qh[i]; float q1=qh[i+half]; qh[i]=q0*fcr - q1*fci; qh[i+half]=q1*fcr + q0*fci; } }
            for(int h=0;h<p.n_kv_heads;h++){ float* kh=engine->k.data()+(size_t)h*head_size; for(int i=0;i<half;i++){ float fcr=cos_ptr[i]; float fci=sin_ptr[i]; float k0=kh[i]; float k1=kh[i+half]; kh[i]=k0*fcr - k1*fci; kh[i+half]=k1*fcr + k0*fci; } }
        }else{
            for(int i=0;i<p.dim;i+=2){ int h_dim=(i%head_size)/2; float fcr=cos_ptr[h_dim]; float fci=sin_ptr[h_dim]; float q0=engine->q[i]; float q1=engine->q[i+1]; engine->q[i]=q0*fcr - q1*fci; engine->q[i+1]=q0*fci + q1*fcr; if(i<kv_dim){ float k0=engine->k[i]; float k1=engine->k[i+1]; engine->k[i]=k0*fcr - k1*fci; engine->k[i+1]=k0*fci + k1*fcr; } }
        }

        int loff=l * p.seq_len * kv_dim;
        std::memcpy(engine->key_cache.data() + (size_t)loff + (size_t)pos * kv_dim, engine->k.data(), (size_t)kv_dim * sizeof(float));
        std::memcpy(engine->val_cache.data() + (size_t)loff + (size_t)pos * kv_dim, engine->v.data(), (size_t)kv_dim * sizeof(float));
        std::fill(engine->attn_out.begin(), engine->attn_out.end(), 0.0f);

        int n_heads=p.n_heads; bool use_parallel_heads=false;
#ifdef _OPENMP
        if(n_heads>=4 && pos>=16) use_parallel_heads=true;
#endif
        if(use_parallel_heads){
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for(int h=0;h<n_heads;h++){
                const float* q_head=engine->q.data()+(size_t)h*head_size; float* att_head=engine->att.data()+(size_t)h*p.seq_len; int kv_h=h/kv_mul;
                for(int t=0;t<=pos;t++){ const float* k_past=engine->key_cache.data()+(size_t)loff+(size_t)t*kv_dim+(size_t)kv_h*head_size; att_head[t]=dot_product_simd(q_head,k_past,head_size)*inv_sqrt_head; }
                softmax(att_head,pos+1);
                float* out_head=engine->attn_out.data()+(size_t)h*head_size;
                for(int t=0;t<=pos;t++){ const float* v_past=engine->val_cache.data()+(size_t)loff+(size_t)t*kv_dim+(size_t)kv_h*head_size; float a=att_head[t];
#if defined(__AVX512F__)
                    __m512 va=_mm512_set1_ps(a); int d=0; for(; d+15<head_size; d+=16){ __m512 vout=_mm512_loadu_ps(out_head+d); __m512 vv=_mm512_loadu_ps(v_past+d); vout=_mm512_fmadd_ps(vv,va,vout); _mm512_storeu_ps(out_head+d,vout); } if(d+7<head_size){ __m256 va256=_mm256_set1_ps(a); __m256 vout=_mm256_loadu_ps(out_head+d); __m256 vv=_mm256_loadu_ps(v_past+d); vout=_mm256_fmadd_ps(vv,va256,vout); _mm256_storeu_ps(out_head+d,vout); d+=8; } for(;d<head_size;d++) out_head[d]+=a*v_past[d];
#elif defined(__AVX2__)
                    __m256 va=_mm256_set1_ps(a); int d=0; for(; d+7<head_size; d+=8){ __m256 vout=_mm256_loadu_ps(out_head+d); __m256 vv=_mm256_loadu_ps(v_past+d); vout=_mm256_fmadd_ps(vv,va,vout); _mm256_storeu_ps(out_head+d,vout); } for(;d<head_size;d++) out_head[d]+=a*v_past[d];
#else
                    for(int d=0;d<head_size;d++) out_head[d]+=a*v_past[d];
#endif
                }
            }
        }else{
            for(int h=0;h<n_heads;h++){
                const float* q_head=engine->q.data()+(size_t)h*head_size; float* att_head=engine->att.data()+(size_t)h*p.seq_len; int kv_h=h/kv_mul;
                for(int t=0;t<=pos;t++){ const float* k_past=engine->key_cache.data()+(size_t)loff+(size_t)t*kv_dim+(size_t)kv_h*head_size; att_head[t]=dot_product_simd(q_head,k_past,head_size)*inv_sqrt_head; }
                softmax(att_head,pos+1);
                float* out_head=engine->attn_out.data()+(size_t)h*head_size;
                for(int t=0;t<=pos;t++){ const float* v_past=engine->val_cache.data()+(size_t)loff+(size_t)t*kv_dim+(size_t)kv_h*head_size; float a=att_head[t];
#if defined(__AVX512F__)
                    __m512 va=_mm512_set1_ps(a); int d=0; for(; d+15<head_size; d+=16){ __m512 vout=_mm512_loadu_ps(out_head+d); __m512 vv=_mm512_loadu_ps(v_past+d); vout=_mm512_fmadd_ps(vv,va,vout); _mm512_storeu_ps(out_head+d,vout); } if(d+7<head_size){ __m256 va256=_mm256_set1_ps(a); __m256 vout=_mm256_loadu_ps(out_head+d); __m256 vv=_mm256_loadu_ps(v_past+d); vout=_mm256_fmadd_ps(vv,va256,vout); _mm256_storeu_ps(out_head+d,vout); d+=8; } for(;d<head_size;d++) out_head[d]+=a*v_past[d];
#elif defined(__AVX2__)
                    __m256 va=_mm256_set1_ps(a); int d=0; for(; d+7<head_size; d+=8){ __m256 vout=_mm256_loadu_ps(out_head+d); __m256 vv=_mm256_loadu_ps(v_past+d); vout=_mm256_fmadd_ps(vv,va,vout); _mm256_storeu_ps(out_head+d,vout); } for(;d<head_size;d++) out_head[d]+=a*v_past[d];
#else
                    for(int d=0;d<head_size;d++) out_head[d]+=a*v_past[d];
#endif
                }
            }
        }

        if(engine->use_i8){
            if(engine->use_vnni){
                matmul_forward_int8_vnni(engine->xb.data(), engine->attn_out.data(), engine->wo_i8.data() + (size_t)l * p.dim * p.dim, engine->wo_s.data() + (size_t)l * p.dim, engine->wo_sum.data() + (size_t)l * p.dim, p.dim, p.dim);
            }else{
                matmul_forward_int8(engine->xb.data(), engine->attn_out.data(), engine->wo_i8.data() + (size_t)l * p.dim * p.dim, engine->wo_s.data() + (size_t)l * p.dim, p.dim, p.dim);
            }
        }else if(engine->use_fp16){
            matmul_forward_fp16(engine->xb.data(), engine->attn_out.data(), engine->wo_fp16.data() + (size_t)l * p.dim * p.dim, p.dim, p.dim);
        }else{
            matmul_forward(engine->xb.data(), engine->attn_out.data(), w.wo + (size_t)l * p.dim * p.dim, p.dim, p.dim);
        }

#if defined(__AVX512F__)
        { int i=0; for(; i+15 < p.dim; i+=16){ __m512 vx=_mm512_loadu_ps(engine->x.data()+i); __m512 vxb=_mm512_loadu_ps(engine->xb.data()+i); _mm512_storeu_ps(engine->x.data()+i,_mm512_add_ps(vx,vxb)); } if(i+7 < p.dim){ __m256 vx=_mm256_loadu_ps(engine->x.data()+i); __m256 vxb=_mm256_loadu_ps(engine->xb.data()+i); _mm256_storeu_ps(engine->x.data()+i,_mm256_add_ps(vx,vxb)); i+=8; } for(; i<p.dim; i++) engine->x[i]+=engine->xb[i]; }
#elif defined(__AVX2__)
        { int i=0; for(; i+7 < p.dim; i+=8){ __m256 vx=_mm256_loadu_ps(engine->x.data()+i); __m256 vxb=_mm256_loadu_ps(engine->xb.data()+i); _mm256_storeu_ps(engine->x.data()+i,_mm256_add_ps(vx,vxb)); } for(; i<p.dim; i++) engine->x[i]+=engine->xb[i]; }
#else
        for(int i=0;i<p.dim;i++) engine->x[i]+=engine->xb[i];
#endif

        rmsnorm_forward(engine->xb.data(), engine->x.data(), w.rms_ffn_weight + (size_t)l * p.dim, p.dim);

        if(engine->use_i8){
            if(engine->use_vnni){
                matmul_forward_int8_vnni(engine->hb1.data(), engine->xb.data(), engine->w1_i8.data() + (size_t)l * p.hidden_dim * p.dim, engine->w1_s.data() + (size_t)l * p.hidden_dim, engine->w1_sum.data() + (size_t)l * p.hidden_dim, p.dim, p.hidden_dim);
                matmul_forward_int8_vnni(engine->hb3.data(), engine->xb.data(), engine->w3_i8.data() + (size_t)l * p.hidden_dim * p.dim, engine->w3_s.data() + (size_t)l * p.hidden_dim, engine->w3_sum.data() + (size_t)l * p.hidden_dim, p.dim, p.hidden_dim);
            }else{
                matmul_forward_int8(engine->hb1.data(), engine->xb.data(), engine->w1_i8.data() + (size_t)l * p.hidden_dim * p.dim, engine->w1_s.data() + (size_t)l * p.hidden_dim, p.dim, p.hidden_dim);
                matmul_forward_int8(engine->hb3.data(), engine->xb.data(), engine->w3_i8.data() + (size_t)l * p.hidden_dim * p.dim, engine->w3_s.data() + (size_t)l * p.hidden_dim, p.dim, p.hidden_dim);
            }
        }else if(engine->use_fp16){
            matmul_forward_fp16(engine->hb1.data(), engine->xb.data(), engine->w1_fp16.data() + (size_t)l * p.hidden_dim * p.dim, p.dim, p.hidden_dim);
            matmul_forward_fp16(engine->hb3.data(), engine->xb.data(), engine->w3_fp16.data() + (size_t)l * p.hidden_dim * p.dim, p.dim, p.hidden_dim);
        }else{
            matmul_forward(engine->hb1.data(), engine->xb.data(), w.w1 + (size_t)l * p.hidden_dim * p.dim, p.dim, p.hidden_dim);
            matmul_forward(engine->hb3.data(), engine->xb.data(), w.w3 + (size_t)l * p.hidden_dim * p.dim, p.dim, p.hidden_dim);
        }

        for(int j=0;j<p.hidden_dim;j++) engine->hb[j]=fast_silu(engine->hb1[j])*engine->hb3[j];

        if(engine->use_i8){
            if(engine->use_vnni){
                matmul_forward_int8_vnni(engine->xb.data(), engine->hb.data(), engine->w2_i8.data() + (size_t)l * p.dim * p.hidden_dim, engine->w2_s.data() + (size_t)l * p.dim, engine->w2_sum.data() + (size_t)l * p.dim, p.hidden_dim, p.dim);
            }else{
                matmul_forward_int8(engine->xb.data(), engine->hb.data(), engine->w2_i8.data() + (size_t)l * p.dim * p.hidden_dim, engine->w2_s.data() + (size_t)l * p.dim, p.hidden_dim, p.dim);
            }
        }else if(engine->use_fp16){
            matmul_forward_fp16(engine->xb.data(), engine->hb.data(), engine->w2_fp16.data() + (size_t)l * p.dim * p.hidden_dim, p.hidden_dim, p.dim);
        }else{
            matmul_forward(engine->xb.data(), engine->hb.data(), w.w2 + (size_t)l * p.dim * p.hidden_dim, p.hidden_dim, p.dim);
        }

#if defined(__AVX512F__)
        { int i=0; for(; i+15 < p.dim; i+=16){ __m512 vx=_mm512_loadu_ps(engine->x.data()+i); __m512 vxb=_mm512_loadu_ps(engine->xb.data()+i); _mm512_storeu_ps(engine->x.data()+i,_mm512_add_ps(vx,vxb)); } if(i+7 < p.dim){ __m256 vx=_mm256_loadu_ps(engine->x.data()+i); __m256 vxb=_mm256_loadu_ps(engine->xb.data()+i); _mm256_storeu_ps(engine->x.data()+i,_mm256_add_ps(vx,vxb)); i+=8; } for(; i<p.dim; i++) engine->x[i]+=engine->xb[i]; }
#elif defined(__AVX2__)
        { int i=0; for(; i+7 < p.dim; i+=8){ __m256 vx=_mm256_loadu_ps(engine->x.data()+i); __m256 vxb=_mm256_loadu_ps(engine->xb.data()+i); _mm256_storeu_ps(engine->x.data()+i,_mm256_add_ps(vx,vxb)); } for(; i<p.dim; i++) engine->x[i]+=engine->xb[i]; }
#else
        for(int i=0;i<p.dim;i++) engine->x[i]+=engine->xb[i];
#endif
    }

    rmsnorm_forward(engine->x.data(), engine->x.data(), w.rms_final_weight, p.dim);

    if(out_logits){
        if(engine->use_i8){
            if(engine->use_vnni){
                matmul_forward_int8_vnni(engine->logits.data(), engine->x.data(), engine->wcls_i8.data(), engine->wcls_s.data(), engine->wcls_sum.data(), p.dim, p.vocab_size);
            }else{
                matmul_forward_int8(engine->logits.data(), engine->x.data(), engine->wcls_i8.data(), engine->wcls_s.data(), p.dim, p.vocab_size);
            }
        }else if(engine->use_fp16){
            matmul_forward_fp16(engine->logits.data(), engine->x.data(), engine->wcls_fp16.data(), p.dim, p.vocab_size);
        }else{
            const float* cls_w = w.wcls ? w.wcls : w.token_embedding_table;
            matmul_forward(engine->logits.data(), engine->x.data(), cls_w, p.dim, p.vocab_size);
        }
        std::memcpy(out_logits, engine->logits.data(), (size_t)p.vocab_size*sizeof(float));
    }
}

int llama_sample_token(LlamaCppEngine* engine, float temperature, float top_p){
    if(!engine) return 0; const int vocab_size=engine->config.vocab_size;
    if(temperature<=0.0f){ int best_i=0; float best_v=engine->logits[0]; for(int i=1;i<vocab_size;i++) if(engine->logits[i]>best_v){ best_v=engine->logits[i]; best_i=i; } return best_i; }
    std::memcpy(engine->sample_probs.data(), engine->logits.data(), (size_t)vocab_size*sizeof(float));
    float inv_temp=1.0f/temperature; for(int i=0;i<vocab_size;i++) engine->sample_probs[i]*=inv_temp;
    softmax(engine->sample_probs.data(), vocab_size);
    if(top_p < 1.0f){
        for(int i=0;i<vocab_size;i++) engine->sample_candidates[i]={engine->sample_probs[i], i};
        int K=std::min(vocab_size,64);
        std::partial_sort(engine->sample_candidates.begin(), engine->sample_candidates.begin()+K, engine->sample_candidates.end(), [](const auto& a, const auto& b){return a.first > b.first;});
        float cumsum=0.0f; int cutoff_idx=K;
        for(int i=0;i<K;i++){ cumsum+=engine->sample_candidates[i].first; if(cumsum>top_p && i>0){ cutoff_idx=i+1; break; } }
        if(cumsum < top_p && K < vocab_size){
            std::sort(engine->sample_candidates.begin()+K, engine->sample_candidates.end(), [](const auto& a, const auto& b){return a.first > b.first;});
            for(int i=K;i<vocab_size;i++){ cumsum+=engine->sample_candidates[i].first; if(cumsum>top_p && i>0){ cutoff_idx=i+1; break; } }
        }
        float renorm_sum=0.0f; for(int i=0;i<cutoff_idx;i++) renorm_sum+=engine->sample_candidates[i].first;
        std::uniform_real_distribution<float> dist(0.0f, renorm_sum); float r=dist(engine->rng); float acc=0.0f;
        for(int i=0;i<cutoff_idx;i++){ acc+=engine->sample_candidates[i].first; if(r<=acc) return engine->sample_candidates[i].second; }
        return engine->sample_candidates[0].second;
    }
    std::uniform_real_distribution<float> dist(0.0f,1.0f); float r=dist(engine->rng); float acc=0.0f;
    for(int i=0;i<vocab_size;i++){ acc+=engine->sample_probs[i]; if(r<=acc) return i; }
    return vocab_size-1;
}

int llama_generate(LlamaCppEngine* engine, const int* prompt_tokens, int prompt_len, int max_new_tokens, float temperature, float top_p, int* out_tokens){
    if(!engine||!prompt_tokens||prompt_len<=0||!out_tokens||prompt_len>=engine->config.seq_len||max_new_tokens<0) return 0;
    engine->reset_kv_cache(); int pos=0;
    for(int i=0;i<prompt_len;i++){ if(i==prompt_len-1) llama_forward(engine,prompt_tokens[i],pos,engine->logits.data()); else llama_forward(engine,prompt_tokens[i],pos,nullptr); pos++; }
    int generated_count=0;
    for(int step=0; step<max_new_tokens; step++){
        if(pos>=engine->config.seq_len-1) break;
        int next_token=llama_sample_token(engine,temperature,top_p);
        out_tokens[generated_count++]=next_token;
        if(next_token==2) break;
        if(step+1 < max_new_tokens) llama_forward(engine,next_token,pos,engine->logits.data()); else llama_forward(engine,next_token,pos,nullptr);
        pos++;
    }
    return generated_count;
}

float llama_full_train_step(LlamaCppEngine* engine, const int* input_tokens, const int* target_tokens, int seq_len, float lr, float weight_decay, float beta1, float beta2, float eps){
    if (!engine || !input_tokens || !target_tokens || seq_len <= 0 || seq_len >= engine->config.seq_len) return 0.0f;
    const LlamaCppConfig& p = engine->config; LlamaCppWeights& w = engine->weights;
    const int D = p.dim; const int H = p.n_heads; const int HD = D / H; const int KV = (D * p.n_kv_heads) / H;
    const int Hidden = p.hidden_dim; const int L = p.n_layers; const float inv_head = 1.0f / std::sqrt((float)HD);
    FullTrainWorkspace& ws = engine->full_workspace;
    ws.resize(seq_len, L, D, KV, H, Hidden);
    for (int t = 0; t < seq_len; t++) {
        if (input_tokens[t] < 0 || input_tokens[t] >= p.vocab_size || target_tokens[t] < 0 || target_tokens[t] >= p.vocab_size) return 0.0f;
        std::memcpy(ws.states.data() + (size_t)t * D, w.token_embedding_table + (size_t)input_tokens[t] * D, (size_t)D * sizeof(float));
    }
    std::fill(engine->grad_buffer.begin(), engine->grad_buffer.end(), 0.0f);
    std::fill(ws.d_states.begin(), ws.d_states.end(), 0.0f);
    std::fill(ws.d_final.begin(), ws.d_final.end(), 0.0f);
    std::fill(ws.d_norm_att.begin(), ws.d_norm_att.end(), 0.0f);
    std::fill(ws.d_q.begin(), ws.d_q.end(), 0.0f);
    std::fill(ws.d_k.begin(), ws.d_k.end(), 0.0f);
    std::fill(ws.d_v.begin(), ws.d_v.end(), 0.0f);
    std::fill(ws.d_norm_ffn.begin(), ws.d_norm_ffn.end(), 0.0f);
    std::fill(ws.d_gate.begin(), ws.d_gate.end(), 0.0f);
    std::fill(ws.d_up.begin(), ws.d_up.end(), 0.0f);
    std::fill(ws.d_swiglu.begin(), ws.d_swiglu.end(), 0.0f);
    std::fill(ws.d_attn_out.begin(), ws.d_attn_out.end(), 0.0f);
    std::fill(ws.d_attn_state.begin(), ws.d_attn_state.end(), 0.0f);
    size_t off_emb = 0; size_t off_rms_att = off_emb + (size_t)p.vocab_size * D; size_t off_wq = off_rms_att + (size_t)L * D; size_t off_wk = off_wq + (size_t)L * D * D; size_t off_wv = off_wk + (size_t)L * KV * D; size_t off_wo = off_wv + (size_t)L * KV * D; size_t off_rms_ffn = off_wo + (size_t)L * D * D; size_t off_w1 = off_rms_ffn + (size_t)L * D; size_t off_w2 = off_w1 + (size_t)L * Hidden * D; size_t off_w3 = off_w2 + (size_t)L * D * Hidden; size_t off_rms_final = off_w3 + (size_t)L * Hidden * D; size_t off_wcls = off_rms_final + D; float* grad = engine->grad_buffer.data();
    auto state_at = [&](int layer, int t) -> float* { return ws.states.data() + ((size_t)layer * seq_len + t) * D; };
    auto layer_dim_at = [&](std::vector<float>& data, int layer, int t) -> float* { return data.data() + ((size_t)layer * seq_len + t) * D; };
    auto layer_kv_at = [&](std::vector<float>& data, int layer, int t) -> float* { return data.data() + ((size_t)layer * seq_len + t) * KV; };
    auto layer_hidden_at = [&](std::vector<float>& data, int layer, int t) -> float* { return data.data() + ((size_t)layer * seq_len + t) * Hidden; };
    auto score_at = [&](int layer, int t, int h) -> float* { return ws.scores.data() + (((size_t)layer * seq_len + t) * H + h) * seq_len; };
    for (int l = 0; l < L; l++) {
        for (int t = 0; t < seq_len; t++) {
            float* x = state_at(l, t); float* norm = layer_dim_at(ws.norm_att, l, t);
            rmsnorm_forward(norm, x, w.rms_att_weight + (size_t)l * D, D);
            matmul_forward(layer_dim_at(ws.q, l, t), norm, w.wq + (size_t)l * D * D, D, D);
            matmul_forward(layer_kv_at(ws.k, l, t), norm, w.wk + (size_t)l * KV * D, D, KV);
            matmul_forward(layer_kv_at(ws.v, l, t), norm, w.wv + (size_t)l * KV * D, D, KV);
            float* q = layer_dim_at(ws.q, l, t); float* k = layer_kv_at(ws.k, l, t);
            const float* cos_ptr = engine->cos_cache.data() + (size_t)t * (HD / 2); const float* sin_ptr = engine->sin_cache.data() + (size_t)t * (HD / 2);
            if (p.rope_type == 1) {
                const int half = HD / 2;
                for (int h = 0; h < H; h++) { float* qh = q + (size_t)h * HD; for (int i = 0; i < half; i++) { float q0 = qh[i], q1 = qh[i + half]; qh[i] = q0 * cos_ptr[i] - q1 * sin_ptr[i]; qh[i + half] = q1 * cos_ptr[i] + q0 * sin_ptr[i]; } }
                for (int h = 0; h < p.n_kv_heads; h++) { float* kh = k + (size_t)h * HD; for (int i = 0; i < half; i++) { float k0 = kh[i], k1 = kh[i + half]; kh[i] = k0 * cos_ptr[i] - k1 * sin_ptr[i]; kh[i + half] = k1 * cos_ptr[i] + k0 * sin_ptr[i]; } }
            } else {
                for (int i = 0; i < D; i += 2) { int r = (i % HD) / 2; float q0 = q[i], q1 = q[i + 1]; q[i] = q0 * cos_ptr[r] - q1 * sin_ptr[r]; q[i + 1] = q0 * sin_ptr[r] + q1 * cos_ptr[r]; if (i < KV) { float k0 = k[i], k1 = k[i + 1]; k[i] = k0 * cos_ptr[r] - k1 * sin_ptr[r]; k[i + 1] = k0 * sin_ptr[r] + k1 * cos_ptr[r]; } }
            }
        }
        for (int t = 0; t < seq_len; t++) {
            float* out = layer_dim_at(ws.attn_out, l, t); std::fill(out, out + D, 0.0f); const int kv_mul = H / p.n_kv_heads;
            for (int h = 0; h < H; h++) {
                float* weights = score_at(l, t, h); const float* qh = layer_dim_at(ws.q, l, t) + (size_t)h * HD; const int kv_h = h / kv_mul; float max_score = -1e30f;
                for (int u = 0; u <= t; u++) { const float* kh = layer_kv_at(ws.k, l, u) + (size_t)kv_h * HD; weights[u] = dot_product_simd(qh, kh, HD) * inv_head; if (weights[u] > max_score) max_score = weights[u]; }
                float sum = 0.0f; for (int u = 0; u <= t; u++) { weights[u] = std::exp(weights[u] - max_score); sum += weights[u]; } float inv_sum = 1.0f / std::max(sum, 1e-20f);
                for (int u = 0; u < seq_len; u++) weights[u] = u <= t ? weights[u] * inv_sum : 0.0f;
                float* out_head = out + (size_t)h * HD;
                for (int u = 0; u <= t; u++) { const float* vh = layer_kv_at(ws.v, l, u) + (size_t)kv_h * HD; for (int d = 0; d < HD; d++) out_head[d] += weights[u] * vh[d]; }
            }
            float* attn_state = layer_dim_at(ws.attn_state, l, t); matmul_forward(engine->xb.data(), out, w.wo + (size_t)l * D * D, D, D); const float* x = state_at(l, t); for (int d = 0; d < D; d++) attn_state[d] = x[d] + engine->xb[d];
            float* norm_ffn = layer_dim_at(ws.norm_ffn, l, t); rmsnorm_forward(norm_ffn, attn_state, w.rms_ffn_weight + (size_t)l * D, D);
            float* gate = layer_hidden_at(ws.gate, l, t); float* up = layer_hidden_at(ws.up, l, t); float* swiglu = layer_hidden_at(ws.swiglu, l, t);
            matmul_forward(gate, norm_ffn, w.w1 + (size_t)l * Hidden * D, D, Hidden); matmul_forward(up, norm_ffn, w.w3 + (size_t)l * Hidden * D, D, Hidden);
            for (int j = 0; j < Hidden; j++) swiglu[j] = silu(gate[j]) * up[j];
            float* next = state_at(l + 1, t); matmul_forward(engine->xb.data(), swiglu, w.w2 + (size_t)l * D * Hidden, Hidden, D); for (int d = 0; d < D; d++) next[d] = attn_state[d] + engine->xb[d];
        }
    }
    float total_loss = 0.0f; std::fill(ws.final_norm.begin(), ws.final_norm.end(), 0.0f); const float inv_seq = 1.0f / seq_len; const float* cls_w = w.wcls ? w.wcls : w.token_embedding_table;
    for (int t = 0; t < seq_len; t++) {
        float* final = ws.final_norm.data() + (size_t)t * D; rmsnorm_forward(final, state_at(L, t), w.rms_final_weight, D);
        float max_logit = -1e30f; for (int v = 0; v < p.vocab_size; v++) { float value = dot_product_simd(cls_w + (size_t)v * D, final, D); engine->train_step_logits[v] = value; if (value > max_logit) max_logit = value; }
        float sum_exp = 0.0f; for (int v = 0; v < p.vocab_size; v++) { engine->train_probs[v] = std::exp(engine->train_step_logits[v] - max_logit); sum_exp += engine->train_probs[v]; }
        float inv_sum = 1.0f / std::max(sum_exp, 1e-20f); int target = target_tokens[t]; float target_prob = std::max(engine->train_probs[target] * inv_sum, 1e-20f); total_loss -= std::log(target_prob);
        float* d_final = ws.d_final.data() + (size_t)t * D; std::fill(d_final, d_final + D, 0.0f);
        for (int v = 0; v < p.vocab_size; v++) { float dlogit = (engine->train_probs[v] * inv_sum - (v == target ? 1.0f : 0.0f)) * inv_seq; float* dcls = grad + (w.shared_classifier ? off_emb : off_wcls) + (size_t)v * D; const float* cls_row = cls_w + (size_t)v * D; for (int d = 0; d < D; d++) { dcls[d] += dlogit * final[d]; d_final[d] += dlogit * cls_row[d]; } }
    }
    for (int t = 0; t < seq_len; t++) { float* d_final = ws.d_final.data() + (size_t)t * D; float* dx_final = ws.d_states.data() + ((size_t)L * seq_len + t) * D; rmsnorm_backward(dx_final, grad + off_rms_final, d_final, state_at(L, t), w.rms_final_weight, D); }
    for (int l = L - 1; l >= 0; l--) {
        const size_t wq_off = off_wq + (size_t)l * D * D; const size_t wk_off = off_wk + (size_t)l * KV * D; const size_t wv_off = off_wv + (size_t)l * KV * D; const size_t wo_off = off_wo + (size_t)l * D * D; const size_t w1_off = off_w1 + (size_t)l * Hidden * D; const size_t w2_off = off_w2 + (size_t)l * D * Hidden; const size_t w3_off = off_w3 + (size_t)l * Hidden * D;
        std::fill(ws.d_q.begin() + (size_t)l * seq_len * D, ws.d_q.begin() + (size_t)(l + 1) * seq_len * D, 0.0f); std::fill(ws.d_k.begin() + (size_t)l * seq_len * KV, ws.d_k.begin() + (size_t)(l + 1) * seq_len * KV, 0.0f); std::fill(ws.d_v.begin() + (size_t)l * seq_len * KV, ws.d_v.begin() + (size_t)(l + 1) * seq_len * KV, 0.0f);
        for (int t = 0; t < seq_len; t++) {
            float* d_out = ws.d_states.data() + ((size_t)(l + 1) * seq_len + t) * D; float* d_attn_state = layer_dim_at(ws.d_attn_state, l, t); float* d_norm_ffn = layer_dim_at(ws.d_norm_ffn, l, t); float* d_gate = layer_hidden_at(ws.d_gate, l, t); float* d_up = layer_hidden_at(ws.d_up, l, t); float* d_swiglu = layer_hidden_at(ws.d_swiglu, l, t);
            const float* gate = layer_hidden_at(ws.gate, l, t); const float* up = layer_hidden_at(ws.up, l, t); const float* norm_ffn = layer_dim_at(ws.norm_ffn, l, t); const float* attn_state = layer_dim_at(ws.attn_state, l, t);
            std::fill(d_norm_ffn, d_norm_ffn + D, 0.0f); std::fill(d_gate, d_gate + Hidden, 0.0f); std::fill(d_up, d_up + Hidden, 0.0f);
            matmul_backward(d_swiglu, grad + w2_off, d_out, layer_hidden_at(ws.swiglu, l, t), w.w2 + (size_t)l * D * Hidden, Hidden, D);
            for (int j = 0; j < Hidden; j++) { float s = silu(gate[j]); d_gate[j] = d_swiglu[j] * up[j] * silu_deriv(gate[j]); d_up[j] = d_swiglu[j] * s; }
            matmul_backward(d_norm_ffn, grad + w1_off, d_gate, norm_ffn, w.w1 + (size_t)l * Hidden * D, D, Hidden);
            matmul_backward(engine->xb.data(), grad + w3_off, d_up, norm_ffn, w.w3 + (size_t)l * Hidden * D, D, Hidden);
            for (int d = 0; d < D; d++) d_norm_ffn[d] += engine->xb[d];
            rmsnorm_backward(d_attn_state, grad + off_rms_ffn + (size_t)l * D, d_norm_ffn, attn_state, w.rms_ffn_weight + (size_t)l * D, D);
            for (int d = 0; d < D; d++) d_attn_state[d] += d_out[d];
            float* d_attn_out = layer_dim_at(ws.d_attn_out, l, t); matmul_backward(d_attn_out, grad + wo_off, d_attn_state, layer_dim_at(ws.attn_out, l, t), w.wo + (size_t)l * D * D, D, D);
            float* d_input = ws.d_states.data() + ((size_t)l * seq_len + t) * D; for (int d = 0; d < D; d++) d_input[d] += d_attn_state[d];
        }
        const int kv_mul = H / p.n_kv_heads;
        for (int t = 0; t < seq_len; t++) {
            for (int h = 0; h < H; h++) {
                float* weights = score_at(l, t, h); const float* dcontext = layer_dim_at(ws.d_attn_out, l, t) + (size_t)h * HD; const int kv_h = h / kv_mul; float weighted = 0.0f;
                for (int u = 0; u <= t; u++) weighted += weights[u] * dot_product_simd(dcontext, layer_kv_at(ws.v, l, u) + (size_t)kv_h * HD, HD);
                for (int u = 0; u <= t; u++) { float dscore = weights[u] * (dot_product_simd(dcontext, layer_kv_at(ws.v, l, u) + (size_t)kv_h * HD, HD) - weighted); float* dqh = layer_dim_at(ws.d_q, l, t) + (size_t)h * HD; float* dkh = layer_kv_at(ws.d_k, l, u) + (size_t)kv_h * HD; const float* kh = layer_kv_at(ws.k, l, u) + (size_t)kv_h * HD; const float* qh = layer_dim_at(ws.q, l, t) + (size_t)h * HD; float* dvh = layer_kv_at(ws.d_v, l, u) + (size_t)kv_h * HD; for (int d = 0; d < HD; d++) { dqh[d] += dscore * kh[d] * inv_head; dkh[d] += dscore * qh[d] * inv_head; dvh[d] += weights[u] * dcontext[d]; } }
            }
        }
        for (int t = 0; t < seq_len; t++) {
            float* dq = layer_dim_at(ws.d_q, l, t); float* dk = layer_kv_at(ws.d_k, l, t); const float* cos_ptr = engine->cos_cache.data() + (size_t)t * (HD / 2); const float* sin_ptr = engine->sin_cache.data() + (size_t)t * (HD / 2);
            if (p.rope_type == 1) {
                const int half = HD / 2;
                for (int h = 0; h < H; h++) { float* dqh = dq + (size_t)h * HD; for (int i = 0; i < half; i++) { float d0 = dqh[i], d1 = dqh[i + half]; dqh[i] = d0 * cos_ptr[i] + d1 * sin_ptr[i]; dqh[i + half] = -d0 * sin_ptr[i] + d1 * cos_ptr[i]; } }
                for (int h = 0; h < p.n_kv_heads; h++) { float* dkh = dk + (size_t)h * HD; for (int i = 0; i < half; i++) { float d0 = dkh[i], d1 = dkh[i + half]; dkh[i] = d0 * cos_ptr[i] + d1 * sin_ptr[i]; dkh[i + half] = -d0 * sin_ptr[i] + d1 * cos_ptr[i]; } }
            } else {
                for (int i = 0; i < D; i += 2) { int r = (i % HD) / 2; float d0 = dq[i], d1 = dq[i + 1]; dq[i] = d0 * cos_ptr[r] + d1 * sin_ptr[r]; dq[i + 1] = -d0 * sin_ptr[r] + d1 * cos_ptr[r]; if (i < KV) { float k0 = dk[i], k1 = dk[i + 1]; dk[i] = k0 * cos_ptr[r] + k1 * sin_ptr[r]; dk[i + 1] = -k0 * sin_ptr[r] + k1 * cos_ptr[r]; } }
            }
            const float* norm = layer_dim_at(ws.norm_att, l, t); float* dnorm = layer_dim_at(ws.d_norm_att, l, t); std::fill(dnorm, dnorm + D, 0.0f);
            matmul_backward(dnorm, grad + wq_off, dq, norm, w.wq + (size_t)l * D * D, D, D);
            matmul_backward(engine->xb.data(), grad + wk_off, dk, norm, w.wk + (size_t)l * KV * D, D, KV);
            for (int d = 0; d < D; d++) dnorm[d] += engine->xb[d];
            matmul_backward(engine->x.data(), grad + wv_off, layer_kv_at(ws.d_v, l, t), norm, w.wv + (size_t)l * KV * D, D, KV);
            for (int d = 0; d < D; d++) dnorm[d] += engine->x[d];
            float* d_input = ws.d_states.data() + ((size_t)l * seq_len + t) * D; rmsnorm_backward(engine->x.data(), grad + off_rms_att + (size_t)l * D, dnorm, state_at(l, t), w.rms_att_weight + (size_t)l * D, D); for (int d = 0; d < D; d++) d_input[d] += engine->x[d];
        }
    }
    float* gemb = grad + off_emb; for (int t = 0; t < seq_len; t++) { float* row = gemb + (size_t)input_tokens[t] * D; const float* dx = ws.d_states.data() + (size_t)t * D; for (int d = 0; d < D; d++) row[d] += dx[d]; }
    engine->adam_step++; const float b1_corr = 1.0f - std::pow(beta1, (float)engine->adam_step); const float b2_corr = 1.0f - std::pow(beta2, (float)engine->adam_step); const float step_size = lr * std::sqrt(b2_corr) / std::max(b1_corr, 1e-12f);
    auto update_group = [&](float* params, size_t offset, size_t count){ float* g = grad + offset; float* m = engine->m_buffer.data() + offset; float* v = engine->v_buffer.data() + offset; for(size_t i=0;i<count;i++){ m[i]=beta1*m[i]+(1.0f-beta1)*g[i]; v[i]=beta2*v[i]+(1.0f-beta2)*g[i]*g[i]; params[i]-=lr*weight_decay*params[i]; params[i]-=step_size*m[i]/(std::sqrt(v[i])+eps); } };
    update_group(w.token_embedding_table, off_emb, (size_t)p.vocab_size * D); update_group(w.rms_att_weight, off_rms_att, (size_t)L * D); update_group(w.wq, off_wq, (size_t)L * D * D); update_group(w.wk, off_wk, (size_t)L * KV * D); update_group(w.wv, off_wv, (size_t)L * KV * D); update_group(w.wo, off_wo, (size_t)L * D * D); update_group(w.rms_ffn_weight, off_rms_ffn, (size_t)L * D); update_group(w.w1, off_w1, (size_t)L * Hidden * D); update_group(w.w2, off_w2, (size_t)L * D * Hidden); update_group(w.w3, off_w3, (size_t)L * Hidden * D); update_group(w.rms_final_weight, off_rms_final, D);
    if (!w.shared_classifier && w.wcls && w.wcls != w.token_embedding_table) update_group(w.wcls, off_wcls, (size_t)p.vocab_size * D);
    return total_loss * inv_seq;
}

float llama_train_step(LlamaCppEngine* engine, const int* input_tokens, const int* target_tokens, int seq_len, float lr, float weight_decay, float beta1, float beta2, float eps){
    if (!engine || !input_tokens || !target_tokens || seq_len <= 0) return 0.0f;
    const LlamaCppConfig& p = engine->config; LlamaCppWeights& w = engine->weights; float total_loss = 0.0f; engine->reset_kv_cache();
    size_t emb_size = (size_t)p.vocab_size * p.dim;
    if (engine->m_buffer.size() < emb_size) { engine->m_buffer.assign(emb_size, 0.0f); engine->v_buffer.assign(emb_size, 0.0f); }
    if (w.wcls && engine->m_cls.size() < emb_size) { engine->m_cls.assign(emb_size, 0.0f); engine->v_cls.assign(emb_size, 0.0f); }
    float* step_logits = engine->train_step_logits.data(); float* dlogits = engine->train_dlogits.data(); float* probs = engine->train_probs.data();
    int active_targets = 0; for (int pos = 0; pos < seq_len; pos++) { int t = target_tokens[pos]; if (t >= 0 && t < p.vocab_size) active_targets++; }
    if (active_targets == 0) return 0.0f; float inv_targets = 1.0f / active_targets;
    engine->adam_step++; int t_step = engine->adam_step; float beta1_corr = 1.0f - std::pow(beta1, (float)t_step); float beta2_corr = 1.0f - std::pow(beta2, (float)t_step);
    float inv_beta1_corr = 1.0f / std::max(1e-7f, beta1_corr); float inv_beta2_corr = 1.0f / std::max(1e-7f, beta2_corr);
    for (int pos = 0; pos < seq_len; pos++) {
        int in_tok = input_tokens[pos]; int target_tok = target_tokens[pos]; llama_forward(engine, in_tok, pos, step_logits);
        if (target_tok < 0 || target_tok >= p.vocab_size) continue;
        std::memcpy(probs, step_logits, (size_t)p.vocab_size * sizeof(float)); softmax(probs, p.vocab_size);
        float target_prob = std::max(1e-12f, probs[target_tok]); total_loss += -std::log(target_prob);
        for (int i = 0; i < p.vocab_size; i++) dlogits[i] = (probs[i] - (i == target_tok ? 1.0f : 0.0f)) * inv_targets;
        const float* cls_w = w.wcls ? w.wcls : w.token_embedding_table; float* d_emb_row = w.token_embedding_table + (size_t)in_tok * p.dim; float* m_emb_row = engine->m_buffer.data() + (size_t)in_tok * p.dim; float* v_emb_row = engine->v_buffer.data() + (size_t)in_tok * p.dim;
        std::fill(engine->train_grad_embedding.begin(), engine->train_grad_embedding.end(), 0.0f); float* g = engine->train_grad_embedding.data();
        for (int v = 0; v < p.vocab_size; v++) { float dv = dlogits[v]; if (v != target_tok && std::abs(dv) < 1e-5f) continue; const float* cls_row = cls_w + (size_t)v * p.dim; for (int d = 0; d < p.dim; d++) g[d] += dv * cls_row[d]; }
        for (int d = 0; d < p.dim; d++) { float grad = g[d]; m_emb_row[d] = beta1 * m_emb_row[d] + (1.0f - beta1) * grad; v_emb_row[d] = beta2 * v_emb_row[d] + (1.0f - beta2) * (grad * grad); float m_hat = m_emb_row[d] * inv_beta1_corr; float v_hat = v_emb_row[d] * inv_beta2_corr; float step_val = m_hat / (std::sqrt(v_hat) + eps); d_emb_row[d] -= lr * (step_val + weight_decay * d_emb_row[d]); }
        float* cls_base = (w.wcls ? w.wcls : w.token_embedding_table); float* m_cls_base = (w.wcls ? engine->m_cls.data() : engine->m_buffer.data()); float* v_cls_base = (w.wcls ? engine->v_cls.data() : engine->v_buffer.data()); const float* x_vec = engine->x.data();
        float* cls_target = cls_base + (size_t)target_tok * p.dim; float* m_target = m_cls_base + (size_t)target_tok * p.dim; float* v_target = v_cls_base + (size_t)target_tok * p.dim; float d_target = dlogits[target_tok];
        for (int d = 0; d < p.dim; d++) { float grad = d_target * x_vec[d]; m_target[d] = beta1 * m_target[d] + (1.0f - beta1) * grad; v_target[d] = beta2 * v_target[d] + (1.0f - beta2) * (grad * grad); float m_hat = m_target[d] * inv_beta1_corr; float v_hat = v_target[d] * inv_beta2_corr; float step_val = m_hat / (std::sqrt(v_hat) + eps); cls_target[d] -= lr * (step_val + weight_decay * cls_target[d]); }
        float prob_threshold = (p.vocab_size <= 1024) ? 0.0f : 0.005f;
        for (int v = 0; v < p.vocab_size; v++) { if (v == target_tok) continue; if (prob_threshold > 0.0f && probs[v] < prob_threshold) continue; float* cls_row = cls_base + (size_t)v * p.dim; float* m_row = m_cls_base + (size_t)v * p.dim; float* v_row = v_cls_base + (size_t)v * p.dim; float dv = dlogits[v]; for (int d = 0; d < p.dim; d++) { float grad = dv * x_vec[d]; m_row[d] = beta1 * m_row[d] + (1.0f - beta1) * grad; v_row[d] = beta2 * v_row[d] + (1.0f - beta2) * (grad * grad); float m_hat = m_row[d] * inv_beta1_corr; float v_hat = v_row[d] * inv_beta2_corr; float step_val = m_hat / (std::sqrt(v_hat) + eps); cls_row[d] -= lr * (step_val + weight_decay * cls_row[d]); } }
    }
    return total_loss / active_targets;
}

} // extern C
