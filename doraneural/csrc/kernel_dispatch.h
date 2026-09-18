#ifndef DORANEURAL_KERNEL_DISPATCH_H
#define DORANEURAL_KERNEL_DISPATCH_H

#include "cpu_features.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(DORANEURAL_ARCH_ARM64)
#include <arm_neon.h>
#elif defined(DORANEURAL_ARCH_X86)
#include <immintrin.h>
#endif

// ============================================================================
// 1. Portable Reference Implementations (Scalar Fallback)
// ============================================================================

namespace scalar_kernels {

inline float silu(float x) {
    if (x > 88.0f) return x;
    if (x < -88.0f) return 0.0f;
    return x / (1.0f + std::exp(-x));
}

inline float dot_product(const float* a, const float* b, int n) {
    float acc = 0.0f;
    for (int j = 0; j < n; j++) acc += a[j] * b[j];
    return acc;
}

inline void rmsnorm_forward(float* __restrict__ out, const float* __restrict__ x, const float* __restrict__ weight, int size, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) sum_sq += x[i] * x[i];
    float inv_rms = 1.0f / std::sqrt(sum_sq / size + eps);
    for (int i = 0; i < size; i++) out[i] = x[i] * inv_rms * weight[i];
}

inline void matmul_fp32(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ W, int n, int d) {
    if (d <= 0 || n <= 0) return;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((size_t)d * n >= 16384)
#endif
    for (int i = 0; i < d; i++) {
        const float* row = W + (size_t)i * n;
        float acc = 0.0f;
        for (int j = 0; j < n; j++) acc += row[j] * x[j];
        y[i] = acc;
    }
}

inline void matmul_int8(float* __restrict__ y, const float* __restrict__ x, const int8_t* __restrict__ W, const float* __restrict__ scales, int n, int d) {
    if (d <= 0 || n <= 0) return;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((size_t)d * n >= 16384)
#endif
    for (int i = 0; i < d; i++) {
        const int8_t* row = W + (size_t)i * n;
        float acc = 0.0f;
        for (int j = 0; j < n; j++) acc += (float)row[j] * x[j];
        y[i] = acc * scales[i];
    }
}

} // namespace scalar_kernels

// ============================================================================
// 2. ARM64 NEON Vectorized Implementations
// ============================================================================

#if defined(DORANEURAL_ARCH_ARM64)
namespace neon_kernels {

inline float dot_product(const float* a, const float* b, int n) {
    int j = 0;
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    for (; j + 7 < n; j += 8) {
        float32x4_t a0 = vld1q_f32(a + j);
        float32x4_t b0 = vld1q_f32(b + j);
        sum0 = vfmaq_f32(sum0, a0, b0);
        float32x4_t a1 = vld1q_f32(a + j + 4);
        float32x4_t b1 = vld1q_f32(b + j + 4);
        sum1 = vfmaq_f32(sum1, a1, b1);
    }
    for (; j + 3 < n; j += 4) {
        float32x4_t a0 = vld1q_f32(a + j);
        float32x4_t b0 = vld1q_f32(b + j);
        sum0 = vfmaq_f32(sum0, a0, b0);
    }
    float acc = vaddvq_f32(vaddq_f32(sum0, sum1));
    for (; j < n; j++) acc += a[j] * b[j];
    return acc;
}

inline void rmsnorm_forward(float* __restrict__ out, const float* __restrict__ x, const float* __restrict__ weight, int size, float eps) {
    int i = 0;
    float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
    for (; i + 7 < size; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        sum_vec0 = vfmaq_f32(sum_vec0, v0, v0);
        sum_vec1 = vfmaq_f32(sum_vec1, v1, v1);
    }
    for (; i + 3 < size; i += 4) {
        float32x4_t v0 = vld1q_f32(x + i);
        sum_vec0 = vfmaq_f32(sum_vec0, v0, v0);
    }
    float sum_sq = vaddvq_f32(vaddq_f32(sum_vec0, sum_vec1));
    for (; i < size; i++) sum_sq += x[i] * x[i];

    float inv_rms = 1.0f / std::sqrt(sum_sq / size + eps);
    float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);
    int i2 = 0;
    for (; i2 + 7 < size; i2 += 8) {
        float32x4_t v0 = vld1q_f32(x + i2);
        float32x4_t w0 = vld1q_f32(weight + i2);
        float32x4_t v1 = vld1q_f32(x + i2 + 4);
        float32x4_t w1 = vld1q_f32(weight + i2 + 4);
        vst1q_f32(out + i2, vmulq_f32(vmulq_f32(v0, inv_rms_vec), w0));
        vst1q_f32(out + i2 + 4, vmulq_f32(vmulq_f32(v1, inv_rms_vec), w1));
    }
    for (; i2 + 3 < size; i2 += 4) {
        float32x4_t v0 = vld1q_f32(x + i2);
        float32x4_t w0 = vld1q_f32(weight + i2);
        vst1q_f32(out + i2, vmulq_f32(vmulq_f32(v0, inv_rms_vec), w0));
    }
    for (; i2 < size; i2++) out[i2] = x[i2] * inv_rms * weight[i2];
}

inline void matmul_fp32(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ W, int n, int d) {
    if (d <= 0 || n <= 0) return;
    if ((size_t)d * n < 16384) {
        for (int i = 0; i < d; i++) y[i] = dot_product(W + (size_t)i * n, x, n);
        return;
    }
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < d; i += 4) {
        int rows = std::min(4, d - i);
        const float* r0 = W + (size_t)(i + 0) * n;
        const float* r1 = rows > 1 ? W + (size_t)(i + 1) * n : nullptr;
        const float* r2 = rows > 2 ? W + (size_t)(i + 2) * n : nullptr;
        const float* r3 = rows > 3 ? W + (size_t)(i + 3) * n : nullptr;

        float32x4_t s0_0 = vdupq_n_f32(0.0f), s0_1 = vdupq_n_f32(0.0f);
        float32x4_t s1_0 = vdupq_n_f32(0.0f), s1_1 = vdupq_n_f32(0.0f);
        float32x4_t s2_0 = vdupq_n_f32(0.0f), s2_1 = vdupq_n_f32(0.0f);
        float32x4_t s3_0 = vdupq_n_f32(0.0f), s3_1 = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 7 < n; j += 8) {
            float32x4_t vx0 = vld1q_f32(x + j);
            float32x4_t vx1 = vld1q_f32(x + j + 4);
            if (rows > 0) { s0_0 = vfmaq_f32(s0_0, vld1q_f32(r0 + j), vx0); s0_1 = vfmaq_f32(s0_1, vld1q_f32(r0 + j + 4), vx1); }
            if (rows > 1) { s1_0 = vfmaq_f32(s1_0, vld1q_f32(r1 + j), vx0); s1_1 = vfmaq_f32(s1_1, vld1q_f32(r1 + j + 4), vx1); }
            if (rows > 2) { s2_0 = vfmaq_f32(s2_0, vld1q_f32(r2 + j), vx0); s2_1 = vfmaq_f32(s2_1, vld1q_f32(r2 + j + 4), vx1); }
            if (rows > 3) { s3_0 = vfmaq_f32(s3_0, vld1q_f32(r3 + j), vx0); s3_1 = vfmaq_f32(s3_1, vld1q_f32(r3 + j + 4), vx1); }
        }
        for (; j + 3 < n; j += 4) {
            float32x4_t vx = vld1q_f32(x + j);
            if (rows > 0) s0_0 = vfmaq_f32(s0_0, vld1q_f32(r0 + j), vx);
            if (rows > 1) s1_0 = vfmaq_f32(s1_0, vld1q_f32(r1 + j), vx);
            if (rows > 2) s2_0 = vfmaq_f32(s2_0, vld1q_f32(r2 + j), vx);
            if (rows > 3) s3_0 = vfmaq_f32(s3_0, vld1q_f32(r3 + j), vx);
        }
        float acc0 = vaddvq_f32(vaddq_f32(s0_0, s0_1));
        float acc1 = rows > 1 ? vaddvq_f32(vaddq_f32(s1_0, s1_1)) : 0.0f;
        float acc2 = rows > 2 ? vaddvq_f32(vaddq_f32(s2_0, s2_1)) : 0.0f;
        float acc3 = rows > 3 ? vaddvq_f32(vaddq_f32(s3_0, s3_1)) : 0.0f;
        for (; j < n; j++) {
            float xj = x[j];
            acc0 += r0[j] * xj;
            if (rows > 1) acc1 += r1[j] * xj;
            if (rows > 2) acc2 += r2[j] * xj;
            if (rows > 3) acc3 += r3[j] * xj;
        }
        y[i] = acc0; if (rows > 1) y[i+1] = acc1; if (rows > 2) y[i+2] = acc2; if (rows > 3) y[i+3] = acc3;
    }
}

inline void matmul_int8(float* __restrict__ y, const float* __restrict__ x, const int8_t* __restrict__ W, const float* __restrict__ scales, int n, int d) {
    if (d <= 0 || n <= 0) return;
    if ((size_t)d * n < 16384) {
        for (int i = 0; i < d; i++) {
            const int8_t* row = W + (size_t)i * n;
            float acc = 0.0f;
            for (int j = 0; j < n; j++) acc += (float)row[j] * x[j];
            y[i] = acc * scales[i];
        }
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

        float32x4_t s0_0 = vdupq_n_f32(0.0f), s0_1 = vdupq_n_f32(0.0f);
        float32x4_t s1_0 = vdupq_n_f32(0.0f), s1_1 = vdupq_n_f32(0.0f);
        float32x4_t s2_0 = vdupq_n_f32(0.0f), s2_1 = vdupq_n_f32(0.0f);
        float32x4_t s3_0 = vdupq_n_f32(0.0f), s3_1 = vdupq_n_f32(0.0f);
        int j = 0;
        for (; j + 7 < n; j += 8) {
            float32x4_t xv0 = vld1q_f32(x + j);
            float32x4_t xv1 = vld1q_f32(x + j + 4);
            if (rows > 0) {
                int8x8_t b8 = vld1_s8(r0 + j);
                int16x8_t b16 = vmovl_s8(b8);
                float32x4_t w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16)));
                float32x4_t w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16)));
                s0_0 = vfmaq_f32(s0_0, w0, xv0); s0_1 = vfmaq_f32(s0_1, w1, xv1);
            }
            if (rows > 1) {
                int8x8_t b8 = vld1_s8(r1 + j);
                int16x8_t b16 = vmovl_s8(b8);
                float32x4_t w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16)));
                float32x4_t w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16)));
                s1_0 = vfmaq_f32(s1_0, w0, xv0); s1_1 = vfmaq_f32(s1_1, w1, xv1);
            }
            if (rows > 2) {
                int8x8_t b8 = vld1_s8(r2 + j);
                int16x8_t b16 = vmovl_s8(b8);
                float32x4_t w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16)));
                float32x4_t w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16)));
                s2_0 = vfmaq_f32(s2_0, w0, xv0); s2_1 = vfmaq_f32(s2_1, w1, xv1);
            }
            if (rows > 3) {
                int8x8_t b8 = vld1_s8(r3 + j);
                int16x8_t b16 = vmovl_s8(b8);
                float32x4_t w0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(b16)));
                float32x4_t w1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(b16)));
                s3_0 = vfmaq_f32(s3_0, w0, xv0); s3_1 = vfmaq_f32(s3_1, w1, xv1);
            }
        }
        float acc0 = vaddvq_f32(vaddq_f32(s0_0, s0_1)) * s0;
        float acc1 = rows > 1 ? vaddvq_f32(vaddq_f32(s1_0, s1_1)) * s1 : 0.0f;
        float acc2 = rows > 2 ? vaddvq_f32(vaddq_f32(s2_0, s2_1)) * s2 : 0.0f;
        float acc3 = rows > 3 ? vaddvq_f32(vaddq_f32(s3_0, s3_1)) * s3 : 0.0f;
        for (; j < n; j++) {
            float xj = x[j];
            acc0 += (float)r0[j] * xj * s0;
            if (rows > 1) acc1 += (float)r1[j] * xj * s1;
            if (rows > 2) acc2 += (float)r2[j] * xj * s2;
            if (rows > 3) acc3 += (float)r3[j] * xj * s3;
        }
        y[i] = acc0; if (rows > 1) y[i+1] = acc1; if (rows > 2) y[i+2] = acc2; if (rows > 3) y[i+3] = acc3;
    }
}

} // namespace neon_kernels
#endif // DORANEURAL_ARCH_ARM64

// ============================================================================
// 3. Dispatch Struct and Resolver
// ============================================================================

typedef float (*DotProductFn)(const float* a, const float* b, int n);
typedef void (*RMSNormFn)(float* __restrict__ out, const float* __restrict__ x, const float* __restrict__ weight, int size, float eps);
typedef void (*MatmulFP32Fn)(float* __restrict__ y, const float* __restrict__ x, const float* __restrict__ W, int n, int d);
typedef void (*MatmulINT8Fn)(float* __restrict__ y, const float* __restrict__ x, const int8_t* __restrict__ W, const float* __restrict__ scales, int n, int d);

struct KernelDispatch {
    const char* backend_name;
    DotProductFn dot_product;
    RMSNormFn rmsnorm_forward;
    MatmulFP32Fn matmul_fp32;
    MatmulINT8Fn matmul_int8;
};

inline KernelDispatch resolve_kernel_dispatch() {
    CPUFeatures cpu = detect_cpu_features();
    const char* env_backend = std::getenv("DORANEURAL_BACKEND");

    KernelDispatch disp;
    disp.backend_name = "SCALAR";
    disp.dot_product = scalar_kernels::dot_product;
    disp.rmsnorm_forward = scalar_kernels::rmsnorm_forward;
    disp.matmul_fp32 = scalar_kernels::matmul_fp32;
    disp.matmul_int8 = scalar_kernels::matmul_int8;

    if (env_backend && std::strcmp(env_backend, "scalar") == 0) {
        return disp;
    }

#if defined(DORANEURAL_ARCH_ARM64)
    if (cpu.has_arm_neon) {
        disp.backend_name = "ARM_NEON";
        disp.dot_product = neon_kernels::dot_product;
        disp.rmsnorm_forward = neon_kernels::rmsnorm_forward;
        disp.matmul_fp32 = neon_kernels::matmul_fp32;
        disp.matmul_int8 = neon_kernels::matmul_int8;
    }
#endif

    return disp;
}

#endif // DORANEURAL_KERNEL_DISPATCH_H
