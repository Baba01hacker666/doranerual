#include "llm_engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

inline float silu(float x) {
    if (x > 88.0f) return x;
    if (x < -88.0f) return 0.0f;
    return x / (1.0f + std::exp(-x));
}

inline float silu_deriv(float x) {
    if (x > 88.0f) return 1.0f;
    if (x < -88.0f) return 0.0f;
    float s = 1.0f / (1.0f + std::exp(-x));
    return s + x * s * (1.0f - s);
}

inline float dot_product_simd(const float* a, const float* b, int n) {
#if defined(__ARM_NEON) || defined(__aarch64__)
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
    for (; j < n; j++) {
        acc += a[j] * b[j];
    }
    return acc;
#elif defined(__AVX2__)
    int j = 0;
    __m256 sum0 = _mm256_setzero_ps();
    for (; j + 7 < n; j += 8) {
        __m256 va = _mm256_loadu_ps(a + j);
        __m256 vb = _mm256_loadu_ps(b + j);
        sum0 = _mm256_fmadd_ps(va, vb, sum0);
    }
    __m128 vlow = _mm256_castps256_ps128(sum0);
    __m128 vhigh = _mm256_extractf128_ps(sum0, 1);
    __m128 vsum = _mm_add_ps(vlow, vhigh);
    vsum = _mm_hadd_ps(vsum, vsum);
    vsum = _mm_hadd_ps(vsum, vsum);
    float acc = _mm_cvtss_f32(vsum);
    for (; j < n; j++) {
        acc += a[j] * b[j];
    }
    return acc;
#else
    float acc = 0.0f;
    #pragma omp simd reduction(+:acc)
    for (int j = 0; j < n; j++) {
        acc += a[j] * b[j];
    }
    return acc;
#endif
}

void rmsnorm_forward(float* out, const float* x, const float* weight, int size, float eps = 1e-5f) {
    float sum_sq = 0.0f;
#if defined(__ARM_NEON) || defined(__aarch64__)
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
    sum_sq = vaddvq_f32(vaddq_f32(sum_vec0, sum_vec1));
    for (; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
#else
    for (int i = 0; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
#endif

    float inv_rms = 1.0f / std::sqrt(sum_sq / size + eps);

#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t inv_rms_vec = vdupq_n_f32(inv_rms);
    i = 0;
    for (; i + 7 < size; i += 8) {
        float32x4_t v0 = vld1q_f32(x + i);
        float32x4_t w0 = vld1q_f32(weight + i);
        float32x4_t v1 = vld1q_f32(x + i + 4);
        float32x4_t w1 = vld1q_f32(weight + i + 4);
        vst1q_f32(out + i, vmulq_f32(vmulq_f32(v0, inv_rms_vec), w0));
        vst1q_f32(out + i + 4, vmulq_f32(vmulq_f32(v1, inv_rms_vec), w1));
    }
    for (; i < size; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
#else
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
#endif
}

void rmsnorm_backward(
    float* dx,
    float* dweight,
    const float* dout,
    const float* x,
    const float* weight,
    int size,
    float eps = 1e-5f
) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
    float mean_sq = sum_sq / size + eps;
    float inv_rms = 1.0f / std::sqrt(mean_sq);

    float sum_dout_x = 0.0f;
    for (int i = 0; i < size; i++) {
        dweight[i] += dout[i] * x[i] * inv_rms;
        sum_dout_x += dout[i] * weight[i] * x[i];
    }

    float factor = sum_dout_x / (size * mean_sq);
    for (int i = 0; i < size; i++) {
        dx[i] = (dout[i] * weight[i] * inv_rms) - (x[i] * inv_rms * factor);
    }
}

// 4-row tiled SIMD GEMV: shares loaded x vector across 4 rows of W
void matmul_forward(float* y, const float* x, const float* W, int n, int d) {
    int i = 0;
    for (; i + 3 < d; i += 4) {
        const float* r0 = W + i * n;
        const float* r1 = W + (i + 1) * n;
        const float* r2 = W + (i + 2) * n;
        const float* r3 = W + (i + 3) * n;

#if defined(__ARM_NEON) || defined(__aarch64__)
        float32x4_t s0_0 = vdupq_n_f32(0.0f), s0_1 = vdupq_n_f32(0.0f);
        float32x4_t s1_0 = vdupq_n_f32(0.0f), s1_1 = vdupq_n_f32(0.0f);
        float32x4_t s2_0 = vdupq_n_f32(0.0f), s2_1 = vdupq_n_f32(0.0f);
        float32x4_t s3_0 = vdupq_n_f32(0.0f), s3_1 = vdupq_n_f32(0.0f);

        int j = 0;
        for (; j + 7 < n; j += 8) {
            float32x4_t vx0 = vld1q_f32(x + j);
            float32x4_t vx1 = vld1q_f32(x + j + 4);

            s0_0 = vfmaq_f32(s0_0, vld1q_f32(r0 + j), vx0);
            s0_1 = vfmaq_f32(s0_1, vld1q_f32(r0 + j + 4), vx1);

            s1_0 = vfmaq_f32(s1_0, vld1q_f32(r1 + j), vx0);
            s1_1 = vfmaq_f32(s1_1, vld1q_f32(r1 + j + 4), vx1);

            s2_0 = vfmaq_f32(s2_0, vld1q_f32(r2 + j), vx0);
            s2_1 = vfmaq_f32(s2_1, vld1q_f32(r2 + j + 4), vx1);

            s3_0 = vfmaq_f32(s3_0, vld1q_f32(r3 + j), vx0);
            s3_1 = vfmaq_f32(s3_1, vld1q_f32(r3 + j + 4), vx1);
        }
        for (; j + 3 < n; j += 4) {
            float32x4_t vx = vld1q_f32(x + j);
            s0_0 = vfmaq_f32(s0_0, vld1q_f32(r0 + j), vx);
            s1_0 = vfmaq_f32(s1_0, vld1q_f32(r1 + j), vx);
            s2_0 = vfmaq_f32(s2_0, vld1q_f32(r2 + j), vx);
            s3_0 = vfmaq_f32(s3_0, vld1q_f32(r3 + j), vx);
        }
        float acc0 = vaddvq_f32(vaddq_f32(s0_0, s0_1));
        float acc1 = vaddvq_f32(vaddq_f32(s1_0, s1_1));
        float acc2 = vaddvq_f32(vaddq_f32(s2_0, s2_1));
        float acc3 = vaddvq_f32(vaddq_f32(s3_0, s3_1));
        for (; j < n; j++) {
            float xj = x[j];
            acc0 += r0[j] * xj;
            acc1 += r1[j] * xj;
            acc2 += r2[j] * xj;
            acc3 += r3[j] * xj;
        }
        y[i] = acc0;
        y[i + 1] = acc1;
        y[i + 2] = acc2;
        y[i + 3] = acc3;
#else
        y[i] = dot_product_simd(r0, x, n);
        y[i + 1] = dot_product_simd(r1, x, n);
        y[i + 2] = dot_product_simd(r2, x, n);
        y[i + 3] = dot_product_simd(r3, x, n);
#endif
    }
    for (; i < d; i++) {
        y[i] = dot_product_simd(W + i * n, x, n);
    }
}

// Matmul backward: dW += dy @ x^T, dx = W^T @ dy
void matmul_backward(
    float* dx,
    float* dW,
    const float* dy,
    const float* x,
    const float* W,
    int n,
    int d
) {
    if (dx) {
        if ((size_t)d * n >= 32768 && n >= 16) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < n; j++) {
                float acc = 0.0f;
                for (int i = 0; i < d; i++) {
                    acc += W[i * n + j] * dy[i];
                }
                dx[j] = acc;
            }
        } else {
            for (int j = 0; j < n; j++) {
                float acc = 0.0f;
                for (int i = 0; i < d; i++) {
                    acc += W[i * n + j] * dy[i];
                }
                dx[j] = acc;
            }
        }
    }

    if (dW) {
        if ((size_t)d * n >= 32768 && d >= 16) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < d; i++) {
                float dy_i = dy[i];
                float* row = dW + i * n;
                int j = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
                float32x4_t vdy = vdupq_n_f32(dy_i);
                for (; j + 3 < n; j += 4) {
                    float32x4_t vr = vld1q_f32(row + j);
                    float32x4_t vx = vld1q_f32(x + j);
                    vr = vfmaq_f32(vr, vdy, vx);
                    vst1q_f32(row + j, vr);
                }
#endif
                for (; j < n; j++) {
                    row[j] += dy_i * x[j];
                }
            }
        } else {
            for (int i = 0; i < d; i++) {
                float dy_i = dy[i];
                float* row = dW + i * n;
                int j = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
                float32x4_t vdy = vdupq_n_f32(dy_i);
                for (; j + 3 < n; j += 4) {
                    float32x4_t vr = vld1q_f32(row + j);
                    float32x4_t vx = vld1q_f32(x + j);
                    vr = vfmaq_f32(vr, vdy, vx);
                    vst1q_f32(row + j, vr);
                }
#endif
                for (; j < n; j++) {
                    row[j] += dy_i * x[j];
                }
            }
        }
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
#if defined(__ARM_NEON) || defined(__aarch64__)
    int i = 0;
    float32x4_t max_vec = vdupq_n_f32(max_val);
    for (; i + 3 < size; i += 4) {
        max_vec = vmaxq_f32(max_vec, vld1q_f32(x + i));
    }
    max_val = vmaxvq_f32(max_vec);
    for (; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
#else
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
#endif

    float sum = 0.0f;
    for (int j = 0; j < size; j++) {
        x[j] = std::exp(x[j] - max_val);
        sum += x[j];
    }
    float inv_sum = 1.0f / (sum > 0.0f ? sum : 1e-5f);
#if defined(__ARM_NEON) || defined(__aarch64__)
    float32x4_t inv_vec = vdupq_n_f32(inv_sum);
    i = 0;
    for (; i + 3 < size; i += 4) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), inv_vec));
    }
    for (; i < size; i++) {
        x[i] *= inv_sum;
    }
#else
    for (int j = 0; j < size; j++) {
        x[j] *= inv_sum;
    }
#endif
}

} // namespace

struct LlamaCppEngine {
    LlamaCppConfig config;
    LlamaCppWeights weights;

    // KV Cache
    std::vector<float> key_cache;   // (n_layers, seq_len, kv_dim)
    std::vector<float> val_cache;   // (n_layers, seq_len, kv_dim)

    // Precomputed RoPE Tables
    std::vector<float> cos_cache;   // (seq_len, head_size / 2)
    std::vector<float> sin_cache;   // (seq_len, head_size / 2)

    // Inference Activation Buffers
    std::vector<float> x;
    std::vector<float> xb;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> att;
    std::vector<float> attn_out;
    std::vector<float> hb1;
    std::vector<float> hb3;
    std::vector<float> hb;
    std::vector<float> logits;

    // Sampling Buffers
    std::vector<float> sample_probs;
    std::vector<std::pair<float, int>> sample_candidates;

    // Training Buffers
    std::vector<float> train_step_logits;
    std::vector<float> train_dlogits;
    std::vector<float> train_probs;

    // AdamW Optimizer State
    std::vector<float> grad_buffer;
    std::vector<float> m_buffer;
    std::vector<float> v_buffer;
    int adam_step;

    // PRNG
    std::mt19937 rng;

    LlamaCppEngine(const LlamaCppConfig* cfg, LlamaCppWeights* w)
        : config(*cfg), weights(*w), adam_step(0), rng(42) {
        int head_size = config.dim / config.n_heads;
        int half = head_size / 2;
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

        key_cache.resize(config.n_layers * config.seq_len * kv_dim, 0.0f);
        val_cache.resize(config.n_layers * config.seq_len * kv_dim, 0.0f);

        x.resize(config.dim, 0.0f);
        xb.resize(config.dim, 0.0f);
        q.resize(config.dim, 0.0f);
        k.resize(kv_dim, 0.0f);
        v.resize(kv_dim, 0.0f);
        att.resize(config.n_heads * config.seq_len, 0.0f);
        attn_out.resize(config.dim, 0.0f);
        hb1.resize(config.hidden_dim, 0.0f);
        hb3.resize(config.hidden_dim, 0.0f);
        hb.resize(config.hidden_dim, 0.0f);
        logits.resize(config.vocab_size, 0.0f);

        sample_probs.resize(config.vocab_size, 0.0f);
        sample_candidates.resize(config.vocab_size);

        train_step_logits.resize(config.vocab_size, 0.0f);
        train_dlogits.resize(config.vocab_size, 0.0f);
        train_probs.resize(config.vocab_size, 0.0f);

        // Precompute RoPE cos/sin cache across all positions and half-dimensions
        cos_cache.resize(config.seq_len * half, 0.0f);
        sin_cache.resize(config.seq_len * half, 0.0f);
        for (int p_idx = 0; p_idx < config.seq_len; p_idx++) {
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / std::pow(10000.0f, (2.0f * i) / static_cast<float>(head_size));
                float val = p_idx * freq;
                cos_cache[p_idx * half + i] = std::cos(val);
                sin_cache[p_idx * half + i] = std::sin(val);
            }
        }

        size_t total_weights = calculate_total_parameters();
        grad_buffer.resize(total_weights, 0.0f);
        m_buffer.resize(total_weights, 0.0f);
        v_buffer.resize(total_weights, 0.0f);
    }

    size_t calculate_total_parameters() const {
        size_t kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
        size_t total = 0;

        total += (size_t)config.vocab_size * config.dim; // embedding
        total += (size_t)config.n_layers * config.dim;   // rms_att
        total += (size_t)config.n_layers * config.dim * config.dim; // wq
        total += (size_t)config.n_layers * kv_dim * config.dim;     // wk
        total += (size_t)config.n_layers * kv_dim * config.dim;     // wv
        total += (size_t)config.n_layers * config.dim * config.dim; // wo
        total += (size_t)config.n_layers * config.dim;              // rms_ffn
        total += (size_t)config.n_layers * config.hidden_dim * config.dim; // w1
        total += (size_t)config.n_layers * config.dim * config.hidden_dim; // w2
        total += (size_t)config.n_layers * config.hidden_dim * config.dim; // w3
        total += (size_t)config.dim;                                      // rms_final
        if (!weights.shared_classifier && weights.wcls != weights.token_embedding_table) {
            total += (size_t)config.vocab_size * config.dim;
        }
        return total;
    }

    void reset_kv_cache() {
        std::fill(key_cache.begin(), key_cache.end(), 0.0f);
        std::fill(val_cache.begin(), val_cache.end(), 0.0f);
    }
};

extern "C" {

LlamaCppEngine* llama_create(const LlamaCppConfig* config, LlamaCppWeights* weights) {
    if (!config || !weights) return nullptr;
    return new LlamaCppEngine(config, weights);
}

void llama_free(LlamaCppEngine* engine) {
    if (engine) delete engine;
}

void llama_reset_cache(LlamaCppEngine* engine) {
    if (engine) engine->reset_kv_cache();
}

int llama_get_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

void llama_set_threads(int num_threads) {
#ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
#endif
}

void llama_forward(LlamaCppEngine* engine, int token, int pos, float* out_logits) {
    if (!engine || token < 0 || token >= engine->config.vocab_size) return;
    if (pos >= engine->config.seq_len) return;

    const LlamaCppConfig& p = engine->config;
    const LlamaCppWeights& w = engine->weights;
    int head_size = p.dim / p.n_heads;
    int half = head_size / 2;
    int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    int kv_mul = p.n_heads / p.n_kv_heads;

    // 1. Token embedding lookup
    const float* emb_row = w.token_embedding_table + token * p.dim;
    std::memcpy(engine->x.data(), emb_row, p.dim * sizeof(float));

    // Pointer to precalculated RoPE cache for this position
    const float* cos_ptr = engine->cos_cache.data() + pos * half;
    const float* sin_ptr = engine->sin_cache.data() + pos * half;
    float inv_sqrt_head = 1.0f / std::sqrt(static_cast<float>(head_size));

    for (int l = 0; l < p.n_layers; l++) {
        // Pre-attention RMSNorm
        rmsnorm_forward(engine->xb.data(), engine->x.data(), w.rms_att_weight + l * p.dim, p.dim);

        // Q, K, V projections
        matmul_forward(engine->q.data(), engine->xb.data(), w.wq + l * p.dim * p.dim, p.dim, p.dim);
        matmul_forward(engine->k.data(), engine->xb.data(), w.wk + l * kv_dim * p.dim, p.dim, kv_dim);
        matmul_forward(engine->v.data(), engine->xb.data(), w.wv + l * kv_dim * p.dim, p.dim, kv_dim);

        // RoPE relative positional encoding (precomputed sin/cos)
        if (p.rope_type == 1) {
            // HuggingFace split-half RoPE: [q_first_half, q_second_half]
            for (int h = 0; h < p.n_heads; h++) {
                float* qh = engine->q.data() + h * head_size;
                int i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
                for (; i + 3 < half; i += 4) {
                    float32x4_t fcr = vld1q_f32(cos_ptr + i);
                    float32x4_t fci = vld1q_f32(sin_ptr + i);
                    float32x4_t q0 = vld1q_f32(qh + i);
                    float32x4_t q1 = vld1q_f32(qh + i + half);
                    vst1q_f32(qh + i, vsubq_f32(vmulq_f32(q0, fcr), vmulq_f32(q1, fci)));
                    vst1q_f32(qh + i + half, vaddq_f32(vmulq_f32(q1, fcr), vmulq_f32(q0, fci)));
                }
#endif
                for (; i < half; i++) {
                    float fcr = cos_ptr[i];
                    float fci = sin_ptr[i];
                    float q0 = qh[i];
                    float q1 = qh[i + half];
                    qh[i] = q0 * fcr - q1 * fci;
                    qh[i + half] = q1 * fcr + q0 * fci;
                }
            }
            for (int h = 0; h < p.n_kv_heads; h++) {
                float* kh = engine->k.data() + h * head_size;
                int i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
                for (; i + 3 < half; i += 4) {
                    float32x4_t fcr = vld1q_f32(cos_ptr + i);
                    float32x4_t fci = vld1q_f32(sin_ptr + i);
                    float32x4_t k0 = vld1q_f32(kh + i);
                    float32x4_t k1 = vld1q_f32(kh + i + half);
                    vst1q_f32(kh + i, vsubq_f32(vmulq_f32(k0, fcr), vmulq_f32(k1, fci)));
                    vst1q_f32(kh + i + half, vaddq_f32(vmulq_f32(k1, fcr), vmulq_f32(k0, fci)));
                }
#endif
                for (; i < half; i++) {
                    float fcr = cos_ptr[i];
                    float fci = sin_ptr[i];
                    float k0 = kh[i];
                    float k1 = kh[i + half];
                    kh[i] = k0 * fcr - k1 * fci;
                    kh[i + half] = k1 * fcr + k0 * fci;
                }
            }
        } else {
            // Standard llama2.c interleaved RoPE: [q0, q1, q2, q3, ...]
            for (int i = 0; i < p.dim; i += 2) {
                int h_dim = (i % head_size) / 2;
                float fcr = cos_ptr[h_dim];
                float fci = sin_ptr[h_dim];

                float q0 = engine->q[i];
                float q1 = engine->q[i + 1];
                engine->q[i] = q0 * fcr - q1 * fci;
                engine->q[i + 1] = q0 * fci + q1 * fcr;

                if (i < kv_dim) {
                    float k0 = engine->k[i];
                    float k1 = engine->k[i + 1];
                    engine->k[i] = k0 * fcr - k1 * fci;
                    engine->k[i + 1] = k0 * fci + k1 * fcr;
                }
            }
        }

        // Cache Key and Value at current position
        int loff = l * p.seq_len * kv_dim;
        std::memcpy(engine->key_cache.data() + loff + pos * kv_dim, engine->k.data(), kv_dim * sizeof(float));
        std::memcpy(engine->val_cache.data() + loff + pos * kv_dim, engine->v.data(), kv_dim * sizeof(float));

        // Multi-head attention into preallocated attn_out buffer
        std::fill(engine->attn_out.begin(), engine->attn_out.end(), 0.0f);

        for (int h = 0; h < p.n_heads; h++) {
            const float* q_head = engine->q.data() + h * head_size;
            float* att_head = engine->att.data() + h * p.seq_len;
            int kv_h = h / kv_mul;

            for (int t = 0; t <= pos; t++) {
                const float* k_past = engine->key_cache.data() + loff + t * kv_dim + kv_h * head_size;
                att_head[t] = dot_product_simd(q_head, k_past, head_size) * inv_sqrt_head;
            }

            softmax(att_head, pos + 1);

            float* out_head = engine->attn_out.data() + h * head_size;
            for (int t = 0; t <= pos; t++) {
                const float* v_past = engine->val_cache.data() + loff + t * kv_dim + kv_h * head_size;
                float a = att_head[t];
                int d = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
                float32x4_t va = vdupq_n_f32(a);
                for (; d + 3 < head_size; d += 4) {
                    float32x4_t vout = vld1q_f32(out_head + d);
                    float32x4_t vv = vld1q_f32(v_past + d);
                    vout = vfmaq_f32(vout, vv, va);
                    vst1q_f32(out_head + d, vout);
                }
#endif
                for (; d < head_size; d++) {
                    out_head[d] += a * v_past[d];
                }
            }
        }

        // Attention output projection: wo @ attn_out
        matmul_forward(engine->xb.data(), engine->attn_out.data(), w.wo + l * p.dim * p.dim, p.dim, p.dim);
        int i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
        for (; i + 3 < p.dim; i += 4) {
            float32x4_t vx = vld1q_f32(engine->x.data() + i);
            float32x4_t vxb = vld1q_f32(engine->xb.data() + i);
            vst1q_f32(engine->x.data() + i, vaddq_f32(vx, vxb));
        }
#endif
        for (; i < p.dim; i++) {
            engine->x[i] += engine->xb[i];
        }

        // Pre-FFN RMSNorm
        rmsnorm_forward(engine->xb.data(), engine->x.data(), w.rms_ffn_weight + l * p.dim, p.dim);

        // SwiGLU FFN: w2 @ (silu(w1 @ xb) * (w3 @ xb))
        matmul_forward(engine->hb1.data(), engine->xb.data(), w.w1 + l * p.hidden_dim * p.dim, p.dim, p.hidden_dim);
        matmul_forward(engine->hb3.data(), engine->xb.data(), w.w3 + l * p.hidden_dim * p.dim, p.dim, p.hidden_dim);

        for (int j = 0; j < p.hidden_dim; j++) {
            engine->hb[j] = silu(engine->hb1[j]) * engine->hb3[j];
        }

        matmul_forward(engine->xb.data(), engine->hb.data(), w.w2 + l * p.dim * p.hidden_dim, p.hidden_dim, p.dim);
        i = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
        for (; i + 3 < p.dim; i += 4) {
            float32x4_t vx = vld1q_f32(engine->x.data() + i);
            float32x4_t vxb = vld1q_f32(engine->xb.data() + i);
            vst1q_f32(engine->x.data() + i, vaddq_f32(vx, vxb));
        }
#endif
        for (; i < p.dim; i++) {
            engine->x[i] += engine->xb[i];
        }
    }

    // Final RMSNorm
    rmsnorm_forward(engine->x.data(), engine->x.data(), w.rms_final_weight, p.dim);

    // Classifier projection into logits
    const float* cls_w = w.wcls ? w.wcls : w.token_embedding_table;
    matmul_forward(engine->logits.data(), engine->x.data(), cls_w, p.dim, p.vocab_size);

    if (out_logits) {
        std::memcpy(out_logits, engine->logits.data(), p.vocab_size * sizeof(float));
    }
}

int llama_sample_token(LlamaCppEngine* engine, float temperature, float top_p) {
    if (!engine) return 0;
    const int vocab_size = engine->config.vocab_size;

    if (temperature <= 0.0f) {
        // Fast greedy argmax
        int best_i = 0;
        float best_v = engine->logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (engine->logits[i] > best_v) {
                best_v = engine->logits[i];
                best_i = i;
            }
        }
        return best_i;
    }

    std::memcpy(engine->sample_probs.data(), engine->logits.data(), vocab_size * sizeof(float));
    float inv_temp = 1.0f / temperature;
    for (int i = 0; i < vocab_size; i++) {
        engine->sample_probs[i] *= inv_temp;
    }
    softmax(engine->sample_probs.data(), vocab_size);

    if (top_p < 1.0f) {
        for (int i = 0; i < vocab_size; i++) {
            engine->sample_candidates[i] = {engine->sample_probs[i], i};
        }

        int K = std::min(vocab_size, 64);
        std::partial_sort(
            engine->sample_candidates.begin(),
            engine->sample_candidates.begin() + K,
            engine->sample_candidates.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; }
        );

        float cumsum = 0.0f;
        int cutoff_idx = K;
        for (int i = 0; i < K; i++) {
            cumsum += engine->sample_candidates[i].first;
            if (cumsum > top_p && i > 0) {
                cutoff_idx = i + 1;
                break;
            }
        }

        if (cumsum < top_p && K < vocab_size) {
            std::sort(
                engine->sample_candidates.begin() + K,
                engine->sample_candidates.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; }
            );
            for (int i = K; i < vocab_size; i++) {
                cumsum += engine->sample_candidates[i].first;
                if (cumsum > top_p && i > 0) {
                    cutoff_idx = i + 1;
                    break;
                }
            }
        }

        float renorm_sum = 0.0f;
        for (int i = 0; i < cutoff_idx; i++) {
            renorm_sum += engine->sample_candidates[i].first;
        }

        std::uniform_real_distribution<float> dist(0.0f, renorm_sum);
        float r = dist(engine->rng);
        float acc = 0.0f;
        for (int i = 0; i < cutoff_idx; i++) {
            acc += engine->sample_candidates[i].first;
            if (r <= acc) {
                return engine->sample_candidates[i].second;
            }
        }
        return engine->sample_candidates[0].second;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(engine->rng);
    float acc = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        acc += engine->sample_probs[i];
        if (r <= acc) return i;
    }
    return vocab_size - 1;
}

int llama_generate(
    LlamaCppEngine* engine,
    const int* prompt_tokens,
    int prompt_len,
    int max_new_tokens,
    float temperature,
    float top_p,
    int* out_tokens
) {
    if (!engine || !prompt_tokens || prompt_len <= 0 || !out_tokens) return 0;

    engine->reset_kv_cache();

    int pos = 0;
    for (int i = 0; i < prompt_len; i++) {
        llama_forward(engine, prompt_tokens[i], pos, nullptr);
        pos++;
    }

    int generated_count = 0;
    for (int step = 0; step < max_new_tokens; step++) {
        if (pos >= engine->config.seq_len - 1) break;

        int next_token = llama_sample_token(engine, temperature, top_p);
        out_tokens[generated_count++] = next_token;

        if (next_token == 2) break; // EOS token

        llama_forward(engine, next_token, pos, nullptr);
        pos++;
    }

    return generated_count;
}

float llama_train_step(
    LlamaCppEngine* engine,
    const int* input_tokens,
    const int* target_tokens,
    int seq_len,
    float lr,
    float weight_decay,
    float beta1,
    float beta2,
    float eps
) {
    if (!engine || !input_tokens || !target_tokens || seq_len <= 0) return 0.0f;

    const LlamaCppConfig& p = engine->config;
    LlamaCppWeights& w = engine->weights;
    float total_loss = 0.0f;

    // Reset KV cache for fresh training sequence
    engine->reset_kv_cache();

    // Zero gradient buffer
    std::fill(engine->grad_buffer.begin(), engine->grad_buffer.end(), 0.0f);

    float* step_logits = engine->train_step_logits.data();
    float* dlogits = engine->train_dlogits.data();
    float* probs = engine->train_probs.data();

    for (int pos = 0; pos < seq_len; pos++) {
        int in_tok = input_tokens[pos];
        int target_tok = target_tokens[pos];

        llama_forward(engine, in_tok, pos, step_logits);

        // Softmax & Cross-Entropy Loss
        std::memcpy(probs, step_logits, p.vocab_size * sizeof(float));
        softmax(probs, p.vocab_size);

        float target_prob = std::max(1e-12f, probs[target_tok]);
        total_loss += -std::log(target_prob);

        // dL/dLogits = (probs - 1_{target}) / seq_len
        float inv_seq = 1.0f / seq_len;
        for (int i = 0; i < p.vocab_size; i++) {
            dlogits[i] = (probs[i] - (i == target_tok ? 1.0f : 0.0f)) * inv_seq;
        }

        // Gradient backprop through classifier into embeddings
        const float* cls_w = w.wcls ? w.wcls : w.token_embedding_table;
        float* d_emb_row = w.token_embedding_table + in_tok * p.dim;

        // Fast vector-matrix product: g = cls_w^T @ dlogits
        // Outer loop over active v, inner loop over d (contiguous memory & SIMD vectorized)
        std::vector<float> g(p.dim, 0.0f);
        for (int v = 0; v < p.vocab_size; v++) {
            float dv = dlogits[v];
            if (v != target_tok && std::abs(dv) < 1e-5f) continue;
            const float* cls_row = cls_w + v * p.dim;
            for (int d = 0; d < p.dim; d++) {
                g[d] += dv * cls_row[d];
            }
        }
        for (int d = 0; d < p.dim; d++) {
            d_emb_row[d] -= lr * (g[d] + weight_decay * d_emb_row[d]);
        }

        // Gradient update for classifier weights:
        // Logits = cls_w @ x, so dL / d(cls_w[v, d]) = dlogits[v] * x[d]
        float* cls_base = (w.wcls ? w.wcls : w.token_embedding_table);
        const float* x_vec = engine->x.data();

        // 1. Target token classifier row (increases target logit)
        float* cls_target = cls_base + target_tok * p.dim;
        float d_target = dlogits[target_tok];
        for (int d = 0; d < p.dim; d++) {
            cls_target[d] -= lr * (d_target * x_vec[d] + weight_decay * cls_target[d]);
        }

        // 2. Competing tokens classifier rows (decreases competing logits)
        if (p.vocab_size <= 1024) {
            for (int v = 0; v < p.vocab_size; v++) {
                if (v == target_tok) continue;
                float* cls_row = cls_base + v * p.dim;
                float dv = dlogits[v];
                for (int d = 0; d < p.dim; d++) {
                    cls_row[d] -= lr * (dv * x_vec[d] + weight_decay * cls_row[d]);
                }
            }
        } else {
            for (int v = 0; v < p.vocab_size; v++) {
                if (v == target_tok || probs[v] < 0.005f) continue;
                float* cls_row = cls_base + v * p.dim;
                float dv = dlogits[v];
                for (int d = 0; d < p.dim; d++) {
                    cls_row[d] -= lr * (dv * x_vec[d] + weight_decay * cls_row[d]);
                }
            }
        }
    }

    engine->adam_step++;
    return total_loss / seq_len;
}

} // extern "C"
