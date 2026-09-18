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
#ifdef _OPENMP
    if ((size_t)d * n >= 16384 && d >= 16) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < d; i++) {
            y[i] = dot_product_simd(W + i * n, x, n);
        }
        return;
    }
#endif
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


// Sequence activations used by the native full-transformer trainer. The
// workspace is retained by the engine and resized only when the training
// sequence shape changes, avoiding per-step heap churn.
struct FullTrainWorkspace {
    int seq = 0;
    int layers = 0;
    int dim = 0;
    int kv_dim = 0;
    int heads = 0;
    int hidden = 0;

    std::vector<float> states;       // (layers + 1, seq, dim)
    std::vector<float> norm_att;     // (layers, seq, dim)
    std::vector<float> q;            // (layers, seq, dim)
    std::vector<float> k;            // (layers, seq, kv_dim)
    std::vector<float> v;            // (layers, seq, kv_dim)
    std::vector<float> scores;       // (layers, seq, heads, seq)
    std::vector<float> attn_out;     // (layers, seq, dim)
    std::vector<float> attn_state;   // (layers, seq, dim)
    std::vector<float> norm_ffn;     // (layers, seq, dim)
    std::vector<float> gate;         // (layers, seq, hidden)
    std::vector<float> up;           // (layers, seq, hidden)
    std::vector<float> swiglu;       // (layers, seq, hidden)
    std::vector<float> final_norm;   // (seq, dim)
    std::vector<float> d_final;      // (seq, dim), gradient of final norm

    std::vector<float> d_states;
    std::vector<float> d_norm_att;
    std::vector<float> d_q;
    std::vector<float> d_k;
    std::vector<float> d_v;
    std::vector<float> d_norm_ffn;
    std::vector<float> d_gate;
    std::vector<float> d_up;
    std::vector<float> d_swiglu;
    std::vector<float> d_attn_out;
    std::vector<float> d_attn_state;

    void resize(int new_seq, int new_layers, int new_dim, int new_kv_dim,
                int new_heads, int new_hidden) {
        if (seq == new_seq && layers == new_layers && dim == new_dim &&
            kv_dim == new_kv_dim && heads == new_heads && hidden == new_hidden) {
            return;
        }
        seq = new_seq;
        layers = new_layers;
        dim = new_dim;
        kv_dim = new_kv_dim;
        heads = new_heads;
        hidden = new_hidden;
        const size_t layer_dim = static_cast<size_t>(layers) * seq * dim;
        const size_t layer_kv = static_cast<size_t>(layers) * seq * kv_dim;
        const size_t layer_hidden = static_cast<size_t>(layers) * seq * hidden;
        const size_t layer_scores = static_cast<size_t>(layers) * seq * heads * seq;
        states.assign(static_cast<size_t>(layers + 1) * seq * dim, 0.0f);
        norm_att.assign(layer_dim, 0.0f);
        q.assign(layer_dim, 0.0f);
        k.assign(layer_kv, 0.0f);
        v.assign(layer_kv, 0.0f);
        scores.assign(layer_scores, 0.0f);
        attn_out.assign(layer_dim, 0.0f);
        attn_state.assign(layer_dim, 0.0f);
        norm_ffn.assign(layer_dim, 0.0f);
        gate.assign(layer_hidden, 0.0f);
        up.assign(layer_hidden, 0.0f);
        swiglu.assign(layer_hidden, 0.0f);
        final_norm.assign(static_cast<size_t>(seq) * dim, 0.0f);
        d_final.assign(static_cast<size_t>(seq) * dim, 0.0f);
        d_states.assign(states.size(), 0.0f);
        d_norm_att.assign(layer_dim, 0.0f);
        d_q.assign(layer_dim, 0.0f);
        d_k.assign(layer_kv, 0.0f);
        d_v.assign(layer_kv, 0.0f);
        d_norm_ffn.assign(layer_dim, 0.0f);
        d_gate.assign(layer_hidden, 0.0f);
        d_up.assign(layer_hidden, 0.0f);
        d_swiglu.assign(layer_hidden, 0.0f);
        d_attn_out.assign(layer_dim, 0.0f);
        d_attn_state.assign(layer_dim, 0.0f);
    }
};

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
    std::vector<float> train_grad_embedding;

    // Native full-transformer training workspace
    FullTrainWorkspace full_workspace;

    // AdamW Optimizer State
    std::vector<float> grad_buffer;
    std::vector<float> m_buffer;
    std::vector<float> v_buffer;
    std::vector<float> m_cls;
    std::vector<float> v_cls;
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
        train_grad_embedding.resize(config.dim, 0.0f);

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
    if (
        !engine || !prompt_tokens || prompt_len <= 0 || !out_tokens
        || prompt_len >= engine->config.seq_len || max_new_tokens < 0
    ) return 0;

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

float llama_full_train_step(
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
    if (!engine || !input_tokens || !target_tokens || seq_len <= 0 ||
        seq_len >= engine->config.seq_len) return 0.0f;

    const LlamaCppConfig& p = engine->config;
    LlamaCppWeights& w = engine->weights;
    const int D = p.dim;
    const int H = p.n_heads;
    const int HD = D / H;
    const int KV = (D * p.n_kv_heads) / H;
    const int Hidden = p.hidden_dim;
    const int L = p.n_layers;
    const float inv_head = 1.0f / std::sqrt(static_cast<float>(HD));
    FullTrainWorkspace& ws = engine->full_workspace;
    ws.resize(seq_len, L, D, KV, H, Hidden);

    for (int t = 0; t < seq_len; t++) {
        if (input_tokens[t] < 0 || input_tokens[t] >= p.vocab_size ||
            target_tokens[t] < 0 || target_tokens[t] >= p.vocab_size) return 0.0f;
        std::memcpy(ws.states.data() + static_cast<size_t>(t) * D,
                    w.token_embedding_table + input_tokens[t] * D,
                    D * sizeof(float));
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

    // Parameter-gradient offsets match calculate_total_parameters().
    size_t off_emb = 0;
    size_t off_rms_att = off_emb + static_cast<size_t>(p.vocab_size) * D;
    size_t off_wq = off_rms_att + static_cast<size_t>(L) * D;
    size_t off_wk = off_wq + static_cast<size_t>(L) * D * D;
    size_t off_wv = off_wk + static_cast<size_t>(L) * KV * D;
    size_t off_wo = off_wv + static_cast<size_t>(L) * KV * D;
    size_t off_rms_ffn = off_wo + static_cast<size_t>(L) * D * D;
    size_t off_w1 = off_rms_ffn + static_cast<size_t>(L) * D;
    size_t off_w2 = off_w1 + static_cast<size_t>(L) * Hidden * D;
    size_t off_w3 = off_w2 + static_cast<size_t>(L) * D * Hidden;
    size_t off_rms_final = off_w3 + static_cast<size_t>(L) * Hidden * D;
    size_t off_wcls = off_rms_final + D;
    float* grad = engine->grad_buffer.data();

    auto state_at = [&](int layer, int t) -> float* {
        return ws.states.data() + (static_cast<size_t>(layer) * seq_len + t) * D;
    };
    auto layer_dim_at = [&](std::vector<float>& data, int layer, int t) -> float* {
        return data.data() + (static_cast<size_t>(layer) * seq_len + t) * D;
    };
    auto layer_kv_at = [&](std::vector<float>& data, int layer, int t) -> float* {
        return data.data() + (static_cast<size_t>(layer) * seq_len + t) * KV;
    };
    auto layer_hidden_at = [&](std::vector<float>& data, int layer, int t) -> float* {
        return data.data() + (static_cast<size_t>(layer) * seq_len + t) * Hidden;
    };
    auto score_at = [&](int layer, int t, int h) -> float* {
        return ws.scores.data() +
            ((static_cast<size_t>(layer) * seq_len + t) * H + h) * seq_len;
    };

    // Forward pass over the complete causal sequence, retaining activations.
    for (int l = 0; l < L; l++) {
        for (int t = 0; t < seq_len; t++) {
            float* x = state_at(l, t);
            float* norm = layer_dim_at(ws.norm_att, l, t);
            rmsnorm_forward(norm, x, w.rms_att_weight + l * D, D);
            matmul_forward(layer_dim_at(ws.q, l, t), norm, w.wq + static_cast<size_t>(l) * D * D, D, D);
            matmul_forward(layer_kv_at(ws.k, l, t), norm, w.wk + static_cast<size_t>(l) * KV * D, D, KV);
            matmul_forward(layer_kv_at(ws.v, l, t), norm, w.wv + static_cast<size_t>(l) * KV * D, D, KV);

            float* q = layer_dim_at(ws.q, l, t);
            float* k = layer_kv_at(ws.k, l, t);
            const float* cos_ptr = engine->cos_cache.data() + t * (HD / 2);
            const float* sin_ptr = engine->sin_cache.data() + t * (HD / 2);
            if (p.rope_type == 1) {
                const int half = HD / 2;
                for (int h = 0; h < H; h++) {
                    float* qh = q + h * HD;
                    for (int i = 0; i < half; i++) {
                        float q0 = qh[i], q1 = qh[i + half];
                        qh[i] = q0 * cos_ptr[i] - q1 * sin_ptr[i];
                        qh[i + half] = q1 * cos_ptr[i] + q0 * sin_ptr[i];
                    }
                }
                for (int h = 0; h < p.n_kv_heads; h++) {
                    float* kh = k + h * HD;
                    for (int i = 0; i < half; i++) {
                        float k0 = kh[i], k1 = kh[i + half];
                        kh[i] = k0 * cos_ptr[i] - k1 * sin_ptr[i];
                        kh[i + half] = k1 * cos_ptr[i] + k0 * sin_ptr[i];
                    }
                }
            } else {
                for (int i = 0; i < D; i += 2) {
                    int r = (i % HD) / 2;
                    float q0 = q[i], q1 = q[i + 1];
                    q[i] = q0 * cos_ptr[r] - q1 * sin_ptr[r];
                    q[i + 1] = q0 * sin_ptr[r] + q1 * cos_ptr[r];
                    if (i < KV) {
                        float k0 = k[i], k1 = k[i + 1];
                        k[i] = k0 * cos_ptr[r] - k1 * sin_ptr[r];
                        k[i + 1] = k0 * sin_ptr[r] + k1 * cos_ptr[r];
                    }
                }
            }
        }

        for (int t = 0; t < seq_len; t++) {
            float* out = layer_dim_at(ws.attn_out, l, t);
            std::fill(out, out + D, 0.0f);
            const int kv_mul = H / p.n_kv_heads;
            for (int h = 0; h < H; h++) {
                float* weights = score_at(l, t, h);
                const float* qh = layer_dim_at(ws.q, l, t) + h * HD;
                const int kv_h = h / kv_mul;
                float max_score = -1e30f;
                for (int u = 0; u <= t; u++) {
                    const float* kh = layer_kv_at(ws.k, l, u) + kv_h * HD;
                    weights[u] = dot_product_simd(qh, kh, HD) * inv_head;
                    if (weights[u] > max_score) max_score = weights[u];
                }
                float sum = 0.0f;
                for (int u = 0; u <= t; u++) {
                    weights[u] = std::exp(weights[u] - max_score);
                    sum += weights[u];
                }
                float inv_sum = 1.0f / std::max(sum, 1e-20f);
                for (int u = 0; u < seq_len; u++) weights[u] = u <= t ? weights[u] * inv_sum : 0.0f;
                float* out_head = out + h * HD;
                for (int u = 0; u <= t; u++) {
                    const float* vh = layer_kv_at(ws.v, l, u) + kv_h * HD;
                    for (int d = 0; d < HD; d++) out_head[d] += weights[u] * vh[d];
                }
            }
            float* attn_state = layer_dim_at(ws.attn_state, l, t);
            matmul_forward(engine->xb.data(), out, w.wo + static_cast<size_t>(l) * D * D, D, D);
            const float* x = state_at(l, t);
            for (int d = 0; d < D; d++) attn_state[d] = x[d] + engine->xb[d];

            float* norm_ffn = layer_dim_at(ws.norm_ffn, l, t);
            rmsnorm_forward(norm_ffn, attn_state, w.rms_ffn_weight + l * D, D);
            float* gate = layer_hidden_at(ws.gate, l, t);
            float* up = layer_hidden_at(ws.up, l, t);
            float* swiglu = layer_hidden_at(ws.swiglu, l, t);
            matmul_forward(gate, norm_ffn, w.w1 + static_cast<size_t>(l) * Hidden * D, D, Hidden);
            matmul_forward(up, norm_ffn, w.w3 + static_cast<size_t>(l) * Hidden * D, D, Hidden);
            for (int j = 0; j < Hidden; j++) swiglu[j] = silu(gate[j]) * up[j];
            float* next = state_at(l + 1, t);
            matmul_forward(engine->xb.data(), swiglu, w.w2 + static_cast<size_t>(l) * D * Hidden, Hidden, D);
            for (int d = 0; d < D; d++) next[d] = attn_state[d] + engine->xb[d];
        }
    }

    // Final norm, classifier cross-entropy, and output-side gradients.
    float total_loss = 0.0f;
    std::fill(ws.final_norm.begin(), ws.final_norm.end(), 0.0f);
    const float inv_seq = 1.0f / seq_len;
    const float* cls_w = w.wcls ? w.wcls : w.token_embedding_table;
    for (int t = 0; t < seq_len; t++) {
        float* final = ws.final_norm.data() + t * D;
        rmsnorm_forward(final, state_at(L, t), w.rms_final_weight, D);
        float max_logit = -1e30f;
        for (int v = 0; v < p.vocab_size; v++) {
            float value = dot_product_simd(cls_w + static_cast<size_t>(v) * D, final, D);
            engine->train_step_logits[v] = value;
            if (value > max_logit) max_logit = value;
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < p.vocab_size; v++) {
            engine->train_probs[v] = std::exp(engine->train_step_logits[v] - max_logit);
            sum_exp += engine->train_probs[v];
        }
        float inv_sum = 1.0f / std::max(sum_exp, 1e-20f);
        int target = target_tokens[t];
        float target_prob = std::max(engine->train_probs[target] * inv_sum, 1e-20f);
        total_loss -= std::log(target_prob);
        float* d_final = ws.d_final.data() + static_cast<size_t>(t) * D;
        std::fill(d_final, d_final + D, 0.0f);
        for (int v = 0; v < p.vocab_size; v++) {
            float dlogit = (engine->train_probs[v] * inv_sum - (v == target ? 1.0f : 0.0f)) * inv_seq;
            float* dcls = grad + (w.shared_classifier ? off_emb : off_wcls) + static_cast<size_t>(v) * D;
            const float* cls_row = cls_w + static_cast<size_t>(v) * D;
            for (int d = 0; d < D; d++) {
                dcls[d] += dlogit * final[d];
                d_final[d] += dlogit * cls_row[d];
            }
        }
    }

    // Reverse final RMSNorm and every decoder block.
    for (int t = 0; t < seq_len; t++) {
        float* d_final = ws.d_final.data() + static_cast<size_t>(t) * D;
        float* dx_final = ws.d_states.data() + (static_cast<size_t>(L) * seq_len + t) * D;
        rmsnorm_backward(dx_final, grad + off_rms_final, d_final,
                         state_at(L, t), w.rms_final_weight, D);
    }

    for (int l = L - 1; l >= 0; l--) {
        const size_t wq_off = off_wq + static_cast<size_t>(l) * D * D;
        const size_t wk_off = off_wk + static_cast<size_t>(l) * KV * D;
        const size_t wv_off = off_wv + static_cast<size_t>(l) * KV * D;
        const size_t wo_off = off_wo + static_cast<size_t>(l) * D * D;
        const size_t w1_off = off_w1 + static_cast<size_t>(l) * Hidden * D;
        const size_t w2_off = off_w2 + static_cast<size_t>(l) * D * Hidden;
        const size_t w3_off = off_w3 + static_cast<size_t>(l) * Hidden * D;
        std::fill(ws.d_q.begin() + static_cast<size_t>(l) * seq_len * D,
                  ws.d_q.begin() + static_cast<size_t>(l + 1) * seq_len * D, 0.0f);
        std::fill(ws.d_k.begin() + static_cast<size_t>(l) * seq_len * KV,
                  ws.d_k.begin() + static_cast<size_t>(l + 1) * seq_len * KV, 0.0f);
        std::fill(ws.d_v.begin() + static_cast<size_t>(l) * seq_len * KV,
                  ws.d_v.begin() + static_cast<size_t>(l + 1) * seq_len * KV, 0.0f);

        // FFN and attention output projection backward.
        for (int t = 0; t < seq_len; t++) {
            float* d_out = ws.d_states.data() + (static_cast<size_t>(l + 1) * seq_len + t) * D;
            float* d_attn_state = layer_dim_at(ws.d_attn_state, l, t);
            float* d_norm_ffn = layer_dim_at(ws.d_norm_ffn, l, t);
            float* d_gate = layer_hidden_at(ws.d_gate, l, t);
            float* d_up = layer_hidden_at(ws.d_up, l, t);
            float* d_swiglu = layer_hidden_at(ws.d_swiglu, l, t);
            const float* gate = layer_hidden_at(ws.gate, l, t);
            const float* up = layer_hidden_at(ws.up, l, t);
            const float* norm_ffn = layer_dim_at(ws.norm_ffn, l, t);
            const float* attn_state = layer_dim_at(ws.attn_state, l, t);
            std::fill(d_norm_ffn, d_norm_ffn + D, 0.0f);
            std::fill(d_gate, d_gate + Hidden, 0.0f);
            std::fill(d_up, d_up + Hidden, 0.0f);
            matmul_backward(d_swiglu, grad + w2_off, d_out,
                            layer_hidden_at(ws.swiglu, l, t), w.w2 + static_cast<size_t>(l) * D * Hidden,
                            Hidden, D);
            for (int j = 0; j < Hidden; j++) {
                float s = silu(gate[j]);
                d_gate[j] = d_swiglu[j] * up[j] * silu_deriv(gate[j]);
                d_up[j] = d_swiglu[j] * s;
            }
            matmul_backward(d_norm_ffn, grad + w1_off, d_gate, norm_ffn,
                            w.w1 + static_cast<size_t>(l) * Hidden * D, D, Hidden);
            matmul_backward(engine->xb.data(), grad + w3_off, d_up, norm_ffn,
                            w.w3 + static_cast<size_t>(l) * Hidden * D, D, Hidden);
            for (int d = 0; d < D; d++) d_norm_ffn[d] += engine->xb[d];
            rmsnorm_backward(d_attn_state, grad + off_rms_ffn + static_cast<size_t>(l) * D,
                             d_norm_ffn, attn_state, w.rms_ffn_weight + l * D, D);
            for (int d = 0; d < D; d++) d_attn_state[d] += d_out[d];

            float* d_attn_out = layer_dim_at(ws.d_attn_out, l, t);
            matmul_backward(d_attn_out, grad + wo_off, d_attn_state,
                            layer_dim_at(ws.attn_out, l, t), w.wo + static_cast<size_t>(l) * D * D,
                            D, D);
            float* d_input = ws.d_states.data() + (static_cast<size_t>(l) * seq_len + t) * D;
            for (int d = 0; d < D; d++) d_input[d] += d_attn_state[d];
        }

        // Attention backward and projection gradients.
        const int kv_mul = H / p.n_kv_heads;
        for (int t = 0; t < seq_len; t++) {
            for (int h = 0; h < H; h++) {
                float* weights = score_at(l, t, h);
                const float* dcontext = layer_dim_at(ws.d_attn_out, l, t) + h * HD;
                const int kv_h = h / kv_mul;
                float weighted = 0.0f;
                for (int u = 0; u <= t; u++) {
                    weighted += weights[u] * dot_product_simd(dcontext, layer_kv_at(ws.v, l, u) + kv_h * HD, HD);
                }
                for (int u = 0; u <= t; u++) {
                    float dscore = weights[u] * (dot_product_simd(dcontext, layer_kv_at(ws.v, l, u) + kv_h * HD, HD) - weighted);
                    float* dqh = layer_dim_at(ws.d_q, l, t) + h * HD;
                    float* dkh = layer_kv_at(ws.d_k, l, u) + kv_h * HD;
                    const float* kh = layer_kv_at(ws.k, l, u) + kv_h * HD;
                    const float* qh = layer_dim_at(ws.q, l, t) + h * HD;
                    float* dvh = layer_kv_at(ws.d_v, l, u) + kv_h * HD;
                    const float* vh = layer_kv_at(ws.v, l, u) + kv_h * HD;
                    for (int d = 0; d < HD; d++) {
                        dqh[d] += dscore * kh[d] * inv_head;
                        dkh[d] += dscore * qh[d] * inv_head;
                        dvh[d] += weights[u] * dcontext[d];
                    }
                    (void)vh;
                }
            }
        }

        // Backpropagate through RoPE and Q/K/V projections.
        for (int t = 0; t < seq_len; t++) {
            float* dq = layer_dim_at(ws.d_q, l, t);
            float* dk = layer_kv_at(ws.d_k, l, t);
            const float* cos_ptr = engine->cos_cache.data() + t * (HD / 2);
            const float* sin_ptr = engine->sin_cache.data() + t * (HD / 2);
            if (p.rope_type == 1) {
                const int half = HD / 2;
                for (int h = 0; h < H; h++) {
                    float* dqh = dq + h * HD;
                    for (int i = 0; i < half; i++) {
                        float d0 = dqh[i], d1 = dqh[i + half];
                        dqh[i] = d0 * cos_ptr[i] + d1 * sin_ptr[i];
                        dqh[i + half] = -d0 * sin_ptr[i] + d1 * cos_ptr[i];
                    }
                }
                for (int h = 0; h < p.n_kv_heads; h++) {
                    float* dkh = dk + h * HD;
                    for (int i = 0; i < half; i++) {
                        float d0 = dkh[i], d1 = dkh[i + half];
                        dkh[i] = d0 * cos_ptr[i] + d1 * sin_ptr[i];
                        dkh[i + half] = -d0 * sin_ptr[i] + d1 * cos_ptr[i];
                    }
                }
            } else {
                for (int i = 0; i < D; i += 2) {
                    int r = (i % HD) / 2;
                    float d0 = dq[i], d1 = dq[i + 1];
                    dq[i] = d0 * cos_ptr[r] + d1 * sin_ptr[r];
                    dq[i + 1] = -d0 * sin_ptr[r] + d1 * cos_ptr[r];
                    if (i < KV) {
                        float k0 = dk[i], k1 = dk[i + 1];
                        dk[i] = k0 * cos_ptr[r] + k1 * sin_ptr[r];
                        dk[i + 1] = -k0 * sin_ptr[r] + k1 * cos_ptr[r];
                    }
                }
            }

            const float* norm = layer_dim_at(ws.norm_att, l, t);
            float* dnorm = layer_dim_at(ws.d_norm_att, l, t);
            std::fill(dnorm, dnorm + D, 0.0f);
            matmul_backward(dnorm, grad + wq_off, dq, norm,
                            w.wq + static_cast<size_t>(l) * D * D, D, D);
            matmul_backward(engine->xb.data(), grad + wk_off, dk, norm,
                            w.wk + static_cast<size_t>(l) * KV * D, D, KV);
            for (int d = 0; d < D; d++) dnorm[d] += engine->xb[d];
            matmul_backward(engine->x.data(), grad + wv_off, layer_kv_at(ws.d_v, l, t), norm,
                            w.wv + static_cast<size_t>(l) * KV * D, D, KV);
            for (int d = 0; d < D; d++) dnorm[d] += engine->x[d];

            float* d_input = ws.d_states.data() + (static_cast<size_t>(l) * seq_len + t) * D;
            rmsnorm_backward(engine->x.data(), grad + off_rms_att + static_cast<size_t>(l) * D,
                             dnorm, state_at(l, t), w.rms_att_weight + l * D, D);
            for (int d = 0; d < D; d++) d_input[d] += engine->x[d];
        }
    }

    // Input embedding gradients include both classifier tying and token lookup.
    float* gemb = grad + off_emb;
    for (int t = 0; t < seq_len; t++) {
        float* row = gemb + static_cast<size_t>(input_tokens[t]) * D;
        const float* dx = ws.d_states.data() + static_cast<size_t>(t) * D;
        for (int d = 0; d < D; d++) row[d] += dx[d];
    }

    // Increment once per full sequence, not once per tensor group.
    engine->adam_step++;
    const float b1_corr = 1.0f - std::pow(beta1, static_cast<float>(engine->adam_step));
    const float b2_corr = 1.0f - std::pow(beta2, static_cast<float>(engine->adam_step));
    const float step_size = lr * std::sqrt(b2_corr) / std::max(b1_corr, 1e-12f);
    auto update_group = [&](float* params, size_t offset, size_t count) {
        float* g = grad + offset;
        float* m = engine->m_buffer.data() + offset;
        float* v = engine->v_buffer.data() + offset;
        for (size_t i = 0; i < count; i++) {
            m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * g[i] * g[i];
            params[i] -= lr * weight_decay * params[i];
            params[i] -= step_size * m[i] / (std::sqrt(v[i]) + eps);
        }
    };
    update_group(w.token_embedding_table, off_emb, static_cast<size_t>(p.vocab_size) * D);
    update_group(w.rms_att_weight, off_rms_att, static_cast<size_t>(L) * D);
    update_group(w.wq, off_wq, static_cast<size_t>(L) * D * D);
    update_group(w.wk, off_wk, static_cast<size_t>(L) * KV * D);
    update_group(w.wv, off_wv, static_cast<size_t>(L) * KV * D);
    update_group(w.wo, off_wo, static_cast<size_t>(L) * D * D);
    update_group(w.rms_ffn_weight, off_rms_ffn, static_cast<size_t>(L) * D);
    update_group(w.w1, off_w1, static_cast<size_t>(L) * Hidden * D);
    update_group(w.w2, off_w2, static_cast<size_t>(L) * D * Hidden);
    update_group(w.w3, off_w3, static_cast<size_t>(L) * Hidden * D);
    update_group(w.rms_final_weight, off_rms_final, D);
    if (!w.shared_classifier && w.wcls && w.wcls != w.token_embedding_table) {
        update_group(w.wcls, off_wcls, static_cast<size_t>(p.vocab_size) * D);
    }
    return total_loss * inv_seq;
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

    // Ensure AdamW moment buffers are allocated
    size_t emb_size = static_cast<size_t>(p.vocab_size) * p.dim;
    if (engine->m_buffer.size() < emb_size) {
        engine->m_buffer.assign(emb_size, 0.0f);
        engine->v_buffer.assign(emb_size, 0.0f);
    }
    if (w.wcls && engine->m_cls.size() < emb_size) {
        engine->m_cls.assign(emb_size, 0.0f);
        engine->v_cls.assign(emb_size, 0.0f);
    }

    float* step_logits = engine->train_step_logits.data();
    float* dlogits = engine->train_dlogits.data();
    float* probs = engine->train_probs.data();

    // Count active supervised target tokens (skip masked tokens where target < 0)
    int active_targets = 0;
    for (int pos = 0; pos < seq_len; pos++) {
        int t = target_tokens[pos];
        if (t >= 0 && t < p.vocab_size) active_targets++;
    }
    if (active_targets == 0) return 0.0f;
    float inv_targets = 1.0f / active_targets;

    engine->adam_step++;
    int t_step = engine->adam_step;
    float beta1_corr = 1.0f - std::pow(beta1, static_cast<float>(t_step));
    float beta2_corr = 1.0f - std::pow(beta2, static_cast<float>(t_step));
    float inv_beta1_corr = 1.0f / std::max(1e-7f, beta1_corr);
    float inv_beta2_corr = 1.0f / std::max(1e-7f, beta2_corr);

    for (int pos = 0; pos < seq_len; pos++) {
        int in_tok = input_tokens[pos];
        int target_tok = target_tokens[pos];

        llama_forward(engine, in_tok, pos, step_logits);

        // If target is masked (e.g., prompt token in SFT), forward ran to update KV cache,
        // but we skip cross-entropy loss and gradient backpropagation for this token
        if (target_tok < 0 || target_tok >= p.vocab_size) {
            continue;
        }

        // Softmax & Cross-Entropy Loss
        std::memcpy(probs, step_logits, p.vocab_size * sizeof(float));
        softmax(probs, p.vocab_size);

        float target_prob = std::max(1e-12f, probs[target_tok]);
        total_loss += -std::log(target_prob);

        // dL/dLogits = (probs - 1_{target}) / active_targets
        for (int i = 0; i < p.vocab_size; i++) {
            dlogits[i] = (probs[i] - (i == target_tok ? 1.0f : 0.0f)) * inv_targets;
        }

        // Gradient backprop through classifier into embeddings
        const float* cls_w = w.wcls ? w.wcls : w.token_embedding_table;
        float* d_emb_row = w.token_embedding_table + in_tok * p.dim;
        float* m_emb_row = engine->m_buffer.data() + in_tok * p.dim;
        float* v_emb_row = engine->v_buffer.data() + in_tok * p.dim;

        // Fast vector-matrix product: g = cls_w^T @ dlogits.
        // Reuse a preallocated buffer: allocating a std::vector for every
        // token was a significant hot-loop cost on CPU-only fine-tuning.
        std::fill(engine->train_grad_embedding.begin(), engine->train_grad_embedding.end(), 0.0f);
        float* g = engine->train_grad_embedding.data();
        for (int v = 0; v < p.vocab_size; v++) {
            float dv = dlogits[v];
            if (v != target_tok && std::abs(dv) < 1e-5f) continue;
            const float* cls_row = cls_w + v * p.dim;
            for (int d = 0; d < p.dim; d++) {
                g[d] += dv * cls_row[d];
            }
        }
        for (int d = 0; d < p.dim; d++) {
            float grad = g[d];
            m_emb_row[d] = beta1 * m_emb_row[d] + (1.0f - beta1) * grad;
            v_emb_row[d] = beta2 * v_emb_row[d] + (1.0f - beta2) * (grad * grad);
            float m_hat = m_emb_row[d] * inv_beta1_corr;
            float v_hat = v_emb_row[d] * inv_beta2_corr;
            float step_val = m_hat / (std::sqrt(v_hat) + eps);
            d_emb_row[d] -= lr * (step_val + weight_decay * d_emb_row[d]);
        }

        // Gradient update for classifier weights with AdamW:
        float* cls_base = (w.wcls ? w.wcls : w.token_embedding_table);
        float* m_cls_base = (w.wcls ? engine->m_cls.data() : engine->m_buffer.data());
        float* v_cls_base = (w.wcls ? engine->v_cls.data() : engine->v_buffer.data());
        const float* x_vec = engine->x.data();

        // 1. Target token classifier row
        float* cls_target = cls_base + target_tok * p.dim;
        float* m_target = m_cls_base + target_tok * p.dim;
        float* v_target = v_cls_base + target_tok * p.dim;
        float d_target = dlogits[target_tok];
        for (int d = 0; d < p.dim; d++) {
            float grad = d_target * x_vec[d];
            m_target[d] = beta1 * m_target[d] + (1.0f - beta1) * grad;
            v_target[d] = beta2 * v_target[d] + (1.0f - beta2) * (grad * grad);
            float m_hat = m_target[d] * inv_beta1_corr;
            float v_hat = v_target[d] * inv_beta2_corr;
            float step_val = m_hat / (std::sqrt(v_hat) + eps);
            cls_target[d] -= lr * (step_val + weight_decay * cls_target[d]);
        }

        // 2. Competing tokens classifier rows
        float prob_threshold = (p.vocab_size <= 1024) ? 0.0f : 0.005f;
        for (int v = 0; v < p.vocab_size; v++) {
            if (v == target_tok) continue;
            if (prob_threshold > 0.0f && probs[v] < prob_threshold) continue;
            float* cls_row = cls_base + v * p.dim;
            float* m_row = m_cls_base + v * p.dim;
            float* v_row = v_cls_base + v * p.dim;
            float dv = dlogits[v];
            for (int d = 0; d < p.dim; d++) {
                float grad = dv * x_vec[d];
                m_row[d] = beta1 * m_row[d] + (1.0f - beta1) * grad;
                v_row[d] = beta2 * v_row[d] + (1.0f - beta2) * (grad * grad);
                float m_hat = m_row[d] * inv_beta1_corr;
                float v_hat = v_row[d] * inv_beta2_corr;
                float step_val = m_hat / (std::sqrt(v_hat) + eps);
                cls_row[d] -= lr * (step_val + weight_decay * cls_row[d]);
            }
        }
    }

    return total_loss / active_targets;
}

} // extern "C"
