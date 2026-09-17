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

void rmsnorm_forward(float* out, const float* x, const float* weight, int size, float eps = 1e-5f) {
    float sum_sq = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_sq += x[i] * x[i];
    }
    float inv_rms = 1.0f / std::sqrt(sum_sq / size + eps);
    for (int i = 0; i < size; i++) {
        out[i] = x[i] * inv_rms * weight[i];
    }
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

// Matmul: y (d,) = W (d, n) @ x (n,)
void matmul_forward(float* y, const float* x, const float* W, int n, int d) {
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        const float* row = W + i * n;
        float acc = 0.0f;
        for (int j = 0; j < n; j++) {
            acc += row[j] * x[j];
        }
        y[i] = acc;
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
        #pragma omp parallel for
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int i = 0; i < d; i++) {
                acc += W[i * n + j] * dy[i];
            }
            dx[j] = acc;
        }
    }

    if (dW) {
        #pragma omp parallel for
        for (int i = 0; i < d; i++) {
            float dy_i = dy[i];
            float* row = dW + i * n;
            for (int j = 0; j < n; j++) {
                row[j] += dy_i * x[j];
            }
        }
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    float inv_sum = 1.0f / (sum > 0.0f ? sum : 1e-5f);
    for (int i = 0; i < size; i++) {
        x[i] *= inv_sum;
    }
}

} // namespace

struct LlamaCppEngine {
    LlamaCppConfig config;
    LlamaCppWeights weights;

    // KV Cache
    std::vector<float> key_cache;   // (n_layers, seq_len, kv_dim)
    std::vector<float> val_cache;   // (n_layers, seq_len, kv_dim)

    // Inference Activation Buffers
    std::vector<float> x;
    std::vector<float> xb;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> att;
    std::vector<float> hb1;
    std::vector<float> hb3;
    std::vector<float> hb;
    std::vector<float> logits;

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
        int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;

        key_cache.resize(config.n_layers * config.seq_len * kv_dim, 0.0f);
        val_cache.resize(config.n_layers * config.seq_len * kv_dim, 0.0f);

        x.resize(config.dim, 0.0f);
        xb.resize(config.dim, 0.0f);
        q.resize(config.dim, 0.0f);
        k.resize(kv_dim, 0.0f);
        v.resize(kv_dim, 0.0f);
        att.resize(config.n_heads * config.seq_len, 0.0f);
        hb1.resize(config.hidden_dim, 0.0f);
        hb3.resize(config.hidden_dim, 0.0f);
        hb.resize(config.hidden_dim, 0.0f);
        logits.resize(config.vocab_size, 0.0f);

        size_t total_weights = calculate_total_parameters();
        grad_buffer.resize(total_weights, 0.0f);
        m_buffer.resize(total_weights, 0.0f);
        v_buffer.resize(total_weights, 0.0f);
    }

    size_t calculate_total_parameters() const {
        size_t head_size = config.dim / config.n_heads;
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

    const LlamaCppConfig& p = engine->config;
    const LlamaCppWeights& w = engine->weights;
    int head_size = p.dim / p.n_heads;
    int kv_dim = (p.dim * p.n_kv_heads) / p.n_heads;
    int kv_mul = p.n_heads / p.n_kv_heads;

    // 1. Token embedding lookup
    const float* emb_row = w.token_embedding_table + token * p.dim;
    std::memcpy(engine->x.data(), emb_row, p.dim * sizeof(float));

    for (int l = 0; l < p.n_layers; l++) {
        // Pre-attention RMSNorm
        rmsnorm_forward(engine->xb.data(), engine->x.data(), w.rms_att_weight + l * p.dim, p.dim);

        // Q, K, V projections
        matmul_forward(engine->q.data(), engine->xb.data(), w.wq + l * p.dim * p.dim, p.dim, p.dim);
        matmul_forward(engine->k.data(), engine->xb.data(), w.wk + l * kv_dim * p.dim, p.dim, kv_dim);
        matmul_forward(engine->v.data(), engine->xb.data(), w.wv + l * kv_dim * p.dim, p.dim, kv_dim);

        // RoPE relative positional encoding
        for (int i = 0; i < p.dim; i += 2) {
            int h_dim = i % head_size;
            float freq = 1.0f / std::pow(10000.0f, static_cast<float>(h_dim) / head_size);
            float val = pos * freq;
            float fcr = std::cos(val);
            float fci = std::sin(val);

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

        // Cache Key and Value at current position
        int loff = l * p.seq_len * kv_dim;
        std::memcpy(engine->key_cache.data() + loff + pos * kv_dim, engine->k.data(), kv_dim * sizeof(float));
        std::memcpy(engine->val_cache.data() + loff + pos * kv_dim, engine->v.data(), kv_dim * sizeof(float));

        // Multi-head attention
        std::vector<float> attn_out(p.dim, 0.0f);

        #pragma omp parallel for
        for (int h = 0; h < p.n_heads; h++) {
            const float* q_head = engine->q.data() + h * head_size;
            float* att_head = engine->att.data() + h * p.seq_len;
            int kv_h = h / kv_mul;

            for (int t = 0; t <= pos; t++) {
                const float* k_past = engine->key_cache.data() + loff + t * kv_dim + kv_h * head_size;
                float score = 0.0f;
                for (int d = 0; d < head_size; d++) {
                    score += q_head[d] * k_past[d];
                }
                score /= std::sqrt(static_cast<float>(head_size));
                att_head[t] = score;
            }

            softmax(att_head, pos + 1);

            float* out_head = attn_out.data() + h * head_size;
            for (int t = 0; t <= pos; t++) {
                const float* v_past = engine->val_cache.data() + loff + t * kv_dim + kv_h * head_size;
                float a = att_head[t];
                for (int d = 0; d < head_size; d++) {
                    out_head[d] += a * v_past[d];
                }
            }
        }

        // Attention output projection: wo @ attn_out
        matmul_forward(engine->xb.data(), attn_out.data(), w.wo + l * p.dim * p.dim, p.dim, p.dim);
        for (int i = 0; i < p.dim; i++) {
            engine->x[i] += engine->xb[i];
        }

        // Pre-FFN RMSNorm
        rmsnorm_forward(engine->xb.data(), engine->x.data(), w.rms_ffn_weight + l * p.dim, p.dim);

        // SwiGLU FFN: w2 @ (silu(w1 @ xb) * (w3 @ xb))
        matmul_forward(engine->hb1.data(), engine->xb.data(), w.w1 + l * p.hidden_dim * p.dim, p.dim, p.hidden_dim);
        matmul_forward(engine->hb3.data(), engine->xb.data(), w.w3 + l * p.hidden_dim * p.dim, p.dim, p.hidden_dim);

        for (int i = 0; i < p.hidden_dim; i++) {
            engine->hb[i] = silu(engine->hb1[i]) * engine->hb3[i];
        }

        matmul_forward(engine->xb.data(), engine->hb.data(), w.w2 + l * p.dim * p.hidden_dim, p.hidden_dim, p.dim);
        for (int i = 0; i < p.dim; i++) {
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
    const int vocab_size = engine->config.vocab_size;
    std::vector<float> probs(engine->logits);

    if (temperature <= 0.0f) {
        // Argmax greedy
        int best_i = 0;
        float best_v = probs[0];
        for (int i = 1; i < vocab_size; i++) {
            if (probs[i] > best_v) {
                best_v = probs[i];
                best_i = i;
            }
        }
        return best_i;
    }

    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= temperature;
    }
    softmax(probs.data(), vocab_size);

    if (top_p < 1.0f) {
        std::vector<std::pair<float, int>> sorted_probs(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            sorted_probs[i] = {probs[i], i};
        }
        std::sort(sorted_probs.begin(), sorted_probs.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        float cumsum = 0.0f;
        int cutoff_idx = vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            cumsum += sorted_probs[i].first;
            if (cumsum > top_p && i > 0) {
                cutoff_idx = i + 1;
                break;
            }
        }

        float renorm_sum = 0.0f;
        for (int i = 0; i < cutoff_idx; i++) {
            renorm_sum += sorted_probs[i].first;
        }

        std::uniform_real_distribution<float> dist(0.0f, renorm_sum);
        float r = dist(engine->rng);
        float acc = 0.0f;
        for (int i = 0; i < cutoff_idx; i++) {
            acc += sorted_probs[i].first;
            if (r <= acc) {
                return sorted_probs[i].second;
            }
        }
        return sorted_probs[0].second;
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(engine->rng);
    float acc = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        acc += probs[i];
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

    std::vector<float> step_logits(p.vocab_size);
    std::vector<float> dlogits(p.vocab_size);

    for (int pos = 0; pos < seq_len; pos++) {
        int in_tok = input_tokens[pos];
        int target_tok = target_tokens[pos];

        llama_forward(engine, in_tok, pos, step_logits.data());

        // Softmax & Cross-Entropy Loss
        std::vector<float> probs = step_logits;
        softmax(probs.data(), p.vocab_size);

        float target_prob = std::max(1e-12f, probs[target_tok]);
        total_loss += -std::log(target_prob);

        // dL/dLogits = (probs - 1_{target}) / seq_len
        for (int i = 0; i < p.vocab_size; i++) {
            dlogits[i] = (probs[i] - (i == target_tok ? 1.0f : 0.0f)) / seq_len;
        }

        // Gradient backprop through classifier into embeddings
        const float* cls_w = w.wcls ? w.wcls : w.token_embedding_table;
        float* d_emb_row = w.token_embedding_table + in_tok * p.dim;

        for (int d = 0; d < p.dim; d++) {
            float g = 0.0f;
            for (int v = 0; v < p.vocab_size; v++) {
                g += dlogits[v] * cls_w[v * p.dim + d];
            }
            d_emb_row[d] -= lr * (g + weight_decay * d_emb_row[d]);
        }
    }

    engine->adam_step++;
    return total_loss / seq_len;
}

} // extern "C"
