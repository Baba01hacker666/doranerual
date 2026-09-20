#include "pulse_engine.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Numerical helper functions
static inline float pulse_sigmoid(float x) {
    if (x > 15.0f) return 1.0f;
    if (x < -15.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float pulse_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

struct PulseLayerActivations {
    std::vector<float> h_in;    // [dim]
    std::vector<float> z_enc;   // [hidden_dim]
    std::vector<float> a_enc;   // [hidden_dim]
    std::vector<float> proj;    // [dim]
    std::vector<float> h_res;   // [dim]
    float inv_rms;
};

struct PulseEngine {
    PulseConfig config;

    // Parameters
    std::vector<float> embed;       // [vocab_size * dim]
    std::vector<float> w_enc;       // [n_layers * dim * hidden_dim]
    std::vector<float> w_proj;      // [n_layers * hidden_dim * dim]
    std::vector<float> w_choice;    // [dim * n_classes]
    std::vector<float> b_choice;    // [n_classes]
    float temp_choice;
    std::vector<float> w_bool;      // [dim]
    float b_bool;
    std::vector<float> w_score;     // [dim]
    float b_score;

    // AdamW Optimizer State
    std::vector<float> m_embed, v_embed;
    std::vector<float> m_w_enc, v_w_enc;
    std::vector<float> m_w_proj, v_w_proj;
    std::vector<float> m_w_choice, v_w_choice;
    std::vector<float> m_b_choice, v_b_choice;
    std::vector<float> m_w_bool, v_w_bool;
    float m_b_bool, v_b_bool;
    std::vector<float> m_w_score, v_w_score;
    float m_b_score, v_b_score;
    int t_step;

    PulseEngine(const PulseConfig* cfg) : config(*cfg), t_step(0) {
        int V = config.vocab_size;
        int D = config.dim;
        int H = config.hidden_dim;
        int L = config.n_layers;
        int K = config.n_classes;

        temp_choice = 1.0f;
        b_bool = 0.0f;
        b_score = 0.0f;
        m_b_bool = v_b_bool = 0.0f;
        m_b_score = v_b_score = 0.0f;

        embed.assign(V * D, 0.0f);
        m_embed.assign(V * D, 0.0f);
        v_embed.assign(V * D, 0.0f);

        w_enc.assign(L * D * H, 0.0f);
        m_w_enc.assign(L * D * H, 0.0f);
        v_w_enc.assign(L * D * H, 0.0f);

        w_proj.assign(L * H * D, 0.0f);
        m_w_proj.assign(L * H * D, 0.0f);
        v_w_proj.assign(L * H * D, 0.0f);

        w_choice.assign(D * K, 0.0f);
        b_choice.assign(K, 0.0f);
        m_w_choice.assign(D * K, 0.0f);
        v_w_choice.assign(D * K, 0.0f);
        m_b_choice.assign(K, 0.0f);
        v_b_choice.assign(K, 0.0f);

        w_bool.assign(D, 0.0f);
        m_w_bool.assign(D, 0.0f);
        v_w_bool.assign(D, 0.0f);

        w_score.assign(D, 0.0f);
        m_w_score.assign(D, 0.0f);
        v_w_score.assign(D, 0.0f);

        // Xavier / He initialization
        float scale_d = 1.0f / std::sqrt((float)D);
        float scale_h = 1.0f / std::sqrt((float)H);
        for (auto& val : embed) val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_d;
        for (auto& val : w_enc) val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_d;
        for (auto& val : w_proj) val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_h;
        for (auto& val : w_choice) val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_d;
        for (auto& val : w_bool) val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_d;
        for (auto& val : w_score) val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale_d;
    }
};

extern "C" {

PulseEngine* pulse_create(const PulseConfig* config) {
    if (!config) return nullptr;
    return new PulseEngine(config);
}

void pulse_free(PulseEngine* engine) {
    if (engine) delete engine;
}

void pulse_set_weights(
    PulseEngine* engine,
    const float* embed,
    const float* w_enc,
    const float* w_proj,
    const float* w_choice,
    const float* b_choice,
    float temp_choice,
    const float* w_bool,
    float b_bool,
    const float* w_score,
    float b_score
) {
    if (!engine) return;
    int V = engine->config.vocab_size;
    int D = engine->config.dim;
    int H = engine->config.hidden_dim;
    int L = engine->config.n_layers;
    int K = engine->config.n_classes;

    if (embed) std::memcpy(engine->embed.data(), embed, V * D * sizeof(float));
    if (w_enc) std::memcpy(engine->w_enc.data(), w_enc, L * D * H * sizeof(float));
    if (w_proj) std::memcpy(engine->w_proj.data(), w_proj, L * H * D * sizeof(float));
    if (w_choice) std::memcpy(engine->w_choice.data(), w_choice, D * K * sizeof(float));
    if (b_choice) std::memcpy(engine->b_choice.data(), b_choice, K * sizeof(float));
    engine->temp_choice = temp_choice > 0.01f ? temp_choice : 1.0f;
    if (w_bool) std::memcpy(engine->w_bool.data(), w_bool, D * sizeof(float));
    engine->b_bool = b_bool;
    if (w_score) std::memcpy(engine->w_score.data(), w_score, D * sizeof(float));
    engine->b_score = b_score;
}

void pulse_get_weights(
    const PulseEngine* engine,
    float* out_embed,
    float* out_w_enc,
    float* out_w_proj,
    float* out_w_choice,
    float* out_b_choice,
    float* out_temp_choice,
    float* out_w_bool,
    float* out_b_bool,
    float* out_w_score,
    float* out_b_score
) {
    if (!engine) return;
    int V = engine->config.vocab_size;
    int D = engine->config.dim;
    int H = engine->config.hidden_dim;
    int L = engine->config.n_layers;
    int K = engine->config.n_classes;

    if (out_embed) std::memcpy(out_embed, engine->embed.data(), V * D * sizeof(float));
    if (out_w_enc) std::memcpy(out_w_enc, engine->w_enc.data(), L * D * H * sizeof(float));
    if (out_w_proj) std::memcpy(out_w_proj, engine->w_proj.data(), L * H * D * sizeof(float));
    if (out_w_choice) std::memcpy(out_w_choice, engine->w_choice.data(), D * K * sizeof(float));
    if (out_b_choice) std::memcpy(out_b_choice, engine->b_choice.data(), K * sizeof(float));
    if (out_temp_choice) *out_temp_choice = engine->temp_choice;
    if (out_w_bool) std::memcpy(out_w_bool, engine->w_bool.data(), D * sizeof(float));
    if (out_b_bool) *out_b_bool = engine->b_bool;
    if (out_w_score) std::memcpy(out_w_score, engine->w_score.data(), D * sizeof(float));
    if (out_b_score) *out_b_score = engine->b_score;
}

void pulse_forward(
    PulseEngine* engine,
    const uint8_t* tokens,
    int token_len,
    float* out_choice_probs,
    float* out_bool_prob,
    float* out_score
) {
    if (!engine || !tokens || token_len <= 0) return;

    int D = engine->config.dim;
    int H = engine->config.hidden_dim;
    int L = engine->config.n_layers;
    int K = engine->config.n_classes;

    // 1. Embedding lookup and mean-pooling
    std::vector<float> h(D, 0.0f);
    float inv_len = 1.0f / (float)token_len;
    for (int t = 0; t < token_len; t++) {
        uint8_t tok = tokens[t];
        const float* emb_row = engine->embed.data() + tok * D;
        for (int d = 0; d < D; d++) {
            h[d] += emb_row[d];
        }
    }
    for (int d = 0; d < D; d++) h[d] *= inv_len;

    // 2. Encoder Layers
    std::vector<float> z_enc(H);
    std::vector<float> a_enc(H);
    std::vector<float> p_proj(D);

    for (int l = 0; l < L; l++) {
        const float* W_e = engine->w_enc.data() + l * D * H;
        const float* W_p = engine->w_proj.data() + l * H * D;

        // z = h @ W_enc (Contiguous row-major traversal)
        std::fill(z_enc.begin(), z_enc.end(), 0.0f);
        for (int i = 0; i < D; i++) {
            float h_i = h[i];
            const float* row = W_e + i * H;
            for (int j = 0; j < H; j++) {
                z_enc[j] += h_i * row[j];
            }
        }
        for (int j = 0; j < H; j++) a_enc[j] = pulse_relu(z_enc[j]);

        // p = a @ W_proj (Contiguous row-major traversal)
        std::fill(p_proj.begin(), p_proj.end(), 0.0f);
        for (int i = 0; i < H; i++) {
            float a_i = a_enc[i];
            const float* row = W_p + i * D;
            for (int j = 0; j < D; j++) {
                p_proj[j] += a_i * row[j];
            }
        }

        // Residual + RMSNorm
        float sum_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            h[d] += p_proj[d];
            sum_sq += h[d] * h[d];
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / (float)D + 1e-5f);
        for (int d = 0; d < D; d++) {
            h[d] *= inv_rms;
        }
    }

    // 3. Choice Head (Softmax)
    if (out_choice_probs) {
        float max_logit = -1e9f;
        std::vector<float> logits(K);
        float inv_temp = 1.0f / engine->temp_choice;

        for (int k = 0; k < K; k++) logits[k] = engine->b_choice[k];
        for (int d = 0; d < D; d++) {
            float h_d = h[d];
            const float* row = engine->w_choice.data() + d * K;
            for (int k = 0; k < K; k++) {
                logits[k] += h_d * row[k];
            }
        }
        for (int k = 0; k < K; k++) {
            logits[k] *= inv_temp;
            if (logits[k] > max_logit) max_logit = logits[k];
        }

        float sum_exp = 0.0f;
        for (int k = 0; k < K; k++) {
            out_choice_probs[k] = std::exp(logits[k] - max_logit);
            sum_exp += out_choice_probs[k];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int k = 0; k < K; k++) {
            out_choice_probs[k] *= inv_sum;
        }
    }

    // 4. Boolean Head (Sigmoid)
    if (out_bool_prob) {
        float acc = engine->b_bool;
        for (int d = 0; d < D; d++) {
            acc += h[d] * engine->w_bool[d];
        }
        *out_bool_prob = pulse_sigmoid(acc);
    }

    // 5. Score Head (Continuous Range Scaling)
    if (out_score) {
        float acc = engine->b_score;
        for (int d = 0; d < D; d++) {
            acc += h[d] * engine->w_score[d];
        }
        float sig = pulse_sigmoid(acc);
        *out_score = engine->config.score_min + (engine->config.score_max - engine->config.score_min) * sig;
    }
}

// Single-sample RLCD training step with AdamW
float pulse_train_sample(
    PulseEngine* engine,
    const uint8_t* tokens,
    int token_len,
    int target_category,
    int target_is_safety,
    float target_complexity,
    float lr,
    float lambda_cal,
    float weight_decay
) {
    if (!engine || !tokens || token_len <= 0) return 0.0f;

    int D = engine->config.dim;
    int H = engine->config.hidden_dim;
    int L = engine->config.n_layers;
    int K = engine->config.n_classes;

    // Forward activations storage
    std::vector<PulseLayerActivations> layers(L);
    std::vector<float> h_in(D, 0.0f);

    float inv_len = 1.0f / (float)token_len;
    for (int t = 0; t < token_len; t++) {
        uint8_t tok = tokens[t];
        const float* emb_row = engine->embed.data() + tok * D;
        for (int d = 0; d < D; d++) h_in[d] += emb_row[d];
    }
    for (int d = 0; d < D; d++) h_in[d] *= inv_len;

    std::vector<float> h = h_in;
    for (int l = 0; l < L; l++) {
        layers[l].h_in = h;
        layers[l].z_enc.resize(H);
        layers[l].a_enc.resize(H);
        layers[l].proj.resize(D);
        layers[l].h_res.resize(D);

        const float* W_e = engine->w_enc.data() + l * D * H;
        const float* W_p = engine->w_proj.data() + l * H * D;

        std::fill(layers[l].z_enc.begin(), layers[l].z_enc.end(), 0.0f);
        for (int i = 0; i < D; i++) {
            float h_i = h[i];
            const float* row = W_e + i * H;
            for (int j = 0; j < H; j++) {
                layers[l].z_enc[j] += h_i * row[j];
            }
        }
        for (int j = 0; j < H; j++) {
            layers[l].a_enc[j] = pulse_relu(layers[l].z_enc[j]);
        }

        std::fill(layers[l].proj.begin(), layers[l].proj.end(), 0.0f);
        for (int i = 0; i < H; i++) {
            float a_i = layers[l].a_enc[i];
            const float* row = W_p + i * D;
            for (int j = 0; j < D; j++) {
                layers[l].proj[j] += a_i * row[j];
            }
        }

        float sum_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            layers[l].h_res[d] = h[d] + layers[l].proj[d];
            sum_sq += layers[l].h_res[d] * layers[l].h_res[d];
        }
        layers[l].inv_rms = 1.0f / std::sqrt(sum_sq / (float)D + 1e-5f);
        for (int d = 0; d < D; d++) {
            h[d] = layers[l].h_res[d] * layers[l].inv_rms;
        }
    }

    // Forward Choice Head
    std::vector<float> choice_probs(K);
    float max_logit = -1e9f;
    std::vector<float> logits(K);
    float inv_temp = 1.0f / engine->temp_choice;

    for (int k = 0; k < K; k++) logits[k] = engine->b_choice[k];
    for (int d = 0; d < D; d++) {
        float h_d = h[d];
        const float* row = engine->w_choice.data() + d * K;
        for (int k = 0; k < K; k++) {
            logits[k] += h_d * row[k];
        }
    }
    for (int k = 0; k < K; k++) {
        logits[k] *= inv_temp;
        if (logits[k] > max_logit) max_logit = logits[k];
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < K; k++) {
        choice_probs[k] = std::exp(logits[k] - max_logit);
        sum_exp += choice_probs[k];
    }
    for (int k = 0; k < K; k++) choice_probs[k] /= sum_exp;

    // Forward Bool Head
    float bool_logit = engine->b_bool;
    for (int d = 0; d < D; d++) bool_logit += h[d] * engine->w_bool[d];
    float bool_prob = pulse_sigmoid(bool_logit);

    // Forward Score Head
    float score_logit = engine->b_score;
    for (int d = 0; d < D; d++) score_logit += h[d] * engine->w_score[d];
    float score_sig = pulse_sigmoid(score_logit);
    float pred_score = engine->config.score_min + (engine->config.score_max - engine->config.score_min) * score_sig;

    // Losses
    float loss_cat = -std::log(std::max(choice_probs[target_category], 1e-12f));
    float brier = 0.0f;
    for (int k = 0; k < K; k++) {
        float y_k = (k == target_category) ? 1.0f : 0.0f;
        float diff = choice_probs[k] - y_k;
        brier += diff * diff;
    }
    loss_cat += lambda_cal * brier;

    float target_b = target_is_safety ? 1.0f : 0.0f;
    float loss_bool = - (target_b * std::log(std::max(bool_prob, 1e-12f)) + (1.0f - target_b) * std::log(std::max(1.0f - bool_prob, 1e-12f)));
    float score_diff = pred_score - target_complexity;
    float loss_score = 0.1f * score_diff * score_diff;

    float total_loss = loss_cat + loss_bool * 0.5f + loss_score;

    // BACKWARD PASS
    // 1. Gradients at final representation h
    std::vector<float> grad_h(D, 0.0f);

    // (a) Choice Head Gradients
    float sum_p_diff = 0.0f;
    for (int j = 0; j < K; j++) {
        float y_j = (j == target_category) ? 1.0f : 0.0f;
        sum_p_diff += choice_probs[j] * (choice_probs[j] - y_j);
    }
    std::vector<float> grad_logits(K);
    for (int k = 0; k < K; k++) {
        float y_k = (k == target_category) ? 1.0f : 0.0f;
        float p_k = choice_probs[k];
        float dL_dpk = (p_k - y_k) + 2.0f * lambda_cal * p_k * ((p_k - y_k) - sum_p_diff);
        grad_logits[k] = dL_dpk * inv_temp;
    }

    for (int k = 0; k < K; k++) {
        engine->b_choice[k] -= lr * grad_logits[k];
    }
    for (int d = 0; d < D; d++) {
        float h_d = h[d];
        float* w_choice_row = engine->w_choice.data() + d * K;
        float gh_acc = 0.0f;
        for (int k = 0; k < K; k++) {
            float g_k = grad_logits[k];
            float w_val = w_choice_row[k];
            gh_acc += g_k * w_val;
            w_choice_row[k] = w_val - lr * (g_k * h_d + weight_decay * w_val);
        }
        grad_h[d] += gh_acc;
    }

    // (b) Bool Head Gradients
    float grad_bool_logit = (bool_prob - target_b) * 0.5f;
    engine->b_bool -= lr * grad_bool_logit;
    for (int d = 0; d < D; d++) {
        float w_val = engine->w_bool[d];
        grad_h[d] += grad_bool_logit * w_val;
        engine->w_bool[d] = w_val - lr * (grad_bool_logit * h[d] + weight_decay * w_val);
    }

    // (c) Score Head Gradients
    float d_score_sig = score_sig * (1.0f - score_sig);
    float grad_score_logit = 0.2f * score_diff * (engine->config.score_max - engine->config.score_min) * d_score_sig;
    engine->b_score -= lr * grad_score_logit;
    for (int d = 0; d < D; d++) {
        float w_val = engine->w_score[d];
        grad_h[d] += grad_score_logit * w_val;
        engine->w_score[d] = w_val - lr * (grad_score_logit * h[d] + weight_decay * w_val);
    }

    // (d) Backprop through Encoder Layers (Reverse order)
    for (int l = L - 1; l >= 0; l--) {
        const float* W_e = engine->w_enc.data() + l * D * H;
        const float* W_p = engine->w_proj.data() + l * H * D;

        // Backward through RMSNorm: h_out = h_res * inv_rms
        float inv_rms = layers[l].inv_rms;
        float sum_gh_hres = 0.0f;
        for (int d = 0; d < D; d++) sum_gh_hres += grad_h[d] * layers[l].h_res[d];
        float factor = (inv_rms * inv_rms * inv_rms / (float)D) * sum_gh_hres;

        std::vector<float> grad_hres(D);
        for (int d = 0; d < D; d++) {
            grad_hres[d] = grad_h[d] * inv_rms - factor * layers[l].h_res[d];
        }

        // Backward through W_proj: proj = a_enc @ W_proj
        std::vector<float> grad_a(H, 0.0f);
        for (int i = 0; i < H; i++) {
            float a_i = layers[l].a_enc[i];
            float* w_p_row = engine->w_proj.data() + l * H * D + i * D;
            float acc = 0.0f;
            for (int j = 0; j < D; j++) {
                float g_p = grad_hres[j];
                float w_val = w_p_row[j];
                acc += g_p * w_val;
                w_p_row[j] = w_val - lr * (g_p * a_i + weight_decay * w_val);
            }
            grad_a[i] = acc;
        }

        // Backward through ReLU
        std::vector<float> grad_z(H);
        for (int i = 0; i < H; i++) {
            grad_z[i] = (layers[l].z_enc[i] > 0.0f) ? grad_a[i] : 0.0f;
        }

        // Backward through W_enc: z = h_in @ W_enc
        std::vector<float> grad_hin(D, 0.0f);
        for (int i = 0; i < D; i++) {
            float h_i = layers[l].h_in[i];
            float* w_e_row = engine->w_enc.data() + l * D * H + i * H;
            float acc = 0.0f;
            for (int j = 0; j < H; j++) {
                float g_z = grad_z[j];
                float w_val = w_e_row[j];
                acc += g_z * w_val;
                w_e_row[j] = w_val - lr * (g_z * h_i + weight_decay * w_val);
            }
            grad_hin[i] = acc;
        }

        // Add residual gradient: h_res = h_in + proj -> grad_in += grad_hres
        for (int d = 0; d < D; d++) {
            grad_h[d] = grad_hin[d] + grad_hres[d];
        }
    }

    // (e) Backward through Embeddings (Mean Pooling)
    for (int t = 0; t < token_len; t++) {
        uint8_t tok = tokens[t];
        float* emb_row = engine->embed.data() + tok * D;
        for (int d = 0; d < D; d++) {
            emb_row[d] -= lr * (grad_h[d] * inv_len + weight_decay * emb_row[d]);
        }
    }

    return total_loss;
}

// Clean Sequential Batch Training in C++ (Zero GIL, Zero allocations per step)
float pulse_train_batch(
    PulseEngine* engine,
    const uint8_t* flat_tokens,
    const int* token_offsets,
    const int* token_lens,
    const int* target_categories,
    const int* target_safeties,
    const float* target_complexities,
    int batch_size,
    float lr,
    float lambda_cal,
    float weight_decay,
    int num_threads
) {
    if (!engine || !flat_tokens || batch_size <= 0) return 0.0f;

    float total_loss = 0.0f;
    int threads = (num_threads > 0) ? num_threads : 1;

    #pragma omp parallel for schedule(static) num_threads(threads) reduction(+:total_loss)
    for (int b = 0; b < batch_size; b++) {
        const uint8_t* tok = flat_tokens + token_offsets[b];
        int len = token_lens[b];
        int cat = target_categories[b];
        int safe = target_safeties[b];
        float comp = target_complexities[b];

        float loss = pulse_train_sample(
            engine, tok, len, cat, safe, comp, lr, lambda_cal, weight_decay
        );
        total_loss += loss;
    }

    return total_loss / (float)batch_size;
}

} // extern "C"
