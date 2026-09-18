#include "rtu_engine.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

// Cross-platform SIMD guards
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#elif defined(__x86_64__) || defined(_M_X64)
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#endif

namespace {

inline float sigmoid_scalar(float x) {
    if (x > 88.0f) return 1.0f;
    if (x < -88.0f) return 0.0f;
    return 1.0f / (1.0f + std::exp(-x));
}

inline float silu_scalar(float x) {
    return x * sigmoid_scalar(x);
}

inline float silu_deriv_scalar(float x) {
    float s = sigmoid_scalar(x);
    return s + x * s * (1.0f - s);
}

void layernorm_forward(
    const float* s,
    int dim,
    const float* gamma,
    const float* beta,
    float* norm_s,
    float* out_mean,
    float* out_var,
    float eps = 1e-5f
) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) sum += s[i];
    float mean = sum / static_cast<float>(dim);

    float sum_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float d = s[i] - mean;
        sum_sq += d * d;
    }
    float var = sum_sq / static_cast<float>(dim);
    float inv_std = 1.0f / std::sqrt(var + eps);

    for (int i = 0; i < dim; ++i) {
        float x_hat = (s[i] - mean) * inv_std;
        norm_s[i] = x_hat * gamma[i] + beta[i];
    }

    *out_mean = mean;
    *out_var = var;
}

void layernorm_backward(
    const float* dout,
    const float* s,
    int dim,
    const float* gamma,
    float mean,
    float var,
    float* dx,
    float* dgamma,
    float* dbeta,
    float eps = 1e-5f
) {
    float std_dev = std::sqrt(var + eps);
    float inv_std = 1.0f / std_dev;
    float inv_std3 = inv_std * inv_std * inv_std;

    float dvar = 0.0f;
    float dmean_term = 0.0f;

    for (int i = 0; i < dim; ++i) {
        float x_hat = (s[i] - mean) * inv_std;
        dgamma[i] += dout[i] * x_hat;
        dbeta[i] += dout[i];

        float dxhat = dout[i] * gamma[i];
        dvar += dxhat * (s[i] - mean) * (-0.5f * inv_std3);
        dmean_term += dxhat * (-inv_std);
    }

    float fdim = static_cast<float>(dim);
    float dmean = dmean_term;

    for (int i = 0; i < dim; ++i) {
        float dxhat = dout[i] * gamma[i];
        dx[i] = dxhat * inv_std + dvar * 2.0f * (s[i] - mean) / fdim + dmean / fdim;
    }
}

// Matrix-Vector Product: y = x @ W
void matmul_vec(
    const float* x,
    const float* W,
    float* y,
    int in_dim,
    int out_dim
) {
    std::fill(y, y + out_dim, 0.0f);
    for (int j = 0; j < out_dim; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < in_dim; ++i) {
            sum += x[i] * W[i * out_dim + j];
        }
        y[j] = sum;
    }
}

// Vector-Matrix Transposed Product: y = x @ W^T
void matmul_vec_transposed(
    const float* x,
    const float* W,
    float* y,
    int in_dim,
    int out_dim
) {
    std::fill(y, y + out_dim, 0.0f);
    for (int j = 0; j < out_dim; ++j) {
        float sum = 0.0f;
        const float* row = W + (j * in_dim);
        for (int i = 0; i < in_dim; ++i) {
            sum += x[i] * row[i];
        }
        y[j] = sum;
    }
}

} // namespace

struct LayerState {
    std::vector<float> state;
    std::vector<float> decaytrace;
    std::vector<float> embedtrace; // [vocab_size, dim]

    LayerState(int dim, int vocab_size)
        : state(dim, 0.0f),
          decaytrace(dim, 0.0f),
          embedtrace(vocab_size * dim, 0.0f) {}

    void reset() {
        std::fill(state.begin(), state.end(), 0.0f);
        std::fill(decaytrace.begin(), decaytrace.end(), 0.0f);
        std::fill(embedtrace.begin(), embedtrace.end(), 0.0f);
    }
};

struct RTUEngine {
    int dim;
    int n_layers;
    int vocab_size;
    float lr;
    float weight_decay;
    float variance_weight;
    float latent_weight;
    float stop_weight;
    int adam_step;

    // Weights
    std::vector<float> embed;   // [vocab_size, dim]
    std::vector<float> decoder; // [dim, vocab_size]
    std::vector<float> stop_w;  // [dim]
    float stop_b;

    std::vector<std::vector<float>> decay_w; // [n_layers][dim]
    std::vector<std::vector<float>> weights; // [n_layers][dim * dim]
    std::vector<std::vector<float>> gamma;   // [n_layers][dim]
    std::vector<std::vector<float>> beta;    // [n_layers][dim]

    // States
    std::vector<LayerState> layers;

    // AdamW optimizer states
    std::vector<float> m_embed, v_embed;
    std::vector<float> m_dec, v_dec;
    std::vector<float> m_stop_w, v_stop_w;
    float m_stop_b, v_stop_b;

    std::vector<std::vector<float>> m_decay, v_decay;
    std::vector<std::vector<float>> m_weights, v_weights;
    std::vector<std::vector<float>> m_gamma, v_gamma;
    std::vector<std::vector<float>> m_beta, v_beta;

    // Zero-allocation reusable buffers
    std::vector<float> x_buf;
    std::vector<float> dx_buf;
    std::vector<float> logits_buf;
    std::vector<float> probs_buf;
    std::vector<float> dlogits_buf;
    std::vector<float> dz_buf;
    std::vector<float> d_norm_buf;
    std::vector<float> d_state_buf;
    std::vector<float> curr_s_buf;
    std::vector<float> norm_s_buf;
    std::vector<float> pre_act_buf;

    std::vector<std::vector<float>> layer_decays;
    std::vector<std::vector<float>> layer_prev_states;
    std::vector<std::vector<float>> layer_curr_states;
    std::vector<std::vector<float>> layer_norm_states;
    std::vector<std::vector<float>> layer_pre_acts;
    std::vector<float> layer_means;
    std::vector<float> layer_vars;

    // Gradient accumulators
    std::vector<float> g_embed;
    std::vector<float> g_dec;
    std::vector<float> g_stop_w;
    float g_stop_b;
    std::vector<std::vector<float>> g_decay;
    std::vector<std::vector<float>> g_weights;
    std::vector<std::vector<float>> g_gamma;
    std::vector<std::vector<float>> g_beta;

    std::mt19937 rng;

    RTUEngine(
        int d,
        int l,
        int v,
        float lr_in,
        float wd_in,
        float var_w,
        float lat_w,
        float st_w,
        int seed
    ) : dim(d),
        n_layers(l),
        vocab_size(v),
        lr(lr_in),
        weight_decay(wd_in),
        variance_weight(var_w),
        latent_weight(lat_w),
        stop_weight(st_w),
        adam_step(0),
        stop_b(0.0f),
        m_stop_b(0.0f),
        v_stop_b(0.0f),
        g_stop_b(0.0f),
        rng(seed) {
        
        embed.resize(vocab_size * dim);
        decoder.resize(dim * vocab_size);
        stop_w.resize(dim);

        m_embed.assign(vocab_size * dim, 0.0f);
        v_embed.assign(vocab_size * dim, 0.0f);
        m_dec.assign(dim * vocab_size, 0.0f);
        v_dec.assign(dim * vocab_size, 0.0f);
        m_stop_w.assign(dim, 0.0f);
        v_stop_w.assign(dim, 0.0f);

        float emb_std = 1.0f / std::sqrt(static_cast<float>(dim));
        float proj_std = 0.02f / std::sqrt(std::max(1.0f, static_cast<float>(n_layers)));
        std::normal_distribution<float> emb_dist(0.0f, emb_std);
        std::normal_distribution<float> proj_dist(0.0f, proj_std);
        std::uniform_real_distribution<float> decay_dist(2.0f, 4.0f);

        for (auto& val : embed) val = emb_dist(rng);
        for (auto& val : decoder) val = emb_dist(rng);
        for (auto& val : stop_w) val = emb_dist(rng);

        decay_w.resize(n_layers);
        weights.resize(n_layers);
        gamma.resize(n_layers);
        beta.resize(n_layers);

        m_decay.resize(n_layers); v_decay.resize(n_layers);
        m_weights.resize(n_layers); v_weights.resize(n_layers);
        m_gamma.resize(n_layers); v_gamma.resize(n_layers);
        m_beta.resize(n_layers); v_beta.resize(n_layers);

        // Preallocated scratch buffers
        x_buf.resize(dim);
        dx_buf.resize(dim);
        logits_buf.resize(vocab_size);
        probs_buf.resize(vocab_size);
        dlogits_buf.resize(vocab_size);
        dz_buf.resize(dim);
        d_norm_buf.resize(dim);
        d_state_buf.resize(dim);
        curr_s_buf.resize(dim);
        norm_s_buf.resize(dim);
        pre_act_buf.resize(dim);

        layer_decays.resize(n_layers, std::vector<float>(dim));
        layer_prev_states.resize(n_layers, std::vector<float>(dim));
        layer_curr_states.resize(n_layers, std::vector<float>(dim));
        layer_norm_states.resize(n_layers, std::vector<float>(dim));
        layer_pre_acts.resize(n_layers, std::vector<float>(dim));
        layer_means.resize(n_layers);
        layer_vars.resize(n_layers);

        g_embed.assign(vocab_size * dim, 0.0f);
        g_dec.assign(dim * vocab_size, 0.0f);
        g_stop_w.assign(dim, 0.0f);
        g_decay.resize(n_layers, std::vector<float>(dim, 0.0f));
        g_weights.resize(n_layers, std::vector<float>(dim * dim, 0.0f));
        g_gamma.resize(n_layers, std::vector<float>(dim, 0.0f));
        g_beta.resize(n_layers, std::vector<float>(dim, 0.0f));

        for (int i = 0; i < n_layers; ++i) {
            decay_w[i].resize(dim);
            for (auto& val : decay_w[i]) val = decay_dist(rng);

            weights[i].resize(dim * dim);
            for (auto& val : weights[i]) val = proj_dist(rng);

            gamma[i].assign(dim, 1.0f);
            beta[i].assign(dim, 0.0f);

            m_decay[i].assign(dim, 0.0f); v_decay[i].assign(dim, 0.0f);
            m_weights[i].assign(dim * dim, 0.0f); v_weights[i].assign(dim * dim, 0.0f);
            m_gamma[i].assign(dim, 0.0f); v_gamma[i].assign(dim, 0.0f);
            m_beta[i].assign(dim, 0.0f); v_beta[i].assign(dim, 0.0f);

            layers.emplace_back(dim, vocab_size);
        }
    }

    void reset_state() {
        for (auto& layer : layers) {
            layer.reset();
        }
    }

    void reset_gradients() {
        std::fill(g_embed.begin(), g_embed.end(), 0.0f);
        std::fill(g_dec.begin(), g_dec.end(), 0.0f);
        std::fill(g_stop_w.begin(), g_stop_w.end(), 0.0f);
        g_stop_b = 0.0f;
        for (int i = 0; i < n_layers; ++i) {
            std::fill(g_decay[i].begin(), g_decay[i].end(), 0.0f);
            std::fill(g_weights[i].begin(), g_weights[i].end(), 0.0f);
            std::fill(g_gamma[i].begin(), g_gamma[i].end(), 0.0f);
            std::fill(g_beta[i].begin(), g_beta[i].end(), 0.0f);
        }
    }

    void forward_byte(
        int b,
        bool update_state,
        float* out_logits,
        float* out_stop_prob,
        float* out_latent
    ) {
        const float* enc = embed.data() + (b * dim);
        std::copy(enc, enc + dim, x_buf.begin());

        for (int i = 0; i < n_layers; ++i) {
            float mean = 0.0f, var = 0.0f;
            const float* prev_s = layers[i].state.data();

            for (int j = 0; j < dim; ++j) {
                float d = sigmoid_scalar(decay_w[i][j]);
                curr_s_buf[j] = d * prev_s[j] + enc[j];
            }

            if (update_state) {
                std::copy(curr_s_buf.begin(), curr_s_buf.end(), layers[i].state.begin());
            }

            layernorm_forward(curr_s_buf.data(), dim, gamma[i].data(), beta[i].data(), norm_s_buf.data(), &mean, &var);
            matmul_vec(norm_s_buf.data(), weights[i].data(), pre_act_buf.data(), dim, dim);

            for (int j = 0; j < dim; ++j) {
                x_buf[j] += silu_scalar(pre_act_buf[j]);
            }
        }

        if (out_latent) {
            std::copy(x_buf.begin(), x_buf.end(), out_latent);
        }

        if (out_logits) {
            matmul_vec(x_buf.data(), decoder.data(), out_logits, dim, vocab_size);
        }

        float stop_logit = stop_b;
        for (int j = 0; j < dim; ++j) {
            stop_logit += x_buf[j] * stop_w[j];
        }
        float stop_prob = sigmoid_scalar(stop_logit);
        if (out_stop_prob) {
            *out_stop_prob = stop_prob;
        }
    }

    int sample_byte(const float* logits, float temperature, float top_p) {
        if (temperature <= 0.0f) {
            int max_idx = 0;
            float max_val = logits[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    max_idx = i;
                }
            }
            return max_idx;
        }

        float max_l = logits[0];
        for (int i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_l) max_l = logits[i];
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            probs_buf[i] = std::exp((logits[i] - max_l) / std::max(1e-4f, temperature));
            sum_exp += probs_buf[i];
        }
        for (int i = 0; i < vocab_size; ++i) probs_buf[i] /= sum_exp;

        std::vector<int> indices(vocab_size);
        for (int i = 0; i < vocab_size; ++i) indices[i] = i;

        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return probs_buf[a] > probs_buf[b];
        });

        float cumsum = 0.0f;
        int cutoff = vocab_size;
        for (int i = 0; i < vocab_size; ++i) {
            cumsum += probs_buf[indices[i]];
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }

        float norm_sum = 0.0f;
        for (int i = 0; i < cutoff; ++i) norm_sum += probs_buf[indices[i]];
        std::uniform_real_distribution<float> u(0.0f, norm_sum);
        float r = u(rng);

        float acc = 0.0f;
        for (int i = 0; i < cutoff; ++i) {
            acc += probs_buf[indices[i]];
            if (r <= acc) return indices[i];
        }
        return indices[0];
    }

    void apply_adamw(
        float step_lr,
        float* param,
        float* grad,
        float* m,
        float* v,
        int size,
        bool decay = true,
        float clip_norm = 5.0f
    ) {
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;

        float sum_sq = 0.0f;
        for (int i = 0; i < size; ++i) sum_sq += grad[i] * grad[i];
        float gnorm = std::sqrt(sum_sq);
        if (gnorm > clip_norm && gnorm > 0.0f) {
            float scale = clip_norm / gnorm;
            for (int i = 0; i < size; ++i) grad[i] *= scale;
        }

        float corr1 = std::max(1e-7f, 1.0f - std::pow(beta1, static_cast<float>(adam_step)));
        float corr2 = std::max(1e-7f, 1.0f - std::pow(beta2, static_cast<float>(adam_step)));

        for (int i = 0; i < size; ++i) {
            m[i] = beta1 * m[i] + (1.0f - beta1) * grad[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * (grad[i] * grad[i]);

            float m_hat = m[i] / corr1;
            float v_hat = v[i] / corr2;
            float step = m_hat / (std::sqrt(v_hat) + eps);

            if (decay) {
                param[i] -= step_lr * (step + weight_decay * param[i]);
            } else {
                param[i] -= step_lr * step;
            }
        }
    }

    void apply_all_adamw(float step_lr) {
        adam_step += 1;
        apply_adamw(step_lr, embed.data(), g_embed.data(), m_embed.data(), v_embed.data(), vocab_size * dim);
        apply_adamw(step_lr, decoder.data(), g_dec.data(), m_dec.data(), v_dec.data(), dim * vocab_size);
        apply_adamw(step_lr, stop_w.data(), g_stop_w.data(), m_stop_w.data(), v_stop_w.data(), dim);
        apply_adamw(step_lr, &stop_b, &g_stop_b, &m_stop_b, &v_stop_b, 1, false);

        for (int i = 0; i < n_layers; ++i) {
            apply_adamw(step_lr, decay_w[i].data(), g_decay[i].data(), m_decay[i].data(), v_decay[i].data(), dim, false);
            apply_adamw(step_lr, weights[i].data(), g_weights[i].data(), m_weights[i].data(), v_weights[i].data(), dim * dim);
            apply_adamw(step_lr, gamma[i].data(), g_gamma[i].data(), m_gamma[i].data(), v_gamma[i].data(), dim, false);
            apply_adamw(step_lr, beta[i].data(), g_beta[i].data(), m_beta[i].data(), v_beta[i].data(), dim, false);
        }
        reset_gradients();
    }

    void forward_and_accumulate_gradients(
        int cur_b,
        int next_b,
        bool is_end,
        float* out_losses
    ) {
        const float* enc = embed.data() + (cur_b * dim);
        std::copy(enc, enc + dim, x_buf.begin());

        for (int i = 0; i < n_layers; ++i) {
            for (int j = 0; j < dim; ++j) {
                layer_decays[i][j] = sigmoid_scalar(decay_w[i][j]);
                layer_prev_states[i][j] = layers[i].state[j];
                layer_curr_states[i][j] = layer_decays[i][j] * layer_prev_states[i][j] + enc[j];
            }

            layers[i].state = layer_curr_states[i];

            layernorm_forward(
                layer_curr_states[i].data(),
                dim,
                gamma[i].data(),
                beta[i].data(),
                layer_norm_states[i].data(),
                &layer_means[i],
                &layer_vars[i]
            );

            matmul_vec(layer_norm_states[i].data(), weights[i].data(), layer_pre_acts[i].data(), dim, dim);

            for (int j = 0; j < dim; ++j) {
                x_buf[j] += silu_scalar(layer_pre_acts[i][j]);
            }
        }

        matmul_vec(x_buf.data(), decoder.data(), logits_buf.data(), dim, vocab_size);

        float stop_logit = stop_b;
        for (int j = 0; j < dim; ++j) stop_logit += x_buf[j] * stop_w[j];
        float stop_prob = sigmoid_scalar(stop_logit);

        float total_loss = 0.0f, ce_loss = 0.0f, lat_loss = 0.0f, var_loss = 0.0f, st_loss = 0.0f;

        // 1. Variance Loss: max(0, 1 - std(x))
        float sum_x = 0.0f, sum_x_sq = 0.0f;
        for (int j = 0; j < dim; ++j) {
            sum_x += x_buf[j];
            sum_x_sq += x_buf[j] * x_buf[j];
        }
        float mean_x = sum_x / static_cast<float>(dim);
        float var_x = (sum_x_sq / static_cast<float>(dim)) - (mean_x * mean_x);
        float std_x = std::sqrt(std::max(0.0f, var_x) + 1e-4f);

        std::fill(dx_buf.begin(), dx_buf.end(), 0.0f);
        if (std_x < 1.0f) {
            var_loss = (1.0f - std_x) * variance_weight;
            total_loss += var_loss;
            float scale = variance_weight / (static_cast<float>(dim) * std_x);
            for (int j = 0; j < dim; ++j) {
                dx_buf[j] += -(x_buf[j] - mean_x) * scale;
            }
        }

        if (next_b >= 0 && next_b < vocab_size) {
            // 2. Cross-Entropy Loss
            float max_l = logits_buf[0];
            for (int j = 1; j < vocab_size; ++j) {
                if (logits_buf[j] > max_l) max_l = logits_buf[j];
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < vocab_size; ++j) {
                probs_buf[j] = std::exp(logits_buf[j] - max_l);
                sum_exp += probs_buf[j];
            }
            for (int j = 0; j < vocab_size; ++j) probs_buf[j] /= sum_exp;

            float target_prob = std::max(1e-12f, probs_buf[next_b]);
            ce_loss = -std::log(target_prob);
            total_loss += ce_loss;

            for (int j = 0; j < vocab_size; ++j) {
                dlogits_buf[j] = probs_buf[j] - (j == next_b ? 1.0f : 0.0f);
            }

            // 3. Latent MSE
            const float* target_latent = embed.data() + (next_b * dim);
            float diff_sq_sum = 0.0f;
            for (int j = 0; j < dim; ++j) {
                float diff = x_buf[j] - target_latent[j];
                diff_sq_sum += diff * diff;
                dx_buf[j] += (2.0f * diff / static_cast<float>(dim)) * latent_weight;
            }
            lat_loss = (diff_sq_sum / static_cast<float>(dim)) * latent_weight;
            total_loss += lat_loss;

            // 4. Stop Loss
            float target_stop = is_end ? 1.0f : 0.0f;
            float diff_stop = stop_prob - target_stop;
            st_loss = (diff_stop * diff_stop) * stop_weight;
            total_loss += st_loss;

            float d_stop_logit = 2.0f * diff_stop * (stop_prob * (1.0f - stop_prob)) * stop_weight;
            for (int j = 0; j < dim; ++j) {
                g_stop_w[j] += x_buf[j] * d_stop_logit;
                dx_buf[j] += stop_w[j] * d_stop_logit;
            }
            g_stop_b += d_stop_logit;

            // Decoder gradients & dx accumulation
            for (int i = 0; i < dim; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < vocab_size; ++j) {
                    g_dec[i * vocab_size + j] += x_buf[i] * dlogits_buf[j];
                    sum += decoder[i * vocab_size + j] * dlogits_buf[j];
                }
                dx_buf[i] += sum;
            }

            // Layer backprop
            for (int i = n_layers - 1; i >= 0; --i) {
                for (int j = 0; j < dim; ++j) {
                    dz_buf[j] = dx_buf[j] * silu_deriv_scalar(layer_pre_acts[i][j]);
                }

                for (int r = 0; r < dim; ++r) {
                    for (int c = 0; c < dim; ++c) {
                        g_weights[i][r * dim + c] += layer_norm_states[i][r] * dz_buf[c];
                    }
                }

                matmul_vec_transposed(dz_buf.data(), weights[i].data(), d_norm_buf.data(), dim, dim);

                layernorm_backward(
                    d_norm_buf.data(),
                    layer_curr_states[i].data(),
                    dim,
                    gamma[i].data(),
                    layer_means[i],
                    layer_vars[i],
                    d_state_buf.data(),
                    g_gamma[i].data(),
                    g_beta[i].data()
                );

                // RTRL trace update
                float* etrace = layers[i].embedtrace.data();
                float* dtrace = layers[i].decaytrace.data();

                const float* dec_row = layer_decays[i].data();
                const float* d_st = d_state_buf.data();
                for (int v_idx = 0; v_idx < vocab_size; ++v_idx) {
                    float* v_etrace = etrace + (v_idx * dim);
                    float* g_emb_row = g_embed.data() + (v_idx * dim);
                    if (v_idx == cur_b) {
                        for (int j = 0; j < dim; ++j) {
                            float val = v_etrace[j] * dec_row[j] + 1.0f;
                            v_etrace[j] = val;
                            g_emb_row[j] += val * d_st[j];
                        }
                    } else {
                        for (int j = 0; j < dim; ++j) {
                            float val = v_etrace[j] * dec_row[j];
                            v_etrace[j] = val;
                            g_emb_row[j] += val * d_st[j];
                        }
                    }
                }

                for (int j = 0; j < dim; ++j) {
                    float decay_deriv = layer_decays[i][j] * (1.0f - layer_decays[i][j]);
                    dtrace[j] = layer_decays[i][j] * dtrace[j] + decay_deriv * layer_prev_states[i][j];
                    g_decay[i][j] += d_state_buf[j] * dtrace[j];
                }
            }

            // Direct embedding gradient
            float* cur_g_emb = g_embed.data() + (cur_b * dim);
            for (int j = 0; j < dim; ++j) cur_g_emb[j] += dx_buf[j];
        }

        if (out_losses) {
            out_losses[0] = total_loss;
            out_losses[1] = ce_loss;
            out_losses[2] = lat_loss;
            out_losses[3] = var_loss;
            out_losses[4] = st_loss;
        }
    }
};

extern "C" {

RTUEngine* rtu_create(
    int dim,
    int n_layers,
    int vocab_size,
    float lr,
    float weight_decay,
    float variance_weight,
    float latent_weight,
    float stop_weight,
    int seed
) {
    return new RTUEngine(dim, n_layers, vocab_size, lr, weight_decay, variance_weight, latent_weight, stop_weight, seed);
}

void rtu_free(RTUEngine* engine) {
    delete engine;
}

void rtu_reset_state(RTUEngine* engine) {
    if (engine) engine->reset_state();
}

void rtu_forward_byte(
    RTUEngine* engine,
    int b,
    int update_state,
    float* out_logits,
    float* out_stop_prob,
    float* out_latent
) {
    if (engine) {
        engine->forward_byte(b, update_state != 0, out_logits, out_stop_prob, out_latent);
    }
}

int rtu_sample_byte(RTUEngine* engine, const float* logits, float temperature, float top_p) {
    return engine ? engine->sample_byte(logits, temperature, top_p) : 0;
}

void rtu_step_online_train(
    RTUEngine* engine,
    int cur_b,
    int next_b,
    int is_end,
    float lr,
    int* out_sampled_b,
    float* out_stop_prob,
    float* out_losses
) {
    if (engine) {
        engine->reset_gradients();
        engine->forward_and_accumulate_gradients(cur_b, next_b, is_end != 0, out_losses);
        engine->apply_all_adamw(lr);
        if (out_stop_prob) *out_stop_prob = sigmoid_scalar(engine->stop_b);
        if (out_sampled_b) *out_sampled_b = engine->sample_byte(engine->logits_buf.data(), 0.7f, 0.9f);
    }
}

void rtu_train_sequence(
    RTUEngine* engine,
    const uint8_t* bytes,
    int length,
    float lr,
    int reset_state,
    float* out_total_loss,
    float* out_ce_loss,
    float* out_latent_loss,
    float* out_variance_loss
) {
    if (!engine || length < 2) return;

    if (reset_state) engine->reset_state();

    float sum_tot = 0.0f, sum_ce = 0.0f, sum_lat = 0.0f, sum_var = 0.0f;
    float losses[5];
    int n_steps = length - 1;
    int accum_window = 32;

    engine->reset_gradients();

    for (int i = 0; i < n_steps; ++i) {
        int cur_b = static_cast<int>(bytes[i]);
        int next_b = static_cast<int>(bytes[i + 1]);
        bool is_end = (i == n_steps - 1);

        engine->forward_and_accumulate_gradients(cur_b, next_b, is_end, losses);

        sum_tot += losses[0];
        sum_ce += losses[1];
        sum_lat += losses[2];
        sum_var += losses[3];

        if ((i + 1) % accum_window == 0 || is_end) {
            engine->apply_all_adamw(lr);
        }
    }

    if (out_total_loss) *out_total_loss = sum_tot / static_cast<float>(n_steps);
    if (out_ce_loss) *out_ce_loss = sum_ce / static_cast<float>(n_steps);
    if (out_latent_loss) *out_latent_loss = sum_lat / static_cast<float>(n_steps);
    if (out_variance_loss) *out_variance_loss = sum_var / static_cast<float>(n_steps);
}

void rtu_get_weights(
    RTUEngine* engine,
    float* embed,
    float* decoder,
    float* stop_w,
    float* stop_b,
    float** decay_w,
    float** weights,
    float** gamma,
    float** beta,
    float** states
) {
    if (!engine) return;

    if (embed) std::memcpy(embed, engine->embed.data(), engine->embed.size() * sizeof(float));
    if (decoder) std::memcpy(decoder, engine->decoder.data(), engine->decoder.size() * sizeof(float));
    if (stop_w) std::memcpy(stop_w, engine->stop_w.data(), engine->stop_w.size() * sizeof(float));
    if (stop_b) *stop_b = engine->stop_b;

    for (int i = 0; i < engine->n_layers; ++i) {
        if (decay_w && decay_w[i]) std::memcpy(decay_w[i], engine->decay_w[i].data(), engine->dim * sizeof(float));
        if (weights && weights[i]) std::memcpy(weights[i], engine->weights[i].data(), engine->dim * engine->dim * sizeof(float));
        if (gamma && gamma[i]) std::memcpy(gamma[i], engine->gamma[i].data(), engine->dim * sizeof(float));
        if (beta && beta[i]) std::memcpy(beta[i], engine->beta[i].data(), engine->dim * sizeof(float));
        if (states && states[i]) std::memcpy(states[i], engine->layers[i].state.data(), engine->dim * sizeof(float));
    }
}

void rtu_set_weights(
    RTUEngine* engine,
    const float* embed,
    const float* decoder,
    const float* stop_w,
    const float* stop_b,
    const float* const* decay_w,
    const float* const* weights,
    const float* const* gamma,
    const float* const* beta,
    const float* const* states
) {
    if (!engine) return;

    if (embed) std::memcpy(engine->embed.data(), embed, engine->embed.size() * sizeof(float));
    if (decoder) std::memcpy(engine->decoder.data(), decoder, engine->decoder.size() * sizeof(float));
    if (stop_w) std::memcpy(engine->stop_w.data(), stop_w, engine->stop_w.size() * sizeof(float));
    if (stop_b) engine->stop_b = *stop_b;

    for (int i = 0; i < engine->n_layers; ++i) {
        if (decay_w && decay_w[i]) std::memcpy(engine->decay_w[i].data(), decay_w[i], engine->dim * sizeof(float));
        if (weights && weights[i]) std::memcpy(engine->weights[i].data(), weights[i], engine->dim * engine->dim * sizeof(float));
        if (gamma && gamma[i]) std::memcpy(engine->gamma[i].data(), gamma[i], engine->dim * sizeof(float));
        if (beta && beta[i]) std::memcpy(engine->beta[i].data(), beta[i], engine->dim * sizeof(float));
        if (states && states[i]) std::memcpy(engine->layers[i].state.data(), states[i], engine->dim * sizeof(float));
    }
}

} // extern "C"
