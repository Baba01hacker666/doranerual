#ifndef DORANEURAL_LLM_ENGINE_H
#define DORANEURAL_LLM_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    int rope_type; // 0 = llama2c interleaved, 1 = HuggingFace split-half
    int eos_token_id; // EOS token ID (default 2)
    float rope_theta; // Base frequency theta (default 10000.0)
} LlamaCppConfig;

typedef struct {
    float* token_embedding_table;
    float* rms_att_weight;
    float* wq;
    float* wk;
    float* wv;
    float* wo;
    float* rms_ffn_weight;
    float* w1;
    float* w2;
    float* w3;
    float* rms_final_weight;
    float* wcls;
    int shared_classifier;
} LlamaCppWeights;

typedef struct LlamaCppEngine LlamaCppEngine;

// Create and free engine
LlamaCppEngine* llama_create(const LlamaCppConfig* config, LlamaCppWeights* weights);
void llama_free(LlamaCppEngine* engine);

// Forward single token with KV-cache
void llama_forward(LlamaCppEngine* engine, int token, int pos, float* out_logits);

// Forward single token with fused classifier argmax (zero logit memory bandwidth, for greedy search)
int llama_forward_argmax(LlamaCppEngine* engine, int token, int pos);

// Clear KV cache
void llama_reset_cache(LlamaCppEngine* engine);

// Multi-token autoregressive generation in C++
int llama_generate(
    LlamaCppEngine* engine,
    const int* prompt_tokens,
    int prompt_len,
    int max_new_tokens,
    float temperature,
    float top_p,
    int* out_tokens
);

// Extended generation supporting top-k, custom eos, and fused greedy mode
int llama_generate_ex(
    LlamaCppEngine* engine,
    const int* prompt_tokens,
    int prompt_len,
    int max_new_tokens,
    float temperature,
    float top_p,
    int top_k,
    int eos_token_id,
    int* out_tokens
);

// Training / Fine-tuning step with Cross Entropy & AdamW optimizer
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
);

// Native full-transformer forward/backpropagation through every decoder layer.
float llama_full_train_step(
    LlamaCppEngine* engine,
    const int* input_tokens,
    const int* target_tokens,
    int seq_len,
    float lr,
    float weight_decay,
    float beta1,
    float beta2,
    float eps,
    float grad_clip
);

// Threading controls
int llama_get_threads();
void llama_set_threads(int num_threads);

// Sampling
int llama_sample_token(LlamaCppEngine* engine, float temperature, float top_p);
int llama_sample_token_ex(LlamaCppEngine* engine, float temperature, float top_p, int top_k);

// Performance profiling hooks
void llama_set_profile(LlamaCppEngine* engine, int enable);
void llama_reset_profile(LlamaCppEngine* engine);
void llama_get_profile(LlamaCppEngine* engine, double* out_stats);

// Hardware Architecture Detection & Dispatch
const char* llama_get_cpu_arch();
const char* llama_get_cpu_backend();
void llama_print_cpu_features();

#ifdef __cplusplus
}
#endif

#endif // DORANEURAL_LLM_ENGINE_H
