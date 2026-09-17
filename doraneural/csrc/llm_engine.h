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

// Threading controls
int llama_get_threads();
void llama_set_threads(int num_threads);

#ifdef __cplusplus
}
#endif

#endif // DORANEURAL_LLM_ENGINE_H
