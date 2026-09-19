#ifndef DORANEURAL_PULSE_ENGINE_H
#define DORANEURAL_PULSE_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;   // Default 256 for byte-level
    int dim;          // Representation dimension (e.g. 128, 256, 512)
    int hidden_dim;   // Hidden projection dimension (e.g. 256, 512, 1024)
    int n_layers;     // Number of encoder layers (e.g. 2, 4, 6)
    int n_classes;    // Number of routing domains (e.g. 8)
    float score_min;  // Minimum continuous score (default 0.0)
    float score_max;  // Maximum continuous score (default 10.0)
} PulseConfig;

typedef struct PulseEngine PulseEngine;

// Create and free high-performance C++ Decision Engine
PulseEngine* pulse_create(const PulseConfig* config);
void pulse_free(PulseEngine* engine);

// Set model weights from external memory (e.g. NumPy / Python)
void pulse_set_weights(
    PulseEngine* engine,
    const float* embed,      // [vocab_size * dim]
    const float* w_enc,      // [n_layers * dim * hidden_dim]
    const float* w_proj,     // [n_layers * hidden_dim * dim]
    const float* w_choice,   // [dim * n_classes]
    const float* b_choice,   // [n_classes]
    float temp_choice,
    const float* w_bool,     // [dim * 1]
    float b_bool,
    const float* w_score,    // [dim * 1]
    float b_score
);

// Get model weights into external memory
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
);

// High-speed Single-Pass Inference (< 0.02ms in C++)
void pulse_forward(
    PulseEngine* engine,
    const uint8_t* tokens,
    int token_len,
    float* out_choice_probs,  // [n_classes]
    float* out_bool_prob,     // [1]
    float* out_score          // [1]
);

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
);

// High-throughput parallel OpenMP batch training step
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
);

#ifdef __cplusplus
}
#endif

#endif // DORANEURAL_PULSE_ENGINE_H
