#ifndef RTU_ENGINE_H
#define RTU_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RTUEngine RTUEngine;

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
);

void rtu_free(RTUEngine* engine);

void rtu_reset_state(RTUEngine* engine);

void rtu_forward_byte(
    RTUEngine* engine,
    int b,
    int update_state,
    float* out_logits,
    float* out_stop_prob,
    float* out_latent
);

int rtu_sample_byte(
    RTUEngine* engine,
    const float* logits,
    float temperature,
    float top_p
);

void rtu_step_online_train(
    RTUEngine* engine,
    int cur_b,
    int next_b,
    int is_end,
    float lr,
    int* out_sampled_b,
    float* out_stop_prob,
    float* out_losses // [total, ce, latent, variance, stop]
);

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
);

// Weights synchronization with NumPy arrays
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
);

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
);

#ifdef __cplusplus
}
#endif

#endif // RTU_ENGINE_H
