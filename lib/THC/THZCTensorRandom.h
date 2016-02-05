#ifndef TH_CUDA_TENSOR_RANDOM_INC
#define TH_CUDA_TENSOR_RANDOM_INC

#include "THZCTensor.h"

/* Generator */
typedef struct _Generator {
  struct curandStateMtgp32* gen_states;
  struct mtgp32_kernel_params *kernel_params;
  int initf;
  unsigned long initial_seed;
} Generator;

typedef struct THCRNGState {
  /* One generator per GPU */
  Generator* gen;
  Generator* current_gen;
  int num_devices;
} THCRNGState;

struct THCState;

THZC_API void THZCRandom_init(struct THCState *state, int num_devices, int current_device);
THZC_API void THZCRandom_shutdown(struct THCState *state);
THZC_API void THZCRandom_setGenerator(struct THCState *state, int device);
THZC_API unsigned long THZCRandom_seed(struct THCState *state);
THZC_API unsigned long THZCRandom_seedAll(struct THCState *state);
THZC_API void THZCRandom_manualSeed(struct THCState *state, unsigned long the_seed_);
THZC_API void THZCRandom_manualSeedAll(struct THCState *state, unsigned long the_seed_);
THZC_API unsigned long THZCRandom_initialSeed(struct THCState *state);
THZC_API void THZCRandom_getRNGState(struct THCState *state, THByteTensor *rng_state);
THZC_API void THZCRandom_setRNGState(struct THCState *state, THByteTensor *rng_state);
THZC_API void THZCudaTensor_geometric(struct THCState *state, THZCudaTensor *self, double p);
THZC_API void THZCudaTensor_bernoulli(struct THCState *state, THZCudaTensor *self, double p);
THZC_API void THZCudaTensor_uniform(struct THCState *state, THZCudaTensor *self, double a, double b);
THZC_API void THZCudaTensor_normal(struct THCState *state, THZCudaTensor *self, double mean, double stdv);
THZC_API void THZCudaTensor_exponential(struct THCState *state, THZCudaTensor *self, double lambda);
THZC_API void THZCudaTensor_cauchy(struct THCState *state, THZCudaTensor *self, double median, double sigma);
THZC_API void THZCudaTensor_logNormal(struct THCState *state, THZCudaTensor *self, double mean, double stdv);

THZC_API void THZCudaTensor_multinomial(struct THCState *state, THZCudaTensor *self, THZCudaTensor *prob_dist, int n_sample, int with_replacement);

#endif
