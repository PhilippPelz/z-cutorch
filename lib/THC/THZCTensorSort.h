#ifndef TH_CUDA_TENSOR_SORT_INC
#define TH_CUDA_TENSOR_SORT_INC

#include "THZCTensor.h"

THZC_API void THZCudaTensor_sort(THCState* state,
                               THZCudaTensor* sorted,
                               THZCudaTensor* indices,
                               THZCudaTensor* input,
                               int dim, int order);

#endif
