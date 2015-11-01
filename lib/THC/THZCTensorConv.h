#ifndef TH_CUDA_TENSOR_CONV_INC
#define TH_CUDA_TENSOR_CONV_INC

#include "THC/THCTensor.h"
#include "THZCTensor.h"

struct THCState;

// THZC_API void THZCudaTensor_conv2Dmv(struct THCState *state,
//                                      THZCudaTensor *output, float beta,
//                                      THZCudaTensor *input,
//                                      THZCudaTensor *kernel, long srow,
//                                      long scol, const char *type);
// THZC_API void THZCudaTensor_conv2Dmm(struct THCState *state,
//                                      THZCudaTensor *output, float beta,
//                                      THZCudaTensor *input,
//                                      THZCudaTensor *kernel, long srow,
//                                      long scol, const char *type);

// THZC_API void THZCudaTensor_conv2DRevger(struct THCState *state,
//                                          THZCudaTensor *output, float beta,
//                                          float alpha, THZCudaTensor *input,
//                                          THZCudaTensor *kernel, long srow,
//                                          long scol);
// THZC_API void THZCudaTensor_conv2DRevgerm(struct THCState *state,
//                                           THZCudaTensor *output, float beta,
//                                           float alpha, THZCudaTensor *input,
//                                           THZCudaTensor *kernel, long srow,
//                                           long scol);

// THZC_API void THZCudaTensor_conv2Dmap(struct THCState *state,
//                                       THZCudaTensor *output,
//                                       THZCudaTensor *input,
//                                       THZCudaTensor *kernel, long stride_x,
//                                       long stride_y, THCudaTensor *table,
//                                       long fanin);

#endif
