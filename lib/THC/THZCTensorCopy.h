#ifndef TH_CUDA_TENSOR_COPY_INC
#define TH_CUDA_TENSOR_COPY_INC

#include "THZCTensor.h"
// #include "THZCGeneral.h"

THZC_API void THZCudaTensor_copy(THCState *state, THZCudaTensor *self,
                                 THZCudaTensor *src);
THZC_API void THZCudaTensor_copyIm(THCState *state, THZCudaTensor *self, THCudaTensor *src);
THZC_API void THZCudaTensor_copyRe(THCState *state, THZCudaTensor *self, THCudaTensor *src);
THZC_API void THZCudaTensor_copyByte(THCState *state, THZCudaTensor *self,
                                     THByteTensor *src);
THZC_API void THZCudaTensor_copyChar(THCState *state, THZCudaTensor *self,
                                     THCharTensor *src);
THZC_API void THZCudaTensor_copyShort(THCState *state, THZCudaTensor *self,
                                      THShortTensor *src);
THZC_API void THZCudaTensor_copyInt(THCState *state, THZCudaTensor *self,
                                    THIntTensor *src);
THZC_API void THZCudaTensor_copyLong(THCState *state, THZCudaTensor *self,
                                     THLongTensor *src);
THZC_API void THZCudaTensor_copyFloat(THCState *state, THZCudaTensor *self,
                                      THFloatTensor *src);
THZC_API void THZCudaTensor_copyDouble(THCState *state, THZCudaTensor *self,
                                       THDoubleTensor *src);
THZC_API void THZCudaTensor_copyZFloat(THCState *state, THZCudaTensor *self,
                                       THZFloatTensor *src);

THZC_API void THZFloatTensor_copyZCuda(THCState *state, THZFloatTensor *self,
                                       THZCudaTensor *src);
THZC_API void THZCudaTensor_copyZCuda(THCState *state, THZCudaTensor *self,
                                      THZCudaTensor *src);

THZC_API void THZCudaTensor_copyAsyncZFloat(THCState *state,
                                            THZCudaTensor *self,
                                            THZFloatTensor *src);
THZC_API void THZFloatTensor_copyAsyncZCuda(THCState *state,
                                            THZFloatTensor *self,
                                            THZCudaTensor *src);

#endif
