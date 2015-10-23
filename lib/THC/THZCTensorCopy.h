#ifndef TH_CUDA_TENSOR_COPY_INC
#define TH_CUDA_TENSOR_COPY_INC

#include "THZCTensor.h"
#include "THZCGeneral.h"

THZC_API void THZCudaTensor_copy(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_copyByte(THCState *state, THZCudaTensor *self, THByteTensor *src);
THZC_API void THZCudaTensor_copyChar(THCState *state, THZCudaTensor *self, THCharTensor *src);
THZC_API void THZCudaTensor_copyShort(THCState *state, THZCudaTensor *self, THShortTensor *src);
THZC_API void THZCudaTensor_copyInt(THCState *state, THZCudaTensor *self, THIntTensor *src);
THZC_API void THZCudaTensor_copyLong(THCState *state, THZCudaTensor *self, THLongTensor *src);
THZC_API void THZCudaTensor_copyFloat(THCState *state, THZCudaTensor *self, THFloatTensor *src);
THZC_API void THZCudaTensor_copyDouble(THCState *state, THZCudaTensor *self, THDoubleTensor *src);

THZC_API void THByteTensor_copyCuda(THCState *state, THByteTensor *self, THZCudaTensor *src);
THZC_API void THCharTensor_copyCuda(THCState *state, THCharTensor *self, THZCudaTensor *src);
THZC_API void THShortTensor_copyCuda(THCState *state, THShortTensor *self, THZCudaTensor *src);
THZC_API void THIntTensor_copyCuda(THCState *state, THIntTensor *self, THZCudaTensor *src);
THZC_API void THLongTensor_copyCuda(THCState *state, THLongTensor *self, THZCudaTensor *src);
THZC_API void THFloatTensor_copyCuda(THCState *state, THFloatTensor *self, THZCudaTensor *src);
THZC_API void THDoubleTensor_copyCuda(THCState *state, THDoubleTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_copyCuda(THCState *state, THZCudaTensor *self, THZCudaTensor *src);

THZC_API void THZCudaTensor_copyAsyncFloat(THCState *state, THZCudaTensor *self, THFloatTensor *src);
THZC_API void THFloatTensor_copyAsyncCuda(THCState *state, THFloatTensor *self, THZCudaTensor *src);

#endif
