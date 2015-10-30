#ifndef THZC_STORAGE_COPY_INC
#define THZC_STORAGE_COPY_INC

#include "THZCStorage.h"
#include "THZCGeneral.h"


/* Support for copy between different Storage types */

THZC_API void THZCudaStorage_rawCopy(THCState *state, THZCudaStorage *storage, float *src);
THZC_API void THZCudaStorage_copy(THCState *state, THZCudaStorage *storage, THZCudaStorage *src);
THZC_API void THZCudaStorage_copyZCuda(THCState *state, THZCudaStorage *self, THZCudaStorage *src);

THZC_API void THZCudaStorage_copyByte(THCState *state, THZCudaStorage *storage, struct THByteStorage *src);
THZC_API void THZCudaStorage_copyChar(THCState *state, THZCudaStorage *storage, struct THCharStorage *src);
THZC_API void THZCudaStorage_copyShort(THCState *state, THZCudaStorage *storage, struct THShortStorage *src);
THZC_API void THZCudaStorage_copyInt(THCState *state, THZCudaStorage *storage, struct THIntStorage *src);
THZC_API void THZCudaStorage_copyLong(THCState *state, THZCudaStorage *storage, struct THLongStorage *src);
THZC_API void THZCudaStorage_copyFloat(THCState *state, THZCudaStorage *storage, struct THFloatStorage *src);
THZC_API void THZCudaStorage_copyDouble(THCState *state, THZCudaStorage *storage, struct THDoubleStorage *src);
THZC_API void THZCudaStorage_copyZFloat(THCState *state, THZCudaStorage *storage, struct THDoubleStorage *src);

THZC_API void THZFloatStorage_copyZCuda(THCState *state, THZFloatStorage *self, struct THZCudaStorage *src);

#endif
