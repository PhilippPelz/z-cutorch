#ifndef THZC_STORAGE_COPY_INC
#define THZC_STORAGE_COPY_INC

#include "THZCStorage.h"
#include "THZCGeneral.h"

/* Support for copy between different Storage types */

THZC_API void THZCudaStorage_rawCopy(THCState *state, THZCudaStorage *storage, float *src);
THZC_API void THZCudaStorage_copy(THCState *state, THZCudaStorage *storage, THZCudaStorage *src);
THZC_API void THZCudaStorage_copyByte(THCState *state, THZCudaStorage *storage, struct THByteStorage *src);
THZC_API void THZCudaStorage_copyChar(THCState *state, THZCudaStorage *storage, struct THCharStorage *src);
THZC_API void THZCudaStorage_copyShort(THCState *state, THZCudaStorage *storage, struct THShortStorage *src);
THZC_API void THZCudaStorage_copyInt(THCState *state, THZCudaStorage *storage, struct THIntStorage *src);
THZC_API void THZCudaStorage_copyLong(THCState *state, THZCudaStorage *storage, struct THLongStorage *src);
THZC_API void THZCudaStorage_copyFloat(THCState *state, THZCudaStorage *storage, struct THFloatStorage *src);
THZC_API void THZCudaStorage_copyDouble(THCState *state, THZCudaStorage *storage, struct THDoubleStorage *src);

THZC_API void THByteStorage_copyCuda(THCState *state, THByteStorage *self, struct THZCudaStorage *src);
THZC_API void THCharStorage_copyCuda(THCState *state, THCharStorage *self, struct THZCudaStorage *src);
THZC_API void THShortStorage_copyCuda(THCState *state, THShortStorage *self, struct THZCudaStorage *src);
THZC_API void THIntStorage_copyCuda(THCState *state, THIntStorage *self, struct THZCudaStorage *src);
THZC_API void THLongStorage_copyCuda(THCState *state, THLongStorage *self, struct THZCudaStorage *src);
THZC_API void THFloatStorage_copyCuda(THCState *state, THFloatStorage *self, struct THZCudaStorage *src);
THZC_API void THDoubleStorage_copyCuda(THCState *state, THDoubleStorage *self, struct THZCudaStorage *src);
THZC_API void THZCudaStorage_copyCuda(THCState *state, THZCudaStorage *self, THZCudaStorage *src);

#endif
