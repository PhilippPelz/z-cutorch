#include "THZCStorageCopy.h"
#include "THZCGeneral.h"

void THZCudaStorage_copyZFloat(THCState *state, THZCudaStorage *self, struct THZFloatStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THZCudaCheck(cudaMemcpy(self->data, src->data, self->size * sizeof(cx),
                          cudaMemcpyHostToDevice));
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC)                           \
  void THZCudaStorage_copy##TYPEC(THCState *state, THZCudaStorage *self, struct TH##TYPEC##Storage *src) \
  {                                                                     \
    THZFloatStorage *buffer;                                             \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    buffer = THZFloatStorage_newWithSize(src->size);                     \
    THZFloatStorage_copy##TYPEC(buffer, src);                            \
    THZCudaStorage_copyZFloat(state, self, buffer);                              \
    THZFloatStorage_free(buffer);                                        \
  }

TH_CUDA_STORAGE_IMPLEMENT_COPY(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Double)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Float)

void THZFloatStorage_copyZCuda(THCState *state, THZFloatStorage *self, struct THZCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THZCudaCheck(cudaMemcpy(self->data, src->data, self->size * sizeof(cx), cudaMemcpyDeviceToHost));
}
