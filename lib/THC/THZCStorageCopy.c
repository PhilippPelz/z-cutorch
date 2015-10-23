#include "THZCStorageCopy.h"
#include "THZCGeneral.h"

void THZCudaStorage_copyFloat(THCState *state, THZCudaStorage *self, struct THFloatStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THZCudaCheck(cudaMemcpy(self->data, src->data, self->size * sizeof(float), cudaMemcpyHostToDevice));
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPY(TYPEC)                           \
  void THZCudaStorage_copy##TYPEC(THCState *state, THZCudaStorage *self, struct TH##TYPEC##Storage *src) \
  {                                                                     \
    THFloatStorage *buffer;                                             \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    buffer = THFloatStorage_newWithSize(src->size);                     \
    THFloatStorage_copy##TYPEC(buffer, src);                            \
    THZCudaStorage_copyFloat(state, self, buffer);                              \
    THFloatStorage_free(buffer);                                        \
  }

TH_CUDA_STORAGE_IMPLEMENT_COPY(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPY(Double)

void THFloatStorage_copyCuda(THCState *state, THFloatStorage *self, struct THZCudaStorage *src)
{
  THArgCheck(self->size == src->size, 2, "size does not match");
  THZCudaCheck(cudaMemcpy(self->data, src->data, self->size * sizeof(float), cudaMemcpyDeviceToHost));
}

#define TH_CUDA_STORAGE_IMPLEMENT_COPYTO(TYPEC)                           \
  void TH##TYPEC##Storage_copyCuda(THCState *state, TH##TYPEC##Storage *self, struct THZCudaStorage *src) \
  {                                                                     \
    THFloatStorage *buffer;                                             \
    THArgCheck(self->size == src->size, 2, "size does not match");      \
    buffer = THFloatStorage_newWithSize(src->size);                     \
    THFloatStorage_copyCuda(state, buffer, src);                               \
    TH##TYPEC##Storage_copyFloat(self, buffer);                         \
    THFloatStorage_free(buffer);                                        \
  }

TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Byte)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Char)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Short)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Int)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Long)
TH_CUDA_STORAGE_IMPLEMENT_COPYTO(Double)
