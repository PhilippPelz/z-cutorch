#include "THZCStorage.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

void THZCudaStorage_fill(THCState *state, THZCudaStorage *self, float value)
{
  thrust::device_ptr<float> self_data(self->data);
  thrust::fill(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    self_data, self_data+self->size, value);
}

void THZCudaStorage_resize(THCState *state, THZCudaStorage *self, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(!(self->flag & TH_STORAGE_RESIZABLE))
    return;

  if(size == 0)
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      THZCudaCheck(THZCudaFree(state, self->data));
      THZCHeapUpdate(state, -self->size * sizeof(float));
    }
    self->data = NULL;
    self->size = 0;
  }
  else
  {
    float *data = NULL;
    // update heap *before* attempting malloc, to free space for the malloc
    THZCHeapUpdate(state, size * sizeof(float));
    cudaError_t err = THZCudaMalloc(state, (void**)(&data), size * sizeof(float));
    if(err != cudaSuccess) {
      THZCHeapUpdate(state, -size * sizeof(float));
    }
    THZCudaCheck(err);

    if (self->data) {
      THZCudaCheck(cudaMemcpyAsync(data,
                                  self->data,
                                  THMin(self->size, size) * sizeof(float),
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
      THZCudaCheck(THZCudaFree(state, self->data));
      THZCHeapUpdate(state, -self->size * sizeof(float));
    }

    self->data = data;
    self->size = size;
  }
}
