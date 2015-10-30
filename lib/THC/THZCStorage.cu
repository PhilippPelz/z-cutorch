#include "THZCStorage.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

#include <cusp/complex.h>
typedef cusp::complex<float> ccx;

void THZCudaStorage_fill(THCState *state, THZCudaStorage *self, cx value)
{
  thrust::device_ptr<ccx> self_data((ccx*)self->data);
  thrust::fill(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    self_data, self_data+self->size, ccx(crealf(value),cimagf(value)));
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
      THZCHeapUpdate(state, -self->size * sizeof(cux));
    }
    self->data = NULL;
    self->size = 0;
  }
  else
  {
    cux *data = NULL;
    // update heap *before* attempting malloc, to free space for the malloc
    THZCHeapUpdate(state, size * sizeof(cux));
    cudaError_t err = THZCudaMalloc(state, (void**)(&data), size * sizeof(cux));
    if(err != cudaSuccess) {
      THZCHeapUpdate(state, -size * sizeof(cux));
    }
    THZCudaCheck(err);

    if (self->data) {
      THZCudaCheck(cudaMemcpyAsync(data,
                                  self->data,
                                  THMin(self->size, size) * sizeof(cux),
                                  cudaMemcpyDeviceToDevice,
                                  THCState_getCurrentStream(state)));
      THZCudaCheck(THZCudaFree(state, self->data));
      THZCHeapUpdate(state, -self->size * sizeof(cux));
    }

    self->data = data;
    self->size = size;
  }
}
