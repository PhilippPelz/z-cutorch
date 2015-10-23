#include "THZCTensorCopy.h"
#include "THZCGeneral.h"
#include "THZCTensor.h"

/* specific methods */

void THZCudaTensor_copyFloat(THCState *state, THZCudaTensor *self, struct THFloatTensor *src)
{
  THArgCheck(THZCudaTensor_nElement(state, self) == THFloatTensor_nElement(src), 2, "sizes do not match");

  {
    THZCudaTensor *selfc = THZCudaTensor_newContiguous(state, self);
    src = THFloatTensor_newContiguous(src);

    THZCudaCheck(cudaMemcpy(THZCudaTensor_data(state, selfc),
                           THFloatTensor_data(src),
                           THFloatTensor_nElement(src) * sizeof(float),
                           cudaMemcpyHostToDevice));

    THFloatTensor_free(src);
    THZCudaTensor_freeCopyTo(state, selfc, self);
  }
}

/* everything comes down to copy to a tensor of floats */
#define IMPLEMENT_TH_CUDA_TENSOR_COPY(TYPEC)                            \
void THZCudaTensor_copy##TYPEC(THCState *state, THZCudaTensor *self, struct TH##TYPEC##Tensor *src) \
{                                                                       \
  THArgCheck(THZCudaTensor_nElement(state, self) == TH##TYPEC##Tensor_nElement(src), 2, "sizes do not match"); \
                                                                        \
  {                                                                     \
    THLongStorage *size = TH##TYPEC##Tensor_newSizeOf(src);             \
    THFloatTensor *srcf = THFloatTensor_newWithSize(size, NULL);        \
                                                                        \
    THFloatTensor_copy##TYPEC(srcf, src);                               \
    THZCudaTensor_copyFloat(state, self, srcf);                                 \
                                                                        \
    THLongStorage_free(size);                                           \
    THFloatTensor_free(srcf);                                           \
  }                                                                     \
}

IMPLEMENT_TH_CUDA_TENSOR_COPY(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Double)

/* copyCuda */

void THFloatTensor_copyCuda(THCState *state, THFloatTensor *self, struct THZCudaTensor *src)
{
  THArgCheck(THFloatTensor_nElement(self) == THZCudaTensor_nElement(state, src), 2, "sizes do not match");

  {
    THFloatTensor *selfc = THFloatTensor_newContiguous(self);
    src = THZCudaTensor_newContiguous(state, src);

    THZCudaCheck(cudaMemcpy(THFloatTensor_data(selfc),
                           THZCudaTensor_data(state, src),
                           THZCudaTensor_nElement(state, src) * sizeof(float),
                           cudaMemcpyDeviceToHost));

    THZCudaTensor_free(state, src);
    THFloatTensor_freeCopyTo(selfc, self);
  }
}

#define IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(TYPEC)                                                          \
  void TH##TYPEC##Tensor_copyCuda(THCState *state, TH##TYPEC##Tensor *self, struct THZCudaTensor *src) \
  {                                                                                                      \
    THArgCheck(TH##TYPEC##Tensor_nElement(self) == THZCudaTensor_nElement(state, src), 2, "sizes do not match"); \
                                                                                                         \
    {                                                                                                    \
      THLongStorage *size = THZCudaTensor_newSizeOf(state, src);                                          \
      THFloatTensor *srcf = THFloatTensor_newWithSize(size, NULL);                                       \
                                                                                                         \
      THFloatTensor_copyCuda(state, srcf, src);                                                          \
      TH##TYPEC##Tensor_copyFloat(self, srcf);                                                           \
                                                                                                         \
      THLongStorage_free(size);                                                                          \
      THFloatTensor_free(srcf);                                                                          \
    }                                                                                                    \
  }

IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY_TO(Double)

void THZCudaTensor_copyCuda(THCState *state, THZCudaTensor *self, THZCudaTensor *src)
{
  THZCudaTensor_copy(state, self, src);
}

void THZCudaTensor_copyAsyncFloat(THCState *state, THZCudaTensor *self, struct THFloatTensor *src)
{
  THArgCheck(THZCudaTensor_nElement(state, self) == THFloatTensor_nElement(src), 2, "sizes do not match");
  THArgCheck(THZCudaTensor_isContiguous(state, self), 2, "Target tensor must be contiguous");
  THArgCheck(THFloatTensor_isContiguous(src), 3, "Source tensor must be contiguous");

  if (THZCudaTensor_nElement(state, self) == 0) return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THZCudaTensor_getDevice(state, self);
  int currentDevice;
  THZCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THZCudaCheck(cudaSetDevice(tensorDevice));
  }

  THZCudaCheck(cudaMemcpyAsync(THZCudaTensor_data(state, self),
                              THFloatTensor_data(src),
                              THFloatTensor_nElement(src) * sizeof(float),
                              cudaMemcpyHostToDevice,
                              THCState_getDeviceStream(state, tensorDevice,
                                                       THCState_getCurrentStreamIndex(state))));

  if (currentDevice != tensorDevice) {
    THZCudaCheck(cudaSetDevice(currentDevice));
  }
}

void THFloatTensor_copyAsyncCuda(THCState *state, THFloatTensor *self, struct THZCudaTensor *src)
{
  THArgCheck(THFloatTensor_nElement(self) == THZCudaTensor_nElement(state, src), 2, "sizes do not match");
  THArgCheck(THFloatTensor_isContiguous(self), 2, "Target tensor must be contiguous");
  THArgCheck(THZCudaTensor_isContiguous(state, src), 3, "Source tensor must be contiguous");

  if (THFloatTensor_nElement(self) == 0) return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THZCudaTensor_getDevice(state, src);
  int currentDevice;
  THZCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THZCudaCheck(cudaSetDevice(tensorDevice));
  }

  THZCudaCheck(cudaMemcpyAsync(THFloatTensor_data(self),
                              THZCudaTensor_data(state, src),
                              THZCudaTensor_nElement(state, src) * sizeof(float),
                              cudaMemcpyDeviceToHost,
                              THCState_getDeviceStream(state, tensorDevice,
                                                       THCState_getCurrentStreamIndex(state))));

  if (currentDevice != tensorDevice) {
    THZCudaCheck(cudaSetDevice(currentDevice));
  }
}
