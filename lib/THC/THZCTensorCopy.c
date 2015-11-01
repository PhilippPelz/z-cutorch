#include "THZCTensorCopy.h"
#include "THZCGeneral.h"
#include "THZCTensor.h"

/* specific methods */

void THZCudaTensor_copyZFloat(THCState *state, THZCudaTensor *self,
                              struct THZFloatTensor *src) {
  THArgCheck(THZCudaTensor_nElement(state, self) ==
                 THZFloatTensor_nElement(src),
             2, "sizes do not match");

  {
    THZCudaTensor *selfc = THZCudaTensor_newContiguous(state, self);
    src = THZFloatTensor_newContiguous(src);

    THZCudaCheck(cudaMemcpy(
        THZCudaTensor_data(state, selfc), THZFloatTensor_data(src),
        THZFloatTensor_nElement(src) * sizeof(cx), cudaMemcpyHostToDevice));

    THZFloatTensor_free(src);
    THZCudaTensor_freeCopyTo(state, selfc, self);
  }
}

/* everything comes down to copy to a tensor of floats */
#define IMPLEMENT_TH_CUDA_TENSOR_COPY(TYPEC)                                   \
  void THZCudaTensor_copy##TYPEC(THCState *state, THZCudaTensor *self,         \
                                 struct TH##TYPEC##Tensor *src) {              \
    THArgCheck(THZCudaTensor_nElement(state, self) ==                          \
                   TH##TYPEC##Tensor_nElement(src),                            \
               2, "sizes do not match");                                       \
                                                                               \
    {                                                                          \
      THLongStorage *size = TH##TYPEC##Tensor_newSizeOf(src);                  \
      THZFloatTensor *srcf = THZFloatTensor_newWithSize(size, NULL);           \
                                                                               \
      THZFloatTensor_copy##TYPEC(srcf, src);                                   \
      THZCudaTensor_copyZFloat(state, self, srcf);                             \
                                                                               \
      THLongStorage_free(size);                                                \
      THZFloatTensor_free(srcf);                                               \
    }                                                                          \
  }

IMPLEMENT_TH_CUDA_TENSOR_COPY(Byte)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Char)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Short)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Int)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Long)
IMPLEMENT_TH_CUDA_TENSOR_COPY(Double)

/* copyCuda */

void THZFloatTensor_copyZCuda(THCState *state, THZFloatTensor *self,
                              struct THZCudaTensor *src) {
  THArgCheck(THZFloatTensor_nElement(self) ==
                 THZCudaTensor_nElement(state, src),
             2, "sizes do not match");

  {
    THZFloatTensor *selfc = THZFloatTensor_newContiguous(self);
    src = THZCudaTensor_newContiguous(state, src);

    THZCudaCheck(cudaMemcpy(THZFloatTensor_data(selfc),
                            THZCudaTensor_data(state, src),
                            THZCudaTensor_nElement(state, src) * sizeof(cux),
                            cudaMemcpyDeviceToHost));

    THZCudaTensor_free(state, src);
    THZFloatTensor_freeCopyTo(selfc, self);
  }
}

void THZCudaTensor_copyZCuda(THCState *state, THZCudaTensor *self,
                             THZCudaTensor *src) {
  THZCudaTensor_copy(state, self, src);
}

void THZCudaTensor_copyAsyncZFloat(THCState *state, THZCudaTensor *self,
                                   struct THZFloatTensor *src) {
  THArgCheck(THZCudaTensor_nElement(state, self) ==
                 THZFloatTensor_nElement(src),
             2, "sizes do not match");
  THArgCheck(THZCudaTensor_isContiguous(state, self), 2,
             "Target tensor must be contiguous");
  THArgCheck(THZFloatTensor_isContiguous(src), 3,
             "Source tensor must be contiguous");

  if (THZCudaTensor_nElement(state, self) == 0)
    return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THZCudaTensor_getDevice(state, self);
  int currentDevice;
  THZCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THZCudaCheck(cudaSetDevice(tensorDevice));
  }

  THZCudaCheck(cudaMemcpyAsync(
      THZCudaTensor_data(state, self), THZFloatTensor_data(src),
      THZFloatTensor_nElement(src) * sizeof(cux), cudaMemcpyHostToDevice,
      THCState_getDeviceStream(state, tensorDevice,
                               THCState_getCurrentStreamIndex(state))));

  if (currentDevice != tensorDevice) {
    THZCudaCheck(cudaSetDevice(currentDevice));
  }
}

void THZFloatTensor_copyAsyncZCuda(THCState *state, THZFloatTensor *self,
                                   struct THZCudaTensor *src) {
  THArgCheck(THZFloatTensor_nElement(self) ==
                 THZCudaTensor_nElement(state, src),
             2, "sizes do not match");
  THArgCheck(THZFloatTensor_isContiguous(self), 2,
             "Target tensor must be contiguous");
  THArgCheck(THZCudaTensor_isContiguous(state, src), 3,
             "Source tensor must be contiguous");

  if (THZFloatTensor_nElement(self) == 0)
    return;

  // Perform the copy wrt the current stream on the CudaTensor's device.
  int tensorDevice = THZCudaTensor_getDevice(state, src);
  int currentDevice;
  THZCudaCheck(cudaGetDevice(&currentDevice));

  if (currentDevice != tensorDevice) {
    THZCudaCheck(cudaSetDevice(tensorDevice));
  }

  THZCudaCheck(cudaMemcpyAsync(
      THZFloatTensor_data(self), THZCudaTensor_data(state, src),
      THZCudaTensor_nElement(state, src) * sizeof(cx), cudaMemcpyDeviceToHost,
      THCState_getDeviceStream(state, tensorDevice,
                               THCState_getCurrentStreamIndex(state))));

  if (currentDevice != tensorDevice) {
    THZCudaCheck(cudaSetDevice(currentDevice));
  }
}
