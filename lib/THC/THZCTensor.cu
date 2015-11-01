#include "THZCTensor.h"

cudaTextureObject_t THZCudaTensor_getTextureObject(THCState *state, THZCudaTensor *self)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  cudaTextureObject_t texObj;
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeLinear;
  resDesc.res.linear.devPtr = THZCudaTensor_dataCu(state, self);
  resDesc.res.linear.sizeInBytes = THZCudaTensor_nElement(state, self) * 4;
  resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0,
                                                  cudaChannelFormatKindFloat);
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess) {
    if (THZCudaTensor_nElement(state, self) > 2>>27)
      THError("Failed to create texture object, "
              "nElement:%ld exceeds 27-bit addressing required for tex1Dfetch. Cuda Error: %s",
              THZCudaTensor_nElement(state, self), cudaGetErrorString(errcode));
    else
      THError("Failed to create texture object: %s", cudaGetErrorString(errcode));
  }
  return texObj;
}

THZC_API int THZCudaTensor_getDevice(THCState* state, const THZCudaTensor* thc) {
  if (!thc->storage) return -1;
  cudaPointerAttributes attr;
  THZCudaCheck(cudaPointerGetAttributes(&attr, thc->storage->data));
  return attr.device;
}
