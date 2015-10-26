#include "THZCApply.cuh"

static inline int curGPU() {
  int curDev;
  THZCudaCheck(cudaGetDevice(&curDev));
  return curDev;
}

THZC_API void
THZCudaTensor_copy(THCState* state, THZCudaTensor* dst, THZCudaTensor* src) {
  long totalElements = THZCudaTensor_nElement(state, dst);

  THArgCheck(totalElements == THZCudaTensor_nElement(state, src), 2,
             "sizes do not match");

  if (THZCudaTensor_nDimension(state, dst) == 0) {
    // Zero-dim tensor; copy nothing
    return;
  }

  // We can memcpy the memory if:
  // -both tensors are contiguous; or,
  // -there is only one element to copy; or,
  // -FIXME: if both tensors have matching size and stride arrays, and no
  // holes within (in other words, there is some permutation that can be applied
  // to the size/strides such that the resulting tensor is contiguous).
  bool srcContig = THZCudaTensor_isContiguous(state, src);
  bool dstContig = THZCudaTensor_isContiguous(state, dst);
  bool memcpyEligible = (srcContig && dstContig) || (totalElements == 1);

  int oldDev = curGPU();
  int srcDev = THZCudaTensor_getDevice(state, src);
  int dstDev = THZCudaTensor_getDevice(state, dst);

  // empirically, running the kernel on the device that holds the
  // non-contiguous tensor is faster by 5-10x
  int copyDev   = dstContig ? srcDev : dstDev;
  int remoteDev = dstContig ? dstDev : srcDev;

  if (srcDev == dstDev) {
    if (oldDev != srcDev) {
      THZCudaCheck(cudaSetDevice(srcDev));
    }
  } else {
    // synchronize remote device before copy
    cudaEvent_t dataReady;
    THZCudaCheck(cudaSetDevice(remoteDev));
    THZCudaCheck(cudaEventCreate(&dataReady));
    THZCudaCheck(cudaEventRecord(
                  dataReady,
                  THCState_getDeviceStream(state, remoteDev, THCState_getCurrentStreamIndex(state))));
    THZCudaCheck(cudaSetDevice(copyDev));
    THZCudaCheck(cudaStreamWaitEvent(
                  THCState_getDeviceStream(state, copyDev, THCState_getCurrentStreamIndex(state)),
                  dataReady, 0));
    THZCudaCheck(cudaEventDestroy(dataReady));
  }

  if (memcpyEligible) {
    THZCudaCheck(cudaMemcpyAsync(THZCudaTensor_data(state, dst),
                                THZCudaTensor_data(state, src),
                                totalElements * sizeof(cuComplex),
                                cudaMemcpyDeviceToDevice,
                                THCState_getCurrentStream(state)));
  } else {
      bool succ =
        THZCudaTensor_pointwiseApply2(state, dst, src, CopyOp<cuComplex>());
      THArgCheck(succ, 2, CUTORCH_DIM_WARNING);
  }

  if (srcDev != dstDev) {
    // synchronize remote device after copy
    cudaEvent_t doneCopying;
    THZCudaCheck(cudaEventCreate(&doneCopying));
    THZCudaCheck(cudaEventRecord(
                  doneCopying,
                  THCState_getDeviceStream(state, copyDev, THCState_getCurrentStreamIndex(state))));
    THZCudaCheck(cudaSetDevice(remoteDev));
    THZCudaCheck(cudaStreamWaitEvent(
                  THCState_getDeviceStream(state, remoteDev, THCState_getCurrentStreamIndex(state)),
                  doneCopying, 0));
    THZCudaCheck(cudaEventDestroy(doneCopying));
  }

  if (curGPU() != oldDev) {
    THZCudaCheck(cudaSetDevice(oldDev));
  }

  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}
