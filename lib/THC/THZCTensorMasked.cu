#include "THZCTensorMath.h"
#include "THZCGeneral.h"
#include "THZCDeviceUtils.cuh"
#include "THZCBlas.h"
#include "THZCTensorCopy.h"
#include "THZCTensorRandom.h"
#include "THZCApply.cuh"
#include "THZCReduce.cuh"

#include <thrust/device_ptr.h>
#include <thrust/scan.h>



#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

struct TensorMaskedFillOp {
  TensorMaskedFillOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* t, float* mask) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      *t = cx(value,0);
    }
  }

  float value;
};
struct cxTensorMaskedFillOp {
  TensorMaskedFillOp(cx v) : value(v) {}
  __device__ __forceinline__ void operator()(cx* t, float* mask) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      *t = value;
    }
  }

  cx value;
};

void THZCudaTensor_maskedFill(THCState* state,
                             THZCudaTensor *tensor, THZCudaTensor *mask, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, tensor, mask));
  THArgCheck(THZCudaTensor_nElement(state, tensor) ==
             THZCudaTensor_nElement(state, mask),
             2, "sizes do not match");

  if (!THZCudaTensor_pointwiseApply2(state, tensor, mask, TensorMaskedFillOp(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_maskedFill(THCState* state,
                             THZCudaTensor *tensor, THZCudaTensor *mask, cx value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, tensor, mask));
  THArgCheck(THZCudaTensor_nElement(state, tensor) ==
             THZCudaTensor_nElement(state, mask),
             2, "sizes do not match");

  if (!THZCudaTensor_pointwiseApply2(state, tensor, mask, cxTensorMaskedFillOp((cx)value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorMaskedCopyOp {
  TensorMaskedCopyOp(cx* s) : src(s) {}

  __device__ __forceinline__ void operator()(cx* out, float* mask, float* maskPrefixSum) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      // We've already checked that this offset is <= 2^24, so this is ok.
      *out = src[(int) *maskPrefixSum];
    }
  }

  // Where we are copying from
  cx* src;
};


void THZCudaTensor_maskedCopy(THCState* state,
                             THZCudaTensor *tensor, THCudaTensor *mask, THZCudaTensor *src)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, tensor, src, mask));
  long maskSize = THZCudaTensor_nElement(state, mask);
  long tensorSize = THZCudaTensor_nElement(state, tensor);
  long srcSize = THZCudaTensor_nElement(state, src);

  // Since we are performing a prefix sum of mask, it cannot exceed
  // the size allowed in consecutive integers in float32
  THArgCheck(maskSize <= (long) FLOAT32_MAX_CONSECUTIVE_INT,
             3, "mask nElements exceeds single-precision float "
             "consecutive integer precision size (2^24)");

  // `mask` and `tensor` must have the same number of elements
  THArgCheck(maskSize == tensorSize, 2,
             "mask and tensor must have the same number of elements");

  THZCudaTensor* contigMask = THZCudaTensor_newContiguous(state, mask);
  long oneElements = (long) THZCudaTensor_sumall(state, contigMask);

  // The number of `1` elements present in the mask must be <= the
  // number of elements available in `src`
  if (oneElements > srcSize) {
    THZCudaTensor_free(state, contigMask);
    THArgCheck(false, 2, "source nElements must be == mask `1` elements");
  }

  // Use a prefix sum to determine the copy locations of the masked elements
  THZCudaTensor* maskPrefixSum = THZCudaTensor_new(state);
  THZCudaTensor_resizeAs(state, maskPrefixSum, contigMask);

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  THZCudaTensor* contigSrc = THZCudaTensor_newContiguous(state, src);

  thrust::device_ptr<cx>
    maskData(THZCudaTensor_data(state, contigMask));
  thrust::device_ptr<cx>
    maskPrefixSumData(THZCudaTensor_data(state, maskPrefixSum));
  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
                         maskData,
                         maskData + THZCudaTensor_nElement(state, contigMask),
                         maskPrefixSumData);

  // update `tensor` where `mask` == 1 but pull from `src` at
  // maskPrefixSum
  bool status = THZCudaTensor_pointwiseApply3(
    state, tensor, contigMask, maskPrefixSum,
    TensorMaskedCopyOp(THZCudaTensor_data(state, contigSrc)));

  THZCudaTensor_free(state, contigSrc);
  THZCudaTensor_free(state, maskPrefixSum);
  THZCudaTensor_free(state, contigMask);

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THZCudaCheck(cudaGetLastError());
}

struct TensorMaskedSelectOp {
  TensorMaskedSelectOp(float* t) : out(t) {}
  __device__ __forceinline__ void operator()(float* mask, float* maskPrefixSum, float* in) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      out[(int) *maskPrefixSum] = *in;
    }
  }

  float* out;
};

void THZCudaTensor_maskedSelect(THCState* state,
                               THZCudaTensor *tensor, THZCudaTensor *src, THCudaTensor *mask)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, tensor, src, mask));
  THArgCheck(THZCudaTensor_nElement(state, mask) == THZCudaTensor_nElement(state, src),
             2, "sizes do not match");

  // Since we are performing a prefix sum of mask, it cannot exceed
  // the size allowed in consecutive integers in float32
  THArgCheck(THZCudaTensor_nElement(state, mask) <=
             (long) FLOAT32_MAX_CONSECUTIVE_INT,
             3, "mask nElements exceeds single-precision float "
             "consecutive integer precision size (2^24)");

  // Determine our output size
  THZCudaTensor* contigMask = THZCudaTensor_newContiguous(state, mask);
  long totalElements = (long) THZCudaTensor_sumall(state, contigMask);

  // This should be contiguous already, so no need to make it contig
  // for the apply kernel
  THZCudaTensor_resize1d(state, tensor, totalElements);

  // Use a prefix sum to determine the output locations of the masked elements
  THZCudaTensor* maskPrefixSum = THZCudaTensor_new(state);
  THZCudaTensor_resizeAs(state, maskPrefixSum, contigMask);

  thrust::device_ptr<float>
    maskData(THZCudaTensor_data(state, contigMask));
  thrust::device_ptr<float>
    maskPrefixSumData(THZCudaTensor_data(state, maskPrefixSum));
  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
                         maskData,
                         maskData + THZCudaTensor_nElement(state, contigMask),
                         maskPrefixSumData);

  // Then copy over the masked elements at their desired output index
  bool status = THZCudaTensor_pointwiseApply3(
    state, contigMask, maskPrefixSum,
    src, TensorMaskedSelectOp(THZCudaTensor_data(state, tensor)));

  THZCudaTensor_free(state, contigMask);
  THZCudaTensor_free(state, maskPrefixSum);

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_maskedFillByte(THCState* state, THZCudaTensor *tensor, THByteTensor *mask, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, tensor));
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THZCudaTensor* maskCuda = THZCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THZCudaTensor_copyByte(state, maskCuda, mask);
  THZCudaTensor_maskedFill(state, tensor, maskCuda, value);
  THZCudaTensor_free(state, maskCuda);
}

void THZCudaTensor_maskedCopyByte(THCState* state, THZCudaTensor *tensor, THByteTensor *mask, THZCudaTensor *src)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, tensor, src));
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THZCudaTensor* maskCuda = THZCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THZCudaTensor_copyByte(state, maskCuda, mask);
  THZCudaTensor_maskedCopy(state, tensor, maskCuda, src);
  THZCudaTensor_free(state, maskCuda);
}

void THZCudaTensor_maskedSelectByte(THCState* state, THZCudaTensor *tensor, THZCudaTensor *src, THByteTensor *mask)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, tensor, src));
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THZCudaTensor* maskCuda = THZCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THZCudaTensor_copyByte(state, maskCuda, mask);
  THZCudaTensor_maskedSelect(state, tensor, src, maskCuda);
  THZCudaTensor_free(state, maskCuda);
}
