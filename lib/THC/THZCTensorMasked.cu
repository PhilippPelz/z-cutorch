#include "THC/THCTensorMath.h"
#include "THZCTensorMath.h"
#include "THZCGeneral.h"
#include "THZCGeneral.cuh"
#include "THZCDeviceUtils.cuh"
#include "THZCBlas.h"
#include "THZCTensorCopy.h"
//#include "THZCTensorRandom.h"
#include "THZCApply.cuh"
#include "THZCReduce.cuh"


#include <thrust/device_ptr.h>
#include <thrust/scan.h>

// #include <thrust/complex.h>
// typedef thrust::complex<float> ccx;

#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif


struct cxTensorMaskedFillOp {
  cxTensorMaskedFillOp(ccx v) : value(v) {}
  __device__ __forceinline__ void operator()(ccx* t, float* mask) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      *t = value;
    }
  }

  ccx value;
};

void THZCudaTensor_maskedFill(THCState* state, THZCudaTensor *tensor, THCudaTensor *mask, cx value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, tensor, mask));
  THArgCheck(THZCudaTensor_nElement(state, tensor) ==
             THCudaTensor_nElement(state, mask),
             2, "sizes do not match");

  if (!THZCudaTensor_pointwiseApply2ZF(state, tensor, mask, cxTensorMaskedFillOp(toCcx(value)))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorMaskedCopyOp {
  TensorMaskedCopyOp(ccx* s) : src(s) {}

  __device__ __forceinline__ void operator()(float* mask, float* maskPrefixSum, ccx* out) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      // We've already checked that this offset is <= 2^24, so this is ok.
      *out = src[(int) *maskPrefixSum];
    }
  }

  // Where we are copying from
  ccx* src;
};


void THZCudaTensor_maskedCopy(THCState* state, THZCudaTensor *tensor, THCudaTensor *mask, THZCudaTensor *src)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, tensor, src, mask));
  long maskSize = THCudaTensor_nElement(state, mask);
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

  THCudaTensor* contigMask = THCudaTensor_newContiguous(state, mask);
  long oneElements = (long) THCudaTensor_sumall(state, contigMask);

  // The number of `1` elements present in the mask must be <= the
  // number of elements available in `src`
  if (oneElements > srcSize) {
    THCudaTensor_free(state, contigMask);
    THArgCheck(false, 2, "source nElements must be == mask `1` elements");
  }

  // Use a prefix sum to determine the copy locations of the masked elements
  THCudaTensor* maskPrefixSum = THCudaTensor_new(state);
  THCudaTensor_resizeAs(state, maskPrefixSum, contigMask);

  // We are getting elements from `src` based on an offset from
  // `maskPrefixSum`, so that should be made contiguous too
  THZCudaTensor* contigSrc = THZCudaTensor_newContiguous(state, src);

  thrust::device_ptr<float>
    maskData(THCudaTensor_data(state, contigMask));
  thrust::device_ptr<float>
    maskPrefixSumData(THCudaTensor_data(state, maskPrefixSum));
  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
     maskData,
     maskData + THCudaTensor_nElement(state, contigMask),
     maskPrefixSumData);

  // update `tensor` where `mask` == 1 but pull from `src` at
  // maskPrefixSum
  bool status = THZCudaTensor_pointwiseApply3FFZ(
    state, contigMask, maskPrefixSum, tensor,
    TensorMaskedCopyOp((ccx*)THZCudaTensor_data(state, contigSrc)));

  THZCudaTensor_free(state, contigSrc);
  THCudaTensor_free(state, maskPrefixSum);
  THCudaTensor_free(state, contigMask);

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THZCudaCheck(cudaGetLastError());
}

struct TensorMaskedSelectOp {
  TensorMaskedSelectOp(ccx* t) : out(t) {}
  __device__ __forceinline__ void operator()(float* mask, float* maskPrefixSum,ccx* in) {
    // Really mask should be `0` or `1` but we can't propagate errors here.
    if (*mask != 0.0f) {
      out[(int) *maskPrefixSum] = *in;
    }
  }

  ccx* out;
};

void THZCudaTensor_maskedSelect(THCState* state,
                               THZCudaTensor *tensor, THZCudaTensor *src, THCudaTensor *mask)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, tensor, src, mask));
  THArgCheck(THCudaTensor_nElement(state, mask) == THZCudaTensor_nElement(state, src),
             2, "sizes do not match");

  // Since we are performing a prefix sum of mask, it cannot exceed
  // the size allowed in consecutive integers in float32
  THArgCheck(THCudaTensor_nElement(state, mask) <=
             (long) FLOAT32_MAX_CONSECUTIVE_INT,
             3, "mask nElements exceeds single-precision float "
             "consecutive integer precision size (2^24)");

  // Determine our output size
  THCudaTensor* contigMask = THCudaTensor_newContiguous(state, mask);
  long totalElements = (long) THCudaTensor_sumall(state, contigMask);

  // This should be contiguous already, so no need to make it contig
  // for the apply kernel
  THZCudaTensor_resize1d(state, tensor, totalElements);

  // Use a prefix sum to determine the output locations of the masked elements
  THCudaTensor* maskPrefixSum = THCudaTensor_new(state);
  THCudaTensor_resizeAs(state, maskPrefixSum, contigMask);

  thrust::device_ptr<float>
    maskData(THCudaTensor_data(state, contigMask));
  thrust::device_ptr<float>
    maskPrefixSumData(THCudaTensor_data(state, maskPrefixSum));
  thrust::exclusive_scan(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
                         maskData,
                         maskData + THCudaTensor_nElement(state, contigMask),
                         maskPrefixSumData);

  // Then copy over the masked elements at their desired output index
  bool status = THZCudaTensor_pointwiseApply3FFZ(
    state, contigMask, maskPrefixSum,
    src, TensorMaskedSelectOp((ccx*)THZCudaTensor_data(state, tensor)));

  THCudaTensor_free(state, contigMask);
  THCudaTensor_free(state, maskPrefixSum);

  THArgCheck(status, 2, CUTORCH_DIM_WARNING);
  THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_maskedFillByte(THCState* state, THZCudaTensor *tensor, THByteTensor *mask, cx value)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, tensor));
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THCudaTensor* maskCuda = THCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THCudaTensor_copyByte(state, maskCuda, mask);
  THZCudaTensor_maskedFill(state, tensor, maskCuda, value);
  THCudaTensor_free(state, maskCuda);
}

void THZCudaTensor_maskedCopyByte(THCState* state, THZCudaTensor *tensor, THByteTensor *mask, THZCudaTensor *src)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, tensor, src));
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THCudaTensor* maskCuda = THCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THCudaTensor_copyByte(state, maskCuda, mask);
  THZCudaTensor_maskedCopy(state, tensor, maskCuda, src);
  THCudaTensor_free(state, maskCuda);
}

void THZCudaTensor_maskedSelectByte(THCState* state, THZCudaTensor *tensor, THZCudaTensor *src, THByteTensor *mask)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, tensor, src));
  THLongStorage* maskSize = THByteTensor_newSizeOf(mask);
  THCudaTensor* maskCuda = THCudaTensor_newWithSize(state, maskSize, NULL);
  THLongStorage_free(maskSize);
  THCudaTensor_copyByte(state, maskCuda, mask);
  THZCudaTensor_maskedSelect(state, tensor, src, maskCuda);
  THCudaTensor_free(state, maskCuda);
}
