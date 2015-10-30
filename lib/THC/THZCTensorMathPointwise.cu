#include "THZCTensorMath.h"
#include "THZCGeneral.h"
#include "THZCBlas.h"
#include "THZCTensorCopy.h"
#include "THZCApply.cuh"
#include "THZCReduce.cuh"

#include <cusp/complex.h>
typedef cusp::complex<float> ccx;

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC)                   \
  struct Tensor##NAME##Op {                                             \
    __device__ __forceinline__ void operator()(cx* out, cx* in) const { \
      ccx *o = (ccx*)out;                                               \
      ccx *i = (ccx*)in;                                                 \
      *o = CFUNC(*i);                                                   \
    }                                                                   \
                                                                        \
    __device__ __forceinline__ void operator()(cx* v) const {        \
      ccx *vc = (ccx*)v;                                                    \
      *vc = CFUNC(*vc);                                                   \
    }                                                                   \
  };                                                                    \
                                                                        \
  void THZCudaTensor_##NAME(THCState* state, THZCudaTensor* self_, THZCudaTensor* src) { \
    THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));                \
    if (self_ == src) {                                                 \
      if (!THZCudaTensor_pointwiseApply1(state, self_, Tensor##NAME##Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING); \
      }                                                                 \
    } else {                                                            \
      THZCudaTensor_resizeAs(state, self_, src);                         \
                                                                        \
      if (!THZCudaTensor_pointwiseApply2(state, self_, src, Tensor##NAME##Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING); \
      }                                                                 \
    }                                                                   \
                                                                        \
    THZCudaCheck(cudaGetLastError());                                    \
  }

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log, cusp::log)
// IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, log1p)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(exp, cusp::exp)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cos, cusp::cos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(acos, cusp::acos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cosh, cusp::cosh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sin, cusp::sin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(asin, cusp::asin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sinh, cusp::sinh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tan, cusp::tan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(atan, cusp::atan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tanh, cusp::tanh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sqrt, cusp::sqrt)
// IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(ceil, ceil)
// IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, floor)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(abs, cusp::abs)
// IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(round, roundf)

#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC

struct TensorAddOp {
  __device__ __forceinline__ void operator()(cx* out, cx* in) {
    ccx *o = (ccx*)out;
    ccx *i = (ccx*)in;
    *o += *i;
  }

  __device__ __forceinline__ void operator()(cx* out, cx* in1, cx* in2) {
    ccx *o = (ccx*)out;
    ccx *i1 = (ccx*)in1;
    ccx *i2 = (ccx*)in2;
    *o = *i1 + *i2;
  }
};

struct TensorCAddOp {
  TensorCAddOp(ccx v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    ccx *o = (ccx*)out;
    ccx *i = (ccx*)in;
    *out += val * *in;
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    ccx *o = (ccx*)out;
    ccx *i1 = (ccx*)in1;
    ccx *i2 = (ccx*)in2;
    *out = *in1 + val * *in2;
  }

  ccx val;
};

void THZCudaTensor_cadd(THCState *state, THZCudaTensor *self_, THZCudaTensor* src1, cx value, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THArgCheck(THZCudaTensor_nElement(state, src1) ==
             THZCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    if (value == 1.0f) {
      // self += src2
      if (!THZCudaTensor_pointwiseApply2(state, self_, src2, TensorAddOp())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self += value * src2
      if (!THZCudaTensor_pointwiseApply2(state, self_, src2, TensorCAddOp((ccx)value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src1);

    if (value == 1.0f) {
      // self = src1 + src2
      if (!THZCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorAddOp())) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    } else {
      // self = src1 + value * src2
      if (!THZCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorCAddOp((ccx)value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorMulOp {
  __device__ __forceinline__ void operator()(cx* out, cx* in) {
    *out = cuCmulf(*out,*in);
  }

  __device__ __forceinline__ void operator()(cx* out, cx* in1, cx* in2) {
    *out = cuCmulf(*in1,*in2);
  }
};

void THZCudaTensor_cmul(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THArgCheck(THZCudaTensor_nElement(state, src1) ==
             THZCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    if (!THZCudaTensor_pointwiseApply2(state, self_, src2, TensorMulOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src1);

    // self = src1 * src2
    if (!THZCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorMulOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorMaxOp {
  __device__ __forceinline__ void operator()(cx* out, cx* in) {
    ccx *o = (ccx*)out;
    ccx *i = (ccx*)in;
    *o = max(cusp::abs(*o), cusp::abs(*i));
  }

  __device__ __forceinline__ void operator()(cx* out, cx* in1, cx* in2) {
    ccx *o = (ccx*)out;
    ccx *i1 = (ccx*)in1;
    ccx *i2 = (ccx*)in2;
    *o = max(cusp::abs(*i1), cusp::abs(*i2));
  }
};

void THZCudaTensor_cmax(THCState *state, THZCudaTensor *self, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self, src1, src2));
  THArgCheck(THZCudaTensor_nElement(state, src1) ==
             THZCudaTensor_nElement(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THZCudaTensor_pointwiseApply2(state, self, src2, TensorMaxOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self, src1);
    if (!THZCudaTensor_pointwiseApply3(state, self, src1, src2, TensorMaxOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

struct TensorMinOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    ccx *o = (ccx*)out;
    ccx *i = (ccx*)in;
    *o = min(cusp::abs(*o), cusp::abs(*i));
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    ccx *o = (ccx*)out;
    ccx *i1 = (ccx*)in1;
    ccx *i2 = (ccx*)in2;
    *o = min(cusp::abs(*i1), cusp::abs(*i2));
  }
};

void THZCudaTensor_cmin(THCState *state, THZCudaTensor *self, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self, src1, src2));
  THArgCheck(THZCudaTensor_nElement(state, src1) ==
             THZCudaTensor_nElement(state, src2), 2, "sizes do not match");

  if (self == src1) {
    if (!THZCudaTensor_pointwiseApply2(state, self, src2, TensorMinOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self, src1);
    if (!THZCudaTensor_pointwiseApply3(state, self, src1, src2, TensorMinOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}
