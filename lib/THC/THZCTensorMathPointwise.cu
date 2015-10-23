#include "THZCTensorMath.h"
#include "THZCGeneral.h"
#include "THZCBlas.h"
#include "THZCTensorCopy.h"
#include "THZCApply.cuh"
#include "THZCReduce.cuh"

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC)                   \
  struct Tensor##NAME##Op {                                             \
    __device__ __forceinline__ void operator()(float* out, float* in) const { \
      *out = CFUNC(*in);                                                \
    }                                                                   \
                                                                        \
    __device__ __forceinline__ void operator()(float* v) const {        \
      *v = CFUNC(*v);                                                   \
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

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log, log)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log1p, log1p)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(exp, exp)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cos, cos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(acos, acos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cosh, cosh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sin, sin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(asin, asin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sinh, sinh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tan, tan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(atan, atan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tanh, tanh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sqrt, sqrt)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(ceil, ceil)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(floor, floor)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(abs, fabs)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(round, roundf)

#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC

struct TensorAddOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out += *in;
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = *in1 + *in2;
  }
};

struct TensorCAddOp {
  TensorCAddOp(float v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out += val * *in;
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = *in1 + val * *in2;
  }

  float val;
};

void THZCudaTensor_cadd(THCState *state, THZCudaTensor *self_, THZCudaTensor* src1, float value, THZCudaTensor *src2)
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
      if (!THZCudaTensor_pointwiseApply2(state, self_, src2, TensorCAddOp(value))) {
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
      if (!THZCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorCAddOp(value))) {
        THArgCheck(false, 2, CUTORCH_DIM_WARNING);
      }
    }
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorMulOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out *= *in;
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = *in1 * *in2;
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
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = max(*out, *in);
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = max(*in1, *in2);
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
    *out = min(*out, *in);
  }

  __device__ __forceinline__ void operator()(float* out, float* in1, float* in2) {
    *out = min(*in1, *in2);
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

struct TensorMaxValueOp {
  TensorMaxValueOp(float v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out) {
    *out = max(*out, val);
  }

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = max(*in, val);
  }

  float val;
};

void THZCudaTensor_cmaxValue(THCState *state, THZCudaTensor *self, THZCudaTensor *src, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self, src));

  if (self == src) {
    if (!THZCudaTensor_pointwiseApply1(state, self, TensorMaxValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self, src);
    if (!THZCudaTensor_pointwiseApply2(state, self, src, TensorMaxValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}

struct TensorMinValueOp {
  TensorMinValueOp(float v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out) {
    *out = min(*out, val);
  }

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = min(*in, val);
  }

  float val;
};

void THZCudaTensor_cminValue(THCState *state, THZCudaTensor *self, THZCudaTensor *src, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self, src));

  if (self == src) {
    if (!THZCudaTensor_pointwiseApply1(state, self, TensorMinValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self, src);
    if (!THZCudaTensor_pointwiseApply2(state, self, src, TensorMinValueOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }
}
