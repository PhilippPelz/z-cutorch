#include "THZCTensorMath.h"
#include "THZCGeneral.h"
#include "THZCBlas.h"
#include "THZCTensorCopy.h"
#include "THZCTensorRandom.h"
#include "THZCApply.cuh"
#include "THZCReduce.cuh"

template<class Op>
void THZCudaTensor_logicalValue(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, Op op)
{
  THZCudaTensor_resizeAs(state, self_, src);

  if (!THZCudaTensor_pointwiseApply2(state, self_, src, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorLTValueOp {
  TensorLTValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in < value);
  }

  const float value;
};

void THZCudaTensor_ltValue(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  THZCudaTensor_logicalValue(state, self_, src, TensorLTValueOp(value));
}

struct TensorGTValueOp {
  TensorGTValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in > value);
  }

  const float value;
};

void THZCudaTensor_gtValue(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  THZCudaTensor_logicalValue(state, self_, src, TensorGTValueOp(value));
}

struct TensorLEValueOp {
  TensorLEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in <= value);
  }

  const float value;
};

void THZCudaTensor_leValue(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  THZCudaTensor_logicalValue(state, self_, src, TensorLEValueOp(value));
}

struct TensorGEValueOp {
  TensorGEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in >= value);
  }

  const float value;
};

void THZCudaTensor_geValue(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  THZCudaTensor_logicalValue(state, self_, src, TensorGEValueOp(value));
}

struct TensorEQValueOp {
  TensorEQValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in == value);
  }

  const float value;
};

void THZCudaTensor_eqValue(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  THZCudaTensor_logicalValue(state, self_, src, TensorEQValueOp(value));
}

struct TensorNEValueOp {
  TensorNEValueOp(float v) : value(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = (*in != value);
  }

  const float value;
};

void THZCudaTensor_neValue(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  THZCudaTensor_logicalValue(state, self_, src, TensorNEValueOp(value));
}
