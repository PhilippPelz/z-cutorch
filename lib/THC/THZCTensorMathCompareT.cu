#include "THZCTensorMath.h"
#include "THZCGeneral.h"
#include "THZCBlas.h"
#include "THZCTensorCopy.h"
#include "THZCApply.cuh"
#include "THZCReduce.cuh"

template<class Op>
void THZCudaTensor_logicalTensor(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2, Op op)
{
  THZCudaTensor_resizeAs(state, self_, src1);
  THArgCheck(THZCudaTensor_nElement(state, src1) == THZCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (!THZCudaTensor_pointwiseApply3(state, self_, src1, src2, op)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorLTOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a < *b);
  }
};

struct TensorGTOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a > *b);
  }
};

struct TensorLEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a <= *b);
  }
};

struct TensorGEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a >= *b);
  }
};

struct TensorEQOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a == *b);
  }
};

struct TensorNEOp {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = (float) (*a != *b);
  }
};

void THZCudaTensor_ltTensor(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THZCudaTensor_logicalTensor(state, self_, src1, src2, TensorLTOp());
}


void THZCudaTensor_gtTensor(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THZCudaTensor_logicalTensor(state, self_, src1, src2, TensorGTOp());
}


void THZCudaTensor_leTensor(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THZCudaTensor_logicalTensor(state, self_, src1, src2, TensorLEOp());
}


void THZCudaTensor_geTensor(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THZCudaTensor_logicalTensor(state, self_, src1, src2, TensorGEOp());
}


void THZCudaTensor_eqTensor(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THZCudaTensor_logicalTensor(state, self_, src1, src2, TensorEQOp());
}


void THZCudaTensor_neTensor(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THZCudaTensor_logicalTensor(state, self_, src1, src2, TensorNEOp());
}
