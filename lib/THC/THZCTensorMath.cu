#include "THZCTensorMath.h"
#include "THZCGeneral.h"
#include "THZCTensorCopy.h"
#include "THZCTensorRandom.h"
#include "THZCApply.cuh"
#include "THZCReduce.cuh"
#include "THZCReduceAll.cuh"

#include <thrust/functional.h>

struct TensorFillOp {
  TensorFillOp(cuComplex v) : val(v) {}
  __device__ __forceinline__ void operator()(cuComplex* v) { *v = val; }

  const cuComplex val;
};
struct TensorFillReOp {
  TensorFillOp(cuComplex v) : val(v) {}
  __device__ __forceinline__ void operator()(cuComplex* v) { *v = make_cuComplex(val,cuCimagf(*v)); }

  const cuComplex val;
};
struct TensorFillImOp {
  TensorFillOp(cuComplex v) : val(v) {}
  __device__ __forceinline__ void operator()(cuComplex* v) { *v = make_cuComplex(cuCrealf(*v),val); }

  const cuComplex val;
};

void THZCudaTensor_fill(THCState* state, THZCudaTensor *self_, cuComplex value)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self_));
  if (!THZCudaTensor_pointwiseApply1(state, self_, TensorFillOp(value))) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_fillim(THCState* state, THZCudaTensor *self_, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self_));
  if (!THZCudaTensor_pointwiseApply1(state, self_, TensorFillImOp(value))) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_fillre(THCState* state, THZCudaTensor *self_, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self_));
  if (!THZCudaTensor_pointwiseApply1(state, self_, TensorFillReOp(value))) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_zero(THCState *state, THZCudaTensor *self_)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self_));
  if (THZCudaTensor_isContiguous(state, self_)) {
    THZCudaCheck(cudaMemsetAsync(THZCudaTensor_data(state, self_),
                                0,
                                sizeof(cuComplex) * THZCudaTensor_nElement(state, self_),
                                THCState_getCurrentStream(state)));
  } else {
    if (!THZCudaTensor_pointwiseApply1(state, self_, TensorFillOp(0))) {
      THArgCheck(false, 1, CUTORCH_DIM_WARNING);
    }
  }

  THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_zeros(THCState *state, THZCudaTensor *r_, THLongStorage *size)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, r_));
  THZCudaTensor_resize(state, r_, size, NULL);
  THZCudaTensor_zero(state, r_);
}

void THZCudaTensor_ones(THCState *state, THZCudaTensor *r_, THLongStorage *size)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, r_));
  THZCudaTensor_resize(state, r_, size, NULL);
  THZCudaTensor_fill(state, r_, 1);
}

void THZCudaTensor_reshape(THCState *state, THZCudaTensor *r_, THZCudaTensor *t, THLongStorage *size)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, r_, t));
  THZCudaTensor_resize(state, r_, size, NULL);
  THZCudaTensor_copy(state, r_, t);
}

long THZCudaTensor_numel(THCState *state, THZCudaTensor *t)
{
  return THZCudaTensor_nElement(state, t);
}

void THZCudaTensor_cat(THCState *state, THZCudaTensor *result, THZCudaTensor *ta, THZCudaTensor *tb, int dimension)
{
  THZCudaTensor* inputs[2];
  inputs[0] = ta;
  inputs[1] = tb;
  THZCudaTensor_catArray(state, result, inputs, 2, dimension);
}

void THZCudaTensor_catArray(THCState *state, THZCudaTensor *result, THZCudaTensor **inputs, int numInputs, int dimension)
{
  THLongStorage *size;
  int i, j;
  long offset;
  int ndim = dimension + 1;
  for (i = 0; i < numInputs; i++)
  {
    ndim = THMax(ndim, THZCudaTensor_nDimension(state, inputs[i]));
  }

  THArgCheck(numInputs > 0, 3, "invalid number of inputs %d", numInputs);
  THArgCheck(dimension >= 0, 4, "invalid dimension %d", dimension+1);

  size = THLongStorage_newWithSize(ndim);
  for(i = 0; i < ndim; i++)
  {
    long dimSize = i < THZCudaTensor_nDimension(state, inputs[0])
                       ? THZCudaTensor_size(state, inputs[0], i)
                       : 1;
    if (i == dimension)
    {
      for (j = 1; j < numInputs; j++)
      {
        dimSize += i < THZCudaTensor_nDimension(state, inputs[j])
                       ? THZCudaTensor_size(state, inputs[j], i)
                       : 1;
      }
    }
    else
    {
      for (j = 1; j < numInputs; j++)
      {
        if (dimSize != (i < THZCudaTensor_nDimension(state, inputs[j])
                            ? THZCudaTensor_size(state, inputs[j], i)
                            : 1)) {
          THLongStorage_free(size);
          THError("inconsistent tensor sizes");
        }
      }
    }
    size->data[i] = dimSize;
  }

  THZCudaTensor_resize(state, result, size, NULL);
  THLongStorage_free(size);

  offset = 0;
  for (j = 0; j < numInputs; j++)
  {
    long dimSize = dimension < THZCudaTensor_nDimension(state, inputs[j])
                       ? THZCudaTensor_size(state, inputs[j], dimension)
                       : 1;
    THZCudaTensor *nt = THZCudaTensor_newWithTensor(state, result);
    THZCudaTensor_narrow(state, nt, NULL, dimension, offset, dimSize);
    THZCudaTensor_copy(state, nt, inputs[j]);
    THZCudaTensor_free(state, nt);
    offset += dimSize;
  }
}

struct TensorDivOp {
  __device__ __forceinline__ void
  operator()(cuComplex* out, cuComplex* in) {
    *out = cuCdivf(*out,*in);
  }

  __device__ __forceinline__ void
  operator()(cuComplex* out, cuComplex* in1, cuComplex* in2) {
    *out = cuCdivf(*in1,*in2);
  }
};

void THZCudaTensor_cdiv(THCState* state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
  THArgCheck(THZCudaTensor_nElement(state, src1) ==
             THZCudaTensor_nElement(state, src2), 3, "sizes do not match");

  if (self_ == src1) {
    // self *= src2
    if (!THZCudaTensor_pointwiseApply2(state, self_, src2, TensorDivOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src1);

    // self = src1 * src2
    if (!THZCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorDivOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorAddCMulOp {
  TensorAddCMulOp(float v) : val(v) {}

  __device__ __forceinline__ void
  operator()(cuComplex* out, cuComplex* in1, cuComplex* in2) {
    *out = cuCaddf(*out,cuCmulf(make_cuComplex(val*cuCrealf(*in1),val*cuCimagf(*in1)),*in2));
  }

  float val;
};

void THZCudaTensor_addcmul(THCState *state, THZCudaTensor *self_, THZCudaTensor *t, float value, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 4, self_, t, src1, src2));
  if(self_ != t)
  {
    THZCudaTensor_resizeAs(state, self_, t);
    THZCudaTensor_copy(state, self_, t);
  }
  else
  {
    THArgCheck(THZCudaTensor_nElement(state, self_) == THZCudaTensor_nElement(state, src1),
               1, "sizes do not match");
  }

  THArgCheck(THZCudaTensor_nElement(state, src1) == THZCudaTensor_nElement(state, src2),
             3, "sizes do not match");

  if (!THZCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorAddCMulOp(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorAddCDivOp {
  TensorAddCDivOp(float v) : val(v) {}

  __device__ __forceinline__ void
  operator()(cuComplex* out, cuComplex* in1, cuComplex* in2) {
    *out += cuCaddf(*out,cuCdivf(make_cuComplex(val*cuCrealf(*in1),val*cuCimagf(*in1)),*in2));
  }

  float val;
};

void THZCudaTensor_addcdiv(THCState *state, THZCudaTensor *self_, THZCudaTensor *t, float value, THZCudaTensor *src1, THZCudaTensor *src2)
{
  THAssert(THZCudaTensor_checkGPU(state, 4, self_, t, src1, src2));
  if(self_ != t)
  {
    THZCudaTensor_resizeAs(state, self_, t);
    THZCudaTensor_copy(state, self_, t);
  }
  else
  {
    THArgCheck(THZCudaTensor_nElement(state, self_) == THZCudaTensor_nElement(state, src1),
               1, "sizes do not match");
  }
  THArgCheck(THZCudaTensor_nElement(state, src1) == THZCudaTensor_nElement(state, src2),
             3, "sizes do not match");

  if (!THZCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorAddCDivOp(value))) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

struct AbsOp {
  AbsOp(){}

  __device__ __forceinline__ void
  operator()(cuComplex* out, cuComplex* in1, cuComplex* in2) {
    *out += cuCaddf(*out,cuCdivf(make_cuComplex(val*cuCrealf(*in1),val*cuCimagf(*in1)),*in2));
  }

  float val;
};

template <typename T>
struct AbsOp : public thrust::unary_function<T,T>{
  __host__ __device__
  float operator()(cuComplex v){
    return cuCabsf(v);
  }
};

float THZCudaTensor_minall(THCState *state, THZCudaTensor *self)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  float val = (float) THInf;
  if (!THZCudaTensor_reduceAll(state, self,
                              AbsOp(),
                              thrust::minimum<float>(),
                              (float) THInf, &val, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
  return val;
}

float THZCudaTensor_maxall(THCState *state, THZCudaTensor *self)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  float val = (float) -THInf;
  if (!THZCudaTensor_reduceAll(state, self,
                              AbsOp(),
                              thrust::maximum<float>(),
                              (float) -THInf, &val, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
  return val;
}
template <typename T>
struct Plus : public thrust::binary_function<T,T>{
  __host__ __device__
  float operator()(const cuComplex& v1,const cuComplex& v2){
    return cuCaddf(v1,v2);
  }
};
template <typename T>
struct Mul : public thrust::binary_function<T,T>{
  __host__ __device__
  float operator()(const cuComplex& v1,const cuComplex& v2){
    return cuCmulf(v1,v2);
  }
};
float THZCudaTensor_sumall(THCState *state, THZCudaTensor *self)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  float val = 0.0f;
  if (!THZCudaTensor_reduceAll(state, self,
                              thrust::identity<float>(),
                              Plus(),
                              0.0f, &val, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
  return val;
}

float THZCudaTensor_prodall(THCState *state, THZCudaTensor *self)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  float val = 1.0f;
  if (!THZCudaTensor_reduceAll(state, self,
                              thrust::identity<float>(),
                              Mul(),
                              1.0f, &val, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
  return val;
}

void THZCudaTensor_sum(THCState* state, THZCudaTensor *self, THZCudaTensor *src, long dimension)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self, src));
  if (!THZCudaTensor_reduceDim(
        state, self, src,
        thrust::identity<float>(), Plus(), 0.0f, dimension)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_prod(THCState* state, THZCudaTensor *self, THZCudaTensor *src, long dimension)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self, src));
  if (!THZCudaTensor_reduceDim(
        state, self, src,
        thrust::identity<float>(), Mul(), 1.0f, dimension)) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

struct logicalall_functor
{
  __device__ inline float operator()(float x, float y) const
  {
    return x && y;
  }
};

struct logicalany_functor
{
  __device__ float operator()(float x, float y) const
  {
    return x || y;
  }
};

int THZCudaTensor_logicalall(THCState *state, THZCudaTensor *self) {
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  float result = 0.0f;
  if (!THZCudaTensor_reduceAll(state, self,
                              thrust::identity<float>(),
                              logicalall_functor(),
                              1.0f, &result, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  return (int) result;
}

int THZCudaTensor_logicalany(THCState *state, THZCudaTensor *self) {
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  float result = 0.0f;
  if (!THZCudaTensor_reduceAll(state, self,
                              thrust::identity<float>(),
                              logicalany_functor(),
                              0.0f, &result, 0)) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  return (int) result;
}
