#include "THZCTensorMath.h"
// #include "THZCGeneral.h"
#include "THZCBlas.h"
#include "THZCTensorCopy.h"
// #include "THZCTensorRandom.h"
#include "THZCApply.cuh"
#include "THZCReduce.cuh"

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

// #include <thrust/complex.h>
// typedef thrust::complex<float> ccx;

#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

// ccx toCcx(cx val) {
// 	return ccx(crealf(val), cimagf(val));
// }

struct ZTensorPowOp {
  ZTensorPowOp(ccx v) : val(v) {}
  __device__ __forceinline__ void operator()(ccx* out, ccx* in) {
    *out = thrust::pow(*in, val);
  }

  __device__ __forceinline__ void operator()(ccx* v) {
    *v = thrust::pow(*v, val);
  }

  const ccx val;
};

void THZCudaTensor_pow(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, cx value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    if (!THZCudaTensor_pointwiseApply1(state, self_, ZTensorPowOp(toCcx(value)))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src);

    if (!THZCudaTensor_pointwiseApply2(state, self_, src, ZTensorPowOp(toCcx(value)))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THZCudaCheck(cudaGetLastError());
}

cx THZCudaTensor_meanall(THCState *state, THZCudaTensor *self)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  THArgCheck(self->nDimension > 0, 1, "empty Tensor");
  return THZCudaTensor_sumall(state, self)/THZCudaTensor_nElement(state, self);
}

void
THZCudaTensor_mean(THCState *state, THZCudaTensor *self, THZCudaTensor *src, long dim)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self, src));
  THZCudaTensor_sum(state, self, src, dim);
  THZCudaTensor_div(state, self, self, THZCudaTensor_size(state, src, dim));
}

struct square_functor
{
  const ccx mean;

  square_functor(ccx mean_) : mean(mean) {}

    __host__ __device__ ccx operator()(const ccx& x) const
  {
    float x1 = thrust::abs((ccx)x-mean);
    return ccx(x1*x1,0);
  }
};

float THZCudaTensor_varall(THCState *state, THZCudaTensor *self)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  self = THZCudaTensor_newContiguous(state, self);
  long size = THZCudaTensor_nElement(state, self);
  thrust::device_ptr<ccx> self_data((ccx*)THZCudaTensor_data(state, self));

  cx mean = THZCudaTensor_meanall(state, self);
  ccx result =
    thrust::transform_reduce(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      self_data, self_data+size, square_functor(toCcx(mean)),
      ccx(0,0), thrust::plus<ccx>());

  float res = result.real()/(float)(THZCudaTensor_nElement(state, self)-1);

  THZCudaTensor_free(state, self);
  return res;
}

float THZCudaTensor_stdall(THCState *state, THZCudaTensor *self)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  return sqrt(THZCudaTensor_varall(state, self));
}

template <int StaticExp>
struct TensorNormOp
{
  TensorNormOp(float exp) : exponent(exp) {}

  __host__ __device__ ccx operator()(ccx y) const {
    if (StaticExp == 1) {
      return ccx(thrust::abs(y),0);
    } else if (StaticExp == 2) {
      float x = thrust::abs(y);
      return ccx(x * x,0);
    } else {
      return ccx(powf(thrust::abs(y), exponent),0);
    }
  }

  const float exponent;
};

struct TensorNonZeroOp
{
  TensorNonZeroOp() {}
  __host__ __device__ ccx operator()(ccx lhs) const { return thrust::abs(lhs) != 0.0f ? ccx(1,0) : ccx(0,0); }
};

float THZCudaTensor_normall(THCState *state, THZCudaTensor *self, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  self = THZCudaTensor_newContiguous(state, self);
  long size = THZCudaTensor_nElement(state, self);
  thrust::device_ptr<ccx> self_data((ccx*)THZCudaTensor_data(state, self));

  ccx result;
	float res = 0;
  if (value == 0.0f) {
    result = thrust::transform_reduce(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      self_data, self_data+size, TensorNonZeroOp(),
      ccx(0.0,0), thrust::plus<ccx>());
			res = result.real();
  } else if (value == 1.0f) {
    result = thrust::transform_reduce(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      self_data, self_data+size, TensorNormOp<1>(value),
      ccx(0.0,0), thrust::plus<ccx>());

  } else if (value == 2.0f) {
    result = thrust::transform_reduce(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      self_data, self_data+size, TensorNormOp<2>(value),
      ccx(0.0,0), thrust::plus<ccx>());
    res = powf(result.real(), 0.5f);

  } else {
    result = thrust::transform_reduce(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      self_data, self_data+size, TensorNormOp<-1>(value),
      ccx(0.0,0), thrust::plus<ccx>());
    res = powf(result.real(), 1.0f / value);
  }

  THZCudaTensor_free(state, self);
  return res;
}

void THZCudaTensor_normDim(THCState *state, THZCudaTensor* self, THZCudaTensor* src, float value, long dimension)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self, src));
  if (value == 0.0f) {
    THZCudaTensor_reduceDim(state, self, src,
                           TensorNonZeroOp(), thrust::plus<ccx>(),
                           0.0f, dimension);
  } else if (value == 1.0f) {
    THZCudaTensor_reduceDim(state, self, src,
                           TensorNormOp<1>(value), thrust::plus<ccx>(),
                           0.0f, dimension);

  } else if (value == 2.0f) {
    THZCudaTensor_reduceDim(state, self, src,
                           TensorNormOp<2>(value), thrust::plus<ccx>(),
                           0.0f, dimension);
    THZCudaTensor_pow(state, self, self, 0.5f);

  } else {
    THZCudaTensor_reduceDim(state, self, src,
                           TensorNormOp<-1>(value), thrust::plus<ccx>(),
                           0.0f, dimension);
    THZCudaTensor_pow(state, self, self, 1.0f / value);
  }

  THZCudaCheck(cudaGetLastError());
}

__global__ void THZCudaTensor_kernel_renorm(ccx *data, const float value, const long size, const float maxnorm)
{
  __shared__ float buffer[32];
  long tx = threadIdx.x;
  long bx = blockIdx.x;
  long step = blockDim.x;
  ccx *row = data + size*bx;

  buffer[tx] = 0;

  // get norm of axis
  for (long i=tx; i<size; i+=step)
  {
    buffer[tx] += powf(thrust::abs(row[i]), value);
  }
  // add (reduce)
  for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
  {
    __syncthreads();
    if (tx < stride)
      buffer[tx] += buffer[tx+stride];
  }
  // clip norms
  __syncthreads();
  float norm = powf(buffer[0], 1/value);
  if (norm > maxnorm)
  {
    norm = maxnorm / (norm + 1e-7);
    // renormalize
    for (long i=tx; i<size; i+=step)
    {
      row[i] *= norm;
    }
  }
}

void THZCudaTensor_renorm(THCState *state, THZCudaTensor* self, THZCudaTensor* src, float value, long dimension, float maxnorm)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self, src));
  THZCudaTensor *self_;
  THZCudaTensor *src_ = THZCudaTensor_newTranspose(state, src, dimension, 0);
  THZCudaTensor *data = THZCudaTensor_newClone(state, src_);
  long size = THZCudaTensor_nElement(state, data)/data->size[0];

  THArgCheck(dimension >= 0 && dimension < THZCudaTensor_nDimension(state, src), 3, "invalid dimension");
  THArgCheck(value > 0, 2, "non-positive-norm not supported");
  THArgCheck(THZCudaTensor_nDimension(state, src) > 1, 1, "need at least 2 dimensions");

  dim3 grid(data->size[0]);
  dim3 threads(32);

  THZCudaTensor_kernel_renorm<<<grid, threads, 0, THCState_getCurrentStream(state)>>>((ccx*)THZCudaTensor_data(state, data), value, size, maxnorm);

  cudaError errcode = cudaGetLastError();
  if(errcode != cudaSuccess)
    THError(cudaGetErrorString(errcode));

  THZCudaTensor_free(state, src_);
  self_ = THZCudaTensor_newTranspose(state, data, dimension, 0);
  THZCudaTensor_resizeAs(state, self, self_);
  THZCudaTensor_freeCopyTo(state, self_, self);
  THZCudaTensor_free(state, data);
}

struct dist_functor
{
  const float exponent;

  dist_functor(float exponent_) : exponent(exponent_) {}

  __host__ __device__ ccx operator()(const ccx& x, const ccx& y) const
  {
    return ccx(powf(thrust::abs(x-y), exponent),0);
  }
};

float THZCudaTensor_dist(THCState *state, THZCudaTensor *self, THZCudaTensor *src, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self, src));
  self = THZCudaTensor_newContiguous(state, self);
  long size = THZCudaTensor_nElement(state, self);
  src = THZCudaTensor_newContiguous(state, src);
  thrust::device_ptr<ccx> self_data((ccx*)THZCudaTensor_data(state, self));
  thrust::device_ptr<ccx> src_data((ccx*)THZCudaTensor_data(state, src));
 ccx result;
//   ccx result = thrust::inner_product(
// #if CUDA_VERSION >= 7000
//     thrust::cuda::par.on(THCState_getCurrentStream(state)),
// #endif
//     self_data, self_data+size, src_data, ccx(0,0),
//     thrust::plus<ccx>, dist_functor(value));

  THZCudaTensor_free(state, src);
  THZCudaTensor_free(state, self);

  return powf(result.real(), (float)1.0/value);
}
