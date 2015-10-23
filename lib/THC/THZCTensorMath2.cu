#include "THZCTensorMath.h"
#include "THZCGeneral.h"
#include "THZCBlas.h"
#include "THZCTensorCopy.h"
#include "THZCTensorRandom.h"
#include "THZCApply.cuh"
#include "THZCReduce.cuh"

#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

struct TensorPowOp {
  TensorPowOp(float v) : val(v) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = powf(*in, val);
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v = powf(*v, val);
  }

  const float val;
};

void THZCudaTensor_pow(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    if (!THZCudaTensor_pointwiseApply1(state, self_, TensorPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src);

    if (!THZCudaTensor_pointwiseApply2(state, self_, src, TensorPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorTPowOp {
  TensorTPowOp(float v) : val(v) {}

  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = powf(val, *in);
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v = powf(val, *v);
  }

  const float val;
};

void THZCudaTensor_tpow(THCState *state, THZCudaTensor *self_, float value, THZCudaTensor *src)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    if (!THZCudaTensor_pointwiseApply1(state, self_, TensorTPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src);

    if (!THZCudaTensor_pointwiseApply2(state, self_, src, TensorTPowOp(value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorATan2Op {
  __device__ __forceinline__ void operator()(float* out, float* a, float* b) {
    *out = atan2f(*a, *b);
  }
};

void THZCudaTensor_atan2(THCState *state, THZCudaTensor *self_, THZCudaTensor *tx, THZCudaTensor *ty)
{
  THAssert(THZCudaTensor_checkGPU(state, 3, self_, tx, ty));
  THArgCheck(THZCudaTensor_nElement(state, tx) ==
             THZCudaTensor_nElement(state, ty), 3, "sizes do not match");
  THZCudaTensor_resizeAs(state, self_, tx);

  if (!THZCudaTensor_pointwiseApply3(state, self_, tx, ty, TensorATan2Op())) {
    THArgCheck(false, 2, CUTORCH_DIM_WARNING);
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorClampOp {
  TensorClampOp(float min, float max) : minValue(min), maxValue(max) {}
  __device__ __forceinline__ void operator()(float* out, float* in) {
    *out = max(min(*in, maxValue), minValue);
  }

  __device__ __forceinline__ void operator()(float* v) {
    *v = max(min(*v, maxValue), minValue);
  }

  const float minValue;
  const float maxValue;
};

void THZCudaTensor_clamp(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, float min_value,
  float max_value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    if (!THZCudaTensor_pointwiseApply1(state, self_, TensorClampOp(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src);

    if (!THZCudaTensor_pointwiseApply2(state, self_, src, TensorClampOp(min_value, max_value))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorSignOp {
  __device__ __forceinline__ void operator()(float* out, float* in) {
    float orig = *in;
    *out = (orig > 0) - (orig < 0);
  }

  __device__ __forceinline__ void operator()(float* v) {
    float orig = *v;
    *v = (orig > 0) - (orig < 0);
  }
};

void THZCudaTensor_sign(THCState *state, THZCudaTensor *self_, THZCudaTensor *src)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  if (self_ == src) {
    if (!THZCudaTensor_pointwiseApply1(state, self_, TensorSignOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src);

    if (!THZCudaTensor_pointwiseApply2(state, self_, src, TensorSignOp())) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THZCudaCheck(cudaGetLastError());
}

float THZCudaTensor_meanall(THCState *state, THZCudaTensor *self)
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
  const float mean;

  square_functor(float mean_) : mean(mean_) {}

    __host__ __device__ float operator()(const float& x) const
  {
    return (x-mean)*(x-mean);
  }
};

float THZCudaTensor_varall(THCState *state, THZCudaTensor *self)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  self = THZCudaTensor_newContiguous(state, self);
  long size = THZCudaTensor_nElement(state, self);
  thrust::device_ptr<float> self_data(THZCudaTensor_data(state, self));

  float mean = THZCudaTensor_meanall(state, self);
  float result =
    thrust::transform_reduce(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      self_data, self_data+size, square_functor(mean),
      (float)0, thrust::plus<float>());

  result = result/(THZCudaTensor_nElement(state, self)-1);

  THZCudaTensor_free(state, self);
  return result;
}

float THZCudaTensor_stdall(THCState *state, THZCudaTensor *self)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  return sqrt(THZCudaTensor_varall(state, self));
}

// Given the sum of values and the sum of squares, compute the variance or standard deviation.
template<bool flag, bool apply_sqrt>
__forceinline__ __device__ float THZCudaTensor_computeVar(float sum, float sum2, unsigned row_size) {
  if (flag) {
    sum /= row_size;
    sum2 /= row_size;
    sum2 -= sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  else {
    sum /= row_size;
    sum2 /= row_size - 1;
    sum2 -= ((float)row_size) / ((float)(row_size - 1)) * sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  if (apply_sqrt)
    return sqrt(sum2);
  else
    return sum2;
}

/* Compute the variance (or standard deviation) along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to compute the variance;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same blockIdx.y process an "outer row" (i.e. an element of the flattened
 * outer dimensions, which contains several "inner rows").
 * Each thread processes a single inner row at a time.
 */
template<bool flag, bool apply_sqrt>
__global__ void THZCudaTensor_kernel_varOuterDim(float *tgt, float *src_, unsigned num_orows, unsigned num_irows, unsigned row_size)
{
  for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      float *src = src_ + orow * row_size * num_irows + irow;
      float sum = 0, sum2 = 0;

      for (unsigned col = 0; col < row_size; ++col) {
        float val = *src;
        sum += val;
        sum2 += val * val;

        src += num_irows;
      }

      tgt[orow * num_irows + irow] = THZCudaTensor_computeVar<flag, apply_sqrt>(sum, sum2, row_size);
    }
  }
}

template<bool apply_sqrt>
__host__ void THZCudaTensor_varOuterDim(THCState *state, THZCudaTensor *tgt, THZCudaTensor *src, long dimension, int flag)
{
  unsigned ndim = THZCudaTensor_nDimension(state, src);
  // Treat all outer dimensions (i.e. dim < dimension) as one.
  unsigned num_orows = 1;
  for (unsigned dim = 0; dim < dimension; dim++) {
    num_orows *= THZCudaTensor_size(state, src, dim);
  }
  unsigned row_size = THZCudaTensor_size(state, src, dimension);
  // Treat all inner dimensions (i.e. dim > dimension) as one.
  unsigned num_irows = 1;
  for (unsigned dim = dimension + 1; dim < ndim; dim++) {
    num_irows *= THZCudaTensor_size(state, src, dim);
  }

  dim3 threads(min(512, num_irows));
  unsigned maxGridDim = 1024;
  dim3 grid(min(maxGridDim, num_orows), min(maxGridDim, THZCCeilDiv(num_irows, threads.x)));

  if (flag) {
    THZCudaTensor_kernel_varOuterDim<true, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        THZCudaTensor_data(state, tgt), THZCudaTensor_data(state, src), num_orows, num_irows, row_size);
  } else {
    THZCudaTensor_kernel_varOuterDim<false, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        THZCudaTensor_data(state, tgt), THZCudaTensor_data(state, src), num_orows, num_irows, row_size);
  }
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}


/* Compute the variance (or standard deviation) of the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<bool flag, bool apply_sqrt>
__global__ void THZCudaTensor_kernel_varInnermostDim(float *tgt, float *src_, unsigned num_rows, unsigned row_size)
{
  __shared__ float ssum[32][16];
  __shared__ float ssum2[32][16];

  for (unsigned block_row = blockIdx.x * blockDim.y; block_row < num_rows; block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;
    float sum = 0, sum2 = 0;
    if (row < num_rows) {
      float *src = src_ + row * row_size;
      // Sequential reduction within a thread.
      for (unsigned col = threadIdx.x; col < row_size; col += blockDim.x) {
        float val = src[col];
        sum += val;
        sum2 += val * val;
      }
    }
    ssum[threadIdx.y][threadIdx.x] = sum;
    ssum2[threadIdx.y][threadIdx.x] = sum2;
    __syncthreads();

    // Reduce intermediate values to single value.
    for (unsigned s = 8; s > 1; s >>= 1) {
      if (row < num_rows && threadIdx.x < s) {
        ssum[threadIdx.y][threadIdx.x] += ssum[threadIdx.y][threadIdx.x + s];
        ssum2[threadIdx.y][threadIdx.x] += ssum2[threadIdx.y][threadIdx.x + s];
      }
      __syncthreads();
    }

    if (row < num_rows && threadIdx.x == 0) {
      sum = ssum[threadIdx.y][0] + ssum[threadIdx.y][1];
      sum2 = ssum2[threadIdx.y][0] + ssum2[threadIdx.y][1];
      tgt[row] = THZCudaTensor_computeVar<flag, apply_sqrt>(sum, sum2, row_size);
    }
    __syncthreads();
  }
}

template<bool apply_sqrt>
__host__ void THZCudaTensor_varInnermostDim(THCState *state, THZCudaTensor *tgt, THZCudaTensor *src, int flag)
{
  unsigned ndim = THZCudaTensor_nDimension(state, src);
  // Treat all outer dimensions as a single dimension.
  unsigned num_rows = 1;
  for (unsigned dim = 0; dim < ndim - 1; dim++) {
    num_rows *= THZCudaTensor_size(state, src, dim);
  }
  unsigned row_size = THZCudaTensor_size(state, src, ndim - 1);

  // From limited testing, 16x32 seemed a good compromise for handling both long and short dimensions.
  dim3 threads(16, 32);
  dim3 grid(min(1024, THZCCeilDiv(num_rows, threads.y)));

  if (flag) {
    THZCudaTensor_kernel_varInnermostDim<true, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        THZCudaTensor_data(state, tgt), THZCudaTensor_data(state, src), num_rows, row_size);
  } else {
    THZCudaTensor_kernel_varInnermostDim<false, apply_sqrt><<<grid, threads, 0, THCState_getCurrentStream(state)>>>(
        THZCudaTensor_data(state, tgt), THZCudaTensor_data(state, src), num_rows, row_size);
  }
  cudaError errcode = cudaGetLastError();
  if (errcode != cudaSuccess) {
    THError(cudaGetErrorString(errcode));
  }
}

void THZCudaTensor_var(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, long dimension, int flag)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  THLongStorage *dim = THZCudaTensor_newSizeOf(state, src);
  THLongStorage_set(dim, dimension, 1);
  THZCudaTensor_resize(state, self_, dim, NULL);
  THLongStorage_free(dim);

  THZCudaTensor *self = THZCudaTensor_newContiguous(state, self_);
  src = THZCudaTensor_newContiguous(state, src);

  if (dimension == THZCudaTensor_nDimension(state, src) - 1) {
    THZCudaTensor_varInnermostDim<false>(state, self, src, flag);
  } else {
    THZCudaTensor_varOuterDim<false>(state, self, src, dimension, flag);
  }

  THZCudaTensor_free(state, src);
  THZCudaTensor_freeCopyTo(state, self, self_);
}

void THZCudaTensor_std(THCState *state, THZCudaTensor *self_, THZCudaTensor *src, long dimension, int flag)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src));
  THLongStorage *dim = THZCudaTensor_newSizeOf(state, src);
  THLongStorage_set(dim, dimension, 1);
  THZCudaTensor_resize(state, self_, dim, NULL);
  THLongStorage_free(dim);

  THZCudaTensor *self = THZCudaTensor_newContiguous(state, self_);
  src = THZCudaTensor_newContiguous(state, src);

  if (dimension == THZCudaTensor_nDimension(state, src) - 1) {
    THZCudaTensor_varInnermostDim<true>(state, self, src, flag);
  } else {
    THZCudaTensor_varOuterDim<true>(state, self, src, dimension, flag);
  }

  THZCudaTensor_free(state, src);
  THZCudaTensor_freeCopyTo(state, self, self_);
}

template <int StaticExp>
struct TensorNormOp
{
  TensorNormOp(float exp) : exponent(exp) {}

  __host__ __device__ float operator()(float x) const {
    if (StaticExp == 1) {
      return fabsf(x);
    } else if (StaticExp == 2) {
      return x * x;
    } else {
      return powf(fabsf(x), exponent);
    }
  }

  const float exponent;
};

struct TensorNonZeroOp
{
  TensorNonZeroOp() {}
  __host__ __device__ bool operator()(float lhs) const { return lhs != 0.0f; }
};

float THZCudaTensor_normall(THCState *state, THZCudaTensor *self, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, self));
  self = THZCudaTensor_newContiguous(state, self);
  long size = THZCudaTensor_nElement(state, self);
  thrust::device_ptr<float> self_data(THZCudaTensor_data(state, self));

  float result;

  if (value == 0.0f) {
    result = thrust::transform_reduce(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      self_data, self_data+size, TensorNonZeroOp(),
      0.0f, thrust::plus<float>());
  } else if (value == 1.0f) {
    result = thrust::transform_reduce(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      self_data, self_data+size, TensorNormOp<1>(value),
      0.0f, thrust::plus<float>());

  } else if (value == 2.0f) {
    result = thrust::transform_reduce(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      self_data, self_data+size, TensorNormOp<2>(value),
      0.0f, thrust::plus<float>());
    result = powf(result, 0.5f);

  } else {
    result = thrust::transform_reduce(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      self_data, self_data+size, TensorNormOp<-1>(value),
      0.0f, thrust::plus<float>());
    result = powf(result, 1.0f / value);
  }

  THZCudaTensor_free(state, self);
  return result;
}

void THZCudaTensor_norm(THCState *state, THZCudaTensor* self, THZCudaTensor* src, float value, long dimension)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self, src));
  if (value == 0.0f) {
    THZCudaTensor_reduceDim(state, self, src,
                           TensorNonZeroOp(), thrust::plus<float>(),
                           0.0f, dimension);
  } else if (value == 1.0f) {
    THZCudaTensor_reduceDim(state, self, src,
                           TensorNormOp<1>(value), thrust::plus<float>(),
                           0.0f, dimension);

  } else if (value == 2.0f) {
    THZCudaTensor_reduceDim(state, self, src,
                           TensorNormOp<2>(value), thrust::plus<float>(),
                           0.0f, dimension);
    THZCudaTensor_pow(state, self, self, 0.5f);

  } else {
    THZCudaTensor_reduceDim(state, self, src,
                           TensorNormOp<-1>(value), thrust::plus<float>(),
                           0.0f, dimension);
    THZCudaTensor_pow(state, self, self, 1.0f / value);
  }

  THZCudaCheck(cudaGetLastError());
}

__global__ void THZCudaTensor_kernel_renorm(float *data, const float value, const long size, const float maxnorm)
{
  __shared__ float buffer[32];
  long tx = threadIdx.x;
  long bx = blockIdx.x;
  long step = blockDim.x;
  float *row = data + size*bx;

  buffer[tx] = 0;

  // get norm of axis
  for (long i=tx; i<size; i+=step)
  {
    buffer[tx] += pow(fabs(row[i]), value);
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
  float norm = pow(buffer[0], 1/value);
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

  THZCudaTensor_kernel_renorm<<<grid, threads, 0, THCState_getCurrentStream(state)>>>(THZCudaTensor_data(state, data), value, size, maxnorm);

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

  __host__ __device__ float operator()(const float& x, const float& y) const
  {
    return pow(fabs(x-y), exponent);
  }
};

float THZCudaTensor_dist(THCState *state, THZCudaTensor *self, THZCudaTensor *src, float value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self, src));
  self = THZCudaTensor_newContiguous(state, self);
  long size = THZCudaTensor_nElement(state, self);
  src = THZCudaTensor_newContiguous(state, src);
  thrust::device_ptr<float> self_data(THZCudaTensor_data(state, self));
  thrust::device_ptr<float> src_data(THZCudaTensor_data(state, src));

  float result = thrust::inner_product(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
    self_data, self_data+size, src_data, (float) 0,
    thrust::plus<float>(), dist_functor(value));

  THZCudaTensor_free(state, src);
  THZCudaTensor_free(state, self);

  return pow(result, (float)1.0/value);
}

void THZCudaTensor_rand(THCState *state, THZCudaTensor *r_, THLongStorage *size)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, r_));
  THZCudaTensor_resize(state, r_, size, NULL);
  THZCudaTensor_uniform(state, r_, 0, 1);
}

void THZCudaTensor_randn(THCState *state, THZCudaTensor *r_, THLongStorage *size)
{
  THAssert(THZCudaTensor_checkGPU(state, 1, r_));
  THZCudaTensor_resize(state, r_, size, NULL);
  THZCudaTensor_normal(state, r_, 0, 1);
}
