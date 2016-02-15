#include "THZCTensorMath.h"
#include "THZCGeneral.h"
#include "THZCGeneral.cuh"
#include "THZCBlas.h"
#include "THZCTensorCopy.h"
#include "THZCApply.cuh"
#include "THZCReduce.cuh"

// #include <thrust/complex.h>
// typedef thrust::complex<float> ccx;

// ccx toCcx(cx val) {
// 	return ccx(crealf(val), cimagf(val));
// }
struct TensorFillOp {
	TensorFillOp(ccx v) :
			val(v) {
	}
	__device__ __forceinline__ void operator()(ccx* v) {
		*v = val;
	}

	const ccx val;
};
struct TensorFillReOp {
	TensorFillReOp(float v) :
			val(v) {
	}
	__device__ __forceinline__ void operator()(ccx* v) {
		*v = ccx(val, v->imag());
	}

	const float val;
};
struct TensorFillImOp {
	TensorFillImOp(float v) :
			val(v) {
	}
	__device__ __forceinline__ void operator()(ccx* v) {
		*v = ccx(v->real(), val);
	}

	const float val;
};
struct TensorAddConstantOp {
  TensorAddConstantOp(ccx v) : val(v) {}
  __device__ __forceinline__ void operator()(ccx* out, ccx* in) {
    ccx *o = (ccx*)out;
    ccx *i = (ccx*)in;
    *o = *i + val;
  }

  __device__ __forceinline__ void operator()(ccx* v) {
    ccx *vo = (ccx*)v;
    *vo += val;
  }

  const ccx val;
};

void THZCudaTensor_add(THCState *state, THZCudaTensor *self_, THZCudaTensor *src_, cx value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src_));
  if (self_ == src_) {
		// printf("path1\n");
    if (!THZCudaTensor_pointwiseApply1(state, self_, TensorAddConstantOp(toCcx(value)))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src_);
		// printf("path2\n");
    if (!THZCudaTensor_pointwiseApply2(state, self_, src_, TensorAddConstantOp(toCcx(value)))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THZCudaCheck(cudaGetLastError());
}

struct TensorMulConstantOp {
  TensorMulConstantOp(ccx v) : val(v) {}
  __device__ __forceinline__ void operator()(ccx* o, ccx* i) {
    *o = *i * val;
  }

  __device__ __forceinline__ void operator()(ccx* v) {
    *v *= val;
  }

  const ccx val;
};

void THZCudaTensor_fill(THCState* state, THZCudaTensor *self_, cx value) {
	THAssert(THZCudaTensor_checkGPU(state, 1, self_));
	if (!THZCudaTensor_pointwiseApply1(state, self_, TensorFillOp(toCcx(value)))) {
		THArgCheck(false, 1, CUTORCH_DIM_WARNING);
	}

	THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_fillim(THCState* state, THZCudaTensor *self_, float value) {
	THAssert(THZCudaTensor_checkGPU(state, 1, self_));
	if (!THZCudaTensor_pointwiseApply1(state, self_, TensorFillImOp(value))) {
		THArgCheck(false, 1, CUTORCH_DIM_WARNING);
	}

	THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_fillre(THCState* state, THZCudaTensor *self_, float value) {
	THAssert(THZCudaTensor_checkGPU(state, 1, self_));
	if (!THZCudaTensor_pointwiseApply1(state, self_, TensorFillReOp(value))) {
		THArgCheck(false, 1, CUTORCH_DIM_WARNING);
	}

	THZCudaCheck(cudaGetLastError());
}
void THZCudaTensor_ones(THCState *state, THZCudaTensor *r_,
		THLongStorage *size) {
	THAssert(THZCudaTensor_checkGPU(state, 1, r_));
	THZCudaTensor_resize(state, r_, size, NULL);
	THZCudaTensor_fill(state, r_, 1);
}
void THZCudaTensor_zero(THCState *state, THZCudaTensor *self_) {
	THAssert(THZCudaTensor_checkGPU(state, 1, self_));
	if (THZCudaTensor_isContiguous(state, self_)) {
		THZCudaCheck(
				cudaMemsetAsync(THZCudaTensor_data(state, self_), 0, sizeof(cx) * THZCudaTensor_nElement(state, self_), THCState_getCurrentStream(state)));
	} else {
		if (!THZCudaTensor_pointwiseApply1(state, self_, TensorFillOp(0))) {
			THArgCheck(false, 1, CUTORCH_DIM_WARNING);
		}
	}

	THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_zeros(THCState *state, THZCudaTensor *r_,
		THLongStorage *size) {
	THAssert(THZCudaTensor_checkGPU(state, 1, r_));
	THZCudaTensor_resize(state, r_, size, NULL);
	THZCudaTensor_zero(state, r_);
}
void THZCudaTensor_mul(THCState *state, THZCudaTensor *self_, THZCudaTensor *src_, cx value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src_));
  if (self_ == src_) {
    if (!THZCudaTensor_pointwiseApply1(state, self_, TensorMulConstantOp(toCcx(value)))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src_);

    if (!THZCudaTensor_pointwiseApply2(state, self_, src_, TensorMulConstantOp(toCcx(value)))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_div(THCState* state, THZCudaTensor *self_, THZCudaTensor *src_, cx value)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(value != 0.0f, 3, "divide by zero");

  if (self_ == src_) {
    if (!THZCudaTensor_pointwiseApply1(state, self_, TensorMulConstantOp(1.0f / toCcx(value)))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src_);

    if (!THZCudaTensor_pointwiseApply2(state, self_, src_, TensorMulConstantOp(1.0f / toCcx(value)))) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  THZCudaCheck(cudaGetLastError());
}

template <int Upper>
struct TensorTriOp {
  TensorTriOp(cx *start_, long stride0_, long stride1_, long k_)
    : start(start_), stride0(stride0_), stride1(stride1_), k(k_) {}

  __device__ __forceinline__ int mask(ccx *in) {
    ptrdiff_t n = in - (ccx*)start;
    long row, col;
    if (stride0 > stride1)
    {
      row = (long) (n / stride0);
      col = (long) ((n % stride0) / stride1);
    }
    else
    {
      row = (long) ((n % stride1) / stride0);
      col = (long) (n / stride1);
    }

    return Upper ? (col - row >= k) : (col - row <= k);
  }

  __device__ __forceinline__ void operator()(ccx* out, ccx* in) {
    *out = mask(in) ? *in : 0;
  }

  __device__ __forceinline__ void operator()(ccx* v) {
    if (!mask(v))
      *v = 0;
  }

  const cx *start;
  const long stride0, stride1, k;
};

void THZCudaTensor_tril(THCState *state, THZCudaTensor *self_, THZCudaTensor *src_, long k)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THZCudaTensor *src = src_;
  if (self_ == src_)
    src = THZCudaTensor_newContiguous(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  cx *start = THZCudaTensor_data(state, src) + src->storageOffset;

  TensorTriOp<0> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THZCudaTensor_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src);

    if (!THZCudaTensor_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THZCudaTensor_freeCopyTo(state, src, src_);

  THZCudaCheck(cudaGetLastError());
}

void THZCudaTensor_triu(THCState *state, THZCudaTensor *self_, THZCudaTensor *src_, long k)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self_, src_));
  THArgCheck(src_->nDimension == 2, 1, "expected a matrix");

  THZCudaTensor *src = src_;
  if (self_ == src_)
    src = THZCudaTensor_newContiguous(state, src_);

  long stride0 = src->stride[0];
  long stride1 = src->stride[1];
  cx *start = THZCudaTensor_data(state, src) + src->storageOffset;

  TensorTriOp<1> op(start, stride0, stride1, k);

  if (self_ == src_) {
    if (!THZCudaTensor_pointwiseApply1(state, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  } else {
    THZCudaTensor_resizeAs(state, self_, src);

    if (!THZCudaTensor_pointwiseApply2(state, self_, src, op)) {
      THArgCheck(false, 2, CUTORCH_DIM_WARNING);
    }
  }

  if (self_ == src_)
    THZCudaTensor_freeCopyTo(state, src, src_);

  THZCudaCheck(cudaGetLastError());
}
