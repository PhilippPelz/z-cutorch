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

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(NAME, CFUNC)                   \
  struct Tensor##NAME##Op {                                             \
    __device__ __forceinline__ void operator()(ccx* o, ccx* i) const { \
      *o = CFUNC(*i);                                                   \
    }                                                                   \
                                                                        \
    __device__ __forceinline__ void operator()(ccx* v) const {        \
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

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log, thrust::log)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(log10, thrust::log10)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(exp, thrust::exp)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cos, thrust::cos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(acos, thrust::acos)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(cosh, thrust::cosh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(acosh, thrust::acosh)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sin, thrust::sin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(asin, thrust::asin)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sinh, thrust::sinh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(asinh, thrust::asinh)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tan, thrust::tan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(atan, thrust::atan)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(tanh, thrust::tanh)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(atanh, thrust::atanh)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(sqrt, thrust::sqrt)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(conj, thrust::conj)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(proj, thrust::proj)

IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(pow, thrust::pow)


#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNCZ(NAME, CFUNC)                   \
  struct Tensor##NAME##Op {                                             \
    __device__ __forceinline__ void operator()(ccx* o, ccx* i) const { \
      *o = ccx(CFUNC(*i),0);                                                   \
    }                                                                   \
                                                                        \
    __device__ __forceinline__ void operator()(ccx* v) const {        \
      *v = ccx(CFUNC(*v),0);                                                   \
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

IMPLEMENT_CUDA_TENSOR_BASIC_FUNCZ(zabs, thrust::abs)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNCZ(zarg, thrust::arg)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNCZ(znorm, thrust::norm)

#define IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_RET_FLOAT(NAME, CFUNC)                   \
  struct Tensor##NAME##Op {                                             \
    __device__ __forceinline__ void operator()(ccx* i,float* o) const { \
      *o = CFUNC(*i);                                                   \
    }                                                                   \
                                                                        \
  };                                                                    \
                                                                        \
  void THZCudaTensor_##NAME(THCState* state, THCudaTensor* self_, THZCudaTensor* src) { \
    THAssert(THZCudaTensor_checkGPU(state, 1, src));                \
    THAssert(THCudaTensor_checkGPU(state, 1, self_));                \
    THLongStorage *size = THCudaTensor_newSizeOf(state, src);                 \
    THLongStorage *stride = THCudaTensor_newStrideOf(state, src);               \
    THCudaTensor_resize(state,self_,size,stride);                               \
                                                                        \
      if (!THZCudaTensor_pointwiseApply2ZF(state, src, self_, Tensor##NAME##Op())) { \
        THArgCheck(false, 2, CUTORCH_DIM_WARNING); 						\
      }                                                                 \
    }                                                                   \
                                                                        \
    THZCudaCheck(cudaGetLastError());                                    \
    THLongStorage_free(size);                                           \
    THLongStorage_free(stride);                                           \
  }



IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_RET_FLOAT(abs, thrust::abs)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_RET_FLOAT(arg, thrust::arg)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_RET_FLOAT(norm, thrust::norm)
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_RET_FLOAT(real, [ccx* i,float* o] () { *o = (*i).real();   })
IMPLEMENT_CUDA_TENSOR_BASIC_FUNC_RET_FLOAT(imag, [ccx* i,float* o] () { *o = (*i).imag();   })

// IMPLEMENT_CUDA_TENSOR_BASIC_FUNC(round, roundf)

#undef IMPLEMENT_CUDA_TENSOR_BASIC_FUNC

struct TensorAddOp {
	__device__ __forceinline__ void operator()(ccx* o, ccx* i) {
		*o += *i;
	}

	__device__ __forceinline__ void operator()(ccx* o, ccx* i1, ccx* i2) {
		*o = *i1 + *i2;
	}
};

struct TensorCAddOp {
	TensorCAddOp(ccx v) :
			val(v) {
	}

	__device__ __forceinline__ void operator()(ccx* out, ccx* in) {
		*out += val * *in;
	}

	__device__ __forceinline__ void operator()(ccx* out, ccx* in1, ccx* in2) {
		*out = *in1 + val * *in2;
	}

	ccx val;
};

void THZCudaTensor_cadd(THCState *state, THZCudaTensor *self_, THZCudaTensor* src1, cx value, THZCudaTensor *src2) {
	THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
	THArgCheck(THZCudaTensor_nElement(state, src1) == THZCudaTensor_nElement(state, src2), 3, "sizes do not match");

	if (self_ == src1) {
		if (value == 1.0f) {
			// self += src2
			if (!THZCudaTensor_pointwiseApply2(state, self_, src2, TensorAddOp())) {
				THArgCheck(false, 2, CUTORCH_DIM_WARNING);
			}
		} else {
			// self += value * src2
			if (!THZCudaTensor_pointwiseApply2(state, self_, src2, TensorCAddOp(toCcx(value)))) {
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
			if (!THZCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorCAddOp(toCcx(value)))) {
				THArgCheck(false, 2, CUTORCH_DIM_WARNING);
			}
		}
	}

	THZCudaCheck(cudaGetLastError());
}

struct TensorPolarOp {
	__device__ __forceinline__ void operator()(ccx* out, float* abs, float* arg) {
		*out = thrust::polar(*abs,*arg);
	}
};

void THZCudaTensor_polar(THCState *state, THZCudaTensor *self_, THCudaTensor *src1, THCudaTensor *src2) {
	THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
	THArgCheck(THZCudaTensor_nElement(state, src1) == THZCudaTensor_nElement(state, src2), 3, "sizes do not match");


    THLongStorage *size = THCudaTensor_newSizeOf(state, src1);
    THLongStorage *stride = THCudaTensor_newStrideOf(state, src1);
    THZCudaTensor_resize(state,self_,size,stride);

		// self = src1 * src2
		if (!THZCudaTensor_pointwiseApply3(state, self_, src1, src2, TensorPolarOp())) {
			THArgCheck(false, 2, CUTORCH_DIM_WARNING);
		}

	   THZCudaCheck(cudaGetLastError());
}


struct TensorMulOp {
	__device__ __forceinline__ void operator()(ccx* out, ccx* in) {
		*out = *out * *in;
	}

	__device__ __forceinline__ void operator()(ccx* out, ccx* in1, ccx* in2) {
		*out = *in1 * *in2;
	}
};

void THZCudaTensor_cmul(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2) {
	THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
	THArgCheck(THZCudaTensor_nElement(state, src1) == THZCudaTensor_nElement(state, src2), 3, "sizes do not match");

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

struct TensorDivOp {
	__device__ __forceinline__ void operator()(ccx* out, ccx* in) {
		*out = *out / *in;
	}

	__device__ __forceinline__ void operator()(ccx* out, ccx* in1, ccx* in2) {
		*out = *in1 / *in2;
	}
};

void THZCudaTensor_cdiv(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2) {
	THAssert(THZCudaTensor_checkGPU(state, 3, self_, src1, src2));
	THArgCheck(THZCudaTensor_nElement(state, src1) == THZCudaTensor_nElement(state, src2), 3, "sizes do not match");

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

struct TensorMaxOp {
	__device__ __forceinline__ void operator()(ccx* o, ccx* i) {
		*o = max(thrust::abs(*o), thrust::abs(*i));
	}

	__device__ __forceinline__ void operator()(ccx* o, ccx* i1, ccx* i2) {
		*o = max(thrust::abs(*i1), thrust::abs(*i2));
	}
};

void THZCudaTensor_cmax(THCState *state, THZCudaTensor *self, THZCudaTensor *src1, THZCudaTensor *src2) {
	THAssert(THZCudaTensor_checkGPU(state, 3, self, src1, src2));
	THArgCheck(THZCudaTensor_nElement(state, src1) == THZCudaTensor_nElement(state, src2), 2, "sizes do not match");

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
	__device__ __forceinline__ void operator()(ccx* o, ccx* i) {
		*o = min(thrust::abs(*o), thrust::abs(*i));
	}

	__device__ __forceinline__ void operator()(ccx* o, ccx* i1, ccx* i2) {
		*o = min(thrust::abs(*i1), thrust::abs(*i2));
	}
};

void THZCudaTensor_cmin(THCState *state, THZCudaTensor *self, THZCudaTensor *src1, THZCudaTensor *src2) {
	THAssert(THZCudaTensor_checkGPU(state, 3, self, src1, src2));
	THArgCheck(THZCudaTensor_nElement(state, src1) == THZCudaTensor_nElement(state, src2), 2, "sizes do not match");

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