#ifndef TH_CUDA_TENSOR_MATH_INC
#define TH_CUDA_TENSOR_MATH_INC

#include "THZCTensor.h"
#include "THZCGeneral.h"

//Tensormath.cu
THZC_API void THZCudaTensor_fill(THCState *state, THZCudaTensor *self, cx value);
THZC_API void THZCudaTensor_fillim(THCState *state, THZCudaTensor *self, float value);
THZC_API void THZCudaTensor_fillre(THCState *state, THZCudaTensor *self, float value);
THZC_API void THZCudaTensor_zero(THCState *state, THZCudaTensor *self);

THZC_API void THZCudaTensor_zeros(THCState *state, THZCudaTensor *r_, THLongStorage *size);
THZC_API void THZCudaTensor_ones(THCState *state, THZCudaTensor *r_, THLongStorage *size);
THZC_API void THZCudaTensor_reshape(THCState *state, THZCudaTensor *r_, THZCudaTensor *t, THLongStorage *size);
THZC_API long THZCudaTensor_numel(THCState *state, THZCudaTensor *t);

THZC_API void THZCudaTensor_cat(THCState *state, THZCudaTensor *result, THZCudaTensor *ta, THZCudaTensor *tb, int dimension);
THZC_API void THZCudaTensor_catArray(THCState *state, THZCudaTensor *result, THZCudaTensor **inputs, int numInputs, int dimension);

THZC_API void THZCudaTensor_add(THCState *state, THZCudaTensor *self, THZCudaTensor *src, cx value);
THZC_API void THZCudaTensor_mul(THCState *state, THZCudaTensor *self, THZCudaTensor *src, cx value);
THZC_API void THZCudaTensor_div(THCState *state, THZCudaTensor *self, THZCudaTensor *src, cx value);



THZC_API void THZCudaTensor_cpow(THCState *state, THZCudaTensor *self_, THZCudaTensor *src1, THZCudaTensor *src2);
THZC_API void THZCudaTensor_cdiv(THCState *state, THZCudaTensor *self, THZCudaTensor *src1, THZCudaTensor *src2);

THZC_API void THZCudaTensor_addcmul(THCState *state, THZCudaTensor *self, THZCudaTensor* t, cx value, THZCudaTensor *src1, THZCudaTensor *src2);
THZC_API void THZCudaTensor_addcdiv(THCState *state, THZCudaTensor *self, THZCudaTensor* t, cx value, THZCudaTensor *src1, THZCudaTensor *src2);

THZC_API void THZCudaTensor_sum(THCState *state, THZCudaTensor *self, THZCudaTensor *src, long dim);
THZC_API void THZCudaTensor_prod(THCState *state, THZCudaTensor *self, THZCudaTensor *src, long dim);
THZC_API cx THZCudaTensor_minall(THCState *state, THZCudaTensor *self);
THZC_API cx THZCudaTensor_maxall(THCState *state, THZCudaTensor *self);
THZC_API cx THZCudaTensor_sumall(THCState *state, THZCudaTensor *self);
THZC_API cx THZCudaTensor_prodall(THCState *state, THZCudaTensor *self);

// TensorMath2.cu
THZC_API cx THZCudaTensor_meanall(THCState *state, THZCudaTensor *self);
THZC_API void  THZCudaTensor_mean(THCState *state, THZCudaTensor *self, THZCudaTensor *src, long dim);
THZC_API cx THZCudaTensor_varall(THCState *state, THZCudaTensor *self);
THZC_API void  THZCudaTensor_var(THCState *state, THZCudaTensor *self, THZCudaTensor *src, long dim, int flag);
THZC_API cx THZCudaTensor_stdall(THCState *state, THZCudaTensor *self);
THZC_API void  THZCudaTensor_std(THCState *state, THZCudaTensor *self, THZCudaTensor *src, long dim, int flag);
THZC_API float THZCudaTensor_normall(THCState *state, THZCudaTensor *self, float value);
THZC_API void  THZCudaTensor_norm(THCState *state, THZCudaTensor* self, THZCudaTensor* src, float value, long dimension);
THZC_API void  THZCudaTensor_renorm(THCState *state, THZCudaTensor* self, THZCudaTensor* src, float value, long dimension, float max_norm);
THZC_API float THZCudaTensor_dist(THCState *state, THZCudaTensor *self, THZCudaTensor *src, float value);

//TensorMasked.cu
THZC_API void THZCudaTensor_maskedFill(THCState *state, THZCudaTensor *tensor, THCudaTensor *mask, cx value);
THZC_API void THZCudaTensor_maskedFill(THCState *state, THZCudaTensor *tensor, THCudaTensor *mask, cx value);
THZC_API void THZCudaTensor_maskedCopy(THCState *state, THZCudaTensor *tensor, THCudaTensor *mask, THZCudaTensor *src);
THZC_API void THZCudaTensor_maskedSelect(THCState *state, THZCudaTensor *tensor, THZCudaTensor *src, THCudaTensor *mask);

THZC_API void THZCudaTensor_maskedFillByte(THCState *state, THZCudaTensor *tensor, THByteTensor *mask, cx value);
THZC_API void THZCudaTensor_maskedFillByte(THCState *state, THZCudaTensor *tensor, THByteTensor *mask, cx value);
THZC_API void THZCudaTensor_maskedCopyByte(THCState *state, THZCudaTensor *tensor, THByteTensor *mask, THZCudaTensor *src);
THZC_API void THZCudaTensor_maskedSelectByte(THCState *state, THZCudaTensor *tensor, THZCudaTensor *src, THByteTensor *mask);


//TensorMathBlas
THZC_API cx THZCudaTensor_dot(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_addmv(THCState *state, THZCudaTensor *self, cx beta, THZCudaTensor *t, cx alpha, THZCudaTensor *mat, THZCudaTensor *vec);
THZC_API void THZCudaTensor_addmm(THCState *state, THZCudaTensor *self, cx beta, THZCudaTensor *t, cx alpha, THZCudaTensor *mat1, THZCudaTensor *mat2);
THZC_API void THZCudaTensor_addr(THCState *state, THZCudaTensor *self, cx beta, THZCudaTensor *t, cx alpha, THZCudaTensor *vec1, THZCudaTensor *vec2);
THZC_API void THZCudaTensor_baddbmm(THCState *state, THZCudaTensor *result, cx beta, THZCudaTensor *t,
                                  cx alpha, THZCudaTensor *batch1, THZCudaTensor *batch2);

// MAGMA (i.e. CUDA implementation of LAPACK functions)
THZC_API void THZCudaTensor_gesv(THCState *state, THZCudaTensor *rb_, THZCudaTensor *ra_, THZCudaTensor *b_, THZCudaTensor *a_);
THZC_API void THZCudaTensor_gels(THCState *state, THZCudaTensor *rb_, THZCudaTensor *ra_, THZCudaTensor *b_, THZCudaTensor *a_);
THZC_API void THZCudaTensor_syev(THCState *state, THZCudaTensor *re_, THZCudaTensor *rv_, THZCudaTensor *a_, const char *jobz, const char *uplo);
THZC_API void THZCudaTensor_geev(THCState *state, THZCudaTensor *re_, THZCudaTensor *rv_, THZCudaTensor *a_, const char *jobvr);
THZC_API void THZCudaTensor_gesvd(THCState *state, THZCudaTensor *ru_, THZCudaTensor *rs_, THZCudaTensor *rv_, THZCudaTensor *a, const char *jobu);
THZC_API void THZCudaTensor_gesvd2(THCState *state, THZCudaTensor *ru_, THZCudaTensor *rs_, THZCudaTensor *rv_, THZCudaTensor *ra_, THZCudaTensor *a, const char *jobu);
THZC_API void THZCudaTensor_getri(THCState *state, THZCudaTensor *ra_, THZCudaTensor *a);
THZC_API void THZCudaTensor_potri(THCState *state, THZCudaTensor *ra_, THZCudaTensor *a);
THZC_API void THZCudaTensor_potrf(THCState *state, THZCudaTensor *ra_, THZCudaTensor *a);
THZC_API void THZCudaTensor_qr(THCState *state, THZCudaTensor *rq_, THZCudaTensor *rr_, THZCudaTensor *a);

//MathPairwise
THZC_API void THZCudaTensor_tril(THCState *state, THZCudaTensor *self, THZCudaTensor *src, long k);
THZC_API void THZCudaTensor_triu(THCState *state, THZCudaTensor *self, THZCudaTensor *src, long k);

//MathPointwise
THZC_API void THZCudaTensor_cmin(THCState *state, THZCudaTensor *self, THZCudaTensor *src1, THZCudaTensor *src2);
THZC_API void THZCudaTensor_cmax(THCState *state, THZCudaTensor *self, THZCudaTensor *src1, THZCudaTensor *src2);
THZC_API void THZCudaTensor_log(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
// THZC_API void THZCudaTensor_log1p(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_exp(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_cos(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_acos(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_cosh(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_sin(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_asin(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_sinh(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_tan(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_atan(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_tanh(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_pow(THCState *state, THZCudaTensor *self, THZCudaTensor *src, cx value);
// THZC_API void THZCudaTensor_tpow(THCState *state, THZCudaTensor *self, float value, THZCudaTensor *src);
THZC_API void THZCudaTensor_clamp(THCState *state, THZCudaTensor *self, THZCudaTensor *src, float min_value, float max_value);
THZC_API void THZCudaTensor_sqrt(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
// THZC_API void THZCudaTensor_ceil(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
// THZC_API void THZCudaTensor_floor(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_abs(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_sign(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
// THZC_API void THZCudaTensor_round(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
// THZC_API void THZCudaTensor_atan2(THCState *state, THZCudaTensor *r_, THZCudaTensor *tx, THZCudaTensor *ty);

THZC_API void THZCudaTensor_cmul(THCState *state, THZCudaTensor *self, THZCudaTensor *src1, THZCudaTensor *src2);
THZC_API void THZCudaTensor_cadd(THCState *state, THZCudaTensor *self, THZCudaTensor *src1, cx value, THZCudaTensor *src2);

//MathTransformReduce
THZC_API void THZCudaTensor_min(THCState *state, THZCudaTensor *values, THZCudaTensor *indices, THZCudaTensor *src, long dim);
THZC_API void THZCudaTensor_max(THCState *state, THZCudaTensor *values, THZCudaTensor *indices, THZCudaTensor *src, long dim);

THZC_API void THZCudaTensor_cumsum(THCState *state, THZCudaTensor *self, THZCudaTensor *src, long dim);
THZC_API void THZCudaTensor_cumprod(THCState *state, THZCudaTensor *self, THZCudaTensor *src, long dim);


// THZC_API void THZCudaTensor_cminValue(THCState *state, THZCudaTensor *self, THZCudaTensor *src, float value);
// THZC_API void THZCudaTensor_cmaxValue(THCState *state, THZCudaTensor *self, THZCudaTensor *src, float value);
// THZC_API void THZCudaTensor_cminValue(THCState *state, THZCudaTensor *self, THZCudaTensor *src, cx value);
// THZC_API void THZCudaTensor_cmaxValue(THCState *state, THZCudaTensor *self, THZCudaTensor *src, cx value);

//TensorScatterGather
THZC_API void THZCudaTensor_gather(THCState* state, THZCudaTensor *tensor, THZCudaTensor *src, int dim, THZCudaTensor *index);
THZC_API void THZCudaTensor_scatter(THCState* state, THZCudaTensor *tensor, int dim, THZCudaTensor *index, THZCudaTensor *src);
THZC_API void THZCudaTensor_scatterFill(THCState* state, THZCudaTensor *tensor, int dim, THZCudaTensor *index, cx value);
THZC_API void THZCudaTensor_scatterFill(THCState* state, THZCudaTensor *tensor, int dim, THZCudaTensor *index, cx value);



THZC_API void THZCudaTensor_indexCopy(THCState *state, THZCudaTensor *res_, int dim, THCudaTensor *indices, THZCudaTensor *src);
THZC_API void THZCudaTensor_indexFill(THCState *state, THZCudaTensor *tensor, int dim, THCudaTensor *index, cx val);
THZC_API void THZCudaTensor_indexFillf(THCState *state, THZCudaTensor *res_, int dim, THCudaTensor *indices, float val);
THZC_API void THZCudaTensor_indexSelect(THCState *state, THZCudaTensor *tensor, THZCudaTensor *src, int dim, THCudaTensor *index);

THZC_API void THZCudaTensor_indexCopy_long(THCState *state, THZCudaTensor *res_, int dim, THLongTensor *indices, THZCudaTensor *src);
THZC_API void THZCudaTensor_indexAdd_long(THCState *state, THZCudaTensor *res_, int dim, THLongTensor *indices, THZCudaTensor *src);
THZC_API void THZCudaTensor_indexFill_long(THCState *state, THZCudaTensor *tensor, int dim, THLongTensor *index, cx val);
THZC_API void THZCudaTensor_indexFill_long(THCState *state, THZCudaTensor *tensor, int dim, THLongTensor *index, cx val);
THZC_API void THZCudaTensor_indexSelect_long(THCState *state, THZCudaTensor *tensor, THZCudaTensor *src, int dim, THLongTensor *index);





#endif
