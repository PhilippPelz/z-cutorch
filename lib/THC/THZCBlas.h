#ifndef THZC_BLAS_INC
#define THZC_BLAS_INC

#include "THZCGeneral.h"

/* Level 1 */
THZC_API void THZCudaBlas_swap(THCState *state, long n, float *x, long incx, float *y, long incy);
THZC_API void THZCudaBlas_scal(THCState *state, long n, float a, float *x, long incx);
THZC_API void THZCudaBlas_copy(THCState *state, long n, float *x, long incx, float *y, long incy);
THZC_API void THZCudaBlas_axpy(THCState *state, long n, float a, float *x, long incx, float *y, long incy);
THZC_API cx THZCudaBlas_dot(THCState *state, long n, float *x, long incx, float *y, long incy);

/* Level 2 */
THZC_API void THZCudaBlas_gemv(THCState *state, char trans, long m, long n, float alpha, float *a, long lda, float *x, long incx, float beta, float *y, long incy);
THZC_API void THZCudaBlas_ger(THCState *state, long m, long n, float alpha, float *x, long incx, float *y, long incy, float *a, long lda);

/* Level 3 */
THZC_API void THZCudaBlas_gemm(THCState *state, char transa, char transb, long m, long n, long k, float alpha, float *a, long lda, float *b, long ldb, float beta, float *c, long ldc);
THZC_API void THZCudaBlas_gemmBatched(THCState *state, char transa, char transb, long m, long n, long k,
                                    float alpha, const float *a[], long lda, const float *b[], long ldb,
                                    float beta, float *c[], long ldc, long batchCount);

#endif
