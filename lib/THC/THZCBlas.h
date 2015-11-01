#ifndef THZC_BLAS_INC
#define THZC_BLAS_INC

#include "THZCGeneral.h"

/* Level 1 */
THZC_API void THZCudaBlas_swap(THCState *state, long n, cx *x, long incx, cx *y,
                               long incy);
THZC_API void THZCudaBlas_scal(THCState *state, long n, cx a, cx *x, long incx);
THZC_API void THZCudaBlas_copy(THCState *state, long n, cx *x, long incx, cx *y,
                               long incy);
THZC_API void THZCudaBlas_axpy(THCState *state, long n, cx a, cx *x, long incx,
                               cx *y, long incy);
THZC_API cx
THZCudaBlas_dot(THCState *state, long n, cx *x, long incx, cx *y, long incy);

/* Level 2 */
THZC_API void THZCudaBlas_gemv(THCState *state, char trans, long m, long n,
                               cx alpha, cx *a, long lda, cx *x, long incx,
                               cx beta, cx *y, long incy);
THZC_API void THZCudaBlas_ger(THCState *state, long m, long n, cx alpha, cx *x,
                              long incx, cx *y, long incy, cx *a, long lda);

/* Level 3 */
THZC_API void THZCudaBlas_gemm(THCState *state, char transa, char transb,
                               long m, long n, long k, cx alpha, cx *a,
                               long lda, cx *b, long ldb, cx beta, cx *c,
                               long ldc);
THZC_API void THZCudaBlas_gemmBatched(THCState *state, char transa, char transb,
                                      long m, long n, long k, cx alpha,
                                      const cx *a[], long lda, const cx *b[],
                                      long ldb, cx beta, cx *c[], long ldc,
                                      long batchCount);

#endif
