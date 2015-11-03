#include "THZCTensorMath.h"
#include "THZCTensorCopy.h"

#include "THZCGeneral.h"
#include "THZCGeneral.cuh"



#include "THC/THC.h"

#include <algorithm>

#ifdef USE_MAGMA
#include <magma.h>
#endif

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif

#define NoMagma(name) "No CUDA implementation of '" #name "'. Install MAGMA and rebuild cutorch (http://icl.cs.utk.edu/magma/)"

void THZCudaTensor_copy(THCState *state, THZCudaTensor *self,
                                 THZCudaTensor *src);

void THZCMagma_init(THCState *state)
{
#ifdef USE_MAGMA
  magma_init();
#endif
}

#ifdef USE_MAGMA
static inline cux* th_magma_cmalloc_pinned(size_t n)
{
  cux* ptr;
  if (MAGMA_SUCCESS != magma_cmalloc_pinned(&ptr, n))
    THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", 2*n/268435456);
  return ptr;
}
static inline float* th_magma_smalloc_pinned(size_t n)
{
  float* ptr;
  if (MAGMA_SUCCESS != magma_smalloc_pinned(&ptr, n))
    THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", n/268435456);
  return ptr;
}
// static inline double* th_magma_dmalloc_pinned(size_t n)
// {
//   double* ptr;
//   if (MAGMA_SUCCESS != magma_dmalloc_pinned(&ptr, n))
//     THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", 2*n/268435456);
//   return ptr;
// }
static inline int* th_magma_imalloc_pinned(size_t n)
{
  int* ptr;
  if (MAGMA_SUCCESS != magma_imalloc_pinned(&ptr, n))
    THError("$ Torch: not enough memory: you tried to allocate %dGB. Buy new RAM!", n/268435456);
  return ptr;
}

// static void THZCudaTensor_copyArray1d(THCState *state, THZCudaTensor *self, cux *src, int k)
// {
//   long size[1] = { k };
//   long stride[1] = { 1 };
//   THZCudaTensor_rawResize(state, self, 1, size, stride);
//   size_t len = k * sizeof(cux);
//   THZCudaCheck(cudaMemcpy(self->storage->data + self->storageOffset, src, len, cudaMemcpyHostToDevice));
// }

static void THCudaTensor_copyArray1d(THCState *state, THCudaTensor *self, float *src, int k)
{
  long size[1] = { k };
  long stride[1] = { 1 };
  THCudaTensor_rawResize(state, self, 1, size, stride);
  size_t len = k * sizeof(cux);
  THCudaCheck(cudaMemcpy(self->storage->data + self->storageOffset, src, len, cudaMemcpyHostToDevice));
}

static void THZCudaTensor_copyArray2d(THCState *state, THZCudaTensor *self, cux *src, int m, int n)
{
  long size[2] = { m, n };
  long stride[2] = { 1, m };
  THZCudaTensor_rawResize(state, self, 2, size, stride);
  size_t len = m * n * sizeof(cux);
  THZCudaCheck(cudaMemcpy(self->storage->data + self->storageOffset, src, len, cudaMemcpyHostToDevice));
}

static void THZCudaTensor_copyTensor2d(THCState *state, cux *dst, THZCudaTensor *self)
{
  THAssert(self->nDimension == 2);
  size_t len = THZCudaTensor_nElement(state, self)*sizeof(cux);
  THZCudaTensor *temp = THZCudaTensor_newTranspose(state, self, 0, 1);
  THZCudaTensor *selfc = THZCudaTensor_newContiguous(state, temp);
  THZCudaCheck(cudaMemcpy(dst, selfc->storage->data + selfc->storageOffset, len, cudaMemcpyDeviceToHost));
  THZCudaTensor_free(state, temp);
  THZCudaTensor_free(state, selfc);
}

static THZCudaTensor* THZCudaTensor_newColumnMajor(THCState *state, THZCudaTensor *self, THZCudaTensor *src)
{
  THAssert(src->nDimension == 2);
  if (self == src && self->stride[0] == 1 && self->stride[1] == self->size[0])
  {
    THZCudaTensor_retain(state, self);
    return self;
  }

  if (self == src)
    self = THZCudaTensor_new(state);
  else
    THZCudaTensor_retain(state, self);

  long size[2] = { src->size[0], src->size[1] };
  long stride[2] = { 1, src->size[0] };

  THZCudaTensor_rawResize(state, self, 2, size, stride);
  THZCudaTensor_copy(state, self, src);
  return self;
}
#endif

void THZCudaTensor_gesv(THCState *state, THZCudaTensor *rb_, THZCudaTensor *ra_, THZCudaTensor *b_, THZCudaTensor *a_)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(b_->nDimension == 2, 2, "b should be 2 dimensional");
  THArgCheck(a_->size[0] == a_->size[1], 1, "A should be square");
  THArgCheck(b_->size[0] == a_->size[0], 2, "A,b size incompatible");

  int n = a_->size[0];
  int nrhs = b_->size[1];

  THZCudaTensor *a = THZCudaTensor_newColumnMajor(state, ra_, a_);
  THZCudaTensor *b = THZCudaTensor_newColumnMajor(state, rb_, b_);
  cux *a_data = THZCudaTensor_dataCu(state, a);
  cux *b_data = THZCudaTensor_dataCu(state, b);

  int *ipiv = th_magma_imalloc_pinned(n);

  int info;
  magma_cgesv_gpu(n, nrhs, a_data, n, ipiv, b_data, n, &info);

  if (info < 0)
    THError("MAGMA gesv : Argument %d : illegal value", -info);
  else if (info > 0)
    THError("MAGMA gesv : U(%d,%d) is zero, singular U.", info, info);

  magma_free_pinned(ipiv);
  THZCudaTensor_freeCopyTo(state, a, ra_);
  THZCudaTensor_freeCopyTo(state, b, rb_);
#else
  THError(NoMagma(gesv));
#endif
}

void THZCudaTensor_gels(THCState *state, THZCudaTensor *rb_, THZCudaTensor *ra_, THZCudaTensor *b_, THZCudaTensor *a_)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 1, "A should be 2 dimensional");
  THArgCheck(b_->nDimension == 2, 1, "b should be 2 dimensional");
  THArgCheck(a_->size[0] == b_->size[0], 2, "size incompatible A,b");
  THArgCheck(a_->size[0] >= a_->size[1], 2, "A should have m >= n");

  THZCudaTensor *a = THZCudaTensor_newColumnMajor(state, ra_, a_);
  THZCudaTensor *b = THZCudaTensor_newColumnMajor(state, rb_, b_);
  cux *a_data = THZCudaTensor_dataCu(state, a);
  cux *b_data = THZCudaTensor_dataCu(state, b);

  int m = a->size[0];
  int n = a->size[1];
  int nrhs = b->size[1];
  magmaFloatComplex wkopt;

  int info;
  magma_cgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, &wkopt, -1, &info);

  int lwork = (int)wkopt.x;
  cux *hwork = th_magma_cmalloc_pinned((size_t)wkopt.x);
  magma_cgels_gpu(MagmaNoTrans, m, n, nrhs, a_data, m, b_data, m, hwork, lwork, &info);
  magma_free_pinned(hwork);

  if (info != 0)
    THError("MAGMA gels : Argument %d : illegal value", -info);

  THZCudaTensor_freeCopyTo(state, a, ra_);
  THZCudaTensor_freeCopyTo(state, b, rb_);
#else
  THError(NoMagma(gels));
#endif
}

// http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magma__cheev__driver.html#gae8eb14e95ecbd97680a9c92ca3224261
void THZCudaTensor_cheev(THCState *state, THCudaTensor *re_, THZCudaTensor *rv_, THZCudaTensor *a, const char *jobzs, const char *uplos)
{
#ifdef USE_MAGMA
  int n = a->size[0];
  int ldda = n;
  int ldwa = n;

  magma_uplo_t uplo = uplos[0] == 'U' ?  MagmaUpper : MagmaLower;
  magma_vec_t jobz = jobzs[0] == 'N' ? MagmaNoVec : MagmaVec;

  THZCudaTensor *input = THZCudaTensor_newColumnMajor(state, rv_, a);
  cux *input_data = THZCudaTensor_dataCu(state, input);

  // compute optimal size of work array
  int info;
  magmaFloatComplex workq;
  float rworkq;
  int iworkq;

  // eigen values and workspace
  cux *wA = th_magma_cmalloc_pinned((size_t)ldwa*n);
  float *w = th_magma_smalloc_pinned(n);

  magma_cheevd_gpu(jobz, uplo, n, input_data, ldda, w, wA, ldwa, &workq, -1, &rworkq, -1, &iworkq, -1, &info);

  int lwork = (int)workq.x;
  int lrwork = (int)rworkq;
  int liwork = (int)iworkq;

  cux *work = th_magma_cmalloc_pinned((size_t)lwork);
  int *iwork = th_magma_imalloc_pinned(liwork);
  float *rwork = th_magma_smalloc_pinned(lrwork);

  // compute eigenvalues and, optionally, eigenvectors
  magma_cheevd_gpu(jobz, uplo, n, input_data, ldda, w, wA, ldwa, work, lwork, rwork, lrwork, iwork, liwork, &info);

  // copy eigen values from w to re_
  if (info == 0)
    THCudaTensor_copyArray1d(state, re_, w, n);

  magma_free_pinned(wA);
  magma_free_pinned(w);
  magma_free_pinned(work);
  magma_free_pinned(iwork);
  magma_free_pinned(rwork);

  // check error value
  if (info > 0)
    THError("MAGMA magma_cheevd_gpu : Failed to converge. %d off-diagonal elements of an didn't converge to zero", info);
  else if (info < 0)
    THError("MAGMA magma_cheevd_gpu : Argument %d : illegal value", -info);

    //copy eigenvectors
  THZCudaTensor_freeCopyTo(state, input, rv_);
#else
  THError(NoMagma(syev));
#endif
}

void THZCudaTensor_geev(THCState *state, THZCudaTensor *re_, THZCudaTensor *rv_, THZCudaTensor *a_, const char *jobvrs)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 3, "A should be 2 dimensional");
  THArgCheck(a_->size[0] == a_->size[1], 3, "A should be square");

  magma_vec_t jobvr = jobvrs[0] == 'N' ? MagmaNoVec : MagmaVec;
  int n = a_->size[0];

  cux *a_data = th_magma_cmalloc_pinned(n * n);
  THZCudaTensor_copyTensor2d(state, a_data, a_);

  cux *w = th_magma_cmalloc_pinned(n);

  cux *vr_data = NULL;
  int ldvr = 1;
  if (jobvr == MagmaVec)
  {
    vr_data = th_magma_cmalloc_pinned(n * n);
    ldvr = n;
  }

  magmaFloatComplex wkopt;
  int info;
  float rworkq;

  magma_cgeev(MagmaNoVec, jobvr, n, a_data, n, w, NULL, 1, vr_data, ldvr, &wkopt, -1, &rworkq, &info);

  int lwork = (int) wkopt.x;
  cux *work_data = th_magma_cmalloc_pinned(lwork);
  float *rwork = th_magma_smalloc_pinned(2*n);

  magma_cgeev(MagmaNoVec, jobvr, n, a_data, n, w, NULL, 1, vr_data, ldvr, work_data, lwork,rwork, &info);

  if (info > 0)
    THError("MAGMA geev : Failed to converge. %d off-diagonal elements of an didn't converge to zero", info);
  else if (info < 0)
    THError("MAGMA geev : Argument %d : illegal value", -info);
  {
    THZCudaTensor_resize2d(state, re_, 2, n);
    THZCudaTensor *re = THZCudaTensor_newContiguous(state, re_);
    THZCudaCheck(cudaMemcpy(re->storage->data + re->storageOffset, w, n*sizeof(cux), cudaMemcpyHostToDevice));
    THZCudaTensor_freeCopyTo(state, re, re_);
    THZCudaTensor_transpose(state, re_, NULL, 0, 1);
  }

  if (jobvr == MagmaVec)
    THZCudaTensor_copyArray2d(state, rv_, vr_data, n, n);

  magma_free_pinned(work_data);
  magma_free_pinned(vr_data);
  magma_free_pinned(rwork);
  magma_free_pinned(w);
  magma_free_pinned(a_data);

#else
  THError(NoMagma(geev));
#endif
}

// http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magma__cgesvd__driver.html#ga4fabaf4f2eb541d45768429d7ef8979a
void THZCudaTensor_gesvd2(THCState *state, THZCudaTensor *ru_, THCudaTensor *rs_, THZCudaTensor *rv_, THZCudaTensor *ra_, THZCudaTensor *a, const char *jobus)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");

  magma_vec_t jobu = jobus[0] == 'A' ? MagmaAllVec : jobus[0] == 'S' ? MagmaSomeVec : jobus[0] == 'O' ? MagmaOverwriteVec : MagmaNoVec;
  magma_vec_t jobvt = jobu;

  int m = a->size[0];
  int n = a->size[1];
  int k = m < n ? m : n;
  int j = (jobu == MagmaAllVec) ? m : k;

  cux *a_data = th_magma_cmalloc_pinned(m * n);
  THZCudaTensor_copyTensor2d(state, a_data, a);

  float *rs_data = th_magma_smalloc_pinned(k);
  cux *ru_data = th_magma_cmalloc_pinned(m * j);
  cux *rv_data = th_magma_cmalloc_pinned(n * n);

  magmaFloatComplex wkopt;
  float tmp;
  int info;
  magma_cgesvd(jobu, jobvt, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, &wkopt, -1,&tmp, &info);

  int lwork = (int) wkopt.x;
  cux *work_data = th_magma_cmalloc_pinned(lwork);
  float* rwork = th_magma_smalloc_pinned(5*k);

  magma_cgesvd(jobu, jobvt, m, n, a_data, m, rs_data, ru_data, m, rv_data, n, work_data, lwork,rwork, &info);

  if (info > 0)
    THError("MAGMA gesvd : %d superdiagonals failed to converge", info);
  else if (info < 0)
    THError("MAGMA gesvd : Argument %d : illegal value", -info);

  THZCudaTensor_copyArray2d(state, rv_, rv_data, n, n);
  THZCudaTensor_transpose(state, rv_, NULL, 0, 1);
  THZCudaTensor_copyArray2d(state, ru_, ru_data, m, j);
  THCudaTensor_copyArray1d(state, rs_, rs_data, k);
  THZCudaTensor_copyArray2d(state, ra_, a_data,  m, n);

  magma_free_pinned(work_data);
  magma_free_pinned(rv_data);
  magma_free_pinned(rwork);
  magma_free_pinned(ru_data);
  magma_free_pinned(rs_data);
  magma_free_pinned(a_data);
#else
  THError(NoMagma(gesvd2));
#endif
}

void THZCudaTensor_gesvd(THCState *state, THZCudaTensor *ru_, THCudaTensor *rs_, THZCudaTensor *rv_, THZCudaTensor *a, const char *jobu)
{
#ifdef USE_MAGMA
  THZCudaTensor *ra_ = THZCudaTensor_new(state);
  THZCudaTensor_gesvd2(state, ru_, rs_, rv_,  ra_, a, jobu);
  THZCudaTensor_free(state, ra_);
#else
  THError(NoMagma(gesvd));
#endif
}

void THZCudaTensor_getri(THCState *state, THZCudaTensor *ra_, THZCudaTensor *a)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int info;
  int n = a->size[0];
  int lwork = n * magma_get_sgetri_nb(n);

  THZCudaTensor *input = THZCudaTensor_newColumnMajor(state, ra_, a);
  cux *input_data = THZCudaTensor_dataCu(state, input);

  int *ipiv = th_magma_imalloc_pinned(n);

  THZCudaTensor *work = THZCudaTensor_newWithSize1d(state, lwork);
  cux *work_data = THZCudaTensor_dataCu(state, work);

  // Run LU
  magma_cgetrf_gpu(n, n, input_data, n, ipiv, &info);
  if (info > 0)
    THError("MAGMA getrf : U(%d,%d) is 0, U is singular", info, info);
  else if (info < 0)
    THError("MAGMA getrf : Argument %d : illegal value", -info);

  // Inverse
  magma_cgetri_gpu(n, input_data, n, ipiv, work_data, lwork, &info);
  if (info > 0)
    THError("MAGMA getri : U(%d,%d) is 0, U is singular", info, info);
  else if (info < 0)
    THError("MAGMA getri : Argument %d : illegal value", -info);

  THZCudaTensor_free(state, work);
  magma_free_pinned(ipiv);
  THZCudaTensor_freeCopyTo(state, input, ra_);
#else
  THError(NoMagma(getri));
#endif
}

__global__ void THZCudaTensor_copyUpperSymmetric(cux *input, int n, int len)
{
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < len; idx += 65535) {
    const int r = idx % n;
    const int c = idx / n;
    if (r > c) {
      input[idx] = input[r*n + c];
    }
  }
}

void THZCudaTensor_potri(THCState *state, THZCudaTensor *ra_, THZCudaTensor *a)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];

  THZCudaTensor *input = THZCudaTensor_newColumnMajor(state, ra_, a);
  cux *input_data = THZCudaTensor_dataCu(state, input);

  int info;
  magma_cpotrf_gpu(MagmaUpper, n, input_data, n, &info);
  if (info > 0)
    THError("MAGMA potrf : A(%d,%d) is 0, A cannot be factorized", info, info);
  else if (info < 0)
    THError("MAGMA potrf : Argument %d : illegal value", -info);

  magma_cpotri_gpu(MagmaUpper, n, input_data, n, &info);
  if (info > 0)
    THError("MAGMA potri : A(%d,%d) is 0, A cannot be factorized", info, info);
  else if (info < 0)
    THError("MAGMA potri : Argument %d : illegal value", -info);

  cudaStream_t stream = THCState_getCurrentStream(state);
  const int len = n*n;
  dim3 blocks(std::min(DIVUP(len, 128), 65535));
  dim3 threads(128);
  THZCudaTensor_copyUpperSymmetric<<<blocks, threads, 0, stream>>>(input_data, n, len);

  THZCudaTensor_freeCopyTo(state, input, ra_);
#else
  THError(NoMagma(potri));
#endif
}

void THZCudaTensor_potrf(THCState *state, THZCudaTensor *ra_, THZCudaTensor *a)
{
#ifdef USE_MAGMA
  THArgCheck(a->nDimension == 2, 2, "A should be 2 dimensional");
  THArgCheck(a->size[0] == a->size[1], 2, "A should be square");

  int n = a->size[0];

  THZCudaTensor *input = THZCudaTensor_newColumnMajor(state, ra_, a);
  cux *input_data = THZCudaTensor_dataCu(state, input);

  int info;
  magma_cpotrf_gpu(MagmaUpper, n, input_data, n, &info);

  // check error value
  if (info > 0)
    THError("MAGMA potrf : A(%d,%d) is 0, A cannot be factorized", info, info);
  else if (info < 0)
    THError("MAGMA potrf : Argument %d : illegal value", -info);

  THZCudaTensor_triu(state, ra_, input, 0);
  THZCudaTensor_free(state, input);
#else
  THError(NoMagma(potrf));
#endif
}

void THZCudaTensor_qr(THCState *state, THZCudaTensor *rq_, THZCudaTensor *rr_, THZCudaTensor *a_)
{
#ifdef USE_MAGMA
  THArgCheck(a_->nDimension == 2, 2, "A should be 2 dimensional");

  THZCudaTensor *a = THZCudaTensor_newColumnMajor(state, rr_, a_);
  int m = a->size[0];
  int n = a->size[1];
  int k = (m < n ? m : n);
  int nb = magma_get_cgeqrf_nb(m);

  cux *a_data = THZCudaTensor_dataCu(state, a);
  cux *tau_data = th_magma_cmalloc_pinned(n*n);

  THZCudaTensor *work = THZCudaTensor_newWithSize1d(state, (2*k + ((n+31)/32)*32)*nb);
  cux *work_data = THZCudaTensor_dataCu(state, work);

  int info;
  magma_cgeqrf_gpu(m, n, a_data, m, tau_data, work_data, &info);

  if (info != 0)
    THError("MAGMA geqrf : Argument %d : illegal value.", -info);

  THZCudaTensor *q = THZCudaTensor_newColumnMajor(state, rq_, a);
  cux *q_data = THZCudaTensor_dataCu(state, q);

  THZCudaTensor_narrow(state, a, a, 0, 0, k);
  THZCudaTensor_triu(state, rr_, a, 0);
  THZCudaTensor_free(state, a);

  magma_cungqr_gpu(m, n, k, q_data, m, tau_data, work_data, nb, &info);

  if (info != 0)
    THError("MAGMA orgqr : Argument %d : illegal value.", -info);

  THZCudaTensor_free(state, work);
  magma_free_pinned(tau_data);

  THZCudaTensor_narrow(state, q, q, 1, 0, k);
  THZCudaTensor_freeCopyTo(state, q, rq_);
#else
  THError(NoMagma(qr));
#endif
}
