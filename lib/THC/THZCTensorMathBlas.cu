#include "THZCTensorMath.h"
#include "THZCGeneral.h"
#include "THZCBlas.h"
#include "THZCTensorCopy.h"
#include "THZCTensorRandom.h"
#include "THZCApply.cuh"
#include "THZCReduce.cuh"



cx THZCudaTensor_dot(THCState *state, THZCudaTensor *self, THZCudaTensor *src)
{
  THAssert(THZCudaTensor_checkGPU(state, 2, self, src));
  THArgCheck(THZCudaTensor_nElement(state, self) == THZCudaTensor_nElement(state, src), 2, "sizes do not match");

  {
    self = THZCudaTensor_newContiguous(state, self);
    src = THZCudaTensor_newContiguous(state, src);

    cx result = THZCudaBlas_dot(state,
                                  THZCudaTensor_nElement(state, self),
                                  THZCudaTensor_data(state, self), 1,
                                  THZCudaTensor_data(state, src), 1);
    THZCudaTensor_free(state, src);
    THZCudaTensor_free(state, self);

    cx result;
  }
}

void THZCudaTensor_addmv(THCState *state, THZCudaTensor *r_, cx beta, THZCudaTensor *t, cx alpha, THZCudaTensor *mat, THZCudaTensor *vec)
{
  THAssert(THZCudaTensor_checkGPU(state, 4, r_, t, mat, vec));
  if( (mat->nDimension != 2) || (vec->nDimension != 1) )
    THError("matrix and vector expected");

  if( mat->size[1] != vec->size[0] )
    THError("size mismatch");

  if(t->nDimension != 1)
    THError("size mismatch");

  if(t->size[0] != mat->size[0])
    THError("size mismatch");

  if(r_ != t)
  {
    THZCudaTensor_resizeAs(state, r_, t);
    THZCudaTensor_copy(state, r_, t);
  }

  if(mat->stride[0] == 1)
  {
    THZCudaBlas_gemv(state, 'n', mat->size[0], mat->size[1],
                    alpha, THZCudaTensor_data(state, mat), mat->stride[1],
                    THZCudaTensor_data(state, vec), vec->stride[0],
                    beta, THZCudaTensor_data(state, r_), r_->stride[0]);
  }
  else if(mat->stride[1] == 1)
  {
    THZCudaBlas_gemv(state, 't',  mat->size[1], mat->size[0],
                    alpha, THZCudaTensor_data(state, mat), mat->stride[0],
                    THZCudaTensor_data(state, vec), vec->stride[0],
                    beta, THZCudaTensor_data(state, r_), r_->stride[0]);
  }
  else
  {
    THZCudaTensor *cmat = THZCudaTensor_newContiguous(state, mat);

    THZCudaBlas_gemv(state, 't',  mat->size[1], mat->size[0],
                    alpha, THZCudaTensor_data(state, cmat), cmat->stride[0],
                    THZCudaTensor_data(state, vec), vec->stride[0],
                    beta, THZCudaTensor_data(state, r_), r_->stride[0]);

    THZCudaTensor_free(state, cmat);
  }
}

void THZCudaTensor_addmm(THCState *state, THZCudaTensor *r_, float beta, THZCudaTensor *t, float alpha, THZCudaTensor *m1, THZCudaTensor *m2)
{
  THAssert(THZCudaTensor_checkGPU(state, 4, r_, t, m1, m2));
  char transpose_r, transpose_m1, transpose_m2;
  THZCudaTensor *r__, *m1_, *m2_;

  if( (m1->nDimension != 2) || (m2->nDimension != 2) )
    THError("matrix and matrix expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != m1->size[0]) || (t->size[1] != m2->size[1]) || (m1->size[1] != m2->size[0]) )
    THError("size mismatch");

  if(t != r_)
  {
    THZCudaTensor_resizeAs(state, r_, t);
    THZCudaTensor_copy(state, r_, t);
  }

  /* r_ */
  if(r_->stride[0] == 1 &&
     r_->stride[1] != 0)
  {
    transpose_r = 'n';
    r__ = r_;
  }
  else if(r_->stride[1] == 1 &&
          r_->stride[0] != 0)
  {
    THZCudaTensor *swap = m2;
    m2 = m1;
    m1 = swap;
    transpose_r = 't';
    r__ = r_;
  }
  else
  {
    transpose_r = 'n';

    r__ = THZCudaTensor_newWithSize2d(state, r_->size[1], r_->size[0]);
    THZCudaTensor_copy(state, r__, r_);
    THZCudaTensor_transpose(state, r__, NULL, 0, 1);
  }

  /* m1 */
  if(m1->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
     m1->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
  {
    transpose_m1 = 'n';
    m1_ = m1;
  }
  else if(m1->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
          m1->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
  {
    transpose_m1 = 't';
    m1_ = m1;
  }
  else
  {
    transpose_m1 = (transpose_r == 'n' ? 't' : 'n');
    m1_ = THZCudaTensor_newContiguous(state, m1);
  }

  /* m2 */
  if(m2->stride[(transpose_r == 'n' ? 0 : 1)] == 1 &&
     m2->stride[(transpose_r == 'n' ? 1 : 0)] != 0)
  {
    transpose_m2 = 'n';
    m2_ = m2;
  }
  else if(m2->stride[(transpose_r == 'n' ? 1 : 0)] == 1 &&
          m2->stride[(transpose_r == 'n' ? 0 : 1)] != 0)
  {
    transpose_m2 = 't';
    m2_ = m2;
  }
  else
  {
    transpose_m2 = (transpose_r == 'n' ? 't' : 'n');
    m2_ = THZCudaTensor_newContiguous(state, m2);
  }

  /* do the operation */
  THZCudaBlas_gemm(state,
                  transpose_m1,
                  transpose_m2,
                  r__->size[(transpose_r == 'n' ? 0 : 1)],
                  r__->size[(transpose_r == 'n' ? 1 : 0)],
                  m1_->size[(transpose_r == 'n' ? 1 : 0)],
                  alpha,
                  THZCudaTensor_data(state, m1_),
                  (transpose_m1 == 'n' ? m1_->stride[(transpose_r == 'n' ? 1 : 0)] : m1_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  THZCudaTensor_data(state, m2_),
                  (transpose_m2 == 'n' ? m2_->stride[(transpose_r == 'n' ? 1 : 0)] : m2_->stride[(transpose_r == 'n' ? 0 : 1)]),
                  beta,
                  THZCudaTensor_data(state, r__),
                  r__->stride[(transpose_r == 'n' ? 1 : 0)]);

  /* free intermediate variables */
  if(m1_ != m1)
    THZCudaTensor_free(state, m1_);

  if(m2_ != m2)
    THZCudaTensor_free(state, m2_);

  if(r__ != r_)
    THZCudaTensor_freeCopyTo(state, r__, r_);
}

void THZCudaTensor_addr(THCState *state, THZCudaTensor *r_, float beta, THZCudaTensor *t, float alpha, THZCudaTensor *vec1, THZCudaTensor *vec2)
{
  THAssert(THZCudaTensor_checkGPU(state, 4, r_, t, vec1, vec2));
  if( (vec1->nDimension != 1) || (vec2->nDimension != 1) )
    THError("vector and vector expected");

  if(t->nDimension != 2)
    THError("size mismatch");

  if( (t->size[0] != vec1->size[0]) || (t->size[1] != vec2->size[0]) )
    THError("size mismatch");

  if(r_ != t)
  {
    THZCudaTensor_resizeAs(state, r_, t);
    THZCudaTensor_copy(state, r_, t);
  }

  if(beta != 1)
    THZCudaTensor_mul(state, r_, r_, beta);

  if(r_->stride[0] == 1)
  {
    THZCudaBlas_ger(state, vec1->size[0], vec2->size[0],
                   alpha, THZCudaTensor_data(state, vec1), vec1->stride[0],
                   THZCudaTensor_data(state, vec2), vec2->stride[0],
                   THZCudaTensor_data(state, r_), r_->stride[1]);
  }
  else if(r_->stride[1] == 1)
  {
    THZCudaBlas_ger(state, vec2->size[0], vec1->size[0],
                   alpha, THZCudaTensor_data(state, vec2), vec2->stride[0],
                   THZCudaTensor_data(state, vec1), vec1->stride[0],
                   THZCudaTensor_data(state, r_), r_->stride[0]);
  }
  else
  {
    THZCudaTensor *cr = THZCudaTensor_newClone(state, r_);

    THZCudaBlas_ger(state, vec2->size[0], vec1->size[0],
                   alpha, THZCudaTensor_data(state, vec2), vec2->stride[0],
                   THZCudaTensor_data(state, vec1), vec1->stride[0],
                   THZCudaTensor_data(state, cr), cr->stride[0]);

    THZCudaTensor_freeCopyTo(state, cr, r_);
  }
}

void THZCudaTensor_baddbmm(THCState *state, THZCudaTensor *result, float beta, THZCudaTensor *t,
                          float alpha, THZCudaTensor *batch1, THZCudaTensor *batch2) {
  THAssert(THZCudaTensor_checkGPU(state, 4, result, t, batch1, batch2));
  THArgCheck(THZCudaTensor_nDimension(state, t) == 3, 4, "expected 3D tensor");
  THArgCheck(THZCudaTensor_nDimension(state, batch1) == 3, 6, "expected 3D tensor");
  THArgCheck(THZCudaTensor_nDimension(state, batch2) == 3, 7, "expected 3D tensor");
  THArgCheck(THZCudaTensor_size(state, t, 0) == THZCudaTensor_size(state, batch1, 0), 6,
             "equal number of batches expected");
  THArgCheck(THZCudaTensor_size(state, t, 0) == THZCudaTensor_size(state, batch2, 0), 7,
             "equal number of batches expected");
  THArgCheck(THZCudaTensor_size(state, t, 1) == THZCudaTensor_size(state, batch1, 1), 6,
             "wrong matrix size");
  THArgCheck(THZCudaTensor_size(state, t, 2) == THZCudaTensor_size(state, batch2, 2), 7,
             "wrong matrix size");
  THArgCheck(THZCudaTensor_size(state, batch1, 2) == THZCudaTensor_size(state, batch2, 1), 6,
             "wrong matrix size");

  if (t != result) {
    THZCudaTensor_resizeAs(state, result, t);
    THZCudaTensor_copy(state, result, t);
  }

  bool transpose_result;
  char transpose_batch1, transpose_batch2;
  long lda, ldb, ldc;
  THZCudaTensor *result_, *batch1_, *batch2_;
  if (result->stride[1] == 1)
  {
    transpose_result = false;
    result_ = result;
    ldc = result_->stride[2];
  }
  else if (result->stride[2] == 1)
  {
    transpose_result = true;

    THZCudaTensor *swap = batch2;
    batch2 = batch1;
    batch1 = swap;

    result_ = result;
    ldc = result_->stride[1];
  }
  else
  {
    transpose_result = false;

    result_ = THZCudaTensor_newWithSize3d(state, result->size[0], result->size[2], result->size[1]);
    THZCudaTensor_copy(state, result_, result);
    THZCudaTensor_transpose(state, result_, NULL, 1, 2);

    ldc = result_->stride[2];
  }

  if (batch1->stride[transpose_result ? 2 : 1] == 1)
  {
    transpose_batch1 = 'n';
    batch1_ = batch1;
    lda = batch1_->stride[transpose_result ? 1 : 2];
  }
  else if (batch1->stride[transpose_result ? 1 : 2] == 1)
  {
    transpose_batch1 = 't';
    batch1_ = batch1;
    lda = batch1_->stride[transpose_result ? 2 : 1];
  }
  else
  {
    transpose_batch1 = transpose_result ? 'n' : 't';
    batch1_ = THZCudaTensor_newContiguous(state, batch1);
    lda = batch1_->stride[1];
  }

  if (batch2->stride[transpose_result ? 2 : 1] == 1)
  {
    transpose_batch2 = 'n';
    batch2_ = batch2;
    ldb = batch2_->stride[transpose_result ? 1 : 2];
  }
  else if (batch2->stride[transpose_result ? 1 : 2] == 1)
  {
    transpose_batch2 = 't';
    batch2_ = batch2;
    ldb = batch2_->stride[transpose_result ? 2 : 1];
  }
  else
  {
    transpose_batch2 = transpose_result ? 'n' : 't';
    batch2_ = THZCudaTensor_newContiguous(state, batch2);
    ldb = batch2_->stride[1];
  }

  // Compute pointers to matrices in each batch.
  long num_batches = result_->size[0];
  size_t matrices_size = num_batches * sizeof(cx*);
  const cx **matrices1 = (const cx **)THAlloc(matrices_size);
  const cx **matrices2 = (const cx **)THAlloc(matrices_size);
  cx **result_matrices = (cx **)THAlloc(matrices_size);
  for (int i = 0; i < num_batches; ++i)
  {
    matrices1[i] = THZCudaTensor_data(state, batch1_) + i * batch1_->stride[0];
    matrices2[i] = THZCudaTensor_data(state, batch2_) + i * batch2_->stride[0];
    result_matrices[i] = THZCudaTensor_data(state, result_) + i * result_->stride[0];
  }

  // Copy pointers to device.
  const cx **d_matrices1, **d_matrices2;
  cx **d_result_matrices;
  THZCudaCheck(THZCudaMalloc(state, (void**)&d_matrices1, matrices_size));
  THZCudaCheck(THZCudaMalloc(state, (void**)&d_matrices2, matrices_size));
  THZCudaCheck(THZCudaMalloc(state, (void**)&d_result_matrices, matrices_size));

  THZCudaCheck(cudaMemcpyAsync(d_matrices1, matrices1, matrices_size,
                              cudaMemcpyHostToDevice, THCState_getCurrentStream(state)));
  THZCudaCheck(cudaMemcpyAsync(d_matrices2, matrices2, matrices_size,
                              cudaMemcpyHostToDevice, THCState_getCurrentStream(state)));
  THZCudaCheck(cudaMemcpyAsync(d_result_matrices, result_matrices, matrices_size,
                              cudaMemcpyHostToDevice, THCState_getCurrentStream(state)));

  THZCudaBlas_gemmBatched(
      state,
      transpose_batch1,
      transpose_batch2,
      result_->size[transpose_result ? 2 : 1],
      result_->size[transpose_result ? 1 : 2],
      batch1_->size[transpose_result ? 1 : 2],
      alpha,
      d_matrices1, lda,
      d_matrices2, ldb,
      beta,
      d_result_matrices, ldc,
      num_batches);

  THZCudaFree(state, d_matrices1);
  THZCudaFree(state, d_matrices2);
  THZCudaFree(state, d_result_matrices);
  THFree(matrices1);
  THFree(matrices2);
  THFree(result_matrices);

  if (batch1_ != batch1)
    THZCudaTensor_free(state, batch1_);

  if (batch2_ != batch2)
    THZCudaTensor_free(state, batch2_);

  if (result_ != result)
    THZCudaTensor_freeCopyTo(state, result_, result);
}
