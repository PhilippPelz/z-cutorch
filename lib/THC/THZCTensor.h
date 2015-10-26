#ifndef THZC_TENSOR_INC
#define THZC_TENSOR_INC

#include "THTensor.h"
#include "THZCStorage.h"
#include "THZCGeneral.h"
#include "cuComplex.h"

#define TH_TENSOR_REFCOUNTED 1

typedef struct THZCudaTensor
{
    long *size;
    long *stride;
    int nDimension;

    THZCudaStorage *storage;
    long storageOffset;
    int refcount;

    char flag;

} THZCudaTensor;


/**** access methods ****/
THZC_API THZCudaStorage* THZCudaTensor_storage(THCState *state, const THZCudaTensor *self);
THZC_API long THZCudaTensor_storageOffset(THCState *state, const THZCudaTensor *self);
THZC_API int THZCudaTensor_nDimension(THCState *state, const THZCudaTensor *self);
THZC_API long THZCudaTensor_size(THCState *state, const THZCudaTensor *self, int dim);
THZC_API long THZCudaTensor_stride(THCState *state, const THZCudaTensor *self, int dim);
THZC_API THLongStorage *THZCudaTensor_newSizeOf(THCState *state, THZCudaTensor *self);
THZC_API THLongStorage *THZCudaTensor_newStrideOf(THCState *state, THZCudaTensor *self);
THZC_API cuComplex *THZCudaTensor_data(THCState *state, const THZCudaTensor *self);

THZC_API void THZCudaTensor_setFlag(THCState *state, THZCudaTensor *self, const char flag);
THZC_API void THZCudaTensor_clearFlag(THCState *state, THZCudaTensor *self, const char flag);


/**** creation methods ****/
THZC_API THZCudaTensor *THZCudaTensor_new(THCState *state);
THZC_API THZCudaTensor *THZCudaTensor_newWithTensor(THCState *state, THZCudaTensor *tensor);
/* stride might be NULL */
THZC_API THZCudaTensor *THZCudaTensor_newWithStorage(THCState *state, THZCudaStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THZC_API THZCudaTensor *THZCudaTensor_newWithStorage1d(THCState *state, THZCudaStorage *storage_, long storageOffset_,
                                long size0_, long stride0_);
THZC_API THZCudaTensor *THZCudaTensor_newWithStorage2d(THCState *state, THZCudaStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_);
THZC_API THZCudaTensor *THZCudaTensor_newWithStorage3d(THCState *state, THZCudaStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_);
THZC_API THZCudaTensor *THZCudaTensor_newWithStorage4d(THCState *state, THZCudaStorage *storage_, long storageOffset_,
                                long size0_, long stride0_,
                                long size1_, long stride1_,
                                long size2_, long stride2_,
                                long size3_, long stride3_);

/* stride might be NULL */
THZC_API THZCudaTensor *THZCudaTensor_newWithSize(THCState *state, THLongStorage *size_, THLongStorage *stride_);
THZC_API THZCudaTensor *THZCudaTensor_newWithSize1d(THCState *state, long size0_);
THZC_API THZCudaTensor *THZCudaTensor_newWithSize2d(THCState *state, long size0_, long size1_);
THZC_API THZCudaTensor *THZCudaTensor_newWithSize3d(THCState *state, long size0_, long size1_, long size2_);
THZC_API THZCudaTensor *THZCudaTensor_newWithSize4d(THCState *state, long size0_, long size1_, long size2_, long size3_);

THZC_API THZCudaTensor *THZCudaTensor_newClone(THCState *state, THZCudaTensor *self);
THZC_API THZCudaTensor *THZCudaTensor_newContiguous(THCState *state, THZCudaTensor *tensor);
THZC_API THZCudaTensor *THZCudaTensor_newSelect(THCState *state, THZCudaTensor *tensor, int dimension_, long sliceIndex_);
THZC_API THZCudaTensor *THZCudaTensor_newNarrow(THCState *state, THZCudaTensor *tensor, int dimension_, long firstIndex_, long size_);
THZC_API THZCudaTensor *THZCudaTensor_newTranspose(THCState *state, THZCudaTensor *tensor, int dimension1_, int dimension2_);
THZC_API THZCudaTensor *THZCudaTensor_newUnfold(THCState *state, THZCudaTensor *tensor, int dimension_, long size_, long step_);

THZC_API void THZCudaTensor_resize(THCState *state, THZCudaTensor *tensor, THLongStorage *size, THLongStorage *stride);
THZC_API void THZCudaTensor_resizeAs(THCState *state, THZCudaTensor *tensor, THZCudaTensor *src);
THZC_API void THZCudaTensor_resize1d(THCState *state, THZCudaTensor *tensor, long size0_);
THZC_API void THZCudaTensor_resize2d(THCState *state, THZCudaTensor *tensor, long size0_, long size1_);
THZC_API void THZCudaTensor_resize3d(THCState *state, THZCudaTensor *tensor, long size0_, long size1_, long size2_);
THZC_API void THZCudaTensor_resize4d(THCState *state, THZCudaTensor *tensor, long size0_, long size1_, long size2_, long size3_);
THZC_API void THZCudaTensor_resize5d(THCState *state, THZCudaTensor *tensor, long size0_, long size1_, long size2_, long size3_, long size4_);
THZC_API void THZCudaTensor_rawResize(THCState *state, THZCudaTensor *self, int nDimension, long *size, long *stride);

THZC_API void THZCudaTensor_set(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_setStorage(THCState *state, THZCudaTensor *self, THZCudaStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_);
THZC_API void THZCudaTensor_setStorage1d(THCState *state, THZCudaTensor *self, THZCudaStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_);
THZC_API void THZCudaTensor_setStorage2d(THCState *state, THZCudaTensor *self, THZCudaStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_);
THZC_API void THZCudaTensor_setStorage3d(THCState *state, THZCudaTensor *self, THZCudaStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_);
THZC_API void THZCudaTensor_setStorage4d(THCState *state, THZCudaTensor *self, THZCudaStorage *storage_, long storageOffset_,
                                    long size0_, long stride0_,
                                    long size1_, long stride1_,
                                    long size2_, long stride2_,
                                    long size3_, long stride3_);

THZC_API void THZCudaTensor_narrow(THCState *state, THZCudaTensor *self, THZCudaTensor *src, int dimension_, long firstIndex_, long size_);
THZC_API void THZCudaTensor_select(THCState *state, THZCudaTensor *self, THZCudaTensor *src, int dimension_, long sliceIndex_);
THZC_API void THZCudaTensor_transpose(THCState *state, THZCudaTensor *self, THZCudaTensor *src, int dimension1_, int dimension2_);
THZC_API void THZCudaTensor_unfold(THCState *state, THZCudaTensor *self, THZCudaTensor *src, int dimension_, long size_, long step_);

THZC_API void THZCudaTensor_squeeze(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
THZC_API void THZCudaTensor_squeeze1d(THCState *state, THZCudaTensor *self, THZCudaTensor *src, int dimension_);

THZC_API int THZCudaTensor_isContiguous(THCState *state, const THZCudaTensor *self);
THZC_API int THZCudaTensor_isSameSizeAs(THCState *state, const THZCudaTensor *self, const THZCudaTensor *src);
THZC_API long THZCudaTensor_nElement(THCState *state, const THZCudaTensor *self);

THZC_API void THZCudaTensor_retain(THCState *state, THZCudaTensor *self);
THZC_API void THZCudaTensor_free(THCState *state, THZCudaTensor *self);
THZC_API void THZCudaTensor_freeCopyTo(THCState *state, THZCudaTensor *self, THZCudaTensor *dst);

/* Slow access methods [check everything] */
THZC_API void THZCudaTensor_set1d(THCState *state, THZCudaTensor *tensor, long x0, cuComplex value);
THZC_API void THZCudaTensor_set2d(THCState *state, THZCudaTensor *tensor, long x0, long x1, cuComplex value);
THZC_API void THZCudaTensor_set3d(THCState *state, THZCudaTensor *tensor, long x0, long x1, long x2, cuComplex value);
THZC_API void THZCudaTensor_set4d(THCState *state, THZCudaTensor *tensor, long x0, long x1, long x2, long x3, cuComplex value);

THZC_API cuComplex THZCudaTensor_get1d(THCState *state, const THZCudaTensor *tensor, long x0);
THZC_API cuComplex THZCudaTensor_get2d(THCState *state, const THZCudaTensor *tensor, long x0, long x1);
THZC_API cuComplex THZCudaTensor_get3d(THCState *state, const THZCudaTensor *tensor, long x0, long x1, long x2);
THZC_API cuComplex THZCudaTensor_get4d(THCState *state, const THZCudaTensor *tensor, long x0, long x1, long x2, long x3);

/* CUDA-specific functions */
THZC_API cudaTextureObject_t THZCudaTensor_getTextureObject(THCState *state, THZCudaTensor *self);
THZC_API int THZCudaTensor_getDevice(THCState *state, const THZCudaTensor *self);
THZC_API int THZCudaTensor_checkGPU(THCState *state, unsigned int nTensors, ...);

#endif
