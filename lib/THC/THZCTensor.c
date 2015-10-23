#include "THZCGeneral.h"
#include "THZCTensor.h"
#include "THZCTensorCopy.h"
#include "THAtomic.h"

/**** access methods ****/
THZCudaStorage *THZCudaTensor_storage(THCState *state, const THZCudaTensor *self)
{
  return self->storage;
}

long THZCudaTensor_storageOffset(THCState *state, const THZCudaTensor *self)
{
  return self->storageOffset;
}

int THZCudaTensor_nDimension(THCState *state, const THZCudaTensor *self)
{
  return self->nDimension;
}

long THZCudaTensor_size(THCState *state, const THZCudaTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->size[dim];
}

long THZCudaTensor_stride(THCState *state, const THZCudaTensor *self, int dim)
{
  THArgCheck((dim >= 0) && (dim < self->nDimension), 2, "out of range");
  return self->stride[dim];
}

THLongStorage *THZCudaTensor_newSizeOf(THCState *state, THZCudaTensor *self)
{
  THLongStorage *size = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(size, self->size);
  return size;
}

THLongStorage *THZCudaTensor_newStrideOf(THCState *state, THZCudaTensor *self)
{
  THLongStorage *stride = THLongStorage_newWithSize(self->nDimension);
  THLongStorage_rawCopy(stride, self->stride);
  return stride;
}

float *THZCudaTensor_data(THCState *state, const THZCudaTensor *self)
{
  if(self->storage)
    return (self->storage->data+self->storageOffset);
  else
    return NULL;
}

void THZCudaTensor_setFlag(THCState *state, THZCudaTensor *self, const char flag)
{
  self->flag |= flag;
}

void THZCudaTensor_clearFlag(THCState *state, THZCudaTensor *self, const char flag)
{
  self->flag &= ~flag;
}

/**** creation methods ****/

static void THZCudaTensor_rawInit(THCState *state, THZCudaTensor *self);
static void THZCudaTensor_rawSet(THCState *state, THZCudaTensor *self, THZCudaStorage *storage, long storageOffset, int nDimension, long *size, long *stride);


/* Empty init */
THZCudaTensor *THZCudaTensor_new(THCState *state)
{
  THZCudaTensor *self = (THZCudaTensor*)THAlloc(sizeof(THZCudaTensor));
  THZCudaTensor_rawInit(state, self);
  return self;
}

/* Pointer-copy init */
THZCudaTensor *THZCudaTensor_newWithTensor(THCState *state, THZCudaTensor *tensor)
{
  THZCudaTensor *self = (THZCudaTensor*)THAlloc(sizeof(THZCudaTensor));
  THZCudaTensor_rawInit(state, self);
  THZCudaTensor_rawSet(state,
                      self,
                      tensor->storage,
                      tensor->storageOffset,
                      tensor->nDimension,
                      tensor->size,
                      tensor->stride);
  return self;
}

/* Storage init */
THZCudaTensor *THZCudaTensor_newWithStorage(THCState *state, THZCudaStorage *storage, long storageOffset, THLongStorage *size, THLongStorage *stride)
{
  THZCudaTensor *self = (THZCudaTensor*)THAlloc(sizeof(THZCudaTensor));
  if(size && stride)
    THArgCheck(size->size == stride->size, 4, "inconsistent size");

  THZCudaTensor_rawInit(state, self);
  THZCudaTensor_rawSet(state,
                      self,
                      storage,
                      storageOffset,
                      (size ? size->size : (stride ? stride->size : 0)),
                      (size ? size->data : NULL),
                      (stride ? stride->data : NULL));

  return self;
}
THZCudaTensor *THZCudaTensor_newWithStorage1d(THCState *state, THZCudaStorage *storage, long storageOffset,
                               long size0, long stride0)
{
  return THZCudaTensor_newWithStorage4d(state, storage, storageOffset, size0, stride0, -1, -1,  -1, -1,  -1, -1);
}

THZCudaTensor *THZCudaTensor_newWithStorage2d(THCState *state, THZCudaStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1)
{
  return THZCudaTensor_newWithStorage4d(state, storage, storageOffset, size0, stride0, size1, stride1,  -1, -1,  -1, -1);
}

THZCudaTensor *THZCudaTensor_newWithStorage3d(THCState *state, THZCudaStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2)
{
  return THZCudaTensor_newWithStorage4d(state, storage, storageOffset, size0, stride0, size1, stride1,  size2, stride2,  -1, -1);
}

THZCudaTensor *THZCudaTensor_newWithStorage4d(THCState *state, THZCudaStorage *storage, long storageOffset,
                               long size0, long stride0,
                               long size1, long stride1,
                               long size2, long stride2,
                               long size3, long stride3)
{
  long size[4] = {size0, size1, size2, size3};
  long stride[4] = {stride0, stride1, stride2, stride3};

  THZCudaTensor *self = (THZCudaTensor*)THAlloc(sizeof(THZCudaTensor));
  THZCudaTensor_rawInit(state, self);
  THZCudaTensor_rawSet(state, self, storage, storageOffset, 4, size, stride);

  return self;
}

THZCudaTensor *THZCudaTensor_newWithSize(THCState *state, THLongStorage *size, THLongStorage *stride)
{
  return THZCudaTensor_newWithStorage(state, NULL, 0, size, stride);
}

THZCudaTensor *THZCudaTensor_newWithSize1d(THCState *state, long size0)
{
  return THZCudaTensor_newWithSize4d(state, size0, -1, -1, -1);
}

THZCudaTensor *THZCudaTensor_newWithSize2d(THCState *state, long size0, long size1)
{
  return THZCudaTensor_newWithSize4d(state, size0, size1, -1, -1);
}

THZCudaTensor *THZCudaTensor_newWithSize3d(THCState *state, long size0, long size1, long size2)
{
  return THZCudaTensor_newWithSize4d(state, size0, size1, size2, -1);
}

THZCudaTensor *THZCudaTensor_newWithSize4d(THCState *state, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THZCudaTensor *self = (THZCudaTensor*)THAlloc(sizeof(THZCudaTensor));
  THZCudaTensor_rawInit(state, self);
  THZCudaTensor_rawResize(state, self, 4, size, NULL);

  return self;
}

THZCudaTensor *THZCudaTensor_newClone(THCState *state, THZCudaTensor *self)
{
  THZCudaTensor *tensor = THZCudaTensor_new(state);
  THZCudaTensor_resizeAs(state, tensor, self);
  THZCudaTensor_copy(state, tensor, self);
  return tensor;
}

THZCudaTensor *THZCudaTensor_newContiguous(THCState *state, THZCudaTensor *self)
{
  if(!THZCudaTensor_isContiguous(state, self))
    return THZCudaTensor_newClone(state, self);
  else
  {
    THZCudaTensor_retain(state, self);
    return self;
  }
}

THZCudaTensor *THZCudaTensor_newSelect(THCState *state, THZCudaTensor *tensor, int dimension_, long sliceIndex_)
{
  THZCudaTensor *self = THZCudaTensor_newWithTensor(state, tensor);
  THZCudaTensor_select(state, self, NULL, dimension_, sliceIndex_);
  return self;
}

THZCudaTensor *THZCudaTensor_newNarrow(THCState *state, THZCudaTensor *tensor, int dimension_, long firstIndex_, long size_)
{
  THZCudaTensor *self = THZCudaTensor_newWithTensor(state, tensor);
  THZCudaTensor_narrow(state, self, NULL, dimension_, firstIndex_, size_);
  return self;
}

THZCudaTensor *THZCudaTensor_newTranspose(THCState *state, THZCudaTensor *tensor, int dimension1_, int dimension2_)
{
  THZCudaTensor *self = THZCudaTensor_newWithTensor(state, tensor);
  THZCudaTensor_transpose(state, self, NULL, dimension1_, dimension2_);
  return self;
}

THZCudaTensor *THZCudaTensor_newUnfold(THCState *state, THZCudaTensor *tensor, int dimension_, long size_, long step_)
{
  THZCudaTensor *self = THZCudaTensor_newWithTensor(state, tensor);
  THZCudaTensor_unfold(state, self, NULL, dimension_, size_, step_);
  return self;
}

/* Resize */
void THZCudaTensor_resize(THCState *state, THZCudaTensor *self, THLongStorage *size, THLongStorage *stride)
{
  THArgCheck(size != NULL, 2, "invalid size");
  if(stride)
    THArgCheck(stride->size == size->size, 3, "invalid stride");

  THZCudaTensor_rawResize(state, self, size->size, size->data, (stride ? stride->data : NULL));
}

void THZCudaTensor_resizeAs(THCState *state, THZCudaTensor *self, THZCudaTensor *src)
{
  int isSame = 0;
  int d;
  if(self->nDimension == src->nDimension)
  {
    isSame = 1;
    for(d = 0; d < self->nDimension; d++)
    {
      if(self->size[d] != src->size[d])
      {
        isSame = 0;
        break;
      }
    }
  }

  if(!isSame)
    THZCudaTensor_rawResize(state, self, src->nDimension, src->size, NULL);
}

void THZCudaTensor_resize1d(THCState *state, THZCudaTensor *tensor, long size0)
{
  THZCudaTensor_resize4d(state, tensor, size0, -1, -1, -1);
}

void THZCudaTensor_resize2d(THCState *state, THZCudaTensor *tensor, long size0, long size1)
{
  THZCudaTensor_resize4d(state, tensor, size0, size1, -1, -1);
}

void THZCudaTensor_resize3d(THCState *state, THZCudaTensor *tensor, long size0, long size1, long size2)
{
  THZCudaTensor_resize4d(state, tensor, size0, size1, size2, -1);
}

void THZCudaTensor_resize4d(THCState *state, THZCudaTensor *self, long size0, long size1, long size2, long size3)
{
  long size[4] = {size0, size1, size2, size3};

  THZCudaTensor_rawResize(state, self, 4, size, NULL);
}

void THZCudaTensor_resize5d(THCState *state, THZCudaTensor *self, long size0, long size1, long size2, long size3, long size4)
{
    long size[5] = {size0, size1, size2, size3, size4};

  THZCudaTensor_rawResize(state, self, 5, size, NULL);
}

void THZCudaTensor_set(THCState *state, THZCudaTensor *self, THZCudaTensor *src)
{
  if(self != src)
    THZCudaTensor_rawSet(state,
                        self,
                        src->storage,
                        src->storageOffset,
                        src->nDimension,
                        src->size,
                        src->stride);
}

void THZCudaTensor_setStorage(THCState *state, THZCudaTensor *self, THZCudaStorage *storage_, long storageOffset_, THLongStorage *size_, THLongStorage *stride_)
{
  if(size_ && stride_)
    THArgCheck(size_->size == stride_->size, 5, "inconsistent size/stride sizes");

  THZCudaTensor_rawSet(state,
                      self,
                      storage_,
                      storageOffset_,
                      (size_ ? size_->size : (stride_ ? stride_->size : 0)),
                      (size_ ? size_->data : NULL),
                      (stride_ ? stride_->data : NULL));
}

void THZCudaTensor_setStorage1d(THCState *state, THZCudaTensor *self, THZCudaStorage *storage_, long storageOffset_,
                             long size0_, long stride0_)
{
  THZCudaTensor_setStorage4d(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            -1, -1,
                            -1, -1,
                            -1, -1);
}

void THZCudaTensor_setStorage2d(THCState *state, THZCudaTensor *self, THZCudaStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_)
{
  THZCudaTensor_setStorage4d(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            -1, -1,
                            -1, -1);
}

void THZCudaTensor_setStorage3d(THCState *state, THZCudaTensor *self, THZCudaStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_)
{
  THZCudaTensor_setStorage4d(state, self, storage_, storageOffset_,
                            size0_, stride0_,
                            size1_, stride1_,
                            size2_, stride2_,
                            -1, -1);
}

void THZCudaTensor_setStorage4d(THCState *state, THZCudaTensor *self, THZCudaStorage *storage_, long storageOffset_,
                             long size0_, long stride0_,
                             long size1_, long stride1_,
                             long size2_, long stride2_,
                             long size3_, long stride3_)
{

  long size[4] = {size0_, size1_, size2_, size3_};
  long stride[4] = {stride0_, stride1_, stride2_, stride3_};

  THZCudaTensor_rawSet(state, self, storage_, storageOffset_, 4, size, stride);
}


void THZCudaTensor_narrow(THCState *state, THZCudaTensor *self, THZCudaTensor *src, int dimension, long firstIndex, long size)
{
  if(!src)
    src = self;

  THArgCheck( (dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck( (firstIndex >= 0) && (firstIndex < src->size[dimension]), 4, "out of range");
  THArgCheck( (size > 0) && (firstIndex+size <= src->size[dimension]), 5, "out of range");

  THZCudaTensor_set(state, self, src);

  if(firstIndex > 0)
    self->storageOffset += firstIndex*self->stride[dimension];

  self->size[dimension] = size;
}

void THZCudaTensor_select(THCState *state, THZCudaTensor *self, THZCudaTensor *src, int dimension, long sliceIndex)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(src->nDimension > 1, 1, "cannot select on a vector");
  THArgCheck((dimension >= 0) && (dimension < src->nDimension), 3, "out of range");
  THArgCheck((sliceIndex >= 0) && (sliceIndex < src->size[dimension]), 4, "out of range");

  THZCudaTensor_set(state, self, src);
  THZCudaTensor_narrow(state, self, NULL, dimension, sliceIndex, 1);
  for(d = dimension; d < self->nDimension-1; d++)
  {
    self->size[d] = self->size[d+1];
    self->stride[d] = self->stride[d+1];
  }
  self->nDimension--;
}

void THZCudaTensor_transpose(THCState *state, THZCudaTensor *self, THZCudaTensor *src, int dimension1, int dimension2)
{
  long z;

  if(!src)
    src = self;

  THArgCheck( (dimension1 >= 0) && (dimension1 < src->nDimension), 1, "out of range");
  THArgCheck( (dimension2 >= 0) && (dimension2 < src->nDimension), 2, "out of range");

  THZCudaTensor_set(state, self, src);

  if(dimension1 == dimension2)
    return;

  z = self->stride[dimension1];
  self->stride[dimension1] = self->stride[dimension2];
  self->stride[dimension2] = z;
  z = self->size[dimension1];
  self->size[dimension1] = self->size[dimension2];
  self->size[dimension2] = z;
}

void THZCudaTensor_unfold(THCState *state, THZCudaTensor *self, THZCudaTensor *src, int dimension, long size, long step)
{
  long *newSize;
  long *newStride;
  int d;

  if(!src)
    src = self;

  THArgCheck( (src->nDimension > 0), 1, "cannot unfold an empty tensor");
  THArgCheck(dimension < src->nDimension, 2, "out of range");
  THArgCheck(size <= src->size[dimension], 3, "out of range");
  THArgCheck(step > 0, 4, "invalid step");

  THZCudaTensor_set(state, self, src);

  newSize = (long*)THAlloc(sizeof(long)*(self->nDimension+1));
  newStride = (long*)THAlloc(sizeof(long)*(self->nDimension+1));

  newSize[self->nDimension] = size;
  newStride[self->nDimension] = self->stride[dimension];
  for(d = 0; d < self->nDimension; d++)
  {
    if(d == dimension)
    {
      newSize[d] = (self->size[d] - size) / step + 1;
      newStride[d] = step*self->stride[d];
    }
    else
    {
      newSize[d] = self->size[d];
      newStride[d] = self->stride[d];
    }
  }

  THFree(self->size);
  THFree(self->stride);

  self->size = newSize;
  self->stride = newStride;
  self->nDimension++;
}

/* we have to handle the case where the result is a number */
void THZCudaTensor_squeeze(THCState *state, THZCudaTensor *self, THZCudaTensor *src)
{
  int ndim = 0;
  int d;

  if(!src)
    src = self;

  THZCudaTensor_set(state, self, src);

  for(d = 0; d < src->nDimension; d++)
  {
    if(src->size[d] != 1)
    {
      if(d != ndim)
      {
        self->size[ndim] = src->size[d];
        self->stride[ndim] = src->stride[d];
      }
      ndim++;
    }
  }

  /* right now, we do not handle 0-dimension tensors */
  if(ndim == 0 && src->nDimension > 0)
  {
    self->size[0] = 1;
    self->stride[0] = 1;
    ndim = 1;
  }
  self->nDimension = ndim;
}

void THZCudaTensor_squeeze1d(THCState *state, THZCudaTensor *self, THZCudaTensor *src, int dimension)
{
  int d;

  if(!src)
    src = self;

  THArgCheck(dimension < src->nDimension, 3, "dimension out of range");

  THZCudaTensor_set(state, self, src);

  if(src->size[dimension] == 1 && src->nDimension > 1)
  {
    for(d = dimension; d < self->nDimension-1; d++)
    {
      self->size[d] = self->size[d+1];
      self->stride[d] = self->stride[d+1];
    }
    self->nDimension--;
  }
}

int THZCudaTensor_isContiguous(THCState *state, const THZCudaTensor *self)
{
  long z = 1;
  int d;
  for(d = self->nDimension-1; d >= 0; d--)
  {
    if(self->size[d] != 1)
    {
      if(self->stride[d] == z)
        z *= self->size[d];
      else
        return 0;
    }
  }
  return 1;
}

int THZCudaTensor_isSameSizeAs(THCState *state, const THZCudaTensor *self, const THZCudaTensor* src)
{
  int d;
  if (self->nDimension != src->nDimension)
    return 0;
  for(d = 0; d < self->nDimension; ++d)
  {
    if(self->size[d] != src->size[d])
      return 0;
  }
  return 1;
}

long THZCudaTensor_nElement(THCState *state, const THZCudaTensor *self)
{
  if(self->nDimension == 0)
    return 0;
  else
  {
    long nElement = 1;
    int d;
    for(d = 0; d < self->nDimension; d++)
      nElement *= self->size[d];
    return nElement;
  }
}

void THZCudaTensor_retain(THCState *state, THZCudaTensor *self)
{
  if(self->flag & TH_TENSOR_REFCOUNTED)
    THAtomicIncrementRef(&self->refcount);
}

void THZCudaTensor_free(THCState *state, THZCudaTensor *self)
{
  if(!self)
    return;

  if(self->flag & TH_TENSOR_REFCOUNTED)
  {
    if(THAtomicDecrementRef(&self->refcount))
    {
      THFree(self->size);
      THFree(self->stride);
      if(self->storage)
        THZCudaStorage_free(state, self->storage);
      THFree(self);
    }
  }
}

void THZCudaTensor_freeCopyTo(THCState *state, THZCudaTensor *self, THZCudaTensor *dst)
{
  if(self != dst)
    THZCudaTensor_copy(state, dst, self);

  THZCudaTensor_free(state, self);
}

/*******************************************************************************/

static void THZCudaTensor_rawInit(THCState *state, THZCudaTensor *self)
{
  self->refcount = 1;
  self->storage = NULL;
  self->storageOffset = 0;
  self->size = NULL;
  self->stride = NULL;
  self->nDimension = 0;
  self->flag = TH_TENSOR_REFCOUNTED;
}

static void THZCudaTensor_rawSet(THCState *state, THZCudaTensor *self, THZCudaStorage *storage, long storageOffset, int nDimension, long *size, long *stride)
{
  /* storage */
  if(self->storage != storage)
  {
    if(self->storage)
      THZCudaStorage_free(state, self->storage);

    if(storage)
    {
      self->storage = storage;
      THZCudaStorage_retain(state, self->storage);
    }
    else
      self->storage = NULL;
  }

  /* storageOffset */
  if(storageOffset < 0)
    THError("Tensor: invalid storage offset");
  self->storageOffset = storageOffset;

  /* size and stride */
  THZCudaTensor_rawResize(state, self, nDimension, size, stride);
}

void THZCudaTensor_rawResize(THCState *state, THZCudaTensor *self, int nDimension, long *size, long *stride)
{
  int d;
  int nDimension_;
  long totalSize;
  int hascorrectsize = 1;

  nDimension_ = 0;
  for(d = 0; d < nDimension; d++)
  {
    if(size[d] > 0)
    {
      nDimension_++;
      if((self->nDimension > d) && (size[d] != self->size[d]))
        hascorrectsize = 0;

      if((self->nDimension > d) && stride && (stride[d] >= 0) && (stride[d] != self->stride[d]))
        hascorrectsize = 0;
    }
    else
      break;
  }
  nDimension = nDimension_;

  if(nDimension != self->nDimension)
    hascorrectsize = 0;

  if(hascorrectsize)
    return;

  if(nDimension > 0)
  {
    if(nDimension != self->nDimension)
    {
      self->size = (long*)THRealloc(self->size, sizeof(long)*nDimension);
      self->stride = (long*)THRealloc(self->stride, sizeof(long)*nDimension);
      self->nDimension = nDimension;
    }

    totalSize = 1;
    for(d = self->nDimension-1; d >= 0; d--)
    {
      self->size[d] = size[d];
      if(stride && (stride[d] >= 0) )
        self->stride[d] = stride[d];
      else
      {
        if(d == self->nDimension-1)
          self->stride[d] = 1;
        else
          self->stride[d] = self->size[d+1]*self->stride[d+1];
      }
      totalSize += (self->size[d]-1)*self->stride[d];
    }

    if(totalSize+self->storageOffset > 0)
    {
      if(!self->storage)
        self->storage = THZCudaStorage_new(state);
      if(totalSize+self->storageOffset > self->storage->size)
        THZCudaStorage_resize(state, self->storage, totalSize+self->storageOffset);
    }
  }
  else
    self->nDimension = 0;
}

void THZCudaTensor_set1d(THCState *state, THZCudaTensor *tensor, long x0, float value)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  THZCudaStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0], value);
}

float THZCudaTensor_get1d(THCState *state, const THZCudaTensor *tensor, long x0)
{
  THArgCheck(tensor->nDimension == 1, 1, "tensor must have one dimension");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]), 2, "out of range");
  return THZCudaStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]);
}

void THZCudaTensor_set2d(THCState *state, THZCudaTensor *tensor, long x0, long x1, float value)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  THZCudaStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1], value);
}

float THZCudaTensor_get2d(THCState *state, const THZCudaTensor *tensor, long x0, long x1)
{
  THArgCheck(tensor->nDimension == 2, 1, "tensor must have two dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]), 2, "out of range");
  return THZCudaStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]);
}

void THZCudaTensor_set3d(THCState *state, THZCudaTensor *tensor, long x0, long x1, long x2, float value)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  THZCudaStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2], value);
}

float THZCudaTensor_get3d(THCState *state, const THZCudaTensor *tensor, long x0, long x1, long x2)
{
  THArgCheck(tensor->nDimension == 3, 1, "tensor must have three dimensions");
  THArgCheck( (x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]), 2, "out of range");
  return THZCudaStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]);
}

void THZCudaTensor_set4d(THCState *state, THZCudaTensor *tensor, long x0, long x1, long x2, long x3, float value)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  THZCudaStorage_set(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3], value);
}

float THZCudaTensor_get4d(THCState *state, const THZCudaTensor *tensor, long x0, long x1, long x2, long x3)
{
  THArgCheck(tensor->nDimension == 4, 1, "tensor must have four dimensions");
  THArgCheck((x0 >= 0) && (x0 < tensor->size[0]) && (x1 >= 0) && (x1 < tensor->size[1]) && (x2 >= 0) && (x2 < tensor->size[2]) && (x3 >= 0) && (x3 < tensor->size[3]), 2, "out of range");
  return THZCudaStorage_get(state, tensor->storage, tensor->storageOffset+x0*tensor->stride[0]+x1*tensor->stride[1]+x2*tensor->stride[2]+x3*tensor->stride[3]);
}

int THZCudaTensor_checkGPU(THCState *state, unsigned int nTensors, ...)
{
#ifdef DISABLE_CHECK_GPU
  return 1;  // Disable GPU checks.
#else
  int curDev = -1;
  THZCudaCheck(cudaGetDevice(&curDev));
  va_list(args);
  va_start(args, nTensors);
  int valid = 1;
  for (unsigned int i = 0; i < nTensors; i++) {
    THZCudaTensor* tensor = va_arg(args, THZCudaTensor*);
    if (tensor == NULL) {
      continue;
    }
    int tensorDev = THZCudaTensor_getDevice(state, tensor);
    if (tensorDev != -1 && tensorDev != curDev) {
      valid = 0;
      break;
    }
  }
  va_end(args);
  return valid;
#endif
}
