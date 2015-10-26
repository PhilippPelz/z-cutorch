#include "THZCStorage.h"
#include "THZCGeneral.h"
#include "THAtomic.h"

void THZCudaStorage_set(THCState *state, THZCudaStorage *self, long index, float value)
{
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  THZCudaCheck(cudaMemcpy(self->data + index, &value, sizeof(float), cudaMemcpyHostToDevice));
}

float THZCudaStorage_get(THCState *state, const THZCudaStorage *self, long index)
{
  float value;
  THArgCheck((index >= 0) && (index < self->size), 2, "index out of bounds");
  THZCudaCheck(cudaMemcpy(&value, self->data + index, sizeof(float), cudaMemcpyDeviceToHost));
  return value;
}

THZCudaStorage* THZCudaStorage_new(THCState *state)
{
  THZCudaStorage *storage = (THZCudaStorage*)THAlloc(sizeof(THZCudaStorage));
  storage->data = NULL;
  storage->size = 0;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

THZCudaStorage* THZCudaStorage_newWithSize(THCState *state, long size)
{
  THArgCheck(size >= 0, 2, "invalid size");

  if(size > 0)
  {
    THZCudaStorage *storage = (THZCudaStorage*)THAlloc(sizeof(THZCudaStorage));

    // update heap *before* attempting malloc, to free space for the malloc
    THZCHeapUpdate(state, size * sizeof(float));
    cudaError_t err =
      THZCudaMalloc(state, (void**)&(storage->data), size * sizeof(float));
    if(err != cudaSuccess){
      THZCHeapUpdate(state, -size * sizeof(float));
    }
    THZCudaCheck(err);

    storage->size = size;
    storage->refcount = 1;
    storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
    return storage;
  }
  else
  {
    return THZCudaStorage_new(state);
  }
}

THZCudaStorage* THZCudaStorage_newWithSize1(THCState *state, float data0)
{
  THZCudaStorage *self = THZCudaStorage_newWithSize(state, 1);
  THZCudaStorage_set(state, self, 0, data0);
  return self;
}

THZCudaStorage* THZCudaStorage_newWithSize2(THCState *state, float data0, float data1)
{
  THZCudaStorage *self = THZCudaStorage_newWithSize(state, 2);
  THZCudaStorage_set(state, self, 0, data0);
  THZCudaStorage_set(state, self, 1, data1);
  return self;
}

THZCudaStorage* THZCudaStorage_newWithSize3(THCState *state, float data0, float data1, float data2)
{
  THZCudaStorage *self = THZCudaStorage_newWithSize(state, 3);
  THZCudaStorage_set(state, self, 0, data0);
  THZCudaStorage_set(state, self, 1, data1);
  THZCudaStorage_set(state, self, 2, data2);
  return self;
}

THZCudaStorage* THZCudaStorage_newWithSize4(THCState *state, float data0, float data1, float data2, float data3)
{
  THZCudaStorage *self = THZCudaStorage_newWithSize(state, 4);
  THZCudaStorage_set(state, self, 0, data0);
  THZCudaStorage_set(state, self, 1, data1);
  THZCudaStorage_set(state, self, 2, data2);
  THZCudaStorage_set(state, self, 3, data3);
  return self;
}

THZCudaStorage* THZCudaStorage_newWithMapping(THCState *state, const char *fileName, long size, int isShared)
{
  THError("not available yet for THZCudaStorage");
  return NULL;
}

THZCudaStorage* THZCudaStorage_newWithData(THCState *state, cuComplex *data, long size)
{
  THZCudaStorage *storage = (THZCudaStorage*)THAlloc(sizeof(THZCudaStorage));
  storage->data = data;
  storage->size = size;
  storage->refcount = 1;
  storage->flag = TH_STORAGE_REFCOUNTED | TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM;
  return storage;
}

void THZCudaStorage_retain(THCState *state, THZCudaStorage *self)
{
  if(self && (self->flag & TH_STORAGE_REFCOUNTED))
    THAtomicIncrementRef(&self->refcount);
}

void THZCudaStorage_free(THCState *state, THZCudaStorage *self)
{
  if(!(self->flag & TH_STORAGE_REFCOUNTED))
    return;

  if (THAtomicDecrementRef(&self->refcount))
  {
    if(self->flag & TH_STORAGE_FREEMEM) {
      THZCHeapUpdate(state, -self->size * sizeof(float));
      THZCudaCheck(THZCudaFree(state, self->data));
    }
    THFree(self);
  }
}
