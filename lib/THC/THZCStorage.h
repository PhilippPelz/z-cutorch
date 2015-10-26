#ifndef THZC_STORAGE_INC
#define THZC_STORAGE_INC

#include "THStorage.h"
#include "THZCGeneral.h"
#include "cuComplex.h"

#define TH_STORAGE_REFCOUNTED 1
#define TH_STORAGE_RESIZABLE  2
#define TH_STORAGE_FREEMEM    4


typedef struct THZCudaStorage
{
    cuComplex *data;
    long size;
    int refcount;
    char flag;
    THZAllocator *allocator;
    void *allocatorContext;
    struct THZCudaStorage *view;
} THZCudaStorage;


THZC_API float* THZCudaStorage_data(THCState *state, const THZCudaStorage*);
THZC_API long THZCudaStorage_size(THCState *state, const THZCudaStorage*);

/* slow access -- checks everything */
THZC_API void THZCudaStorage_set(THCState *state, THZCudaStorage*, long, float);
THZC_API float THZCudaStorage_get(THCState *state, const THZCudaStorage*, long);

THZC_API THZCudaStorage* THZCudaStorage_new(THCState *state);
THZC_API THZCudaStorage* THZCudaStorage_newWithSize(THCState *state, long size);
THZC_API THZCudaStorage* THZCudaStorage_newWithSize1(THCState *state, float);
THZC_API THZCudaStorage* THZCudaStorage_newWithSize2(THCState *state, float, float);
THZC_API THZCudaStorage* THZCudaStorage_newWithSize3(THCState *state, float, float, float);
THZC_API THZCudaStorage* THZCudaStorage_newWithSize4(THCState *state, float, float, float, float);
THZC_API THZCudaStorage* THZCudaStorage_newWithMapping(THCState *state, const char *filename, long size, int shared);

/* takes ownership of data */
THZC_API THZCudaStorage* THZCudaStorage_newWithData(THCState *state, cuComplex *data, long size);

THZC_API THZCudaStorage* THZCudaStorage_newWithAllocator(THCState *state, long size,
                                                      THAllocator* allocator,
                                                      void *allocatorContext);
THZC_API THZCudaStorage* THZCudaStorage_newWithDataAndAllocator(
    THCState *state, cuComplex* data, long size, THAllocator* allocator, void *allocatorContext);

THZC_API void THZCudaStorage_setFlag(THCState *state, THZCudaStorage *storage, const char flag);
THZC_API void THZCudaStorage_clearFlag(THCState *state, THZCudaStorage *storage, const char flag);
THZC_API void THZCudaStorage_retain(THCState *state, THZCudaStorage *storage);

THZC_API void THZCudaStorage_free(THCState *state, THZCudaStorage *storage);
THZC_API void THZCudaStorage_resize(THCState *state, THZCudaStorage *storage, long size);
THZC_API void THZCudaStorage_fill(THCState *state, THZCudaStorage *storage, float value);

#endif
