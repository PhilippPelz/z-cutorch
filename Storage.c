#include "torch/utils.h"
#include "THZC.h"
#include "THFile.h"
#include "luaT.h"

/* everything is as the generic Storage.c, except few things (see below) */

#define real float complex
#define Real ZCuda
#define TH_GENERIC_FILE "generic/Storage.c"

#define torch_Storage_(NAME) TH_CONCAT_4(torch_,Real,Storage_,NAME)

#define THFile_readRealRaw(file, data, size)                            \
  {                                                                     \
    float *fdata = (float*)THAlloc(sizeof(float)*size);                 \
    THFile_readFloatRaw(file, fdata, size);                             \
    THZCudaCheck(cudaMemcpy(data, fdata, size * sizeof(float), cudaMemcpyHostToDevice)); \
    THFree(fdata);                                                      \
  }

#define THFile_writeRealRaw(file, data, size)                           \
  {                                                                     \
    float *fdata = (float*)THAlloc(sizeof(float)*size);                 \
    THZCudaCheck(cudaMemcpy(fdata, data, size * sizeof(float), cudaMemcpyDeviceToHost)); \
    THFile_writeFloatRaw(file, fdata, size);                            \
    THFree(fdata);                                                      \
  }

#define torch_Storage TH_CONCAT_STRING_3(torch.,Real,Storage)

#include "generic/Storage.c"

#undef real
#undef Real
#undef TH_GENERIC_FILE

/* now we overwrite some methods specific to CudaStorage */
static int zcutorch_ZCudaStorage_copy(lua_State *L)
{
  THCState *state = cutorch_getstate(L);
  THZCudaStorage *storage = luaT_checkudata(L, 1, "torch.CudaStorage");
  void *src;
  if( (src = luaT_toudata(L, 2, "torch.ZCudaStorage")) )
    THZCudaStorage_copy(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )
    THZCudaStorage_copyByte(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CharStorage")) )
    THZCudaStorage_copyChar(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )
    THZCudaStorage_copyShort(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.IntStorage")) )
    THZCudaStorage_copyInt(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.LongStorage")) )
    THZCudaStorage_copyLong(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )
    THZCudaStorage_copyFloat(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )
    THZCudaStorage_copyDouble(state, storage, src);
  else if( (src = luaT_toudata(L, 2, "torch.CudaStorage")) )
    THZCudaStorage_copyZCuda(state, storage, src);
  else
    luaL_typerror(L, 2, "torch.*Storage");

  lua_settop(L, 1);
  return 1;
}

#define CUDA_IMPLEMENT_STORAGE_COPY(TYPEC)                              \
  static int zcutorch_##TYPEC##Storage_copy(lua_State *L)                \
  {                                                                     \
    TH##TYPEC##Storage *storage = luaT_checkudata(L, 1, "torch." #TYPEC "Storage"); \
    void *src;                                                          \
    if( (src = luaT_toudata(L, 2, "torch." #TYPEC "Storage")) )         \
      TH##TYPEC##Storage_copy(storage, src);                            \
    else if( (src = luaT_toudata(L, 2, "torch.ByteStorage")) )          \
      TH##TYPEC##Storage_copyByte(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.CharStorage")) )          \
      TH##TYPEC##Storage_copyChar(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.ShortStorage")) )         \
      TH##TYPEC##Storage_copyShort(storage, src);                       \
    else if( (src = luaT_toudata(L, 2, "torch.IntStorage")) )           \
      TH##TYPEC##Storage_copyInt(storage, src);                         \
    else if( (src = luaT_toudata(L, 2, "torch.LongStorage")) )          \
      TH##TYPEC##Storage_copyLong(storage, src);                        \
    else if( (src = luaT_toudata(L, 2, "torch.FloatStorage")) )         \
      TH##TYPEC##Storage_copyFloat(storage, src);                       \
    else if( (src = luaT_toudata(L, 2, "torch.DoubleStorage")) )        \
      TH##TYPEC##Storage_copyDouble(storage, src);                      \
    else if( (src = luaT_toudata(L, 2, "torch.ZCudaStorage")) )          \
      TH##TYPEC##Storage_copyCuda(cutorch_getstate(L), storage, src);   \
    else                                                                \
      luaL_typerror(L, 2, "torch.*Storage");                            \
                                                                        \
    lua_settop(L, 1);                                                   \
    return 1;                                                           \
}

CUDA_IMPLEMENT_STORAGE_COPY(Byte)
CUDA_IMPLEMENT_STORAGE_COPY(Char)
CUDA_IMPLEMENT_STORAGE_COPY(Short)
CUDA_IMPLEMENT_STORAGE_COPY(Int)
CUDA_IMPLEMENT_STORAGE_COPY(Long)
CUDA_IMPLEMENT_STORAGE_COPY(Float)
CUDA_IMPLEMENT_STORAGE_COPY(Double)

void zcutorch_ZCudaStorage_init(lua_State* L)
{
  /* the standard stuff */
  torch_ZCudaStorage_init(L);

  /* the copy methods */
  {
    int i;

    const void* tnames[8] = {"torch.ByteStorage",
                             "torch.CharStorage",
                             "torch.ShortStorage",
                             "torch.IntStorage",
                             "torch.LongStorage",
                             "torch.FloatStorage",
                             "torch.DoubleStorage",
                             "torch.ZCudaStorage"};

    static int (*funcs[8])(lua_State*) = {zcutorch_ByteStorage_copy,
                                          zcutorch_CharStorage_copy,
                                          zcutorch_ShortStorage_copy,
                                          zcutorch_IntStorage_copy,
                                          zcutorch_LongStorage_copy,
                                          zcutorch_FloatStorage_copy,
                                          zcutorch_DoubleStorage_copy,
                                          zcutorch_ZCudaStorage_copy};

    for(i = 0; i < 8; i++)
    {
      luaT_pushmetatable(L, tnames[i]);
      lua_pushcfunction(L, funcs[i]);
      lua_setfield(L, -2, "copy");
      lua_pop(L, 1);
    }
  }
}
