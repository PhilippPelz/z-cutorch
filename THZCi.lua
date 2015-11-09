local ffi = require 'ffi'

ffi.cdef([[
typedef struct float2 cuComplex;
typedef struct THZCudaStorage
{
    cuComplex *data;
    long size;
    int refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
    struct THZCudaStorage *view;
} THZCudaStorage;

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

void THZCudaTensor_abs(THCState *state, THCudaTensor *self, THZCudaTensor *src);
void THZCudaTensor_arg(THCState *state, THCudaTensor *self, THZCudaTensor *src);
void THZCudaTensor_norm(THCState *state, THCudaTensor *self, THZCudaTensor *src);

void THZCudaTensor_real(THCState *state, THCudaTensor *self, THZCudaTensor *src);
void THZCudaTensor_imag(THCState *state, THCudaTensor *self, THZCudaTensor *src);

void THZCudaTensor_zabs(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
void THZCudaTensor_zarg(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
void THZCudaTensor_znorm(THCState *state, THZCudaTensor *self, THZCudaTensor *src);

void THZCudaTensor_zreal(THCState *state, THZCudaTensor *self, THZCudaTensor *src);
void THZCudaTensor_zimag(THCState *state, THZCudaTensor *self, THZCudaTensor *src);

float THZCudaTensor_normall(THCState *state, THZCudaTensor *self, float value);
void THZCudaTensor_normDim(THCState *state, THZCudaTensor *self, THZCudaTensor *src, float value, long dimension);

void THZCudaTensor_polar(THCState *state, THZCudaTensor *self, THCudaTensor *src1, THCudaTensor *src2);

void THZCudaTensor_cim(THCState *state, THZCudaTensor *self, THZCudaTensor *src1, THCudaTensor *src2);
void THZCudaTensor_cre(THCState *state, THZCudaTensor *self, THZCudaTensor *src1, THCudaTensor *src2);
]])

C = ffi.load('THZC')

local Storage = torch.getmetatable('torch.ZCudaStorage')
local Storage_tt = ffi.typeof('THZCudaStorage**')

rawset(Storage, "cdata", function(self) return Storage_tt(self)[0] end)
rawset(Storage, "data", function(self) return Storage_tt(self)[0].data end)
-- Tensor
local Tensor = torch.getmetatable('torch.ZCudaTensor')
local Tensor_tt = ffi.typeof('THZCudaTensor**')

rawset(Tensor, "cdata", function(self) return Tensor_tt(self)[0] end)

rawset(Tensor, "data",
       function(self)
          self = Tensor_tt(self)[0]
          return self.storage ~= nil and self.storage.data + self.storageOffset or nil
       end
)
return C
