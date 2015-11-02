local ok, ffi = pcall(require, 'ffi')
if ok then

   local cdefs = [[
typedef struct float2 float2;
typedef struct THZCudaStorage
{
    float2 *data;
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
]]
   ffi.cdef(cdefs)

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

end
