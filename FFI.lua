local ok, ffi = pcall(require, 'ffi')
if ok then

   local cdefs = [[
typedef struct CUstream_st *cudaStream_t;

struct cublasContext;
typedef struct cublasContext *cublasHandle_t;
typedef struct CUhandle_st *cublasHandle_t;

typedef struct _THCCudaResourcesPerDevice {
  cudaStream_t* streams;
  cublasHandle_t* blasHandles;
  size_t scratchSpacePerStream;
  void** devScratchSpacePerStream;
} THCCudaResourcesPerDevice;


typedef struct THCState
{
  struct THCRNGState* rngState;
  struct cudaDeviceProp* deviceProperties;
  cudaStream_t currentStream;
  cublasHandle_t currentBlasHandle;
  THCCudaResourcesPerDevice* resourcesPerDevice;
  int numDevices;
  int numUserStreams;
  int numUserBlasHandles;
  int currentPerDeviceStream;
  int currentPerDeviceBlasHandle;
  struct THAllocator* cudaHostAllocator;
} THCState;

cudaStream_t THCState_getCurrentStream(THCState *state);

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
