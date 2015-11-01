#ifndef THZC_GENERAL_INC
#define THZC_GENERAL_INC

#include "THGeneral.h"
#include "THAllocator.h"
#include "THZ.h"
#include "THC/THC.h"


#undef log1p

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <cuComplex.h>
#include <complex.h>

#define cx float _Complex           // used in all public interfaces
typedef cuComplex cux;              // used in ZCudaStorage

/* #undef USE_MAGMA */

#ifdef __cplusplus
# define THZC_EXTERNC extern "C"
#else
# define THZC_EXTERNC extern
#endif

#ifdef _WIN32
# ifdef THZC_EXPORTS
#  define THZC_API THZC_EXTERNC __declspec(dllexport)
# else
#  define THZC_API THZC_EXTERNC __declspec(dllimport)
# endif
#else
# define THZC_API THZC_EXTERNC
#endif

#ifndef THAssert
#define THAssert(exp)                                                   \
  do {                                                                  \
    if (!(exp)) {                                                       \
      _THError(__FILE__, __LINE__, "assert(%s) failed", #exp);          \
    }                                                                   \
  } while(0)
#endif

struct THCRNGState;  /* Random number generator state. */

THZC_API void THZCudaInit(THCState* state);
THZC_API void THZCudaShutdown(THCState* state);
THZC_API void THZCudaEnablePeerToPeerAccess(THCState* state);

THZC_API struct cudaDeviceProp* THCState_getCurrentDeviceProperties(THCState* state);

THZC_API void THZCMagma_init(THCState *state);

/* State manipulators and accessors */
THZC_API int THCState_getNumDevices(THCState* state);
THZC_API void THCState_reserveStreams(THCState* state, int numStreams);
THZC_API int THCState_getNumStreams(THCState* state);

THZC_API cudaStream_t THCState_getDeviceStream(THCState *state, int device, int stream);
THZC_API cudaStream_t THCState_getCurrentStream(THCState *state);
THZC_API int THCState_getCurrentStreamIndex(THCState *state);
THZC_API void THCState_setStream(THCState *state, int device, int stream);
THZC_API void THCState_setStreamForCurrentDevice(THCState *state, int stream);

THZC_API void THCState_reserveBlasHandles(THCState* state, int numHandles);
THZC_API int THCState_getNumBlasHandles(THCState* state);

THZC_API cublasHandle_t THCState_getDeviceBlasHandle(THCState *state, int device, int handle);
THZC_API cublasHandle_t THCState_getCurrentBlasHandle(THCState *state);
THZC_API int THCState_getCurrentBlasHandleIndex(THCState *state);
THZC_API void THCState_setBlasHandle(THCState *state, int device, int handle);
THZC_API void THCState_setBlasHandleForCurrentDevice(THCState *state, int handle);

/* For the current device and stream, returns the allocated scratch space */
THZC_API void* THCState_getCurrentDeviceScratchSpace(THCState* state);
THZC_API void* THCState_getDeviceScratchSpace(THCState* state, int device, int stream);
THZC_API size_t THCState_getCurrentDeviceScratchSpaceSize(THCState* state);
THZC_API size_t THCState_getDeviceScratchSpaceSize(THCState* state, int device);

#define THZCudaCheck(err)  __THZCudaCheck(err, __FILE__, __LINE__)
#define THZCublasCheck(err)  __THZCublasCheck(err,  __FILE__, __LINE__)

THZC_API void __THZCudaCheck(cudaError_t err, const char *file, const int line);
THZC_API void __THZCublasCheck(cublasStatus_t status, const char *file, const int line);

THZC_API cudaError_t THZCudaMalloc(THCState *state, void **ptr, size_t size);
THZC_API cudaError_t THZCudaFree(THCState *state, void *ptr);
THZC_API void THZCSetGCHandler(THCState *state,
                             void (*torchGCHandlerFunction)(void *data),
                             void *data );
THZC_API void THZCHeapUpdate(THCState *state, long size);

#endif
