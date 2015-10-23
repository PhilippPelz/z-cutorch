#ifndef THZC_DEVICE_TENSOR_UTILS_INC
#define THZC_DEVICE_TENSOR_UTILS_INC

#include "THZCDeviceTensor.cuh"
#include "THZCTensor.h"

/// Constructs a THZCDeviceTensor initialized from a THZCudaTensor. Will
/// error if the dimensionality does not match exactly.
template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
THZCDeviceTensor<T, Dim, IndexT, PtrTraits>
toDeviceTensor(THCState* state, THZCudaTensor* t);

template <typename T, int Dim, typename IndexT>
THZCDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>
toDeviceTensor(THCState* state, THZCudaTensor* t) {
  return toDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>(state, t);
}

template <typename T, int Dim>
THZCDeviceTensor<T, Dim, int, DefaultPtrTraits>
toDeviceTensor(THCState* state, THZCudaTensor* t) {
  return toDeviceTensor<T, Dim, int, DefaultPtrTraits>(state, t);
}

/// Constructs a DeviceTensor initialized from a THZCudaTensor by
/// upcasting or downcasting the tensor to that of a different
/// dimension.
template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
THZCDeviceTensor<T, Dim, IndexT, PtrTraits>
toDeviceTensorCast(THCState* state, THZCudaTensor* t);

template <typename T, int Dim, typename IndexT>
THZCDeviceTensor<T, Dim, IndexT, DefaultPtrTraits>
toDeviceTensorCast(THCState* state, THZCudaTensor* t) {
  return toDeviceTensorCast<T, Dim, IndexT, DefaultPtrTraits>(state, t);
}

template <typename T, int Dim>
THZCDeviceTensor<T, Dim, int, DefaultPtrTraits>
toDeviceTensorCast(THCState* state, THZCudaTensor* t) {
  return toDeviceTensorCast<T, Dim, int, DefaultPtrTraits>(state, t);
}

#include "THZCDeviceTensorUtils-inl.cuh"

#endif // THZC_DEVICE_TENSOR_UTILS_INC
