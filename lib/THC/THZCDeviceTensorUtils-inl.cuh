#include <limits>

template <typename T, int Dim,
          typename IndexT, template <typename U> class PtrTraits>
THZCDeviceTensor<T, Dim, IndexT, PtrTraits>
toDeviceTensor(THCState* state, THZCudaTensor* t) {
  if (Dim != THZCudaTensor_nDimension(state, t)) {
    THError("THZCudaTensor dimension mismatch");
  }

  // Determine the maximum offset into the tensor achievable; `IndexT`
  // must be smaller than this type in order to use it.
  long maxOffset = 0;
  IndexT sizes[Dim];
  IndexT strides[Dim];

  for (int i = 0; i < Dim; ++i) {
    long size = THZCudaTensor_size(state, t, i);
    long stride = THZCudaTensor_stride(state, t, i);

    maxOffset += (size - 1) * stride;

    sizes[i] = (IndexT) size;
    strides[i] = (IndexT) stride;
  }

  if (maxOffset > std::numeric_limits<IndexT>::max()) {
    THError("THZCudaTensor sizes too large for THZCDeviceTensor conversion");
  }

  return THZCDeviceTensor<T, Dim, IndexT, PtrTraits>(
    THZCudaTensor_data(state, t), sizes, strides);
}

namespace detail {

// Add a layer of SFINAE to support static_assert
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct UpcastTHZCRoot {
  static THZCDeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THZCudaTensor* t);
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct UpcastTHZC :
      UpcastTHZCRoot<T, Dim, IndexT, PtrTraits, NewDim, B> {
};

// Never instantiated SFINAE purposes only
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct UpcastTHZC<T, Dim, IndexT, PtrTraits, NewDim, false> :
      UpcastTHZCRoot<T, Dim, IndexT, PtrTraits, NewDim, false> {
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct UpcastTHZC<T, Dim, IndexT, PtrTraits, NewDim, true> :
      UpcastTHZCRoot<T, Dim, IndexT, PtrTraits, NewDim, true>  {
  static THZCDeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THZCudaTensor* t) {
    thc_static_assert(NewDim > Dim);
    return toDeviceTensor<T, Dim, IndexT, PtrTraits>(state, t).
      template upcastOuter<NewDim>();
  }
};

// Add a layer of SFINAE to support static_assert
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct DowncastTHZCRoot {
  static THZCDeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THZCudaTensor* t);
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim, bool B>
struct DowncastTHZC :
      DowncastTHZCRoot<T, Dim, IndexT, PtrTraits, NewDim, B> {
};

// Never instantiated SFINAE purposes only
template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct DowncastTHZC<T, Dim, IndexT, PtrTraits, NewDim, false> :
      DowncastTHZCRoot<T, Dim, IndexT, PtrTraits, NewDim, false> {
};

template <typename T, int Dim, typename IndexT,
          template <typename U> class PtrTraits,
          int NewDim>
struct DowncastTHZC<T, Dim, IndexT, PtrTraits, NewDim, true> :
      DowncastTHZCRoot<T, Dim, IndexT, PtrTraits, NewDim, true>  {
  static THZCDeviceTensor<T, NewDim, IndexT, PtrTraits>
  make(THCState* state, THZCudaTensor* t) {
    thc_static_assert(NewDim < Dim);
    return toDeviceTensor<T, Dim, IndexT, PtrTraits>(state, t).
      template downcastOuter<NewDim>();
  }
};

} // namespace detail

#define SWITCH_UNROLL_CUDA_CAST_FACTORY(i)                              \
  case i:                                                               \
  if (NewDim > i) {                                                     \
    return detail::UpcastTHZC<T, i, IndexT,                              \
                             PtrTraits, NewDim, (NewDim > i)>::         \
      make(state, t);                                                   \
  } else if (NewDim == i) {                                             \
    return toDeviceTensor<T, NewDim, IndexT, PtrTraits>(state, t);      \
  } else {                                                              \
    return detail::DowncastTHZC<T, i, IndexT,                            \
                               PtrTraits, NewDim, (NewDim < i)>::       \
      make(state, t);                                                   \
  }                                                                     \
  /* break; */

template <typename T, int NewDim,
          typename IndexT, template <typename U> class PtrTraits>
THZCDeviceTensor<T, NewDim, IndexT, PtrTraits>
toDeviceTensorCast(THCState* state, THZCudaTensor* t) {
  switch (THZCudaTensor_nDimension(state, t)) {
    SWITCH_UNROLL_CUDA_CAST_FACTORY(1);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(2);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(3);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(4);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(5);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(6);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(7);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(8);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(9);
    SWITCH_UNROLL_CUDA_CAST_FACTORY(10);
    default:
      ;
  }

  // Not implemented
  THError("THZCDeviceTensor dimension size not supported");
  return NULL; /* never enters this piece, appeasing compiler warnings */
}

#undef SWITCH_UNROLL_CUDA_CAST_FACTORY
