#include "THZCReduceApplyUtils.cuh"
#include "THZCTensorCopy.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

template <typename IndexType, int Power2SortSize>
__device__ __forceinline__ IndexType
getSortSliceLinearIndex() {
  // linear block ID -> slice we are sorting (one per block)
  return getLinearBlockId<IndexType>();
}

// Returns 2^(ceil(lg(n)) from Stanford bit twiddling hacks
unsigned long nextHighestPowerOf2(unsigned long n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n |= n >> 32;
  n++;

  return n;
}

template <typename T>
struct LTComp {
  __device__ __forceinline__ bool operator()(const T& a, const T& b) const {
    return (a < b);
  }
};

template <typename T>
struct GTComp {
  __device__ __forceinline__ bool operator()(const T& a, const T& b) const {
    return (a > b);
  }
};

template <typename Comparator, typename K, typename V>
__device__ __forceinline__ void bitonicSwap(K& kA, V& vA,
                                            K& kB, V& vB,
                                            bool dir,
                                            const Comparator& comp) {
  // Entries with -1 indices (not real data; out of bounds) always
  // sort to the end
  bool val = (comp(kA, kB) && (vA != -1)) || (vB == -1);
  if (val == dir) {
    K k = kA;
    kA = kB;
    kB = k;

    V v = vA;
    vA = vB;
    vB = v;
  }
};

template <typename Comparator, typename K, typename V,
          typename IndexType, int Power2SortSize>
__device__ inline void bitonicSort(K keys[Power2SortSize],
                                   V values[Power2SortSize],
                                   const Comparator& comp) {
#pragma unroll
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((threadIdx.x & (size / 2)) != 0);

#pragma unroll
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {

      // Single warp per slice is completely synchronous
      if (Power2SortSize > 64) {
        __syncthreads();
      }

      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      bitonicSwap<Comparator, K, V>(
        keys[pos], values[pos], keys[pos + stride], values[pos + stride],
        flag, comp);
    }
  }

#pragma unroll
  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {
    // Single warp per slice is completely synchronous
    if (Power2SortSize > 64) {
      __syncthreads();
    }

    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    bitonicSwap<Comparator, K, V>(
      keys[pos], values[pos], keys[pos + stride], values [pos + stride],
      false, comp);
  }

  // Single warp per slice is completely synchronous
  if (Power2SortSize > 64) {
    __syncthreads();
  }
}

template <typename Comparator, typename IndexType, int Dims, int Power2SortSize>
__global__ void
THZCudaTensor_bitonicSortWithIndex(TensorInfo<IndexType> sorted,
                                  TensorInfo<IndexType> indices,
                                  TensorInfo<IndexType> input,
                                  int dim,
                                  IndexType totalSlices,
                                  IndexType sliceSize,
                                  IndexType sliceStride,
                                  IndexType outSize,
                                  IndexType outStride,
                                  const Comparator comp) {
  // Find the slice of the tensor that we are sorting
  const IndexType linearIndex =
    getSortSliceLinearIndex<IndexType, Power2SortSize>();

  // Tiling the slices could have us be out of bounds, if there are a
  // lot of slices to sort
  if (linearIndex >= totalSlices) {
    return;
  }

  __shared__ float keys[Power2SortSize];
  __shared__ int values[Power2SortSize];

  // Read unsorted values
  const IndexType inputStartOffset =
    IndexToOffset<IndexType, Dims>::get(linearIndex, input);

  // Each thread is responsible for loading and storing 2 elements
  const int elem1 = threadIdx.x;
  const int elem2 = threadIdx.x + (Power2SortSize / 2);

  keys[elem1] = (elem1 < sliceSize) ?
    input.data[inputStartOffset + elem1 * sliceStride] :
    0.0f; // doesn't matter, element val out of bounds
  // Torch indices are 1-based (hence the +1)
  values[elem1] = (elem1 < sliceSize) ? (elem1 + 1) :
    -1; // out of bounds
  keys[elem2] = (elem2 < sliceSize) ?
    input.data[inputStartOffset + elem2 * sliceStride] :
    0.0f; // doesn't matter, element val out of bounds
  // Torch indices are 1-based (hence the +1)
  values[elem2] = (elem2 < sliceSize) ? (elem2 + 1) :
    -1; // out of bounds

  // Sort!
  bitonicSort<Comparator, float, int, IndexType, Power2SortSize>(
    keys, values, comp);

  // Write sorted values; indices have same layout
  const IndexType sortedStartOffset =
    IndexToOffset<IndexType, -1>::get(linearIndex, sorted);

  const IndexType out1 = sortedStartOffset + elem1 * outStride;
  // elem1 values are always valid, since otherwise we would have
  // chosen the next smallest power-of-2 for sorting
  sorted.data[out1] = keys[elem1];
  indices.data[out1] = values[elem1];

  const IndexType out2 = sortedStartOffset + elem2 * outStride;
  // elem2 values might be out-of-range, if the data size we are
  // sorting is not a power-of-2
  if (values[elem2] != -1) {
    sorted.data[out2] = keys[elem2];
    indices.data[out2] = values[elem2];
  }
}

bool THZCudaTensor_sortImpl(THCState* state,
                           THZCudaTensor* sorted,
                           THZCudaTensor* indices,
                           THZCudaTensor* input,
                           int dim, bool dir) {
  long inElements = THZCudaTensor_nElement(state, input);

  long sliceSize = THZCudaTensor_size(state, input, dim);
  long sliceStride = THZCudaTensor_stride(state, input, dim);
  long slices = inElements / sliceSize;

  long outSize = THZCudaTensor_size(state, sorted, dim);
  long outStride = THZCudaTensor_stride(state, sorted, dim);

  if (THZCudaTensor_nDimension(state, input) > MAX_CUTORCH_DIMS) {
    // Too many dimensions
    return false;
  }

  if (THZCudaTensor_nDimension(state, input) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  // The amount of shared memory and block size is based on
  // 2^ceil(lg(n)); we choose that sorting implementation for a given
  // size.
  long ceilPowerOf2 = nextHighestPowerOf2(sliceSize);

  // Only handle 1-2048 at the moment
  if (ceilPowerOf2 > 2048) {
    return false;
  }

  const dim3 block(ceilPowerOf2 / 2);

  // The grid is based on the number of independent slices that we
  // have to sort; one block per slice
  dim3 grid;
  if (!THZC_getGridFromTiles(slices, grid)) {
    return false;
  }

#define HANDLE_CASE(TYPE, A, SIZE)                                      \
  if (dir) {                                                            \
    THZCudaTensor_bitonicSortWithIndex<GTComp<float>, TYPE, A, SIZE>     \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(           \
        sortedInfo, indicesInfo, inputInfo,                             \
        dim,                                                            \
        slices, (TYPE) sliceSize, (TYPE) sliceStride,                   \
        (TYPE) outSize, (TYPE) outStride,                               \
        GTComp<float>());                                               \
  } else {                                                              \
    THZCudaTensor_bitonicSortWithIndex<LTComp<float>, TYPE, A, SIZE>     \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(           \
        sortedInfo, indicesInfo, inputInfo,                             \
        dim,                                                            \
        slices, (TYPE) sliceSize, (TYPE) sliceStride,                   \
        (TYPE) outSize, (TYPE) outStride,                               \
        LTComp<float>());                                               \
  }

#define HANDLE_SORT_CASE(TYPE, A)               \
  {                                             \
    switch (ceilPowerOf2) {                     \
      case 2048:                                \
      HANDLE_CASE(TYPE, A, 2048);               \
      break;                                    \
      case 1024:                                \
      HANDLE_CASE(TYPE, A, 1024);               \
      break;                                    \
      case 512:                                 \
      HANDLE_CASE(TYPE, A, 512);                \
      break;                                    \
      case 256:                                 \
      HANDLE_CASE(TYPE, A, 256);                \
      break;                                    \
      case 128:                                 \
      HANDLE_CASE(TYPE, A, 128);                \
      break;                                    \
      case 64:                                  \
      HANDLE_CASE(TYPE, A, 64);                 \
      break;                                    \
      case 32:                                  \
      HANDLE_CASE(TYPE, A, 32);                 \
      break;                                    \
      case 16:                                  \
      HANDLE_CASE(TYPE, A, 16);                 \
      break;                                    \
      case 8:                                   \
      HANDLE_CASE(TYPE, A, 8);                  \
      break;                                    \
      case 4:                                   \
      HANDLE_CASE(TYPE, A, 4);                  \
      break;                                    \
      case 2:                                   \
      HANDLE_CASE(TYPE, A, 2);                  \
      break;                                    \
      case 1:                                   \
      HANDLE_CASE(TYPE, A, 1);                  \
      break;                                    \
      default:                                  \
      assert(false);                            \
    }                                           \
  }

#define HANDLE_A_CASE(TYPE, A)                      \
  {                                                 \
    if (inputInfo.isContiguous()) {                 \
      HANDLE_SORT_CASE(TYPE, -2);                   \
    } else {                                        \
      switch (A) {                                  \
        case 1:                                     \
        HANDLE_SORT_CASE(TYPE, 1);                  \
          break;                                    \
        case 2:                                     \
        HANDLE_SORT_CASE(TYPE, 2);                  \
          break;                                    \
        case 3:                                     \
        HANDLE_SORT_CASE(TYPE, 3);                  \
          break;                                    \
        default:                                    \
        HANDLE_SORT_CASE(TYPE, -1);                 \
          break;                                    \
      }                                             \
    }                                               \
  }

  if (THZC_canUse32BitIndexMath(state, input)) {
    // In order to get to the right offset for the slice we are
    // sorting, set `dim` size to 1 (the `dropDim` argument)
    TensorInfo<unsigned int> sortedInfo(state, sorted, dim);
    TensorInfo<unsigned int> indicesInfo(state, indices, dim);
    TensorInfo<unsigned int> inputInfo(state, input, dim);

    HANDLE_A_CASE(unsigned int, inputInfo.dims);
  } else {
    // In order to get to the right offset for the slice we are
    // sorting, set `dim` size to 1 (the `dropDim` argument)
    TensorInfo<unsigned long> sortedInfo(state, sorted, dim);
    TensorInfo<unsigned long> indicesInfo(state, indices, dim);
    TensorInfo<unsigned long> inputInfo(state, input, dim);

    // long case is rare, just instantiate these versions
    if (inputInfo.isContiguous()) {
      HANDLE_SORT_CASE(unsigned long, -2);
    } else {
      HANDLE_SORT_CASE(unsigned long, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_SORT_CASE
#undef HANDLE_A_CASE

  return true;
}

// `base` is the base address of a tensor
// For each slice (defined as a linear point of `out`, from 0 ->
// (sliceSize - 1) * sliceStride, we fill that slice from `0` to
// `sliceSize - 1`.
__global__ void
THZCudaTensor_fillSliceWithIndex(TensorInfo<unsigned long> out,
                                long totalSlices,
                                long sliceSize,
                                long sliceStride) {
  long slice = getLinearBlockId<long>();

  if (slice >= totalSlices) {
    return;
  }

  const unsigned long offset =
    IndexToOffset<unsigned long, -1>::get(slice, out);

  for (long i = threadIdx.x; i < sliceSize; i += blockDim.x) {
    // Torch indices are 1-based (hence the +1)
    out.data[offset + i * sliceStride] = (float) i + 1;
  }
}

bool shouldSortThrust(THCState* state, THZCudaTensor* input, int dim) {
  long totalElements = THZCudaTensor_nElement(state, input);
  long sliceSize = THZCudaTensor_size(state, input, dim);
  long numSlices = totalElements / sliceSize;

  // Only bother deferring to Thrust if the sort slice is contiguous,
  // the number of slices are small, and they are large
  return ((THZCudaTensor_stride(state, input, dim) == 1) &&
          numSlices <= 16 &&
          sliceSize > 2048);
}

void THZCudaTensor_sortImplThrust(THCState* state,
                                 THZCudaTensor* sorted,
                                 THZCudaTensor* indices,
                                 THZCudaTensor* input,
                                 int dim, bool dir) {
  // Fill the indices as values that Thrust can use for key/value sorting
  long totalElements = THZCudaTensor_nElement(state, input);
  long sliceSize = THZCudaTensor_size(state, input, dim);
  long sliceStride = THZCudaTensor_stride(state, input, dim);
  long numSlices = totalElements / sliceSize;

  // Copy input to sorted, since we sort in place
  if (sorted != input) {
    THZCudaTensor_copy(state, sorted, input);
  }

  TensorInfo<unsigned long> sortedInfo(state, sorted, dim);
  TensorInfo<unsigned long> indicesInfo(state, indices, dim);

  dim3 grid;
  THZC_getGridFromTiles(numSlices, grid);

  THZCudaTensor_fillSliceWithIndex<<<grid, min((long long)sliceSize, 1024LL),
                                    0, THCState_getCurrentStream(state)>>>(
    indicesInfo, numSlices, sliceSize, sliceStride);
  THZCudaCheck(cudaGetLastError());

  for (long slice = 0; slice < numSlices; ++slice) {
    unsigned long sortedStart =
      IndexToOffset<unsigned long, -1>::get(slice, sortedInfo);
    unsigned long indicesStart =
      IndexToOffset<unsigned long, -1>::get(slice, indicesInfo);

    thrust::device_ptr<float>
      sortedSliceStart(THZCudaTensor_data(state, sorted) +
                       sortedStart);
    thrust::device_ptr<float>
      sortedSliceEnd(THZCudaTensor_data(state, sorted) +
                     sortedStart + sliceSize);
    thrust::device_ptr<float>
      indicesSliceStart(THZCudaTensor_data(state, indices) +
                        indicesStart);

    if (dir) {
      thrust::sort_by_key(
#if CUDA_VERSION >= 7000
        thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
        sortedSliceStart, sortedSliceEnd, indicesSliceStart,
        thrust::greater<float>());
    } else {
      thrust::sort_by_key(
#if CUDA_VERSION >= 7000
        thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
        sortedSliceStart, sortedSliceEnd, indicesSliceStart,
        thrust::less<float>());
    }
  }
}

THZC_API void THZCudaTensor_sort(THCState* state,
                               THZCudaTensor *sorted,
                               THZCudaTensor *indices,
                               THZCudaTensor *input,
                               int dim, int order) {
  THAssert(THZCudaTensor_checkGPU(state, 3, sorted, indices, input));
  // Make sure sufficient output space is allocated
  THZCudaTensor_resizeAs(state, sorted, input);
  THZCudaTensor_resizeAs(state, indices, input);

  // If we think Thrust will be more efficient, use that
  if (shouldSortThrust(state, input, dim)) {
    THZCudaTensor_sortImplThrust(state, sorted, indices, input,
                                dim, (bool) order);
    return;
  }

  // Otherwise, try to use our blockwide sort kernel per each reduction slice
  if (THZCudaTensor_sortImpl(state, sorted, indices, input,
                            dim, (bool) order)) {
    return;
  }

  // Fall back to Thrust if our kernel can't handle the input
  THZCudaTensor_sortImplThrust(state, sorted, indices, input,
                              dim, (bool) order);

  THZCudaCheck(cudaGetLastError());
}
