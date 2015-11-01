#ifndef THZC_REDUCE_INC
#define THZC_REDUCE_INC

//
// This file contains dimension reduction operation functions and
// kernels that work on both contiguous and non-contiguous tensor
// arguments of arbitrary (up to MAX_CUTORCH_DIMS) dimensioned
// arguments without copying or temporary storage.
//

#include "THZCReduceApplyUtils.cuh"
#include <complex.h>
#define cx float _Complex

// Threads per thread block
#define THZC_NONCONTIG_REDUCE_BLOCK_SIZE 32 * 16

template <typename IndexType>
__device__ __forceinline__ IndexType getReduceNoncontigDimSliceIndex() {
  // Each thread handles one slice
  return getLinearBlockId<IndexType>() * THZC_NONCONTIG_REDUCE_BLOCK_SIZE +
         threadIdx.x;
}

// Kernel that handles an entire reduction of a slice of a tensor per each
// thread
template <typename ModifyOp, typename ReduceOp, typename IndexType, int ADims,
          int BDims>
__launch_bounds__(32 * 16, 4) __global__ void THZCudaTensor_reduceNoncontigDim(
    ZTensorInfo<IndexType> out, ZTensorInfo<IndexType> in,
    IndexType reductionStride, IndexType reductionSize, IndexType totalSlices,
    ccx init, ModifyOp modifyOp, ReduceOp reduceOp) {
  const IndexType sliceIndex = getReduceNoncontigDimSliceIndex<IndexType>();

  if (sliceIndex >= totalSlices) {
    return;
  }

  // Each thread picks a point in `out` and `in` for which it is
  // producing the reduction
  const IndexType outOffset =
      ZIndexToOffset<IndexType, ADims>::get(sliceIndex, out);
  const IndexType inBaseOffset =
      ZIndexToOffset<IndexType, BDims>::get(sliceIndex, in);

  // For each point in reductionSize, reduce into `r`
  IndexType inOffset = inBaseOffset;
  ccx r = (ccx)init;

  for (IndexType i = 0; i < reductionSize; ++i) {
    r = reduceOp(r, modifyOp(in.data[inOffset]));
    inOffset += reductionStride;
  }

  // Write out reduced value
  out.data[outOffset] = (ccx)r;
}

template <typename IndexType>
__device__ __forceinline__ IndexType getReduceContigDimSliceIndex() {
  // Each block handles one slice
  return getLinearBlockId<IndexType>();
}

// Kernel that handles an entire reduction of a slice of a tensor per
// each block
template <typename ModifyOp, typename ReduceOp, typename IndexType, int ADims,
          int BDims>
__global__ void
THZCudaTensor_reduceContigDim(ZTensorInfo<IndexType> out,
                              ZTensorInfo<IndexType> in, IndexType reductionSize,
                              IndexType totalSlices, ccx init,
                              ModifyOp modifyOp, ReduceOp reduceOp) {
  const IndexType sliceIndex = getReduceContigDimSliceIndex<IndexType>();

  if (sliceIndex >= totalSlices) {
    return;
  }

  // Get the offset in `out` for the reduction
  const IndexType outOffset =
      ZIndexToOffset<IndexType, ADims>::get(sliceIndex, out);

  // Get the base offset in `in` for this block's reduction
  const IndexType inBaseOffset =
      ZIndexToOffset<IndexType, BDims>::get(sliceIndex, in);

  // Each thread in the block will reduce some subset of elements in
  // the slice. The elements are guaranteed contiguous starting at
  // `inBaseOffset`.
  ccx r = (ccx)init;
  for (IndexType i = threadIdx.x; i < reductionSize; i += blockDim.x) {
    r = reduceOp(r, modifyOp(in.data[inBaseOffset + i]));
  }

  // Reduce within the block
  extern __shared__ ccx smem[];
  r = reduceBlock<ccx, ReduceOp>(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out reduced value
    out.data[outOffset] = r;
  }
}

inline dim3 getNoncontigReduceBlock() {
  return dim3(THZC_NONCONTIG_REDUCE_BLOCK_SIZE);
}

inline dim3 getContigReduceBlock(long numSlices, long reductionSize) {
  // If the number of slices is low but the reduction dimension size
  // is high, then we should increase block size for greater parallelism.
  // Aim for at least 32 warps per SM (assume 15 SMs; don't bother
  // inquiring the real number for now).
  int maxWarps = 4; // better occupancy if many blocks are around
  // For numSlices > 15 * 8, there are > 32 warps active per SM.
  if (numSlices < 15 * 8) {
    maxWarps = 8;
    if (numSlices < 15 * 4) {
      maxWarps = 16;
      if (numSlices < 15 * 2) {
        maxWarps = 32;
      }
    }
  }

  // Scale up block size based on the reduction dimension size
  long warpsInReductionSize = THZCCeilDiv(reductionSize, 32L);
  int numWarps = warpsInReductionSize > (long)maxWarps
                     ? maxWarps
                     : (int)warpsInReductionSize;
  return dim3(numWarps * 32);
}

inline bool getNoncontigReduceGrid(long elements, dim3 &grid) {
  // One output point per thread
  return THZC_getGridFromTiles(
      THZCCeilDiv(elements, (long)THZC_NONCONTIG_REDUCE_BLOCK_SIZE), grid);
}

inline bool getContigReduceGrid(long elements, dim3 &grid) {
  // One output point per block
  return THZC_getGridFromTiles(elements, grid);
}

// Performs a reduction out[..., 0, ...] = reduce_i(modify(in[..., i, ...])) for
// all in where i and the out's 0 are indexed at dimension `dim`
template <typename ModifyOp, typename ReduceOp>
bool THZCudaTensor_reduceDim(THCState *state, THZCudaTensor *out,
                             THZCudaTensor *in, const ModifyOp &modifyOp,
                             const ReduceOp &reduceOp, ccx init, int dim) {
  long inElements = THZCudaTensor_nElement(state, in);

  long reductionSize = THZCudaTensor_size(state, in, dim);
  long reductionStride = THZCudaTensor_stride(state, in, dim);
  long outElements = inElements / reductionSize;

  if (THZCudaTensor_nDimension(state, out) > MAX_CUTORCH_DIMS ||
      THZCudaTensor_nDimension(state, in) > MAX_CUTORCH_DIMS) {
    return false;
  }

  if (THZCudaTensor_nDimension(state, in) == 0) {
    // Zero-dim tensor; do nothing
    return true;
  }

  // Is the reduction dimension contiguous? If so, then we can use a
  // shared memory reduction kernel to increase performance.
  bool contigReduction = (reductionStride == 1);

  dim3 block;
  dim3 grid;
  int smemSize = 0; // contiguous reduction uses smem
  if (contigReduction) {
    if (!getContigReduceGrid(outElements, grid)) {
      return false;
    }

    block = getContigReduceBlock(outElements, reductionSize);
    smemSize = sizeof(ccx) * block.x;
  } else {
    if (!getNoncontigReduceGrid(outElements, grid)) {
      return false;
    }

    block = getNoncontigReduceBlock();
  }

  // Resize out to correspond to the reduced size
  THLongStorage *sizes = THZCudaTensor_newSizeOf(state, in);
  THLongStorage_set(sizes, dim, 1);
  THZCudaTensor_resize(state, out, sizes, NULL);
  THLongStorage_free(sizes);

// It is possible that the tensor dimensions are able to be collapsed,
// and thus we can reduce the actual code complexity of the copy by
// exploiting this knowledge statically, since the div/mod is the
// most expensive part of the operation, more so than memory accesses.
// For instance, when copying a non-contiguous to a contiguous tensor
// (or vice versa), the contiguous tensor can be collapsed to one
// dimension, and the loop to translate the linear index to the array
// index can be similarly collapsed. That is what this unrolling is for.
#define HANDLE_CASE(TYPE, OUT, IN)                                             \
  if (contigReduction) {                                                       \
    THZCudaTensor_reduceContigDim<ModifyOp, ReduceOp, TYPE, OUT, IN> << <      \
        grid, block, smemSize, THCState_getCurrentStream(state)>>>             \
        (outInfo, inInfo, reductionSize, (TYPE)outElements,                    \
         init, modifyOp, reduceOp);                 \
  } else {                                                                     \
    THZCudaTensor_reduceNoncontigDim<ModifyOp, ReduceOp, TYPE, OUT, IN> << <   \
        grid, block, 0, THCState_getCurrentStream(state)>>>                    \
        (outInfo, inInfo, reductionStride, reductionSize, (TYPE)outElements,   \
         init, modifyOp, reduceOp);                 \
  }

#define HANDLE_IN_CASE(TYPE, OUT, IN)                                          \
  {                                                                            \
    if (inInfo.isContiguous()) {                                               \
      HANDLE_CASE(TYPE, OUT, -2);                                              \
    } else {                                                                   \
      switch (IN) {                                                            \
      case 1:                                                                  \
        HANDLE_CASE(TYPE, OUT, 1);                                             \
        break;                                                                 \
      case 2:                                                                  \
        HANDLE_CASE(TYPE, OUT, 2);                                             \
        break;                                                                 \
      case 3:                                                                  \
        HANDLE_CASE(TYPE, OUT, 3);                                             \
        break;                                                                 \
      default:                                                                 \
        HANDLE_CASE(TYPE, OUT, -1);                                            \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
  }

#define HANDLE_OUT_CASE(TYPE, OUT, IN)                                         \
  {                                                                            \
    if (outInfo.isContiguous()) {                                              \
      HANDLE_IN_CASE(TYPE, -2, IN);                                            \
    } else {                                                                   \
      switch (OUT) {                                                           \
      case 1:                                                                  \
        HANDLE_IN_CASE(TYPE, 1, IN);                                           \
        break;                                                                 \
      case 2:                                                                  \
        HANDLE_IN_CASE(TYPE, 2, IN);                                           \
        break;                                                                 \
      case 3:                                                                  \
        HANDLE_IN_CASE(TYPE, 3, IN);                                           \
        break;                                                                 \
      default:                                                                 \
        HANDLE_IN_CASE(TYPE, -1, IN);                                          \
        break;                                                                 \
      }                                                                        \
    }                                                                          \
  }

  if (THZC_canUse32BitIndexMath(state, out) &&
      THZC_canUse32BitIndexMath(state, in)) {
    ZTensorInfo<unsigned int> outInfo(state, out);
    ZTensorInfo<unsigned int> inInfo(state, in, dim);

    HANDLE_OUT_CASE(unsigned int, outInfo.dims, inInfo.dims);
  } else {
    ZTensorInfo<unsigned long> outInfo(state, out);
    ZTensorInfo<unsigned long> inInfo(state, in, dim);

    // For large tensors, we only compile the completely contiguous
    // version and the completely generic version, to reduce
    // compilation time.
    if (outInfo.isContiguous() && inInfo.isContiguous()) {
      HANDLE_CASE(unsigned long, -2, -2);
    } else {
      HANDLE_CASE(unsigned long, -1, -1);
    }
  }
#undef HANDLE_CASE
#undef HANDLE_IN_CASE
#undef HANDLE_OUT_CASE

  return true;
}

#undef THZC_NONCONTIG_REDUCE_BLOCK_SIZE

#endif // THZC_REDUCE_INC
