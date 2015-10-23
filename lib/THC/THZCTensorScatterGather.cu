#include "THZCTensorMath.h"
#include "THZCGeneral.h"
#include "THZCApply.cuh"


// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename IndexType, int Dims>
struct IndexToScatterGatherOffsets {
  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<IndexType>& index, IndexType* indexOffset,
      const TensorInfo<IndexType>& t1, IndexType* t1Offset,
      const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }

  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<IndexType>& index, IndexType* indexOffset,
      const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }
};

// Same as above but using a dynamic number of dimensions.
template <typename IndexType>
struct IndexToScatterGatherOffsets<IndexType, -1> {
  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<IndexType>& index, IndexType* indexOffset,
      const TensorInfo<IndexType>& t1, IndexType* t1Offset,
      const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }

  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<IndexType>& index, IndexType* indexOffset,
      const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }
};


template <typename IndexType, int Dims>
__global__ void THZCudaTensor_gatherKernel(
    TensorInfo<IndexType> tensor,
    TensorInfo<IndexType> src,
    TensorInfo<IndexType> index,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          tensor, &tensorOffset,
                                                          src, &srcOffset);

    IndexType indexValue = (IndexType)index.data[indexOffset] - 1;
    srcOffset += indexValue * src.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}

#define RUN(TYPE, DIMS)                                              \
  THZCudaTensor_gatherKernel<TYPE, DIMS>                              \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(        \
          tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

void THZCudaTensor_gather(THCState* state, THZCudaTensor *tensor, THZCudaTensor *src, int dim, THZCudaTensor *index) {
  THAssert(THZCudaTensor_checkGPU(state, 3, tensor, src, index));

  THArgCheck(THZCudaTensor_nDimension(state, src) == THZCudaTensor_nDimension(state, tensor), 2,
             "Input tensor must have same dimensions as output tensor");
  THArgCheck(dim >= 0 && dim < THZCudaTensor_nDimension(state, tensor), 3,
             "Index dimension is out of bounds");
  THArgCheck(THZCudaTensor_nDimension(state, index) == THZCudaTensor_nDimension(state, src), 4,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(THZCudaTensor_isSameSizeAs(state, tensor, index), 4,
             "Index tensor must have the same size as output tensor.");

  for (int d = 0; d < THZCudaTensor_nDimension(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THZCudaTensor_size(state, tensor, d) == THZCudaTensor_size(state, src, d), 2,
                 "Input tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  if (THZCudaTensor_nDimension(state, tensor) > MAX_CUTORCH_DIMS) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  const long totalElements = THZCudaTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THZCudaTensor* oldTensor = NULL;
  if (THZC_overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THZCudaTensor_newContiguous(state, tensor);
  }

  if (THZC_canUse32BitIndexMath(state, tensor) &&
      THZC_canUse32BitIndexMath(state, src) &&
      THZC_canUse32BitIndexMath(state, index)) {
    TensorInfo<unsigned int> tensorInfo(state, tensor, NoCollapseDims);
    TensorInfo<unsigned int> srcInfo(state, src, NoCollapseDims);
    TensorInfo<unsigned int> indexInfo(state, index, NoCollapseDims);

    // Specialize for a small number of dimensions.
    switch (indexInfo.dims) {
      case 1:
        RUN(unsigned int, 1);
        break;
      case 2:
        RUN(unsigned int, 2);
        break;
      case 3:
        RUN(unsigned int, 3);
        break;
      default:
        RUN(unsigned int, -1);
        break;
    }
  } else {
    TensorInfo<unsigned long> tensorInfo(state, tensor, NoCollapseDims);
    TensorInfo<unsigned long> srcInfo(state, src, NoCollapseDims);
    TensorInfo<unsigned long> indexInfo(state, index, NoCollapseDims);

    RUN(unsigned long, -1)
  }

  if (oldTensor) {
    THZCudaTensor_copyIgnoringOverlaps(state, oldTensor, tensor);
    THZCudaTensor_free(state, tensor);
    tensor = oldTensor;
  }
}

#undef RUN


template <typename IndexType, int Dims>
__global__ void THZCudaTensor_scatterKernel(
    TensorInfo<IndexType> tensor,
    TensorInfo<IndexType> src,
    TensorInfo<IndexType> index,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          src, &srcOffset,
                                                          tensor, &tensorOffset);

    IndexType indexValue = (IndexType)index.data[indexOffset] - 1;
    tensorOffset += indexValue * tensor.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}

#define RUN(TYPE, DIMS)                                              \
  THZCudaTensor_scatterKernel<TYPE, DIMS>                             \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(        \
          tensorInfo, srcInfo, indexInfo, dim, (TYPE)totalElements);

void THZCudaTensor_scatter(THCState* state, THZCudaTensor *tensor, int dim, THZCudaTensor *index, THZCudaTensor *src) {
  THAssert(THZCudaTensor_checkGPU(state, 3, tensor, src, index));

  THArgCheck(dim >= 0 && dim < THZCudaTensor_nDimension(state, tensor), 2,
             "Index dimension is out of bounds");
  THArgCheck(THZCudaTensor_nDimension(state, index) == THZCudaTensor_nDimension(state, src), 3,
             "Index tensor must have same dimensions as input tensor");
  THArgCheck(THZCudaTensor_nDimension(state, src) == THZCudaTensor_nDimension(state, tensor), 4,
             "Input tensor must have same dimensions as output tensor");
  THArgCheck(THZCudaTensor_isSameSizeAs(state, src, index), 3,
             "Index tensor must have the same size as input tensor.");

  for (int d = 0; d < THZCudaTensor_nDimension(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THZCudaTensor_size(state, tensor, d) == THZCudaTensor_size(state, src, d), 4,
                 "Input tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  if (THZCudaTensor_nDimension(state, tensor) > MAX_CUTORCH_DIMS) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  const long totalElements = THZCudaTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THZCudaTensor* oldTensor = NULL;
  if (THZC_overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THZCudaTensor_newContiguous(state, tensor);
  }

  if (THZC_canUse32BitIndexMath(state, tensor) &&
      THZC_canUse32BitIndexMath(state, src) &&
      THZC_canUse32BitIndexMath(state, index)) {
    TensorInfo<unsigned int> tensorInfo(state, tensor, NoCollapseDims);
    TensorInfo<unsigned int> srcInfo(state, src, NoCollapseDims);
    TensorInfo<unsigned int> indexInfo(state, index, NoCollapseDims);

    // Specialize for a small number of dimensions.
    switch (indexInfo.dims) {
      case 1:
        RUN(unsigned int, 1);
        break;
      case 2:
        RUN(unsigned int, 2);
        break;
      case 3:
        RUN(unsigned int, 3);
        break;
      default:
        RUN(unsigned int, -1);
        break;
    }
  } else {
    TensorInfo<unsigned long> tensorInfo(state, tensor, NoCollapseDims);
    TensorInfo<unsigned long> srcInfo(state, src, NoCollapseDims);
    TensorInfo<unsigned long> indexInfo(state, index, NoCollapseDims);

    RUN(unsigned long, -1)
  }

  if (oldTensor) {
    THZCudaTensor_copyIgnoringOverlaps(state, oldTensor, tensor);
    THZCudaTensor_free(state, tensor);
    tensor = oldTensor;
  }
}

#undef RUN


template <typename IndexType, int Dims>
__global__ void THZCudaTensor_scatterFillKernel(
    TensorInfo<IndexType> tensor,
    TensorInfo<IndexType> index,
    float value,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          tensor, &tensorOffset);

    IndexType indexValue = (IndexType)index.data[indexOffset] - 1;
    tensorOffset += indexValue * tensor.strides[dim];

    tensor.data[tensorOffset] = value;
  }
}

#define RUN(TYPE, DIMS)                                            \
  THZCudaTensor_scatterFillKernel<TYPE, DIMS>                       \
      <<<grid, block, 0, THCState_getCurrentStream(state)>>>(      \
          tensorInfo, indexInfo, value, dim, (TYPE)totalElements);

void THZCudaTensor_scatterFill(THCState* state, THZCudaTensor *tensor, int dim, THZCudaTensor *index, float value) {
  THAssert(THZCudaTensor_checkGPU(state, 2, tensor, index));

  THArgCheck(dim >= 0 && dim < THZCudaTensor_nDimension(state, tensor), 2,
             "Index dimension is out of bounds");
  THArgCheck(THZCudaTensor_nDimension(state, index) == THZCudaTensor_nDimension(state, tensor), 3,
             "Index tensor must have same dimensions as output tensor");

  for (int d = 0; d < THZCudaTensor_nDimension(state, tensor); d++) {
    if (d != dim) {
      THArgCheck(THZCudaTensor_size(state, tensor, d) == THZCudaTensor_size(state, index, d), 4,
                 "Index tensor must have same size as output tensor apart from the specified dimension");
    }
  }

  if (THZCudaTensor_nDimension(state, tensor) > MAX_CUTORCH_DIMS) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  const long totalElements = THZCudaTensor_nElement(state, index);
  const dim3 block = getApplyBlock();
  dim3 grid;
  if (!getApplyGrid(state, totalElements, grid)) {
    return THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THZCudaTensor* oldTensor = NULL;
  if (THZC_overlappingIndices(state, tensor)) {
    oldTensor = tensor;
    tensor = THZCudaTensor_newContiguous(state, tensor);
  }

  if (THZC_canUse32BitIndexMath(state, tensor) &&
      THZC_canUse32BitIndexMath(state, index)) {
    TensorInfo<unsigned int> tensorInfo(state, tensor, NoCollapseDims);
    TensorInfo<unsigned int> indexInfo(state, index, NoCollapseDims);

    // Specialize for a small number of dimensions.
    switch (indexInfo.dims) {
      case 1:
        RUN(unsigned int, 1);
        break;
      case 2:
        RUN(unsigned int, 2);
        break;
      case 3:
        RUN(unsigned int, 3);
        break;
      default:
        RUN(unsigned int, -1);
        break;
    }
  } else {
    TensorInfo<unsigned long> tensorInfo(state, tensor, NoCollapseDims);
    TensorInfo<unsigned long> indexInfo(state, index, NoCollapseDims);

    RUN(unsigned long, -1);
  }

  if (oldTensor) {
    THZCudaTensor_copyIgnoringOverlaps(state, oldTensor, tensor);
    THZCudaTensor_free(state, tensor);
    tensor = oldTensor;
  }
}

#undef RUN
