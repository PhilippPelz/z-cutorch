#include "THZCApply.cuh"

// Implementation of copyIgnoringOverlaps, defined after pointwiseApply2.
void THZCudaTensor_copyIgnoringOverlaps(THCState* state,
                                       THZCudaTensor* dst,
                                       THZCudaTensor* src) {
  THZCudaTensor_pointwiseApply2(state, dst, src, CopyOp<ccx>(),
                               ReadOnly, // ignore overwrites
                               ReadOnly);
}
