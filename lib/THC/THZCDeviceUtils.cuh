#ifndef THZC_DEVICE_UTILS_INC
#define THZC_DEVICE_UTILS_INC

/* The largest consecutive integer representable in float32 (2^24) */
#define FLOAT32_MAX_CONSECUTIVE_INT 16777216.0f

/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T THZCCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

#endif // THZC_DEVICE_UTILS_INC
