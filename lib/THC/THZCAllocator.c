#include "THZCAllocator.h"

static void *THZCudaHostAllocator_alloc(void* ctx, long size) {
  void* ptr;

  if (size < 0) THError("Invalid memory size: %ld", size);

  if (size == 0) return NULL;

  THZCudaCheck(cudaMallocHost(&ptr, size));

  return ptr;
}

static void THZCudaHostAllocator_free(void* ctx, void* ptr) {
  if (!ptr) return;

  THZCudaCheck(cudaFreeHost(ptr));
}

static void *THZCudaHostAllocator_realloc(void* ctx, void* ptr, long size) {
  if (size < 0) THError("Invalid memory size: %ld", size);

  THZCudaHostAllocator_free(ctx, ptr);

  if (size == 0) return NULL;

  THZCudaCheck(cudaMallocHost(&ptr, size));

  return ptr;
}

void THZCAllocator_init(THCState *state) {
  state->cudaHostAllocator = (THAllocator*)malloc(sizeof(THAllocator));
  state->cudaHostAllocator->malloc = &THZCudaHostAllocator_alloc;
  state->cudaHostAllocator->realloc = &THZCudaHostAllocator_realloc;
  state->cudaHostAllocator->free = &THZCudaHostAllocator_free;
}

void THZCAllocator_shutdown(THCState *state) {
  free(state->cudaHostAllocator);
}
