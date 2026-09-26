/* Copyright (c) 2026 The Raven authors. ISC License. */
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct context { int device; } *CUcontext;
typedef struct allocation {
  CUcontext owner;
  max_align_t alignment;
} allocation;
static _Thread_local CUcontext current;

int cuInit(unsigned flags) { assert(flags == 0); return 0; }
int cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }
int cuDeviceComputeCapability(int *major, int *minor, int device) {
  (void)device; *major = 8; *minor = 0; return 0;
}
int cuCtxCreate_v2(CUcontext *context, unsigned flags, int device) {
  assert(flags == 0);
  *context = malloc(sizeof(**context));
  if (*context == NULL) return 2;
  (*context)->device = device;
  current = *context;
  return 0;
}
int cuCtxSetCurrent(CUcontext context) { current = context; return 0; }
int cuCtxSynchronize(void) { assert(current != NULL); return 0; }
int cuCtxDestroy_v2(CUcontext context) { free(context); return 0; }
int cuStreamCreate(void **stream, unsigned flags) {
  assert(flags == 1); *stream = malloc(1); return *stream == NULL ? 2 : 0;
}
int cuStreamDestroy_v2(void *stream) { free(stream); return 0; }
int cuStreamQuery(void *stream) { assert(stream != NULL); return 0; }
int cuMemAlloc_v2(uint64_t *address, size_t size) {
  allocation *block = malloc(sizeof(*block) + size);
  if (block == NULL) return 2;
  assert(current != NULL);
  block->owner = current;
  *address = (uint64_t)(uintptr_t)(block + 1);
  return 0;
}
int cuMemFree_v2(uint64_t address) {
  allocation *block = (allocation *)(uintptr_t)address - 1;
  /* Free must use the allocation's original context after name replacement. */
  assert(block->owner == current);
  free(block);
  return 0;
}
int cuMemHostAlloc(void **address, size_t size, unsigned flags) {
  assert(flags == 1);
  uint64_t allocated = 0;
  int status = cuMemAlloc_v2(&allocated, size);
  *address = (void *)(uintptr_t)allocated;
  return status;
}
int cuMemFreeHost(void *address) { return cuMemFree_v2((uint64_t)(uintptr_t)address); }
int cuGetErrorString(int status, const char **message) {
  (void)status; *message = "mock CUDA allocation failure"; return 0;
}

/* The fixture deliberately executes no GPU work or foreign host imports. */
#define UNEXPECTED(name) int name(void) { abort(); }
UNEXPECTED(cuMemHostRegister_v2)
UNEXPECTED(cuMemHostUnregister)
UNEXPECTED(cuMemcpyAsync)
UNEXPECTED(cuModuleLoadData)
UNEXPECTED(cuModuleGetFunction)
UNEXPECTED(cuModuleUnload)
UNEXPECTED(cuLaunchKernel)
UNEXPECTED(cuLaunchHostFunc)
UNEXPECTED(cuStreamWaitValue64_v2)
UNEXPECTED(cuStreamWriteValue64_v2)
