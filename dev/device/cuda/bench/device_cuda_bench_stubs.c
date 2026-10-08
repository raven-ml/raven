/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The floors of the CUDA bench: the CUDA calls each driver row makes, from C
   alone, on streams of their own in GPU 0's primary context, which stays
   current on the calling thread. Each round trip writes the next value into
   a word of page-locked host memory and spins until it reads it. CUDA's
   functions are those a device's capability finds. A failing call raises
   Failure with its line and status. Every stub holds the runtime. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#if defined(_WIN32)
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

typedef int CUresult;
typedef uint64_t CUdeviceptr;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUevent;

/* cuda.h's CUstreamBatchMemOpParams, as device_cuda_stubs.c declares it. */
typedef union {
  struct {
    int operation;
    CUdeviceptr address;
    uint64_t value;
    unsigned int flags;
    CUdeviceptr alias;
  } wait;
  uint64_t pad[6];
} memop;

#define CUDA(X)                                                                \
  X(cuDevicePrimaryCtxRetain, (CUcontext *, int))                              \
  X(cuCtxPushCurrent_v2, (CUcontext))                                          \
  X(cuStreamCreate, (CUstream *, unsigned int))                                \
  X(cuEventCreate, (CUevent *, unsigned int))                                  \
  X(cuEventRecord, (CUevent, CUstream))                                        \
  X(cuStreamWaitEvent, (CUstream, CUevent, unsigned int))                      \
  X(cuStreamWriteValue64_v2, (CUstream, CUdeviceptr, uint64_t, unsigned int)) \
  X(cuStreamBatchMemOp_v2, (CUstream, unsigned int, memop *, unsigned int))   \
  X(cuMemHostAlloc, (void **, size_t, unsigned int))                           \
  X(cuMemAlloc_v2, (CUdeviceptr *, size_t))                                    \
  X(cuMemFree_v2, (CUdeviceptr))                                               \
  X(cuMemHostRegister_v2, (void *, size_t, unsigned int))                      \
  X(cuMemHostUnregister, (void *))                                             \
  X(cuMemcpyAsync, (CUdeviceptr, CUdeviceptr, size_t, CUstream))               \
  X(cuLaunchKernel, (void *, unsigned int, unsigned int, unsigned int,         \
                     unsigned int, unsigned int, unsigned int, unsigned int,   \
                     CUstream, void **, void **))

#define DECLARE(name, args) static CUresult(CUDAAPI *p_##name) args;
CUDA(DECLARE)
#undef DECLARE

static void check(CUresult s, int line) {
  char msg[64];
  if (s == 0) return;
  snprintf(msg, sizeof msg, "line %d: CUDA error %d", line, s);
  caml_failwith(msg);
}

#define CHECK(call) check(call, __LINE__)

/* Binds the functions of the table above, in its order. */
value device_cuda_bench_bind(value v_f) {
  int i = 0;
#define BIND(name, args)                                                       \
  p_##name = (CUresult(CUDAAPI *) args)Nativeint_val(Field(v_f, i++));
  CUDA(BIND)
#undef BIND
  return Val_unit;
}

value device_cuda_bench_names(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(r);
  static const char *const names[] = {
#define NAME(name, args) #name,
      CUDA(NAME)
#undef NAME
  };
  int n = (int)(sizeof names / sizeof names[0]);
  r = caml_alloc(n, 0);
  for (int i = 0; i < n; i++) Store_field(r, i, caml_copy_string(names[i]));
  CAMLreturn(r);
}

/* The floor's streams, the event that orders them, and its word. */
static CUstream streams[2];
static CUevent event;
static _Atomic uint64_t *word;
static uint64_t last;

value device_cuda_bench_start(value unit) {
  CUcontext c;
  void *w;
  (void)unit;
  CHECK(p_cuDevicePrimaryCtxRetain(&c, 0));
  CHECK(p_cuCtxPushCurrent_v2(c));
  for (int q = 0; q < 2; q++) CHECK(p_cuStreamCreate(&streams[q], 1));
  CHECK(p_cuEventCreate(&event, 2 /* CU_EVENT_DISABLE_TIMING */));
  CHECK(p_cuMemHostAlloc(&w, 8, 3 /* PORTABLE | DEVICEMAP */));
  word = w;
  atomic_store(word, 0);
  return Val_unit;
}

static void release(int q) {
  CHECK(p_cuStreamWriteValue64_v2(streams[q], (CUdeviceptr)(uintptr_t)word,
                                  ++last, 0));
}

static void spin(void) {
  while (atomic_load_explicit(word, memory_order_acquire) < last) {
  }
}

/* [v_k] releases on COMPUTE, then the wait for the last. */
value device_cuda_bench_release(value v_k) {
  for (long i = 0; i < Long_val(v_k); i++) release(0);
  spin();
  return Val_unit;
}

/* A release on the stream the last one did not use, after it. */
value device_cuda_bench_switch(value unit) {
  int q = (int)(last & 1);
  (void)unit;
  CHECK(p_cuStreamWaitEvent(streams[q], event, 0));
  release(q);
  CHECK(p_cuEventRecord(event, streams[q]));
  spin();
  return Val_unit;
}

/* [v_n] satisfied 64-bit waits on the host word [v_at], in one batch, then
   a release. */
value device_cuda_bench_waits(value v_at, value v_n) {
  memop ops[16];
  int n = Int_val(v_n);
  memset(ops, 0, sizeof ops);
  for (int i = 0; i < n; i++) {
    ops[i].wait.operation = 4; /* CU_STREAM_MEM_OP_WAIT_VALUE_64 */
    ops[i].wait.address = (CUdeviceptr)Long_val(v_at);
    ops[i].wait.value = 1;
  }
  CHECK(p_cuStreamBatchMemOp_v2(streams[0], (unsigned int)n, ops, 0));
  release(0);
  spin();
  return Val_unit;
}

/* [v_k] launches of the kernel [v_f], of two 64-bit parameters, over one
   thread, then a release. */
value device_cuda_bench_launch(value v_f, value v_k) {
  uint64_t a = 0, b = 0;
  void *params[2] = {&a, &b};
  for (long i = 0; i < Long_val(v_k); i++)
    CHECK(p_cuLaunchKernel((void *)Nativeint_val(v_f), 1, 1, 1, 1, 1, 1, 0,
                           streams[0], params, NULL));
  release(0);
  spin();
  return Val_unit;
}

/* [v_n] bytes of GPU memory, or of page-locked host memory if [v_host]. */
value device_cuda_bench_buffer(value v_host, value v_n) {
  size_t n = Long_val(v_n);
  if (Bool_val(v_host)) {
    void *p;
    CHECK(p_cuMemHostAlloc(&p, n, 3));
    return caml_copy_nativeint((intnat)p);
  }
  CUdeviceptr d;
  CHECK(p_cuMemAlloc_v2(&d, n));
  return caml_copy_nativeint((intnat)d);
}

/* A copy of [v_n] bytes, then a release. */
value device_cuda_bench_copy(value v_dst, value v_src, value v_n) {
  CHECK(p_cuMemcpyAsync((CUdeviceptr)Nativeint_val(v_dst),
                        (CUdeviceptr)Nativeint_val(v_src), Long_val(v_n),
                        streams[0]));
  release(0);
  spin();
  return Val_unit;
}

value device_cuda_bench_alloc(value v_n) {
  CUdeviceptr d;
  CHECK(p_cuMemAlloc_v2(&d, Long_val(v_n)));
  CHECK(p_cuMemFree_v2(d));
  return Val_unit;
}

value device_cuda_bench_map_host(value v_p, value v_n) {
  void *p = (void *)Long_val(v_p);
  CHECK(p_cuMemHostRegister_v2(p, Long_val(v_n), 3));
  CHECK(p_cuMemHostUnregister(p));
  return Val_unit;
}
