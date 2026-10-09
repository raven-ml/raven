/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The floors of the CUDA bench: the CUDA calls each driver row makes, from C
   alone, on streams of their own in GPU 0's primary context, which stays
   current on the calling thread. Each round trip writes the next value into
   a word of page-locked host memory and spins until it reads it. CUDA's
   functions are those a device's capability finds. A failing call raises
   Failure with its line and status. One more stub calls the driver's own C
   submit, for the waits rig cannot make. Every stub holds the
   runtime. */

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

#include "rig_cuda.h"

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
typedef void *CUgraph;
typedef void *CUgraphNode;
typedef void *CUgraphExec;

/* cuda.h's CUDA_KERNEL_NODE_PARAMS_v2. */
typedef struct {
  void *func;
  unsigned int grid[3], block[3], shared;
  void **params, **extra;
  void *kern;
  CUcontext context;
} kernel_node;

/* cuda.h's CUstreamBatchMemOpParams, as rig_cuda_stubs.c declares it. */
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
                     CUstream, void **, void **))                              \
  X(cuGraphCreate, (CUgraph *, unsigned int))                                  \
  X(cuGraphAddKernelNode_v2, (CUgraphNode *, CUgraph, const CUgraphNode *,     \
                              size_t, const kernel_node *))                    \
  X(cuGraphInstantiateWithFlags, (CUgraphExec *, CUgraph, unsigned long long)) \
  X(cuGraphExecKernelNodeSetParams_v2, (CUgraphExec, CUgraphNode,              \
                                        const kernel_node *))                  \
  X(cuGraphLaunch, (CUgraphExec, CUstream))

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
value rig_cuda_bench_bind(value v_f) {
  int i = 0;
#define BIND(name, args)                                                       \
  p_##name = (CUresult(CUDAAPI *) args)Nativeint_val(Field(v_f, i++));
  CUDA(BIND)
#undef BIND
  return Val_unit;
}

value rig_cuda_bench_names(value unit) {
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

value rig_cuda_bench_start(value unit) {
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
value rig_cuda_bench_release(value v_k) {
  for (long i = 0; i < Long_val(v_k); i++) release(0);
  spin();
  return Val_unit;
}

/* A release on the stream the last one did not use, after it. */
value rig_cuda_bench_switch(value unit) {
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
value rig_cuda_bench_waits(value v_at, value v_n) {
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

/* The driver's submit of value [v_v] on the device [v_self], with no part
   and [v_n] satisfied waits on the 64-bit word at [v_at], and its commit. */
value rig_cuda_bench_entry_waits(value v_self, value v_v, value v_at,
                                    value v_n) {
  struct rig_wait w[16];
  int n = Int_val(v_n);
  void *self = (void *)Nativeint_val(v_self);
  uint64_t v = (uint64_t)Long_val(v_v);
  for (int i = 0; i < n; i++)
    w[i] = (struct rig_wait){.at = (uint64_t)Long_val(v_at), .value = 1,
                            .kind = RIG_WORD};
  const char *why = NULL;
  int r = rig_cuda_submit(self, v, w, n, NULL, 0, NULL, 0, &why);
  if (r == RIG_OK) r = rig_cuda_commit(self, v, &why);
  if (r == RIG_FAILED) caml_failwith(why);
  return Val_unit;
}

/* [v_k] launches of the kernel [v_f], of two 64-bit parameters, over one
   thread, then a release. */
value rig_cuda_bench_launch(value v_f, value v_k) {
  uint64_t a = 0, b = 0;
  void *params[2] = {&a, &b};
  for (long i = 0; i < Long_val(v_k); i++)
    CHECK(p_cuLaunchKernel((void *)Nativeint_val(v_f), 1, 1, 1, 1, 1, 1, 0,
                           streams[0], params, NULL));
  release(0);
  spin();
  return Val_unit;
}

/* The floor's graph: [nodes] kernels of the function [func], each after the
   one before, with the two 64-bit parameters [args], as the driver makes
   them. */
#define NODES_MAX 64
static CUgraphExec exec;
static CUgraphNode nodes[NODES_MAX];
static int count;
static void *func;
static uint64_t args[2];

/* The parameters of a node: [func] over one thread, its arguments [args]
   passed as a buffer, the form compiled code uses. */
static void node_params(kernel_node *p, void **extra, size_t *size) {
  *size = sizeof args;
  extra[0] = (void *)1; /* CU_LAUNCH_PARAM_BUFFER_POINTER */
  extra[1] = args;
  extra[2] = (void *)2; /* CU_LAUNCH_PARAM_BUFFER_SIZE */
  extra[3] = size;
  extra[4] = NULL; /* CU_LAUNCH_PARAM_END */
  *p = (kernel_node){func, {1, 1, 1}, {1, 1, 1}, 0, NULL, extra, NULL, NULL};
}

/* Makes the floor's graph of [v_n] kernels [v_f]. */
value rig_cuda_bench_graph(value v_f, value v_n) {
  CUgraph g;
  kernel_node p;
  void *extra[5];
  size_t size;
  func = (void *)Nativeint_val(v_f);
  count = Int_val(v_n);
  if (count > NODES_MAX) caml_invalid_argument("rig_cuda_bench_graph");
  node_params(&p, extra, &size);
  CHECK(p_cuGraphCreate(&g, 0));
  for (int i = 0; i < count; i++)
    CHECK(p_cuGraphAddKernelNode_v2(&nodes[i], g, i > 0 ? &nodes[i - 1] : NULL,
                                    i > 0, &p));
  CHECK(p_cuGraphInstantiateWithFlags(&exec, g, 0));
  return Val_unit;
}

/* The floor's graph launched, after updating every node's first argument to
   the run's parity if [v_updated], then a release. */
value rig_cuda_bench_graph_launch(value v_updated) {
  if (Bool_val(v_updated)) {
    kernel_node p;
    void *extra[5];
    size_t size;
    args[0] = last & 1;
    node_params(&p, extra, &size);
    for (int i = 0; i < count; i++)
      CHECK(p_cuGraphExecKernelNodeSetParams_v2(exec, nodes[i], &p));
  }
  CHECK(p_cuGraphLaunch(exec, streams[0]));
  release(0);
  spin();
  return Val_unit;
}

/* [v_n] bytes of GPU memory, or of page-locked host memory if [v_host]. */
value rig_cuda_bench_buffer(value v_host, value v_n) {
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
value rig_cuda_bench_copy(value v_dst, value v_src, value v_n) {
  CHECK(p_cuMemcpyAsync((CUdeviceptr)Nativeint_val(v_dst),
                        (CUdeviceptr)Nativeint_val(v_src), Long_val(v_n),
                        streams[0]));
  release(0);
  spin();
  return Val_unit;
}

value rig_cuda_bench_alloc(value v_n) {
  CUdeviceptr d;
  CHECK(p_cuMemAlloc_v2(&d, Long_val(v_n)));
  CHECK(p_cuMemFree_v2(d));
  return Val_unit;
}

value rig_cuda_bench_map_host(value v_p, value v_n) {
  void *p = (void *)Long_val(v_p);
  CHECK(p_cuMemHostRegister_v2(p, Long_val(v_n), 3));
  CHECK(p_cuMemHostUnregister(p));
  return Val_unit;
}
