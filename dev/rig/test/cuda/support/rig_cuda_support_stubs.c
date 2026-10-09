/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fills, as compiled code would write them, and CUDA's own view of host
   memory. CUDA's functions are those the device's capability finds, bound
   once. A fill's argument is a bigarray's C memory. Every stub holds the
   runtime: none blocks. */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#if defined(_WIN32)
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

#define Ptr_val(v) ((void *)Nativeint_val(v))
#define Addr_val(v) ((void *)Long_val(v))

typedef int CUresult;
typedef void *CUcontext;

/* CUDA, as the capability finds it */

static CUresult(CUDAAPI *launch_kernel)(void *, unsigned int, unsigned int,
                                        unsigned int, unsigned int,
                                        unsigned int, unsigned int,
                                        unsigned int, void *, void **,
                                        void **);
static CUresult(CUDAAPI *get_current)(CUcontext *);
static CUresult(CUDAAPI *retain)(CUcontext *, int);
static CUresult(CUDAAPI *push)(CUcontext);
static CUresult(CUDAAPI *pop)(CUcontext *);
static CUresult(CUDAAPI *device_pointer)(uint64_t *, void *, unsigned int);
static CUresult(CUDAAPI *get_attribute)(int *, int, int);
static CUresult(CUDAAPI *memcpy_dtoh)(void *, uint64_t, size_t);
static CUresult(CUDAAPI *memcpy_htod)(uint64_t, const void *, size_t);
static CUresult(CUDAAPI *host_register)(void *, size_t, unsigned int);
static CUresult(CUDAAPI *host_unregister)(void *);
static CUresult(CUDAAPI *mem_get_info)(size_t *, size_t *);
static CUresult(CUDAAPI *set_params)(void *, void *, const void *);
static CUresult(CUDAAPI *graph_launch_fn)(void *, void *);
static CUresult(CUDAAPI *func_is_loaded)(int *, void *);
static CUresult(CUDAAPI *func_get_module)(void **, void *);
static CUresult(CUDAAPI *function_count)(unsigned int *, void *);
static CUresult(CUDAAPI *enumerate_functions)(void **, unsigned int, void *);
static CUresult(CUDAAPI *synchronize)(void);

/* Binds cuLaunchKernel, cuCtxGetCurrent, cuDevicePrimaryCtxRetain,
   cuCtxPushCurrent_v2, cuCtxPopCurrent_v2, cuMemHostGetDevicePointer_v2,
   cuDeviceGetAttribute, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2,
   cuMemHostRegister_v2, cuMemHostUnregister, cuMemGetInfo_v2,
   cuGraphExecKernelNodeSetParams_v2, cuGraphLaunch, cuFuncIsLoaded,
   cuFuncGetModule, cuModuleGetFunctionCount, cuModuleEnumerateFunctions and
   cuCtxSynchronize, in this order. */
value rig_cuda_test_bind(value v_f) {
  launch_kernel = Ptr_val(Field(v_f, 0));
  get_current = Ptr_val(Field(v_f, 1));
  retain = Ptr_val(Field(v_f, 2));
  push = Ptr_val(Field(v_f, 3));
  pop = Ptr_val(Field(v_f, 4));
  device_pointer = Ptr_val(Field(v_f, 5));
  get_attribute = Ptr_val(Field(v_f, 6));
  memcpy_dtoh = Ptr_val(Field(v_f, 7));
  memcpy_htod = Ptr_val(Field(v_f, 8));
  host_register = Ptr_val(Field(v_f, 9));
  host_unregister = Ptr_val(Field(v_f, 10));
  mem_get_info = Ptr_val(Field(v_f, 11));
  set_params = Ptr_val(Field(v_f, 12));
  graph_launch_fn = Ptr_val(Field(v_f, 13));
  func_is_loaded = Ptr_val(Field(v_f, 14));
  func_get_module = Ptr_val(Field(v_f, 15));
  function_count = Ptr_val(Field(v_f, 16));
  enumerate_functions = Ptr_val(Field(v_f, 17));
  synchronize = Ptr_val(Field(v_f, 18));
  return Val_unit;
}

/* The calling thread's current context. */
value rig_cuda_test_current(value unit) {
  CUcontext c = NULL;
  (void)unit;
  if (get_current(&c) != 0) c = NULL;
  return caml_copy_nativeint((intnat)c);
}

/* Makes the primary context of CUDA's device 0 current above the
   thread's own. */
static void push_primary(void) {
  static CUcontext context = NULL;
  if (context == NULL && retain(&context, 0) != 0) caml_failwith("retain");
  if (push(context) != 0) caml_failwith("push");
}

/* Whether CUDA holds the host memory at [v_p] page-locked and mapped. */
value rig_cuda_test_locked(value v_p) {
  CUcontext popped;
  uint64_t d = 0;
  push_primary();
  CUresult s = device_pointer(&d, Addr_val(v_p), 0);
  pop(&popped);
  return Val_bool(s == 0);
}

/* Page-locks the [v_n] bytes of host memory at [v_p] for every device and
   maps them (CU_MEMHOST_PORTABLE | CU_MEMHOST_DEVICEMAP), as another library
   would. */
value rig_cuda_test_register(value v_p, value v_n) {
  CUcontext popped;
  push_primary();
  CUresult s = host_register(Addr_val(v_p), Long_val(v_n), 0x3);
  pop(&popped);
  if (s != 0) caml_failwith("cuMemHostRegister");
  return Val_unit;
}

/* The bytes of GPU 0's memory CUDA reports free. */
value rig_cuda_test_free_memory(value unit) {
  CUcontext popped;
  size_t free = 0, total = 0;
  (void)unit;
  push_primary();
  CUresult s = mem_get_info(&free, &total);
  pop(&popped);
  if (s != 0) caml_failwith("cuMemGetInfo");
  return Val_long((intnat)free);
}

value rig_cuda_test_unregister(value v_p) {
  CUcontext popped;
  push_primary();
  CUresult s = host_unregister(Addr_val(v_p));
  pop(&popped);
  if (s != 0) caml_failwith("cuMemHostUnregister");
  return Val_unit;
}

/* The [v_n] bytes of GPU memory at [v_a], copied by CUDA once every stream
   of the context is done with them. */
value rig_cuda_test_read_gpu(value v_a, value v_n) {
  CAMLparam2(v_a, v_n);
  CAMLlocal1(r);
  CUcontext popped;
  size_t n = Long_val(v_n);
  char *buf = malloc(n);
  if (buf == NULL) caml_raise_out_of_memory();
  push_primary();
  CUresult s = memcpy_dtoh(buf, (uint64_t)Nativeint_val(v_a), n);
  pop(&popped);
  if (s != 0) {
    free(buf);
    caml_failwith("cuMemcpyDtoH");
  }
  r = caml_alloc_initialized_string(n, buf);
  free(buf);
  CAMLreturn(r);
}

/* cuMemcpyHtoD from pageable memory returns once the bytes are staged, and
   CUDA's default stream DMAs them later, which rig's streams, created
   non-blocking, do not wait for: the context's synchronisation lands them. */
value rig_cuda_test_write_gpu(value v_a, value v_s) {
  CUcontext popped;
  push_primary();
  CUresult s = memcpy_htod((uint64_t)Nativeint_val(v_a), String_val(v_s),
                           caml_string_length(v_s));
  if (s == 0) s = synchronize();
  pop(&popped);
  if (s != 0) caml_failwith("cuMemcpyHtoD");
  return Val_unit;
}

/* Runs the kernel [v_spin], with the parameters [v_flag] and [v_ns], on
   CUDA's default stream. */
value rig_cuda_test_stall(value v_spin, value v_flag, value v_ns) {
  CUcontext popped;
  uint64_t flag = (uint64_t)Long_val(v_flag), ns = (uint64_t)Long_val(v_ns);
  void *params[2] = {&flag, &ns};
  push_primary();
  CUresult s = launch_kernel((void *)Long_val(v_spin), 1, 1, 1, 1, 1, 1, 0,
                             NULL, params, NULL);
  pop(&popped);
  if (s != 0) caml_failwith("cuLaunchKernel");
  return Val_unit;
}

/* The count of the functions of the module of the function [v_f], and of
   those whose code CUDA holds loaded (CU_FUNCTION_LOADING_STATE_LOADED). */
value rig_cuda_test_loaded(value v_f) {
  CAMLparam1(v_f);
  CAMLlocal1(r);
  void *m = NULL, **fs;
  unsigned int n = 0, loaded = 0;
  if (func_get_module(&m, Ptr_val(v_f)) != 0 || function_count(&n, m) != 0)
    caml_failwith("cuModuleGetFunctionCount");
  fs = calloc(n + 1, sizeof *fs);
  if (fs == NULL) caml_raise_out_of_memory();
  if (enumerate_functions(fs, n, m) != 0) {
    free(fs);
    caml_failwith("cuModuleEnumerateFunctions");
  }
  for (unsigned int i = 0; i < n; i++) {
    int state = 0;
    if (func_is_loaded(&state, fs[i]) == 0 && state == 1) loaded++;
  }
  free(fs);
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(n));
  Store_field(r, 1, Val_int(loaded));
  CAMLreturn(r);
}

/* CUDA device 0's attribute [v_a]. */
value rig_cuda_test_attribute(value v_a) {
  int a = 0;
  if (get_attribute(&a, Int_val(v_a), 0) != 0) caml_failwith("attribute");
  return Val_int(a);
}

/* Fills */

#define Arg_val(v) Caml_ba_data_val(v)

/* [n] zeroed bytes of C memory as a bigarray, the argument of a fill: the
   suite hands it to rig as a host buffer. */
static value arg(size_t n) {
  value v = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                               (intnat)n);
  memset(Caml_ba_data_val(v), 0, n);
  return v;
}

/* A fill that returns [code]. */
struct failing {
  int code;
};

static int failing(void *queue, void *arg, uint64_t v) {
  (void)queue, (void)v;
  return ((struct failing *)arg)->code;
}

value rig_cuda_test_failing(value v_code) {
  value v = arg(sizeof(struct failing));
  ((struct failing *)Arg_val(v))->code = Int_val(v_code);
  return v;
}

value rig_cuda_test_failing_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)failing);
}

/* A fill that records the context current while it runs. */
struct context {
  CUcontext seen;
};

static int context(void *queue, void *arg, uint64_t v) {
  struct context *c = arg;
  (void)queue, (void)v;
  if (get_current(&c->seen) != 0) c->seen = NULL;
  return 0;
}

value rig_cuda_test_context(value unit) {
  (void)unit;
  return arg(sizeof(struct context));
}

value rig_cuda_test_context_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)context);
}

value rig_cuda_test_seen(value v_arg) {
  return caml_copy_nativeint((intnat)((struct context *)Arg_val(v_arg))->seen);
}

/* A fill that updates nodes of a graph, then launches it. */

#define ARGS_MAX 64

/* CUDA_KERNEL_NODE_PARAMS_v2. */
struct kernel_node {
  void *func;
  unsigned int grid[3], block[3], shared;
  void **params, **extra;
  void *kern;
  CUcontext context;
};

struct update {
  void *node, *func;
  unsigned int sizes[7]; /* grid x y z, block x y z, shared memory bytes */
  size_t length;
  unsigned char args[ARGS_MAX];
};

struct graph_launch {
  void *exec;
  int n;
  struct update u[];
};

static int graph_launch(void *queue, void *arg, uint64_t v) {
  struct graph_launch *g = arg;
  (void)v;
  for (int i = 0; i < g->n; i++) {
    struct update *u = &g->u[i];
    size_t length = u->length;
    void *extra[] = {(void *)1, u->args, (void *)2, &length, (void *)0};
    const unsigned int *z = u->sizes;
    struct kernel_node p = {u->func, {z[0], z[1], z[2]}, {z[3], z[4], z[5]},
                            z[6], NULL, length > 0 ? extra : NULL, NULL, NULL};
    CUresult s = set_params(g->exec, u->node, &p);
    if (s != 0) return s;
  }
  return graph_launch_fn(g->exec, queue);
}

/* The argument of a fill that launches the executable [v_exec] after
   updating, for each [i], the node [v_nodes.(i)] to the kernel [v_funcs.(i)]
   of the sizes [v_sizes] (seven per update) and the argument block
   [v_args.(i)], of at most ARGS_MAX bytes. */
value rig_cuda_test_graph(value v_exec, value v_nodes, value v_funcs,
                          value v_sizes, value v_args) {
  CAMLparam5(v_exec, v_nodes, v_funcs, v_sizes, v_args);
  CAMLlocal1(v);
  int n = (int)Wosize_val(v_nodes);
  v = arg(sizeof(struct graph_launch) + (size_t)n * sizeof(struct update));
  struct graph_launch *g = Arg_val(v);
  g->exec = Ptr_val(v_exec);
  g->n = n;
  for (int i = 0; i < n; i++) {
    struct update *u = &g->u[i];
    value a = Field(v_args, i);
    if (caml_string_length(a) > ARGS_MAX)
      caml_invalid_argument("rig_cuda_test_graph: argument block too long");
    u->node = Ptr_val(Field(v_nodes, i));
    u->func = (void *)Long_val(Field(v_funcs, i));
    for (int k = 0; k < 7; k++)
      u->sizes[k] = (unsigned int)Long_val(Field(v_sizes, 7 * i + k));
    u->length = caml_string_length(a);
    memcpy(u->args, String_val(a), u->length);
  }
  CAMLreturn(v);
}

value rig_cuda_test_graph_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)graph_launch);
}
