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
static CUresult(CUDAAPI *memcpy_async)(uint64_t, uint64_t, size_t, void *);
static CUresult(CUDAAPI *memcpy_dtoh)(void *, uint64_t, size_t);
static CUresult(CUDAAPI *memcpy_htod)(uint64_t, const void *, size_t);
static CUresult(CUDAAPI *host_register)(void *, size_t, unsigned int);
static CUresult(CUDAAPI *host_unregister)(void *);
static CUresult(CUDAAPI *mem_get_info)(size_t *, size_t *);
static CUresult(CUDAAPI *set_params)(void *, void *, const void *);
static CUresult(CUDAAPI *graph_launch_fn)(void *, void *);

/* Binds cuLaunchKernel, cuCtxGetCurrent, cuDevicePrimaryCtxRetain,
   cuCtxPushCurrent_v2, cuCtxPopCurrent_v2, cuMemHostGetDevicePointer_v2,
   cuDeviceGetAttribute, cuMemcpyAsync, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2,
   cuMemHostRegister_v2, cuMemHostUnregister, cuMemGetInfo_v2,
   cuGraphExecKernelNodeSetParams_v2 and cuGraphLaunch, in this order. */
value rig_cuda_test_bind(value v_f) {
  launch_kernel = Ptr_val(Field(v_f, 0));
  get_current = Ptr_val(Field(v_f, 1));
  retain = Ptr_val(Field(v_f, 2));
  push = Ptr_val(Field(v_f, 3));
  pop = Ptr_val(Field(v_f, 4));
  device_pointer = Ptr_val(Field(v_f, 5));
  get_attribute = Ptr_val(Field(v_f, 6));
  memcpy_async = Ptr_val(Field(v_f, 7));
  memcpy_dtoh = Ptr_val(Field(v_f, 8));
  memcpy_htod = Ptr_val(Field(v_f, 9));
  host_register = Ptr_val(Field(v_f, 10));
  host_unregister = Ptr_val(Field(v_f, 11));
  mem_get_info = Ptr_val(Field(v_f, 12));
  set_params = Ptr_val(Field(v_f, 13));
  graph_launch_fn = Ptr_val(Field(v_f, 14));
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

value rig_cuda_test_write_gpu(value v_a, value v_s) {
  CUcontext popped;
  push_primary();
  CUresult s = memcpy_htod((uint64_t)Nativeint_val(v_a), String_val(v_s),
                           caml_string_length(v_s));
  pop(&popped);
  if (s != 0) caml_failwith("cuMemcpyHtoD");
  return Val_unit;
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

/* A fill that launches [count] times the kernel [f] over [grid] blocks of
   [block] threads with the two 64-bit parameters [a] and [b], and records
   the context current while it ran. */
struct launch {
  void *f;
  unsigned int grid, block;
  int count;
  uint64_t a, b;
  CUcontext seen;
};

static int launch(void *queue, void *arg, uint64_t v) {
  struct launch *l = arg;
  void *params[2] = {&l->a, &l->b};
  (void)v;
  if (get_current(&l->seen) != 0) l->seen = NULL;
  for (int i = 0; i < l->count; i++) {
    CUresult s = launch_kernel(l->f, l->grid, 1, 1, l->block, 1, 1, 0, queue,
                               params, NULL);
    if (s != 0) return s;
  }
  return 0;
}

value rig_cuda_test_launch(value v_f, value v_grid, value v_block,
                              value v_count, value v_a, value v_b) {
  value v = arg(sizeof(struct launch));
  struct launch *l = Arg_val(v);
  l->f = (void *)Long_val(v_f);
  l->grid = (unsigned int)Long_val(v_grid);
  l->block = (unsigned int)Long_val(v_block);
  l->count = Int_val(v_count);
  l->a = (uint64_t)Long_val(v_a);
  l->b = (uint64_t)Long_val(v_b);
  return v;
}

value rig_cuda_test_launch_byte(value *argv, int argn) {
  (void)argn;
  return rig_cuda_test_launch(argv[0], argv[1], argv[2], argv[3], argv[4],
                                 argv[5]);
}

value rig_cuda_test_launch_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)launch);
}

value rig_cuda_test_seen(value v_arg) {
  return caml_copy_nativeint((intnat)((struct launch *)Arg_val(v_arg))->seen);
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

/* A fill that runs the kernel [spin] for [ns] nanoseconds, then copies [n]
   bytes from [src] to [dst]: a copy that starts late. */
struct delayed {
  void *spin;
  uint64_t flag, ns, dst, src, n;
};

static int delayed(void *queue, void *arg, uint64_t v) {
  struct delayed *d = arg;
  void *params[2] = {&d->flag, &d->ns};
  (void)v;
  CUresult s =
      launch_kernel(d->spin, 1, 1, 1, 1, 1, 1, 0, queue, params, NULL);
  if (s != 0) return s;
  return memcpy_async(d->dst, d->src, d->n, queue);
}

value rig_cuda_test_delayed(value v_spin, value v_flag, value v_ns,
                               value v_dst, value v_src, value v_n) {
  value v = arg(sizeof(struct delayed));
  struct delayed *d = Arg_val(v);
  d->spin = (void *)Long_val(v_spin);
  d->flag = (uint64_t)Long_val(v_flag);
  d->ns = (uint64_t)Long_val(v_ns);
  d->dst = (uint64_t)Long_val(v_dst);
  d->src = (uint64_t)Long_val(v_src);
  d->n = (uint64_t)Long_val(v_n);
  return v;
}

value rig_cuda_test_delayed_byte(value *argv, int argn) {
  (void)argn;
  return rig_cuda_test_delayed(argv[0], argv[1], argv[2], argv[3], argv[4],
                                  argv[5]);
}

value rig_cuda_test_delayed_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)delayed);
}
