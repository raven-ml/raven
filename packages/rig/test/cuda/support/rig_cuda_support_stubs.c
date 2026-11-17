/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Fills, as compiled code would write them, CUDA's own view of host
   memory, and the device's C submit. CUDA's functions are those the
   device's capability finds, bound once. A fill's argument is a bigarray's
   C memory. Every stub holds the runtime: none blocks. */

#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig_edge.h"

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
static CUresult(CUDAAPI *set_attribute)(void *, int, int);
static CUresult(CUDAAPI *func_is_loaded)(int *, void *);
static CUresult(CUDAAPI *func_get_module)(void **, void *);
static CUresult(CUDAAPI *function_count)(unsigned int *, void *);
static CUresult(CUDAAPI *enumerate_functions)(void **, unsigned int, void *);
static CUresult(CUDAAPI *synchronize)(void);
static CUresult(CUDAAPI *primary_state)(int, unsigned int *, int *);
static CUresult(CUDAAPI *primary_set_flags)(int, unsigned int);

/* Binds cuLaunchKernel, cuCtxGetCurrent, cuDevicePrimaryCtxRetain,
   cuCtxPushCurrent_v2, cuCtxPopCurrent_v2, cuMemHostGetDevicePointer_v2,
   cuDeviceGetAttribute, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2,
   cuMemHostRegister_v2, cuMemHostUnregister, cuMemGetInfo_v2,
   cuFuncSetAttribute, cuFuncIsLoaded, cuFuncGetModule,
   cuModuleGetFunctionCount, cuModuleEnumerateFunctions, cuCtxSynchronize, cuDevicePrimaryCtxGetState and
   cuDevicePrimaryCtxSetFlags_v2, in this order. */
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
  set_attribute = Ptr_val(Field(v_f, 12));
  func_is_loaded = Ptr_val(Field(v_f, 13));
  func_get_module = Ptr_val(Field(v_f, 14));
  function_count = Ptr_val(Field(v_f, 15));
  enumerate_functions = Ptr_val(Field(v_f, 16));
  synchronize = Ptr_val(Field(v_f, 17));
  primary_state = Ptr_val(Field(v_f, 18));
  primary_set_flags = Ptr_val(Field(v_f, 19));
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

/* cuda.h's CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES. */
#define MAX_DYNAMIC_SHARED 8

/* Sets the most dynamic shared memory a launch of the function [v_f] may
   take to [v_n] bytes. */
value rig_cuda_test_limit_shared(value v_f, value v_n) {
  CUcontext popped;
  push_primary();
  CUresult s = set_attribute((void *)Long_val(v_f), MAX_DYNAMIC_SHARED,
                             Int_val(v_n));
  pop(&popped);
  if (s != 0) caml_failwith("cuFuncSetAttribute");
  return Val_unit;
}

/* CUDA device 0's attribute [v_a]. */
value rig_cuda_test_attribute(value v_a) {
  int a = 0;
  if (get_attribute(&a, Int_val(v_a), 0) != 0) caml_failwith("attribute");
  return Val_int(a);
}

/* The flags of CUDA device 0's primary context. */
value rig_cuda_test_context_flags(value unit) {
  unsigned int flags = 0;
  int active = 0;
  (void)unit;
  if (primary_state(0, &flags, &active) != 0)
    caml_failwith("cuDevicePrimaryCtxGetState");
  return Val_long(flags);
}

/* Sets the flags of CUDA device 0's primary context to [v_flags]. */
value rig_cuda_test_set_context_flags(value v_flags) {
  if (primary_set_flags(0, (unsigned int)Long_val(v_flags)) != 0)
    caml_failwith("cuDevicePrimaryCtxSetFlags");
  return Val_unit;
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

/* Mutable fixture state is borrowed by the GPU submission. Its bigarrays
   cover whole aligned pages. */
#define STATE_ALIGNMENT 65536

CAMLextern value caml_ba_sub(value vb, value vofs, value vlen);

/* The aligned pages are a slice of a larger bigarray the runtime allocates
   and frees, as no C library offers an aligned allocation on every system. */
static value state(size_t n) {
  CAMLparam0();
  CAMLlocal1(v);
  size_t bytes = (n + STATE_ALIGNMENT - 1) & ~(size_t)(STATE_ALIGNMENT - 1);
  v = caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT, 1, NULL,
                         (intnat)(bytes + STATE_ALIGNMENT));
  uintptr_t p = (uintptr_t)Caml_ba_data_val(v);
  uintptr_t at = (p + STATE_ALIGNMENT - 1) & ~(uintptr_t)(STATE_ALIGNMENT - 1);
  memset((void *)at, 0, bytes);
  CAMLreturn(caml_ba_sub(v, Val_long(at - p), Val_long(bytes)));
}

value rig_cuda_test_data(value array) {
  return Val_long((intnat)Caml_ba_data_val(array));
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

/* A fill that records the current context at the address its argument holds. */
struct context {
  CUcontext seen;
};

static int context(void *queue, void *arg, uint64_t v) {
  uint64_t at;
  memcpy(&at, arg, sizeof at);
  struct context *c = (struct context *)(uintptr_t)at;
  (void)queue, (void)v;
  if (get_current(&c->seen) != 0) c->seen = NULL;
  return 0;
}

value rig_cuda_test_context(value unit) {
  CAMLparam1(unit);
  CAMLreturn(state(sizeof(struct context)));
}

value rig_cuda_test_context_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)context);
}

value rig_cuda_test_seen(value v_arg) {
  return caml_copy_nativeint((intnat)((struct context *)Arg_val(v_arg))->seen);
}

/* A fill that runs a kernel, as rig_cuda_test_stall does, on its queue. */
struct spinning {
  uint64_t spin, flag, ns;
};

static int spinning(void *queue, void *arg, uint64_t v) {
  struct spinning s;
  memcpy(&s, arg, sizeof s);
  void *params[2] = {&s.flag, &s.ns};
  (void)v;
  return launch_kernel((void *)(uintptr_t)s.spin, 1, 1, 1, 1, 1, 1, 0, queue,
                       params, NULL);
}

value rig_cuda_test_spinning(value v_spin, value v_flag, value v_ns) {
  value v = arg(sizeof(struct spinning));
  struct spinning s = {(uint64_t)Long_val(v_spin), (uint64_t)Long_val(v_flag),
                       (uint64_t)Long_val(v_ns)};
  memcpy(Arg_val(v), &s, sizeof s);
  return v;
}

value rig_cuda_test_spinning_fill(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)spinning);
}

/* The C entry */

/* What the C submit of the device [v_self] answers for the fills [v_parts],
   each [(queue, fn, params)], as the value [v_v] with no waits: [None], or
   [Some why] for RIG_FAILED. */
value rig_cuda_test_submit(value v_self, value v_v, value v_parts) {
  CAMLparam3(v_self, v_v, v_parts);
  CAMLlocal1(v_why);
  int n = (int)Wosize_val(v_parts);
  size_t bytes = 0;
  for (int i = 0; i < n; i++)
    bytes += rig_block_bytes(caml_string_length(Field(Field(v_parts, i), 2)));
  struct rig_part *p = calloc((size_t)n + 1, sizeof *p);
  void *scratch = calloc((size_t)n + 1, RIG_SCRATCH);
  uint8_t *args = calloc(bytes + 1, 1);
  if (p == NULL || scratch == NULL || args == NULL) {
    free(p);
    free(scratch);
    free(args);
    caml_raise_out_of_memory();
  }
  size_t at = 0;
  for (int i = 0; i < n; i++) {
    value k = Field(v_parts, i), v_params = Field(k, 2);
    size_t params = caml_string_length(v_params);
    p[i].queue = Int_val(Field(k, 0));
    p[i].kind = RIG_FILL;
    p[i].fill.fn = (int (*)(void *, void *, uint64_t))Nativeint_val(Field(k, 1));
    p[i].fill.args.block = (uint32_t)at;
    p[i].fill.args.params = (uint32_t)params;
    memcpy(args + at + sizeof(struct rig_block), String_val(v_params), params);
    at += rig_block_bytes(params);
  }
  void *self = Ptr_val(v_self);
  const char *failure = NULL;
  int r = rig_driver_of(self)->submit(self, (uint64_t)Long_val(v_v), NULL, 0,
                                      p, n, scratch, args, NULL, 0, NULL,
                                      &failure);
  free(p);
  free(scratch);
  free(args);
  if (r != RIG_FAILED) CAMLreturn(Val_none);
  v_why = caml_copy_string(failure != NULL ? failure : "");
  CAMLreturn(caml_alloc_some(v_why));
}
