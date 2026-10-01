/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The CUDA driver API, loaded at run time. The subset used is declared here,
   so building needs no CUDA installation; the driver is loaded by its
   standard name from the platform's library search path. Every call pushes
   the device's context on the calling thread and pops it before returning,
   error paths included: OCaml domains run on threads of their own, and other
   CUDA libraries in the process keep the context they had current. */

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nx_device.h"

#ifdef _WIN32
#include <windows.h>
#define CUDAAPI __stdcall
static void *library_open(const char *name) { return LoadLibraryA(name); }
static void *library_symbol(void *lib, const char *name) {
  return (void *)GetProcAddress((HMODULE)lib, name);
}
static void yield(void) { SwitchToThread(); }
static int64_t now_ms(void) { return (int64_t)GetTickCount64(); }
static const char *const library_names[] = {"nvcuda.dll", NULL};
#else
#include <dlfcn.h>
#include <sched.h>
#include <time.h>
#define CUDAAPI
static void *library_open(const char *name) {
  return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}
static void *library_symbol(void *lib, const char *name) {
  return dlsym(lib, name);
}
static void yield(void) { sched_yield(); }
static int64_t now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}
#ifdef __APPLE__
static const char *const library_names[] = {"libcuda.dylib", NULL};
#else
static const char *const library_names[] = {"libcuda.so.1", "libcuda.so",
                                            NULL};
#endif
#endif

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void(CUDAAPI *CUhostFn)(void *);

#define CUDA_SUCCESS 0
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NOT_READY 600
#define CU_STREAM_NON_BLOCKING 1
#define CU_MEMHOST_PORTABLE_DEVICEMAP 3
#define CU_STREAM_WAIT_VALUE_GEQ 0

/* The driver */

#define DRIVER(X)                                                              \
  X(cuInit, (unsigned int))                                                    \
  X(cuDriverGetVersion, (int *))                                               \
  X(cuDeviceGetCount, (int *))                                                 \
  X(cuDeviceGet, (CUdevice *, int))                                            \
  X(cuDeviceGetAttribute, (int *, int, CUdevice))                              \
  X(cuDeviceTotalMem_v2, (size_t *, CUdevice))                                 \
  X(cuDevicePrimaryCtxRetain, (CUcontext *, CUdevice))                         \
  X(cuDevicePrimaryCtxRelease_v2, (CUdevice))                                  \
  X(cuCtxPushCurrent_v2, (CUcontext))                                          \
  X(cuCtxPopCurrent_v2, (CUcontext *))                                         \
  X(cuStreamCreate, (CUstream *, unsigned int))                                \
  X(cuStreamDestroy_v2, (CUstream))                                            \
  X(cuStreamQuery, (CUstream))                                                 \
  X(cuStreamWaitValue64_v2, (CUstream, CUdeviceptr, uint64_t, unsigned int))  \
  X(cuStreamWriteValue64_v2, (CUstream, CUdeviceptr, uint64_t, unsigned int)) \
  X(cuMemAlloc_v2, (CUdeviceptr *, size_t))                                    \
  X(cuMemFree_v2, (CUdeviceptr))                                               \
  X(cuMemHostAlloc, (void **, size_t, unsigned int))                           \
  X(cuMemFreeHost, (void *))                                                   \
  X(cuMemHostRegister_v2, (void *, size_t, unsigned int))                      \
  X(cuMemHostUnregister, (void *))                                             \
  X(cuMemHostGetDevicePointer_v2, (CUdeviceptr *, void *, unsigned int))       \
  X(cuMemcpyAsync, (CUdeviceptr, CUdeviceptr, size_t, CUstream))               \
  X(cuLaunchHostFunc, (CUstream, CUhostFn, void *))                            \
  X(cuMemcpyPeerAsync,                                                         \
    (CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t, CUstream))        \
  X(cuPointerGetAttribute, (void *, int, CUdeviceptr))                         \
  X(cuModuleLoadData, (CUmodule *, const void *))                              \
  X(cuModuleGetFunction, (CUfunction *, CUmodule, const char *))               \
  X(cuModuleUnload, (CUmodule))                                                \
  X(cuGetErrorName, (CUresult, const char **))                                 \
  X(cuGetErrorString, (CUresult, const char **))

#define DECLARE(name, args) static CUresult(CUDAAPI *p_##name) args;
DRIVER(DECLARE)
#undef DECLARE

static char load_error[256];

/* The driver, once loaded and initialized. */
static void *driver_library = NULL;

/* Loads the driver and initializes it. The caller calls it once, holding the
   library's lock. [None] on success, [Some msg] otherwise. */
value caml_nx_cuda_load(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(msg);
  caml_release_runtime_system();
  void *lib = NULL;
  for (int i = 0; library_names[i] != NULL && lib == NULL; i++)
    lib = library_open(library_names[i]);
  if (lib == NULL)
    snprintf(load_error, sizeof(load_error),
             "no driver library (%s) on the library search path",
             library_names[0]);
#define RESOLVE(name, args)                                                    \
  if (load_error[0] == '\0') {                                                 \
    p_##name = (CUresult(CUDAAPI *) args)library_symbol(lib, #name);           \
    if (p_##name == NULL)                                                      \
      snprintf(load_error, sizeof(load_error),                                 \
               "the driver has no %s", #name);                           \
  }
  DRIVER(RESOLVE)
#undef RESOLVE
  if (load_error[0] == '\0') {
    CUresult status = p_cuInit(0);
    if (status != CUDA_SUCCESS) {
      const char *name = "unknown error";
      p_cuGetErrorName(status, &name);
      snprintf(load_error, sizeof(load_error), "cuInit failed: %s", name);
    }
  }
  caml_acquire_runtime_system();
  if (load_error[0] == '\0') driver_library = lib;
  if (load_error[0] == '\0') CAMLreturn(Val_none);
  msg = caml_copy_string(load_error);
  CAMLreturn(caml_alloc_some(msg));
}

/* Errors */

static void describe(CUresult status, char *buf, size_t size) {
  const char *name = NULL, *text = NULL;
  p_cuGetErrorName(status, &name);
  p_cuGetErrorString(status, &text);
  if (name == NULL) snprintf(buf, size, "CUDA error %d", status);
  else snprintf(buf, size, "%s: %s", name, text != NULL ? text : "");
}

static void check(CUresult status) {
  if (status != CUDA_SUCCESS) {
    char buf[256];
    describe(status, buf, sizeof(buf));
    caml_failwith(buf);
  }
}

/* [describe status] as an OCaml string. */
value caml_nx_cuda_describe(value v_status) {
  CAMLparam1(v_status);
  char buf[256];
  describe(Int_val(v_status), buf, sizeof(buf));
  CAMLreturn(caml_copy_string(buf));
}

/* Contexts. [push ctx] makes [ctx] current on the calling thread above the
   thread's own; [pop status] restores the thread's own and returns [status]. */

static CUresult push(CUcontext ctx) { return p_cuCtxPushCurrent_v2(ctx); }

static CUresult pop(CUresult status) {
  CUcontext ctx;
  CUresult popped = p_cuCtxPopCurrent_v2(&ctx);
  return status != CUDA_SUCCESS ? status : popped;
}

#define Ptr_val(v) ((void *)Nativeint_val(v))
#define Dptr_val(v) ((CUdeviceptr)(uintptr_t)Nativeint_val(v))

/* [Some address]. */
static value some_address(intnat a) {
  CAMLparam0();
  CAMLlocal1(v);
  v = caml_copy_nativeint(a);
  CAMLreturn(caml_alloc_some(v));
}

/* [Some (host, device)]. */
static value some_addresses(intnat host, intnat device) {
  CAMLparam0();
  CAMLlocal3(pair, h, d);
  h = caml_copy_nativeint(host);
  d = caml_copy_nativeint(device);
  pair = caml_alloc_tuple(2);
  Store_field(pair, 0, h);
  Store_field(pair, 1, d);
  CAMLreturn(caml_alloc_some(pair));
}

/* Devices */

value caml_nx_cuda_count(value unit) {
  CAMLparam1(unit);
  int count = 0;
  check(p_cuDeviceGetCount(&count));
  CAMLreturn(Val_int(count));
}

value caml_nx_cuda_driver_version(value unit) {
  CAMLparam1(unit);
  int version = 0;
  check(p_cuDriverGetVersion(&version));
  CAMLreturn(Val_int(version));
}

/* An attribute the driver does not know is unsupported. */
static int supported(CUdevice device, int which) {
  int v = 0;
  return p_cuDeviceGetAttribute(&v, which, device) == CUDA_SUCCESS && v != 0;
}

/* [(device, major, minor, total bytes, stream memory ops, unified)] */
value caml_nx_cuda_device(value v_ordinal) {
  CAMLparam1(v_ordinal);
  CAMLlocal1(r);
  CUdevice device = 0;
  size_t total = 0;
  int major = 0, minor = 0;
  check(p_cuDeviceGet(&device, Int_val(v_ordinal)));
  check(p_cuDeviceTotalMem_v2(&total, device));
  check(p_cuDeviceGetAttribute(&major, 75, device));
  check(p_cuDeviceGetAttribute(&minor, 76, device));
  r = caml_alloc_tuple(6);
  Store_field(r, 0, Val_int(device));
  Store_field(r, 1, Val_int(major));
  Store_field(r, 2, Val_int(minor));
  Store_field(r, 3, Val_long(total));
  Store_field(r, 4, Val_bool(supported(device, 122)));
  Store_field(r, 5, Val_bool(supported(device, 41)));
  CAMLreturn(r);
}

value caml_nx_cuda_retain(value v_device) {
  CAMLparam1(v_device);
  CUcontext ctx = NULL;
  caml_release_runtime_system();
  CUresult status = p_cuDevicePrimaryCtxRetain(&ctx, Int_val(v_device));
  caml_acquire_runtime_system();
  check(status);
  CAMLreturn(caml_copy_nativeint((intnat)ctx));
}

value caml_nx_cuda_release(value v_device) {
  CAMLparam1(v_device);
  check(p_cuDevicePrimaryCtxRelease_v2(Int_val(v_device)));
  CAMLreturn(Val_unit);
}

value caml_nx_cuda_stream(value v_ctx) {
  CAMLparam1(v_ctx);
  CUstream s = NULL;
  CUresult status = push(Ptr_val(v_ctx));
  if (status == CUDA_SUCCESS)
    status = pop(p_cuStreamCreate(&s, CU_STREAM_NON_BLOCKING));
  check(status);
  CAMLreturn(caml_copy_nativeint((intnat)s));
}

value caml_nx_cuda_stream_destroy(value v_ctx, value v_stream) {
  CAMLparam2(v_ctx, v_stream);
  CUresult status = push(Ptr_val(v_ctx));
  if (status == CUDA_SUCCESS)
    status = pop(p_cuStreamDestroy_v2(Ptr_val(v_stream)));
  check(status);
  CAMLreturn(Val_unit);
}

/* Memory */

value caml_nx_cuda_alloc(value v_ctx, value v_size) {
  CAMLparam2(v_ctx, v_size);
  CUcontext ctx = Ptr_val(v_ctx);
  size_t size = Long_val(v_size);
  CUdeviceptr d = 0;
  caml_release_runtime_system();
  CUresult status = push(ctx);
  if (status == CUDA_SUCCESS) status = pop(p_cuMemAlloc_v2(&d, size));
  caml_acquire_runtime_system();
  if (status == CUDA_ERROR_OUT_OF_MEMORY) CAMLreturn(Val_none);
  check(status);
  CAMLreturn(some_address((intnat)d));
}

value caml_nx_cuda_free(value v_ctx, value v_ptr) {
  CAMLparam2(v_ctx, v_ptr);
  CUcontext ctx = Ptr_val(v_ctx);
  CUdeviceptr d = Dptr_val(v_ptr);
  caml_release_runtime_system();
  CUresult status = push(ctx);
  if (status == CUDA_SUCCESS) status = pop(p_cuMemFree_v2(d));
  caml_acquire_runtime_system();
  check(status);
  CAMLreturn(Val_unit);
}

value caml_nx_cuda_host_alloc(value v_ctx, value v_size) {
  CAMLparam2(v_ctx, v_size);
  CUcontext ctx = Ptr_val(v_ctx);
  size_t size = Long_val(v_size);
  void *p = NULL;
  CUdeviceptr d = 0;
  caml_release_runtime_system();
  CUresult status = push(ctx);
  if (status == CUDA_SUCCESS) {
    CUresult s = p_cuMemHostAlloc(&p, size, CU_MEMHOST_PORTABLE_DEVICEMAP);
    if (s == CUDA_SUCCESS) {
      s = p_cuMemHostGetDevicePointer_v2(&d, p, 0);
      if (s != CUDA_SUCCESS) p_cuMemFreeHost(p);
    }
    status = pop(s);
  }
  caml_acquire_runtime_system();
  if (status == CUDA_ERROR_OUT_OF_MEMORY) CAMLreturn(Val_none);
  check(status);
  CAMLreturn(some_addresses((intnat)p, (intnat)d));
}

value caml_nx_cuda_host_free(value v_ctx, value v_ptr) {
  CAMLparam2(v_ctx, v_ptr);
  CUcontext ctx = Ptr_val(v_ctx);
  void *p = Ptr_val(v_ptr);
  caml_release_runtime_system();
  CUresult status = push(ctx);
  if (status == CUDA_SUCCESS) status = pop(p_cuMemFreeHost(p));
  caml_acquire_runtime_system();
  check(status);
  CAMLreturn(Val_unit);
}

/* The device pointer, in [ctx], of page-locked host memory at [p]. */
value caml_nx_cuda_device_pointer(value v_ctx, value v_ptr) {
  CAMLparam2(v_ctx, v_ptr);
  CUdeviceptr d = 0;
  CUresult status = push(Ptr_val(v_ctx));
  if (status == CUDA_SUCCESS)
    status = pop(p_cuMemHostGetDevicePointer_v2(&d, Ptr_val(v_ptr), 0));
  check(status);
  CAMLreturn(caml_copy_nativeint((intnat)d));
}

/* Page-locks host memory for every context: [0] on success, the driver's
   status otherwise. */
value caml_nx_cuda_register(value v_ctx, value v_ptr, value v_size) {
  CAMLparam3(v_ctx, v_ptr, v_size);
  CUcontext ctx = Ptr_val(v_ctx);
  void *p = Ptr_val(v_ptr);
  size_t size = Long_val(v_size);
  caml_release_runtime_system();
  CUresult status = push(ctx);
  if (status == CUDA_SUCCESS)
    status =
        pop(p_cuMemHostRegister_v2(p, size, CU_MEMHOST_PORTABLE_DEVICEMAP));
  caml_acquire_runtime_system();
  CAMLreturn(Val_int(status));
}

value caml_nx_cuda_unregister(value v_ctx, value v_ptr) {
  CAMLparam2(v_ctx, v_ptr);
  CUcontext ctx = Ptr_val(v_ctx);
  void *p = Ptr_val(v_ptr);
  caml_release_runtime_system();
  CUresult status = push(ctx);
  if (status == CUDA_SUCCESS) status = pop(p_cuMemHostUnregister(p));
  caml_acquire_runtime_system();
  check(status);
  CAMLreturn(Val_unit);
}

/* [(context, range start, range size)] of the memory at a device address, or
   [None] if the driver does not know it. */
value caml_nx_cuda_pointer(value v_ctx, value v_ptr) {
  CAMLparam2(v_ctx, v_ptr);
  CAMLlocal2(r, v);
  CUdeviceptr d = Dptr_val(v_ptr), start = 0;
  CUcontext owner = NULL;
  size_t size = 0;
  CUresult status = push(Ptr_val(v_ctx));
  check(status);
  int known = p_cuPointerGetAttribute(&owner, 1, d) == CUDA_SUCCESS &&
              p_cuPointerGetAttribute(&start, 11, d) == CUDA_SUCCESS &&
              p_cuPointerGetAttribute(&size, 12, d) == CUDA_SUCCESS;
  check(pop(CUDA_SUCCESS));
  if (!known) CAMLreturn(Val_none);
  r = caml_alloc_tuple(3);
  v = caml_copy_nativeint((intnat)owner);
  Store_field(r, 0, v);
  v = caml_copy_nativeint((intnat)start);
  Store_field(r, 1, v);
  Store_field(r, 2, Val_long(size));
  CAMLreturn(caml_alloc_some(r));
}

/* The host address of page-locked memory at a device address, or [0]. */
value caml_nx_cuda_host_pointer(value v_ctx, value v_ptr) {
  CAMLparam2(v_ctx, v_ptr);
  void *host = NULL;
  check(push(Ptr_val(v_ctx)));
  if (p_cuPointerGetAttribute(&host, 4, Dptr_val(v_ptr)) != CUDA_SUCCESS)
    host = NULL;
  check(pop(CUDA_SUCCESS));
  CAMLreturn(caml_copy_nativeint((intnat)host));
}

/* Copies. Each is work on the device's timeline: on the copy stream, a wait
   for the signal to reach [v - 1], the copy, then the signal of [v]. */

value caml_nx_cuda_copy(value v_ctx, value v_stream, value v_signal,
                        value v_dst, value v_src, value v_n, value v_v) {
  CAMLparam5(v_ctx, v_stream, v_signal, v_dst, v_src);
  CAMLxparam2(v_n, v_v);
  CUstream stream = Ptr_val(v_stream);
  CUdeviceptr signal = Dptr_val(v_signal);
  uint64_t v = (uint64_t)Long_val(v_v);
  CUresult status = push(Ptr_val(v_ctx));
  if (status == CUDA_SUCCESS) {
    CUresult s = p_cuStreamWaitValue64_v2(stream, signal, v - 1,
                                          CU_STREAM_WAIT_VALUE_GEQ);
    if (s == CUDA_SUCCESS)
      s = p_cuMemcpyAsync(Dptr_val(v_dst), Dptr_val(v_src), Long_val(v_n),
                          stream);
    if (s == CUDA_SUCCESS) s = p_cuStreamWriteValue64_v2(stream, signal, v, 0);
    status = pop(s);
  }
  check(status);
  CAMLreturn(Val_unit);
}

value caml_nx_cuda_copy_byte(value *argv, int argn) {
  (void)argn;
  return caml_nx_cuda_copy(argv[0], argv[1], argv[2], argv[3], argv[4],
                           argv[5], argv[6]);
}

value caml_nx_cuda_peer(value v_ctx, value v_stream, value v_signal,
                        value v_dst, value v_dst_ctx, value v_src, value v_n,
                        value v_v) {
  CAMLparam5(v_ctx, v_stream, v_signal, v_dst, v_dst_ctx);
  CAMLxparam3(v_src, v_n, v_v);
  CUcontext ctx = Ptr_val(v_ctx);
  CUstream stream = Ptr_val(v_stream);
  CUdeviceptr signal = Dptr_val(v_signal);
  uint64_t v = (uint64_t)Long_val(v_v);
  CUresult status = push(ctx);
  if (status == CUDA_SUCCESS) {
    CUresult s = p_cuStreamWaitValue64_v2(stream, signal, v - 1,
                                          CU_STREAM_WAIT_VALUE_GEQ);
    if (s == CUDA_SUCCESS)
      s = p_cuMemcpyPeerAsync(Dptr_val(v_dst), Ptr_val(v_dst_ctx),
                              Dptr_val(v_src), ctx, Long_val(v_n), stream);
    if (s == CUDA_SUCCESS) s = p_cuStreamWriteValue64_v2(stream, signal, v, 0);
    status = pop(s);
  }
  check(status);
  CAMLreturn(Val_unit);
}

value caml_nx_cuda_peer_byte(value *argv, int argn) {
  (void)argn;
  return caml_nx_cuda_peer(argv[0], argv[1], argv[2], argv[3], argv[4],
                           argv[5], argv[6], argv[7]);
}

/* Stores the host clock into the word at [word], from the driver's thread
   once the stream reached it. */
static void CUDAAPI host_stamp(void *word) {
  atomic_store_explicit((_Atomic uint64_t *)word, nx_device_now_ns(),
                        memory_order_release);
}

value caml_nx_cuda_host_stamp(value unit) {
  (void)unit;
  return caml_copy_nativeint((intnat)(void *)host_stamp);
}

/* The address of the driver's entry point [v_name], if the driver is loaded
   and has one. */
value caml_nx_cuda_driver_function(value v_name) {
  CAMLparam1(v_name);
  CAMLlocal1(address);
  void *f = driver_library == NULL
                ? NULL
                : library_symbol(driver_library, String_val(v_name));
  if (f == NULL) CAMLreturn(Val_none);
  address = caml_copy_nativeint((intnat)f);
  CAMLreturn(caml_alloc_some(address));
}

/* A timestamp as work on the device's timeline: on the copy stream, a wait
   for the signal to reach [v - 1], the host clock stored into the host word
   [v_word] by a host function, then the signal of [v]. */
value caml_nx_cuda_stamp(value v_ctx, value v_stream, value v_signal,
                         value v_word, value v_v) {
  CAMLparam5(v_ctx, v_stream, v_signal, v_word, v_v);
  CUstream stream = Ptr_val(v_stream);
  CUdeviceptr signal = Dptr_val(v_signal);
  uint64_t v = (uint64_t)Long_val(v_v);
  CUresult status = push(Ptr_val(v_ctx));
  if (status == CUDA_SUCCESS) {
    CUresult s = p_cuStreamWaitValue64_v2(stream, signal, v - 1,
                                          CU_STREAM_WAIT_VALUE_GEQ);
    if (s == CUDA_SUCCESS)
      s = p_cuLaunchHostFunc(stream, host_stamp, Ptr_val(v_word));
    if (s == CUDA_SUCCESS) s = p_cuStreamWriteValue64_v2(stream, signal, v, 0);
    status = pop(s);
  }
  check(status);
  CAMLreturn(Val_unit);
}

/* Signals */

static uint64_t load_signal(void *word) {
  return atomic_load_explicit((_Atomic uint64_t *)word, memory_order_acquire);
}

value caml_nx_cuda_signaled(value v_word) {
  return Val_long(load_signal(Ptr_val(v_word)));
}

/* Waits until the signal word reaches [v]. The timeout restarts whenever the
   word moves. About every millisecond it queries the streams: one that reports
   an error faulted, and the wait raises it. */
value caml_nx_cuda_wait(value v_ctx, value v_compute, value v_copy,
                        value v_word, value v_v, value v_timeout) {
  CAMLparam5(v_ctx, v_compute, v_copy, v_word, v_v);
  CAMLxparam1(v_timeout);
  CUcontext ctx = Ptr_val(v_ctx);
  CUstream streams[2] = {Ptr_val(v_compute), Ptr_val(v_copy)};
  void *word = Ptr_val(v_word);
  uint64_t target = (uint64_t)Long_val(v_v);
  int64_t timeout = Long_val(v_timeout);
  int signaled = 0;
  caml_release_runtime_system();
  CUresult status = push(ctx);
  if (status == CUDA_SUCCESS) {
    CUresult fault = CUDA_SUCCESS;
    uint64_t seen = load_signal(word);
    int64_t start = now_ms(), queried = start;
    for (;;) {
      uint64_t now = load_signal(word);
      if (now >= target) {
        signaled = 1;
        break;
      }
      int64_t t = now_ms();
      if (t - queried >= 1) {
        queried = t;
        for (int i = 0; i < 2 && fault == CUDA_SUCCESS; i++) {
          CUresult s = p_cuStreamQuery(streams[i]);
          if (s != CUDA_SUCCESS && s != CUDA_ERROR_NOT_READY) fault = s;
        }
        if (fault != CUDA_SUCCESS) break;
      }
      if (now != seen) {
        seen = now;
        start = t;
      } else if (t - start > timeout)
        break;
      yield();
    }
    status = pop(fault);
  }
  caml_acquire_runtime_system();
  check(status);
  CAMLreturn(Val_bool(signaled));
}

value caml_nx_cuda_wait_byte(value *argv, int argn) {
  (void)argn;
  return caml_nx_cuda_wait(argv[0], argv[1], argv[2], argv[3], argv[4],
                           argv[5]);
}

/* Programs */

value caml_nx_cuda_module(value v_ctx, value v_image) {
  CAMLparam2(v_ctx, v_image);
  CUcontext ctx = Ptr_val(v_ctx);
  size_t n = caml_string_length(v_image);
  char *image = malloc(n + 1);
  if (image == NULL) caml_raise_out_of_memory();
  memcpy(image, String_val(v_image), n);
  image[n] = '\0';
  CUmodule module = NULL;
  caml_release_runtime_system();
  CUresult status = push(ctx);
  if (status == CUDA_SUCCESS) status = pop(p_cuModuleLoadData(&module, image));
  caml_acquire_runtime_system();
  free(image);
  check(status);
  CAMLreturn(caml_copy_nativeint((intnat)module));
}

value caml_nx_cuda_module_unload(value v_ctx, value v_module) {
  CAMLparam2(v_ctx, v_module);
  CUresult status = push(Ptr_val(v_ctx));
  if (status == CUDA_SUCCESS)
    status = pop(p_cuModuleUnload(Ptr_val(v_module)));
  check(status);
  CAMLreturn(Val_unit);
}

value caml_nx_cuda_function(value v_ctx, value v_module, value v_name) {
  CAMLparam3(v_ctx, v_module, v_name);
  CUfunction f = NULL;
  CUresult status = push(Ptr_val(v_ctx));
  if (status == CUDA_SUCCESS)
    status = pop(p_cuModuleGetFunction(&f, Ptr_val(v_module),
                                        String_val(v_name)));
  check(status);
  CAMLreturn(caml_copy_nativeint((intnat)f));
}
