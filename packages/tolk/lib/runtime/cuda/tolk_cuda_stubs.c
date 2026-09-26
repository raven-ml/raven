/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include "tolk_dl.h"
#include <stdint.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

/* Hand-declared subset of the CUDA driver API. The library is resolved with
   dlopen at first use so this library builds and loads on machines without
   an NVIDIA driver; device creation fails cleanly there instead. */

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
#define CU_LAUNCH_PARAM_END ((void *)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((void *)0x02)
#define CU_MEMHOSTALLOC_PORTABLE 0x01
static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDeviceComputeCapability)(int *, int *, CUdevice);
static CUresult (*p_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuCtxDestroy)(CUcontext);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuMemAlloc)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemFree)(CUdeviceptr);
static CUresult (*p_cuMemHostAlloc)(void **, size_t, unsigned int);
static CUresult (*p_cuMemFreeHost)(void *);
static CUresult (*p_cuMemHostRegister)(void *, size_t, unsigned int);
static CUresult (*p_cuMemHostUnregister)(void *);
static CUresult (*p_cuMemcpyAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
static CUresult (*p_cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*p_cuModuleUnload)(CUmodule);
static CUresult (*p_cuLaunchHostFunc)(CUstream, void (*)(void *), void *);
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, CUstream,
                                    void **, void **);
static CUresult (*p_cuGetErrorString)(CUresult, const char **);
static CUresult (*p_cuStreamCreate)(CUstream *, unsigned int);
static CUresult (*p_cuStreamDestroy)(CUstream);
static CUresult (*p_cuStreamQuery)(CUstream);
static CUresult (*p_cuStreamWaitValue64)(CUstream, CUdeviceptr, uint64_t, unsigned int);
static CUresult (*p_cuStreamWriteValue64)(CUstream, CUdeviceptr, uint64_t, unsigned int);

static void *cuda_handle = NULL;
static pthread_once_t cuda_once = PTHREAD_ONCE_INIT;
static const char *cuda_load_error = NULL;
static CUresult cuda_init_status;

/* Publish the function table only after initialization. This callback cannot
   raise into OCaml: pthread_once must finish even when a symbol is missing. */
static void load_cuda(void) {
  static const char *names[] = {
#if defined(_WIN32)
      "nvcuda.dll",
#else
      "libcuda.so.1", "libcuda.so",
#endif
      NULL};
  for (int i = 0; cuda_handle == NULL && names[i] != NULL; ++i)
    cuda_handle = tolk_dlopen(names[i]);
  if (cuda_handle == NULL) {
    cuda_load_error = "CUDA driver library not found";
    return;
  }
#define LOAD_CUDA(var, name)                                          \
  do {                                                                \
    var = tolk_dlsym(cuda_handle, name);                                   \
    if (var == NULL) { cuda_load_error = "CUDA driver is missing " name; return; } \
  } while (0)
  LOAD_CUDA(p_cuInit, "cuInit");
  LOAD_CUDA(p_cuDeviceGet, "cuDeviceGet");
  LOAD_CUDA(p_cuDeviceComputeCapability, "cuDeviceComputeCapability");
  LOAD_CUDA(p_cuCtxCreate, "cuCtxCreate_v2");
  LOAD_CUDA(p_cuCtxSetCurrent, "cuCtxSetCurrent");
  LOAD_CUDA(p_cuCtxDestroy, "cuCtxDestroy_v2");
  LOAD_CUDA(p_cuCtxSynchronize, "cuCtxSynchronize");
  LOAD_CUDA(p_cuMemAlloc, "cuMemAlloc_v2");
  LOAD_CUDA(p_cuMemFree, "cuMemFree_v2");
  LOAD_CUDA(p_cuMemHostAlloc, "cuMemHostAlloc");
  LOAD_CUDA(p_cuMemFreeHost, "cuMemFreeHost");
  LOAD_CUDA(p_cuMemHostRegister, "cuMemHostRegister_v2");
  LOAD_CUDA(p_cuMemHostUnregister, "cuMemHostUnregister");
  LOAD_CUDA(p_cuMemcpyAsync, "cuMemcpyAsync");
  LOAD_CUDA(p_cuModuleLoadData, "cuModuleLoadData");
  LOAD_CUDA(p_cuModuleGetFunction, "cuModuleGetFunction");
  LOAD_CUDA(p_cuModuleUnload, "cuModuleUnload");
  LOAD_CUDA(p_cuLaunchKernel, "cuLaunchKernel");
  LOAD_CUDA(p_cuLaunchHostFunc, "cuLaunchHostFunc");
  LOAD_CUDA(p_cuGetErrorString, "cuGetErrorString");
  LOAD_CUDA(p_cuStreamCreate, "cuStreamCreate");
  LOAD_CUDA(p_cuStreamDestroy, "cuStreamDestroy_v2");
  LOAD_CUDA(p_cuStreamQuery, "cuStreamQuery");
  LOAD_CUDA(p_cuStreamWaitValue64, "cuStreamWaitValue64_v2");
  LOAD_CUDA(p_cuStreamWriteValue64, "cuStreamWriteValue64_v2");
#undef LOAD_CUDA
  cuda_init_status = p_cuInit(0);
}

static void ensure_cuda(void) {
  caml_release_runtime_system();
  int status = pthread_once(&cuda_once, load_cuda);
  caml_acquire_runtime_system();
  if (status != 0) caml_failwith("CUDA driver initialization lock failed");
  if (cuda_load_error != NULL) caml_failwith(cuda_load_error);
}

static void cuda_check(CUresult status) {
  if (status != 0) {
    const char *error = NULL;
    char buf[256];
    if (p_cuGetErrorString != NULL) p_cuGetErrorString(status, &error);
    snprintf(buf, sizeof(buf), "CUDA Error %d, %s", status,
             error != NULL ? error : "unknown error");
    caml_failwith(buf);
  }
}

/* The generated host program calls these ordinary C functions. They never
   enter the OCaml runtime, including when submission fails. The first error
   remains sticky until synchronization reports it; polling also recognizes
   asynchronous stream faults so a failed launch cannot hang a replay fence. */
typedef struct {
  CUcontext context;
  pthread_mutex_t lock;
  CUstream streams[2];
  CUresult status;
} tolk_cuda_queue;

static CUresult queue_status(tolk_cuda_queue *q, CUresult status) {
  if (q->status == 0) q->status = status;
  return q->status;
}

static void tolk_cuda_hcq_begin(tolk_cuda_queue *q) {
  pthread_mutex_lock(&q->lock);
  if (q->status == 0) queue_status(q, p_cuCtxSetCurrent(q->context));
  pthread_mutex_unlock(&q->lock);
}

static void tolk_cuda_hcq_launch(tolk_cuda_queue *q, CUfunction function,
    uint64_t gx, uint64_t gy, uint64_t gz, uint64_t lx, uint64_t ly, uint64_t lz,
    void *args, uint64_t bytes) {
  size_t size = (size_t)bytes;
  void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, args,
    CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
  pthread_mutex_lock(&q->lock);
  if (q->status == 0)
    queue_status(q, p_cuLaunchKernel(function, gx, gy, gz, lx, ly, lz, 0,
                                    q->streams[0], NULL, extra));
  pthread_mutex_unlock(&q->lock);
}

static void tolk_cuda_hcq_copy(tolk_cuda_queue *q, uint64_t dst, uint64_t src,
                               uint64_t bytes) {
  pthread_mutex_lock(&q->lock);
  if (q->status == 0)
    queue_status(q, p_cuMemcpyAsync(dst, src, bytes, q->streams[1]));
  pthread_mutex_unlock(&q->lock);
}

static void tolk_cuda_hcq_wait(tolk_cuda_queue *q, uint64_t stream,
                               uint64_t address, uint64_t value) {
  pthread_mutex_lock(&q->lock);
  if (q->status == 0)
    queue_status(q, p_cuStreamWaitValue64(q->streams[stream], address, value, 0));
  pthread_mutex_unlock(&q->lock);
}

static void tolk_cuda_hcq_signal(tolk_cuda_queue *q, uint64_t stream,
                                 uint64_t address, uint64_t value) {
  pthread_mutex_lock(&q->lock);
  if (q->status == 0)
    queue_status(q, p_cuStreamWriteValue64(q->streams[stream], address, value, 0));
  pthread_mutex_unlock(&q->lock);
}

/* CUDA orders this native callback with the stream. No OCaml state is
   touched on the driver thread; pinned slot storage survives synchronization. */
static void tolk_cuda_host_stamp(void *slot) {
#if defined(_WIN32)
  LARGE_INTEGER tick, frequency;
  QueryPerformanceCounter(&tick);
  QueryPerformanceFrequency(&frequency);
  *(uint64_t *)slot = (uint64_t)((double)tick.QuadPart * 1e9 / (double)frequency.QuadPart);
#else
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  *(uint64_t *)slot = (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
#endif
}

static void tolk_cuda_hcq_timestamp(tolk_cuda_queue *q, uint64_t stream, uint64_t address) {
  pthread_mutex_lock(&q->lock);
  if (q->status == 0)
    queue_status(q, p_cuLaunchHostFunc(q->streams[stream], tolk_cuda_host_stamp, (void *)(uintptr_t)address));
  pthread_mutex_unlock(&q->lock);
}

CAMLprim value caml_tolk_cuda_profile_clock(value unit) {
  CAMLparam1(unit);
  uint64_t stamp;
  tolk_cuda_host_stamp(&stamp);
  CAMLreturn(caml_copy_double((double)stamp / 1000.0));
}

static uint64_t tolk_cuda_hcq_poll(tolk_cuda_queue *q, volatile uint64_t *signal) {
  pthread_mutex_lock(&q->lock);
  if (q->status == 0) queue_status(q, p_cuCtxSetCurrent(q->context));
  for (int i = 0; i < 2 && q->status == 0; i++) {
    CUresult status = p_cuStreamQuery(q->streams[i]);
    if (status != 600) queue_status(q, status); /* CUDA_ERROR_NOT_READY */
  }
  uint64_t value = q->status == 0 ? *signal : UINT64_MAX;
  pthread_mutex_unlock(&q->lock);
  return value;
}

CAMLprim value caml_tolk_cuda_hcq_create(value v_context) {
  CAMLparam1(v_context);
  CAMLlocal1(result);
  result = caml_copy_nativeint(0);
  tolk_cuda_queue *q = calloc(1, sizeof(*q));
  if (q == NULL) caml_raise_out_of_memory();
  if (pthread_mutex_init(&q->lock, NULL) != 0) {
    free(q);
    caml_failwith("CUDA queue mutex initialization failed");
  }
  q->context = (CUcontext)Nativeint_val(v_context);
  CUresult status = p_cuStreamCreate(&q->streams[0], 1); /* nonblocking */
  if (status == 0) status = p_cuStreamCreate(&q->streams[1], 1);
  if (status != 0) {
    if (q->streams[0] != NULL) p_cuStreamDestroy(q->streams[0]);
    if (q->streams[1] != NULL) p_cuStreamDestroy(q->streams[1]);
    pthread_mutex_destroy(&q->lock);
    free(q);
    cuda_check(status);
  }
  Nativeint_val(result) = (intnat)q;
  CAMLreturn(result);
}

CAMLprim value caml_tolk_cuda_hcq_destroy(value v_queue) {
  CAMLparam1(v_queue);
  tolk_cuda_queue *q = (tolk_cuda_queue *)Nativeint_val(v_queue);
  caml_release_runtime_system();
  CUresult status = q->status;
  CUresult result = p_cuCtxSetCurrent(q->context);
  if (result == 0) result = p_cuCtxSynchronize();
  if (status == 0) status = result;
  /* Attempt every release even if synchronization or an earlier release
     fails. Context destruction follows on the OCaml side. */
  for (int i = 0; i < 2; i++) {
    result = p_cuStreamDestroy(q->streams[i]);
    if (status == 0) status = result;
  }
  pthread_mutex_destroy(&q->lock);
  free(q);
  caml_acquire_runtime_system();
  cuda_check(status);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_ctx_destroy(value v_context) {
  CAMLparam1(v_context);
  CUcontext context = (CUcontext)Nativeint_val(v_context);
  caml_release_runtime_system();
  CUresult status = p_cuCtxDestroy(context);
  caml_acquire_runtime_system();
  cuda_check(status);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_hcq_synchronize(value v_queue) {
  CAMLparam1(v_queue);
  tolk_cuda_queue *q = (tolk_cuda_queue *)Nativeint_val(v_queue);
  pthread_mutex_lock(&q->lock);
  CUresult pending = q->status;
  pthread_mutex_unlock(&q->lock);
  cuda_check(pending);
  cuda_check(p_cuCtxSetCurrent(q->context));
  caml_release_runtime_system();
  CUresult status = p_cuCtxSynchronize();
  caml_acquire_runtime_system();
  pthread_mutex_lock(&q->lock);
  status = queue_status(q, status);
  pthread_mutex_unlock(&q->lock);
  cuda_check(status);
  CAMLreturn(Val_unit);
}

/* Await only a captured timeline value. Synchronizing the context here
   would also drain later submissions that never accessed the host owner. */
static double hcq_clock_ms(void) {
#if defined(_WIN32)
  LARGE_INTEGER tick, frequency;
  QueryPerformanceCounter(&tick);
  QueryPerformanceFrequency(&frequency);
  return 1000.0 * (double)tick.QuadPart / (double)frequency.QuadPart;
#else
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1e6;
#endif
}

CAMLprim value caml_tolk_cuda_hcq_await(value v_queue, value v_signal,
                                      value v_goal, value v_timeout) {
  CAMLparam4(v_queue, v_signal, v_goal, v_timeout);
  tolk_cuda_queue *q = (tolk_cuda_queue *)Nativeint_val(v_queue);
  volatile uint64_t *signal = (volatile uint64_t *)Nativeint_val(v_signal);
  uint64_t goal = Int64_val(v_goal);
  int timeout_ms = Int_val(v_timeout);
  caml_release_runtime_system();
  double started = hcq_clock_ms();
  uint64_t previous = 0;
  for (;;) {
    uint64_t done = tolk_cuda_hcq_poll(q, signal);
    if (done >= goal) break;
    if (done != previous) { previous = done; started = hcq_clock_ms(); }
    else if (hcq_clock_ms() - started > timeout_ms) {
      pthread_mutex_lock(&q->lock);
      queue_status(q, 702); /* CUDA_ERROR_LAUNCH_TIMEOUT */
      pthread_mutex_unlock(&q->lock);
      break;
    }
#if defined(_WIN32)
    Sleep(0);
#else
    struct timespec pause = {0, 10000};
    nanosleep(&pause, NULL);
#endif
  }
  pthread_mutex_lock(&q->lock);
  CUresult status = q->status;
  pthread_mutex_unlock(&q->lock);
  caml_acquire_runtime_system();
  cuda_check(status);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_hcq_symbol(value v_name) {
  CAMLparam1(v_name);
  const char *name = String_val(v_name);
  void *symbol = NULL;
#define HCQ_SYMBOL(fn) if (strcmp(name, #fn) == 0) symbol = (void *)fn
  HCQ_SYMBOL(tolk_cuda_hcq_begin);
  HCQ_SYMBOL(tolk_cuda_hcq_launch);
  HCQ_SYMBOL(tolk_cuda_hcq_copy);
  HCQ_SYMBOL(tolk_cuda_hcq_wait);
  HCQ_SYMBOL(tolk_cuda_hcq_signal);
  HCQ_SYMBOL(tolk_cuda_hcq_timestamp);
  HCQ_SYMBOL(tolk_cuda_hcq_poll);
#undef HCQ_SYMBOL
  if (symbol == NULL) caml_invalid_argument("Unknown CUDA submission helper");
  CAMLreturn(caml_copy_nativeint((intnat)symbol));
}

CAMLprim value caml_tolk_cuda_init(value unit) {
  CAMLparam1(unit);
  ensure_cuda();
  cuda_check(cuda_init_status);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_device_get(value v_ordinal) {
  CAMLparam1(v_ordinal);
  CUdevice dev = 0;
  cuda_check(p_cuDeviceGet(&dev, Int_val(v_ordinal)));
  CAMLreturn(Val_int(dev));
}

CAMLprim value caml_tolk_cuda_compute_capability(value v_device) {
  CAMLparam1(v_device);
  CAMLlocal1(v_pair);
  int major = 0, minor = 0;
  cuda_check(
      p_cuDeviceComputeCapability(&major, &minor, (CUdevice)Int_val(v_device)));
  v_pair = caml_alloc_tuple(2);
  Store_field(v_pair, 0, Val_int(major));
  Store_field(v_pair, 1, Val_int(minor));
  CAMLreturn(v_pair);
}

CAMLprim value caml_tolk_cuda_ctx_create(value v_device) {
  CAMLparam1(v_device);
  CAMLlocal1(result);
  result = caml_copy_nativeint(0);
  CUcontext ctx = NULL;
  cuda_check(p_cuCtxCreate(&ctx, 0, (CUdevice)Int_val(v_device)));
  Nativeint_val(result) = (intnat)ctx;
  CAMLreturn(result);
}

CAMLprim value caml_tolk_cuda_ctx_set_current(value v_ctx) {
  CAMLparam1(v_ctx);
  cuda_check(p_cuCtxSetCurrent((CUcontext)Nativeint_val(v_ctx)));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_mem_alloc(value v_size) {
  CAMLparam1(v_size);
  CUdeviceptr ptr = 0;
  cuda_check(p_cuMemAlloc(&ptr, (size_t)Long_val(v_size)));
  CAMLreturn(caml_copy_nativeint((intnat)ptr));
}

CAMLprim value caml_tolk_cuda_mem_free(value v_ptr) {
  CAMLparam1(v_ptr);
  cuda_check(p_cuMemFree((CUdeviceptr)Nativeint_val(v_ptr)));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_mem_host_alloc(value v_size) {
  CAMLparam1(v_size);
  void *ptr = NULL;
  cuda_check(
      p_cuMemHostAlloc(&ptr, (size_t)Long_val(v_size), CU_MEMHOSTALLOC_PORTABLE));
  CAMLreturn(caml_copy_nativeint((intnat)ptr));
}

CAMLprim value caml_tolk_cuda_mem_free_host(value v_ptr) {
  CAMLparam1(v_ptr);
  cuda_check(p_cuMemFreeHost((void *)Nativeint_val(v_ptr)));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_mem_host_register(value v_ptr, value v_size) {
  CAMLparam2(v_ptr, v_size);
#if defined(_WIN32)
  SYSTEM_INFO system_info;
  GetSystemInfo(&system_info);
  size_t page_size = system_info.dwPageSize;
#else
  long system_page_size = sysconf(_SC_PAGESIZE);
  if (system_page_size <= 0) caml_failwith("Cannot determine CUDA host page size");
  size_t page_size = (size_t)system_page_size;
#endif
  if ((uintptr_t)Nativeint_val(v_ptr) % page_size != 0)
    CAMLreturn(Val_int(-1));
  CUresult status = p_cuMemHostRegister((void *)Nativeint_val(v_ptr),
                                      (size_t)Long_val(v_size), 0);
  if (status == 1 || status == 801) CAMLreturn(Val_int(-1)); /* unsupported host range */
  if (status != 712) cuda_check(status); /* already registered */
  CAMLreturn(Val_int(status == 0));
}

CAMLprim value caml_tolk_cuda_mem_host_unregister(value v_ptr) {
  CAMLparam1(v_ptr);
  cuda_check(p_cuMemHostUnregister((void *)Nativeint_val(v_ptr)));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_host_read(value v_bytes, value v_host) {
  CAMLparam2(v_bytes, v_host);
  memcpy(Bytes_val(v_bytes), (const void *)Nativeint_val(v_host),
         caml_string_length(v_bytes));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_module_load(value v_lib) {
  CAMLparam1(v_lib);
  CAMLlocal1(v_module);
  v_module = caml_copy_nativeint(0);
  CUmodule module = NULL;
  /* OCaml strings carry a terminating NUL byte, so the PTX image is always
     NUL-terminated as cuModuleLoadData requires. */
  cuda_check(p_cuModuleLoadData(&module, String_val(v_lib)));
  Nativeint_val(v_module) = (intnat)module;
  CAMLreturn(v_module);
}

CAMLprim value caml_tolk_cuda_module_function(value v_module, value v_name) {
  CAMLparam2(v_module, v_name);
  CAMLlocal1(v_function);
  v_function = caml_copy_nativeint(0);
  CUfunction function = NULL;
  cuda_check(p_cuModuleGetFunction(&function,
      (CUmodule)Nativeint_val(v_module), String_val(v_name)));
  Nativeint_val(v_function) = (intnat)function;
  CAMLreturn(v_function);
}

CAMLprim value caml_tolk_cuda_module_unload(value v_module) {
  CAMLparam1(v_module);
  cuda_check(p_cuModuleUnload((CUmodule)Nativeint_val(v_module)));
  CAMLreturn(Val_unit);
}
