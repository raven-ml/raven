/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The CUDA library, loaded at run time, and the devices it drives. The
   subset of the CUDA driver API used is declared here, so building needs no
   CUDA installation.

   A stub answers CUDA's failure as data: one with a numeric result returns
   it as a non-negative int, or CUDA's status negated; one without returns
   the status. A stub that calls CUDA in a context pushes the device's
   context on the calling thread and pops it before returning: OCaml domains
   run on threads of their own, and other CUDA libraries in the process keep
   the context they had current. The submit and the commit push it only
   where another context is current.

   A stub whose comment says it releases the runtime does so without running
   pending signal handlers, so no OCaml code runs between CUDA's answer and
   its caller; every other stub holds the runtime. */

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
#include <caml/signals.h>

#include "rig_cuda.h"

/* The platform */

/* sleep asks CUDA each POLL_MS milliseconds whether a stream failed, and
   sleeps between the questions: a fault is found within POLL_MS of CUDA's
   report, and the wake latency falls on waits rig has already spun
   on, which are long. */
#define POLL_MS 1

#if defined(_WIN32)
#include <windows.h>
#define CUDAAPI __stdcall
static const char *const library_names[] = {"nvcuda.dll", NULL};
static void *library_open(const char *name) {
  return (void *)LoadLibraryA(name);
}
static void *library_symbol(void *lib, const char *name) {
  return (void *)GetProcAddress((HMODULE)lib, name);
}
static int64_t now_ms(void) { return (int64_t)GetTickCount64(); }
static void poll_pause(void) { Sleep(POLL_MS); }
static long host_page_size(void) {
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return (long)info.dwPageSize;
}
#else
#include <dlfcn.h>
#include <time.h>
#include <unistd.h>
#define CUDAAPI
#if defined(__APPLE__)
static const char *const library_names[] = {"libcuda.dylib", NULL};
#else
static const char *const library_names[] = {"libcuda.so.1", "libcuda.so",
                                            NULL};
#endif
static void *library_open(const char *name) {
  return dlopen(name, RTLD_NOW | RTLD_LOCAL);
}
static void *library_symbol(void *lib, const char *name) {
  return dlsym(lib, name);
}
static int64_t now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}
static void poll_pause(void) {
  struct timespec t = {0, POLL_MS * 1000000L};
  nanosleep(&t, NULL);
}
static long host_page_size(void) { return sysconf(_SC_PAGESIZE); }
#endif

/* The CUDA driver API, from cuda.h */

typedef int CUresult;
typedef int CUdevice;
typedef uint64_t CUdeviceptr;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUevent;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUgraph;
typedef void *CUgraphNode;
typedef void *CUgraphExec;

/* CUDA_KERNEL_NODE_PARAMS_v2. */
typedef struct {
  CUfunction func;
  unsigned int grid[3], block[3], shared;
  void **params, **extra;
  void *kern;
  CUcontext context;
} kernel_node;

/* CUstreamBatchMemOpParams: a union of 48 bytes, of which waits use this
   member. */
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

enum {
  CUDA_SUCCESS = 0,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_READY = 600,
  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2, /* then Y and Z */
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,  /* then Y and Z */
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
  CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
  CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
  CU_STREAM_NON_BLOCKING = 0x1,
  CU_EVENT_DISABLE_TIMING = 0x2,
  CU_MEMHOST_PORTABLE_DEVICEMAP = 0x3,
  CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
  CU_STREAM_WAIT_VALUE_GEQ = 0x0,
  CU_STREAM_WAIT_VALUE_FLUSH = 1 << 30,
};

/* The keys of cuLaunchKernel's [extra] list. */
#define CU_LAUNCH_PARAM_END ((void *)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((void *)0x02)

/* A batch holds fewer than 256 operations. */
#define BATCH 255

#define CUDA(X)                                                                \
  X(cuInit, (unsigned int))                                                    \
  X(cuDriverGetVersion, (int *))                                               \
  X(cuDeviceGetCount, (int *))                                                 \
  X(cuDeviceGet, (CUdevice *, int))                                            \
  X(cuDeviceGetAttribute, (int *, int, CUdevice))                              \
  X(cuDeviceTotalMem_v2, (size_t *, CUdevice))                                 \
  X(cuDeviceCanAccessPeer, (int *, CUdevice, CUdevice))                        \
  X(cuDevicePrimaryCtxRetain, (CUcontext *, CUdevice))                         \
  X(cuDevicePrimaryCtxRelease_v2, (CUdevice))                                  \
  X(cuCtxPushCurrent_v2, (CUcontext))                                          \
  X(cuCtxPopCurrent_v2, (CUcontext *))                                         \
  X(cuCtxGetCurrent, (CUcontext *))                                            \
  X(cuCtxEnablePeerAccess, (CUcontext, unsigned int))                          \
  X(cuStreamCreate, (CUstream *, unsigned int))                                \
  X(cuStreamDestroy_v2, (CUstream))                                            \
  X(cuStreamQuery, (CUstream))                                                 \
  X(cuStreamWaitEvent, (CUstream, CUevent, unsigned int))                      \
  X(cuStreamWriteValue64_v2, (CUstream, CUdeviceptr, uint64_t, unsigned int))  \
  X(cuStreamBatchMemOp_v2, (CUstream, unsigned int, memop *, unsigned int))    \
  X(cuEventCreate, (CUevent *, unsigned int))                                  \
  X(cuEventDestroy_v2, (CUevent))                                              \
  X(cuEventRecord, (CUevent, CUstream))                                        \
  X(cuMemAlloc_v2, (CUdeviceptr *, size_t))                                    \
  X(cuMemFree_v2, (CUdeviceptr))                                               \
  X(cuMemHostAlloc, (void **, size_t, unsigned int))                           \
  X(cuMemFreeHost, (void *))                                                   \
  X(cuMemHostRegister_v2, (void *, size_t, unsigned int))                      \
  X(cuMemHostUnregister, (void *))                                             \
  X(cuMemHostGetDevicePointer_v2, (CUdeviceptr *, void *, unsigned int))       \
  X(cuPointerGetAttribute, (void *, int, CUdeviceptr))                         \
  X(cuMemcpyAsync, (CUdeviceptr, CUdeviceptr, size_t, CUstream))               \
  X(cuModuleLoadData, (CUmodule *, const void *))                              \
  X(cuModuleGetFunction, (CUfunction *, CUmodule, const char *))               \
  X(cuModuleGetFunctionCount, (unsigned int *, CUmodule))                      \
  X(cuModuleEnumerateFunctions, (CUfunction *, unsigned int, CUmodule))        \
  X(cuFuncLoad, (CUfunction))                                                  \
  X(cuFuncGetAttribute, (int *, int, CUfunction))                              \
  X(cuFuncSetAttribute, (CUfunction, int, int))                                \
  X(cuModuleUnload, (CUmodule))                                                \
  X(cuLaunchKernel, (CUfunction, unsigned int, unsigned int, unsigned int,     \
                     unsigned int, unsigned int, unsigned int, unsigned int,   \
                     CUstream, void **, void **))                              \
  X(cuGraphCreate, (CUgraph *, unsigned int))                                  \
  X(cuGraphAddKernelNode_v2, (CUgraphNode *, CUgraph, const CUgraphNode *,     \
                              size_t, const kernel_node *))                    \
  X(cuGraphInstantiateWithFlags, (CUgraphExec *, CUgraph, unsigned long long)) \
  X(cuGraphExecDestroy, (CUgraphExec))                                         \
  X(cuGraphDestroy, (CUgraph))                                                 \
  X(cuGetErrorName, (CUresult, const char **))                                 \
  X(cuGetErrorString, (CUresult, const char **))

#define DECLARE(name, args) static CUresult(CUDAAPI *p_##name) args;
CUDA(DECLARE)
#undef DECLARE

/* The CUDA library, once loaded and initialised. */
static void *library = NULL;

/* Loads the CUDA library and initialises it: [(0, "")]; [(-1, name)] if no
   library loads, [name] the first it tried; [(-2, name)] if the library
   lacks the function [name]; or [(status, "")] if cuInit fails. The caller
   calls it once at a time, until it answers [0]. Releases the runtime:
   cuInit may take long. */
value caml_rig_cuda_load(value unit) {
  CAMLparam1(unit);
  CAMLlocal2(r, s);
  const char *missing = NULL;
  int code = 0;
  void *lib = NULL;
  caml_enter_blocking_section_no_pending();
  for (int i = 0; library_names[i] != NULL && lib == NULL; i++)
    lib = library_open(library_names[i]);
  if (lib == NULL) {
    code = -1;
    missing = library_names[0];
  }
#define RESOLVE(name, args)                                                    \
  if (code == 0) {                                                             \
    p_##name = (CUresult(CUDAAPI *) args)library_symbol(lib, #name);           \
    if (p_##name == NULL) {                                                    \
      code = -2;                                                               \
      missing = #name;                                                         \
    }                                                                          \
  }
  CUDA(RESOLVE)
#undef RESOLVE
  if (code == 0) code = p_cuInit(0);
  if (code == 0) library = lib;
  caml_leave_blocking_section();
  s = caml_copy_string(missing != NULL ? missing : "");
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_int(code));
  Store_field(r, 1, s);
  CAMLreturn(r);
}

/* Errors */

/* "NAME: text" for [status], CUDA's name and description, into [buf]. */
static void describe(CUresult status, char *buf, size_t size) {
  const char *name = NULL, *text = NULL;
  if (p_cuGetErrorName(status, &name) != CUDA_SUCCESS || name == NULL) {
    snprintf(buf, size, "CUDA error %d", status);
    return;
  }
  if (p_cuGetErrorString(status, &text) != CUDA_SUCCESS || text == NULL)
    text = "";
  snprintf(buf, size, "%s: %s", name, text);
}

value caml_rig_cuda_error(value v_status) {
  CAMLparam1(v_status);
  char buf[256];
  describe(Int_val(v_status), buf, sizeof buf);
  CAMLreturn(caml_copy_string(buf));
}

/* A numeric result [x], or CUDA's status [s] negated. */
static value answer(CUresult s, intnat x) {
  return Val_long(s == CUDA_SUCCESS ? x : -(intnat)s);
}

/* The library */

/* The address of the library's function [v_name], or [0]. */
value caml_rig_cuda_symbol(value v_name) {
  void *f = library == NULL ? NULL : library_symbol(library, String_val(v_name));
  return Val_long((intnat)f);
}

value caml_rig_cuda_page_size(value unit) {
  (void)unit;
  return Val_long(host_page_size());
}

value caml_rig_cuda_driver_version(value unit) {
  int version = 0;
  CUresult s = p_cuDriverGetVersion(&version);
  (void)unit;
  return answer(s, version);
}

value caml_rig_cuda_count(value unit) {
  int count = 0;
  CUresult s = p_cuDeviceGetCount(&count);
  (void)unit;
  return answer(s, count);
}

/* The CUdevice of ordinal [v_ordinal]. */
value caml_rig_cuda_device(value v_ordinal) {
  CUdevice device = 0;
  CUresult s = p_cuDeviceGet(&device, Int_val(v_ordinal));
  return answer(s, device);
}

value caml_rig_cuda_attribute(value v_device, value v_attribute) {
  int a = 0;
  CUresult s = p_cuDeviceGetAttribute(&a, Int_val(v_attribute),
                                      Int_val(v_device));
  return answer(s, a);
}

value caml_rig_cuda_total_memory(value v_device) {
  size_t total = 0;
  CUresult s = p_cuDeviceTotalMem_v2(&total, Int_val(v_device));
  return answer(s, (intnat)total);
}

/* Contexts. [push c] makes [c] current on the calling thread above the
   thread's own; [pop status] restores the thread's own, and answers
   [status], or pop's failure if [status] is a success. */

static CUresult push(CUcontext c) { return p_cuCtxPushCurrent_v2(c); }

static CUresult pop(CUresult status) {
  CUcontext c;
  CUresult popped = p_cuCtxPopCurrent_v2(&c);
  return status != CUDA_SUCCESS ? status : popped;
}

/* Makes [c] current on the calling thread where another context is:
   whether it pushed [c], which [restore] pops. */
static CUresult own(CUcontext c, int *pushed) {
  CUcontext current = NULL;
  CUresult s = p_cuCtxGetCurrent(&current);
  *pushed = 0;
  if (s != CUDA_SUCCESS || current == c) return s;
  s = push(c);
  *pushed = s == CUDA_SUCCESS;
  return s;
}

static CUresult restore(int pushed, CUresult status) {
  return pushed ? pop(status) : status;
}

/* Sets [s] to the status of [call], made with [d]'s context current, or to
   the push's if the context cannot be made current, and [call] is not
   made. */
#define IN_CONTEXT(s, d, call)                                                 \
  do {                                                                         \
    (s) = push((d)->context);                                                  \
    if ((s) == CUDA_SUCCESS) (s) = pop(call);                                  \
  } while (0)

/* Runs [stmt] with the runtime released, running no pending action. */
#define RELEASED(stmt)                                                         \
  do {                                                                         \
    caml_enter_blocking_section_no_pending();                                  \
    stmt;                                                                      \
    caml_leave_blocking_section();                                             \
  } while (0)

/* Devices */

/* The values a device encodes past its last commit before it commits on
   its own: a commit, a stream write, costs about a kernel launch. */
#define COMMIT_EVERY 64

/* Whether [v]'s parts hold a copy. A value with a copy commits as it is
   encoded: the copy hides the write's cost, and a commit made later lands
   after the work encoded since, so a wait for one leg of a staged copy
   would wait for the next leg too. */
static int copies(const struct rig_part *p, int n) {
  for (int i = 0; i < n; i++)
    if (p[i].kind == RIG_COPY) return 1;
  return 0;
}

/* A device: its context, its two streams and the events that order them,
   and its timeline word, whose host address CUDA's work also uses
   (cuMemHostAlloc's memory, not write-combined, under unified
   addressing). A value ends on one stream, after its work on both; a value
   that starts on the other stream waits for that end. It is never freed:
   Rig_cuda.edge. */
struct device {
  const struct rig_driver *driver;
  CUdevice device;
  CUcontext context;
  CUstream streams[2]; /* COMPUTE:0, COPY:0 */
  CUevent done[2];     /* the latest part of a stream another one waits for */
  CUevent ended;       /* the end of the last value, for the other stream */
  CUevent waited;      /* the foreign waits of the value being submitted */
  unsigned int groups[3], threads[3]; /* a launch's most along x, y, z */
  _Atomic uint64_t *word;
  uint64_t last;       /* the last value submit received */
  uint64_t committed;  /* the last value written to the word */
  int tail;            /* the stream the last value ended on */
  int flush;           /* the last foreign wait flushes remote writes */
  int failed;
  char failure[256];
};

#define Device_val(v) ((struct device *)Long_val(v))

/* Destroys the streams and events [d] has; CUDA's answers are dropped. */
static void destroy(struct device *d) {
  for (int q = 0; q < 2; q++) {
    if (d->streams[q] != NULL) p_cuStreamDestroy_v2(d->streams[q]);
    if (d->done[q] != NULL) p_cuEventDestroy_v2(d->done[q]);
    d->streams[q] = d->done[q] = NULL;
  }
  if (d->ended != NULL) p_cuEventDestroy_v2(d->ended);
  if (d->waited != NULL) p_cuEventDestroy_v2(d->waited);
  d->ended = d->waited = NULL;
}

/* Makes [d]'s word, streams and events, or none of them. The events record
   no timing, the form CUDA waits on fastest. */
static CUresult start(struct device *d) {
  void *word = NULL;
  CUresult s = p_cuMemHostAlloc(&word, sizeof(uint64_t),
                                CU_MEMHOST_PORTABLE_DEVICEMAP);
  for (int q = 0; q < 2 && s == CUDA_SUCCESS; q++) {
    s = p_cuStreamCreate(&d->streams[q], CU_STREAM_NON_BLOCKING);
    if (s == CUDA_SUCCESS)
      s = p_cuEventCreate(&d->done[q], CU_EVENT_DISABLE_TIMING);
  }
  if (s == CUDA_SUCCESS)
    s = p_cuEventCreate(&d->ended, CU_EVENT_DISABLE_TIMING);
  if (s == CUDA_SUCCESS)
    s = p_cuEventCreate(&d->waited, CU_EVENT_DISABLE_TIMING);
  if (s == CUDA_SUCCESS) {
    d->word = word;
    atomic_store_explicit(d->word, 0, memory_order_relaxed);
    return s;
  }
  destroy(d);
  if (word != NULL) p_cuMemFreeHost(word);
  return s;
}

static const struct rig_driver driver = {rig_cuda_room, rig_cuda_submit,
                                         rig_cuda_commit};

/* Reads the GPU's limits on a launch's groups and threads into [d]. */
static CUresult limits(struct device *d) {
  CUresult s = CUDA_SUCCESS;
  for (int k = 0; k < 3 && s == CUDA_SUCCESS; k++) {
    int groups = 0, threads = 0;
    s = p_cuDeviceGetAttribute(&groups, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X + k,
                               d->device);
    if (s == CUDA_SUCCESS)
      s = p_cuDeviceGetAttribute(
          &threads, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X + k, d->device);
    d->groups[k] = (unsigned int)groups;
    d->threads[k] = (unsigned int)threads;
  }
  return s;
}

/* Opens a device on the primary context of the CUdevice [v_device]: its
   state's address, or CUDA's status negated. A device that does not open
   gives back its retain of the context. Releases the runtime: CUDA may
   create the context. */
value caml_rig_cuda_open(value v_device) {
  struct device *d = calloc(1, sizeof *d);
  int flush = 0;
  if (d == NULL) caml_raise_out_of_memory();
  d->driver = &driver;
  d->device = Int_val(v_device);
  caml_enter_blocking_section_no_pending();
  CUresult s = p_cuDeviceGetAttribute(
      &flush, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, d->device);
  if (s == CUDA_SUCCESS) s = limits(d);
  if (s == CUDA_SUCCESS)
    s = p_cuDevicePrimaryCtxRetain(&d->context, d->device);
  if (s == CUDA_SUCCESS) {
    IN_CONTEXT(s, d, start(d));
    if (s != CUDA_SUCCESS) p_cuDevicePrimaryCtxRelease_v2(d->device);
  }
  if (s != CUDA_SUCCESS) free(d);
  caml_leave_blocking_section();
  if (s != CUDA_SUCCESS) return Val_long(-s);
  d->flush = flush != 0;
  return Val_long((intnat)d);
}

/* Whether [v_self]'s GPU addresses the memory of [v_home]'s GPU: [1], after
   enabling the access; [0] if CUDA says it cannot; or CUDA's status
   negated. Releases the runtime: enabling maps the peer's allocations,
   which may take long (unmeasured: kimchi has one GPU). */
value caml_rig_cuda_peer(value v_self, value v_home) {
  struct device *d = Device_val(v_self), *h = Device_val(v_home);
  int can = 0;
  if (d->device == h->device) return Val_long(1);
  CUresult s;
  caml_enter_blocking_section_no_pending();
  s = p_cuDeviceCanAccessPeer(&can, d->device, h->device);
  if (s == CUDA_SUCCESS && can != 0)
    IN_CONTEXT(s, d, p_cuCtxEnablePeerAccess(h->context, 0));
  caml_leave_blocking_section();
  return answer(s, can != 0);
}

/* Memory */

/* [v_n] bytes of GPU memory, or of page-locked host memory mapped for
   every device if [v_host]: their address, or CUDA's status negated.
   Releases the runtime. */
value caml_rig_cuda_alloc(value v_self, value v_host, value v_n) {
  struct device *d = Device_val(v_self);
  size_t n = Long_val(v_n);
  CUdeviceptr a = 0;
  void *p = NULL;
  CUresult s;
  if (Bool_val(v_host))
    RELEASED(IN_CONTEXT(
        s, d, p_cuMemHostAlloc(&p, n, CU_MEMHOST_PORTABLE_DEVICEMAP)));
  else
    RELEASED(IN_CONTEXT(s, d, p_cuMemAlloc_v2(&a, n)));
  return answer(s, Bool_val(v_host) ? (intnat)p : (intnat)a);
}

/* Frees what caml_rig_cuda_alloc gave: CUDA's status. Releases the
   runtime: cuMemFree may wait for the GPU. */
value caml_rig_cuda_free(value v_self, value v_host, value v_address) {
  struct device *d = Device_val(v_self);
  intnat a = Long_val(v_address);
  CUresult s;
  if (Bool_val(v_host))
    RELEASED(IN_CONTEXT(s, d, p_cuMemFreeHost((void *)a)));
  else
    RELEASED(IN_CONTEXT(s, d, p_cuMemFree_v2((CUdeviceptr)a)));
  return Val_int(s);
}

/* The address by which CUDA's work reaches the page-locked host memory at
   [v_address], or CUDA's status negated: CUDA_ERROR_INVALID_VALUE if the
   memory is not page-locked. */
value caml_rig_cuda_mapped(value v_self, value v_address) {
  CUdeviceptr a = 0;
  CUresult s;
  IN_CONTEXT(s, Device_val(v_self),
             p_cuMemHostGetDevicePointer_v2(&a, (void *)Long_val(v_address),
                                            0));
  return answer(s, (intnat)a);
}

/* The start of the CUDA allocation that holds [v_address], or CUDA's
   status negated: host memory CUDA allocated or page-locked, under unified
   addressing. */
value caml_rig_cuda_allocation(value v_self, value v_address) {
  CUdeviceptr start = 0;
  CUresult s;
  IN_CONTEXT(s, Device_val(v_self),
             p_cuPointerGetAttribute(&start,
                                     CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                     (CUdeviceptr)Long_val(v_address)));
  return answer(s, (intnat)start);
}

/* Page-locks the [v_n] bytes of host memory at [v_address] for every
   device if [v_lock], or unlocks the range that starts there: CUDA's
   status. Releases the runtime: CUDA locks or unlocks every page. */
value caml_rig_cuda_lock(value v_self, value v_lock, value v_address,
                            value v_n) {
  struct device *d = Device_val(v_self);
  void *p = (void *)Long_val(v_address);
  size_t n = Long_val(v_n);
  CUresult s;
  if (Bool_val(v_lock))
    RELEASED(IN_CONTEXT(
        s, d, p_cuMemHostRegister_v2(p, n, CU_MEMHOST_PORTABLE_DEVICEMAP)));
  else
    RELEASED(IN_CONTEXT(s, d, p_cuMemHostUnregister(p)));
  return Val_int(s);
}

/* Images */

/* Loads the module of [image] into [*m] with every function's code: under
   lazy loading, CUDA's default, a function's code is placed at its first
   use, where an out-of-memory would read as no such kernel. Each function
   is allowed as much dynamic shared memory as [d]'s GPU gives a block,
   less its static shared memory: CUDA allows 48 KiB unless told. On a
   failure the module is unloaded and the failure is the answer: an unload
   CUDA refuses then could only repeat a failed context's error, which the
   next call meets. */
static CUresult load_module(struct device *d, CUmodule *m,
                            const char *image) {
  unsigned int n = 0;
  int largest = 0, fixed = 0;
  CUfunction *fs = NULL;
  CUresult s = p_cuModuleLoadData(m, image);
  if (s != CUDA_SUCCESS) return s;
  s = p_cuDeviceGetAttribute(
      &largest, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
      d->device);
  if (s == CUDA_SUCCESS) s = p_cuModuleGetFunctionCount(&n, *m);
  if (s == CUDA_SUCCESS && n > 0) {
    fs = malloc(n * sizeof *fs);
    s = fs == NULL ? CUDA_ERROR_OUT_OF_MEMORY
                   : p_cuModuleEnumerateFunctions(fs, n, *m);
  }
  for (unsigned int i = 0; i < n && s == CUDA_SUCCESS; i++) {
    s = p_cuFuncLoad(fs[i]);
    if (s == CUDA_SUCCESS)
      s = p_cuFuncGetAttribute(&fixed, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                               fs[i]);
    if (s == CUDA_SUCCESS)
      s = p_cuFuncSetAttribute(
          fs[i], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          largest - fixed);
  }
  free(fs);
  if (s != CUDA_SUCCESS) p_cuModuleUnload(*m);
  return s;
}

/* The module of [v_image], every function loaded, or CUDA's status
   negated. PTX is text, which CUDA reads up to a NUL: the image is copied
   with one. Releases the runtime: CUDA may compile it, and waits for the
   GPU's running work. */
value caml_rig_cuda_load_module(value v_self, value v_image) {
  CAMLparam1(v_image);
  struct device *d = Device_val(v_self);
  size_t n = caml_string_length(v_image);
  char *image = malloc(n + 1);
  CUmodule m = NULL;
  CUresult s;
  if (image == NULL) caml_raise_out_of_memory();
  memcpy(image, String_val(v_image), n);
  image[n] = '\0';
  RELEASED(IN_CONTEXT(s, d, load_module(d, &m, image)));
  free(image);
  CAMLreturn(answer(s, (intnat)m));
}

/* What a launch of a function reads: the function, the most threads a
   group of it has, and the most dynamic shared memory a group of it
   takes, which load_module set. */
struct function {
  CUfunction f;
  unsigned int threads, shared;
};

static CUresult find(struct function *fn, CUmodule m, const char *name) {
  int threads = 0, shared = 0;
  CUresult s = p_cuModuleGetFunction(&fn->f, m, name);
  if (s == CUDA_SUCCESS)
    s = p_cuFuncGetAttribute(&threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                             fn->f);
  if (s == CUDA_SUCCESS)
    s = p_cuFuncGetAttribute(
        &shared, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, fn->f);
  fn->threads = (unsigned int)threads;
  fn->shared = (unsigned int)shared;
  return s;
}

/* The struct function of [v_module]'s function [v_name], in memory of its
   own that caml_rig_cuda_forget frees, or CUDA's status negated. */
value caml_rig_cuda_entry(value v_self, value v_module, value v_name) {
  struct function *fn = malloc(sizeof *fn);
  CUresult s;
  if (fn == NULL) caml_raise_out_of_memory();
  IN_CONTEXT(s, Device_val(v_self),
             find(fn, (CUmodule)Long_val(v_module), String_val(v_name)));
  if (s == CUDA_SUCCESS) return Val_long((intnat)fn);
  free(fn);
  return Val_long(-s);
}

/* The CUfunction of the struct function [v_fn]. */
value caml_rig_cuda_code(value v_fn) {
  return Val_long((intnat)((struct function *)Long_val(v_fn))->f);
}

value caml_rig_cuda_forget(value v_fn) {
  free((void *)Long_val(v_fn));
  return Val_unit;
}

/* Unloads [v_module]: CUDA's status. Releases the runtime: CUDA waits for
   the GPU's running work, whichever module it runs. */
value caml_rig_cuda_unload(value v_self, value v_module) {
  struct device *d = Device_val(v_self);
  CUmodule m = (CUmodule)Long_val(v_module);
  CUresult s;
  RELEASED(IN_CONTEXT(s, d, p_cuModuleUnload(m)));
  return Val_int(s);
}

/* Graphs */

/* A graph of [n] kernels: their functions, their sizes (grid x y z, block
   x y z and shared memory bytes, seven per kernel) and their argument
   blocks, [n] blocks back to back whose lengths [lengths] gives. */
struct kernels {
  int n;
  CUfunction *funcs;
  unsigned int *sizes;
  char *args;
  size_t *lengths;
};

/* Makes the graph of [k] as a chain, each kernel after the one before, and
   instantiates it: [*graph], [*exec] and [nodes], or CUDA's status and
   nothing. CUDA copies each node's arguments as it adds it. */
static CUresult make_graph(const struct kernels *k, CUgraph *graph,
                           CUgraphExec *exec, CUgraphNode *nodes) {
  CUresult s = p_cuGraphCreate(graph, 0);
  char *args = k->args;
  for (int i = 0; i < k->n && s == CUDA_SUCCESS; i++) {
    size_t length = k->lengths[i];
    void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, args,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &length, CU_LAUNCH_PARAM_END};
    const unsigned int *z = k->sizes + 7 * i;
    kernel_node p = {k->funcs[i], {z[0], z[1], z[2]}, {z[3], z[4], z[5]}, z[6],
                     NULL, length > 0 ? extra : NULL, NULL, NULL};
    const CUgraphNode *before = i > 0 ? &nodes[i - 1] : NULL;
    s = p_cuGraphAddKernelNode_v2(&nodes[i], *graph, before, i > 0, &p);
    args += length;
  }
  if (s == CUDA_SUCCESS) s = p_cuGraphInstantiateWithFlags(exec, *graph, 0);
  if (s != CUDA_SUCCESS && *graph != NULL) p_cuGraphDestroy(*graph);
  return s;
}

/* The graph of the kernels [v_funcs], [v_sizes] (seven per kernel, each
   checked to fit 32 bits) and [v_args] in [v_self]'s context, as
   [(0, exec, graph, nodes)], or [(status, 0, 0, [||])] with CUDA's status.
   Releases the runtime: CUDA instantiates the graph. */
value caml_rig_cuda_graph(value v_self, value v_funcs, value v_sizes,
                          value v_args) {
  CAMLparam3(v_funcs, v_sizes, v_args);
  CAMLlocal3(r, v_nodes, x);
  struct device *d = Device_val(v_self);
  int n = (int)Wosize_val(v_funcs);
  size_t total = 0;
  for (int i = 0; i < n; i++) total += caml_string_length(Field(v_args, i));
  struct kernels k = {n, calloc(n + 1, sizeof(CUfunction)),
                      calloc(7 * n + 1, sizeof(unsigned int)),
                      malloc(total + 1), calloc(n + 1, sizeof(size_t))};
  CUgraphNode *nodes = calloc(n + 1, sizeof(CUgraphNode));
  if (k.funcs == NULL || k.sizes == NULL || k.args == NULL ||
      k.lengths == NULL || nodes == NULL) {
    free(k.funcs), free(k.sizes), free(k.args), free(k.lengths), free(nodes);
    caml_raise_out_of_memory();
  }
  char *at = k.args;
  for (int i = 0; i < n; i++) {
    value a = Field(v_args, i);
    k.funcs[i] = (CUfunction)Long_val(Field(v_funcs, i));
    k.lengths[i] = caml_string_length(a);
    memcpy(at, String_val(a), k.lengths[i]);
    at += k.lengths[i];
  }
  for (int i = 0; i < 7 * n; i++)
    k.sizes[i] = (unsigned int)Long_val(Field(v_sizes, i));
  CUgraph graph = NULL;
  CUgraphExec exec = NULL;
  CUresult s;
  RELEASED(IN_CONTEXT(s, d, make_graph(&k, &graph, &exec, nodes)));
  free(k.funcs), free(k.sizes), free(k.args), free(k.lengths);
  v_nodes = caml_alloc(s == CUDA_SUCCESS ? n : 0, 0);
  for (int i = 0; s == CUDA_SUCCESS && i < n; i++) {
    x = caml_copy_nativeint((intnat)nodes[i]);
    Store_field(v_nodes, i, x);
  }
  free(nodes);
  r = caml_alloc_tuple(4);
  Store_field(r, 0, Val_int(s));
  x = caml_copy_nativeint(s == CUDA_SUCCESS ? (intnat)exec : 0);
  Store_field(r, 1, x);
  x = caml_copy_nativeint(s == CUDA_SUCCESS ? (intnat)graph : 0);
  Store_field(r, 2, x);
  Store_field(r, 3, v_nodes);
  CAMLreturn(r);
}

/* Destroys the executable [v_exec] and the graph [v_graph] in [v_self]'s
   context. CUDA's answers are dropped: after a fault or a stop the context
   remains, and its error is the only answer. Releases the runtime. */
value caml_rig_cuda_graph_release(value v_self, value v_exec, value v_graph) {
  struct device *d = Device_val(v_self);
  CUgraphExec exec = (CUgraphExec)Nativeint_val(v_exec);
  CUgraph graph = (CUgraph)Nativeint_val(v_graph);
  caml_enter_blocking_section_no_pending();
  if (push(d->context) == CUDA_SUCCESS) {
    p_cuGraphExecDestroy(exec);
    p_cuGraphDestroy(graph);
    pop(CUDA_SUCCESS);
  }
  caml_leave_blocking_section();
  return Val_unit;
}

/* A stream's error, or success while its work runs. */
static CUresult query(CUstream q) {
  CUresult s = p_cuStreamQuery(q);
  return s == CUDA_ERROR_NOT_READY ? CUDA_SUCCESS : s;
}

/* Submissions */

/* One submission's progress: the streams it entered, whether its foreign
   waits are placed, whether both streams run it, and the streams given
   work since their [done] event was last recorded; and its launches'
   blocks and slots' addresses. */
struct submission {
  struct device *d;
  const struct rig_wait *waits;
  int nwaits;
  const uint8_t *args;
  const uint64_t *slots;
  int entered[2];
  int waited;
  int both;
  int unrecorded[2];
  const char *step;  /* the step that failed; NULL: ordering the streams */
  uint64_t bytes;    /* the bytes of the last copy */
};

/* The steps a failure names, beside ordering the streams. */
static const char filling[] = "running a fill", copying[] = "copying",
                  launching[] = "launching a kernel",
                  waiting[] = "waiting on a word", writing[] = "writing the word";

/* [r], the status of a call of [step], recorded as the failing step. */
static CUresult in_step(struct submission *s, const char *step, CUresult r) {
  if (r != CUDA_SUCCESS && s->step == NULL) s->step = step;
  return r;
}

/* Places [n] waits on [q] in batches; where the GPU can, the last flushes
   the remote writes made before the words it waited for. */
static CUresult wait_words(struct device *d, CUstream q,
                           const struct rig_wait *w, int n) {
  memop ops[BATCH];
  for (int i = 0; i < n; i += BATCH) {
    int k = n - i < BATCH ? n - i : BATCH;
    memset(ops, 0, (size_t)k * sizeof *ops);
    for (int j = 0; j < k; j++) {
      ops[j].wait.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
      ops[j].wait.address = w[i + j].at;
      ops[j].wait.value = w[i + j].value;
      ops[j].wait.flags = CU_STREAM_WAIT_VALUE_GEQ;
    }
    if (d->flush && i + k == n)
      ops[k - 1].wait.flags |= CU_STREAM_WAIT_VALUE_FLUSH;
    CUresult s = p_cuStreamBatchMemOp_v2(q, (unsigned int)k, ops, 0);
    if (s != CUDA_SUCCESS) return s;
  }
  return CUDA_SUCCESS;
}

/* Before the first command of the submission on stream [q]: the end of the
   last value, if it ended on the other stream, and the foreign waits,
   placed on the first stream entered and reached by the other through an
   event. */
static CUresult enter(struct submission *s, int q) {
  struct device *d = s->d;
  CUstream stream = d->streams[q];
  CUresult r = CUDA_SUCCESS;
  if (s->entered[q]) return r;
  if (d->tail != q) r = p_cuStreamWaitEvent(stream, d->ended, 0);
  if (r != CUDA_SUCCESS) return r;
  s->entered[q] = 1;
  if (s->nwaits == 0) return r;
  if (s->waited) return p_cuStreamWaitEvent(stream, d->waited, 0);
  s->waited = 1;
  r = in_step(s, waiting, wait_words(d, stream, s->waits, s->nwaits));
  if (r == CUDA_SUCCESS && s->both) r = p_cuEventRecord(d->waited, stream);
  return r;
}

/* Makes stream [q]'s next command wait for the work [s] gave the other
   stream so far, recording the other stream's [done] event only if it was
   given work since its last record. */
static CUresult follow(struct submission *s, int q) {
  struct device *d = s->d;
  int o = 1 - q;
  CUresult e = CUDA_SUCCESS;
  if (s->unrecorded[o]) e = p_cuEventRecord(d->done[o], d->streams[o]);
  if (e != CUDA_SUCCESS) return e;
  s->unrecorded[o] = 0;
  return p_cuStreamWaitEvent(d->streams[q], d->done[o], 0);
}

/* Launches the function of the launch [p] on [stream] over the grid of its
   block, with its parameters copied and each ref's word raised by its
   slot's address: CUDA copies the parameters at the call. */
static CUresult launch(const struct submission *s, CUstream stream,
                       const struct rig_part *p) {
  const struct function *fn = p->launch.launch;
  const struct rig_block *b = (const void *)(s->args + p->launch.block);
  _Alignas(16) uint8_t params[RIG_PARAMS];
  size_t size = p->launch.params;
  void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, params,
                   CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
  rig_params(params, p, s->args, s->slots);
  return p_cuLaunchKernel(fn->f, b->groups[0], b->groups[1], b->groups[2],
                          b->threads[0], b->threads[1], b->threads[2],
                          b->shared, stream, NULL, size > 0 ? extra : NULL);
}

/* Enqueues the work of the part [p] on [stream]. */
static CUresult work(struct submission *s, CUstream stream,
                     const struct rig_part *p, uint64_t v) {
  s->bytes = p->kind == RIG_COPY ? p->copy.bytes : 0;
  switch (p->kind) {
  case RIG_FILL:
    return in_step(s, filling, p->fill.fn(stream, p->fill.arg, v));
  case RIG_LAUNCH:
    return in_step(s, launching, launch(s, stream, p));
  default:
    return in_step(s, copying,
                   p_cuMemcpyAsync(p->copy.dst + p->copy.dst_offset,
                                   p->copy.src + p->copy.src_offset,
                                   p->copy.bytes, stream));
  }
}

/* Enqueues the parts in array order, and ends [v] on the stream of the
   last part (on the last value's stream for none) once both streams' work
   is done. A part that names a part of the other stream waits for all the
   work queued there before it, a superset of the parts it names: the order
   costs one event per stream and no look ahead. */
static CUresult run(struct submission *s, uint64_t v, const struct rig_part *p,
                    int n) {
  struct device *d = s->d;
  int last[2] = {-1, -1};
  for (int i = 0; i < n; i++) last[p[i].queue] = i;
  int r = n > 0 ? p[n - 1].queue : d->tail, o = 1 - r;
  CUresult e = CUDA_SUCCESS;
  s->both = last[0] >= 0 && last[1] >= 0;
  if (last[1 - d->tail] >= 0)
    e = p_cuEventRecord(d->ended, d->streams[d->tail]);
  for (int i = 0; i < n && e == CUDA_SUCCESS; i++) {
    int q = p[i].queue, across = 0;
    e = enter(s, q);
    for (int k = 0; k < p[i].nafter; k++)
      across |= p[p[i].after[k]].queue != q;
    if (e == CUDA_SUCCESS && across) e = follow(s, q);
    if (e == CUDA_SUCCESS) e = work(s, d->streams[q], &p[i], v);
    s->unrecorded[q] = 1;
  }
  if (e == CUDA_SUCCESS) e = enter(s, r);
  if (e == CUDA_SUCCESS && last[o] >= 0) e = follow(s, r);
  if (e == CUDA_SUCCESS) d->tail = r;
  return e;
}

/* Writes [v] into the word after the last value's end, which follows every
   earlier value's. */
static CUresult write_word(struct device *d, uint64_t v) {
  CUresult e = p_cuStreamWriteValue64_v2(
      d->streams[d->tail], (CUdeviceptr)(uintptr_t)d->word, v, 0);
  if (e == CUDA_SUCCESS) d->committed = v;
  return e;
}

/* Records "step: NAME: text" for the failure [e] of [s]. */
static void fail(struct device *d, const struct submission *s, CUresult e) {
  char cause[192];
  describe(e, cause, sizeof cause);
  if (s->step == copying)
    snprintf(d->failure, sizeof d->failure, "copying %llu bytes: %s",
             (unsigned long long)s->bytes, cause);
  else
    snprintf(d->failure, sizeof d->failure, "%s: %s",
             s->step != NULL ? s->step : "ordering the streams", cause);
}

/* After a failure, writes [v] after the work [s] queued, so that the word
   still reaches every value: on COMPUTE:0 after COPY:0's work, or on the one
   stream entered. It writes nothing if the context failed, whose work no
   longer runs, or if CUDA refuses a call that orders the write so. */
static void drain(struct submission *s, uint64_t v) {
  struct device *d = s->d;
  int r = s->entered[1] && !s->entered[0];
  CUresult e = query(d->streams[0]);
  if (s->entered[0] && s->entered[1]) {
    e = p_cuEventRecord(d->done[1], d->streams[1]);
    if (e == CUDA_SUCCESS) e = p_cuStreamWaitEvent(d->streams[0], d->done[1], 0);
  }
  if (e == CUDA_SUCCESS) e = enter(s, r);
  if (e == CUDA_SUCCESS) d->tail = r;
  if (e == CUDA_SUCCESS) write_word(d, v);
}

/* Whether [d]'s GPU and the function run the launch [p] over its block in
   [args]: no size is 0, and none exceeds a limit. */
static int runs(const struct device *d, const struct rig_part *p,
                const uint8_t *args) {
  const struct function *fn = p->launch.launch;
  const struct rig_block *b = (const void *)(args + p->launch.block);
  uint64_t threads = 1;
  for (int k = 0; k < 3; k++) {
    if (b->groups[k] == 0 || b->groups[k] > d->groups[k]) return 0;
    if (b->threads[k] == 0 || b->threads[k] > d->threads[k]) return 0;
    threads *= b->threads[k];
  }
  return threads <= fn->threads && b->shared <= fn->shared;
}

int rig_cuda_room(void *self, const struct rig_part *p, int n,
                  const uint8_t *args) {
  const struct device *d = self;
  for (int i = 0; i < n; i++) {
    if (p[i].queue < 0 || p[i].queue > 1) return RIG_NEVER;
    switch (p[i].kind) {
    case RIG_FILL:
      if (p[i].fill.ring_units != 0 || p[i].fill.segment_bytes != 0)
        return RIG_NEVER;
      break;
    case RIG_COPY:
      break;
    case RIG_LAUNCH:
      if (p[i].queue != 0 || !runs(d, &p[i], args)) return RIG_NEVER;
      break;
    default:
      return RIG_NEVER;
    }
  }
  return RIG_FITS;
}

int rig_cuda_submit(void *self, uint64_t v, const struct rig_wait *waits,
                    int nwaits, const struct rig_part *parts, int nparts,
                    const uint8_t *args, const uint64_t *slots, int nslots,
                    const uint64_t *handles, int nhandles,
                    uint64_t *times, const char **failure) {
  (void)times;
  struct device *d = self;
  struct submission s = {
      .d = d, .waits = waits, .nwaits = nwaits, .args = args, .slots = slots};
  int pushed;
  (void)nslots;
  (void)handles;
  (void)nhandles;
  d->last = v;
  CUresult e = own(d->context, &pushed);
  if (e == CUDA_SUCCESS) {
    if (!d->failed) e = run(&s, v, parts, nparts);
    if (e == CUDA_SUCCESS && !d->failed &&
        (v - d->committed >= COMMIT_EVERY || copies(parts, nparts)))
      e = in_step(&s, writing, write_word(d, v));
    if (d->failed || e != CUDA_SUCCESS) drain(&s, v);
    e = restore(pushed, e);
  }
  if (e != CUDA_SUCCESS && !d->failed) {
    fail(d, &s, e);
    d->failed = 1;
  }
  if (!d->failed) return d->committed == v ? RIG_COMMITTED : RIG_OK;
  *failure = d->failure;
  return RIG_FAILED;
}

int rig_cuda_commit(void *self, uint64_t v, const char **failure) {
  struct device *d = self;
  int pushed;
  if (!d->failed && v > d->committed) {
    CUresult e = own(d->context, &pushed);
    if (e == CUDA_SUCCESS) e = restore(pushed, write_word(d, v));
    if (e != CUDA_SUCCESS) {
      struct submission s = {.d = d, .step = writing};
      fail(d, &s, e);
      d->failed = 1;
    }
  }
  if (!d->failed) return RIG_OK;
  *failure = d->failure;
  return RIG_FAILED;
}


/* Timeline */

value caml_rig_cuda_signaled(value v_self) {
  struct device *d = Device_val(v_self);
  return Val_long(atomic_load_explicit(d->word, memory_order_acquire));
}

/* Gives back the stopped device's word, which holds its last value: the
   streams and events of a stop that found work running go first, as that
   work has ended. Releases the runtime: CUDA may wait for the GPU. */
value caml_rig_cuda_free_word(value v_self) {
  struct device *d = Device_val(v_self);
  caml_enter_blocking_section_no_pending();
  if (push(d->context) == CUDA_SUCCESS) {
    destroy(d);
    p_cuMemFreeHost((void *)d->word);
    d->word = NULL;
    pop(CUDA_SUCCESS);
  }
  caml_leave_blocking_section();
  return Val_unit;
}

value caml_rig_cuda_word(value v_self) {
  return Val_long((intnat)Device_val(v_self)->word);
}

/* The error that ended the context's work, which CUDA answers to every call
   on its streams, or [0]. */
value caml_rig_cuda_failed(value v_self) {
  struct device *d = Device_val(v_self);
  CUresult s;
  IN_CONTEXT(s, d, query(d->streams[0]));
  return Val_int(s);
}

/* Waits until the word differs from [seen], for at most [ms] milliseconds,
   asking each POLL_MS whether a stream met an error: success, or that
   error. */
static CUresult watch(struct device *d, uint64_t seen, int64_t ms) {
  int64_t start = now_ms();
  CUresult fault = CUDA_SUCCESS;
  while (atomic_load_explicit(d->word, memory_order_acquire) == seen) {
    fault = query(d->streams[0]);
    if (fault == CUDA_SUCCESS) fault = query(d->streams[1]);
    if (fault != CUDA_SUCCESS || now_ms() - start >= ms) break;
    poll_pause();
  }
  return fault;
}

/* [watch] unless the word already differs from [v_seen]: [0], or a stream's
   error. Releases the runtime. */
value caml_rig_cuda_sleep(value v_self, value v_seen, value v_ms) {
  struct device *d = Device_val(v_self);
  uint64_t seen = (uint64_t)Long_val(v_seen);
  CUresult s = CUDA_SUCCESS;
  if (atomic_load_explicit(d->word, memory_order_acquire) == seen)
    RELEASED(IN_CONTEXT(s, d, watch(d, seen, Long_val(v_ms))));
  return Val_int(s);
}

/* Stops a device: [true] if its work no longer writes memory, after
   raising the word to the last value by compare-and-set and destroying the
   streams and events; [false] if work still runs, or the context cannot be
   made current to ask. The work no longer writes once the word holds the
   last value (its write is the work's last command), or once both streams
   are idle or one met an error. Work that still runs gets the write of the
   last value, if no commit wrote it, so the word reaches it once the work
   ends. */
value caml_rig_cuda_stop(value v_self) {
  struct device *d = Device_val(v_self);
  uint64_t w = atomic_load_explicit(d->word, memory_order_acquire);
  int stopped = (int64_t)(w - d->last) >= 0, running = 0;
  if (push(d->context) != CUDA_SUCCESS) return Val_false;
  if (!stopped && !d->failed && d->committed != d->last)
    write_word(d, d->last);
  for (int q = 0; q < 2 && !stopped; q++) {
    CUresult s = p_cuStreamQuery(d->streams[q]);
    if (s == CUDA_ERROR_NOT_READY) running = 1;
    else if (s != CUDA_SUCCESS) stopped = 1;
  }
  stopped |= !running;
  if (stopped) {
    rig_raise(d->word, d->last);
    destroy(d);
  }
  pop(CUDA_SUCCESS);
  return Val_bool(stopped);
}
