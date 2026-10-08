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
   the context they had current.

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

#include "device_cuda.h"

/* The platform */

/* sleep asks CUDA each POLL_MS milliseconds whether a stream failed, and
   sleeps between the questions: a fault is found within POLL_MS of CUDA's
   report, and the wake latency falls on waits nx.device has already spun
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
  CUDA_ERROR_NOT_READY = 600,
  CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
  CU_STREAM_NON_BLOCKING = 0x1,
  CU_EVENT_DISABLE_TIMING = 0x2,
  CU_MEMHOST_PORTABLE_DEVICEMAP = 0x3,
  CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
  CU_STREAM_WAIT_VALUE_GEQ = 0x0,
  CU_STREAM_WAIT_VALUE_EQ = 0x1,
  CU_STREAM_WAIT_VALUE_FLUSH = 1 << 30,
};

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
  X(cuCtxPushCurrent_v2, (CUcontext))                                          \
  X(cuCtxPopCurrent_v2, (CUcontext *))                                         \
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
  X(cuMemcpyAsync, (CUdeviceptr, CUdeviceptr, size_t, CUstream))               \
  X(cuModuleLoadData, (CUmodule *, const void *))                              \
  X(cuModuleGetFunction, (CUfunction *, CUmodule, const char *))               \
  X(cuModuleUnload, (CUmodule))                                                \
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
value caml_device_cuda_load(value unit) {
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

value caml_device_cuda_error(value v_status) {
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
value caml_device_cuda_symbol(value v_name) {
  void *f = library == NULL ? NULL : library_symbol(library, String_val(v_name));
  return Val_long((intnat)f);
}

value caml_device_cuda_page_size(value unit) {
  (void)unit;
  return Val_long(host_page_size());
}

value caml_device_cuda_driver_version(value unit) {
  int version = 0;
  CUresult s = p_cuDriverGetVersion(&version);
  (void)unit;
  return answer(s, version);
}

value caml_device_cuda_count(value unit) {
  int count = 0;
  CUresult s = p_cuDeviceGetCount(&count);
  (void)unit;
  return answer(s, count);
}

/* The CUdevice of ordinal [v_ordinal]. */
value caml_device_cuda_device(value v_ordinal) {
  CUdevice device = 0;
  CUresult s = p_cuDeviceGet(&device, Int_val(v_ordinal));
  return answer(s, device);
}

value caml_device_cuda_attribute(value v_device, value v_attribute) {
  int a = 0;
  CUresult s = p_cuDeviceGetAttribute(&a, Int_val(v_attribute),
                                      Int_val(v_device));
  return answer(s, a);
}

value caml_device_cuda_total_memory(value v_device) {
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

/* A device: its context, its two streams and the events that order them,
   and its timeline word, whose host address CUDA's work also uses
   (cuMemHostAlloc's memory, not write-combined, under unified
   addressing). It is never freed: Device_cuda.self. */
struct device {
  CUdevice device;
  CUcontext context;
  CUstream streams[2]; /* COMPUTE:0, COPY:0 */
  CUevent done[2];     /* the latest part of a stream another one waits for */
  CUevent released;    /* the release of the last value */
  CUevent waited;      /* the foreign waits of the value being submitted */
  _Atomic uint64_t *word;
  uint64_t last;       /* the last value submit received */
  int released_on;     /* the stream that released it */
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
  if (d->released != NULL) p_cuEventDestroy_v2(d->released);
  if (d->waited != NULL) p_cuEventDestroy_v2(d->waited);
  d->released = d->waited = NULL;
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
    s = p_cuEventCreate(&d->released, CU_EVENT_DISABLE_TIMING);
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

/* Opens a device on the primary context of the CUdevice [v_device]: its
   state's address, or CUDA's status negated. Releases the runtime: CUDA
   may create the context. */
value caml_device_cuda_open(value v_device) {
  struct device *d = calloc(1, sizeof *d);
  int flush = 0;
  if (d == NULL) caml_raise_out_of_memory();
  d->device = Int_val(v_device);
  caml_enter_blocking_section_no_pending();
  CUresult s = p_cuDeviceGetAttribute(
      &flush, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, d->device);
  if (s == CUDA_SUCCESS)
    s = p_cuDevicePrimaryCtxRetain(&d->context, d->device);
  if (s == CUDA_SUCCESS) IN_CONTEXT(s, d, start(d));
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
value caml_device_cuda_peer(value v_self, value v_home) {
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
value caml_device_cuda_alloc(value v_self, value v_host, value v_n) {
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

/* Frees what caml_device_cuda_alloc gave: CUDA's status. Releases the
   runtime: cuMemFree may wait for the GPU. */
value caml_device_cuda_free(value v_self, value v_host, value v_address) {
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
value caml_device_cuda_mapped(value v_self, value v_address) {
  CUdeviceptr a = 0;
  CUresult s;
  IN_CONTEXT(s, Device_val(v_self),
             p_cuMemHostGetDevicePointer_v2(&a, (void *)Long_val(v_address),
                                            0));
  return answer(s, (intnat)a);
}

/* Page-locks the [v_n] bytes of host memory at [v_address] for every
   device if [v_lock], or unlocks the range that starts there: CUDA's
   status. Releases the runtime: CUDA locks or unlocks every page. */
value caml_device_cuda_lock(value v_self, value v_lock, value v_address,
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

/* The module of [v_image], or CUDA's status negated. PTX is text, which
   CUDA reads up to a NUL: the image is copied with one. Releases the
   runtime: CUDA may compile it, and waits for the GPU's running work. */
value caml_device_cuda_load_module(value v_self, value v_image) {
  struct device *d = Device_val(v_self);
  size_t n = caml_string_length(v_image);
  char *image = malloc(n + 1);
  CUmodule m = NULL;
  CUresult s;
  if (image == NULL) caml_raise_out_of_memory();
  memcpy(image, String_val(v_image), n);
  image[n] = '\0';
  RELEASED(IN_CONTEXT(s, d, p_cuModuleLoadData(&m, image)));
  free(image);
  return answer(s, (intnat)m);
}

value caml_device_cuda_function(value v_self, value v_module, value v_name) {
  CUfunction f = NULL;
  CUresult s;
  IN_CONTEXT(s, Device_val(v_self),
             p_cuModuleGetFunction(&f, (CUmodule)Long_val(v_module),
                                   String_val(v_name)));
  return answer(s, (intnat)f);
}

/* Unloads [v_module]: CUDA's status. Releases the runtime: CUDA waits for
   the GPU's running work, whichever module it runs. */
value caml_device_cuda_unload(value v_self, value v_module) {
  struct device *d = Device_val(v_self);
  CUmodule m = (CUmodule)Long_val(v_module);
  CUresult s;
  RELEASED(IN_CONTEXT(s, d, p_cuModuleUnload(m)));
  return Val_int(s);
}

/* A stream's error, or success while its work runs. */
static CUresult query(CUstream q) {
  CUresult s = p_cuStreamQuery(q);
  return s == CUDA_ERROR_NOT_READY ? CUDA_SUCCESS : s;
}

/* Submissions */

/* One submission's progress: the streams it entered, whether its foreign
   waits are placed, and whether both streams run it. */
struct submission {
  struct device *d;
  const struct nx_wait *waits;
  int nwaits;
  int entered[2];
  int waited;
  int both;
  const char *step;  /* the step that failed; NULL: ordering the streams */
  uint64_t bytes;    /* the bytes of the last copy */
};

/* The steps a failure names, beside ordering the streams. */
static const char filling[] = "running a fill", copying[] = "copying",
                  waiting[] = "waiting on a word", writing[] = "writing the word";

/* [r], the status of a call of [step], recorded as the failing step. */
static CUresult in_step(struct submission *s, const char *step, CUresult r) {
  if (r != CUDA_SUCCESS && s->step == NULL) s->step = step;
  return r;
}

/* Places [n] waits on [q] in batches; where the GPU can, the last flushes
   the remote writes made before the words it waited for. */
static CUresult wait_words(struct device *d, CUstream q,
                           const struct nx_wait *w, int n) {
  memop ops[BATCH];
  for (int i = 0; i < n; i += BATCH) {
    int k = n - i < BATCH ? n - i : BATCH;
    memset(ops, 0, (size_t)k * sizeof *ops);
    for (int j = 0; j < k; j++) {
      ops[j].wait.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
      ops[j].wait.address = w[i + j].at;
      ops[j].wait.value = w[i + j].value;
      ops[j].wait.flags = w[i + j].kind == NX_EQUAL ? CU_STREAM_WAIT_VALUE_EQ
                                                    : CU_STREAM_WAIT_VALUE_GEQ;
    }
    if (d->flush && i + k == n)
      ops[k - 1].wait.flags |= CU_STREAM_WAIT_VALUE_FLUSH;
    CUresult s = p_cuStreamBatchMemOp_v2(q, (unsigned int)k, ops, 0);
    if (s != CUDA_SUCCESS) return s;
  }
  return CUDA_SUCCESS;
}

/* Before the first command of the submission on stream [q]: the release of
   the last value, if another stream made it, and the foreign waits, placed
   on the first stream entered and reached by the other through an event. */
static CUresult enter(struct submission *s, int q) {
  struct device *d = s->d;
  CUstream stream = d->streams[q];
  CUresult r = CUDA_SUCCESS;
  if (s->entered[q]) return r;
  if (d->released_on != q) r = p_cuStreamWaitEvent(stream, d->released, 0);
  if (r != CUDA_SUCCESS) return r;
  s->entered[q] = 1;
  if (s->nwaits == 0) return r;
  if (s->waited) return p_cuStreamWaitEvent(stream, d->waited, 0);
  s->waited = 1;
  r = in_step(s, waiting, wait_words(d, stream, s->waits, s->nwaits));
  if (r == CUDA_SUCCESS && s->both) r = p_cuEventRecord(d->waited, stream);
  return r;
}

/* Whether a part after part [i], on the other stream, runs after it. */
static int awaited(const struct nx_part *p, int n, int i) {
  for (int j = i + 1; j < n; j++) {
    if (p[j].queue == p[i].queue) continue;
    for (int k = 0; k < p[j].nafter; k++)
      if (p[j].after[k] == i) return 1;
  }
  return 0;
}

/* Enqueues the parts in array order, each after the parts of the other
   stream it names, then releases [v] on the stream of the last part (on
   COMPUTE:0 for none) once both streams' work is done. A part's
   completion is recorded when a later part of the other stream, or the
   release, waits for it. */
static CUresult run(struct submission *s, uint64_t v, const struct nx_part *p,
                    int n) {
  struct device *d = s->d;
  int last[2] = {-1, -1};
  for (int i = 0; i < n; i++) last[p[i].queue] = i;
  int r = n > 0 ? p[n - 1].queue : 0, o = 1 - r;
  CUresult e = CUDA_SUCCESS;
  s->both = last[0] >= 0 && last[1] >= 0;
  for (int i = 0; i < n && e == CUDA_SUCCESS; i++) {
    int q = p[i].queue, across = 0;
    CUstream stream = d->streams[q];
    e = enter(s, q);
    for (int k = 0; k < p[i].nafter; k++)
      across |= p[p[i].after[k]].queue != q;
    if (e == CUDA_SUCCESS && across)
      e = p_cuStreamWaitEvent(stream, d->done[1 - q], 0);
    s->bytes = p[i].copy_bytes;
    if (e == CUDA_SUCCESS && p[i].fill != NULL)
      e = in_step(s, filling, p[i].fill(stream, p[i].arg, v));
    else if (e == CUDA_SUCCESS)
      e = in_step(s, copying,
             p_cuMemcpyAsync(p[i].copy_dst + p[i].copy_dst_offset,
                             p[i].copy_src + p[i].copy_src_offset,
                             p[i].copy_bytes, stream));
    if (e == CUDA_SUCCESS && ((i == last[q] && q != r) || awaited(p, n, i)))
      e = p_cuEventRecord(d->done[q], stream);
  }
  if (e == CUDA_SUCCESS) e = enter(s, r);
  if (e == CUDA_SUCCESS && last[o] >= 0)
    e = p_cuStreamWaitEvent(d->streams[r], d->done[o], 0);
  if (e == CUDA_SUCCESS)
    e = in_step(s, writing,
           p_cuStreamWriteValue64_v2(d->streams[r],
                                     (CUdeviceptr)(uintptr_t)d->word, v, 0));
  if (e == CUDA_SUCCESS) e = p_cuEventRecord(d->released, d->streams[r]);
  if (e == CUDA_SUCCESS) d->released_on = r;
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

/* After a failure, releases [v] after the work [s] queued, so that the word
   still reaches every value: on COMPUTE:0 after COPY:0's work, or on the one
   stream entered. It writes nothing if the context failed, whose work no
   longer runs, or if CUDA refuses a call that orders the release so. */
static void drain(struct submission *s, uint64_t v) {
  struct device *d = s->d;
  int r = s->entered[1] && !s->entered[0];
  CUresult e = query(d->streams[0]);
  if (s->entered[0] && s->entered[1]) {
    e = p_cuEventRecord(d->done[1], d->streams[1]);
    if (e == CUDA_SUCCESS) e = p_cuStreamWaitEvent(d->streams[0], d->done[1], 0);
  }
  if (e == CUDA_SUCCESS) e = enter(s, r);
  if (e == CUDA_SUCCESS)
    e = p_cuStreamWriteValue64_v2(d->streams[r],
                                  (CUdeviceptr)(uintptr_t)d->word, v, 0);
  if (e == CUDA_SUCCESS) e = p_cuEventRecord(d->released, d->streams[r]);
  if (e == CUDA_SUCCESS) d->released_on = r;
}

int device_cuda_room(void *self, const struct nx_part *p, int n) {
  (void)self;
  for (int i = 0; i < n; i++) {
    if (p[i].queue < 0 || p[i].queue > 1 || p[i].words != NULL ||
        p[i].n != 0 || p[i].ring_units != 0 || p[i].segment_bytes != 0)
      return NX_NEVER;
  }
  return NX_FITS;
}

int device_cuda_submit(void *self, uint64_t v, const struct nx_wait *waits,
                       int nwaits, const struct nx_part *parts, int nparts,
                       const uint64_t *handles, int nhandles,
                       const char **failure) {
  struct device *d = self;
  struct submission s = {d, waits, nwaits, {0, 0}, 0, 0, NULL, 0};
  (void)handles;
  (void)nhandles;
  d->last = v;
  CUresult e = push(d->context);
  if (e == CUDA_SUCCESS) {
    if (!d->failed) e = run(&s, v, parts, nparts);
    if (d->failed || e != CUDA_SUCCESS) drain(&s, v);
    e = pop(e);
  }
  if (e != CUDA_SUCCESS && !d->failed) {
    fail(d, &s, e);
    d->failed = 1;
  }
  if (!d->failed) return NX_OK;
  *failure = d->failure;
  return NX_FAILED;
}

/* Assigned to the edge's types, so a signature that drifts from nx_edge.h
   is a compile error. */
static nx_room_fn *const room_entry = device_cuda_room;
static nx_submit_fn *const submit_entry = device_cuda_submit;

value caml_device_cuda_room_entry(value unit) {
  (void)unit;
  return Val_long((intnat)room_entry);
}

value caml_device_cuda_submit_entry(value unit) {
  (void)unit;
  return Val_long((intnat)submit_entry);
}

/* Device_cuda.submit's C side. [v_waits] holds a kind, an address and a
   value per wait. A part is a record whose second field holds ints:
   nx_part's int fields in the order below, then its [after] indices. Waits
   and parts are copied out of the OCaml heap, then submitted without the
   runtime, which runs nothing before the answer, NX_OK or NX_FAILED,
   reaches the caller. */
enum {
  part_queue,
  part_fill,
  part_arg,
  part_ring_units,
  part_segment_bytes,
  part_copy_dst,
  part_copy_dst_offset,
  part_copy_src,
  part_copy_src_offset,
  part_copy_bytes,
  part_after
};

static intnat at(value ints, int f) { return Long_val(Field(ints, f)); }

value caml_device_cuda_submit(value v_self, value v_v, value v_waits,
                              value v_parts) {
  struct device *d = Device_val(v_self);
  int nwaits = (int)(Wosize_val(v_waits) / 3);
  int nparts = (int)Wosize_val(v_parts);
  size_t nafter = 0;
  for (int i = 0; i < nparts; i++)
    nafter += Wosize_val(Field(Field(v_parts, i), 1)) - part_after;
  size_t size = nwaits * sizeof(struct nx_wait) +
                nparts * sizeof(struct nx_part) + nafter * sizeof(int);
  char *mem = size == 0 ? NULL : malloc(size);
  if (size != 0 && mem == NULL) caml_raise_out_of_memory();
  struct nx_wait *w = (struct nx_wait *)mem;
  struct nx_part *p = (struct nx_part *)(w + nwaits);
  int *after = (int *)(p + nparts);
  for (int i = 0; i < nwaits; i++) {
    w[i].kind = (int)at(v_waits, 3 * i);
    w[i].at = (uint64_t)at(v_waits, 3 * i + 1);
    w[i].value = (uint64_t)at(v_waits, 3 * i + 2);
  }
  for (int i = 0; i < nparts; i++) {
    value k = Field(Field(v_parts, i), 1);
    p[i] = (struct nx_part){
        .queue = (int)at(k, part_queue),
        .fill = (int (*)(void *, void *, uint64_t))at(k, part_fill),
        .arg = (void *)at(k, part_arg),
        .ring_units = (size_t)at(k, part_ring_units),
        .segment_bytes = (size_t)at(k, part_segment_bytes),
        .copy_dst = (uint64_t)at(k, part_copy_dst),
        .copy_dst_offset = (uint64_t)at(k, part_copy_dst_offset),
        .copy_src = (uint64_t)at(k, part_copy_src),
        .copy_src_offset = (uint64_t)at(k, part_copy_src_offset),
        .copy_bytes = (uint64_t)at(k, part_copy_bytes),
        .after = after,
        .nafter = (int)(Wosize_val(k) - part_after)};
    for (int j = 0; j < p[i].nafter; j++) *after++ = (int)at(k, part_after + j);
  }
  const char *failure = NULL;
  caml_enter_blocking_section_no_pending();
  int r = device_cuda_submit(d, (uint64_t)Long_val(v_v), w, nwaits, p, nparts,
                             NULL, 0, &failure);
  free(mem);
  caml_leave_blocking_section();
  return Val_int(r);
}

value caml_device_cuda_failure(value v_self) {
  return caml_copy_string(Device_val(v_self)->failure);
}

/* Timeline */

value caml_device_cuda_signaled(value v_self) {
  struct device *d = Device_val(v_self);
  return Val_long(atomic_load_explicit(d->word, memory_order_acquire));
}

value caml_device_cuda_last(value v_self) {
  return Val_long(Device_val(v_self)->last);
}

value caml_device_cuda_word(value v_self) {
  return Val_long((intnat)Device_val(v_self)->word);
}

/* The error that ended the context's work, which CUDA answers to every call
   on its streams, or [0]. */
value caml_device_cuda_failed(value v_self) {
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
value caml_device_cuda_sleep(value v_self, value v_seen, value v_ms) {
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
   last value (what follows a release only records an event), or once both
   streams are idle or one met an error. */
value caml_device_cuda_stop(value v_self) {
  struct device *d = Device_val(v_self);
  uint64_t w = atomic_load_explicit(d->word, memory_order_acquire);
  int stopped = (int64_t)(w - d->last) >= 0, running = 0;
  if (push(d->context) != CUDA_SUCCESS) return Val_false;
  for (int q = 0; q < 2 && !stopped; q++) {
    CUresult s = p_cuStreamQuery(d->streams[q]);
    if (s == CUDA_ERROR_NOT_READY) running = 1;
    else if (s != CUDA_SUCCESS) stopped = 1;
  }
  stopped |= !running;
  if (stopped) {
    while ((int64_t)(w - d->last) < 0 &&
           !atomic_compare_exchange_weak(d->word, &w, d->last)) {
    }
    destroy(d);
  }
  pop(CUDA_SUCCESS);
  return Val_bool(stopped);
}
