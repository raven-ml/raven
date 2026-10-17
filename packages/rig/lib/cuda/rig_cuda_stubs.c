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

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/callback.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>

#include "rig_cuda.h"

/* The platform */

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
#else
#include <dlfcn.h>
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
  CUDA_ERROR_INVALID_VALUE = 1,
  CUDA_ERROR_OUT_OF_MEMORY = 2,
  CUDA_ERROR_NOT_READY = 600,
  CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
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
  X(cuEventSynchronize, (CUevent))                                             \
  X(cuEventQuery, (CUevent))                                                   \
  X(cuEventElapsedTime, (float *, CUevent, CUevent))                           \
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
  X(cuFuncGetParamInfo, (CUfunction, size_t, size_t *, size_t *))              \
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

/* Guards */

/* A guard: a mutex of its own memory, which a custom block holds and
   destroys once collected. No section of the driver's takes a second guard,
   so a thread holds at most one: a signal handler, finaliser or Gc.Memprof
   callback that runs while its thread holds a guard and takes one would
   wait for good on its own, or take two in no fixed order. A guard counts
   among the locks its thread holds (rig_hold in rig_edge.h) from before its
   take to after its give. */
#define Guard_val(v) (*(pthread_mutex_t **)Data_custom_val(v))

/* Whether this thread holds a guard. */
static _Thread_local int holding;

static void guard_finalize(value v) {
  pthread_mutex_t *m = Guard_val(v);
  if (m == NULL) return;
  pthread_mutex_destroy(m);
  free(m);
}

static struct custom_operations guard_ops = {
    "rig.cuda.guard",           guard_finalize,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

value caml_rig_cuda_guard(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(v);
  v = caml_alloc_custom(&guard_ops, sizeof(pthread_mutex_t *), 0, 1);
  Guard_val(v) = NULL;
  pthread_mutex_t *m = malloc(sizeof *m);
  if (m == NULL) caml_raise_out_of_memory();
  int err = pthread_mutex_init(m, NULL);
  if (err != 0) {
    free(m);
    if (err == ENOMEM || err == EAGAIN) caml_raise_out_of_memory();
    char why[96];
    snprintf(why, sizeof why, "Rig_cuda: a guard: %s", strerror(err));
    caml_failwith(why);
  }
  Guard_val(v) = m;
  CAMLreturn(v);
}

/* Takes the guard [v_g]: by try-lock, and otherwise after releasing the
   runtime without running pending signal handlers, which run at the
   caller's next poll point. Raises Rig_cuda.Refused, which rig_cuda.ml
   registers, before it takes anything, if this thread holds a guard; no
   other exception can end the take, so a caller that takes it as the last
   step before its handler gives it back on every exit. */
value caml_rig_cuda_guard_take(value v_g) {
  if (holding)
    caml_raise_with_string(*caml_named_value("Rig_cuda.Refused"),
                           "Rig_cuda: called from a signal handler, "
                           "finaliser or Gc.Memprof callback while this "
                           "thread holds a lock of the driver's");
  rig_hold();
  pthread_mutex_t *m = Guard_val(v_g);
  if (pthread_mutex_trylock(m) != 0) {
    caml_enter_blocking_section_no_pending();
    int err = pthread_mutex_lock(m);
    caml_leave_blocking_section();
    if (err != 0) {
      fprintf(stderr, "Rig_cuda: a guard: %s\n", strerror(err));
      abort();
    }
  }
  holding = 1;
  return Val_unit;
}

value caml_rig_cuda_guard_give(value v_g) {
  holding = 0;
  pthread_mutex_unlock(Guard_val(v_g));
  rig_unhold();
  return Val_unit;
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

/* Sets [s] to the status of [call], made with the context [c] current, or
   to the push's if the context cannot be made current, and [call] is not
   made. */
#define IN_CONTEXT(s, c, call)                                                 \
  do {                                                                         \
    (s) = push(c);                                                             \
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

/* Times

   A timed value records an event on each stream it enters, after its
   waits, and one after its work on the stream it ends on, into the stamp
   of its value modulo STAMPS. A submit whose value's stamp still holds an
   earlier value's events waits for that value's end. An event's time is
   its float of milliseconds since the reference event, plus the host's
   time at the reference, measured between two readings of the host's
   clock: at open, and again by a timed submit that finds the reference
   REFERENCE_NS old, which may block. A float's precision falls as the two
   events grow apart, and the GPU timer drifts from the host's clock, so a
   reference is taken afresh rather than carried over from the last. Times
   are converted in value order, and a value's start is never before the
   end of the value converted before it: a value's work starts once every
   earlier value's work has completed, on both streams. */

#define STAMPS 1024
#define REFERENCE_NS UINT64_C(8000000000)

/* The stamp of a timed value: the events recorded when its work started on
   each stream it entered and when it ended, and where its times go, NULL
   once they are written or dropped. */
struct stamp {
  CUevent start[2], end;
  int entered[2];
  uint64_t v;
  uint64_t *times;
};

/* A device: its context, its two streams and the events that order them, and
   its timeline word: the word rig gave its open, in rig's array, which every
   context reaches registered ([register_words]), or, on a GPU that registers no
   host memory or for a word outside the array, a word of its own from
   cuMemHostAlloc, whose value rig reads through [signaled]. [word] is its host
   address and [word_at] the address CUDA's work writes. A value ends on one
   stream, after its work on both; a value that starts on the other stream waits
   for that end. It lives from the open to the stop. The context is the GPU's
   primary context, which the process keeps: memory, modules and graphs are the
   context's, and outlive the device. */
struct device {
  CUdevice device;
  CUcontext context;
  int keeps; /* its open registered rig's array: its context stays */
  CUstream streams[2]; /* COMPUTE:0, COPY:0 */
  CUevent done[2];     /* the latest part of a stream another one waits for */
  CUevent ended;       /* the end of the last value, for the other stream */
  CUevent waited;      /* the foreign waits of the value being submitted */
  /* Times: a stream that runs nothing but references; the reference event
     and the host's time at it; the event a measure records into; the
     stamps; every timed value up to [converted] converted, and the timed
     values whose times are still to write; and the end of the last value
     converted. */
  CUstream clock;
  CUevent reference, trial;
  uint64_t reference_ns;
  struct stamp stamps[STAMPS];
  uint64_t converted, unconverted, last_end;
  unsigned int groups[3], threads[3]; /* a launch's most along x, y, z */
  _Atomic uint64_t *word;
  CUdeviceptr word_at;
  uint64_t last;       /* the last value submit received */
  uint64_t committed;  /* the last value written to the word */
  int tail;            /* the stream the last value ended on */
  int flush;           /* the last foreign wait flushes remote writes */
  int failed;
  char failure[256];
};

/* A device's edge, kept for the process: its word, the last value its
   submit received, as the stop found it, and its state, NULL once
   stopped. A stop that finds work running leaves its context, an event
   behind the work on each stream, which [caml_rig_cuda_ended] asks, or,
   where it cannot ask CUDA, the state, which [caml_rig_cuda_ended] stops
   again. A word of the device's own goes back once rig frees it, and the
   edge's word is then [kept], which holds its last value
   ([caml_rig_cuda_keep_word]). */
struct edge {
  const struct rig_driver *driver;
  _Atomic uint64_t *word;
  _Atomic uint64_t kept;
  uint64_t last;
  struct device *d;
  CUcontext context;
  CUevent ends[2];
  struct device *unstopped;
};

#define Edge_val(v) ((struct edge *)Long_val(v))
#define Device_val(v) (Edge_val(v)->d)
#define Context_val(v) ((CUcontext)Long_val(v))

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
  for (int i = 0; i < STAMPS; i++) {
    struct stamp *t = &d->stamps[i];
    CUevent *e[3] = {&t->start[0], &t->start[1], &t->end};
    for (int k = 0; k < 3; k++) {
      if (*e[k] != NULL) p_cuEventDestroy_v2(*e[k]);
      *e[k] = NULL;
    }
  }
  if (d->reference != NULL) p_cuEventDestroy_v2(d->reference);
  if (d->trial != NULL) p_cuEventDestroy_v2(d->trial);
  if (d->clock != NULL) p_cuStreamDestroy_v2(d->clock);
  d->reference = d->trial = NULL;
  d->clock = NULL;
}

/* The round trips a reference is chosen from. */
#define REFERENCES 8

/* The round trip a measure's closest must be within, unless the measure
   runs out of its REFERENCES_MOST tries. */
#define ROUND_TRIP_BOUND_NS UINT64_C(10000)
#define REFERENCES_MOST 16

/* How long a round trip of the open's measure may take. */
#define ROUND_TRIP_NS UINT64_C(1000000000)

/* Records the trial event on the idle clock stream and waits for it by
   asking CUDA until it has completed, so that no wake-up of a blocked
   thread, which a loaded host may delay for milliseconds, adds to the
   round trip: CUDA's status, or CUDA_ERROR_NOT_READY past ROUND_TRIP_NS. */
static CUresult round_trip(struct device *d, uint64_t before) {
  CUresult s = p_cuEventRecord(d->trial, d->clock);
  while (s == CUDA_SUCCESS || s == CUDA_ERROR_NOT_READY) {
    s = p_cuEventQuery(d->trial);
    if (s != CUDA_ERROR_NOT_READY) return s;
    if (rig_now_ns() - before > ROUND_TRIP_NS) return s;
  }
  return s;
}

/* Takes [d]'s reference: records an event on the idle clock stream
   between two readings of the host's clock, REFERENCES times, and more
   while the closest is over ROUND_TRIP_BOUND_NS, and keeps the event whose
   readings lie closest, with the middle of them. A GPU that idled may take
   milliseconds to answer the first. */
static CUresult refer(struct device *d) {
  uint64_t best = UINT64_MAX;
  CUresult s = CUDA_SUCCESS;
  for (int i = 0; s == CUDA_SUCCESS && i < REFERENCES_MOST &&
                  (i < REFERENCES || best > ROUND_TRIP_BOUND_NS);
       i++) {
    uint64_t before = rig_now_ns();
    s = round_trip(d, before);
    uint64_t took = rig_now_ns() - before;
    if (s != CUDA_SUCCESS || took >= best) continue;
    best = took;
    CUevent e = d->reference;
    d->reference = d->trial;
    d->trial = e;
    d->reference_ns = before + took / 2;
  }
  return s;
}

/* Makes the stream and the events of [d]'s references, and its first
   reference. A stamp's events are made at its first timed value, so that
   a device that never times its work makes none of them. */
static CUresult clock_start(struct device *d) {
  CUresult s = p_cuStreamCreate(&d->clock, CU_STREAM_NON_BLOCKING);
  if (s == CUDA_SUCCESS) s = p_cuEventCreate(&d->reference, 0);
  if (s == CUDA_SUCCESS) s = p_cuEventCreate(&d->trial, 0);
  if (s == CUDA_SUCCESS) s = refer(d);
  return s;
}

/* Makes [t]'s events unless it has them. */
static CUresult stamp_events(struct stamp *t) {
  CUresult s = CUDA_SUCCESS;
  if (t->start[0] == NULL) s = p_cuEventCreate(&t->start[0], 0);
  if (s == CUDA_SUCCESS && t->start[1] == NULL)
    s = p_cuEventCreate(&t->start[1], 0);
  if (s == CUDA_SUCCESS && t->end == NULL) s = p_cuEventCreate(&t->end, 0);
  return s;
}

/* Whether rig's array of timeline words is registered. The first device
   that opens registers it, PORTABLE so that every context reaches it, in
   its GPU's primary context, which the process keeps from then on, even if
   that open fails ([keeps]): the registration lives as long as that
   context. A second registration, from an open that raced the first,
   answers that the memory is registered. */
static _Atomic int words_registered;

static CUresult register_words(struct device *d) {
  if (atomic_load_explicit(&words_registered, memory_order_acquire))
    return CUDA_SUCCESS;
  CUresult s = p_cuMemHostRegister_v2(rig_words(), RIG_WORDS_BYTES,
                                      CU_MEMHOST_PORTABLE_DEVICEMAP);
  if (s == CUDA_SUCCESS) d->keeps = 1;
  if (s == CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED) s = CUDA_SUCCESS;
  if (s == CUDA_SUCCESS)
    atomic_store_explicit(&words_registered, 1, memory_order_release);
  return s;
}

/* Makes [d]'s word, on [slot] in rig's array, or of its own if [own], with
   the value [slot] holds; or none. */
static CUresult word_start(struct device *d, _Atomic uint64_t *slot, int own) {
  uint64_t base = atomic_load_explicit(slot, memory_order_acquire);
  void *word = slot;
  CUresult s = own ? p_cuMemHostAlloc(&word, sizeof(uint64_t),
                                      CU_MEMHOST_PORTABLE_DEVICEMAP)
                   : register_words(d);
  CUdeviceptr at = 0;
  if (s == CUDA_SUCCESS) s = p_cuMemHostGetDevicePointer_v2(&at, word, 0);
  if (s != CUDA_SUCCESS) {
    if (own && word != slot) p_cuMemFreeHost(word);
    return s;
  }
  d->word = word;
  d->word_at = at;
  if (own) atomic_store_explicit(d->word, base, memory_order_relaxed);
  d->last = d->committed = base;
  return s;
}

/* Makes [d]'s streams and events, or none of them. The events that order
   the streams record no timing, the form CUDA waits on fastest. */
static CUresult start(struct device *d) {
  CUresult s = CUDA_SUCCESS;
  for (int q = 0; q < 2 && s == CUDA_SUCCESS; q++) {
    s = p_cuStreamCreate(&d->streams[q], CU_STREAM_NON_BLOCKING);
    if (s == CUDA_SUCCESS)
      s = p_cuEventCreate(&d->done[q], CU_EVENT_DISABLE_TIMING);
  }
  if (s == CUDA_SUCCESS)
    s = p_cuEventCreate(&d->ended, CU_EVENT_DISABLE_TIMING);
  if (s == CUDA_SUCCESS)
    s = p_cuEventCreate(&d->waited, CU_EVENT_DISABLE_TIMING);
  if (s == CUDA_SUCCESS) s = clock_start(d);
  if (s != CUDA_SUCCESS) destroy(d);
  return s;
}

/* Makes [d]'s streams and events, then its word ([word_start]), or none of
   them. */
static CUresult begin(struct device *d, _Atomic uint64_t *slot, int own) {
  CUresult s = start(d);
  if (s != CUDA_SUCCESS) return s;
  s = word_start(d, slot, own);
  if (s != CUDA_SUCCESS) destroy(d);
  return s;
}

static const struct rig_driver driver = {rig_cuda_room, rig_cuda_submit,
                                         rig_cuda_commit, rig_cuda_settle};

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

/* Opens a device on the primary context of the CUdevice [v_device], with
   the word rig gave at [v_word], or a word of its own if [v_own]: its
   state's address, or CUDA's status negated. A device that does not open
   gives back its retain of the context. The word comes last, so that a
   registration is made in a context the process keeps. Releases the
   runtime: CUDA may create the context. */
value caml_rig_cuda_open(value v_device, value v_word, value v_own) {
  _Atomic uint64_t *slot = (_Atomic uint64_t *)Nativeint_val(v_word);
  int own = Bool_val(v_own);
  struct device *d = calloc(1, sizeof *d);
  struct edge *e = calloc(1, sizeof *e);
  int flush = 0;
  if (d == NULL || e == NULL) {
    free(d);
    free(e);
    caml_raise_out_of_memory();
  }
  d->device = Int_val(v_device);
  caml_enter_blocking_section_no_pending();
  CUresult s = p_cuDeviceGetAttribute(
      &flush, CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES, d->device);
  if (s == CUDA_SUCCESS) s = limits(d);
  if (s == CUDA_SUCCESS)
    s = p_cuDevicePrimaryCtxRetain(&d->context, d->device);
  if (s == CUDA_SUCCESS) {
    IN_CONTEXT(s, d->context, begin(d, slot, own));
    if (s != CUDA_SUCCESS && !d->keeps)
      p_cuDevicePrimaryCtxRelease_v2(d->device);
  }
  if (s != CUDA_SUCCESS) free(d), free(e);
  caml_leave_blocking_section();
  if (s != CUDA_SUCCESS) return Val_long(-s);
  d->flush = flush != 0;
  e->driver = &driver;
  e->word = d->word;
  e->d = d;
  return Val_long((intnat)e);
}

/* The primary context of [v_self]'s GPU. */
value caml_rig_cuda_context(value v_self) {
  return Val_long((intnat)Device_val(v_self)->context);
}

/* Whether the CUdevice [v_device], of primary context [v_context],
   addresses the memory of the CUdevice [v_home], of primary context
   [v_home_context]: [1], after enabling the access; [0] if CUDA says it
   cannot; or CUDA's status negated. Releases the runtime: enabling maps
   the peer's allocations, which may take long (unmeasured: kimchi has one
   GPU). */
value caml_rig_cuda_peer(value v_device, value v_context, value v_home,
                         value v_home_context) {
  CUdevice d = Int_val(v_device), h = Int_val(v_home);
  int can = 0;
  if (d == h) return Val_long(1);
  CUresult s;
  caml_enter_blocking_section_no_pending();
  s = p_cuDeviceCanAccessPeer(&can, d, h);
  if (s == CUDA_SUCCESS && can != 0)
    IN_CONTEXT(s, Context_val(v_context),
               p_cuCtxEnablePeerAccess(Context_val(v_home_context), 0));
  caml_leave_blocking_section();
  return answer(s, can != 0);
}

/* Memory */

/* [v_n] bytes of GPU memory, or of page-locked host memory mapped for
   every device if [v_host]: their address, or CUDA's status negated.
   Releases the runtime. */
value caml_rig_cuda_alloc(value v_context, value v_host, value v_n) {
  CUcontext c = Context_val(v_context);
  size_t n = Long_val(v_n);
  CUdeviceptr a = 0;
  void *p = NULL;
  CUresult s;
  if (Bool_val(v_host))
    RELEASED(IN_CONTEXT(
        s, c, p_cuMemHostAlloc(&p, n, CU_MEMHOST_PORTABLE_DEVICEMAP)));
  else
    RELEASED(IN_CONTEXT(s, c, p_cuMemAlloc_v2(&a, n)));
  return answer(s, Bool_val(v_host) ? (intnat)p : (intnat)a);
}

/* Frees what caml_rig_cuda_alloc gave: CUDA's status. Releases the
   runtime: cuMemFree may wait for the GPU. */
value caml_rig_cuda_free(value v_context, value v_host, value v_address) {
  CUcontext c = Context_val(v_context);
  intnat a = Long_val(v_address);
  CUresult s;
  if (Bool_val(v_host))
    RELEASED(IN_CONTEXT(s, c, p_cuMemFreeHost((void *)a)));
  else
    RELEASED(IN_CONTEXT(s, c, p_cuMemFree_v2((CUdeviceptr)a)));
  return Val_int(s);
}

/* The address by which CUDA's work reaches the page-locked host memory at
   [v_address], or CUDA's status negated: CUDA_ERROR_INVALID_VALUE if the
   memory is not page-locked. */
value caml_rig_cuda_mapped(value v_context, value v_address) {
  CUdeviceptr a = 0;
  CUresult s;
  IN_CONTEXT(s, Context_val(v_context),
             p_cuMemHostGetDevicePointer_v2(&a, (void *)Long_val(v_address),
                                            0));
  return answer(s, (intnat)a);
}

/* The start of the CUDA allocation or registration that holds [v_address],
   or CUDA's status negated: host memory CUDA allocated or page-locked, under
   unified addressing. */
value caml_rig_cuda_allocation(value v_context, value v_address) {
  CUdeviceptr start = 0;
  CUresult s;
  IN_CONTEXT(s, Context_val(v_context),
             p_cuPointerGetAttribute(&start,
                                     CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                                     (CUdeviceptr)Long_val(v_address)));
  return answer(s, (intnat)start);
}

/* Page-locks the [v_n] bytes of host memory at [v_address] for every
   device if [v_lock], or unlocks the range that starts there: CUDA's
   status. Releases the runtime: CUDA locks or unlocks every page. */
value caml_rig_cuda_lock(value v_context, value v_lock, value v_address,
                            value v_n) {
  CUcontext c = Context_val(v_context);
  void *p = (void *)Long_val(v_address);
  size_t n = Long_val(v_n);
  CUresult s;
  if (Bool_val(v_lock))
    RELEASED(IN_CONTEXT(
        s, c, p_cuMemHostRegister_v2(p, n, CU_MEMHOST_PORTABLE_DEVICEMAP)));
  else
    RELEASED(IN_CONTEXT(s, c, p_cuMemHostUnregister(p)));
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
  RELEASED(IN_CONTEXT(s, d->context, load_module(d, &m, image)));
  free(image);
  CAMLreturn(answer(s, (intnat)m));
}

/* What a launch of a function reads: the function, the most threads a
   group of it has, and the most dynamic shared memory a group of it
   takes, which load_module set; and the bytes of its parameters. */
struct function {
  CUfunction f;
  unsigned int threads, shared;
  size_t params;
};

/* The end of [f]'s last parameter into [*end]. cuFuncGetParamInfo answers
   CUDA_ERROR_INVALID_VALUE for the index past the last. */
static CUresult params_end(CUfunction f, size_t *end) {
  size_t offset = 0, size = 0;
  CUresult s = CUDA_SUCCESS;
  *end = 0;
  for (size_t i = 0; s == CUDA_SUCCESS; i++) {
    s = p_cuFuncGetParamInfo(f, i, &offset, &size);
    if (s == CUDA_SUCCESS && offset + size > *end) *end = offset + size;
  }
  return s == CUDA_ERROR_INVALID_VALUE ? CUDA_SUCCESS : s;
}

static CUresult find(struct function *fn, CUmodule m, const char *name) {
  int threads = 0, shared = 0;
  CUresult s = p_cuModuleGetFunction(&fn->f, m, name);
  if (s == CUDA_SUCCESS)
    s = p_cuFuncGetAttribute(&threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                             fn->f);
  if (s == CUDA_SUCCESS)
    s = p_cuFuncGetAttribute(
        &shared, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, fn->f);
  if (s == CUDA_SUCCESS) s = params_end(fn->f, &fn->params);
  fn->threads = (unsigned int)threads;
  fn->shared = (unsigned int)shared;
  return s;
}

/* The struct function of [v_module]'s function [v_name], in memory of its
   own that caml_rig_cuda_forget frees, or CUDA's status negated. */
value caml_rig_cuda_entry(value v_context, value v_module, value v_name) {
  struct function *fn = malloc(sizeof *fn);
  CUresult s;
  if (fn == NULL) caml_raise_out_of_memory();
  IN_CONTEXT(s, Context_val(v_context),
             find(fn, (CUmodule)Long_val(v_module), String_val(v_name)));
  if (s == CUDA_SUCCESS) return Val_long((intnat)fn);
  free(fn);
  return Val_long(-s);
}

/* The CUfunction of the struct function [v_fn]. */
value caml_rig_cuda_code(value v_fn) {
  return Val_long((intnat)((struct function *)Long_val(v_fn))->f);
}

/* The bytes of parameters of the struct function [v_fn]. */
value caml_rig_cuda_params(value v_fn) {
  return Val_long((intnat)((struct function *)Long_val(v_fn))->params);
}

/* The most threads a group of the struct function [v_fn] has. */
value caml_rig_cuda_threads(value v_fn) {
  return Val_long((intnat)((struct function *)Long_val(v_fn))->threads);
}

value caml_rig_cuda_forget(value v_fn) {
  free((void *)Long_val(v_fn));
  return Val_unit;
}

/* Unloads [v_module]: CUDA's status. Releases the runtime: CUDA waits for
   the GPU's running work, whichever module it runs. */
value caml_rig_cuda_unload(value v_context, value v_module) {
  CUmodule m = (CUmodule)Long_val(v_module);
  CUresult s;
  RELEASED(IN_CONTEXT(s, Context_val(v_context), p_cuModuleUnload(m)));
  return Val_int(s);
}

/* Graphs */

value caml_rig_cuda_process(value unit) {
  (void)unit;
  return Val_long(getpid());
}

/* Destroys a graph in the context [c], which outlives the device that made
   the graph. */
static void release_graph(CUcontext c, CUgraphExec exec, CUgraph graph) {
  if (push(c) == CUDA_SUCCESS) {
    if (exec != NULL) p_cuGraphExecDestroy(exec);
    if (graph != NULL) p_cuGraphDestroy(graph);
    pop(CUDA_SUCCESS);
  }
}

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
  if (s != CUDA_SUCCESS && *graph != NULL) {
    p_cuGraphDestroy(*graph);
    *graph = NULL;
  }
  return s;
}

/* The graph of the kernels [v_funcs], [v_sizes] (seven per kernel, each
   checked to fit 32 bits) and [v_args] in the context [v_context], as
   [(0, exec, graph, nodes)], or [(status, 0, 0, [||])] with CUDA's status.
   Releases the runtime: CUDA instantiates the graph. */
value caml_rig_cuda_graph(value v_context, value v_funcs, value v_sizes,
                          value v_args) {
  CAMLparam4(v_context, v_funcs, v_sizes, v_args);
  CAMLlocal3(r, v_nodes, x);
  CUcontext c = Context_val(v_context);
  int n = (int)Wosize_val(v_funcs);
  /* Every OCaml allocation precedes native ownership. On success only the
     preallocated nativeint payloads are filled. */
  r = caml_alloc_tuple(4);
  for (int i = 0; i < 4; i++) Store_field(r, i, Val_unit);
  x = caml_copy_nativeint(0);
  Store_field(r, 1, x);
  x = caml_copy_nativeint(0);
  Store_field(r, 2, x);
  v_nodes = caml_alloc(n, 0);
  for (int i = 0; i < n; i++) Store_field(v_nodes, i, Val_unit);
  Store_field(r, 3, v_nodes);
  for (int i = 0; i < n; i++) {
    x = caml_copy_nativeint(0);
    Store_field(v_nodes, i, x);
  }
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
  RELEASED({
    IN_CONTEXT(s, c, make_graph(&k, &graph, &exec, nodes));
    if (s != CUDA_SUCCESS) release_graph(c, exec, graph);
  });
  free(k.funcs), free(k.sizes), free(k.args), free(k.lengths);
  Store_field(r, 0, Val_int(s));
  if (s == CUDA_SUCCESS) {
    Nativeint_val(Field(r, 1)) = (intnat)exec;
    Nativeint_val(Field(r, 2)) = (intnat)graph;
    for (int i = 0; i < n; i++)
      Nativeint_val(Field(v_nodes, i)) = (intnat)nodes[i];
  } else {
    Store_field(r, 3, Atom(0));
  }
  free(nodes);
  CAMLreturn(r);
}

/* Destroys the executable [v_exec] and the graph [v_graph] in the context
   [v_context]. CUDA's answers are dropped: after a fault or a stop the
   context remains, and its error is the only answer. Releases the
   runtime. */
value caml_rig_cuda_graph_release(value v_context, value v_exec,
                                  value v_graph) {
  CAMLparam3(v_context, v_exec, v_graph);
  CUcontext c = Context_val(v_context);
  CUgraphExec exec = (CUgraphExec)Nativeint_val(v_exec);
  CUgraph graph = (CUgraph)Nativeint_val(v_graph);
  caml_enter_blocking_section_no_pending();
  release_graph(c, exec, graph);
  caml_leave_blocking_section();
  CAMLreturn(Val_unit);
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
  uint8_t *args;
  const uint64_t *slots;
  int entered[2];
  int waited;
  int both;
  int unrecorded[2];
  const char *step;  /* the step that failed; NULL: ordering the streams */
  uint64_t bytes;    /* the bytes of the last copy */
  struct stamp *stamp; /* NULL unless the value is timed */
};

/* The steps a failure names, beside ordering the streams. */
static const char filling[] = "running a fill", copying[] = "copying",
                  launching[] = "launching a kernel",
                  waiting[] = "waiting on a word",
                  writing[] = "writing the word", timing[] = "timing the work";

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

/* The foreign waits on [stream], placed on the first stream entered and
   reached by the other through an event. */
static CUresult place_waits(struct submission *s, CUstream stream) {
  struct device *d = s->d;
  if (s->nwaits == 0) return CUDA_SUCCESS;
  if (s->waited) return p_cuStreamWaitEvent(stream, d->waited, 0);
  s->waited = 1;
  CUresult r = in_step(s, waiting, wait_words(d, stream, s->waits, s->nwaits));
  if (r == CUDA_SUCCESS && s->both) r = p_cuEventRecord(d->waited, stream);
  return r;
}

/* Before the first command of the submission on stream [q]: the end of the
   last value, if it ended on the other stream, the foreign waits, and, for
   a timed value, the record of when its work starts there. */
static CUresult enter(struct submission *s, int q) {
  struct device *d = s->d;
  CUstream stream = d->streams[q];
  CUresult r = CUDA_SUCCESS;
  if (s->entered[q]) return r;
  if (d->tail != q) r = p_cuStreamWaitEvent(stream, d->ended, 0);
  if (r != CUDA_SUCCESS) return r;
  s->entered[q] = 1;
  r = place_waits(s, stream);
  if (r != CUDA_SUCCESS || s->stamp == NULL) return r;
  s->stamp->entered[q] = 1;
  return in_step(s, timing, p_cuEventRecord(s->stamp->start[q], stream));
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
   block, with its parameters placed in the block's place and each ref's
   word raised by its slot's address: CUDA copies the parameters at the
   call. CUDA refuses a buffer of another size than the function's
   parameters, so the launch's bytes past them, which rig allows, are left
   out. */
static CUresult launch(const struct submission *s, CUstream stream,
                       const struct rig_part *p) {
  const struct function *fn = p->launch.launch;
  const struct rig_block *b = (const void *)(s->args + p->launch.args.block);
  uint8_t *params = rig_place(&p->launch.args, s->args);
  size_t size = fn->params;
  void *extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, params,
                   CU_LAUNCH_PARAM_BUFFER_SIZE, &size, CU_LAUNCH_PARAM_END};
  rig_params(params, &p->launch.args, s->args, s->slots);
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
    return in_step(s, filling, rig_fill(stream, p, s->args, s->slots, v));
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

/* Times */

/* [ms] milliseconds as nanoseconds. */
static int64_t ns_of_ms(float ms) { return (int64_t)((double)ms * 1e6); }

/* The host's time at the completed event [e]: [*ns], or CUDA's status. */
static CUresult host_time(const struct device *d, CUevent e, uint64_t *ns) {
  float ms;
  CUresult s = p_cuEventElapsedTime(&ms, d->reference, e);
  if (s == CUDA_SUCCESS) *ns = d->reference_ns + (uint64_t)ns_of_ms(ms);
  return s;
}

static uint64_t later(uint64_t a, uint64_t b) { return a > b ? a : b; }

/* Writes the times of [t], whose work has completed and whose value is
   the next to convert, from its events: the earliest of its streams'
   starts and its end, each no earlier than the end of the value converted
   before it, which completed first. Its times go, written or not. */
static CUresult convert(struct device *d, struct stamp *t) {
  uint64_t *times = t->times, start = UINT64_MAX, end, x;
  t->times = NULL;
  d->unconverted--;
  for (int q = 0; q < 2; q++) {
    if (!t->entered[q]) continue;
    CUresult s = host_time(d, t->start[q], &x);
    if (s != CUDA_SUCCESS) return s;
    if (x < start) start = x;
  }
  CUresult s = host_time(d, t->end, &end);
  if (s != CUDA_SUCCESS) return s;
  start = later(start, d->last_end);
  end = later(end, start);
  d->last_end = end;
  times[0] = start;
  times[1] = end;
  return CUDA_SUCCESS;
}

/* Writes, in value order, the times of every timed value up to [v] not
   yet written, whose work has completed with [v]'s. */
static CUresult convert_upto(struct device *d, uint64_t v) {
  uint64_t u = d->converted + 1;
  if (v >= STAMPS && u < v - STAMPS + 1) u = v - STAMPS + 1;
  for (; u <= v && d->unconverted > 0; u++) {
    struct stamp *t = &d->stamps[u % STAMPS];
    if (t->times == NULL || t->v != u) continue;
    CUresult s = convert(d, t);
    if (s != CUDA_SUCCESS) return s;
  }
  d->converted = later(d->converted, v);
  return CUDA_SUCCESS;
}

/* The stamp of [v] into [*t], once the earlier value that held it has
   completed and its times, with every earlier value's, are written. While
   no timed value waits for its times, every value before [v] counts as
   converted, so that a conversion never walks values nobody timed. */
static CUresult take_stamp(struct device *d, uint64_t v, struct stamp **t) {
  struct stamp *x = &d->stamps[v % STAMPS];
  CUresult s = CUDA_SUCCESS;
  if (x->times != NULL) s = p_cuEventSynchronize(x->end);
  if (s == CUDA_SUCCESS && x->times != NULL) s = convert_upto(d, x->v);
  if (s == CUDA_SUCCESS) s = stamp_events(x);
  if (d->unconverted == 0) d->converted = v - 1;
  x->entered[0] = x->entered[1] = 0;
  *t = x;
  return s;
}

/* Writes [v] into the word after the last value's end, which follows every
   earlier value's. */
static CUresult write_word(struct device *d, uint64_t v) {
  CUresult e =
      p_cuStreamWriteValue64_v2(d->streams[d->tail], d->word_at, v, 0);
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
   still reaches every value: on COMPUTE:0 after COPY:0's work, on the one
   stream entered, or on the last value's stream if [s] entered none, since
   [ended] then need not hold the last value's end. It writes nothing if the
   context failed, whose work no longer runs, or if CUDA refuses a call that
   orders the write so. */
static void drain(struct submission *s, uint64_t v) {
  struct device *d = s->d;
  int r = s->entered[0] ? 0 : s->entered[1] ? 1 : d->tail;
  CUresult e = query(d->streams[0]);
  if (e == CUDA_SUCCESS && s->entered[0] && s->entered[1]) {
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
  const struct rig_block *b = (const void *)(args + p->launch.args.block);
  uint64_t threads = 1;
  for (int k = 0; k < 3; k++) {
    if (b->groups[k] == 0 || b->groups[k] > d->groups[k]) return 0;
    if (b->threads[k] == 0 || b->threads[k] > d->threads[k]) return 0;
    threads *= b->threads[k];
  }
  return threads <= fn->threads && b->shared <= fn->shared;
}

int rig_cuda_room(void *self, const struct rig_part *p, int n,
                  void *scratch, const uint8_t *args) {
  (void)scratch;
  const struct device *d = ((struct edge *)self)->d;
  for (int i = 0; i < n; i++) {
    if (p[i].queue < 0 || p[i].queue > 1) return RIG_NEVER;
    switch (p[i].kind) {
    case RIG_FILL:
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
                    void *scratch, uint8_t *args,
                    const uint64_t *slots, int nslots, uint64_t *times,
                    const char **failure) {
  (void)scratch;
  struct device *d = ((struct edge *)self)->d;
  struct submission s = {
      .d = d, .waits = waits, .nwaits = nwaits, .args = args, .slots = slots};
  int pushed;
  (void)nslots;
  d->last = v;
  CUresult e = own(d->context, &pushed);
  if (e == CUDA_SUCCESS) {
    if (!d->failed && times != NULL && nparts > 0) {
      if (rig_now_ns() - d->reference_ns >= REFERENCE_NS)
        e = in_step(&s, timing, refer(d));
      if (e == CUDA_SUCCESS)
        e = in_step(&s, timing, take_stamp(d, v, &s.stamp));
    } else if (d->unconverted == 0)
      d->converted = v;
    if (!d->failed && e == CUDA_SUCCESS) e = run(&s, v, parts, nparts);
    if (e == CUDA_SUCCESS && s.stamp != NULL)
      e = in_step(&s, timing,
                  p_cuEventRecord(s.stamp->end, d->streams[d->tail]));
    if (e == CUDA_SUCCESS && s.stamp != NULL) {
      s.stamp->v = v;
      s.stamp->times = times;
      d->unconverted++;
    }
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

/* Writes [v]'s times, with every earlier timed value's. The end event
   precedes the word's write of [v] on its stream, so the synchronize
   returns at once. A failure loses the device, as a failed submit does. */
int rig_cuda_settle(void *self, uint64_t v, const char **failure) {
  struct device *d = ((struct edge *)self)->d;
  struct stamp *t = &d->stamps[v % STAMPS];
  struct submission s = {.d = d, .step = timing};
  int pushed;
  if (d->failed) {
    *failure = d->failure;
    return RIG_FAILED;
  }
  CUresult e = own(d->context, &pushed);
  if (e == CUDA_SUCCESS) {
    if (t->times != NULL && t->v == v) e = p_cuEventSynchronize(t->end);
    if (e == CUDA_SUCCESS) e = convert_upto(d, v);
    e = restore(pushed, e);
  }
  if (e == CUDA_SUCCESS) return RIG_OK;
  fail(d, &s, e);
  d->failed = 1;
  *failure = d->failure;
  return RIG_FAILED;
}

int rig_cuda_commit(void *self, uint64_t v, const char **failure) {
  struct device *d = ((struct edge *)self)->d;
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
  return Val_long(
      atomic_load_explicit(Edge_val(v_self)->word, memory_order_acquire));
}

/* Points [v_self]'s word at the edge's [kept], which takes the word's
   value, so that the word can be freed: a later claim of the GPU asks the
   edge whether its work ended ([caml_rig_cuda_ended]) and reads [kept].
   The caller holds the claims' lock, under which every such read runs.
   Answers the host address of the word to free. */
value caml_rig_cuda_keep_word(value v_self) {
  struct edge *e = Edge_val(v_self);
  _Atomic uint64_t *w = e->word;
  atomic_store_explicit(&e->kept, atomic_load_explicit(w, memory_order_acquire),
                        memory_order_relaxed);
  e->word = &e->kept;
  return Val_long((intnat)w);
}

/* The host address of the device's word. */
value caml_rig_cuda_word(value v_self) {
  return Val_long((intnat)Edge_val(v_self)->word);
}

/* The address at which CUDA's work writes the device's word. */
value caml_rig_cuda_word_at(value v_self) {
  return Val_long((intnat)Device_val(v_self)->word_at);
}

/* The error a stream of the device met, which ended the context's work and
   which CUDA answers to every call on its streams, or [0]; [0] once
   stopped, its streams gone. */
static CUresult streams_failed(struct device *d) {
  CUresult s = query(d->streams[0]);
  return s != CUDA_SUCCESS ? s : query(d->streams[1]);
}

value caml_rig_cuda_failed(value v_self) {
  struct device *d = Device_val(v_self);
  CUresult s;
  if (d == NULL) return Val_int(0);
  IN_CONTEXT(s, d->context, streams_failed(d));
  return Val_int(s);
}

/* Whether the word of [e]'s device holds the last value. */
static int reached(const struct edge *e) {
  uint64_t w = atomic_load_explicit(e->word, memory_order_acquire);
  return (int64_t)(w - e->last) >= 0;
}

/* Stops [d], the state of [e]'s device, in its context, made current:
   [true] if its work no longer writes memory, after raising the word to
   the last value by compare-and-set. The work no longer writes once the
   word holds the last value (its write is the work's last command), or
   once both streams are idle or one met an error. The streams are asked
   before any write is enqueued, since a write would keep an idle stream
   busy. A fault counts as an end because CUDA then runs none of the
   context's queued work: its reference does not state this, and the 615
   driver behaves so.

   Work that still runs gets the write of the last value, if no commit
   wrote it, so the word reaches it once the work ends; and each stream
   records its [done] event behind its work, which the edge takes, so a
   later [caml_rig_cuda_ended] finds the work's end or its fault once the
   streams are gone. The streams and the other events are then destroyed:
   CUDA frees them once their work ends. If CUDA records no event, [d]
   stays whole in the edge, unstopped. */
static int stop_in(struct edge *e, struct device *d) {
  int stopped = reached(e), running = 0;
  for (int q = 0; q < 2 && !stopped; q++) {
    CUresult s = p_cuStreamQuery(d->streams[q]);
    if (s == CUDA_ERROR_NOT_READY) running = 1;
    else if (s != CUDA_SUCCESS) stopped = 1;
  }
  stopped |= !running;
  if (stopped) rig_raise(e->word, e->last);
  else {
    if (!d->failed && d->committed != d->last) write_word(d, d->last);
    for (int q = 0; q < 2; q++)
      if (p_cuEventRecord(d->done[q], d->streams[q]) != CUDA_SUCCESS) {
        e->unstopped = d;
        return 0;
      }
    for (int q = 0; q < 2; q++) {
      e->ends[q] = d->done[q];
      d->done[q] = NULL;
    }
  }
  destroy(d);
  free(d);
  return stopped;
}

/* Stops [d], the state of [e]'s device, made current in its context
   ([stop_in]), or leaves it to [caml_rig_cuda_ended] if the context cannot
   be made current. */
static int stop_edge(struct edge *e, struct device *d) {
  if (push(d->context) != CUDA_SUCCESS) {
    e->unstopped = d;
    return 0;
  }
  int stopped = stop_in(e, d);
  pop(CUDA_SUCCESS);
  return stopped;
}

/* Stops a device: [true] if its work no longer writes memory ([stop_in]);
   [false] if work still runs, or the context cannot be made current to
   ask, which leaves the state to [caml_rig_cuda_ended]. The state but for
   the edge goes. CUDA's calls may wait for other threads' calls in the
   context, so they run with the runtime released. */
value caml_rig_cuda_stop(value v_self) {
  struct edge *e = Edge_val(v_self);
  struct device *d = e->d;
  int stopped;
  e->last = d->last;
  e->context = d->context;
  e->d = NULL;
  RELEASED(stopped = stop_edge(e, d));
  return Val_bool(stopped);
}

/* Whether the work that [e]'s stop found running has since ended, asked
   in its context ([caml_rig_cuda_ended]): [1] if it has, [0] if it runs,
   [-1] if the context cannot be made current to ask. */
static int ended_in(struct edge *e) {
  if (push(e->context) != CUDA_SUCCESS) return -1;
  int ended;
  if (e->unstopped != NULL) {
    struct device *d = e->unstopped;
    e->unstopped = NULL;
    ended = stop_in(e, d);
  } else {
    ended = 1;
    for (int q = 0; q < 2; q++) {
      CUresult s = p_cuEventQuery(e->ends[q]);
      if (s == CUDA_ERROR_NOT_READY) ended = 0;
      else if (s != CUDA_SUCCESS) {
        ended = 1;
        break;
      }
    }
    ended |= reached(e);
  }
  if (ended && e->ends[0] != NULL) {
    rig_raise(e->word, e->last);
    for (int q = 0; q < 2; q++) {
      p_cuEventDestroy_v2(e->ends[q]);
      e->ends[q] = NULL;
    }
  }
  pop(CUDA_SUCCESS);
  return ended;
}

/* Whether the work a stop found running has since ended: its word holds
   the last value, or both of its streams' end events have completed or
   met an error, a fault which ends the context's work. A state the stop
   left unstopped is stopped again first. Once the work has ended, the
   word is raised to the last value, which a fault or a refused write left
   short, and the events go. It answers as [ended_in], and the next call
   after an answer of [0] or [-1] asks again. CUDA's calls run with the
   runtime released, as [caml_rig_cuda_stop]'s do. */
value caml_rig_cuda_ended(value v_self) {
  struct edge *e = Edge_val(v_self);
  int ended;
  if (e->unstopped == NULL && e->ends[0] == NULL) return Val_int(reached(e));
  RELEASED(ended = ended_in(e));
  return Val_int(ended);
}
