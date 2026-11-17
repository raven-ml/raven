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

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if !defined(_WIN32)
#include <fcntl.h>
#endif

#define CAML_NAME_SPACE
#include <caml/alloc.h>
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

/* CUDA_KERNEL_NODE_PARAMS_v3. */
typedef struct {
  CUfunction func;
  unsigned int grid[3], block[3], shared;
  void **params, **extra;
  void *kern;
  CUcontext context;
} kernel_node;

/* CUgraphNodeParams: the node's type, then its parameters in a union of 232
   bytes, of which a kernel node's use [kernel]. Every byte unused is 0. */
typedef struct {
  int type;
  int reserved0[3];
  union {
    long long reserved1[29];
    kernel_node kernel;
  } u;
  long long reserved2;
} node_params;

_Static_assert(sizeof(node_params) == 256, "CUgraphNodeParams is 256 bytes");

/* CUDA_MEMCPY2D_v2: a box of [Height] rows of [WidthInBytes] bytes, each
   side at its address with its pitch. rig's sides are unified addresses,
   so neither names a host pointer or an array. */
typedef struct {
  size_t srcXInBytes, srcY;
  int srcMemoryType;
  const void *srcHost;
  CUdeviceptr srcDevice;
  void *srcArray;
  size_t srcPitch;
  size_t dstXInBytes, dstY;
  int dstMemoryType;
  void *dstHost;
  CUdeviceptr dstDevice;
  void *dstArray;
  size_t dstPitch;
  size_t WidthInBytes, Height;
} memcpy2d;

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
  CUDA_ERROR_NOT_SUPPORTED = 801,
  CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
  CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
  CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,
  CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
  CU_GRAPH_NODE_TYPE_KERNEL = 0,
  CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2, /* then Y and Z */
  CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,  /* then Y and Z */
  CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
  CU_MEMORYTYPE_UNIFIED = 4,
  CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
  CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
  CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
  CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
  CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1,
  CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 0x05,
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
  X(cuDeviceGetPCIBusId, (char *, int, CUdevice))                              \
  X(cuDeviceCanAccessPeer, (int *, CUdevice, CUdevice))                        \
  X(cuDevicePrimaryCtxRetain, (CUcontext *, CUdevice))                         \
  X(cuDevicePrimaryCtxRelease_v2, (CUdevice))                                  \
  X(cuCtxPushCurrent_v2, (CUcontext))                                          \
  X(cuCtxPopCurrent_v2, (CUcontext *))                                         \
  X(cuCtxGetCurrent, (CUcontext *))                                            \
  X(cuCtxGetLimit, (size_t *, int))                                            \
  X(cuCtxSetLimit, (int, size_t))                                              \
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
  X(cuMemGetInfo_v2, (size_t *, size_t *))                                     \
  X(cuMemHostAlloc, (void **, size_t, unsigned int))                           \
  X(cuMemFreeHost, (void *))                                                   \
  X(cuMemHostRegister_v2, (void *, size_t, unsigned int))                      \
  X(cuMemHostUnregister, (void *))                                             \
  X(cuMemHostGetDevicePointer_v2, (CUdeviceptr *, void *, unsigned int))       \
  X(cuPointerGetAttribute, (void *, int, CUdeviceptr))                         \
  X(cuMemGetHandleForAddressRange,                                             \
    (void *, CUdeviceptr, size_t, int, unsigned long long))                    \
  X(cuMemcpyAsync, (CUdeviceptr, CUdeviceptr, size_t, CUstream))               \
  X(cuMemcpy2DAsync_v2, (const memcpy2d *, CUstream))                          \
  X(cuMemsetD8Async, (CUdeviceptr, unsigned char, size_t, CUstream))           \
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
  X(cuGraphAddNode,                                                            \
    (CUgraphNode *, CUgraph, const CUgraphNode *, size_t, node_params *))      \
  X(cuGraphInstantiateWithFlags, (CUgraphExec *, CUgraph, unsigned long long)) \
  X(cuGraphExecNodeSetParams, (CUgraphExec, CUgraphNode, node_params *))       \
  X(cuGraphLaunch, (CUgraphExec, CUstream))                                    \
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

/* The PCI bus address of the CUdevice [v_device] into [v_id], NUL-ended:
   [0], or CUDA's status negated. */
value caml_rig_cuda_bus(value v_device, value v_id) {
  CUresult s = p_cuDeviceGetPCIBusId((char *)Bytes_val(v_id),
                                     (int)caml_string_length(v_id),
                                     Int_val(v_device));
  return answer(s, 0);
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

/* Replay

   A submission's runs of launches play through chains: graphs of n kernel
   nodes, each after the one before, one of each length n from 2 to CHAIN
   for a device. A chain is made from the placeholder kernel, which does
   nothing, at the first submit that needs its length: rig_cuda_room
   answers RIG_GROW, and caml_rig_cuda_grow makes it outside the turn. A
   device that holds no placeholder plays every piece plain.

   A run is a sequence of consecutive launches, cut before a launch that
   starts after a part of COPY:0 that COMPUTE:0 has not yet waited for
   ([across]). A run of one launches plain. A longer one is cut into the
   fewest pieces of at most CHAIN launches, of lengths that differ by at
   most one ([piece]); each piece updates every node of the chain of its
   length to its launches, then launches the chain, one call for its
   kernels:

     parts    L1 ... L40   C   L41 L42 L43     (L a launch, C a copy)
     pieces   [20]  [20]   C   [    3    ]

   [held] is the module of each node's function, NULL where unknown: an
   unload points the nodes of its module, and the unknown ones, back at the
   placeholder before CUDA unloads the module. A chain keeps its graph,
   whose nodes its updates name, until it is destroyed. */

#define CHAIN 32

struct chain {
  CUgraph graph;
  CUgraphExec exec;
  CUgraphNode nodes[CHAIN];
  CUmodule held[CHAIN];
};

/* The chain of a length CUDA refused to make, for a reason other than
   memory or a failed context: the pieces of that length launch plain. */
static struct chain unmade;

/* A device: its context, its two streams and the events that order them, and
   its timeline word: the word rig gave its open, in rig's array, which every
   context reaches registered ([register_words]), or, on a GPU that registers no
   host memory or for a word outside the array, a word of its own from
   cuMemHostAlloc, whose value rig reads through [signaled]. [word] is its host
   address and [word_at] the address CUDA's work writes. A value ends on one
   stream, after its work on both; a value that starts on the other stream waits
   for that end. It lives from the open to the stop. The context is the GPU's
   primary context, which the process keeps: memory and modules are the
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
  uint64_t max_pitch; /* the most bytes of a 2D copy's pitch */
  size_t total;       /* the context's memory, cuMemGetInfo's at open */
  _Atomic uint64_t *word;
  CUdeviceptr word_at;
  uint64_t last;       /* the last value submit received */
  uint64_t committed;  /* the last value written to the word */
  int tail;            /* the stream the last value ended on */
  int flush;           /* the last foreign wait flushes remote writes */
  int failed;
  char failure[256];
  /* Replay: the chains by length, which room installs from [handed], where
     grow leaves what it made; the placeholder kernel and its module; and
     the pieces played through a chain, the launches sent plain and the
     chains made. [mutex] guards [chains], [handed] and the placeholder,
     holds a chain's nodes for the updates of a submit or an unload, and is
     held through a grow's makes. */
  pthread_mutex_t mutex;
  struct chain *chains[CHAIN + 1], *handed[CHAIN + 1];
  CUfunction placeholder;
  CUmodule placeholder_module;
  _Atomic uint64_t pieces, plain, made;
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

/* The L2's fetch from DRAM on a miss, in bytes: the limit CUDA reports
   before any set, and the one rig sets. */
#define CUDA_FETCH 64
#define FETCH 128

/* Widens the L2's fetches from DRAM in the current context to FETCH bytes
   where the context's limit still reads CUDA_FETCH. On an RTX 5000 Ada,
   until a set, the L2 fetches the 32-byte sectors a miss names. A
   kernel that reads 16 bytes of each of many rows a trip, as a product
   over quantised weights does, then misses on every second trip; at 128
   bytes, on every eighth. A random scalar read from DRAM brings four times
   the bytes it brought before. The limit is the context's, so it holds for
   other libraries' kernels too, and a limit of CUDA_FETCH set on purpose
   reads as the default. A hint: a refusal leaves the context as it was. */
static void fetch(void) {
  size_t limit = 0;
  CUresult s = p_cuCtxGetLimit(&limit, CU_LIMIT_MAX_L2_FETCH_GRANULARITY);
  if (s == CUDA_SUCCESS && limit == CUDA_FETCH)
    p_cuCtxSetLimit(CU_LIMIT_MAX_L2_FETCH_GRANULARITY, FETCH);
}

/* Reads the context's memory, then makes [d]'s streams and events, then its
   word ([word_start]), then widens the context's L2 fetches ([fetch]); or
   none of them. */
static CUresult begin(struct device *d, _Atomic uint64_t *slot, int own) {
  size_t avail = 0;
  CUresult s = p_cuMemGetInfo_v2(&avail, &d->total);
  if (s != CUDA_SUCCESS) return s;
  s = start(d);
  if (s != CUDA_SUCCESS) return s;
  s = word_start(d, slot, own);
  if (s != CUDA_SUCCESS) destroy(d);
  else fetch();
  return s;
}

static const struct rig_driver driver = {rig_cuda_room, rig_cuda_submit,
                                         rig_cuda_commit, rig_cuda_settle,
                                         NULL};

/* Reads the GPU's limits on a launch's groups and threads, and on a 2D
   copy's pitch, into [d]. */
static CUresult limits(struct device *d) {
  int pitch = 0;
  CUresult s =
      p_cuDeviceGetAttribute(&pitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, d->device);
  d->max_pitch = (uint64_t)pitch;
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
  if (d == NULL || e == NULL || pthread_mutex_init(&d->mutex, NULL) != 0) {
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
  if (s != CUDA_SUCCESS) {
    pthread_mutex_destroy(&d->mutex);
    free(d);
    free(e);
  }
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

/* The bytes of memory of [v_self]'s context, as its open read them. */
value caml_rig_cuda_total(value v_self) {
  return Val_long((intnat)Device_val(v_self)->total);
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

/* The bytes of GPU memory CUDA reports free in the context [v_context] and
   in all (cuMemGetInfo), stored in [v_out]'s fields 0 and 1: CUDA's status,
   which leaves [v_out] as it was where it is not a success. */
value caml_rig_cuda_memory(value v_context, value v_out) {
  size_t avail = 0, total = 0;
  CUresult s;
  IN_CONTEXT(s, Context_val(v_context), p_cuMemGetInfo_v2(&avail, &total));
  if (s == CUDA_SUCCESS) {
    Field(v_out, 0) = Val_long((intnat)avail);
    Field(v_out, 1) = Val_long((intnat)total);
  }
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

/* The bytes of the CUDA allocation that holds [v_address], or CUDA's status
   negated, as caml_rig_cuda_allocation finds its start. */
value caml_rig_cuda_range_size(value v_context, value v_address) {
  size_t n = 0;
  CUresult s;
  IN_CONTEXT(s, Context_val(v_context),
             p_cuPointerGetAttribute(&n, CU_POINTER_ATTRIBUTE_RANGE_SIZE,
                                     (CUdeviceptr)Long_val(v_address)));
  return answer(s, (intnat)n);
}

/* A new dma-buf descriptor over the [v_n] bytes at [v_address], which start
   and end on the host's page: the descriptor, made close-on-exec, or CUDA's
   status negated. Releases the runtime: CUDA pins the memory for export. */
value caml_rig_cuda_dmabuf(value v_context, value v_address, value v_n) {
  CUcontext c = Context_val(v_context);
  CUdeviceptr a = (CUdeviceptr)Long_val(v_address);
  size_t n = Long_val(v_n);
  int fd = -1;
  CUresult s;
  RELEASED(IN_CONTEXT(s, c,
                      p_cuMemGetHandleForAddressRange(
                          &fd, a, n, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0)));
#if !defined(_WIN32)
  if (s == CUDA_SUCCESS) (void)fcntl(fd, F_SETFD, FD_CLOEXEC);
#endif
  return answer(s, fd);
}

/* The host's page, in bytes. */
value caml_rig_cuda_page(value unit) {
  (void)unit;
#if defined(_WIN32)
  SYSTEM_INFO i;
  GetSystemInfo(&i);
  return Val_long(i.dwPageSize);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
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
   with one. Releases the runtime: CUDA may compile it, and waits for all of
   the context's queued work. */
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

/* What a launch of a function reads: the function and its module, the
   most threads a group of it has, and the most dynamic shared memory a
   group of it takes, which load_module set; the bytes of its parameters;
   and whether it launches plain, never through a chain: one that needs
   clusters, which CUDA checks only at a launch, or whose update CUDA
   refused ([play]). The submits write [plain] under the turn. */
struct function {
  CUfunction f;
  CUmodule module;
  unsigned int threads, shared;
  size_t params;
  int plain;
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

/* Whether [f] needs clusters: a cluster size set at each launch, or one of
   its own. A query CUDA refuses counts as a need. */
static int clustered(CUfunction f) {
  int must = 0, width = 0;
  CUresult s = p_cuFuncGetAttribute(
      &must, CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET, f);
  if (s == CUDA_SUCCESS)
    s = p_cuFuncGetAttribute(&width, CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH,
                             f);
  return s != CUDA_SUCCESS || must != 0 || width > 0;
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
  fn->module = m;
  fn->threads = (unsigned int)threads;
  fn->shared = (unsigned int)shared;
  fn->plain = s == CUDA_SUCCESS && clustered(fn->f);
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
   all of the context's queued work, whichever module it runs. */
value caml_rig_cuda_unload(value v_context, value v_module) {
  CUmodule m = (CUmodule)Long_val(v_module);
  CUresult s;
  RELEASED(IN_CONTEXT(s, Context_val(v_context), p_cuModuleUnload(m)));
  return Val_int(s);
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
  int seen[2]; /* stream q waited for the other's parts before seen[q] */
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

/* Adds one to the count [n], which only the turn's holder writes. */
static void count(_Atomic uint64_t *n) {
  atomic_store_explicit(n, atomic_load_explicit(n, memory_order_relaxed) + 1,
                        memory_order_relaxed);
}

/* An [extra] list of cuLaunchKernel's keys, which passes the parameters as
   one buffer, and that buffer's size. */
struct extra {
  void *keys[5];
  size_t size;
};

/* The kernel node of the launch [p]: its function over the grid of its
   block, with its parameters placed in the block's place and each ref's
   word raised by its slot's address. CUDA copies the parameters at the
   launch or the update that is given them. CUDA refuses a buffer of
   another size than the function's parameters, so the launch's bytes past
   them, which rig allows, are left out. [x] holds the node's [extra]. */
static kernel_node node_of(const struct submission *s, const struct rig_part *p,
                           struct extra *x) {
  const struct function *fn = p->launch.launch;
  const struct rig_block *b = (const void *)(s->args + p->launch.args.block);
  uint8_t *params = rig_place(&p->launch.args, s->args);
  rig_params(params, &p->launch.args, s->args, s->slots);
  x->size = fn->params;
  x->keys[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
  x->keys[1] = params;
  x->keys[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
  x->keys[3] = &x->size;
  x->keys[4] = CU_LAUNCH_PARAM_END;
  kernel_node k = {fn->f,
                   {b->groups[0], b->groups[1], b->groups[2]},
                   {b->threads[0], b->threads[1], b->threads[2]},
                   b->shared,
                   NULL,
                   x->size > 0 ? x->keys : NULL,
                   NULL,
                   NULL};
  return k;
}

/* Launches the launch [p] plain, on [stream], with cuLaunchKernel. */
static CUresult launch(const struct submission *s, CUstream stream,
                       const struct rig_part *p) {
  struct extra x;
  kernel_node k = node_of(s, p, &x);
  count(&s->d->plain);
  return p_cuLaunchKernel(k.func, k.grid[0], k.grid[1], k.grid[2], k.block[0],
                          k.block[1], k.block[2], k.shared, stream, NULL,
                          k.extra);
}

/* Enqueues the copy part [p] on [stream]. A run, and a box whose rows lie
   end to end on both sides, is one copy, and a fill one memset. Another
   box is one 2D copy, or a copy per row where a pitch exceeds the GPU's
   most or CUDA refuses the pitches, as cuMemcpy2DAsync may for two
   regions of one GPU whose pitches cuMemAllocPitch did not make; a refusal
   of its arguments leaves the context usable. */
static CUresult copy(const struct device *d, CUstream stream,
                     const struct rig_part *p) {
  CUdeviceptr dst = p->copy.dst + p->copy.dst_offset;
  if (p->copy.fill)
    return p->copy.width == 0
               ? CUDA_SUCCESS
               : p_cuMemsetD8Async(dst, p->copy.byte, p->copy.width, stream);
  CUdeviceptr src = p->copy.src + p->copy.src_offset;
  uint64_t width = p->copy.width, rows = p->copy.rows;
  uint64_t sp = p->copy.src_pitch, dp = p->copy.dst_pitch;
  if (rows <= 1 || (sp == width && dp == width))
    return p_cuMemcpyAsync(dst, src, width * rows, stream);
  if (width == 0) return CUDA_SUCCESS;
  if (sp <= d->max_pitch && dp <= d->max_pitch) {
    memcpy2d m = {.srcMemoryType = CU_MEMORYTYPE_UNIFIED,
                  .srcDevice = src,
                  .srcPitch = sp,
                  .dstMemoryType = CU_MEMORYTYPE_UNIFIED,
                  .dstDevice = dst,
                  .dstPitch = dp,
                  .WidthInBytes = width,
                  .Height = rows};
    CUresult e = p_cuMemcpy2DAsync_v2(&m, stream);
    if (e != CUDA_ERROR_INVALID_VALUE) return e;
  }
  CUresult e = CUDA_SUCCESS;
  for (uint64_t i = 0; i < rows && e == CUDA_SUCCESS; i++)
    e = p_cuMemcpyAsync(dst + i * dp, src + i * sp, width, stream);
  return e;
}

/* Enqueues the work of the part [p] on [stream]. */
static CUresult work(struct submission *s, CUstream stream,
                     const struct rig_part *p, uint64_t v) {
  s->bytes = p->kind == RIG_COPY ? p->copy.width * p->copy.rows : 0;
  switch (p->kind) {
  case RIG_FILL:
    return in_step(s, filling, rig_fill(stream, p, s->args, s->slots, v));
  case RIG_LAUNCH:
    return in_step(s, launching, launch(s, stream, p));
  default:
    return in_step(s, copying, copy(s->d, stream, p));
  }
}

/* Runs and pieces (Replay) */

/* The steps of a chain's play a failure names. */
static const char updating[] = "updating a chain of kernels",
                  replaying[] = "launching a chain of kernels";

/* Whether the part [p] is a launch that may play through a chain. */
static int chained(const struct rig_part *p) {
  return p->kind == RIG_LAUNCH &&
         !((const struct function *)p->launch.launch)->plain;
}

/* Whether part [i] of [p] starts after a part of the other stream that its
   stream has not waited for: one at or past [seen], the part at which its
   stream last waited for all the other stream's work before it. */
static int across(const struct rig_part *p, int i, int seen) {
  for (int k = 0; k < p[i].nafter; k++) {
    int a = p[i].after[k];
    if (a >= seen && p[a].queue != p[i].queue) return 1;
  }
  return 0;
}

/* The end of the run of [p]'s [n] parts that starts at the chained launch
   [i], once COMPUTE:0 waited for COPY:0's parts before [seen]: the first
   part after it that is no chained launch or that starts after a part of
   COPY:0 at or past [seen]. */
static int run_end(const struct rig_part *p, int n, int i, int seen) {
  int j = i + 1;
  while (j < n && chained(&p[j]) && !across(p, j, seen)) j++;
  return j;
}

/* The length of the first piece of a run of [left] launches. A run of
   L > CHAIN launches takes m = ceil(L / CHAIN) pieces of floor(L / m) or
   ceil(L / m) launches; the first takes ceil(L / m), which leaves m - 1
   pieces' worth that the same rule splits the same way: 33 is 17 + 16, 65
   is 22 + 22 + 21. */
static int piece(int left) {
  int pieces = (left + CHAIN - 1) / CHAIN;
  return (left + pieces - 1) / pieces;
}

/* Points node [i] of [c] at [k], recording its module [m]. */
static CUresult set_node(struct chain *c, int i, const kernel_node *k,
                         CUmodule m) {
  node_params n;
  memset(&n, 0, sizeof n);
  n.type = CU_GRAPH_NODE_TYPE_KERNEL;
  n.u.kernel = *k;
  CUresult e = p_cuGraphExecNodeSetParams(c->exec, c->nodes[i], &n);
  c->held[i] = e == CUDA_SUCCESS ? m : NULL;
  return e;
}

/* Plays the [k] launches [p] through [c], the chain of their length: it
   updates every node under the mutex, then launches [c] on COMPUTE:0 with
   the mutex released, since cuGraphLaunch blocks while the stream is full.
   An update CUDA refuses (CUDA_ERROR_INVALID_VALUE or
   CUDA_ERROR_NOT_SUPPORTED) marks its function plain, sets [*refused] and
   leaves [c] unlaunched; the caller then launches the piece plain, where
   another error of the function meets it again. Any other error is the
   context's. */
static CUresult play(struct submission *s, struct chain *c,
                     const struct rig_part *p, int k, int *refused) {
  struct device *d = s->d;
  CUresult e = CUDA_SUCCESS;
  *refused = 0;
  pthread_mutex_lock(&d->mutex);
  for (int i = 0; i < k && e == CUDA_SUCCESS; i++) {
    const struct function *fn = p[i].launch.launch;
    struct extra x;
    kernel_node node = node_of(s, &p[i], &x);
    e = set_node(c, i, &node, fn->module);
    if (e == CUDA_ERROR_INVALID_VALUE || e == CUDA_ERROR_NOT_SUPPORTED) {
      ((struct function *)fn)->plain = 1;
      *refused = 1;
    }
  }
  pthread_mutex_unlock(&d->mutex);
  if (*refused) return CUDA_SUCCESS;
  if (e != CUDA_SUCCESS) return in_step(s, updating, e);
  e = in_step(s, replaying, p_cuGraphLaunch(c->exec, d->streams[0]));
  if (e == CUDA_SUCCESS) count(&d->pieces);
  return e;
}

/* Enqueues on COMPUTE:0, once [order] placed the first, the run of
   chained launches of [p] from [i] to [end]: each piece through the
   chain of its length, or plain for a run of one or a length with no chain
   made. After a refused update the rest of the run launches plain. */
static CUresult play_run(struct submission *s, const struct rig_part *p,
                         int i, int end) {
  struct device *d = s->d;
  CUresult e = CUDA_SUCCESS;
  int refused = 0;
  while (i < end && e == CUDA_SUCCESS) {
    int k = refused ? end - i : piece(end - i);
    struct chain *c = refused || k == 1 ? NULL : d->chains[k];
    int played = 0;
    if (c != NULL && c != &unmade) {
      e = play(s, c, &p[i], k, &refused);
      played = e == CUDA_SUCCESS && !refused;
    }
    for (int j = i; !played && j < i + k && e == CUDA_SUCCESS; j++)
      e = in_step(s, launching, launch(s, d->streams[0], &p[j]));
    i += k;
  }
  s->unrecorded[0] = 1;
  return e;
}

/* Enters the submission on the stream [q] of part [i] of [p], then, if the
   part starts after a part of the other stream that [q] has not waited
   for, makes [q] wait for all the other stream's work so far. */
static CUresult order(struct submission *s, const struct rig_part *p, int i) {
  int q = p[i].queue;
  CUresult e = enter(s, q);
  if (e != CUDA_SUCCESS || !across(p, i, s->seen[q])) return e;
  e = follow(s, q);
  if (e == CUDA_SUCCESS) s->seen[q] = i;
  return e;
}

/* Enqueues the parts in array order, and ends [v] on the stream of the
   last part (on the last value's stream for none) once both streams' work
   is done. A part that names a part of the other stream its stream has not
   waited for waits for all the work queued there before it, a superset of
   the parts it names that also covers each later part naming only those:
   after one copy on COPY:0, only the first launch that names it waits.
   The order costs one event per stream and no look ahead. A run of
   launches is cut before a launch that waits, so that the wait precedes
   the piece that holds it and no earlier launch waits for COPY:0. */
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
  for (int i = 0; i < n && e == CUDA_SUCCESS;) {
    int q = p[i].queue;
    e = order(s, p, i);
    if (e != CUDA_SUCCESS) break;
    if (chained(&p[i])) {
      int end = run_end(p, n, i, s->seen[0]);
      e = play_run(s, p, i, end);
      i = end;
      continue;
    }
    e = work(s, d->streams[q], &p[i], v);
    s->unrecorded[q] = 1;
    i++;
  }
  if (e == CUDA_SUCCESS) e = enter(s, r);
  if (e == CUDA_SUCCESS && last[o] >= s->seen[r]) e = follow(s, r);
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

/* The lengths of the pieces of [p]'s [n] parts with no chain in [d], as
   bit n for length n, as rig_cuda_submit cuts them. */
static uint64_t unchained(const struct device *d, const struct rig_part *p,
                          int n) {
  uint64_t missing = 0;
  int seen[2] = {0, 0};
  for (int i = 0; i < n;) {
    int q = p[i].queue;
    if (across(p, i, seen[q])) seen[q] = i;
    if (!chained(&p[i])) {
      i++;
      continue;
    }
    int end = run_end(p, n, i, seen[0]);
    while (i < end) {
      int k = piece(end - i);
      if (k > 1 && d->chains[k] == NULL) missing |= UINT64_C(1) << k;
      i += k;
    }
  }
  return missing;
}

/* Moves the chains grow handed into [d]'s, under the mutex: whether [d]
   holds the placeholder that grow makes chains from. */
static int install(struct device *d) {
  pthread_mutex_lock(&d->mutex);
  int placed = d->placeholder != NULL;
  for (int k = 2; k <= CHAIN; k++) {
    if (d->chains[k] != NULL || d->handed[k] == NULL) continue;
    d->chains[k] = d->handed[k];
    d->handed[k] = NULL;
  }
  pthread_mutex_unlock(&d->mutex);
  return placed;
}

/* The lengths of chains the last room of the calling thread that answered
   RIG_GROW found missing, which caml_rig_cuda_grow makes: rig calls grow
   on that thread before the thread's next room. */
static _Thread_local uint64_t asked;

int rig_cuda_room(void *self, int nwaits, const struct rig_part *p, int n,
                  void *scratch, const uint8_t *args) {
  (void)nwaits;
  (void)scratch;
  struct device *d = ((struct edge *)self)->d;
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
  /* A device without the placeholder plays its pieces plain. */
  if (unchained(d, p, n) == 0 || !install(d)) return RIG_FITS;
  asked = unchained(d, p, n);
  return asked == 0 ? RIG_FITS : RIG_GROW;
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
    if (e == CUDA_SUCCESS && !d->failed && v - d->committed >= COMMIT_EVERY)
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

/* Chains (Replay) */

/* The placeholder's node: the kernel that does nothing, over one thread of
   one group, with no parameters. */
static kernel_node placeholder_node(const struct device *d) {
  kernel_node k = {d->placeholder, {1, 1, 1}, {1, 1, 1}, 0,
                   NULL,           NULL,      NULL,      NULL};
  return k;
}

/* Makes [c], a chain of [n] nodes of the placeholder [k] of the module [m],
   each after the one before, and instantiates it, or makes nothing and
   answers CUDA's error. */
static CUresult make_chain(struct chain *c, int n, const kernel_node *k,
                           CUmodule m) {
  CUresult s = p_cuGraphCreate(&c->graph, 0);
  for (int i = 0; i < n && s == CUDA_SUCCESS; i++) {
    node_params p;
    memset(&p, 0, sizeof p);
    p.type = CU_GRAPH_NODE_TYPE_KERNEL;
    p.u.kernel = *k;
    s = p_cuGraphAddNode(&c->nodes[i], c->graph,
                         i > 0 ? &c->nodes[i - 1] : NULL, i > 0, &p);
    c->held[i] = m;
  }
  if (s == CUDA_SUCCESS)
    s = p_cuGraphInstantiateWithFlags(&c->exec, c->graph, 0);
  if (s != CUDA_SUCCESS && c->graph != NULL) p_cuGraphDestroy(c->graph);
  return s;
}

/* Destroys [c], made or handed, unless it is [unmade]. CUDA frees a chain
   whose launches still run once they end. */
static void drop(struct chain *c) {
  if (c == NULL || c == &unmade) return;
  p_cuGraphExecDestroy(c->exec);
  p_cuGraphDestroy(c->graph);
  free(c);
}

/* Destroys [d]'s chains, made and handed, in its context, made current. */
static void unchain(struct device *d) {
  for (int k = 2; k <= CHAIN; k++) {
    drop(d->chains[k]);
    drop(d->handed[k]);
    d->chains[k] = d->handed[k] = NULL;
  }
}

/* Whether [d] holds or was handed the chain of length [n], under the
   mutex. */
static int has_chain(const struct device *d, int n) {
  return d->chains[n] != NULL || d->handed[n] != NULL;
}

/* Makes [d]'s chain of length [n] from the placeholder's node [k] and
   hands it to the next room, under the mutex: CUDA's status, as [grow]
   answers. A chain CUDA refuses to make while the context stands is handed
   as [unmade]. */
static CUresult hand(struct device *d, int n, const kernel_node *k) {
  struct chain *c = calloc(1, sizeof *c);
  if (c == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
  CUresult s = make_chain(c, n, k, d->placeholder_module);
  if (s != CUDA_SUCCESS) {
    free(c);
    if (s == CUDA_ERROR_OUT_OF_MEMORY) return s;
    CUresult failed = streams_failed(d);
    if (failed != CUDA_SUCCESS) return failed;
    c = &unmade;
  }
  if (c != &unmade)
    atomic_fetch_add_explicit(&d->made, 1, memory_order_relaxed);
  d->handed[n] = c;
  return CUDA_SUCCESS;
}

/* Makes the chain of each length of [want], as bit n for length n, that
   [d] neither holds nor was handed, in [d]'s context, made current. It
   holds the mutex throughout, so that no two grows make one length; a play,
   an install, an unload or a placeholder's gift that meets it waits for
   its makes. A make that succeeds or hands [unmade] happens once a length
   in the device's life; one refused for lack of memory hands nothing, so
   each later grow of its length, at every round and submit while memory
   stays short, makes it again under the mutex. Answers
   CUDA_ERROR_OUT_OF_MEMORY if the GPU or the host had no memory for one,
   the context's error if it failed, or CUDA_SUCCESS; what it made before
   stays handed.

   On an RTX 5000 Ada under the 615 driver, a make took 6 us for 2 nodes to
   115 us for 32. The 31 chains held 6.7 MiB of the GPU's memory and 4.5
   MiB of the host's as made, and a node's first update to 4,096 bytes of
   parameters added 8 to 12 KB of host memory. An update cost the host 95
   to 107 ns a node at up to 256 bytes, 330 ns at 4,096, and 270 to 478 ns
   with a change of function, and added 0.14 to 0.41 us a node to the
   chain's time on the GPU. */
static CUresult grow(struct device *d, uint64_t want) {
  CUresult s = CUDA_SUCCESS;
  pthread_mutex_lock(&d->mutex);
  kernel_node k = placeholder_node(d);
  for (int n = 2; n <= CHAIN && s == CUDA_SUCCESS; n++)
    if ((want & (UINT64_C(1) << n)) && !has_chain(d, n)) s = hand(d, n, &k);
  pthread_mutex_unlock(&d->mutex);
  return s;
}

/* Makes the chains the last room of the calling thread found missing
   ([asked]): CUDA's status, as [grow] answers. Releases the runtime: CUDA
   instantiates the graphs. */
value caml_rig_cuda_grow(value v_self) {
  struct device *d = Device_val(v_self);
  uint64_t want = asked;
  CUresult s;
  RELEASED(IN_CONTEXT(s, d->context, grow(d, want)));
  return Val_int(s);
}

/* Points every node of [d]'s chains whose function may be of [m], that of
   [m] or an unknown one, back at the placeholder, in [d]'s context, made
   current, under the mutex. */
static CUresult repoint(struct device *d, CUmodule m) {
  kernel_node k = placeholder_node(d);
  CUresult s = CUDA_SUCCESS;
  pthread_mutex_lock(&d->mutex);
  for (int n = 2; n <= CHAIN && s == CUDA_SUCCESS; n++) {
    struct chain *c = d->chains[n];
    if (c == NULL || c == &unmade) continue;
    for (int i = 0; i < n && s == CUDA_SUCCESS; i++)
      if (c->held[i] == m || c->held[i] == NULL)
        s = set_node(c, i, &k, d->placeholder_module);
  }
  pthread_mutex_unlock(&d->mutex);
  return s;
}

/* Points every node of [v_self]'s chains that may hold a function of the
   module [v_module] back at the placeholder ([repoint]): CUDA's status, or
   [0] once stopped, its chains gone. rig unloads a module only once no
   work that runs it can run, so no update of a submit is pending between
   its node's update and its chain's launch. Releases the runtime. */
value caml_rig_cuda_repoint(value v_self, value v_module) {
  struct device *d = Device_val(v_self);
  CUmodule m = (CUmodule)Long_val(v_module);
  CUresult s;
  if (d == NULL) return Val_int(0);
  RELEASED(IN_CONTEXT(s, d->context, repoint(d, m)));
  return Val_int(s);
}

/* Gives [v_self]'s device the placeholder kernel [v_function] of the module
   [v_module], which the process keeps. Releases the runtime: the mutex may
   be held through another thread's grow. */
value caml_rig_cuda_placeholder(value v_self, value v_module,
                                value v_function) {
  struct device *d = Device_val(v_self);
  CUmodule m = (CUmodule)Long_val(v_module);
  CUfunction f = (CUfunction)Long_val(v_function);
  caml_enter_blocking_section_no_pending();
  pthread_mutex_lock(&d->mutex);
  d->placeholder_module = m;
  d->placeholder = f;
  pthread_mutex_unlock(&d->mutex);
  caml_leave_blocking_section();
  return Val_unit;
}

/* The chains [v_self]'s grows made since it opened; [0] once stopped. */
value caml_rig_cuda_made(value v_self) {
  struct device *d = Device_val(v_self);
  if (d == NULL) return Val_long(0);
  return Val_long(atomic_load_explicit(&d->made, memory_order_relaxed));
}

/* The pieces [v_self]'s submits played through a chain and the launches
   they sent plain, since it opened; [(0, 0)] once stopped. */
value caml_rig_cuda_replays(value v_self) {
  CAMLparam1(v_self);
  CAMLlocal1(r);
  struct device *d = Device_val(v_self);
  uint64_t pieces = 0, plain = 0;
  if (d != NULL) {
    pieces = atomic_load_explicit(&d->pieces, memory_order_relaxed);
    plain = atomic_load_explicit(&d->plain, memory_order_relaxed);
  }
  r = caml_alloc_tuple(2);
  Store_field(r, 0, Val_long(pieces));
  Store_field(r, 1, Val_long(plain));
  CAMLreturn(r);
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
   streams are gone. The chains, the streams and the other events are
   destroyed: CUDA frees them once their work ends. If CUDA records no
   event, [d] stays in the edge, unstopped, but for its chains. */
static int stop_in(struct edge *e, struct device *d) {
  int stopped = reached(e), running = 0;
  unchain(d);
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
  pthread_mutex_destroy(&d->mutex);
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
