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
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Hand-declared subset of the CUDA driver and NVRTC APIs. Both libraries are
   resolved with dlopen at first use so this library builds and loads on
   machines without the CUDA toolkit or an NVIDIA driver; device creation
   fails cleanly there instead. */

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUctx_st *CUcontext;
typedef struct CUmod_st *CUmodule;
typedef struct CUfunc_st *CUfunction;
typedef struct CUstream_st *CUstream;
typedef struct CUevent_st *CUevent;
typedef struct CUgraph_st *CUgraph;
typedef struct CUgraphNode_st *CUgraphNode;
typedef struct CUgraphExec_st *CUgraphExec;

#define CU_LAUNCH_PARAM_END ((void *)0x00)
#define CU_LAUNCH_PARAM_BUFFER_POINTER ((void *)0x01)
#define CU_LAUNCH_PARAM_BUFFER_SIZE ((void *)0x02)
#define CU_MEMHOSTALLOC_PORTABLE 0x01
#define CU_MEMORYTYPE_DEVICE 0x02

/* Driver-ABI layouts of the graph node parameter structs, as declared in
   cuda.h for cuGraphAddKernelNode (v1 params) and cuGraphAddMemcpyNode. */
typedef struct {
  CUfunction func;
  unsigned int gridDimX, gridDimY, gridDimZ;
  unsigned int blockDimX, blockDimY, blockDimZ;
  unsigned int sharedMemBytes;
  void **kernelParams;
  void **extra;
} CUDA_KERNEL_NODE_PARAMS_v1;

typedef struct {
  size_t srcXInBytes, srcY, srcZ, srcLOD;
  unsigned int srcMemoryType;
  const void *srcHost;
  CUdeviceptr srcDevice;
  void *srcArray;
  void *reserved0;
  size_t srcPitch, srcHeight;
  size_t dstXInBytes, dstY, dstZ, dstLOD;
  unsigned int dstMemoryType;
  void *dstHost;
  CUdeviceptr dstDevice;
  void *dstArray;
  void *reserved1;
  size_t dstPitch, dstHeight;
  size_t WidthInBytes, Height, Depth;
} CUDA_MEMCPY3D_v2;

static CUresult (*p_cuInit)(unsigned int);
static CUresult (*p_cuDeviceGet)(CUdevice *, int);
static CUresult (*p_cuDeviceComputeCapability)(int *, int *, CUdevice);
static CUresult (*p_cuCtxCreate)(CUcontext *, unsigned int, CUdevice);
static CUresult (*p_cuCtxSetCurrent)(CUcontext);
static CUresult (*p_cuCtxSynchronize)(void);
static CUresult (*p_cuMemAlloc)(CUdeviceptr *, size_t);
static CUresult (*p_cuMemFree)(CUdeviceptr);
static CUresult (*p_cuMemHostAlloc)(void **, size_t, unsigned int);
static CUresult (*p_cuMemFreeHost)(void *);
static CUresult (*p_cuMemcpyHtoDAsync)(CUdeviceptr, const void *, size_t,
                                       CUstream);
static CUresult (*p_cuMemcpyDtoH)(void *, CUdeviceptr, size_t);
static CUresult (*p_cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t,
                                       CUstream);
static CUresult (*p_cuModuleLoadData)(CUmodule *, const void *);
static CUresult (*p_cuModuleGetFunction)(CUfunction *, CUmodule, const char *);
static CUresult (*p_cuModuleUnload)(CUmodule);
static CUresult (*p_cuLaunchKernel)(CUfunction, unsigned int, unsigned int,
                                    unsigned int, unsigned int, unsigned int,
                                    unsigned int, unsigned int, CUstream,
                                    void **, void **);
static CUresult (*p_cuGetErrorString)(CUresult, const char **);
static CUresult (*p_cuEventCreate)(CUevent *, unsigned int);
static CUresult (*p_cuEventRecord)(CUevent, CUstream);
static CUresult (*p_cuEventSynchronize)(CUevent);
static CUresult (*p_cuEventElapsedTime)(float *, CUevent, CUevent);
static CUresult (*p_cuEventDestroy)(CUevent);
static CUresult (*p_cuGraphCreate)(CUgraph *, unsigned int);
static CUresult (*p_cuGraphAddKernelNode)(CUgraphNode *, CUgraph,
                                          const CUgraphNode *, size_t,
                                          const CUDA_KERNEL_NODE_PARAMS_v1 *);
static CUresult (*p_cuGraphAddMemcpyNode)(CUgraphNode *, CUgraph,
                                          const CUgraphNode *, size_t,
                                          const CUDA_MEMCPY3D_v2 *, CUcontext);
static CUresult (*p_cuGraphInstantiate)(CUgraphExec *, CUgraph, CUgraphNode *,
                                        char *, size_t);
static CUresult (*p_cuGraphExecKernelNodeSetParams)(
    CUgraphExec, CUgraphNode, const CUDA_KERNEL_NODE_PARAMS_v1 *);
static CUresult (*p_cuGraphExecMemcpyNodeSetParams)(CUgraphExec, CUgraphNode,
                                                    const CUDA_MEMCPY3D_v2 *,
                                                    CUcontext);
static CUresult (*p_cuGraphLaunch)(CUgraphExec, CUstream);
static CUresult (*p_cuGraphExecDestroy)(CUgraphExec);
static CUresult (*p_cuGraphDestroy)(CUgraph);

static void *cuda_handle = NULL;

static void ensure_cuda(void) {
  if (cuda_handle != NULL) return;
  cuda_handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (cuda_handle == NULL)
    cuda_handle = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
  if (cuda_handle == NULL)
    caml_failwith("CUDA driver library (libcuda.so.1) not found");
#define LOAD_CUDA(var, name)                                          \
  do {                                                                \
    var = dlsym(cuda_handle, name);                                   \
    if (var == NULL) caml_failwith("CUDA driver is missing " name);   \
  } while (0)
  LOAD_CUDA(p_cuInit, "cuInit");
  LOAD_CUDA(p_cuDeviceGet, "cuDeviceGet");
  LOAD_CUDA(p_cuDeviceComputeCapability, "cuDeviceComputeCapability");
  LOAD_CUDA(p_cuCtxCreate, "cuCtxCreate_v2");
  LOAD_CUDA(p_cuCtxSetCurrent, "cuCtxSetCurrent");
  LOAD_CUDA(p_cuCtxSynchronize, "cuCtxSynchronize");
  LOAD_CUDA(p_cuMemAlloc, "cuMemAlloc_v2");
  LOAD_CUDA(p_cuMemFree, "cuMemFree_v2");
  LOAD_CUDA(p_cuMemHostAlloc, "cuMemHostAlloc");
  LOAD_CUDA(p_cuMemFreeHost, "cuMemFreeHost");
  LOAD_CUDA(p_cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
  LOAD_CUDA(p_cuMemcpyDtoH, "cuMemcpyDtoH_v2");
  LOAD_CUDA(p_cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
  LOAD_CUDA(p_cuModuleLoadData, "cuModuleLoadData");
  LOAD_CUDA(p_cuModuleGetFunction, "cuModuleGetFunction");
  LOAD_CUDA(p_cuModuleUnload, "cuModuleUnload");
  LOAD_CUDA(p_cuLaunchKernel, "cuLaunchKernel");
  LOAD_CUDA(p_cuGetErrorString, "cuGetErrorString");
  LOAD_CUDA(p_cuEventCreate, "cuEventCreate");
  LOAD_CUDA(p_cuEventRecord, "cuEventRecord");
  LOAD_CUDA(p_cuEventSynchronize, "cuEventSynchronize");
  LOAD_CUDA(p_cuEventElapsedTime, "cuEventElapsedTime");
  LOAD_CUDA(p_cuEventDestroy, "cuEventDestroy_v2");
  LOAD_CUDA(p_cuGraphCreate, "cuGraphCreate");
  LOAD_CUDA(p_cuGraphAddKernelNode, "cuGraphAddKernelNode");
  LOAD_CUDA(p_cuGraphAddMemcpyNode, "cuGraphAddMemcpyNode");
  LOAD_CUDA(p_cuGraphInstantiate, "cuGraphInstantiate_v2");
  LOAD_CUDA(p_cuGraphExecKernelNodeSetParams, "cuGraphExecKernelNodeSetParams");
  LOAD_CUDA(p_cuGraphExecMemcpyNodeSetParams, "cuGraphExecMemcpyNodeSetParams");
  LOAD_CUDA(p_cuGraphLaunch, "cuGraphLaunch");
  LOAD_CUDA(p_cuGraphExecDestroy, "cuGraphExecDestroy");
  LOAD_CUDA(p_cuGraphDestroy, "cuGraphDestroy");
#undef LOAD_CUDA
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

CAMLprim value caml_tolk_cuda_init(value unit) {
  CAMLparam1(unit);
  ensure_cuda();
  cuda_check(p_cuInit(0));
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
  CUcontext ctx = NULL;
  cuda_check(p_cuCtxCreate(&ctx, 0, (CUdevice)Int_val(v_device)));
  CAMLreturn(caml_copy_nativeint((intnat)ctx));
}

CAMLprim value caml_tolk_cuda_ctx_set_current(value v_ctx) {
  CAMLparam1(v_ctx);
  cuda_check(p_cuCtxSetCurrent((CUcontext)Nativeint_val(v_ctx)));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_ctx_synchronize(value unit) {
  CAMLparam1(unit);
  CUresult status;
  caml_release_runtime_system();
  status = p_cuCtxSynchronize();
  caml_acquire_runtime_system();
  cuda_check(status);
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

CAMLprim value caml_tolk_cuda_host_write(value v_host, value v_bytes) {
  CAMLparam2(v_host, v_bytes);
  memcpy((void *)Nativeint_val(v_host), Bytes_val(v_bytes),
         caml_string_length(v_bytes));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_memcpy_htod_async(value v_dst, value v_src,
                                                value v_size) {
  CAMLparam3(v_dst, v_src, v_size);
  cuda_check(p_cuMemcpyHtoDAsync((CUdeviceptr)Nativeint_val(v_dst),
                                 (const void *)Nativeint_val(v_src),
                                 (size_t)Long_val(v_size), NULL));
  CAMLreturn(Val_unit);
}

/* Device-to-host copy into raw (pinned) host memory. The destination is not
   OCaml-managed, so the runtime lock is released for the duration of the
   blocking copy. */
CAMLprim value caml_tolk_cuda_memcpy_dtoh_ptr(value v_dst, value v_src,
                                              value v_size) {
  CAMLparam3(v_dst, v_src, v_size);
  void *dst = (void *)Nativeint_val(v_dst);
  CUdeviceptr src = (CUdeviceptr)Nativeint_val(v_src);
  size_t size = (size_t)Long_val(v_size);
  CUresult status;
  caml_release_runtime_system();
  status = p_cuMemcpyDtoH(dst, src, size);
  caml_acquire_runtime_system();
  cuda_check(status);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_host_read(value v_bytes, value v_host) {
  CAMLparam2(v_bytes, v_host);
  memcpy(Bytes_val(v_bytes), (const void *)Nativeint_val(v_host),
         caml_string_length(v_bytes));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_memcpy_dtod_async(value v_dst, value v_src,
                                                value v_size) {
  CAMLparam3(v_dst, v_src, v_size);
  cuda_check(p_cuMemcpyDtoDAsync((CUdeviceptr)Nativeint_val(v_dst),
                                 (CUdeviceptr)Nativeint_val(v_src),
                                 (size_t)Long_val(v_size), NULL));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_module_load(value v_lib) {
  CAMLparam1(v_lib);
  CUmodule module = NULL;
  /* OCaml strings carry a terminating NUL byte, so the PTX image is always
     NUL-terminated as cuModuleLoadData requires. */
  cuda_check(p_cuModuleLoadData(&module, String_val(v_lib)));
  CAMLreturn(caml_copy_nativeint((intnat)module));
}

CAMLprim value caml_tolk_cuda_module_get_function(value v_module,
                                                  value v_name) {
  CAMLparam2(v_module, v_name);
  CUfunction func = NULL;
  cuda_check(p_cuModuleGetFunction(&func, (CUmodule)Nativeint_val(v_module),
                                   String_val(v_name)));
  CAMLreturn(caml_copy_nativeint((intnat)func));
}

CAMLprim value caml_tolk_cuda_module_unload(value v_module) {
  CAMLparam1(v_module);
  cuda_check(p_cuModuleUnload((CUmodule)Nativeint_val(v_module)));
  CAMLreturn(Val_unit);
}

/* Launch a kernel. Arguments are encoded as a single parameter buffer of
   packed 8-byte device pointers followed by 4-byte int32 values, passed via
   CU_LAUNCH_PARAM_BUFFER_POINTER. When [wait] is true, the launch is timed
   with a pair of events and the elapsed GPU time in seconds is returned. */
CAMLprim value caml_tolk_cuda_launch_kernel(value v_func, value v_bufs,
                                            value v_vals, value v_global,
                                            value v_local, value v_wait) {
  CAMLparam5(v_func, v_bufs, v_vals, v_global, v_local);
  CAMLxparam1(v_wait);
  CAMLlocal2(v_time, v_some);
  CUfunction func = (CUfunction)Nativeint_val(v_func);
  mlsize_t nbufs = Wosize_val(v_bufs);
  mlsize_t nvals = Wosize_val(v_vals);
  if (Wosize_val(v_global) != 3 || Wosize_val(v_local) != 3)
    caml_failwith("CUDA launch expects 3D sizes");
  unsigned int gx = (unsigned int)Long_val(Field(v_global, 0));
  unsigned int gy = (unsigned int)Long_val(Field(v_global, 1));
  unsigned int gz = (unsigned int)Long_val(Field(v_global, 2));
  unsigned int lx = (unsigned int)Long_val(Field(v_local, 0));
  unsigned int ly = (unsigned int)Long_val(Field(v_local, 1));
  unsigned int lz = (unsigned int)Long_val(Field(v_local, 2));
  int wait = Bool_val(v_wait);

  size_t args_size = nbufs * 8 + nvals * 4;
  char *c_args = (char *)malloc(args_size > 0 ? args_size : 1);
  if (c_args == NULL) caml_failwith("CUDA kernel argument allocation failed");
  for (mlsize_t i = 0; i < nbufs; ++i) {
    CUdeviceptr ptr = (CUdeviceptr)Nativeint_val(Field(v_bufs, i));
    memcpy(c_args + i * 8, &ptr, 8);
  }
  for (mlsize_t i = 0; i < nvals; ++i) {
    int32_t val = (int32_t)Long_val(Field(v_vals, i));
    memcpy(c_args + nbufs * 8 + i * 4, &val, 4);
  }
  void *config[5] = {CU_LAUNCH_PARAM_BUFFER_POINTER, c_args,
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &args_size,
                     CU_LAUNCH_PARAM_END};

  CUevent start = NULL, stop = NULL;
  CUresult status = 0;
  if (wait) {
    status = p_cuEventCreate(&start, 0);
    if (status == 0) status = p_cuEventCreate(&stop, 0);
    if (status == 0) status = p_cuEventRecord(start, NULL);
  }
  if (status == 0)
    status = p_cuLaunchKernel(func, gx, gy, gz, lx, ly, lz, 0, NULL, NULL,
                              config);
  if (wait && status == 0) status = p_cuEventRecord(stop, NULL);
  free(c_args);

  float elapsed_ms = 0.0f;
  if (wait && status == 0) {
    caml_release_runtime_system();
    status = p_cuEventSynchronize(stop);
    caml_acquire_runtime_system();
    if (status == 0) status = p_cuEventElapsedTime(&elapsed_ms, start, stop);
  }
  if (start != NULL) p_cuEventDestroy(start);
  if (stop != NULL) p_cuEventDestroy(stop);
  cuda_check(status);

  if (!wait) CAMLreturn(Val_none);
  v_time = caml_copy_double((double)elapsed_ms * 1e-3);
  v_some = caml_alloc_some(v_time);
  CAMLreturn(v_some);
}

CAMLprim value caml_tolk_cuda_launch_kernel_bc(value *argv, int argc) {
  (void)argc;
  return caml_tolk_cuda_launch_kernel(argv[0], argv[1], argv[2], argv[3],
                                      argv[4], argv[5]);
}

/* Execution graphs. A graph is built once from a fixed sequence of kernel
   and memcpy nodes, instantiated, and relaunched with a single call. Each
   node keeps its parameter structs in C memory so replays can patch buffer
   addresses, scalar values, and launch dimensions before committing them
   with cuGraphExec*NodeSetParams. The node array is allocated at its final
   capacity up front: kernel nodes point into their own struct (extra ->
   config -> args), so the array must never be reallocated. */

typedef struct {
  CUgraphNode node;
  int is_copy;
  CUDA_KERNEL_NODE_PARAMS_v1 kparams;
  CUDA_MEMCPY3D_v2 cparams;
  CUcontext copy_ctx;
  char *args;
  size_t args_size;
  size_t nbufs;
  size_t nvals;
  void *config[5];
} tolk_graph_node;

typedef struct {
  CUgraph graph;
  CUgraphExec exec;
  int n;
  int cap;
  tolk_graph_node *nodes;
} tolk_graph;

static tolk_graph *graph_val(value v) { return (tolk_graph *)Nativeint_val(v); }

static tolk_graph_node *graph_node(value v_graph, value v_node) {
  tolk_graph *g = graph_val(v_graph);
  int i = Int_val(v_node);
  if (i < 0 || i >= g->n) caml_failwith("CUDA graph node index out of range");
  return &g->nodes[i];
}

/* Collect dependency handles into [deps]; returns the count. */
static size_t graph_deps(tolk_graph *g, value v_deps, CUgraphNode *deps) {
  mlsize_t ndeps = Wosize_val(v_deps);
  for (mlsize_t i = 0; i < ndeps; ++i) {
    long d = Long_val(Field(v_deps, i));
    if (d < 0 || d >= g->n) caml_failwith("CUDA graph dependency out of range");
    deps[i] = g->nodes[d].node;
  }
  return (size_t)ndeps;
}

CAMLprim value caml_tolk_cuda_graph_create(value v_count) {
  CAMLparam1(v_count);
  long count = Long_val(v_count);
  if (count < 0) caml_failwith("CUDA graph node count must be non-negative");
  tolk_graph *g = (tolk_graph *)calloc(1, sizeof(tolk_graph));
  if (g == NULL) caml_failwith("CUDA graph allocation failed");
  g->cap = (int)count;
  g->nodes = (tolk_graph_node *)calloc(count > 0 ? count : 1,
                                       sizeof(tolk_graph_node));
  if (g->nodes == NULL) {
    free(g);
    caml_failwith("CUDA graph allocation failed");
  }
  CUresult status = p_cuGraphCreate(&g->graph, 0);
  if (status != 0) {
    free(g->nodes);
    free(g);
    cuda_check(status);
  }
  CAMLreturn(caml_copy_nativeint((intnat)g));
}

CAMLprim value caml_tolk_cuda_graph_add_kernel(value v_graph, value v_func,
                                               value v_global, value v_local,
                                               value v_bufs, value v_vals,
                                               value v_deps) {
  CAMLparam5(v_graph, v_func, v_global, v_local, v_bufs);
  CAMLxparam2(v_vals, v_deps);
  tolk_graph *g = graph_val(v_graph);
  if (g->n >= g->cap) caml_failwith("CUDA graph node capacity exceeded");
  if (Wosize_val(v_global) != 3 || Wosize_val(v_local) != 3)
    caml_failwith("CUDA graph kernel expects 3D sizes");
  tolk_graph_node *node = &g->nodes[g->n];
  node->is_copy = 0;
  node->nbufs = Wosize_val(v_bufs);
  node->nvals = Wosize_val(v_vals);
  node->args_size = node->nbufs * 8 + node->nvals * 4;
  node->args = (char *)malloc(node->args_size > 0 ? node->args_size : 1);
  if (node->args == NULL)
    caml_failwith("CUDA graph kernel argument allocation failed");
  for (mlsize_t i = 0; i < node->nbufs; ++i) {
    CUdeviceptr ptr = (CUdeviceptr)Nativeint_val(Field(v_bufs, i));
    memcpy(node->args + i * 8, &ptr, 8);
  }
  for (mlsize_t i = 0; i < node->nvals; ++i) {
    int32_t val = (int32_t)Long_val(Field(v_vals, i));
    memcpy(node->args + node->nbufs * 8 + i * 4, &val, 4);
  }
  node->config[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
  node->config[1] = node->args;
  node->config[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
  node->config[3] = &node->args_size;
  node->config[4] = CU_LAUNCH_PARAM_END;
  node->kparams.func = (CUfunction)Nativeint_val(v_func);
  node->kparams.gridDimX = (unsigned int)Long_val(Field(v_global, 0));
  node->kparams.gridDimY = (unsigned int)Long_val(Field(v_global, 1));
  node->kparams.gridDimZ = (unsigned int)Long_val(Field(v_global, 2));
  node->kparams.blockDimX = (unsigned int)Long_val(Field(v_local, 0));
  node->kparams.blockDimY = (unsigned int)Long_val(Field(v_local, 1));
  node->kparams.blockDimZ = (unsigned int)Long_val(Field(v_local, 2));
  node->kparams.sharedMemBytes = 0;
  node->kparams.kernelParams = NULL;
  node->kparams.extra = node->config;
  CUgraphNode deps[Wosize_val(v_deps) + 1];
  size_t ndeps = graph_deps(g, v_deps, deps);
  CUresult status = p_cuGraphAddKernelNode(&node->node, g->graph,
                                           ndeps > 0 ? deps : NULL, ndeps,
                                           &node->kparams);
  if (status != 0) {
    free(node->args);
    node->args = NULL;
    cuda_check(status);
  }
  g->n += 1;
  CAMLreturn(Val_int(g->n - 1));
}

CAMLprim value caml_tolk_cuda_graph_add_kernel_bc(value *argv, int argc) {
  (void)argc;
  return caml_tolk_cuda_graph_add_kernel(argv[0], argv[1], argv[2], argv[3],
                                         argv[4], argv[5], argv[6]);
}

CAMLprim value caml_tolk_cuda_graph_add_copy(value v_graph, value v_ctx,
                                             value v_dest, value v_src,
                                             value v_nbytes, value v_deps) {
  CAMLparam5(v_graph, v_ctx, v_dest, v_src, v_nbytes);
  CAMLxparam1(v_deps);
  tolk_graph *g = graph_val(v_graph);
  if (g->n >= g->cap) caml_failwith("CUDA graph node capacity exceeded");
  tolk_graph_node *node = &g->nodes[g->n];
  size_t nbytes = (size_t)Long_val(v_nbytes);
  node->is_copy = 1;
  node->copy_ctx = (CUcontext)Nativeint_val(v_ctx);
  node->cparams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  node->cparams.srcDevice = (CUdeviceptr)Nativeint_val(v_src);
  node->cparams.srcPitch = nbytes;
  node->cparams.srcHeight = 1;
  node->cparams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  node->cparams.dstDevice = (CUdeviceptr)Nativeint_val(v_dest);
  node->cparams.dstPitch = nbytes;
  node->cparams.dstHeight = 1;
  node->cparams.WidthInBytes = nbytes;
  node->cparams.Height = 1;
  node->cparams.Depth = 1;
  CUgraphNode deps[Wosize_val(v_deps) + 1];
  size_t ndeps = graph_deps(g, v_deps, deps);
  cuda_check(p_cuGraphAddMemcpyNode(&node->node, g->graph,
                                    ndeps > 0 ? deps : NULL, ndeps,
                                    &node->cparams, node->copy_ctx));
  g->n += 1;
  CAMLreturn(Val_int(g->n - 1));
}

CAMLprim value caml_tolk_cuda_graph_add_copy_bc(value *argv, int argc) {
  (void)argc;
  return caml_tolk_cuda_graph_add_copy(argv[0], argv[1], argv[2], argv[3],
                                       argv[4], argv[5]);
}

CAMLprim value caml_tolk_cuda_graph_instantiate(value v_graph) {
  CAMLparam1(v_graph);
  tolk_graph *g = graph_val(v_graph);
  CUresult status;
  caml_release_runtime_system();
  status = p_cuGraphInstantiate(&g->exec, g->graph, NULL, NULL, 0);
  caml_acquire_runtime_system();
  cuda_check(status);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_graph_set_buf(value v_graph, value v_node,
                                            value v_pos, value v_addr) {
  CAMLparam4(v_graph, v_node, v_pos, v_addr);
  tolk_graph_node *node = graph_node(v_graph, v_node);
  CUdeviceptr addr = (CUdeviceptr)Nativeint_val(v_addr);
  long pos = Long_val(v_pos);
  if (node->is_copy) {
    if (pos == 1)
      node->cparams.srcDevice = addr;
    else
      node->cparams.dstDevice = addr;
  } else {
    if (pos < 0 || (size_t)pos >= node->nbufs)
      caml_failwith("CUDA graph buffer position out of range");
    memcpy(node->args + pos * 8, &addr, 8);
  }
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_graph_set_val(value v_graph, value v_node,
                                            value v_idx, value v_val) {
  CAMLparam4(v_graph, v_node, v_idx, v_val);
  tolk_graph_node *node = graph_node(v_graph, v_node);
  long idx = Long_val(v_idx);
  if (node->is_copy || idx < 0 || (size_t)idx >= node->nvals)
    caml_failwith("CUDA graph value index out of range");
  int32_t val = (int32_t)Long_val(v_val);
  memcpy(node->args + node->nbufs * 8 + idx * 4, &val, 4);
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_graph_set_launch(value v_graph, value v_node,
                                               value v_global, value v_local) {
  CAMLparam4(v_graph, v_node, v_global, v_local);
  tolk_graph_node *node = graph_node(v_graph, v_node);
  if (node->is_copy)
    caml_failwith("CUDA graph launch dims on a memcpy node");
  if (Wosize_val(v_global) != 3 || Wosize_val(v_local) != 3)
    caml_failwith("CUDA graph launch expects 3D sizes");
  node->kparams.gridDimX = (unsigned int)Long_val(Field(v_global, 0));
  node->kparams.gridDimY = (unsigned int)Long_val(Field(v_global, 1));
  node->kparams.gridDimZ = (unsigned int)Long_val(Field(v_global, 2));
  node->kparams.blockDimX = (unsigned int)Long_val(Field(v_local, 0));
  node->kparams.blockDimY = (unsigned int)Long_val(Field(v_local, 1));
  node->kparams.blockDimZ = (unsigned int)Long_val(Field(v_local, 2));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_cuda_graph_set_params(value v_graph, value v_node) {
  CAMLparam2(v_graph, v_node);
  tolk_graph *g = graph_val(v_graph);
  tolk_graph_node *node = graph_node(v_graph, v_node);
  if (g->exec == NULL) caml_failwith("CUDA graph is not instantiated");
  if (node->is_copy)
    cuda_check(p_cuGraphExecMemcpyNodeSetParams(g->exec, node->node,
                                                &node->cparams,
                                                node->copy_ctx));
  else
    cuda_check(p_cuGraphExecKernelNodeSetParams(g->exec, node->node,
                                                &node->kparams));
  CAMLreturn(Val_unit);
}

/* Launch the instantiated graph. When [wait] is true, the launch is timed
   with a pair of events and the elapsed GPU time in seconds is returned. */
CAMLprim value caml_tolk_cuda_graph_launch(value v_graph, value v_wait) {
  CAMLparam2(v_graph, v_wait);
  CAMLlocal2(v_time, v_some);
  tolk_graph *g = graph_val(v_graph);
  int wait = Bool_val(v_wait);
  if (g->exec == NULL) caml_failwith("CUDA graph is not instantiated");

  CUevent start = NULL, stop = NULL;
  CUresult status = 0;
  if (wait) {
    status = p_cuEventCreate(&start, 0);
    if (status == 0) status = p_cuEventCreate(&stop, 0);
    if (status == 0) status = p_cuEventRecord(start, NULL);
  }
  if (status == 0) status = p_cuGraphLaunch(g->exec, NULL);
  if (wait && status == 0) status = p_cuEventRecord(stop, NULL);

  float elapsed_ms = 0.0f;
  if (wait && status == 0) {
    caml_release_runtime_system();
    status = p_cuEventSynchronize(stop);
    caml_acquire_runtime_system();
    if (status == 0) status = p_cuEventElapsedTime(&elapsed_ms, start, stop);
  }
  if (start != NULL) p_cuEventDestroy(start);
  if (stop != NULL) p_cuEventDestroy(stop);
  cuda_check(status);

  if (!wait) CAMLreturn(Val_none);
  v_time = caml_copy_double((double)elapsed_ms * 1e-3);
  v_some = caml_alloc_some(v_time);
  CAMLreturn(v_some);
}

CAMLprim value caml_tolk_cuda_graph_destroy(value v_graph) {
  CAMLparam1(v_graph);
  tolk_graph *g = graph_val(v_graph);
  if (g->exec != NULL) p_cuGraphExecDestroy(g->exec);
  if (g->graph != NULL) p_cuGraphDestroy(g->graph);
  for (int i = 0; i < g->n; ++i) free(g->nodes[i].args);
  free(g->nodes);
  free(g);
  CAMLreturn(Val_unit);
}

/* NVRTC */

typedef int nvrtcResult;
typedef struct _nvrtcProgram *nvrtcProgram;

static nvrtcResult (*p_nvrtcVersion)(int *, int *);
static nvrtcResult (*p_nvrtcCreateProgram)(nvrtcProgram *, const char *,
                                           const char *, int,
                                           const char *const *,
                                           const char *const *);
static nvrtcResult (*p_nvrtcCompileProgram)(nvrtcProgram, int,
                                            const char *const *);
static nvrtcResult (*p_nvrtcGetPTXSize)(nvrtcProgram, size_t *);
static nvrtcResult (*p_nvrtcGetPTX)(nvrtcProgram, char *);
static nvrtcResult (*p_nvrtcGetProgramLogSize)(nvrtcProgram, size_t *);
static nvrtcResult (*p_nvrtcGetProgramLog)(nvrtcProgram, char *);
static const char *(*p_nvrtcGetErrorString)(nvrtcResult);
static nvrtcResult (*p_nvrtcDestroyProgram)(nvrtcProgram *);

static void *nvrtc_handle = NULL;

static void ensure_nvrtc(void) {
  static const char *names[] = {"libnvrtc.so", "libnvrtc.so.13",
                                "libnvrtc.so.12",
                                "/usr/local/cuda/lib64/libnvrtc.so", NULL};
  if (nvrtc_handle != NULL) return;
  for (int i = 0; nvrtc_handle == NULL && names[i] != NULL; ++i)
    nvrtc_handle = dlopen(names[i], RTLD_LAZY | RTLD_LOCAL);
  if (nvrtc_handle == NULL)
    caml_failwith("NVRTC library (libnvrtc.so) not found");
#define LOAD_NVRTC(var, name)                                    \
  do {                                                           \
    var = dlsym(nvrtc_handle, name);                             \
    if (var == NULL) caml_failwith("NVRTC is missing " name);    \
  } while (0)
  LOAD_NVRTC(p_nvrtcVersion, "nvrtcVersion");
  LOAD_NVRTC(p_nvrtcCreateProgram, "nvrtcCreateProgram");
  LOAD_NVRTC(p_nvrtcCompileProgram, "nvrtcCompileProgram");
  LOAD_NVRTC(p_nvrtcGetPTXSize, "nvrtcGetPTXSize");
  LOAD_NVRTC(p_nvrtcGetPTX, "nvrtcGetPTX");
  LOAD_NVRTC(p_nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize");
  LOAD_NVRTC(p_nvrtcGetProgramLog, "nvrtcGetProgramLog");
  LOAD_NVRTC(p_nvrtcGetErrorString, "nvrtcGetErrorString");
  LOAD_NVRTC(p_nvrtcDestroyProgram, "nvrtcDestroyProgram");
#undef LOAD_NVRTC
}

CAMLprim value caml_tolk_cuda_nvrtc_version(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(v_pair);
  int major = 0, minor = 0;
  nvrtcResult status;
  ensure_nvrtc();
  status = p_nvrtcVersion(&major, &minor);
  if (status != 0) {
    char buf[256];
    snprintf(buf, sizeof(buf), "Nvrtc Error %d, %s", status,
             p_nvrtcGetErrorString(status));
    caml_failwith(buf);
  }
  v_pair = caml_alloc_tuple(2);
  Store_field(v_pair, 0, Val_int(major));
  Store_field(v_pair, 1, Val_int(minor));
  CAMLreturn(v_pair);
}

/* Compile CUDA C source to PTX. Returns [Ok ptx] or [Error message] where the
   message carries the NVRTC error string and the program log. The source and
   options are copied to C memory so the OCaml runtime lock can be released
   during compilation. */
CAMLprim value caml_tolk_cuda_nvrtc_compile(value v_src, value v_opts) {
  CAMLparam2(v_src, v_opts);
  CAMLlocal3(v_result, v_payload, v_msg);
  ensure_nvrtc();

  size_t src_len = caml_string_length(v_src);
  char *src = (char *)malloc(src_len + 1);
  if (src == NULL) caml_failwith("NVRTC source allocation failed");
  memcpy(src, String_val(v_src), src_len + 1);

  mlsize_t nopts = Wosize_val(v_opts);
  char **opts = (char **)calloc(nopts > 0 ? nopts : 1, sizeof(char *));
  if (opts == NULL) {
    free(src);
    caml_failwith("NVRTC options allocation failed");
  }
  for (mlsize_t i = 0; i < nopts; ++i) {
    opts[i] = strdup(String_val(Field(v_opts, i)));
    if (opts[i] == NULL) {
      for (mlsize_t j = 0; j < i; ++j) free(opts[j]);
      free(opts);
      free(src);
      caml_failwith("NVRTC options allocation failed");
    }
  }

  nvrtcProgram prog = NULL;
  nvrtcResult status;
  char *log = NULL;
  char *ptx = NULL;
  size_t ptx_size = 0;

  caml_release_runtime_system();
  status = p_nvrtcCreateProgram(&prog, src, "<null>", 0, NULL, NULL);
  if (status == 0) {
    status = p_nvrtcCompileProgram(prog, (int)nopts, (const char *const *)opts);
    if (status != 0) {
      size_t log_size = 0;
      if (p_nvrtcGetProgramLogSize(prog, &log_size) == 0 && log_size > 0 &&
          (log = (char *)malloc(log_size + 1)) != NULL) {
        if (p_nvrtcGetProgramLog(prog, log) == 0)
          log[log_size] = '\0';
        else {
          free(log);
          log = NULL;
        }
      }
    } else {
      status = p_nvrtcGetPTXSize(prog, &ptx_size);
      if (status == 0) {
        ptx = (char *)malloc(ptx_size > 0 ? ptx_size : 1);
        if (ptx != NULL)
          status = p_nvrtcGetPTX(prog, ptx);
        else
          status = -1;
      }
    }
    p_nvrtcDestroyProgram(&prog);
  }
  caml_acquire_runtime_system();

  for (mlsize_t i = 0; i < nopts; ++i) free(opts[i]);
  free(opts);
  free(src);

  if (status != 0 || ptx == NULL) {
    const char *error =
        status != 0 ? p_nvrtcGetErrorString(status) : "out of memory";
    size_t msg_size =
        strlen(error) + (log != NULL ? strlen(log) : 0) + 64;
    char *msg = (char *)malloc(msg_size);
    if (msg != NULL)
      snprintf(msg, msg_size, "Nvrtc Error %d, %s\n%s", status, error,
               log != NULL ? log : "");
    free(log);
    free(ptx);
    v_msg = caml_copy_string(msg != NULL ? msg : "Nvrtc Error");
    free(msg);
    v_result = caml_alloc(1, 1); /* Error */
    Store_field(v_result, 0, v_msg);
    CAMLreturn(v_result);
  }

  free(log);
  v_payload = caml_alloc_string(ptx_size);
  memcpy(Bytes_val(v_payload), ptx, ptx_size);
  free(ptx);
  v_result = caml_alloc(1, 0); /* Ok */
  Store_field(v_result, 0, v_payload);
  CAMLreturn(v_result);
}
