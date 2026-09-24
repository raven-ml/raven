/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Exercise the production stubs with the CUDA driver boundary replaced.
   This test executable does not link tolk.cuda, so these entry points have
   one definition and the injected function table cannot affect a device. */
#include "tolk_cuda_stubs.c"
#include <assert.h>

static char captured[256];
static size_t captured_size;
static void capture(void **extra) {
  assert(extra[0] == CU_LAUNCH_PARAM_BUFFER_POINTER);
  assert(extra[2] == CU_LAUNCH_PARAM_BUFFER_SIZE);
  captured_size = *(size_t *)extra[3];
  assert(captured_size <= sizeof(captured));
  memcpy(captured, extra[1], captured_size);
}
static CUresult fake_get(CUfunction *f, CUmodule module, const char *name) {
  (void)module; (void)name; *f = (CUfunction)0xcafe; return 0;
}
static CUresult fake_launch(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
    unsigned lx, unsigned ly, unsigned lz, unsigned shared, CUstream stream,
    void **params, void **extra) {
  (void)gx; (void)gy; (void)gz; (void)lx; (void)ly; (void)lz;
  (void)shared; (void)stream; assert(params == NULL); assert(f == (CUfunction)0xcafe);
  capture(extra); return 0;
}
static CUresult fake_graph_create(CUgraph *g, unsigned flags) {
  (void)flags; *g = (CUgraph)1; return 0;
}
static CUresult fake_graph_add(CUgraphNode *n, CUgraph g, const CUgraphNode *deps,
    size_t count, const CUDA_KERNEL_NODE_PARAMS_v1 *p) {
  (void)g; (void)deps; (void)count; *n = (CUgraphNode)2;
  capture(p->extra); return 0;
}
static CUresult fake_instantiate(CUgraphExec *e, CUgraph g, CUgraphNode *n, char *log, size_t size) {
  (void)g; (void)n; (void)log; (void)size; *e = (CUgraphExec)3; return 0;
}
static CUresult fake_set(CUgraphExec e, CUgraphNode n, const CUDA_KERNEL_NODE_PARAMS_v1 *p) {
  (void)e; (void)n; capture(p->extra); return 0;
}
static CUresult fake_graph_destroy(CUgraph g) { (void)g; return 0; }
static CUresult fake_exec_destroy(CUgraphExec e) { (void)e; return 0; }
CAMLprim value caml_test_cuda_abi_setup(value unit) {
  (void)unit;
  p_cuModuleGetFunction = fake_get;
  p_cuLaunchKernel = fake_launch;
  p_cuGraphCreate = fake_graph_create;
  p_cuGraphAddKernelNode = fake_graph_add;
  p_cuGraphInstantiate = fake_instantiate;
  p_cuGraphExecKernelNodeSetParams = fake_set;
  p_cuGraphDestroy = fake_graph_destroy;
  p_cuGraphExecDestroy = fake_exec_destroy;
  return Val_unit;
}
CAMLprim value caml_test_cuda_abi_captured(value unit) {
  CAMLparam1(unit);
  CAMLreturn(caml_alloc_initialized_string(captured_size, captured));
}
