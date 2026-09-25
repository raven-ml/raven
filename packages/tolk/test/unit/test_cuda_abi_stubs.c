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
static tolk_cuda_queue queue = {.lock = PTHREAD_MUTEX_INITIALIZER};
static unsigned handoffs;
static CUresult fake_context(CUcontext ctx) { (void)ctx; return 0; }
static CUresult fake_record(CUevent event, CUstream stream) {
  (void)event; (void)stream; handoffs++; return 0;
}
static CUresult fake_wait(CUstream stream, CUevent event, unsigned flags) {
  (void)stream; (void)event; (void)flags; return 0;
}
CAMLprim value caml_test_cuda_abi_setup(value unit) {
  CAMLparam1(unit);
  p_cuModuleGetFunction = fake_get;
  p_cuLaunchKernel = fake_launch;
  p_cuCtxSetCurrent = fake_context;
  p_cuEventRecord = fake_record;
  p_cuStreamWaitEvent = fake_wait;
  queue.status = 0;
  queue.direct_pending = queue.queue_pending = 0;
  handoffs = 0;
  CAMLreturn(caml_copy_nativeint((intnat)&queue));
}
CAMLprim value caml_test_cuda_abi_submit(value function, value arguments) {
  CAMLparam2(function, arguments);
  tolk_cuda_hcq_begin(&queue);
  tolk_cuda_hcq_launch(&queue, (CUfunction)Nativeint_val(function),
      1, 1, 1, 1, 1, 1, Bytes_val(arguments), caml_string_length(arguments));
  cuda_check(queue.status);
  CAMLreturn(Val_unit);
}
CAMLprim value caml_test_cuda_abi_handoffs(value unit) {
  (void)unit;
  return Val_int(handoffs);
}
CAMLprim value caml_test_cuda_abi_captured(value unit) {
  CAMLparam1(unit);
  CAMLreturn(caml_alloc_initialized_string(captured_size, captured));
}
