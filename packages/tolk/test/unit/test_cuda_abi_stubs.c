/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Exercise the production stubs with the CUDA driver boundary replaced.
   This test executable does not link tolk.cuda, so these entry points have
   one definition and the injected function table cannot affect a device. */
#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Replace dynamic loading before including the production stubs. */
#define TOLK_DL_H
static void *tolk_dlopen(const char *name);
static void *tolk_dlsym(void *handle, const char *name);
#include "tolk_cuda_stubs.c"

static int init_calls, symbol_calls;
static CUresult fake_init(unsigned flags) { assert(flags == 0); init_calls++; return 0; }
static CUresult fake_unused(void) { return 0; }
static void *tolk_dlopen(const char *name) { (void)name; return (void *)1; }
static void *tolk_dlsym(void *handle, const char *name) {
  assert(handle == (void *)1);
  symbol_calls++;
  if (getenv("TOLK_TEST_CUDA_MISSING") && strcmp(name, "cuMemAlloc_v2") == 0) return NULL;
  return strcmp(name, "cuInit") == 0 ? (void *)fake_init : (void *)fake_unused;
}
CAMLprim value caml_test_cuda_init_counts(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(counts);
  counts = caml_alloc_tuple(2);
  Store_field(counts, 0, Val_int(init_calls));
  Store_field(counts, 1, Val_int(symbol_calls));
  CAMLreturn(counts);
}

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

static int destroy_steps, fail_sync;
static CUresult fake_synchronize(void) { destroy_steps++; return fail_sync ? 719 : 0; }
static CUresult fake_event_destroy(CUevent event) {
  assert(event == (CUevent)0x3); destroy_steps++; return 0;
}
static CUresult fake_stream_destroy(CUstream stream) {
  assert(stream == (CUstream)0x1 || stream == (CUstream)0x2);
  destroy_steps++; return 0;
}
static CUresult fake_context_destroy(CUcontext context) {
  assert(context == (CUcontext)0x4); destroy_steps++; return 0;
}
static CUresult fake_error(CUresult status, const char **message) {
  (void)status; *message = "injected synchronization failure"; return 0;
}
CAMLprim value caml_test_cuda_shutdown_setup(value failure) {
  CAMLparam1(failure);
  CAMLlocal1(result);
  result = caml_copy_nativeint(0);
  tolk_cuda_queue *q = calloc(1, sizeof(*q));
  if (q == NULL) caml_raise_out_of_memory();
  assert(pthread_mutex_init(&q->lock, NULL) == 0);
  q->context = (CUcontext)0x4;
  q->streams[0] = (CUstream)0x1;
  q->streams[1] = (CUstream)0x2;
  q->handoff = (CUevent)0x3;
  destroy_steps = 0;
  fail_sync = Bool_val(failure);
  p_cuCtxSetCurrent = fake_context;
  p_cuCtxSynchronize = fake_synchronize;
  p_cuCtxDestroy = fake_context_destroy;
  p_cuEventDestroy = fake_event_destroy;
  p_cuStreamDestroy = fake_stream_destroy;
  p_cuGetErrorString = fake_error;
  Nativeint_val(result) = (intnat)q;
  CAMLreturn(result);
}
CAMLprim value caml_test_cuda_shutdown_steps(value unit) {
  (void)unit;
  return Val_int(destroy_steps);
}

static unsigned stamp_callbacks;
static CUresult fake_host_stamp(CUstream stream, void (*callback)(void *), void *data) {
  assert(stream == (CUstream)0x12);
  stamp_callbacks++;
  callback(data);
  return 0;
}
CAMLprim value caml_test_cuda_timestamp(value unit) {
  CAMLparam1(unit);
  uint64_t stamp = 0;
  queue.streams[1] = (CUstream)0x12;
  queue.status = 0;
  stamp_callbacks = 0;
  p_cuLaunchHostFunc = fake_host_stamp;
  tolk_cuda_hcq_timestamp(&queue, 1, (uint64_t)(uintptr_t)&stamp);
  assert(stamp_callbacks == 1);
  assert(stamp > 0);
  queue.status = 719;
  tolk_cuda_hcq_timestamp(&queue, 1, (uint64_t)(uintptr_t)&stamp);
  assert(stamp_callbacks == 1);
  queue.status = 0;
  CAMLreturn(caml_copy_int64((int64_t)stamp));
}

static CUresult registration_status;
static CUresult fake_register(void *ptr, size_t size, unsigned flags) {
  assert(ptr == (void *)0x10000); assert(size == 16); assert(flags == 0);
  return registration_status;
}
CAMLprim value caml_test_cuda_registration_status(value status) {
  CAMLparam1(status);
  registration_status = Int_val(status);
  p_cuMemHostRegister = fake_register;
  p_cuGetErrorString = fake_error;
  CAMLreturn(Val_unit);
}
