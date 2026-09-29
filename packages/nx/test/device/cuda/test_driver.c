/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The driver calls that a submitter and other CUDA libraries make, through
   the driver the runtime loaded. Each pushes the context it uses and pops it
   before returning. */

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <stdint.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#define CUDAAPI __stdcall
static void *symbol(const char *name) {
  HMODULE lib = LoadLibraryA("nvcuda.dll");
  void *s = lib ? (void *)GetProcAddress(lib, name) : NULL;
  if (s == NULL) caml_failwith(name);
  return s;
}
#else
#include <dlfcn.h>
#define CUDAAPI
static void *symbol(const char *name) {
#ifdef __APPLE__
  void *lib = dlopen("libcuda.dylib", RTLD_NOW | RTLD_LOCAL);
#else
  void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
#endif
  void *s = lib ? dlsym(lib, name) : NULL;
  if (s == NULL) caml_failwith(name);
  return s;
}
#endif

#define Ptr_val(v) ((void *)Nativeint_val(v))

static void check(int status, const char *what) {
  char buf[128];
  if (status != 0) {
    snprintf(buf, sizeof(buf), "%s: CUDA error %d", what, status);
    caml_failwith(buf);
  }
}

static void push(void *ctx) {
  int(CUDAAPI * f)(void *) = symbol("cuCtxPushCurrent_v2");
  check(f(ctx), "cuCtxPushCurrent_v2");
}

static void pop(void) {
  int(CUDAAPI * f)(void **) = symbol("cuCtxPopCurrent_v2");
  void *ctx;
  check(f(&ctx), "cuCtxPopCurrent_v2");
}

/* Enqueues on [stream] a wait for the signal to reach [wait_for], a launch of
   [function] over 256 threads with the one argument [out] if [function] is
   not null, then the signal of [v]. */
static void submit(value ctx, value stream, value signal, uint64_t wait_for,
                   void *function, uint64_t out, uint64_t v) {
  int(CUDAAPI * wait)(void *, uint64_t, uint64_t, unsigned) =
      symbol("cuStreamWaitValue64_v2");
  int(CUDAAPI * write)(void *, uint64_t, uint64_t, unsigned) =
      symbol("cuStreamWriteValue64_v2");
  int(CUDAAPI * launch)(void *, unsigned, unsigned, unsigned, unsigned,
                        unsigned, unsigned, unsigned, void *, void **,
                        void **) = symbol("cuLaunchKernel");
  uint64_t word = (uint64_t)Nativeint_val(signal);
  void *params[] = {&out};
  push(Ptr_val(ctx));
  int status = wait(Ptr_val(stream), word, wait_for, 0);
  if (status == 0 && function != NULL)
    status = launch(function, 1, 1, 1, 256, 1, 1, 0, Ptr_val(stream), params,
                    NULL);
  if (status == 0) status = write(Ptr_val(stream), word, v, 0);
  pop();
  check(status, "submission");
}

/* As a submitter does: timeline work [v] that launches [function]. */
value test_launch(value ctx, value stream, value signal, value function,
                  value out, value v) {
  submit(ctx, stream, signal, (uint64_t)Long_val(v) - 1, Ptr_val(function),
         (uint64_t)Nativeint_val(out), (uint64_t)Long_val(v));
  return Val_unit;
}

value test_launch_byte(value *argv, int argn) {
  (void)argn;
  return test_launch(argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
}

/* Timeline work [v] that waits for a value past it, which never arrives. */
value test_stall(value ctx, value stream, value signal, value v) {
  submit(ctx, stream, signal, (uint64_t)Long_val(v) + 1, NULL, 0,
         (uint64_t)Long_val(v));
  return Val_unit;
}

/* As another library does: [size] bytes of device memory on [ctx]. */
value test_alloc(value ctx, value size) {
  CAMLparam2(ctx, size);
  int(CUDAAPI * alloc)(uint64_t *, size_t) = symbol("cuMemAlloc_v2");
  uint64_t d = 0;
  push(Ptr_val(ctx));
  int status = alloc(&d, Long_val(size));
  pop();
  check(status, "cuMemAlloc_v2");
  CAMLreturn(caml_copy_nativeint((intnat)d));
}

/* [(host, device)] addresses of [size] bytes of page-locked memory made on
   [ctx]. */
value test_host_alloc(value ctx, value size) {
  CAMLparam2(ctx, size);
  CAMLlocal2(r, v);
  int(CUDAAPI * alloc)(void **, size_t, unsigned) = symbol("cuMemHostAlloc");
  int(CUDAAPI * device)(uint64_t *, void *, unsigned) =
      symbol("cuMemHostGetDevicePointer_v2");
  void *p = NULL;
  uint64_t d = 0;
  push(Ptr_val(ctx));
  int status = alloc(&p, Long_val(size), 3);
  if (status == 0) status = device(&d, p, 0);
  pop();
  check(status, "cuMemHostAlloc");
  r = caml_alloc_tuple(2);
  v = caml_copy_nativeint((intnat)p);
  Store_field(r, 0, v);
  v = caml_copy_nativeint((intnat)d);
  Store_field(r, 1, v);
  CAMLreturn(r);
}

/* [size] bytes of device memory on a context of its own, on GPU [ordinal]. */
value test_other_context_alloc(value ordinal, value size) {
  CAMLparam2(ordinal, size);
  int(CUDAAPI * get)(int *, int) = symbol("cuDeviceGet");
  int(CUDAAPI * create)(void **, unsigned, int) = symbol("cuCtxCreate_v2");
  int(CUDAAPI * alloc)(uint64_t *, size_t) = symbol("cuMemAlloc_v2");
  int dev = 0;
  void *ctx = NULL;
  uint64_t d = 0;
  check(get(&dev, Int_val(ordinal)), "cuDeviceGet");
  check(create(&ctx, 0, dev), "cuCtxCreate_v2");
  int status = alloc(&d, Long_val(size));
  pop();
  check(status, "cuMemAlloc_v2");
  CAMLreturn(caml_copy_nativeint((intnat)d));
}
