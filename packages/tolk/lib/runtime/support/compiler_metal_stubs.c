/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

/* MTLCompiler, whose two functions are declared here and found in the
   framework at run time. Elsewhere than on macOS, loading it fails. */

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <stdlib.h>
#include <string.h>

static value result(int tag, value v) {
  CAMLparam1(v);
  CAMLlocal1(r);
  r = caml_alloc_small(1, tag);
  Field(r, 0) = v;
  CAMLreturn(r);
}

#define OK(v) result(0, v)
#define ERROR(...) result(1, caml_alloc_sprintf(__VA_ARGS__))

#ifdef __APPLE__
#include <dlfcn.h>
#include <stdint.h>
#include <sys/sysctl.h>

/* The request type that compiles source to a Metal library. It is not
   documented, and the framework has no symbol for it. */
#define REQUEST_TYPE_COMPILE 13

typedef void (^reply_t)(int32_t error, const void *data, size_t size,
                        const char *message);
static void *(*MTLCodeGenServiceCreate)(const char *);
static void (*MTLCodeGenServiceBuildRequest)(void *, void *, int32_t,
                                             const void *, size_t, reply_t);
static void *service;

/* Called once, by the caller's lock: what it sets is only read by compiles
   after it returned. */
value caml_tolk_metal_load(value v_path) {
  CAMLparam1(v_path);
  void *lib = dlopen(String_val(v_path), RTLD_NOW | RTLD_LOCAL);
  if (lib == NULL) CAMLreturn(ERROR("%s", dlerror()));
  if ((MTLCodeGenServiceCreate = dlsym(lib, "MTLCodeGenServiceCreate")) ==
          NULL ||
      (MTLCodeGenServiceBuildRequest =
           dlsym(lib, "MTLCodeGenServiceBuildRequest")) == NULL)
    CAMLreturn(ERROR("%s", dlerror()));
  service = MTLCodeGenServiceCreate("tolk");
  CAMLreturn(OK(Val_unit));
}

value caml_tolk_metal_macos_major(value unit) {
  char version[32];
  size_t size = sizeof version;
  if (sysctlbyname("kern.osproductversion", version, &size, NULL, 0) != 0)
    return Val_int(0);
  return Val_int(atoi(version));
}

/* The build of macOS, as 25D125: MTLCompiler is part of it. */
value caml_tolk_metal_macos_build(value unit) {
  char build[64];
  size_t size = sizeof build;
  if (sysctlbyname("kern.osversion", build, &size, NULL, 0) != 0)
    return caml_copy_string("");
  return caml_copy_string(build);
}

/* The reply comes through the block, before the request returns. Compiling
   runs without the runtime lock, on a copy of the request. */
value caml_tolk_metal_compile(value v_request) {
  CAMLparam1(v_request);
  CAMLlocal1(v_reply);
  size_t size = caml_string_length(v_request);
  char *request = malloc(size);
  if (request == NULL) caml_raise_out_of_memory();
  memcpy(request, String_val(v_request), size);
  __block char *reply = NULL, *message = NULL;
  __block size_t reply_size = 0;
  __block int called = 0;
  caml_release_runtime_system();
  MTLCodeGenServiceBuildRequest(
      service, NULL, REQUEST_TYPE_COMPILE, request, size,
      ^(int32_t error, const void *data, size_t data_size, const char *msg) {
        called = 1;
        if (error == 0) {
          if ((reply = malloc(data_size ? data_size : 1)) != NULL)
            memcpy(reply, data, reply_size = data_size);
        } else
          message = strdup(msg != NULL ? msg : "");
      });
  caml_acquire_runtime_system();
  free(request);
  if (!called)
    CAMLreturn(ERROR(
        "MTLCodeGenServiceBuildRequest returned without calling the callback"));
  if (reply == NULL && message == NULL) caml_raise_out_of_memory();
  if (message != NULL) {
    value v_error = ERROR("%s", message);
    free(message);
    CAMLreturn(v_error);
  }
  v_reply = caml_alloc_initialized_string(reply_size, reply);
  free(reply);
  CAMLreturn(OK(v_reply));
}

#else

value caml_tolk_metal_load(value v_path) {
  return ERROR("MTLCompiler is macOS's");
}

value caml_tolk_metal_macos_major(value unit) {
  return Val_int(0);
}

value caml_tolk_metal_macos_build(value unit) {
  return caml_copy_string("");
}

value caml_tolk_metal_compile(value v_request) {
  caml_invalid_argument("MTLCompiler is not loaded");
}

#endif
