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
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Hand-declared subset of the NVRTC API. The library is resolved with dlopen
   at first use so this library builds and loads on machines without the CUDA
   toolkit; compilation fails cleanly there instead. */

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
static nvrtcResult (*p_nvrtcGetCUBINSize)(nvrtcProgram, size_t *);
static nvrtcResult (*p_nvrtcGetCUBIN)(nvrtcProgram, char *);
static nvrtcResult (*p_nvrtcGetProgramLogSize)(nvrtcProgram, size_t *);
static nvrtcResult (*p_nvrtcGetProgramLog)(nvrtcProgram, char *);
static const char *(*p_nvrtcGetErrorString)(nvrtcResult);
static nvrtcResult (*p_nvrtcDestroyProgram)(nvrtcProgram *);

static void *nvrtc_handle = NULL;

/* Guarded lazy init: beam search can call nvrtc_compile from several OCaml
   domains concurrently, and the first of them loads the library. A mutex and
   an explicit state rather than pthread_once: the failure path raises an
   OCaml exception, and the longjmp out of a pthread_once init routine would
   leave the once control permanently in progress — every later caller would
   deadlock instead of seeing the error. The raise happens outside the lock,
   and a failed load is retried on the next call, as before. */
static pthread_mutex_t nvrtc_mutex = PTHREAD_MUTEX_INITIALIZER;
static int nvrtc_loaded = 0;
static char nvrtc_error[128];

static void load_nvrtc(void) {
  static const char *names[] = {"libnvrtc.so", "libnvrtc.so.13",
                                "libnvrtc.so.12",
                                "/usr/local/cuda/lib64/libnvrtc.so", NULL};
  for (int i = 0; nvrtc_handle == NULL && names[i] != NULL; ++i)
    nvrtc_handle = dlopen(names[i], RTLD_LAZY | RTLD_LOCAL);
  if (nvrtc_handle == NULL) {
    snprintf(nvrtc_error, sizeof(nvrtc_error),
             "NVRTC library (libnvrtc.so) not found");
    return;
  }
#define LOAD_NVRTC(var, name)                                          \
  do {                                                                 \
    var = dlsym(nvrtc_handle, name);                                   \
    if (var == NULL) {                                                 \
      snprintf(nvrtc_error, sizeof(nvrtc_error),                       \
               "NVRTC is missing " name);                              \
      return;                                                          \
    }                                                                  \
  } while (0)
  LOAD_NVRTC(p_nvrtcVersion, "nvrtcVersion");
  LOAD_NVRTC(p_nvrtcCreateProgram, "nvrtcCreateProgram");
  LOAD_NVRTC(p_nvrtcCompileProgram, "nvrtcCompileProgram");
  LOAD_NVRTC(p_nvrtcGetPTXSize, "nvrtcGetPTXSize");
  LOAD_NVRTC(p_nvrtcGetPTX, "nvrtcGetPTX");
  LOAD_NVRTC(p_nvrtcGetCUBINSize, "nvrtcGetCUBINSize");
  LOAD_NVRTC(p_nvrtcGetCUBIN, "nvrtcGetCUBIN");
  LOAD_NVRTC(p_nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize");
  LOAD_NVRTC(p_nvrtcGetProgramLog, "nvrtcGetProgramLog");
  LOAD_NVRTC(p_nvrtcGetErrorString, "nvrtcGetErrorString");
  LOAD_NVRTC(p_nvrtcDestroyProgram, "nvrtcDestroyProgram");
#undef LOAD_NVRTC
  nvrtc_loaded = 1;
}

static void ensure_nvrtc(void) {
  char err[sizeof(nvrtc_error)];
  int loaded;
  pthread_mutex_lock(&nvrtc_mutex);
  if (!nvrtc_loaded) load_nvrtc();
  loaded = nvrtc_loaded;
  if (!loaded) snprintf(err, sizeof(err), "%s", nvrtc_error);
  pthread_mutex_unlock(&nvrtc_mutex);
  if (!loaded) caml_failwith(err);
}

CAMLprim value caml_tolk_nvrtc_version(value unit) {
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

/* Compile CUDA C source to PTX when [v_ptx] is true, else to a cubin. Returns
   [Ok lib] or [Error message] where the message carries the NVRTC error
   string and the program log. The source and options are copied to C memory
   so the OCaml runtime lock can be released during compilation. The cubin
   payload is raw binary, so the output is always copied by its reported size,
   never measured as a C string. */
CAMLprim value caml_tolk_nvrtc_compile(value v_src, value v_opts, value v_ptx) {
  CAMLparam3(v_src, v_opts, v_ptx);
  CAMLlocal3(v_result, v_payload, v_msg);
  ensure_nvrtc();

  int to_ptx = Bool_val(v_ptx);
  nvrtcResult (*get_size)(nvrtcProgram, size_t *) =
      to_ptx ? p_nvrtcGetPTXSize : p_nvrtcGetCUBINSize;
  nvrtcResult (*get_data)(nvrtcProgram, char *) =
      to_ptx ? p_nvrtcGetPTX : p_nvrtcGetCUBIN;

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
  char *out = NULL;
  size_t out_size = 0;

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
      status = get_size(prog, &out_size);
      if (status == 0) {
        out = (char *)malloc(out_size > 0 ? out_size : 1);
        if (out != NULL)
          status = get_data(prog, out);
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

  if (status != 0 || out == NULL) {
    const char *error =
        status != 0 ? p_nvrtcGetErrorString(status) : "out of memory";
    size_t msg_size =
        strlen(error) + (log != NULL ? strlen(log) : 0) + 64;
    char *msg = (char *)malloc(msg_size);
    if (msg != NULL)
      snprintf(msg, msg_size, "Nvrtc Error %d, %s\n%s", status, error,
               log != NULL ? log : "");
    free(log);
    free(out);
    v_msg = caml_copy_string(msg != NULL ? msg : "Nvrtc Error");
    free(msg);
    v_result = caml_alloc(1, 1); /* Error */
    Store_field(v_result, 0, v_msg);
    CAMLreturn(v_result);
  }

  free(log);
  v_payload = caml_alloc_string(out_size);
  memcpy(Bytes_val(v_payload), out, out_size);
  free(out);
  v_result = caml_alloc(1, 0); /* Ok */
  Store_field(v_result, 0, v_payload);
  CAMLreturn(v_result);
}
