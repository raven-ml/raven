/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

/* NVRTC, whose functions are declared here and found in the library at run
   time, so that tolk builds and runs without CUDA. */

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
static void *open_library(const char *path) { return LoadLibraryA(path); }
static void *find_symbol(void *lib, const char *name) {
  return (void *)GetProcAddress((HMODULE)lib, name);
}
static const char *library_error(void) { return "LoadLibrary failed"; }
#else
#include <dlfcn.h>
static void *open_library(const char *path) {
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}
static void *find_symbol(void *lib, const char *name) {
  return dlsym(lib, name);
}
static const char *library_error(void) { return dlerror(); }
#endif

typedef int nvrtcResult;
typedef struct nvrtcProgram_ *nvrtcProgram;
typedef nvrtcResult (*get_size_t)(nvrtcProgram, size_t *);
typedef nvrtcResult (*get_bytes_t)(nvrtcProgram, char *);

static struct {
  nvrtcResult (*Version)(int *, int *);
  const char *(*GetErrorString)(nvrtcResult);
  nvrtcResult (*CreateProgram)(nvrtcProgram *, const char *, const char *,
                               int, const char *const *, const char *const *);
  nvrtcResult (*CompileProgram)(nvrtcProgram, int, const char *const *);
  nvrtcResult (*DestroyProgram)(nvrtcProgram *);
  get_size_t GetProgramLogSize, GetPTXSize, GetCUBINSize;
  get_bytes_t GetProgramLog, GetPTX, GetCUBIN;
} nvrtc;

#define SYMBOL(f) {(void **)&nvrtc.f, "nvrtc" #f}
static const struct {
  void **fn;
  const char *name;
} symbols[] = {SYMBOL(Version),       SYMBOL(GetErrorString),
               SYMBOL(CreateProgram), SYMBOL(CompileProgram),
               SYMBOL(DestroyProgram), SYMBOL(GetProgramLogSize),
               SYMBOL(GetPTXSize),    SYMBOL(GetCUBINSize),
               SYMBOL(GetProgramLog), SYMBOL(GetPTX),
               SYMBOL(GetCUBIN)};
#undef SYMBOL

static value result(int tag, value v) {
  CAMLparam1(v);
  CAMLlocal1(r);
  r = caml_alloc_small(1, tag);
  Field(r, 0) = v;
  CAMLreturn(r);
}

#define OK(v) result(0, v)
#define ERROR(...) result(1, caml_alloc_sprintf(__VA_ARGS__))

/* Called once, by the caller's lock: the pointers it sets are only read by
   compilers made after it returned. */
value caml_tolk_nvrtc_load(value v_path) {
  CAMLparam1(v_path);
  CAMLlocal1(v_version);
  void *lib = open_library(String_val(v_path));
  if (lib == NULL) CAMLreturn(ERROR("%s", library_error()));
  for (size_t i = 0; i < sizeof symbols / sizeof *symbols; i++)
    if ((*symbols[i].fn = find_symbol(lib, symbols[i].name)) == NULL)
      CAMLreturn(ERROR("%s: undefined symbol %s", String_val(v_path),
                       symbols[i].name));
  int major, minor;
  nvrtcResult status = nvrtc.Version(&major, &minor);
  if (status != 0)
    CAMLreturn(ERROR("Nvrtc Error %d, %s\n", status,
                     nvrtc.GetErrorString(status)));
  v_version = caml_alloc_tuple(2);
  Store_field(v_version, 0, Val_int(major));
  Store_field(v_version, 1, Val_int(minor));
  CAMLreturn(OK(v_version));
}

/* The [size] bytes that [get] writes, or NULL with [*status] set. */
static char *get_bytes(nvrtcProgram prog, get_size_t get_size, get_bytes_t get,
                       size_t *size, nvrtcResult *status) {
  char *bytes = NULL;
  if ((*status = get_size(prog, size)) == 0 &&
      (bytes = malloc(*size ? *size : 1)) != NULL &&
      (*status = get(prog, bytes)) != 0) {
    free(bytes);
    bytes = NULL;
  }
  return bytes;
}

/* Compiling runs without the runtime lock, so that domains compile at once;
   the source and options are copied out of the OCaml heap first. */
value caml_tolk_nvrtc_compile(value v_src, value v_options, value v_ptx) {
  CAMLparam3(v_src, v_options, v_ptx);
  CAMLlocal1(v_lib);
  int ptx = Bool_val(v_ptx), noptions = Wosize_val(v_options);
  char *src = strdup(String_val(v_src));
  char **options = calloc(noptions ? noptions : 1, sizeof *options);
  if (src == NULL || options == NULL) caml_raise_out_of_memory();
  for (int i = 0; i < noptions; i++)
    if ((options[i] = strdup(String_val(Field(v_options, i)))) == NULL)
      caml_raise_out_of_memory();

  nvrtcProgram prog;
  nvrtcResult status;
  char *lib = NULL, *log = NULL;
  size_t size = 0, log_size = 0;
  caml_release_runtime_system();
  status = nvrtc.CreateProgram(&prog, src, "<null>", 0, NULL, NULL);
  if (status == 0) {
    status = nvrtc.CompileProgram(prog, noptions, (const char *const *)options);
    if (status != 0) {
      nvrtcResult log_status;
      log = get_bytes(prog, nvrtc.GetProgramLogSize, nvrtc.GetProgramLog,
                      &log_size, &log_status);
    } else
      lib = get_bytes(prog, ptx ? nvrtc.GetPTXSize : nvrtc.GetCUBINSize,
                      ptx ? nvrtc.GetPTX : nvrtc.GetCUBIN, &size, &status);
    nvrtc.DestroyProgram(&prog);
  }
  caml_acquire_runtime_system();
  for (int i = 0; i < noptions; i++) free(options[i]);
  free(options);
  free(src);

  if (status != 0) {
    /* The log ends with its NUL. */
    value v_error =
        ERROR("Nvrtc Error %d, %s\n%s", status, nvrtc.GetErrorString(status),
              log != NULL && log_size > 0 ? log : "");
    free(log);
    CAMLreturn(v_error);
  }
  if (lib == NULL) caml_raise_out_of_memory();
  v_lib = caml_alloc_initialized_string(size, lib);
  free(lib);
  CAMLreturn(OK(v_lib));
}
