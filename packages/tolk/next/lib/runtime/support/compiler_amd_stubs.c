/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

/* comgr, whose functions are declared here and found in the library at run
   time, so that tolk builds and runs without ROCm. */

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
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

typedef uint32_t status_t;
typedef struct {
  uint64_t handle;
} data_t;
typedef struct {
  uint64_t handle;
} data_set_t;
typedef struct {
  uint64_t handle;
} action_info_t;

static struct {
  void (*get_version)(uint64_t *, uint64_t *);
  status_t (*status_string)(status_t, const char **);
  status_t (*create_action_info)(action_info_t *);
  status_t (*destroy_action_info)(action_info_t);
  status_t (*action_info_set_language)(action_info_t, uint32_t);
  status_t (*action_info_set_isa_name)(action_info_t, const char *);
  status_t (*action_info_set_logging)(action_info_t, bool);
  status_t (*action_info_set_option_list)(action_info_t, const char *const *,
                                          size_t);
  status_t (*create_data_set)(data_set_t *);
  status_t (*destroy_data_set)(data_set_t);
  status_t (*create_data)(uint32_t, data_t *);
  status_t (*release_data)(data_t);
  status_t (*set_data)(data_t, size_t, const char *);
  status_t (*set_data_name)(data_t, const char *);
  status_t (*data_set_add)(data_set_t, data_t);
  status_t (*do_action)(uint32_t, action_info_t, data_set_t, data_set_t);
  status_t (*action_data_get_data)(data_set_t, uint32_t, size_t, data_t *);
  status_t (*get_data)(data_t, size_t *, char *);
} comgr;

#define SYMBOL(f) {(void **)&comgr.f, "amd_comgr_" #f}
static const struct {
  void **fn;
  const char *name;
} symbols[] = {SYMBOL(get_version),
               SYMBOL(status_string),
               SYMBOL(create_action_info),
               SYMBOL(destroy_action_info),
               SYMBOL(action_info_set_language),
               SYMBOL(action_info_set_isa_name),
               SYMBOL(action_info_set_logging),
               SYMBOL(action_info_set_option_list),
               SYMBOL(create_data_set),
               SYMBOL(destroy_data_set),
               SYMBOL(create_data),
               SYMBOL(release_data),
               SYMBOL(set_data),
               SYMBOL(set_data_name),
               SYMBOL(data_set_add),
               SYMBOL(do_action),
               SYMBOL(action_data_get_data),
               SYMBOL(get_data)};
#undef SYMBOL

/* The data kinds of both versions. */
enum { DATA_KIND_SOURCE = 1, DATA_KIND_LOG = 5, DATA_KIND_EXECUTABLE = 8 };

/* Version 3 renumbered the languages and actions. */
static struct {
  uint32_t language_hip, compile_source_with_device_libs_to_bc,
      codegen_bc_to_relocatable, link_relocatable_to_executable,
      assemble_source_to_relocatable;
} enums;

static value result(int tag, value v) {
  CAMLparam1(v);
  CAMLlocal1(r);
  r = caml_alloc_small(1, tag);
  Field(r, 0) = v;
  CAMLreturn(r);
}

#define OK(v) result(0, v)
#define ERROR(...) result(1, caml_alloc_sprintf(__VA_ARGS__))

/* Called once, by the caller's lock: what it sets is only read by compilers
   made after it returned. */
value caml_tolk_comgr_load(value v_path) {
  CAMLparam1(v_path);
  void *lib = open_library(String_val(v_path));
  if (lib == NULL) CAMLreturn(ERROR("%s", library_error()));
  for (size_t i = 0; i < sizeof symbols / sizeof *symbols; i++)
    if ((*symbols[i].fn = find_symbol(lib, symbols[i].name)) == NULL)
      CAMLreturn(ERROR("%s: undefined symbol %s", String_val(v_path),
                       symbols[i].name));
  uint64_t major, minor;
  comgr.get_version(&major, &minor);
  if (major >= 3)
    enums.language_hip = 3, enums.compile_source_with_device_libs_to_bc = 12,
    enums.codegen_bc_to_relocatable = 4,
    enums.link_relocatable_to_executable = 7,
    enums.assemble_source_to_relocatable = 8;
  else
    enums.language_hip = 4, enums.compile_source_with_device_libs_to_bc = 15,
    enums.codegen_bc_to_relocatable = 6,
    enums.link_relocatable_to_executable = 9,
    enums.assemble_source_to_relocatable = 10;
  CAMLreturn(OK(Val_unit));
}

/* The first data of [kind] in [set], NUL-terminated, or NULL. */
static char *get_data(data_set_t set, uint32_t kind, size_t *size) {
  data_t data;
  char *bytes = NULL;
  if (comgr.action_data_get_data(set, kind, 0, &data) != 0) return NULL;
  if (comgr.get_data(data, size, NULL) == 0 &&
      (bytes = malloc(*size + 1)) != NULL) {
    if (comgr.get_data(data, size, bytes) == 0)
      bytes[*size] = '\0';
    else {
      free(bytes);
      bytes = NULL;
    }
  }
  comgr.release_data(data);
  return bytes;
}

struct options {
  const char **list;
  size_t length;
};

/* A message on the C heap, or NULL if memory ran out. */
static char *format(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int length = vsnprintf(NULL, 0, fmt, args);
  va_end(args);
  char *message = malloc(length + 1);
  if (message != NULL) {
    va_start(args, fmt);
    vsnprintf(message, length + 1, fmt, args);
    va_end(args);
  }
  return message;
}

/* Runs without the runtime lock. On success, [*lib] holds [*size] bytes;
   otherwise [*error] holds the message, or is NULL if memory ran out. */
static void compile_hip(const char *src, size_t src_size, const char *isa,
                        bool assemble, struct options compile,
                        struct options codegen, char **lib, size_t *size,
                        char **error) {
  action_info_t info;
  data_set_t sets[4]; /* source, bitcode, relocatable, executable */
  data_t data_src;
  int nsets = 0;
  bool have_info = false, have_src = false;
  status_t status;
  const char *failed = NULL; /* An action that failed, reported with its log */
  char *log = NULL;
  size_t log_size;
  /* Options stay set from one action to the next: linking clears them. */
  const char *no_option = "";
  *lib = *error = NULL;

#define CHECK(call)                                                            \
  if ((status = (call)) != 0) goto fail
#define ACTION(action, from, to, message)                                      \
  if ((status = comgr.do_action(action, info, from, to)) != 0) {               \
    failed = message;                                                          \
    log = get_data(to, DATA_KIND_LOG, &log_size);                              \
    goto fail;                                                                 \
  }
  CHECK(comgr.create_action_info(&info));
  have_info = true;
  CHECK(comgr.action_info_set_language(info, enums.language_hip));
  CHECK(comgr.action_info_set_isa_name(info, isa));
  CHECK(comgr.action_info_set_logging(info, true));
  for (; nsets < 4; nsets++) CHECK(comgr.create_data_set(&sets[nsets]));
  CHECK(comgr.create_data(DATA_KIND_SOURCE, &data_src));
  have_src = true;
  CHECK(comgr.set_data(data_src, src_size, src));
  if (assemble) {
    CHECK(comgr.set_data_name(data_src, "<null>.s"));
    CHECK(comgr.data_set_add(sets[0], data_src));
    ACTION(enums.assemble_source_to_relocatable, sets[0], sets[2],
           "assemble failed");
  } else {
    CHECK(comgr.set_data_name(data_src, "<null>"));
    CHECK(comgr.data_set_add(sets[0], data_src));
    CHECK(comgr.action_info_set_option_list(info, compile.list, compile.length));
    ACTION(enums.compile_source_with_device_libs_to_bc, sets[0], sets[1],
           "compile failed");
    CHECK(comgr.action_info_set_option_list(info, codegen.list, codegen.length));
    CHECK(comgr.do_action(enums.codegen_bc_to_relocatable, info, sets[1],
                          sets[2]));
  }
  CHECK(comgr.action_info_set_option_list(info, &no_option, 1));
  CHECK(comgr.do_action(enums.link_relocatable_to_executable, info, sets[2],
                        sets[3]));
#undef ACTION
#undef CHECK
  if ((*lib = get_data(sets[3], DATA_KIND_EXECUTABLE, size)) != NULL)
    goto release;
  status = 1; /* AMD_COMGR_STATUS_ERROR */

fail:
  if (failed != NULL)
    *error = format("%s\n%s", failed, log != NULL ? log : "");
  else {
    const char *reason = NULL;
    if (comgr.status_string(status, &reason) != 0 || reason == NULL)
      reason = "unknown status";
    *error = format("comgr fail %u, %s", (unsigned)status, reason);
  }
  free(log);
release:
  if (have_src) comgr.release_data(data_src);
  for (int i = 0; i < nsets; i++) comgr.destroy_data_set(sets[i]);
  if (have_info) comgr.destroy_action_info(info);
}

static struct options options_of_array(value v_options) {
  struct options o = {calloc(Wosize_val(v_options) + 1, sizeof(char *)),
                      Wosize_val(v_options)};
  if (o.list == NULL) caml_raise_out_of_memory();
  for (size_t i = 0; i < o.length; i++)
    if ((o.list[i] = strdup(String_val(Field(v_options, i)))) == NULL)
      caml_raise_out_of_memory();
  return o;
}

static void free_options(struct options o) {
  for (size_t i = 0; i < o.length; i++) free((char *)o.list[i]);
  free(o.list);
}

/* Compiling runs without the runtime lock, so that domains compile at once;
   its arguments are copied out of the OCaml heap first. */
value caml_tolk_comgr_compile(value v_src, value v_isa, value v_assemble,
                              value v_compile, value v_codegen) {
  CAMLparam5(v_src, v_isa, v_assemble, v_compile, v_codegen);
  CAMLlocal1(v_lib);
  size_t src_size = caml_string_length(v_src), size = 0;
  char *src = malloc(src_size + 1), *isa = strdup(String_val(v_isa));
  if (src == NULL || isa == NULL) caml_raise_out_of_memory();
  memcpy(src, String_val(v_src), src_size);
  struct options compile = options_of_array(v_compile),
                 codegen = options_of_array(v_codegen);
  char *lib, *error;
  caml_release_runtime_system();
  compile_hip(src, src_size, isa, Bool_val(v_assemble), compile, codegen, &lib,
              &size, &error);
  caml_acquire_runtime_system();
  free_options(compile);
  free_options(codegen);
  free(isa);
  free(src);
  if (lib == NULL) {
    if (error == NULL) caml_raise_out_of_memory();
    value v_error = ERROR("%s", error);
    free(error);
    CAMLreturn(v_error);
  }
  v_lib = caml_alloc_initialized_string(size, lib);
  free(lib);
  CAMLreturn(OK(v_lib));
}
