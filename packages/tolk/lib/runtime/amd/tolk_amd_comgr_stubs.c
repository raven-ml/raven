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
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Hand-declared subset of the AMD code object manager (comgr) API. The
   library is resolved with dlopen at first use so this library builds and
   loads on machines without ROCm; compilation fails cleanly there instead. */

typedef uint32_t amd_comgr_status_t;

/* comgr handles are 64-bit structs passed by value. */
typedef struct {
  uint64_t handle;
} amd_comgr_data_t;
typedef struct {
  uint64_t handle;
} amd_comgr_data_set_t;
typedef struct {
  uint64_t handle;
} amd_comgr_action_info_t;

/* comgr 3 renumbered the language and action enums; both value sets are
   declared and the loaded library's major version selects between them. The
   data kinds are identical in both versions. */
#define COMGR2_LANGUAGE_HIP 4
#define COMGR2_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC 15
#define COMGR2_ACTION_CODEGEN_BC_TO_RELOCATABLE 6
#define COMGR2_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE 9
#define COMGR2_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE 10

#define COMGR3_LANGUAGE_HIP 3
#define COMGR3_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC 12
#define COMGR3_ACTION_CODEGEN_BC_TO_RELOCATABLE 4
#define COMGR3_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE 7
#define COMGR3_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE 8

#define COMGR_DATA_KIND_SOURCE 1
#define COMGR_DATA_KIND_LOG 5
#define COMGR_DATA_KIND_EXECUTABLE 8

static void (*p_amd_comgr_get_version)(uint64_t *, uint64_t *);
static amd_comgr_status_t (*p_amd_comgr_status_string)(amd_comgr_status_t,
                                                       const char **);
static amd_comgr_status_t (*p_amd_comgr_create_action_info)(
    amd_comgr_action_info_t *);
static amd_comgr_status_t (*p_amd_comgr_destroy_action_info)(
    amd_comgr_action_info_t);
static amd_comgr_status_t (*p_amd_comgr_action_info_set_language)(
    amd_comgr_action_info_t, uint32_t);
static amd_comgr_status_t (*p_amd_comgr_action_info_set_isa_name)(
    amd_comgr_action_info_t, const char *);
static amd_comgr_status_t (*p_amd_comgr_action_info_set_logging)(
    amd_comgr_action_info_t, bool);
static amd_comgr_status_t (*p_amd_comgr_action_info_set_option_list)(
    amd_comgr_action_info_t, const char *const *, size_t);
static amd_comgr_status_t (*p_amd_comgr_create_data_set)(
    amd_comgr_data_set_t *);
static amd_comgr_status_t (*p_amd_comgr_destroy_data_set)(
    amd_comgr_data_set_t);
static amd_comgr_status_t (*p_amd_comgr_create_data)(uint32_t,
                                                     amd_comgr_data_t *);
static amd_comgr_status_t (*p_amd_comgr_release_data)(amd_comgr_data_t);
static amd_comgr_status_t (*p_amd_comgr_set_data)(amd_comgr_data_t, size_t,
                                                  const char *);
static amd_comgr_status_t (*p_amd_comgr_set_data_name)(amd_comgr_data_t,
                                                       const char *);
static amd_comgr_status_t (*p_amd_comgr_data_set_add)(amd_comgr_data_set_t,
                                                      amd_comgr_data_t);
static amd_comgr_status_t (*p_amd_comgr_do_action)(uint32_t,
                                                   amd_comgr_action_info_t,
                                                   amd_comgr_data_set_t,
                                                   amd_comgr_data_set_t);
static amd_comgr_status_t (*p_amd_comgr_action_data_get_data)(
    amd_comgr_data_set_t, uint32_t, size_t, amd_comgr_data_t *);
static amd_comgr_status_t (*p_amd_comgr_get_data)(amd_comgr_data_t, size_t *,
                                                  char *);

static void *comgr_handle = NULL;

/* Enum values selected once at load from the library's major version. */
static uint64_t comgr_major = 0, comgr_minor = 0;
static uint32_t comgr_language_hip;
static uint32_t comgr_action_compile_to_bc;
static uint32_t comgr_action_codegen_to_reloc;
static uint32_t comgr_action_link_to_exec;

/* Guarded lazy init: beam search can call comgr_compile from several OCaml
   domains concurrently, and the first of them loads the library. A mutex and
   an explicit state rather than pthread_once: the failure path raises an
   OCaml exception, and the longjmp out of a pthread_once init routine would
   leave the once control permanently in progress — every later caller would
   deadlock instead of seeing the error. The raise happens outside the lock,
   and a failed load is retried on the next call. */
static pthread_mutex_t comgr_mutex = PTHREAD_MUTEX_INITIALIZER;
static int comgr_loaded = 0;
static char comgr_load_error[128];

static void load_comgr(void) {
  char rocm_lib[4096];
  const char *rocm_path = getenv("ROCM_PATH");
  snprintf(rocm_lib, sizeof(rocm_lib), "%s/lib/libamd_comgr.so",
           rocm_path != NULL ? rocm_path : "/opt/rocm");
  const char *names[] = {rocm_lib, "libamd_comgr.so", "libamd_comgr.so.3",
                         "libamd_comgr.so.2", NULL};
  for (int i = 0; comgr_handle == NULL && names[i] != NULL; ++i)
    comgr_handle = dlopen(names[i], RTLD_LAZY | RTLD_LOCAL);
  if (comgr_handle == NULL) {
    snprintf(comgr_load_error, sizeof(comgr_load_error),
             "comgr library (libamd_comgr.so) not found");
    return;
  }
#define LOAD_COMGR(var, name)                                          \
  do {                                                                 \
    var = dlsym(comgr_handle, name);                                   \
    if (var == NULL) {                                                 \
      snprintf(comgr_load_error, sizeof(comgr_load_error),             \
               "comgr is missing " name);                              \
      return;                                                          \
    }                                                                  \
  } while (0)
  LOAD_COMGR(p_amd_comgr_get_version, "amd_comgr_get_version");
  LOAD_COMGR(p_amd_comgr_status_string, "amd_comgr_status_string");
  LOAD_COMGR(p_amd_comgr_create_action_info, "amd_comgr_create_action_info");
  LOAD_COMGR(p_amd_comgr_destroy_action_info,
             "amd_comgr_destroy_action_info");
  LOAD_COMGR(p_amd_comgr_action_info_set_language,
             "amd_comgr_action_info_set_language");
  LOAD_COMGR(p_amd_comgr_action_info_set_isa_name,
             "amd_comgr_action_info_set_isa_name");
  LOAD_COMGR(p_amd_comgr_action_info_set_logging,
             "amd_comgr_action_info_set_logging");
  LOAD_COMGR(p_amd_comgr_action_info_set_option_list,
             "amd_comgr_action_info_set_option_list");
  LOAD_COMGR(p_amd_comgr_create_data_set, "amd_comgr_create_data_set");
  LOAD_COMGR(p_amd_comgr_destroy_data_set, "amd_comgr_destroy_data_set");
  LOAD_COMGR(p_amd_comgr_create_data, "amd_comgr_create_data");
  LOAD_COMGR(p_amd_comgr_release_data, "amd_comgr_release_data");
  LOAD_COMGR(p_amd_comgr_set_data, "amd_comgr_set_data");
  LOAD_COMGR(p_amd_comgr_set_data_name, "amd_comgr_set_data_name");
  LOAD_COMGR(p_amd_comgr_data_set_add, "amd_comgr_data_set_add");
  LOAD_COMGR(p_amd_comgr_do_action, "amd_comgr_do_action");
  LOAD_COMGR(p_amd_comgr_action_data_get_data,
             "amd_comgr_action_data_get_data");
  LOAD_COMGR(p_amd_comgr_get_data, "amd_comgr_get_data");
#undef LOAD_COMGR
  p_amd_comgr_get_version(&comgr_major, &comgr_minor);
  if (comgr_major >= 3) {
    comgr_language_hip = COMGR3_LANGUAGE_HIP;
    comgr_action_compile_to_bc =
        COMGR3_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC;
    comgr_action_codegen_to_reloc = COMGR3_ACTION_CODEGEN_BC_TO_RELOCATABLE;
    comgr_action_link_to_exec = COMGR3_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE;
  } else {
    comgr_language_hip = COMGR2_LANGUAGE_HIP;
    comgr_action_compile_to_bc =
        COMGR2_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC;
    comgr_action_codegen_to_reloc = COMGR2_ACTION_CODEGEN_BC_TO_RELOCATABLE;
    comgr_action_link_to_exec = COMGR2_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE;
  }
  comgr_loaded = 1;
}

static void ensure_comgr(void) {
  char err[sizeof(comgr_load_error)];
  int loaded;
  pthread_mutex_lock(&comgr_mutex);
  if (!comgr_loaded) load_comgr();
  loaded = comgr_loaded;
  if (!loaded) snprintf(err, sizeof(err), "%s", comgr_load_error);
  pthread_mutex_unlock(&comgr_mutex);
  if (!loaded) caml_failwith(err);
}

/* Extracts the first data object of [kind] from [ds] using the two-call
   size/data protocol. Returns a malloc'd NUL-terminated buffer (and its size
   through [size_out]) or NULL. */
static char *comgr_get_output(amd_comgr_data_set_t ds, uint32_t kind,
                              size_t *size_out) {
  amd_comgr_data_t data = {0};
  size_t sz = 0;
  char *buf = NULL;
  if (p_amd_comgr_action_data_get_data(ds, kind, 0, &data) != 0) return NULL;
  if (p_amd_comgr_get_data(data, &sz, NULL) == 0 &&
      (buf = (char *)malloc(sz + 1)) != NULL) {
    if (p_amd_comgr_get_data(data, &sz, buf) == 0)
      buf[sz] = '\0';
    else {
      free(buf);
      buf = NULL;
    }
  }
  p_amd_comgr_release_data(data);
  if (buf != NULL && size_out != NULL) *size_out = sz;
  return buf;
}

/* Runs the HIP-to-code-object pipeline. Called with the OCaml runtime lock
   released, so it must not touch the OCaml heap. On success returns 0 with
   [*out]/[*out_size] set to a malloc'd code object; on failure returns -1
   with [*err] set to a malloc'd message (NULL on allocation failure)
   carrying the comgr status string and, after a failed action, the log
   emitted into that action's output data set. */
static int comgr_compile_hip(const char *src, size_t src_len,
                             const char *arch, char **out, size_t *out_size,
                             char **err) {
  amd_comgr_action_info_t action_info = {0};
  /* Pipeline data sets: source, bitcode, relocatable, executable. */
  amd_comgr_data_set_t sets[4] = {{0}, {0}, {0}, {0}};
  amd_comgr_data_t data_src = {0};
  int have_action = 0, have_sets = 0, have_src = 0;
  amd_comgr_status_t status = 0;
  char *log = NULL;
  char isa_name[128], offload_arch[128];
  int ret = -1;

  *out = NULL;
  *out_size = 0;
  *err = NULL;

  snprintf(isa_name, sizeof(isa_name), "amdgcn-amd-amdhsa--%s", arch);
  snprintf(offload_arch, sizeof(offload_arch), "--offload-arch=%s", arch);

  const char *compile_opts[] = {
      "-O3", "-mcumode", "--hip-version=6.0.32830", "-DHIP_VERSION_MAJOR=6",
      "-DHIP_VERSION_MINOR=0", "-DHIP_VERSION_PATCH=32830", "-D__HIPCC_RTC__",
      "-std=c++14", "-nogpuinc", "-Wno-gnu-line-marker",
      "-Wno-missing-prototypes", offload_arch, "-I/opt/rocm/include",
      "-Xclang", "-disable-llvm-passes", "-Xclang", "-aux-triple", "-Xclang",
      "x86_64-unknown-linux-gnu"};
  const char *codegen_opts[] = {"-O3", "-mllvm",
                                "-amdgpu-internalize-symbols"};
  /* The link step takes a single empty option, not an empty option list. */
  const char *link_opts[] = {""};

#define CHECK(expr)                                                    \
  do {                                                                 \
    status = (expr);                                                   \
    if (status != 0) goto fail;                                        \
  } while (0)

  CHECK(p_amd_comgr_create_action_info(&action_info));
  have_action = 1;
  CHECK(p_amd_comgr_action_info_set_language(action_info,
                                             comgr_language_hip));
  CHECK(p_amd_comgr_action_info_set_isa_name(action_info, isa_name));
  CHECK(p_amd_comgr_action_info_set_logging(action_info, true));

  for (int i = 0; i < 4; ++i) {
    CHECK(p_amd_comgr_create_data_set(&sets[i]));
    have_sets = i + 1;
  }

  CHECK(p_amd_comgr_create_data(COMGR_DATA_KIND_SOURCE, &data_src));
  have_src = 1;
  CHECK(p_amd_comgr_set_data(data_src, src_len, src));
  CHECK(p_amd_comgr_set_data_name(data_src, "<null>"));
  CHECK(p_amd_comgr_data_set_add(sets[0], data_src));

  CHECK(p_amd_comgr_action_info_set_option_list(
      action_info, compile_opts,
      sizeof(compile_opts) / sizeof(*compile_opts)));
  status = p_amd_comgr_do_action(comgr_action_compile_to_bc, action_info,
                                 sets[0], sets[1]);
  if (status != 0) {
    log = comgr_get_output(sets[1], COMGR_DATA_KIND_LOG, NULL);
    goto fail;
  }

  CHECK(p_amd_comgr_action_info_set_option_list(
      action_info, codegen_opts,
      sizeof(codegen_opts) / sizeof(*codegen_opts)));
  status = p_amd_comgr_do_action(comgr_action_codegen_to_reloc, action_info,
                                 sets[1], sets[2]);
  if (status != 0) {
    log = comgr_get_output(sets[2], COMGR_DATA_KIND_LOG, NULL);
    goto fail;
  }

  CHECK(p_amd_comgr_action_info_set_option_list(action_info, link_opts, 1));
  status = p_amd_comgr_do_action(comgr_action_link_to_exec, action_info,
                                 sets[2], sets[3]);
  if (status != 0) {
    log = comgr_get_output(sets[3], COMGR_DATA_KIND_LOG, NULL);
    goto fail;
  }
#undef CHECK

  *out = comgr_get_output(sets[3], COMGR_DATA_KIND_EXECUTABLE, out_size);
  if (*out == NULL) {
    status = 1; /* AMD_COMGR_STATUS_ERROR */
    goto fail;
  }
  ret = 0;
  goto cleanup;

fail: {
  const char *status_str = "unknown status";
  const char *s = NULL;
  if (p_amd_comgr_status_string(status, &s) == 0 && s != NULL)
    status_str = s;
  size_t msg_size = strlen(status_str) + (log != NULL ? strlen(log) : 0) + 64;
  char *msg = (char *)malloc(msg_size);
  if (msg != NULL)
    snprintf(msg, msg_size, "comgr fail %u, %s%s%s", (unsigned)status,
             status_str, log != NULL ? "\n" : "", log != NULL ? log : "");
  *err = msg;
}
cleanup:
  free(log);
  if (have_src) p_amd_comgr_release_data(data_src);
  for (int i = 0; i < have_sets; ++i) p_amd_comgr_destroy_data_set(sets[i]);
  if (have_action) p_amd_comgr_destroy_action_info(action_info);
  return ret;
}

CAMLprim value caml_tolk_amd_comgr_version(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(v_pair);
  ensure_comgr();
  v_pair = caml_alloc_tuple(2);
  Store_field(v_pair, 0, Val_int((int)comgr_major));
  Store_field(v_pair, 1, Val_int((int)comgr_minor));
  CAMLreturn(v_pair);
}

/* Compile HIP source to an HSA code object. Returns [Ok lib] or
   [Error message] where the message carries the comgr status string and the
   compile log. The source and architecture are copied to C memory so the
   OCaml runtime lock can be released during compilation. */
CAMLprim value caml_tolk_amd_comgr_compile(value v_src, value v_arch) {
  CAMLparam2(v_src, v_arch);
  CAMLlocal3(v_result, v_payload, v_msg);
  ensure_comgr();

  size_t src_len = caml_string_length(v_src);
  char *src = (char *)malloc(src_len + 1);
  if (src == NULL) caml_failwith("comgr source allocation failed");
  memcpy(src, String_val(v_src), src_len + 1);
  char *arch = strdup(String_val(v_arch));
  if (arch == NULL) {
    free(src);
    caml_failwith("comgr arch allocation failed");
  }

  char *obj = NULL, *err = NULL;
  size_t obj_size = 0;
  int rc;
  caml_release_runtime_system();
  rc = comgr_compile_hip(src, src_len, arch, &obj, &obj_size, &err);
  caml_acquire_runtime_system();
  free(arch);
  free(src);

  if (rc != 0) {
    v_msg = caml_copy_string(err != NULL ? err : "comgr fail");
    free(err);
    v_result = caml_alloc(1, 1); /* Error */
    Store_field(v_result, 0, v_msg);
    CAMLreturn(v_result);
  }

  v_payload = caml_alloc_string(obj_size);
  memcpy(Bytes_val(v_payload), obj, obj_size);
  free(obj);
  v_result = caml_alloc(1, 0); /* Ok */
  Store_field(v_result, 0, v_payload);
  CAMLreturn(v_result);
}
