/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

/* The program that runs comgr compiles in a process of its own, embedded in
   tolk and spawned by Compiler_amd. comgr serialises every compile in a process
   on a mutex of its own: a process per worker lets compiles run at once. It
   serves compiles until its input ends, so that loading comgr, about 10 ms of a
   process's first compile, is paid once per process.

     worker COMGR

   COMGR is comgr's library. A request holds the ISA, the assemble flag ("0" or
   "1"), the count then the options of the compile, the count then the options
   of code generation, the major and minor version from which comgr is given
   the next options in both steps, the count then those options, each
   NUL-terminated, then the source. A reply holds the code object, as
   compiler_worker.h says. Diagnostics (comgr's, LLVM's) go to standard
   error. */

#include "compiler_worker.h"

/* comgr, whose functions are declared here and found in the library at run
   time. */

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

/* comgr's version, once loaded. */
static struct {
  uint64_t major, minor;
} version;

/* comgr keeps the code it generates in a cache of its own, under
   ~/.cache/comgr up to 5 GB unless AMD_COMGR_CACHE_DIR says otherwise: from
   ROCm 7.0 unless AMD_COMGR_CACHE is 0, in ROCm 6.4 when it is set to another
   value. That is work and disk the caller did not place, beside tolk's own
   cache of code objects. It is disabled before the library loads, which reads
   the variable once. */
static void disable_cache(void) {
#ifdef _WIN32
  _putenv_s("AMD_COMGR_CACHE", "0");
#else
  setenv("AMD_COMGR_CACHE", "0", 1);
#endif
}

/* Why comgr did not load, or "" once it has. */
static char load_error[4096];

/* Loads comgr, or says why it cannot in [load_error]. */
static bool load(const char *path) {
  void *lib = open_library(path);
  if (lib == NULL) {
    snprintf(load_error, sizeof load_error, "comgr not available: %s",
             library_error(path));
    return false;
  }
  for (size_t i = 0; i < sizeof symbols / sizeof *symbols; i++)
    if ((*symbols[i].fn = find_symbol(lib, symbols[i].name)) == NULL) {
      snprintf(load_error, sizeof load_error,
               "comgr not available: %s: undefined symbol %s", path,
               symbols[i].name);
      return false;
    }
  comgr.get_version(&version.major, &version.minor);
  if (version.major >= 3)
    enums.language_hip = 3, enums.compile_source_with_device_libs_to_bc = 12,
    enums.codegen_bc_to_relocatable = 4,
    enums.link_relocatable_to_executable = 7,
    enums.assemble_source_to_relocatable = 8;
  else
    enums.language_hip = 4, enums.compile_source_with_device_libs_to_bc = 15,
    enums.codegen_bc_to_relocatable = 6,
    enums.link_relocatable_to_executable = 9,
    enums.assemble_source_to_relocatable = 10;
  return true;
}

/* The first data of [kind] in [set], or NULL. */
static char *get_data(data_set_t set, uint32_t kind, size_t *size) {
  data_t data;
  char *bytes = NULL;
  if (comgr.action_data_get_data(set, kind, 0, &data) != 0) return NULL;
  if (comgr.get_data(data, size, NULL) == 0 &&
      (bytes = malloc(*size + 1)) != NULL &&
      comgr.get_data(data, size, bytes) != 0) {
    free(bytes);
    bytes = NULL;
  }
  comgr.release_data(data);
  return bytes;
}

/* Compiles, or assembles, [src] and replies with the code object or why
   comgr failed. A step that fails leaves the steps after it undone. comgr's
   objects are released before it returns: the process serves other compiles. */
static void compile_hip(const char *src, size_t src_size, const char *isa,
                        bool assemble, struct options compile,
                        struct options codegen) {
  action_info_t info;
  data_set_t sets[4]; /* source, bitcode, relocatable, executable */
  data_t data_src;
  bool has_info = false, has_src = false;
  int made = 0;
  status_t status;
  char *log = NULL, *lib = NULL;
  size_t log_size = 0, size;
  /* Options stay set from one action to the next: linking clears them. */
  const char *no_option = "";
#define CHECK(call)                                                            \
  if ((status = (call)) != 0) goto fail
#define ACTION(action, from, to, message)                                      \
  if ((status = comgr.do_action(action, info, from, to)) != 0) {               \
    log = get_data(to, DATA_KIND_LOG, &log_size);                              \
    reply_error("%s\n%.*s", message, (int)log_size, log != NULL ? log : "");   \
    goto done;                                                                 \
  }
  CHECK(comgr.create_action_info(&info));
  has_info = true;
  CHECK(comgr.action_info_set_language(info, enums.language_hip));
  CHECK(comgr.action_info_set_isa_name(info, isa));
  CHECK(comgr.action_info_set_logging(info, true));
  for (; made < 4; made++) CHECK(comgr.create_data_set(&sets[made]));
  CHECK(comgr.create_data(DATA_KIND_SOURCE, &data_src));
  has_src = true;
  CHECK(comgr.set_data(data_src, src_size, src));
  CHECK(comgr.set_data_name(data_src, assemble ? "<null>.s" : "<null>"));
  CHECK(comgr.data_set_add(sets[0], data_src));
  if (assemble) {
    ACTION(enums.assemble_source_to_relocatable, sets[0], sets[2],
           "assemble failed");
  } else {
    CHECK(comgr.action_info_set_option_list(info, compile.list, compile.length));
    ACTION(enums.compile_source_with_device_libs_to_bc, sets[0], sets[1],
           "compile failed");
    CHECK(comgr.action_info_set_option_list(info, codegen.list, codegen.length));
    ACTION(enums.codegen_bc_to_relocatable, sets[1], sets[2],
           "code generation failed");
  }
  CHECK(comgr.action_info_set_option_list(info, &no_option, 1));
  ACTION(enums.link_relocatable_to_executable, sets[2], sets[3],
         "link failed");
#undef ACTION
#undef CHECK
  if ((lib = get_data(sets[3], DATA_KIND_EXECUTABLE, &size)) != NULL) {
    reply_with("ok", lib, size);
    goto done;
  }
  status = 1; /* AMD_COMGR_STATUS_ERROR */
fail:;
  const char *reason = NULL;
  if (comgr.status_string(status, &reason) != 0 || reason == NULL)
    reason = "unknown status";
  reply_error("comgr fail %u, %s", (unsigned)status, reason);
done:
  free(lib);
  free(log);
  if (has_src) comgr.release_data(data_src);
  while (made > 0) comgr.destroy_data_set(sets[--made]);
  if (has_info) comgr.destroy_action_info(info);
}

/* Replies to the request [request], [size] bytes. */
static void answer(const char *request, size_t size) {
  const char *at = request, *end = request + size, *isa = field(&at, end),
             *assemble = field(&at, end), *from_major, *from_minor;
  struct options compile = {NULL, 0}, codegen = {NULL, 0}, later = {NULL, 0};
  if (isa == NULL || assemble == NULL || !option_list(&at, end, &compile) ||
      !option_list(&at, end, &codegen) ||
      (from_major = field(&at, end)) == NULL ||
      (from_minor = field(&at, end)) == NULL ||
      !option_list(&at, end, &later))
    reply_error("the request is malformed");
  else {
    uint64_t m = strtoull(from_major, NULL, 10),
             n = strtoull(from_minor, NULL, 10);
    if ((version.major > m || (version.major == m && version.minor >= n)) &&
        (!join(&compile, later) || !join(&codegen, later)))
      reply_error("out of memory");
    else
      compile_hip(at, end - at, isa, assemble[0] == '1', compile, codegen);
  }
  free(compile.list);
  free(codegen.list);
  free(later.list);
}

int main(int argc, char **argv) {
  serve(argc, argv, "COMGR");
  disable_cache();
  bool loaded = load(argv[1]);
  size_t size;
  for (char *request; (request = next_request(&size)) != NULL;) {
    if (loaded)
      answer(request, size);
    else
      reply_error("%s", load_error);
    free(request);
  }
  return 0;
}
