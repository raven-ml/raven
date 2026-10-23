/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

/* The program that runs NVRTC compiles in a process of its own, embedded in
   tolk and spawned by Compiler_cuda. NVRTC's assembler crashes on some valid
   sources: in a process of its own, the crash ends that process only. It
   serves compiles until its input ends, so that NVRTC's setup, about 100 ms of
   a process's first compile and most of it starting CUDA's driver, is paid
   once per process.

     worker NVRTC

   NVRTC is NVRTC's library. A request holds the output ("1" for PTX, "0" for
   a cubin), the count then the options, the major and minor version from which
   NVRTC takes the next options, the count then those options, each
   NUL-terminated, then the source. A reply holds the PTX or cubin, as
   compiler_worker.h says. */

#include "compiler_worker.h"

/* NVRTC, whose functions are declared here and found in the library at run
   time. */

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
} symbols[] = {SYMBOL(Version),        SYMBOL(GetErrorString),
               SYMBOL(CreateProgram),  SYMBOL(CompileProgram),
               SYMBOL(DestroyProgram), SYMBOL(GetProgramLogSize),
               SYMBOL(GetPTXSize),     SYMBOL(GetCUBINSize),
               SYMBOL(GetProgramLog),  SYMBOL(GetPTX),
               SYMBOL(GetCUBIN)};
#undef SYMBOL

/* NVRTC 13.0 to 13.4 keep compiled binaries in a cache of their own, shared
   by every process of the user, whose key ignores some of the options: the
   math ones --use_fast_math, --prec-sqrt, --prec-div and --ftz, so that a
   binary built under other math is served, and -I, -D and --std, so that a
   kernel including CUDA's headers is served a binary built against others
   after a change of CUDA_PATH. tolk's own cache keys binaries by every option.
   NVRTC's is disabled before the library loads. */
static void disable_cache(void) {
#ifdef _WIN32
  _putenv_s("CUDA_CACHE_DISABLE", "1");
#else
  setenv("CUDA_CACHE_DISABLE", "1", 1);
#endif
}

/* Why NVRTC did not load, or "" once it has. */
static char load_error[4096];

/* Loads NVRTC and gives its version, or says why it cannot in
   [load_error]. */
static bool load(const char *path, int *major, int *minor) {
  void *lib = open_library(path);
  if (lib == NULL) {
    snprintf(load_error, sizeof load_error, "failed to load library nvrtc: %s",
             library_error(path));
    return false;
  }
  for (size_t i = 0; i < sizeof symbols / sizeof *symbols; i++)
    if ((*symbols[i].fn = find_symbol(lib, symbols[i].name)) == NULL) {
      snprintf(load_error, sizeof load_error,
               "failed to load library nvrtc: %s: undefined symbol %s", path,
               symbols[i].name);
      return false;
    }
  nvrtcResult status = nvrtc.Version(major, minor);
  if (status != 0) {
    snprintf(load_error, sizeof load_error,
             "failed to load library nvrtc: Nvrtc Error %d, %s\n", status,
             nvrtc.GetErrorString(status));
    return false;
  }
  return true;
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

/* Compiles [src] with [options] and replies with its PTX if [ptx], else its
   cubin, or with NVRTC's error and log. */
static void compile(const char *src, bool ptx, struct options options) {
  nvrtcProgram prog;
  char *lib = NULL, *log = NULL;
  size_t size = 0, log_size = 0;
  nvrtcResult status =
      nvrtc.CreateProgram(&prog, src, "<null>", 0, NULL, NULL);
  if (status == 0) {
    status = nvrtc.CompileProgram(prog, (int)options.length, options.list);
    if (status != 0) {
      nvrtcResult log_status;
      log = get_bytes(prog, nvrtc.GetProgramLogSize, nvrtc.GetProgramLog,
                      &log_size, &log_status);
    } else
      lib = get_bytes(prog, ptx ? nvrtc.GetPTXSize : nvrtc.GetCUBINSize,
                      ptx ? nvrtc.GetPTX : nvrtc.GetCUBIN, &size, &status);
    nvrtc.DestroyProgram(&prog);
  }
  /* The log ends with its NUL. */
  if (status != 0)
    reply_error("Nvrtc Error %d, %s\n%s", status, nvrtc.GetErrorString(status),
                log != NULL && log_size > 0 ? log : "");
  else if (lib == NULL)
    reply_error("out of memory");
  else
    reply_with("ok", lib, size);
  free(lib);
  free(log);
}

/* Appends [later] to [*o], whose list it replaces. */
static bool join(struct options *o, struct options later) {
  const char **list = calloc(o->length + later.length + 1, sizeof *list);
  if (list == NULL) return false;
  memcpy(list, o->list, o->length * sizeof *list);
  memcpy(list + o->length, later.list, later.length * sizeof *list);
  free(o->list);
  o->list = list;
  o->length += later.length;
  return true;
}

/* Replies to the request [request], [size] bytes, of NVRTC [major].[minor]. */
static void answer(const char *request, size_t size, int major, int minor) {
  const char *at = request, *end = request + size, *ptx = field(&at, end),
             *from_major, *from_minor;
  struct options options = {NULL, 0}, later = {NULL, 0};
  if (ptx == NULL || !option_list(&at, end, &options) ||
      (from_major = field(&at, end)) == NULL ||
      (from_minor = field(&at, end)) == NULL ||
      !option_list(&at, end, &later))
    reply_error("the request is malformed");
  else {
    long m = strtol(from_major, NULL, 10), n = strtol(from_minor, NULL, 10);
    if ((major > m || (major == m && minor >= n)) && !join(&options, later))
      reply_error("out of memory");
    else
      compile(at, ptx[0] == '1', options);
  }
  free(options.list);
  free(later.list);
}

int main(int argc, char **argv) {
  serve(argc, argv, "NVRTC");
  disable_cache();
  int major = 0, minor = 0;
  bool loaded = load(argv[1], &major, &minor);
  size_t size;
  for (char *request; (request = next_request(&size)) != NULL;) {
    if (loaded)
      answer(request, size, major, minor);
    else
      reply_error("%s", load_error);
    free(request);
  }
  return 0;
}
