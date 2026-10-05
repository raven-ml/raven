/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

/* The program that runs one comgr compile in a process of its own, embedded in
   tolk and spawned by Compiler_amd. comgr serialises every compile in a process
   on a mutex of its own: a process per compile lets compiles run at once.

     worker COMGR PARENT REQUEST REPLY

   COMGR is comgr's library, PARENT the spawning process's pid. REQUEST holds
   the ISA, the assemble flag ("0" or "1"), the count then the options of the
   compile, the count then the options of code generation, each
   NUL-terminated, then the source to the end of the file. REPLY is written
   once, at the end:

     ok N\n<N bytes of code object>     or     error N\n<N bytes of message>

   so that a reply is complete iff it is N bytes past its header. Diagnostics
   (comgr's, LLVM's) go to standard output and error. */

#include <signal.h>
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
#include <dirent.h>
#include <dlfcn.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
static void *open_library(const char *path) {
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}
static void *find_symbol(void *lib, const char *name) {
  return dlsym(lib, name);
}
static const char *library_error(void) { return dlerror(); }
#endif

/* The reply */

static const char *reply_path;

static void reply_with(const char *kind, const char *data, size_t size) {
  FILE *f = fopen(reply_path, "wb");
  if (f == NULL) return;
  fprintf(f, "%s %zu\n", kind, size);
  fwrite(data, 1, size, f);
  fclose(f);
}

static void reply_error(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int n = vsnprintf(NULL, 0, fmt, args);
  va_end(args);
  char *message = n < 0 ? NULL : malloc(n + 1);
  if (message == NULL) {
    reply_with("error", "out of memory", 13);
    return;
  }
  va_start(args, fmt);
  vsnprintf(message, n + 1, fmt, args);
  va_end(args);
  reply_with("error", message, n);
}

#ifndef _WIN32
/* Closes the descriptors from 3 on that /dev/fd lists, or else those below
   the limit of open files: the limit can be in the millions. */
static void close_listed(void) {
  int fds[1024], n;
  do {
    DIR *dir = opendir("/dev/fd");
    if (dir == NULL) {
      struct rlimit files;
      long last = getrlimit(RLIMIT_NOFILE, &files) == 0 &&
                          files.rlim_cur != RLIM_INFINITY
                      ? (long)files.rlim_cur
                      : 1024;
      for (long fd = 3; fd < last; fd++) close(fd);
      return;
    }
    int listing = dirfd(dir);
    n = 0;
    for (struct dirent *e; n < 1024 && (e = readdir(dir)) != NULL;) {
      int fd = atoi(e->d_name);
      if (fd >= 3 && fd != listing) fds[n++] = fd;
    }
    closedir(dir);
    for (int i = 0; i < n; i++) close(fds[i]);
  } while (n == 1024);
}
#endif

/* The process's state, reset before anything else: what the spawner leaves
   open, ignored, blocked or unlimited would otherwise reach the compile. */
static void reset(void) {
#ifndef _WIN32
  /* Descriptors opened without close-on-exec (sockets, devices) would live as
     long as the compile. */
  int closed = -1;
#ifdef SYS_close_range
  closed = syscall(SYS_close_range, 3, ~0U, 0);
#endif
  if (closed != 0) close_listed();
  /* An ignored signal survives exec, and a blocked mask is inherited. */
  sigset_t none;
  sigemptyset(&none);
  sigprocmask(SIG_SETMASK, &none, NULL);
  for (int s = 1; s < NSIG; s++) signal(s, SIG_DFL);
  /* A terminal's Ctrl-C goes to the program, which decides. */
  setpgid(0, 0);
  /* A crash is an outcome the parent reports: no core file. */
  struct rlimit no_core = {0, 0};
  setrlimit(RLIMIT_CORE, &no_core);
#endif
}

#ifdef __linux__
/* The compile dies with the thread that spawned it, or exits if that thread
   died before it asked. */
static void die_with(long parent) {
  prctl(PR_SET_PDEATHSIG, SIGKILL);
  if (getppid() != parent) _exit(1);
}
#endif

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

/* Loads comgr, or replies why it cannot. */
static bool load(const char *path) {
  void *lib = open_library(path);
  if (lib == NULL) {
    reply_error("comgr not available: %s", library_error());
    return false;
  }
  for (size_t i = 0; i < sizeof symbols / sizeof *symbols; i++)
    if ((*symbols[i].fn = find_symbol(lib, symbols[i].name)) == NULL) {
      reply_error("comgr not available: %s: undefined symbol %s", path,
                  symbols[i].name);
      return false;
    }
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

struct options {
  const char **list;
  size_t length;
};

/* Compiles, or assembles, [src] and replies with the code object or why
   comgr failed. */
static void compile_hip(const char *src, size_t src_size, const char *isa,
                        bool assemble, struct options compile,
                        struct options codegen) {
  action_info_t info;
  data_set_t sets[4]; /* source, bitcode, relocatable, executable */
  data_t data_src;
  status_t status;
  char *log = NULL, *lib;
  size_t log_size = 0, size;
  /* Options stay set from one action to the next: linking clears them. */
  const char *no_option = "";
  /* The process ends after the reply: comgr's objects are never released. */
#define CHECK(call)                                                            \
  if ((status = (call)) != 0) goto fail
#define ACTION(action, from, to, message)                                      \
  if ((status = comgr.do_action(action, info, from, to)) != 0) {               \
    log = get_data(to, DATA_KIND_LOG, &log_size);                              \
    reply_error("%s\n%.*s", message, (int)log_size, log != NULL ? log : "");   \
    return;                                                                    \
  }
  CHECK(comgr.create_action_info(&info));
  CHECK(comgr.action_info_set_language(info, enums.language_hip));
  CHECK(comgr.action_info_set_isa_name(info, isa));
  CHECK(comgr.action_info_set_logging(info, true));
  for (int i = 0; i < 4; i++) CHECK(comgr.create_data_set(&sets[i]));
  CHECK(comgr.create_data(DATA_KIND_SOURCE, &data_src));
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
    CHECK(comgr.do_action(enums.codegen_bc_to_relocatable, info, sets[1],
                          sets[2]));
  }
  CHECK(comgr.action_info_set_option_list(info, &no_option, 1));
  CHECK(comgr.do_action(enums.link_relocatable_to_executable, info, sets[2],
                        sets[3]));
#undef ACTION
#undef CHECK
  if ((lib = get_data(sets[3], DATA_KIND_EXECUTABLE, &size)) != NULL) {
    reply_with("ok", lib, size);
    return;
  }
  status = 1; /* AMD_COMGR_STATUS_ERROR */
fail:;
  const char *reason = NULL;
  if (comgr.status_string(status, &reason) != 0 || reason == NULL)
    reason = "unknown status";
  reply_error("comgr fail %u, %s", (unsigned)status, reason);
}

/* The request */

/* The whole file at [path], or NULL. */
static char *read_file(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) return NULL;
  size_t capacity = 1 << 16;
  char *bytes = malloc(capacity);
  *size = 0;
  for (size_t n; bytes != NULL &&
                 (n = fread(bytes + *size, 1, capacity - *size, f)) > 0;) {
    *size += n;
    if (*size == capacity) bytes = realloc(bytes, capacity *= 2);
  }
  fclose(f);
  return bytes;
}

/* The next NUL-terminated field of [*at], before [end], or NULL. */
static const char *field(const char **at, const char *end) {
  const char *f = *at, *nul = memchr(f, '\0', end - f);
  if (nul == NULL) return NULL;
  *at = nul + 1;
  return f;
}

static bool option_list(const char **at, const char *end, struct options *o) {
  const char *count = field(at, end);
  if (count == NULL) return false;
  o->length = strtoul(count, NULL, 10);
  if ((o->list = calloc(o->length + 1, sizeof *o->list)) == NULL) return false;
  for (size_t i = 0; i < o->length; i++)
    if ((o->list[i] = field(at, end)) == NULL) return false;
  return true;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s COMGR PARENT REQUEST REPLY\n", argv[0]);
    return 2;
  }
  reset();
#ifdef __linux__
  die_with(strtol(argv[2], NULL, 10));
#endif
  reply_path = argv[4];
  size_t size;
  char *request = read_file(argv[3], &size);
  if (request == NULL) {
    reply_error("the request %s does not read", argv[3]);
    return 1;
  }
  const char *at = request, *end = request + size, *isa = field(&at, end),
             *assemble = field(&at, end);
  struct options compile, codegen;
  if (isa == NULL || assemble == NULL || !option_list(&at, end, &compile) ||
      !option_list(&at, end, &codegen)) {
    reply_error("the request %s is malformed", argv[3]);
    return 1;
  }
  if (!load(argv[1])) return 1;
  compile_hip(at, end - at, isa, assemble[0] == '1', compile, codegen);
  return 0;
}
