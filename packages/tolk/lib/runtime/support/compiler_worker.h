/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What every worker shares: a program that runs one compile in a process of
   its own, embedded in tolk and spawned by Compiler_worker.

     worker LIBRARY PARENT REQUEST REPLY

   LIBRARY is the vendor's library, PARENT the spawning process's pid. REQUEST
   holds the worker's fields, each NUL-terminated, then the source to the end of
   the file. REPLY is written once, at the end:

     ok N\n<N bytes of binary>     or     error N\n<N bytes of message>

   so that a reply is complete iff it is N bytes past its header. Diagnostics
   (the library's) go to standard output and error. */

#define _GNU_SOURCE

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
/* Why [path] did not load, named first, as dlerror names it. A static buffer:
   loading is not concurrent. */
static const char *library_error(const char *path) {
  static char message[1024];
  DWORD code = GetLastError();
  int n = snprintf(message, sizeof message, "%s: ", path);
  if (n < 0 || (size_t)n >= sizeof message) return path;
  DWORD m = FormatMessageA(
      FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, code,
      0, message + n, (DWORD)(sizeof message - n), NULL);
  if (m == 0) {
    snprintf(message + n, sizeof message - n, "error %lu",
             (unsigned long)code);
    return message;
  }
  /* the system's message ends with a line break */
  while (m > 0 && strchr(" \r\n", message[n + m - 1]) != NULL)
    message[n + --m] = '\0';
  return message;
}
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
/* Why [path] did not load: dlerror's message, which names it. */
static const char *library_error(const char *path) {
  const char *e = dlerror();
  return e != NULL ? e : path;
}
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

/* The request */

/* The whole file at [path], NUL-terminated past its [*size] bytes, or NULL. */
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
  if (bytes != NULL) bytes[*size] = '\0';
  return bytes;
}

/* The next NUL-terminated field of [*at], before [end], or NULL. */
static const char *field(const char **at, const char *end) {
  const char *f = *at, *nul = memchr(f, '\0', end - f);
  if (nul == NULL) return NULL;
  *at = nul + 1;
  return f;
}

struct options {
  const char **list;
  size_t length;
};

/* A count, then that many fields. */
static bool option_list(const char **at, const char *end, struct options *o) {
  const char *count = field(at, end);
  if (count == NULL) return false;
  o->length = strtoul(count, NULL, 10);
  if ((o->list = calloc(o->length + 1, sizeof *o->list)) == NULL) return false;
  for (size_t i = 0; i < o->length; i++)
    if ((o->list[i] = field(at, end)) == NULL) return false;
  return true;
}

/* Starts the worker whose library is called [library] in its usage: resets the
   process and returns the request, [*size] bytes long. Exits with 2 on a usage
   error, and with 1 having replied if the request does not read. */
static char *start(int argc, char **argv, const char *library, size_t *size) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s %s PARENT REQUEST REPLY\n", argv[0], library);
    exit(2);
  }
  reset();
#ifdef __linux__
  die_with(strtol(argv[2], NULL, 10));
#endif
  reply_path = argv[4];
  char *request = read_file(argv[3], size);
  if (request == NULL) {
    reply_error("the request %s does not read", argv[3]);
    exit(1);
  }
  return request;
}
