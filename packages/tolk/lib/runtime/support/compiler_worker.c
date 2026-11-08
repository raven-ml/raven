/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What every worker shares, as compiler_worker.h declares it. */

#include "compiler_worker.h"

#include <errno.h>
#include <signal.h>
#include <stdarg.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
void *open_library(const char *path) { return LoadLibraryA(path); }
void *find_symbol(void *lib, const char *name) {
  return (void *)GetProcAddress((HMODULE)lib, name);
}
/* The name first, as dlerror names it. A static buffer: loading is not
   concurrent. */
const char *library_error(const char *path) {
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
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
void *open_library(const char *path) {
  return dlopen(path, RTLD_NOW | RTLD_LOCAL);
}
void *find_symbol(void *lib, const char *name) { return dlsym(lib, name); }
/* dlerror's message, which names it. */
const char *library_error(const char *path) {
  const char *e = dlerror();
  return e != NULL ? e : path;
}
#endif

/* The reply */

/* The descriptor of the replies. */
static int reply_fd = -1;

/* Writes the [size] bytes of [data] to [fd]; a failure ends the process, whose
   parent then reads no whole reply. */
static void write_all(int fd, const char *data, size_t size) {
  while (size > 0) {
#ifdef _WIN32
    int n = _write(fd, data, size > 1 << 30 ? 1 << 30 : (unsigned)size);
#else
    ssize_t n = write(fd, data, size);
#endif
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) exit(1);
    data += n;
    size -= n;
  }
}

void reply_with(const char *kind, const char *data, size_t size) {
  char header[64];
  int n = snprintf(header, sizeof header, "%s %zu\n", kind, size);
  write_all(reply_fd, header, n);
  write_all(reply_fd, data, size);
}

void reply_error(const char *fmt, ...) {
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
  free(message);
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

/* The request */

const char *field(const char **at, const char *end) {
  const char *f = *at, *nul = memchr(f, '\0', end - f);
  if (nul == NULL) return NULL;
  *at = nul + 1;
  return f;
}

bool option_list(const char **at, const char *end, struct options *o) {
  const char *count = field(at, end);
  if (count == NULL) return false;
  o->length = strtoul(count, NULL, 10);
  if ((o->list = calloc(o->length + 1, sizeof *o->list)) == NULL) return false;
  for (size_t i = 0; i < o->length; i++)
    if ((o->list[i] = field(at, end)) == NULL) return false;
  return true;
}

bool join(struct options *o, struct options later) {
  const char **list = calloc(o->length + later.length + 1, sizeof *list);
  if (list == NULL) return false;
  memcpy(list, o->list, o->length * sizeof *list);
  memcpy(list + o->length, later.list, later.length * sizeof *list);
  free(o->list);
  o->list = list;
  o->length += later.length;
  return true;
}

/* Serving */

static int request_fd = -1;

/* Reads [size] bytes into [bytes]: false if the input ends first. */
static bool read_all(char *bytes, size_t size) {
  while (size > 0) {
#ifdef _WIN32
    int n = _read(request_fd, bytes, size > 1 << 30 ? 1 << 30 : (unsigned)size);
#else
    ssize_t n = read(request_fd, bytes, size);
#endif
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return false;
    bytes += n;
    size -= n;
  }
  return true;
}

/* A worker never dies with the thread that spawned it: that thread may end
   long before the process. It ends when its input does, which the parent's
   end closes, with the parent. Between requests it reads its input, which
   ends; during a compile, which may never return, a thread of its own waits
   for the parent's end to close and ends the process. Elsewhere than
   Windows that is the hang-up poll reports; on Windows, whose pipes poll
   cannot wait on, a peek every second that finds the pipe broken. */
#ifdef _WIN32
static DWORD WINAPI watch(LPVOID unused) {
  (void)unused;
  HANDLE requests = (HANDLE)_get_osfhandle(request_fd);
  for (;;) {
    DWORD available;
    if (!PeekNamedPipe(requests, NULL, 0, NULL, &available, NULL) &&
        GetLastError() == ERROR_BROKEN_PIPE)
      _exit(1);
    Sleep(1000);
  }
}
#else
static void *watch(void *unused) {
  (void)unused;
  struct pollfd requests = {request_fd, 0, 0};
  for (;;)
    if (poll(&requests, 1, -1) > 0 &&
        (requests.revents & (POLLHUP | POLLERR | POLLNVAL)) != 0)
      _exit(1);
}
#endif

static bool start_watch(void) {
#ifdef _WIN32
  HANDLE thread = CreateThread(NULL, 0, watch, NULL, 0, NULL);
  if (thread == NULL) return false;
  CloseHandle(thread);
  return true;
#else
  pthread_t thread;
  if (pthread_create(&thread, NULL, watch, NULL) != 0) return false;
  pthread_detach(thread);
  return true;
#endif
}

void serve(int argc, char **argv, const char *library) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s %s TEMP\n", argv[0], library);
    exit(2);
  }
  reset();
  /* The library's temporary files go to TEMP, which the parent removes when
     the worker ends: a compile that aborts leaves its files there. */
#ifdef _WIN32
  _putenv_s("TMP", argv[2]);
  _putenv_s("TEMP", argv[2]);
#else
  setenv("TMPDIR", argv[2], 1);
#endif
#ifdef _WIN32
  _setmode(0, _O_BINARY);
  _setmode(1, _O_BINARY);
  request_fd = _dup(0);
  reply_fd = _dup(1);
  int none = _open("NUL", _O_RDONLY);
  if (request_fd >= 0 && reply_fd >= 0 && none >= 0)
    _dup2(none, 0), _dup2(2, 1);
#else
  request_fd = dup(0);
  reply_fd = dup(1);
  int none = open("/dev/null", O_RDONLY);
  if (request_fd >= 0 && reply_fd >= 0 && none >= 0)
    dup2(none, 0), dup2(2, 1);
#endif
  if (request_fd < 0 || reply_fd < 0 || none < 0) {
    fprintf(stderr, "%s: no descriptor for its requests\n", argv[0]);
    exit(1);
  }
  if (!start_watch()) {
    fprintf(stderr, "%s: no thread to watch its parent\n", argv[0]);
    exit(1);
  }
}

char *next_request(size_t *size) {
  char header[32];
  size_t n = 0;
  for (;;) {
    if (!read_all(header + n, 1)) {
      if (n == 0) return NULL;
      fprintf(stderr, "the input ends inside a request\n");
      exit(1);
    }
    if (header[n] == '\n') break;
    if (++n == sizeof header) {
      fprintf(stderr, "a request's header is malformed\n");
      exit(1);
    }
  }
  header[n] = '\0';
  char *end;
  unsigned long long length = strtoull(header, &end, 10);
  if (n == 0 || *end != '\0' || length >= SIZE_MAX) {
    fprintf(stderr, "a request's header is malformed: %s\n", header);
    exit(1);
  }
  char *bytes = malloc(length + 1);
  if (bytes == NULL) {
    fprintf(stderr, "out of memory for a request of %llu bytes\n", length);
    exit(1);
  }
  if (!read_all(bytes, length)) {
    fprintf(stderr, "the input ends inside a request\n");
    exit(1);
  }
  bytes[length] = '\0';
  *size = length;
  return bytes;
}
