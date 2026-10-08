/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A launcher's variables and report descriptor: what OCaml's Unix lacks to
   take a variable out of the environment, and to write on a descriptor known
   by its number. */

#define _GNU_SOURCE

#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#endif

/* Takes the variable [name] out of the environment. Holds the runtime. */
value caml_rig_remote_unsetenv(value name) {
#ifdef _WIN32
  _putenv_s(String_val(name), "");
#else
  unsetenv(String_val(name));
#endif
  return Val_unit;
}

/* Keeps descriptor [n] from the programs the process runs: [false] if [n]
   is no open descriptor. Holds the runtime. */
value caml_rig_remote_report_open(value vn) {
  int n = Int_val(vn);
#ifdef _WIN32
  intptr_t h = _get_osfhandle(n);
  if (h == -1) return Val_false;
  return Val_bool(SetHandleInformation((HANDLE)h, HANDLE_FLAG_INHERIT, 0));
#else
  int flags = fcntl(n, F_GETFD);
  if (flags < 0) return Val_false;
  return Val_bool(fcntl(n, F_SETFD, flags | FD_CLOEXEC) == 0);
#endif
}

/* Writes [s] on descriptor [n], ignoring every error. A pipe whose reader
   is gone raises SIGPIPE, which would end the process: the signal is
   blocked in this thread while it writes, and the one the write raised is
   taken before it is unblocked. Releases the runtime: a full pipe blocks. */
value caml_rig_remote_report_write(value vn, value s) {
  int n = Int_val(vn);
  size_t len = caml_string_length(s);
  char *buf = malloc(len);
  if (buf == NULL) return Val_unit;
  memcpy(buf, String_val(s), len);
  caml_release_runtime_system();
#ifdef _WIN32
  size_t off = 0;
  while (off < len) {
    int k = _write(n, buf + off, (unsigned)(len - off));
    if (k <= 0) break;
    off += (size_t)k;
  }
#else
  sigset_t pipe, old, pending;
  sigemptyset(&pipe);
  sigaddset(&pipe, SIGPIPE);
  pthread_sigmask(SIG_BLOCK, &pipe, &old);
  sigpending(&pending);
  int was_pending = sigismember(&pending, SIGPIPE);
  size_t off = 0;
  while (off < len) {
    ssize_t k = write(n, buf + off, len - off);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) break;
    off += (size_t)k;
  }
  sigpending(&pending);
  if (!was_pending && sigismember(&pending, SIGPIPE)) {
    int sig;
    sigwait(&pipe, &sig);
  }
  pthread_sigmask(SIG_SETMASK, &old, NULL);
#endif
  caml_acquire_runtime_system();
  free(buf);
  return Val_unit;
}
