/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What OCaml's Unix lacks for an agent: taking a launcher's report
   descriptor by its number, writing a line on it with no poll point, and the
   errors of accept it does not name. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <caml/unixsupport.h>
#include <io.h>
#include <windows.h>
extern value caml_win32_alloc_handle(HANDLE);
#else
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#endif

/* [report_open n] is descriptor [n], kept from the programs the process
   runs and, where the system can say so, raising no SIGPIPE on a write:
   [None] if [n] is no open descriptor, as one past an [int] is. Holds the
   runtime. */
value caml_rig_remote_report_open(value vn) {
  CAMLparam1(vn);
  CAMLlocal1(fd);
  intnat v = Long_val(vn);
  if (v < 0 || v > INT_MAX) CAMLreturn(Val_none);
  int n = (int)v;
#ifdef _WIN32
  intptr_t h = _get_osfhandle(n);
  if (h == -1 || !SetHandleInformation((HANDLE)h, HANDLE_FLAG_INHERIT, 0))
    CAMLreturn(Val_none);
  fd = caml_win32_alloc_handle((HANDLE)h);
#else
  int flags = fcntl(n, F_GETFD);
  if (flags < 0) CAMLreturn(Val_none);
#ifdef F_SETNOSIGPIPE
  /* macOS raises a pipe's SIGPIPE on the process, where any thread that
     does not block it takes it: the descriptor itself raises none. */
  (void)fcntl(n, F_SETNOSIGPIPE, 1);
#endif
  if (fcntl(n, F_SETFD, flags | FD_CLOEXEC) != 0) CAMLreturn(Val_none);
  fd = Val_int(n);
#endif
  CAMLreturn(caml_alloc_some(fd));
}

/* Writes [n] bytes of [b] on [fd], up to the first failure other than an
   interruption. */
#ifdef _WIN32
static void write_all(HANDLE fd, const char *b, size_t n) {
  while (n > 0) {
    DWORD k = 0;
    DWORD ask = n > MAXDWORD ? MAXDWORD : (DWORD)n;
    if (!WriteFile(fd, b, ask, &k, NULL) || k == 0) return;
    b += k;
    n -= k;
  }
}
#else
static void write_all(int fd, const char *b, size_t n) {
  while (n > 0) {
    ssize_t k = write(fd, b, n);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return;
    b += k;
    n -= (size_t)k;
  }
}

/* [write_all], with SIGPIPE blocked in this thread: a pipe whose reader is
   gone raises it on the writing thread, which would end the process. The
   one the write raised is taken before the signal is unblocked. */
static void write_quiet(int fd, const char *b, size_t n) {
  sigset_t pipe_set, old, pending;
  sigemptyset(&pipe_set);
  sigaddset(&pipe_set, SIGPIPE);
  pthread_sigmask(SIG_BLOCK, &pipe_set, &old);
  sigpending(&pending);
  int was_pending = sigismember(&pending, SIGPIPE);
  write_all(fd, b, n);
  sigpending(&pending);
  if (!was_pending && sigismember(&pending, SIGPIPE)) {
    int sig;
    sigwait(&pipe_set, &sig);
  }
  pthread_sigmask(SIG_SETMASK, &old, NULL);
}
#endif

/* [process_id ()] is this process's id, as caml_rig_remote_report_line
   compares it: on Windows, Unix.getpid answers a handle to the process. */
value caml_rig_remote_process_id(value unit) {
  (void)unit;
#ifdef _WIN32
  return Val_long((intnat)GetCurrentProcessId());
#else
  return Val_long((intnat)getpid());
#endif
}

/* [report_line fd pid line] writes [line] on [fd] if this process is [pid],
   ignoring every error. It runs no OCaml code: it releases the runtime with
   caml_enter_blocking_section_no_pending, so the signal handlers pending run
   at the caller's next poll point, and with no memory for a copy of [line]
   it writes holding the runtime. */
value caml_rig_remote_report_line(value vfd, value vpid, value vline) {
#ifdef _WIN32
  if ((intnat)GetCurrentProcessId() != Long_val(vpid)) return Val_unit;
  HANDLE fd = Handle_val(vfd);
#else
  if ((intnat)getpid() != Long_val(vpid)) return Val_unit;
  int fd = Int_val(vfd);
#endif
  size_t n = caml_string_length(vline);
  char *b = malloc(n ? n : 1);
  if (b == NULL) {
#ifdef _WIN32
    write_all(fd, String_val(vline), n);
#else
    write_quiet(fd, String_val(vline), n);
#endif
    return Val_unit;
  }
  memcpy(b, String_val(vline), n);
  caml_enter_blocking_section_no_pending();
#ifdef _WIN32
  write_all(fd, b, n);
#else
  write_quiet(fd, b, n);
#endif
  caml_leave_blocking_section();
  free(b);
  return Val_unit;
}

/* The system's EPROTO and ENONET, -1 where it has none. Linux's accept(2)
   reports a connection's protocol errors through them, which Unix.error
   gives as EUNKNOWNERR. */
value caml_rig_remote_eproto(value unit) {
  (void)unit;
#ifdef EPROTO
  return Val_int(EPROTO);
#else
  return Val_int(-1);
#endif
}

value caml_rig_remote_enonet(value unit) {
  (void)unit;
#ifdef ENONET
  return Val_int(ENONET);
#else
  return Val_int(-1);
#endif
}
