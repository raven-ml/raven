/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What OCaml's Unix and threads lack for an agent: taking a launcher's
   report descriptor by its number, writing a line on it with no poll point,
   the errors of accept it does not name, and a wait for a change that
   returns to OCaml after a bounded time. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <caml/threads.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "rig_remote_proxy_stubs.h"

#ifdef _WIN32
#include <caml/unixsupport.h>
#include <io.h>
#include <windows.h>
extern value caml_win32_alloc_handle(HANDLE);
#else
#include <fcntl.h>
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

/* Changes

   A count of an agent's changes, which a wait for the next one bounds in
   time: OCaml's Condition has no timed wait, and a wait that never returns
   to OCaml runs no signal's handler. A caller reads the count with the
   state it guards, under its own lock, and waits from that count: a change
   after the read ends the wait. */

struct changes {
  pthread_mutex_t mu;
  pthread_cond_t cv;
  intnat count;
};

#define Changes_val(v) (*(struct changes **)Data_custom_val(v))

static void changes_free(value v) {
  struct changes *c = Changes_val(v);
  if (c == NULL) return;
  pthread_cond_destroy(&c->cv);
  pthread_mutex_destroy(&c->mu);
  free(c);
}

static struct custom_operations changes_ops = {
    "rig.remote.changes",       changes_free,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* A new count of no change. Raises Out_of_memory if the C heap has no room
   for it. */
value caml_rig_remote_changes(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(v);
  v = caml_alloc_custom(&changes_ops, sizeof(struct changes *), 0, 1);
  Changes_val(v) = NULL;
  struct changes *c = malloc(sizeof *c);
  if (c == NULL) caml_raise_out_of_memory();
  pthread_mutex_init(&c->mu, NULL);
  rig_remote_cond_init(&c->cv);
  c->count = 0;
  Changes_val(v) = c;
  CAMLreturn(v);
}

/* Counts one more change and wakes the waits. Holds the runtime. */
value caml_rig_remote_change(value v) {
  struct changes *c = Changes_val(v);
  pthread_mutex_lock(&c->mu);
  c->count++;
  pthread_cond_broadcast(&c->cv);
  pthread_mutex_unlock(&c->mu);
  return Val_unit;
}

/* The changes counted so far. Holds the runtime. */
value caml_rig_remote_changes_count(value v) {
  struct changes *c = Changes_val(v);
  pthread_mutex_lock(&c->mu);
  intnat n = c->count;
  pthread_mutex_unlock(&c->mu);
  return Val_long(n);
}

/* Waits until more than [vn] changes are counted, or [RIG_REMOTE_STILL_NS]
   has passed, or the condition wakes spuriously. Releases the runtime. */
value caml_rig_remote_changes_await(value v, value vn) {
  CAMLparam2(v, vn);
  struct changes *c = Changes_val(v);
  intnat n = Long_val(vn);
  caml_release_runtime_system();
  pthread_mutex_lock(&c->mu);
  if (c->count == n)
    rig_remote_wait_ns(&c->cv, &c->mu, RIG_REMOTE_STILL_NS);
  pthread_mutex_unlock(&c->mu);
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}
