/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What OCaml's Unix lacks for an agent: taking a launcher's report
   descriptor by its number, and the errors of accept it does not name. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <errno.h>
#include <limits.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
extern value caml_win32_alloc_handle(HANDLE);
#else
#include <fcntl.h>
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
