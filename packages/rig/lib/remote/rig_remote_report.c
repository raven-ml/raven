/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* A launcher's report descriptor: what OCaml's Unix lacks to take a
   descriptor known by its number. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <limits.h>

#ifndef _WIN32
#include <fcntl.h>
#endif

/* [report_open n] is descriptor [n], kept from the programs the process
   runs and, where the system can say so, raising no SIGPIPE on a write:
   [None] if [n] is no open descriptor, as one past an [int] is, and on
   Windows, where launchers pass no descriptor by its number. Holds the
   runtime. */
#ifdef _WIN32
value caml_rig_remote_report_open(value vn) {
  (void)vn;
  return Val_none;
}
#else
value caml_rig_remote_report_open(value vn) {
  CAMLparam1(vn);
  CAMLlocal1(fd);
  intnat v = Long_val(vn);
  if (v < 0 || v > INT_MAX) CAMLreturn(Val_none);
  int n = (int)v;
  int flags = fcntl(n, F_GETFD);
  if (flags < 0) CAMLreturn(Val_none);
#ifdef F_SETNOSIGPIPE
  /* macOS raises a pipe's SIGPIPE on the process, where any thread that
     does not block it takes it: the descriptor itself raises none. */
  (void)fcntl(n, F_SETNOSIGPIPE, 1);
#endif
  if (fcntl(n, F_SETFD, flags | FD_CLOEXEC) != 0) CAMLreturn(Val_none);
  fd = Val_int(n);
  CAMLreturn(caml_alloc_some(fd));
}
#endif
