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

#ifdef _WIN32
#include <io.h>
#include <windows.h>
extern value caml_win32_alloc_handle(HANDLE);
#else
#include <fcntl.h>
#endif

/* [report_open n] is descriptor [n], kept from the programs the process
   runs and, where the system can say so, raising no SIGPIPE on a write:
   [None] if [n] is no open descriptor. Holds the runtime. */
value caml_rig_remote_report_open(value vn) {
  CAMLparam1(vn);
  CAMLlocal1(fd);
  int n = Int_val(vn);
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
