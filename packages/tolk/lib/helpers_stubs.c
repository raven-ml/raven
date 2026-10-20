/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Setting a file's times through a descriptor, which OCaml's Unix library
   does only by path ([Unix.utimes]). */

#define _GNU_SOURCE

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>
#include <caml/signals.h>
#include <caml/unixsupport.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/stat.h>
#endif

/* [tolk_touch fd] sets the access and modification times of the file [fd]
   is open on to now. On Windows the handle needs FILE_WRITE_ATTRIBUTES,
   which a descriptor opened for writing has. Raises [Unix.Unix_error]. */
CAMLprim value tolk_touch(value fd)
{
#ifdef _WIN32
  HANDLE h = Handle_val(fd);
  FILETIME now;
  BOOL ok;
  GetSystemTimeAsFileTime(&now);
  caml_enter_blocking_section();
  ok = SetFileTime(h, NULL, &now, &now);
  caml_leave_blocking_section();
  if (!ok) {
    caml_win32_maperr(GetLastError());
    caml_uerror("SetFileTime", Nothing);
  }
#else
  int d = Int_val(fd);
  int r;
  caml_enter_blocking_section();
  r = futimens(d, NULL);
  caml_leave_blocking_section();
  if (r == -1) caml_uerror("futimens", Nothing);
#endif
  return Val_unit;
}
