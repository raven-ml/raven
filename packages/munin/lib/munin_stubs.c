/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/mlvalues.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

/* This process's ID. On Windows Unix.getpid returns a handle to the process,
   which no other program names it by. */
CAMLprim value caml_munin_getpid(value unit) {
  (void)unit;
#if defined(_WIN32)
  return Val_long(GetCurrentProcessId());
#else
  return Val_long(getpid());
#endif
}
