/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The address-space limit of test_widths, which runs on Linux alone. */

#include <errno.h>
#include <sys/resource.h>

#include <caml/mlvalues.h>

/* Lowers the process's soft limit of its address space to [n] bytes where
   the limit is higher, so that it never raises a limit set outside and
   stays within the hard limit: 0, or the errno of a refusal. Holds the
   runtime: setrlimit does not block. RLIM_INFINITY is rlim_t's largest
   value. */
value nx_test_limit_address_space(value v_n) {
  struct rlimit l;
  if (getrlimit(RLIMIT_AS, &l) != 0) return Val_int(errno);
  rlim_t n = (rlim_t)Long_val(v_n);
  if (n < l.rlim_cur) l.rlim_cur = n;
  return Val_int(setrlimit(RLIMIT_AS, &l) == 0 ? 0 : errno);
}
