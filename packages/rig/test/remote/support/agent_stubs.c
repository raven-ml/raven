/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/mlvalues.h>

#if !defined(_WIN32)
#include <errno.h>
#include <sys/resource.h>
#endif

/* [limit_descriptors n] sets this process's soft limit of its descriptors to
   [n]: 0, the errno, or -1 on Windows, which has no such limit. */
value rig_remote_test_limit_descriptors(value v_n) {
#if defined(_WIN32)
  (void)v_n;
  return Val_int(-1);
#else
  struct rlimit l;
  if (getrlimit(RLIMIT_NOFILE, &l) != 0) return Val_int(errno);
  l.rlim_cur = (rlim_t)Long_val(v_n);
  return Val_int(setrlimit(RLIMIT_NOFILE, &l) == 0 ? 0 : errno);
#endif
}
