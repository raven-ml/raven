/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The process's limit on its address space, which the NVIDIA path's walk
   lowers in a forked child. */

#define _GNU_SOURCE

#include <errno.h>

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>

#if !defined(_WIN32)
#include <sys/resource.h>
#endif

/* [set_space n] sets the soft limit on the process's address space to [n]
   bytes, or none for -1: 0, or the errno; -1 on Windows. Holds the runtime:
   setrlimit does not block. */
value rig_nv_nvidia_test_set_space(value v_n) {
#if defined(_WIN32)
  (void)v_n;
  return Val_int(-1);
#else
  struct rlimit r;
  if (getrlimit(RLIMIT_AS, &r) != 0) return Val_int(errno);
  r.rlim_cur = Long_val(v_n) < 0 ? RLIM_INFINITY : (rlim_t)Long_val(v_n);
  return Val_int(setrlimit(RLIMIT_AS, &r) == 0 ? 0 : errno);
#endif
}
