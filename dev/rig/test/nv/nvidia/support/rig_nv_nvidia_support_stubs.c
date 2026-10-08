/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The process's files and addresses. Every stub holds the runtime: none
   blocks. Off Linux the file limit is not the suites' concern: the path
   opens no GPU there. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>

#define CAML_NAME_SPACE
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
/* The file numbers the suites look at: past the limits they set. */
#define FILES 4096

static int is_open(int fd) { return fcntl(fd, F_GETFD) != -1 || errno != EBADF; }
#endif

/* The number of the process's open files below FILES. */
value rig_nv_nvidia_test_files(value unit) {
  (void)unit;
  int n = 0;
#if !defined(_WIN32)
  for (int fd = 0; fd < FILES; fd++) n += is_open(fd);
#endif
  return Val_int(n);
}

/* The soft limit on the process's files. */
value rig_nv_nvidia_test_limit(value unit) {
  (void)unit;
#if defined(_WIN32)
  return Val_long(0);
#else
  struct rlimit r;
  if (getrlimit(RLIMIT_NOFILE, &r) != 0) caml_failwith("getrlimit");
  return Val_long(r.rlim_cur == RLIM_INFINITY ? -1 : (long)r.rlim_cur);
#endif
}

/* Sets the soft limit on the process's files to [v_n], or none for -1. */
value rig_nv_nvidia_test_set_limit(value v_n) {
#if defined(_WIN32)
  (void)v_n;
#else
  struct rlimit r;
  if (getrlimit(RLIMIT_NOFILE, &r) != 0) caml_failwith("getrlimit");
  r.rlim_cur = Long_val(v_n) < 0 ? RLIM_INFINITY : (rlim_t)Long_val(v_n);
  if (setrlimit(RLIMIT_NOFILE, &r) != 0) caml_failwith("setrlimit");
#endif
  return Val_unit;
}

/* The limit under which the process may open exactly [v_k] more files: a new
   file takes the lowest free number, and the limit bounds the numbers. */
value rig_nv_nvidia_test_limit_for(value v_k) {
  int k = Int_val(v_k), fd = 0;
#if !defined(_WIN32)
  for (int free = 0; fd < FILES; fd++) {
    if (is_open(fd)) continue;
    if (free == k) break;
    free++;
  }
#else
  (void)k;
#endif
  return Val_int(fd);
}

/* Maps [v_n] inaccessible bytes at [v_at], unless something is mapped there:
   [true] if it did. */
value rig_nv_nvidia_test_occupy(value v_at, value v_n) {
#if defined(_WIN32)
  (void)v_at;
  (void)v_n;
  return Val_false;
#else
  void *at = (void *)(uintptr_t)Long_val(v_at);
  void *p = mmap(at, (size_t)Long_val(v_n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) return Val_false;
  if (p != at) {
    munmap(p, (size_t)Long_val(v_n));
    return Val_false;
  }
  return Val_true;
#endif
}

/* Unmaps the [v_n] bytes at [v_at]. */
value rig_nv_nvidia_test_vacate(value v_at, value v_n) {
#if !defined(_WIN32)
  munmap((void *)(uintptr_t)Long_val(v_at), (size_t)Long_val(v_n));
#else
  (void)v_at;
  (void)v_n;
#endif
  return Val_unit;
}
