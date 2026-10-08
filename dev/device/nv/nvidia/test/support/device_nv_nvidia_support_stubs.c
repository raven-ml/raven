/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The GPU lock and the process's files. Every stub holds the runtime: none
   blocks. Off Linux the file limit is not the suites' concern: the path opens
   no GPU there. */

#define _GNU_SOURCE

#define CAML_NAME_SPACE
#include <caml/fail.h>
#include <caml/mlvalues.h>

#if !defined(_WIN32)
#include <errno.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

/* Whether the exclusive lock of the file [v_path] was taken without
   waiting. It is held until the process exits. */
value device_nv_nvidia_test_lock(value v_path) {
#if defined(_WIN32)
  (void)v_path;
  return Val_false;
#else
  int fd = open(String_val(v_path), O_RDONLY | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) return Val_false;
  if (flock(fd, LOCK_EX | LOCK_NB) == 0) return Val_true;
  close(fd);
  return Val_false;
#endif
}

#if !defined(_WIN32)
/* The file numbers the suites look at: past the limits they set. */
#define FILES 4096

static int is_open(int fd) { return fcntl(fd, F_GETFD) != -1 || errno != EBADF; }
#endif

/* The number of the process's open files below FILES. */
value device_nv_nvidia_test_files(value unit) {
  (void)unit;
  int n = 0;
#if !defined(_WIN32)
  for (int fd = 0; fd < FILES; fd++) n += is_open(fd);
#endif
  return Val_int(n);
}

/* The soft limit on the process's files. */
value device_nv_nvidia_test_limit(value unit) {
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
value device_nv_nvidia_test_set_limit(value v_n) {
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
value device_nv_nvidia_test_limit_for(value v_k) {
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
