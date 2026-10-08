/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The machine's GPU lock and the process's files. Every stub but the
   lock's holds the runtime: none blocks. Off Linux the file limit is not
   the suites' concern: the path opens no GPU there. */

#define _GNU_SOURCE

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

/* One try at the exclusive lock of the file [v_path], which the process
   then holds until it exits. A missing file is made writable by every user
   of the machine. Once taken, the file names [v_holder] and the process's
   id, for the processes that wait. Answers [0] once the process holds the
   lock, [-1] after a nap of 100 ms if another process holds it, or the
   errno of a failing call. Releases the runtime for the nap. */
value rig_nv_nvidia_test_lock(value v_path, value v_holder) {
  CAMLparam2(v_path, v_holder);
#if defined(_WIN32)
  CAMLreturn(Val_int(ENOSYS));
#else
  /* The descriptor that holds the lock once taken. The suites take it from
     one domain. */
  static int held = -1;
  if (held >= 0) CAMLreturn(Val_int(0));
  const char *path = String_val(v_path);
  int fd = open(path, O_RDWR | O_CLOEXEC);
  /* O_EXCL: Linux refuses O_CREAT on another user's file in /tmp
     (fs.protected_regular). */
  if (fd < 0 && errno == ENOENT) {
    fd = open(path, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0666);
    if (fd < 0 && errno == EEXIST) fd = open(path, O_RDWR | O_CLOEXEC);
    else if (fd >= 0 && fchmod(fd, 0666) != 0) {
      int e = errno;
      close(fd);
      CAMLreturn(Val_int(e));
    }
  }
  if (fd < 0) CAMLreturn(Val_int(errno));
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    int e = errno;
    close(fd);
    if (e != EWOULDBLOCK) CAMLreturn(Val_int(e));
    struct timespec nap = {0, 100 * 1000 * 1000};
    caml_release_runtime_system();
    nanosleep(&nap, NULL);
    caml_acquire_runtime_system();
    CAMLreturn(Val_int(-1));
  }
  char note[1024] = "";
  snprintf(note, sizeof note, "%s, pid %ld\n", String_val(v_holder),
           (long)getpid());
  size_t len = strlen(note);
  if (ftruncate(fd, 0) != 0 || pwrite(fd, note, len, 0) != (ssize_t)len) {
    int e = errno;
    close(fd);
    CAMLreturn(Val_int(e));
  }
  held = fd;
  CAMLreturn(Val_int(0));
#endif
}

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
