/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the disk's suites ask of the system that OCaml's Unix does not give:
   limits on this process's open files and file sizes, dropping a file's
   cached pages, and the bytes the C heap holds. */

#define _GNU_SOURCE

#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <errno.h>
#include <stddef.h>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RIG_DISK_TEST_ASAN
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define RIG_DISK_TEST_ASAN
#endif

#if defined(RIG_DISK_TEST_ASAN)
#include <sanitizer/allocator_interface.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__GLIBC__)
#include <malloc.h>
#endif

#ifndef _WIN32
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

/* [set_open_files n] sets this process's soft limit of open files to [n]: 0,
   the errno, or -1 on Windows, which has no such limit. Keeps the runtime:
   setrlimit does not block. */
value rig_disk_test_set_open_files(value v_n) {
#ifdef _WIN32
  (void)v_n;
  return Val_int(-1);
#else
  struct rlimit l;
  if (getrlimit(RLIMIT_NOFILE, &l) != 0) return Val_int(errno);
  l.rlim_cur = (rlim_t)Long_val(v_n);
  return Val_int(setrlimit(RLIMIT_NOFILE, &l) == 0 ? 0 : errno);
#endif
}

/* [open_files ()] is this process's soft limit of open files, at most
   [max_int], or -1 on Windows. Keeps the runtime: getrlimit does not block. */
value rig_disk_test_open_files(value unit) {
  (void)unit;
#ifdef _WIN32
  return Val_long(-1);
#else
  struct rlimit l;
  if (getrlimit(RLIMIT_NOFILE, &l) != 0) return Val_long(-1);
  return Val_long(l.rlim_cur > (rlim_t)Max_long ? Max_long : (intnat)l.rlim_cur);
#endif
}

/* [set_file_size n] sets this process's soft limit of a file's size to [n]
   bytes and ignores SIGXFSZ, so growing a file past it fails with EFBIG: 0,
   the errno, or -1 on Windows, which has no such limit. Keeps the runtime:
   setrlimit does not block. */
value rig_disk_test_set_file_size(value v_n) {
#ifdef _WIN32
  (void)v_n;
  return Val_int(-1);
#else
  struct rlimit l;
  if (getrlimit(RLIMIT_FSIZE, &l) != 0) return Val_int(errno);
  l.rlim_cur = (rlim_t)Long_val(v_n);
  if (signal(SIGXFSZ, SIG_IGN) == SIG_ERR) return Val_int(errno);
  return Val_int(setrlimit(RLIMIT_FSIZE, &l) == 0 ? 0 : errno);
#endif
}

/* [drop_pages path] writes the file at [path]'s dirty pages to storage and asks
   the system to drop its cached pages, so that the next read reaches storage:
   0, or the errno. Linux only (posix_fadvise's POSIX_FADV_DONTNEED drops clean
   pages there); -1 elsewhere. Keeps the runtime: a test calls it alone. */
value rig_disk_test_drop_pages(value v_path) {
#if defined(__linux__)
  int fd = open(String_val(v_path), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return Val_int(errno);
  int code =
      fsync(fd) != 0 ? errno : posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
  close(fd);
  return Val_int(code);
#else
  (void)v_path;
  return Val_int(-1);
#endif
}

/* [heap_bytes ()] is the bytes the C heap holds allocated, or -1 where its
   allocator does not say. Keeps the runtime. */
value rig_disk_test_heap_bytes(value unit) {
  (void)unit;
#if defined(RIG_DISK_TEST_ASAN)
  return Val_long((intnat)__sanitizer_get_current_allocated_bytes());
#elif defined(__APPLE__)
  malloc_statistics_t s;
  malloc_zone_statistics(NULL, &s);
  return Val_long((intnat)s.size_in_use);
#elif defined(__GLIBC__)
  struct mallinfo2 m = mallinfo2();
  return Val_long((intnat)(m.uordblks + m.hblkhd));
#else
  return Val_long(-1);
#endif
}
