/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the disk's suite asks of the system that OCaml's Unix does not give: a
   limit on this process's open files, dropping a file's cached pages, and
   whether AddressSanitizer watches this process. */

#define _GNU_SOURCE

#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <errno.h>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/resource.h>
#include <unistd.h>
#endif

/* [set_open_files n] sets this process's soft limit of open files to [n]: 0,
   the errno, or -1 on Windows, which has no such limit. Keeps the runtime:
   setrlimit does not block. */
value device_disk_test_set_open_files(value v_n) {
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

/* [drop_pages path] writes the file at [path]'s dirty pages to storage and asks
   the system to drop its cached pages, so that the next read reaches storage:
   0, or the errno. Linux only (posix_fadvise's POSIX_FADV_DONTNEED drops clean
   pages there); -1 elsewhere. Keeps the runtime: a test calls it alone. */
value device_disk_test_drop_pages(value v_path) {
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

/* [sanitized ()] is whether this executable was compiled with
   AddressSanitizer, which ends the process at a use of freed memory. Keeps
   the runtime. */
value device_disk_test_sanitized(value v_unit) {
  (void)v_unit;
#if defined(__SANITIZE_ADDRESS__)
  return Val_true;
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
  return Val_true;
#else
  return Val_false;
#endif
#else
  return Val_false;
#endif
}
