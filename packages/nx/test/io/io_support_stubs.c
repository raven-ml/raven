/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What nx.io's suites ask of the system that OCaml's Unix does not give:
   dropping a file's cached pages, which of them the system holds, and the
   storage a file takes. */

#define _GNU_SOURCE

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/mlvalues.h>
#include <errno.h>
#include <string.h>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#elif !defined(_WIN32)
#include <sys/stat.h>
#endif

/* [drop_pages path] writes the file at [path]'s dirty pages to storage and asks
   the system to drop its cached pages, so that the next read reaches storage:
   0, or the errno. Linux only (posix_fadvise's POSIX_FADV_DONTNEED drops clean
   pages there); -1 elsewhere. Keeps the runtime: a test calls it alone. */
value nx_io_test_drop_pages(value v_path) {
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

/* [resident path] is a byte for each page of the file at [path], 1 where the
   system holds that page in memory and 0 where it does not, from mincore on a
   mapping of the file that touches none of its pages. Linux only; raises
   [Failure] elsewhere and when a call fails. */
value nx_io_test_resident(value v_path) {
#if defined(__linux__)
  int fd = open(String_val(v_path), O_RDONLY | O_CLOEXEC);
  if (fd < 0) caml_failwith(strerror(errno));
  struct stat st;
  if (fstat(fd, &st) != 0) {
    int code = errno;
    close(fd);
    caml_failwith(strerror(code));
  }
  size_t page = (size_t)sysconf(_SC_PAGESIZE);
  size_t len = (size_t)st.st_size;
  size_t pages = (len + page - 1) / page;
  value r = caml_alloc_string(pages);
  if (pages == 0) {
    close(fd);
    return r;
  }
  void *m = mmap(NULL, len, PROT_READ, MAP_SHARED, fd, 0);
  int code = m == MAP_FAILED ? errno : 0;
  close(fd);
  if (code != 0) caml_failwith(strerror(code));
  unsigned char *v = (unsigned char *)Bytes_val(r);
  code = mincore(m, len, v) != 0 ? errno : 0;
  munmap(m, len);
  if (code != 0) caml_failwith(strerror(code));
  for (size_t i = 0; i < pages; i++) v[i] &= 1;
  return r;
#else
  (void)v_path;
  caml_failwith("mincore of a file: Linux only");
#endif
}

/* [page_size ()] is the system's page size in bytes, the unit of
   [resident]. Linux only; raises [Failure] elsewhere. */
value nx_io_test_page_size(value unit) {
  (void)unit;
#if defined(__linux__)
  return Val_long(sysconf(_SC_PAGESIZE));
#else
  caml_failwith("page size: Linux only");
#endif
}

/* [storage path] is the bytes of storage the file at [path] takes, from its
   blocks of 512 bytes, or -1 where the system states none (Windows) or the
   file cannot be read. */
value nx_io_test_storage(value v_path) {
#if !defined(_WIN32)
  struct stat st;
  if (stat(String_val(v_path), &st) != 0) return Val_long(-1);
  return Val_long((long)st.st_blocks * 512);
#else
  (void)v_path;
  return Val_long(-1);
#endif
}
