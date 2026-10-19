/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the bench rows need from C: a file's pages dropped from the page
   cache, and a host kernel's claims on its operands. */

#define _GNU_SOURCE

#include <errno.h>

#define CAML_NAME_SPACE
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig.h"

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* Drops the file [v_path]'s pages from the page cache, so that its next
   read comes from its medium: writes back its dirty pages, then asks the
   kernel to drop them (Linux's posix_fadvise DONTNEED; macOS's msync
   MS_INVALIDATE over a mapping of the file). Answers [0], or the errno of a
   failing call. Holds the runtime: it serves bench setup only. */
value rig_bench_evict(value v_path) {
#if defined(_WIN32)
  (void)v_path;
  return Val_int(ENOSYS);
#else
  int fd = open(String_val(v_path), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return Val_int(errno);
  int e = 0;
  struct stat st;
  if (fsync(fd) != 0 || fstat(fd, &st) != 0) e = errno;
#if defined(__APPLE__)
  if (e == 0 && st.st_size > 0) {
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (p == MAP_FAILED) e = errno;
    else {
      if (msync(p, (size_t)st.st_size, MS_INVALIDATE) != 0) e = errno;
      munmap(p, (size_t)st.st_size);
    }
  }
#else
  if (e == 0) e = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
#endif
  close(fd);
  return Val_int(e);
#endif
}

/* Whether a claim answer holds the claim. */
static int held(enum rig_claim c) {
  return c == RIG_CLAIMED || c == RIG_WAIT || c == RIG_LOST;
}

/* Claims the memory of three buffers for reading, then releases each, as a
   host kernel with three operands does around its loop. */
value rig_bench_claim_3(value a, value b, value c) {
  enum rig_claim ca = rig_buffer_claim(a, RIG_READ);
  enum rig_claim cb = rig_buffer_claim(b, RIG_READ);
  enum rig_claim cc = rig_buffer_claim(c, RIG_READ);
  if (held(ca)) rig_buffer_release(a);
  if (held(cb)) rig_buffer_release(b);
  if (held(cc)) rig_buffer_release(c);
  return Val_unit;
}
