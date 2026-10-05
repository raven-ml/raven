/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The comgr worker's program in memory, on Linux: a file of no file system,
   so that it runs where the home directory is mounted noexec. */

/* memfd_create and file seals are GNU extensions of the C library. */
#ifdef __linux__
#define _GNU_SOURCE
#endif

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#ifdef __linux__
#include <caml/alloc.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MFD_EXEC
#define MFD_EXEC 0x0010U
#endif

static void refused(const char *call, int fd) {
  value message = caml_alloc_sprintf("%s: %s", call, strerror(errno));
  if (fd >= 0) close(fd);
  caml_failwith_value(message);
}

/* A sealed memory file holding [v_program], open close-on-exec: the
   descriptor of the program the path /proc/self/fd/N runs. Raises Failure
   naming the call the kernel refused. */
value caml_tolk_comgr_worker_memfd(value v_program) {
  CAMLparam1(v_program);
  unsigned flags = MFD_CLOEXEC | MFD_ALLOW_SEALING;
  /* Kernels from 6.3 on make memory files executable only on request, and
     older ones reject the request. */
  int fd = memfd_create("tolk-comgr-worker", flags | MFD_EXEC);
  if (fd < 0 && errno == EINVAL) fd = memfd_create("tolk-comgr-worker", flags);
  if (fd < 0) refused("memfd_create", -1);
  size_t size = caml_string_length(v_program), done = 0;
  while (done < size) {
    ssize_t n = write(fd, String_val(v_program) + done, size - done);
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) refused("write", fd);
    done += n;
  }
  if (fcntl(fd, F_ADD_SEALS,
            F_SEAL_SEAL | F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE) < 0)
    refused("fcntl", fd);
  CAMLreturn(Val_int(fd));
}
#else
value caml_tolk_comgr_worker_memfd(value v_program) {
  (void)v_program;
  caml_failwith("memfd_create: Linux only");
}
#endif
