/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Files of this machine's PCI functions: lock files, and mappings of BARs
   and VFIO regions. Descriptors are Unix.file_descr, an int on POSIX. */

#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#ifndef _WIN32
#include <caml/unixsupport.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

/* Takes an exclusive lock on the file at [path], creating it readable by
   every user, and is its descriptor. Raises Unix_error EWOULDBLOCK if
   another holder has it. A link is never followed and the mode of a file
   another process made is never changed, so a lock file names nothing
   else. The lock is flock's, which other programs driving these GPUs take,
   and which two descriptors of one process exclude as two processes do. */
value caml_device_pci_lock(value path) {
  CAMLparam1(path);
#ifdef _WIN32
  caml_failwith("Locking a PCI function needs a POSIX system");
  CAMLreturn(Val_unit);
#else
  int fd = open(String_val(path), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC,
                0644);
  if (fd < 0) caml_uerror("open", path);
  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    close(fd);
    caml_failwith("the lock file is not a regular file");
  }
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    int e = errno;
    close(fd);
    errno = e;
    caml_uerror("flock", path);
  }
  CAMLreturn(Val_int(fd));
#endif
}

/* Maps [n] bytes of [fd] from [off], on a page, shared, not inherited by
   children. */
value caml_device_pci_map(value fd, value off, value n) {
#ifdef __linux__
  void *p = mmap(NULL, Long_val(n), PROT_READ | PROT_WRITE, MAP_SHARED,
                 Int_val(fd), Long_val(off));
  if (p == MAP_FAILED) {
    char msg[256];
    snprintf(msg, sizeof msg, "mapping a PCI BAR: %s", strerror(errno));
    caml_failwith(msg);
  }
  madvise(p, Long_val(n), MADV_DONTFORK);
  return Val_long((intnat)p);
#else
  (void)fd;
  (void)off;
  (void)n;
  caml_failwith("Mapping a PCI BAR needs Linux");
  return Val_unit;
#endif
}

value caml_device_pci_unmap(value a, value n) {
#ifndef _WIN32
  munmap((void *)Long_val(a), Long_val(n));
#else
  (void)a;
  (void)n;
#endif
  return Val_unit;
}
