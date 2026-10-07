/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Files of this machine's PCI functions: their locks, and mappings of BARs
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
#include <sys/file.h>
#include <sys/mman.h>
#endif

/* Takes flock's exclusive lock on [fd] without waiting; raises Unix_error
   EWOULDBLOCK if another open file holds it. flock locks the file itself,
   so two descriptors of one process exclude each other as two processes
   do. */
value caml_device_pci_flock(value fd) {
#ifdef _WIN32
  (void)fd;
  caml_failwith("Locking a PCI function needs a POSIX system");
#else
  if (flock(Int_val(fd), LOCK_EX | LOCK_NB) != 0) caml_uerror("flock", Nothing);
#endif
  return Val_unit;
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
