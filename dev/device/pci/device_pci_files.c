/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Files of this machine's PCI functions: lock files, and mappings of BARs
   and VFIO regions. */

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
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static void fail_errno(const char *what) {
  char msg[512];
  snprintf(msg, sizeof msg, "%s: %s", what, strerror(errno));
  caml_failwith(msg);
}

/* Takes an exclusive lock on the file at [path], creating it readable by
   every user: its descriptor, or -1 if another holder has it. A link is never
   followed and the mode of a file another process made is never changed, so
   a lock file names nothing else. The lock is flock's, which other programs
   driving these GPUs take, and which two descriptors of one process exclude
   as two processes do. */
value caml_device_pci_lock(value path) {
  CAMLparam1(path);
#ifdef _WIN32
  caml_failwith("Locking a PCI function needs a POSIX system");
  CAMLreturn(Val_unit);
#else
  int fd = open(String_val(path), O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC,
                0644);
  if (fd < 0) fail_errno(String_val(path));
  struct stat st;
  if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
    close(fd);
    caml_failwith("the lock file is not a regular file");
  }
  if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    close(fd);
    CAMLreturn(Val_int(-1));
  }
  CAMLreturn(Val_int(fd));
#endif
}

/* Maps [n] bytes of [fd] from [off], shared, not inherited by children. */
value caml_device_pci_map(value fd, value off, value n) {
#ifdef __linux__
  void *p = mmap(NULL, Long_val(n), PROT_READ | PROT_WRITE, MAP_SHARED,
                 Int_val(fd), Long_val(off));
  if (p == MAP_FAILED) fail_errno("mapping a PCI BAR");
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

value caml_device_pci_close(value fd) {
#ifndef _WIN32
  close(Int_val(fd));
#else
  (void)fd;
#endif
  return Val_unit;
}

/* Opens [path] for reading, and writing if [write]; raises Failure naming
   it. */
value caml_device_pci_open(value path, value write) {
  CAMLparam2(path, write);
#ifdef _WIN32
  caml_failwith("Opening device files needs a POSIX system");
  CAMLreturn(Val_unit);
#else
  int fd = open(String_val(path),
                (Bool_val(write) ? O_RDWR : O_RDONLY) | O_SYNC | O_CLOEXEC);
  if (fd < 0) fail_errno(String_val(path));
  CAMLreturn(Val_int(fd));
#endif
}

/* The [n] bytes, at most 64, of [fd] at [off]: fewer where the file ends,
   or where the kernel shows the process less, as configuration space past
   64 bytes without CAP_SYS_ADMIN. */
value caml_device_pci_pread(value fd, value off, value n) {
  CAMLparam3(fd, off, n);
  CAMLlocal1(s);
#ifdef _WIN32
  caml_failwith("Reading device files needs a POSIX system");
#else
  char buf[64];
  size_t len = Long_val(n) < (long)sizeof buf ? Long_val(n) : sizeof buf;
  ssize_t r = pread(Int_val(fd), buf, len, Long_val(off));
  if (r < 0) fail_errno("reading configuration space");
  s = caml_alloc_initialized_string(r, buf);
#endif
  CAMLreturn(s);
}

/* Writes [s] to [fd] at [off]; raises Failure unless all of it went. */
value caml_device_pci_pwrite(value fd, value off, value s) {
#ifdef _WIN32
  (void)fd;
  (void)off;
  (void)s;
  caml_failwith("Writing device files needs a POSIX system");
#else
  ssize_t r =
      pwrite(Int_val(fd), String_val(s), caml_string_length(s), Long_val(off));
  if (r != (ssize_t)caml_string_length(s)) {
    if (r >= 0) errno = EIO;
    fail_errno("writing configuration space");
  }
#endif
  return Val_unit;
}
