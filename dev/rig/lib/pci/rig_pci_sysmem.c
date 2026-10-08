/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* This process's mappings: the files of PCI functions (their locks, BARs and
   VFIO regions) and memory for GPUs (reservations, locked memory).

   A failing system call raises Unix.Unix_error with its errno and its name;
   the OCaml caller says what it was doing. Descriptors are Unix.file_descr,
   an int on Linux. Linux only: elsewhere every call but the page size raises
   Unix_error ENOSYS. */

#define _GNU_SOURCE
#include <errno.h>

#define CAML_NAME_SPACE
#include <caml/mlvalues.h>
#include <caml/threads.h>
#include <caml/unixsupport.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

value caml_rig_pci_page_size(value unit) {
  (void)unit;
#ifdef _WIN32
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return Val_long(info.dwPageSize);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
}

#ifdef __linux__
#include <fcntl.h>
#include <sys/file.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif


#define PTR(v) ((void *)Long_val(v))

/* mmap and munmap with the runtime released, errno kept: populating memory
   and giving it back take time. */
static void *map_released(void *at, size_t len, int prot, int flags) {
  caml_release_runtime_system();
  void *p = mmap(at, len, prot, flags, -1, 0);
  int e = errno;
  caml_acquire_runtime_system();
  errno = e;
  return p;
}

static int unmap_released(void *at, size_t len) {
  caml_release_runtime_system();
  int r = munmap(at, len);
  int e = errno;
  caml_acquire_runtime_system();
  errno = e;
  return r;
}

/* Files */

/* Takes flock's exclusive lock on [fd] without waiting, so it holds the
   runtime; raises Unix_error EWOULDBLOCK if another open file holds it.
   flock locks the file itself, so two descriptors of one process exclude
   each other as two processes do. */
value caml_rig_pci_flock(value fd) {
  if (flock(Int_val(fd), LOCK_EX | LOCK_NB) != 0) caml_uerror("flock", Nothing);
  return Val_unit;
}

/* Makes the file [path] in the directory [dir], and is its descriptor,
   which holds flock's shared lock from before the file has a name: the file
   is made nameless (O_TMPFILE), locked, then linked at [path] through the
   process's link to it (open(2)), so that no process meets it unlocked. The
   process holds the lock on its memory's file while it lives. Raises
   Unix_error on a failing call, the file then gone. Holds the runtime:
   nothing waits. */
value caml_rig_pci_sysmem_create(value dir, value path) {
  int fd = open(String_val(dir), O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);
  if (fd < 0) caml_uerror("open", dir);
  if (flock(fd, LOCK_SH | LOCK_NB) != 0) {
    int e = errno;
    close(fd);
    caml_unix_error(e, "flock", path);
  }
  char self[32];
  snprintf(self, sizeof self, "/proc/self/fd/%d", fd);
  if (linkat(AT_FDCWD, self, AT_FDCWD, String_val(path),
             AT_SYMLINK_FOLLOW) != 0) {
    int e = errno;
    close(fd);
    caml_unix_error(e, "linkat", path);
  }
  return Val_int(fd);
}

/* Maps [n] bytes of [fd] from [off], on a page, shared, not inherited by
   children. Nothing is populated, so it holds the runtime. madvise is
   advice: the mapping serves without it. */
value caml_rig_pci_map(value fd, value off, value n) {
  void *p = mmap(NULL, Long_val(n), PROT_READ | PROT_WRITE, MAP_SHARED,
                 Int_val(fd), Long_val(off));
  if (p == MAP_FAILED) caml_uerror("mmap", Nothing);
  madvise(p, Long_val(n), MADV_DONTFORK);
  return Val_long((intnat)p);
}

/* Unmaps what caml_rig_pci_map mapped, holding the runtime: no memory
   of the process goes back with it. */
value caml_rig_pci_unmap(value a, value n) {
  if (munmap(PTR(a), Long_val(n)) != 0) caml_uerror("munmap", Nothing);
  return Val_unit;
}

/* Memory */

/* Reserves [n] addresses at [base] with an inaccessible mapping, holding the
   runtime: nothing is populated. A kernel before 4.17 takes
   MAP_FIXED_NOREPLACE as a hint and maps elsewhere when the range is taken,
   which is reported as the range in use, EEXIST, as later kernels do. */
value caml_rig_pci_reserve(value base, value n) {
  void *want = PTR(base);
  void *p = mmap(want, Long_val(n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                     MAP_FIXED_NOREPLACE,
                 -1, 0);
  if (p == MAP_FAILED) caml_uerror("mmap", Nothing);
  if (p != want) {
    munmap(p, Long_val(n));
    caml_unix_error(EEXIST, "mmap", Nothing);
  }
  return Val_unit;
}

/* Maps [n] bytes of shared, populated memory at [va], inside a reservation,
   or where the kernel chooses if [va] is 0. Releases the runtime. */
value caml_rig_pci_sysmem_map(value va, value n) {
  void *at = PTR(va);
  int flags = MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE | (at ? MAP_FIXED : 0);
  void *p = map_released(at, Long_val(n), PROT_READ | PROT_WRITE, flags);
  if (p == MAP_FAILED) caml_uerror("mmap", Nothing);
  return Val_long((intnat)p);
}

/* Maps [n] bytes of the file [fd] from [off], shared and populated, at [va],
   inside a reservation, or where the kernel chooses if [va] is 0. Releases
   the runtime. */
value caml_rig_pci_sysmem_map_file(value fd, value off, value va, value n) {
  void *at = PTR(va);
  int flags = MAP_SHARED | MAP_POPULATE | (at ? MAP_FIXED : 0);
  caml_release_runtime_system();
  void *p = mmap(at, Long_val(n), PROT_READ | PROT_WRITE, flags, Int_val(fd),
                 Long_val(off));
  int e = errno;
  caml_acquire_runtime_system();
  if (p == MAP_FAILED) caml_unix_error(e, "mmap", Nothing);
  return Val_long((intnat)p);
}

/* Gives back the pages of [n] bytes of the file [fd] from [off], leaving its
   size. Releases the runtime. */
value caml_rig_pci_sysmem_punch(value fd, value off, value n) {
  caml_release_runtime_system();
  int r = fallocate(Int_val(fd), FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE,
                    Long_val(off), Long_val(n));
  int e = errno;
  caml_acquire_runtime_system();
  if (r != 0) caml_unix_error(e, "fallocate", Nothing);
  return Val_unit;
}

/* Gives the file [fd] its pages for [n] bytes from [off], growing it.
   Releases the runtime. */
value caml_rig_pci_sysmem_allocate(value fd, value off, value n) {
  caml_release_runtime_system();
  int r = fallocate(Int_val(fd), 0, Long_val(off), Long_val(n));
  int e = errno;
  caml_acquire_runtime_system();
  if (r != 0) caml_unix_error(e, "fallocate", Nothing);
  return Val_unit;
}

/* Reserves [n] addresses on 2 MiB where the kernel chooses, with an
   inaccessible mapping, and is the first: [n] and 2 MiB more are reserved,
   and what lies outside the aligned range given back. Holds the runtime:
   nothing is populated. */
value caml_rig_pci_sysmem_region(value n) {
  size_t len = Long_val(n), huge = 2 << 20;
  char *p = mmap(NULL, len + huge, PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  if (p == MAP_FAILED) caml_uerror("mmap", Nothing);
  char *at = (char *)(((uintptr_t)p + huge - 1) & ~(uintptr_t)(huge - 1));
  if (at > p) munmap(p, at - p);
  if (p + huge > at) munmap(at + len, p + huge - at);
  return Val_long((intnat)at);
}

/* Zeroes the [n] bytes at [va], which the process maps. Releases the
   runtime. */
value caml_rig_pci_sysmem_zero(value va, value n) {
  void *at = PTR(va);
  size_t len = Long_val(n);
  caml_release_runtime_system();
  memset(at, 0, len);
  caml_acquire_runtime_system();
  return Val_unit;
}

/* Unmaps [n] bytes at [va] that no reservation holds. Releases the
   runtime. */
value caml_rig_pci_sysmem_unmap(value va, value n) {
  if (unmap_released(PTR(va), Long_val(n)) != 0)
    caml_uerror("munmap", Nothing);
  return Val_unit;
}

/* Returns [n] bytes at [va] to their reservation. Releases the runtime. */
value caml_rig_pci_sysmem_release(value va, value n) {
  void *p = map_released(PTR(va), Long_val(n), PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                             MAP_FIXED);
  if (p == MAP_FAILED) caml_uerror("mmap", Nothing);
  return Val_unit;
}

#else

/* Without Linux every call raises Unix_error ENOSYS. */

#define FAILS1(f, call)                   \
  value f(value a) {                      \
    (void)a;                              \
    caml_unix_error(ENOSYS, call, Nothing); \
  }

#define FAILS2(f, call)                   \
  value f(value a, value b) {             \
    (void)a;                              \
    (void)b;                              \
    caml_unix_error(ENOSYS, call, Nothing); \
  }

FAILS1(caml_rig_pci_flock, "flock")
FAILS2(caml_rig_pci_sysmem_create, "open")
FAILS2(caml_rig_pci_unmap, "munmap")
FAILS2(caml_rig_pci_reserve, "mmap")
FAILS2(caml_rig_pci_sysmem_unmap, "munmap")
FAILS2(caml_rig_pci_sysmem_release, "mmap")
FAILS1(caml_rig_pci_sysmem_region, "mmap")
FAILS2(caml_rig_pci_sysmem_zero, "memset")

value caml_rig_pci_map(value fd, value off, value n) {
  (void)fd;
  (void)off;
  (void)n;
  caml_unix_error(ENOSYS, "mmap", Nothing);
}

FAILS2(caml_rig_pci_sysmem_map, "mmap")

value caml_rig_pci_sysmem_map_file(value fd, value off, value va, value n) {
  (void)fd;
  (void)off;
  (void)va;
  (void)n;
  caml_unix_error(ENOSYS, "mmap", Nothing);
}

value caml_rig_pci_sysmem_punch(value fd, value off, value n) {
  (void)fd;
  (void)off;
  (void)n;
  caml_unix_error(ENOSYS, "fallocate", Nothing);
}

value caml_rig_pci_sysmem_allocate(value fd, value off, value n) {
  (void)fd;
  (void)off;
  (void)n;
  caml_unix_error(ENOSYS, "fallocate", Nothing);
}
#endif
