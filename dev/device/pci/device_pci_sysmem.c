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

#ifndef _WIN32
#include <unistd.h>
#endif

value caml_device_pci_page_size(value unit) {
  (void)unit;
#ifdef _WIN32
  return Val_long(4096);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
}

#ifdef __linux__
#include <sys/file.h>
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
value caml_device_pci_flock(value fd) {
  if (flock(Int_val(fd), LOCK_EX | LOCK_NB) != 0) caml_uerror("flock", Nothing);
  return Val_unit;
}

/* Maps [n] bytes of [fd] from [off], on a page, shared, not inherited by
   children. Nothing is populated, so it holds the runtime. madvise is
   advice: the mapping serves without it. */
value caml_device_pci_map(value fd, value off, value n) {
  void *p = mmap(NULL, Long_val(n), PROT_READ | PROT_WRITE, MAP_SHARED,
                 Int_val(fd), Long_val(off));
  if (p == MAP_FAILED) caml_uerror("mmap", Nothing);
  madvise(p, Long_val(n), MADV_DONTFORK);
  return Val_long((intnat)p);
}

/* Unmaps what caml_device_pci_map mapped, holding the runtime: no memory
   of the process goes back with it. */
value caml_device_pci_unmap(value a, value n) {
  if (munmap(PTR(a), Long_val(n)) != 0) caml_uerror("munmap", Nothing);
  return Val_unit;
}

/* Memory */

/* Reserves [n] addresses at [base] with an inaccessible mapping, holding the
   runtime: nothing is populated. A kernel before 4.17 takes
   MAP_FIXED_NOREPLACE as a hint and maps elsewhere when the range is taken,
   which is reported as the range in use, EEXIST, as later kernels do. */
value caml_device_pci_reserve(value base, value n) {
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
   or where the kernel chooses if [va] is 0, from a huge page if [huge],
   locked if [locked]. Releases the runtime. */
value caml_device_pci_sysmem_map(value va, value n, value huge, value locked) {
  void *at = PTR(va);
  int flags = MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE |
              (Bool_val(locked) ? MAP_LOCKED : 0) | (at ? MAP_FIXED : 0) |
              (Bool_val(huge) ? MAP_HUGETLB : 0);
  void *p = map_released(at, Long_val(n), PROT_READ | PROT_WRITE, flags);
  if (p == MAP_FAILED) caml_uerror("mmap", Nothing);
  return Val_long((intnat)p);
}

/* Unmaps [n] bytes at [va] that no reservation holds. Releases the
   runtime. */
value caml_device_pci_sysmem_unmap(value va, value n) {
  if (unmap_released(PTR(va), Long_val(n)) != 0)
    caml_uerror("munmap", Nothing);
  return Val_unit;
}

/* Returns [n] bytes at [va] to their reservation. Releases the runtime. */
value caml_device_pci_sysmem_release(value va, value n) {
  void *p = map_released(PTR(va), Long_val(n), PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                             MAP_FIXED);
  if (p == MAP_FAILED) caml_uerror("mmap", Nothing);
  return Val_unit;
}

/* Locking faults the memory in: the runtime is released. */
value caml_device_pci_sysmem_lock(value a, value n) {
  void *at = PTR(a);
  size_t len = Long_val(n);
  caml_release_runtime_system();
  int r = mlock(at, len);
  int e = errno;
  caml_acquire_runtime_system();
  if (r != 0) caml_unix_error(e, "mlock", Nothing);
  return Val_unit;
}

/* Unlocking frees nothing: it holds the runtime. */
value caml_device_pci_sysmem_unlock(value a, value n) {
  if (munlock(PTR(a), Long_val(n)) != 0) caml_uerror("munlock", Nothing);
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

FAILS1(caml_device_pci_flock, "flock")
FAILS2(caml_device_pci_unmap, "munmap")
FAILS2(caml_device_pci_reserve, "mmap")
FAILS2(caml_device_pci_sysmem_unmap, "munmap")
FAILS2(caml_device_pci_sysmem_release, "mmap")
FAILS2(caml_device_pci_sysmem_lock, "mlock")
FAILS2(caml_device_pci_sysmem_unlock, "munlock")

value caml_device_pci_map(value fd, value off, value n) {
  (void)fd;
  (void)off;
  (void)n;
  caml_unix_error(ENOSYS, "mmap", Nothing);
}

value caml_device_pci_sysmem_map(value va, value n, value huge, value locked) {
  (void)va;
  (void)n;
  (void)huge;
  (void)locked;
  caml_unix_error(ENOSYS, "mmap", Nothing);
}
#endif
