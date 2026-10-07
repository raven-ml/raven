/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* This machine's memory for functions: reservations, locked and physically
   addressed memory. Linux only; elsewhere every call but the page size
   raises Failure. */

#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/fail.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#ifndef _WIN32
#include <sys/mman.h>
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
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

static void fail_errno(const char *what) {
  char msg[512];
  snprintf(msg, sizeof msg, "%s: %s", what, strerror(errno));
  caml_failwith(msg);
}

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

#define PTR(v) ((void *)Long_val(v))

/* Reserves [n] addresses at [base] with an inaccessible mapping. */
value caml_device_pci_reserve(value base, value n) {
  void *want = PTR(base);
  void *p = mmap(want, Long_val(n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                     MAP_FIXED_NOREPLACE,
                 -1, 0);
  if (p == MAP_FAILED) fail_errno("reserving the GPU's address range");
  if (p != want) {
    munmap(p, Long_val(n));
    caml_failwith("reserving the GPU's address range: the kernel placed it "
                  "elsewhere");
  }
  return Val_unit;
}

/* Maps [n] bytes of shared, populated memory at [va], inside a reservation,
   or where the kernel chooses if [va] is 0, from a huge page if [huge],
   locked if [locked]. */
value caml_device_pci_sysmem_map(value va, value n, value huge, value locked) {
  void *at = PTR(va);
  int flags = MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE |
              (Bool_val(locked) ? MAP_LOCKED : 0) | (at ? MAP_FIXED : 0) |
              (Bool_val(huge) ? MAP_HUGETLB : 0);
  void *p = map_released(at, Long_val(n), PROT_READ | PROT_WRITE, flags);
  if (p == MAP_FAILED) fail_errno("allocating system memory");
  return Val_long((intnat)p);
}

/* Unmaps [n] bytes at [va] that no reservation holds. */
value caml_device_pci_sysmem_unmap(value va, value n) {
  if (unmap_released(PTR(va), Long_val(n)) != 0)
    fail_errno("freeing system memory");
  return Val_unit;
}

/* Returns [n] bytes at [va] to their reservation. */
value caml_device_pci_sysmem_release(value va, value n) {
  void *p = map_released(PTR(va), Long_val(n), PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE |
                             MAP_FIXED);
  if (p == MAP_FAILED) fail_errno("freeing system memory");
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
  errno = e;
  if (r != 0)
    fail_errno("locking memory for a GPU (the locked-memory limit, "
               "ulimit -l, may be too low)");
  return Val_unit;
}

value caml_device_pci_sysmem_unlock(value a, value n) {
  if (munlock(PTR(a), Long_val(n)) != 0) fail_errno("unlocking memory");
  return Val_unit;
}

#else

/* Without Linux every call raises Failure. */

static value no_sysmem(void) {
  caml_failwith("System memory for a GPU needs Linux");
  return Val_unit;
}

value caml_device_pci_reserve(value base, value n) {
  (void)base;
  (void)n;
  return no_sysmem();
}

value caml_device_pci_sysmem_map(value va, value n, value huge, value locked) {
  (void)va;
  (void)n;
  (void)huge;
  (void)locked;
  return no_sysmem();
}

#define FAILS2(f)             \
  value f(value a, value b) { \
    (void)a;                  \
    (void)b;                  \
    return no_sysmem();       \
  }

FAILS2(caml_device_pci_sysmem_unmap)
FAILS2(caml_device_pci_sysmem_release)
FAILS2(caml_device_pci_sysmem_lock)
FAILS2(caml_device_pci_sysmem_unlock)
#endif
