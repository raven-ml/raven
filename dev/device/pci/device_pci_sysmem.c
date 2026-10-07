/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* This machine's memory for functions: reservations, locked and physically
   addressed memory, page maps. Linux only; elsewhere every call but the page
   size raises Failure. */

#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#ifdef __linux__
#ifndef MAP_FIXED_NOREPLACE
#define MAP_FIXED_NOREPLACE 0x100000
#endif

static void fail_errno(const char *what) {
  char msg[512];
  snprintf(msg, sizeof msg, "%s: %s", what, strerror(errno));
  caml_failwith(msg);
}
#else
static void fail_linux(const char *what) {
  char msg[256];
  snprintf(msg, sizeof msg, "%s needs Linux", what);
  caml_failwith(msg);
}
#endif

#define PTR(v) ((void *)Long_val(v))

value caml_device_pci_page_size(value unit) {
  (void)unit;
#ifdef _WIN32
  return Val_long(4096);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
}

/* Reserves [n] addresses at [base] with an inaccessible mapping. */
value caml_device_pci_reserve(value base, value n) {
#ifdef __linux__
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
#else
  (void)base;
  (void)n;
  fail_linux("Reserving addresses for a GPU");
#endif
  return Val_unit;
}

/* Maps [n] bytes of shared, populated memory at [va], inside a reservation,
   or where the kernel chooses if [va] is 0, from a huge page if [huge],
   locked if [locked]. Populating takes time: the runtime is released. */
value caml_device_pci_sysmem_map(value va, value n, value huge, value locked) {
#ifdef __linux__
  void *at = PTR(va);
  int flags = MAP_SHARED | MAP_ANONYMOUS | MAP_POPULATE |
              (Bool_val(locked) ? MAP_LOCKED : 0) | (at ? MAP_FIXED : 0) |
              (Bool_val(huge) ? MAP_HUGETLB : 0);
  size_t len = Long_val(n);
  caml_release_runtime_system();
  void *p = mmap(at, len, PROT_READ | PROT_WRITE, flags, -1, 0);
  int e = errno;
  caml_acquire_runtime_system();
  errno = e;
  if (p == MAP_FAILED) fail_errno("allocating system memory");
  return Val_long((intnat)p);
#else
  (void)va;
  (void)n;
  (void)huge;
  (void)locked;
  fail_linux("System memory for a GPU");
  return Val_unit;
#endif
}

/* Unmaps [n] bytes at [va] that no reservation holds. */
value caml_device_pci_sysmem_unmap(value va, value n) {
#ifdef __linux__
  if (munmap(PTR(va), Long_val(n)) != 0) fail_errno("freeing system memory");
#else
  (void)va;
  (void)n;
  fail_linux("System memory for a GPU");
#endif
  return Val_unit;
}

/* Returns [n] bytes at [va] to their reservation. */
value caml_device_pci_sysmem_release(value va, value n) {
#ifdef __linux__
  void *p = mmap(PTR(va), Long_val(n), PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1,
                 0);
  if (p == MAP_FAILED) fail_errno("freeing system memory");
#else
  (void)va;
  (void)n;
  fail_linux("System memory for a GPU");
#endif
  return Val_unit;
}

/* Locking faults the memory in: the runtime is released. */
value caml_device_pci_sysmem_lock(value a, value n) {
#ifdef __linux__
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
#else
  (void)a;
  (void)n;
  fail_linux("Locking memory");
#endif
  return Val_unit;
}

value caml_device_pci_sysmem_unlock(value a, value n) {
#ifdef __linux__
  if (munlock(PTR(a), Long_val(n)) != 0) fail_errno("unlocking memory");
#else
  (void)a;
  (void)n;
  fail_linux("Locking memory");
#endif
  return Val_unit;
}

/* The page-map entries of the [pages] pages from [va], as
   /proc/self/pagemap gives them. The kernel walks the page tables for them:
   the runtime is released while it reads into a buffer of its own. */
value caml_device_pci_pagemap(value va, value pages) {
  CAMLparam2(va, pages);
  CAMLlocal1(out);
#ifdef __linux__
  long page = sysconf(_SC_PAGESIZE);
  size_t n = Long_val(pages) * 8;
  off_t at = (off_t)((uintptr_t)Long_val(va) / page) * 8;
  uint8_t *buf = malloc(n ? n : 1);
  if (buf == NULL) caml_raise_out_of_memory();
  const char *failed = NULL;
  int e = 0;
  caml_release_runtime_system();
  int fd = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    failed = "opening /proc/self/pagemap";
    e = errno;
  }
  for (size_t got = 0; !failed && got < n;) {
    ssize_t r = pread(fd, buf + got, n - got, at + got);
    if (r <= 0) {
      failed = "reading /proc/self/pagemap";
      e = r == 0 ? EIO : errno;
    } else
      got += r;
  }
  if (fd >= 0) close(fd);
  caml_acquire_runtime_system();
  if (failed) {
    free(buf);
    errno = e;
    fail_errno(failed);
  }
  out = caml_alloc_initialized_string(n, (const char *)buf);
  free(buf);
#else
  (void)va;
  (void)pages;
  fail_linux("Reading physical addresses");
#endif
  CAMLreturn(out);
}
