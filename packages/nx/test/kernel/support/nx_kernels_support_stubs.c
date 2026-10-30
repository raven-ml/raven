/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

value nx_kernels_support_x86_64(value unit) {
  (void)unit;
#if defined(__x86_64__)
  return Val_true;
#else
  return Val_false;
#endif
}

/* Guard pages */

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <caml/bigarray.h>

static intnat page_bytes(void) {
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (intnat)si.dwPageSize;
#else
  return (intnat)sysconf(_SC_PAGESIZE);
#endif
}

/* [vbytes] bytes, a char bigarray over memory of its own, whose bytes from
   [vfrom] on are readable up to an inaccessible page: their last byte just
   before it where [vafter], else their first byte, [vfrom], just after it.
   The bytes before [vfrom] lie in the page where [vafter] is false. With
   the mapping's start and length, to give to unguard. */
value nx_kernels_support_guarded(value vafter, value vfrom, value vbytes) {
  CAMLparam0();
  CAMLlocal2(ba, r);
  intnat page = page_bytes(), bytes = Long_val(vbytes);
  intnat from = Long_val(vfrom), readable = bytes - from;
  intnat pages = 1 + (readable + page - 1) / page;
  if (pages < 2) pages = 2;
  size_t total = (size_t)(pages * page);
#if defined(_WIN32)
  uint8_t *base = VirtualAlloc(NULL, total, MEM_RESERVE | MEM_COMMIT,
                               PAGE_READWRITE);
  if (base == NULL) caml_raise_out_of_memory();
#else
  uint8_t *base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON, -1, 0);
  if (base == MAP_FAILED) caml_raise_out_of_memory();
#endif
  uint8_t *guard, *data;
  if (Bool_val(vafter)) {
    guard = base + total - page;
    data = guard - bytes;
  } else {
    guard = base;
    data = base + page - from;
  }
#if defined(_WIN32)
  DWORD old;
  VirtualProtect(guard, (size_t)page, PAGE_NOACCESS, &old);
#else
  mprotect(guard, (size_t)page, PROT_NONE);
#endif
  ba = caml_ba_alloc_dims(CAML_BA_CHAR | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL,
                          1, data, bytes);
  r = caml_alloc_tuple(3);
  Store_field(r, 0, ba);
  Store_field(r, 1, caml_copy_nativeint((intnat)base));
  Store_field(r, 2, Val_long((intnat)total));
  CAMLreturn(r);
}

value nx_kernels_support_unguard(value vbase, value vtotal) {
  void *base = (void *)Nativeint_val(vbase);
#if defined(_WIN32)
  (void)vtotal;
  VirtualFree(base, 0, MEM_RELEASE);
#else
  munmap(base, (size_t)Long_val(vtotal));
#endif
  return Val_unit;
}

/* Address space */

#if defined(__linux__)
#include <stdio.h>
#include <sys/resource.h>
#endif

/* Sets the process's address-space limit [vextra] bytes above the size of
   its address space, read from /proc/self/statm: true iff set. Linux
   only. */
value nx_kernels_support_cap_memory(value vextra) {
#if defined(__linux__)
  FILE *f = fopen("/proc/self/statm", "r");
  if (f == NULL) return Val_false;
  unsigned long pages;
  int got = fscanf(f, "%lu", &pages);
  fclose(f);
  if (got != 1) return Val_false;
  struct rlimit l;
  if (getrlimit(RLIMIT_AS, &l) != 0) return Val_false;
  rlim_t cap = (rlim_t)pages * (rlim_t)page_bytes() + (rlim_t)Long_val(vextra);
  if (l.rlim_max != RLIM_INFINITY && cap > l.rlim_max) return Val_false;
  l.rlim_cur = cap;
  return Val_bool(setrlimit(RLIMIT_AS, &l) == 0);
#else
  (void)vextra;
  return Val_false;
#endif
}
