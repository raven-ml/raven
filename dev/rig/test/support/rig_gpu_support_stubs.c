/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Host memory the GPU suites own, by address. None of these blocks, so
   each holds the runtime. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#define Addr_val(v) ((void *)Long_val(v))

value rig_gpu_host_page(value unit) {
  (void)unit;
#if defined(_WIN32)
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return Val_long(info.dwPageSize);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
}

/* [v_n] zeroed bytes from a page, writable, or read-only if [v_read_only].
   The pages are mapped untouched, so a large area costs nothing until
   used. */
value rig_gpu_host_pages(value v_n, value v_read_only) {
  size_t n = Long_val(v_n);
#if defined(_WIN32)
  void *p = VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE,
                         Bool_val(v_read_only) ? PAGE_READONLY
                                               : PAGE_READWRITE);
  if (p == NULL) caml_raise_out_of_memory();
#else
  int prot = Bool_val(v_read_only) ? PROT_READ : PROT_READ | PROT_WRITE;
  void *p = mmap(NULL, n, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) caml_raise_out_of_memory();
#endif
  return Val_long((intnat)p);
}

value rig_gpu_host_free_pages(value v_p, value v_n) {
#if defined(_WIN32)
  (void)v_n;
  VirtualFree(Addr_val(v_p), 0, MEM_RELEASE);
#else
  munmap(Addr_val(v_p), Long_val(v_n));
#endif
  return Val_unit;
}

value rig_gpu_host_get8(value v_p) {
  return Val_int(*(volatile uint8_t *)Addr_val(v_p));
}

value rig_gpu_host_set8(value v_p, value v_x) {
  *(volatile uint8_t *)Addr_val(v_p) = (uint8_t)Long_val(v_x);
  return Val_unit;
}

value rig_gpu_host_get32(value v_p) {
  return Val_long(*(volatile uint32_t *)Addr_val(v_p));
}

value rig_gpu_host_set32(value v_p, value v_x) {
  *(volatile uint32_t *)Addr_val(v_p) = (uint32_t)Long_val(v_x);
  return Val_unit;
}

value rig_gpu_host_get64(value v_p) {
  _Atomic uint64_t *p = Addr_val(v_p);
  return Val_long((intnat)atomic_load_explicit(p, memory_order_acquire));
}

value rig_gpu_host_set64(value v_p, value v_x) {
  _Atomic uint64_t *p = Addr_val(v_p);
  atomic_store_explicit(p, (uint64_t)Long_val(v_x), memory_order_release);
  return Val_unit;
}

value rig_gpu_host_read(value v_p, value v_n) {
  CAMLparam2(v_p, v_n);
  CAMLreturn(caml_alloc_initialized_string(Long_val(v_n), Addr_val(v_p)));
}

value rig_gpu_host_write(value v_p, value v_s) {
  memcpy(Addr_val(v_p), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}
