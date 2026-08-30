/*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*/

#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

/* MAP_ANON is the BSD spelling of MAP_ANONYMOUS; older systems may only define
   one. */
#ifndef MAP_ANON
#define MAP_ANON MAP_ANONYMOUS
#endif

static void raise_errno(const char *what) {
  char buf[256];
  snprintf(buf, sizeof(buf), "%s: %s", what, strerror(errno));
  caml_failwith(buf);
}

/* Files and mappings */

CAMLprim value caml_tolk_amd_constants(value unit) {
  CAMLparam1(unit);
  CAMLlocal1(v);
  v = caml_alloc_tuple(9);
  Store_field(v, 0, Val_int(O_RDONLY));
  Store_field(v, 1, Val_int(O_RDWR));
  Store_field(v, 2, Val_int(PROT_NONE));
  Store_field(v, 3, Val_int(PROT_READ));
  Store_field(v, 4, Val_int(PROT_WRITE));
  Store_field(v, 5, Val_int(MAP_SHARED));
  Store_field(v, 6, Val_int(MAP_PRIVATE));
  Store_field(v, 7, Val_int(MAP_ANON));
  Store_field(v, 8, Val_int(MAP_FIXED));
  CAMLreturn(v);
}

CAMLprim value caml_tolk_amd_open(value v_path, value v_flags) {
  CAMLparam2(v_path, v_flags);
  int fd = open(String_val(v_path), Int_val(v_flags) | O_CLOEXEC);
  if (fd < 0) raise_errno(String_val(v_path));
  CAMLreturn(Val_int(fd));
}

CAMLprim value caml_tolk_amd_close(value v_fd) {
  CAMLparam1(v_fd);
  if (close(Int_val(v_fd)) != 0) raise_errno("close");
  CAMLreturn(Val_unit);
}

CAMLprim value caml_tolk_amd_mmap(value v_addr, value v_size, value v_prot,
                                  value v_flags, value v_fd, value v_offset) {
  CAMLparam5(v_addr, v_size, v_prot, v_flags, v_fd);
  CAMLxparam1(v_offset);
  void *p = mmap((void *)Nativeint_val(v_addr), (size_t)Long_val(v_size),
                 Int_val(v_prot), Int_val(v_flags), Int_val(v_fd),
                 (off_t)Int64_val(v_offset));
  if (p == MAP_FAILED) raise_errno("mmap");
  CAMLreturn(caml_copy_nativeint((intnat)p));
}

CAMLprim value caml_tolk_amd_mmap_bc(value *argv, int argn) {
  return caml_tolk_amd_mmap(argv[0], argv[1], argv[2], argv[3], argv[4],
                            argv[5]);
}

CAMLprim value caml_tolk_amd_munmap(value v_addr, value v_size) {
  CAMLparam2(v_addr, v_size);
  if (munmap((void *)Nativeint_val(v_addr), (size_t)Long_val(v_size)) != 0)
    raise_errno("munmap");
  CAMLreturn(Val_unit);
}

/* Volatile access to mapped device memory. The copies must keep the OCaml
   runtime lock held: the bytes value may move under the GC otherwise. */

CAMLprim value caml_tolk_amd_read32(value v_addr) {
  CAMLparam1(v_addr);
  volatile uint32_t *p = (volatile uint32_t *)Nativeint_val(v_addr);
  CAMLreturn(caml_copy_int32(*p));
}

CAMLprim value caml_tolk_amd_write32(value v_addr, value v_v) {
  volatile uint32_t *p = (volatile uint32_t *)Nativeint_val(v_addr);
  *p = (uint32_t)Int32_val(v_v);
  return Val_unit;
}

CAMLprim value caml_tolk_amd_read64(value v_addr) {
  CAMLparam1(v_addr);
  volatile uint64_t *p = (volatile uint64_t *)Nativeint_val(v_addr);
  CAMLreturn(caml_copy_int64(*p));
}

CAMLprim value caml_tolk_amd_write64(value v_addr, value v_v) {
  volatile uint64_t *p = (volatile uint64_t *)Nativeint_val(v_addr);
  *p = (uint64_t)Int64_val(v_v);
  return Val_unit;
}

/* Unboxed 64-bit read for polling loops; Val_long drops the top bit. */
CAMLprim value caml_tolk_amd_read64_int(value v_addr) {
  volatile uint64_t *p = (volatile uint64_t *)Nativeint_val(v_addr);
  return Val_long((intnat)*p);
}

CAMLprim value caml_tolk_amd_fence(value unit) {
  atomic_thread_fence(memory_order_seq_cst);
  return Val_unit;
}

CAMLprim value caml_tolk_amd_monotonic_ms(value unit) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return Val_long((intnat)ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}

CAMLprim value caml_tolk_amd_memcpy_to_ptr(value v_dst, value v_src,
                                           value v_src_off, value v_len) {
  memcpy((void *)Nativeint_val(v_dst), Bytes_val(v_src) + Long_val(v_src_off),
         (size_t)Long_val(v_len));
  return Val_unit;
}

CAMLprim value caml_tolk_amd_memcpy_from_ptr(value v_dst, value v_dst_off,
                                             value v_src, value v_len) {
  memcpy(Bytes_val(v_dst) + Long_val(v_dst_off), (void *)Nativeint_val(v_src),
         (size_t)Long_val(v_len));
  return Val_unit;
}
