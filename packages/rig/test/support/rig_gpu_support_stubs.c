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
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "rig.h"
#include "rig_edge.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#define Addr_val(v) ((void *)Long_val(v))

value rig_gpu_host_data(value array) {
  return Val_long((intnat)Caml_ba_data_val(array));
}

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

/* The host address of [v_b]'s first byte, with no wait or claim. */
value rig_gpu_test_buffer_at(value v_b) {
  return Val_long((intnat)rig_buffer_host(v_b));
}

/* A driver's copies */

/* The fields of [rig_gpu_edge_copy]'s [v_fields], in order. */
enum {
  f_dst,
  f_dst_offset,
  f_src,
  f_src_offset,
  f_width,
  f_rows,
  f_src_pitch,
  f_dst_pitch,
  f_local,
  f_device_memory,
  f_fill,
  f_byte,
  f_fields
};

/* Hands the device whose C state is [v_edge] the value [v_v]: one copy part
   on its queue [v_queue], whose fields [v_fields] holds as above, through
   its room check, submit and commit, as rig calls them. Answers the room
   check's answer if it is not RIG_FITS, -1 if the submit or the commit
   fails, and RIG_FITS once the value is handed over and committed. */
value rig_gpu_edge_copy(value v_edge, value v_v, value v_queue,
                        value v_fields) {
  void *self = (void *)Nativeint_val(v_edge);
  const struct rig_driver *d = rig_driver_of(self);
  if (Wosize_val(v_fields) != f_fields)
    caml_invalid_argument("rig_gpu_edge_copy: fields");
  uint64_t f[f_fields];
  for (int i = 0; i < f_fields; i++)
    f[i] = (uint64_t)Nativeint_val(Field(v_fields, i));
  struct rig_part p = {.queue = Int_val(v_queue), .kind = RIG_COPY};
  p.copy.dst = f[f_dst];
  p.copy.dst_offset = f[f_dst_offset];
  p.copy.src = f[f_src];
  p.copy.src_offset = f[f_src_offset];
  p.copy.width = f[f_width];
  p.copy.rows = f[f_rows];
  p.copy.src_pitch = f[f_src_pitch];
  p.copy.dst_pitch = f[f_dst_pitch];
  p.copy.local = (int)f[f_local];
  p.copy.device_memory = (int)f[f_device_memory];
  p.copy.fill = (int)f[f_fill];
  p.copy.byte = (uint8_t)f[f_byte];
  _Alignas(8) uint8_t scratch[RIG_SCRATCH];
  int r = d->room(self, &p, 1, scratch, NULL);
  if (r != RIG_FITS) return Val_int(r);
  const char *failure = NULL;
  uint64_t v = (uint64_t)Long_val(v_v);
  r = d->submit(self, v, NULL, 0, &p, 1, scratch, NULL, NULL, 0, NULL,
                &failure);
  if (r == RIG_FAILED) return Val_int(-1);
  if (r == RIG_OK && d->commit(self, v, &failure) != RIG_OK)
    return Val_int(-1);
  return Val_int(RIG_FITS);
}
