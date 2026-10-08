/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The GPU lock, host memory and the C room check. Every stub holds the
   runtime: none blocks. */

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
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "device_nv.h"

#define Ptr_val(v) ((void *)Long_val(v))

/* The GPU lock */

/* Whether the exclusive lock of the file [v_path] was taken without
   waiting. It is held until the process exits. */
value device_nv_test_lock(value v_path) {
#if defined(_WIN32)
  (void)v_path;
  return Val_false;
#else
  int fd = open(String_val(v_path), O_RDONLY | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0) return Val_false;
  if (flock(fd, LOCK_EX | LOCK_NB) == 0) return Val_true;
  close(fd);
  return Val_false;
#endif
}

/* Host memory */

value device_nv_test_page_size(value unit) {
  (void)unit;
#if defined(_WIN32)
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  return Val_long(info.dwPageSize);
#else
  return Val_long(sysconf(_SC_PAGESIZE));
#endif
}

/* [v_n] zeroed writable bytes from a page. */
value device_nv_test_pages(value v_n) {
  size_t n = Long_val(v_n);
#if defined(_WIN32)
  void *p = VirtualAlloc(NULL, n, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
  if (p == NULL) caml_raise_out_of_memory();
#else
  void *p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS,
                 -1, 0);
  if (p == MAP_FAILED) caml_raise_out_of_memory();
#endif
  return Val_long((intnat)p);
}

value device_nv_test_free_pages(value v_p, value v_n) {
#if defined(_WIN32)
  (void)v_n;
  VirtualFree(Ptr_val(v_p), 0, MEM_RELEASE);
#else
  munmap(Ptr_val(v_p), Long_val(v_n));
#endif
  return Val_unit;
}

value device_nv_test_get64(value v_p) {
  _Atomic uint64_t *p = Ptr_val(v_p);
  return Val_long((intnat)atomic_load_explicit(p, memory_order_acquire));
}

value device_nv_test_set64(value v_p, value v_x) {
  _Atomic uint64_t *p = Ptr_val(v_p);
  atomic_store_explicit(p, (uint64_t)Long_val(v_x), memory_order_release);
  return Val_unit;
}

value device_nv_test_read(value v_p, value v_n) {
  CAMLparam2(v_p, v_n);
  CAMLlocal1(s);
  s = caml_alloc_string(Long_val(v_n));
  memcpy(Bytes_val(s), Ptr_val(v_p), Long_val(v_n));
  CAMLreturn(s);
}

value device_nv_test_write(value v_p, value v_s) {
  memcpy(Ptr_val(v_p), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}

/* Byte [i] of the pattern of [seed]. */
static uint8_t pattern_byte(uint64_t seed, uint64_t i) {
  uint64_t h = (i + (seed << 40)) * 0x9E3779B97F4A7C1ULL;
  return (uint8_t)((h ^ (h >> 29)) >> 17);
}

value device_nv_test_pattern(value v_p, value v_n, value v_seed) {
  uint8_t *p = Ptr_val(v_p);
  uint64_t n = Long_val(v_n), seed = Long_val(v_seed);
  for (uint64_t i = 0; i < n; i++) p[i] = pattern_byte(seed, i);
  return Val_unit;
}

value device_nv_test_mismatch(value v_p, value v_n, value v_seed) {
  const uint8_t *p = Ptr_val(v_p);
  uint64_t n = Long_val(v_n), seed = Long_val(v_seed);
  for (uint64_t i = 0; i < n; i++)
    if (p[i] != pattern_byte(seed, i)) return Val_long(i);
  return Val_long(-1);
}

/* The room check */

static int no_fill(void *queue, void *arg, uint64_t v) {
  (void)queue;
  (void)arg;
  (void)v;
  return 0;
}

/* What device_nv_room answers for one part on queue [v_queue]: [v_words]
   zero words, a fill iff [v_fill], [v_units] ring units, [v_bytes] segment
   bytes, a copy of one byte between the handles 0 iff [v_copy], and the
   indices [v_after]. */
value device_nv_test_room(value v_self, value v_queue, value v_words,
                          value v_fill, value v_units, value v_bytes,
                          value v_copy, value v_after) {
  static const uint32_t words[8];
  int after[8];
  struct nx_part p;
  memset(&p, 0, sizeof p);
  p.queue = Int_val(v_queue);
  p.n = Long_val(v_words);
  if (p.n > 8) caml_invalid_argument("device_nv_test_room");
  p.words = p.n > 0 ? words : NULL;
  p.fill = Bool_val(v_fill) ? no_fill : NULL;
  p.ring_units = Long_val(v_units);
  p.segment_bytes = Long_val(v_bytes);
  p.copy_bytes = Bool_val(v_copy) ? 1 : 0;
  p.nafter = (int)Wosize_val(v_after);
  if (p.nafter > 8) caml_invalid_argument("device_nv_test_room");
  for (int i = 0; i < p.nafter; i++) after[i] = Int_val(Field(v_after, i));
  p.after = after;
  return Val_int(device_nv_room((void *)Nativeint_val(v_self), &p, 1));
}

value device_nv_test_room_byte(value *argv, int argn) {
  (void)argn;
  return device_nv_test_room(argv[0], argv[1], argv[2], argv[3], argv[4],
                             argv[5], argv[6], argv[7]);
}
