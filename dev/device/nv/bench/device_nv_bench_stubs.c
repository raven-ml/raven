/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The floors of the NV bench: each row's submissions through
   device_nv_room and device_nv_submit, called from C in a loop on the
   device the row opened, then a spin until the device's timeline word
   holds the last value. The submission is the least sequence the hardware
   needs (the driver's words in a segment, a ring entry, GP_PUT, the
   doorbell), so a floor's distance to its row is the OCaml side of the
   driver.

   Also host memory for the rows: page-aligned bytes, and stores into memory
   the host addresses. A failing call raises Failure. Every stub holds the
   runtime. */

#define _GNU_SOURCE

#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include <device_nv.h>

#define Ptr_val(v) ((void *)Nativeint_val(v))

/* The device, its timeline word, and the last value given to it. */
static void *self;
static _Atomic uint64_t *word;
static uint64_t last;

value device_nv_bench_start(value v_self, value v_word, value v_last) {
  self = Ptr_val(v_self);
  word = (_Atomic uint64_t *)Long_val(v_word);
  last = (uint64_t)Long_val(v_last);
  return Val_unit;
}

static void submit(const struct nx_wait *waits, int nwaits,
                   const struct nx_part *parts, int nparts) {
  const char *failure = NULL;
  if (device_nv_room(self, parts, nparts) != NX_FITS)
    caml_failwith("device_nv_room: the parts do not fit");
  if (device_nv_submit(self, ++last, waits, nwaits, parts, nparts, NULL, 0,
                       &failure) != NX_OK)
    caml_failwith(failure);
}

static void spin(void) {
  while (atomic_load_explicit(word, memory_order_acquire) < last) {
  }
}

/* [v_k] submissions of no parts, then the wait for the last. */
value device_nv_bench_release(value v_k) {
  for (long i = 0; i < Long_val(v_k); i++) submit(NULL, 0, NULL, 0);
  spin();
  return Val_unit;
}

/* A part of no entries on COPY:0 at even values, no part at odd ones, so
   that the release changes channel at each value. */
value device_nv_bench_switch(value unit) {
  struct nx_part copy = {.queue = 1};
  (void)unit;
  submit(NULL, 0, &copy, (last & 1) == 0);
  spin();
  return Val_unit;
}

/* [v_n] waits for the 64-bit word at [v_at] to reach 1, then a release. */
value device_nv_bench_waits(value v_at, value v_n) {
  struct nx_wait waits[16];
  int n = Int_val(v_n);
  if (n > 16) caml_invalid_argument("device_nv_bench_waits: more than 16");
  for (int i = 0; i < n; i++)
    waits[i] = (struct nx_wait){(uint64_t)Long_val(v_at), 1, NX_WORD};
  submit(waits, n, NULL, 0);
  spin();
  return Val_unit;
}

/* The ring entry of two words [v_lo], [v_hi] on COMPUTE:0. */
value device_nv_bench_entry(value v_lo, value v_hi) {
  uint32_t words[2] = {(uint32_t)Long_val(v_lo), (uint32_t)Long_val(v_hi)};
  struct nx_part p = {.queue = 0, .words = words, .n = 2};
  submit(NULL, 0, &p, 1);
  spin();
  return Val_unit;
}

/* A copy of [v_n] bytes from the GPU address [v_src] to [v_dst] on
   COPY:0. */
value device_nv_bench_copy(value v_dst, value v_src, value v_n) {
  struct nx_part p = {.queue = 1,
                      .copy_dst = (uint64_t)Long_val(v_dst),
                      .copy_src = (uint64_t)Long_val(v_src),
                      .copy_bytes = (uint64_t)Long_val(v_n)};
  submit(NULL, 0, &p, 1);
  spin();
  return Val_unit;
}

/* Host memory */

/* [v_n] zeroed bytes from a page boundary, kept for the process's life. */
value device_nv_bench_pages(value v_n) {
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

value device_nv_bench_write(value v_p, value v_s) {
  memcpy((void *)Long_val(v_p), String_val(v_s), caml_string_length(v_s));
  return Val_unit;
}

value device_nv_bench_set64(value v_p, value v_x) {
  _Atomic uint64_t *p = (_Atomic uint64_t *)Long_val(v_p);
  atomic_store_explicit(p, (uint64_t)Long_val(v_x), memory_order_release);
  return Val_unit;
}
