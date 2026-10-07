/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* What the suites need from C: process memory to map, a far machine reached
   through a transport, the accesses of device_pci.h, and the monotonic
   clock.

   A far machine holds [size] bytes at addresses [base, base + size) and
   nothing else. Its transport logs every access, fails on request, and
   fails on an access outside its bytes, so an access beyond a window whose
   bytes are the machine's is a failure the suite sees. Held, its accesses
   block until it is let go, as a link's round trip does. */

#define _POSIX_C_SOURCE 200809L

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "device_pci.h"

value test_memory(value n) {
  void *p = calloc(1, Long_val(n) + 1);
  if (p == NULL) caml_raise_out_of_memory();
  return Val_long((intnat)p);
}

/* Far machines */

#define LOG 4096

struct access {
  int write;
  uint64_t address;
  size_t n;
};

struct far {
  struct device_pci_transport transport; /* first: its address is the far's */
  uint64_t base;
  size_t size;
  uint8_t *bytes;
  int broken, outside, held, waiting;
  size_t logged;
  struct access log[LOG];
};

/* The longest a held access waits before it fails. */
#define HOLD_MS 2000

/* Whether [f] was let go within HOLD_MS. */
static int let_go(struct far *f) {
  struct timespec ms = {0, 1000000};
  __atomic_store_n(&f->waiting, 1, __ATOMIC_RELEASE);
  for (int i = 0; i < HOLD_MS; i++) {
    if (!__atomic_load_n(&f->held, __ATOMIC_ACQUIRE)) return 1;
    nanosleep(&ms, NULL);
  }
  return 0;
}

static int far_access(struct far *f, int write, uint64_t a, void *p,
                      size_t n) {
  if (__atomic_load_n(&f->broken, __ATOMIC_ACQUIRE)) return -1;
  if (__atomic_load_n(&f->held, __ATOMIC_ACQUIRE) && !let_go(f)) return -1;
  if (a < f->base || a - f->base > f->size || n > f->size - (a - f->base)) {
    __atomic_store_n(&f->outside, 1, __ATOMIC_RELEASE);
    return -1;
  }
  size_t i = __atomic_fetch_add(&f->logged, 1, __ATOMIC_ACQ_REL);
  if (i < LOG) f->log[i] = (struct access){write, a, n};
  if (write)
    memcpy(f->bytes + (a - f->base), p, n);
  else
    memcpy(p, f->bytes + (a - f->base), n);
  return 0;
}

static int far_read(void *ctx, uint64_t a, void *dst, size_t n) {
  return far_access(ctx, 0, a, dst, n);
}

static int far_write(void *ctx, uint64_t a, const void *src, size_t n) {
  return far_access(ctx, 1, a, (void *)src, n);
}

static const char *far_failed(void *ctx) {
  struct far *f = ctx;
  if (__atomic_load_n(&f->broken, __ATOMIC_ACQUIRE))
    return "far: the link broke";
  if (__atomic_load_n(&f->outside, __ATOMIC_ACQUIRE))
    return "far: an access outside the machine's bytes";
  return NULL;
}

value test_far(value base, value size) {
  struct far *f = calloc(1, sizeof *f);
  uint8_t *bytes = calloc(1, Long_val(size) + 1);
  if (f == NULL || bytes == NULL) caml_raise_out_of_memory();
  f->transport = (struct device_pci_transport){f, far_read, far_write,
                                                far_failed};
  f->base = Long_val(base);
  f->size = Long_val(size);
  f->bytes = bytes;
  return Val_long((intnat)f);
}

value test_far_break(value far) {
  struct far *f = (struct far *)Long_val(far);
  __atomic_store_n(&f->broken, 1, __ATOMIC_RELEASE);
  return Val_unit;
}

value test_far_hold(value far) {
  struct far *f = (struct far *)Long_val(far);
  __atomic_store_n(&f->waiting, 0, __ATOMIC_RELEASE);
  __atomic_store_n(&f->held, 1, __ATOMIC_RELEASE);
  return Val_unit;
}

/* Whether an access waits on the hold. */
value test_far_waiting(value far) {
  struct far *f = (struct far *)Long_val(far);
  return Val_bool(__atomic_load_n(&f->waiting, __ATOMIC_ACQUIRE));
}

value test_far_let_go(value far) {
  struct far *f = (struct far *)Long_val(far);
  __atomic_store_n(&f->held, 0, __ATOMIC_RELEASE);
  return Val_unit;
}

/* The accesses since the last call, oldest first, as (write, address, n). */
value test_far_log(value far) {
  CAMLparam1(far);
  CAMLlocal3(l, a, cell);
  struct far *f = (struct far *)Long_val(far);
  if (f->logged > LOG) caml_failwith("test_far_log: the log overflowed");
  l = Val_emptylist;
  for (size_t i = f->logged; i > 0; i--) {
    struct access *x = &f->log[i - 1];
    a = caml_alloc_tuple(3);
    Store_field(a, 0, Val_bool(x->write));
    Store_field(a, 1, Val_long(x->address));
    Store_field(a, 2, Val_long(x->n));
    cell = caml_alloc_small(2, Tag_cons);
    Field(cell, 0) = a;
    Field(cell, 1) = l;
    l = cell;
  }
  f->logged = 0;
  CAMLreturn(l);
}

/* The accesses of device_pci.h */

static struct device_pci_window window(value w) {
  struct device_pci_window x;
  device_pci_window_of(w, &x);
  return x;
}

/* (address, length, mapped pointer or 0, has a transport) */
value test_window_of(value w) {
  CAMLparam1(w);
  CAMLlocal1(r);
  struct device_pci_window x = window(w);
  r = caml_alloc_tuple(4);
  Store_field(r, 0, Val_long(x.address));
  Store_field(r, 1, Val_long(x.length));
  Store_field(r, 2, Val_long((intnat)x.mapped));
  Store_field(r, 3, Val_bool(x.transport != NULL));
  CAMLreturn(r);
}

value test_store32(value w, value off, value x) {
  struct device_pci_window v = window(w);
  return Val_int(device_pci_store32(&v, Long_val(off), (uint32_t)Long_val(x)));
}

value test_store64(value w, value off, value x) {
  struct device_pci_window v = window(w);
  return Val_int(device_pci_store64(&v, Long_val(off), Int64_val(x)));
}

value test_load32(value w, value off) {
  struct device_pci_window v = window(w);
  uint32_t x;
  if (device_pci_load32(&v, Long_val(off), &x)) return Val_none;
  return caml_alloc_some(Val_long(x));
}

value test_load64(value w, value off) {
  CAMLparam2(w, off);
  struct device_pci_window v = window(w);
  uint64_t x;
  if (device_pci_load64(&v, Long_val(off), &x)) CAMLreturn(Val_none);
  CAMLreturn(caml_alloc_some(caml_copy_int64(x)));
}

value test_write(value w, value off, value s) {
  struct device_pci_window v = window(w);
  return Val_int(
      device_pci_write(&v, Long_val(off), String_val(s), caml_string_length(s)));
}

/* The monotonic clock */

value test_now_ns(value unit) {
  (void)unit;
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return Val_long((intnat)t.tv_sec * 1000000000 + t.tv_nsec);
}
