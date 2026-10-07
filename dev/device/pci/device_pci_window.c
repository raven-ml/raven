/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Windows: accesses to mapped ranges, calls of transports, and the C entry
   points of device_pci.h. A write takes the [n] bytes of [s] from [off]. */

#define _GNU_SOURCE

#include <stdint.h>
#include <string.h>

#define CAML_NAME_SPACE
#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include "device_pci.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "device_pci stores values little-endian, as its hosts are"
#endif

#define AT(a) ((volatile uint8_t *)(a))

/* Mapped windows: every access is volatile and of exactly its width, and
   holds the runtime: it is one load or store, at a base address [a] and an
   offset [off] from it. */

intnat caml_device_pci_get8(intnat a, intnat off) { return *AT(a + off); }
value caml_device_pci_get8_byte(value a, value off) {
  return Val_long(caml_device_pci_get8(Long_val(a), Long_val(off)));
}

value caml_device_pci_set8(intnat a, intnat off, intnat x) {
  *AT(a + off) = (uint8_t)x;
  return Val_unit;
}
value caml_device_pci_set8_byte(value a, value off, value x) {
  return caml_device_pci_set8(Long_val(a), Long_val(off), Long_val(x));
}

intnat caml_device_pci_get32(intnat a, intnat off) {
  return *(volatile uint32_t *)AT(a + off);
}
value caml_device_pci_get32_byte(value a, value off) {
  return Val_long(caml_device_pci_get32(Long_val(a), Long_val(off)));
}

value caml_device_pci_set32(intnat a, intnat off, intnat x) {
  *(volatile uint32_t *)AT(a + off) = (uint32_t)x;
  return Val_unit;
}
value caml_device_pci_set32_byte(value a, value off, value x) {
  return caml_device_pci_set32(Long_val(a), Long_val(off), Long_val(x));
}

int64_t caml_device_pci_get64(intnat a, intnat off) {
  return (int64_t)*(volatile uint64_t *)AT(a + off);
}
value caml_device_pci_get64_byte(value a, value off) {
  return caml_copy_int64(caml_device_pci_get64(Long_val(a), Long_val(off)));
}

value caml_device_pci_set64(intnat a, intnat off, int64_t x) {
  *(volatile uint64_t *)AT(a + off) = (uint64_t)x;
  return Val_unit;
}
value caml_device_pci_set64_byte(value a, value off, value x) {
  return caml_device_pci_set64(Long_val(a), Long_val(off), Int64_val(x));
}

/* Bulk accesses go a 32-bit word at a time wherever the mapped side is
   aligned: memory behind a BAR need not accept the wider or narrower
   accesses the C library's copies make. The process's side takes any
   alignment. */
static void read_words(uint8_t *dst, const volatile uint8_t *src, size_t n) {
  size_t i = 0;
  for (; i < n && ((uintptr_t)(src + i) & 3); i++) dst[i] = src[i];
  for (; i + 4 <= n; i += 4) {
    uint32_t w = *(const volatile uint32_t *)(src + i);
    memcpy(dst + i, &w, 4);
  }
  for (; i < n; i++) dst[i] = src[i];
}

static void write_words(volatile uint8_t *dst, const uint8_t *src, size_t n) {
  size_t i = 0;
  for (; i < n && ((uintptr_t)(dst + i) & 3); i++) dst[i] = src[i];
  for (; i + 4 <= n; i += 4) {
    uint32_t w;
    memcpy(&w, src + i, 4);
    *(volatile uint32_t *)(dst + i) = w;
  }
  for (; i < n; i++) dst[i] = src[i];
}

static void fill_words(volatile uint8_t *p, size_t n, uint8_t b) {
  uint32_t w = b * 0x01010101u;
  size_t i = 0;
  for (; i < n && ((uintptr_t)(p + i) & 3); i++) p[i] = b;
  for (; i + 4 <= n; i += 4) *(volatile uint32_t *)(p + i) = w;
  for (; i < n; i++) p[i] = b;
}

/* Copies of at least PIECE bytes release the runtime, so their externals are
   not [@@noalloc]. They move a piece at a time through a buffer on the C
   stack, since OCaml strings may move while the runtime is released; shorter
   copies hold it and copy in place. A read from a BAR is a PCIe round trip,
   about 1 us a word, and every other domain waits at its next collection for
   one that holds the runtime: 8 KiB bounds that wait to about 2 ms. On an M1
   Max a release and reacquire take 42 ns and a copy of process memory a word
   at a time runs at 10 GB/s, 800 ns for 8 KiB; a piece through the buffer
   adds about a quarter to that, the pair 5% and the buffer's memcpy the rest,
   and under 0.01% to the 2 ms of a BAR's. */
#define PIECE 8192

static size_t piece(size_t at, size_t n) {
  return n - at < PIECE ? n - at : PIECE;
}

static value read_pieces(const volatile uint8_t *src, size_t n) {
  CAMLparam0();
  CAMLlocal1(s);
  uint8_t buf[PIECE];
  s = caml_alloc_string(n);
  for (size_t at = 0; at < n; at += PIECE) {
    size_t k = piece(at, n);
    caml_release_runtime_system();
    read_words(buf, src + at, k);
    caml_acquire_runtime_system();
    memcpy(Bytes_val(s) + at, buf, k);
  }
  CAMLreturn(s);
}

value caml_device_pci_read(value a, value n) {
  CAMLparam2(a, n);
  CAMLlocal1(s);
  size_t len = Long_val(n);
  if (len >= PIECE) CAMLreturn(read_pieces(AT(Long_val(a)), len));
  s = caml_alloc_string(len);
  read_words(Bytes_val(s), AT(Long_val(a)), len);
  CAMLreturn(s);
}

static void write_pieces(volatile uint8_t *dst, value s, size_t off, size_t n) {
  CAMLparam1(s);
  uint8_t buf[PIECE];
  for (size_t at = 0; at < n; at += PIECE) {
    size_t k = piece(at, n);
    memcpy(buf, String_val(s) + off + at, k);
    caml_release_runtime_system();
    write_words(dst + at, buf, k);
    caml_acquire_runtime_system();
  }
  CAMLreturn0;
}

value caml_device_pci_write_at(value a, value s, value off, value n) {
  size_t len = Long_val(n);
  if (len >= PIECE)
    write_pieces(AT(Long_val(a)), s, Long_val(off), len);
  else
    write_words(AT(Long_val(a)), (const uint8_t *)String_val(s) + Long_val(off),
                len);
  return Val_unit;
}

/* A fill reads no OCaml value: one of PIECE bytes or more runs with the
   runtime released throughout. */
value caml_device_pci_fill(value a, value n, value c) {
  volatile uint8_t *p = AT(Long_val(a));
  size_t len = Long_val(n);
  uint8_t b = (uint8_t)Long_val(c);
  if (len < PIECE) {
    fill_words(p, len, b);
    return Val_unit;
  }
  caml_release_runtime_system();
  fill_words(p, len, b);
  caml_acquire_runtime_system();
  return Val_unit;
}

value caml_device_pci_barrier(value unit) {
  (void)unit;
  device_pci_barrier();
  return Val_unit;
}

/* The bytes at an address as a bigarray that owns nothing. */
value caml_device_pci_bigarray(value a, value n) {
  return caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL,
                            1, (void *)Long_val(a), (intnat)Long_val(n));
}

/* Transports, from OCaml. A transport may block for a round trip, so its
   functions run without the OCaml runtime, on a copy of the bytes: other
   domains collect meanwhile. Once the transport failed, a read gives all
   ones and a write is dropped, as for a function that left the bus: no
   access raises. */

#define TRANSPORT(v) ((const struct device_pci_transport *)Long_val(v))

/* Copies of at most this many bytes stay on the C stack; longer ones are
   allocated. */
#define SMALL 64

/* The reason the transport [tr] failed, if it did; none for the transport 0
   of this machine, which nothing fails. [failed] answers at once: it holds
   the runtime. */
value caml_device_pci_transport_failed(value tr) {
  CAMLparam1(tr);
  CAMLlocal1(why);
  const struct device_pci_transport *t = TRANSPORT(tr);
  if (t == NULL) CAMLreturn(Val_none);
  const char *s = t->failed(t->ctx);
  if (s == NULL) CAMLreturn(Val_none);
  why = caml_copy_string(s);
  CAMLreturn(caml_alloc_some(why));
}

/* A register of [n] bytes, 1, 4 or 8, at [a] through the transport [tr], its
   value in the low bytes of a word: a read and a write each release the
   runtime around one call of the transport, and no OCaml value is held. */
int64_t caml_device_pci_transport_get(intnat tr, intnat a, intnat n) {
  const struct device_pci_transport *t =
      (const struct device_pci_transport *)tr;
  uint64_t x = 0;
  caml_release_runtime_system();
  if (t->read(t->ctx, (uint64_t)a, &x, (size_t)n) != 0) memset(&x, 0xff, n);
  caml_acquire_runtime_system();
  return (int64_t)x;
}
value caml_device_pci_transport_get_byte(value tr, value a, value n) {
  return caml_copy_int64(
      caml_device_pci_transport_get(Long_val(tr), Long_val(a), Long_val(n)));
}

value caml_device_pci_transport_set(intnat tr, intnat a, intnat n, int64_t x) {
  const struct device_pci_transport *t =
      (const struct device_pci_transport *)tr;
  uint64_t v = (uint64_t)x;
  caml_release_runtime_system();
  (void)t->write(t->ctx, (uint64_t)a, &v, (size_t)n);
  caml_acquire_runtime_system();
  return Val_unit;
}
value caml_device_pci_transport_set_byte(value tr, value a, value n, value x) {
  return caml_device_pci_transport_set(Long_val(tr), Long_val(a), Long_val(n),
                                       Int64_val(x));
}

value caml_device_pci_transport_read(value tr, value a, value n) {
  CAMLparam3(tr, a, n);
  CAMLlocal1(s);
  const struct device_pci_transport *t = TRANSPORT(tr);
  uint64_t at = (uint64_t)Long_val(a);
  size_t len = Long_val(n);
  char small[SMALL];
  char *buf = len <= SMALL ? small : caml_stat_alloc(len);
  caml_release_runtime_system();
  if (t->read(t->ctx, at, buf, len) != 0) memset(buf, 0xff, len);
  caml_acquire_runtime_system();
  s = caml_alloc_initialized_string(len, buf);
  if (buf != small) caml_stat_free(buf);
  CAMLreturn(s);
}

value caml_device_pci_transport_write(value tr, value a, value s, value off,
                                      value n) {
  CAMLparam5(tr, a, s, off, n);
  const struct device_pci_transport *t = TRANSPORT(tr);
  uint64_t at = (uint64_t)Long_val(a);
  size_t len = Long_val(n);
  char small[SMALL];
  char *buf = len <= SMALL ? small : caml_stat_alloc(len);
  memcpy(buf, String_val(s) + Long_val(off), len);
  caml_release_runtime_system();
  (void)t->write(t->ctx, at, buf, len);
  caml_acquire_runtime_system();
  if (buf != small) caml_stat_free(buf);
  CAMLreturn(Val_unit);
}

/* device_pci.h */

/* The fields of Window.t in their order: keep the two in sync. */
enum window_field { WINDOW_ADDRESS, WINDOW_LENGTH, WINDOW_TRANSPORT };

void device_pci_window_of(value w, struct device_pci_window *out) {
  const struct device_pci_transport *tr =
      TRANSPORT(Field(w, WINDOW_TRANSPORT));
  intnat a = Long_val(Field(w, WINDOW_ADDRESS));
  out->address = (uint64_t)a;
  out->length = (size_t)Long_val(Field(w, WINDOW_LENGTH));
  out->mapped = tr ? NULL : AT(a);
  out->transport = tr;
}

void device_pci_write(const struct device_pci_window *w, size_t off,
                      const void *src, size_t n) {
  if (w->mapped)
    write_words(w->mapped + off, src, n);
  else
    (void)w->transport->write(w->transport->ctx, w->address + off, src, n);
}
