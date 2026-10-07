/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Windows: accesses to mapped ranges, calls of transports, and the C entry
   points of device_pci.h. */

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

/* Mapped windows: every access is volatile and of exactly its width. */

intnat caml_device_pci_get8(intnat a) { return *AT(a); }
value caml_device_pci_get8_byte(value a) {
  return Val_long(caml_device_pci_get8(Long_val(a)));
}

value caml_device_pci_set8(intnat a, intnat x) {
  *AT(a) = (uint8_t)x;
  return Val_unit;
}
value caml_device_pci_set8_byte(value a, value x) {
  return caml_device_pci_set8(Long_val(a), Long_val(x));
}

intnat caml_device_pci_get32(intnat a) {
  return *(volatile uint32_t *)AT(a);
}
value caml_device_pci_get32_byte(value a) {
  return Val_long(caml_device_pci_get32(Long_val(a)));
}

value caml_device_pci_set32(intnat a, intnat x) {
  *(volatile uint32_t *)AT(a) = (uint32_t)x;
  return Val_unit;
}
value caml_device_pci_set32_byte(value a, value x) {
  return caml_device_pci_set32(Long_val(a), Long_val(x));
}

int64_t caml_device_pci_get64(intnat a) {
  return (int64_t)*(volatile uint64_t *)AT(a);
}
value caml_device_pci_get64_byte(value a) {
  return caml_copy_int64(caml_device_pci_get64(Long_val(a)));
}

value caml_device_pci_set64(intnat a, int64_t x) {
  *(volatile uint64_t *)AT(a) = (uint64_t)x;
  return Val_unit;
}
value caml_device_pci_set64_byte(value a, value x) {
  return caml_device_pci_set64(Long_val(a), Int64_val(x));
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

value caml_device_pci_read(value a, value n) {
  CAMLparam2(a, n);
  CAMLlocal1(s);
  s = caml_alloc_string(Long_val(n));
  read_words(Bytes_val(s), AT(Long_val(a)), Long_val(n));
  CAMLreturn(s);
}

value caml_device_pci_write_at(value a, value s, value off, value n) {
  write_words(AT(Long_val(a)), (const uint8_t *)String_val(s) + Long_val(off),
              Long_val(n));
  return Val_unit;
}

value caml_device_pci_fill(value a, value n, value c) {
  volatile uint8_t *p = AT(Long_val(a));
  size_t len = Long_val(n), i = 0;
  uint8_t b = (uint8_t)Long_val(c);
  uint32_t w = b * 0x01010101u;
  for (; i < len && ((uintptr_t)(p + i) & 3); i++) p[i] = b;
  for (; i + 4 <= len; i += 4) *(volatile uint32_t *)(p + i) = w;
  for (; i < len; i++) p[i] = b;
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
   domains collect meanwhile. */

#define TRANSPORT(v) ((const struct device_pci_transport *)Long_val(v))

/* Copies of at most this many bytes, registers among them, stay on the C
   stack; longer ones are allocated. */
#define SMALL 64

static void transport_failed(const struct device_pci_transport *tr) {
  const char *why = tr->failed(tr->ctx);
  caml_failwith(why ? why : "the machine's transport failed");
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
  int r = t->read(t->ctx, at, buf, len);
  caml_acquire_runtime_system();
  if (r == 0) s = caml_alloc_initialized_string(len, buf);
  if (buf != small) caml_stat_free(buf);
  if (r != 0) transport_failed(t);
  CAMLreturn(s);
}

value caml_device_pci_transport_write(value tr, value a, value s) {
  CAMLparam3(tr, a, s);
  const struct device_pci_transport *t = TRANSPORT(tr);
  uint64_t at = (uint64_t)Long_val(a);
  size_t len = caml_string_length(s);
  char small[SMALL];
  char *buf = len <= SMALL ? small : caml_stat_alloc(len);
  memcpy(buf, String_val(s), len);
  caml_release_runtime_system();
  int r = t->write(t->ctx, at, buf, len);
  caml_acquire_runtime_system();
  if (buf != small) caml_stat_free(buf);
  if (r != 0) transport_failed(t);
  CAMLreturn(Val_unit);
}

/* device_pci.h */

void device_pci_window_of(value w, struct device_pci_window *out) {
  const struct device_pci_transport *tr = TRANSPORT(Field(w, 2));
  out->address = (uint64_t)Long_val(Field(w, 0));
  out->length = (size_t)Long_val(Field(w, 1));
  out->mapped = tr ? NULL : AT(Long_val(Field(w, 0)));
  out->transport = tr;
}

int device_pci_write(const struct device_pci_window *w, size_t off,
                     const void *src, size_t n) {
  if (w->mapped) {
    write_words(w->mapped + off, src, n);
    return 0;
  }
  return w->transport->write(w->transport->ctx, w->address + off, src, n);
}
