/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Windows: accesses to mapped ranges, calls of transports, and the C entry
   points of rig_pci.h. A write takes the [n] bytes of [s] from [off]. */

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

#include "rig_pci.h"

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "rig_pci stores values little-endian, as its hosts are"
#endif

#define AT(a) ((volatile uint8_t *)(a))

/* Mapped windows: every access is volatile and of exactly its width, and
   holds the runtime: it is one load or store, at a base address [a] and an
   offset [off] from it. */

intnat caml_rig_pci_get8(intnat a, intnat off) { return *AT(a + off); }
value caml_rig_pci_get8_byte(value a, value off) {
  return Val_long(caml_rig_pci_get8(Long_val(a), Long_val(off)));
}

value caml_rig_pci_set8(intnat a, intnat off, intnat x) {
  *AT(a + off) = (uint8_t)x;
  return Val_unit;
}
value caml_rig_pci_set8_byte(value a, value off, value x) {
  return caml_rig_pci_set8(Long_val(a), Long_val(off), Long_val(x));
}

intnat caml_rig_pci_get32(intnat a, intnat off) {
  return *(volatile uint32_t *)AT(a + off);
}
value caml_rig_pci_get32_byte(value a, value off) {
  return Val_long(caml_rig_pci_get32(Long_val(a), Long_val(off)));
}

value caml_rig_pci_set32(intnat a, intnat off, intnat x) {
  *(volatile uint32_t *)AT(a + off) = (uint32_t)x;
  return Val_unit;
}
value caml_rig_pci_set32_byte(value a, value off, value x) {
  return caml_rig_pci_set32(Long_val(a), Long_val(off), Long_val(x));
}

int64_t caml_rig_pci_get64(intnat a, intnat off) {
  return (int64_t)*(volatile uint64_t *)AT(a + off);
}
value caml_rig_pci_get64_byte(value a, value off) {
  return caml_copy_int64(caml_rig_pci_get64(Long_val(a), Long_val(off)));
}

value caml_rig_pci_set64(intnat a, intnat off, int64_t x) {
  *(volatile uint64_t *)AT(a + off) = (uint64_t)x;
  return Val_unit;
}
value caml_rig_pci_set64_byte(value a, value off, value x) {
  return caml_rig_pci_set64(Long_val(a), Long_val(off), Int64_val(x));
}

/* Bulk accesses go a 32-bit word at a time wherever the mapped side is
   aligned: memory behind a BAR need not accept the wider or narrower
   accesses the C library's copies make. A combining window, memory behind a
   prefetchable BAR, takes them 16 bytes at a time from its first 16-byte
   boundary: a write-combining buffer then fills in four stores, and on x86
   a streaming load reads 64 bytes from write-combined memory into a buffer
   that the next three loads hit (unverified on hardware). The process's
   side takes any alignment. Each wide access returns the offset where it
   stopped, before the last 16 bytes the copy does not hold. */

#if defined(__x86_64__)
#include <cpuid.h>
#include <emmintrin.h>
#include <smmintrin.h>

/* Whether the processor has SSE4.1's streaming load (CPUID leaf 1, ECX bit
   19), asked once: -1 until known. */
static int streaming = -1;

static int streams(void) {
  int s = __atomic_load_n(&streaming, __ATOMIC_RELAXED);
  if (s < 0) {
    unsigned a, b, c, d;
    s = __get_cpuid(1, &a, &b, &c, &d) && (c & (1u << 19));
    __atomic_store_n(&streaming, s, __ATOMIC_RELAXED);
  }
  return s;
}

__attribute__((target("sse4.1"))) static size_t
load16_streaming(uint8_t *dst, const volatile uint8_t *src, size_t i,
                 size_t n) {
  for (; i + 16 <= n; i += 16)
    _mm_storeu_si128((__m128i *)(dst + i),
                     _mm_stream_load_si128((__m128i *)(uintptr_t)(src + i)));
  return i;
}

static size_t load16(uint8_t *dst, const volatile uint8_t *src, size_t i,
                     size_t n) {
  if (streams()) return load16_streaming(dst, src, i, n);
  for (; i + 16 <= n; i += 16)
    _mm_storeu_si128((__m128i *)(dst + i), *(const volatile __m128i *)(src + i));
  return i;
}

static size_t store16(volatile uint8_t *dst, const uint8_t *src, size_t i,
                      size_t n) {
  for (; i + 16 <= n; i += 16)
    *(volatile __m128i *)(dst + i) = _mm_loadu_si128((const __m128i *)(src + i));
  return i;
}

static size_t fill16(volatile uint8_t *p, size_t i, size_t n, uint8_t b) {
  __m128i v = _mm_set1_epi8((char)b);
  for (; i + 16 <= n; i += 16) *(volatile __m128i *)(p + i) = v;
  return i;
}

#elif defined(__aarch64__)
#include <arm_neon.h>

static size_t load16(uint8_t *dst, const volatile uint8_t *src, size_t i,
                     size_t n) {
  for (; i + 16 <= n; i += 16)
    vst1q_u8(dst + i, *(const volatile uint8x16_t *)(src + i));
  return i;
}

static size_t store16(volatile uint8_t *dst, const uint8_t *src, size_t i,
                      size_t n) {
  for (; i + 16 <= n; i += 16)
    *(volatile uint8x16_t *)(dst + i) = vld1q_u8(src + i);
  return i;
}

static size_t fill16(volatile uint8_t *p, size_t i, size_t n, uint8_t b) {
  uint8x16_t v = vdupq_n_u8(b);
  for (; i + 16 <= n; i += 16) *(volatile uint8x16_t *)(p + i) = v;
  return i;
}

#else
static size_t load16(uint8_t *dst, const volatile uint8_t *src, size_t i,
                     size_t n) {
  (void)dst, (void)src, (void)n;
  return i;
}

static size_t store16(volatile uint8_t *dst, const uint8_t *src, size_t i,
                      size_t n) {
  (void)dst, (void)src, (void)n;
  return i;
}

static size_t fill16(volatile uint8_t *p, size_t i, size_t n, uint8_t b) {
  (void)p, (void)n, (void)b;
  return i;
}
#endif

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

/* The bytes from [p] to its next 16-byte boundary, at most [n]. */
static size_t head(const volatile uint8_t *p, size_t n) {
  size_t k = (16 - ((uintptr_t)p & 15)) & 15;
  return k < n ? k : n;
}

/* Loads from write-combined memory may complete out of order with later
   accesses: a wide read ends with a barrier. */
static void read_wide(uint8_t *dst, const volatile uint8_t *src, size_t n) {
  size_t i = head(src, n);
  read_words(dst, src, i);
  i = load16(dst, src, i, n);
  read_words(dst + i, src + i, n - i);
  rig_pci_barrier();
}

static void write_wide(volatile uint8_t *dst, const uint8_t *src, size_t n) {
  size_t i = head(dst, n);
  write_words(dst, src, i);
  i = store16(dst, src, i, n);
  write_words(dst + i, src + i, n - i);
}

static void fill_wide(volatile uint8_t *p, size_t n, uint8_t b) {
  size_t i = head(p, n);
  fill_words(p, i, b);
  i = fill16(p, i, n, b);
  fill_words(p + i, n - i, b);
}

static void read_bytes(uint8_t *dst, const volatile uint8_t *src, size_t n,
                       int wide) {
  if (wide)
    read_wide(dst, src, n);
  else
    read_words(dst, src, n);
}

static void write_bytes(volatile uint8_t *dst, const uint8_t *src, size_t n,
                        int wide) {
  if (wide)
    write_wide(dst, src, n);
  else
    write_words(dst, src, n);
}

static void fill_bytes(volatile uint8_t *p, size_t n, uint8_t b, int wide) {
  if (wide)
    fill_wide(p, n, b);
  else
    fill_words(p, n, b);
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

static value read_pieces(const volatile uint8_t *src, size_t n, int wide) {
  CAMLparam0();
  CAMLlocal1(s);
  uint8_t buf[PIECE];
  s = caml_alloc_string(n);
  for (size_t at = 0; at < n; at += PIECE) {
    size_t k = piece(at, n);
    caml_release_runtime_system();
    read_bytes(buf, src + at, k, wide);
    caml_acquire_runtime_system();
    memcpy(Bytes_val(s) + at, buf, k);
  }
  CAMLreturn(s);
}

value caml_rig_pci_read(value a, value n, value wide) {
  CAMLparam3(a, n, wide);
  CAMLlocal1(s);
  size_t len = Long_val(n);
  if (len >= PIECE)
    CAMLreturn(read_pieces(AT(Long_val(a)), len, Bool_val(wide)));
  s = caml_alloc_string(len);
  read_bytes(Bytes_val(s), AT(Long_val(a)), len, Bool_val(wide));
  CAMLreturn(s);
}

static void write_pieces(volatile uint8_t *dst, value s, size_t off, size_t n,
                         int wide) {
  CAMLparam1(s);
  uint8_t buf[PIECE];
  for (size_t at = 0; at < n; at += PIECE) {
    size_t k = piece(at, n);
    memcpy(buf, String_val(s) + off + at, k);
    caml_release_runtime_system();
    write_bytes(dst + at, buf, k, wide);
    caml_acquire_runtime_system();
  }
  CAMLreturn0;
}

value caml_rig_pci_write_at(value a, value s, value off, value n,
                               value wide) {
  size_t len = Long_val(n);
  if (len >= PIECE)
    write_pieces(AT(Long_val(a)), s, Long_val(off), len, Bool_val(wide));
  else
    write_bytes(AT(Long_val(a)), (const uint8_t *)String_val(s) + Long_val(off),
                len, Bool_val(wide));
  return Val_unit;
}

/* A fill reads no OCaml value: one of PIECE bytes or more runs with the
   runtime released throughout. */
value caml_rig_pci_fill(value a, value n, value c, value wide) {
  volatile uint8_t *p = AT(Long_val(a));
  size_t len = Long_val(n);
  uint8_t b = (uint8_t)Long_val(c);
  if (len < PIECE) {
    fill_bytes(p, len, b, Bool_val(wide));
    return Val_unit;
  }
  caml_release_runtime_system();
  fill_bytes(p, len, b, Bool_val(wide));
  caml_acquire_runtime_system();
  return Val_unit;
}

value caml_rig_pci_barrier(value unit) {
  (void)unit;
  rig_pci_barrier();
  return Val_unit;
}

/* The bytes at an address as a bigarray that owns nothing. */
value caml_rig_pci_bigarray(value a, value n) {
  return caml_ba_alloc_dims(CAML_BA_UINT8 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL,
                            1, (void *)Long_val(a), (intnat)Long_val(n));
}

/* Transports, from OCaml. A transport may block for a round trip, so its
   functions run without the OCaml runtime, on a copy of the bytes: other
   domains collect meanwhile. Once the transport failed, a read gives all
   ones and a write is dropped, as for a function that left the bus: no
   access raises. */

#define TRANSPORT(v) ((const struct rig_pci_transport *)Long_val(v))

/* Copies of at most this many bytes stay on the C stack; longer ones are
   allocated. */
#define SMALL 64

/* The reason the transport [tr] failed, if it did; none for the transport 0
   of this machine, which nothing fails. [failed] answers at once: it holds
   the runtime. */
value caml_rig_pci_transport_failed(value tr) {
  CAMLparam1(tr);
  CAMLlocal1(why);
  const struct rig_pci_transport *t = TRANSPORT(tr);
  if (t == NULL) CAMLreturn(Val_none);
  const char *s = t->failed(t->ctx);
  if (s == NULL) CAMLreturn(Val_none);
  why = caml_copy_string(s);
  CAMLreturn(caml_alloc_some(why));
}

/* A register of [n] bytes, 1, 4 or 8, at [a] through the transport [tr], its
   value in the low bytes of a word: a read and a write each release the
   runtime around one call of the transport, and no OCaml value is held. */
int64_t caml_rig_pci_transport_get(intnat tr, intnat a, intnat n) {
  const struct rig_pci_transport *t =
      (const struct rig_pci_transport *)tr;
  uint64_t x = 0;
  caml_release_runtime_system();
  if (t->read(t->ctx, (uint64_t)a, &x, (size_t)n) != 0) memset(&x, 0xff, n);
  caml_acquire_runtime_system();
  return (int64_t)x;
}
value caml_rig_pci_transport_get_byte(value tr, value a, value n) {
  return caml_copy_int64(
      caml_rig_pci_transport_get(Long_val(tr), Long_val(a), Long_val(n)));
}

value caml_rig_pci_transport_set(intnat tr, intnat a, intnat n, int64_t x) {
  const struct rig_pci_transport *t =
      (const struct rig_pci_transport *)tr;
  uint64_t v = (uint64_t)x;
  caml_release_runtime_system();
  (void)t->write(t->ctx, (uint64_t)a, &v, (size_t)n);
  caml_acquire_runtime_system();
  return Val_unit;
}
value caml_rig_pci_transport_set_byte(value tr, value a, value n, value x) {
  return caml_rig_pci_transport_set(Long_val(tr), Long_val(a), Long_val(n),
                                       Int64_val(x));
}

value caml_rig_pci_transport_read(value tr, value a, value n) {
  CAMLparam3(tr, a, n);
  CAMLlocal1(s);
  const struct rig_pci_transport *t = TRANSPORT(tr);
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

value caml_rig_pci_transport_write(value tr, value a, value s, value off,
                                      value n) {
  CAMLparam5(tr, a, s, off, n);
  const struct rig_pci_transport *t = TRANSPORT(tr);
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

/* rig_pci.h */

/* The fields of Window.t in their order: keep the two in sync. */
enum window_field {
  WINDOW_ADDRESS,
  WINDOW_LENGTH,
  WINDOW_TRANSPORT,
  WINDOW_COMBINES
};

void rig_pci_window_of(value w, struct rig_pci_window *out) {
  const struct rig_pci_transport *tr =
      TRANSPORT(Field(w, WINDOW_TRANSPORT));
  intnat a = Long_val(Field(w, WINDOW_ADDRESS));
  out->address = (uint64_t)a;
  out->length = (size_t)Long_val(Field(w, WINDOW_LENGTH));
  out->mapped = tr ? NULL : AT(a);
  out->transport = tr;
  out->combines = Bool_val(Field(w, WINDOW_COMBINES));
}

void rig_pci_write(const struct rig_pci_window *w, size_t off,
                      const void *src, size_t n) {
  if (w->mapped)
    write_bytes(w->mapped + off, src, n, w->combines);
  else
    (void)w->transport->write(w->transport->ctx, w->address + off, src, n);
}
