/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  WAV samples between a file's containers and an array's elements. A
  container of [k] little-endian bytes holds a sample in its top bits
  (Multimedia Programming Interface and Data Specifications 1.0, p. 59);
  an element of [w >= k] bytes in the host's order holds the container's
  bytes as its top [k], zeros below. G.711 codes decode to the 13- or
  14-bit values of ITU-T G.711, in the top bits of an int16.

  The stubs capture each bigarray's pointer before they release the
  runtime lock, and then touch only bigarray storage and their locals.
  --------------------------------------------------------------------------*/

#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

#include <stdint.h>
#include <string.h>

#define HOST_BIG (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)

/* [put p w v] stores the [w] low bytes of [v] at [p] in the host's
   order. */
static inline void put(uint8_t *p, int w, uint64_t v) {
  for (int i = 0; i < w; i++) {
    int at = HOST_BIG ? w - 1 - i : i;
    p[at] = (uint8_t)(v >> (8 * i));
  }
}

/* [caml_nx_io_wav_widen src at dst n k w valid] reads the [n] containers
   of [k] bytes of [src] from byte [at] into elements of [w] bytes of
   [dst]. It is the index of the first container with a set bit below its
   top [valid] bits, whose element and those after it are not written, or
   -1. */
CAMLprim value caml_nx_io_wav_widen(value vsrc, value vat, value vdst,
                                    value vn, value vk, value vw,
                                    value vvalid) {
  CAMLparam2(vsrc, vdst);
  const uint8_t *src = (const uint8_t *)Caml_ba_data_val(vsrc) + Long_val(vat);
  uint8_t *dst = Caml_ba_data_val(vdst);
  size_t n = Long_val(vn);
  int k = Int_val(vk), w = Int_val(vw), valid = Int_val(vvalid);
  uint64_t low = valid >= 8 * k ? 0 : (UINT64_C(1) << (8 * k - valid)) - 1;
  intnat bad = -1;
  caml_release_runtime_system();
  for (size_t i = 0; i < n; i++, src += k, dst += w) {
    uint64_t v = 0;
    for (int j = 0; j < k; j++) v |= (uint64_t)src[j] << (8 * j);
    if (v & low) {
      bad = (intnat)i;
      break;
    }
    put(dst, w, v << (8 * (w - k)));
  }
  caml_acquire_runtime_system();
  CAMLreturn(Val_long(bad));
}

CAMLprim value caml_nx_io_wav_widen_byte(value *argv, int argn) {
  (void)argn;
  return caml_nx_io_wav_widen(argv[0], argv[1], argv[2], argv[3], argv[4],
                              argv[5], argv[6]);
}

/* G.711's A-law code [c] as a 13-bit value times 8: its bits inverted at
   even positions, segment [s] and mantissa [m] give [(2 * m + 33) << (s - 1)]
   for [s > 0] and [2 * m + 1] for [s = 0], so the largest is 4032 * 8. */
static int16_t alaw(uint8_t c) {
  c ^= 0x55;
  int m = c & 0x0f, s = (c >> 4) & 7;
  int v = s == 0 ? (m << 4) + 8 : ((m << 4) + 0x108) << (s - 1);
  return (int16_t)(c & 0x80 ? v : -v);
}

/* G.711's mu-law code [c] as a 14-bit value times 4: its bits inverted,
   segment [s] and mantissa [m] give [((2 * m + 33) << s) - 33], so the
   largest is 8031 * 4 and codes 0x7f and 0xff are both 0. */
static int16_t mulaw(uint8_t c) {
  c = (uint8_t)~c;
  int m = c & 0x0f, s = (c >> 4) & 7;
  int v = (((m << 3) + 0x84) << s) - 0x84;
  return (int16_t)(c & 0x80 ? -v : v);
}

/* [caml_nx_io_wav_g711 src at dst n alaw] decodes the [n] codes of [src]
   from byte [at] into int16 elements of [dst], A-law if [alaw] and mu-law
   otherwise. */
CAMLprim value caml_nx_io_wav_g711(value vsrc, value vat, value vdst, value vn,
                                   value valaw) {
  CAMLparam2(vsrc, vdst);
  const uint8_t *src = (const uint8_t *)Caml_ba_data_val(vsrc) + Long_val(vat);
  uint8_t *dst = Caml_ba_data_val(vdst);
  size_t n = Long_val(vn);
  int16_t table[256];
  for (int c = 0; c < 256; c++)
    table[c] = Bool_val(valaw) ? alaw((uint8_t)c) : mulaw((uint8_t)c);
  caml_release_runtime_system();
  for (size_t i = 0; i < n; i++) memcpy(dst + 2 * i, &table[src[i]], 2);
  caml_acquire_runtime_system();
  CAMLreturn(Val_unit);
}
