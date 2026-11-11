/*--------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC

  TIFF's packed samples and predictors over the rows of a strip or tile.
  Samples of a width other than 8, 16 or 32 bits are packed, most
  significant bit first, each row padded to a byte (TIFF 6.0, section 8,
  Compression); [unpack] reads them into whole bytes. The predictors are
  undone in place: horizontal differencing (TIFF 6.0, section 14), which
  replaces each sample but a row's first pixel by its difference from the
  same sample of the pixel before, modulo 2^bits; and the floating point
  predictor (Adobe Photoshop TIFF Technical Note 3), which reorders each
  row's bytes into planes, most significant byte of every sample first,
  then replaces each byte but the first pixel's by its difference from the
  byte one pixel before. The floating point predictor's planes are in the
  same order whatever the file's byte order.
  --------------------------------------------------------------------------*/

#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include <stdint.h>
#include <string.h>

#define HOST_BIG (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)

/* The predictors' codes (TIFF 6.0, section 14; Technical Note 3). */
enum { NONE = 1, HORIZONTAL = 2, FLOATING_POINT = 3 };

/* [swap p n bytes] reverses the bytes of each of the [n] samples of
   [bytes] bytes at [p]. */
static void swap(uint8_t *p, size_t n, int bytes) {
  for (size_t i = 0; i < n; i++, p += bytes)
    for (int a = 0, b = bytes - 1; a < b; a++, b--) {
      uint8_t t = p[a];
      p[a] = p[b];
      p[b] = t;
    }
}

/* Horizontal differencing over [n] samples of a row in the host's byte
   order, [spp] to a pixel: [sum] undoes it, [diff] applies it. */
#define HORIZONTAL_FNS(T)                                                     \
  static void sum_##T(uint8_t *row, size_t n, size_t spp) {                   \
    for (size_t i = spp; i < n; i++) {                                        \
      T a, b;                                                                 \
      memcpy(&a, row + i * sizeof(T), sizeof(T));                             \
      memcpy(&b, row + (i - spp) * sizeof(T), sizeof(T));                     \
      a = (T)(a + b);                                                         \
      memcpy(row + i * sizeof(T), &a, sizeof(T));                             \
    }                                                                         \
  }                                                                           \
  static void diff_##T(uint8_t *row, size_t n, size_t spp) {                  \
    for (size_t i = n; i-- > spp;) {                                          \
      T a, b;                                                                 \
      memcpy(&a, row + i * sizeof(T), sizeof(T));                             \
      memcpy(&b, row + (i - spp) * sizeof(T), sizeof(T));                     \
      a = (T)(a - b);                                                         \
      memcpy(row + i * sizeof(T), &a, sizeof(T));                             \
    }                                                                         \
  }

HORIZONTAL_FNS(uint8_t)
HORIZONTAL_FNS(uint16_t)
HORIZONTAL_FNS(uint32_t)
HORIZONTAL_FNS(uint64_t)

static void horizontal(uint8_t *row, size_t n, size_t spp, int bytes,
                       int undo) {
  switch (bytes) {
  case 1:
    undo ? sum_uint8_t(row, n, spp) : diff_uint8_t(row, n, spp);
    break;
  case 2:
    undo ? sum_uint16_t(row, n, spp) : diff_uint16_t(row, n, spp);
    break;
  case 4:
    undo ? sum_uint32_t(row, n, spp) : diff_uint32_t(row, n, spp);
    break;
  default:
    undo ? sum_uint64_t(row, n, spp) : diff_uint64_t(row, n, spp);
  }
}

/* The index in a host-order sample of [bytes] bytes of its byte of
   significance [k], 0 the most significant. */
static inline int host_byte(int k, int bytes) {
  return HOST_BIG ? k : bytes - 1 - k;
}

/* The floating point predictor's planes of a row of [n] host-order
   samples, written to [planes]: [undo] reads [planes] back into [row]. */
static void floating_point(uint8_t *row, uint8_t *planes, size_t n,
                           size_t spp, int bytes, int undo) {
  size_t len = n * (size_t)bytes;
  if (undo) {
    memcpy(planes, row, len);
    for (size_t i = spp; i < len; i++)
      planes[i] = (uint8_t)(planes[i] + planes[i - spp]);
    for (int k = 0; k < bytes; k++) {
      int at = host_byte(k, bytes);
      const uint8_t *plane = planes + (size_t)k * n;
      for (size_t j = 0; j < n; j++) row[j * bytes + at] = plane[j];
    }
  } else {
    for (int k = 0; k < bytes; k++) {
      int at = host_byte(k, bytes);
      uint8_t *plane = planes + (size_t)k * n;
      for (size_t j = 0; j < n; j++) plane[j] = row[j * bytes + at];
    }
    for (size_t i = len; i-- > spp;)
      planes[i] = (uint8_t)(planes[i] - planes[i - spp]);
    memcpy(row, planes, len);
  }
}

/* [caml_nx_io_tiff_unpredict buf scratch rows n spp bytes predictor big]
   turns the first [rows] rows of [n] samples of [buf], as a file of byte
   order [big] stores them with [predictor], into samples in the host's
   byte order. [scratch] holds a row. */
CAMLprim value caml_nx_io_tiff_unpredict(value vbuf, value vscratch,
                                         value vrows, value vn, value vspp,
                                         value vbytes, value vpredictor,
                                         value vbig) {
  uint8_t *buf = Caml_ba_data_val(vbuf), *scratch = Caml_ba_data_val(vscratch);
  size_t rows = Long_val(vrows), n = Long_val(vn), spp = Long_val(vspp);
  int bytes = Int_val(vbytes), predictor = Int_val(vpredictor);
  int swapped = Bool_val(vbig) != HOST_BIG;
  size_t len = n * (size_t)bytes;
  for (size_t r = 0; r < rows; r++) {
    uint8_t *row = buf + r * len;
    if (predictor == FLOATING_POINT) {
      floating_point(row, scratch, n, spp, bytes, 1);
      continue;
    }
    if (swapped && bytes > 1) swap(row, n, bytes);
    if (predictor == HORIZONTAL) horizontal(row, n, spp, bytes, 1);
  }
  return Val_unit;
}

CAMLprim value caml_nx_io_tiff_unpredict_byte(value *argv, int argn) {
  (void)argn;
  return caml_nx_io_tiff_unpredict(argv[0], argv[1], argv[2], argv[3],
                                   argv[4], argv[5], argv[6], argv[7]);
}

/* [caml_nx_io_tiff_predict buf scratch rows n spp bytes predictor] turns
   the first [rows] rows of [n] little-endian samples of [buf] into what a
   little-endian file stores with [predictor]. [scratch] holds a row. */
CAMLprim value caml_nx_io_tiff_predict(value vbuf, value vscratch, value vrows,
                                       value vn, value vspp, value vbytes,
                                       value vpredictor) {
  uint8_t *buf = Caml_ba_data_val(vbuf), *scratch = Caml_ba_data_val(vscratch);
  size_t rows = Long_val(vrows), n = Long_val(vn), spp = Long_val(vspp);
  int bytes = Int_val(vbytes), predictor = Int_val(vpredictor);
  size_t len = n * (size_t)bytes;
  if (predictor == NONE) return Val_unit;
  for (size_t r = 0; r < rows; r++) {
    uint8_t *row = buf + r * len;
    if (HOST_BIG && bytes > 1) swap(row, n, bytes);
    if (predictor == FLOATING_POINT)
      floating_point(row, scratch, n, spp, bytes, 0);
    else {
      horizontal(row, n, spp, bytes, 0);
      if (HOST_BIG && bytes > 1) swap(row, n, bytes);
    }
  }
  return Val_unit;
}

CAMLprim value caml_nx_io_tiff_predict_byte(value *argv, int argn) {
  (void)argn;
  return caml_nx_io_tiff_predict(argv[0], argv[1], argv[2], argv[3], argv[4],
                                 argv[5], argv[6]);
}

/* [caml_nx_io_tiff_copy src src_at src_stride dst dst_at dst_stride rows
   span] copies [rows] rows of [span] bytes from [src] to [dst], row [i]
   from [src_at + i * src_stride] to [dst_at + i * dst_stride]. The caller
   checks that every row lies in its buffer. */
CAMLprim value caml_nx_io_tiff_copy(value vsrc, value vsrc_at,
                                    value vsrc_stride, value vdst,
                                    value vdst_at, value vdst_stride,
                                    value vrows, value vspan) {
  const uint8_t *src = Caml_ba_data_val(vsrc);
  uint8_t *dst = Caml_ba_data_val(vdst);
  size_t src_at = Long_val(vsrc_at), dst_at = Long_val(vdst_at);
  size_t src_stride = Long_val(vsrc_stride), dst_stride = Long_val(vdst_stride);
  size_t rows = Long_val(vrows), span = Long_val(vspan);
  for (size_t i = 0; i < rows; i++)
    memcpy(dst + dst_at + i * dst_stride, src + src_at + i * src_stride, span);
  return Val_unit;
}

CAMLprim value caml_nx_io_tiff_copy_byte(value *argv, int argn) {
  (void)argn;
  return caml_nx_io_tiff_copy(argv[0], argv[1], argv[2], argv[3], argv[4],
                              argv[5], argv[6], argv[7]);
}

/* [caml_nx_io_tiff_unpack src src_at src_stride first dst dst_at
   dst_stride rows n bits bytes] reads [rows] rows of [n] samples of [bits]
   bits, 1 to 32, packed most significant bit first, row [i] from sample
   [first] of the row at [src_at + i * src_stride], into samples of [bytes]
   bytes, 1, 2 or 4, in the host's byte order from [dst_at + i *
   dst_stride]. A sample spans at most 5 bytes. The caller checks that
   every row lies in its buffer. */
CAMLprim value caml_nx_io_tiff_unpack(value vsrc, value vsrc_at,
                                      value vsrc_stride, value vfirst,
                                      value vdst, value vdst_at,
                                      value vdst_stride, value vrows,
                                      value vn, value vbits, value vbytes) {
  const uint8_t *src = Caml_ba_data_val(vsrc);
  uint8_t *dst = Caml_ba_data_val(vdst);
  size_t src_at = Long_val(vsrc_at), dst_at = Long_val(vdst_at);
  size_t src_stride = Long_val(vsrc_stride), dst_stride = Long_val(vdst_stride);
  size_t first = Long_val(vfirst), rows = Long_val(vrows), n = Long_val(vn);
  int bits = Int_val(vbits), bytes = Int_val(vbytes);
  uint64_t mask = (UINT64_C(1) << bits) - 1;
  for (size_t i = 0; i < rows; i++) {
    const uint8_t *row = src + src_at + i * src_stride;
    uint8_t *out = dst + dst_at + i * dst_stride;
    size_t bit = first * (size_t)bits;
    for (size_t j = 0; j < n; j++, bit += (size_t)bits, out += bytes) {
      const uint8_t *p = row + (bit >> 3);
      int shift = (int)(bit & 7), span = (shift + bits + 7) >> 3;
      uint64_t acc = 0;
      for (int k = 0; k < span; k++) acc = (acc << 8) | p[k];
      uint32_t v = (uint32_t)((acc >> (span * 8 - shift - bits)) & mask);
      if (bytes == 1) {
        out[0] = (uint8_t)v;
      } else if (bytes == 2) {
        uint16_t h = (uint16_t)v;
        memcpy(out, &h, 2);
      } else {
        memcpy(out, &v, 4);
      }
    }
  }
  return Val_unit;
}

CAMLprim value caml_nx_io_tiff_unpack_byte(value *argv, int argn) {
  (void)argn;
  return caml_nx_io_tiff_unpack(argv[0], argv[1], argv[2], argv[3], argv[4],
                                argv[5], argv[6], argv[7], argv[8], argv[9],
                                argv[10]);
}
