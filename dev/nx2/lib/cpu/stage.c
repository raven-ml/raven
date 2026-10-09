/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The stage: moving a block of an operand into a buffer in its carrier, and
   a buffer back into an operand, converted.

   The target's runs read and write contiguous elements of byte-wide
   dtypes. The stage brings every operand to that form: it gathers a
   strided or transposed block into contiguous rows, unpacks sub-byte
   elements one per byte, and decodes narrow floats into float32. The
   unstage goes the other way through the target's conversion from the
   buffer's dtype, then scatters into a strided destination or packs into a
   sub-byte one, whose end bytes it writes with a compare-and-swap. Both
   keep the work of a block in L1: a block holds at most NX_CPU_SLOT
   bytes in any of its forms. */

#include "cpu.h"

/* Sub-byte rows: row j of operand [k] of [b], one element per byte, from
   or to [buf + j·row]. A strided row loads or stores element by element.
   Each run is called with its dtype's bits as constants. */

static void unpack_run(const uint8_t *base, int dt, int64_t p, uint8_t *d,
                       int64_t n) {
  switch (dt) {
    case NX_INT4: nx_sub_unpack_run(base, NX_INT4, p, d, n); return;
    case NX_BIT: nx_sub_unpack_run(base, NX_BIT, p, d, n); return;
    default: nx_sub_unpack_run(base, NX_UINT4, p, d, n); return;
  }
}

static void unpack(const nx_array *a, const nx_cpu_block *b, int k,
                   uint8_t *buf, int64_t row) {
  uint32_t half = a->dtype == NX_INT4 ? 1u << (a->bits - 1) : 0;
  for (int64_t j = 0; j < b->n1; j++) {
    int64_t p = b->at[k] + j * b->s1[k];
    uint8_t *d = buf + j * row;
    if (b->s0[k] == 1) unpack_run(a->base, a->dtype, p, d, b->n0);
    else
      for (int64_t i = 0; i < b->n0; i++, p += b->s0[k])
        d[i] = (uint8_t)((nx_sub_load(a->base, a->bits, p) ^ half) - half);
  }
}

static void pack(const nx_array *a, const nx_cpu_block *b, int k,
                 const uint8_t *buf, int64_t row) {
  for (int64_t j = 0; j < b->n1; j++) {
    int64_t p = b->at[k] + j * b->s1[k];
    const uint8_t *s = buf + j * row;
    if (b->s0[k] != 1)
      for (int64_t i = 0; i < b->n0; i++, p += b->s0[k])
        nx_sub_store(a->base, a->bits, p, s[i]);
    else if (a->bits == 4)
      nx_sub_pack_run(a->base, 4, p, s, b->n0);
    else
      nx_sub_pack_run(a->base, 1, p, s, b->n0);
  }
}

void nx_cpu_stage(const nx_array *a, const nx_cpu_block *b, int k, uint8_t *dst,
                  int64_t row) {
  int dt = a->dtype;
  int64_t n0 = b->n0, n1 = b->n1;
  nx_cpu_run decode = nx_cpu_table->convert[dt][NX_FLOAT32];
  _Alignas(64) uint8_t raw[NX_CPU_SLOT];
  if (a->bits < 8) {
    if (dt != NX_FLOAT4_E2M1FN) {
      unpack(a, b, k, dst, row);
      return;
    }
    unpack(a, b, k, raw, n0);
    for (int64_t j = 0; j < n1; j++) decode(raw + j * n0, dst + j * row, n0);
    return;
  }
  int w = a->bits / 8;
  if (dt == nx_cpu_carrier(dt)) {
    nx_copy_box(dst, a->base,
                &(nx_box){{1, n1, n0},
                          {0, b->at[k]},
                          {{0, row / w, 1}, {0, b->s1[k], b->s0[k]}}},
                a->bits);
    return;
  }
  /* A narrow float: decoded from its memory where its rows are runs, else
     from its block gathered into [raw]. */
  const uint8_t *s = a->base + b->at[k] * w;
  int64_t so = b->s1[k] * w;
  if (b->s0[k] != 1) {
    nx_copy_box(raw, a->base,
                &(nx_box){{1, n1, n0},
                          {0, b->at[k]},
                          {{0, n0, 1}, {0, b->s1[k], b->s0[k]}}},
                a->bits);
    s = raw;
    so = n0 * w;
  }
  for (int64_t j = 0; j < n1; j++) decode(s + j * so, dst + j * row, n0);
}

void nx_cpu_unstage(const nx_array *a, const nx_cpu_block *b, int k,
                    const uint8_t *src, int64_t row, int c) {
  nx_cpu_run convert = nx_cpu_table->convert[c][a->dtype];
  int64_t n0 = b->n0, n1 = b->n1, cw = nx_cpu_width(c);
  int w = nx_cpu_width(a->dtype);
  _Alignas(64) uint8_t raw[NX_CPU_SLOT];
  if (a->bits < 8) {
    /* int4 and uint4 keep a byte integer's low bits: it packs as it is. */
    if ((c == NX_INT8 || c == NX_UINT8) &&
        (a->dtype == NX_INT4 || a->dtype == NX_UINT4)) {
      pack(a, b, k, src, row);
      return;
    }
    for (int64_t j = 0; j < n1; j++) convert(src + j * row, raw + j * n0, n0);
    pack(a, b, k, raw, n0);
    return;
  }
  uint8_t *d = a->base + b->at[k] * w;
  int64_t di = b->s0[k] * w, dout = b->s1[k] * w;
  if (di == w) {
    if (row == n0 * cw && dout == n0 * w)
      convert(src, d, n0 * n1);
    else
      for (int64_t j = 0; j < n1; j++) convert(src + j * row, d + j * dout, n0);
    return;
  }
  /* A strided destination: converted into [raw], then scattered. */
  for (int64_t j = 0; j < n1; j++) convert(src + j * row, raw + j * n0 * w, n0);
  nx_copy_box(a->base, raw,
              &(nx_box){{1, n1, n0},
                        {b->at[k], 0},
                        {{0, b->s1[k], b->s0[k]}, {0, n0, 1}}},
              a->bits);
}

void nx_cpu_stage_as(const nx_array *a, int64_t at, int64_t s0, int64_t s1,
                     int64_t n0, int64_t n1, int d, uint8_t *dst,
                     int64_t pitch) {
  int w = nx_cpu_width(d);
  nx_cpu_block b = {.n0 = n0, .n1 = n1, .n2 = 1};
  b.s0[0] = s0;
  b.s1[0] = s1;
  if (a->dtype == d) {
    b.at[0] = at;
    nx_cpu_stage(a, &b, 0, dst, pitch * w);
    return;
  }
  int c = nx_cpu_carrier(a->dtype), cw = nx_cpu_width(c);
  int widest = cw > w ? cw : w;
  if (a->bits / 8 > widest) widest = a->bits / 8;
  int64_t cols = n0 < NX_CPU_SLOT / widest ? n0 : NX_CPU_SLOT / widest;
  nx_cpu_run convert = nx_cpu_table->convert[c][d];
  _Alignas(64) uint8_t slot[NX_CPU_SLOT];
  for (int64_t i = 0; i < n0; i += cols) {
    int64_t n = cols < n0 - i ? cols : n0 - i;
    int64_t rows = NX_CPU_SLOT / (n * widest);
    for (int64_t j = 0; j < n1; j += rows) {
      b.n0 = n;
      b.n1 = rows < n1 - j ? rows : n1 - j;
      b.at[0] = at + i * s0 + j * s1;
      uint8_t *o = dst + (j * pitch + i) * w;
      if (c == d) {
        nx_cpu_stage(a, &b, 0, o, pitch * w);
        continue;
      }
      nx_cpu_stage(a, &b, 0, slot, n * cw);
      for (int64_t r = 0; r < b.n1; r++)
        convert(slot + r * n * cw, o + r * pitch * w, n);
    }
  }
}
