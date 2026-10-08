/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <string.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
#include <caml/memory.h>
#include <caml/minor_gc.h>
#include <caml/mlvalues.h>

#include "nx_array.h"
#include "nx_layout.h"

/* The row of the dtype [code] in nx_dtype.h, as (name, bits, kind), or
   None past the last code. */
value nx_array_support_row(value code) {
  CAMLparam1(code);
  CAMLlocal2(row, name);
  intnat c = Long_val(code);
  if (c < 0 || c >= NX_DTYPE_COUNT) CAMLreturn(Val_none);
  nx_dtype_row r = nx_dtype_row_of((int)c);
  name = caml_copy_string(r.name);
  row = caml_alloc_tuple(3);
  Store_field(row, 0, name);
  Store_field(row, 1, Val_int(r.bits));
  Store_field(row, 2, Val_int(r.kind));
  CAMLreturn(caml_alloc_some(row));
}

/* The fields of the layout [l] as C reads them (nx_layout.h): rank, flags,
   offset, lo, hi, then the extents and strides. */
value nx_array_support_layout(value l) {
  CAMLparam1(l);
  CAMLlocal1(out);
  int64_t r = (int64_t)Wosize_val(Field(l, NX_LAYOUT_SHAPE));
  out = caml_alloc_tuple(5 + 2 * r);
  Store_field(out, 0, Val_long(r));
  Store_field(out, 1, Field(l, NX_LAYOUT_FLAGS));
  Store_field(out, 2, Field(l, NX_LAYOUT_OFFSET));
  Store_field(out, 3, Field(l, NX_LAYOUT_LO));
  Store_field(out, 4, Field(l, NX_LAYOUT_HI));
  for (int64_t i = 0; i < r; i++) {
    Store_field(out, 5 + i, Field(Field(l, NX_LAYOUT_SHAPE), i));
    Store_field(out, 5 + r + i, Field(Field(l, NX_LAYOUT_STRIDES), i));
  }
  CAMLreturn(out);
}

/* The binary32 bits of the code [c] of the narrow float dtype [dt], as the
   decoder kernels call reads it. */
value nx_array_support_decode(value dt, value c) {
  float f = nx_bits_to_float((int)Long_val(dt), (uint32_t)Long_val(c));
  uint32_t i;
  memcpy(&i, &f, 4);
  return Val_long(i);
}

/* The bits a store of the int64 [v], or of the uint64 whose bits [v] holds,
   writes into an element of the narrow float [dt], rounded once. */
value nx_array_support_of_i64(value dt, value v) {
  return Val_long(
      nx_double_to_bits((int)Long_val(dt), nx_i64_odd(Int64_val(v))));
}

value nx_array_support_of_u64(value dt, value v) {
  return Val_long(
      nx_double_to_bits((int)Long_val(dt), nx_u64_odd((uint64_t)Int64_val(v))));
}

/* A kernel: z <- x + y over float32, through the door and the coalescer. */

static void add_loop(const nx_array *a, const nx_loop *l) {
  int r = l->rank;
  int64_t len = l->extent[r - 1];
  if (len == 0) return;
  int64_t idx[NX_MAX_RANK] = {0}, at[3];
  for (int k = 0; k < 3; k++) at[k] = l->first[k];
  float *z = (float *)a[0].base;
  const float *x = (const float *)a[1].base, *y = (const float *)a[2].base;
  int64_t sz = l->step[0][r - 1], sx = l->step[1][r - 1], sy = l->step[2][r - 1];
  for (;;) {
    for (int64_t j = 0; j < len; j++)
      z[at[0] + j * sz] = x[at[1] + j * sx] + y[at[2] + j * sy];
    int i = r - 2;
    for (; i >= 0; i--) {
      for (int k = 0; k < 3; k++) at[k] += l->step[k][i];
      if (++idx[i] < l->extent[i]) break;
      for (int k = 0; k < 3; k++) at[k] -= l->step[k][i] * l->extent[i];
      idx[i] = 0;
    }
    if (i < 0) return;
  }
}

value nx_array_support_add(value z, value x, value y) {
  nx_operand in[3] = {{z, NX_FLOAT32, 1}, {x, NX_FLOAT32, 0}, {y, NX_FLOAT32, 0}};
  nx_array a[3];
  nx_loop l;
  int e = nx_read(3, in, a);
  if (e) return Val_int(e);
  if (!(e = nx_coalesce(3, a, &l))) add_loop(a, &l);
  nx_done(3, a);
  return Val_int(e);
}

/* Reads [v] through the door, then empties the minor heap and compacts the
   major one while it holds the read, moving [v] and its buffer. */
extern value caml_gc_compaction(value);

value nx_array_support_collect(value v) {
  CAMLparam1(v);
  nx_operand in = {v, nx_array_dtype(v), 0};
  nx_array a;
  int e = nx_read(1, &in, &a);
  if (e) CAMLreturn(Val_int(e));
  in.array = Val_unit;
  v = Val_unit;
  caml_minor_collection();
  caml_gc_compaction(Val_unit);
  nx_done(1, &a);
  CAMLreturn(Val_int(NX_OK));
}

/* A bigarray of [n] int16, at most 120, over static memory [at] bytes past a
   16-byte boundary: misaligned for its elements if [at] is odd. */
value nx_array_support_int16_at(value at, value n) {
  static _Alignas(16) uint8_t bytes[256];
  intnat dim = Long_val(n);
  return caml_ba_alloc(CAML_BA_SINT16 | CAML_BA_C_LAYOUT | CAML_BA_EXTERNAL, 1,
                       bytes + Long_val(at), &dim);
}

/* Copies [len] bytes between host addresses and a bigarray, for an io
   device over bigarrays. */
value nx_array_support_blit_in(value ba, value at, value src, value len) {
  memcpy((uint8_t *)Caml_ba_data_val(ba) + Long_val(at),
         (const void *)Long_val(src), Long_val(len));
  return Val_unit;
}

value nx_array_support_blit_out(value ba, value at, value dst, value len) {
  memcpy((void *)Long_val(dst),
         (const uint8_t *)Caml_ba_data_val(ba) + Long_val(at), Long_val(len));
  return Val_unit;
}
