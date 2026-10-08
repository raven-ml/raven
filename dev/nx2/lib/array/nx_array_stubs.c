/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <string.h>

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_array.h"
#include "nx_layout.h"
#include "rig.h"

#ifndef FLAT_FLOAT_ARRAY
#error "nx_array writes float arrays as flat arrays of doubles"
#endif

/* An array is the record { dtype; layout; buffer }, in this order: the
   dtype an immediate whose value is its code, the layout a record
   (nx_layout.h). This file is the only C that reads it. */
enum { ARRAY_DTYPE, ARRAY_LAYOUT, ARRAY_BUFFER };

/* Layouts */

/* Copies the layout [v] into [a]'s rank, flags, offset and dims, and its
   span into [lo] and [hi]: the record lives in the OCaml heap, which moves.
   Answers NX_LAYOUT if its arrays do not make a layout. */
static int read_layout(value v, nx_array *a, int64_t *lo, int64_t *hi) {
  value shape = Field(v, NX_LAYOUT_SHAPE), strides = Field(v, NX_LAYOUT_STRIDES);
  mlsize_t r = Wosize_val(shape);
  if (r > NX_MAX_RANK || Wosize_val(strides) != r) return NX_LAYOUT;
  a->rank = (int)r;
  a->offset = Long_val(Field(v, NX_LAYOUT_OFFSET));
  a->flags = (int)Long_val(Field(v, NX_LAYOUT_FLAGS));
  *lo = Long_val(Field(v, NX_LAYOUT_LO));
  *hi = Long_val(Field(v, NX_LAYOUT_HI));
  for (mlsize_t i = 0; i < r; i++) {
    a->dim[i] = Long_val(Field(shape, i));
    a->dim[r + i] = Long_val(Field(strides, i));
  }
  return NX_OK;
}

/* The door */

int nx_array_dtype(value v) { return (int)Long_val(Field(v, ARRAY_DTYPE)); }

static int claim_code(enum rig_claim c) {
  switch (c) {
    case RIG_CLAIMED: return NX_OK;
    case RIG_PENDING: return NX_PENDING;
    case RIG_DEAD: return NX_DEAD;
    case RIG_EXCLUSIVE: return NX_EXCLUSIVE;
    default: return NX_READ_ONLY;
  }
}

int nx_read(int n, const nx_operand *in, nx_array *out) {
  if (n <= 0) return NX_OK;
  int64_t first[n], last[n]; /* each operand's bytes, as host addresses */
  for (int k = 0; k < n; k++) {
    value v = in[k].array;
    nx_array *a = &out[k];
    int64_t lo, hi;
    a->dtype = nx_array_dtype(v);
    if (a->dtype != in[k].dtype) return NX_DTYPE;
    a->bits = nx_dtype_row_of(a->dtype).bits;
    int e = read_layout(Field(v, ARRAY_LAYOUT), a, &lo, &hi);
    if (e) return e;
    a->buffer = Field(v, ARRAY_BUFFER);
    if (rig_buffer_why(a->buffer) != NULL) return NX_DEAD;
    if (a->flags & NX_EMPTY) {
      a->base = NULL;
      first[k] = last[k] = 0;
    } else {
      a->base = rig_buffer_host(a->buffer);
      if (a->base == NULL) return NX_NOT_HOST;
      first[k] = (int64_t)(intptr_t)a->base + lo * a->bits / 8;
      last[k] = (int64_t)(intptr_t)a->base + (hi * a->bits + 7) / 8;
    }
    if (in[k].written && !(a->flags & NX_DISTINCT)) return NX_NOT_DISTINCT;
  }
  for (int k = 0; k < n; k++) {
    if (!in[k].written || first[k] == last[k]) continue;
    for (int j = 0; j < n; j++)
      if (j != k && first[j] < last[k] && first[k] < last[j])
        return NX_OVERLAP;
  }
  for (int k = 0; k < n; k++) {
    enum rig_claim c = rig_buffer_claim(
        out[k].buffer, in[k].written ? RIG_READ_WRITE : RIG_READ);
    if (c != RIG_CLAIMED) {
      while (k-- > 0) rig_buffer_release(out[k].buffer);
      return claim_code(c);
    }
  }
  /* Each buffer becomes a local root, so that a kernel that allocates or
     releases the domain lock keeps it reachable and finds it again in
     nx_done. They are chained here and linked at once: where the runtime
     has no thread-local variables to share (macOS), each access to the
     domain's state is a function call. */
  struct caml__roots_block *top = CAML_LOCAL_ROOTS;
  for (int k = 0; k < n; k++) {
    out[k].roots.next = top;
    out[k].roots.ntables = 1;
    out[k].roots.nitems = 1;
    out[k].roots.tables[0] = &out[k].buffer;
    top = &out[k].roots;
  }
  CAML_LOCAL_ROOTS = top;
  return NX_OK;
}

void nx_done(int n, nx_array *a) {
  if (n <= 0) return;
  for (int k = 0; k < n; k++) rig_buffer_release(a[k].buffer);
  /* Unlink the roots nx_read pushed, under any pushed since. */
  struct caml__roots_block **p = &CAML_LOCAL_ROOTS;
  while (*p != &a[n - 1].roots) {
    if (*p == NULL)
      caml_fatal_error("nx_done: the descriptors' roots were popped before "
                       "nx_done");
    p = &(*p)->next;
  }
  *p = a[0].roots.next;
}

/* Coalescing */

int nx_coalesce(int n, const nx_array *a, nx_loop *l) {
  if (n < 1 || n > NX_MAX_OPERANDS) return NX_ARITY;
  int r = a[0].rank;
  for (int k = 1; k < n; k++) {
    if (a[k].rank != r) return NX_SHAPE;
    for (int i = 0; i < r; i++)
      if (a[k].dim[i] != a[0].dim[i]) return NX_SHAPE;
  }
  for (int k = 0; k < n; k++) l->first[k] = a[k].offset;
  int out = 0;
  for (int i = 0; i < r; i++) {
    int64_t d = a[0].dim[i];
    if (d == 0) {
      l->rank = 1;
      l->extent[0] = 0;
      for (int k = 0; k < n; k++) l->step[k][0] = 0;
      return NX_OK;
    }
    if (d == 1) continue;
    /* The axis joins the previous one if every operand's previous stride
       is this stride times this extent: the two axes are one run. */
    int joins = out > 0;
    for (int k = 0; k < n && joins; k++)
      joins = l->step[k][out - 1] == a[k].dim[a[k].rank + i] * d;
    if (joins) {
      l->extent[out - 1] *= d;
      for (int k = 0; k < n; k++) l->step[k][out - 1] = a[k].dim[r + i];
    } else {
      l->extent[out] = d;
      for (int k = 0; k < n; k++) l->step[k][out] = a[k].dim[r + i];
      out++;
    }
  }
  if (out == 0) {
    l->extent[0] = 1;
    for (int k = 0; k < n; k++) l->step[k][0] = 0;
    out = 1;
  }
  l->rank = out;
  return NX_OK;
}

/* Calls [run] on each run of the loop [l] over [n] operands: the positions
   [at] of each operand's first element, and the run's length; the run's
   steps are the loop's innermost. */
static void walk(int n, const nx_loop *l, void *ctx,
                 void (*run)(void *ctx, const int64_t *at, int64_t len)) {
  int r = l->rank;
  int64_t len = l->extent[r - 1];
  if (len == 0) return;
  int64_t idx[NX_MAX_RANK] = {0};
  int64_t at[NX_MAX_OPERANDS];
  for (int k = 0; k < n; k++) at[k] = l->first[k];
  for (;;) {
    run(ctx, at, len);
    /* The odometer over the outer axes. */
    int i = r - 2;
    for (; i >= 0; i--) {
      for (int k = 0; k < n; k++) at[k] += l->step[k][i];
      if (++idx[i] < l->extent[i]) break;
      for (int k = 0; k < n; k++) at[k] -= l->step[k][i] * l->extent[i];
      idx[i] = 0;
    }
    if (i < 0) return;
  }
}

/* Layout.coalesce: coalesces the layouts [ls] into [out], an int array of
   1 + rank + n·(1 + rank) words: the rank, the extents, then per operand
   its first position and steps. Answers a code. */
value nx_array_coalesce(value ls, value out) {
  int n = (int)Wosize_val(ls);
  nx_array a[NX_MAX_OPERANDS];
  nx_loop l;
  int64_t lo, hi;
  if (n < 1 || n > NX_MAX_OPERANDS) return Val_int(NX_ARITY);
  for (int k = 0; k < n; k++) {
    int e = read_layout(Field(ls, k), &a[k], &lo, &hi);
    if (e) return Val_int(e);
  }
  int e = nx_coalesce(n, a, &l);
  if (e) return Val_int(e);
  int r = l.rank;
  Field(out, 0) = Val_long(r);
  for (int i = 0; i < r; i++) Field(out, 1 + i) = Val_long(l.extent[i]);
  for (int k = 0; k < n; k++) {
    mlsize_t at = 1 + r + k * (1 + r);
    Field(out, at) = Val_long(l.first[k]);
    for (int i = 0; i < r; i++)
      Field(out, at + 1 + i) = Val_long(l.step[k][i]);
  }
  return Val_int(NX_OK);
}

/* Elements

   Loads and stores of one element at position [p] from [base]. A float
   dtype's element is a double; an integer's, a boolean's or a narrow
   float's bits are an int64_t, sign-extended for signed integers. A complex
   element is two components, [part] 0 the real one. */

static double load_float(const uint8_t *base, int dt, int64_t p, int part) {
  switch (dt) {
    case NX_FLOAT64: {
      double x;
      memcpy(&x, base + 8 * p, 8);
      return x;
    }
    case NX_FLOAT32: {
      float x;
      memcpy(&x, base + 4 * p, 4);
      return x;
    }
    case NX_FLOAT16:
    case NX_BFLOAT16: {
      uint16_t c;
      memcpy(&c, base + 2 * p, 2);
      return nx_bits_to_float(dt, c);
    }
    case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2: return nx_bits_to_float(dt, base[p]);
    case NX_FLOAT4_E2M1FN: return nx_bits_to_float(dt, nx_sub_load(base, 4, p));
    case NX_COMPLEX128: return load_float(base, NX_FLOAT64, 2 * p + part, 0);
    default: return load_float(base, NX_FLOAT32, 2 * p + part, 0);
  }
}

static int64_t load_int(const uint8_t *base, int dt, int64_t p) {
  switch (dt) {
    case NX_INT64:
    case NX_UINT64: {
      int64_t x;
      memcpy(&x, base + 8 * p, 8);
      return x;
    }
    case NX_INT32: {
      int32_t x;
      memcpy(&x, base + 4 * p, 4);
      return x;
    }
    case NX_UINT32: {
      uint32_t x;
      memcpy(&x, base + 4 * p, 4);
      return x;
    }
    case NX_INT16: {
      int16_t x;
      memcpy(&x, base + 2 * p, 2);
      return x;
    }
    case NX_UINT16: {
      uint16_t x;
      memcpy(&x, base + 2 * p, 2);
      return x;
    }
    case NX_INT8: return (int8_t)base[p];
    case NX_UINT8: return base[p];
    case NX_INT4: return ((int32_t)(nx_sub_load(base, 4, p) << 28)) >> 28;
    case NX_UINT4: return nx_sub_load(base, 4, p);
    case NX_BOOL: return base[p] != 0;
    default: return nx_sub_load(base, 1, p); /* bit */
  }
}

static void store_float(uint8_t *base, int dt, int64_t p, int part,
                        double x) {
  switch (dt) {
    case NX_FLOAT64: memcpy(base + 8 * p, &x, 8); return;
    case NX_FLOAT32: {
      float f = (float)x;
      memcpy(base + 4 * p, &f, 4);
      return;
    }
    case NX_FLOAT16:
    case NX_BFLOAT16: {
      uint16_t c = (uint16_t)nx_double_to_bits(dt, x);
      memcpy(base + 2 * p, &c, 2);
      return;
    }
    case NX_FLOAT8_E4M3FN:
    case NX_FLOAT8_E5M2: base[p] = (uint8_t)nx_double_to_bits(dt, x); return;
    case NX_FLOAT4_E2M1FN:
      nx_sub_store(base, 4, p, (uint32_t)nx_double_to_bits(dt, x));
      return;
    case NX_COMPLEX128: store_float(base, NX_FLOAT64, 2 * p + part, 0, x); return;
    default: store_float(base, NX_FLOAT32, 2 * p + part, 0, x); return;
  }
}

static void store_int(uint8_t *base, int dt, int64_t p, int64_t x) {
  switch (dt) {
    case NX_INT64:
    case NX_UINT64: memcpy(base + 8 * p, &x, 8); return;
    case NX_INT32:
    case NX_UINT32: {
      uint32_t c = (uint32_t)x;
      memcpy(base + 4 * p, &c, 4);
      return;
    }
    case NX_INT16:
    case NX_UINT16: {
      uint16_t c = (uint16_t)x;
      memcpy(base + 2 * p, &c, 2);
      return;
    }
    case NX_INT8:
    case NX_UINT8:
    case NX_BOOL: base[p] = (uint8_t)x; return;
    case NX_INT4:
    case NX_UINT4: nx_sub_store(base, 4, p, (uint32_t)x); return;
    default: nx_sub_store(base, 1, p, (uint32_t)x); return; /* bit */
  }
}

/* Element access from OCaml, on a buffer the caller claimed and waited
   for, at a position its layout reaches. */

intnat nx_array_host(value b) {
  void *p = rig_buffer_host(b);
  return p == NULL ? -1 : (intnat)(intptr_t)p;
}

double nx_array_get_float(value b, intnat dt, intnat p, intnat part) {
  return load_float(rig_buffer_host(b), (int)dt, p, (int)part);
}

int64_t nx_array_get_int(value b, intnat dt, intnat p) {
  return load_int(rig_buffer_host(b), (int)dt, p);
}

value nx_array_set_float(value b, intnat dt, intnat p, intnat part,
                         double x) {
  store_float(rig_buffer_host(b), (int)dt, p, (int)part, x);
  return Val_unit;
}

value nx_array_set_int(value b, intnat dt, intnat p, int64_t x) {
  store_int(rig_buffer_host(b), (int)dt, p, x);
  return Val_unit;
}

value nx_array_host_byte(value b) { return Val_long(nx_array_host(b)); }

value nx_array_get_float_byte(value b, value dt, value p, value part) {
  return caml_copy_double(
      nx_array_get_float(b, Long_val(dt), Long_val(p), Long_val(part)));
}

value nx_array_get_int_byte(value b, value dt, value p) {
  return caml_copy_int64(nx_array_get_int(b, Long_val(dt), Long_val(p)));
}

value nx_array_set_float_byte(value b, value dt, value p, value part,
                              value x) {
  return nx_array_set_float(b, Long_val(dt), Long_val(p), Long_val(part),
                            Double_val(x));
}

value nx_array_set_int_byte(value b, value dt, value p, value x) {
  return nx_array_set_int(b, Long_val(dt), Long_val(p), Int64_val(x));
}

/* Bulk access */

enum { TO_FLAT, TO_IMMEDIATE, TO_BOXED };

/* How a dtype's values lie in an OCaml array: flat doubles, immediates, or
   boxes. */
static int representation(int dt) {
  switch (nx_dtype_row_of(dt).kind) {
    case NX_KIND_FLOAT: return TO_FLAT;
    case NX_KIND_COMPLEX: return TO_BOXED;
    default:
      return nx_dtype_row_of(dt).bits >= 32 ? TO_BOXED : TO_IMMEDIATE;
  }
}

typedef struct {
  const nx_array *a;
  const nx_loop *l;
  value *out; /* a root: allocating boxes moves the array */
  int64_t k;  /* the next element of [out] */
} to_ctx;

static void to_run(void *ctx, const int64_t *at, int64_t len) {
  to_ctx *c = ctx;
  const nx_array *a = c->a;
  int dt = a->dtype;
  int64_t step = c->l->step[0][c->l->rank - 1];
  int64_t p = at[0];
  switch (representation(dt)) {
    case TO_FLAT: {
      double *dst = (double *)Op_val(*c->out) + c->k;
      c->k += len;
      if (dt == NX_FLOAT32) {
        const float *src = (const float *)a->base + p;
        for (int64_t j = 0; j < len; j++) dst[j] = src[j * step];
      } else if (dt == NX_FLOAT64) {
        const double *src = (const double *)a->base + p;
        for (int64_t j = 0; j < len; j++) dst[j] = src[j * step];
      } else if (dt == NX_FLOAT16 && step == 1) {
        nx_f16_to_double_run((const uint16_t *)a->base + p, dst, (size_t)len);
      } else {
        for (int64_t j = 0; j < len; j++, p += step)
          dst[j] = load_float(a->base, dt, p, 0);
      }
      return;
    }
    case TO_IMMEDIATE:
      for (int64_t j = 0; j < len; j++, p += step)
        Field(*c->out, c->k++) = Val_long(load_int(a->base, dt, p));
      return;
    default:
      for (int64_t j = 0; j < len; j++, p += step) {
        value box;
        if (dt == NX_INT32 || dt == NX_UINT32)
          box = caml_copy_int32((int32_t)load_int(a->base, dt, p));
        else if (dt == NX_INT64 || dt == NX_UINT64)
          box = caml_copy_int64(load_int(a->base, dt, p));
        else {
          box = caml_alloc_small(2 * Double_wosize, Double_array_tag);
          Store_double_flat_field(box, 0, load_float(a->base, dt, p, 0));
          Store_double_flat_field(box, 1, load_float(a->base, dt, p, 1));
        }
        caml_modify(&Field(*c->out, c->k++), box);
      }
      return;
  }
}

/* The loop of one operand, its strides as they come: [to_array] and
   [of_array] walk in C order of indices, so axes are not reordered. */
static int loop1(const nx_array *a, nx_loop *l) { return nx_coalesce(1, a, l); }

/* to_array: reads [v]'s elements, in C order of indices, into [out], an
   OCaml array of their number in the dtype's representation. */
value nx_array_to_array(value v, value out) {
  CAMLparam2(v, out);
  nx_operand in = {v, nx_array_dtype(v), 0};
  nx_array a;
  nx_loop l;
  int e = nx_read(1, &in, &a);
  if (e) CAMLreturn(Val_int(e));
  if (!(e = loop1(&a, &l))) {
    to_ctx c = {&a, &l, &out, 0};
    walk(1, &l, &c, to_run);
  }
  nx_done(1, &a);
  CAMLreturn(Val_int(e));
}

typedef struct {
  const nx_array *a;
  const nx_loop *l;
  value values;
  int64_t k;
} of_ctx;

static void of_run(void *ctx, const int64_t *at, int64_t len) {
  of_ctx *c = ctx;
  const nx_array *a = c->a;
  int dt = a->dtype;
  int64_t step = c->l->step[0][c->l->rank - 1];
  int64_t p = at[0], k = c->k;
  c->k += len;
  switch (representation(dt)) {
    case TO_FLAT: {
      const double *src = (const double *)Op_val(c->values) + k;
      if (dt == NX_FLOAT32) {
        float *dst = (float *)a->base + p;
        for (int64_t j = 0; j < len; j++) dst[j * step] = (float)src[j];
      } else if (dt == NX_FLOAT64) {
        double *dst = (double *)a->base + p;
        for (int64_t j = 0; j < len; j++) dst[j * step] = src[j];
      } else if (dt == NX_FLOAT16 && step == 1) {
        nx_double_to_f16_run(src, (uint16_t *)a->base + p, (size_t)len);
      } else {
        for (int64_t j = 0; j < len; j++, p += step)
          store_float(a->base, dt, p, 0, src[j]);
      }
      return;
    }
    case TO_IMMEDIATE:
      for (int64_t j = 0; j < len; j++, p += step)
        store_int(a->base, dt, p, Long_val(Field(c->values, k + j)));
      return;
    default:
      for (int64_t j = 0; j < len; j++, p += step) {
        value x = Field(c->values, k + j);
        if (dt == NX_INT32 || dt == NX_UINT32)
          store_int(a->base, dt, p, Int32_val(x));
        else if (dt == NX_INT64 || dt == NX_UINT64)
          store_int(a->base, dt, p, Int64_val(x));
        else {
          store_float(a->base, dt, p, 0, Double_flat_field(x, 0));
          store_float(a->base, dt, p, 1, Double_flat_field(x, 1));
        }
      }
      return;
  }
}

/* of_array: writes [values], an OCaml array of [v]'s number of elements, in
   C order of indices. It allocates nothing. */
value nx_array_of_array(value v, value values) {
  nx_operand in = {v, nx_array_dtype(v), 1};
  nx_array a;
  nx_loop l;
  int e = nx_read(1, &in, &a);
  if (e) return Val_int(e);
  if (!(e = loop1(&a, &l))) {
    of_ctx c = {&a, &l, values, 0};
    walk(1, &l, &c, of_run);
  }
  nx_done(1, &a);
  return Val_int(e);
}

/* copy: gathers [src]'s elements into [dst], a fresh contiguous array of
   its dtype and shape, bits for bits.

   Each run of the loop is a row of [dst]. A source that steps through the
   row with a stride, a transposed one say, reads one cache line per
   element; the next rows read the same lines. So when an outer axis steps
   less in [src] than the row does, the copy goes in square tiles over that
   axis and the row, TILE bytes of elements a side: each tile reads its
   source lines once. Where the axis steps by one element in [src], a tile
   moves 4x4 blocks, each four contiguous loads and four contiguous stores.
   An element of under a byte keeps to rows. */

#define TILE 256

typedef struct {
  const nx_array *a; /* dst, src */
  const nx_loop *l;
} copy_ctx;

/* Copies [n] elements of [w] bytes, the [k]th from [s + k·bs] to
   [d + k·bd]. Called with a constant [w], the copies are loads and stores. */
static inline __attribute__((always_inline)) void strided(
    uint8_t *d, int64_t bd, const uint8_t *s, int64_t bs, int64_t n,
    size_t w) {
  for (int64_t k = 0; k < n; k++, d += bd, s += bs) memcpy(d, s, w);
}

/* Copies the 4x4 block whose column q is the [4·w] bytes at [s + q·sc]
   into the rows at [d + p·dr], [4·w] bytes each: a transpose. */
static inline __attribute__((always_inline)) void block(
    uint8_t *d, int64_t dr, const uint8_t *s, int64_t sc, size_t w) {
  uint8_t x[4][4 * 16];
  for (int q = 0; q < 4; q++) memcpy(x[q], s + q * sc, 4 * w);
  for (int p = 0; p < 4; p++)
    for (int q = 0; q < 4; q++) memcpy(d + p * dr + q * w, x[q] + p * w, w);
}

static void copy_run(void *ctx, const int64_t *at, int64_t len) {
  copy_ctx *c = ctx;
  const nx_array *dst = &c->a[0], *src = &c->a[1];
  int bits = dst->bits, r = c->l->rank;
  int64_t sd = c->l->step[0][r - 1], ss = c->l->step[1][r - 1];
  int64_t pd = at[0], ps = at[1];
  if (bits < 8) {
    for (int64_t j = 0; j < len; j++, pd += sd, ps += ss)
      nx_sub_store(dst->base, bits, pd, nx_sub_load(src->base, bits, ps));
    return;
  }
  int64_t w = bits / 8;
  uint8_t *d = dst->base + pd * w;
  const uint8_t *s = src->base + ps * w;
  if (sd == 1 && ss == 1) {
    memcpy(d, s, (size_t)(len * w));
    return;
  }
  switch (w) {
    case 1: strided(d, sd, s, ss, len, 1); return;
    case 2: strided(d, 2 * sd, s, 2 * ss, len, 2); return;
    case 4: strided(d, 4 * sd, s, 4 * ss, len, 4); return;
    case 8: strided(d, 8 * sd, s, 8 * ss, len, 8); return;
    default: strided(d, 16 * sd, s, 16 * ss, len, 16); return;
  }
}

/* A tiled copy: [rows] rows of [cols] elements of [w] bytes, rows [dr] and
   [sr] bytes apart in [d] and [s], elements [w] bytes apart in [d] and [sc]
   in [s]. */
static inline __attribute__((always_inline)) void tile(
    uint8_t *d, int64_t dr, const uint8_t *s, int64_t sr, int64_t sc,
    int64_t rows, int64_t cols, size_t w) {
  int64_t i = 0;
  if (sr == (int64_t)w)
    for (; i + 4 <= rows; i += 4) {
      int64_t j = 0;
      for (; j + 4 <= cols; j += 4)
        block(d + i * dr + j * w, dr, s + i * sr + j * sc, sc, w);
      for (int p = 0; p < 4; p++)
        strided(d + (i + p) * dr + j * w, w, s + (i + p) * sr + j * sc, sc,
                cols - j, w);
    }
  for (; i < rows; i++) strided(d + i * dr, w, s + i * sr, sc, cols, w);
}

/* Copies a plane in tiles: [len] rows along the loop's axis r - 2, each as
   long as its innermost extent. */
static void tile_run(void *ctx, const int64_t *at, int64_t len) {
  copy_ctx *c = ctx;
  const nx_loop *l = c->l;
  int r = l->rank;
  int64_t w = c->a[0].bits / 8, cols = l->extent[r - 1];
  int64_t dr = l->step[0][r - 2] * w, sr = l->step[1][r - 2] * w;
  int64_t sc = l->step[1][r - 1] * w, side = TILE / w;
  for (int64_t i = 0; i < len; i += side)
    for (int64_t j = 0; j < cols; j += side) {
      uint8_t *d = c->a[0].base + at[0] * w + i * dr + j * w;
      const uint8_t *s = c->a[1].base + at[1] * w + i * sr + j * sc;
      int64_t m = len - i < side ? len - i : side;
      int64_t n = cols - j < side ? cols - j : side;
      switch (w) {
        case 1: tile(d, dr, s, sr, sc, m, n, 1); break;
        case 2: tile(d, dr, s, sr, sc, m, n, 2); break;
        case 4: tile(d, dr, s, sr, sc, m, n, 4); break;
        case 8: tile(d, dr, s, sr, sc, m, n, 8); break;
        default: tile(d, dr, s, sr, sc, m, n, 16); break;
      }
    }
}

static int64_t magnitude(int64_t x) { return x < 0 ? -x : x; }

/* The outer axis of [l] on which [src] (operand 1) steps least, if it
   steps less there than on the innermost axis, where [dst] steps by one
   element; -1 otherwise. */
static int tile_axis(const nx_loop *l) {
  int r = l->rank, t = -1;
  if (l->step[0][r - 1] != 1) return -1;
  int64_t least = magnitude(l->step[1][r - 1]);
  for (int i = 0; i < r - 1; i++)
    if (magnitude(l->step[1][i]) < least) {
      least = magnitude(l->step[1][i]);
      t = i;
    }
  return t;
}

value nx_array_copy(value dst, value src) {
  int dt = nx_array_dtype(src);
  nx_operand in[2] = {{dst, dt, 1}, {src, dt, 0}};
  nx_array a[2];
  nx_loop l;
  int e = nx_read(2, in, a);
  if (e) return Val_int(e);
  if (!(e = nx_coalesce(2, a, &l))) {
    int r = l.rank, t = a[0].bits < 8 ? -1 : tile_axis(&l);
    copy_ctx c = {a, &l};
    if (t < 0) walk(2, &l, &c, copy_run);
    else {
      /* Axis t becomes the rows, next to the innermost; the walk visits
         the axes before them, and each call copies the tiles of a plane. */
      int64_t x = l.extent[t];
      l.extent[t] = l.extent[r - 2];
      l.extent[r - 2] = x;
      for (int k = 0; k < 2; k++) {
        x = l.step[k][t];
        l.step[k][t] = l.step[k][r - 2];
        l.step[k][r - 2] = x;
      }
      nx_loop planes = l;
      planes.rank = r - 1;
      walk(2, &planes, &c, tile_run);
    }
  }
  nx_done(2, a);
  return Val_int(e);
}
