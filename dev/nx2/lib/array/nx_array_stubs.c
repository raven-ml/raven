/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

#include <string.h>

#include <caml/alloc.h>
#include <caml/bigarray.h>
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

/* Copies the layout [v] into [a]'s rank, flags, offset and dims: the record
   lives in the OCaml heap, which moves. Answers NX_LAYOUT if its arrays do
   not make a layout. */
static int read_layout(value v, nx_array *a) {
  value shape = Field(v, NX_LAYOUT_SHAPE);
  value strides = Field(v, NX_LAYOUT_STRIDES);
  mlsize_t r = Wosize_val(shape);
  if (r > NX_MAX_RANK || Wosize_val(strides) != r) return NX_LAYOUT;
  a->rank = (int)r;
  a->offset = Long_val(Field(v, NX_LAYOUT_OFFSET));
  a->flags = (int)Long_val(Field(v, NX_LAYOUT_FLAGS));
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
    case RIG_READ_ONLY: return NX_READ_ONLY;
  }
  caml_fatal_error("nx_read: rig_buffer_claim answered %d", (int)c);
}

/* The bytes [[first, last)] that the array [v], read into [a], reaches, as
   host addresses; none for an array with no element. */
static void reach(value v, const nx_array *a, int64_t *first, int64_t *last) {
  if (a->base == NULL) {
    *first = *last = 0;
    return;
  }
  value l = Field(v, ARRAY_LAYOUT);
  int64_t base = (int64_t)(intptr_t)a->base;
  *first = base + Long_val(Field(l, NX_LAYOUT_LO)) * a->bits / 8;
  *last = base + (Long_val(Field(l, NX_LAYOUT_HI)) * a->bits + 7) / 8;
}

int nx_read(int n, const nx_operand *in, nx_array *out) {
  if (n <= 0) return NX_OK;
  for (int k = 0; k < n; k++) {
    value v = in[k].array;
    nx_array *a = &out[k];
    a->dtype = nx_array_dtype(v);
    if (a->dtype != in[k].dtype) return NX_DTYPE;
    a->bits = nx_dtype_row_of(a->dtype).bits;
    int e = read_layout(Field(v, ARRAY_LAYOUT), a);
    if (e) return e;
    a->buffer = Field(v, ARRAY_BUFFER);
    if (rig_buffer_why(a->buffer) != NULL) return NX_DEAD;
    if (a->flags & NX_EMPTY)
      a->base = NULL;
    else if ((a->base = rig_buffer_host(a->buffer)) == NULL)
      return NX_NOT_HOST;
    if (in[k].written && !(a->flags & NX_DISTINCT)) return NX_NOT_DISTINCT;
  }
  for (int k = 0; k < n; k++) {
    int64_t first, last, first_j, last_j;
    if (!in[k].written) continue;
    reach(in[k].array, &out[k], &first, &last);
    if (first == last) continue;
    for (int j = 0; j < n; j++) {
      if (j == k) continue;
      reach(in[j].array, &out[j], &first_j, &last_j);
      if (first_j < last && first < last_j) return NX_OVERLAP;
    }
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
  struct caml__roots_block **roots = &CAML_LOCAL_ROOTS, *top = *roots;
  for (int k = 0; k < n; k++) {
    out[k].roots.next = top;
    out[k].roots.ntables = 1;
    out[k].roots.nitems = 1;
    out[k].roots.tables[0] = &out[k].buffer;
    top = &out[k].roots;
  }
  *roots = top;
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
  /* Operands all in C order merge into one run: the loop is their number
     of elements, in steps of one. */
  int contiguous = 1;
  for (int k = 0; k < n; k++) contiguous &= (a[k].flags & NX_CONTIGUOUS) != 0;
  if (contiguous) {
    int64_t e = 1;
    for (int i = 0; i < r; i++) e *= a[0].dim[i];
    l->rank = 1;
    l->extent[0] = e;
    for (int k = 0; k < n; k++) l->step[k][0] = e > 1;
    return NX_OK;
  }
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
      joins = l->step[k][out - 1] == a[k].dim[r + i] * d;
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
   steps are the loop's innermost. It is inlined at each call, where [run]
   is known, so that [run] is inlined into the odometer: clang at -O3
   otherwise calls it through the pointer, once per run. */
static inline __attribute__((always_inline)) void walk(
    int n, const nx_loop *l, void *ctx,
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
  if (n < 1 || n > NX_MAX_OPERANDS) return Val_int(NX_ARITY);
  for (int k = 0; k < n; k++) {
    int e = read_layout(Field(ls, k), &a[k]);
    if (e) return Val_int(e);
  }
  int e = nx_coalesce(n, a, &l);
  if (e) return Val_int(e);
  int r = l.rank;
  /* [out] is an int array: immediates need no write barrier. */
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
    case NX_COMPLEX128:
      store_float(base, NX_FLOAT64, 2 * p + part, 0, x);
      return;
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
  value out;
  int64_t k; /* the next element of [out] */
} to_ctx;

/* Writes a run of flat or immediate elements: no allocation. */
static void to_run(void *ctx, const int64_t *at, int64_t len) {
  to_ctx *c = ctx;
  const nx_array *a = c->a;
  int dt = a->dtype;
  int64_t step = c->l->step[0][c->l->rank - 1];
  int64_t p = at[0];
  if (representation(dt) == TO_IMMEDIATE) {
    /* Immediates need no write barrier. */
    for (int64_t j = 0; j < len; j++, p += step)
      Field(c->out, c->k++) = Val_long(load_int(a->base, dt, p));
    return;
  }
  double *dst = (double *)Op_val(c->out) + c->k;
  c->k += len;
  /* A loop per format, the format's decoder inlined in it. */
#define DECODE(T, decode)                                                 \
  do {                                                                    \
    const T *src = (const T *)a->base + p;                                \
    for (int64_t j = 0; j < len; j++) dst[j] = decode(src[j * step]);     \
  } while (0)
  switch (dt) {
    case NX_FLOAT64: DECODE(double, (double)); return;
    case NX_FLOAT32: DECODE(float, (double)); return;
    case NX_FLOAT16:
      if (step == 1)
        nx_f16_to_double_run((const uint16_t *)a->base + p, dst, (size_t)len);
      else DECODE(uint16_t, nx_f16_to_float);
      return;
    case NX_BFLOAT16: DECODE(uint16_t, nx_bf16_to_float); return;
    case NX_FLOAT8_E4M3FN: DECODE(uint8_t, nx_e4m3fn_to_float); return;
    case NX_FLOAT8_E5M2: DECODE(uint8_t, nx_e5m2_to_float); return;
    default: /* float4: elements share bytes */
      for (int64_t j = 0; j < len; j++, p += step)
        dst[j] = load_float(a->base, dt, p, 0);
      return;
  }
#undef DECODE
}

/* to_array: reads [v]'s elements, in C order of indices, into [out], an
   OCaml array of their number, flat floats or immediates. It allocates
   nothing; elements that box go through nx_array_to_bigarray. */
value nx_array_to_array(value v, value out) {
  nx_operand in = {v, nx_array_dtype(v), 0};
  nx_array a;
  nx_loop l;
  if (representation(in.dtype) == TO_BOXED) return Val_int(NX_DTYPE);
  int e = nx_read(1, &in, &a);
  if (e) return Val_int(e);
  if (!(e = nx_coalesce(1, &a, &l))) {
    to_ctx c = {&a, &l, out, 0};
    walk(1, &l, &c, to_run);
  }
  nx_done(1, &a);
  return Val_int(e);
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
      /* A loop per format, the format's encoder inlined in it. */
#define ENCODE(T, encode)                                                 \
  do {                                                                    \
    T *dst = (T *)a->base + p;                                            \
    for (int64_t j = 0; j < len; j++) dst[j * step] = encode(src[j]);     \
  } while (0)
      switch (dt) {
        case NX_FLOAT64: ENCODE(double, (double)); return;
        case NX_FLOAT32: ENCODE(float, (float)); return;
        case NX_FLOAT16:
          if (step == 1)
            nx_double_to_f16_run(src, (uint16_t *)a->base + p, (size_t)len);
          else ENCODE(uint16_t, nx_double_to_f16);
          return;
        case NX_BFLOAT16: ENCODE(uint16_t, nx_double_to_bf16); return;
        case NX_FLOAT8_E4M3FN: ENCODE(uint8_t, nx_double_to_e4m3fn); return;
        case NX_FLOAT8_E5M2: ENCODE(uint8_t, nx_double_to_e5m2); return;
        default: /* float4: elements share bytes */
          for (int64_t j = 0; j < len; j++, p += step)
            store_float(a->base, dt, p, 0, src[j]);
          return;
      }
#undef ENCODE
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
  if (!(e = nx_coalesce(1, &a, &l))) {
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

/* Sub-byte runs

   A run of sub-byte elements written one after another covers whole bytes,
   except at most one partial byte at each end, which it shares with
   elements outside it. The whole bytes are plain stores: no other write
   reaches them. The partial ones are one compare-and-swap each, which keeps
   the other elements' bits against stores from other threads. */

/* Stores [x]'s bits under [mask] into [*b], keeping its other bits. */
static inline void put_bits(uint8_t *b, uint8_t mask, uint8_t x) {
  uint8_t old = __atomic_load_n(b, __ATOMIC_RELAXED);
  while (!__atomic_compare_exchange_n(b, &old, (uint8_t)((old & ~mask) | x), 1,
                                      __ATOMIC_RELAXED, __ATOMIC_RELAXED))
    ;
}

/* The [k] elements of [bits] bits from position [p] of [s], [step] apart,
   packed LSB first. */
static inline uint8_t pack(const uint8_t *s, int bits, int64_t p, int64_t step,
                           int k) {
  uint8_t x = 0;
  for (int j = 0; j < k; j++, p += step)
    x |= (uint8_t)(nx_sub_load(s, bits, p) << (j * bits));
  return x;
}

/* Copies [len] elements of [bits] bits from position [ps] of [s], [ss]
   apart, to positions [pd], [pd + 1], … of [d]. The whole bytes of [d] come
   from [memcpy] when both runs start at the same bit of a byte, from two
   bytes of [s] shifted when they start at different bits, and packed from
   loads otherwise. It stays out of line so that copy_run, inlined into the
   walk, stays small. */
static __attribute__((noinline)) void sub_run(uint8_t *d, int64_t pd,
                                              const uint8_t *s, int64_t ps,
                                              int64_t ss, int64_t len,
                                              int bits) {
  int per = 8 / bits;
  int head = (int)((per - pd % per) % per);
  if (head > len) head = (int)len;
  if (head > 0) {
    int at = (int)(pd % per) * bits;
    uint8_t mask = (uint8_t)(((1u << (head * bits)) - 1) << at);
    put_bits(d + pd / per, mask, (uint8_t)(pack(s, bits, ps, ss, head) << at));
    pd += head;
    ps += head * ss;
    len -= head;
  }
  int64_t whole = len / per;
  uint8_t *db = d + pd / per;
  if (whole > 0 && ss == 1) {
    int64_t bit = ps * bits;
    int sh = (int)(bit & 7);
    const uint8_t *sb = s + (bit >> 3);
    if (sh == 0)
      memcpy(db, sb, (size_t)whole);
    else {
      /* Byte i takes the top of sb[i] and the bottom of sb[i + 1]. sb[0]
         and sb[whole] hold elements outside the run, which other threads
         may store: they are loaded atomically. */
      uint8_t first = __atomic_load_n(sb, __ATOMIC_RELAXED);
      uint8_t last = __atomic_load_n(sb + whole, __ATOMIC_RELAXED);
      if (whole == 1)
        db[0] = (uint8_t)((first >> sh) | (last << (8 - sh)));
      else {
        db[0] = (uint8_t)((first >> sh) | (sb[1] << (8 - sh)));
        int64_t i = 1;
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        /* Eight bytes at a time: the word at sb + i, shifted, and the
           bottom of the word after it. */
        if (i + 17 < whole) {
          uint64_t lo, hi;
          memcpy(&lo, sb + i, 8);
          for (; i + 17 < whole; i += 8, lo = hi) {
            memcpy(&hi, sb + i + 8, 8);
            uint64_t w = (lo >> sh) | (hi << (64 - sh));
            memcpy(db + i, &w, 8);
          }
        }
#endif
        for (; i < whole - 1; i++)
          db[i] = (uint8_t)((sb[i] >> sh) | (sb[i + 1] << (8 - sh)));
        db[whole - 1] = (uint8_t)((sb[whole - 1] >> sh) | (last << (8 - sh)));
      }
    }
  } else
    for (int64_t i = 0; i < whole; i++)
      db[i] = pack(s, bits, ps + i * per * ss, ss, per);
  pd += whole * per;
  ps += whole * per * ss;
  len -= whole * per;
  if (len > 0)
    put_bits(d + pd / per, (uint8_t)((1u << (len * bits)) - 1),
             pack(s, bits, ps, ss, (int)len));
}

static void copy_run(void *ctx, const int64_t *at, int64_t len) {
  copy_ctx *c = ctx;
  const nx_array *dst = &c->a[0], *src = &c->a[1];
  int bits = dst->bits, r = c->l->rank;
  int64_t sd = c->l->step[0][r - 1], ss = c->l->step[1][r - 1];
  int64_t pd = at[0], ps = at[1];
  if (bits < 8 && sd == 1) {
    sub_run(dst->base, pd, src->base, ps, ss, len, bits);
    return;
  }
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

/* Swaps the axes [i] and [j] of the loop [l] over two operands. */
static void swap_axes(nx_loop *l, int i, int j) {
  int64_t x = l->extent[i];
  l->extent[i] = l->extent[j];
  l->extent[j] = x;
  for (int k = 0; k < 2; k++) {
    x = l->step[k][i];
    l->step[k][i] = l->step[k][j];
    l->step[k][j] = x;
  }
}

/* The fewest elements a run of the gather takes when an axis has as many. */
#define SHORT 8

/* The gather: copies the elements of a[1] into a[0], operands of one dtype
   read through the door. Answers NX_SHAPE if their shapes differ. It is the
   layer's one tiled walk.

   Elements are independent, so the order of the walk decides no bit. An
   innermost axis of fewer than SHORT elements, as a small window's, would
   cost a run per few elements: the nearest outer axis of at least SHORT
   becomes the innermost. Then where a[0] steps by one element along the
   innermost axis and a[1] steps less across rows than along them, the copy
   goes in tiles. */
static int gather(nx_array a[2]) {
  nx_loop l;
  int e = nx_coalesce(2, a, &l);
  if (e) return e;
  int r = l.rank, t = -1;
  copy_ctx c = {a, &l};
  if (a[0].bits >= 8) {
    if (l.extent[r - 1] < SHORT)
      for (int i = r - 2; i >= 0; i--)
        if (l.extent[i] >= SHORT) {
          swap_axes(&l, i, r - 1);
          break;
        }
    t = tile_axis(&l);
  }
  if (t < 0) {
    walk(2, &l, &c, copy_run);
    return NX_OK;
  }
  /* Axis t becomes the rows, next to the innermost; the walk visits the
     axes before them, and each call copies the tiles of a plane. */
  swap_axes(&l, t, r - 2);
  nx_loop planes = l;
  planes.rank = r - 1;
  walk(2, &planes, &c, tile_run);
  return NX_OK;
}

value nx_array_copy(value dst, value src) {
  int dt = nx_array_dtype(src);
  nx_operand in[2] = {{dst, dt, 1}, {src, dt, 0}};
  nx_array a[2];
  int e = nx_read(2, in, a);
  if (e) return Val_int(e);
  e = gather(a);
  nx_done(2, a);
  return Val_int(e);
}

/* to_bigarray: gathers [v]'s elements, in C order of indices, into [out], a
   bigarray of their number and width, bit for bit. to_array boxes elements
   from there once the claim is released, so that an allocation that raises
   holds no claim. */
value nx_array_to_bigarray(value v, value out) {
  nx_operand in = {v, nx_array_dtype(v), 0};
  nx_array a[2];
  int e = nx_read(1, &in, &a[1]);
  if (e) return Val_int(e);
  /* [out] as a C-contiguous array of [v]'s dtype and shape. */
  nx_array *d = &a[0];
  int r = a[1].rank;
  d->base = Caml_ba_data_val(out);
  d->dtype = a[1].dtype;
  d->bits = a[1].bits;
  d->rank = r;
  d->flags = NX_CONTIGUOUS | NX_DISTINCT;
  d->offset = 0;
  int64_t stride = 1;
  for (int i = r - 1; i >= 0; i--) {
    d->dim[i] = a[1].dim[i];
    d->dim[r + i] = stride;
    stride *= a[1].dim[i];
  }
  e = gather(a);
  nx_done(1, &a[1]);
  return Val_int(e);
}
