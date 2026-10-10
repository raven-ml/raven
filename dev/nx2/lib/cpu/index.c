/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Gathers and scatters.

   A gather is a loop over its positions' shape of three operands: the
   destination, the positions, and the source read with step 0 along the
   gathered axis, whose stride is kept apart. Each element reads its
   position p, then the source at its own index plus p strides, or zero
   bits for a p outside the axis: one unsigned compare. Where the positions
   step 0 along a block's row, as a row take's or an embedding's do, the
   row is one copy from the source's row p.

   A scatter copies [into] to the destination, unless the two are one
   array, then applies the updates. Update j's target has j's indices off
   the axis, so those indices cut the targets into disjoint slices, which
   any thread may take; within a slice the updates apply in increasing
   index along the axis, which is C order of the updates for each target.
   A unit is a run of at most UNIT_ROW slices along the innermost axis off
   the scattered one, and walks the scattered axis once, the whole run at
   each step: where the positions step 0 along the run, as an embedding
   gradient's do, the step is one row of the kind. Where runs are fewer
   than the job's threads and the axis long, as a 1-d scatter's is, each
   thread also takes a part of the targets along the axis and walks every
   update of its run, applying those that land in its part: the order per
   target stays C order.

   Add adds left to right from +0 + x. nx_add keeps the first NaN operand,
   and a zero sum is stored as +0: a sum that starts from +0 is never -0,
   and (+0 + x) + u differs from x + u only in that sign. A narrow float
   accumulates in float32 in scratch, beside a flag per target that an
   update reached, and encodes those targets once at the end, so that an
   untouched element keeps its bits. Max and Min keep the first NaN, the
   target's value first; on narrow floats and complex numbers they select
   an operand by the order of nx_kinds.h's domains, so that the result
   keeps its bits. */

#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"
#include "nx_kinds.h"

/* Elements of a gather's block: a copy's. */
#define GATHER_MOST (64 * 1024)

/* Slices of a scatter's unit: a run of 1024 float32 targets is a 4 KiB
   row, and a unit walks the whole axis of updates per run. */
#define UNIT_ROW 1024

/* Updates along the axis from which threads split the targets into parts,
   each reading every update. */
#define PART_UPDATES (64 * 1024)

typedef struct {
  uint64_t lo, hi;
} u128;

/* Gathers */

typedef struct {
  const nx_array *a; /* the destination, the positions, the source */
  int64_t sa;        /* the source's stride along the axis */
  uint64_t dim;      /* its extent there */
  int w;             /* bytes per element, 0 for a sub-byte dtype */
} gather;

/* A row of [n] elements of the type [T]: element i of [d], stepping [sd],
   is the source [x] at i·sx plus its position's [sa] strides, its
   position at i·si of [idx]. */
#define GATHER_ROW(NAME, T)                                                  \
  static void NAME(uint8_t *d_, const int64_t *idx, const uint8_t *x_,      \
                   int64_t n, int64_t sd, int64_t si, int64_t sx,           \
                   int64_t sa, uint64_t dim) {                              \
    T *d = (T *)d_;                                                          \
    const T *x = (const T *)x_;                                              \
    const T zero = {0};                                                      \
    if (si == 0) {                                                           \
      uint64_t p = (uint64_t)idx[0];                                         \
      if (p >= dim) {                                                        \
        for (int64_t i = 0; i < n; i++) d[i * sd] = zero;                    \
        return;                                                              \
      }                                                                      \
      const T *r = x + (int64_t)p * sa;                                      \
      if (sd == 1 && sx == 1) memcpy(d, r, (size_t)n * sizeof(T));           \
      else                                                                   \
        for (int64_t i = 0; i < n; i++) d[i * sd] = r[i * sx];               \
      return;                                                                \
    }                                                                        \
    for (int64_t i = 0; i < n; i++) {                                        \
      uint64_t p = (uint64_t)idx[i * si];                                    \
      d[i * sd] = p < dim ? x[i * sx + (int64_t)p * sa] : zero;              \
    }                                                                        \
  }

GATHER_ROW(gather_1, uint8_t)
GATHER_ROW(gather_2, uint16_t)
GATHER_ROW(gather_4, uint32_t)
GATHER_ROW(gather_8, uint64_t)
GATHER_ROW(gather_16, u128)

static void gather_block(const nx_cpu_block *b, void *ctx) {
  const gather *g = ctx;
  const nx_array *a = g->a;
  const int64_t *idx = (const int64_t *)a[1].base;
  for (int64_t q = 0; q < b->n2; q++)
    for (int64_t j = 0; j < b->n1; j++) {
      int64_t pd = b->at[0] + q * b->s2[0] + j * b->s1[0];
      int64_t pi = b->at[1] + q * b->s2[1] + j * b->s1[1];
      int64_t px = b->at[2] + q * b->s2[2] + j * b->s1[2];
      int64_t sd = b->s0[0], si = b->s0[1], sx = b->s0[2];
      if (g->w == 0) {
        int bits = a[0].bits;
        for (int64_t i = 0; i < b->n0; i++) {
          uint64_t p = (uint64_t)idx[pi + i * si];
          uint32_t v = p < g->dim ? nx_sub_load(a[2].base, bits,
                                                px + i * sx + (int64_t)p * g->sa)
                                  : 0;
          nx_sub_store(a[0].base, bits, pd + i * sd, v);
        }
        continue;
      }
      uint8_t *d = a[0].base + pd * g->w;
      const uint8_t *x = a[2].base + px * g->w;
      void (*row)(uint8_t *, const int64_t *, const uint8_t *, int64_t,
                  int64_t, int64_t, int64_t, int64_t, uint64_t) =
          g->w == 1   ? gather_1
          : g->w == 2 ? gather_2
          : g->w == 4 ? gather_4
          : g->w == 8 ? gather_8
                      : gather_16;
      row(d, idx + pi, x, b->n0, sd, si, sx, g->sa, g->dim);
    }
}

/* The shape of [v] into [s], its rank answered. */
static int shape_of(value v, int64_t *s) {
  int64_t dim[2 * NX_MAX_RANK], off;
  int r = nx_array_layout(v, dim, &off);
  for (int i = 0; i < r; i++) s[i] = dim[i];
  return r;
}

/* Whether [x] of rank [r] and [y] of rank [ry] have one rank, above [axis],
   and one extent along every other axis. */
static int fits_off_axis(const int64_t *x, int r, const int64_t *y, int ry,
                         int axis) {
  if (r != ry || axis >= r) return 0;
  for (int i = 0; i < r; i++)
    if (i != axis && x[i] != y[i]) return 0;
  return 1;
}

static int same_shape(const int64_t *x, int r, const int64_t *y, int ry) {
  if (r != ry) return 0;
  for (int i = 0; i < r; i++)
    if (x[i] != y[i]) return 0;
  return 1;
}

value nx_cpu_gather(value s, value vd, value vi, value vx) {
  CAMLparam4(s, vd, vi, vx);
  int axis = ((const nx_spec_axis *)String_val(s))->axis;
  int dt = nx_array_dtype(vx);
  int64_t ys[NX_MAX_RANK], is[NX_MAX_RANK], xs[NX_MAX_RANK];
  int ry = shape_of(vd, ys), ri = shape_of(vi, is), rx = shape_of(vx, xs);
  if (!fits_off_axis(is, ri, xs, rx, axis) || !same_shape(ys, ry, is, ri))
    CAMLreturn(Val_int(NX_SHAPE));
  nx_operand in[3] = {{vd, dt, 1}, {vi, NX_INT64, 0}, {vx, dt, 0}};
  nx_array a[3];
  int e = nx_read(3, in, a);
  if (e) CAMLreturn(Val_int(e));
  int bits = nx_dtype_row_of(dt).bits;
  gather g = {.a = a,
              .sa = a[2].dim[rx + axis],
              .dim = (uint64_t)a[2].dim[axis],
              .w = bits < 8 ? 0 : bits / 8};
  nx_loop l = {.rank = ri, .first = {a[0].offset, a[1].offset, a[2].offset}};
  for (int i = 0; i < ri; i++) {
    l.extent[i] = a[1].dim[i];
    l.step[0][i] = a[0].dim[ry + i];
    l.step[1][i] = a[1].dim[ri + i];
    l.step[2][i] = i == axis ? 0 : a[2].dim[rx + i];
  }
  l.rank = nx_coalesce_dims(3, ri, l.extent, l.step);
  nx_cpu_walk(3, a, &l, GATHER_MOST, gather_block, &g);
  nx_done(3, a);
  CAMLreturn(Val_int(NX_OK));
}

/* Scatters */

typedef struct scatter scatter;

/* Applies the updates of a run of [n] slices from the positions [pd], [pi]
   and [pu] of the destination, the positions and the updates, those whose
   position lies in [lo, lo + span). */
typedef void (*scatter_run)(const scatter *c, int64_t pd, int64_t pi,
                            int64_t pu, int64_t n, uint64_t lo,
                            uint64_t span);

struct scatter {
  const nx_array *a; /* the destination, into, the positions, the updates */
  scatter_run run;
  int dt, combine;
  /* The slices: a loop of the destination, the positions and the updates
     off the axis; [sd], [si] and [su] their steps along its last axis. */
  nx_loop l;
  int64_t sd, si, su;
  int64_t pieces; /* runs per row of slices */
  int64_t parts;  /* parts of the axis's targets, each a unit's */
  /* Along the axis: the updates' extent, their steps through the positions
     and the updates, the destination's stride and extent. */
  int64_t m, ia, ua, da;
  uint64_t dim;
  /* A narrow float's sums, by the destination's element, and whether an
     update reached each. */
  float *acc;
  uint8_t *touched;
};

/* Sums that start from +0: the first NaN operand, or the sum, +0 for a
   zero. */
static inline float add_f32(float a, float b) {
  float s = nx_add_f32(a, b);
  return s == 0 ? 0.0f : s;
}

static inline double add_f64(double a, double b) {
  double s = nx_add_f64(a, b);
  return s == 0 ? 0.0 : s;
}

/* Typed runs: [F] combines the target's element with the update. Where the
   positions step 0 along the run, each step of the axis is one row. */
#define SCATTER_RUN(NAME, T, F)                                              \
  static void NAME(const scatter *c, int64_t pd, int64_t pi, int64_t pu,    \
                   int64_t n, uint64_t lo, uint64_t span) {                  \
    T *d = (T *)c->a[0].base;                                                \
    const int64_t *x = (const int64_t *)c->a[2].base;                        \
    const T *u = (const T *)c->a[3].base;                                    \
    int64_t sd = c->sd, si = c->si, su = c->su, da = c->da;                  \
    int64_t m = c->m, ia = c->ia, ua = c->ua;                                \
    if (n == 1) {                                                            \
      d += pd;                                                               \
      for (int64_t t = 0; t < m; t++, pi += ia, pu += ua) {                  \
        uint64_t p = (uint64_t)x[pi];                                        \
        if (p - lo >= span) continue;                                        \
        T *e = d + (int64_t)p * da;                                          \
        *e = F(*e, u[pu]);                                                   \
      }                                                                      \
      return;                                                                \
    }                                                                        \
    for (int64_t t = 0; t < m; t++, pi += ia, pu += ua) {                    \
      if (si == 0) {                                                         \
        uint64_t p = (uint64_t)x[pi];                                        \
        if (p - lo >= span) continue;                                        \
        T *e = d + pd + (int64_t)p * da;                                     \
        const T *v = u + pu;                                                 \
        for (int64_t i = 0; i < n; i++) e[i * sd] = F(e[i * sd], v[i * su]); \
        continue;                                                            \
      }                                                                      \
      for (int64_t i = 0; i < n; i++) {                                      \
        uint64_t p = (uint64_t)x[pi + i * si];                               \
        if (p - lo >= span) continue;                                        \
        T *e = d + pd + (int64_t)p * da + i * sd;                            \
        *e = F(*e, u[pu + i * su]);                                          \
      }                                                                      \
    }                                                                        \
  }

#define SET(a, b) (b)
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((b) < (a) ? (b) : (a))
#define ADD_U8(a, b) ((uint8_t)((a) + (b)))
#define ADD_U16(a, b) ((uint16_t)((a) + (b)))
#define ADD_U32(a, b) ((uint32_t)((a) + (b)))
#define ADD_U64(a, b) ((uint64_t)((a) + (b)))

SCATTER_RUN(set_1, uint8_t, SET)
SCATTER_RUN(set_2, uint16_t, SET)
SCATTER_RUN(set_4, uint32_t, SET)
SCATTER_RUN(set_8, uint64_t, SET)
SCATTER_RUN(set_16, u128, SET)
SCATTER_RUN(add_f32_run, float, add_f32)
SCATTER_RUN(add_f64_run, double, add_f64)
SCATTER_RUN(add_8, uint8_t, ADD_U8)
SCATTER_RUN(add_16, uint16_t, ADD_U16)
SCATTER_RUN(add_32, uint32_t, ADD_U32)
SCATTER_RUN(add_64, uint64_t, ADD_U64)
SCATTER_RUN(max_f32, float, nx_maximum_f32)
SCATTER_RUN(min_f32, float, nx_minimum_f32)
SCATTER_RUN(max_f64, double, nx_maximum_f64)
SCATTER_RUN(min_f64, double, nx_minimum_f64)
SCATTER_RUN(max_i8, int8_t, MAX)
SCATTER_RUN(min_i8, int8_t, MIN)
SCATTER_RUN(max_u8, uint8_t, MAX)
SCATTER_RUN(min_u8, uint8_t, MIN)
SCATTER_RUN(max_i16, int16_t, MAX)
SCATTER_RUN(min_i16, int16_t, MIN)
SCATTER_RUN(max_u16, uint16_t, MAX)
SCATTER_RUN(min_u16, uint16_t, MIN)
SCATTER_RUN(max_i32, int32_t, MAX)
SCATTER_RUN(min_i32, int32_t, MIN)
SCATTER_RUN(max_u32, uint32_t, MAX)
SCATTER_RUN(min_u32, uint32_t, MIN)
SCATTER_RUN(max_i64, int64_t, MAX)
SCATTER_RUN(min_i64, int64_t, MIN)
SCATTER_RUN(max_u64, uint64_t, MAX)
SCATTER_RUN(min_u64, uint64_t, MIN)

/* Elements one at a time: sub-byte dtypes, narrow floats and complex
   numbers. */

/* The rank of a float in the order, -0 below +0, for a value that is not
   NaN. */
static inline uint32_t rank_f32(float x) {
  uint32_t b = nx_float_bits(x);
  return b & 0x80000000u ? ~b : b | 0x80000000u;
}

static inline uint64_t rank_f64(double x) {
  uint64_t b = nx_double_bits(x);
  return b & 0x8000000000000000ull ? ~b : b | 0x8000000000000000ull;
}

/* Whether the extreme of [combine] of a and b is a: the first NaN, else the
   greater (lesser) by rank, a on a tie. */
static inline int keeps_f32(int combine, float a, float b) {
  if (nx_float_nan(a)) return 1;
  if (nx_float_nan(b)) return 0;
  return combine == NX_SCATTER_MAX ? rank_f32(a) >= rank_f32(b)
                                   : rank_f32(a) <= rank_f32(b);
}

/* A complex number's order: its real part, then its imaginary part; one
   with a NaN part is a NaN. */
static inline int keeps_c64(int combine, const float *a, const float *b) {
  int na = nx_float_nan(a[0]) || nx_float_nan(a[1]);
  int nb = nx_float_nan(b[0]) || nx_float_nan(b[1]);
  if (na) return 1;
  if (nb) return 0;
  uint32_t ra = rank_f32(a[0]), rb = rank_f32(b[0]);
  if (ra == rb) {
    ra = rank_f32(a[1]);
    rb = rank_f32(b[1]);
  }
  return combine == NX_SCATTER_MAX ? ra >= rb : ra <= rb;
}

static inline int keeps_c128(int combine, const double *a, const double *b) {
  int na = a[0] != a[0] || a[1] != a[1];
  int nb = b[0] != b[0] || b[1] != b[1];
  if (na) return 1;
  if (nb) return 0;
  uint64_t ra = rank_f64(a[0]), rb = rank_f64(b[0]);
  if (ra == rb) {
    ra = rank_f64(a[1]);
    rb = rank_f64(b[1]);
  }
  return combine == NX_SCATTER_MAX ? ra >= rb : ra <= rb;
}

/* The code of a narrow float's element nearest [f]. */
static uint32_t narrow_of_float(int dt, float f) {
  switch (dt) {
    case NX_FLOAT16: return nx_float_to_f16(f);
    case NX_BFLOAT16: return nx_float_to_bf16(f);
    case NX_FLOAT8_E4M3FN: return nx_float_to_e4m3fn(f);
    case NX_FLOAT8_E5M2: return nx_float_to_e5m2(f);
    default: return nx_float_to_e2m1fn(f);
  }
}

static int is_narrow(int dt) {
  return dt == NX_FLOAT16 || dt == NX_BFLOAT16 || dt == NX_FLOAT8_E4M3FN ||
         dt == NX_FLOAT8_E5M2 || dt == NX_FLOAT4_E2M1FN;
}

/* The code at position [p] of [a]: an element of at most 16 bits. */
static inline uint32_t code_at(const nx_array *a, int64_t p) {
  switch (a->bits) {
    case 1:
    case 4: return nx_sub_load(a->base, a->bits, p);
    case 8: return a->base[p];
    default: return ((const uint16_t *)a->base)[p];
  }
}

static inline void code_put(const nx_array *a, int64_t p, uint32_t v) {
  switch (a->bits) {
    case 1:
    case 4: nx_sub_store(a->base, a->bits, p, v); return;
    case 8: a->base[p] = (uint8_t)v; return;
    default: ((uint16_t *)a->base)[p] = (uint16_t)v;
  }
}

/* A sub-byte integer's value: int4's sign-extended. */
static inline int32_t sub_value(int dt, uint32_t c) {
  return dt == NX_INT4 ? (int32_t)((c ^ 8u) - 8u) : (int32_t)c;
}

/* Combines the update at position [pu] into the destination's element at
   [pd]. */
static void combine_one(const scatter *c, int64_t pd, int64_t pu) {
  const nx_array *d = &c->a[0], *u = &c->a[3];
  int dt = c->dt;
  if (dt == NX_COMPLEX64 || dt == NX_COMPLEX128) {
    int w = dt == NX_COMPLEX64 ? 8 : 16;
    uint8_t *e = d->base + pd * w;
    const uint8_t *v = u->base + pu * w;
    if (c->combine == NX_SCATTER_ADD && dt == NX_COMPLEX64) {
      float *x = (float *)e;
      const float *y = (const float *)v;
      x[0] = add_f32(x[0], y[0]);
      x[1] = add_f32(x[1], y[1]);
    } else if (c->combine == NX_SCATTER_ADD) {
      double *x = (double *)e;
      const double *y = (const double *)v;
      x[0] = add_f64(x[0], y[0]);
      x[1] = add_f64(x[1], y[1]);
    } else if (c->combine == NX_SCATTER_SET ||
               !(dt == NX_COMPLEX64
                     ? keeps_c64(c->combine, (float *)e, (const float *)v)
                     : keeps_c128(c->combine, (double *)e,
                                  (const double *)v)))
      memcpy(e, v, (size_t)w);
    return;
  }
  uint32_t a = code_at(d, pd), b = code_at(u, pu);
  if (c->combine == NX_SCATTER_SET) {
    code_put(d, pd, b);
    return;
  }
  if (is_narrow(dt)) {
    if (c->combine == NX_SCATTER_ADD) {
      int64_t k = pd - d->offset;
      if (!c->touched[k]) {
        c->acc[k] = add_f32(0.0f, nx_bits_to_float(dt, a));
        c->touched[k] = 1;
      }
      c->acc[k] = add_f32(c->acc[k], nx_bits_to_float(dt, b));
      return;
    }
    if (!keeps_f32(c->combine, nx_bits_to_float(dt, a),
                   nx_bits_to_float(dt, b)))
      code_put(d, pd, b);
    return;
  }
  /* int4, uint4 and bit; Add wraps modulo 16. */
  if (c->combine == NX_SCATTER_ADD) {
    code_put(d, pd, a + b);
    return;
  }
  int32_t x = sub_value(dt, a), y = sub_value(dt, b);
  int keep = c->combine == NX_SCATTER_MAX ? x >= y : x <= y;
  if (!keep) code_put(d, pd, b);
}

static void element_run(const scatter *c, int64_t pd, int64_t pi, int64_t pu,
                        int64_t n, uint64_t lo, uint64_t span) {
  const int64_t *x = (const int64_t *)c->a[2].base;
  for (int64_t t = 0; t < c->m; t++, pi += c->ia, pu += c->ua)
    for (int64_t i = 0; i < n; i++) {
      uint64_t p = (uint64_t)x[pi + i * c->si];
      if (p - lo >= span) continue;
      combine_one(c, pd + (int64_t)p * c->da + i * c->sd, pu + i * c->su);
    }
}

/* The typed run of [combine] at [dt], or element_run. */
static scatter_run run_of(int combine, int dt) {
  int bits = nx_dtype_row_of(dt).bits;
  if (combine == NX_SCATTER_SET) switch (bits) {
      case 8: return set_1;
      case 16: return set_2;
      case 32: return set_4;
      case 64: return set_8;
      case 128: return set_16;
      default: return element_run;
    }
  int add = combine == NX_SCATTER_ADD, max = combine == NX_SCATTER_MAX;
#define PICK(A, MX, MN) return add ? A : max ? MX : MN
  switch (dt) {
    case NX_FLOAT32: PICK(add_f32_run, max_f32, min_f32);
    case NX_FLOAT64: PICK(add_f64_run, max_f64, min_f64);
    case NX_INT8: PICK(add_8, max_i8, min_i8);
    case NX_UINT8:
    case NX_BOOL: PICK(add_8, max_u8, min_u8);
    case NX_INT16: PICK(add_16, max_i16, min_i16);
    case NX_UINT16: PICK(add_16, max_u16, min_u16);
    case NX_INT32: PICK(add_32, max_i32, min_i32);
    case NX_UINT32: PICK(add_32, max_u32, min_u32);
    case NX_INT64: PICK(add_64, max_i64, min_i64);
    case NX_UINT64: PICK(add_64, max_u64, min_u64);
    default: return element_run;
  }
#undef PICK
}

static void scatter_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const scatter *c = ctx;
  const nx_loop *l = &c->l;
  int r = l->rank;
  int64_t len = l->extent[r - 1];
  for (int64_t u = lo; u < hi; u++) {
    /* The unit's part of the axis's targets, then its run of slices. */
    int64_t part = u % c->parts, v = u / c->parts;
    uint64_t first = c->dim * (uint64_t)part / (uint64_t)c->parts;
    uint64_t next = c->dim * (uint64_t)(part + 1) / (uint64_t)c->parts;
    int64_t i0 = v % c->pieces * UNIT_ROW;
    v /= c->pieces;
    int64_t pd = l->first[0] + i0 * c->sd, pi = l->first[1] + i0 * c->si,
            pu = l->first[2] + i0 * c->su;
    for (int i = r - 2; i >= 0; i--) {
      int64_t x = v % l->extent[i];
      v /= l->extent[i];
      pd += x * l->step[0][i];
      pi += x * l->step[1][i];
      pu += x * l->step[2][i];
    }
    int64_t n = len - i0 < UNIT_ROW ? len - i0 : UNIT_ROW;
    c->run(c, pd, pi, pu, n, first, next - first);
  }
}

/* Stores each narrow float sum into its target. */
static void encode_sums(const scatter *c, int64_t total) {
  const nx_array *d = &c->a[0];
  for (int64_t k = 0; k < total; k++)
    if (c->touched[k])
      code_put(d, d->offset + k, narrow_of_float(c->dt, c->acc[k]));
}

value nx_cpu_scatter(value s, value vd, value vinto, value vi, value vu) {
  CAMLparam5(s, vd, vinto, vi, vu);
  const nx_spec_axis *sp = (const nx_spec_axis *)String_val(s);
  int axis = sp->axis, combine = sp->combine;
  int dt = nx_array_dtype(vinto);
  if (combine == NX_SCATTER_ADD &&
      nx_dtype_row_of(dt).kind == NX_KIND_BOOLEAN)
    CAMLreturn(Val_int(NX_DTYPE));
  int64_t ys[NX_MAX_RANK], ts[NX_MAX_RANK], is[NX_MAX_RANK], us[NX_MAX_RANK];
  int ry = shape_of(vd, ys), rt = shape_of(vinto, ts), ri = shape_of(vi, is),
      ru = shape_of(vu, us);
  if (!same_shape(ys, ry, ts, rt) || !same_shape(is, ri, us, ru) ||
      !fits_off_axis(us, ru, ts, rt, axis))
    CAMLreturn(Val_int(NX_SHAPE));
  nx_operand in[4] = {
      {vd, dt, 1}, {vinto, dt, 0}, {vi, NX_INT64, 0}, {vu, dt, 0}};
  nx_array a[4];
  int e = nx_read(4, in, a);
  if (e) CAMLreturn(Val_int(e));
  int64_t total = 1, updates = 1;
  for (int i = 0; i < ry; i++) total *= ys[i];
  for (int i = 0; i < ru; i++) updates *= us[i];
  if (total > 0 && !a[1].alias) {
    nx_loop l;
    nx_coalesce(2, a, &l);
    nx_cpu_copy_loop(a[0].base, a[1].base, a[0].bits, &l);
  }
  if (total == 0 || updates == 0) {
    nx_done(4, a);
    CAMLreturn(Val_int(NX_OK));
  }
  scatter c = {.a = a,
               .run = run_of(combine, dt),
               .dt = dt,
               .combine = combine,
               .m = us[axis],
               .ia = a[2].dim[ri + axis],
               .ua = a[3].dim[ru + axis],
               .da = a[0].dim[ry + axis],
               .dim = (uint64_t)ys[axis]};
  /* The slices: the destination, positions and updates off the axis, as
     the loop's operands 0, 1 and 2. */
  c.l.rank = ru;
  c.l.first[0] = a[0].offset;
  c.l.first[1] = a[2].offset;
  c.l.first[2] = a[3].offset;
  for (int i = 0; i < ru; i++) {
    c.l.extent[i] = i == axis ? 1 : us[i];
    c.l.step[0][i] = a[0].dim[ry + i];
    c.l.step[1][i] = a[2].dim[ri + i];
    c.l.step[2][i] = a[3].dim[ru + i];
  }
  c.l.rank = nx_coalesce_dims(3, ru, c.l.extent, c.l.step);
  int r = c.l.rank;
  c.sd = c.l.step[0][r - 1];
  c.si = c.l.step[1][r - 1];
  c.su = c.l.step[2][r - 1];
  int64_t len = c.l.extent[r - 1], rows = updates / c.m / len;
  int64_t bytes = updates * (8 + a[3].bits / 8 + a[0].bits / 8);
  int threads = nx_cpu_threads(bytes, bytes);
  c.pieces = (len + UNIT_ROW - 1) / UNIT_ROW;
  int64_t units = rows * c.pieces;
  /* Fewer units than threads, over many updates: each thread takes a part
     of the targets and walks every update, as a 1-d scatter or an
     embedding gradient needs. Narrower runs read the updates' rows in
     pieces, which ran slower on the embedding gradient's row. */
  c.parts = 1;
  if (units < threads && c.m >= PART_UPDATES)
    c.parts = (threads + units - 1) / units;
  if ((uint64_t)c.parts > c.dim) c.parts = (int64_t)c.dim;
  int narrow_add = combine == NX_SCATTER_ADD && is_narrow(dt);
  if (narrow_add) {
    c.acc = malloc((size_t)total * sizeof(float));
    c.touched = calloc((size_t)total, 1);
    if (c.acc == NULL || c.touched == NULL) {
      free(c.acc);
      free(c.touched);
      nx_done(4, a);
      caml_raise_out_of_memory();
    }
  }
  nx_cpu_job(units * c.parts, bytes, bytes, scatter_units, &c);
  if (narrow_add) {
    encode_sums(&c, total);
    free(c.acc);
    free(c.touched);
  }
  nx_done(4, a);
  CAMLreturn(Val_int(NX_OK));
}
