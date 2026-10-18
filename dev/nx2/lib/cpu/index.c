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
   array, then applies the updates; a sorted scatter copies each bin's
   targets just before its updates. Update j's target has j's indices off
   the axis, so those indices cut the targets into disjoint slices, which
   any thread may take; within a slice the updates apply in increasing
   index along the axis, which is C order of the updates for each target.
   A unit is a run of at most UNIT_ROW slices along the innermost axis off
   the scattered one, and walks the scattered axis once, the whole run at
   each step: where the positions step 0 along the run, as an embedding
   gradient's do, the step is one row of the kind. Where runs are fewer
   than the job's threads and the axis long, as a 1-d scatter's is, each
   run's updates are sorted into bins of their targets (Sorted scatters
   below), or, where a run has several slices whose positions differ
   along it, each thread also takes a part of the targets along the axis
   and walks every update of its run, applying those that land in its
   part: the order per target stays C order.

   Add adds left to right from +0 + x. nx_add keeps the first NaN operand,
   and a zero sum is stored as +0: a sum that starts from +0 is never -0,
   and (+0 + x) + u differs from x + u only in that sign. A narrow float
   accumulates in float32 in scratch, beside a flag per target that an
   update reached, and encodes those targets once at the end, so that an
   untouched element keeps its bits. Max and Min keep the first NaN, the
   target's value first, quieted as IEEE 754 returns a NaN operand: its
   quiet bit, the top of its fraction, set, its sign and the rest of its
   payload kept. Quieting is idempotent, so a NaN target stays the first
   NaN through later updates. On narrow floats and complex numbers they
   select an operand by the order of nx_kinds.h's domains, so that the
   result is an operand's bits, quieted where a NaN. */

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
   each reading every update, or sort the updates into bins. One count
   serves every target size: float32 adds at random positions on fudge (Core
   Ultra 5 235, six P-cores, gcc 14, median of six interleaved rounds, load
   1.6-3.5), sorted against parts, in us:

     updates     into 1K targets     into 1M targets
     64K          42.6 /   44.4       59.6 /   51.5
     128K         81.4 /   88.9       96.9 /   88.7
     192K          121 /    132        136 /    127
     256K          154 /    176        174 /    168
     512K          300 /    365        339 /    327
     1M            637 /    726        689 /    656
     2M           1280 /   1515       1525 /   2100
     4M           2745 /   4620       3080 /   5000

   The sort wins into 1K targets at every count and into 1M from 2M
   updates; from 64K to 1M updates into 1M targets the parts run 3-13%
   faster. A second bound on the targets would win that back by a rule
   that follows each machine's caches. */
#define PART_UPDATES (64 * 1024)

/* Updates a sorted scatter sorts at once: about as many as the axis has
   targets, so that a target meets more than a few of them while its bin
   is in the cache, within these bounds. */
#define FEWEST_BLOCK (512 * 1024)
#define MOST_BLOCK (4 * 1024 * 1024)

/* At least two bins per thread, so that threads finish together; at most
   MOST_BINS. */
#define BINS_PER_THREAD 2
#define MOST_BINS 1024

/* Updates a chunk gathers per bin before it writes them out: writing each
   to its bin's run straight away ran several times slower, the bins' runs
   being more than the cache keeps open. */
#define GATHER 16

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
                         intnat axis) {
  if (r != ry || axis < 0 || axis >= r) return 0;
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

/* What a call takes (cpu.h, Calls, scalar): a gathered element, 51 read
   at random over 2^24 elements (M1); a scatter's update, 325 (a float16
   Max, fudge); a target its copy from into, and for a narrow float sum,
   kept in float32 then encoded (encode_sums), 40 more (fudge). */
#define GATHER_COST 128
#define UPDATE_COST 512
#define NARROW_SUM_COST 64

value nx_cpu_gather(value vaxis, value vd, value vi, value vx) {
  CAMLparam4(vaxis, vd, vi, vx);
  intnat axis = Long_val(vaxis);
  int dt = nx_array_dtype(vx);
  int64_t ys[NX_MAX_RANK], is[NX_MAX_RANK], xs[NX_MAX_RANK];
  int ry = shape_of(vd, ys), ri = shape_of(vi, is), rx = shape_of(vx, xs);
  if (!fits_off_axis(is, ri, xs, rx, axis) || !same_shape(ys, ry, is, ri))
    CAMLreturn(Val_int(NX_SHAPE));
  nx_operand in[3] = {{vd, dt, 1}, {vi, NX_INT64, 0}, {vx, dt, 0}};
  nx_array a[3];
  int e = nx_read(3, in, a);
  if (e) CAMLreturn(Val_int(e));
  int r = a[0].rank;
  int bits = nx_dtype_row_of(dt).bits;
  double elements = nx_cpu_elements(&a[1]);
  int released = nx_cpu_release(GATHER_COST * elements);
  gather g = {.a = a,
              .sa = a[2].dim[r + axis],
              .dim = (uint64_t)a[2].dim[axis],
              .w = bits < 8 ? 0 : bits / 8};
  nx_loop l = {.rank = r, .first = {a[0].offset, a[1].offset, a[2].offset}};
  for (int i = 0; i < r; i++) {
    l.extent[i] = a[1].dim[i];
    l.step[0][i] = a[0].dim[r + i];
    l.step[1][i] = a[1].dim[r + i];
    l.step[2][i] = i == axis ? 0 : a[2].dim[r + i];
  }
  l.rank = nx_coalesce_dims(3, r, l.extent, l.step);
  nx_cpu_walk(3, a, &l, GATHER_MOST, 1, gather_block, &g);
  nx_cpu_reacquire(released);
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
  /* The slices: a loop of the destination, the positions, the updates and
     into off the axis; [sd], [si], [su] and [st] their steps along its last
     axis, whose [len] slices are a row of slices. */
  nx_loop l;
  int64_t sd, si, su, st, len;
  int64_t pieces; /* runs per row of slices */
  int64_t parts;  /* parts of the axis's targets, each a unit's */
  /* Along the axis: the updates' extent, their steps through the positions
     and the updates, the destination's and into's strides, the
     destination's extent. */
  int64_t m, ia, ua, da, ta;
  uint64_t dim;
  /* A narrow float's sums, by the destination's element, and whether an
     update reached each. */
  float *acc;
  uint8_t *touched;
  /* A sorted scatter (Sorted scatters): the run's [n] slices, at [pd],
     [pi], [pu] and [pt] in the destination, the positions, the updates and
     into; the block's first step [b0], its [steps] and [chunks], of at most
     [block] steps; the [bins] of 2^[shift] targets. Chunk k's updates of
     bin q end at seg[k·bins + q] in [pos] and [vals], whose entries are
     the updates, of [w] bytes, or their rows' steps (entry_bytes), and
     start at the end of bin q - 1, or at the chunk's first. Each chunk
     gathers [gather_bytes] at [gather]. */
  int64_t pd, pi, pu, pt, n, b0, steps, chunks, bins, block, w, gather_bytes;
  int shift;
  int64_t *pos, *seg;
  uint8_t *vals, *gather;
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

static inline nx_c64 add_c64(nx_c64 a, nx_c64 b) {
  return nx_c64_of(add_f32(a.re, b.re), add_f32(a.im, b.im));
}

static inline nx_c128 add_c128(nx_c128 a, nx_c128 b) {
  return nx_c128_of(add_f64(a.re, b.re), add_f64(a.im, b.im));
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
        if (sd == 1 && su == 1)                                              \
          for (int64_t i = 0; i < n; i++) e[i] = F(e[i], v[i]);              \
        else                                                                 \
          for (int64_t i = 0; i < n; i++)                                    \
            e[i * sd] = F(e[i * sd], v[i * su]);                             \
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

/* [g] quieted where a NaN, by a mask of its NaN test: no branch. */
static inline float quiet_f32(float g) {
  uint32_t nan = (uint32_t)0 - (uint32_t)(g != g);
  return nx_bits_float(nx_float_bits(g) | (nan & ((uint32_t)1 << 22)));
}

static inline double quiet_f64(double g) {
  uint64_t nan = (uint64_t)0 - (uint64_t)(g != g);
  return nx_bits_double(nx_double_bits(g) | (nan & ((uint64_t)1 << 51)));
}

static inline nx_c64 quiet_c64(nx_c64 z) {
  return nx_c64_of(quiet_f32(z.re), quiet_f32(z.im));
}

static inline nx_c128 quiet_c128(nx_c128 z) {
  return nx_c128_of(quiet_f64(z.re), quiet_f64(z.im));
}

/* The extremes of floats and complex numbers, a NaN result quieted. */
#define QUIETED(NAME, T, F, Q) \
  static inline T NAME(T a, T b) { return Q(F(a, b)); }

QUIETED(maximum_f32, float, nx_maximum_f32, quiet_f32)
QUIETED(minimum_f32, float, nx_minimum_f32, quiet_f32)
QUIETED(maximum_f64, double, nx_maximum_f64, quiet_f64)
QUIETED(minimum_f64, double, nx_minimum_f64, quiet_f64)
QUIETED(maximum_c64, nx_c64, nx_maximum_c64, quiet_c64)
QUIETED(minimum_c64, nx_c64, nx_minimum_c64, quiet_c64)
QUIETED(maximum_c128, nx_c128, nx_maximum_c128, quiet_c128)
QUIETED(minimum_c128, nx_c128, nx_minimum_c128, quiet_c128)

#define SET(a, b) (b)
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((b) < (a) ? (b) : (a))
/* A boolean is true where its byte is not zero; its extremes are Or and
   And of truths, stored as 0 or 1. */
#define OR_B(a, b) ((uint8_t)(((a) | (b)) != 0))
#define AND_B(a, b) ((uint8_t)((a) != 0 && (b) != 0))
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
SCATTER_RUN(max_f32, float, maximum_f32)
SCATTER_RUN(min_f32, float, minimum_f32)
SCATTER_RUN(max_f64, double, maximum_f64)
SCATTER_RUN(min_f64, double, minimum_f64)
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
SCATTER_RUN(max_b, uint8_t, OR_B)
SCATTER_RUN(min_b, uint8_t, AND_B)
SCATTER_RUN(add_c64_run, nx_c64, add_c64)
SCATTER_RUN(max_c64, nx_c64, maximum_c64)
SCATTER_RUN(min_c64, nx_c64, minimum_c64)
SCATTER_RUN(add_c128_run, nx_c128, add_c128)
SCATTER_RUN(max_c128, nx_c128, maximum_c128)
SCATTER_RUN(min_c128, nx_c128, minimum_c128)

/* Elements one at a time: sub-byte dtypes and narrow floats. */

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

/* The quiet bit of a narrow float's NaNs, the top of its fraction; 0 for
   float8_e4m3fn, whose NaNs have it, and float4_e2m1fn, which has none. */
static uint32_t narrow_quiet(int dt) {
  switch (dt) {
    case NX_FLOAT16: return 0x200;
    case NX_BFLOAT16: return 0x40;
    case NX_FLOAT8_E5M2: return 0x2;
    default: return 0;
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
  uint32_t a = code_at(d, pd), b = code_at(u, pu);
  if (c->combine == NX_COMBINE_SET) {
    code_put(d, pd, b);
    return;
  }
  if (is_narrow(dt)) {
    if (c->combine == NX_COMBINE_ADD) {
      int64_t k = pd - d->offset;
      if (!c->touched[k]) {
        c->acc[k] = add_f32(0.0f, nx_bits_to_float(dt, a));
        c->touched[k] = 1;
      }
      c->acc[k] = add_f32(c->acc[k], nx_bits_to_float(dt, b));
      return;
    }
    /* The update's code where the extreme is not the element's own value,
       else the element's, quieted where a NaN. */
    float x = nx_bits_to_float(dt, a), y = nx_bits_to_float(dt, b);
    float g = c->combine == NX_COMBINE_MAX ? nx_maximum_f32(x, y)
                                        : nx_minimum_f32(x, y);
    uint32_t r = nx_float_bits(g) != nx_float_bits(x) ? b : a;
    if (g != g) r |= narrow_quiet(dt);
    if (r != a) code_put(d, pd, r);
    return;
  }
  /* int4, uint4 and bit; Add wraps modulo 16. */
  if (c->combine == NX_COMBINE_ADD) {
    code_put(d, pd, a + b);
    return;
  }
  int32_t x = sub_value(dt, a), y = sub_value(dt, b);
  int keep = c->combine == NX_COMBINE_MAX ? x >= y : x <= y;
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
  if (combine == NX_COMBINE_SET) switch (bits) {
      case 8: return set_1;
      case 16: return set_2;
      case 32: return set_4;
      case 64: return set_8;
      case 128: return set_16;
      default: return element_run;
    }
  int add = combine == NX_COMBINE_ADD, max = combine == NX_COMBINE_MAX;
#define PICK(A, MX, MN) return add ? A : max ? MX : MN
  switch (dt) {
    case NX_FLOAT32: PICK(add_f32_run, max_f32, min_f32);
    case NX_FLOAT64: PICK(add_f64_run, max_f64, min_f64);
    case NX_INT8: PICK(add_8, max_i8, min_i8);
    case NX_UINT8: PICK(add_8, max_u8, min_u8);
    case NX_BOOL: PICK(add_8, max_b, min_b);
    case NX_INT16: PICK(add_16, max_i16, min_i16);
    case NX_UINT16: PICK(add_16, max_u16, min_u16);
    case NX_INT32: PICK(add_32, max_i32, min_i32);
    case NX_UINT32: PICK(add_32, max_u32, min_u32);
    case NX_INT64: PICK(add_64, max_i64, min_i64);
    case NX_UINT64: PICK(add_64, max_u64, min_u64);
    case NX_COMPLEX64: PICK(add_c64_run, max_c64, min_c64);
    case NX_COMPLEX128: PICK(add_c128_run, max_c128, min_c128);
    default: return element_run;
  }
#undef PICK
}

/* Run [v] of slices: its positions in the destination, the positions, the
   updates and into, into [p]; its slices answered. */
static int64_t run_at(const scatter *c, int64_t v, int64_t *p) {
  const nx_loop *l = &c->l;
  int r = l->rank;
  int64_t len = l->extent[r - 1];
  int64_t i0 = v % c->pieces * UNIT_ROW;
  v /= c->pieces;
  for (int k = 0; k < 4; k++) p[k] = l->first[k] + i0 * l->step[k][r - 1];
  for (int i = r - 2; i >= 0; i--) {
    int64_t x = v % l->extent[i];
    v /= l->extent[i];
    for (int k = 0; k < 4; k++) p[k] += x * l->step[k][i];
  }
  return len - i0 < UNIT_ROW ? len - i0 : UNIT_ROW;
}

static void scatter_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const scatter *c = ctx;
  for (int64_t u = lo; u < hi; u++) {
    /* The unit's part of the axis's targets, then its run of slices. */
    int64_t part = u % c->parts, p[4];
    uint64_t first = c->dim * (uint64_t)part / (uint64_t)c->parts;
    uint64_t next = c->dim * (uint64_t)(part + 1) / (uint64_t)c->parts;
    int64_t n = run_at(c, u / c->parts, p);
    c->run(c, p[0], p[1], p[2], n, first, next - first);
  }
}

/* Sorted scatters

   Where the runs are fewer than the threads and the axis long, as a 1-d
   scatter's is or an embedding gradient's, whose positions step 0 along
   its run, a run's updates are sorted by their targets' bins of 2^shift
   positions along the axis, a block at a time. A job cuts the block into
   chunks; each chunk's thread counts its updates by bin, then copies
   their positions into its bins' runs in their order, with the updates
   where a row of slices is one slice and their rows' steps where it has
   several, its last run perhaps of one slice, dropping positions outside
   the axis. A second job takes the bins: each
   copies its targets from into, in the first block, then applies its
   chunks' runs in chunk order with the scatter's own run, so that a
   target's updates apply in C order, into targets that fit half a core's
   L2. Threads that split the targets instead each read every update: 1-d
   scatters of 16 Mi updates ran 1.2 to 1.7 times slower so on an x86-64
   of six cores, and the embedding gradient's row 1.3 times. */

/* The bytes of a sorted entry: the update where a row of slices is one
   slice, else its row's step. */
static int64_t entry_bytes(const scatter *c) { return c->len > 1 ? 8 : c->w; }

/* The first step of chunk [k] of the block. */
static int64_t chunk_first(const scatter *c, int64_t k) {
  return c->steps * k / c->chunks;
}

/* Lays out a chunk's updates from [t0] to [t1] by bin, from the bins'
   firsts at [seg], with each a value [V] of the type [T], the update or
   its row's step: each bin's gather in [gp] and [gv], GATHER at a time. */
#define PLACE(T, V)                                                          \
  {                                                                          \
    T *gv = (T *)(gp + bins * GATHER);                                       \
    for (int64_t t = t0; t < t1; t++) {                                      \
      uint64_t p = (uint64_t)x[t * ia];                                      \
      if (p >= dim) continue;                                                \
      int64_t q = (int64_t)(p >> shift), g = fill[q]++;                      \
      gp[q * GATHER + g] = (int64_t)p;                                       \
      gv[q * GATHER + g] = V;                                                \
      if (g + 1 < GATHER) continue;                                          \
      memcpy(pos + seg[q], gp + q * GATHER, sizeof(int64_t) * GATHER);       \
      memcpy((T *)vals + seg[q], gv + q * GATHER, sizeof(T) * GATHER);       \
      seg[q] += GATHER;                                                      \
      fill[q] = 0;                                                           \
    }                                                                        \
    for (int64_t q = 0; q < bins; q++) {                                     \
      memcpy(pos + seg[q], gp + q * GATHER, sizeof(int64_t) * fill[q]);      \
      memcpy((T *)vals + seg[q], gv + q * GATHER, sizeof(T) * fill[q]);      \
      seg[q] += fill[q];                                                     \
    }                                                                        \
  }

static void sort_chunks(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const scatter *c = ctx;
  int64_t bins = c->bins, ia = c->ia, ua = c->ua, b0 = c->b0;
  uint64_t dim = c->dim;
  int shift = c->shift;
  int64_t *pos = c->pos;
  uint8_t *vals = c->vals;
  const int64_t *x = (const int64_t *)c->a[2].base + c->pi + c->b0 * ia;
  const uint8_t *u = c->a[3].base + (c->pu + c->b0 * ua) * c->w;
  /* Counts of bin q at seg[q + 1], made the bins' firsts, then their
     cursors: once laid out, seg[q] is bin q's end. They stay on this
     thread's stack: chunks' counters sharing a cache line made the sort
     several times slower. */
  int64_t seg[MOST_BINS + 1], fill[MOST_BINS];
  for (int64_t k = lo; k < hi; k++) {
    int64_t t0 = chunk_first(c, k), t1 = chunk_first(c, k + 1);
    int64_t *gp = (int64_t *)(c->gather + k * c->gather_bytes);
    memset(seg, 0, (size_t)(bins + 1) * sizeof *seg);
    memset(fill, 0, (size_t)bins * sizeof *fill);
    for (int64_t t = t0; t < t1; t++) {
      uint64_t p = (uint64_t)x[t * ia];
      if (p < dim) seg[(p >> shift) + 1]++;
    }
    seg[0] = t0;
    for (int64_t q = 0; q < bins; q++) seg[q + 1] += seg[q];
#define AT(T) ((const T *)u)[t * ua]
    if (c->len > 1) PLACE(int64_t, b0 + t)
    else switch (c->w) {
        case 1: PLACE(uint8_t, AT(uint8_t)) break;
        case 2: PLACE(uint16_t, AT(uint16_t)) break;
        case 4: PLACE(uint32_t, AT(uint32_t)) break;
        case 8: PLACE(uint64_t, AT(uint64_t)) break;
        default: PLACE(u128, AT(u128))
      }
#undef AT
    memcpy(c->seg + k * bins, seg, (size_t)bins * sizeof *seg);
  }
}

#undef PLACE

/* Copies the run's targets of the bin from [first] from into, unless
   into is the destination. */
static void copy_bin(const scatter *c, uint64_t first) {
  if (c->a[1].alias) return;
  uint64_t end = first + ((uint64_t)1 << c->shift);
  if (end > c->dim) end = c->dim;
  nx_loop l = {.rank = 2,
               .extent = {(int64_t)(end - first), c->n},
               .first = {c->pd + (int64_t)first * c->da,
                         c->pt + (int64_t)first * c->ta}};
  l.step[0][0] = c->da;
  l.step[0][1] = c->sd;
  l.step[1][0] = c->ta;
  l.step[1][1] = c->st;
  l.rank = nx_coalesce_dims(2, 2, l.extent, l.step);
  nx_cpu_copy_loop(c->a[0].base, c->a[1].base, c->a[0].bits, &l);
}

/* Applies each bin's runs with the scatter's run, over the sorted
   positions and updates. */
static void apply_bins(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const scatter *c = ctx;
  /* The run reads the destination's base, width and offset, and the
     positions' and updates' bases and widths. */
  const nx_array *d = &c->a[0];
  nx_array a[4] = {{.base = d->base, .bits = d->bits, .offset = d->offset},
                   {.bits = d->bits},
                   {.bits = 64},
                   {.base = c->a[3].base, .bits = c->a[3].bits}};
  scatter s = *c;
  s.a = a;
  s.ia = s.ua = 1;
  uint64_t span = (uint64_t)1 << c->shift;
  for (int64_t q = lo; q < hi; q++) {
    uint64_t first = (uint64_t)q << c->shift;
    if (c->b0 == 0) copy_bin(c, first);
    for (int64_t k = 0; k < c->chunks; k++) {
      const int64_t *seg = c->seg + k * c->bins;
      int64_t from = q == 0 ? chunk_first(c, k) : seg[q - 1];
      s.m = seg[q] - from;
      if (s.m == 0) continue;
      a[2].base = (uint8_t *)(c->pos + from);
      if (c->len == 1) {
        a[3].base = c->vals + from * c->w;
        s.run(&s, c->pd, 0, 0, 1, first, span);
        continue;
      }
      /* A row of updates at a time, from its step, the next row's bytes
         fetched ahead where they are contiguous: rows read out of order
         stall on memory otherwise. */
      const int64_t *t = (const int64_t *)c->vals + from;
      int64_t m = s.m, bytes = c->su == 1 ? c->n * c->w : 0;
      s.m = 1;
      for (int64_t j = 0; j < m; j++) {
        if (j + 1 < m)
          for (int64_t b = 0; b < bytes; b += 64)
            __builtin_prefetch(c->a[3].base +
                               (c->pu + t[j + 1] * c->ua) * c->w + b);
        s.run(&s, c->pd, j, c->pu + t[j] * c->ua, c->n, first, span);
      }
    }
  }
}

/* The shift of the bins: as many targets as fit half a core's L2, the
   share the contraction's blocks take, fewer while the threads would have
   under BINS_PER_THREAD bins each, more past MOST_BINS bins. */
static int bin_shift(uint64_t dim, int64_t w, int threads) {
  int64_t most = nx_cpu_table->l2 / 2;
  int s = 0;
  while (s < 62 && ((int64_t)2 << s) * w <= most) s++;
  while (s > 0 && ((dim - 1) >> s) + 1 < (uint64_t)threads * BINS_PER_THREAD)
    s--;
  while (s < 62 && ((dim - 1) >> s) + 1 > MOST_BINS) s++;
  return s;
}

/* Scatters run [v] by sorted blocks. */
static void sorted(scatter *c, int64_t v, int threads) {
  int64_t p[4];
  c->n = run_at(c, v, p);
  c->pd = p[0];
  c->pi = p[1];
  c->pu = p[2];
  c->pt = p[3];
  for (c->b0 = 0; c->b0 < c->m; c->b0 += c->block) {
    c->steps = c->m - c->b0 < c->block ? c->m - c->b0 : c->block;
    c->chunks = c->steps < threads ? c->steps : threads;
    int64_t e = entry_bytes(c);
    int64_t bytes = c->steps * 2 * (8 + e);
    nx_cpu_job(c->chunks, bytes, (double)bytes, sort_chunks, c);
    bytes = c->steps * (8 + e + 3 * c->n * c->w);
    nx_cpu_job(c->bins, bytes, (double)bytes, apply_bins, c);
  }
}

/* Stores each narrow float sum into its target. */
static void encode_sums(const scatter *c, int64_t total) {
  const nx_array *d = &c->a[0];
  for (int64_t k = 0; k < total; k++)
    if (c->touched[k])
      code_put(d, d->offset + k, narrow_of_float(c->dt, c->acc[k]));
}

/* Fills the loop of slices from the shape [us] of rank [r], off [axis],
   with into's strides [ts], and answers its units. Only a sorted
   scatter's copy_bin reads into's steps; other scatters pass zeros, which
   never part an axis. */
static int64_t slices(scatter *c, const int64_t *us, int r, intnat axis,
                      const int64_t *ts) {
  const nx_array *a = c->a;
  nx_loop *l = &c->l;
  l->first[0] = a[0].offset;
  l->first[1] = a[2].offset;
  l->first[2] = a[3].offset;
  l->first[3] = a[1].offset;
  for (int i = 0; i < r; i++) {
    l->extent[i] = i == axis ? 1 : us[i];
    l->step[0][i] = a[0].dim[r + i];
    l->step[1][i] = a[2].dim[r + i];
    l->step[2][i] = a[3].dim[r + i];
    l->step[3][i] = ts[i];
  }
  l->rank = nx_coalesce_dims(4, r, l->extent, l->step);
  int lr = l->rank;
  c->sd = l->step[0][lr - 1];
  c->si = l->step[1][lr - 1];
  c->su = l->step[2][lr - 1];
  c->st = l->step[3][lr - 1];
  c->len = l->extent[lr - 1];
  c->pieces = (c->len + UNIT_ROW - 1) / UNIT_ROW;
  int64_t rows = 1;
  for (int i = 0; i < lr - 1; i++) rows *= l->extent[i];
  return rows * c->pieces;
}

/* Copies into to the destination, unless the two are one array. */
static void copy_into(const nx_array *a, int64_t total) {
  if (total == 0 || a[1].alias) return;
  nx_loop l;
  nx_coalesce(2, a, &l);
  nx_cpu_copy_loop(a[0].base, a[1].base, a[0].bits, &l);
}

value nx_cpu_scatter(value vcombine, value vaxis, value vd, value vinto,
                     value vi, value vu) {
  CAMLparam5(vcombine, vaxis, vd, vinto, vi);
  CAMLxparam1(vu);
  int combine = Int_val(vcombine);
  intnat axis = Long_val(vaxis);
  int dt = nx_array_dtype(vinto);
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
  /* One rank: the destination's, into's, the positions' and the
     updates'. */
  int r = a[0].rank;
  int64_t total = 1, updates = 1;
  for (int i = 0; i < r; i++) {
    total *= ys[i];
    updates *= us[i];
  }
  int narrow_add = combine == NX_COMBINE_ADD && is_narrow(dt);
  double target = 2 * NX_CPU_STRIDED_COST * nx_cpu_width(dt) +
                  (narrow_add ? NARROW_SUM_COST : 0);
  int released = nx_cpu_release((double)total * target +
                                (double)updates * UPDATE_COST);
  if (total == 0 || updates == 0) {
    copy_into(a, total);
    nx_cpu_reacquire(released);
    nx_done(4, a);
    CAMLreturn(Val_int(NX_OK));
  }
  scatter c = {.a = a,
               .run = run_of(combine, dt),
               .dt = dt,
               .combine = combine,
               .m = us[axis],
               .ia = a[2].dim[r + axis],
               .ua = a[3].dim[r + axis],
               .da = a[0].dim[r + axis],
               .ta = a[1].dim[r + axis],
               .dim = (uint64_t)ys[axis]};
  static const int64_t unread[NX_MAX_RANK];
  int64_t units = slices(&c, us, r, axis, unread);
  int64_t bytes = updates * (8 + a[3].bits / 8 + a[0].bits / 8);
  int threads = nx_cpu_threads(bytes, bytes);
  /* Fewer units than threads, over many updates. Rows of one slice, or of
     slices whose positions step 0 along them, as a 1-d scatter's or an
     embedding gradient's, sort their updates; a dtype of under a byte
     shares its bytes across bins, so doesn't. Otherwise each thread takes
     a part of the targets and walks every update of its run. */
  c.parts = 1;
  int few = units < threads && c.m >= PART_UPDATES;
  int sort = few && (c.len == 1 || c.si == 0) && a[0].bits >= 8;
  if (few && !sort) c.parts = (threads + units - 1) / units;
  /* Into's steps may part the loop's axes further; a row of one slice, or
     of positions stepping 0, stays one. */
  if (sort) units = slices(&c, us, r, axis, a[1].dim + r);
  if ((uint64_t)c.parts > c.dim) c.parts = (int64_t)c.dim;
  if (!sort) copy_into(a, total);
  if (narrow_add) {
    c.acc = malloc((size_t)total * sizeof(float));
    c.touched = calloc((size_t)total, 1);
  }
  if (sort) {
    c.w = a[0].bits / 8;
    int64_t width = c.len < UNIT_ROW ? c.len : UNIT_ROW;
    c.shift = bin_shift(c.dim, width * c.w, threads);
    c.bins = (int64_t)((c.dim - 1) >> c.shift) + 1;
    c.block = (int64_t)c.dim < FEWEST_BLOCK ? FEWEST_BLOCK
              : (int64_t)c.dim > MOST_BLOCK ? MOST_BLOCK
                                            : (int64_t)c.dim;
    int64_t most = c.m < c.block ? c.m : c.block, e = entry_bytes(&c);
    c.gather_bytes = c.bins * GATHER * (8 + e);
    c.pos = malloc((size_t)most * sizeof(int64_t));
    c.vals = malloc((size_t)(most * e));
    c.seg = malloc((size_t)(threads * c.bins) * sizeof(int64_t));
    c.gather = malloc((size_t)(threads * c.gather_bytes));
  }
  if ((narrow_add && (c.acc == NULL || c.touched == NULL)) ||
      (sort && (c.pos == NULL || c.vals == NULL || c.seg == NULL ||
                c.gather == NULL))) {
    free(c.acc);
    free(c.touched);
    free(c.pos);
    free(c.vals);
    free(c.seg);
    free(c.gather);
    nx_cpu_reacquire(released);
    nx_done(4, a);
    caml_raise_out_of_memory();
  }
  if (sort)
    for (int64_t v = 0; v < units; v++) sorted(&c, v, threads);
  else
    nx_cpu_job(units * c.parts, bytes, bytes, scatter_units, &c);
  if (narrow_add) encode_sums(&c, total);
  free(c.acc);
  free(c.touched);
  free(c.pos);
  free(c.vals);
  free(c.seg);
  free(c.gather);
  nx_cpu_reacquire(released);
  nx_done(4, a);
  CAMLreturn(Val_int(NX_OK));
}

value nx_cpu_scatter_byte(value *argv, int argn) {
  (void)argn;
  return nx_cpu_scatter(argv[0], argv[1], argv[2], argv[3], argv[4],
                        argv[5]);
}
