/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Assemblies, and folds: the adjoint of a padded, windowed load.

   An assembly fills what no piece covers, then copies each piece into its
   region in order, so the last piece holding an element wins. Pieces of
   unit steps whose regions are disjoint and fill the result, as a
   concatenation's do, leave nothing to fill. One such piece inside the
   result, as a pad's, is one job over the result's rows, each row filled
   before the piece, copied, and filled after it, so that every element is
   written once; of a sub-byte dtype, whose rows may share bytes, it fills
   the 2·rank boxes around the piece, then copies it. Any other set fills
   the whole result first. A first piece that is the destination over the
   whole result is already in place.

   A fold's result starts at -0, the sum of no term with no +0 in it: a
   float's -0 + x is x, so each element is the sum of its taps alone, a
   single -0 tap included. Then each tap, the indices along the window axes
   in C order, adds the operand's elements of that tap into the result:
   within a tap each result element receives at most one, so every element
   adds its taps in C order, as nx_cpu.mli states. An element no tap
   reaches is set to +0 last. Taps are a product of each axis's spans, so
   an element no tap reaches is one whose index along some axis no span
   holds: those slices are set, axis by axis. Along an
   axis, the operand's indices whose padded coordinate w·step + j·dilation
   lands on an element, lo + i·(interior + 1) with 0 <= i < d, step by a
   constant and so do the i they land on: a tap is one box of the operand
   added into one box of the result, rows by rows.c's add. An axis no
   window takes is one of step 1 and tap 0, and its span is every tap's:
   the leading such axes, as a convolution's batch and channels, cut the
   boxes into slabs of disjoint results that threads take whole, each
   starting its slab at -0, then adding its taps in order while it holds
   the operand's slab in cache. Where the leading axes' indices miss some
   slab, as padding along them does, the whole result starts at -0 first.

   An operand whose taps lie innermost, as a convolution's [n; c; h; w; kh;
   kw] does, gives each tap's rows a step of the taps' count: adding them
   ran at about three cycles an element on one fudge core. On a target
   that copies strided rows (cpu.h's NX_CPU_COPY_STRIDED), where the
   slab fits half the L2, a unit first copies it into scratch with the
   taps outermost, through the copy's transposed tiles, and adds each
   tap's rows from there in unit steps. A narrow float's or sub-byte
   dtype's rows, which the stage gathers into slots anyway, do not. */

#include <stdlib.h>
#include <string.h>

#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"

/* Assemblies */

/* A fill of the element of bits [fill] into [dst]. */
typedef struct {
  const nx_array *dst;
  const uint8_t *fill;
} filling;

/* A fill's block. */
static void fill_block(const nx_cpu_block *b, void *ctx) {
  const filling *f = ctx;
  nx_cpu_fill_block(f->dst, b, f->fill);
}

/* Stores the element of bits [fill] into the box of [dst] from [first],
   of [r] axes of [extent] stepping [step] elements, on the job's threads
   as a fill's walk does. */
static void fill_box(const nx_array *dst, const uint8_t *fill, int r,
                     const int64_t *extent, const int64_t *step,
                     int64_t first) {
  nx_loop l = {.first = {first}};
  for (int i = 0; i < r; i++) {
    if (extent[i] == 0) return;
    l.extent[i] = extent[i];
    l.step[0][i] = step[i];
  }
  l.rank = nx_coalesce_dims(1, r, l.extent, l.step);
  int w = dst->bits < 8 ? 1 : dst->bits / 8;
  filling f = {dst, fill};
  nx_cpu_walk(1, dst, &l, NX_CPU_SLOT / w, w, fill_block, &f);
}

/* Elements of a pad's unit: a longer row is cut into parts of this many,
   so that a long 1-d pad shares the job's threads. It is copy.c's
   COPY_MOST, for its reason: at 64 Ki a unit's call costs next to nothing
   beside its copy, and a job still has units to share. */
#define PAD_PART (64 * 1024)

/* What a call takes past its bytes (cpu.h, Calls, scalar): a piece's
   copy loop, 6.1 KiB for a piece of one element (M1); a pair of pieces'
   test along an axis (tiles), 79 a pair of 2048 one-element pieces
   (fudge); a span's step over a position of a tap (span_of), 313 for a
   fold of one axis, its sums included (fudge). */
#define PIECE_COST 16384
#define PAIR_COST 128
#define SPAN_COST 512

/* A pad: the result and its one piece, of whole bytes, the piece's box
   [lo, hi) per axis of the result's [shape], their steps [sd] and [sp];
   rows along the last axis, [parts] units each. */
typedef struct {
  const nx_array *a;
  filling f;
  int r;
  int64_t parts;
  const int64_t *shape, *lo, *hi, *sd, *sp;
} padding;

/* Fills [n] elements of the result's row from position [pd]. */
static void fill_run(const padding *p, int64_t pd, int64_t n) {
  if (n <= 0) return;
  nx_cpu_block b = {.n0 = n, .n1 = 1, .n2 = 1, .at = {pd},
                    .s0 = {p->sd[p->r - 1]}};
  nx_cpu_fill_block(p->f.dst, &b, p->f.fill);
}

static void pad_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const padding *p = ctx;
  int last = p->r - 1;
  int64_t n = p->shape[last], sd = p->sd[last], sp = p->sp[last];
  for (int64_t u = lo; u < hi; u++) {
    int64_t row = u / p->parts, c0 = u % p->parts * PAD_PART;
    int64_t c1 = c0 + PAD_PART < n ? c0 + PAD_PART : n;
    /* The row's positions in the result and the piece, and whether the
       piece holds it. */
    int64_t pd = p->a[0].offset, pp = p->a[1].offset;
    int inside = 1;
    for (int i = last - 1; i >= 0; i--) {
      int64_t x = row % p->shape[i];
      row /= p->shape[i];
      pd += x * p->sd[i];
      pp += (x - p->lo[i]) * p->sp[i];
      inside &= x >= p->lo[i] && x < p->hi[i];
    }
    /* The piece's columns of the part: none outside the piece. */
    int64_t x0 = inside ? p->lo[last] : c1, x1 = inside ? p->hi[last] : c1;
    x0 = x0 < c0 ? c0 : x0 > c1 ? c1 : x0;
    x1 = x1 < x0 ? x0 : x1 > c1 ? c1 : x1;
    fill_run(p, pd + c0 * sd, x0 - c0);
    if (x1 > x0)
      nx_copy_box(p->a[0].base, p->a[1].base,
                  &(nx_box){{1, 1, x1 - x0},
                            {pd + x0 * sd, pp + (x0 - p->lo[last]) * sp},
                            {{0, 0, sd}, {0, 0, sp}}},
                  p->a[0].bits);
    fill_run(p, pd + x1 * sd, c1 - x1);
  }
}

/* The regions of unit steps, as [lo, hi) per axis, in [lo] and [hi]: 0 if
   one steps otherwise. */
static int intervals(int np, int r, const int64_t *ranges, int64_t (*lo)[NX_MAX_RANK],
                     int64_t (*hi)[NX_MAX_RANK]) {
  for (int j = 0; j < np; j++)
    for (int i = 0; i < r; i++) {
      const int64_t *x = ranges + 3 * (j * r + i);
      if (x[2] != 1 && x[1] > 1) return 0;
      lo[j][i] = x[0];
      hi[j][i] = x[0] + x[1];
    }
  return 1;
}

/* Whether the boxes [lo, hi) of the [np] pieces are disjoint and fill the
   result of [total] elements. */
static int tiles(int np, int r, int64_t (*lo)[NX_MAX_RANK],
                 int64_t (*hi)[NX_MAX_RANK], int64_t total) {
  int64_t sum = 0;
  for (int j = 0; j < np; j++) {
    int64_t n = 1;
    for (int i = 0; i < r; i++) n *= hi[j][i] - lo[j][i];
    sum += n;
    for (int k = 0; k < j; k++) {
      int apart = 0;
      for (int i = 0; i < r && !apart; i++)
        apart = hi[j][i] <= lo[k][i] || hi[k][i] <= lo[j][i] ||
                hi[j][i] == lo[j][i];
      if (!apart) return 0;
    }
  }
  return sum == total;
}

/* The pieces' descriptors, boxes and ranges: one C-heap block of their
   count. A custom block owns it, so that an exception the door raises
   leaves it to the finalizer. */
typedef struct {
  nx_array *a;
  int64_t (*lo)[NX_MAX_RANK], (*hi)[NX_MAX_RANK];
  int64_t *ranges;
} pieces;

static void free_scratch(value v) {
  free(*(void **)Data_custom_val(v));
  *(void **)Data_custom_val(v) = NULL;
}

/* Frees the scratch [v] owns, that of [many] pieces with the domain lock
   released (nx_free_released). */
static void drop_scratch(value v, int many) {
  void *b = *(void **)Data_custom_val(v);
  *(void **)Data_custom_val(v) = NULL;
  if (many) nx_free_released(b);
  else free(b);
}

static struct custom_operations scratch_ops = {
    "nx.cpu.assemble",          free_scratch,
    custom_compare_default,     custom_hash_default,
    custom_serialize_default,   custom_deserialize_default,
    custom_compare_ext_default, custom_fixed_length_default};

/* Carves [p] for [np] pieces of rank [r] out of a block that [owner] then
   owns, allocated with the domain lock released for [many] pieces
   (nx_alloc_released); 0 if host memory runs out. Each part is a multiple
   of 8 bytes, so each starts aligned. */
static int pieces_of(value owner, int np, int r, int many, pieces *p) {
  size_t n = (size_t)np + 1;
  size_t sa = n * sizeof *p->a, sb = n * sizeof *p->lo,
         sr = (3 * (size_t)r * (size_t)np + 1) * sizeof *p->ranges;
  size_t bytes = sa + 2 * sb + sr;
  uint8_t *b = many ? nx_alloc_released(bytes) : malloc(bytes);
  if (b == NULL) return 0;
  *(void **)Data_custom_val(owner) = b;
  p->a = (nx_array *)b;
  p->lo = (int64_t(*)[NX_MAX_RANK])(b + sa);
  p->hi = (int64_t(*)[NX_MAX_RANK])(b + sa + sb);
  p->ranges = (int64_t *)(b + sa + 2 * sb);
  return 1;
}

/* Whether the range [x], a Move.range's start, count and step, keeps
   elements of an axis of extent [d], as Move.Slice takes it. */
static int keeps(const int64_t *x, int64_t d) {
  int64_t start = x[0], count = x[1], step = x[2], last;
  if (step == 0 || count < 0) return 0;
  if (count == 0) return 1;
  if (__builtin_mul_overflow(count - 1, step, &last) ||
      __builtin_add_overflow(start, last, &last))
    return 0;
  return start >= 0 && start < d && last >= 0 && last < d;
}

/* The operands of an assembly as the door reads them, the destination
   then each piece, from the roots [vd] and [vpieces]: each piece is read
   once, its region into [ranges] and checked there, as another domain may
   write [vpieces]. [elements] counts the pieces', for the price. */
typedef struct {
  const value *vd, *vpieces;
  int r, dt;
  const int64_t *shape;
  int64_t *ranges, elements;
} reading;

static int operand(void *ctx, int k, nx_operand *in) {
  reading *c = ctx;
  if (k == 0) {
    *in = (nx_operand){*c->vd, c->dt, 1};
    return NX_OK;
  }
  int j = k - 1, r = c->r;
  value piece = Field(*c->vpieces, j), region = Field(piece, 0);
  int64_t ps[2 * NX_MAX_RANK], off, n = 1;
  *in = (nx_operand){Field(piece, 1), c->dt, 0};
  if ((int)Wosize_val(region) != r || nx_array_layout(in->array, ps, &off) != r)
    return NX_SHAPE;
  for (int i = 0; i < r; i++) {
    int64_t *x = c->ranges + 3 * (j * r + i);
    for (int f = 0; f < 3; f++) x[f] = Long_val(Field(Field(region, i), f));
    if (!keeps(x, c->shape[i]) || ps[i] != x[1]) return NX_SHAPE;
    n *= x[1];
  }
  c->elements += n;
  return NX_OK;
}

/* The external. [vfill] is the fill's bits and [vpieces] the pieces, each a
   region, a Move.range per axis, and its array, which the door reads with
   the regions (operand). */
value nx_cpu_assemble(value vfill, value vd, value vpieces) {
  CAMLparam3(vfill, vd, vpieces);
  CAMLlocal1(owner);
  int np = (int)Wosize_val(vpieces);
  int64_t shape[2 * NX_MAX_RANK], off;
  int r = nx_array_layout(vd, shape, &off);
  owner = caml_alloc_custom(&scratch_ops, sizeof(void *), 0, 1);
  *(void **)Data_custom_val(owner) = NULL;
  int n = 1 + np, many = n > NX_MAX_OPERANDS;
  pieces p;
  if (!pieces_of(owner, np, r, many, &p)) caml_raise_out_of_memory();
  uint8_t fill[16] = {0};
  size_t nfill = caml_string_length(vfill);
  memcpy(fill, String_val(vfill), nfill < sizeof fill ? nfill : sizeof fill);
  int dt = nx_array_dtype(vd), bits = nx_dtype_row_of(dt).bits, e;
  int64_t *ranges = p.ranges;
  nx_array *a = p.a;
  reading c = {&vd, &vpieces, r, dt, shape, ranges, 0};
  e = nx_read_each(n, operand, &c, a);
  if (e) {
    drop_scratch(owner, many);
    CAMLreturn(Val_int(e));
  }
  double pairs = (double)np * (np - 1) / 2 * r;
  double cost = NX_CPU_STRIDED_COST * nx_cpu_width(dt) *
                    (nx_cpu_elements(&a[0]) + (double)c.elements) +
                PAIR_COST * pairs + PIECE_COST * (double)np;
  int released = nx_cpu_release(cost);
  int64_t total = 1;
  for (int i = 0; i < r; i++) total *= shape[i];
  const int64_t *step = a[0].dim + r;
  int64_t(*lo)[NX_MAX_RANK] = p.lo, (*hi)[NX_MAX_RANK] = p.hi;
  int unit = intervals(np, r, ranges, lo, hi);
  /* A piece the door found identical to the destination is in place only
     as the first, over the whole result in order; elsewhere the kernel
     would read it at other indices than it writes. */
  for (int j = 0; j < np && !e; j++) {
    if (!a[1 + j].alias) continue;
    int whole = j == 0;
    for (int i = 0; i < r && whole; i++) {
      const int64_t *x = ranges + 3 * i;
      whole = x[0] == 0 && x[1] == shape[i] && (x[2] == 1 || x[1] <= 1);
    }
    if (!whole) e = NX_OVERLAP;
  }
  int in_place = np > 0 && a[1].alias;
  int covered = in_place || (unit && tiles(np, r, lo, hi, total));
  /* A sub-byte dtype's rows may share bytes: its pad keeps the boxes. */
  int padded = !e && total > 0 && !covered && unit && np == 1 && r > 0 &&
               bits >= 8;
  if (padded) {
    padding pd = {.a = a,
                  .f = {&a[0], fill},
                  .r = r,
                  .parts = (shape[r - 1] + PAD_PART - 1) / PAD_PART,
                  .shape = shape,
                  .lo = lo[0],
                  .hi = hi[0],
                  .sd = step,
                  .sp = a[1].dim + r};
    int64_t bytes = 2 * total * (bits / 8);
    nx_cpu_job(total / shape[r - 1] * pd.parts, bytes, bytes, pad_units, &pd);
  }
  if (!e && total > 0 && !covered && !padded) {
    if (unit && np == 1) {
      /* The 2·rank boxes around the piece: before and after it along axis
         i, within it along the axes before i, whole along those after. */
      for (int i = 0; i < r; i++) {
        int64_t extent[NX_MAX_RANK], first = a[0].offset;
        for (int k = 0; k < r; k++) {
          extent[k] = k < i ? hi[0][k] - lo[0][k] : shape[k];
          if (k < i) first += lo[0][k] * step[k];
        }
        extent[i] = lo[0][i];
        fill_box(&a[0], fill, r, extent, step, first);
        extent[i] = shape[i] - hi[0][i];
        fill_box(&a[0], fill, r, extent, step, first + hi[0][i] * step[i]);
      }
    } else
      fill_box(&a[0], fill, r, shape, step, a[0].offset);
  }
  for (int j = 0; j < np && !e && !padded; j++) {
    if (a[1 + j].alias) continue;
    nx_loop l = {.rank = r, .first = {a[0].offset, a[1 + j].offset}};
    int64_t n = 1;
    for (int i = 0; i < r; i++) {
      const int64_t *x = ranges + 3 * (j * r + i);
      l.extent[i] = x[1];
      l.first[0] += x[0] * step[i];
      l.step[0][i] = x[2] * step[i];
      l.step[1][i] = a[1 + j].dim[r + i];
      n *= x[1];
    }
    if (n == 0) continue;
    l.rank = nx_coalesce_dims(2, r, l.extent, l.step);
    nx_cpu_copy_loop(a[0].base, a[1 + j].base, bits, &l, nx_cpu_width(dt));
  }
  nx_cpu_reacquire(released);
  nx_done(n, a);
  drop_scratch(owner, many);
  CAMLreturn(Val_int(e));
}

/* Folds */

/* A padding, Nx_kernel.pad, copied out of OCaml: [lo], [hi] and
   [interior] by axis of its [rank], then by window its axis, size, step
   and dilation. */
typedef struct {
  int rank, nwindows;
  int64_t lo[NX_MAX_RANK], hi[NX_MAX_RANK], interior[NX_MAX_RANK];
  int64_t window[NX_MAX_RANK][4];
} load_pad;

/* Copies the Nx_kernel.pad [v] into [p] and answers NX_OK, or NX_SHAPE
   where [lo], [hi] and [interior] differ in length or the rank and the
   windows number more than NX_MAX_RANK. Reads [v] alone and allocates
   nothing. */
static int pad_of(value v, load_pad *p) {
  value lo = Field(v, 0), hi = Field(v, 1), inner = Field(v, 2);
  value windows = Field(v, 3);
  mlsize_t r = Wosize_val(lo), nw = Wosize_val(windows);
  if (Wosize_val(hi) != r || Wosize_val(inner) != r || r + nw > NX_MAX_RANK)
    return NX_SHAPE;
  p->rank = (int)r;
  p->nwindows = (int)nw;
  for (mlsize_t i = 0; i < r; i++) {
    p->lo[i] = Long_val(Field(lo, i));
    p->hi[i] = Long_val(Field(hi, i));
    p->interior[i] = Long_val(Field(inner, i));
  }
  /* A Move.window: axis, size, step, dilation. */
  for (mlsize_t k = 0; k < nw; k++)
    for (int f = 0; f < 4; f++)
      p->window[k][f] = Long_val(Field(Field(windows, k), f));
  return NX_OK;
}

/* The number of windows of [w] (axis, size, step, dilation) along an axis
   of extent [e], as Move.Window counts them: none where the dilated window
   is longer than the axis. The reach is compared through a division, so no
   product overflows. */
static int64_t nx_cpu_windows(int64_t e, const int64_t *w) {
  if (e < 1 || w[1] - 1 > (e - 1) / w[3]) return 0;
  return (e - 1 - w[3] * (w[1] - 1)) / w[2] + 1;
}

/* Fills [y] with the shape an operand of shape [x] and rank [r] has once
   padded by [p], its windows' axes last, as Nx_kernel.Shape.padded
   computes it. Answers NX_OK, or NX_SHAPE for a pad it cannot apply: [r]
   other than [p]'s rank, an extent below zero or past int64, or a window
   whose size, step or dilation is below one. */
static int padded_shape(const load_pad *p, int r, const int64_t *x,
                        int64_t *y) {
  if (p->rank != r) return NX_SHAPE;
  for (int i = 0; i < r; i++) {
    int64_t d = x[i], gaps = 0, e;
    if (p->interior[i] < 0) return NX_SHAPE;
    if (d > 0 && __builtin_mul_overflow(p->interior[i], d - 1, &gaps))
      return NX_SHAPE;
    if (__builtin_add_overflow(p->lo[i], p->hi[i], &e) ||
        __builtin_add_overflow(e, d, &e) || __builtin_add_overflow(e, gaps, &e))
      return NX_SHAPE;
    if (e < 0) return NX_SHAPE;
    y[i] = e;
  }
  for (int k = 0; k < p->nwindows; k++) {
    const int64_t *w = p->window[k]; /* axis, size, step, dilation */
    if (w[0] < 0 || w[0] >= r || (k > 0 && w[0] <= p->window[k - 1][0]))
      return NX_SHAPE;
    if (w[1] < 1 || w[2] < 1 || w[3] < 1) return NX_SHAPE;
    y[w[0]] = nx_cpu_windows(y[w[0]], w);
    y[r + k] = w[1];
  }
  return NX_OK;
}


/* Zeroes the [n] elements of [a] from position [p], contiguous: a
   sub-byte dtype's as a run (nx_array.h), whose end bytes other units'
   runs may share. */
static void zero_run(const nx_array *a, int64_t p, int64_t n) {
  if (a->bits < 8) nx_sub_fill_run(a->base, a->bits, p, n, 0);
  else memset(a->base + p * (a->bits / 8), 0, (size_t)(n * (a->bits / 8)));
}

/* Stores the element of bits [start] into the [n] elements of [a] from
   position [p], contiguous, as zero_run does. */
static void start_run(const nx_array *a, const uint8_t *start, int64_t p,
                      int64_t n) {
  nx_cpu_block b = {.n0 = n, .n1 = 1, .n2 = 1, .at = {p}, .s0 = {1}};
  nx_cpu_fill_block(a, &b, start);
}

/* The bits of -0 of [dt], a float's or each part of a complex number's
   sign alone, into [start]; 0 for another dtype. Answers whether they are
   -0's. */
static int negative_zero(int dt, uint8_t *start) {
  memset(start, 0, 16);
  int bits = nx_dtype_row_of(dt).bits, kind = nx_dtype_row_of(dt).kind;
  if (kind == NX_KIND_FLOAT) {
    if (bits < 8) start[0] = (uint8_t)(1u << (bits - 1));
    else start[bits / 8 - 1] = 0x80;
    return 1;
  }
  if (kind == NX_KIND_COMPLEX) {
    start[bits / 16 - 1] = 0x80;
    start[bits / 8 - 1] = 0x80;
    return 1;
  }
  return 0;
}

/* An axis of a tap's box: [count] operand indices from [x0] stepping [xs],
   landing on result indices from [d0] stepping [ds]. */
typedef struct {
  int64_t count, x0, xs, d0, ds;
} span;

/* The span of the operand's indices w in [0, n) along an axis of the
   result's extent [d] that land on it at tap [j]: w·step + j·dilation =
   lo + i·s. */
static span span_of(int64_t n, int64_t step, int64_t j, int64_t dilation,
                    int64_t lo, int64_t s, int64_t d) {
  span a = {0, 0, 0, 0, 0};
  for (int64_t w = 0; w < n; w++) {
    int64_t q = w * step + j * dilation - lo;
    if (q < 0 || q % s != 0 || q / s >= d) continue;
    if (a.count == 0) {
      a.x0 = w;
      a.d0 = q / s;
    } else if (a.count == 1) {
      a.xs = w - a.x0;
      a.ds = q / s - a.d0;
    }
    a.count++;
  }
  return a;
}

/* The bytes of a narrow row's piece in its carrier, each operand's: small,
   since the slots live on the stack of every padded load's copy. */
#define ADD_SLOT 2048

/* Adds the [n] elements of operand 1 from position [x], [sx] apart, into
   those of operand 0 from [d], [sd] apart, by [add], the row of their
   carrier. A narrow float or a sub-byte dtype adds through slots of its
   carrier, a piece of ADD_SLOT bytes at a time, each sum rounded to it on
   the store, as Prog.Add rounds; the stage's ends of a sub-byte row are
   atomic, so units whose rows share a byte may add at once. */
static void add_row(const nx_cpu_row2 add, const nx_array *a, int w,
                    int64_t n, int64_t d, int64_t sd, int64_t x, int64_t sx) {
  int c = nx_cpu_carrier(a[0].dtype);
  if (c == a[0].dtype) {
    uint8_t *dp = a[0].base + d * w;
    add(n, dp, sd, dp, sd, a[1].base + x * w, sx);
    return;
  }
  _Alignas(64) uint8_t ds[ADD_SLOT], xs[ADD_SLOT];
  int64_t cw = nx_cpu_width(c), most = ADD_SLOT / cw;
  nx_cpu_block b = {.n1 = 1, .n2 = 1};
  b.s0[0] = sd;
  b.s0[1] = sx;
  for (int64_t i = 0; i < n; i += most) {
    b.n0 = n - i < most ? n - i : most;
    b.at[0] = d + i * sd;
    b.at[1] = x + i * sx;
    nx_cpu_stage(&a[0], &b, 0, ds, b.n0 * cw);
    nx_cpu_stage(&a[1], &b, 1, xs, b.n0 * cw);
    add(b.n0, ds, 1, ds, 1, xs, 1);
    nx_cpu_unstage(&a[0], &b, 0, ds, b.n0 * cw, c);
  }
}

/* Adds the box of [r] spans from the operand's position [px] into the
   result's [pd], each span's steps scaled by the strides [sx] and [sd]. */
static void add_box(const nx_cpu_row2 add, const nx_array *a, int w, int r,
                    const span *b, const int64_t *sx, const int64_t *sd,
                    int64_t px, int64_t pd) {
  int64_t extent[NX_MAX_RANK], step[2][NX_MAX_RANK];
  for (int i = 0; i < r; i++) {
    if (b[i].count == 0) return;
    extent[i] = b[i].count;
    step[0][i] = b[i].ds * sd[i];
    step[1][i] = b[i].xs * sx[i];
    pd += b[i].d0 * sd[i];
    px += b[i].x0 * sx[i];
  }
  int n = nx_coalesce_dims(2, r, extent, step);
  int64_t at[NX_MAX_RANK] = {0};
  int64_t rows = 1;
  for (int i = 0; i < n - 1; i++) rows *= extent[i];
  for (int64_t k = 0; k < rows; k++) {
    int64_t d = pd, x = px;
    for (int i = 0; i < n - 1; i++) {
      d += at[i] * step[0][i];
      x += at[i] * step[1][i];
    }
    add_row(add, a, w, extent[n - 1], d, step[0][n - 1], x, step[1][n - 1]);
    for (int i = n - 2; i >= 0 && ++at[i] == extent[i]; i--) at[i] = 0;
  }
}

typedef struct {
  const nx_array *a; /* the result, the operand */
  nx_cpu_row2 add;
  int w, r, leading; /* bytes, rank, leading axes no window takes */
  int64_t taps;
  int64_t slab; /* elements a unit starts first, or 0 */
  const uint8_t *start; /* the bits each result element starts from */
  /* Whether a unit copies its operand slab into [scratch_bytes] of
     scratch, taps first: the slab's copy there from position 0, its
     elements per tap and its steps along the result's axes. */
  int tap_major;
  int64_t scratch_bytes, plane;
  nx_box copy;
  int64_t ss[NX_MAX_RANK];
  span (*boxes)[NX_MAX_RANK]; /* by tap, then axis */
  int64_t *tap_px;            /* by tap, the operand's position */
  int64_t sx[NX_MAX_RANK], sd[NX_MAX_RANK];
} folder;

/* Units: an index of each leading axis's span, in C order. Each adds its
   slab of every tap's box in tap order, so a thread holds its operand's
   slab in cache across the taps. Where the scratch cannot be had, the
   taps add from the operand, strided. */
static void fold_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const folder *f = ctx;
  uint8_t *scratch = f->tap_major ? nx_cpu_take(f->scratch_bytes) : NULL;
  nx_array s[2] = {f->a[0], {.base = scratch}};
  for (int64_t u = lo; u < hi; u++) {
    int64_t v = u, pd = f->a[0].offset, px = 0;
    for (int i = f->leading - 1; i >= 0; i--) {
      const span *b = &f->boxes[0][i];
      int64_t k = v % b->count;
      v /= b->count;
      pd += (b->d0 + k * b->ds) * f->sd[i];
      px += (b->x0 + k * b->xs) * f->sx[i];
    }
    int rest = f->r - f->leading;
    if (f->slab > 0) start_run(&f->a[0], f->start, pd, f->slab);
    if (scratch == NULL) {
      for (int64_t t = 0; t < f->taps; t++)
        add_box(f->add, f->a, f->w, rest, f->boxes[t] + f->leading,
                f->sx + f->leading, f->sd + f->leading, px + f->tap_px[t],
                pd);
      continue;
    }
    nx_box b = f->copy;
    b.first[1] = f->tap_px[0] + px;
    nx_copy_box(scratch, f->a[1].base, &b, f->a[1].bits);
    for (int64_t t = 0; t < f->taps; t++)
      add_box(f->add, s, f->w, rest, f->boxes[t] + f->leading,
              f->ss + f->leading, f->sd + f->leading, t * f->plane, pd);
  }
  nx_cpu_give(scratch);
}

/* Lays out [f]'s scratch: the operand's slab off the leading axes, of
   extents [xs], with the taps of the [nw] windows [win] outermost, then
   the result's axes in C order. Answers 0 where the slab exceeds half the
   L2 or its copy takes more than a box's three axes; the fold then reads
   none of the layout. */
static int slab_copy(folder *f, const nx_array *a, const int64_t *xs,
                     int64_t (*win)[4], int nw) {
  int r = f->r, rx = r + nw, n = 0;
  int64_t extent[NX_MAX_RANK], step[NX_MAX_OPERANDS][NX_MAX_RANK];
  f->plane = 1;
  for (int i = r - 1; i >= f->leading; i--) {
    f->ss[i] = f->plane;
    f->plane *= xs[i];
  }
  int64_t at = f->plane;
  for (int k = nw - 1; k >= 0; k--) {
    extent[k] = win[k][1];
    step[0][k] = at;
    step[1][k] = a[1].dim[rx + r + k];
    at *= win[k][1];
  }
  for (int i = f->leading; i < r; i++) {
    extent[nw + i - f->leading] = xs[i];
    step[0][nw + i - f->leading] = f->ss[i];
    step[1][nw + i - f->leading] = f->sx[i];
  }
  f->scratch_bytes = at * f->w;
  if (f->scratch_bytes > nx_cpu_l2 / 2) return 0;
  n = nx_coalesce_dims(2, nw + r - f->leading, extent, step);
  if (n > 3) return 0;
  nx_box *b = &f->copy;
  for (int i = 0; i < 3; i++) {
    int j = i - (3 - n);
    b->extent[i] = j < 0 ? 1 : extent[j];
    b->step[0][i] = j < 0 ? 0 : step[0][j];
    b->step[1][i] = j < 0 ? 0 : step[1][j];
  }
  b->first[0] = 0;
  return 1;
}

/* Sets +0 where no tap of [f] lands, in the result of [shape] whose
   windows are [win], [nw] of them: along each axis, the runs of indices no
   tap's span holds, each one box. An axis's spans are those of its window's
   taps alone, the taps where every other window's index is 0, or tap 0's
   for an axis no window takes; an axis one span holds whole is skipped. A
   box two axes share is set twice. Answers 1 where the marks' memory cannot
   be had. */
static int unreached_zeros(const folder *f, const int64_t *shape,
                           int64_t (*win)[4], int nw) {
  static const uint8_t zero[16] = {0};
  int r = f->r;
  uint8_t *held = NULL;
  for (int i = 0; i < r; i++) {
    int64_t n = shape[i], count = 1, every = 0;
    /* The taps of the window along [i]: [count] of them, [every] apart. */
    for (int k = 0; k < nw; k++)
      if (win[k][0] == i) {
        count = win[k][1];
        every = 1;
        for (int q = k + 1; q < nw; q++) every *= win[q][1];
      }
    int whole = 0;
    for (int64_t j = 0; j < count && !whole; j++)
      whole = f->boxes[j * every][i].count == n;
    if (whole) continue;
    if (held == NULL) {
      int64_t widest = 1;
      for (int q = 0; q < r; q++)
        if (shape[q] > widest) widest = shape[q];
      held = malloc((size_t)widest);
      if (held == NULL) return 1;
    }
    memset(held, 0, (size_t)n);
    for (int64_t j = 0; j < count; j++) {
      const span *b = &f->boxes[j * every][i];
      for (int64_t k = 0; k < b->count; k++) held[b->d0 + k * b->ds] = 1;
    }
    int64_t extent[NX_MAX_RANK];
    for (int q = 0; q < r; q++) extent[q] = shape[q];
    for (int64_t d = 0; d < n;) {
      if (held[d]) {
        d++;
        continue;
      }
      int64_t e = d;
      while (e < n && !held[e]) e++;
      extent[i] = e - d;
      fill_box(&f->a[0], zero, r, extent, f->sd,
               f->a[0].offset + d * f->sd[i]);
      d = e;
    }
  }
  free(held);
  return 0;
}

value nx_cpu_fold_pad(value vpad, value vd, value vx) {
  CAMLparam3(vpad, vd, vx);
  int dt = nx_array_dtype(vx);
  if (nx_dtype_row_of(dt).kind == NX_KIND_BOOLEAN)
    CAMLreturn(Val_int(NX_DTYPE));
  nx_cpu_row2 add = nx_cpu_rows.op2[NX_OP2_ADD][nx_cpu_carrier(dt)];
  /* The operand's shape: the padded extents, each window's axis by its
     count, then the windows' sizes. */
  load_pad pad;
  int64_t shape[2 * NX_MAX_RANK], xs[2 * NX_MAX_RANK], off, want[NX_MAX_RANK];
  int r = nx_array_layout(vd, shape, &off), rx = nx_array_layout(vx, xs, &off);
  if (pad_of(vpad, &pad) || padded_shape(&pad, r, shape, want))
    CAMLreturn(Val_int(NX_SHAPE));
  int nw = pad.nwindows, ry = r;
  if (rx != r + nw) CAMLreturn(Val_int(NX_SHAPE));
  for (int i = 0; i < r + nw; i++)
    if (xs[i] != want[i]) CAMLreturn(Val_int(NX_SHAPE));
  int64_t lo[NX_MAX_RANK], s[NX_MAX_RANK];
  for (int i = 0; i < r; i++) {
    lo[i] = pad.lo[i];
    s[i] = pad.interior[i] + 1;
  }
  int64_t(*win)[4] = pad.window; /* by window: axis, size, step, dilation */
  nx_operand in[2] = {{vd, dt, 1}, {vx, dt, 0}};
  nx_array a[2];
  int e = nx_read(2, in, a);
  if (e) CAMLreturn(Val_int(e));
  int w = nx_cpu_width(dt);
  int64_t total = 1, taps = 1, positions = 0;
  for (int i = 0; i < r; i++) total *= shape[i];
  for (int k = 0; k < nw; k++) taps *= win[k][1];
  /* An operand with no element adds nothing. */
  int64_t xn = 1;
  for (int i = 0; i < r + nw; i++) xn *= xs[i];
  for (int i = 0; i < r; i++) positions += xs[i];
  /* The zeros, each tap's sum of a result, and each tap's spans. */
  double taps_total = (double)taps * (double)total;
  int released = nx_cpu_release(
      (double)total * w + NX_CPU_STRIDED_COST * (taps_total + xn) * w +
      SPAN_COST * (double)taps * (double)positions);
  if (total > 0 && xn == 0) zero_run(&a[0], a[0].offset, total);
  if (total > 0 && xn > 0) {
    folder f = {.a = a, .add = add, .w = w, .r = r, .taps = taps};
    uint8_t start[16];
    int signed_zero = negative_zero(dt, start);
    f.start = start;
    f.boxes = malloc((size_t)taps * sizeof *f.boxes);
    f.tap_px = malloc((size_t)taps * sizeof *f.tap_px);
    if (f.boxes == NULL || f.tap_px == NULL) {
      free(f.boxes);
      free(f.tap_px);
      nx_cpu_reacquire(released);
      nx_done(2, a);
      caml_raise_out_of_memory();
    }
    int windowed[NX_MAX_RANK] = {0};
    for (int k = 0; k < nw; k++) windowed[win[k][0]] = 1;
    for (int i = 0; i < r; i++) {
      f.sx[i] = a[1].dim[rx + i];
      f.sd[i] = a[0].dim[ry + i];
    }
    for (int64_t t = 0; t < taps; t++) {
      /* Tap t's indices along the window axes, in C order. */
      int64_t q = t, j[NX_MAX_RANK];
      f.tap_px[t] = a[1].offset;
      for (int k = nw - 1; k >= 0; k--) {
        j[k] = q % win[k][1];
        q /= win[k][1];
        f.tap_px[t] += j[k] * a[1].dim[rx + r + k];
      }
      span *box = f.boxes[t];
      for (int i = 0; i < r; i++)
        box[i] = span_of(xs[i], 1, 0, 0, lo[i], s[i], shape[i]);
      for (int k = 0; k < nw; k++) {
        int64_t ax = win[k][0];
        box[ax] = span_of(xs[ax], win[k][2], j[k], win[k][3], lo[ax], s[ax],
                          shape[ax]);
      }
    }
    /* The leading axes no window takes have one span for every tap: their
       indices cut the work into units of disjoint results. */
    int64_t units = 1, slab = 1;
    while (f.leading < r && !windowed[f.leading])
      units *= f.boxes[0][f.leading++].count;
    for (int i = f.leading; i < r; i++) slab *= shape[i];
    if (units * slab == total) f.slab = slab;
    else start_run(&a[0], start, a[0].offset, total);
    int64_t bytes = xn * w + taps * total * w;
    f.tap_major = NX_CPU_COPY_STRIDED && nx_cpu_carrier(dt) == dt &&
                  r > f.leading && f.sx[r - 1] != 1 &&
                  slab_copy(&f, a, xs, win, nw);
    nx_cpu_job(units, bytes, bytes, fold_units, &f);
    int lost = signed_zero && unreached_zeros(&f, shape, win, nw);
    free(f.boxes);
    free(f.tap_px);
    if (lost) {
      nx_cpu_reacquire(released);
      nx_done(2, a);
      caml_raise_out_of_memory();
    }
  }
  nx_cpu_reacquire(released);
  nx_done(2, a);
  CAMLreturn(Val_int(NX_OK));
}
