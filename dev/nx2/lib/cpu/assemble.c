/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Assemblies, and folds: the adjoint of a padded, windowed load.

   An assembly fills what no piece covers, then copies each piece into its
   region in order, so the last piece holding an element wins. Pieces of
   unit steps whose regions are disjoint and fill the result, as a
   concatenation's do, leave nothing to fill; one such piece leaves the
   2·rank boxes around it, as a pad does; any other set fills the whole
   result first. A first piece that is the destination over the whole
   result is already in place.

   A fold's result starts at +0. Then each tap, the indices along the window
   axes in C order, adds the operand's elements of that tap into the
   result: within a tap each result element receives at most one, so every
   element adds its taps in C order, as Spec states. Along an axis, the
   operand's indices whose padded coordinate w·step + j·dilation lands on
   an element, lo + i·(interior + 1) with 0 <= i < d, step by a constant
   and so do the i they land on: a tap is one box of the operand added into
   one box of the result, rows by the table's add. An axis no window takes
   is one of step 1 and tap 0, and its span is every tap's: the leading
   such axes, as a convolution's batch and channels, cut the boxes into
   slabs of disjoint results that threads take whole, each adding its
   slab's taps in order while it holds the operand's slab in cache. */

#include <stdlib.h>
#include <string.h>

#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "cpu.h"

/* Assemblies */

/* Stores the element of bits [fill] into the box of [dst] from [first],
   of [r] axes of [extent] stepping [step] elements. */
static void fill_box(const nx_array *dst, const uint8_t *fill, int r,
                     const int64_t *extent, const int64_t *step,
                     int64_t first) {
  int64_t e[NX_MAX_RANK], s[NX_MAX_OPERANDS][NX_MAX_RANK];
  for (int i = 0; i < r; i++) {
    if (extent[i] == 0) return;
    e[i] = extent[i];
    s[0][i] = step[i];
  }
  int n = nx_coalesce_dims(1, r, e, s);
  int64_t rows = 1, at[NX_MAX_RANK] = {0};
  for (int i = 0; i < n - 1; i++) rows *= e[i];
  int bits = dst->bits, w = bits / 8;
  int lg = w == 1 ? 0 : w == 2 ? 1 : w == 4 ? 2 : w == 8 ? 3 : 4;
  for (int64_t k = 0; k < rows; k++) {
    int64_t p = first;
    for (int i = 0; i < n - 1; i++) p += at[i] * s[0][i];
    if (bits < 8)
      for (int64_t j = 0; j < e[n - 1]; j++)
        nx_sub_store(dst->base, bits, p + j * s[0][n - 1], fill[0]);
    else
      nx_cpu_table->fill[lg](e[n - 1], dst->base + p * w, s[0][n - 1], fill);
    for (int i = n - 2; i >= 0 && ++at[i] == e[i]; i--) at[i] = 0;
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

/* The pieces' operands, descriptors, ranges and boxes: C-heap scratch of
   their count. */
typedef struct {
  nx_operand *in;
  nx_array *a;
  int64_t *ranges;
  int64_t (*lo)[NX_MAX_RANK], (*hi)[NX_MAX_RANK];
} pieces;

static void free_pieces(pieces *p) {
  free(p->in);
  free(p->a);
  free(p->ranges);
  free(p->lo);
  free(p->hi);
}

value nx_cpu_assemble(value vs, value vd, value vpieces) {
  CAMLparam3(vs, vd, vpieces);
  int np = (int)Wosize_val(vpieces);
  const nx_spec_shaped *sp = (const nx_spec_shaped *)String_val(vs);
  int r = sp->rank, nfill = sp->nfill;
  if (sp->npieces != np) CAMLreturn(Val_int(NX_SHAPE));
  /* Scratch first: allocating runs no OCaml code, so the descriptor stays
     where it is until it is copied. */
  pieces p = {.in = malloc((size_t)(1 + np) * sizeof *p.in),
              .a = malloc((size_t)(1 + np) * sizeof *p.a),
              .ranges = malloc((size_t)(3 * r * np + 1) * sizeof *p.ranges),
              .lo = malloc((size_t)(np + 1) * sizeof *p.lo),
              .hi = malloc((size_t)(np + 1) * sizeof *p.hi)};
  if (!p.in || !p.a || !p.ranges || !p.lo || !p.hi) {
    free_pieces(&p);
    caml_raise_out_of_memory();
  }
  uint8_t fill[16];
  int64_t shape[NX_MAX_RANK], *ranges = p.ranges;
  memcpy(fill, sp->fill, sizeof fill);
  for (int i = 0; i < r; i++) shape[i] = sp->shape[i];
  for (int j = 0; j < np; j++)
    for (int i = 0; i < r; i++)
      memcpy(ranges + 3 * (j * r + i), nx_spec_shaped_range(sp, j, i),
             3 * sizeof(int64_t));
  int dt = nx_array_dtype(vd), bits = nx_dtype_row_of(dt).bits;
  int e = NX_OK;
  if (nfill != (bits < 8 ? 1 : bits / 8) || (bits < 8 && fill[0] >> bits) ||
      (dt == NX_BOOL && fill[0] > 1))
    e = NX_DTYPE;
  int64_t ys[2 * NX_MAX_RANK], ps[2 * NX_MAX_RANK], off;
  if (!e && nx_array_layout(vd, ys, &off) != r) e = NX_SHAPE;
  for (int i = 0; i < r && !e; i++)
    if (ys[i] != shape[i]) e = NX_SHAPE;
  for (int j = 0; j < np && !e; j++) {
    if (nx_array_layout(Field(vpieces, j), ps, &off) != r) e = NX_SHAPE;
    for (int i = 0; i < r && !e; i++)
      if (ps[i] != ranges[3 * (j * r + i) + 1]) e = NX_SHAPE;
  }
  if (e) {
    free_pieces(&p);
    CAMLreturn(Val_int(e));
  }
  nx_operand *in = p.in;
  nx_array *a = p.a;
  in[0] = (nx_operand){vd, dt, 1};
  for (int j = 0; j < np; j++) in[1 + j] = (nx_operand){Field(vpieces, j), dt, 0};
  if ((e = nx_read(1 + np, in, a))) {
    free_pieces(&p);
    CAMLreturn(Val_int(e));
  }
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
  if (!e && total > 0 && !in_place && !(unit && tiles(np, r, lo, hi, total))) {
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
  for (int j = 0; j < np && !e; j++) {
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
    nx_cpu_copy_loop(a[0].base, a[1 + j].base, bits, &l);
  }
  nx_done(1 + np, a);
  free_pieces(&p);
  CAMLreturn(Val_int(e));
}

/* Padded loads, staged as an assembly of one piece: the fill, then the
   operand's elements that land inside the padded array, lo + t·(interior
   + 1) along each axis, then the windows as a layout over the copy. */

int nx_cpu_unpad(const nx_array *a, const nx_spec_pad *p, nx_array *out) {
  int r = a->rank, nw = p->nwindows, bits = a->bits;
  const int64_t *lo = p->geometry, *hi = p->geometry + r,
                *inner = p->geometry + 2 * r;
  int64_t shape[NX_MAX_RANK], stride[NX_MAX_RANK], total = 1;
  for (int i = 0; i < r; i++) {
    int64_t d = a->dim[i];
    shape[i] = lo[i] + hi[i] + d + (d > 0 ? inner[i] * (d - 1) : 0);
  }
  for (int i = r - 1; i >= 0; i--) {
    stride[i] = total;
    total *= shape[i];
  }
  *out = (nx_array){.dtype = a->dtype, .bits = bits, .rank = r + nw};
  if (total > 0) {
    out->base = malloc((size_t)((total * bits + 7) / 8));
    if (out->base == NULL) return 1;
    fill_box(out, p->fill, r, shape, stride, 0);
    /* The operand's indices t that land inside: lo + t·s in [0, shape). */
    nx_loop l = {.rank = r, .first = {0, a->offset}};
    int64_t n = 1;
    for (int i = 0; i < r; i++) {
      int64_t s = inner[i] + 1, t0 = lo[i] >= 0 ? 0 : (-lo[i] + s - 1) / s;
      int64_t last = shape[i] - 1 - lo[i] < 0 ? -1 : (shape[i] - 1 - lo[i]) / s;
      if (last > a->dim[i] - 1) last = a->dim[i] - 1;
      l.extent[i] = last < t0 ? 0 : last - t0 + 1;
      l.first[0] += (lo[i] + t0 * s) * stride[i];
      l.first[1] += t0 * a->dim[r + i];
      l.step[0][i] = s * stride[i];
      l.step[1][i] = a->dim[r + i];
      n *= l.extent[i];
    }
    if (n > 0) {
      l.rank = nx_coalesce_dims(2, r, l.extent, l.step);
      nx_cpu_copy_loop(out->base, a->base, bits, &l);
    }
  }
  /* The windows: each window's axis by its count, stepping [step]
     elements, then an axis of its size, stepping [dilation]. */
  for (int i = 0; i < r; i++) {
    out->dim[i] = shape[i];
    out->dim[r + nw + i] = stride[i];
  }
  for (int k = 0; k < nw; k++) {
    const int64_t *w = p->geometry + 3 * r + 4 * k; /* axis, size, step,
                                                      dilation */
    int64_t ax = w[0];
    out->dim[ax] = (shape[ax] - 1 - w[3] * (w[1] - 1)) / w[2] + 1;
    out->dim[r + nw + ax] = w[2] * stride[ax];
    out->dim[r + k] = w[1];
    out->dim[r + nw + r + k] = w[3] * stride[ax];
  }
  int64_t elements = 1;
  for (int i = 0; i < r + nw; i++) elements *= out->dim[i];
  out->flags = elements == 0 ? NX_EMPTY : nw == 0 ? NX_CONTIGUOUS | NX_DISTINCT : 0;
  return 0;
}

/* Folds */

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

/* Adds the box of [r] spans from the operand's position [px] into the
   result's [pd], each span's steps scaled by the strides [sx] and [sd]. */
static void add_box(const nx_cpu_row2 add, const nx_array *a, int w, int r,
                    const span *b, const int64_t *sx, const int64_t *sd,
                    int64_t px, int64_t pd) {
  int64_t extent[NX_MAX_RANK], step[NX_MAX_OPERANDS][NX_MAX_RANK];
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
    uint8_t *dp = a[0].base + d * w;
    add(extent[n - 1], dp, step[0][n - 1], dp, step[0][n - 1],
        a[1].base + x * w, step[1][n - 1]);
    for (int i = n - 2; i >= 0 && ++at[i] == extent[i]; i--) at[i] = 0;
  }
}

typedef struct {
  const nx_array *a; /* the result, the operand */
  nx_cpu_row2 add;
  int w, r, leading; /* bytes, rank, leading axes no window takes */
  int64_t taps;
  span (*boxes)[NX_MAX_RANK]; /* by tap, then axis */
  int64_t *tap_px;            /* by tap, the operand's position */
  int64_t sx[NX_MAX_RANK], sd[NX_MAX_RANK];
} folder;

/* Units: an index of each leading axis's span, in C order. Each adds its
   slab of every tap's box in tap order, so a thread holds its operand's
   slab in cache across the taps. */
static void fold_units(int64_t lo, int64_t hi, int worker, void *ctx) {
  (void)worker;
  const folder *f = ctx;
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
    for (int64_t t = 0; t < f->taps; t++)
      add_box(f->add, f->a, f->w, rest, f->boxes[t] + f->leading,
              f->sx + f->leading, f->sd + f->leading, px + f->tap_px[t], pd);
  }
}

value nx_cpu_fold_pad(value vs, value vd, value vx) {
  CAMLparam3(vs, vd, vx);
  const nx_spec_shaped *sp = (const nx_spec_shaped *)String_val(vs);
  const nx_spec_pad *pad = nx_spec_shaped_pad(sp);
  int r = sp->rank, nw = pad->nwindows;
  int64_t shape[NX_MAX_RANK], lo[NX_MAX_RANK], s[NX_MAX_RANK];
  int64_t win[NX_MAX_RANK][4]; /* by window: axis, size, step, dilation */
  for (int i = 0; i < r; i++) {
    shape[i] = sp->shape[i];
    lo[i] = pad->geometry[i];
    s[i] = pad->geometry[2 * r + i] + 1;
  }
  for (int k = 0; k < nw; k++)
    for (int f = 0; f < 4; f++) win[k][f] = pad->geometry[3 * r + 4 * k + f];
  int dt = nx_array_dtype(vx);
  if (nx_dtype_row_of(dt).kind == NX_KIND_BOOLEAN)
    CAMLreturn(Val_int(NX_DTYPE));
  nx_cpu_row2 add = nx_cpu_table->op2[NX_OP2_ADD][dt];
  if (add == NULL || nx_cpu_carrier(dt) != dt)
    CAMLreturn(Val_int(NX_DECLINED));
  /* The operand's shape: the padded extents, each window's axis by its
     count, then the windows' sizes. */
  int64_t ys[2 * NX_MAX_RANK], xs[2 * NX_MAX_RANK], off, want[NX_MAX_RANK];
  int ry = nx_array_layout(vd, ys, &off), rx = nx_array_layout(vx, xs, &off);
  if (ry != r || rx != r + nw) CAMLreturn(Val_int(NX_SHAPE));
  for (int i = 0; i < r; i++) {
    if (ys[i] != shape[i]) CAMLreturn(Val_int(NX_SHAPE));
    int64_t d = shape[i];
    want[i] = lo[i] + pad->geometry[r + i] + d + (d > 0 ? (s[i] - 1) * (d - 1) : 0);
  }
  for (int k = 0; k < nw; k++) {
    int64_t a = win[k][0], p = want[a];
    want[a] = (p - 1 - win[k][3] * (win[k][1] - 1)) / win[k][2] + 1;
    want[r + k] = win[k][1];
  }
  for (int i = 0; i < r + nw; i++)
    if (xs[i] != want[i]) CAMLreturn(Val_int(NX_SHAPE));
  nx_operand in[2] = {{vd, dt, 1}, {vx, dt, 0}};
  nx_array a[2];
  int e = nx_read(2, in, a);
  if (e) CAMLreturn(Val_int(e));
  int w = nx_cpu_width(dt);
  int64_t total = 1, taps = 1;
  for (int i = 0; i < r; i++) total *= shape[i];
  for (int k = 0; k < nw; k++) taps *= win[k][1];
  if (total > 0) memset(a[0].base + a[0].offset * w, 0, (size_t)(total * w));
  /* An operand with no element adds nothing. */
  int64_t xn = 1;
  for (int i = 0; i < r + nw; i++) xn *= xs[i];
  if (total > 0 && xn > 0) {
    folder f = {.a = a, .add = add, .w = w, .r = r, .taps = taps};
    f.boxes = malloc((size_t)taps * sizeof *f.boxes);
    f.tap_px = malloc((size_t)taps * sizeof *f.tap_px);
    if (f.boxes == NULL || f.tap_px == NULL) {
      free(f.boxes);
      free(f.tap_px);
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
    int64_t units = 1;
    while (f.leading < r && !windowed[f.leading])
      units *= f.boxes[0][f.leading++].count;
    int64_t bytes = xn * w + taps * total * w;
    nx_cpu_job(units, bytes, bytes, fold_units, &f);
    free(f.boxes);
    free(f.tap_px);
  }
  nx_done(2, a);
  CAMLreturn(Val_int(NX_OK));
}
