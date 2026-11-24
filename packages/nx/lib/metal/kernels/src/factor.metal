/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The factorisations' steps: cholesky, lu, qr and solve_triangular,
   batched, at float32 and complex64 (kernels.h, Factorisations).
   factor.ml plans them.

   Every routine computes in place in a C-contiguous matrix per batch
   element (a destination, or workspace, into which the plan copies the
   operand), by launches of the steps below and of the contraction's
   products. Each element of Cholesky, LU and the triangular solves gets
   the arithmetic of the textbook unblocked algorithm, right-looking,
   whatever the blocking: its updates, one mac each, in increasing step
   order (decreasing for an upper solve), starting from its own value,
   then its division. The blocks only choose which launch performs which
   updates: a product starts each element's chain from the element itself
   and takes its k products in order. QR's sums over rows follow their
   own order (below).

   Every element a step reads from an operand, or from a copy of one, it
   reads flushed: a subnormal as a zero of its sign (nx_metal.mli). Past
   those reads every value is an arithmetic result, which the GPU
   flushes, so a routine computes on its operands' flushed values.

   Steps:
   - trsm: a triangular solve of at most leaf rows, a thread a column.
   - lu_panel: partial pivoting on a panel of columns, its rows held by
     one group, or by P groups over one launch a column (no group waits
     on another group of its launch).
   - gather: a panel's row interchanges on the other columns and on perm.
   - chol: Cholesky of a diagonal block held in threadgroup memory.
   - qr_panel and the finishing steps: see their sections.

   A matrix's G threads are 16 or 32 lanes of one SIMD-group, several
   matrices a group, or the whole group. The lanes of a SIMD-group meet
   at a SIMD-group barrier, the whole group at a threadgroup barrier,
   which a matrix of G <= 32 lanes never takes. Every lane of a
   SIMD-group reaches each of its barriers: a group holds an even count
   of matrices of 16 lanes, and the lanes of one past the batch's last
   compute that last one again without storing it (member). A
   SIMD-group of no matrix returns at once.

   This follows nx.amd's factor.hip, a port of LAPACK's xGETRF, xPOTRF,
   xGEQRT, xLARFG, xLARFT and ReLAPACK's recursive xTRSM (LICENSE-lapack,
   LICENSE-relapack). */

#include "device.h"
#include "nx_kinds.h"

namespace la {

/* Scalars. T is float or nx_c64, R its part's type. Every update of the
   routines is mac or nmac: one fused multiply-add on reals; on complex
   numbers two a part, the real part taking ar br then ai (-bi), the
   imaginary part ar bi then ai br, the order nx_kernel.mli numbers a
   complex contraction's terms in. */

typedef nx_c64 c64;
typedef float R;

static inline float re(float x) { return x; }
static inline float re(c64 x) { return x.re; }
static inline float im(float) { return 0.0f; }
static inline float im(c64 x) { return x.im; }

/* T's number from its parts: a real number drops the imaginary part. */
template <typename T> struct num;
template <> struct num<float> {
  static constexpr bool cplx() { return false; }
  static float mk(float r, float) { return r; }
};
template <> struct num<c64> {
  static constexpr bool cplx() { return true; }
  static c64 mk(float r, float i) { return nx_c64_of(r, i); }
};

template <typename T> static inline T mk(float r, float i) {
  return num<T>::mk(r, i);
}

static inline float nan_f() { return as_type<float>(0x7FC00000u); }
template <typename T> static inline T zero() { return mk<T>(0.0f, 0.0f); }
template <typename T> static inline T one() { return mk<T>(1.0f, 0.0f); }
template <typename T> static inline T nan_() { return mk<T>(nan_f(), nan_f()); }
template <typename T> static inline T neg(T x) { return mk<T>(-re(x), -im(x)); }
template <typename T> static inline T conj_(T x) { return mk<T>(re(x), -im(x)); }
template <typename T> static inline T conj_if(T x, bool c) {
  return c ? conj_(x) : x;
}
template <typename T> static inline bool is_zero(T x) {
  return re(x) == 0.0f && im(x) == 0.0f;
}

/* [x] as its operand's flushed value: each subnormal part a zero of its
   sign. */
static inline float flush(float x) { return nx_flush_f32(x); }
static inline c64 flush(c64 x) {
  return nx_c64_of(nx_flush_f32(x.re), nx_flush_f32(x.im));
}

static inline float fma_(float a, float b, float c) { return nx_fmaf(a, b, c); }
static inline float abs_(float x) { return metal::fabs(x); }
static inline float sqrt_(float x) { return nx_sqrtf(x); }
static inline float ldexp_(float x, int e) { return nx_ldexp_f32(x, e); }
static inline float copysign_(float x, float y) { return metal::copysign(x, y); }

/* The exponent e of a finite normal x > 0 = m 2^e, 1 <= m < 2. */
static inline int ilogb_(float x) { return int(nx_ilogb_f32(x)); }

/* The larger of a and b, the other one where one is NaN: chosen here, not
   by the instruction's NaN rule. */
static inline float max_(float a, float b) {
  if (a != a) return b;
  if (b != b) return a;
  return a < b ? b : a;
}

/* c + a·b. */
static inline float mac(float c, float a, float b) { return fma_(a, b, c); }
static inline c64 mac(c64 c, c64 a, c64 b) {
  float r = fma_(a.re, b.re, c.re);
  r = fma_(-a.im, b.im, r);
  float i = fma_(a.re, b.im, c.im);
  i = fma_(a.im, b.re, i);
  return nx_c64_of(r, i);
}

/* c − a·b, as c + (−a)·b. */
template <typename T> static inline T nmac(T c, T a, T b) {
  return mac(c, neg(a), b);
}
template <typename T> static inline T mul(T a, T b) {
  return mac(zero<T>(), a, b);
}

/* LU's magnitude: |re| + |im|. */
static inline float mag(float x) { return abs_(x); }
static inline float mag(c64 x) { return abs_(x.re) + abs_(x.im); }

/* x / d: on complex numbers each part divided by d's real part where d's
   imaginary part is zero, as Cholesky's diagonal is, else Smith's
   quotient (nx_kinds.h's Fdiv). */
static inline float divd(float x, float d) { return x / d; }
static inline c64 divd(c64 x, c64 d) {
  if (d.im == 0.0f) return nx_c64_of(x.re / d.re, x.im / d.re);
  return nx_fdiv_c64(x, d);
}

/* The most rows of a triangular solve's leaf, the columns of an LU
   panel, the order of a Cholesky block. */
template <typename T> constexpr int leaf_rows() {
  return num<T>::cplx() ? NX_LA_LEAF / 2 : NX_LA_LEAF;
}

static inline long imin(long a, long b) { return a < b ? a : b; }
static inline long imax(long a, long b) { return a < b ? b : a; }
static inline long cdiv(long a, long b) { return (a + b - 1) / b; }

/* A launch's unit u over d: by d's divisor [md] (device.h's divide),
   where the host found the launch's units and d below 2^32, else under
   [wide] in 64 bits, which the GPU runs in software. */
static inline long unit_div(long u, long d, long md, long wide) {
  return wide ? u / d : long(divide(uint(u), ulong(md)));
}

/* A pointer to [T] in threadgroup memory, or in device memory where
   [DEV]: a tile lies in either, by the launch's choice, which changes no
   bit. */
template <typename T, bool DEV> struct space;
template <typename T> struct space<T, false> {
  typedef threadgroup T *ptr;
  static ptr at(device uchar *, threadgroup uchar *t) { return (ptr)t; }
};
template <typename T> struct space<T, true> {
  typedef device T *ptr;
  static ptr at(device uchar *d, threadgroup uchar *) { return (ptr)d; }
};

/* Meetings. Every barrier orders threadgroup and device memory, since a
   tile may lie in either. */

static inline void barrier() {
  metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup |
                             metal::mem_flags::mem_device);
}

static inline void wave_sync() {
  metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup |
                           metal::mem_flags::mem_device);
}

/* A matrix's G threads meet: within a SIMD-group, or the whole group. */
static inline void meet(int G) {
  if (G <= 32)
    wave_sync();
  else
    barrier();
}

/* The matrix the lanes of a group's matrix [g] compute, of a unit's
   [mats] from [unit mats], below [matrices]; -1 where their SIMD-group
   holds none. At G = 16, the lanes of matrix z = matrices, one past the
   batch's last, share their SIMD-group's barriers with matrix z − 1, the
   last: they compute it again and set [ghost], under which they store
   nothing. */
static inline long member(long unit, long mats, long matrices, int G, int g,
                          thread bool &ghost) {
  const long z = unit * mats + g;
  ghost = false;
  if (g >= mats) return -1;
  if (z < matrices) return z;
  if (G != 16 || (g & 1) == 0 || z - 1 >= matrices) return -1;
  ghost = true;
  return z - 1;
}

/* Lanes. A value of another lane of the SIMD-group, 32 bits at a time. */

static inline uint shfl(uint v, int src) {
  return metal::simd_shuffle(v, ushort(src));
}
static inline int shfl(int v, int src) {
  return as_type<int>(shfl(as_type<uint>(v), src));
}
static inline float shfl(float v, int src) {
  return as_type<float>(shfl(as_type<uint>(v), src));
}
static inline long shfl(long v, int src) {
  const uint2 h = as_type<uint2>(v);
  return as_type<long>(uint2(shfl(h.x, src), shfl(h.y, src)));
}
template <typename T> static inline T shfl_t(T x, int src) {
  return mk<T>(shfl(re(x), src), shfl(im(x), src));
}

/* The lane [o] above [l] within its segment of [width] lanes, or [l]
   past the segment's end. */
static inline int down(int l, int o, int width) {
  return l % width + o < width ? l + o : l;
}

/* Triangular solve leaf */

struct trsm_words {
  device const uchar *l;
  device uchar *x;
  device const uchar *src;
  long sl, lrs, lcs, sx, xrs, xcs, n, r, rdiv, wide, count, conj, unit,
      whole, first;
};

NX_FIELD(trsm_words, l, NX_LA_TRSM_L);
NX_FIELD(trsm_words, x, NX_LA_TRSM_X);
NX_FIELD(trsm_words, src, NX_LA_TRSM_SRC);
NX_FIELD(trsm_words, sl, NX_LA_TRSM_SL);
NX_FIELD(trsm_words, r, NX_LA_TRSM_R);
NX_FIELD(trsm_words, count, NX_LA_TRSM_COUNT);
NX_FIELD(trsm_words, first, NX_LA_TRSM_FIRST);
NX_SIZE(trsm_words, NX_LA_TRSM_WORDS);

/* x ← op(l)⁻¹ x for the lower triangular n × n view op(l), n <= the
   leaf's rows, and the n × r view x of each matrix, a thread a column:
   x_i takes −l_ip x_p for p < i in increasing p, one nmac each, then is
   divided by l_ii unless [unit]. op(l)'s element (i, p) is l's at
   i·lrs + p·lcs, conjugated under [conj]: an upper triangle or an
   adjoint is a view of reversed or swapped strides, so its solve runs in
   its own order. Column g of the launch is column g mod r of matrix
   g / r. l's elements are arithmetic operands but the diagonal, which
   the zero test reads flushed. */
template <typename T>
static inline void trsm(constant trsm_words &w, threadgroup uchar *lds,
                        uint gx, uint tid, ushort, ushort) {
  const long g = w.first + long(gx) * NX_LA_TRSM_THREADS + tid;
  if (g >= w.count) return;
  const long r = w.r;
  const long z = unit_div(g, r, w.rdiv, w.wide), col = g - z * r;
  const int n = int(w.n);
  const bool conj = w.conj != 0, unit = w.unit != 0, whole = w.whole != 0;
  device const T *l = (device const T *)w.l + z * w.sl;
  device T *x = (device T *)w.x + z * w.sx + col * w.xcs;
  device const T *b =
      w.src ? (device const T *)w.src + z * w.sx + col * w.xcs : x;
  threadgroup T *xs = (threadgroup T *)lds; /* xs[i T + tid] */
  for (int i = 0; i < n; i++)
    xs[i * NX_LA_TRSM_THREADS + tid] = flush(b[i * w.xrs]);
  bool zero_diag = false;
  for (int i = 0; i < n; i++) {
    T v = xs[i * NX_LA_TRSM_THREADS + tid];
    device const T *li = l + i * w.lrs;
    for (int q = 0; q < i; q++)
      v = nmac(v, conj_if(li[q * w.lcs], conj),
               xs[q * NX_LA_TRSM_THREADS + tid]);
    if (!unit) {
      const T d = conj_if(flush(li[i * w.lcs]), conj);
      zero_diag |= is_zero(d);
      v = divd(v, d);
    }
    xs[i * NX_LA_TRSM_THREADS + tid] = v;
  }
  for (int i = 0; i < n; i++)
    x[i * w.xrs] =
        whole && zero_diag ? nan_<T>() : xs[i * NX_LA_TRSM_THREADS + tid];
}

/* LU panel */

/* A pivot candidate: [forced] where it is the row at the step's own
   position and its magnitude is NaN, which the search never leaves; else
   its magnitude, -1 for NaN, which nothing exceeds and which exceeds no
   number; then its position, the lesser winning a tie. [row] is its row
   in the group's tile, [src] its group. The order is total, so any tree
   of comparisons picks the same row: the sequential search's. */
struct Cand {
  float key;
  int forced;
  long pos;
  int src;
  int pad;
  long row;
};

static_assert(sizeof(Cand) <= NX_LA_SLOT_HEAD,
              "a candidate fits a slot's head and the host's candidates");

static inline bool better(Cand a, Cand b) {
  if (a.forced != b.forced) return b.forced > a.forced;
  if (a.key != b.key) return b.key > a.key;
  return b.pos < a.pos;
}

static inline Cand none() {
  Cand c;
  c.key = -2.0f;
  c.forced = 0;
  c.pos = 0x7FFFFFFFFFFFFFFF;
  c.src = -1;
  c.pad = 0;
  c.row = -1;
  return c;
}

static inline Cand shfl_cand(Cand c, int src) {
  Cand d;
  d.key = shfl(c.key, src);
  d.forced = shfl(c.forced, src);
  d.pos = shfl(c.pos, src);
  d.src = shfl(c.src, src);
  d.pad = 0;
  d.row = shfl(c.row, src);
  return d;
}

/* The best candidate of a matrix's G threads: G of 16 or 32 lanes, or
   the whole group, through [s], a slot a SIMD-group, of the parity a step
   alternates so that its writes never meet the last step's reads. */
static inline Cand group_best(Cand c, int G, threadgroup Cand *s, int ln,
                              int sg) {
  const int width = G < 32 ? G : 32;
  for (int o = width / 2; o >= 1; o /= 2) {
    const Cand d = shfl_cand(c, ln ^ o);
    if (better(c, d)) c = d;
  }
  if (G <= 32) return c;
  const int waves = NX_LA_THREADS / 32;
  if (ln == 0) s[sg] = c;
  barrier();
  c = s[0];
  for (int q = 1; q < waves; q++)
    if (better(c, s[q])) c = s[q];
  return c;
}

struct panel_words {
  device uchar *a;
  device long *piv;
  device long *src_of;
  device const uchar *src;
  device long *perm;
  device uchar *spill;
  device uchar *slots;
  device uchar *levels;
  long sbat, lda, spiv, sperm, m, c0, w, matrices, per, P, G, mats, ld,
      in_lds, spill_bytes, slot_bytes, slots_stride, slots_parity,
      cands_parity, row, head, pos, level, lv_stride, lv_in, lv_out, n_in,
      k, pdiv, runs_div, wide, first, count;
};

NX_FIELD(panel_words, a, NX_LA_PANEL_A);
NX_FIELD(panel_words, slots, NX_LA_PANEL_SLOTS);
NX_FIELD(panel_words, levels, NX_LA_PANEL_LEVELS);
NX_FIELD(panel_words, level, NX_LA_PANEL_LEVEL);
NX_FIELD(panel_words, sbat, NX_LA_PANEL_SBAT);
NX_FIELD(panel_words, G, NX_LA_PANEL_G);
NX_FIELD(panel_words, k, NX_LA_PANEL_K);
NX_FIELD(panel_words, count, NX_LA_PANEL_COUNT);
NX_SIZE(panel_words, NX_LA_PANEL_WORDS);

/* LU of the panel of columns [c0, c0 + w) of rows [c0, m) of each
   matrix, in place: partial pivoting as nx_kernel.mli states it, then
   each later row's element divided by the pivot (not where the pivot is
   zero), then the panel's later columns updated by one nmac each. Rows
   stay where they were loaded; [pos] holds each row's position, and the
   interchange of step j swaps two positions. At the end each row is
   stored at its position, pivots[c0 + j] = c0 + p_j, and src_of[q] = the
   row now at position q (rows from c0), for gather.

   With P = 1, one launch: G threads a matrix, [mats] a group. With
   P > 1, P groups a matrix, group q holding the panel's rows [q·per,
   (q + 1)·per) in its tile at [spill], over steps + 1 launches, launch k
   taking step k − 1's winner and update and step k's candidates: each
   group writes its best candidate and that row's values to its slot, and
   the next launch's groups find the same winner among the slots, the
   order being total. Where the slots are many, launches of levels
   between two steps take each run's best into [levels], the last level
   few, and the next launch searches it instead: the same winner, its
   candidate copied whole. No group waits on another. The tile lies in
   threadgroup memory or, where [DEV], in device memory. */
template <typename T, bool DEV> struct Lu {
  typedef typename space<T, DEV>::ptr TP;
  typedef typename space<long, DEV>::ptr PP;
  constant panel_words &p;
  long z, q, r0, rows, w, ldt, steps;
  int G, lane;
  bool ghost;

  Lu(constant panel_words &p_) : p(p_) {}

  /* Element (r, c) of the tile: in threadgroup memory row-major, [ldt]
     elements a row; in device memory column-major, [ldt] elements a
     column, so that the lanes of a SIMD-group, a row each, read adjacent
     elements of a column. */
  long at(long r, long c) const { return DEV ? r + c * ldt : r * ldt + c; }
  long col_step() const { return DEV ? ldt : 1; }

  /* The tile's rows from a, or from [src], flushed, and their
     positions. */
  void load(TP tile, PP pos, device const T *a) {
    for (long r = lane; r < rows; r += G)
      for (long c = 0; c < w; c++)
        tile[at(r, c)] = flush(
            p.src ? ((device const T *)p.src)[z * p.sbat + (r0 + r) * p.lda + c]
                  : a[(r0 + r) * p.lda + c]);
    for (long r = lane; r < rows; r += G) pos[r] = r0 + r;
  }

  /* A row's candidate for step j. */
  Cand cand(TP tile, long r, long ps, long j) {
    const float mg = mag(tile[at(r, j)]);
    Cand c;
    c.key = mg != mg ? -1.0f : mg;
    c.forced = ps == j && mg != mg;
    c.pos = ps;
    c.src = int(q);
    c.pad = 0;
    c.row = r;
    return c;
  }

  /* Step j with pivot row u, its elements [us] apart, at position pj: the
     interchange, each later row's multiplier and update; the best
     candidate for step j + 1. */
  template <typename UP>
  Cand step(TP tile, PP pos, UP u, long us, long j, long pj) {
    if (q == 0 && lane == 0 && !ghost)
      p.piv[z * p.spiv + p.c0 + j] = p.c0 + pj;
    const T d = u[j * us];
    const bool divide = !is_zero(d);
    Cand next = none();
    for (long r = lane; r < rows; r += G) {
      /* Only the rows at j and pj move: the others' positions are read,
         not rewritten. */
      const long was = pos[r];
      const long ps = was == j ? pj : was == pj ? j : was;
      if (ps != was) pos[r] = ps;
      if (ps <= j) continue;
      TP tr = tile + at(r, 0);
      const long cs = col_step();
      T l = tr[j * cs];
      if (divide) l = divd(l, d);
      tr[j * cs] = l;
      for (long c = j + 1; c < w; c++)
        tr[c * cs] = nmac(tr[c * cs], l, u[c * us]);
      if (j + 1 < steps) {
        const Cand cd = cand(tile, r, ps, j + 1);
        if (better(next, cd)) next = cd;
      }
    }
    return next;
  }

  /* Every row at its position; perm, or src_of. */
  void store(TP tile, PP pos, device T *a) {
    for (long r = lane; r < rows; r += G)
      for (long c = 0; c < w; c++) a[pos[r] * p.lda + c] = tile[at(r, c)];
    device long *src_of = p.src_of + z * (p.m - p.c0);
    for (long r = lane; r < rows; r += G) {
      if (p.perm)
        p.perm[z * p.sperm + pos[r]] = r0 + r;
      else
        src_of[pos[r]] = r0 + r;
    }
  }
};

/* One unit of a panel launch: a group's matrices, or one group's rows of
   a matrix with P > 1. */
template <typename T, bool DEV>
static inline void lu_unit(constant panel_words &p, long unit,
                           threadgroup uchar *lds, int item, int ln, int sg) {
  typedef typename space<T, DEV>::ptr TP;
  typedef typename space<long, DEV>::ptr PP;
  const bool many = p.P > 1;
  Lu<T, DEV> s(p);
  s.G = many ? NX_LA_THREADS : int(p.G);
  /* G is 16, 32 or the group's threads: a power of two. */
  const int g = item >> metal::ctz(s.G);
  s.lane = item & (s.G - 1);
  s.z = many ? unit_div(unit, p.P, p.pdiv, p.wide)
             : member(unit, p.mats, p.matrices, s.G, g, s.ghost);
  s.q = many ? unit - s.z * p.P : 0;
  if (many) s.ghost = false;
  if (s.z < 0) return;
  const long mp = p.m - p.c0;
  s.w = p.w;
  s.ldt = p.ld;
  s.r0 = s.q * p.per;
  s.rows = imax(0, imin(p.per, mp - s.r0));
  s.steps = imin(mp, s.w);
  /* The candidates, then with P > 1 the pivot row; then each matrix's
     tile and positions, or at [spill]. */
  threadgroup Cand *const scr0 = (threadgroup Cand *)lds;
  threadgroup Cand *const scr1 = (threadgroup Cand *)(lds + p.cands_parity);
  threadgroup T *us = (threadgroup T *)(lds + p.row);
  device uchar *dbase = p.spill + (many ? unit : s.z) * p.spill_bytes;
  threadgroup uchar *lbase = lds + p.head + g * p.spill_bytes;
  const TP tile = space<T, DEV>::at(dbase, lbase);
  const PP pos = space<long, DEV>::at(dbase + p.pos, lbase + p.pos);
  device T *a = (device T *)p.a + s.z * p.sbat + p.c0 * p.lda + p.c0;

  /* One launch runs every step; launch k of several, step k − 1 from the
     slots, then step k's candidates. */
  const long k = p.k;
  device uchar *slots = many ? p.slots + s.z * p.slots_stride : nullptr;
  /* Step j's slot of group i. */
  auto slot = [&](long j, long i) {
    return slots + (j & 1) * p.slots_parity + i * p.slot_bytes;
  };
  /* The last level's candidates, where there are levels. */
  device const uchar *last =
      many && p.lv_in >= 0
          ? p.levels + (s.z * p.lv_stride + p.lv_in) * NX_LA_SLOT_HEAD
          : nullptr;
  const long cands = last ? p.n_in : p.P;
  Cand best = none();
  if (!many || k == 0) {
    s.load(tile, pos, (device const T *)a);
    if (!many) meet(s.G);
    for (long r = s.lane; r < s.rows; r += s.G) {
      const Cand c = s.cand(tile, r, pos[r], 0);
      if (better(best, c)) best = c;
    }
  }
  const long from = many ? imax(k - 1, 0) : 0, to = many ? k : s.steps;
  for (long j = from; j < to; j++) {
    if (many) {
      Cand c = none();
      for (long i = item; i < cands; i += NX_LA_THREADS) {
        const Cand d =
            last ? *(device const Cand *)(last + i * NX_LA_SLOT_HEAD)
                 : *(device const Cand *)slot(j, i);
        if (better(c, d)) c = d;
      }
      c = group_best(c, s.G, scr0, ln, sg);
      device const T *src =
          (device const T *)(slot(j, c.src) + NX_LA_SLOT_HEAD);
      for (long c2 = j + item; c2 < s.w; c2 += NX_LA_THREADS) us[c2] = src[c2];
      barrier();
      best = s.step(tile, pos, us, 1, j, c.pos);
    } else {
      best = group_best(best, s.G, (j & 1) ? scr1 : scr0, ln, sg);
      best = s.step(tile, pos, tile + s.at(best.row, 0), s.col_step(), j,
                    best.pos);
      meet(s.G);
    }
  }
  if (many && k < s.steps) {
    best = group_best(best, s.G, scr1, ln, sg);
    device uchar *mine = slot(k, s.q);
    if (item == 0) *(device Cand *)mine = best;
    device T *vals = (device T *)(mine + NX_LA_SLOT_HEAD);
    if (best.src == s.q)
      for (long c = k + item; c < s.w; c += NX_LA_THREADS)
        vals[c] = tile[s.at(best.row, c)];
    return;
  }
  if (!s.ghost) s.store(tile, pos, a);
}

/* A launch of a level: unit u, matrix u / runs, takes the best of its
   run of NX_LA_THREADS candidates, step k's slots at level 1, and stores
   it whole, so that its src still names the slot that holds its row. */
static inline void lu_level(constant panel_words &p, long unit,
                            threadgroup uchar *lds, int item, int ln, int sg) {
  const long runs = cdiv(p.n_in, NX_LA_THREADS);
  const long z = unit_div(unit, runs, p.runs_div, p.wide),
             r = unit - z * runs;
  const long i = r * NX_LA_THREADS + item;
  device uchar *lv = p.levels + z * p.lv_stride * NX_LA_SLOT_HEAD;
  Cand c = none();
  if (i < p.n_in) {
    device const uchar *slots = p.slots + z * p.slots_stride;
    c = p.level == 1
            ? *(device const Cand *)(slots + (p.k & 1) * p.slots_parity +
                                     i * p.slot_bytes)
            : *(device const Cand *)(lv + (p.lv_in + i) * NX_LA_SLOT_HEAD);
  }
  c = group_best(c, NX_LA_THREADS, (threadgroup Cand *)lds, ln, sg);
  if (item == 0) *(device Cand *)(lv + (p.lv_out + r) * NX_LA_SLOT_HEAD) = c;
}

template <typename T>
static inline void lu_panel(constant panel_words &p, threadgroup uchar *lds,
                            uint gx, uint item, ushort ln, ushort sg) {
  const long unit = p.first + long(gx);
  if (unit >= p.count) return;
  if (p.level > 0) {
    lu_level(p, unit, lds, int(item), int(ln), int(sg));
    return;
  }
  if (p.in_lds)
    lu_unit<T, false>(p, unit, lds, int(item), int(ln), int(sg));
  else
    lu_unit<T, true>(p, unit, lds, int(item), int(ln), int(sg));
}

/* Row interchanges */

struct gather_words {
  device uint *a;
  device const long *piv;
  device const long *src_of;
  device long *perm;
  long sbat, lda, n, spiv, sperm, m, c0, w, steps, fmt, per, pdiv, wide,
      first, count;
};

NX_FIELD(gather_words, a, NX_LA_GATHER_A);
NX_FIELD(gather_words, perm, NX_LA_GATHER_PERM);
NX_FIELD(gather_words, sbat, NX_LA_GATHER_SBAT);
NX_FIELD(gather_words, fmt, NX_LA_GATHER_FMT);
NX_FIELD(gather_words, count, NX_LA_GATHER_COUNT);
NX_SIZE(gather_words, NX_LA_GATHER_WORDS);

/* A panel's interchanges on each matrix's columns outside the panel and
   on perm: rows c0 + t and pivots[c0 + t], t < steps, the only rows that
   moved, take the rows src_of names, all read before any is written. A
   group takes NX_LA_GATHER_SPAN 4-byte words of each of those rows of a
   matrix, from the columns left of the panel then right of it, staged in
   threadgroup memory: the moved rows, the rows they take, then their
   perm values, of 2 NX_LA_LEAF words each, then the words. The
   matrix's first group takes perm too. [W] is the words of an element. */
template <int W>
static inline void gather_t(constant gather_words &w, threadgroup uchar *lds,
                            long u, int item) {
  constexpr int T2 = 2 * NX_LA_LEAF, SPAN = NX_LA_GATHER_SPAN;
  threadgroup long *rows = (threadgroup long *)lds;
  threadgroup long *from = rows + T2;
  threadgroup long *pvals = from + T2;
  threadgroup uint *vals = (threadgroup uint *)(pvals + T2);
  const long mp = w.m - w.c0, t2 = 2 * w.steps;
  const long outside = (w.n - w.w) * W;
  const long z = unit_div(u, w.per, w.pdiv, w.wide), blk = u - z * w.per;
  device const long *piv = w.piv + z * w.spiv + w.c0;
  device const long *src_of = w.src_of + z * mp;
  for (long t = item; t < t2; t += NX_LA_THREADS) {
    const long r = t < w.steps ? t : piv[t - w.steps] - w.c0;
    rows[t] = r;
    from[t] = src_of[r];
  }
  barrier();
  /* Word e of a row: column e / W outside the panel. */
  device uint *a = w.a + (z * w.sbat + w.c0 * w.lda) * W;
  const long e0 = blk * SPAN;
  for (long x = item; x < t2 * SPAN; x += NX_LA_THREADS) {
    const long t = x / SPAN, e = e0 + x % SPAN;
    if (e >= outside) continue;
    const long col = e / W < w.c0 ? e / W : e / W + w.w;
    vals[t * SPAN + x % SPAN] = a[from[t] * w.lda * W + col * W + e % W];
  }
  device long *perm = w.perm + z * w.sperm + w.c0;
  if (blk == 0)
    for (long t = item; t < t2; t += NX_LA_THREADS) pvals[t] = perm[from[t]];
  barrier();
  for (long x = item; x < t2 * SPAN; x += NX_LA_THREADS) {
    const long t = x / SPAN, e = e0 + x % SPAN;
    if (e >= outside) continue;
    const long col = e / W < w.c0 ? e / W : e / W + w.w;
    a[rows[t] * w.lda * W + col * W + e % W] = vals[t * SPAN + x % SPAN];
  }
  if (blk == 0)
    for (long t = item; t < t2; t += NX_LA_THREADS) perm[rows[t]] = pvals[t];
}

/* Cholesky */

struct chol_words {
  device uchar *a;
  device int *fail;
  device const uchar *src;
  long sbat, lda, cs, k, nb, matrices, G, mats, ld, stride, upper, first,
      count;
};

NX_FIELD(chol_words, a, NX_LA_CHOL_A);
NX_FIELD(chol_words, src, NX_LA_CHOL_SRC);
NX_FIELD(chol_words, sbat, NX_LA_CHOL_SBAT);
NX_FIELD(chol_words, upper, NX_LA_CHOL_UPPER);
NX_FIELD(chol_words, count, NX_LA_CHOL_COUNT);
NX_SIZE(chol_words, NX_LA_CHOL_WORDS);

/* L of the diagonal block of order nb at (k, k) of each matrix, in
   place, right-looking: L_jj = sqrt (re a_jj), the column below divided
   by it, then each later element (i, c), c <= i, takes −conj(L_cj) L_ij,
   one nmac. Reads the block's lower triangle, flushed, and the real parts
   of its diagonal alone. [fail] is set for a matrix that meets a pivot
   that is not positive, or NaN. G threads a matrix, [mats] a group. */
template <typename T>
static inline void chol(constant chol_words &w, threadgroup uchar *lds,
                        uint gx, uint item, ushort, ushort) {
  const long u = w.first + long(gx);
  if (u >= w.count) return;
  const int nb = int(w.nb), ldt = int(w.ld), G = int(w.G);
  const bool upper = w.upper != 0;
  const int g = int(item) / G, lane = int(item) % G;
  bool ghost;
  const long z = member(u, w.mats, w.matrices, G, g, ghost);
  if (z < 0) return;
  threadgroup T *tile = (threadgroup T *)(lds + g * w.stride);
  device T *a = (device T *)w.a + z * w.sbat + w.k * (w.lda + w.cs);
  device const T *src = w.src ? (device const T *)w.src + z * w.sbat : a;
  for (int r = 0; r < nb; r++)
    for (int c = lane; c <= r; c += G)
      tile[r * ldt + c] = flush(src[r * w.lda + c * w.cs]);
  meet(G);
  bool failed = false;
  for (int j = 0; j < nb; j++) {
    const float d = re(tile[j * ldt + j]);
    failed |= !(d > 0.0f);
    const float s = sqrt_(d);
    for (int i = j + 1 + lane; i < nb; i += G)
      tile[i * ldt + j] = divd(tile[i * ldt + j], mk<T>(s, 0.0f));
    meet(G);
    if (lane == 0) tile[j * ldt + j] = mk<T>(s, 0.0f);
    for (int i = j + 1 + lane; i < nb; i += G) {
      const T l = tile[i * ldt + j];
      for (int c = j + 1; c <= i; c++)
        tile[i * ldt + c] = nmac(tile[i * ldt + c], conj_(tile[c * ldt + j]), l);
    }
    meet(G);
  }
  if (ghost) return;
  if (w.src) {
    for (int r = 0; r < nb; r++)
      for (int c = lane; c < nb; c += G) {
        const T x = c == r  ? tile[r * ldt + r]
                    : upper ? (c > r ? conj_(tile[c * ldt + r]) : zero<T>())
                            : (c < r ? tile[r * ldt + c] : zero<T>());
        a[r * w.lda + c * w.cs] = failed ? nan_<T>() : x;
      }
    return;
  }
  if (failed && lane == 0) w.fail[z] = 1;
  for (int r = 0; r < nb; r++)
    for (int c = lane; c <= r; c += G) a[r * w.lda + c * w.cs] = tile[r * ldt + c];
}

struct chol_finish_words {
  device uchar *a;
  device const int *fail;
  long n, upper, tr, fmt, first, count;
};

NX_FIELD(chol_finish_words, a, NX_LA_FINISH_A);
NX_FIELD(chol_finish_words, n, NX_LA_FINISH_N);
NX_FIELD(chol_finish_words, count, NX_LA_FINISH_COUNT);
NX_SIZE(chol_finish_words, NX_LA_FINISH_WORDS);

/* Each n × n matrix: NaN where [fail] is set; else under [upper] U = Lᴴ,
   each element below the diagonal moved conjugated above it, else the
   elements above the diagonal +0. The diagonal stays. Under [tr], the
   matrix holds Lᵀ, which it leaves as L or U. */
template <typename T>
static inline void chol_finish_t(constant chol_finish_words &w, long g) {
  const long n = w.n, nn = n * n;
  const bool upper = w.upper != 0, tr = w.tr != 0;
  const long z = g / nn, e = g - z * nn;
  const long i = e / n, j = e - i * n;
  device T *a = (device T *)w.a + z * nn;
  if (w.fail[z]) {
    a[e] = nan_<T>();
    return;
  }
  if (tr) {
    /* Element (i, j) above the diagonal holds L_ji: U_ij = conj(L_ji) in
       place, or L_ji moved below. */
    if (i > j && upper) a[e] = zero<T>();
    if (i >= j) return;
    if (upper) {
      a[e] = conj_(a[e]);
    } else {
      a[j * n + i] = a[e];
      a[e] = zero<T>();
    }
    return;
  }
  if (i <= j) {
    if (i < j && !upper) a[e] = zero<T>();
    return;
  }
  if (!upper) return;
  a[j * n + i] = conj_(a[e]);
  a[e] = zero<T>();
}

/* Triangular solves' diagonal */

struct diag_words {
  device const float *a;
  device float *x;
  device int *flag;
  long sa, sd, n, sx, xn, fmt, first, count;
};

NX_FIELD(diag_words, a, NX_LA_DIAG_A);
NX_FIELD(diag_words, flag, NX_LA_DIAG_FLAG);
NX_FIELD(diag_words, sa, NX_LA_DIAG_SA);
NX_FIELD(diag_words, count, NX_LA_DIAG_COUNT);
NX_SIZE(diag_words, NX_LA_DIAG_WORDS);

/* flag[z] = 1 where a diagonal element of matrix z is zero, read
   flushed; P parts an element. */
static inline void diag_zero_t(constant diag_words &w, long g, int P) {
  const long z = g / w.n, i = g - z * w.n;
  device const float *d = w.a + (z * w.sa + i * w.sd) * P;
  if (flush(d[0]) == 0.0f && (P == 1 || flush(d[1]) == 0.0f)) w.flag[z] = 1;
}

/* x NaN for a flagged matrix. */
static inline void nan_where_t(constant diag_words &w, long g, int P) {
  const long z = g / w.xn, i = g - z * w.xn;
  if (!w.flag[z]) return;
  device float *x = w.x + (z * w.sx + i) * P;
  for (int q = 0; q < P; q++) x[q] = nan_f();
}

/* The identity's rows: perm[g] = g mod m. */
struct iota_words {
  device long *perm;
  long m, first, count;
};

NX_FIELD(iota_words, perm, NX_LA_IOTA_PERM);
NX_FIELD(iota_words, count, NX_LA_IOTA_COUNT);
NX_SIZE(iota_words, NX_LA_IOTA_WORDS);

/* QR */

/* Sums over rows run in this order: the rows of the matrix in blocks of
   NX_LA_QBLK from row 0, each block in 16 lanes by row mod 16, each lane
   from +0 in increasing row, lane l taking lane l + 8, then l + 4, l + 2,
   l + 1; then the blocks from the panel's first, 32 to a superblock
   combined by the same tree over 32 (l + 16 first), superblocks added in
   increasing order from the first. A row outside the sum adds nothing to
   its lane. */
#define QSUP NX_LA_QSUP

template <typename T, bool MAX> static inline T combine(T x, T y) {
  if (MAX) return mk<T>(max_(re(x), re(y)), 0.0f);
  return mk<T>(re(x) + re(y), im(x) + im(y));
}

/* [total] and a value's partials of superblocks [from, to) of [nb]
   blocks, [stride] apart, combined by one SIMD-group: each superblock of
   QSUP by the tree, then in order, superblock 0 its own value. Every
   lane returns it. */
template <typename T, bool MAX, typename PP>
static inline T blocks_run(PP part, long stride, long nb, long from, long to,
                           T total, int ln) {
  const long end = imin(nb, to * QSUP);
  for (long s0 = from * QSUP; s0 < end; s0 += QSUP) {
    const long here = imin(QSUP, nb - s0);
    T x = zero<T>();
    if (ln < here) x = part[(s0 + ln) * stride];
    for (int o = 16; o >= 1; o /= 2) {
      const T y = shfl_t(x, down(ln, o, 32));
      if (ln + o < here) x = combine<T, MAX>(x, y);
    }
    x = shfl_t(x, 0);
    total = s0 == 0 ? x : combine<T, MAX>(total, x);
  }
  return total;
}

/* A value's partials of [nb] blocks combined, as blocks_run takes them. */
template <typename T, bool MAX, typename PP>
static inline T blocks_tree(PP part, long stride, long nb, int ln) {
  return blocks_run<T, MAX>(part, stride, nb, 0, cdiv(nb, QSUP), zero<T>(),
                            ln);
}

/* LAPACK's least normal over the unit roundoff. */
static inline float safmin() { return 0x1p-102f; }

/* Below it, or above its reciprocal, x's largest part takes the scaled
   sums, its squares near the normal range's ends; likewise the product
   of x's largest part with the later columns' (prod_lo), below which the
   largest products keep fewer than u's bits above the least normal. */
static inline float safe_lo() { return 0x1p-30f; }
static inline int prod_lo() { return -100; }

/* sqrt (a² + b² + c²), scaled (xLAPY3). */
static inline float lapy3(float a, float b, float c) {
  float xa = abs_(a), xb = abs_(b), xc = abs_(c);
  const float w = max_(xa, max_(xb, xc));
  if (w == 0.0f || w != w) return xa + xb + xc;
  xa /= w;
  xb /= w;
  xc /= w;
  return w * sqrt_(fma_(xa, xa, fma_(xb, xb, xc * xc)));
}

/* What a reduction sums a row: step j's [MAIN] values over x times an
   exact power of two 2^s (|2^s x|², its largest part, the later columns'
   largest part, conj(2^s x) a_c for each later column), or [GRAM] pairs
   of V's columns from pair v0. A pass takes [ALL] its values, the
   [MAXIMA] (MAIN's values 1 and 2) or the [SUMS] (the others). */
enum { MAIN, GRAM };
enum { ALL, MAXIMA, SUMS };

struct qr_words {
  device uchar *a;
  device uchar *tau;
  device uchar *v;
  device uchar *t;
  device uchar *q;
  device uchar *r;
  device uchar *spill;
  device uchar *parts;
  long sbat, lda, stau, sv, st, ldto, qc, n, m, c0, w, matrices, per, P, G,
      mats, ld, in_lds, parts_in_lds, most, spill_bytes, parts_stride,
      stride, tc, ts, gram, mx, parts_at, tile_at, store, drow, gstore, tot,
      summed, from, to, stage, k, first, count;
};

NX_FIELD(qr_words, a, NX_LA_QR_A);
NX_FIELD(qr_words, parts, NX_LA_QR_PARTS);
NX_FIELD(qr_words, sbat, NX_LA_QR_SBAT);
NX_FIELD(qr_words, G, NX_LA_QR_G);
NX_FIELD(qr_words, tot, NX_LA_QR_TOT);
NX_FIELD(qr_words, stage, NX_LA_QR_STAGE);
NX_FIELD(qr_words, count, NX_LA_QR_COUNT);
NX_SIZE(qr_words, NX_LA_QR_WORDS);

/* Householder QR of the panel of columns [c0, c0 + w) of rows [c0, m) of
   each matrix f, in place: LAPACK's xGEQR2 with xLARFG's reflectors, the
   panel's rows held by P groups as LU's are, in whole NX_LA_QBLK blocks
   of rows. Step j's sums over the rows below the diagonal, Σ|x_i|², the
   largest part of x and s_c = Σ conj(x_i) a_ic for the later columns,
   take one reduction (two where the maxima ask for a scale); then
   v = [1; x σ], σ = 1 / (α − β), w_c = a_jc + conj(σ) s_c, and
   a_ic −= (conj(τ) w_c) v_i, one nmac each. Then the dense V (ones on its
   diagonal, zeros above), where asked, and T with
   H_0 ⋯ H_(w−1) = I − V T Vᴴ (xLARFT), from Vᴴ V by the same reductions.
   Where the panel is the whole matrix (P = 1), it stores the first qc
   columns of Q = I − V (T V1ᴴ) and the qc rows of R. With P > 1 each
   meeting of the groups is a launch boundary (see the stages below): the
   results are P = 1's.

   Threadgroup memory, a matrix, at the offsets the words give: vals
   [most], tc [w], T [w²], the pairs [w (w − 1) / 2], M [w qc] where Q is
   stored; then, where they fit, the reductions' store and the tile.
   [PDEV] and [TDEV] place the partials and the tile in device memory. */
template <typename T, bool PDEV, bool TDEV> struct Qr {
  typedef typename space<T, PDEV>::ptr PP;
  typedef typename space<T, TDEV>::ptr TP;
  constant qr_words &p;
  long z, q, w, ldt, c0, m, most, lo, hi, rows, nblocks, b_first;
  int G, lane, ln;
  bool ghost;
  threadgroup T *vals;
  threadgroup T *tc;
  threadgroup T *ts;
  PP parts;
  PP drow;
  PP tot;
  TP tile;
  int parity;

  Qr(constant qr_words &p_) : p(p_) {}

  void sync() { meet(G); }

  /* V's element (i, c) from the tile's row. */
  T vd(TP row, long i, long c) {
    const long rr = i - c0;
    return rr > c ? row[c] : rr == c ? one<T>() : zero<T>();
  }

  T term(int mode, int v, T acc, TP row, long i, long j, int big, long v0) {
    if (mode == GRAM) {
      const long x = v0 + v;
      long pa = 0;
      while ((pa + 1) * (2 * w - pa - 2) / 2 <= x) pa++;
      const long pb = x - pa * (2 * w - pa - 1) / 2 + pa + 1;
      return mac(acc, conj_(vd(row, i, pa)), vd(row, i, pb));
    }
    T x = row[j];
    if (big != 0) /* big: the exponent of the scale s = 2^big, exact */
      x = mk<T>(ldexp_(re(x), big), ldexp_(im(x), big));
    if (v == 0) return mk<T>(fma_(re(x), re(x), fma_(im(x), im(x), re(acc))), 0.0f);
    if (v == 1) return mk<T>(max_(re(acc), max_(abs_(re(x)), abs_(im(x)))), 0.0f);
    if (v == 2) {
      float most_ = re(acc);
      for (long c = j + 1; c < w; c++)
        most_ = max_(most_, max_(abs_(re(row[c])), abs_(im(row[c]))));
      return mk<T>(most_, 0.0f);
    }
    return mac(acc, conj_(x), row[j + v - 2]);
  }

  static bool maximum(int mode, int v) { return mode == MAIN && (v == 1 || v == 2); }
  static bool chosen(int mode, int v, int sel) {
    return sel == ALL || (sel == MAXIMA) == maximum(mode, v);
  }

  /* This group's partials of [nv] values over the rows from [from], one
     per NX_LA_QBLK block of rows, into the store of [parity]. */
  void partials(int mode, int nv, int sel, long from, long j, int big, long v0) {
    PP pp = parts + parity * p.store;
    const int H = G / 16, h = lane / 16, hl = lane % 16;
    const long my_b0 = lo / NX_LA_QBLK,
               my_nb = rows > 0 ? (hi - 1) / NX_LA_QBLK - my_b0 + 1 : 0;
    for (long t = h; t < my_nb * nv; t += H) {
      const long b = my_b0 + t / nv;
      const int v = int(t % nv);
      if (!chosen(mode, v, sel)) continue;
      const bool mx = maximum(mode, v);
      T acc = zero<T>();
      for (long i = b * NX_LA_QBLK + hl; i < (b + 1) * NX_LA_QBLK; i += 16)
        if (i >= from && i >= lo && i < hi)
          acc = term(mode, v, acc, tile + (i - lo) * ldt, i, j, big, v0);
      for (int o = 8; o >= 1; o /= 2) {
        const T y = shfl_t(acc, down(ln, o, 16));
        acc = mx ? combine<T, true>(acc, y) : combine<T, false>(acc, y);
      }
      if (hl == 0) pp[(b - b_first) * most + v] = acc;
    }
  }

  /* Every group's partials of [parity]'s store combined, into vals.
     Values 1 and 2 of MAIN are maxima. */
  void tree(int mode, int nv, int sel) {
    if (p.summed) {
      for (int v = lane; v < nv; v += G)
        if (chosen(mode, v, sel)) vals[v] = tot[v];
      sync();
      return;
    }
    PP pp = parts + parity * p.store;
    if (G >= 32) {
      for (int v = lane / 32; v < nv; v += G / 32) {
        if (!chosen(mode, v, sel)) continue;
        const T x = maximum(mode, v)
                        ? blocks_tree<T, true>(pp + v, most, nblocks, ln)
                        : blocks_tree<T, false>(pp + v, most, nblocks, ln);
        if (lane % 32 == 0) vals[v] = x;
      }
    } else {
      for (int v = lane; v < nv; v += G) /* one block */
        if (chosen(mode, v, sel)) vals[v] = pp[v];
    }
    sync();
  }

  /* A sum stage's launch: each chosen value's total at [tot] carried over
     superblocks [from, to) of [parity]'s store, in tree's order, so that
     a total is tree's whatever launches its superblocks take. */
  void sum(int mode, int nv, int sel) {
    PP pp = parts + parity * p.store;
    for (int v = lane / 32; v < nv; v += G / 32) {
      if (!chosen(mode, v, sel)) continue;
      const T before = p.from == 0 ? zero<T>() : T(tot[v]);
      const T x = maximum(mode, v)
                      ? blocks_run<T, true>(pp + v, most, nblocks, p.from,
                                            p.to, before, ln)
                      : blocks_run<T, false>(pp + v, most, nblocks, p.from,
                                             p.to, before, ln);
      if (ln == 0) tot[v] = x;
    }
  }

  /* A reduction within one launch (P = 1). */
  void reduce(int mode, int nv, long from, long j, int big, long v0) {
    partials(mode, nv, ALL, from, j, big, v0);
    sync();
    tree(mode, nv, ALL);
    parity ^= 1;
  }

  /* The exponent e of step j's sums, from their maxima: 0, or where x's
     squares, or its largest products with the later columns, are near
     the normal range's ends, the exponent of x's largest part, so that
     the sums over x 2^-e have the columns' own magnitude, as LAPACK's
     v_i a_ic do. */
  int scale(float big, float most_) {
    /* The largest product's exponent, from the factors' own: their
       product may underflow to 0. */
    const int pe =
        most_ != 0.0f && most_ - most_ == 0.0f ? ilogb_(big) + ilogb_(most_) : 0;
    if (big != 0.0f && big - big == 0.0f &&
        (big < safe_lo() || big > 1.0f / safe_lo() || pe < prod_lo() ||
         pe > -prod_lo()))
      return ilogb_(big);
    return 0;
  }

  /* This group's rows of the panel, flushed. */
  void load() {
    device const T *a = (device const T *)p.a + z * p.sbat;
    for (long r = lane; r < rows; r += G)
      for (long c = 0; c < w; c++)
        tile[r * ldt + c] = flush(a[(lo + r) * p.lda + c0 + c]);
    sync();
  }

  /* Step j's diagonal row, from its owner. */
  void diagonal(long j) {
    const long jr = c0 + j;
    PP dr = drow + (j & 1) * w;
    if (jr >= lo && jr < hi)
      for (long c = j + lane; c < w; c += G) dr[c] = tile[(jr - lo) * ldt + c];
  }

  /* Step j from its sums in vals, over x 2^-e: the reflector, xLARFG's
     scaling of x, α and β by 1 / safmin until β is normal (each exact,
     the sums' scale following), τ, and the update of this group's
     rows. */
  void finish(long j, int e) {
    const long jr = c0 + j;
    PP dr = drow + (j & 1) * w;
    const float sf = safmin();
    const T alpha = dr[j];
    float ar = re(alpha), ai = im(alpha), beta = 0.0f;
    float xnorm = ldexp_(sqrt_(re(vals[0])), e);
    const bool reflect = !(xnorm == 0.0f && ai == 0.0f);
    int knt = 0;
    if (reflect) {
      beta = -copysign_(lapy3(ar, ai, xnorm), ar);
      while (abs_(beta) < sf && knt < 20) {
        knt++;
        for (long i = imax(lo, jr + 1) + lane; i < hi; i += G) {
          const T x = tile[(i - lo) * ldt + j];
          tile[(i - lo) * ldt + j] = mk<T>(re(x) * (1.0f / sf), im(x) * (1.0f / sf));
        }
        ar *= 1.0f / sf;
        ai *= 1.0f / sf;
        /* A zero x's sums are zero at any scale: e stays, σ's scaled copy
           finite. */
        if (re(vals[1]) != 0.0f) e -= ilogb_(sf);
        xnorm = ldexp_(sqrt_(re(vals[0])), e);
        beta = -copysign_(lapy3(ar, ai, xnorm), ar);
      }
    }
    T tau = zero<T>(), sigma = zero<T>();
    if (reflect) {
      tau = mk<T>((beta - ar) / beta, -ai / beta);
      sigma = divd(one<T>(), mk<T>(ar - beta, ai));
      for (int c = 0; c < knt; c++) beta *= sf;
      /* w_c = a_jc + conj(σ) s_c, s_c = 2^e times the sums. */
      const T se = mk<T>(ldexp_(re(sigma), e), ldexp_(im(sigma), e));
      for (long c = j + 1 + lane; c < w; c += G)
        tc[c] = mul(conj_(tau), mac(dr[c], conj_(se), vals[c - j + 2]));
    }
    if (lane == 0 && q == 0 && !ghost)
      ((device T *)p.tau)[z * p.stau + c0 + j] = tau;
    ts[j * w + j] = tau; /* T's diagonal, for T below */
    sync();
    for (long i = imax(lo, jr) + lane; i < hi && reflect; i += G) {
      TP row = tile + (i - lo) * ldt;
      if (i == jr) {
        row[j] = mk<T>(beta, 0.0f);
        for (long c = j + 1; c < w; c++)
          row[c] = mk<T>(re(row[c]) - re(tc[c]), im(row[c]) - im(tc[c]));
        continue;
      }
      const T v = mul(T(row[j]), sigma);
      row[j] = v;
      for (long c = j + 1; c < w; c++) row[c] = nmac(T(row[c]), T(tc[c]), v);
    }
    sync();
  }

  /* The factored panel and dense V from this group's rows. */
  void panel() {
    device T *a = (device T *)p.a + z * p.sbat;
    device T *vo = p.v ? (device T *)p.v + z * p.sv : nullptr;
    for (long r = lane; r < rows && !p.q && !ghost; r += G)
      for (long c = 0; c < w; c++) {
        a[(lo + r) * p.lda + c0 + c] = tile[r * ldt + c];
        if (vo) vo[(lo + r - c0) * w + c] = vd(tile + r * ldt, lo + r, c);
      }
  }

  /* T: T_ii = τ_i, T[0:i, i] = −τ_i T[0:i, 0:i] g[0:i, i] from the pairs
     g, a column a turn, lane r its row r; one SIMD-group, or the 16
     lanes. */
  void larft(threadgroup const T *gram) {
    const int width = G < 32 ? G : 32;
    if (lane < width) {
      for (long i = 0; i < w; i++) {
        const T ti = ts[i * w + i];
        for (long r = lane; r < w; r += width) {
          if (r >= i) {
            if (r > i) ts[r * w + i] = zero<T>();
            continue;
          }
          T x = zero<T>();
          for (long c = r; c < i; c++)
            x = mac(x, T(ts[r * w + c]), T(gram[c * (2 * w - c - 1) / 2 + (i - c - 1)]));
          ts[r * w + i] = mul(neg(ti), x);
        }
        wave_sync();
      }
    }
    sync();
    if (p.t && q == 0 && !ghost) {
      device T *to = (device T *)p.t + z * p.st;
      for (long r = 0; r < w; r++)
        for (long c = lane; c < w; c += G) to[r * p.ldto + c] = ts[r * w + c];
    }
  }
};

/* One unit of a QR panel launch: a group's matrices, or one group's rows
   of a matrix with P > 1. */
template <typename T, bool PDEV, bool TDEV>
static inline void qr_unit(constant qr_words &p, long unit,
                           threadgroup uchar *lds, int item, int ln) {
  typedef typename space<T, PDEV>::ptr PP;
  const bool many = p.P > 1;
  /* A sum stage's unit is a matrix. */
  const bool summing = p.stage >= NX_LA_QR_SUM_B;
  const int G = many ? NX_LA_THREADS : int(p.G);
  const int g = item / G, lane = item % G;
  Qr<T, PDEV, TDEV> s(p);
  s.z = summing ? unit
        : many  ? unit / p.P
                : member(unit, p.mats, p.matrices, G, g, s.ghost);
  s.q = many && !summing ? unit % p.P : 0;
  if (many) s.ghost = false;
  if (s.z < 0) return;
  const long w = p.w, c0 = p.c0, m = p.m, most = p.most, z = s.z;
  s.w = w;
  s.ldt = p.ld;
  s.c0 = c0;
  s.m = m;
  s.most = most;
  s.G = G;
  s.lane = lane;
  s.ln = ln;
  s.b_first = c0 / NX_LA_QBLK;
  s.nblocks = (m - 1) / NX_LA_QBLK - s.b_first + 1;
  const long bpc = p.per / NX_LA_QBLK;
  s.lo = imax(c0, (s.b_first + s.q * bpc) * NX_LA_QBLK);
  s.hi = imin(m, (s.b_first + (s.q + 1) * bpc) * NX_LA_QBLK);
  s.rows = imax(0, s.hi - s.lo);
  const long npair = w * (w - 1) / 2;
  threadgroup uchar *mine = lds + g * p.stride;
  threadgroup T *vals = (threadgroup T *)mine;
  threadgroup T *ts = (threadgroup T *)(mine + p.ts);
  threadgroup T *gram = (threadgroup T *)(mine + p.gram);
  threadgroup T *M = (threadgroup T *)(mine + p.mx);
  s.vals = vals;
  s.tc = (threadgroup T *)(mine + p.tc);
  s.ts = ts;
  s.parts = space<T, PDEV>::at(
      (device uchar *)((device T *)p.parts + z * p.parts_stride),
      mine + p.parts_at);
  s.tile = space<T, TDEV>::at(p.spill + (many ? unit : z) * p.spill_bytes,
                              mine + p.tile_at);
  s.drow = s.parts + p.drow;    /* two of w, by step parity */
  s.tot = s.parts + p.tot;      /* where summed, a meeting's totals */
  PP gstore = s.parts + p.gstore; /* with P > 1, the pairs */
  s.parity = 0;
  const long steps = imin(m - c0, w);
  /* Gram pairs' batches, and step j's values. */
  const long batches = cdiv(npair, most);
  auto nv = [&](long j) { return int(w - j + 2); };
  auto nvg = [&](long i) { return int(imin(most, npair - i * most)); };

  if (many) {
    const long k = p.k;
    if (p.stage == NX_LA_QR_SUM_B) {
      s.parity = int(k & 1);
      s.sum(MAIN, nv(k), MAXIMA);
      return;
    }
    if (p.stage == NX_LA_QR_SUM_A) {
      s.parity = int((k - 1) & 1);
      s.sum(MAIN, nv(k - 1), SUMS);
      return;
    }
    if (p.stage == NX_LA_QR_SUM_G) {
      s.parity = int((steps + k - 1) & 1);
      s.sum(GRAM, nvg(k - 1), ALL);
      return;
    }
    if (p.stage == NX_LA_QR_STEP_B) {
      /* Step k's sums at the scale its maxima give. */
      s.parity = int(k & 1);
      s.tree(MAIN, nv(k), MAXIMA);
      const int e = s.scale(re(vals[1]), re(vals[2]));
      s.partials(MAIN, nv(k), SUMS, c0 + k + 1, k, -e, 0);
      return;
    }
    if (p.stage == NX_LA_QR_STEP_A) {
      if (k == 0) {
        s.load();
      } else {
        s.parity = int((k - 1) & 1);
        s.tree(MAIN, nv(k - 1), MAXIMA);
        const int e = s.scale(re(vals[1]), re(vals[2]));
        s.tree(MAIN, nv(k - 1), SUMS);
        s.finish(k - 1, e);
      }
      if (k < steps) {
        s.diagonal(k);
        s.parity = int(k & 1);
        s.partials(MAIN, nv(k), MAXIMA, c0 + k + 1, k, 0, 0);
        return;
      }
      s.panel();
      if (!p.t) return;
      s.parity = int(steps & 1);
      s.partials(GRAM, nvg(0), ALL, c0, 0, 0, 0);
      return;
    }
    /* NX_LA_QR_STEP_G, k = i: batch i − 1 into the store, batch i's sums;
       after the last, T. */
    if (s.q == 0) {
      s.parity = int((steps + k - 1) & 1);
      s.tree(GRAM, nvg(k - 1), ALL);
      for (int v = lane; v < nvg(k - 1); v += G) gstore[(k - 1) * most + v] = vals[v];
    }
    if (k < batches) {
      s.parity = int((steps + k) & 1);
      s.partials(GRAM, nvg(k), ALL, c0, 0, 0, k * most);
      return;
    }
    if (s.q != 0) return;
    s.sync();
    for (long v = lane; v < npair; v += G) gram[v] = gstore[v];
    for (long i = lane; i < w; i += G)
      ts[i * w + i] = ((device const T *)p.tau)[z * p.stau + c0 + i];
    s.sync();
    s.larft(gram);
    return;
  }

  s.load();
  for (long j = 0; j < steps; j++) {
    s.diagonal(j);
    s.reduce(MAIN, nv(j), c0 + j + 1, j, 0, 0);
    const int e = s.scale(re(vals[1]), re(vals[2]));
    /* At G = 16 the SIMD-group's two matrices take the pass where either
       asks: at e = 0 it gives the same sums again. */
    if (metal::simd_any(e != 0)) s.reduce(MAIN, nv(j), c0 + j + 1, j, -e, 0);
    s.finish(j, e);
  }

  /* The factored panel; dense V; T from Vᴴ V. */
  s.panel();
  if (!p.t && !p.q) return;
  /* Pair v = (pa < pb), row-major: Σ_i conj(V_i,pa) V_i,pb. */
  for (long i = 0; i < batches; i++) {
    s.reduce(GRAM, nvg(i), c0, 0, 0, i * most);
    for (int v = lane; v < nvg(i); v += G) gram[i * most + v] = vals[v];
    s.sync();
  }
  s.larft(gram);
  if (!p.q) return;
  /* The whole matrix: M = T V1ᴴ (steps × qc), Q = I − V M, R. A panel
     wider than its rows has [steps] reflectors: T's later columns were
     never written, so no sum reads them. */
  const long qc = p.qc;
  for (long c = lane; c < qc; c += G)
    for (long r = 0; r < steps; r++) {
      T x = zero<T>();
      for (long s2 = r; s2 < steps; s2++)
        x = mac(x, T(ts[r * w + s2]), conj_(s.vd(s.tile + c * s.ldt, c, s2)));
      M[r * qc + c] = x;
    }
  s.sync();
  if (s.ghost) return;
  device T *qo = (device T *)p.q + z * m * qc;
  for (long i = 0; i < m; i++)
    for (long c = lane; c < qc; c += G) {
      T x = i == c ? one<T>() : zero<T>();
      for (long s2 = 0; s2 < steps && s2 <= i; s2++)
        x = nmac(x, s.vd(s.tile + i * s.ldt, i, s2), T(M[s2 * qc + c]));
      qo[i * qc + c] = x;
    }
  device T *ro = (device T *)p.r + z * qc * p.n;
  for (long i = 0; i < qc; i++)
    for (long c = lane; c < p.n; c += G)
      ro[i * p.n + c] = c >= i ? T(s.tile[i * s.ldt + c]) : zero<T>();
}

template <typename T>
static inline void qr_panel(constant qr_words &p, threadgroup uchar *lds,
                            uint gx, uint item, ushort ln, ushort) {
  const long unit = p.first + long(gx);
  if (unit >= p.count) return;
  if (!p.parts_in_lds)
    qr_unit<T, true, true>(p, unit, lds, int(item), int(ln));
  else if (!p.in_lds)
    qr_unit<T, false, true>(p, unit, lds, int(item), int(ln));
  else
    qr_unit<T, false, false>(p, unit, lds, int(item), int(ln));
}

/* QR's other steps */

struct qr_out_words {
  device const float *f;
  device float *q;
  device float *r;
  long m, n, qc, fmt, first, count;
};

NX_FIELD(qr_out_words, f, NX_LA_QR_OUT_F);
NX_FIELD(qr_out_words, m, NX_LA_QR_OUT_M);
NX_FIELD(qr_out_words, count, NX_LA_QR_OUT_COUNT);
NX_SIZE(qr_out_words, NX_LA_QR_OUT_WORDS);

/* q = the first qc columns of the m × m identity; r = the upper triangle
   of f's first rows (rows past m zero). P parts an element. f is copied
   unflushed: it is the factorisation's own result, never an operand. */
static inline void qr_out_t(constant qr_out_words &w, long g, int P) {
  const long m = w.m, n = w.n, qc = w.qc;
  const long nq = m * qc, nr = qc * n;
  const long z = g / (nq + nr), e = g - z * (nq + nr);
  if (e < nq) {
    device float *x = w.q + (z * nq + e) * P;
    const long row = e / qc, col = e - row * qc;
    x[0] = row == col ? 1.0f : 0.0f;
    if (P == 2) x[1] = 0.0f;
    return;
  }
  const long i = (e - nq) / n, c = (e - nq) - i * n;
  device float *x = w.r + (z * nr + e - nq) * P;
  const bool kept = c >= i && i < m && w.f;
  for (int q = 0; q < P; q++) x[q] = kept ? w.f[(z * m * n + i * n + c) * P + q] : 0.0f;
}

struct dense_words {
  device const float *f;
  device float *v;
  long m, n, c0, w, fmt, first, count;
};

NX_FIELD(dense_words, f, NX_LA_DENSE_F);
NX_FIELD(dense_words, m, NX_LA_DENSE_M);
NX_FIELD(dense_words, count, NX_LA_DENSE_COUNT);
NX_SIZE(dense_words, NX_LA_DENSE_WORDS);

/* Dense V of a panel from f's columns [c0, c0 + w): rows [c0, m). f is
   copied unflushed: it is the factorisation's own result, never an
   operand. */
static inline void dense_v_t(constant dense_words &w, long g, int P) {
  const long h = w.m - w.c0, nv = h * w.w;
  const long z = g / nv, e = g - z * nv;
  const long r = e / w.w, c = e - r * w.w;
  device const float *x = w.f + (z * w.m * w.n + (w.c0 + r) * w.n + w.c0 + c) * P;
  device float *v = w.v + (z * nv + e) * P;
  v[0] = r > c ? x[0] : r == c ? 1.0f : 0.0f;
  if (P == 2) v[1] = r > c ? x[1] : 0.0f;
}

} /* namespace la */

/* Entries */

#define LA_ARGS(words)                                                        \
  constant la::words &w [[buffer(0)]],                                        \
      threadgroup uchar *lds [[threadgroup(0)]],                              \
      uint gx [[threadgroup_position_in_grid]],                               \
      uint item [[thread_index_in_threadgroup]]

/* LA_ARGS with the thread's lane in its SIMD-group and the SIMD-group's
   index in the group. */
#define LA_SIMD_ARGS(words)                                                   \
  LA_ARGS(words), ushort ln [[thread_index_in_simdgroup]],                    \
      ushort sg [[simdgroup_index_in_threadgroup]]

#define LA_FORMATS(name, threads, words, entry)                               \
  NX_KERNEL(threads) void name##_f32(LA_SIMD_ARGS(words)) {                   \
    la::entry<float>(w, lds, gx, item, ln, sg);                               \
  }                                                                           \
  NX_KERNEL(threads) void name##_c64(LA_SIMD_ARGS(words)) {                   \
    la::entry<la::c64>(w, lds, gx, item, ln, sg);                             \
  }

LA_FORMATS(la_trsm, NX_LA_TRSM_THREADS, trsm_words, trsm)
LA_FORMATS(la_lu_panel, NX_LA_THREADS, panel_words, lu_panel)
LA_FORMATS(la_chol, NX_LA_THREADS, chol_words, chol)
LA_FORMATS(la_qr_panel, NX_LA_THREADS, qr_words, qr_panel)

/* A thread an element: [g] from the launch's first, below its count. */
#define LA_ELEMENT(g)                                                         \
  const long g = w.first + long(gx) * NX_LA_THREADS + item;                   \
  if (g >= w.count) return

NX_KERNEL(NX_LA_THREADS) void la_gather(LA_ARGS(gather_words)) {
  const long u = w.first + long(gx);
  if (u >= w.count) return;
  if (w.fmt == NX_LA_F32)
    la::gather_t<1>(w, lds, u, int(item));
  else
    la::gather_t<2>(w, lds, u, int(item));
}

NX_KERNEL(NX_LA_THREADS) void la_chol_finish(LA_ARGS(chol_finish_words)) {
  LA_ELEMENT(g);
  if (w.fmt == NX_LA_F32)
    la::chol_finish_t<float>(w, g);
  else
    la::chol_finish_t<la::c64>(w, g);
}

NX_KERNEL(NX_LA_THREADS) void la_diag_zero(LA_ARGS(diag_words)) {
  LA_ELEMENT(g);
  la::diag_zero_t(w, g, w.fmt == NX_LA_F32 ? 1 : 2);
}

NX_KERNEL(NX_LA_THREADS) void la_nan_where(LA_ARGS(diag_words)) {
  LA_ELEMENT(g);
  la::nan_where_t(w, g, w.fmt == NX_LA_F32 ? 1 : 2);
}

NX_KERNEL(NX_LA_THREADS) void la_qr_out(LA_ARGS(qr_out_words)) {
  LA_ELEMENT(g);
  la::qr_out_t(w, g, w.fmt == NX_LA_F32 ? 1 : 2);
}

NX_KERNEL(NX_LA_THREADS) void la_dense_v(LA_ARGS(dense_words)) {
  LA_ELEMENT(g);
  la::dense_v_t(w, g, w.fmt == NX_LA_F32 ? 1 : 2);
}

NX_KERNEL(NX_LA_THREADS) void la_iota(LA_ARGS(iota_words)) {
  LA_ELEMENT(g);
  w.perm[g] = g % w.m;
}
