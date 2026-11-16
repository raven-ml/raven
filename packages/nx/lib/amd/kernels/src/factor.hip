/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The factorisations' steps: cholesky, lu, qr and solve_triangular,
   batched, at float32, float64, complex64 and complex128 (kernels.h,
   Factorisations). factor.ml plans them.

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

   Steps:
   - trsm: a triangular solve of at most leaf_rows rows, a thread a column.
   - lu_panel: partial pivoting on a panel of columns, its rows held by
     one group, or by P groups over one launch a column (no group waits
     on another group of its launch).
   - gather: a panel's row interchanges on the other columns and on perm.
   - chol: Cholesky of a diagonal block held in LDS.
   - qr_panel and the finishing steps: see their sections.

   A matrix's group of G work-items is 16 or 32 lanes of one wave, several
   matrices a group, or the whole group. The lanes of a wave run in step,
   so G <= 32 meets at a wave's fence; the whole group at a barrier. */

#include <stdint.h>

#include "device.h"

/* nx_kinds.h's complex division, whose functions are inlined as every
   other device function is. */
#pragma clang attribute push(__attribute__((always_inline)), apply_to = function)
#include <math.h>
#include "nx_kinds.h"
#pragma clang attribute pop

namespace la {

/* A member function, inlined as every device function is. */
#define MEMBER inline __attribute__((device, always_inline))

/* Scalars. T is float, double, c64 or c128 (nx_kinds.h's complex types),
   R its part's type. Every update of the routines is mac or nmac: one
   fused multiply-add on reals; on complex numbers two a part, the real
   part taking ar br then ai (-bi), the imaginary part ar bi then ai br,
   the order nx_kernel.mli numbers a complex contraction's terms in. */

typedef nx_c64 c64;
typedef nx_c128 c128;

template <typename T> struct Part { typedef T type; };
template <> struct Part<c64> { typedef float type; };
template <> struct Part<c128> { typedef double type; };
template <typename T> using R_of = typename Part<T>::type;

template <typename T> constexpr bool cplx = false;
template <> constexpr bool cplx<c64> = true;
template <> constexpr bool cplx<c128> = true;

DEVICE float fma_(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
DEVICE double fma_(double a, double b, double c) { return __builtin_fma(a, b, c); }
DEVICE float abs_(float x) { return __builtin_fabsf(x); }
DEVICE double abs_(double x) { return __builtin_fabs(x); }
DEVICE float sqrt_(float x) { return __builtin_sqrtf(x); }
DEVICE double sqrt_(double x) { return __builtin_sqrt(x); }
DEVICE float ldexp_(float x, int e) { return __builtin_ldexpf(x, e); }
DEVICE double ldexp_(double x, int e) { return __builtin_ldexp(x, e); }
DEVICE float copysign_(float x, float y) { return __builtin_copysignf(x, y); }
DEVICE double copysign_(double x, double y) { return __builtin_copysign(x, y); }

/* The exponent e of a finite nonzero x = m 2^e, 1 <= |m| < 2, subnormal
   or not. */
DEVICE int ilogb_(float x) { return __builtin_amdgcn_frexp_expf(x) - 1; }
DEVICE int ilogb_(double x) { return __builtin_amdgcn_frexp_exp(x) - 1; }

/* The larger of a and b, the other one where one is NaN: chosen here, not
   by the instruction's NaN rule. */
template <typename R> DEVICE R max_(R a, R b) {
  if (a != a) return b;
  if (b != b) return a;
  return a < b ? b : a;
}

DEVICE float re(float x) { return x; }
DEVICE double re(double x) { return x; }
DEVICE float re(c64 x) { return x.re; }
DEVICE double re(c128 x) { return x.re; }
DEVICE float im(float) { return 0; }
DEVICE double im(double) { return 0; }
DEVICE float im(c64 x) { return x.im; }
DEVICE double im(c128 x) { return x.im; }

template <typename T> DEVICE T mk(R_of<T> r, R_of<T> i) {
  if constexpr (cplx<T>) {
    T z;
    z.re = r;
    z.im = i;
    return z;
  } else {
    (void)i;
    return r;
  }
}

template <typename T> DEVICE T zero() { return mk<T>(0, 0); }
template <typename T> DEVICE T one() { return mk<T>(1, 0); }
template <typename T> DEVICE T nan_() { return mk<T>(NAN, NAN); }
template <typename T> DEVICE T neg(T x) { return mk<T>(-re(x), -im(x)); }
template <typename T> DEVICE T conj_(T x) { return mk<T>(re(x), -im(x)); }
template <typename T> DEVICE T conj_if(T x, bool c) { return c ? conj_(x) : x; }
template <typename T> DEVICE bool is_zero(T x) { return re(x) == 0 && im(x) == 0; }

/* c + a·b. */
template <typename T> DEVICE T mac(T c, T a, T b) {
  if constexpr (cplx<T>) {
    R_of<T> r = fma_(a.re, b.re, c.re);
    r = fma_(-a.im, b.im, r);
    R_of<T> i = fma_(a.re, b.im, c.im);
    i = fma_(a.im, b.re, i);
    return mk<T>(r, i);
  } else {
    return fma_(a, b, c);
  }
}

/* c − a·b, as c + (−a)·b. */
template <typename T> DEVICE T nmac(T c, T a, T b) { return mac(c, neg(a), b); }
template <typename T> DEVICE T mul(T a, T b) { return mac(zero<T>(), a, b); }

/* LU's magnitude: |re| + |im|. */
template <typename T> DEVICE R_of<T> mag(T x) {
  if constexpr (cplx<T>) return abs_(re(x)) + abs_(im(x));
  else return abs_(x);
}

/* x / d: on complex numbers each part divided by d's real part where d's
   imaginary part is zero, as Cholesky's diagonal is, else Smith's
   quotient (nx_kinds.h's Fdiv). */
DEVICE float divd(float x, float d) { return x / d; }
DEVICE double divd(double x, double d) { return x / d; }
DEVICE c64 divd(c64 x, c64 d) {
  if (d.im == 0) return mk<c64>(x.re / d.re, x.im / d.re);
  return nx_fdiv_c64(x, d);
}
DEVICE c128 divd(c128 x, c128 d) {
  if (d.im == 0) return mk<c128>(x.re / d.re, x.im / d.re);
  return nx_fdiv_c128(x, d);
}

/* The most rows of a triangular solve's leaf. */
template <typename T> constexpr int leaf_rows() {
  return sizeof(T) == 16 ? NX_LA_LEAF / 2 : NX_LA_LEAF;
}

DEVICE int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

/* The group's LDS beyond its static variables, as the launch sizes it. */
extern "C" __attribute__((shared, aligned(16))) uint8_t nx_la_lds[];

/* Words

   A group stages its kernel's words into LDS once ({!stage}), before any
   other work, and reads them there, so a loop that reads a word reads
   LDS. */

constexpr int cmax(int a, int b) { return a < b ? b : a; }

constexpr int staged_most = cmax(
    cmax(cmax(NX_LA_TRSM_WORDS, NX_LA_PANEL_WORDS),
         cmax(NX_LA_CHOL_WORDS, NX_LA_QR_WORDS)),
    cmax(cmax(NX_LA_GATHER_WORDS, NX_LA_FINISH_WORDS),
         cmax(cmax(NX_LA_DIAG_WORDS, NX_LA_QR_OUT_WORDS),
              cmax(NX_LA_DENSE_WORDS, NX_LA_IOTA_WORDS))));

static SHARED int64_t la_staged[staged_most];

/* The kernel's first [n] words, staged by NX_LA_TRSM_THREADS work-items,
   the fewest a factor kernel's group has. */
DEVICE const int64_t *staged(int n) {
  stage(la_staged, n, NX_LA_TRSM_THREADS);
  return la_staged;
}

/* Lanes

   A value of another lane of the wave: [ds_bpermute] moves 32 bits, so
   wider values move in halves. */

DEVICE int32_t shfl(int32_t v, int src) {
  return __builtin_amdgcn_ds_bpermute(src << 2, v);
}
DEVICE int64_t shfl(int64_t v, int src) {
  const int32_t lo = shfl((int32_t)(uint32_t)v, src);
  const int32_t hi = shfl((int32_t)(uint32_t)((uint64_t)v >> 32), src);
  return (int64_t)(((uint64_t)(uint32_t)hi << 32) | (uint32_t)lo);
}
DEVICE float shfl(float v, int src) {
  return __builtin_bit_cast(float, shfl(__builtin_bit_cast(int32_t, v), src));
}
DEVICE double shfl(double v, int src) {
  return __builtin_bit_cast(double, shfl(__builtin_bit_cast(int64_t, v), src));
}
template <typename T> DEVICE T shfl_t(T x, int src) {
  return mk<T>(shfl(re(x), src), shfl(im(x), src));
}

/* The lane [o] above this one within its segment of [width] lanes, or
   this one past the segment's end. */
DEVICE int down(int o, int width) {
  const int l = (int)lane();
  return l % width + o < width ? l + o : l;
}

/* The wave's lanes meet: each then sees the LDS the others wrote before.
   The wave's instructions run in order, so this orders the compiler's. */
DEVICE void wave_sync(void) {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

/* A matrix's G work-items meet: within a wave, or the whole group. */
DEVICE void meet(int G) {
  if (G <= 32) wave_sync();
  else barrier();
}

/* Triangular solve leaf */

/* x ← op(l)⁻¹ x for the lower triangular n × n view op(l), n <=
   leaf_rows, and the n × r view x of each matrix, a thread a column: x_i
   takes −l_ip x_p for p < i in increasing p, one nmac each, then is
   divided by l_ii unless [unit]. op(l)'s element (i, p) is l's at
   i·lrs + p·lcs, conjugated under [conj]: an upper triangle or an adjoint
   is a view of reversed or swapped strides, so its solve runs in its own
   order. Column g of the launch is column g mod r of matrix g / r. */
template <typename T> DEVICE void trsm(void) {
  const int64_t *w = staged(NX_LA_TRSM_WORDS);
  const T *l0 = address<const T>(w, NX_LA_TRSM_L);
  T *x0 = address<T>(w, NX_LA_TRSM_X);
  const T *src0 = address<const T>(w, NX_LA_TRSM_SRC);
  const int64_t sl = uniform(w[NX_LA_TRSM_SL]), lrs = uniform(w[NX_LA_TRSM_LRS]),
                lcs = uniform(w[NX_LA_TRSM_LCS]), sx = uniform(w[NX_LA_TRSM_SX]),
                xrs = uniform(w[NX_LA_TRSM_XRS]), xcs = uniform(w[NX_LA_TRSM_XCS]);
  const int n = (int)uniform(w[NX_LA_TRSM_N]);
  const int64_t r = uniform(w[NX_LA_TRSM_R]), total = uniform(w[NX_LA_TRSM_MATRICES]) * r;
  const bool conj = uniform(w[NX_LA_TRSM_CONJ]) != 0, unit = uniform(w[NX_LA_TRSM_UNIT]) != 0,
             whole = uniform(w[NX_LA_TRSM_WHOLE]) != 0;
  const int64_t stride = uniform(w[NX_LA_TRSM_GROUPS]) * NX_LA_TRSM_THREADS;
  T *xs = (T *)nx_la_lds; /* xs[i * NX_LA_TRSM_THREADS + item()] */
  const int tid = (int)item();
  const bool small = is_small(total);

  for (int64_t g = (int64_t)group_x() * NX_LA_TRSM_THREADS + tid; g < total;
       g += stride) {
    int64_t z = g;
    const int64_t col = split(z, r, small);
    const T *l = l0 + z * sl;
    T *x = x0 + z * sx + col * xcs;
    const T *b = src0 ? src0 + z * sx + col * xcs : x;
    for (int i = 0; i < n; i++) xs[i * NX_LA_TRSM_THREADS + tid] = b[i * xrs];
    bool zero_diag = false;
    for (int i = 0; i < n; i++) {
      T v = xs[i * NX_LA_TRSM_THREADS + tid];
      const T *li = l + i * lrs;
      for (int q = 0; q < i; q++)
        v = nmac(v, conj_if(li[q * lcs], conj), xs[q * NX_LA_TRSM_THREADS + tid]);
      if (!unit) {
        const T d = conj_if(li[i * lcs], conj);
        zero_diag |= is_zero(d);
        v = divd(v, d);
      }
      xs[i * NX_LA_TRSM_THREADS + tid] = v;
    }
    for (int i = 0; i < n; i++)
      x[i * xrs] = whole && zero_diag ? nan_<T>() : xs[i * NX_LA_TRSM_THREADS + tid];
  }
}

/* LU panel */

/* A pivot candidate: [forced] where it is the row at the step's own
   position and its magnitude is NaN, which the search never leaves; else
   its magnitude, -1 for NaN, which nothing exceeds and which exceeds no
   number; then its position, the lesser winning a tie. [row] is its row
   in the group's tile, [src] its group. The order is total, so any tree
   of comparisons picks the same row: the sequential search's. */
template <typename R> struct Cand {
  R key;
  int64_t pos;
  int32_t forced, src;
  int64_t row;
};

static_assert(sizeof(Cand<float>) <= NX_LA_SLOT_HEAD && sizeof(Cand<double>) <= NX_LA_SLOT_HEAD,
              "a candidate fits a slot's head and the host's candidates' bytes");

template <typename R> DEVICE bool better(const Cand<R> &a, const Cand<R> &b) {
  if (a.forced != b.forced) return b.forced > a.forced;
  if (a.key != b.key) return b.key > a.key;
  return b.pos < a.pos;
}

template <typename R> DEVICE Cand<R> none() {
  Cand<R> c;
  c.key = -2;
  c.pos = INT64_MAX;
  c.forced = 0;
  c.src = -1;
  c.row = -1;
  return c;
}

template <typename R> DEVICE Cand<R> shfl_cand(const Cand<R> &c, int src) {
  Cand<R> d;
  d.key = shfl(c.key, src);
  d.pos = shfl(c.pos, src);
  d.forced = shfl(c.forced, src);
  d.src = shfl(c.src, src);
  d.row = shfl(c.row, src);
  return d;
}

/* The best candidate of a matrix's G work-items: G of 16 or 32 lanes, or
   the whole group, through [s], a slot a wave, of the parity a step
   alternates so that its writes never meet the last step's reads. */
template <typename R>
DEVICE Cand<R> group_best(Cand<R> c, int G, Cand<R> *s) {
  const int width = G < 32 ? G : 32;
  for (int o = width / 2; o >= 1; o /= 2) {
    Cand<R> d = shfl_cand(c, (int)lane() ^ o);
    if (better(c, d)) c = d;
  }
  if (G <= 32) return c;
  const int waves = NX_LA_THREADS / 32, wv = (int)item() / 32;
  if ((int)lane() == 0) s[wv] = c;
  barrier();
  c = s[0];
  for (int q = 1; q < waves; q++)
    if (better(c, s[q])) c = s[q];
  return c;
}

/* The words of a panel launch. */
struct PanelWords {
  uint8_t *a, *spill, *slots;
  int64_t *piv, *src_of, *perm;
  const uint8_t *src;
  int64_t sbat, lda, spiv, sperm, m, c0, w, matrices, per, P, ld,
      spill_bytes, slot_bytes, slots_stride, slots_parity, cands_parity, row, head, pos, k;
  int G, mats;
  bool in_lds;
};

DEVICE PanelWords panel_words(void) {
  const int64_t *w = staged(NX_LA_PANEL_WORDS);
  PanelWords p;
  p.a = address<uint8_t>(w, NX_LA_PANEL_A);
  p.piv = address<int64_t>(w, NX_LA_PANEL_PIV);
  p.src_of = address<int64_t>(w, NX_LA_PANEL_SRC_OF);
  p.src = address<const uint8_t>(w, NX_LA_PANEL_SRC);
  p.perm = address<int64_t>(w, NX_LA_PANEL_PERM);
  p.spill = address<uint8_t>(w, NX_LA_PANEL_SPILL);
  p.slots = address<uint8_t>(w, NX_LA_PANEL_SLOTS);
  p.sbat = uniform(w[NX_LA_PANEL_SBAT]);
  p.lda = uniform(w[NX_LA_PANEL_LDA]);
  p.spiv = uniform(w[NX_LA_PANEL_SPIV]);
  p.sperm = uniform(w[NX_LA_PANEL_SPERM]);
  p.m = uniform(w[NX_LA_PANEL_M]);
  p.c0 = uniform(w[NX_LA_PANEL_C0]);
  p.w = uniform(w[NX_LA_PANEL_W]);
  p.matrices = uniform(w[NX_LA_PANEL_MATRICES]);
  p.per = uniform(w[NX_LA_PANEL_PER]);
  p.P = uniform(w[NX_LA_PANEL_P]);
  p.G = (int)uniform(w[NX_LA_PANEL_G]);
  p.mats = (int)uniform(w[NX_LA_PANEL_MATS]);
  p.ld = uniform(w[NX_LA_PANEL_LD]);
  p.in_lds = uniform(w[NX_LA_PANEL_IN_LDS]) != 0;
  p.spill_bytes = uniform(w[NX_LA_PANEL_SPILL_BYTES]);
  p.slot_bytes = uniform(w[NX_LA_PANEL_SLOT_BYTES]);
  p.slots_stride = uniform(w[NX_LA_PANEL_SLOTS_STRIDE]);
  p.slots_parity = uniform(w[NX_LA_PANEL_SLOTS_PARITY]);
  p.cands_parity = uniform(w[NX_LA_PANEL_CANDS_PARITY]);
  p.row = uniform(w[NX_LA_PANEL_ROW]);
  p.head = uniform(w[NX_LA_PANEL_HEAD]);
  p.pos = uniform(w[NX_LA_PANEL_POS]);
  p.k = uniform(w[NX_LA_PANEL_K]);
  return p;
}

/* LU of the panel of columns [c0, c0 + w) of rows [c0, m) of each matrix,
   in place: partial pivoting as nx_kernel.mli states it, then each later
   row's element divided by the pivot (not where the pivot is zero), then
   the panel's later columns updated by one nmac each. Rows stay where
   they were loaded; [pos] holds each row's position, and the interchange
   of step j swaps two positions. At the end each row is stored at its
   position, pivots[c0 + j] = c0 + p_j, and src_of[q] = the row now at
   position q (rows from c0), for gather.

   With P = 1, one launch: G work-items a matrix, [mats] a group. With
   P > 1, P groups a matrix, group q holding the panel's rows [q·per,
   (q + 1)·per) in its tile at [spill], over steps + 1 launches, launch k
   taking step k − 1's winner and update and step k's candidates: each
   group writes its best candidate and that row's values to its slot, and
   the next launch's groups find the same winner among the slots, the
   order being total. No group waits on another. */
template <typename T> struct Lu {
  typedef R_of<T> R;
  const PanelWords &p;
  int64_t z, q, r0, rows, w, ldt, steps;
  int G, lane;

  /* The tile's rows from a, or from [src], and their positions. */
  MEMBER void load(T *tile, int64_t *pos, const T *a) {
    for (int64_t r = lane; r < rows; r += G)
      for (int64_t c = 0; c < w; c++)
        tile[r * ldt + c] = p.src ? ((const T *)p.src)[z * p.sbat + (r0 + r) * p.lda + c]
                                  : a[(r0 + r) * p.lda + c];
    for (int64_t r = lane; r < rows; r += G) pos[r] = r0 + r;
  }

  /* A row's candidate for step j. */
  MEMBER Cand<R> cand(const T *tile, int64_t r, int64_t ps, int64_t j) {
    R mg = mag(tile[r * ldt + j]);
    Cand<R> c;
    c.key = mg != mg ? (R)-1 : mg;
    c.pos = ps;
    c.forced = ps == j && mg != mg;
    c.src = (int32_t)q;
    c.row = r;
    return c;
  }

  /* Step j with pivot row u at position pj: the interchange, each later
     row's multiplier and update; the best candidate for step j + 1. */
  MEMBER Cand<R> step(T *tile, int64_t *pos, const T *u, int64_t j, int64_t pj) {
    if (q == 0 && lane == 0) p.piv[z * p.spiv + p.c0 + j] = p.c0 + pj;
    const T d = u[j];
    const bool divide = !is_zero(d);
    Cand<R> next = none<R>();
    for (int64_t r = lane; r < rows; r += G) {
      int64_t ps = pos[r];
      if (ps == j) ps = pj;
      else if (ps == pj) ps = j;
      pos[r] = ps;
      if (ps <= j) continue;
      T *tr = tile + r * ldt;
      T l = tr[j];
      if (divide) l = divd(l, d);
      tr[j] = l;
      for (int64_t c = j + 1; c < w; c++) tr[c] = nmac(tr[c], l, u[c]);
      if (j + 1 < steps) {
        Cand<R> cd = cand(tile, r, ps, j + 1);
        if (better(next, cd)) next = cd;
      }
    }
    return next;
  }

  /* Every row at its position; perm, or src_of. */
  MEMBER void store(const T *tile, const int64_t *pos, T *a) {
    for (int64_t r = lane; r < rows; r += G)
      for (int64_t c = 0; c < w; c++) a[pos[r] * p.lda + c] = tile[r * ldt + c];
    int64_t *src_of = p.src_of + z * (p.m - p.c0);
    for (int64_t r = lane; r < rows; r += G) {
      if (p.perm) p.perm[z * p.sperm + pos[r]] = r0 + r;
      else src_of[pos[r]] = r0 + r;
    }
  }
};

/* One unit of a panel launch: a group's matrices, or one group's rows of
   a matrix with P > 1. */
template <typename T> DEVICE void lu_unit(const PanelWords &p, int64_t unit) {
  typedef R_of<T> R;
  uint8_t *lds = nx_la_lds;
  const bool many = p.P > 1;
  Lu<T> s{p};
  s.G = many ? NX_LA_THREADS : p.G;
  const int g = (int)item() / s.G;
  s.lane = (int)item() % s.G;
  s.z = many ? unit / p.P : unit * p.mats + g;
  s.q = many ? unit % p.P : 0;
  /* A matrix's whole lanes: G <= 32 lanes end together. */
  if (g >= p.mats || s.z >= p.matrices) return;
  const int64_t mp = p.m - p.c0;
  s.w = p.w;
  s.ldt = p.ld;
  s.r0 = s.q * p.per;
  s.rows = imax(0, imin(p.per, mp - s.r0));
  s.steps = imin(mp, s.w);
  /* The candidates, then with P > 1 the pivot row; then each matrix's tile
     and positions, or at [spill]. */
  Cand<R> *const scr[2] = {(Cand<R> *)lds, (Cand<R> *)(lds + p.cands_parity)};
  T *us = (T *)(lds + p.row);
  uint8_t *base = p.in_lds ? lds + p.head + g * p.spill_bytes
                           : p.spill + (many ? unit : s.z) * p.spill_bytes;
  T *tile = (T *)base;
  int64_t *pos = (int64_t *)(base + p.pos);
  T *a = (T *)p.a + s.z * p.sbat + p.c0 * p.lda + p.c0;

  /* One launch runs every step; launch k of several, step k − 1 from the
     slots, then step k's candidates. Each of load, step and store is
     called from one place, so that inlining them costs one copy. */
  const int64_t k = p.k;
  uint8_t *slots = many ? p.slots + s.z * p.slots_stride : nullptr;
  /* Step j's slot of group i. */
  auto slot = [&](int64_t j, int64_t i) { return slots + (j & 1) * p.slots_parity + i * p.slot_bytes; };
  Cand<R> best = none<R>();
  if (!many || k == 0) {
    s.load(tile, pos, (const T *)a);
    if (!many) meet(s.G);
    for (int64_t r = s.lane; r < s.rows; r += s.G) {
      Cand<R> c = s.cand(tile, r, pos[r], 0);
      if (better(best, c)) best = c;
    }
  }
  const int64_t from = many ? imax(k - 1, 0) : 0, to = many ? k : s.steps;
  for (int64_t j = from; j < to; j++) {
    const T *u;
    int64_t pj;
    if (many) {
      Cand<R> c = none<R>();
      for (int64_t i = item(); i < p.P; i += NX_LA_THREADS) {
        const Cand<R> d = *(const Cand<R> *)slot(j, i);
        if (better(c, d)) c = d;
      }
      c = group_best(c, s.G, scr[0]);
      const T *src = (const T *)(slot(j, c.src) + NX_LA_SLOT_HEAD);
      for (int64_t c2 = j + item(); c2 < s.w; c2 += NX_LA_THREADS) us[c2] = src[c2];
      barrier();
      u = us;
      pj = c.pos;
    } else {
      best = group_best(best, s.G, scr[j & 1]);
      u = tile + best.row * s.ldt;
      pj = best.pos;
    }
    best = s.step(tile, pos, u, j, pj);
    if (!many) meet(s.G);
  }
  if (many && k < s.steps) {
    best = group_best(best, s.G, scr[1]);
    uint8_t *mine = slot(k, s.q);
    if (item() == 0) *(Cand<R> *)mine = best;
    T *vals = (T *)(mine + NX_LA_SLOT_HEAD);
    if (best.src == s.q)
      for (int64_t c = k + item(); c < s.w; c += NX_LA_THREADS) vals[c] = tile[best.row * s.ldt + c];
    return;
  }
  s.store(tile, pos, a);
}

template <typename T> DEVICE void lu_panel(void) {
  const PanelWords p = panel_words();
  const int64_t units = p.P > 1 ? p.matrices * p.P : cdiv(p.matrices, p.mats);
  const int64_t groups = uniform(la_staged[NX_LA_PANEL_GROUPS]);
  for (int64_t u = group_x(); u < units; u += groups) {
    lu_unit<T>(p, u);
    /* The next unit reuses the LDS. */
    barrier();
  }
}

/* Row interchanges */

/* A panel's interchanges on each matrix's columns outside the panel and
   on perm: rows c0 + t and pivots[c0 + t], t < steps, the only rows that
   moved, take the rows src_of names, all read before any is written. A
   group takes a matrix's 64 words of 4 bytes of each of those rows, from
   the columns left of the panel then right of it, staged in LDS; the
   matrix's first group takes perm too. */
#define GATHER_WORDS 64

/* Row t of the moved ones: the row it is, the row it takes, the words it
   takes and its perm. An LU panel takes at most 64 steps. */
static SHARED int64_t gather_rows[2 * NX_LA_LEAF], gather_from[2 * NX_LA_LEAF];
static SHARED uint32_t gather_vals[2 * NX_LA_LEAF][GATHER_WORDS];
static SHARED int64_t gather_perm[2 * NX_LA_LEAF];

template <int W> DEVICE void gather_words(const int64_t *w) {
  int64_t *rows = gather_rows, *from = gather_from, *pvals = gather_perm;
  uint32_t(*vals)[GATHER_WORDS] = gather_vals;
  uint32_t *a0 = address<uint32_t>(w, NX_LA_GATHER_A);
  const int64_t *piv0 = address<const int64_t>(w, NX_LA_GATHER_PIV);
  const int64_t *src0 = address<const int64_t>(w, NX_LA_GATHER_SRC_OF);
  int64_t *perm0 = address<int64_t>(w, NX_LA_GATHER_PERM);
  const int64_t sbat = uniform(w[NX_LA_GATHER_SBAT]), lda = uniform(w[NX_LA_GATHER_LDA]),
                n = uniform(w[NX_LA_GATHER_N]), spiv = uniform(w[NX_LA_GATHER_SPIV]),
                sperm = uniform(w[NX_LA_GATHER_SPERM]), m = uniform(w[NX_LA_GATHER_M]),
                c0 = uniform(w[NX_LA_GATHER_C0]), pw = uniform(w[NX_LA_GATHER_W]),
                steps = uniform(w[NX_LA_GATHER_STEPS]),
                matrices = uniform(w[NX_LA_GATHER_MATRICES]),
                groups = uniform(w[NX_LA_GATHER_GROUPS]);
  const int64_t mp = m - c0, t2 = 2 * steps;
  const int64_t outside = (n - pw) * W, blocks = cdiv(outside, GATHER_WORDS);
  const int64_t per = blocks > 0 ? blocks : 1, units = matrices * per;
  const bool small = is_small(units);

  for (int64_t u = group_x(); u < units; u += groups) {
    int64_t z = u;
    const int64_t blk = split(z, per, small);
    const int64_t *piv = piv0 + z * spiv + c0;
    const int64_t *src_of = src0 + z * mp;
    barrier();
    for (int64_t t = item(); t < t2; t += NX_LA_THREADS) {
      const int64_t r = t < steps ? t : piv[t - steps] - c0;
      rows[t] = r;
      from[t] = src_of[r];
    }
    barrier();
    /* Word e of a row: column e / W outside the panel. */
    uint32_t *a = a0 + (z * sbat + c0 * lda) * W;
    const int64_t e0 = blk * GATHER_WORDS;
    for (int64_t x = item(); x < t2 * GATHER_WORDS; x += NX_LA_THREADS) {
      const int64_t t = x / GATHER_WORDS, e = e0 + x % GATHER_WORDS;
      if (e >= outside) continue;
      const int64_t col = e / W < c0 ? e / W : e / W + pw;
      vals[t][x % GATHER_WORDS] = a[from[t] * lda * W + col * W + e % W];
    }
    int64_t *perm = perm0 + z * sperm + c0;
    if (blk == 0)
      for (int64_t t = item(); t < t2; t += NX_LA_THREADS) pvals[t] = perm[from[t]];
    barrier();
    for (int64_t x = item(); x < t2 * GATHER_WORDS; x += NX_LA_THREADS) {
      const int64_t t = x / GATHER_WORDS, e = e0 + x % GATHER_WORDS;
      if (e >= outside) continue;
      const int64_t col = e / W < c0 ? e / W : e / W + pw;
      a[rows[t] * lda * W + col * W + e % W] = vals[t][x % GATHER_WORDS];
    }
    if (blk == 0)
      for (int64_t t = item(); t < t2; t += NX_LA_THREADS) perm[rows[t]] = pvals[t];
  }
}

DEVICE void gather(void) {
  const int64_t *w = staged(NX_LA_GATHER_WORDS);
  switch (uniform(w[NX_LA_GATHER_FMT])) {
  case NX_LA_F32: gather_words<1>(w); break;
  case NX_LA_C128: gather_words<4>(w); break;
  default: gather_words<2>(w);
  }
}

/* The identity's rows: perm[g] = g mod m. */
DEVICE void iota(void) {
  const int64_t *w = staged(NX_LA_IOTA_WORDS);
  int64_t *perm = address<int64_t>(w, NX_LA_IOTA_PERM);
  const int64_t m = uniform(w[NX_LA_IOTA_M]), total = uniform(w[NX_LA_IOTA_TOTAL]);
  const int64_t stride = uniform(w[NX_LA_IOTA_GROUPS]) * NX_LA_THREADS;
  for (int64_t g = (int64_t)group_x() * NX_LA_THREADS + item(); g < total; g += stride)
    perm[g] = g % m;
}

/* Cholesky */

/* L of the diagonal block of order nb at (k, k) of each matrix, in place,
   right-looking: L_jj = sqrt (re a_jj), the column below divided by it,
   then each later element (i, c), c <= i, takes −conj(L_cj) L_ij, one
   nmac. Reads the block's lower triangle and the real parts of its
   diagonal alone. [fail] is set for a matrix that meets a pivot that is
   not positive, or NaN. G work-items a matrix, [mats] a group. */
template <typename T> DEVICE void chol(void) {
  typedef R_of<T> R;
  const int64_t *w = staged(NX_LA_CHOL_WORDS);
  T *a0 = address<T>(w, NX_LA_CHOL_A);
  int32_t *fail = address<int32_t>(w, NX_LA_CHOL_FAIL);
  const T *src0 = address<const T>(w, NX_LA_CHOL_SRC);
  const int64_t sbat = uniform(w[NX_LA_CHOL_SBAT]), lda = uniform(w[NX_LA_CHOL_LDA]),
                cs = uniform(w[NX_LA_CHOL_CS]), k = uniform(w[NX_LA_CHOL_K]),
                matrices = uniform(w[NX_LA_CHOL_MATRICES]), groups = uniform(w[NX_LA_CHOL_GROUPS]);
  const int nb = (int)uniform(w[NX_LA_CHOL_NB]), ldt = (int)uniform(w[NX_LA_CHOL_LD]);
  const int G = (int)uniform(w[NX_LA_CHOL_G]), mats = (int)uniform(w[NX_LA_CHOL_MATS]);
  const int64_t stride = uniform(w[NX_LA_CHOL_STRIDE]);
  const bool upper = uniform(w[NX_LA_CHOL_UPPER]) != 0;
  const int g = (int)item() / G, lane = (int)item() % G;
  T *tile = (T *)(nx_la_lds + g * stride);
  const int64_t units = cdiv(matrices, mats);

  for (int64_t u = group_x(); u < units; u += groups) {
    barrier();
    const int64_t z = u * mats + g;
    if (g >= mats || z >= matrices) continue;
    T *a = a0 + z * sbat + k * (lda + cs);
    const T *src = src0 ? src0 + z * sbat : a;
    for (int r = 0; r < nb; r++)
      for (int c = lane; c <= r; c += G) tile[r * ldt + c] = src[r * lda + c * cs];
    meet(G);
    bool failed = false;
    for (int j = 0; j < nb; j++) {
      R d = re(tile[j * ldt + j]);
      failed |= !(d > 0);
      R s = sqrt_(d);
      for (int i = j + 1 + lane; i < nb; i += G) tile[i * ldt + j] = divd(tile[i * ldt + j], mk<T>(s, 0));
      meet(G);
      if (lane == 0) tile[j * ldt + j] = mk<T>(s, 0);
      for (int i = j + 1 + lane; i < nb; i += G) {
        T l = tile[i * ldt + j];
        for (int c = j + 1; c <= i; c++)
          tile[i * ldt + c] = nmac(tile[i * ldt + c], conj_(tile[c * ldt + j]), l);
      }
      meet(G);
    }
    if (src0) {
      for (int r = 0; r < nb; r++)
        for (int c = lane; c < nb; c += G) {
          T x = c == r ? tile[r * ldt + r]
                : upper ? (c > r ? conj_(tile[c * ldt + r]) : zero<T>())
                        : (c < r ? tile[r * ldt + c] : zero<T>());
          a[r * lda + c * cs] = failed ? nan_<T>() : x;
        }
      continue;
    }
    if (failed && lane == 0) fail[z] = 1;
    for (int r = 0; r < nb; r++)
      for (int c = lane; c <= r; c += G) a[r * lda + c * cs] = tile[r * ldt + c];
  }
}

/* Each n × n matrix: NaN where [fail] is set; else under [upper] U = Lᴴ,
   each element below the diagonal moved conjugated above it, else the
   elements above the diagonal +0. The diagonal stays. Under [tr], the
   matrix holds Lᵀ, which it leaves as L or U. */
template <typename T> DEVICE void chol_finish_t(const int64_t *w) {
  T *a0 = address<T>(w, NX_LA_FINISH_A);
  const int32_t *fail = address<const int32_t>(w, NX_LA_FINISH_FAIL);
  const int64_t n = uniform(w[NX_LA_FINISH_N]), nn = n * n,
                total = nn * uniform(w[NX_LA_FINISH_MATRICES]);
  const bool upper = uniform(w[NX_LA_FINISH_UPPER]) != 0,
             tr = uniform(w[NX_LA_FINISH_TR]) != 0;
  const int64_t stride = uniform(w[NX_LA_FINISH_GROUPS]) * NX_LA_THREADS;
  const bool small = is_small(total);
  for (int64_t g = (int64_t)group_x() * NX_LA_THREADS + item(); g < total; g += stride) {
    int64_t z = g;
    const int64_t e = split(z, nn, small);
    int64_t i = e;
    const int64_t j = split(i, n, small);
    T *a = a0 + z * nn;
    if (fail[z]) {
      a[e] = nan_<T>();
      continue;
    }
    if (tr) {
      /* Element (i, j) above the diagonal holds L_ji: U_ij = conj(L_ji)
         in place, or L_ji moved below. */
      if (i > j && upper) a[e] = zero<T>();
      if (i >= j) continue;
      if (upper) {
        a[e] = conj_(a[e]);
      } else {
        a[j * n + i] = a[e];
        a[e] = zero<T>();
      }
      continue;
    }
    if (i <= j) {
      if (i < j && !upper) a[e] = zero<T>();
      continue;
    }
    if (!upper) continue;
    a[j * n + i] = conj_(a[e]);
    a[e] = zero<T>();
  }
}

DEVICE void chol_finish(void) {
  const int64_t *w = staged(NX_LA_FINISH_WORDS);
  switch (uniform(w[NX_LA_FINISH_FMT])) {
  case NX_LA_F32: chol_finish_t<float>(w); break;
  case NX_LA_F64: chol_finish_t<double>(w); break;
  case NX_LA_C64: chol_finish_t<c64>(w); break;
  default: chol_finish_t<c128>(w);
  }
}

/* Triangular solves' diagonal: flag[z] = 1 where a diagonal element of
   matrix z is zero; then x NaN for a flagged matrix. P parts an element. */
template <typename R> DEVICE void diag_zero_parts(const int64_t *w, int P) {
  const R *a0 = address<const R>(w, NX_LA_DIAG_A);
  int32_t *flag = address<int32_t>(w, NX_LA_DIAG_FLAG);
  const int64_t sa = uniform(w[NX_LA_DIAG_SA]), sd = uniform(w[NX_LA_DIAG_SD]), n = uniform(w[NX_LA_DIAG_N]);
  const int64_t total = n * uniform(w[NX_LA_DIAG_MATRICES]);
  const int64_t stride = uniform(w[NX_LA_DIAG_GROUPS]) * NX_LA_THREADS;
  const bool small = is_small(total);
  for (int64_t g = (int64_t)group_x() * NX_LA_THREADS + item(); g < total; g += stride) {
    int64_t z = g;
    const int64_t i = split(z, n, small);
    const R *d = a0 + (z * sa + i * sd) * P;
    if (d[0] == 0 && (P == 1 || d[1] == 0)) flag[z] = 1;
  }
}

DEVICE void diag_zero(void) {
  const int64_t *w = staged(NX_LA_DIAG_WORDS);
  switch (uniform(w[NX_LA_DIAG_FMT])) {
  case NX_LA_F32: diag_zero_parts<float>(w, 1); break;
  case NX_LA_F64: diag_zero_parts<double>(w, 1); break;
  case NX_LA_C64: diag_zero_parts<float>(w, 2); break;
  default: diag_zero_parts<double>(w, 2);
  }
}

template <typename R> DEVICE void nan_where_parts(const int64_t *w, int P) {
  R *x0 = address<R>(w, NX_LA_DIAG_X);
  const int32_t *flag = address<const int32_t>(w, NX_LA_DIAG_FLAG);
  const int64_t sx = uniform(w[NX_LA_DIAG_SX]), xn = uniform(w[NX_LA_DIAG_XN]);
  const int64_t total = xn * uniform(w[NX_LA_DIAG_MATRICES]);
  const int64_t stride = uniform(w[NX_LA_DIAG_GROUPS]) * NX_LA_THREADS;
  const bool small = is_small(total);
  for (int64_t g = (int64_t)group_x() * NX_LA_THREADS + item(); g < total; g += stride) {
    int64_t z = g;
    const int64_t i = split(z, xn, small);
    if (!flag[z]) continue;
    R *x = x0 + (z * sx + i) * P;
    for (int q = 0; q < P; q++) x[q] = NAN;
  }
}

DEVICE void nan_where(void) {
  const int64_t *w = staged(NX_LA_DIAG_WORDS);
  switch (uniform(w[NX_LA_DIAG_FMT])) {
  case NX_LA_F32: nan_where_parts<float>(w, 1); break;
  case NX_LA_F64: nan_where_parts<double>(w, 1); break;
  case NX_LA_C64: nan_where_parts<float>(w, 2); break;
  default: nan_where_parts<double>(w, 2);
  }
}

/* QR */

/* Sums over rows run in this order: the rows of the matrix in blocks of
   NX_LA_QBLK from row 0, each block in 16 lanes by row mod 16, each lane
   from +0 in increasing row, lane l taking lane l + 8, then l + 4, l + 2,
   l + 1; then the blocks from the panel's first, 32 to a superblock
   combined by the same tree over 32 (l + 16 first), superblocks added in
   increasing order from the first. A row outside the sum adds nothing to
   its lane. */
#define QSUP 32

template <typename T, bool MAX> DEVICE T combine(T x, T y) {
  if (MAX) return mk<T>(max_(re(x), re(y)), 0);
  return mk<T>(re(x) + re(y), im(x) + im(y));
}

/* A value's partials of [nb] blocks, [stride] apart, combined by one
   wave: superblocks of QSUP by the tree, then in order. Every lane
   returns it. */
template <typename T, bool MAX> DEVICE T blocks_tree(const T *part, int64_t stride, int64_t nb) {
  const int ln = (int)lane();
  T total = zero<T>();
  for (int64_t s0 = 0; s0 < nb; s0 += QSUP) {
    const int64_t here = imin(QSUP, nb - s0);
    T x = zero<T>();
    if (ln < here) x = part[(s0 + ln) * stride];
    for (int o = 16; o >= 1; o /= 2) {
      T y = shfl_t(x, down(o, 32));
      if (ln + o < here) x = combine<T, MAX>(x, y);
    }
    x = shfl_t(x, 0);
    total = s0 == 0 ? x : combine<T, MAX>(total, x);
  }
  return total;
}

/* LAPACK's least normal over the unit roundoff. */
template <typename R> DEVICE R safmin() {
  return sizeof(R) == 4 ? (R)0x1p-102 : (R)0x1p-969;
}

/* Below it, or above its reciprocal, x's largest part takes the scaled
   sums, its squares near the normal range's ends; likewise the product
   of x's largest part with the later columns' (prod_lo), below which the
   largest products keep fewer than u's bits above the subnormals. */
template <typename R> DEVICE R safe_lo() {
  return sizeof(R) == 4 ? (R)0x1p-30 : (R)0x1p-250;
}
template <typename R> DEVICE int prod_lo() { return sizeof(R) == 4 ? -100 : -900; }

/* sqrt (a² + b² + c²), scaled (xLAPY3). */
template <typename R> DEVICE R lapy3(R a, R b, R c) {
  R xa = abs_(a), xb = abs_(b), xc = abs_(c);
  R w = max_(xa, max_(xb, xc));
  if (w == 0 || w != w) return xa + xb + xc;
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

/* The words of a QR panel launch. */
struct QrWords {
  uint8_t *a, *tau, *v, *t, *q, *r, *spill, *parts;
  int64_t sbat, lda, stau, sv, st, ldto, qc, n, m, c0, w, matrices, per, P,
      ld, most, spill_bytes, parts_stride, stride, tc, ts, gram, mx, parts_at,
      tile_at, store, drow, gstore, stage, k;
  int G, mats;
  bool in_lds, parts_in_lds;
};

DEVICE QrWords qr_words(void) {
  const int64_t *w = staged(NX_LA_QR_WORDS);
  QrWords p;
  p.a = address<uint8_t>(w, NX_LA_QR_A);
  p.tau = address<uint8_t>(w, NX_LA_QR_TAU);
  p.v = address<uint8_t>(w, NX_LA_QR_V);
  p.t = address<uint8_t>(w, NX_LA_QR_T);
  p.q = address<uint8_t>(w, NX_LA_QR_Q);
  p.r = address<uint8_t>(w, NX_LA_QR_R);
  p.spill = address<uint8_t>(w, NX_LA_QR_SPILL);
  p.parts = address<uint8_t>(w, NX_LA_QR_PARTS);
  p.sbat = uniform(w[NX_LA_QR_SBAT]);
  p.lda = uniform(w[NX_LA_QR_LDA]);
  p.stau = uniform(w[NX_LA_QR_STAU]);
  p.sv = uniform(w[NX_LA_QR_SV]);
  p.st = uniform(w[NX_LA_QR_ST]);
  p.ldto = uniform(w[NX_LA_QR_LDTO]);
  p.qc = uniform(w[NX_LA_QR_QC]);
  p.n = uniform(w[NX_LA_QR_N]);
  p.m = uniform(w[NX_LA_QR_M]);
  p.c0 = uniform(w[NX_LA_QR_C0]);
  p.w = uniform(w[NX_LA_QR_W]);
  p.matrices = uniform(w[NX_LA_QR_MATRICES]);
  p.per = uniform(w[NX_LA_QR_PER]);
  p.P = uniform(w[NX_LA_QR_P]);
  p.G = (int)uniform(w[NX_LA_QR_G]);
  p.mats = (int)uniform(w[NX_LA_QR_MATS]);
  p.ld = uniform(w[NX_LA_QR_LD]);
  p.in_lds = uniform(w[NX_LA_QR_IN_LDS]) != 0;
  p.parts_in_lds = uniform(w[NX_LA_QR_PARTS_IN_LDS]) != 0;
  p.most = uniform(w[NX_LA_QR_MOST]);
  p.spill_bytes = uniform(w[NX_LA_QR_SPILL_BYTES]);
  p.parts_stride = uniform(w[NX_LA_QR_PARTS_STRIDE]);
  p.stride = uniform(w[NX_LA_QR_STRIDE]);
  p.tc = uniform(w[NX_LA_QR_TC]);
  p.ts = uniform(w[NX_LA_QR_TS]);
  p.gram = uniform(w[NX_LA_QR_GRAM]);
  p.mx = uniform(w[NX_LA_QR_MX]);
  p.parts_at = uniform(w[NX_LA_QR_PARTS_AT]);
  p.tile_at = uniform(w[NX_LA_QR_TILE_AT]);
  p.store = uniform(w[NX_LA_QR_STORE]);
  p.drow = uniform(w[NX_LA_QR_DROW]);
  p.gstore = uniform(w[NX_LA_QR_GSTORE]);
  p.stage = uniform(w[NX_LA_QR_STAGE]);
  p.k = uniform(w[NX_LA_QR_K]);
  return p;
}

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

   LDS, a matrix, at the offsets the words give: vals [most], tc [w],
   T [w²], the pairs [w (w − 1) / 2], M [w qc] where Q is stored; then,
   where they fit, the reductions' store and the tile. */
template <typename T> struct Qr {
  typedef R_of<T> R;
  const QrWords &p;
  int64_t z, q, w, ldt, c0, m, most, lo, hi, rows, nblocks, b_first;
  int G, lane;
  T *vals, *parts, *tile, *drow, *tc, *ts;
  int parity;

  MEMBER void sync() { meet(G); }

  /* V's element (i, c) from the tile's row. */
  MEMBER T vd(const T *row, int64_t i, int64_t c) {
    int64_t rr = i - c0;
    return rr > c ? row[c] : rr == c ? one<T>() : zero<T>();
  }

  MEMBER T term(int mode, int v, T acc, const T *row, int64_t i, int64_t j, R big, int64_t v0) {
    if (mode == GRAM) {
      int64_t x = v0 + v, pa = 0;
      while ((pa + 1) * (2 * w - pa - 2) / 2 <= x) pa++;
      int64_t pb = x - pa * (2 * w - pa - 1) / 2 + pa + 1;
      return mac(acc, conj_(vd(row, i, pa)), vd(row, i, pb));
    }
    T x = row[j];
    if (big != 0) /* big: the exponent of the scale s = 2^big, exact; ldexp
                     for a subnormal x, whose s may exceed the largest float */
      x = mk<T>(ldexp_(re(x), (int)big), ldexp_(im(x), (int)big));
    if (v == 0) return mk<T>(fma_(re(x), re(x), fma_(im(x), im(x), re(acc))), 0);
    if (v == 1) return mk<T>(max_(re(acc), max_(abs_(re(x)), abs_(im(x)))), 0);
    if (v == 2) {
      R most_ = re(acc);
      for (int64_t c = j + 1; c < w; c++) most_ = max_(most_, max_(abs_(re(row[c])), abs_(im(row[c]))));
      return mk<T>(most_, 0);
    }
    return mac(acc, conj_(x), row[j + v - 2]);
  }

  static MEMBER bool maximum(int mode, int v) { return mode == MAIN && (v == 1 || v == 2); }
  static MEMBER bool chosen(int mode, int v, int sel) {
    return sel == ALL || (sel == MAXIMA) == maximum(mode, v);
  }

  /* This group's partials of [nv] values over the rows from [from], one
     per NX_LA_QBLK block of rows, into the store of [parity]. */
  MEMBER void partials(int mode, int nv, int sel, int64_t from, int64_t j, R big, int64_t v0) {
    T *pp = parts + parity * p.store;
    const int H = G / 16, h = lane / 16, hl = lane % 16;
    const int64_t my_b0 = lo / NX_LA_QBLK, my_nb = rows > 0 ? (hi - 1) / NX_LA_QBLK - my_b0 + 1 : 0;
    for (int64_t t = h; t < my_nb * nv; t += H) {
      const int64_t b = my_b0 + t / nv;
      const int v = (int)(t % nv);
      if (!chosen(mode, v, sel)) continue;
      const bool mx = maximum(mode, v);
      T acc = zero<T>();
      for (int64_t i = b * NX_LA_QBLK + hl; i < (b + 1) * NX_LA_QBLK; i += 16)
        if (i >= from && i >= lo && i < hi) acc = term(mode, v, acc, tile + (i - lo) * ldt, i, j, big, v0);
      for (int o = 8; o >= 1; o /= 2) {
        T y = shfl_t(acc, down(o, 16));
        acc = mx ? combine<T, true>(acc, y) : combine<T, false>(acc, y);
      }
      if (hl == 0) pp[(b - b_first) * most + v] = acc;
    }
  }

  /* Every group's partials of [parity]'s store combined, into vals.
     Values 1 and 2 of MAIN are maxima. */
  MEMBER void tree(int mode, int nv, int sel) {
    T *pp = parts + parity * p.store;
    if (G >= 32) {
      for (int v = lane / 32; v < nv; v += G / 32) {
        if (!chosen(mode, v, sel)) continue;
        T x = maximum(mode, v) ? blocks_tree<T, true>(pp + v, most, nblocks)
                               : blocks_tree<T, false>(pp + v, most, nblocks);
        if (lane % 32 == 0) vals[v] = x;
      }
    } else {
      for (int v = lane; v < nv; v += G) /* one block */
        if (chosen(mode, v, sel)) vals[v] = pp[v];
    }
    sync();
  }

  /* A reduction within one launch (P = 1). */
  MEMBER void reduce(int mode, int nv, int64_t from, int64_t j, R big, int64_t v0) {
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
  MEMBER int scale(R big, R most_) {
    /* The largest product's exponent, from the factors' own: their
       product may underflow to 0. */
    const int pe = most_ != 0 && most_ - most_ == 0 ? ilogb_(big) + ilogb_(most_) : 0;
    if (big != 0 && big - big == 0 &&
        (big < safe_lo<R>() || big > 1 / safe_lo<R>() || pe < prod_lo<R>() || pe > -prod_lo<R>()))
      return ilogb_(big);
    return 0;
  }

  /* This group's rows of the panel. */
  MEMBER void load() {
    const T *a = (const T *)p.a + z * p.sbat;
    for (int64_t r = lane; r < rows; r += G)
      for (int64_t c = 0; c < w; c++) tile[r * ldt + c] = a[(lo + r) * p.lda + c0 + c];
    sync();
  }

  /* Step j's diagonal row, from its owner. */
  MEMBER void diagonal(int64_t j) {
    const int64_t jr = c0 + j;
    T *dr = drow + (j & 1) * w;
    if (jr >= lo && jr < hi)
      for (int64_t c = j + lane; c < w; c += G) dr[c] = tile[(jr - lo) * ldt + c];
  }

  /* Step j from its sums in vals, over x 2^-e: the reflector, xLARFG's
     scaling of x, α and β by 1 / safmin until β is normal (each exact,
     the sums' scale following), τ, and the update of this group's
     rows. */
  MEMBER void finish(int64_t j, int e) {
    const int64_t jr = c0 + j;
    const T *dr = drow + (j & 1) * w;
    const R sf = safmin<R>();
    const T alpha = dr[j];
    R ar = re(alpha), ai = im(alpha), beta = 0;
    R xnorm = ldexp_(sqrt_(re(vals[0])), e);
    const bool reflect = !(xnorm == 0 && ai == 0);
    int knt = 0;
    if (reflect) {
      beta = -copysign_(lapy3(ar, ai, xnorm), ar);
      while (abs_(beta) < sf && knt < 20) {
        knt++;
        for (int64_t i = imax(lo, jr + 1) + lane; i < hi; i += G) {
          T &x = tile[(i - lo) * ldt + j];
          x = mk<T>(re(x) * (1 / sf), im(x) * (1 / sf));
        }
        ar *= 1 / sf;
        ai *= 1 / sf;
        /* A zero x's sums are zero at any scale: e stays, σ's scaled
           copy finite. */
        if (re(vals[1]) != 0) e -= ilogb_(sf);
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
      for (int64_t c = j + 1 + lane; c < w; c += G)
        tc[c] = mul(conj_(tau), mac(dr[c], conj_(se), vals[c - j + 2]));
    }
    if (lane == 0 && q == 0) ((T *)p.tau)[z * p.stau + c0 + j] = tau;
    ts[j * w + j] = tau; /* T's diagonal, for T below */
    sync();
    for (int64_t i = imax(lo, jr) + lane; i < hi && reflect; i += G) {
      T *row = tile + (i - lo) * ldt;
      if (i == jr) {
        row[j] = mk<T>(beta, 0);
        for (int64_t c = j + 1; c < w; c++) row[c] = mk<T>(re(row[c]) - re(tc[c]), im(row[c]) - im(tc[c]));
        continue;
      }
      const T v = mul(row[j], sigma);
      row[j] = v;
      for (int64_t c = j + 1; c < w; c++) row[c] = nmac(row[c], tc[c], v);
    }
    sync();
  }

  /* The factored panel and dense V from this group's rows. */
  MEMBER void panel() {
    T *a = (T *)p.a + z * p.sbat;
    T *vo = p.v ? (T *)p.v + z * p.sv : nullptr;
    for (int64_t r = lane; r < rows && !p.q; r += G)
      for (int64_t c = 0; c < w; c++) {
        a[(lo + r) * p.lda + c0 + c] = tile[r * ldt + c];
        if (vo) vo[(lo + r - c0) * w + c] = vd(tile + r * ldt, lo + r, c);
      }
  }

  /* T: T_ii = τ_i, T[0:i, i] = −τ_i T[0:i, 0:i] g[0:i, i] from the pairs
     g, a column a turn, lane r its row r; one wave, or the 16 lanes. */
  MEMBER void larft(const T *gram) {
    const int width = G < 32 ? G : 32;
    if (lane < width) {
      for (int64_t i = 0; i < w; i++) {
        const T ti = ts[i * w + i];
        for (int64_t r = lane; r < w; r += width) {
          if (r >= i) {
            if (r > i) ts[r * w + i] = zero<T>();
            continue;
          }
          T x = zero<T>();
          for (int64_t c = r; c < i; c++) x = mac(x, ts[r * w + c], gram[c * (2 * w - c - 1) / 2 + (i - c - 1)]);
          ts[r * w + i] = mul(neg(ti), x);
        }
        wave_sync();
      }
    }
    sync();
    if (p.t && q == 0) {
      T *to = (T *)p.t + z * p.st;
      for (int64_t r = 0; r < w; r++)
        for (int64_t c = lane; c < w; c += G) to[r * p.ldto + c] = ts[r * w + c];
    }
  }
};

/* One unit of a QR panel launch: a group's matrices, or one group's rows
   of a matrix with P > 1. */
template <typename T> DEVICE void qr_unit(const QrWords &p, int64_t unit) {
  typedef R_of<T> R;
  uint8_t *lds = nx_la_lds;
  const bool many = p.P > 1;
  const int G = many ? NX_LA_THREADS : p.G;
  const int g = (int)item() / G, lane = (int)item() % G;
  Qr<T> s{p};
  s.z = many ? unit / p.P : unit * p.mats + g;
  s.q = many ? unit % p.P : 0;
  if (g >= p.mats || s.z >= p.matrices) return;
  const int64_t w = p.w, c0 = p.c0, m = p.m, most = p.most, z = s.z;
  s.w = w;
  s.ldt = p.ld;
  s.c0 = c0;
  s.m = m;
  s.most = most;
  s.G = G;
  s.lane = lane;
  s.b_first = c0 / NX_LA_QBLK;
  s.nblocks = (m - 1) / NX_LA_QBLK - s.b_first + 1;
  const int64_t bpc = p.per / NX_LA_QBLK;
  s.lo = imax(c0, (s.b_first + s.q * bpc) * NX_LA_QBLK);
  s.hi = imin(m, (s.b_first + (s.q + 1) * bpc) * NX_LA_QBLK);
  s.rows = imax(0, s.hi - s.lo);
  const int64_t npair = w * (w - 1) / 2;
  uint8_t *mine = lds + g * p.stride;
  T *vals = (T *)mine, *tc = (T *)(mine + p.tc), *ts = (T *)(mine + p.ts),
    *gram = (T *)(mine + p.gram), *M = (T *)(mine + p.mx);
  s.vals = vals;
  s.tc = tc;
  s.ts = ts;
  s.parts = p.parts_in_lds ? (T *)(mine + p.parts_at) : (T *)p.parts + z * p.parts_stride;
  s.tile = p.in_lds ? (T *)(mine + p.tile_at) : (T *)(p.spill + (many ? unit : z) * p.spill_bytes);
  T *tile = s.tile;
  s.drow = s.parts + p.drow;     /* two of w, by step parity */
  T *gstore = s.parts + p.gstore; /* with P > 1, the pairs */
  s.parity = 0;
  const int64_t steps = imin(m - c0, w);
  /* Gram pairs' batches, and step j's values. */
  const int64_t batches = cdiv(npair, most);
  auto nv = [&](int64_t j) { return (int)(w - j + 2); };
  auto nvg = [&](int64_t i) { return (int)imin(most, npair - i * most); };

  if (many) {
    const int64_t k = p.k;
    if (p.stage == NX_LA_QR_STEP_B) {
      /* Step k's sums at the scale its maxima give. */
      s.parity = (int)(k & 1);
      s.tree(MAIN, nv(k), MAXIMA);
      const int e = s.scale(re(vals[1]), re(vals[2]));
      s.partials(MAIN, nv(k), SUMS, c0 + k + 1, k, (R)-e, 0);
      return;
    }
    if (p.stage == NX_LA_QR_STEP_A) {
      if (k == 0) {
        s.load();
      } else {
        s.parity = (int)((k - 1) & 1);
        s.tree(MAIN, nv(k - 1), MAXIMA);
        const int e = s.scale(re(vals[1]), re(vals[2]));
        s.tree(MAIN, nv(k - 1), SUMS);
        s.finish(k - 1, e);
      }
      if (k < steps) {
        s.diagonal(k);
        s.parity = (int)(k & 1);
        s.partials(MAIN, nv(k), MAXIMA, c0 + k + 1, k, 0, 0);
        return;
      }
      s.panel();
      if (!p.t) return;
      s.parity = (int)(steps & 1);
      s.partials(GRAM, nvg(0), ALL, c0, 0, 0, 0);
      return;
    }
    /* NX_LA_QR_STEP_G, k = i: batch i − 1 into the store, batch i's sums;
       after the last, T. */
    if (s.q == 0) {
      s.parity = (int)((steps + k - 1) & 1);
      s.tree(GRAM, nvg(k - 1), ALL);
      for (int v = lane; v < nvg(k - 1); v += G) gstore[(k - 1) * most + v] = vals[v];
    }
    if (k < batches) {
      s.parity = (int)((steps + k) & 1);
      s.partials(GRAM, nvg(k), ALL, c0, 0, 0, k * most);
      return;
    }
    if (s.q != 0) return;
    s.sync();
    for (int64_t v = lane; v < npair; v += G) gram[v] = gstore[v];
    for (int64_t i = lane; i < w; i += G) ts[i * w + i] = ((const T *)p.tau)[z * p.stau + c0 + i];
    s.sync();
    s.larft(gram);
    return;
  }

  s.load();
  for (int64_t j = 0; j < steps; j++) {
    s.diagonal(j);
    s.reduce(MAIN, nv(j), c0 + j + 1, j, 0, 0);
    const int e = s.scale(re(vals[1]), re(vals[2]));
    if (e != 0) s.reduce(MAIN, nv(j), c0 + j + 1, j, (R)-e, 0);
    s.finish(j, e);
  }

  /* The factored panel; dense V; T from Vᴴ V. */
  s.panel();
  if (!p.t && !p.q) return;
  /* Pair v = (pa < pb), row-major: Σ_i conj(V_i,pa) V_i,pb. */
  for (int64_t i = 0; i < batches; i++) {
    s.reduce(GRAM, nvg(i), c0, 0, 0, i * most);
    for (int v = lane; v < nvg(i); v += G) gram[i * most + v] = vals[v];
    s.sync();
  }
  s.larft(gram);
  if (!p.q) return;
  /* The whole matrix: M = T V1ᴴ (steps × qc), Q = I − V M, R. A panel
     wider than its rows has [steps] reflectors: T's later columns were
     never written, so no sum reads them. */
  const int64_t qc = p.qc;
  for (int64_t c = lane; c < qc; c += G)
    for (int64_t r = 0; r < steps; r++) {
      T x = zero<T>();
      for (int64_t s2 = r; s2 < steps; s2++) x = mac(x, ts[r * w + s2], conj_(s.vd(tile + c * s.ldt, c, s2)));
      M[r * qc + c] = x;
    }
  s.sync();
  T *qo = (T *)p.q + z * m * qc;
  for (int64_t i = 0; i < m; i++)
    for (int64_t c = lane; c < qc; c += G) {
      T x = i == c ? one<T>() : zero<T>();
      for (int64_t s2 = 0; s2 < steps && s2 <= i; s2++) x = nmac(x, s.vd(tile + i * s.ldt, i, s2), M[s2 * qc + c]);
      qo[i * qc + c] = x;
    }
  T *ro = (T *)p.r + z * qc * p.n;
  for (int64_t i = 0; i < qc; i++)
    for (int64_t c = lane; c < p.n; c += G) ro[i * p.n + c] = c >= i ? tile[i * s.ldt + c] : zero<T>();
}

template <typename T> DEVICE void qr_panel(void) {
  const QrWords p = qr_words();
  const int64_t units = p.P > 1 ? p.matrices * p.P : cdiv(p.matrices, p.mats);
  const int64_t groups = uniform(la_staged[NX_LA_QR_GROUPS]);
  for (int64_t u = group_x(); u < units; u += groups) {
    qr_unit<T>(p, u);
    /* The next unit reuses the LDS. */
    barrier();
  }
}

/* QR's other steps */

/* q = the first qc columns of the m × m identity; r = the upper triangle
   of f's first rows (rows past m zero). */
template <typename R> DEVICE void qr_out_parts(const int64_t *w, int P) {
  const R *f0 = address<const R>(w, NX_LA_QR_OUT_F);
  R *q0 = address<R>(w, NX_LA_QR_OUT_Q);
  R *r0 = address<R>(w, NX_LA_QR_OUT_R);
  const int64_t m = uniform(w[NX_LA_QR_OUT_M]), n = uniform(w[NX_LA_QR_OUT_N]), qc = uniform(w[NX_LA_QR_OUT_QC]);
  const int64_t nq = m * qc, nr = qc * n;
  const int64_t total = (nq + nr) * uniform(w[NX_LA_QR_OUT_MATRICES]);
  const int64_t stride = uniform(w[NX_LA_QR_OUT_GROUPS]) * NX_LA_THREADS;
  const bool small = is_small(total);
  for (int64_t g = (int64_t)group_x() * NX_LA_THREADS + item(); g < total; g += stride) {
    int64_t z = g;
    const int64_t e = split(z, nq + nr, small);
    if (e < nq) {
      R *x = q0 + (z * nq + e) * P;
      int64_t row = e;
      const int64_t col = split(row, qc, small);
      x[0] = row == col ? 1 : 0;
      if (P == 2) x[1] = 0;
      continue;
    }
    int64_t i = e - nq;
    const int64_t c = split(i, n, small);
    R *x = r0 + (z * nr + e - nq) * P;
    const bool kept = c >= i && i < m && f0;
    for (int q = 0; q < P; q++) x[q] = kept ? f0[(z * m * n + i * n + c) * P + q] : 0;
  }
}

DEVICE void qr_out(void) {
  const int64_t *w = staged(NX_LA_QR_OUT_WORDS);
  switch (uniform(w[NX_LA_QR_OUT_FMT])) {
  case NX_LA_F32: qr_out_parts<float>(w, 1); break;
  case NX_LA_F64: qr_out_parts<double>(w, 1); break;
  case NX_LA_C64: qr_out_parts<float>(w, 2); break;
  default: qr_out_parts<double>(w, 2);
  }
}

/* Dense V of a panel from f's columns [c0, c0 + w): rows [c0, m). */
template <typename R> DEVICE void dense_v_parts(const int64_t *w, int P) {
  const R *f0 = address<const R>(w, NX_LA_DENSE_F);
  R *v0 = address<R>(w, NX_LA_DENSE_V);
  const int64_t m = uniform(w[NX_LA_DENSE_M]), n = uniform(w[NX_LA_DENSE_N]), c0 = uniform(w[NX_LA_DENSE_C0]),
                pw = uniform(w[NX_LA_DENSE_W]);
  const int64_t h = m - c0, nv = h * pw;
  const int64_t total = nv * uniform(w[NX_LA_DENSE_MATRICES]);
  const int64_t stride = uniform(w[NX_LA_DENSE_GROUPS]) * NX_LA_THREADS;
  const bool small = is_small(total);
  for (int64_t g = (int64_t)group_x() * NX_LA_THREADS + item(); g < total; g += stride) {
    int64_t z = g;
    const int64_t e = split(z, nv, small);
    int64_t r = e;
    const int64_t c = split(r, pw, small);
    const R *x = f0 + (z * m * n + (c0 + r) * n + c0 + c) * P;
    R *v = v0 + (z * nv + e) * P;
    v[0] = r > c ? x[0] : r == c ? 1 : 0;
    if (P == 2) v[1] = r > c ? x[1] : 0;
  }
}

DEVICE void dense_v(void) {
  const int64_t *w = staged(NX_LA_DENSE_WORDS);
  switch (uniform(w[NX_LA_DENSE_FMT])) {
  case NX_LA_F32: dense_v_parts<float>(w, 1); break;
  case NX_LA_F64: dense_v_parts<double>(w, 1); break;
  case NX_LA_C64: dense_v_parts<float>(w, 2); break;
  default: dense_v_parts<double>(w, 2);
  }
}

} /* namespace la */

/* Entries */

#define LA_FORMATS(name, threads, words, entry)                               \
  KERNEL SIZED(threads) void name##_f32(Words<words>) { la::entry<float>(); } \
  KERNEL SIZED(threads) void name##_f64(Words<words>) {                       \
    la::entry<double>();                                                      \
  }                                                                           \
  KERNEL SIZED(threads) void name##_c64(Words<words>) {                       \
    la::entry<la::c64>();                                                     \
  }                                                                           \
  KERNEL SIZED(threads) void name##_c128(Words<words>) {                      \
    la::entry<la::c128>();                                                    \
  }

LA_FORMATS(la_trsm, NX_LA_TRSM_THREADS, NX_LA_TRSM_WORDS, trsm)
LA_FORMATS(la_lu_panel, NX_LA_THREADS, NX_LA_PANEL_WORDS, lu_panel)
LA_FORMATS(la_chol, NX_LA_THREADS, NX_LA_CHOL_WORDS, chol)
LA_FORMATS(la_qr_panel, NX_LA_THREADS, NX_LA_QR_WORDS, qr_panel)

#define LA_ONE(name, words, entry)                                            \
  KERNEL SIZED(NX_LA_THREADS) void name(Words<words>) { la::entry(); }

LA_ONE(la_gather, NX_LA_GATHER_WORDS, gather)
LA_ONE(la_chol_finish, NX_LA_FINISH_WORDS, chol_finish)
LA_ONE(la_diag_zero, NX_LA_DIAG_WORDS, diag_zero)
LA_ONE(la_nan_where, NX_LA_DIAG_WORDS, nan_where)
LA_ONE(la_qr_out, NX_LA_QR_OUT_WORDS, qr_out)
LA_ONE(la_dense_v, NX_LA_DENSE_WORDS, dense_v)
LA_ONE(la_iota, NX_LA_IOTA_WORDS, iota)
