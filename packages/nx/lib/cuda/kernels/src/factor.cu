/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* The factorisations' steps: cholesky, lu, qr and solve_triangular,
   batched, at float32, float64, complex64 and complex128 (kernels.h,
   Factorisations). factor.ml plans them, with the contraction's FMA
   kernels for their products.

   Every routine computes in place in a C-contiguous matrix per batch
   element (a destination, or memory of the call's, into which the plan
   copies the operand), by launches of the steps below and of products.
   Each element of Cholesky, LU and the triangular solves gets the
   arithmetic of the textbook unblocked algorithm, right-looking, whatever
   the blocking: its updates, one mac each, in increasing step order
   (decreasing for an upper solve), starting from its own value, then its
   division. The blocks only choose which launch performs which updates: a
   product starts each output's chain from the output itself
   (NX_CONTRACT_INIT_FIRST, one range of k). QR's sums over rows follow
   their own order (below).

   Steps:
   - trsm: a triangular solve of at most leaf_rows rows, a thread a column.
   - lu_panel: partial pivoting on a panel of columns, its rows held by
     one block, or by P blocks over one launch a column (no block waits
     on another block of its launch).
   - gather: a panel's row interchanges on the other columns and on perm.
   - chol: Cholesky of a diagonal block held in shared memory.
   - qr_panel and the finishing steps: see their sections. */

#include <math.h>
#include <stdint.h>

namespace nx_factor {

/* Scalars. T is float, double, c64 or c128 (nx_kinds.h's complex types),
   R its part's type. Every update of the routines is mac or nmac: one
   fused multiply-add on reals; on complex numbers two a part, the real
   part taking ar br then ai (-bi), the imaginary part ar bi then ai br,
   the order nx_kernel.mli numbers a complex contraction's terms in. */

#define HD __device__ __forceinline__

typedef nx_c64 c64;
typedef nx_c128 c128;

template <typename T> struct Part { typedef T type; };
template <> struct Part<c64> { typedef float type; };
template <> struct Part<c128> { typedef double type; };
template <typename T> using R_of = typename Part<T>::type;

template <typename T> constexpr bool cplx = false;
template <> constexpr bool cplx<c64> = true;
template <> constexpr bool cplx<c128> = true;

HD float fma_(float a, float b, float c) { return fmaf(a, b, c); }
HD double fma_(double a, double b, double c) { return fma(a, b, c); }

HD float re(float x) { return x; }
HD double re(double x) { return x; }
HD float re(c64 x) { return x.re; }
HD double re(c128 x) { return x.re; }
HD float im(float) { return 0; }
HD double im(double) { return 0; }
HD float im(c64 x) { return x.im; }
HD double im(c128 x) { return x.im; }

template <typename T> HD T mk(R_of<T> r, R_of<T> i) {
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

template <typename T> HD T zero() { return mk<T>(0, 0); }
template <typename T> HD T one() { return mk<T>(1, 0); }
template <typename T> HD T neg(T x) { return mk<T>(-re(x), -im(x)); }
template <typename T> HD T conj_(T x) { return mk<T>(re(x), -im(x)); }
template <typename T> HD T conj_if(T x, bool c) { return c ? conj_(x) : x; }
template <typename T> HD bool is_zero(T x) { return re(x) == 0 && im(x) == 0; }

/* c + a·b. */
template <typename T> HD T mac(T c, T a, T b) {
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
template <typename T> HD T nmac(T c, T a, T b) { return mac(c, neg(a), b); }

/* LU's magnitude: |re| + |im|. */
template <typename T> HD R_of<T> mag(T x) {
  return cplx<T> ? fabs(re(x)) + fabs(im(x)) : fabs(re(x));
}

/* x / d: on complex numbers each part divided by d's real part where d's
   imaginary part is zero, as Cholesky's diagonal is, else Smith's
   quotient (nx_kinds.h's Fdiv). */
HD float divd(float x, float d) { return x / d; }
HD double divd(double x, double d) { return x / d; }
HD c64 divd(c64 x, c64 d) {
  if (d.im == 0) return mk<c64>(x.re / d.re, x.im / d.re);
  return nx_fdiv_c64(x, d);
}
HD c128 divd(c128 x, c128 d) {
  if (d.im == 0) return mk<c128>(x.re / d.re, x.im / d.re);
  return nx_fdiv_c128(x, d);
}

template <typename T> HD T mul(T a, T b) { return mac(zero<T>(), a, b); }


/* The most rows of a triangular solve's leaf: its columns' values, a
   thread's each, fill 64 KB of shared memory at 128 threads. */
template <typename T> constexpr int leaf_rows() { return sizeof(T) == 16 ? NX_LA_LEAF / 2 : NX_LA_LEAF; }

template <typename T> __device__ __forceinline__ T ld(const T *p) { return *p; }

__device__ __forceinline__ int64_t r16d(int64_t a) { return (a + 15) / 16 * 16; }

__device__ __forceinline__ int64_t cdiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

/* Triangular solve leaf */

/* x ← op(l)⁻¹ x for the lower triangular n × n view op(l), n <= leaf_rows, and
   the n × r view x of each matrix, a thread a column: x_i takes −l_ip x_p
   for p < i in increasing p, one nmac each, then is divided by l_ii unless
   [unit]. op(l)'s element (i, p) is l's at i·lrs + p·lcs, conjugated under
   [conj]: an upper triangle or an adjoint is a view of reversed or swapped
   strides, so its solve runs in its own order. Column g of the launch is
   column g mod r of matrix g / r. */
#define NX_LA_TRSM_THREADS 128

template <typename T> __device__ void trsm(const la_trsm_params &p) {
  extern __shared__ __align__(16) uint8_t smem[];
  T *xs = (T *)smem; /* xs[i * NX_LA_TRSM_THREADS + tid] */
  const int tid = threadIdx.x;
  const int64_t g = (int64_t)blockIdx.x * NX_LA_TRSM_THREADS + tid;
  if (g >= p.matrices * p.r) return;
  const int64_t z = g / p.r, col = g % p.r;
  const T *l = (const T *)p.l + z * p.sl;
  T *x = (T *)p.x + z * p.sx + col * p.xcs;
  const int n = (int)p.n;
  const T *b = p.src ? (const T *)p.src + z * p.sx + col * p.xcs : x;
  for (int i = 0; i < n; i++) xs[i * NX_LA_TRSM_THREADS + tid] = b[i * p.xrs];
  bool zero_diag = false;
  for (int i = 0; i < n; i++) {
    T v = xs[i * NX_LA_TRSM_THREADS + tid];
    const T *li = l + i * p.lrs;
    for (int q = 0; q < i; q++)
      v = nmac(v, conj_if(ld(li + q * p.lcs), p.conj), xs[q * NX_LA_TRSM_THREADS + tid]);
    if (!p.unit) {
      const T d = conj_if(ld(li + i * p.lcs), p.conj);
      zero_diag |= is_zero(d);
      v = divd(v, d);
    }
    xs[i * NX_LA_TRSM_THREADS + tid] = v;
  }
  for (int i = 0; i < n; i++) x[i * p.xrs] = p.whole && zero_diag ? mk<T>(NAN, NAN) : xs[i * NX_LA_TRSM_THREADS + tid];
}

/* LU panel */

/* A pivot candidate: [forced] where it is the row at the step's own
   position and its magnitude is NaN, which the search never leaves; else
   its magnitude, -1 for NaN, which nothing exceeds and which exceeds no
   number; then its position, the lesser winning a tie. [row] is its row
   in the block's tile, [src] its block. The order is total, so any tree
   of comparisons picks the same row: the sequential search's. */
template <typename R> struct Cand {
  R key;
  int64_t pos;
  int32_t forced, src;
  int64_t row;
};

template <typename R> __device__ __forceinline__ bool better(const Cand<R> &a, const Cand<R> &b) {
  if (a.forced != b.forced) return b.forced > a.forced;
  if (a.key != b.key) return b.key > a.key;
  return b.pos < a.pos;
}

template <typename R> __device__ __forceinline__ Cand<R> none() {
  Cand<R> c;
  c.key = -2;
  c.pos = INT64_MAX;
  c.forced = 0;
  c.src = -1;
  c.row = -1;
  return c;
}

template <typename R> __device__ Cand<R> shfl_cand(Cand<R> c, int o, int width, unsigned mask) {
  Cand<R> d;
  d.key = __shfl_xor_sync(mask, c.key, o, width);
  d.pos = __shfl_xor_sync(mask, c.pos, o, width);
  d.forced = __shfl_xor_sync(mask, c.forced, o, width);
  d.src = __shfl_xor_sync(mask, c.src, o, width);
  d.row = __shfl_xor_sync(mask, c.row, o, width);
  return d;
}

/* The best candidate of a group of G threads: G of 16 (a half-warp) or
   the whole block, [scr] two slots a warp, alternated by [parity] so that
   a step's writes never meet the last step's reads. */
template <typename R>
__device__ Cand<R> group_best(Cand<R> c, int G, unsigned mask, Cand<R> *scr, int parity) {
  const int width = G < 32 ? G : 32;
  for (int o = width / 2; o >= 1; o /= 2) {
    Cand<R> d = shfl_cand(c, o, width, mask);
    if (better(c, d)) c = d;
  }
  if (G <= 32) return c;
  const int warps = blockDim.x / 32, w = threadIdx.x / 32;
  Cand<R> *s = scr + parity * 32;
  if (threadIdx.x % 32 == 0) s[w] = c;
  __syncthreads();
  c = s[0];
  for (int q = 1; q < warps; q++)
    if (better(c, s[q])) c = s[q];
  return c;
}

/* LU of the panel of columns [c0, c0 + w) of rows [c0, m) of each matrix,
   in place: partial pivoting as nx_kernel.mli states it, then each later
   row's element divided by the pivot (not where the pivot is zero), then
   the panel's later columns updated by one nmac each. Rows stay where
   they were loaded; [pos] holds each row's position, and the interchange
   of step j swaps two positions. At the end each row is stored at its
   position, pivots[c0 + j] = c0 + p_j, and src_of[q] = the row now at
   position q (rows from c0), for gather_rows.

   With P = 1, one launch: G is 16, a half-warp per matrix, several a
   block, or the block. With P > 1, P blocks a matrix, block q holding the
   panel's rows [q·per, (q + 1)·per) in its tile at [spill], over steps +
   1 launches, launch k taking step k − 1's winner and update and step
   k's candidates: each block writes its best candidate and that row's
   values to its slot, and the next launch's blocks find the same winner
   among the slots, the order being total. No block waits on another. */
template <typename R> struct Lu {
  const la_panel_params &p;
  int64_t z, q, r0, rows, w, ldt, steps;
  int G, lane;
  unsigned mask;
  Cand<R> *scr;

  __device__ void sync() {
    if (G <= 32) __syncwarp(mask);
    else __syncthreads();
  }

  /* The tile's rows from a, or from [src], and their positions. */
  template <typename T> __device__ __forceinline__ void load(T *tile, int64_t *pos, const T *a) {
    for (int64_t r = lane; r < rows; r += G)
      for (int64_t c = 0; c < w; c++)
        tile[r * ldt + c] =
            p.src ? ((const T *)p.src)[z * p.sbat + (r0 + r) * p.lda + c] : a[(r0 + r) * p.lda + c];
    for (int64_t r = lane; r < rows; r += G) pos[r] = r0 + r;
  }

  /* A row's candidate for step j. */
  template <typename T> __device__ Cand<R> cand(const T *tile, int64_t r, int64_t ps, int64_t j) {
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
  template <typename T>
  __device__ __forceinline__ Cand<R> step(T *tile, int64_t *pos, const T *u, int64_t j, int64_t pj) {
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
  template <typename T> __device__ __forceinline__ void store(const T *tile, const int64_t *pos, T *a) {
    for (int64_t r = lane; r < rows; r += G)
      for (int64_t c = 0; c < w; c++) a[pos[r] * p.lda + c] = tile[r * ldt + c];
    int64_t *src_of = p.src_of + z * (p.m - p.c0);
    for (int64_t r = lane; r < rows; r += G) {
      if (p.perm) p.perm[z * p.sperm + pos[r]] = r0 + r;
      else src_of[pos[r]] = r0 + r;
    }
  }
};

template <typename T> __device__ void lu_panel(const la_panel_params &p) {
  typedef R_of<T> R;
  extern __shared__ __align__(16) uint8_t smem[];
  Lu<R> s{p};
  const bool many = p.P > 1;
  s.G = many ? blockDim.x : p.G;
  const int groups = blockDim.x / s.G, g = threadIdx.x / s.G;
  s.lane = threadIdx.x % s.G;
  s.mask = s.G < 32 ? (((1u << s.G) - 1) << (threadIdx.x % 32 / s.G * s.G)) : 0xffffffffu;
  s.z = many ? blockIdx.x / p.P : (int64_t)blockIdx.x * groups + g;
  s.q = many ? blockIdx.x % p.P : 0;
  if (s.z >= p.matrices) return; /* a whole group: G = 16 groups end together */
  const int64_t mp = p.m - p.c0;
  s.w = p.w;
  s.ldt = p.ld;
  s.r0 = s.q * p.per;
  s.rows = max((int64_t)0, min(p.per, mp - s.r0));
  s.steps = min(mp, s.w);
  /* 64 candidates, then with P > 1 the pivot row; then the groups' tiles
     and positions, or at [spill]. */
  s.scr = (Cand<R> *)smem;
  T *us = (T *)(smem + 64 * sizeof(Cand<R>));
  const int64_t head = (64 * sizeof(Cand<R>) + (many ? p.w * sizeof(T) : 0) + 15) / 16 * 16;
  uint8_t *base = p.in_smem ? smem + head + g * p.spill_bytes
                            : (uint8_t *)p.spill + (int64_t)blockIdx.x * p.spill_bytes;
  T *tile = (T *)base;
  int64_t *pos = (int64_t *)(base + ((p.per * s.ldt * sizeof(T) + 15) / 16 * 16));
  T *a = (T *)p.a + s.z * p.sbat + p.c0 * p.lda + p.c0;

  /* One launch runs every step; launch k of several, step k − 1 from the
     slots, then step k's candidates. Each of load, step and store is
     called from one place, so that inlining them costs one copy. */
  const int64_t k = p.k;
  uint8_t *slots = many ? (uint8_t *)p.slots + s.z * 2 * p.P * p.slot_bytes : nullptr;
  Cand<R> best = none<R>();
  if (!many || k == 0) {
    s.load(tile, pos, (const T *)a);
    if (!many) s.sync();
    for (int64_t r = s.lane; r < s.rows; r += s.G) {
      Cand<R> c = s.cand(tile, r, pos[r], 0);
      if (better(best, c)) best = c;
    }
  }
  const int64_t from = many ? max(k - 1, (int64_t)0) : 0, to = many ? k : s.steps;
  for (int64_t j = from; j < to; j++) {
    const T *u;
    int64_t pj;
    if (many) {
      Cand<R> c = none<R>();
      for (int64_t i = threadIdx.x; i < p.P; i += blockDim.x) {
        const Cand<R> d = *(const Cand<R> *)(slots + ((j & 1) * p.P + i) * p.slot_bytes);
        if (better(c, d)) c = d;
      }
      c = group_best(c, s.G, s.mask, s.scr, 0);
      const T *src = (const T *)(slots + ((j & 1) * p.P + c.src) * p.slot_bytes + NX_LA_SLOT_HEAD);
      for (int64_t c2 = j + threadIdx.x; c2 < s.w; c2 += blockDim.x) us[c2] = src[c2];
      __syncthreads();
      u = us;
      pj = c.pos;
    } else {
      best = group_best(best, s.G, s.mask, s.scr, (int)(j & 1));
      u = tile + best.row * s.ldt;
      pj = best.pos;
    }
    best = s.step(tile, pos, u, j, pj);
    if (!many) s.sync();
  }
  if (many && k < s.steps) {
    best = group_best(best, s.G, s.mask, s.scr, 1);
    uint8_t *mine = slots + ((k & 1) * p.P + s.q) * p.slot_bytes;
    if (threadIdx.x == 0) *(Cand<R> *)mine = best;
    T *vals = (T *)(mine + NX_LA_SLOT_HEAD);
    if (best.src == s.q)
      for (int64_t c = k + threadIdx.x; c < s.w; c += blockDim.x) vals[c] = tile[best.row * s.ldt + c];
    return;
  }
  s.store(tile, pos, a);
}

/* Row interchanges */

/* A panel's interchanges on each matrix's columns outside the panel and
   on perm: rows c0 + t and pivots[c0 + t], t < steps, the only rows that
   moved, take the rows src_of names, all read before any is written.
   Column g of the launch is column g mod (cols + 1) of matrix g / (cols +
   1), the last being perm; cols = n − w, the columns left of the panel
   then right of it. */
#define GATHER_MOST 128 /* rows: 2 · the widest panel */

__device__ __forceinline__ void gather_words(const la_gather_params &p, int W) {
  const int64_t cols = p.n - p.w + 1;
  const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= cols * p.matrices) return;
  const int64_t z = g / cols, c = g % cols;
  const int64_t mp = p.m - p.c0;
  const int64_t *piv = p.piv + z * p.spiv + p.c0;
  const int64_t *src_of = p.src_of + z * mp;
  const int t2 = (int)(2 * p.steps);
  if (c == cols - 1) {
    int64_t *perm = p.perm + z * p.sperm + p.c0, v[GATHER_MOST];
    for (int t = 0; t < t2; t++) {
      int64_t r = t < p.steps ? t : piv[t - p.steps] - p.c0;
      v[t] = perm[src_of[r]];
    }
    for (int t = 0; t < t2; t++) {
      int64_t r = t < p.steps ? t : piv[t - p.steps] - p.c0;
      perm[r] = v[t];
    }
    return;
  }
  /* The column's elements as W words of 4 bytes, a word at a time. */
  const int64_t col = c < p.c0 ? c : c + p.w;
  uint32_t *a = (uint32_t *)p.a + (z * p.sbat + p.c0 * p.lda + col) * W, v[GATHER_MOST];
  for (int w = 0; w < W; w++) {
    for (int t = 0; t < t2; t++) {
      int64_t r = t < p.steps ? t : piv[t - p.steps] - p.c0;
      v[t] = a[src_of[r] * p.lda * W + w];
    }
    for (int t = 0; t < t2; t++) {
      int64_t r = t < p.steps ? t : piv[t - p.steps] - p.c0;
      a[r * p.lda * W + w] = v[t];
    }
  }
}

/* One kernel of every format, its words a value by [fmt], so that
   gather_words is inlined once. */
template <typename T> __device__ void gather(const la_gather_params &p) {
  gather_words(p, p.fmt == NX_FMT_F32 ? 1 : p.fmt == NX_FMT_C128 ? 4 : 2);
}

/* The identity's rows: perm[g] = g mod m. */
template <typename T> __device__ void iota(const la_iota_params &p) {
  const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g < p.total) p.perm[g] = g % p.m;
}

/* Words of one value: flags' zeros, T's zero blocks. */
template <typename T> __device__ void fill(const la_fill_params &p) {
  const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g < p.count) p.x[g] = p.value;
}


/* Cholesky */

/* L of the diagonal block of order nb at (k, k) of each matrix, in place,
   right-looking: L_jj = sqrt (re a_jj), the column below divided by it,
   then each later element (i, c), c <= i, takes −conj(L_cj) L_ij, one
   nmac. Reads the block's lower triangle and the real parts of its
   diagonal alone. [fail] is set for a matrix that meets a pivot that is
   not positive, or NaN. G is 16 (a half-warp a matrix) or the block. */
template <typename T> __device__ void chol(const la_chol_params &p) {
  typedef R_of<T> R;
  extern __shared__ __align__(16) uint8_t smem[];
  const int G = p.G, groups = blockDim.x / G;
  const int g = threadIdx.x / G, lane = threadIdx.x % G;
  const unsigned mask = G < 32 ? (((1u << G) - 1) << (threadIdx.x % 32 / G * G)) : 0xffffffffu;
  const int64_t z = (int64_t)blockIdx.x * groups + g;
  if (z >= p.matrices) return;
  const int nb = (int)p.nb, ldt = p.ld;
  T *tile = (T *)smem + (int64_t)g * nb * ldt;
  const int64_t cs = p.cs;
  T *a = (T *)p.a + z * p.sbat + p.k * (p.lda + cs);
  auto sync = [&] {
    if (G <= 32) __syncwarp(mask);
    else __syncthreads();
  };
  const T *src = p.src ? (const T *)p.src + z * p.sbat : a;
  for (int r = 0; r < nb; r++)
    for (int c = lane; c <= r; c += G) tile[r * ldt + c] = src[r * p.lda + c * cs];
  sync();
  bool failed = false;
  for (int j = 0; j < nb; j++) {
    R d = re(tile[j * ldt + j]);
    failed |= !(d > 0);
    R s = sqrt(d);
    for (int i = j + 1 + lane; i < nb; i += G) tile[i * ldt + j] = divd(tile[i * ldt + j], mk<T>(s, 0));
    sync();
    if (lane == 0) tile[j * ldt + j] = mk<T>(s, 0);
    for (int i = j + 1 + lane; i < nb; i += G) {
      T l = tile[i * ldt + j];
      for (int c = j + 1; c <= i; c++)
        tile[i * ldt + c] = nmac(tile[i * ldt + c], conj_(tile[c * ldt + j]), l);
    }
    sync();
  }
  if (p.src) {
    for (int r = 0; r < nb; r++)
      for (int c = lane; c < nb; c += G) {
        T x = c == r ? tile[r * ldt + r]
              : p.upper ? (c > r ? conj_(tile[c * ldt + r]) : zero<T>())
                        : (c < r ? tile[r * ldt + c] : zero<T>());
        a[r * p.lda + c * cs] = failed ? mk<T>(NAN, NAN) : x;
      }
    return;
  }
  if (failed && lane == 0) p.fail[z] = 1;
  for (int r = 0; r < nb; r++)
    for (int c = lane; c <= r; c += G) a[r * p.lda + c * cs] = tile[r * ldt + c];
}

/* Each n × n matrix: NaN where [fail] is set; else under [upper] U = Lᴴ,
   each element below the diagonal moved conjugated above it, else the
   elements above the diagonal +0. With [tr], L lies above the diagonal,
   transposed: each element moved below it, else conjugated in place, and
   the other triangle +0. The diagonal stays. */
template <typename T> __device__ void chol_finish(const la_chol_finish_params &p) {
  const int64_t nn = p.n * p.n;
  const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= nn * p.matrices) return;
  const int64_t z = g / nn, e = g % nn, i = e / p.n, j = e % p.n;
  T *a = (T *)p.a + z * nn;
  if (p.fail[z]) {
    a[e] = mk<T>(NAN, NAN);
    return;
  }
  if (p.tr) {
    if (i > j && p.upper) a[e] = zero<T>();
    if (i >= j) return;
    if (p.upper) a[e] = conj_(a[e]);
    else a[j * p.n + i] = a[e], a[e] = zero<T>();
    return;
  }
  if (i <= j) {
    if (i < j && !p.upper) a[e] = zero<T>();
    return;
  }
  if (!p.upper) return;
  a[j * p.n + i] = conj_(a[e]);
  a[e] = zero<T>();
}

/* Triangular solves' diagonal: flag[z] = 1 where a diagonal element of
   matrix z is zero; then x NaN for a flagged matrix. */
template <typename R> __device__ __noinline__ void diag_zero_parts(const la_diag_params &p, int P) {
  const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= p.n * p.matrices) return;
  const int64_t z = g / p.n, i = g % p.n;
  const R *d = (const R *)p.a + (z * p.sa + i * p.sd) * P;
  if (d[0] == 0 && (P == 1 || d[1] == 0)) p.flag[z] = 1;
}

template <typename T> __device__ void diag_zero(const la_diag_params &p) {
  diag_zero_parts<R_of<T>>(p, cplx<T> ? 2 : 1);
}

template <typename T> __device__ void nan_where(const la_diag_params &p) {
  const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= p.xn * p.matrices) return;
  const int64_t z = g / p.xn;
  if (p.flag[z]) ((T *)p.x)[z * p.sx + g % p.xn] = mk<T>(NAN, NAN);
}


/* QR */

/* Sums over rows run in this order: the rows of the matrix in blocks of
   NX_LA_QBLK from row 0, each block in 16 lanes by row mod 16, each lane from
   +0 in increasing row, lane l taking lane l + 8, then l + 4, l + 2,
   l + 1; then the blocks from the panel's first, 32 to a superblock
   combined by the same tree over 32 (l + 16 first), superblocks added in
   increasing order from the first. A row outside the sum adds nothing to
   its lane. */
#define QSUP 32

/* Householder QR of the panel of columns [c0, c0 + w) of rows [c0, m) of
   each matrix f, in place: LAPACK's xGEQR2 with xLARFG's reflectors, the
   panel's rows held by P blocks as LU's are, in whole NX_LA_QBLK blocks of
   rows. Step j's sums over the rows below the diagonal, Σ|x_i|², the
   largest part of x and s_c = Σ conj(x_i) a_ic for the later columns,
   take one reduction (two where the maxima ask for a scale); then v = [1; x σ], σ = 1 / (α − β), w_c = a_jc +
   conj(σ) s_c, and a_ic −= (conj(τ) w_c) v_i, one nmac each. Then the
   dense V (ones on its diagonal, zeros above), where asked, and T with
   H_0 ⋯ H_(w−1) = I − V T Vᴴ (xLARFT), from Vᴴ V by the same reductions.
   Where the panel is the whole matrix (P = 1), it stores the first qc
   columns of Q = I − V (T V1ᴴ) and the qc rows of R. With P > 1 each
   meeting of the blocks is a launch boundary (see the stages below): the
   results are P = 1's.

   Shared memory, a group: vals [most], tc [w], T [w²], M [w qc] where Q
   is stored; then, where they fit, the reductions' store and the tile. */
template <typename T, bool MAX> __device__ __forceinline__ T combine(T x, T y) {
  if (MAX) return mk<T>(fmax(re(x), re(y)), 0);
  return mk<T>(re(x) + re(y), im(x) + im(y));
}

template <typename T> __device__ __forceinline__ T shfl_down(T x, int o, int width, unsigned mask) {
  return mk<T>(__shfl_down_sync(mask, re(x), o, width), __shfl_down_sync(mask, im(x), o, width));
}

/* A value's partials of [nb] blocks, [stride] apart, combined by one warp:
   superblocks of QSUP by the tree, then in order. Every lane returns it. */
template <typename T, bool MAX>
__device__ T blocks_tree(const T *part, int64_t stride, int64_t nb) {
  const int lane = threadIdx.x % 32;
  T total = zero<T>();
  for (int64_t s0 = 0; s0 < nb; s0 += QSUP) {
    const int64_t here = min((int64_t)QSUP, nb - s0);
    T x = zero<T>();
    if (lane < here) x = part[(s0 + lane) * stride];
    for (int o = 16; o >= 1; o /= 2) {
      T y = shfl_down(x, o, 32, 0xffffffffu);
      if (lane + o < here) x = combine<T, MAX>(x, y);
    }
    x = mk<T>(__shfl_sync(0xffffffffu, re(x), 0), __shfl_sync(0xffffffffu, im(x), 0));
    total = s0 == 0 ? x : combine<T, MAX>(total, x);
  }
  return total;
}

/* LAPACK's least normal over the unit roundoff. */
template <typename R> __device__ __forceinline__ R safmin() {
  return sizeof(R) == 4 ? (R)0x1p-102 : (R)0x1p-969;
}

/* Below it, or above its reciprocal, x's largest part takes the scaled
   sums, its squares near the normal range's ends; likewise the product
   of x's largest part with the later columns' (prod_lo), below which the
   largest products keep fewer than u's bits above the subnormals. */
template <typename R> __device__ __forceinline__ R safe_lo() {
  return sizeof(R) == 4 ? (R)0x1p-30 : (R)0x1p-250;
}
template <typename R> __device__ __forceinline__ int prod_lo() {
  return sizeof(R) == 4 ? -100 : -900;
}

/* sqrt (a² + b² + c²), scaled (xLAPY3). */
template <typename R> __device__ R lapy3(R a, R b, R c) {
  R xa = fabs(a), xb = fabs(b), xc = fabs(c);
  R w = fmax(xa, fmax(xb, xc));
  if (w == 0 || w != w) return xa + xb + xc;
  xa /= w;
  xb /= w;
  xc /= w;
  return w * sqrt(fma_(xa, xa, fma_(xb, xb, xc * xc)));
}

/* What a reduction sums a row: step j's [MAIN] values over x times an
   exact power of two 2^s (|2^s x|², its largest part, the later columns'
   largest part, conj(2^s x) a_c for each later column), or [GRAM] pairs
   of V's columns from pair v0. A pass takes [ALL] its values, the
   [MAXIMA] (MAIN's values 1 and 2) or the [SUMS] (the others). */
enum { MAIN, GRAM };
enum { ALL, MAXIMA, SUMS };

/* A panel's launches with P > 1, la_qr_params' [stage] (kernels.h):
   launch (A, k) finishes step k − 1 and takes step k's maxima, (B, k)
   step k's sums, (G, i) the Gram pairs' batch i − 1 into the pairs'
   store and batch i's partials, and after the last batch T. */
template <typename T> struct Qr {
  typedef R_of<T> R;
  const la_qr_params &p;
  int64_t z, q, w, ldt, c0, m, most, lo, hi, rows, nblocks, b_first;
  int G, lane;
  unsigned hmask;
  T *vals, *parts, *tile, *drow, *tc, *ts;
  int parity;

  __device__ void sync() {
    if (G <= 32) __syncwarp(G == 16 ? hmask : 0xffffffffu);
    else __syncthreads();
  }

  /* V's element (i, c) from the tile's row. */
  __device__ T vd(const T *row, int64_t i, int64_t c) {
    int64_t rr = i - c0;
    return rr > c ? row[c] : rr == c ? one<T>() : zero<T>();
  }

  __device__ T term(int mode, int v, T acc, const T *row, int64_t i, int64_t j, R big, int64_t v0) {
    if (mode == GRAM) {
      int64_t x = v0 + v, pa = 0;
      while ((pa + 1) * (2 * w - pa - 2) / 2 <= x) pa++;
      int64_t pb = x - pa * (2 * w - pa - 1) / 2 + pa + 1;
      return mac(acc, conj_(vd(row, i, pa)), vd(row, i, pb));
    }
    T x = row[j];
    if (big != 0) /* big: the exponent of the scale s = 2^big, exact; scalbn
                     for a subnormal x, whose s may exceed the largest float */
      x = mk<T>(scalbn(re(x), (int)big), scalbn(im(x), (int)big));
    if (v == 0) return mk<T>(fma_(re(x), re(x), fma_(im(x), im(x), re(acc))), 0);
    if (v == 1) return mk<T>(fmax(re(acc), fmax(fabs(re(x)), fabs(im(x)))), 0);
    if (v == 2) {
      R most = re(acc);
      for (int64_t c = j + 1; c < w; c++) most = fmax(most, fmax(fabs(re(row[c])), fabs(im(row[c]))));
      return mk<T>(most, 0);
    }
    return mac(acc, conj_(x), row[j + v - 2]);
  }

  static __device__ bool maximum(int mode, int v) { return mode == MAIN && (v == 1 || v == 2); }
  static __device__ bool chosen(int mode, int v, int sel) {
    return sel == ALL || (sel == MAXIMA) == maximum(mode, v);
  }

  /* This block's partials of [nv] values over the rows from [from], one
     per NX_LA_QBLK block of rows, into the store of [parity]. */
  __device__ __noinline__ void partials(int mode, int nv, int sel, int64_t from, int64_t j, R big,
                                        int64_t v0) {
    T *pp = parts + parity * nblocks * most;
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
        T y = shfl_down(acc, o, 16, hmask);
        acc = mx ? combine<T, true>(acc, y) : combine<T, false>(acc, y);
      }
      if (hl == 0) pp[(b - b_first) * most + v] = acc;
    }
  }

  /* Every block's partials of [parity]'s store combined, into vals. Values
     1 and 2 of MAIN are maxima. */
  __device__ __noinline__ void tree(int mode, int nv, int sel) {
    T *pp = parts + parity * nblocks * most;
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
  __device__ __noinline__ void reduce(int mode, int nv, int64_t from, int64_t j, R big, int64_t v0) {
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
  __device__ int scale(R big, R most) {
    /* The largest product's exponent, from the factors' own: their
       product may underflow to 0. */
    const int pe = most != 0 && most - most == 0 ? ilogb(big) + ilogb(most) : 0;
    if (big != 0 && big - big == 0 &&
        (big < safe_lo<R>() || big > 1 / safe_lo<R>() || pe < prod_lo<R>() || pe > -prod_lo<R>()))
      return ilogb(big);
    return 0;
  }

  /* This block's rows of the panel. */
  __device__ __noinline__ void load() {
    const T *a = (const T *)p.a + z * p.sbat;
    for (int64_t r = lane; r < rows; r += G)
      for (int64_t c = 0; c < w; c++) tile[r * ldt + c] = a[(lo + r) * p.lda + c0 + c];
    sync();
  }

  /* Step j's diagonal row, from its owner. */
  __device__ void diagonal(int64_t j) {
    const int64_t jr = c0 + j;
    T *dr = drow + (j & 1) * w;
    if (jr >= lo && jr < hi)
      for (int64_t c = j + lane; c < w; c += G) dr[c] = tile[(jr - lo) * ldt + c];
  }

  /* Step j from its sums in vals, over x 2^-e: the reflector, xLARFG's
     scaling of x, α and β by 1 / safmin until β is normal (each exact,
     the sums' scale following), τ, and the update of this block's
     rows. */
  __device__ __noinline__ void finish(int64_t j, int e) {
    const int64_t jr = c0 + j;
    const T *dr = drow + (j & 1) * w;
    const R sf = safmin<R>();
    const T alpha = dr[j];
    R ar = re(alpha), ai = im(alpha), beta = 0;
    R xnorm = scalbn(sqrt(re(vals[0])), e);
    const bool reflect = !(xnorm == 0 && ai == 0);
    int knt = 0;
    if (reflect) {
      beta = -copysign(lapy3(ar, ai, xnorm), ar);
      while (fabs(beta) < sf && knt < 20) {
        knt++;
        for (int64_t i = max(lo, jr + 1) + lane; i < hi; i += G) {
          T &x = tile[(i - lo) * ldt + j];
          x = mk<T>(re(x) * (1 / sf), im(x) * (1 / sf));
        }
        ar *= 1 / sf;
        ai *= 1 / sf;
        /* A zero x's sums are zero at any scale: e stays, σ's scaled
           copy finite. */
        if (re(vals[1]) != 0) e -= ilogb(sf);
        xnorm = scalbn(sqrt(re(vals[0])), e);
        beta = -copysign(lapy3(ar, ai, xnorm), ar);
      }
    }
    T tau = zero<T>(), sigma = zero<T>();
    if (reflect) {
      tau = mk<T>((beta - ar) / beta, -ai / beta);
      sigma = divd(one<T>(), mk<T>(ar - beta, ai));
      for (int c = 0; c < knt; c++) beta *= sf;
      /* w_c = a_jc + conj(σ) s_c, s_c = 2^e times the sums. */
      const T se = mk<T>(scalbn(re(sigma), e), scalbn(im(sigma), e));
      for (int64_t c = j + 1 + lane; c < w; c += G)
        tc[c] = mul(conj_(tau), mac(dr[c], conj_(se), vals[c - j + 2]));
    }
    if (lane == 0 && q == 0) ((T *)p.tau)[z * p.stau + c0 + j] = tau;
    ts[j * w + j] = tau; /* T's diagonal, for T below */
    sync();
    for (int64_t i = max(lo, jr) + lane; i < hi && reflect; i += G) {
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

  /* The factored panel and dense V from this block's rows. */
  __device__ __noinline__ void panel() {
    T *a = (T *)p.a + z * p.sbat;
    T *vo = p.v ? (T *)p.v + z * p.sv : nullptr;
    for (int64_t r = lane; r < rows && !p.q; r += G)
      for (int64_t c = 0; c < w; c++) {
        a[(lo + r) * p.lda + c0 + c] = tile[r * ldt + c];
        if (vo) vo[(lo + r - c0) * w + c] = vd(tile + r * ldt, lo + r, c);
      }
  }

  /* T: T_ii = τ_i, T[0:i, i] = −τ_i T[0:i, 0:i] g[0:i, i] from the pairs
     g, a column a turn, lane r its row r; one warp, or the half-warp. */
  __device__ __noinline__ void larft(const T *gram) {
    if (lane < min(G, 32)) {
      const unsigned wm = G == 16 ? hmask : 0xffffffffu;
      for (int64_t i = 0; i < w; i++) {
        const T ti = ts[i * w + i];
        for (int64_t r = lane; r < w; r += min(G, 32)) {
          if (r >= i) {
            if (r > i) ts[r * w + i] = zero<T>();
            continue;
          }
          T x = zero<T>();
          for (int64_t c = r; c < i; c++) x = mac(x, ts[r * w + c], gram[c * (2 * w - c - 1) / 2 + (i - c - 1)]);
          ts[r * w + i] = mul(neg(ti), x);
        }
        __syncwarp(wm);
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

template <typename T> __device__ void qr_panel(const la_qr_params &p) {
  typedef R_of<T> R;
  extern __shared__ __align__(16) uint8_t smem[];
  const bool many = p.P > 1;
  const int G = many ? (int)blockDim.x : (int)p.G, groups = blockDim.x / G;
  const int g = threadIdx.x / G, lane = threadIdx.x % G;
  Qr<T> s{p};
  s.z = many ? blockIdx.x / p.P : (int64_t)blockIdx.x * groups + g;
  s.q = many ? blockIdx.x % p.P : 0;
  if (s.z >= p.matrices) return;
  const int64_t w = p.w, c0 = p.c0, m = p.m, most = p.most, z = s.z;
  s.w = w;
  s.ldt = p.ld;
  s.c0 = c0;
  s.m = m;
  s.most = most;
  s.G = G;
  s.lane = lane;
  s.hmask = 0xffffu << (threadIdx.x % 32 / 16 * 16);
  s.b_first = c0 / NX_LA_QBLK;
  s.nblocks = (m - 1) / NX_LA_QBLK - s.b_first + 1;
  const int64_t bpc = p.per / NX_LA_QBLK;
  s.lo = max(c0, (s.b_first + s.q * bpc) * NX_LA_QBLK);
  s.hi = min(m, (s.b_first + (s.q + 1) * bpc) * NX_LA_QBLK);
  s.rows = max((int64_t)0, s.hi - s.lo);
  const int64_t npair = w * (w - 1) / 2;
  const int64_t hdr = r16d((most + w + w * w + npair + (p.q ? w * p.qc : 0)) * sizeof(T));
  uint8_t *mine = smem + g * (hdr + (p.parts_in_smem ? p.parts_stride * sizeof(T) : 0) +
                              (p.in_smem ? p.spill_bytes : 0));
  T *vals = (T *)mine, *tc = vals + most, *ts = tc + w, *gram = ts + w * w, *M = gram + npair;
  s.vals = vals;
  s.tc = tc;
  s.ts = ts;
  s.parts = p.parts_in_smem ? (T *)(mine + hdr) : (T *)p.parts + z * p.parts_stride;
  s.tile = p.in_smem ? (T *)(mine + hdr + (p.parts_in_smem ? p.parts_stride * sizeof(T) : 0))
                     : (T *)((uint8_t *)p.spill + ((int64_t)blockIdx.x * groups + g) * p.spill_bytes);
  T *tile = s.tile;
  s.drow = s.parts + 2 * s.nblocks * most; /* two of w, by step parity */
  T *gstore = s.drow + 2 * w;               /* with P > 1, the pairs */
  s.parity = 0;
  const int64_t steps = min(m - c0, w);
  /* Gram pairs' batches, and step j's values. */
  const int64_t batches = cdiv(npair, most);
  auto nv = [&](int64_t j) { return (int)(w - j + 2); };
  auto nvg = [&](int64_t i) { return (int)min(most, npair - i * most); };

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
    /* NX_LA_QR_STEP_G, k = i: batch i − 1 into the store, batch i's sums; after
       the last, T. */
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
  /* The whole matrix: M = T V1ᴴ (w × qc), Q = I − V M, R. */
  const int64_t qc = p.qc;
  for (int64_t c = lane; c < qc; c += G)
    for (int64_t r = 0; r < w; r++) {
      T x = zero<T>();
      for (int64_t s2 = r; s2 < w; s2++) x = mac(x, ts[r * w + s2], conj_(s.vd(tile + c * s.ldt, c, s2)));
      M[r * qc + c] = x;
    }
  s.sync();
  T *qo = (T *)p.q + z * m * qc;
  for (int64_t i = 0; i < m; i++)
    for (int64_t c = lane; c < qc; c += G) {
      T x = i == c ? one<T>() : zero<T>();
      for (int64_t s2 = 0; s2 < w && s2 <= i; s2++) x = nmac(x, s.vd(tile + i * s.ldt, i, s2), M[s2 * qc + c]);
      qo[i * qc + c] = x;
    }
  T *ro = (T *)p.r + z * qc * p.n;
  for (int64_t i = 0; i < qc; i++)
    for (int64_t c = lane; c < p.n; c += G) ro[i * p.n + c] = c >= i ? tile[i * s.ldt + c] : zero<T>();
}

/* QR's other steps */

/* q = the first qc columns of the m × m identity; r = the upper triangle
   of f's first rows (rows past m zero). */
template <typename R> __device__ __noinline__ void qr_out_parts(const la_qr_out_params &p, int P) {
  const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t nq = p.m * p.qc, nr = p.qc * p.n;
  if (g >= (nq + nr) * p.matrices) return;
  const int64_t z = g / (nq + nr), e = g % (nq + nr);
  if (e < nq) {
    R *x = (R *)p.q + (z * nq + e) * P;
    x[0] = e / p.qc == e % p.qc ? 1 : 0;
    if (P == 2) x[1] = 0;
    return;
  }
  const int64_t i = (e - nq) / p.n, c = (e - nq) % p.n;
  R *x = (R *)p.r + (z * nr + e - nq) * P;
  const R *f = (const R *)p.f + (z * p.m * p.n + i * p.n + c) * P;
  for (int q = 0; q < P; q++) x[q] = c >= i && i < p.m ? f[q] : 0;
}

template <typename T> __device__ void qr_out(const la_qr_out_params &p) {
  qr_out_parts<R_of<T>>(p, cplx<T> ? 2 : 1);
}

/* Dense V of a panel from f's columns [c0, c0 + w): rows [c0, m). */
template <typename R> __device__ __noinline__ void dense_v_parts(const la_dense_params &p, int P) {
  const int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t h = p.m - p.c0, nv = h * p.w;
  if (g >= nv * p.matrices) return;
  const int64_t z = g / nv, e = g % nv, r = e / p.w, c = e % p.w;
  const R *x = (const R *)p.f + (z * p.m * p.n + (p.c0 + r) * p.n + p.c0 + c) * P;
  R *v = (R *)p.v + (z * nv + e) * P;
  v[0] = r > c ? x[0] : r == c ? 1 : 0;
  if (P == 2) v[1] = r > c ? x[1] : 0;
}

template <typename T> __device__ void dense_v(const la_dense_params &p) {
  dense_v_parts<R_of<T>>(p, cplx<T> ? 2 : 1);
}

} /* namespace nx_factor */

/* Entries */

#define LA_T_f32 float
#define LA_T_f64 double
#define LA_T_c64 nx_factor::c64
#define LA_T_c128 nx_factor::c128
#define LA_T_any float
#define LA_THREADS_trsm NX_LA_TRSM_THREADS
#define LA_THREADS_gather 128
#define LA_THREADS_lu_panel 256
#define LA_THREADS_chol 256
#define LA_THREADS_chol_finish 256
#define LA_THREADS_diag_zero 256
#define LA_THREADS_nan_where 256
#define LA_THREADS_qr_panel 256
#define LA_THREADS_qr_out 256
#define LA_THREADS_dense_v 256
#define LA_THREADS_iota 256
#define LA_THREADS_fill 256

#define LINALG(name, entry, params, fmt)                                      \
  extern "C" __global__ void __launch_bounds__(LA_THREADS_##entry)            \
      name(const __grid_constant__ la_##params##_params p) {                  \
    nx_factor::entry<LA_T_##fmt>(p);                                          \
  }

/* One entry for every format, by the parameters' [fmt]. */
#define LINALG_ANY(name, entry, params)                                       \
  extern "C" __global__ void __launch_bounds__(LA_THREADS_##entry)            \
      name(const __grid_constant__ la_##params##_params p) {                  \
    switch (p.fmt) {                                                          \
    case NX_FMT_F32: nx_factor::entry<LA_T_f32>(p); break;                    \
    case NX_FMT_F64: nx_factor::entry<LA_T_f64>(p); break;                    \
    case NX_FMT_C64: nx_factor::entry<LA_T_c64>(p); break;                    \
    default: nx_factor::entry<LA_T_c128>(p);                                  \
    }                                                                         \
  }
