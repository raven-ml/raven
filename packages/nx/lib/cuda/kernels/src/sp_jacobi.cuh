/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* eigh and svd of small matrices, each in its block's shared memory, by
   cyclic Jacobi in parallel order.

   A round rotates n/2 disjoint pairs of indices at once, the pairs of the
   round-robin tournament (Brent and Luk 1985), so that n − 1 rounds, n
   rounded up to even, rotate every pair once: a sweep. A pair is rotated
   where its coupling is not negligible against its diagonal, and sweeps
   repeat until one rotates nothing (svd: nothing beyond rounding's
   reach). A rotation J = [c, s·e; −s·ē, c], e the coupling's phase,
   makes the pair's 2 × 2 Hermitian problem diagonal (Golub and Van
   Loan, Alg. 8.5.1, made complex by the phase).

   eigh rotates both sides of a, a ← Jᴴ a J, and v ← v J. svd rotates the
   columns of the tall orientation g of a, g ← g J, v ← v J, the pair's
   problem being their Gram matrix (Hestenes's one-sided Jacobi); at the
   end g's columns are the singular values times u's columns.

   A block holds [per] matrices, NT / (their elements), at least one; the
   block's threads split into one group per matrix. Every group runs the
   same rounds until no matrix of the block rotates, so a matrix's bits
   are those it would have alone: a sweep that rotates nothing changes
   nothing. Shared reductions combine in a fixed order or by integer
   maxima and ands, which no order changes. */

#include "sp_common.cuh"

#define NT NX_SP_THREADS
/* Sweeps before a matrix is declared not to converge: Jacobi converges
   quadratically, in about 6 to 12 sweeps here; xGESVJ allows 30. */
#define MAX_SWEEPS 60

/* The pair [i] of round [r] among [np] indices, np even: index np − 1
   meets r, and the others meet by reflection about r. */
D void pair_of(int r, int i, int np, int *p, int *q) {
  int a, b;
  if (i == 0) {
    a = np - 1;
    b = r;
  } else {
    a = (r + i) % (np - 1);
    b = (r - i + np - 1) % (np - 1);
  }
  *p = a < b ? a : b;
  *q = a < b ? b : a;
}

/* The rotation of the Hermitian 2 × 2 [x y; ȳ z], y ≠ 0: column p of J
   is (c, −s·ē), column q (s·e, c), e = y / |y|, and the new diagonal is
   x − t|y|, z + t|y|. */
template <class T, class R>
D void rotation(R x, R z, T y, R g, R *c, T *bp, T *bq, R *t) {
  T e = cdiv(y, elt<T>(g));
  R tau = (z - x) / (2 * g);
  *t = (tau >= 0 ? (R)1 : (R)-1) / (fabs(tau) + hypot((R)1, tau));
  *c = 1 / hypot((R)1, *t);
  R s = *t * *c;
  *bp = elt<T>(-s) * conj(e);
  *bq = elt<T>(s) * e;
}

/* The larger magnitude of [x]'s parts. */
template <class T> D typename real_of<T>::R part(T x) {
  return fmax(fabs(re(x)), fabs(im(x)));
}

/* Shared state: per matrix of the block, its scale's exponent, flags,
   largest magnitude's bits; and the arrays each round reads. */
struct slots {
  int e[NT], finite[NT], rotated[NT], failed[NT];
  unsigned long long big[NT];
};

template <class R> D unsigned long long bits_of(R x) {
  return sizeof(R) == 4 ? (unsigned long long)__float_as_uint((float)x)
                        : (unsigned long long)__double_as_longlong(x);
}
template <class R> D R of_bits(unsigned long long b) {
  return sizeof(R) == 4 ? (R)__uint_as_float((unsigned)b)
                        : (R)__longlong_as_double((long long)b);
}

/* Element [x]'s part of a matrix's scan: its slot made not finite where
   it is NaN or infinite, its largest part's magnitude into the maximum,
   whose bits order as the magnitudes do. */
template <class T> D void scan_one(slots *sl, int slot, T x) {
  typedef typename real_of<T>::R R;
  R nan = (re(x) - re(x)) + (im(x) - im(x));
  if (nan != 0) atomicAnd(&sl->finite[slot], 0);
  R m = fmax(fabs(re(x)), fabs(im(x)));
  if (m == m) atomicMax(&sl->big[slot], bits_of(m));
}

template <class T> D T nan_of() {
  typedef typename real_of<T>::R R;
  R n = (R)NAN;
  return elt<T>(n, n);
}


/* Eigh */

/* Shared memory per matrix: a, v, the round's α and β per index (T),
   partner and rank per index (int), new diagonal (R). */
__host__ __device__ inline int eigh_per(int n) { return nx_sp_eigh_per(n); }

/* w and, where [v] is not NULL, v of the [nb] Hermitian n × n matrices
   at [a], 1 ≤ n ≤ 32. Launch NT threads a block, any grid, eigh_smem
   bytes of dynamic shared memory. */
template <class T>
__device__ void eigh_jacobi(const T *a, typename real_of<T>::R *w, T *v,
                            int64_t nb, int n) {
  typedef typename real_of<T>::R R;
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ slots sl;
  const int nn = n * n, per = eigh_per(n), g = NT / per;
  const int slot = threadIdx.x / g, lt = threadIdx.x % g;
  const int np = n + (n & 1);
  T *A = (T *)smem + (size_t)slot * (2 * nn + 2 * n), *V = A + nn,
    *al = V + nn, *be = al + n;
  int *ints = (int *)((T *)smem + (size_t)per * (2 * nn + 2 * n));
  int *pa = ints + slot * n;
  R *dn = (R *)(ints + per * n + (per * n & 1)) + slot * n;
  const R eps = lim<R>::eps(), tiny = lim<R>::tiny();
  const R rmin = sqrt(tiny / (2 * eps)), rmax = 1 / rmin;
  for (int64_t base = (int64_t)blockIdx.x * per; base < nb;
       base += (int64_t)gridDim.x * per) {
    const int64_t b = base + slot;
    const bool live = slot < per && b < nb;
    __syncthreads();
    if (threadIdx.x < per) {
      sl.finite[threadIdx.x] = 1;
      sl.rotated[threadIdx.x] = 0;
      sl.failed[threadIdx.x] = 0;
      sl.big[threadIdx.x] = 0;
    }
    __syncthreads();
    /* The lower triangle and the diagonal's real parts, which alone are
       read, and the upper triangle their conjugates. */
    if (live)
      #pragma unroll 1
      for (int k = lt; k < nn; k += g) {
        int i = k / n, j = k % n;
        if (i < j) continue;
        T x = a[(size_t)b * nn + k];
        if (i == j) x = elt<T>(re(x));
        scan_one(&sl, slot, x);
        A[k] = x;
        A[j * n + i] = conj(x);
        if (v) {
          V[k] = elt<T>((R)(i == j));
          if (i != j) V[j * n + i] = elt<T>((R)0);
        }
      }
    __syncthreads();
    const int e = scale_exp(of_bits<R>(sl.big[slot]), rmin, rmax);
    const bool run = live && sl.finite[slot];
    if (run && e != 0)
      #pragma unroll 1
      for (int k = lt; k < nn; k += g) A[k] = scaled(A[k], e);
    bool go = true;
    #pragma unroll 1
    for (int sweep = 0; go; sweep++) {
      #pragma unroll 1
      for (int r = 0; r < np - 1; r++) {
        __syncthreads();
        if (run && lt < np / 2) {
          int p, q;
          pair_of(r, lt, np, &p, &q);
          if (q < n) {
            R x = re(A[p * n + p]), z = re(A[q * n + q]);
            T y = A[p * n + q];
            R gy = mag(y);
            pa[p] = pa[q] = -1;
            if (gy > tiny && gy > eps * sqrt(fabs(x)) * sqrt(fabs(z))) {
              R c, t;
              rotation(x, z, y, gy, &c, &be[p], &be[q], &t);
              al[p] = al[q] = elt<T>(c);
              pa[p] = q;
              pa[q] = p;
              dn[p] = x - t * gy;
              dn[q] = z + t * gy;
              sl.rotated[slot] = 1;
            }
          } else if (p < n)
            pa[p] = -1;
        }
        __syncthreads();
        /* a ← Jᴴ a J and v ← v J, each element from the old ones; a
           rotated pair's coupling is 0 and its diagonal the rotation's. */
        T na[4], nv[4];
        if (run)
          #pragma unroll 1
          for (int k = lt, c = 0; k < nn; k += g, c++) {
            int i = k / n, j = k % n, pi = pa[i], pj = pa[j];
            T hi = A[k], hp = pi >= 0 ? A[pi * n + j] : hi;
            if (pj >= 0) {
              hi = al[j] * hi + be[j] * A[i * n + pj];
              if (pi >= 0) hp = al[j] * hp + be[j] * A[pi * n + pj];
            }
            T x = pi >= 0 ? conj(al[i]) * hi + conj(be[i]) * hp : hi;
            if (pi >= 0 && j == pi) x = elt<T>((R)0);
            if (pi >= 0 && j == i) x = elt<T>(dn[i]);
            na[c] = x;
            if (v)
              nv[c] = pj >= 0 ? al[j] * V[k] + be[j] * V[i * n + pj] : V[k];
          }
        __syncthreads();
        if (run)
          #pragma unroll 1
          for (int k = lt, c = 0; k < nn; k += g, c++) {
            A[k] = na[c];
            if (v) V[k] = nv[c];
          }
      }
      __syncthreads();
      const int mine = run && sl.rotated[slot];
      go = __syncthreads_or(mine);
      if (go && sweep + 1 == MAX_SWEEPS) {
        if (mine && lt == 0) sl.failed[slot] = 1;
        go = false;
      }
      if (threadIdx.x < per) sl.rotated[threadIdx.x] = 0;
    }
    __syncthreads();
    const bool ok = run && !sl.failed[slot];
    /* Ascending, ties in index order, and v's columns with them, each
       normalised: a rotation of angle t below √u has c = 1 in floating
       point and lengthens its two columns by 1 + t²/2, always longer. */
    if (ok)
      #pragma unroll 1
      for (int i = lt; i < n; i += g) {
        R x = re(A[i * n + i]);
        int rank = 0;
        #pragma unroll 1
        for (int j = 0; j < n; j++) {
          R y = re(A[j * n + j]);
          rank += y < x || (y == x && j < i);
        }
        dn[i] = x;
        pa[i] = rank;
        if (!v) continue;
        sumsq<R> sum;
        #pragma unroll 1
        for (int j = 0; j < n; j++) sum.sq(V[j * n + i]);
        R norm = sqrt(sum.get());
        #pragma unroll 1
        for (int j = 0; j < n; j++) V[j * n + i] = V[j * n + i] / norm;
      }
    __syncthreads();
    if (live && !ok) {
      #pragma unroll 1
      for (int k = lt; k < n; k += g) w[(size_t)b * n + k] = (R)NAN;
      #pragma unroll 1
      for (int k = lt; v && k < nn; k += g)
        v[(size_t)b * nn + k] = nan_of<T>();
    }
    if (ok) {
      #pragma unroll 1
      for (int i = lt; i < n; i += g)
        w[(size_t)b * n + pa[i]] = scaled(dn[i], -e) + (R)0;
      #pragma unroll 1
      for (int k = lt; v && k < nn; k += g)
        v[(size_t)b * nn + k - k % n + pa[k % n]] = V[k];
    }
  }
}

/* Svd */

/* Shared memory per matrix: g (M × K), v (K × K), u (M × M), α and β per
   column, a projection per row (T); rank and its inverse per column
   (int); norms per column and weights per row (R). */
__host__ __device__ inline size_t svd_span(int M, int K) { return nx_sp_svd_span(M, K); }
/* Matrices per block: as many as the threads take, as long as their
   arrays fit NX_SP_JACOBI_SMEM (kernels.h). */
__host__ __device__ inline int svd_per(int m, int n, int tb, int rb) {
  return nx_sp_svd_per(m, n, tb, rb);
}

/* s and, where [u] is not NULL, u and vh of the [nb] m × n matrices at
   [a], 1 ≤ m, n ≤ 32, [uc] the columns of u and [vr] the rows of vh
   (reduced or complete). g is a, or aᴴ where a is wide, M × K. Launch NT
   threads a block, any grid, svd_smem bytes of dynamic shared memory. */
template <class T>
__device__ void svd_jacobi(const T *a, typename real_of<T>::R *s, T *u,
                           T *vh, int64_t nb, int m, int n, int uc, int vr) {
  typedef typename real_of<T>::R R;
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ slots sl;
  const bool wide = m < n, vec = u != NULL;
  const int M = wide ? n : m, K = wide ? m : n, mk = M * K;
  /* The columns of g's left factor the call stores: u's, or vh's rows. */
  const int gc = wide ? vr : uc;
  const int per = svd_per(m, n, sizeof(T), sizeof(R)), g = NT / per;
  const int slot = threadIdx.x / g, lt = threadIdx.x % g;
  const int np = K + (K & 1);
  const size_t span = svd_span(M, K);
  T *G = (T *)smem + slot * span, *V = G + mk, *U = V + K * K,
    *al = U + M * M, *be = al + K, *pr = be + K;
  int *ints = (int *)((T *)smem + per * span);
  int *pa = ints + slot * 2 * K, *iv = pa + K;
  R *sg = (R *)(ints + per * 2 * K) + slot * (K + M), *sc = sg + K;
  const R eps = lim<R>::eps(), tiny = lim<R>::tiny();
  const R small = sqrt(tiny) / (2 * eps);
  /* xGESVJ's tolerance, √M u, and its test of convergence: a sweep
     whose cosines all lie below √K tol and for which K max cs max |t|
     lies below tol has converged. Rounding can hold a pair's cosine just
     above tol, where a rotation of a few ulps gives it back sweep after
     sweep. A sweep whose rotations all have cosines at most [near] and
     tangents at most [slight] meets the test, and asks for no other. */
  const R tol = sqrt((R)M) * eps, near = sqrt((R)K) * tol;
  const R slight = 1 / (K * sqrt((R)K));
  for (int64_t base = (int64_t)blockIdx.x * per; base < nb;
       base += (int64_t)gridDim.x * per) {
    const int64_t b = base + slot;
    const bool live = slot < per && b < nb;
    __syncthreads();
    if (threadIdx.x < per) {
      sl.finite[threadIdx.x] = 1;
      sl.rotated[threadIdx.x] = 0;
      sl.failed[threadIdx.x] = 0;
      sl.big[threadIdx.x] = 0;
    }
    __syncthreads();
    if (live) {
      #pragma unroll 1
      for (int k = lt; k < mk; k += g) {
        int i = k / n, j = k % n;
        T x = a[(size_t)b * mk + k];
        scan_one(&sl, slot, x);
        if (wide)
          G[j * K + i] = conj(x);
        else
          G[k] = x;
      }
      #pragma unroll 1
      for (int k = lt; vec && k < K * K; k += g)
        V[k] = elt<T>((R)(k / K == k % K));
    }
    __syncthreads();
    const int e = scale_exp(of_bits<R>(sl.big[slot]), small, 1 / small);
    /* A column of norm at most [keep], u times the scaled matrix's
       largest part, is completed, not rotated. On a matrix of lower rank,
       rotations leave rounding of about u ‖a‖ in the columns that span
       nothing, and each sweep shrinks it by only some orders of
       magnitude: carried down to the least normal number, it takes more
       sweeps than the limit. Completing K such columns moves a by at
       most 2 √K u ‖a‖_F, within the bound. [keep] is at least u [small],
       so a column above it has normal digits to divide by. */
    const R keep = eps * scaled(of_bits<R>(sl.big[slot]), e);
    const bool run = live && sl.finite[slot];
    if (run && e != 0)
      #pragma unroll 1
      for (int k = lt; k < mk; k += g) G[k] = scaled(G[k], e);
    bool go = true;
    #pragma unroll 1
    for (int sweep = 0; go; sweep++) {
      #pragma unroll 1
      for (int r = 0; r < np - 1; r++) {
        __syncthreads();
        if (run && lt < np / 2) {
          int p, q;
          pair_of(r, lt, np, &p, &q);
          if (q < K) {
            /* The pair's cosine from the columns each scaled by its
               largest part, so that no square under- or overflows; the
               rotation from it and the norms' ratio (xGESVJ). */
            R sp = 0, sq = 0, x = 0, z = 0;
            T y = elt<T>((R)0);
            for (int i = 0; i < M; i++) {
              sp = fmax(sp, part(G[i * K + p]));
              sq = fmax(sq, part(G[i * K + q]));
            }
            pa[p] = pa[q] = -1;
            if (sp > 0 && sq > 0) {
              for (int i = 0; i < M; i++) {
                T gp = G[i * K + p] / sp, gq = G[i * K + q] / sq;
                x += abs2(gp);
                z += abs2(gq);
                y += conj(gp) * gq;
              }
              R np = sp * sqrt(x), nq = sq * sqrt(z);
              R cs = mag(y) / (sqrt(x) * sqrt(z));
              /* A column at most [keep] is completed, not rotated: its
                 singular value is its norm. */
              if (cs > tol && fmin(np, nq) > keep) {
                R tau = (nq / np - np / nq) / (2 * cs);
                R t = (tau >= 0 ? (R)1 : (R)-1) /
                      (fabs(tau) + hypot((R)1, tau));
                if (t != 0) {
                  R c = 1 / hypot((R)1, t), s = t * c;
                  T e = cdiv(y, elt<T>(mag(y)));
                  be[p] = elt<T>(-s) * conj(e);
                  be[q] = elt<T>(s) * e;
                  al[p] = al[q] = elt<T>(c);
                  pa[p] = q;
                  pa[q] = p;
                  if (cs > near || fabs(t) > slight) sl.rotated[slot] = 1;
                }
              }
            }
          } else if (p < K)
            pa[p] = -1;
        }
        __syncthreads();
        /* g ← g J and v ← v J. */
        T ng[4], nv[4];
        if (run)
          #pragma unroll 1
          for (int k = lt, c = 0; k < mk; k += g, c++) {
            int i = k / K, j = k % K, pj = pa[j];
            ng[c] = pj >= 0 ? al[j] * G[k] + be[j] * G[i * K + pj] : G[k];
            if (vec && k < K * K)
              nv[c] = pj >= 0 ? al[j] * V[k] + be[j] * V[k - j + pj] : V[k];
          }
        __syncthreads();
        if (run)
          #pragma unroll 1
          for (int k = lt, c = 0; k < mk; k += g, c++) {
            G[k] = ng[c];
            if (vec && k < K * K) V[k] = nv[c];
          }
      }
      __syncthreads();
      const int mine = run && sl.rotated[slot];
      go = __syncthreads_or(mine);
      if (go && sweep + 1 == MAX_SWEEPS) {
        if (mine && lt == 0) sl.failed[slot] = 1;
        go = false;
      }
      if (threadIdx.x < per) sl.rotated[threadIdx.x] = 0;
    }
    __syncthreads();
    const bool ok = run && !sl.failed[slot];
    /* The singular values, each column's norm, and v's columns
       normalised, as eigh's are; both norms by compensated sums. */
    if (ok)
      #pragma unroll 1
      for (int j = lt; j < K; j += g) {
        R sj = 0;
        sumsq<R> x;
        #pragma unroll 1
        for (int i = 0; i < M; i++) sj = fmax(sj, part(G[i * K + j]));
        #pragma unroll 1
        for (int i = 0; i < M && sj > 0; i++) x.sq(G[i * K + j] / sj);
        sg[j] = sj * sqrt(x.get());
        if (!vec) continue;
        sumsq<R> y;
        #pragma unroll 1
        for (int i = 0; i < K; i++) y.sq(V[i * K + j]);
        R norm = sqrt(y.get());
        #pragma unroll 1
        for (int i = 0; i < K; i++) V[i * K + j] = V[i * K + j] / norm;
      }
    __syncthreads();
    /* Descending, ties in index order. */
    if (ok)
      #pragma unroll 1
      for (int j = lt; j < K; j += g) {
        int rank = 0;
        #pragma unroll 1
        for (int i = 0; i < K; i++)
          rank += sg[i] > sg[j] || (sg[i] == sg[j] && i < j);
        pa[j] = rank;
        iv[rank] = j;
      }
    __syncthreads();
    /* u's columns, those whose norm exceeds [keep]. */
    int kept = 0;
    #pragma unroll 1
    for (int j = 0; ok && j < K; j++) kept += sg[j] > keep;
    if (ok && vec)
      #pragma unroll 1
      for (int k = lt; k < mk; k += g) {
        int i = k / K, j = k % K;
        if (sg[j] > keep) U[i * M + pa[j]] = G[k] / sg[j];
      }
    /* Columns [kept, gc) of g's left factor: e_i less its projection on
       the columns before, twice, normalised, i the row those columns
       weigh least, the first of equals (Gram-Schmidt completion). Every
       group runs gc steps, so that the block's barriers match. */
    #pragma unroll 1
    for (int c = 0; vec && c < gc; c++) {
      const bool mine = ok && c >= kept && lt < M;
      __syncthreads();
      if (mine) {
        R x = 0;
        #pragma unroll 1
        for (int j = 0; j < c; j++) x += abs2(U[lt * M + j]);
        sc[lt] = x;
      }
      __syncthreads();
      T x = elt<T>((R)0);
      if (mine) {
        int best = 0;
        #pragma unroll 1
        for (int i = 1; i < M; i++)
          if (sc[i] < sc[best]) best = i;
        x = elt<T>((R)(lt == best));
        #pragma unroll 1
        for (int j = 0; j < c; j++) x -= U[lt * M + j] * conj(U[best * M + j]);
      }
      __syncthreads();
      if (mine) U[lt * M + c] = x;
      __syncthreads();
      if (mine && lt < c) {
        T d = elt<T>((R)0);
        #pragma unroll 1
        for (int i = 0; i < M; i++) d += conj(U[i * M + lt]) * U[i * M + c];
        pr[lt] = d;
      }
      __syncthreads();
      if (mine) {
        x = U[lt * M + c];
        #pragma unroll 1
        for (int j = 0; j < c; j++) x -= U[lt * M + j] * pr[j];
      }
      __syncthreads();
      if (mine) U[lt * M + c] = x;
      __syncthreads();
      R nrm = 0;
      #pragma unroll 1
      for (int i = 0; mine && i < M; i++) nrm += abs2(U[i * M + c]);
      __syncthreads();
      if (mine) U[lt * M + c] = x / sqrt(nrm);
    }
    __syncthreads();
    const int64_t K64 = K;
    if (live && !ok) {
      #pragma unroll 1
      for (int k = lt; k < K; k += g) s[b * K64 + k] = (R)NAN;
      #pragma unroll 1
      for (int k = lt; vec && k < m * uc; k += g)
        u[(size_t)b * m * uc + k] = nan_of<T>();
      #pragma unroll 1
      for (int k = lt; vec && k < vr * n; k += g)
        vh[(size_t)b * vr * n + k] = nan_of<T>();
    }
    if (!ok) continue;
    #pragma unroll 1
    for (int j = lt; j < K; j += g)
      s[b * K64 + pa[j]] = scaled(sg[j], -e) + (R)0;
    if (!vec) continue;
    /* Tall: u = U and vh = Vᴴ. Wide: u = V and vh = Uᴴ. */
    #pragma unroll 1
    for (int k = lt; k < m * uc; k += g) {
      int i = k / uc, j = k % uc;
      u[(size_t)b * m * uc + k] = wide ? V[i * K + iv[j]] : U[i * M + j];
    }
    #pragma unroll 1
    for (int k = lt; k < vr * n; k += g) {
      int i = k / n, j = k % n;
      vh[(size_t)b * vr * n + k] =
          wide ? conj(U[j * M + i]) : conj(V[j * K + iv[i]]);
    }
  }
}

