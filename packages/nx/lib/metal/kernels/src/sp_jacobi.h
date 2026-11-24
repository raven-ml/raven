/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* eigh and svd of small matrices, each in its group's threadgroup memory,
   by cyclic Jacobi in parallel order.

   A round rotates n/2 disjoint pairs of indices at once, the pairs of the
   round-robin tournament (Brent and Luk 1985), so that n − 1 rounds, n
   rounded up to even, rotate every pair once: a sweep. A pair is rotated
   where its coupling is not negligible against its diagonal, and sweeps
   repeat until one rotates nothing (svd: nothing beyond rounding's
   reach). A rotation J = [c, s e; −s ē, c], e the coupling's phase,
   makes the pair's 2 x 2 Hermitian problem diagonal (Golub and Van
   Loan, Alg. 8.5.1, made complex by the phase).

   eigh rotates both sides of a, a ← Jᴴ a J, and v ← v J. svd rotates the
   columns of the tall orientation g of a, g ← g J, v ← v J, the pair's
   problem being their Gram matrix (Hestenes's one-sided Jacobi); at the
   end g's columns are the singular values times u's columns.

   A group holds [per] matrices, NT / (their elements), at least one; the
   group's threads split into a set a matrix. Every set runs the group's
   rounds, its barriers among them, until no matrix of the group rotates,
   but a matrix rotates only until its own sweeps stop: eigh's at the
   first sweep that rotates nothing, which changes nothing; svd's at the
   first whose rotations are all within rounding's reach, after which it
   is left as it is. So a matrix's bits are those it has alone, whatever
   [per] and its neighbours. Shared reductions combine in a fixed order
   or by integer maxima and ands, which no order changes.

   A launch takes the groups' units, [per] matrices each, from [first] to
   [count] − 1. A group's threadgroup memory holds the slots (jacobi_head
   bytes) and then the matrices, as kernels.h states.

   This follows nx.amd's sp_jacobi.h. */

#ifndef NX_METAL_SP_JACOBI_H
#define NX_METAL_SP_JACOBI_H

#include "sp_common.h"

namespace sp {

#define NT NX_SP_THREADS
/* The elements of a round a thread holds: NT / per threads cover a
   matrix's at most 1024 elements, or per's at most NT. */
#define ROW 4

/* The pair [i] of round [r] among [np] indices, np even: index np − 1
   meets r, and the others meet by reflection about r. */
static inline void pair_of(int r, int i, int np, thread int &p,
                           thread int &q) {
  int a, b;
  if (i == 0) {
    a = np - 1;
    b = r;
  } else {
    a = (r + i) % (np - 1);
    b = (r - i + np - 1) % (np - 1);
  }
  p = a < b ? a : b;
  q = a < b ? b : a;
}

/* The rotation of the Hermitian 2 x 2 [x y; ȳ z], y ≠ 0: column p of J
   is (c, −s ē), column q (s e, c), e = y / |y|, and the new diagonal is
   x − t|y|, z + t|y|. */
template <class T>
static inline void rotation(float x, float z, T y, float g, thread float &c,
                            threadgroup T *bp, threadgroup T *bq,
                            thread float &t) {
  const T e = cdiv(y, elt<T>(g));
  const float tau = (z - x) / (2 * g);
  t = (tau >= 0 ? 1.0f : -1.0f) / (fabs_(tau) + hypot_(1.0f, tau));
  c = 1 / hypot_(1.0f, t);
  const float s = t * c;
  *bp = elt<T>(-s) * conj(e);
  *bq = elt<T>(s) * e;
}

/* The larger magnitude of [x]'s parts. */
template <class T> static inline float part(T x) {
  return fmax_(fabs_(re(x)), fabs_(im(x)));
}

/* The group's slots, a word each of its [per] matrices: the bits of its
   largest magnitude, whether it is finite, rotated this sweep, failed,
   done; from byte 16 of the group's memory, the meeting's word at byte
   0. */
struct slots {
  threadgroup metal::atomic_uint *big;
  threadgroup metal::atomic_int *finite;
  threadgroup int *rotated, *failed, *done;
};

/* The bytes of the meeting's word and the slots of [per] matrices,
   rounded up to 16: where the matrices start. */
static inline int jacobi_head(int per) {
  return 16 + (20 * per + 15) / 16 * 16;
}

static inline slots slots_of(threadgroup uchar *lds, int per) {
  slots sl;
  sl.big = (threadgroup metal::atomic_uint *)(lds + 16);
  sl.finite = (threadgroup metal::atomic_int *)(sl.big + per);
  sl.rotated = (threadgroup int *)(sl.finite + per);
  sl.failed = sl.rotated + per;
  sl.done = sl.failed + per;
  return sl;
}

static inline void slots_reset(slots sl, int t, int per) {
  if (t >= per) return;
  metal::atomic_store_explicit(&sl.finite[t], 1, metal::memory_order_relaxed);
  metal::atomic_store_explicit(&sl.big[t], 0u, metal::memory_order_relaxed);
  sl.rotated[t] = 0;
  sl.failed[t] = 0;
  sl.done[t] = 0;
}

/* The largest magnitude of matrix [slot]'s scan, 0 for a set past the
   group's matrices. */
static inline float slot_big(slots sl, int slot, int per) {
  if (slot >= per) return 0;
  return as_type<float>(
      metal::atomic_load_explicit(&sl.big[slot], metal::memory_order_relaxed));
}

static inline bool slot_finite(slots sl, int slot) {
  return metal::atomic_load_explicit(&sl.finite[slot],
                                     metal::memory_order_relaxed) != 0;
}

/* Element [x]'s part of a matrix's scan: its slot made not finite where
   it is NaN or infinite, its largest part's magnitude into the maximum,
   whose bits order as the magnitudes do. [x] is flushed, so its bits are
   those of a normal number or a zero. */
template <class T> static inline void scan_one(slots sl, int slot, T x) {
  const float nan = (re(x) - re(x)) + (im(x) - im(x));
  if (nan != 0)
    metal::atomic_fetch_and_explicit(&sl.finite[slot], 0,
                                     metal::memory_order_relaxed);
  const float m = fmax_(fabs_(re(x)), fabs_(im(x)));
  if (m == m)
    metal::atomic_fetch_max_explicit(&sl.big[slot], as_type<uint>(m),
                                     metal::memory_order_relaxed);
}

template <class T> static inline T nan_of() { return elt<T>(nan_(), nan_()); }

/* Eigh */

/* Threadgroup memory per matrix, after the head: a, v, the round's α and
   β per index (T); then for every matrix partner and rank per index
   (int), and new diagonal (float). */

/* w and, where [v] is not null, v of the [nb] Hermitian n x n matrices
   at [a], 1 <= n <= 32. NT threads a group. */
template <class T>
static inline void eigh_jacobi(grp g, long gx, long ngx,
                               threadgroup uchar *lds, device const T *a,
                               device float *w, device T *v, long nb, int n,
                               int per, int max_sweeps, long first,
                               long count) {
  const slots sl = slots_of(lds, per);
  threadgroup uchar *smem = lds + jacobi_head(per);
  const int nn = n * n, gs = NT / per;
  const int slot = g.tid / gs, lt = g.tid % gs;
  const int np = n + (n & 1);
  threadgroup T *A = (threadgroup T *)smem + ulong(slot) * (2 * nn + 2 * n),
                *V = A + nn, *al = V + nn, *be = al + n;
  threadgroup int *ints =
      (threadgroup int *)((threadgroup T *)smem +
                          ulong(per) * (2 * nn + 2 * n));
  threadgroup int *pa = ints + slot * n;
  threadgroup float *dn =
      (threadgroup float *)(ints + per * n + (per * n & 1)) + slot * n;
  const float eps = lim::eps(), tiny = lim::tiny();
  const float rmin = sqrt_(tiny / (2 * eps)), rmax = 1 / rmin;
  for (long u = first + gx; u < count; u += ngx) {
    const long b = u * per + slot;
    const bool live = slot < per && b < nb;
    barrier();
    slots_reset(sl, g.tid, per);
    barrier();
    /* The lower triangle and the diagonal's real parts, which alone are
       read, flushed, and the upper triangle their conjugates. */
    if (live)
      for (int k = lt; k < nn; k += gs) {
        const int i = k / n, j = k % n;
        if (i < j) continue;
        T x = flush(T(a[ulong(b) * nn + k]));
        if (i == j) x = elt<T>(re(x));
        scan_one(sl, slot, x);
        A[k] = x;
        A[j * n + i] = conj(x);
        if (v) {
          V[k] = elt<T>(float(i == j));
          if (i != j) V[j * n + i] = elt<T>(0);
        }
      }
    barrier();
    const int e = scale_exp(slot_big(sl, slot, per), rmin, rmax);
    const bool run = live && slot_finite(sl, slot);
    if (run && e != 0)
      for (int k = lt; k < nn; k += gs) A[k] = scaled(T(A[k]), e);
    bool go = true;
    for (int sweep = 0; go; sweep++) {
      for (int r = 0; r < np - 1; r++) {
        barrier();
        if (run && lt < np / 2) {
          int p, q;
          pair_of(r, lt, np, p, q);
          if (q < n) {
            const float x = re(T(A[p * n + p])), z = re(T(A[q * n + q]));
            const T y = A[p * n + q];
            const float gy = mag(y);
            pa[p] = pa[q] = -1;
            if (gy > tiny && gy > eps * sqrt_(fabs_(x)) * sqrt_(fabs_(z))) {
              float c, t;
              rotation(x, z, y, gy, c, &be[p], &be[q], t);
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
        barrier();
        /* a ← Jᴴ a J and v ← v J, each element from the old ones; a
           rotated pair's coupling is 0 and its diagonal the rotation's. */
        T na[ROW], nv[ROW];
        if (run)
          #pragma unroll
          for (int c = 0; c < ROW; c++) {
            const int k = lt + c * gs;
            if (k >= nn) break;
            const int i = k / n, j = k % n, pi = pa[i], pj = pa[j];
            T hi = A[k], hp = hi;
            if (pi >= 0) hp = A[pi * n + j];
            if (pj >= 0) {
              hi = T(al[j]) * hi + T(be[j]) * T(A[i * n + pj]);
              if (pi >= 0) hp = T(al[j]) * hp + T(be[j]) * T(A[pi * n + pj]);
            }
            T x = pi >= 0 ? conj(T(al[i])) * hi + conj(T(be[i])) * hp : hi;
            if (pi >= 0 && j == pi) x = elt<T>(0);
            if (pi >= 0 && j == i) x = elt<T>(dn[i]);
            na[c] = x;
            if (v)
              nv[c] = pj >= 0 ? T(al[j]) * T(V[k]) + T(be[j]) * T(V[i * n + pj])
                              : T(V[k]);
          }
        barrier();
        if (run)
          #pragma unroll
          for (int c = 0; c < ROW; c++) {
            const int k = lt + c * gs;
            if (k >= nn) break;
            A[k] = na[c];
            if (v) V[k] = nv[c];
          }
      }
      barrier();
      const bool mine = run && sl.rotated[slot];
      go = barrier_or(g, mine);
      if (go && sweep + 1 >= max_sweeps) {
        if (mine && lt == 0) sl.failed[slot] = 1;
        go = false;
      }
      if (g.tid < per) sl.rotated[g.tid] = 0;
    }
    barrier();
    const bool ok = run && !sl.failed[slot];
    /* Ascending, ties in index order, and v's columns with them, each
       normalised: a rotation of angle t below √u has c = 1 in floating
       point and lengthens its two columns by 1 + t²/2, always longer. */
    if (ok)
      for (int i = lt; i < n; i += gs) {
        const float x = re(T(A[i * n + i]));
        int rank = 0;
        for (int j = 0; j < n; j++) {
          const float y = re(T(A[j * n + j]));
          rank += y < x || (y == x && j < i);
        }
        dn[i] = x;
        pa[i] = rank;
        if (!v) continue;
        sumsq sum;
        for (int j = 0; j < n; j++) sum.sq(T(V[j * n + i]));
        const float norm = sqrt_(sum.get());
        for (int j = 0; j < n; j++) V[j * n + i] = T(V[j * n + i]) / norm;
      }
    barrier();
    if (live && !ok) {
      for (int k = lt; k < n; k += gs) w[ulong(b) * n + k] = nan_();
      for (int k = lt; v && k < nn; k += gs) v[ulong(b) * nn + k] = nan_of<T>();
    }
    if (ok) {
      for (int i = lt; i < n; i += gs)
        w[ulong(b) * n + pa[i]] = scaled(dn[i], -e) + 0.0f;
      for (int k = lt; v && k < nn; k += gs)
        v[ulong(b) * nn + k - k % n + pa[k % n]] = V[k];
    }
  }
}

/* Svd */

/* Threadgroup memory per matrix, after the head, [span] elements apart:
   g (M x K), v (K x K), u (M x M), α and β per column, a projection per
   row (T); then for every matrix rank and its inverse per column (int),
   and norms per column and weights per row (float). */

/* s and, where [u] is not null, u and vh of the [nb] m x n matrices at
   [a], 1 <= m, n <= 32, [uc] the columns of u and [vr] the rows of vh
   (reduced or complete). g is a, or aᴴ where a is wide, M x K. NT
   threads a group. */
template <class T>
static inline void svd_jacobi(grp g, long gx, long ngx,
                              threadgroup uchar *lds, device const T *a,
                              device float *s, device T *u, device T *vh,
                              long nb, int m, int n, int uc, int vr, int per,
                              ulong span, int max_sweeps, long first,
                              long count) {
  const slots sl = slots_of(lds, per);
  threadgroup uchar *smem = lds + jacobi_head(per);
  const bool wide = m < n, vec = u != nullptr;
  const int M = wide ? n : m, K = wide ? m : n, mk = M * K;
  /* The columns of g's left factor the call stores: u's, or vh's rows. */
  const int gc = wide ? vr : uc;
  const int gs = NT / per;
  const int slot = g.tid / gs, lt = g.tid % gs;
  const int np = K + (K & 1);
  threadgroup T *G = (threadgroup T *)smem + slot * span, *V = G + mk,
                *U = V + K * K, *al = U + M * M, *be = al + K, *pr = be + K;
  threadgroup int *ints =
      (threadgroup int *)((threadgroup T *)smem + per * span);
  threadgroup int *pa = ints + slot * 2 * K, *iv = pa + K;
  threadgroup float *sg =
      (threadgroup float *)(ints + per * 2 * K) + slot * (K + M);
  threadgroup float *sc = sg + K;
  const float eps = lim::eps(), tiny = lim::tiny();
  const float small = sqrt_(tiny) / (2 * eps);
  /* xGESVJ's tolerance, √M u, and its test of convergence: a sweep
     whose cosines all lie below √K tol and for which K max cs max |t|
     lies below tol has converged. Rounding can hold a pair's cosine just
     above tol, where a rotation of a few ulps gives it back sweep after
     sweep. A sweep whose rotations all have cosines at most [near] and
     tangents at most [slight] meets the test, and asks for no other. */
  const float tol = sqrt_(float(M)) * eps, near = sqrt_(float(K)) * tol;
  const float slight = 1 / (K * sqrt_(float(K)));
  for (long unit = first + gx; unit < count; unit += ngx) {
    const long b = unit * per + slot;
    const bool live = slot < per && b < nb;
    barrier();
    slots_reset(sl, g.tid, per);
    barrier();
    if (live) {
      for (int k = lt; k < mk; k += gs) {
        const int i = k / n, j = k % n;
        const T x = flush(T(a[ulong(b) * mk + k]));
        scan_one(sl, slot, x);
        if (wide)
          G[j * K + i] = conj(x);
        else
          G[k] = x;
      }
      for (int k = lt; vec && k < K * K; k += gs)
        V[k] = elt<T>(float(k / K == k % K));
    }
    barrier();
    const float big = slot_big(sl, slot, per);
    const int e = scale_exp(big, small, 1 / small);
    /* A column of norm at most [keep], u times the scaled matrix's
       largest part, is completed, not rotated. On a matrix of lower rank,
       rotations leave rounding of about u ‖a‖ in the columns that span
       nothing, and each sweep shrinks it by only some orders of
       magnitude: carried down to the least normal number, it takes more
       sweeps than the limit. Completing K such columns moves a by at
       most 2 √K u ‖a‖_F, within the bound. [keep] is at least u [small],
       so a column above it has normal digits to divide by. */
    const float keep = eps * scaled(big, e);
    const bool run = live && slot_finite(sl, slot);
    if (run && e != 0)
      for (int k = lt; k < mk; k += gs) G[k] = scaled(T(G[k]), e);
    bool go = true;
    for (int sweep = 0; go; sweep++) {
      /* Whether the matrix still sweeps: not once a sweep of its own has
         rotated only within rounding's reach. */
      const bool act = run && !sl.done[slot];
      for (int r = 0; r < np - 1; r++) {
        barrier();
        if (act && lt < np / 2) {
          int p, q;
          pair_of(r, lt, np, p, q);
          if (q < K) {
            /* The pair's cosine from the columns each scaled by its
               largest part, so that no square under- or overflows; the
               rotation from it and the norms' ratio (xGESVJ). */
            float sp = 0, sq = 0, x = 0, z = 0;
            T y = elt<T>(0);
            for (int i = 0; i < M; i++) {
              sp = fmax_(sp, part(T(G[i * K + p])));
              sq = fmax_(sq, part(T(G[i * K + q])));
            }
            pa[p] = pa[q] = -1;
            if (sp > 0 && sq > 0) {
              for (int i = 0; i < M; i++) {
                const T gp = T(G[i * K + p]) / sp, gq = T(G[i * K + q]) / sq;
                x += abs2(gp);
                z += abs2(gq);
                y += conj(gp) * gq;
              }
              const float nrp = sp * sqrt_(x), nrq = sq * sqrt_(z);
              const float cs = mag(y) / (sqrt_(x) * sqrt_(z));
              /* A column at most [keep] is completed, not rotated: its
                 singular value is its norm. */
              if (cs > tol && fmin_(nrp, nrq) > keep) {
                const float tau = (nrq / nrp - nrp / nrq) / (2 * cs);
                const float t = (tau >= 0 ? 1.0f : -1.0f) /
                                (fabs_(tau) + hypot_(1.0f, tau));
                if (t != 0) {
                  const float c = 1 / hypot_(1.0f, t), sn = t * c;
                  const T ph = cdiv(y, elt<T>(mag(y)));
                  be[p] = elt<T>(-sn) * conj(ph);
                  be[q] = elt<T>(sn) * ph;
                  al[p] = al[q] = elt<T>(c);
                  pa[p] = q;
                  pa[q] = p;
                  if (cs > near || fabs_(t) > slight) sl.rotated[slot] = 1;
                }
              }
            }
          } else if (p < K)
            pa[p] = -1;
        }
        barrier();
        /* g ← g J and v ← v J. */
        T ng[ROW], nv[ROW];
        if (act)
          #pragma unroll
          for (int c = 0; c < ROW; c++) {
            const int k = lt + c * gs;
            if (k >= mk) break;
            const int i = k / K, j = k % K, pj = pa[j];
            ng[c] = pj >= 0 ? T(al[j]) * T(G[k]) + T(be[j]) * T(G[i * K + pj])
                            : T(G[k]);
            if (vec && k < K * K)
              nv[c] = pj >= 0
                          ? T(al[j]) * T(V[k]) + T(be[j]) * T(V[k - j + pj])
                          : T(V[k]);
          }
        barrier();
        if (act)
          #pragma unroll
          for (int c = 0; c < ROW; c++) {
            const int k = lt + c * gs;
            if (k >= mk) break;
            G[k] = ng[c];
            if (vec && k < K * K) V[k] = nv[c];
          }
      }
      barrier();
      const bool mine = act && sl.rotated[slot];
      if (act && !mine && lt == 0) sl.done[slot] = 1;
      go = barrier_or(g, mine);
      if (go && sweep + 1 >= max_sweeps) {
        if (mine && lt == 0) sl.failed[slot] = 1;
        go = false;
      }
      if (g.tid < per) sl.rotated[g.tid] = 0;
    }
    barrier();
    const bool ok = run && !sl.failed[slot];
    /* The singular values, each column's norm, and v's columns
       normalised, as eigh's are; both norms by compensated sums. */
    if (ok)
      for (int j = lt; j < K; j += gs) {
        float sj = 0;
        sumsq x;
        for (int i = 0; i < M; i++) sj = fmax_(sj, part(T(G[i * K + j])));
        for (int i = 0; i < M && sj > 0; i++) x.sq(T(G[i * K + j]) / sj);
        sg[j] = sj * sqrt_(x.get());
        if (!vec) continue;
        sumsq y;
        for (int i = 0; i < K; i++) y.sq(T(V[i * K + j]));
        const float norm = sqrt_(y.get());
        for (int i = 0; i < K; i++) V[i * K + j] = T(V[i * K + j]) / norm;
      }
    barrier();
    /* Descending, ties in index order. */
    if (ok)
      for (int j = lt; j < K; j += gs) {
        int rank = 0;
        for (int i = 0; i < K; i++)
          rank += sg[i] > sg[j] || (sg[i] == sg[j] && i < j);
        pa[j] = rank;
        iv[rank] = j;
      }
    barrier();
    /* u's columns, those whose norm exceeds [keep]. */
    int kept = 0;
    for (int j = 0; ok && j < K; j++) kept += sg[j] > keep;
    if (ok && vec)
      for (int k = lt; k < mk; k += gs) {
        const int i = k / K, j = k % K;
        if (sg[j] > keep) U[i * M + pa[j]] = T(G[k]) / sg[j];
      }
    /* Columns [kept, gc) of g's left factor: e_i less its projection on
       the columns before, twice, normalised, i the row those columns
       weigh least, the first of equals (Gram-Schmidt completion). Every
       set runs gc steps, so that the group's barriers match. */
    for (int c = 0; vec && c < gc; c++) {
      const bool mine = ok && c >= kept && lt < M;
      barrier();
      if (mine) {
        float x = 0;
        for (int j = 0; j < c; j++) x += abs2(T(U[lt * M + j]));
        sc[lt] = x;
      }
      barrier();
      T x = elt<T>(0);
      if (mine) {
        int best = 0;
        for (int i = 1; i < M; i++)
          if (sc[i] < sc[best]) best = i;
        x = elt<T>(float(lt == best));
        for (int j = 0; j < c; j++)
          x -= T(U[lt * M + j]) * conj(T(U[best * M + j]));
      }
      barrier();
      if (mine) U[lt * M + c] = x;
      barrier();
      if (mine && lt < c) {
        T d = elt<T>(0);
        for (int i = 0; i < M; i++)
          d += conj(T(U[i * M + lt])) * T(U[i * M + c]);
        pr[lt] = d;
      }
      barrier();
      if (mine) {
        x = U[lt * M + c];
        for (int j = 0; j < c; j++) x -= T(U[lt * M + j]) * T(pr[j]);
      }
      barrier();
      if (mine) U[lt * M + c] = x;
      barrier();
      float nrm = 0;
      for (int i = 0; mine && i < M; i++) nrm += abs2(T(U[i * M + c]));
      barrier();
      if (mine) U[lt * M + c] = x / sqrt_(nrm);
    }
    barrier();
    const long K64 = K;
    if (live && !ok) {
      for (int k = lt; k < K; k += gs) s[b * K64 + k] = nan_();
      for (int k = lt; vec && k < m * uc; k += gs)
        u[ulong(b) * m * uc + k] = nan_of<T>();
      for (int k = lt; vec && k < vr * n; k += gs)
        vh[ulong(b) * vr * n + k] = nan_of<T>();
    }
    if (!ok) continue;
    for (int j = lt; j < K; j += gs)
      s[b * K64 + pa[j]] = scaled(sg[j], -e) + 0.0f;
    if (!vec) continue;
    /* Tall: u = U and vh = Vᴴ. Wide: u = V and vh = Uᴴ. */
    for (int k = lt; k < m * uc; k += gs) {
      const int i = k / uc, j = k % uc;
      u[ulong(b) * m * uc + k] = wide ? T(V[i * K + iv[j]]) : T(U[i * M + j]);
    }
    for (int k = lt; k < vr * n; k += gs) {
      const int i = k / n, j = k % n;
      vh[ulong(b) * vr * n + k] =
          wide ? conj(T(U[j * M + i])) : conj(T(V[j * K + iv[i]]));
    }
  }
}

#undef NT
#undef ROW

} // namespace sp

#endif
