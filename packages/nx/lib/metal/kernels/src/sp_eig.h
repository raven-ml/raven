/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* eig: a v = v diag w for general square matrices, as LAPACK's xGEEV
   computes it, one group a matrix.

   a is balanced (xGEBAL), scaled into a safe range by a power of 2,
   reduced to the upper Hessenberg H = Qᴴ a Q (xGEHD2), Z = Q formed
   (xORGHR), then H = Z S Zᴴ in Schur form by the implicit QR iteration
   (xLAHQR): single shifts on complex numbers, Francis double shifts on
   reals, where each complex conjugate pair of eigenvalues stays a 2 x 2
   block in standard form (xLANV2), so that the pair is exactly
   conjugate. The eigenvectors X of S come by back substitution
   (xTREVC3), one thread an eigenvalue (eig_vectors), v = Z X by the
   contraction's product, then eig_finish undoes the balancing and gives
   each column unit norm, its element of largest magnitude real.

   The steps run as launches of bounded work, each keeping its state in
   the workspace:

     eig_in      a into H, flushed, balanced and scaled; Z = I, in the
     .. eig_scale  seven steps of balancing's comment below
     eig_hess    the Hessenberg reflectors of columns [from, to)
     eig_orghr   Z's reflectors of columns [from, to), the last first
     eig_qr      at most q QR sweeps, then the state saved
     eig_w       w, and NaN for a matrix whose iteration failed

   No value depends on where a launch ends: a launch reads every value
   it needs from the workspace, and the sweeps' state is the loop's own.

   Every thread of a group runs the same control flow and computes the
   same scalars from the same memory; loops over rows or columns split
   among the threads, and the group's reductions combine in one fixed
   tree. So the bits depend on the matrix and the group size alone,
   NX_SP_HALF.

   The QR iteration's limit is a total of 30 max(10, h) sweeps per
   matrix, not per eigenvalue, h = hi − lo + 1 (30 is [itq], smaller in
   tests): a matrix that does not converge stops after a bounded number
   of launches.

   Workspace a matrix, [per] bytes apart, column-major (element (i, j)
   at x[i + j n]): H, Z and X of n² elements, a complex scratch of n² for
   the vectors' back substitution, and for each index τ, scale, wr, wi,
   column norms; balancing's counts, n words; four words (ok, the
   scale's exponent, lo, hi) and the state of balancing and the
   iteration, NX_SP_EIG_STATE words.

   This follows nx.amd's sp_eig.h. */

#ifndef NX_METAL_SP_EIG_H
#define NX_METAL_SP_EIG_H

#include "sp_block.h"

namespace sp {

#define KEXSH 10

/* Four values in registers, read by index. */
struct four {
  cf v0, v1, v2, v3;
  cf operator[](int i) const thread {
    cf x = v3;
    if (i == 0) x = v0;
    if (i == 1) x = v1;
    if (i == 2) x = v2;
    return x;
  }
};

template <class T> struct eig_ws {
  device T *h, *z, *x;
  device cf *xs;
  device T *tau;
  device float *scale, *wr, *wi, *cn;
  device int *cnt, *ok, *st;
};

/* Matrix [b]'s workspace, [tb] an element's bytes: a step of parts alone
   takes its format's workspace as T = float. */
template <class T>
static inline eig_ws<T> eig_of(device uchar *base, long b, int n, ulong per,
                               int tb = sizeof(T)) {
  eig_ws<T> ws;
  const ulong nn = ulong(n) * n;
  device uchar *p = base + ulong(b) * per;
  ws.h = (device T *)p;
  ws.z = (device T *)(p += nn * tb);
  ws.x = (device T *)(p += nn * tb);
  ws.xs = (device cf *)(p += nn * tb);
  ws.tau = (device T *)(ws.xs + nn);
  ws.scale = (device float *)((device uchar *)ws.tau + ulong(n) * tb);
  ws.wr = ws.scale + n;
  ws.wi = ws.wr + n;
  ws.cn = ws.wi + n;
  ws.cnt = (device int *)(ws.cn + n);
  ws.ok = ws.cnt + n;
  ws.st = ws.ok + 4;
  return ws;
}

/* Balancing (xGEBAL 'B'): rows and columns that isolate an eigenvalue
   swapped to the ends, then rows and columns [lo, hi] scaled by powers
   of 2 until each row's and column's norms are within 5% of balanced, or
   for at most [passes] passes: a matrix that needs more keeps the scales
   they reach, a similarity all the same. scale takes, outside [lo, hi],
   the index each was swapped with, inside, its scale.

   It runs as launches of bounded work, its state in the workspace:

     eig_in      a into H, flushed, column-major; each row's NaN test
     eig_count   each row's nonzeros off the diagonal
     eig_iso     at most q isolations of a row, then of a column
     eig_colcnt  each column's nonzeros off the diagonal in rows [k, l]
     eig_rows    at most q rows of the scaling's passes
     eig_big     each column's largest part
     eig_scale   H scaled into the safe range, τ zeroed, Z = I

   Isolation keeps each row's (then column's) count of nonzeros off the
   diagonal within the active block, and updates it as a swap exchanges
   two counts and a dropped column (row) removes its elements: the row
   it finds is the one xGEBAL's scan finds, in O(n) a step. */
template <class T>
static inline void swap_cols(grp g, device T *a, int n, int i, int j,
                             int rows) {
  for (int p = g.tid; p < rows; p += g.nthr) {
    const T x = M(a, p, i);
    M(a, p, i) = M(a, p, j);
    M(a, p, j) = x;
  }
}
template <class T>
static inline void swap_rows(grp g, device T *a, int n, int i, int j, int c0) {
  for (int p = c0 + g.tid; p < n; p += g.nthr) {
    const T x = M(a, i, p);
    M(a, i, p) = M(a, j, p);
    M(a, j, p) = x;
  }
}

/* Whether x counts as a nonzero: xGEBAL's test on flushed H, so −0 is
   zero, NaN is not, and a complex value is zero only where both parts
   are. */
template <class T> static inline int nonzero(T x) {
  return !(x == elt<T>(0));
}

/* In: row r of a, row-major, flushed into H, column-major; the row's
   NaN test, a sum of x − x terms, into wr[r]. Row 0's group starts
   balancing. A group a row, unit b n + r. */
template <class T>
static inline void eig_in(grp g, long gx, long ngx, threadgroup float *sh,
                          device const T *a, device uchar *wsp, int n,
                          ulong per, long first, long count) {
  const int t = g.tid, nt = g.nthr;
  const bool small = is_small(count);
  for (long u = first + gx; u < count; u += ngx) {
    long b = u;
    const int r = int(split(b, n, small));
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    device const T *ar = a + (ulong(b) * n + ulong(r)) * ulong(n);
    float nan = 0;
    for (int c = t; c < n; c += nt) {
      const T x = flush(T(ar[c]));
      nan += (re(x) - re(x)) + (im(x) - im(x));
      M(ws.h, r, c) = x;
    }
    nan = bsum(g, nan, sh);
    if (t == 0) {
      ws.wr[r] = nan;
      if (r == 0) {
        ws.st[NX_SP_EIG_BAL] = NX_SP_BAL_START;
        ws.st[NX_SP_EIG_BK] = 0;
        ws.st[NX_SP_EIG_BL] = n - 1;
      }
    }
    barrier();
  }
}

/* Count: row r's nonzeros off the diagonal. A thread a row, unit
   b n + r. */
template <class T>
static inline void eig_count(grp g, long gx, long ngx, device uchar *wsp,
                             int n, ulong per, long first, long count) {
  const bool small = is_small(count);
  for (long u = first + gx * g.nthr + g.tid; u < count; u += ngx * g.nthr) {
    long b = u;
    const int r = int(split(b, n, small));
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    int c = 0;
    for (int j = 0; j < n; j++) c += j != r && nonzero(T(M(ws.h, r, j)));
    ws.cnt[r] = c;
  }
}

/* Column counts: column j's nonzeros off the diagonal in rows [k, l],
   for a matrix whose rows are isolated. A thread a column, unit
   b n + j. */
template <class T>
static inline void eig_colcnt(grp g, long gx, long ngx, device uchar *wsp,
                              int n, ulong per, long first, long count) {
  const bool small = is_small(count);
  for (long u = first + gx * g.nthr + g.tid; u < count; u += ngx * g.nthr) {
    long b = u;
    const int j = int(split(b, n, small));
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    const int k = ws.st[NX_SP_EIG_BK], l = ws.st[NX_SP_EIG_BL];
    if (ws.st[NX_SP_EIG_BAL] != NX_SP_BAL_WAIT || j < k || j > l) continue;
    int c = 0;
    for (int i = k; i <= l; i++) c += i != j && nonzero(T(M(ws.h, i, j)));
    ws.cnt[j] = c;
  }
}

/* The end of isolation: [lo, hi] = [k, l] scaled from 1, the scaling's
   first pass next. */
static inline void iso_end(grp g, device float *scale, device int *ok,
                           thread int &phase, thread int &i, int k, int l) {
  for (int p = k + g.tid; p <= l; p += g.nthr) scale[p] = 1;
  if (g.tid == 0) ok[2] = k, ok[3] = l;
  phase = NX_SP_BAL_SCALE;
  i = k;
}

/* Isolation: at most [q] steps of each matrix in its rows' phase, or,
   where [cols], in its columns'. A step finds the last row of [0, l]
   alone in columns [0, l] (the first column of [k, l] alone in rows
   [k, l]) and swaps it to l (to k), or ends the phase. The first step
   tests the NaN sums: a matrix that holds one is done. One group a
   matrix; [sh] a float a thread. */
template <class T>
static inline void eig_iso(grp g, long gx, long ngx, threadgroup float *sh,
                           device uchar *wsp, int n, ulong per, int cols,
                           int q, long first, long count) {
  const int t = g.tid, nt = g.nthr;
  for (long b = first + gx; b < count; b += ngx) {
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    device T *h = ws.h;
    device int *cnt = ws.cnt;
    int phase = ws.st[NX_SP_EIG_BAL], k = ws.st[NX_SP_EIG_BK],
        l = ws.st[NX_SP_EIG_BL], i = ws.st[NX_SP_EIG_BI];
    if (cols && phase == NX_SP_BAL_WAIT) phase = NX_SP_BAL_COLS;
    const int mine = cols ? NX_SP_BAL_COLS : NX_SP_BAL_ROWS;
    if (phase != mine && phase != NX_SP_BAL_START) continue;
    for (int s = 0; s < q && (phase == mine || phase == NX_SP_BAL_START);
         s++) {
      if (phase == NX_SP_BAL_START) {
        float nan = 0;
        for (int r = t; r < n; r += nt) nan += ws.wr[r];
        const bool ok = bsum(g, nan, sh) == 0;
        if (t == 0) ws.ok[0] = ok;
        if (!ok) {
          if (t == 0) ws.ok[2] = 0, ws.ok[3] = n - 1;
          phase = NX_SP_BAL_DONE;
          break;
        }
        phase = NX_SP_BAL_ROWS;
        if (n == 1) {
          iso_end(g, ws.scale, ws.ok, phase, i, k, l);
          break;
        }
      }
      if (phase == NX_SP_BAL_ROWS) {
        int found = -1;
        for (int r = t; r <= l; r += nt)
          if (cnt[r] == 0) found = metal::max(found, r);
        found = int(bmax(g, float(found), sh));
        if (found < 0) {
          phase = NX_SP_BAL_WAIT;
          break;
        }
        if (t == 0) ws.scale[l] = float(found);
        if (found != l) {
          barrier();
          swap_cols(g, h, n, found, l, l + 1);
          barrier();
          swap_rows(g, h, n, found, l, k);
          barrier();
          if (t == 0) {
            const int c = cnt[found];
            cnt[found] = cnt[l];
            cnt[l] = c;
          }
        }
        barrier();
        if (l == 0) {
          if (t == 0) ws.ok[2] = 0, ws.ok[3] = 0;
          phase = NX_SP_BAL_DONE;
          break;
        }
        l--;
        for (int r = t; r <= l; r += nt) cnt[r] -= nonzero(T(M(h, r, l + 1)));
        barrier();
        continue;
      }
      int found = n;
      for (int j = k + t; j <= l; j += nt)
        if (cnt[j] == 0) found = metal::min(found, j);
      found = -int(bmax(g, float(-found), sh));
      if (found > l) {
        iso_end(g, ws.scale, ws.ok, phase, i, k, l);
        break;
      }
      if (t == 0) ws.scale[k] = float(found);
      if (found != k) {
        barrier();
        swap_cols(g, h, n, found, k, l + 1);
        barrier();
        swap_rows(g, h, n, found, k, k);
        barrier();
        if (t == 0) {
          const int c = cnt[found];
          cnt[found] = cnt[k];
          cnt[k] = c;
        }
      }
      barrier();
      k++;
      for (int j = k + t; j <= l; j += nt) cnt[j] -= nonzero(T(M(h, k - 1, j)));
      barrier();
    }
    barrier();
    if (t == 0) {
      ws.st[NX_SP_EIG_BAL] = phase;
      ws.st[NX_SP_EIG_BK] = k;
      ws.st[NX_SP_EIG_BL] = l;
      ws.st[NX_SP_EIG_BI] = i;
      ws.st[NX_SP_EIG_BPASS] = 0;
      ws.st[NX_SP_EIG_BNOCONV] = 0;
    }
    barrier();
  }
}

/* Scaling: at most [q] steps of each matrix in its scaling, a step row i
   of [k, l] as xGEBAL scales it, or, past l, the pass's end: another
   pass where this one rescaled a row and fewer than [passes] ran, else
   done. One group a matrix; [sh] and [si] a float and an int a
   thread. */
template <class T>
static inline void eig_rows(grp g, long gx, long ngx, threadgroup float *sh,
                            threadgroup int *si, device uchar *wsp, int n,
                            ulong per, int passes, int q, long first,
                            long count) {
  const int t = g.tid, nt = g.nthr;
  const float ulp = 2 * lim::eps();
  const float sfmin1 = lim::tiny() / ulp, sfmax1 = 1 / sfmin1,
              sfmin2 = sfmin1 * 2, sfmax2 = 1 / sfmin2;
  for (long b = first + gx; b < count; b += ngx) {
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    if (ws.st[NX_SP_EIG_BAL] != NX_SP_BAL_SCALE) continue;
    device T *a = ws.h;
    const int k = ws.st[NX_SP_EIG_BK], l = ws.st[NX_SP_EIG_BL];
    int i = ws.st[NX_SP_EIG_BI], pass = ws.st[NX_SP_EIG_BPASS],
        noconv = ws.st[NX_SP_EIG_BNOCONV], phase = NX_SP_BAL_SCALE;
    for (int s = 0; s < q && phase == NX_SP_BAL_SCALE; s++) {
      if (i > l) {
        if (noconv && pass + 1 < passes) {
          pass++;
          i = k;
          noconv = 0;
        } else
          phase = NX_SP_BAL_DONE;
        continue;
      }
      float c = bnorm<T>(g, &M(a, k, i), l - k + 1, 1, sh);
      float r = bnorm<T>(g, &M(a, i, k), l - k + 1, ulong(n), sh);
      float best = -1;
      int bi = 0;
      for (int p = t; p <= l; p += nt)
        if (mag1(T(M(a, p, i))) > best) best = mag1(T(M(a, p, i))), bi = p;
      bi = bargmax(g, best, bi, sh, si);
      const T x = M(a, bi, i);
      best = -1;
      bi = 0;
      for (int p = k + t; p < n; p += nt)
        if (mag1(T(M(a, i, p))) > best) best = mag1(T(M(a, i, p))), bi = p;
      bi = bargmax(g, best, bi, sh, si);
      const T y = M(a, i, bi);
      const int row = i++;
      float ca = lapy3(re(x), im(x), 0.0f), ra = lapy3(re(y), im(y), 0.0f);
      if (c == 0 || r == 0) continue;
      float gg = r / 2, f = 1;
      const float sum = c + r;
      while (c < gg && fmax_(f, c) < sfmax2 && fmax_(ca, c) < sfmax2 &&
             fmin_(r, gg) > sfmin2 && ra > sfmin2)
        f *= 2, c *= 2, ca *= 2, r /= 2, gg /= 2, ra /= 2;
      gg = c / 2;
      while (gg >= r && fmax_(r, ra) < sfmax2 && fmin_(f, c) > sfmin2 &&
             fmin_(gg, ca) > sfmin2)
        f /= 2, c /= 2, gg /= 2, ca /= 2, r *= 2, ra *= 2;
      if (c + r >= 0.95f * sum) continue;
      const float si_ = ws.scale[row];
      if (f < 1 && si_ < 1 && f * si_ <= sfmin1) continue;
      if (f > 1 && si_ > 1 && si_ >= sfmax1 / f) continue;
      noconv = 1;
      barrier();
      if (t == 0) ws.scale[row] = si_ * f;
      for (int j = k + t; j < n; j += nt)
        M(a, row, j) = T(M(a, row, j)) * elt<T>(1 / f);
      barrier();
      for (int j = t; j <= l; j += nt)
        M(a, j, row) = T(M(a, j, row)) * elt<T>(f);
      barrier();
    }
    barrier();
    if (t == 0) {
      ws.st[NX_SP_EIG_BAL] = phase;
      ws.st[NX_SP_EIG_BI] = i;
      ws.st[NX_SP_EIG_BPASS] = pass;
      ws.st[NX_SP_EIG_BNOCONV] = noconv;
    }
    barrier();
  }
}

/* Big: column j's largest part of balanced H into wi[j]. A group a
   column, unit b n + j. */
template <class T>
static inline void eig_big(grp g, long gx, long ngx, threadgroup float *sh,
                           device uchar *wsp, int n, ulong per, long first,
                           long count) {
  const bool small = is_small(count);
  for (long u = first + gx; u < count; u += ngx) {
    long b = u;
    const int j = int(split(b, n, small));
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    if (!ws.ok[0]) continue;
    float big = 0;
    for (int i = g.tid; i < n; i += g.nthr) {
      const T x = M(ws.h, i, j);
      big = fmax_(big, fmax_(fabs_(re(x)), fabs_(im(x))));
    }
    big = bmax(g, big, sh);
    if (g.tid == 0) ws.wi[j] = big;
    barrier();
  }
}

/* Scale: H's column j scaled by the exponent that brings H's largest
   part into the safe range, τ_j zeroed and, for the vectors, Z's column
   j the identity's; column 0's group writes the exponent and the QR's
   start. A group a column, unit b n + j. Balanced first, then scaled:
   xGEEV scales first, which flushes entries balancing would bring near
   the others. */
template <class T>
static inline void eig_scale(grp g, long gx, long ngx, threadgroup float *sh,
                             device uchar *wsp, int vectors, int n, ulong per,
                             long first, long count) {
  const int t = g.tid, nt = g.nthr;
  const float ulp = 2 * lim::eps();
  const bool small = is_small(count);
  for (long u = first + gx; u < count; u += ngx) {
    long b = u;
    const int j = int(split(b, n, small));
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    const bool ok = ws.ok[0];
    int e = 0;
    if (ok) {
      float big = 0;
      for (int p = t; p < n; p += nt) big = fmax_(big, ws.wi[p]);
      big = bmax(g, big, sh);
      const float small_ = sqrt_(lim::tiny()) / ulp;
      e = scale_exp(big, small_, 1 / small_);
      for (int i = t; e != 0 && i < n; i += nt)
        M(ws.h, i, j) = scaled(T(M(ws.h, i, j)), e);
      if (t == 0) ws.tau[j] = elt<T>(0);
      for (int i = t; vectors && i < n; i += nt)
        M(ws.z, i, j) = elt<T>(float(i == j));
    }
    barrier();
    if (t == 0 && j == 0) {
      ws.ok[1] = e;
      ws.st[NX_SP_EIG_PHASE] = ok ? NX_SP_EIG_BEGIN : NX_SP_EIG_DONE;
      ws.st[NX_SP_EIG_I] = ws.ok[3];
      ws.st[NX_SP_EIG_L] = ws.ok[2];
      ws.st[NX_SP_EIG_ITS] = 0;
      ws.st[NX_SP_EIG_KDEFL] = 0;
    }
    barrier();
  }
}

/* Hessenberg (xGEHD2): reflector i in column i below the subdiagonal,
   for i in [from, to) within [lo, hi). One group a matrix. */
template <class T>
static inline void eig_hess(grp g, long gx, long ngx, threadgroup float *sh,
                            device uchar *wsp, int n, int from, int to,
                            ulong per, long first, long count) {
  const int t = g.tid, nt = g.nthr;
  for (long b = first + gx; b < count; b += ngx) {
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    if (!ws.ok[0]) continue;
    const int lo = ws.ok[2], hi = ws.ok[3];
    device T *h = ws.h, *vcol = ws.x;
    for (int i = metal::max(lo, from); i < metal::min(hi, to); i++) {
      device T *alpha = &M(h, i + 1, i);
      const T tau = reflector<T>(g, alpha, alpha + 1, hi - i - 1, 1, sh);
      if (t == 0) ws.tau[i] = tau;
      /* v = [1; column i below], staged so the updates read it. */
      barrier();
      for (int p = t; p < hi - i; p += nt)
        vcol[p] = p == 0 ? elt<T>(1) : T(M(h, i + 1 + p, i));
      barrier();
      larf_right(g, vcol, tau, hi + 1, hi - i, &M(h, 0, i + 1), n);
      larf_left(g, vcol, conj(tau), hi - i, n - i - 1, &M(h, i + 1, i + 1), n);
    }
    barrier();
  }
}

/* Z = Q (xORGHR): the reflectors applied to I, last first, before H's
   subdiagonal is cleared: reflector i for i in [from, to) within
   [lo, hi), from the highest down. One group a matrix. */
template <class T>
static inline void eig_orghr(grp g, long gx, long ngx, device uchar *wsp,
                             int n, int from, int to, ulong per, long first,
                             long count) {
  const int t = g.tid, nt = g.nthr;
  const ulong nn = ulong(n) * n;
  for (long b = first + gx; b < count; b += ngx) {
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    if (!ws.ok[0]) continue;
    const int lo = ws.ok[2], hi = ws.ok[3];
    device T *h = ws.h, *z = ws.z, *v = ws.x + nn - n;
    for (int i = metal::min(hi, to) - 1; i >= metal::max(lo, from); i--) {
      barrier();
      for (int p = t; p < hi - i; p += nt)
        v[p] = p == 0 ? elt<T>(1) : T(M(h, i + 1 + p, i));
      barrier();
      larf_left(g, v, T(ws.tau[i]), hi - i, hi - i, &M(z, i + 1, i + 1), n);
    }
    barrier();
  }
}

/* Whether H's subdiagonal element (k, k − 1) is negligible in the active
   block [lo, hi] (Ahues and Tisseur's test, as xLAHQR). */
template <class T>
static inline bool negligible(device const T *h, int n, int k, int lo, int hi,
                              float smlnum) {
  const float ulp = 2 * lim::eps();
  if (mag1(T(M(h, k, k - 1))) <= smlnum) return true;
  float tst = mag1(T(M(h, k - 1, k - 1))) + mag1(T(M(h, k, k)));
  if (tst == 0) {
    if (k - 2 >= lo) tst += fabs_(re(T(M(h, k - 1, k - 2))));
    if (k + 1 <= hi) tst += fabs_(re(T(M(h, k + 1, k))));
  }
  if (!(fabs_(re(T(M(h, k, k - 1)))) <= ulp * tst)) return false;
  float x = mag1(T(M(h, k, k - 1))), y = mag1(T(M(h, k - 1, k)));
  const float ab = fmax_(x, y), ba = fmin_(x, y);
  x = mag1(T(M(h, k, k)));
  y = mag1(T(M(h, k - 1, k - 1)) - T(M(h, k, k)));
  const float aa = fmax_(x, y), bb = fmin_(x, y), s = aa + ab;
  const float tt = ulp * (bb * (aa / s));
  return ba * (ab / s) <= fmax_(smlnum, tt);
}

/* sqrt of a complex number, principal branch. */
static inline cf csqrt(cf a) {
  const float m = hypot_(a.re, a.im);
  if (m == 0) return C(0, a.im);
  const float r = sqrt_((m + fabs_(a.re)) / 2);
  if (a.re >= 0) return C(r, a.im / (2 * r));
  return C(fabs_(a.im) / (2 * r), copysign_(r, a.im));
}

/* The iteration's state between launches (NX_SP_EIG_*). */
struct qr_state {
  int phase, i, l, its, kdefl;
};

static inline qr_state state_of(device const int *st) {
  qr_state s;
  s.phase = st[NX_SP_EIG_PHASE];
  s.i = st[NX_SP_EIG_I];
  s.l = st[NX_SP_EIG_L];
  s.its = st[NX_SP_EIG_ITS];
  s.kdefl = st[NX_SP_EIG_KDEFL];
  return s;
}

/* H with every element below the subdiagonal 0, the start of the
   iteration. */
template <class T>
static inline void eig_clear(grp g, device T *h, int n) {
  const ulong nn = ulong(n) * n;
  for (ulong k = g.tid; k < nn; k += g.nthr)
    if (int(k % n) > int(k / n) + 1) h[k] = elt<T>(0);
  barrier();
}

/* Complex: single-shift QR sweeps (xLAHQR), at most [q] of them from
   the state [s]. H made upper triangular in columns [i1, i2] under
   [wantt], Z's rows [0, n) rotated where [z]. The state becomes FAILED
   where the matrix's sweeps reach [itmax]: xLAHQR's limit, without its
   last step, whose result xLAHQR never tests. */
static inline void lahqr(grp g, thread qr_state &s, bool wantt, int n, int lo,
                         int hi, device cf *h, device float *wr,
                         device float *wi, device cf *z, int itq, int q) {
  const int t = g.tid, nt = g.nthr;
  const float ulp = 2 * lim::eps();
  if (s.phase == NX_SP_EIG_BEGIN) {
    eig_clear(g, h, n);
    if (lo == hi) {
      wr[lo] = re(cf(M(h, lo, lo)));
      wi[lo] = im(cf(M(h, lo, lo)));
      s.phase = NX_SP_EIG_DONE;
      return;
    }
    if (t == 0) {
      for (int j = lo; j <= hi - 3; j++)
        M(h, j + 2, j) = M(h, j + 3, j) = C(0, 0);
      if (lo <= hi - 2) M(h, hi, hi - 2) = C(0, 0);
    }
    const int jlo = wantt ? 0 : lo, jhi = wantt ? n - 1 : hi;
    /* The subdiagonal made real by a diagonal similarity. */
    for (int i = lo + 1; i <= hi; i++) {
      barrier();
      const cf x = M(h, i, i - 1);
      if (x.im == 0) continue;
      cf sc = x / mag1(x);
      sc = conj(sc) / lapy3(sc.re, sc.im, 0.0f);
      const float ax = lapy3(x.re, x.im, 0.0f);
      barrier();
      for (int j = i + t; j <= jhi; j += nt) M(h, i, j) = cf(M(h, i, j)) * sc;
      barrier();
      for (int j = jlo + t; j <= metal::min(jhi, i + 1); j += nt)
        M(h, j, i) = cf(M(h, j, i)) * conj(sc);
      for (int j = t; z && j < n; j += nt)
        M(z, j, i) = cf(M(z, j, i)) * conj(sc);
      barrier();
      if (t == 0) M(h, i, i - 1) = C(ax, 0);
    }
    s.phase = NX_SP_EIG_RUN;
  }
  barrier();
  const float smlnum = lim::tiny() * (float(hi - lo + 1) / ulp);
  const int itmax = itq * metal::max(10, hi - lo + 1);
  int i1 = 0, i2 = n - 1;
  int i = s.i, l = s.l, its = s.its, kdefl = s.kdefl;
  for (int sweeps = 0; i >= lo;) {
    barrier();
    int k = i;
    while (k > l && !negligible(h, n, k, lo, hi, smlnum)) k--;
    l = k;
    barrier();
    if (l > lo && t == 0) M(h, l, l - 1) = C(0, 0);
    if (l >= i) {
      barrier();
      if (t == 0) {
        wr[i] = cf(M(h, i, i)).re;
        wi[i] = cf(M(h, i, i)).im;
      }
      /* its runs on: the limit is the matrix's total. */
      kdefl = 0;
      i = l - 1;
      l = lo;
      continue;
    }
    if (its == itmax) {
      s.phase = NX_SP_EIG_FAILED;
      break;
    }
    /* The launch ends before a sweep, after the deflations its last
       sweep allows: the next launch resumes here, as one launch would
       go on. */
    if (sweeps == q) break;
    kdefl++;
    if (!wantt) i1 = l, i2 = i;
    cf sh;
    if (kdefl % (2 * KEXSH) == 0)
      sh = C(0.75f * fabs_(cf(M(h, i, i - 1)).re), 0) + cf(M(h, i, i));
    else if (kdefl % KEXSH == 0)
      sh = C(0.75f * fabs_(cf(M(h, l + 1, l)).re), 0) + cf(M(h, l, l));
    else {
      sh = M(h, i, i);
      const cf u = csqrt(cf(M(h, i - 1, i))) * csqrt(cf(M(h, i, i - 1)));
      float su = mag1(u);
      if (su != 0) {
        const cf x = (cf(M(h, i - 1, i - 1)) - sh) / 2.0f;
        const float sx = mag1(x);
        su = fmax_(su, sx);
        cf y = su * csqrt((x / su) * (x / su) + (u / su) * (u / su));
        if (sx > 0) {
          const cf xs = x / sx;
          if (xs.re * y.re + xs.im * y.im < 0) y = -y;
        }
        sh = sh - u * cdiv(u, x + y);
      }
    }
    int m = i - 1;
    cf v0, v1;
    for (;; m--) {
      const cf h11 = M(h, m, m), h22 = M(h, m + 1, m + 1);
      cf h11s = h11 - sh;
      float h21 = cf(M(h, m + 1, m)).re;
      const float ss = mag1(h11s) + fabs_(h21);
      h11s = h11s / ss;
      h21 /= ss;
      v0 = h11s;
      v1 = C(h21, 0);
      if (m == l) break;
      if (fabs_(cf(M(h, m, m - 1)).re) * fabs_(h21) <=
          ulp * (mag1(h11s) * (mag1(h11) + mag1(h22))))
        break;
    }
    for (int k2 = m; k2 < i; k2++) {
      barrier();
      if (k2 > m) v0 = M(h, k2, k2 - 1), v1 = M(h, k2 + 1, k2 - 1);
      /* The 2-vector's reflector, every thread alike. */
      cf t1;
      {
        const float xn = mag(v1);
        float ar = v0.re, ai = v0.im;
        if (xn == 0 && ai == 0)
          t1 = C(0, 0);
        else {
          float beta = ar >= 0 ? -lapy3(ar, ai, xn) : lapy3(ar, ai, xn);
          const float sm = safmin();
          int knt = 0;
          cf vv = v1;
          while (fabs_(beta) < sm && knt < 20) {
            knt++;
            vv = vv * (1 / sm);
            beta *= 1 / sm, ar *= 1 / sm, ai *= 1 / sm;
          }
          if (knt) {
            const float an = lapy3(ar, ai, mag(vv));
            beta = ar >= 0 ? -an : an;
          }
          t1 = C((beta - ar) / beta, -(ai / beta));
          v1 = cdiv(vv, C(ar - beta, ai));
          for (int j = 0; j < knt; j++) beta *= sm;
          v0 = C(beta, 0);
        }
      }
      barrier();
      if (k2 > m && t == 0)
        M(h, k2, k2 - 1) = v0, M(h, k2 + 1, k2 - 1) = C(0, 0);
      const cf v2 = v1;
      const float t2 = (t1 * v2).re;
      for (int j = k2 + t; j <= i2; j += nt) {
        const cf sum = conj(t1) * cf(M(h, k2, j)) + t2 * cf(M(h, k2 + 1, j));
        M(h, k2, j) -= sum;
        M(h, k2 + 1, j) -= sum * v2;
      }
      barrier();
      for (int j = i1 + t; j <= metal::min(k2 + 2, i); j += nt) {
        const cf sum = t1 * cf(M(h, j, k2)) + t2 * cf(M(h, j, k2 + 1));
        M(h, j, k2) -= sum;
        M(h, j, k2 + 1) -= sum * conj(v2);
      }
      for (int j = t; z && j < n; j += nt) {
        const cf sum = t1 * cf(M(z, j, k2)) + t2 * cf(M(z, j, k2 + 1));
        M(z, j, k2) -= sum;
        M(z, j, k2 + 1) -= sum * conj(v2);
      }
      if (k2 == m && m > l) {
        /* H(m + 1, m) made real again, as the sweep started below l. */
        cf x = C(1, 0) - t1;
        x = x / lapy3(x.re, x.im, 0.0f);
        barrier();
        if (t == 0) {
          M(h, m + 1, m) = cf(M(h, m + 1, m)) * conj(x);
          if (m + 2 <= i) M(h, m + 2, m + 1) = cf(M(h, m + 2, m + 1)) * x;
        }
        for (int j = m; j <= i; j++) {
          if (j == m + 1) continue;
          barrier();
          for (int p = j + 1 + t; p <= i2; p += nt)
            M(h, j, p) = cf(M(h, j, p)) * x;
          barrier();
          for (int p = i1 + t; p < j; p += nt)
            M(h, p, j) = cf(M(h, p, j)) * conj(x);
          for (int p = t; z && p < n; p += nt)
            M(z, p, j) = cf(M(z, p, j)) * conj(x);
        }
      }
    }
    barrier();
    cf x = M(h, i, i - 1);
    if (x.im != 0) {
      const float r = lapy3(x.re, x.im, 0.0f);
      x = x / r;
      barrier();
      if (t == 0) M(h, i, i - 1) = C(r, 0);
      for (int p = i + 1 + t; p <= i2; p += nt)
        M(h, i, p) = cf(M(h, i, p)) * conj(x);
      barrier();
      for (int p = i1 + t; p < i; p += nt) M(h, p, i) = cf(M(h, p, i)) * x;
      for (int p = t; z && p < n; p += nt) M(z, p, i) = cf(M(z, p, i)) * x;
    }
    its++;
    sweeps++;
  }
  if (s.phase == NX_SP_EIG_RUN && i < lo) s.phase = NX_SP_EIG_DONE;
  s.i = i, s.l = l, s.its = its, s.kdefl = kdefl;
}

/* The Schur factorisation of the real 2 x 2 [a b; c d] (xLANV2). */
static inline void lanv2(thread float &a, thread float &b, thread float &c,
                         thread float &d, thread float &rt1r,
                         thread float &rt1i, thread float &rt2r,
                         thread float &rt2i, thread float &cs,
                         thread float &sn) {
  const float ulp = 2 * lim::eps();
  const float safmn2 = 0x1p-51f, safmx2 = 1 / safmn2;
  if (c == 0) {
    cs = 1;
    sn = 0;
  } else if (b == 0) {
    cs = 0;
    sn = 1;
    const float tt = d;
    d = a;
    a = tt;
    b = -c;
    c = 0;
  } else if (a - d == 0 && copysign_(1.0f, b) != copysign_(1.0f, c)) {
    cs = 1;
    sn = 0;
  } else {
    float temp = a - d, p = temp / 2;
    const float bcmax = fmax_(fabs_(b), fabs_(c));
    const float bcmis =
        fmin_(fabs_(b), fabs_(c)) * copysign_(1.0f, b) * copysign_(1.0f, c);
    const float scale = fmax_(fabs_(p), bcmax);
    float zz = (p / scale) * p + (bcmax / scale) * bcmis;
    if (zz >= 4 * ulp) {
      zz = p + copysign_(sqrt_(scale) * sqrt_(zz), p);
      a = d + zz;
      d = d - (bcmax / zz) * bcmis;
      const float tau = lapy3(c, zz, 0.0f);
      cs = zz / tau;
      sn = c / tau;
      b = b - c;
      c = 0;
    } else {
      float sigma = b + c;
      for (int count = 1;; count++) {
        const float sc = fmax_(fabs_(temp), fabs_(sigma));
        if (sc >= safmx2) {
          sigma *= safmn2;
          temp *= safmn2;
          if (count <= 20) continue;
        }
        if (sc <= safmn2) {
          sigma *= safmx2;
          temp *= safmx2;
          if (count <= 20) continue;
        }
        break;
      }
      p = temp / 2;
      float tau = lapy3(sigma, temp, 0.0f);
      cs = sqrt_((1 + fabs_(sigma) / tau) / 2);
      sn = -(p / (tau * cs)) * copysign_(1.0f, sigma);
      const float aa = a * cs + b * sn, bb = -a * sn + b * cs;
      const float cc = c * cs + d * sn, dd = -c * sn + d * cs;
      a = aa * cs + cc * sn;
      b = bb * cs + dd * sn;
      c = -aa * sn + cc * cs;
      d = -bb * sn + dd * cs;
      temp = (a + d) / 2;
      a = d = temp;
      if (c != 0 && b == 0) {
        b = -c;
        c = 0;
        temp = cs;
        cs = -sn;
        sn = temp;
      } else if (c != 0 && copysign_(1.0f, b) == copysign_(1.0f, c)) {
        const float sab = sqrt_(fabs_(b)), sac = sqrt_(fabs_(c));
        p = copysign_(sab * sac, c);
        tau = 1 / sqrt_(fabs_(b + c));
        a = temp + p;
        d = temp - p;
        b = b - c;
        c = 0;
        const float cs1 = sab * tau, sn1 = sac * tau;
        temp = cs * cs1 - sn * sn1;
        sn = cs * sn1 + sn * cs1;
        cs = temp;
      }
    }
  }
  rt1r = a;
  rt2r = d;
  rt1i = c == 0 ? 0 : sqrt_(fabs_(b)) * sqrt_(fabs_(c));
  rt2i = -rt1i;
}

/* Real: Francis double-shift QR sweeps (xLAHQR), each complex pair a
   2 x 2 block in xLANV2's form under [wantt], at most [q] of them from
   the state [s]. The state becomes FAILED as the complex one's does. */
static inline void lahqr(grp g, thread qr_state &s, bool wantt, int n, int lo,
                         int hi, device float *h, device float *wr,
                         device float *wi, device float *z, int itq, int q) {
  const int t = g.tid, nt = g.nthr;
  const float ulp = 2 * lim::eps();
  if (s.phase == NX_SP_EIG_BEGIN) {
    eig_clear(g, h, n);
    if (lo == hi) {
      if (t == 0) wr[lo] = M(h, lo, lo), wi[lo] = 0;
      s.phase = NX_SP_EIG_DONE;
      return;
    }
    if (t == 0) {
      for (int j = lo; j <= hi - 3; j++) M(h, j + 2, j) = M(h, j + 3, j) = 0;
      if (lo <= hi - 2) M(h, hi, hi - 2) = 0;
    }
    s.phase = NX_SP_EIG_RUN;
  }
  const float smlnum = lim::tiny() * (float(hi - lo + 1) / ulp);
  const int itmax = itq * metal::max(10, hi - lo + 1);
  int i1 = 0, i2 = n - 1;
  int i = s.i, l = s.l, its = s.its, kdefl = s.kdefl;
  for (int sweeps = 0; i >= lo;) {
    barrier();
    int k = i;
    while (k > l && !negligible(h, n, k, lo, hi, smlnum)) k--;
    l = k;
    barrier();
    if (l > lo && t == 0) M(h, l, l - 1) = 0;
    if (l >= i - 1) {
      barrier();
      if (l == i) {
        if (t == 0) wr[i] = M(h, i, i), wi[i] = 0;
      } else {
        float a = M(h, i - 1, i - 1), b = M(h, i - 1, i), c = M(h, i, i - 1),
              d = M(h, i, i), r1r, r1i, r2r, r2i, cs, sn;
        lanv2(a, b, c, d, r1r, r1i, r2r, r2i, cs, sn);
        barrier();
        if (t == 0) {
          M(h, i - 1, i - 1) = a, M(h, i - 1, i) = b, M(h, i, i - 1) = c,
                       M(h, i, i) = d;
          wr[i - 1] = r1r, wi[i - 1] = r1i, wr[i] = r2r, wi[i] = r2i;
        }
        if (wantt) {
          for (int j = i + 1 + t; j <= i2; j += nt) {
            const float x = M(h, i - 1, j), y = M(h, i, j);
            M(h, i - 1, j) = cs * x + sn * y;
            M(h, i, j) = cs * y - sn * x;
          }
          for (int j = i1 + t; j < i - 1; j += nt) {
            const float x = M(h, j, i - 1), y = M(h, j, i);
            M(h, j, i - 1) = cs * x + sn * y;
            M(h, j, i) = cs * y - sn * x;
          }
        }
        for (int j = t; z && j < n; j += nt) {
          const float x = M(z, j, i - 1), y = M(z, j, i);
          M(z, j, i - 1) = cs * x + sn * y;
          M(z, j, i) = cs * y - sn * x;
        }
      }
      /* its runs on: the limit is the matrix's total. */
      kdefl = 0;
      i = l - 1;
      l = lo;
      continue;
    }
    if (its == itmax) {
      s.phase = NX_SP_EIG_FAILED;
      break;
    }
    /* The launch ends before a sweep, after the deflations its last
       sweep allows: the next launch resumes here, as one launch would
       go on. */
    if (sweeps == q) break;
    kdefl++;
    if (!wantt) i1 = l, i2 = i;
    float h11, h12, h21, h22, sh;
    if (kdefl % (2 * KEXSH) == 0 || kdefl % KEXSH == 0) {
      const int p = kdefl % (2 * KEXSH) == 0 ? i : l;
      sh = kdefl % (2 * KEXSH) == 0
               ? fabs_(M(h, i, i - 1)) + fabs_(M(h, i - 1, i - 2))
               : fabs_(M(h, l + 1, l)) + fabs_(M(h, l + 2, l + 1));
      h11 = h22 = 0.75f * sh + M(h, p, p);
      h12 = -0.4375f * sh;
      h21 = sh;
    } else {
      h11 = M(h, i - 1, i - 1);
      h21 = M(h, i, i - 1);
      h12 = M(h, i - 1, i);
      h22 = M(h, i, i);
    }
    sh = fabs_(h11) + fabs_(h12) + fabs_(h21) + fabs_(h22);
    float rt1r = 0, rt1i = 0, rt2r = 0, rt2i = 0;
    if (sh != 0) {
      h11 /= sh, h21 /= sh, h12 /= sh, h22 /= sh;
      const float tr = (h11 + h22) / 2,
                  det = (h11 - tr) * (h22 - tr) - h12 * h21;
      const float rtdisc = sqrt_(fabs_(det));
      if (det >= 0) {
        rt1r = rt2r = tr * sh;
        rt1i = rtdisc * sh;
        rt2i = -rt1i;
      } else {
        rt1r = tr + rtdisc;
        rt2r = tr - rtdisc;
        rt1r = rt2r =
            (fabs_(rt1r - h22) <= fabs_(rt2r - h22) ? rt1r : rt2r) * sh;
      }
    }
    int m = i - 2;
    float v0, v1, v2;
    for (;; m--) {
      float h21s = M(h, m + 1, m);
      const float hmm = M(h, m, m);
      sh = fabs_(hmm - rt2r) + fabs_(rt2i) + fabs_(h21s);
      h21s = M(h, m + 1, m) / sh;
      v0 = h21s * M(h, m, m + 1) + (hmm - rt1r) * ((hmm - rt2r) / sh) -
           rt1i * (rt2i / sh);
      v1 = h21s * (hmm + M(h, m + 1, m + 1) - rt1r - rt2r);
      v2 = h21s * M(h, m + 2, m + 1);
      sh = fabs_(v0) + fabs_(v1) + fabs_(v2);
      v0 /= sh, v1 /= sh, v2 /= sh;
      if (m == l) break;
      if (fabs_(M(h, m, m - 1)) * (fabs_(v1) + fabs_(v2)) <=
          ulp * fabs_(v0) *
              (fabs_(M(h, m - 1, m - 1)) + fabs_(hmm) +
               fabs_(M(h, m + 1, m + 1))))
        break;
    }
    for (int k2 = m; k2 < i; k2++) {
      barrier();
      const int nr = metal::min(3, i - k2 + 1);
      if (k2 > m) {
        v0 = M(h, k2, k2 - 1);
        v1 = M(h, k2 + 1, k2 - 1);
        if (nr == 3) v2 = M(h, k2 + 2, k2 - 1);
      }
      /* The reflector of [v0; v1; v2], every thread alike. */
      float t1 = 0;
      {
        const float xn = nr == 3 ? lapy3(v1, v2, 0.0f) : fabs_(v1);
        float ar = v0;
        if (xn != 0) {
          const float sm = safmin();
          float beta = ar >= 0 ? -lapy3(ar, 0.0f, xn) : lapy3(ar, 0.0f, xn);
          int knt = 0;
          float x1 = v1, x2 = nr == 3 ? v2 : 0;
          while (fabs_(beta) < sm && knt < 20) {
            knt++;
            x1 *= 1 / sm, x2 *= 1 / sm, beta *= 1 / sm, ar *= 1 / sm;
          }
          if (knt) {
            const float an = lapy3(ar, 0.0f, lapy3(x1, x2, 0.0f));
            beta = ar >= 0 ? -an : an;
          }
          t1 = (beta - ar) / beta;
          v1 = x1 / (ar - beta);
          if (nr == 3) v2 = x2 / (ar - beta);
          for (int j = 0; j < knt; j++) beta *= sm;
          v0 = beta;
        }
      }
      barrier();
      if (t == 0) {
        if (k2 > m) {
          M(h, k2, k2 - 1) = v0;
          M(h, k2 + 1, k2 - 1) = 0;
          if (k2 < i - 1) M(h, k2 + 2, k2 - 1) = 0;
        } else if (m > l)
          M(h, k2, k2 - 1) *= 1 - t1;
      }
      const float u2 = v1, t2 = t1 * u2, u3 = nr == 3 ? v2 : 0, t3 = t1 * u3;
      for (int j = k2 + t; j <= i2; j += nt) {
        const float sum = M(h, k2, j) + u2 * M(h, k2 + 1, j) +
                          (nr == 3 ? u3 * M(h, k2 + 2, j) : 0.0f);
        M(h, k2, j) -= sum * t1;
        M(h, k2 + 1, j) -= sum * t2;
        if (nr == 3) M(h, k2 + 2, j) -= sum * t3;
      }
      barrier();
      const int e = nr == 3 ? metal::min(k2 + 3, i) : i;
      for (int j = i1 + t; j <= e; j += nt) {
        const float sum = M(h, j, k2) + u2 * M(h, j, k2 + 1) +
                          (nr == 3 ? u3 * M(h, j, k2 + 2) : 0.0f);
        M(h, j, k2) -= sum * t1;
        M(h, j, k2 + 1) -= sum * t2;
        if (nr == 3) M(h, j, k2 + 2) -= sum * t3;
      }
      for (int j = t; z && j < n; j += nt) {
        const float sum = M(z, j, k2) + u2 * M(z, j, k2 + 1) +
                          (nr == 3 ? u3 * M(z, j, k2 + 2) : 0.0f);
        M(z, j, k2) -= sum * t1;
        M(z, j, k2 + 1) -= sum * t2;
        if (nr == 3) M(z, j, k2 + 2) -= sum * t3;
      }
    }
    its++;
    sweeps++;
  }
  if (s.phase == NX_SP_EIG_RUN && i < lo) s.phase = NX_SP_EIG_DONE;
  s.i = i, s.l = l, s.its = its, s.kdefl = kdefl;
}

/* QR: at most [q] sweeps of each matrix still iterating, its state read
   and written back. [itq] is the limit over max(10, h). One group a
   matrix. */
template <class T>
static inline void eig_qr(grp g, long gx, long ngx, device uchar *wsp,
                          int vectors, int n, int itq, int q, ulong per,
                          long first, long count) {
  for (long b = first + gx; b < count; b += ngx) {
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    qr_state s = state_of(ws.st);
    if (s.phase == NX_SP_EIG_DONE || s.phase == NX_SP_EIG_FAILED) continue;
    const int lo = ws.ok[2], hi = ws.ok[3];
    lahqr(g, s, vectors != 0, n, lo, hi, ws.h, ws.wr, ws.wi,
          vectors ? ws.z : nullptr, itq, q);
    barrier();
    if (g.tid == 0) {
      ws.st[NX_SP_EIG_PHASE] = s.phase;
      ws.st[NX_SP_EIG_I] = s.i;
      ws.st[NX_SP_EIG_L] = s.l;
      ws.st[NX_SP_EIG_ITS] = s.its;
      ws.st[NX_SP_EIG_KDEFL] = s.kdefl;
    }
    barrier();
  }
}

/* w: the eigenvalues scaled back, NaN for a matrix that is not finite or
   whose iteration failed or did not finish in the launches it had, which
   ok then records. Eigenvalues outside [lo, hi] are the isolated
   diagonal; lahqr set the others. A pair's
   imaginary part below 2^-126 in magnitude scaled back is 2^-126, its
   sign kept: the values stay a pair, as their vectors are, and a
   smaller one would read as 0 in the device's own arithmetic. One group
   a matrix. */
template <class T>
static inline void eig_w(grp g, long gx, long ngx, device float *w,
                         device uchar *wsp, int n, ulong per, long first,
                         long count) {
  const int t = g.tid, nt = g.nthr;
  for (long b = first + gx; b < count; b += ngx) {
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    device const T *h = ws.h;
    const bool ok = ws.ok[0] && ws.st[NX_SP_EIG_PHASE] == NX_SP_EIG_DONE;
    const int e = ws.ok[1], lo = ws.ok[2], hi = ws.ok[3];
    barrier();
    if (t == 0) ws.ok[0] = ok;
    for (int j = t; ok && j < n; j += nt)
      if (j < lo || j > hi)
        ws.wr[j] = re(T(M(h, j, j))), ws.wi[j] = im(T(M(h, j, j)));
    barrier();
    for (int j = t; j < n; j += nt) {
      const float x = ok ? scaled(ws.wr[j], -e) + 0.0f : nan_();
      float y = ok ? scaled(ws.wi[j], -e) + 0.0f : nan_();
      if (!real_of<T>::cplx() && ok && ws.wi[j] != 0 && y == 0)
        y = copysign_(lim::eta(), ws.wi[j]);
      w[2 * (ulong(b) * n + j)] = x;
      w[2 * (ulong(b) * n + j) + 1] = y;
    }
    barrier();
  }
}

static inline float cabs1(cf x) { return fabs_(x.re) + fabs_(x.im); }
static inline void cscale(device cf *x, int k, float f) {
  for (int i = 0; i <= k; i++) x[i] = cf(x[i]) * f;
}

/* Vectors: X's column for eigenvalue [ki] of a matrix by back
   substitution (xTREVC3), one thread each, the matrices' eigenvalues
   from [first] to [count] − 1; on reals a pair's thread is its
   second's, filling both columns (real and imaginary parts). */
template <class T>
static inline void eig_vectors(grp g, long gx, long ngx, device uchar *wsp,
                               int n, ulong per, long first, long count) {
  const bool cplx = real_of<T>::cplx();
  const float ulp = 2 * lim::eps();
  const float smlnum = lim::tiny() * (float(n) / ulp),
              bignum = (1 - ulp) / smlnum;
  const bool small = is_small(count);
  for (long q = first + gx * g.nthr + g.tid; q < count; q += ngx * g.nthr) {
    long b = q;
    const int ki = int(split(b, n, small));
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    if (!ws.ok[0]) continue;
    device const T *s = ws.h;
    device const float *wr = ws.wr, *wi = ws.wi;
    device T *xm = ws.x;
    device cf *x = ws.xs + ulong(ki) * n;
    if (!cplx && wi[ki] > 0) continue; /* the pair's second does it */
    const bool pair = !cplx && wi[ki] != 0;
    const int top = pair ? ki - 1 : ki;
    const cf lambda = C(wr[top], wi[top]);
    const float smin = fmax_(ulp * cabs1(lambda), smlnum);
    if (!pair) {
      x[ki] = C(1, 0);
      for (int k = 0; k < ki; k++)
        x[k] = C(-re(T(M(s, k, ki))), -im(T(M(s, k, ki))));
    } else {
      const float bb = re(T(M(s, ki - 1, ki))), cc = re(T(M(s, ki, ki - 1)));
      if (fabs_(bb) >= fabs_(cc)) {
        x[ki - 1] = C(1, 0);
        x[ki] = C(0, wi[top] / bb);
      } else {
        x[ki - 1] = C(-wi[top] / cc, 0);
        x[ki] = C(0, 1);
      }
      for (int k = 0; k < top; k++)
        x[k] = (-cf(x[ki - 1])) * C(re(T(M(s, k, ki - 1))), 0) -
               cf(x[ki]) * C(re(T(M(s, k, ki))), 0);
    }
    for (int j = top - 1; j >= 0;) {
      const bool block = !cplx && j > 0 && re(T(M(s, j, j - 1))) != 0;
      const int j0 = block ? j - 1 : j;
      float xmax = 0;
      if (!block) {
        cf d = C(re(T(M(s, j, j))), im(T(M(s, j, j)))) - lambda;
        if (cabs1(d) < smin) d = C(smin, 0);
        const float dm = cabs1(d), xj = cabs1(cf(x[j]));
        if (dm < 1 && xj > 1 && xj > bignum * dm) cscale(x, ki, 1 / xj);
        x[j] = cdiv(cf(x[j]), d);
        xmax = cabs1(cf(x[j]));
      } else {
        /* The 2 x 2 block by elimination with complete pivoting
           (xLALN2). */
        four e4;
        e4.v0 = C(re(T(M(s, j0, j0))), 0) - lambda;
        e4.v1 = C(re(T(M(s, j0, j))), 0);
        e4.v2 = C(re(T(M(s, j, j0))), 0);
        e4.v3 = C(re(T(M(s, j, j))), 0) - lambda;
        int p = 0;
        for (int q4 = 1; q4 < 4; q4++)
          if (cabs1(e4[q4]) > cabs1(e4[p])) p = q4;
        if (cabs1(e4[p]) < smin) p = 0;
        const int pr = p / 2, pc = p % 2;
        cf u11 = e4[p], l21, u12, u22;
        if (cabs1(u11) < smin) {
          u11 = C(smin, 0);
          l21 = u12 = C(0, 0);
          u22 = C(smin, 0);
        } else {
          l21 = cdiv(e4[(1 - pr) * 2 + pc], u11);
          u12 = e4[pr * 2 + 1 - pc];
          u22 = e4[(1 - pr) * 2 + 1 - pc] - l21 * u12;
          if (cabs1(u22) < smin) u22 = C(smin, 0);
        }
        cf b1 = x[j0 + pr], b2 = cf(x[j0 + 1 - pr]) - l21 * b1;
        const float bm = fmax_(cabs1(b1), cabs1(b2)),
                    um = fmin_(cabs1(u22), cabs1(u11));
        if (um < 1 && bm > 1 && bm > bignum * um) {
          cscale(x, ki, 1 / bm);
          b1 = b1 / bm;
          b2 = b2 / bm;
        }
        const cf y2 = cdiv(b2, u22), y1 = cdiv(b1 - u12 * y2, u11);
        x[j0 + 1 - pc] = y2;
        x[j0 + pc] = y1;
        xmax = fmax_(cabs1(y1), cabs1(y2));
        float emax = 0;
        for (int q4 = 0; q4 < 4; q4++) emax = fmax_(emax, cabs1(e4[q4]));
        if (xmax > 1 && emax > 1 && xmax > bignum / emax) {
          cscale(x, ki, emax / bignum);
          xmax *= emax / bignum;
        }
      }
      const float cmax = fmax_(ws.cn[j], ws.cn[j0]);
      if (xmax > 1 && cmax > bignum / xmax) {
        cscale(x, ki, 1 / xmax);
        xmax = 1;
      }
      for (int k = 0; k < j0; k++) {
        x[k] -= cf(x[j]) * C(re(T(M(s, k, j))), im(T(M(s, k, j))));
        if (block) x[k] -= cf(x[j0]) * C(re(T(M(s, k, j0))), 0);
      }
      j = j0 - 1;
    }
    for (int k = 0; k < n; k++) {
      const cf xk = k <= ki ? cf(x[k]) : C(0, 0);
      if (cplx)
        M(xm, k, ki) = elt<T>(xk.re, xk.im);
      else {
        M(xm, k, top) = elt<T>(xk.re);
        if (pair) M(xm, k, ki) = elt<T>(xk.im);
      }
    }
  }
}

/* The column norms of S's strict upper part, which back substitution
   reads to avoid overflow: one thread a column. */
template <class T>
static inline void eig_norms(grp g, long gx, long ngx, device uchar *wsp,
                             int n, ulong per, long first, long count) {
  const bool small = is_small(count);
  for (long q = first + gx * g.nthr + g.tid; q < count; q += ngx * g.nthr) {
    long b = q;
    const int j = int(split(b, n, small));
    const eig_ws<T> ws = eig_of<T>(wsp, b, n, per);
    device const T *s = ws.h;
    float c = 0;
    for (int i = 0; i < j; i++) c += mag1(T(M(s, i, j)));
    ws.cn[j] = c;
  }
}

/* Column j of Z X at row i, as a complex number: on reals a pair's real
   and imaginary parts, columns j and j + 1. */
static inline cf zx_at(device const float *y, int n, int i, int j, bool cplx,
                       bool pair) {
  if (cplx) return C(y[2 * (ulong(i) * n + j)], y[2 * (ulong(i) * n + j) + 1]);
  return C(y[ulong(i) * n + j], pair ? y[ulong(i) * n + j + 1] : 0.0f);
}

/* Finish: [vo], row-major Z X in, columns made of largest |re| + |im| 1,
   the balancing undone, unit norm and largest element real; NaN where
   the matrix failed. One group a matrix. vo holds complex numbers of
   float parts; on reals Z X is first read from [zx], real. */
static inline void eig_finish(grp g, long gx, long ngx, device const float *zx,
                              device float *vo, device uchar *wsp, int n,
                              bool cplx, ulong per, long first, long count) {
  const int t = g.tid, nt = g.nthr;
  const ulong nn = ulong(n) * n;
  for (long b = first + gx; b < count; b += ngx) {
    const eig_ws<float> ws =
        eig_of<float>(wsp, b, n, per, (cplx ? 2 : 1) * sizeof(float));
    device float *v = vo + 2 * ulong(b) * nn;
    if (!ws.ok[0]) {
      for (ulong k = t; k < 2 * nn; k += nt) v[k] = nan_();
      continue;
    }
    device const float *y = zx + ulong(b) * nn * (cplx ? 2 : 1);
    const int lo = ws.ok[2], hi = ws.ok[3];
    device const float *wi = ws.wi, *scale = ws.scale;
    for (int j = t; j < n; j += nt) {
      if (!cplx && wi[j] < 0) continue;
      const bool pair = !cplx && wi[j] > 0;
      float emax = 0;
      for (int i = 0; i < n; i++)
        emax = fmax_(emax, cabs1(zx_at(y, n, i, j, cplx, pair)));
      const float f = emax > 0 ? 1 / emax : 1.0f;
      /* Balancing undone: rows [lo, hi] scaled, the others swapped back
         in the order they were swapped out. */
      for (int i = 0; i < n; i++) {
        cf c = zx_at(y, n, i, j, cplx, pair) * f;
        if (i >= lo && i <= hi && lo < hi) c = c * scale[i];
        v[2 * (ulong(i) * n + j)] = c.re;
        v[2 * (ulong(i) * n + j) + 1] = c.im;
      }
      for (int p = 0; p < n; p++) {
        const int i = p < lo ? lo - 1 - p : p;
        if (i >= lo && i <= hi) continue;
        const int k = int(scale[i]);
        if (k == i) continue;
        device float *x = v + 2 * (ulong(i) * n + j),
                     *z = v + 2 * (ulong(k) * n + j);
        const float a0 = x[0], a1 = x[1];
        x[0] = z[0], x[1] = z[1], z[0] = a0, z[1] = a1;
      }
      /* Unit norm, scaled by the largest part, and the element of largest
         magnitude, the first of equals, made real and positive. */
      float big = 0;
      for (int i = 0; i < n; i++)
        big = fmax_(big, fmax_(fabs_(v[2 * (ulong(i) * n + j)]),
                               fabs_(v[2 * (ulong(i) * n + j) + 1])));
      float q = 0;
      for (int i = 0; i < n; i++) {
        const float a0 = v[2 * (ulong(i) * n + j)] / big,
                    a1 = v[2 * (ulong(i) * n + j) + 1] / big;
        q += a0 * a0 + a1 * a1;
      }
      const float nrm = big * sqrt_(q);
      int kb = 0;
      float best = -1;
      for (int i = 0; i < n; i++) {
        device float *x = v + 2 * (ulong(i) * n + j);
        x[0] /= nrm;
        x[1] /= nrm;
        const float m2 = x[0] * x[0] + x[1] * x[1];
        if (m2 > best) best = m2, kb = i;
      }
      device float *xk = v + 2 * (ulong(kb) * n + j);
      const float am = hypot_(xk[0], xk[1]);
      const cf ph = C(xk[0] / am, -xk[1] / am);
      for (int i = 0; i < n; i++) {
        device float *x = v + 2 * (ulong(i) * n + j);
        const cf c = C(x[0], x[1]) * ph;
        x[0] = c.re;
        /* A real value's vector is real: +0 imaginary parts. */
        x[1] = i == kb || (!cplx && !pair) ? 0.0f : c.im;
        if (i == kb) x[0] = am;
      }
      /* A real pair's second column is the first's conjugate. */
      if (pair)
        for (int i = 0; i < n; i++) {
          device float *x = v + 2 * (ulong(i) * n + j);
          x[2] = x[0];
          x[3] = -x[1];
        }
    }
  }
}

#undef KEXSH

} // namespace sp

#endif
