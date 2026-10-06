/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* unfold and fold: the sliding windows over an operand's last [k] axes, and
   their sum back, as nx.cpu computes them (cpu/nx_c_move.c).

   Window w of tap kf along spatial axis d, at kernel coordinate kc and
   window coordinate wc, covers position [wc * stride[d] - before[d] + kc *
   dilation[d]]; the windows per axis are [win]. Leading index [lead] of the
   operand is at [off + place(lead, lext, lstr)].

   - UNFOLD(uint32_t), keyed by width: [u (dst, x, wmeta)] writes element
     (lead, kf, w) of the result, of shape (leading..., kprod, l), as the
     operand's element the tap covers, at spatial strides [xstr] and extents
     [extent], or 0 where it falls in the padding.
   - FOLD(float32): [f (dst, x, wmeta)] writes element (lead, o) of the
     result, of shape (leading..., extent...), as +0 plus the taps that cover
     o in increasing kf, at the compute type, rounded once: the operand's
     tap kf of window w at [kstep] and [lstep]. Integers add in 64 bits and
     wrap, booleans in their byte. */

#include "common.h"

struct wmeta {
  i64 n, groups, lrank, k, kprod, l, off, kstep, lstep;
  i64 lext[MAX_RANK], lstr[MAX_RANK];
  i64 kernel[MAX_RANK], stride[MAX_RANK], dilation[MAX_RANK];
  i64 before[MAX_RANK], extent[MAX_RANK], win[MAX_RANK], xstr[MAX_RANK];
};

DEVICE i64 place(i64 i, i64 rank, const i64 *ext, const i64 *str) {
  i64 o = 0;
  for (i64 d = rank - 1; d >= 0; d--) {
    o += (i % ext[d]) * str[d];
    i /= ext[d];
  }
  return o;
}

template <class T> DEVICE void unfold(T *dst, const T *x, const wmeta &m) {
  for (i64 e = first(); e < m.n; e += m.groups * THREADS) {
    i64 w = e % m.l, t = e / m.l;
    i64 kf = t % m.kprod, lead = t / m.kprod;
    i64 o = m.off + place(lead, m.lrank, m.lext, m.lstr);
    bool inside = true;
    for (i64 d = m.k - 1; d >= 0; d--) {
      i64 kc = kf % m.kernel[d], wc = w % m.win[d];
      kf /= m.kernel[d];
      w /= m.win[d];
      i64 p = wc * m.stride[d] - m.before[d] + kc * m.dilation[d];
      inside = inside && p >= 0 && p < m.extent[d];
      o += p * m.xstr[d];
    }
    dst[e] = inside ? x[o] : (T)0;
  }
}

/* A boolean sum, in its byte. */
DEVICE uint8_t add(uint8_t a, uint8_t b) { return a + b; }

template <class D>
DEVICE void fold(typename D::storage *dst, const typename D::storage *x,
                 const wmeta &m) {
  typedef typename D::compute A;
  i64 spatial = 1;
  for (i64 d = 0; d < m.k; d++) spatial *= m.extent[d];
  for (i64 e = first(); e < m.n; e += m.groups * THREADS) {
    i64 lead = e / spatial, os = e - lead * spatial;
    const typename D::storage *xl =
        x + m.off + place(lead, m.lrank, m.lext, m.lstr);
    A acc = 0;
    for (i64 kf = 0; kf < m.kprod; kf++) {
      i64 r = kf, q = os, w = 0, wstep = 1;
      bool covers = true;
      for (i64 d = m.k - 1; d >= 0 && covers; d--) {
        i64 kc = r % m.kernel[d], oc = q % m.extent[d];
        r /= m.kernel[d];
        q /= m.extent[d];
        i64 num = oc + m.before[d] - kc * m.dilation[d];
        covers = num >= 0 && num % m.stride[d] == 0 && num / m.stride[d] < m.win[d];
        w += (num / m.stride[d]) * wstep;
        wstep *= m.win[d];
      }
      if (covers) acc = add(acc, D::load(xl[kf * m.kstep + w * m.lstep]));
    }
    dst[e] = store<D>(acc);
  }
}

#define UNFOLD(T)                                                              \
  KERNEL void u(T *dst, const T *x, wmeta m) { unfold(dst, x, m); }

#define FOLD(D)                                                                \
  KERNEL void f(D::storage *dst, const D::storage *x, wmeta m) {               \
    fold<D>(dst, x, m);                                                        \
  }
