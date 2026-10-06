/*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* sort and argsort along an axis, as nx.cpu orders (cpu/nx_c_sort.c).
   SORT(float32) defines the kernels of float32.

   Each element becomes an unsigned key that orders like it, paired with its
   position: an unsigned integer or a boolean is its own key, a signed integer
   flips its sign bit, and a float complements a negative value's bits and
   sets a positive one's sign bit, so -0 sorts just below +0. Every NaN takes
   the dtype width's greatest key, which no number takes, so NaNs tie above
   every number. A descending sort complements the keys within the width:
   the exact reverse order, NaN first. Pairs order by key, then position, so
   the order is total and equal elements keep their input order in either
   direction.

   Each row's pairs go to scratch, [keys] and [pos], padded to a power of two
   2^[plog] with pairs of the greatest key and positions past the row, which
   sort after every element. A bitonic network sorts the padded rows; then
   [v] copies the elements at the sorted positions, bit for bit, and [i]
   writes the positions.

   Row r's element j lies at [base(r) + j * step], base(r) the place of r in
   the rows' extents [oext] at strides [ostr], for the operand and for the
   result alike.
   - [k (x, keys, pos, ometa)]: the pairs.
   - [l (keys, pos, total, plog, klo, khi, groups)]: for each stage k from
     [klo] to [khi], the steps of distance at most BLOCK / 2, a block of
     BLOCK slots at a time in the workgroup's memory.
   - [g (keys, pos, total, plog, k, j, groups)]: one step of distance [j] of
     stage [k] over all slots.
   - [v (dst, x, pos, ometa)] and [i (dst, pos, ometa)]. */

#include "common.h"

enum { BLOCK = 2048 };

struct ometa {
  i64 rows, len, plog, groups, orank, xoff, xstep, dstep;
  uint64_t flip;
  i64 oext[MAX_RANK], xostr[MAX_RANK], dostr[MAX_RANK];
};

/* The unsigned word of a storage width. */
template <int W> struct word;
template <> struct word<1> { typedef uint8_t t; };
template <> struct word<2> { typedef uint16_t t; };
template <> struct word<4> { typedef uint32_t t; };
template <> struct word<8> { typedef uint64_t t; };

/* The key of element [x] of [D]. */
template <class D> DEVICE uint64_t key(typename D::storage x) {
  typedef typename word<sizeof(typename D::storage)>::t U;
  const U sign = (U)((U)1 << (sizeof(U) * 8 - 1));
  U b = __builtin_bit_cast(U, x);
  if (D::cat == FLOAT) {
    typename D::compute v = D::load(x);
    if (v != v) return (U)~(U)0;
    return b & sign ? (U)~b : (U)(b | sign);
  }
  if (D::cat == SINT) return (U)(b ^ sign);
  return b;
}

/* Where row [r] starts in a view of the rows' extents at strides [str]. */
DEVICE i64 row(const ometa &m, const i64 *str, i64 r) {
  i64 o = 0;
  for (i64 d = m.orank - 1; d >= 0; d--) {
    o += (r % m.oext[d]) * str[d];
    r /= m.oext[d];
  }
  return o;
}

template <class D>
DEVICE void pairs(const typename D::storage *x, uint64_t *keys, i64 *pos,
                  const ometa &m) {
  i64 p = (i64)1 << m.plog, total = m.rows << m.plog;
  for (i64 s = first(); s < total; s += m.groups * THREADS) {
    i64 r = s >> m.plog, j = s & (p - 1);
    if (j < m.len) {
      keys[s] = key<D>(x[m.xoff + row(m, m.xostr, r) + j * m.xstep]) ^ m.flip;
    } else {
      keys[s] = ~(uint64_t)0;
    }
    pos[s] = j;
  }
}

/* Whether pair a orders after pair b. */
DEVICE bool after(uint64_t ka, i64 pa, uint64_t kb, i64 pb) {
  return ka > kb || (ka == kb && pa > pb);
}

/* The step of distance [j] of stage [k] at slot [s], whose partner [s + j]
   is in its row: the pair goes ascending where bit k of the slot's place in
   its row is clear. */
DEVICE bool ascending(i64 s, i64 plog, i64 k) {
  return ((s & (((i64)1 << plog) - 1)) & k) == 0;
}

DEVICE void local(uint64_t *keys, i64 *pos, i64 total, i64 plog, i64 klo,
                  i64 khi, i64 groups) {
  __attribute__((shared)) static uint64_t sk[BLOCK];
  __attribute__((shared)) static i64 sp[BLOCK];
  int t = __builtin_amdgcn_workitem_id_x();
  for (i64 b = (i64)__builtin_amdgcn_workgroup_id_x() * BLOCK; b < total;
       b += groups * BLOCK) {
    for (int e = t; e < BLOCK; e += THREADS)
      if (b + e < total) {
        sk[e] = keys[b + e];
        sp[e] = pos[b + e];
      }
    barrier();
    for (i64 k = klo; k <= khi; k <<= 1) {
      for (i64 j = (k >> 1) < BLOCK / 2 ? k >> 1 : BLOCK / 2; j > 0; j >>= 1) {
        for (int q = t; q < BLOCK / 2; q += THREADS) {
          int lo = 2 * q - (q & (j - 1)), hi = lo + j;
          if (b + hi < total) {
            bool up = ascending(b + lo, plog, k);
            if (after(sk[lo], sp[lo], sk[hi], sp[hi]) == up) {
              uint64_t kk = sk[lo];
              i64 pp = sp[lo];
              sk[lo] = sk[hi];
              sp[lo] = sp[hi];
              sk[hi] = kk;
              sp[hi] = pp;
            }
          }
        }
        barrier();
      }
    }
    for (int e = t; e < BLOCK; e += THREADS)
      if (b + e < total) {
        keys[b + e] = sk[e];
        pos[b + e] = sp[e];
      }
    barrier();
  }
}

DEVICE void global(uint64_t *keys, i64 *pos, i64 total, i64 plog, i64 k,
                   i64 j, i64 groups) {
  for (i64 q = first(); q < total / 2; q += groups * THREADS) {
    i64 lo = 2 * q - (q & (j - 1)), hi = lo + j;
    bool up = ascending(lo, plog, k);
    if (after(keys[lo], pos[lo], keys[hi], pos[hi]) == up) {
      uint64_t kk = keys[lo];
      i64 pp = pos[lo];
      keys[lo] = keys[hi];
      pos[lo] = pos[hi];
      keys[hi] = kk;
      pos[hi] = pp;
    }
  }
}

/* The sorted rows' first [len] slots, as elements or positions. */
template <class T, class F>
DEVICE void results(T *dst, const ometa &m, F at) {
  i64 n = m.rows * m.len;
  for (i64 e = first(); e < n; e += m.groups * THREADS) {
    i64 r = e / m.len, j = e - r * m.len;
    dst[row(m, m.dostr, r) + j * m.dstep] = at(r, j);
  }
}

#define SORT(D)                                                                \
  KERNEL void NAME(k)(const D::storage *x, uint64_t *keys, i64 *pos,           \
                      ometa m) {                                               \
    pairs<D>(x, keys, pos, m);                                                 \
  }                                                                            \
  KERNEL void NAME(l)(uint64_t *keys, i64 *pos, i64 total, i64 plog, i64 klo,  \
                      i64 khi, i64 groups) {                                   \
    local(keys, pos, total, plog, klo, khi, groups);                           \
  }                                                                            \
  KERNEL void NAME(g)(uint64_t *keys, i64 *pos, i64 total, i64 plog, i64 k,    \
                      i64 j, i64 groups) {                                     \
    global(keys, pos, total, plog, k, j, groups);                              \
  }                                                                            \
  KERNEL void NAME(v)(D::storage *dst, const D::storage *x, const i64 *pos,    \
                      ometa m) {                                               \
    results(dst, m, [&](i64 r, i64 j) {                                        \
      return x[m.xoff + row(m, m.xostr, r) + pos[(r << m.plog) + j] * m.xstep]; \
    });                                                                        \
  }                                                                            \
  KERNEL void NAME(i)(i64 *dst, const i64 *pos, ometa m) {                     \
    results(dst, m, [&](i64 r, i64 j) { return pos[(r << m.plog) + j]; });     \
  }
