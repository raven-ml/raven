/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Contraction, as kernels.h's nx_metal_contract states it: dense products
   on simdgroup matrices, skinny products, of one row of a, on the SIMD
   units, and integer products of wide integers on the SIMD units; and the
   pack, which copies an operand into the layout an instance reads.

   Dense: a threadgroup computes a BM × BN tile of out; its four simdgroups
   interleave, each holding TM × TN accumulators of 8 × 8 float32. Each
   step of BK along k stages a's and b's tiles in threadgroup memory as
   stored, then multiplies them in steps of 8, widening each operand
   exactly to the matrix units' type. Every output sums its terms in
   increasing k, 8 at a time in the GPU's matrix product, then adds init:
   an association fixed by the shape.

   int8 runs there too: it is exact in half, so its sums are exact in
   float32 while they stay under 2^24, and move into wrapping 32-bit sums
   before they could reach it.

   Dense kernels read a stored [m][k], and b stored [k][n] (n) or [n][k]
   (t), as their instance's name says. */

#include "kernels.h"
#include "nx_dtype.h"

using namespace metal;

/* Loops over registers unroll whole, so no array of them is indexed at run
   time, which would put it in memory. */
#define UNROLL _Pragma("clang loop unroll(full)")

#define WM 2
#define WN 2
#define THREADS NX_METAL_THREADS
static_assert(THREADS == 32 * WM * WN, "a dense threadgroup is 4 simdgroups");
#define PAD 4

/* Operand dtypes, by their storage; BF16F is bfloat16 staged as
   float32. */
enum { F32, F16, BF16, BF16F, I8 };

template <int D> struct elt;
/* t: the storage; s: the type tiles stage in threadgroup memory, into
   which stage widens exactly; f: the matrix units' operand type, whose
   products are exact in float32; bk: the steps of k the dense kernel
   stages in its square tiles, the faster as measured on the M1 Max. For
   integers, chunk: how many terms the matrix units may sum in float32 and
   stay exact, every partial sum an integer of at most 2^24 (0 for
   floats). */
template <> struct elt<F32> {
  static constant constexpr uint chunk = 0, bk = NX_METAL_BK;
  typedef float t, s, f;
  static vec<s, 4> stage(vec<t, 4> x) { return x; }
  static float2 get(vec<s, 2> x) { return x; }
  static float4 get4(vec<t, 4> x) { return x; }
  static float get1(t x) { return x; }
};
template <> struct elt<F16> {
  static constant constexpr uint chunk = 0, bk = NX_METAL_BK_HALF;
  typedef half t, s, f;
  static vec<s, 4> stage(vec<t, 4> x) { return x; }
  static half2 get(vec<s, 2> x) { return x; }
  static float4 get4(vec<t, 4> x) { return float4(x); }
  static float get1(t x) { return x; }
};
/* A bfloat16 code is a float32's top half: widening it is a shift, exact
   for every value. A NaN stays a NaN, though not always a quiet one,
   which is all a product needs. */
template <> struct elt<BF16> {
  static constant constexpr uint chunk = 0, bk = NX_METAL_BK;
  typedef ushort t, s;
  typedef float f;
  static vec<s, 4> stage(vec<t, 4> x) { return x; }
  static float2 get(vec<s, 2> x) {
    uint u = as_type<uint>(x);
    return float2(as_type<float>(u << 16), as_type<float>(u & 0xffff0000u));
  }
  static float4 get4(vec<t, 4> x) { return as_type<float4>(uint4(x) << 16); }
  static float get1(t x) { return as_type<float>(uint(x) << 16); }
};

/* bfloat16 widened as it is staged, so the fragments read float32's
   tiles. Measured on the large tile: with b stored [n][k] it runs 2.5%
   faster than BF16 (4096-nt, Llama's up projection, the 512-row
   prefills); with b stored [k][n], 6% slower. */
template <> struct elt<BF16F> {
  static constant constexpr uint chunk = 0, bk = NX_METAL_BK;
  typedef ushort t;
  typedef float s, f;
  static vec<s, 4> stage(vec<t, 4> x) {
    return as_type<float4>(uint4(x) << 16);
  }
  static float2 get(vec<s, 2> x) { return x; }
};

/* Bytes are exact in half, and stage as half; products of int8 are at
   most 2^14. */
template <> struct elt<I8> {
  static constant constexpr uint chunk = 1024, bk = NX_METAL_BK_HALF;
  typedef char t;
  typedef half s, f;
  static vec<s, 4> stage(vec<t, 4> x) { return half4(x); }
  static half2 get(vec<s, 2> x) { return x; }
};

/* x read from the code at [p] of dtype [dt], an out or init dtype. */
static float decode(uint dt, device const void *p, ulong i) {
  switch (dt) {
  case NX_FLOAT16: return float(((device const half *)p)[i]);
  case NX_BFLOAT16: return nx_bf16_to_float(((device const ushort *)p)[i]);
  default: return ((device const float *)p)[i];
  }
}

static void encode(uint dt, device void *p, ulong i, float x) {
  switch (dt) {
  case NX_FLOAT16: ((device half *)p)[i] = half(x); break;
  case NX_BFLOAT16: ((device ushort *)p)[i] = nx_float_to_bf16(x); break;
  default: ((device float *)p)[i] = x;
  }
}

/* The element at [i] of the integer array [p] of dtype [dt], widened to A
   by its sign. */
template <typename A>
static A widen(uint dt, device const uchar *p, ulong i) {
  switch (dt) {
  case NX_INT8: return A(long(((device const char *)p)[i]));
  case NX_UINT8: return A(((device const uchar *)p)[i]);
  case NX_INT16: return A(long(((device const short *)p)[i]));
  case NX_UINT16: return A(((device const ushort *)p)[i]);
  case NX_INT32: return A(long(((device const int *)p)[i]));
  case NX_UINT32: return A(((device const uint *)p)[i]);
  default: return A(((device const ulong *)p)[i]);
  }
}

/* Stores the sum x, wrapped to the accumulator of dtype [acc], at [i] of
   the integer array [p] of dtype [dt], as a cast from the accumulator
   does: its low bits, or, into a wider out, widened by the accumulator's
   sign. */
static void narrow(uint dt, uint acc, device uchar *p, ulong i, ulong x) {
  switch (acc) {
  case NX_INT8: x = ulong(long(char(x))); break;
  case NX_UINT8: x = ulong(uchar(x)); break;
  case NX_INT16: x = ulong(long(short(x))); break;
  case NX_UINT16: x = ulong(ushort(x)); break;
  case NX_INT32: x = ulong(long(int(x))); break;
  case NX_UINT32: x = ulong(uint(x)); break;
  }
  switch (dt) {
  case NX_INT8:
  case NX_UINT8: ((device uchar *)p)[i] = uchar(x); break;
  case NX_INT16:
  case NX_UINT16: ((device ushort *)p)[i] = ushort(x); break;
  case NX_INT32:
  case NX_UINT32: ((device uint *)p)[i] = uint(x); break;
  default: ((device ulong *)p)[i] = x;
  }
}

/* Reads N runs of 4 consecutive elements at src into x, as vectors. With
   WIDE, src lies at a multiple of 16 bytes, and bytes are read 16 at a
   time, four runs from one load: a load reads only at a multiple of its
   alignment. */
template <bool WIDE, uint N, typename T>
static void runs(thread vec<T, 4> *x, device const T *src) {
  UNROLL
  for (uint v = 0; v < N; v++)
    x[v] = vec<T, 4>(*(device const packed_vec<T, 4> *)(src + 4 * v));
}

template <uint N, typename T>
static void bytes(thread vec<T, 4> *x, device const T *src) {
  UNROLL
  for (uint v = 0; v < N / 4; v++) {
    uint4 w = *(device const uint4 *)(src + 16 * v);
    UNROLL
    for (uint e = 0; e < 4; e++) x[4 * v + e] = as_type<vec<T, 4>>(w[e]);
  }
}

template <bool WIDE, uint N>
static void runs(thread char4 *x, device const char *src) {
  if (WIDE) return bytes<N>(x, src);
  runs<false, N, char>(x, src);
}

/* An operand's tile as stored: R rows of C elements, contiguous along a
   row, at step k0 of the k axis, which runs along the rows if K_ROWS and
   along the columns otherwise. Each thread stages PER consecutive
   elements of a row, read as vectors where they lie inside the matrix, 0
   outside. */
template <int D, int R, int C, bool K_ROWS> struct tile {
  static constant constexpr uint per = R * C / THREADS;
  static_assert(per >= 4 && per % 4 == 0,
                "a thread stages whole vectors of 4 elements");
  typedef typename elt<D>::t T;
  typedef typename elt<D>::s S;
  vec<T, 4> x[per / 4];
  device const T *src; /* the thread's first element at step k0 */
  threadgroup S *dst;
  uint ld, rows, cols, r, c;
  bool inside; /* the thread's run lies wholly inside the other axis */

  /* The tile of the rows × cols matrix [m], whose rows lie [ld] elements
     apart, starting at [at] along the axis other than k, staged at [s]. */
  tile(device const T *m, uint ld, uint rows, uint cols, uint at,
       threadgroup S (*s)[C + PAD], uint t)
      : ld(ld), rows(rows), cols(cols) {
    uint tr = t / (C / per), tc = (t % (C / per)) * per;
    r = tr + (K_ROWS ? 0 : at);
    c = tc + (K_ROWS ? at : 0);
    src = m + r * ld + c;
    dst = &s[tr][tc];
    inside = K_ROWS ? c + per <= cols : r < rows;
  }

  /* Reads the thread's elements of the next step along k, which lies
     wholly inside the matrix, and moves to the step after. */
  template <bool WIDE> void fetch() {
    runs<WIDE, per / 4>(x, src);
    src += K_ROWS ? R * ld : C;
  }

  /* fetch, for a step inside the k axis that may reach past the other. */
  void fetch_edge() {
    if (inside) return fetch<false>();
    UNROLL
    for (uint e = 0; e < per; e++)
      x[e / 4][e % 4] = (K_ROWS ? c + e < cols : r < rows) ? src[e] : T(0);
    src += K_ROWS ? R * ld : C;
  }

  /* Reads the thread's elements of the last step along k, k0, which may
     reach past the k axis. */
  void fetch_last(uint k0) {
    uint gr = r + (K_ROWS ? k0 : 0), gc = c + (K_ROWS ? 0 : k0);
    UNROLL
    for (uint e = 0; e < per; e++)
      x[e / 4][e % 4] = gr < rows && gc + e < cols ? src[e] : T(0);
  }

  void put() {
    UNROLL
    for (uint v = 0; v < per / 4; v++)
      *(threadgroup vec<S, 4> *)(dst + 4 * v) = elt<D>::stage(x[v]);
  }
};

/* The two elements a lane holds of an 8 × 8 matrix: for lane l, with
   q = l / 4, those at row (q & 4) + (l / 2) % 4 and columns
   (q & 2)·2 + (l % 2)·2 and the next. */
template <typename U>
static thread vec<U, 2> &elements(thread simdgroup_matrix<U, 8, 8> &x) {
  return reinterpret_cast<thread vec<U, 2> &>(x.thread_elements());
}

/* The lane's elements of an 8 × 8 matrix of a staged tile whose rows lie
   LD apart, [at] elements past the lane's first: the next one along the
   row, or, if TRANS, the next row's, when the tile holds the matrix's
   columns as rows. Offsets are constants, so each read is one instruction
   past a base address computed once. */
template <int D, int LD, bool TRANS>
static vec<typename elt<D>::f, 2>
piece(const threadgroup typename elt<D>::s *base, int at) {
  typedef typename elt<D>::s S;
  if (TRANS) return elt<D>::get(vec<S, 2>(base[at], base[at + LD]));
  return elt<D>::get(*(const threadgroup vec<S, 2> *)(base + at));
}

/* Moves the accumulators' sums, integers, into [sums]. */
template <int TM, int TN>
static void flush(thread simdgroup_float8x8 (&acc)[TM][TN],
                  thread uint2 (&sums)[TM][TN]) {
  UNROLL
  for (uint i = 0; i < TM; i++)
    UNROLL
    for (uint j = 0; j < TN; j++) {
      sums[i][j] += uint2(int2(elements(acc[i][j])));
      acc[i][j] = simdgroup_float8x8(0);
    }
}

/* One step along k: the staged tiles at pa and pb, each lane's first
   element, multiplied into acc in steps of 8. */
template <int D, int TM, int TN, int BK, int ALD, int BLD, bool BT>
static void multiply(thread simdgroup_float8x8 (&acc)[TM][TN],
                     const threadgroup typename elt<D>::s *pa,
                     const threadgroup typename elt<D>::s *pb) {
  UNROLL
  for (uint kk = 0; kk < BK; kk += 8) {
    simdgroup_matrix<typename elt<D>::f, 8, 8> af[TM], bf[TN];
    simdgroup_barrier(mem_flags::mem_none);
    UNROLL
    for (uint i = 0; i < TM; i++)
      elements(af[i]) = piece<D, ALD, false>(pa, 8 * WM * i * ALD + kk);
    simdgroup_barrier(mem_flags::mem_none);
    UNROLL
    for (uint j = 0; j < TN; j++)
      elements(bf[j]) = piece<D, BLD, BT>(
          pb, BT ? 8 * WN * j * BLD + kk : kk * BLD + 8 * WN * j);
    simdgroup_barrier(mem_flags::mem_none);
    UNROLL
    for (uint i = 0; i < TM; i++)
      UNROLL
      for (uint j = 0; j < TN; j++)
        simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
  }
}

template <int D, bool BT, bool EDGE, int BM, int BN, int BK>
kernel void contract(constant nx_metal_contract &p [[buffer(0)]],
                     uint3 g [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  typedef typename elt<D>::t T;
  /* The tile: TM × TN accumulators of 8 × 8 a simdgroup, and BK steps of
     k at a time. */
  constexpr int TM = BM / (8 * WM), TN = BN / (8 * WN);
  /* a's and b's tiles as stored: a's BM × BK; b's BK × BN, or BN × BK if
     BT. */
  constexpr int BR = BT ? BN : BK, BC = BT ? BK : BN;
  typedef typename elt<D>::s S;
  threadgroup S as[BM][BK + PAD], bs[BR][BC + PAD];
  uint z = g.z, k = p.k;
  device const T *a = (device const T *)p.a + z * p.a_batch;
  device const T *b = (device const T *)p.b + z * p.b_batch;
  uint b_ld = BT ? p.b_n : p.b_k;
  /* Threadgroups run in the order of x: 2^swizzle consecutive ones take a
     column of tiles, which read the same tile of b. */
  uint tile_m = (g.y << p.swizzle) + (g.x & ((1u << p.swizzle) - 1));
  uint tile_n = g.x >> p.swizzle;
  uint m0 = tile_m * BM, n0 = tile_n * BN;
  if (m0 >= p.m || n0 >= p.n) return;
  /* The simdgroups interleave: simdgroup s takes the 8 × 8 blocks (i, j)
     of the tile with i = s / WN mod WM and j = s mod WN. */
  uint sm = (sg / WN) * 8, sn = (sg % WN) * 8;
  /* The row and first column of lane's elements. */
  uint q = lane / 4;
  uint fm = (q & 4) + (lane / 2) % 4, fn = (q & 2) * 2 + (lane % 2) * 2;

  /* The lane's first element of a's and b's matrices in the tiles. */
  constexpr int ALD = BK + PAD, BLD = BC + PAD;
  const threadgroup S *pa = &as[sm + fm][fn];
  const threadgroup S *pb =
      BT ? &bs[sn + fn][fm] : &bs[fm][sn + fn];

  simdgroup_float8x8 acc[TM][TN];
  UNROLL
  for (uint i = 0; i < TM; i++)
    UNROLL
    for (uint j = 0; j < TN; j++) acc[i][j] = simdgroup_float8x8(0);
  /* Integers: the exact float32 sums move into 32-bit sums, which wrap,
     every chunk_steps steps, which leaves room for a last partial step. */
  constexpr uint chunk_steps = elt<D>::chunk ? elt<D>::chunk / BK - 1 : 0;
  uint2 sums[TM][TN] = {};

  tile<D, BM, BK, false> ta(a, p.a_m, p.m, k, m0, as, t);
  tile<D, BR, BC, !BT> tb(b, b_ld, BT ? p.n : k, BT ? k : p.n, n0, bs, t);
  /* The steps wholly inside the k axis, read unchecked where the tile
     lies inside m and n, then the step that reaches k's end. Without EDGE,
     every tile lies inside and k is a multiple of BK: the loop the
     compiler sees has no check, which runs it faster. */
  bool inside = !EDGE || (m0 + BM <= p.m && n0 + BN <= p.n);
  uint steps = k / BK;
  if (inside)
    for (uint s = 0; s < steps; s++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      ta.template fetch<!EDGE>();
      tb.template fetch<!EDGE>();
      ta.put();
      tb.put();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      multiply<D, TM, TN, BK, ALD, BLD, BT>(acc, pa, pb);
      if (chunk_steps && (s + 1) % chunk_steps == 0) flush(acc, sums);
    }
  else if (EDGE)
    for (uint s = 0; s < steps; s++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      ta.fetch_edge();
      tb.fetch_edge();
      ta.put();
      tb.put();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      multiply<D, TM, TN, BK, ALD, BLD, BT>(acc, pa, pb);
      if (chunk_steps && (s + 1) % chunk_steps == 0) flush(acc, sums);
    }
  if (EDGE && steps * BK < k) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ta.fetch_last(steps * BK);
    tb.fetch_last(steps * BK);
    ta.put();
    tb.put();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    multiply<D, TM, TN, BK, ALD, BLD, BT>(acc, pa, pb);
  }

  ulong out_at = ulong(g.z) * p.m * p.n;
  device const void *init = (device const uchar *)p.init;
  if (elt<D>::chunk) {
    UNROLL
    for (uint i = 0; i < TM; i++)
      UNROLL
      for (uint j = 0; j < TN; j++) {
        uint r = m0 + sm + 8 * WM * i + fm;
        UNROLL
        for (uint e = 0; e < 2; e++) {
          uint c = n0 + sn + 8 * WN * j + fn + e;
          if (r >= p.m || c >= p.n) continue;
          uint x = sums[i][j][e] + uint(int(elements(acc[i][j])[e]));
          if (p.init_dtype != NX_DTYPE_COUNT)
            x += widen<uint>(p.init_dtype, (device const uchar *)init,
                             z * p.init_batch + ulong(r) * p.init_m +
                                 ulong(c) * p.init_n);
          narrow(p.out_dtype, p.acc, (device uchar *)p.out,
                 out_at + ulong(r) * p.n + c, x);
        }
      }
    return;
  }
  /* Each lane's pairs of a row, stored as pairs where both lie inside. */
  UNROLL
  for (uint i = 0; i < TM; i++)
    UNROLL
    for (uint j = 0; j < TN; j++) {
      uint r = m0 + sm + 8 * WM * i + fm, c = n0 + sn + 8 * WN * j + fn;
      if (!inside && (r >= p.m || c >= p.n)) continue;
      float2 x = elements(acc[i][j]);
      if (p.init_dtype != NX_DTYPE_COUNT) {
        ulong at = z * p.init_batch + ulong(r) * p.init_m + ulong(c) * p.init_n;
        x.x += decode(p.init_dtype, init, at);
        if (inside || c + 1 < p.n)
          x.y += decode(p.init_dtype, init, at + p.init_n);
      }
      ulong o = out_at + ulong(r) * p.n + c;
      if (!inside && c + 1 >= p.n) {
        encode(p.out_dtype, (device void *)p.out, o, x.x);
        continue;
      }
      switch (p.out_dtype) {
      case NX_FLOAT16:
        *(device packed_half2 *)((device half *)p.out + o) = half2(x);
        break;
      case NX_BFLOAT16:
        *(device packed_ushort2 *)((device ushort *)p.out + o) =
            ushort2(nx_float_to_bf16(x.x), nx_float_to_bf16(x.y));
        break;
      default: *(device packed_float2 *)((device float *)p.out + o) = x;
      }
    }
}

/* Skinny products

   Products of one row of a, decode's matrix-vector products, which stream
   b once. Threadgroups of THREADS threads; grid z counts the batch.

   b stored [n][k] (t): a threadgroup computes SKINNY_T columns of out:
   lane l of simdgroup s sums its terms k = 8l + 256s + 1024j + e, e < 8,
   in increasing j and e; a simdgroup's lanes combine by simd_sum, then
   the 4 simdgroups' sums add in increasing s.

   b stored [k][n] (n): a threadgroup computes SKINNY_N columns, 8 per
   lane: lane l of simdgroup s sums columns 8(l mod 4) to 8(l mod 4) + 7
   over the k ≡ l / 4 + 8s (mod 32), in increasing k; the 8 lanes of a
   column add by shuffles across lanes 4, 8 and 16 apart, then the 4
   simdgroups' sums add in increasing s.

   Either way an output's association is a function of the shape. */

#define SKINNY_T NX_METAL_SKINNY_T
#define SKINNY_N NX_METAL_SKINNY_N

/* Adds out's init and stores it, as the dense kernel does. */
static void finish(constant nx_metal_contract &p, uint z, uint c, float x) {
  if (p.init_dtype != NX_DTYPE_COUNT)
    x += decode(p.init_dtype, (device const uchar *)p.init,
                z * p.init_batch + ulong(c) * p.init_n);
  encode(p.out_dtype, (device void *)p.out, ulong(z) * p.n + c, x);
}

/* The 4 elements of a run at q, those at i + e < i_n, 0 past it. */
template <int D>
static float4 run4(device const typename elt<D>::t *q, uint i, uint i_n) {
  typedef typename elt<D>::t T;
  if (i + 4 <= i_n)
    return elt<D>::get4(vec<T, 4>(*(device const packed_vec<T, 4> *)q));
  float4 x = 0;
  for (uint e = 0; i + e < i_n && e < 4; e++) x[e] = elt<D>::get1(q[e]);
  return x;
}

/* The 8 elements of a run at q, inside the matrix, as two float4. With
   V16, q lies on a 16-byte boundary, and 2-byte elements load in one
   16-byte read. */
template <int D> struct run8;
template <> struct run8<F32> {
  template <bool V16>
  static void get(device const float *q, thread float4 &lo,
                  thread float4 &hi) {
    lo = *(device const packed_float4 *)q;
    hi = *(device const packed_float4 *)(q + 4);
  }
};
template <int D> struct run8 {
  typedef typename elt<D>::t T;
  template <bool V16>
  static void get(device const T *q, thread float4 &lo, thread float4 &hi) {
    if (V16) {
      uint4 w = *(device const uint4 *)q;
      lo = elt<D>::get4(as_type<vec<T, 4>>(w.xy));
      hi = elt<D>::get4(as_type<vec<T, 4>>(w.zw));
      return;
    }
    lo = elt<D>::get4(vec<T, 4>(*(device const packed_vec<T, 4> *)q));
    hi = elt<D>::get4(vec<T, 4>(*(device const packed_vec<T, 4> *)(q + 4)));
  }
};

/* Whether runs of 8 elements at multiples of 8 from p, rows ld elements
   apart, lie on 16-byte boundaries. A kernel tests it once and runs the
   loop compiled for its answer: tested in the loop, it made decode 1.5
   times slower. */
template <typename T>
static bool v16(device const T *p, uint ld) {
  return ((ulong(p) | ulong(ld) * sizeof(T)) & 15) == 0;
}

/* The t form's sums over the lane's k from k0, its 8 elements of a run at
   once. */
template <int D, bool V16>
static void sum_t(device const typename elt<D>::t *a,
                  device const typename elt<D>::t *b,
                  constant nx_metal_contract &p, uint n0, uint k0,
                  thread float (&acc)[SKINNY_T]) {
  typedef typename elt<D>::t T;
  for (; k0 < p.k; k0 += THREADS * 8) {
    float av[8];
    if (p.a_k == 1 && k0 + 8 <= p.k) {
      float4 lo, hi;
      run8<D>::template get<V16>(a + k0, lo, hi);
      UNROLL
      for (uint e = 0; e < 4; e++) av[e] = lo[e], av[e + 4] = hi[e];
    } else {
      UNROLL
      for (uint e = 0; e < 8; e++)
        av[e] = k0 + e < p.k ? elt<D>::get1(a[(k0 + e) * p.a_k]) : 0.0f;
    }
    UNROLL
    for (uint j = 0; j < SKINNY_T; j++) {
      device const T *q = b + min(n0 + j, p.n - 1) * p.b_n + k0;
      uint k_n = n0 + j < p.n ? p.k : 0;
      float4 lo, hi;
      if (k0 + 8 <= k_n)
        run8<D>::template get<V16>(q, lo, hi);
      else
        lo = run4<D>(q, k0, k_n), hi = run4<D>(q + 4, k0 + 4, k_n);
      UNROLL
      for (uint e = 0; e < 4; e++) acc[j] = fma(av[e], lo[e], acc[j]);
      UNROLL
      for (uint e = 0; e < 4; e++) acc[j] = fma(av[e + 4], hi[e], acc[j]);
    }
  }
}

/* The n form's sums over its k, its 8 columns inside the matrix. */
template <int D, bool V16>
static void sum_n(device const typename elt<D>::t *a,
                  device const typename elt<D>::t *b,
                  constant nx_metal_contract &p, uint k0, uint n0,
                  thread float4 &lo, thread float4 &hi) {
  for (uint k = k0; k < p.k; k += 32) {
    float av = elt<D>::get1(a[k * p.a_k]);
    float4 bl, bh;
    run8<D>::template get<V16>(b + k * p.b_k + n0, bl, bh);
    lo = fma(float4(av), bl, lo);
    hi = fma(float4(av), bh, hi);
  }
}

template <int D, bool BT>
kernel void skinny(constant nx_metal_contract &p [[buffer(0)]],
                   uint3 g [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
  typedef typename elt<D>::t T;
  device const T *a = (device const T *)p.a + g.z * p.a_batch;
  device const T *b = (device const T *)p.b + g.z * p.b_batch;

  if (BT) {
    threadgroup float part_t[THREADS / 32][SKINNY_T];
    uint n0 = g.x * SKINNY_T;
    if (n0 >= p.n) return;
    float acc[SKINNY_T] = {};
    if (v16(a, 0) && v16(b, p.b_n))
      sum_t<D, true>(a, b, p, n0, 8 * lane + 256 * sg, acc);
    else
      sum_t<D, false>(a, b, p, n0, 8 * lane + 256 * sg, acc);
    UNROLL
    for (uint j = 0; j < SKINNY_T; j++) {
      float x = simd_sum(acc[j]);
      if (lane == 0) part_t[sg][j] = x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg != 0 || lane >= SKINNY_T || n0 + lane >= p.n) return;
    float x = part_t[0][lane];
    UNROLL
    for (uint s = 1; s < THREADS / 32; s++) x += part_t[s][lane];
    finish(p, g.z, n0 + lane, x);
    return;
  }

  threadgroup float4 part[THREADS / 32][SKINNY_N / 4];
  uint n0 = g.x * SKINNY_N + 8 * (lane % 4);
  float4 lo = 0, hi = 0;
  uint k0 = lane / 4 + 8 * sg;
  if (n0 + 8 > p.n)
    for (uint k = k0; k < p.k; k += 32) {
      float av = elt<D>::get1(a[k * p.a_k]);
      device const T *q = b + k * p.b_k + n0;
      lo = fma(float4(av), run4<D>(q, n0, p.n), lo);
      hi = fma(float4(av), run4<D>(q + 4, n0 + 4, p.n), hi);
    }
  else if (v16(b, p.b_k))
    sum_n<D, true>(a, b, p, k0, n0, lo, hi);
  else
    sum_n<D, false>(a, b, p, k0, n0, lo, hi);
  UNROLL
  for (uint d = 4; d < 32; d *= 2) {
    lo += simd_shuffle_xor(lo, d);
    hi += simd_shuffle_xor(hi, d);
  }
  if (lane < 4) {
    part[sg][2 * lane] = lo;
    part[sg][2 * lane + 1] = hi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  float x = part[0][lane / 4][lane % 4];
  UNROLL
  for (uint s = 1; s < THREADS / 32; s++) x += part[s][lane / 4][lane % 4];
  uint c = g.x * SKINNY_N + lane;
  if (c < p.n) finish(p, g.z, c, x);
}

#define SKINNY(name, D, BT)                                               \
  template [[host_name(name)]] kernel void skinny<D, BT>(                \
      constant nx_metal_contract &, uint3, uint, uint);

SKINNY("skinny_f32_n", F32, false)
SKINNY("skinny_f32_t", F32, true)
SKINNY("skinny_f16_n", F16, false)
SKINNY("skinny_f16_t", F16, true)
SKINNY("skinny_bf16_n", BF16, false)
SKINNY("skinny_bf16_t", BF16, true)

/* The parts of a contraction split along k, summed in order, then init,
   rounded once to out. */
kernel void contract_combine(constant nx_metal_combine &p [[buffer(0)]],
                             uint i [[thread_position_in_grid]]) {
  uint mn = p.m * p.n;
  if (i >= p.batch * mn) return;
  uint z = i / mn, r = (i % mn) / p.n, c = i % p.n;
  device const float *parts = (device const float *)p.parts;
  float x = parts[i];
  for (uint q = 1; q < p.split; q++) x += parts[ulong(q) * p.batch * mn + i];
  if (p.init_dtype != NX_DTYPE_COUNT)
    x += decode(p.init_dtype, (device const uchar *)p.init,
                z * p.init_batch + ulong(r) * p.init_m + ulong(c) * p.init_n);
  encode(p.out_dtype, (device void *)p.out, i, x);
}

/* Integer contraction on the SIMD units: operands of any integer dtype
   widen to 64 bits, whose sum wraps; it then wraps to the accumulator and
   reaches out as a cast from it does. Wrapping at every step equals
   wrapping once, so the order of the sum is free, and a sum wrapped to 64
   bits is the accumulator's modulo its width; this kernel sums in
   increasing k. A threadgroup of INT_THREADS threads computes an INT_TILE
   × INT_TILE tile of out, each thread a 4 × 4 block, staging INT_BK steps
   of a's and b's tiles widened. */

#define INT_TILE NX_METAL_INT_TILE
#define INT_THREADS NX_METAL_INT_THREADS
#define INT_BK 16

kernel void contract_int(constant nx_metal_contract &p [[buffer(0)]],
                         uint3 g [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
  /* Unsigned, so the sum wraps as the dtype's arithmetic does. */
  typedef ulong A;
  threadgroup A as[INT_BK][INT_TILE], bs[INT_BK][INT_TILE];
  device const uchar *a = (device const uchar *)p.a;
  device const uchar *b = (device const uchar *)p.b;
  uint m0 = g.y * INT_TILE, n0 = g.x * INT_TILE;
  uint tm = (t / (INT_TILE / 4)) * 4, tn = (t % (INT_TILE / 4)) * 4;
  A acc[4][4] = {};
  for (uint k0 = 0; k0 < p.k; k0 += INT_BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    /* Each thread stages INT_BK·INT_TILE / INT_THREADS elements of each
       tile. */
    for (uint e = t; e < INT_BK * INT_TILE; e += INT_THREADS) {
      uint kk = e / INT_TILE, i = e % INT_TILE, gk = k0 + kk;
      as[kk][i] = m0 + i < p.m && gk < p.k
                      ? widen<A>(p.dtype, a,
                                 g.z * p.a_batch + ulong(m0 + i) * p.a_m +
                                     ulong(gk) * p.a_k)
                      : A(0);
      bs[kk][i] = n0 + i < p.n && gk < p.k
                      ? widen<A>(p.dtype, b,
                                 g.z * p.b_batch + ulong(gk) * p.b_k +
                                     ulong(n0 + i) * p.b_n)
                      : A(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint kk = 0; kk < INT_BK; kk++) {
      vec<A, 4> x = *(threadgroup vec<A, 4> *)&as[kk][tm];
      vec<A, 4> y = *(threadgroup vec<A, 4> *)&bs[kk][tn];
      UNROLL
      for (uint i = 0; i < 4; i++)
        UNROLL
        for (uint j = 0; j < 4; j++) acc[i][j] += x[i] * y[j];
    }
  }
  ulong out_at = ulong(g.z) * p.m * p.n;
  UNROLL
  for (uint i = 0; i < 4; i++)
    UNROLL
    for (uint j = 0; j < 4; j++) {
      uint r = m0 + tm + i, c = n0 + tn + j;
      if (r >= p.m || c >= p.n) continue;
      A x = acc[i][j];
      if (p.init_dtype != NX_DTYPE_COUNT)
        x += widen<A>(p.init_dtype, (device const uchar *)p.init,
                      g.z * p.init_batch + ulong(r) * p.init_m +
                          ulong(c) * p.init_n);
      narrow(p.out_dtype, p.acc, (device uchar *)p.out,
             out_at + ulong(r) * p.n + c, x);
    }
}

/* Packs

   An operand of batch × rows × cols elements of a byte, two or four,
   copied into rows of ld elements with cols contiguous, the ld - cols
   past each row zero. A threadgroup moves a PACK × PACK tile through
   threadgroup memory: each thread reads runs of 4 elements along the
   source's contiguous axis and writes runs of 4 along the copy's, so
   both sides stream whole vectors. */

#define PACK NX_METAL_PACK

/* The run of 4 [bytes]-byte elements at element [i] of p, as words. */
static uint4 get4(uint bytes, device const uchar *p, uint i) {
  switch (bytes) {
  case 1: return uint4(*(device const packed_uchar4 *)(p + i));
  case 2: return uint4(*(device const packed_ushort4 *)(p + 2 * i));
  default: return uint4(*(device const packed_uint4 *)(p + 4 * i));
  }
}

static uint get1(uint bytes, device const uchar *p, uint i) {
  switch (bytes) {
  case 1: return p[i];
  case 2: return ((device const ushort *)p)[i];
  default: return ((device const uint *)p)[i];
  }
}

/* Stores the run x at element [i] of p, on a 16-byte boundary. */
static void set4(uint bytes, device uchar *p, ulong i, uint4 x) {
  switch (bytes) {
  case 1: *(device uchar4 *)(p + i) = uchar4(x); break;
  case 2: *(device ushort4 *)(p + 2 * i) = ushort4(x); break;
  default: *(device uint4 *)(p + 4 * i) = x;
  }
}

kernel void pack(constant nx_metal_pack &p [[buffer(0)]],
                 uint3 g [[threadgroup_position_in_grid]],
                 uint3 t [[thread_position_in_threadgroup]]) {
  threadgroup uint tile[PACK][PACK + 1];
  device const uchar *src =
      (device const uchar *)p.src + g.z * p.src_batch * p.bytes;
  uint r0 = g.y * PACK, c0 = g.x * PACK;
  /* Runs go along cols, unless rows' stride is 1: (u, v) is the tile's
     element (r, c) or (c, r), u along the run. */
  bool down = p.row == 1;
  uint u_n = down ? p.rows : p.cols, v_n = down ? p.cols : p.rows;
  uint u0 = down ? r0 : c0, v0 = down ? c0 : r0;
  uint u_step = down ? p.row : p.col, v_step = down ? p.col : p.row;
  uint u = 4 * t.x;
  for (uint v = t.y; v < PACK; v += NX_METAL_PACK_ROWS) {
    uint4 x = 0;
    if (v0 + v < v_n) {
      uint at = (u0 + u) * u_step + (v0 + v) * v_step;
      if (u_step == 1 && u0 + u + 4 <= u_n)
        x = get4(p.bytes, src, at);
      else
        for (uint e = 0; e < 4 && u0 + u + e < u_n; e++)
          x[e] = get1(p.bytes, src, at + e * u_step);
    }
    for (uint e = 0; e < 4; e++)
      if (down)
        tile[u + e][v] = x[e];
      else
        tile[v][u + e] = x[e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device uchar *dst = (device uchar *)p.dst;
  uint c = c0 + 4 * t.x;
  for (uint i = t.y; i < PACK; i += NX_METAL_PACK_ROWS) {
    uint r = r0 + i;
    if (r >= p.rows || c >= p.ld) continue;
    uint4 x = uint4(tile[i][4 * t.x], tile[i][4 * t.x + 1],
                    tile[i][4 * t.x + 2], tile[i][4 * t.x + 3]);
    set4(p.bytes, dst, g.z * p.dst_batch + ulong(r) * p.ld + c, x);
  }
}

#define CONTRACT(name, D, BT, EDGE, BM, BN, BK)                          \
  template [[host_name(name)]] kernel void                                \
  contract<D, BT, EDGE, BM, BN, BK>(constant nx_metal_contract &, uint3,  \
                                    uint, uint, uint);

/* Each float dtype: large tiles, unchecked, for products of whole tiles
   with b stored either way; small ones (_s) for products of few tiles and
   every product past whole tiles, b stored [k][n]; and wide ones (_w) for
   products of few rows, b stored either way. The small and wide tiles'
   one instance each reads tiles reaching past the matrix: measured, its
   checks cost nothing on whole tiles (256 and 512 squares within 3%;
   16-row products 10-13% faster than an unchecked twin), where the large
   tile's cost 2-11% (f32 4096-nt 9%, 64 x 512 batches 8-11%). The
   large tile with b stored [n][k] stages as DT. */
#define FLOATS(name, D, DT)                                             \
  CONTRACT(name "_n", D, false, false, NX_METAL_LARGE, NX_METAL_LARGE,   \
           elt<D>::bk)                                                    \
  CONTRACT(name "_t", DT, true, false, NX_METAL_LARGE, NX_METAL_LARGE,   \
           elt<DT>::bk)                                                   \
  CONTRACT(name "_s", D, false, true, NX_METAL_SMALL, NX_METAL_SMALL,    \
           elt<D>::bk)                                                    \
  CONTRACT(name "_wn", D, false, true, NX_METAL_WIDE_M, NX_METAL_WIDE_N, \
           NX_METAL_BK_WIDE)                                              \
  CONTRACT(name "_wt", D, true, true, NX_METAL_WIDE_M, NX_METAL_WIDE_N,  \
           NX_METAL_BK_WIDE)

FLOATS("contract_f32", F32, F32)
FLOATS("contract_f16", F16, F16)
FLOATS("contract_bf16", BF16, BF16F)

/* int8 into 32 bits: large tiles, whole, b stored [k][n]. */
CONTRACT("contract_i8", I8, false, false, NX_METAL_LARGE, NX_METAL_LARGE,
         elt<I8>::bk)
