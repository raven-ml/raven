/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Attention (nx_metal.mli, Attention): a group runs a unit of at most
   NX_ATT_ROWS query rows of one key head, four SIMD-groups of 8 rows, over
   the unit's keys in blocks of NX_ATT_BLOCK, in nx.cpu's association.

   Each score is an 8x8 float fragment's element: simdgroup_multiply_
   accumulate is a chain of eight fused multiply-adds in increasing k from
   its addend, so S = Q Kᵀ over the dimensions in increasing order from a
   zero fragment is each score's chain from +0, then one multiply by the
   scale. Dimensions past dk read q = -0 and k = +0, whose product leaves
   every sum as it was. A block's weights stay float32 in the fragments,
   and P V over the block's fragments in increasing key from a zero
   fragment is each element's chain from +0, the block's part p; then
   a = fma(α, a, p) and l = fma(α, l, s), α = 1 where the block's
   greatest is -inf.

   q's rows, k's keys and v's keys stage in threadgroup memory in their
   own dtype and widen exactly where a fragment reads them, so staging
   changes no bit. q stages once a unit where it holds dk; k stages
   transposed, so that a fragment's two columns are adjacent. With pages or
   a mask, each key's element offsets and each row's visible keys are
   found once a block; else keys are dense rows, visible below the length.
   A row of 4 elements loads as one vector where the plan finds it
   contiguous and aligned.

   Loops over dimensions and columns skip the steps past dk and dv rather
   than leave the loop, so that the compiler unrolls them and holds the
   fragments in registers.

   A thread holds row (q & 4) + (i/2 mod 4) of a fragment, columns
   (q & 2)·2 + (i mod 2)·2 and the next, i its lane and q = i/4. So the
   thread's two columns sum a row's weights in nx.cpu's lanes j mod 8, and
   its lanes' tree +4, +2, +1 is a shuffle at xor 8, one at xor 1, then
   the thread's own pair.

   A key past the length, or of a table entry outside [0, pages), is
   absent: its k and v are never read and stage as +0, and its pairs are
   hidden. A hidden pair's weight is +0, whose products with finite values
   change only a zero's sign, which o = a / l + 0 removes. Where a staged
   value is a NaN or an infinity, the columns staged with it take their
   products one by one over the visible pairs, in increasing key, so a
   value hidden from a row never reaches it.

   A full block, 64 dense keys below the length without pages or a mask,
   shows every live row each of its pairs. No pair is hidden, so its
   values take the matrix units whatever they hold, and its scores take
   the general body's tests only where one is a NaN or an infinity. */

#include <metal_simdgroup_matrix>

#include "device.h"
#include "nx_kinds.h"

using metal::simdgroup_float8x8;

struct attention_words {
  device const uchar *q;
  device const uchar *k;
  device const uchar *v;
  device uchar *o;
  device float *lse;
  device const long *lengths;
  device const long *table;
  device const uchar *mask;
  device const float *x;
  device float *scores;
  constant long *lead;
  ulong flags;
  ulong scale;
  long dk;
  long dv;
  long tq;
  long tk;
  long hq;
  long hk;
  long unit_heads;
  long unit_rows;
  long page;
  long pages;
  long leads;
  long nb;
  long first_unit;
  long g;
  long tiles;
  long chunks;
  ulong tiles_div;
  ulong chunks_div;
  ulong hk_div;
  ulong rows_div;
  ulong page_div;
  long q_h, q_t, q_d;
  long k_h, k_t, k_d, k_p;
  long v_h, v_t, v_d, v_p;
  long o_h, o_t, o_d;
  long lse_h, lse_t;
  long tab_p;
  long mask_h, mask_t, mask_k;
  long x_h, x_t, x_k;
};

NX_FIELD(attention_words, q, NX_ATT_Q);
NX_FIELD(attention_words, k, NX_ATT_K);
NX_FIELD(attention_words, v, NX_ATT_V);
NX_FIELD(attention_words, o, NX_ATT_O);
NX_FIELD(attention_words, lse, NX_ATT_LSE);
NX_FIELD(attention_words, lengths, NX_ATT_LENGTHS_BUF);
NX_FIELD(attention_words, table, NX_ATT_TABLE);
NX_FIELD(attention_words, mask, NX_ATT_MASK_BUF);
NX_FIELD(attention_words, x, NX_ATT_X_BUF);
NX_FIELD(attention_words, scores, NX_ATT_SCORES_BUF);
NX_FIELD(attention_words, lead, NX_ATT_LEAD);
NX_FIELD(attention_words, flags, NX_ATT_FLAGS);
NX_FIELD(attention_words, scale, NX_ATT_SCALE);
NX_FIELD(attention_words, dk, NX_ATT_DK);
NX_FIELD(attention_words, dv, NX_ATT_DV);
NX_FIELD(attention_words, tq, NX_ATT_TQ);
NX_FIELD(attention_words, tk, NX_ATT_TK);
NX_FIELD(attention_words, hq, NX_ATT_HQ);
NX_FIELD(attention_words, hk, NX_ATT_HK);
NX_FIELD(attention_words, unit_heads, NX_ATT_UNIT_HEADS);
NX_FIELD(attention_words, unit_rows, NX_ATT_UNIT_ROWS);
NX_FIELD(attention_words, page, NX_ATT_PAGE);
NX_FIELD(attention_words, pages, NX_ATT_PAGES);
NX_FIELD(attention_words, leads, NX_ATT_LEADS);
NX_FIELD(attention_words, nb, NX_ATT_NB);
NX_FIELD(attention_words, first_unit, NX_ATT_FIRST_UNIT);
NX_FIELD(attention_words, g, NX_ATT_G);
NX_FIELD(attention_words, tiles, NX_ATT_TILES_N);
NX_FIELD(attention_words, chunks, NX_ATT_CHUNKS_N);
NX_FIELD(attention_words, tiles_div, NX_ATT_TILES_DIV);
NX_FIELD(attention_words, chunks_div, NX_ATT_CHUNKS_DIV);
NX_FIELD(attention_words, hk_div, NX_ATT_HK_DIV);
NX_FIELD(attention_words, rows_div, NX_ATT_ROWS_DIV);
NX_FIELD(attention_words, page_div, NX_ATT_PAGE_DIV);
NX_FIELD(attention_words, q_h, NX_ATT_Q_H);
NX_FIELD(attention_words, k_h, NX_ATT_K_H);
NX_FIELD(attention_words, k_p, NX_ATT_K_P);
NX_FIELD(attention_words, v_h, NX_ATT_V_H);
NX_FIELD(attention_words, v_p, NX_ATT_V_P);
NX_FIELD(attention_words, o_h, NX_ATT_O_H);
NX_FIELD(attention_words, lse_h, NX_ATT_LSE_H);
NX_FIELD(attention_words, tab_p, NX_ATT_TAB_P);
NX_FIELD(attention_words, mask_h, NX_ATT_MASK_H);
NX_FIELD(attention_words, x_h, NX_ATT_X_H);
NX_FIELD(attention_words, x_k, NX_ATT_X_K);
NX_SIZE(attention_words, NX_ATT_WORDS);

/* Elements of q, k, v and o by their dtype: [T] their stored bits, which
   widen exactly to float, and a float rounds once to nearest to store.
   A NaN widens to a NaN. */
struct att_f32 {
  typedef float T;
  static float neg0() { return -0.f; }
  static float wide(float x) { return x; }
  static float2 wide2(metal::vec<float, 2> x) { return x; }
  static float4 wide4(metal::vec<float, 4> x) { return x; }
  static void put(device uchar *b, long i, float x) {
    ((device float *)b)[i] = x;
  }
};

struct att_f16 {
  typedef ushort T;
  static ushort neg0() { return 0x8000; }
  static float wide(ushort x) { return float(as_type<half>(x)); }
  static float2 wide2(metal::vec<ushort, 2> x) {
    return float2(as_type<half2>(x));
  }
  static float4 wide4(metal::vec<ushort, 4> x) {
    return float4(as_type<half4>(x));
  }
  static void put(device uchar *b, long i, float x) {
    ((device ushort *)b)[i] = nx_metal_float_to_f16(x);
  }
};

struct att_bf16 {
  typedef ushort T;
  static ushort neg0() { return 0x8000; }
  static float wide(ushort x) { return as_type<float>(uint(x) << 16); }
  static float2 wide2(metal::vec<ushort, 2> x) {
    return as_type<float2>(uint2(x) << 16);
  }
  static float4 wide4(metal::vec<ushort, 4> x) {
    return as_type<float4>(uint4(x) << 16);
  }
  static void put(device uchar *b, long i, float x) {
    ((device ushort *)b)[i] = nx_float_to_bf16(x);
  }
};

/* Prog.Exp's bits: its near form where |x| <= NX_EXP_NEAR_F32, which
   gives the kind's bits there, else the kind. */
static inline float att_exp(float x) {
  return metal::fabs(x) <= NX_EXP_NEAR_F32 ? nx_exp_near_f32(x)
                                           : nx_exp_f32(x);
}

/* A fragment's row and first column at [lane], and the lane that holds
   row [r]'s columns [c] and c + 1, c even. */
static inline uint frag_row(uint lane) {
  return ((lane / 4) & 4) + ((lane / 2) % 4);
}
static inline uint frag_col(uint lane) {
  return ((lane / 4) & 2) * 2 + (lane % 2) * 2;
}
static inline ushort frag_lane(uint r, uint c) {
  return ushort(((r & 4) << 2) | ((c & 4) << 1) | ((r & 3) << 1) |
                ((c >> 1) & 1));
}

/* Whether pair [f], [e] of a thread is visible, from its row's key bits
   shifted down by its first column: bit 8 (f mod 4) + e of [lo] for f < 4,
   of [hi] else. With [f] and [e] constant, a constant shift of 32 bits. */
static inline bool att_pair(uint lo, uint hi, int f, int e) {
  return (((f < 4 ? lo : hi) >> ((f % 4) * 8 + e)) & 1) != 0;
}

/* A unit's row [i]: its head and query row, where it has one. */
struct att_row {
  long head;
  long row;
  bool live;
};

static inline att_row unit_row(constant attention_words &w, uint i, long h0,
                               long nh, long i0, long nt) {
  att_row r;
  const uint hh = divide(i, w.rows_div);
  const long t = long(i) - long(hh) * w.unit_rows;
  r.live = long(i) < w.unit_heads * w.unit_rows && long(hh) < nh && t < nt;
  r.head = h0 + long(hh);
  r.row = i0 + t;
  return r;
}

/* An offset of 32 bits, whose arithmetic wraps, read as the signed offset
   it stands for; one of 64 bits as itself. */
static inline int att_at(uint x) { return as_type<int>(x); }
static inline long att_at(long x) { return x; }

/* Four elements of a row at [p], from element [at] on, at step [s]: those
   below [n] read, the others [z]. One vector load where [vec], which the
   plan sets where s = 1, n is a multiple of 4 and each row starts on a
   vector. Offsets [I] within the row are uint where the plan finds them
   all below 2^31 in magnitude (NX_ATT_NARROW), else long. */
template <typename D, typename I>
static inline metal::vec<typename D::T, 4>
att_four(device const typename D::T *p, I at, I n, I s, bool vec,
         typename D::T z) {
  typedef typename D::T T;
  if (vec)
    return at < n ? *(device const metal::vec<T, 4> *)(p + att_at(at))
                  : metal::vec<T, 4>(z);
  metal::vec<T, 4> x = metal::vec<T, 4>(z);
  _Pragma("clang loop unroll(full)")
  for (int i = 0; i < 4; i++)
    if (at + I(i) < n) x[i] = p[att_at((at + I(i)) * s)];
  return x;
}

/* q's rows into [qs], CW dims from [d0], dims past dk -0. */
template <typename D, int CW>
static inline void att_stage_q(threadgroup typename D::T *qs,
                               device const typename D::T *qp,
                               threadgroup const long *qrow,
                               threadgroup const uchar *rlive, long d0,
                               long dk, long q_d, bool qvec, uint tid) {
  typedef metal::vec<typename D::T, 4> T4;
  constexpr int LDQ = NX_ATT_LDQ(sizeof(typename D::T), CW);
  for (uint e = tid * 4; e < uint(NX_ATT_ROWS * CW);
       e += NX_ATT_THREADS * 4) {
    const uint i = e / CW, c = e % CW;
    T4 x = T4(D::neg0());
    if (rlive[i])
      x = att_four<D, long>(qp + qrow[i], d0 + long(c), dk, q_d, qvec,
                            D::neg0());
    *(threadgroup T4 *)(qs + i * LDQ + c) = x;
  }
}

/* Key [j] of a block: whether it is present, and its row. A plain block's
   keys are the dense rows from the block's first, [p], at [step], present
   below [nk]; others' are the block pass's [seen] and [ent], from [p]. */
template <bool PLAIN, typename I, typename T>
static inline bool att_key(uint j, long nk, device const T *p, I step,
                           threadgroup const uchar *seen,
                           threadgroup const long *ent,
                           thread device const T *&row) {
  if (PLAIN) {
    row = p + att_at(I(j) * step);
    return long(j) < nk;
  }
  row = p + ent[j];
  return seen[j] != 0;
}

/* k's keys [j0, j0 + SK) of a block into [st] transposed, CW dims from
   [d0], +0 past dk and for an absent key. Where the plan finds rows of
   vectors and the chunk lies below dk, 4 dims are one load and no test. */
template <typename D, int CW, int SK, bool PLAIN, typename I>
static inline void att_stage_k(threadgroup typename D::T *st,
                               device const typename D::T *kp, uint j0,
                               long nk, I step, threadgroup const uchar *seen,
                               threadgroup const long *ent, I d0, I dk, I k_d,
                               bool kvec, uint tid) {
  typedef typename D::T T;
  typedef metal::vec<T, 4> T4;
  constexpr int LDK = NX_ATT_LDK(sizeof(T), SK);
  constexpr uint STEP = NX_ATT_THREADS * 4;
  const bool whole = kvec && d0 + I(CW) <= dk;
  for (uint e = tid * 4; e < uint(SK * CW); e += STEP) {
    const uint kk = e / CW, c = e % CW;
    device const T *row;
    T4 x = T4(0);
    if (att_key<PLAIN, I, T>(j0 + kk, nk, kp, step, seen, ent, row)) {
      if (whole)
        x = *(device const T4 *)(row + att_at(d0 + I(c)));
      else
        x = att_four<D, I>(row, d0 + I(c), dk, k_d, kvec, T(0));
    }
    st[(c + 0) * LDK + kk] = x[0];
    st[(c + 1) * LDK + kk] = x[1];
    st[(c + 2) * LDK + kk] = x[2];
    st[(c + 3) * LDK + kk] = x[3];
  }
}

/* v's keys [j0, j0 + SK) of a block into [st], C columns from [col0], +0
   past dv and for an absent key, as att_stage_k; whether one this thread
   staged is a NaN or an infinity. */
template <typename D, int C, int SK, bool PLAIN, typename I>
static inline bool att_stage_v(threadgroup typename D::T *st,
                               device const typename D::T *vp, uint j0,
                               long nk, I step, threadgroup const uchar *seen,
                               threadgroup const long *ent, I col0, I dv,
                               I v_d, bool vvec, uint tid) {
  typedef typename D::T T;
  typedef metal::vec<T, 4> T4;
  constexpr int LDV = NX_ATT_LDV(sizeof(T), C);
  constexpr uint STEP = NX_ATT_THREADS * 4;
  const bool whole = vvec && col0 + I(C) <= dv;
  bool odd = false;
  for (uint e = tid * 4; e < uint(SK * C); e += STEP) {
    const uint kk = e / C, c = e % C;
    device const T *row;
    T4 x = T4(0);
    if (att_key<PLAIN, I, T>(j0 + kk, nk, vp, step, seen, ent, row)) {
      if (whole)
        x = *(device const T4 *)(row + att_at(col0 + I(c)));
      else
        x = att_four<D, I>(row, col0 + I(c), dv, v_d, vvec, T(0));
    }
    odd |= !metal::all(metal::isfinite(D::wide4(x)));
    *(threadgroup T4 *)(st + kk * LDV + c) = x;
  }
  return odd;
}

/* att_stage_k and att_stage_v of a plain block or by the block pass, at
   32-bit offsets within a block where [narrow], else 64-bit ones. */
template <typename D, int CW, int SK>
static inline void att_stage_kb(threadgroup typename D::T *st,
                                device const typename D::T *kp, uint j0,
                                long nk, bool plain, bool narrow, long step,
                                threadgroup const uchar *seen,
                                threadgroup const long *ent, long d0, long dk,
                                long k_d, bool kvec, uint tid) {
  if (narrow && plain)
    att_stage_k<D, CW, SK, true, uint>(st, kp, j0, nk, uint(step), seen, ent,
                                       uint(d0), uint(dk), uint(k_d), kvec,
                                       tid);
  else if (narrow)
    att_stage_k<D, CW, SK, false, uint>(st, kp, j0, nk, uint(step), seen, ent,
                                        uint(d0), uint(dk), uint(k_d), kvec,
                                        tid);
  else if (plain)
    att_stage_k<D, CW, SK, true, long>(st, kp, j0, nk, step, seen, ent, d0,
                                       dk, k_d, kvec, tid);
  else
    att_stage_k<D, CW, SK, false, long>(st, kp, j0, nk, step, seen, ent, d0,
                                        dk, k_d, kvec, tid);
}

template <typename D, int C, int SK>
static inline bool att_stage_vb(threadgroup typename D::T *st,
                                device const typename D::T *vp, uint j0,
                                long nk, bool plain, bool narrow, long step,
                                threadgroup const uchar *seen,
                                threadgroup const long *ent, long col0,
                                long dv, long v_d, bool vvec, uint tid) {
  if (narrow && plain)
    return att_stage_v<D, C, SK, true, uint>(st, vp, j0, nk, uint(step), seen,
                                             ent, uint(col0), uint(dv),
                                             uint(v_d), vvec, tid);
  if (narrow)
    return att_stage_v<D, C, SK, false, uint>(st, vp, j0, nk, uint(step),
                                              seen, ent, uint(col0), uint(dv),
                                              uint(v_d), vvec, tid);
  if (plain)
    return att_stage_v<D, C, SK, true, long>(st, vp, j0, nk, step, seen, ent,
                                             col0, dv, v_d, vvec, tid);
  return att_stage_v<D, C, SK, false, long>(st, vp, j0, nk, step, seen, ent,
                                            col0, dv, v_d, vvec, tid);
}

/* A unit of a key head's rows over its keys, a piece of the columns: dk
   held whole where DK is 64 or 128, in chunks of NX_ATT_CHUNK where it is
   0. */
template <typename D, int DK, int NC>
static void attention_tile(constant attention_words &w,
                           threadgroup uchar *mem, uint unit_g, uint piece,
                           uint tid, uint sg, uint lane) {
  typedef typename D::T T;
  constexpr int SZ = sizeof(T);
  constexpr bool HELD = DK > 0;
  constexpr int CW = HELD ? DK : NX_ATT_CHUNK;
  constexpr int SK = NX_ATT_STAGE_KEYS(SZ);
  constexpr int NS = SK / 8;
  constexpr int LDQ = NX_ATT_LDQ(SZ, CW);
  constexpr int LDK = NX_ATT_LDK(SZ, SK);
  typedef metal::vec<T, 2> T2;

  threadgroup ushort *vis16 = (threadgroup ushort *)(mem + NX_ATT_AT_VIS);
  threadgroup uchar *bad = mem + NX_ATT_AT_BAD;
  threadgroup uchar *seen = mem + NX_ATT_AT_SEEN;
  threadgroup uchar *rlive = mem + NX_ATT_AT_LIVE;
  threadgroup long *kent = (threadgroup long *)(mem + NX_ATT_AT_KENT);
  threadgroup long *vent = (threadgroup long *)(mem + NX_ATT_AT_VENT);
  threadgroup long *qrow = (threadgroup long *)(mem + NX_ATT_AT_QROW);
  threadgroup T *qs = (threadgroup T *)(mem + NX_ATT_AT_Q);
  threadgroup T *st = (threadgroup T *)(mem + NX_ATT_AT_STAGE(SZ, CW));

  const ulong flags = w.flags;
  const bool paged = (flags & NX_ATT_PAGED) != 0;
  const bool qvec = (flags & NX_ATT_QVEC) != 0;
  const bool kvec = (flags & NX_ATT_KVEC) != 0;
  const bool vvec = (flags & NX_ATT_VVEC) != 0;
  const float scale = as_type<float>(uint(w.scale));
  device const T *qp = (device const T *)w.q;
  device const T *kp = (device const T *)w.k;
  device const T *vp = (device const T *)w.v;
  const long dk = w.dk, dv = w.dv;

  /* The unit: lead index b, key head kh, a chunk of its heads and a tile
     of their rows, in nx.cpu's order. */
  const long g = w.g;
  const ulong u = ulong(w.first_unit) + unit_g;
  const ulong u1 = nx_divide(u, ulong(w.tiles), w.tiles_div);
  const ulong u2 = nx_divide(u1, ulong(w.chunks), w.chunks_div);
  const ulong ub = nx_divide(u2, ulong(w.hk), w.hk_div);
  const long tile = long(u) - long(u1) * w.tiles;
  const long chunk = long(u1) - long(u2) * w.chunks;
  const long kh = long(u2) - long(ub) * w.hk, b = long(ub);
  const long h0 = kh * g + chunk * w.unit_heads;
  const long nh = metal::min(w.unit_heads, g - chunk * w.unit_heads);
  const long i0 = tile * w.unit_rows;
  const long nt = metal::min(w.unit_rows, w.tq - i0);

  /* Each array's offset at lead index b: the lead table holds the lead
     extents, their divisors, then each array's steps. */
  long off[NX_ATT_LEAD_ARRAYS];
  for (int j = 0; j < NX_ATT_LEAD_ARRAYS; j++) off[j] = 0;
  ulong rest = ub;
  for (long a = w.leads - 1; a >= 0; a--) {
    const ulong up =
        nx_divide(rest, ulong(w.lead[a]), ulong(w.lead[w.leads + a]));
    const long i = long(rest) - long(up) * w.lead[a];
    rest = up;
    for (int j = 0; j < NX_ATT_LEAD_ARRAYS; j++)
      off[j] += i * w.lead[2 * w.leads + j * w.leads + a];
  }
  long len = w.tk;
  if (flags & NX_ATT_LENGTHS)
    len = metal::max(0l, metal::min(w.lengths[off[NX_ATT_LLEN]], w.tk));

  /* This thread's fragment row and columns, and the row whose visible
     keys it finds. */
  const uint fm = frag_row(lane), fn = frag_col(lane);
  const uint ri = sg * 8 + fm;
  const att_row me = unit_row(w, ri, h0, nh, i0, nt);
  const long col0 = long(piece) * NX_ATT_PIECE;
  const long xrow = off[NX_ATT_LX] + me.head * w.x_h + me.row * w.x_t;
  const bool scored = (flags & NX_ATT_X) != 0;
  const uint vr = tid / 4, q4 = (tid % 4) * 16;
  const att_row vrow = unit_row(w, vr, h0, nh, i0, nt);
  const long mrow =
      off[NX_ATT_LMASK] + vrow.head * w.mask_h + vrow.row * w.mask_t;

  /* q's rows: each one's offset and whether it is one. */
  if (tid < NX_ATT_ROWS) {
    const att_row r = unit_row(w, tid, h0, nh, i0, nt);
    rlive[tid] = r.live;
    qrow[tid] = off[NX_ATT_LQ] + r.head * w.q_h + r.row * w.q_t;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (HELD)
    att_stage_q<D, CW>(qs, qp, qrow, rlive, 0, dk, w.q_d, qvec, tid);

  simdgroup_float8x8 O[NC];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NC; j++)
    O[j] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.f);
  float m = -INFINITY, l = 0.f;

  /* Without pages or a mask, a block takes no pass: its keys are dense
     rows, and every pair of a row before the length is visible. */
  const bool plain = (flags & (NX_ATT_PAGED | NX_ATT_MASK)) == 0;

  /* k's and v's rows from their key head's first element, or the pool's
     where paged; offsets of 32 bits within a block and a row where the
     plan finds them below 2^31 in magnitude. */
  const bool narrow = (flags & NX_ATT_NARROW) != 0;
  device const T *kq = paged ? kp : kp + off[NX_ATT_LK] + kh * w.k_h;
  device const T *vq = paged ? vp : vp + off[NX_ATT_LV] + kh * w.v_h;

  /* ib counts the unit's blocks in 32 bits: a unit of 2^32 blocks passes
     a launch's budget and declines (Too_long). */
  for (uint ib = 0; long(ib) * NX_ATT_BLOCK < len; ib++) {
    const long kb = long(ib) * NX_ATT_BLOCK;
    float alpha = 1.f;
    const long nk = metal::min(long(NX_ATT_BLOCK), len - kb);
    /* A full block's keys are every bit: a shift by 64 is undefined, and
       select would compute it. */
    const ulong keys = nk < NX_ATT_BLOCK ? (1ul << nk) - 1 : ~0ul;
    ulong mine = me.live ? keys : 0ul;

    /* A dense block of 64 present keys every live row sees whole. */
    const bool full = plain && kb + NX_ATT_BLOCK <= len;

    /* Else each key's presence and the offsets of its k and v rows; then
       each row's visible keys, a quarter of a row a thread. */
    if (!plain) {
      if (tid < NX_ATT_BLOCK) {
        bool here = long(tid) < nk;
        long ka = 0, va = 0;
        if (here) {
          const long key = kb + long(tid);
          if (paged) {
            const ulong pi = nx_divide(ulong(key), ulong(w.page), w.page_div);
            const long slot = key - long(pi) * w.page;
            const long e = w.table[off[NX_ATT_LTAB] + long(pi) * w.tab_p];
            here = e >= 0 && e < w.pages;
            ka = e * w.k_p + kh * w.k_h + slot * w.k_t;
            va = e * w.v_p + kh * w.v_h + slot * w.v_t;
          } else {
            ka = key * w.k_t;
            va = key * w.v_t;
          }
        }
        seen[tid] = here;
        kent[tid] = ka;
        vent[tid] = va;
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      {
        uint bits = 0;
        for (uint j = 0; j < 16; j++) {
          const long key = kb + long(q4 + j);
          bool vis = vrow.live && seen[q4 + j] != 0;
          if (vis && (flags & NX_ATT_MASK))
            vis = w.mask[mrow + key * w.mask_k] != 0;
          bits |= uint(vis) << j;
        }
        vis16[vr * 4 + tid % 4] = ushort(bits);
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);

      /* A block no row sees leaves every state as it was. */
      const uint seen32 =
          uint(vis16[lane * 4]) | uint(vis16[lane * 4 + 1]) |
          uint(vis16[lane * 4 + 2]) | uint(vis16[lane * 4 + 3]);
      if (metal::simd_or(seen32) == 0) continue;
      mine = ulong(vis16[ri * 4]) | (ulong(vis16[ri * 4 + 1]) << 16) |
             (ulong(vis16[ri * 4 + 2]) << 32) |
             (ulong(vis16[ri * 4 + 3]) << 48);
    }

    /* S = Q Kᵀ, SK keys of k staged at a time, transposed. */
    simdgroup_float8x8 S[8];
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < 8; f++)
      S[f] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    _Pragma("clang loop unroll(full)")
    for (int sub = 0; sub < NX_ATT_BLOCK / SK; sub++)
      for (long d0 = 0; !scored && d0 < (HELD ? 1 : dk); d0 += CW) {
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (!HELD)
          att_stage_q<D, CW>(qs, qp, qrow, rlive, d0, dk, w.q_d, qvec, tid);
        att_stage_kb<D, CW, SK>(st, plain ? kq + kb * w.k_t : kq,
                                uint(sub) * SK, nk, plain, narrow, w.k_t,
                                seen, kent, d0, dk, w.k_d, kvec, tid);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)")
        for (int d8 = 0; d8 < CW; d8 += 8) {
          if (d0 + d8 >= dk) continue;
          simdgroup_float8x8 qf;
          const float2 qq =
              D::wide2(*(threadgroup T2 *)(qs + ri * LDQ + d8 + fn));
          qf.thread_elements()[0] = qq[0];
          qf.thread_elements()[1] = qq[1];
          _Pragma("clang loop unroll(full)")
          for (int f = 0; f < NS; f++) {
            simdgroup_float8x8 kf;
            const float2 k2 = D::wide2(
                *(threadgroup T2 *)(st + (d8 + fm) * LDK + f * 8 + fn));
            kf.thread_elements()[0] = k2[0];
            kf.thread_elements()[1] = k2[1];
            metal::simdgroup_multiply_accumulate(S[sub * NS + f], qf, kf,
                                                 S[sub * NS + f]);
          }
        }
      }

    /* The scores; where the first launch stores them, nothing else. */
    _Pragma("clang loop unroll(full)")
    for (int f = 0; f < 8; f++) {
      S[f].thread_elements()[0] = S[f].thread_elements()[0] * scale;
      S[f].thread_elements()[1] = S[f].thread_elements()[1] * scale;
    }
    if (flags & NX_ATT_SCORES) {
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < 8; f++)
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 2; e++) {
          const long key = kb + long(f * 8 + int(fn) + e);
          if (me.live && key < w.tk)
            w.scores[((b * w.hq + me.head) * w.tq + me.row) * w.tk + key] =
                S[f].thread_elements()[e];
        }
      continue;
    }

    /* This thread's pairs' visibility, its column 8 f + fn + e at bit
       8 f + e (att_pair). */
    const ulong mf = mine >> fn;
    const uint mlo = uint(mf), mhi = uint(mf >> 32);

    /* Output 1 at each visible pair. */
    if (flags & (NX_ATT_X | NX_ATT_ADD | NX_ATT_SUB | NX_ATT_RSUB | NX_ATT_MUL))
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < 8; f++)
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 2; e++) {
          const uint j = uint(f) * 8 + fn + uint(e);
          if (full ? !me.live : !att_pair(mlo, mhi, f, e)) continue;
          const float y = w.x[xrow + (kb + long(j)) * w.x_k];
          float s = S[f].thread_elements()[e];
          if (flags & NX_ATT_X) s = y;
          else if (flags & NX_ATT_ADD) s = s + y;
          else if (flags & NX_ATT_SUB) s = s - y;
          else if (flags & NX_ATT_RSUB) s = y - s;
          else s = s * y;
          S[f].thread_elements()[e] = s;
        }

    /* A full block: the thread's greatest and least score, and whether a
       score of the SIMD-group is a NaN or an infinity (0 s is then a
       NaN), which sends the block to the general body. */
    bool general = !full;
    if (full) {
      float hi = -INFINITY, lw = INFINITY;
      float bd = 0.f;
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < 8; f++)
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 2; e++) {
          const float s = S[f].thread_elements()[e];
          hi = s > hi ? s : hi;
          lw = s < lw ? s : lw;
          bd = metal::fma(0.f, s, bd);
        }
      general = metal::simd_any(bd != bd);
      if (!general) {
        float fx = hi > m ? hi : m;
        const float y1 = metal::simd_shuffle_xor(fx, 1);
        fx = y1 > fx ? y1 : fx;
        const float y8 = metal::simd_shuffle_xor(fx, 8);
        fx = y8 > fx ? y8 : fx;
        alpha = att_exp(m - fx);
        const bool far = !(lw - fx >= -NX_EXP_NEAR_F32);
        float l0 = 0.f, l1 = 0.f;
        if (metal::simd_any(far)) {
          _Pragma("clang loop unroll(full)")
          for (int f = 0; f < 8; f++) {
            const float w0 = att_exp(S[f].thread_elements()[0] - fx);
            const float w1 = att_exp(S[f].thread_elements()[1] - fx);
            S[f].thread_elements()[0] = w0;
            S[f].thread_elements()[1] = w1;
            l0 = l0 + w0;
            l1 = l1 + w1;
          }
        } else {
          _Pragma("clang loop unroll(full)")
          for (int f = 0; f < 8; f++) {
            const float w0 = nx_exp_near_f32(S[f].thread_elements()[0] - fx);
            const float w1 = nx_exp_near_f32(S[f].thread_elements()[1] - fx);
            S[f].thread_elements()[0] = w0;
            S[f].thread_elements()[1] = w1;
            l0 = l0 + w0;
            l1 = l1 + w1;
          }
        }
        l0 = l0 + metal::simd_shuffle_xor(l0, 8);
        l1 = l1 + metal::simd_shuffle_xor(l1, 8);
        l0 = l0 + metal::simd_shuffle_xor(l0, 1);
        l1 = l1 + metal::simd_shuffle_xor(l1, 1);
        l = metal::fma(alpha, l, l0 + l1);
        m = fx;
      }
    }

    /* Else -inf at a hidden pair; the greatest, a NaN where one is, as
       nx.cpu's; and the thread's least score above -inf. */
    if (general) {
      float mx = m, lo = INFINITY;
      uint nan = 0;
      _Pragma("clang loop unroll(full)")
      for (int f = 0; f < 8; f++)
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 2; e++) {
          const float s =
              att_pair(mlo, mhi, f, e) ? S[f].thread_elements()[e] : -INFINITY;
          S[f].thread_elements()[e] = s;
          nan |= uint(s != s);
          mx = s > mx ? s : mx;
          lo = s < lo && s != -INFINITY ? s : lo;
        }
      {
        const float y1 = metal::simd_shuffle_xor(mx, 1);
        mx = y1 > mx ? y1 : mx;
        const float y8 = metal::simd_shuffle_xor(mx, 8);
        mx = y8 > mx ? y8 : mx;
        nan |= metal::simd_shuffle_xor(nan, 1);
        nan |= metal::simd_shuffle_xor(nan, 8);
        if (nan != 0) mx = NAN;
      }

      /* Weights, l in nx.cpu's lanes, and α. A row whose greatest is -inf
         takes weights 0, α 1, and keeps its state. */
      if (mx == -INFINITY) {
        _Pragma("clang loop unroll(full)")
        for (int f = 0; f < 8; f++) {
          S[f].thread_elements()[0] = 0.f;
          S[f].thread_elements()[1] = 0.f;
        }
      } else {
        alpha = att_exp(m - mx);

        /* Where each pair of the SIMD-group lies in exp's near range or at
           -inf, whose weight is +0, the near form alone gives Exp's bits.
           With mx finite, a pair's s - mx is -inf or, rounding being
           monotone, within [lo - mx, 0], so lo - mx decides for each of the
           thread's pairs; it is +inf where none is above -inf. A NaN or +inf
           mx, at which some s - mx may be a NaN, is far. */
        const bool far = !(lo - mx >= -NX_EXP_NEAR_F32);
        float l0 = 0.f, l1 = 0.f;
        if (metal::simd_any(far)) {
          _Pragma("clang loop unroll(full)")
          for (int f = 0; f < 8; f++) {
            const float w0 = att_exp(S[f].thread_elements()[0] - mx);
            const float w1 = att_exp(S[f].thread_elements()[1] - mx);
            S[f].thread_elements()[0] = w0;
            S[f].thread_elements()[1] = w1;
            l0 = l0 + w0;
            l1 = l1 + w1;
          }
        } else {
          _Pragma("clang loop unroll(full)")
          for (int f = 0; f < 8; f++) {
            const float d0 = S[f].thread_elements()[0] - mx;
            const float d1 = S[f].thread_elements()[1] - mx;
            const float w0 = d0 == -INFINITY ? 0.f : nx_exp_near_f32(d0);
            const float w1 = d1 == -INFINITY ? 0.f : nx_exp_near_f32(d1);
            S[f].thread_elements()[0] = w0;
            S[f].thread_elements()[1] = w1;
            l0 = l0 + w0;
            l1 = l1 + w1;
          }
        }
        l0 = l0 + metal::simd_shuffle_xor(l0, 8);
        l1 = l1 + metal::simd_shuffle_xor(l1, 8);
        l0 = l0 + metal::simd_shuffle_xor(l0, 1);
        l1 = l1 + metal::simd_shuffle_xor(l1, 1);
        l = metal::fma(alpha, l, l0 + l1);
        m = mx;
      }
    }

    /* p = P V from +0 over the block's keys, NX_ATT_VCOLS columns of v
       staged at a time, then a = fma(α, a, p). Each SIMD-group flags the
       values it staged that are not finite, outside a full block. */
    constexpr int NP = NX_ATT_VCOLS / 8;
    constexpr int LDV = NX_ATT_LDV(SZ, NX_ATT_VCOLS);
    _Pragma("clang loop unroll(full)")
    for (int c0 = 0; c0 < NC * 8; c0 += NX_ATT_VCOLS) {
      if (col0 + c0 >= dv) continue;
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      bool odd = att_stage_vb<D, NX_ATT_VCOLS, NX_ATT_BLOCK>(
          st, plain ? vq + kb * w.v_t : vq, 0, nk, plain, narrow, w.v_t, seen,
          vent, col0 + c0, dv, w.v_d, vvec, tid);
      if (!full) {
        odd = metal::simd_any(odd);
        if (lane == 0) bad[sg] = odd;
      }
      threadgroup_barrier(metal::mem_flags::mem_threadgroup);
      const bool one = !full && (bad[0] | bad[1] | bad[2] | bad[3]) != 0;

      _Pragma("clang loop unroll(full)")
      for (int jj = 0; jj < NP; jj++) {
        const int j = c0 / 8 + jj;
        if (col0 + j * 8 >= dv) continue;
        simdgroup_float8x8 P =
            metal::make_filled_simdgroup_matrix<float, 8, 8>(0.f);
        if (!one) {
          _Pragma("clang loop unroll(full)")
          for (int f = 0; f < 8; f++) {
            simdgroup_float8x8 vf;
            const float2 v2 = D::wide2(
                *(threadgroup T2 *)(st + (f * 8 + fm) * LDV + jj * 8 + fn));
            vf.thread_elements()[0] = v2[0];
            vf.thread_elements()[1] = v2[1];
            metal::simdgroup_multiply_accumulate(P, S[f], vf, P);
          }
        } else {
          /* One product at a time over the visible pairs, in increasing
             key, each weight from the lane that holds it. */
          _Pragma("clang loop unroll(full)")
          for (int kk = 0; kk < NX_ATT_BLOCK; kk++) {
            const int f = kk / 8, c = kk % 8;
            const float wt = metal::simd_shuffle(
                S[f].thread_elements()[c & 1], frag_lane(fm, uint(c & 6)));
            if (((mine >> kk) & 1) == 0) continue;
            const float2 v2 =
                D::wide2(*(threadgroup T2 *)(st + kk * LDV + jj * 8 + fn));
            P.thread_elements()[0] =
                metal::fma(wt, v2[0], P.thread_elements()[0]);
            P.thread_elements()[1] =
                metal::fma(wt, v2[1], P.thread_elements()[1]);
          }
        }
        _Pragma("clang loop unroll(full)")
        for (int e = 0; e < 2; e++)
          O[j].thread_elements()[e] = metal::fma(
              alpha, O[j].thread_elements()[e], P.thread_elements()[e]);
      }
    }
  }
  if ((flags & NX_ATT_SCORES) || !me.live) return;

  /* o = a / l + 0, rounded once to o's dtype; lse = m + log l. */
  const bool none = l == 0.f;
  const long orow = off[NX_ATT_LO] + me.head * w.o_h + me.row * w.o_t;
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < NC; j++)
    for (int e = 0; e < 2; e++) {
      const long col = col0 + j * 8 + long(fn) + e;
      if (col >= dv) continue;
      const float a = O[j].thread_elements()[e];
      const float y = none ? 0.f : a / l + 0.f;
      if (flags & NX_ATT_O32) att_f32::put(w.o, orow + col * w.o_d, y);
      else D::put(w.o, orow + col * w.o_d, y);
    }
  if (piece == 0 && fn == 0)
    w.lse[off[NX_ATT_LLSE] + me.head * w.lse_h + me.row * w.lse_t] =
        none ? -INFINITY : (m == INFINITY ? m : m + nx_log_f32(l));
}

#define ATT_TILE(code, D, d, dk, c)                                            \
  NX_KERNEL(NX_ATT_THREADS)                                                    \
  void attention_tile_##code##_##d##_##c(                                      \
      constant attention_words &w [[buffer(0)]],                               \
      threadgroup uchar *mem [[threadgroup(0)]],                               \
      uint3 g [[threadgroup_position_in_grid]],                                \
      uint tid [[thread_index_in_threadgroup]],                                \
      uint sg [[simdgroup_index_in_threadgroup]],                              \
      uint lane [[thread_index_in_simdgroup]]) {                               \
    attention_tile<D, dk, c / 8>(w, mem, g.x, g.y, tid, sg, lane);             \
  }

#define ATT_TILES_D(code, D, d, dk)                                            \
  ATT_TILE(code, D, d, dk, 32)                                                 \
  ATT_TILE(code, D, d, dk, 64) ATT_TILE(code, D, d, dk, 128)

#define ATT_TILES(code, D)                                                     \
  ATT_TILES_D(code, D, 64, 64)                                                 \
  ATT_TILES_D(code, D, 128, 128) ATT_TILES_D(code, D, w, 0)

ATT_TILES(f32, att_f32)
ATT_TILES(f16, att_f16)
ATT_TILES(bf16, att_bf16)
