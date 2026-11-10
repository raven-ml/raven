/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Attention (Nx_kernel.S.attention): each query row's softmax over its
   visible keys' scores, times their values, by blocks of NX_ATT_KEYS keys
   with a running maximum m, sum l and accumulator a per row
   (FlashAttention-2's forward, Dao 2023, arXiv 2307.08691, Algorithm 1).

   A unit is a lead index's key head, some of its group's query heads
   ([hg] at most) and some of their rows ([tr] at most): a tile of rows
   that share each block of keys and values. A group of NX_ATT_THREADS
   threads takes a unit and NX_ATT_COLUMNS columns of o (64 in float64),
   and walks the unit's blocks from key 0, up to its lead index's length:

     classes (output 0's ranges, a chunk of blocks at once)
       --> per block: keys' offsets, absent where a table entry is out of
           range; each row's visible keys: output 0 on a mixed block
       --> S = q k^T, times the scale; output 1 on the visible pairs
       --> m' = max (m, S); weights exp (s - m'); l = α l + Σ, the sum in
           nx.cpu's eight lanes and their tree; a = α a
       --> a += p v
     then o = a / l + 0 and lse = m + log l, or a part's (m, l, a)

   The FFMA kernel (simt) computes nx.cpu's arithmetic, so a row's bits are
   nx.cpu's: each score's products fused in increasing dimension from +0,
   each value product fused in increasing key. The matrix-unit kernel
   (mma) takes float16 or bfloat16 q, k and v into float32: q k^T and p v
   on mma.sync m16n8k16, each weight rounded once to v's dtype for p v,
   the softmax in registers in nx.cpu's lane order. A block whose values
   hold one that is not finite takes, in either kernel, p v by fused
   products over each row's visible pairs alone, so a value hidden from a
   row never reaches its o.

   The pair program runs on loop.cu's tape over NX_ATT_THREADS x V
   elements, the kinds NX_ATT_KINDS lists in line, its planes first in
   dynamic shared memory: the kernel fills
   the planes of the nodes the host gave (the score, coordinates, loads,
   or their bounds over a block's box), runs the tape and reads its result.
   A thread fills and reads its own elements alone, so a tape runs without
   barriers.

   Parts: where the units are few, the keys are cut into parts of [pblocks]
   blocks; each part's (m, l, a) goes to the workspace, and merge combines
   them in part order as a block does. */

#ifndef NX_CUDA_ATTENTION_CU
#define NX_CUDA_ATTENTION_CU

#include <type_traits>

#include "loop.cu"

#define KEYS NX_ATT_KEYS
#define AT NX_ATT_THREADS

namespace att {

using nx_loop::held;
using nx_loop::word;

enum { HIDDEN, MIXED, SHOWN };

/* Words */

__device__ __forceinline__ const u64 *desc(const u64 *w, int f) {
  return w + w[f];
}

__device__ __forceinline__ const uint8_t *base(const u64 *w,
                                               const u64 *d) {
  return (const uint8_t *)w[NX_ATT_HEADER + (d[0] & 0xffff)];
}

__device__ __forceinline__ int dtype(const u64 *d) {
  return (int)((d[0] >> 16) & 0xff);
}

__device__ __forceinline__ int64_t stride(const u64 *d, int k) {
  return ((const int64_t *)d)[2 + k];
}

__device__ __forceinline__ int64_t hw(const u64 *w, int f) {
  return (int64_t)w[f];
}

/* Lead index [b]'s first position in the array of descriptor [d], whose
   first strides are the lead axes'. */
__device__ __noinline__ int64_t lead_at(const u64 *w, const u64 *d,
                                        int64_t b) {
  int nl = (int)w[NX_ATT_NLEAD];
  const u64 *ext = w + w[NX_ATT_LEAD];
  int64_t at = (int64_t)d[1];
  for (int x = nl - 1; x > 0; x--) {
    int64_t e = (int64_t)ext[x], q = b / e;
    at += (b - q * e) * stride(d, x);
    b = q;
  }
  return nl > 0 ? at + b * stride(d, 0) : at;
}

/* Lead index [b]'s coordinate along lead axis [a]. */
__device__ __noinline__ int64_t lead_coord(const u64 *w, int64_t b,
                                           int a) {
  int nl = (int)w[NX_ATT_NLEAD];
  const u64 *ext = w + w[NX_ATT_LEAD];
  for (int x = nl - 1; x > a; x--) b /= (int64_t)ext[x];
  return a == 0 ? b : b % (int64_t)ext[a];
}

/* Elements */

/* Element [i] of [x], of the float dtype [dt], widened to A. */
template <class A>
__device__ __forceinline__ A rd(const void *x, int64_t i, int dt) {
  switch (dt) {
  case NX_FLOAT16: return (A)nx_f16_to_float(((const uint16_t *)x)[i]);
  case NX_BFLOAT16: return (A)nx_bf16_to_float(((const uint16_t *)x)[i]);
  case NX_FLOAT32: return (A)((const float *)x)[i];
  default: return (A)((const double *)x)[i];
  }
}

/* [x] rounded once into [dt], at element [i] of [o]. */
__device__ __noinline__ void wr(void *o, int64_t i, int dt, float x) {
  switch (dt) {
  case NX_FLOAT16: ((uint16_t *)o)[i] = nx_float_to_f16(x); break;
  case NX_BFLOAT16: ((uint16_t *)o)[i] = nx_float_to_bf16(x); break;
  default: ((float *)o)[i] = x;
  }
}

__device__ __noinline__ void wr(void *o, int64_t i, int dt, double x) {
  switch (dt) {
  case NX_FLOAT16: ((uint16_t *)o)[i] = (uint16_t)nx_double_to_bits(NX_FLOAT16, x); break;
  case NX_BFLOAT16: ((uint16_t *)o)[i] = (uint16_t)nx_double_to_bits(NX_BFLOAT16, x); break;
  case NX_FLOAT32: ((float *)o)[i] = (float)x; break;
  default: ((double *)o)[i] = x;
  }
}

/* The greatest of a and b, a NaN where either is. */
template <class A> __device__ __forceinline__ A greatest(A a, A b) {
  if (a != a || b != b) return a + b;
  return b > a ? b : a;
}

__device__ __forceinline__ float exp_(float x) { return nx_exp_f32(x); }
__device__ __forceinline__ double exp_(double x) { return nx_exp_f64(x); }
__device__ __forceinline__ float log_(float x) { return nx_log_f32(x); }
__device__ __forceinline__ double log_(double x) { return nx_log_f64(x); }
__device__ __forceinline__ float fma_(float a, float b, float c) {
  return fmaf(a, b, c);
}
__device__ __forceinline__ double fma_(double a, double b, double c) {
  return fma(a, b, c);
}

/* Units */

/* A unit: its lead index, key head, first head and row, its heads and
   rows of a head, its rows in all, its lead index's keys, and the lead
   positions of q, k, v, the table, o and lse. */
typedef struct {
  int64_t b, kh, h0, i0, len;
  int nh, nt, rows;
  int64_t qa, ka, va, ta, oa, la;
} unit;

/* Unit [u], the heaviest rows first, as a causal mask leaves them. */
__device__ __noinline__ unit unit_of(const u64 *w, int64_t u) {
  unit c;
  int64_t tiles = hw(w, NX_ATT_TILES), chunks = hw(w, NX_ATT_CHUNKS);
  int64_t hk = hw(w, NX_ATT_HK), g = hw(w, NX_ATT_HQ) / hk;
  int64_t hg = hw(w, NX_ATT_HG), tr = hw(w, NX_ATT_TR), tq = hw(w, NX_ATT_TQ);
  int64_t tile = tiles - 1 - u % tiles, rest = u / tiles;
  int64_t chunk = rest % chunks;
  rest /= chunks;
  c.kh = rest % hk;
  c.b = rest / hk;
  c.h0 = c.kh * g + chunk * hg;
  c.i0 = tile * tr;
  c.nh = (int)(g - chunk * hg < hg ? g - chunk * hg : hg);
  c.nt = (int)(tq - c.i0 < tr ? tq - c.i0 : tr);
  c.rows = c.nh * c.nt;
  int64_t tk = hw(w, NX_ATT_TK), l = tk;
  if (w[NX_ATT_LENGTHS]) {
    const u64 *d = desc(w, NX_ATT_LENGTHS);
    l = ((const int64_t *)base(w, d))[lead_at(w, d, c.b)];
    l = l < 0 ? 0 : l > tk ? tk : l;
  }
  c.len = l;
  c.qa = lead_at(w, desc(w, NX_ATT_Q), c.b);
  c.oa = lead_at(w, desc(w, NX_ATT_O), c.b);
  c.la = lead_at(w, desc(w, NX_ATT_LSE), c.b);
  if (w[NX_ATT_PAGE] == 0) {
    c.ka = lead_at(w, desc(w, NX_ATT_K), c.b);
    c.va = lead_at(w, desc(w, NX_ATT_VALUES), c.b);
    c.ta = 0;
  } else {
    c.ka = (int64_t)desc(w, NX_ATT_K)[1];
    c.va = (int64_t)desc(w, NX_ATT_VALUES)[1];
    c.ta = lead_at(w, desc(w, NX_ATT_TABLE), c.b);
  }
  return c;
}

/* Key [t] of the unit: its first elements in k and v, or -1 where its
   table entry is outside [0, pages). */
__device__ __forceinline__ void key_at(const u64 *w, const unit &c,
                                    int64_t t, int64_t *ko, int64_t *vo) {
  const u64 *k = desc(w, NX_ATT_K), *v = desc(w, NX_ATT_VALUES);
  int64_t page = hw(w, NX_ATT_PAGE);
  if (page == 0) {
    int nl = (int)w[NX_ATT_NLEAD];
    *ko = c.ka + c.kh * stride(k, nl) + t * stride(k, nl + 1);
    *vo = c.va + c.kh * stride(v, nl) + t * stride(v, nl + 1);
    return;
  }
  const u64 *tb = desc(w, NX_ATT_TABLE);
  int nl = (int)w[NX_ATT_NLEAD];
  int64_t e = ((const int64_t *)base(w, tb))[c.ta + t / page * stride(tb, nl)];
  if (e < 0 || e >= hw(w, NX_ATT_PAGES)) {
    *ko = *vo = -1;
    return;
  }
  int64_t s = t % page;
  *ko = c.ka + e * stride(k, 0) + c.kh * stride(k, 1) + s * stride(k, 2);
  *vo = c.va + e * stride(v, 0) + c.kh * stride(v, 1) + s * stride(v, 2);
}

/* The dimension's stride of k or v. */
__device__ __forceinline__ int64_t dim_stride(const u64 *w, int f) {
  return stride(desc(w, f), w[NX_ATT_PAGE] ? 3 : (int)w[NX_ATT_NLEAD] + 2);
}

/* Tapes */

/* [x] held of [from] cast into [to], both of float32, float64, int64 and
   bool, as loop.cu's cast stores it. */
__device__ __forceinline__ held cast_in(int from, int to, held x) {
  if (from == to) return x;
  bool integer = from == NX_INT64 || from == NX_BOOL;
  int64_t s = from == NX_BOOL ? (int64_t)(x.lo != 0) : (int64_t)x.lo;
  double f = from == NX_FLOAT64 ? __longlong_as_double((long long)x.lo)
                                : (double)__uint_as_float((uint32_t)x.lo);
  held r = {0, 0};
  switch (to) {
  case NX_BOOL: r.lo = integer ? s != 0 : f != 0; break;
  case NX_FLOAT64:
    r.lo = (u64)__double_as_longlong(integer ? (double)s : f);
    break;
  case NX_FLOAT32: r.lo = __float_as_uint(integer ? (float)s : (float)f); break;
  default: r.lo = integer ? (u64)s : (u64)nx_double_to_bits(NX_INT64, f);
  }
  return r;
}

#define ATT_DO_1(S, T, n) put<T>(E, d, i, nx_##n##_##S(get<T>(E, a, i)))
#define ATT_DO_2(S, T, n)                                                      \
  put<T>(E, d, i, nx_##n##_##S(get<T>(E, a, i), get<T>(E, b, i)))
#define ATT_DO_C(S, T, n)                                                      \
  word(E, d, i) = nx_##n##_##S(get<T>(E, a, i), get<T>(E, b, i)) != 0
#define ATT_CASE(S, T, A, U, n)                                                \
  case NX_ATT_OP(S, A, U): ATT_DO_##A(S, T, n); break;

/* Runs the tape's [n] words from [from] over the thread's [V] elements of
   [E], an element at a time: the kinds NX_ATT_KINDS lists, constants,
   casts among float32, float64, int64 and bool, selections of one or two
   words and truths, each in line, as loop.cu runs them. The host lowers a
   program to no other instruction here, nor to a load, a coordinate or a
   store: the kernel gives their planes. */
__device__ __noinline__ void tape(const u64 *w, int E, int V, int from,
                                  int n) {
  using namespace nx_loop;
  for (int at = from; at < from + n; at++) {
    u64 x = w[at];
    int op = (int)(x & 0xffff), d = (int)((x >> 16) & 0xfff);
    int a = (int)((x >> 28) & 0xfff), b = (int)((x >> 40) & 0xfff);
    int c = (int)((x >> 52) & 0xfff);
    for (int v = 0; v < V; v++) {
      int i = v * AT + threadIdx.x;
      switch (op) {
        NX_ATT_KINDS(ATT_CASE)
      case NX_LOOP_CONST: put_held(E, d, i, b, held{w[at + 1], w[at + 2]}); break;
      case NX_LOOP_CAST:
        put_held(E, d, i, words_of(c), cast_in(b, c, get_held(E, a, i, words_of(b))));
        break;
      case NX_LOOP_WHERE1:
        word(E, d, i) = word(E, a, i) != 0 ? word(E, b, i) : word(E, c, i);
        break;
      case NX_LOOP_WHERE2:
        put64(E, d, i, word(E, a, i) != 0 ? get64(E, b, i) : get64(E, c, i));
        break;
      case NX_LOOP_TRUTH: word(E, d, i) = word(E, a, i) != 0; break;
      }
    }
    if (op == NX_LOOP_CONST) at += 2;
  }
}

#undef ATT_DO_1
#undef ATT_DO_2
#undef ATT_DO_C
#undef ATT_CASE

/* A thread's element slot [v] of a run of V elements a thread. */
__device__ __forceinline__ int slot(int v) { return v * AT + threadIdx.x; }

/* Load [k]'s element at head [h], row [i] and key [j] of the unit. */
__device__ __noinline__ held load_at(const u64 *w, int k, const unit &c,
                                     int64_t h, int64_t i, int64_t j) {
  const u64 *d = w + w[w[NX_ATT_LOADS] + k];
  int nl = (int)w[NX_ATT_NLEAD];
  int64_t at = lead_at(w, d, c.b) + h * stride(d, nl) + i * stride(d, nl + 1) +
               j * stride(d, nl + 2);
  return nx_loop::read(base(w, d), at, dtype(d));
}

/* The planes of the table at word [g] for the pair (h, i, j) of the unit
   at element [e] of a tape of E elements, with the score [x]. */
template <class A>
__device__ __noinline__ void give(const u64 *w, int g, int E, int e,
                                  const unit &c, int64_t h, int64_t i,
                                  int64_t j, A x) {
  int nl = (int)w[NX_ATT_NLEAD], n = (int)w[g];
  for (int k = 1; k <= n; k++) {
    u64 t = w[g + k];
    int kind = (int)(t & 0xff), x8 = (int)((t >> 8) & 0xffffff);
    int pl = (int)(t >> 32);
    switch (kind) {
    case NX_ATT_GIVE_SCORE: nx_loop::put<A>(E, pl, e, x); break;
    case NX_ATT_GIVE_COORD: {
      int64_t y = x8 < nl ? lead_coord(w, c.b, x8)
                  : x8 == nl ? h : x8 == nl + 1 ? i : j;
      nx_loop::put<int64_t>(E, pl, e, y);
      break;
    }
    case NX_ATT_GIVE_LOAD: {
      const u64 *d = w + w[w[NX_ATT_LOADS] + x8];
      nx_loop::put_held(E, pl, e, nx_loop::words_of(dtype(d)),
                        load_at(w, x8, c, h, i, j));
      break;
    }
    }
  }
}

/* Whether held [a] of the integer or boolean dtype [dt] comes before [b]
   in its order. */
__device__ __forceinline__ bool before(int dt, held a, held b) {
  if (!nx_loop::is_signed(dt)) return a.lo < b.lo;
  if (dt == NX_INT64) return (int64_t)a.lo < (int64_t)b.lo;
  return (int32_t)(uint32_t)a.lo < (int32_t)(uint32_t)b.lo;
}

/* The least or the greatest held value of the integer or boolean [dt]. */
__device__ held end_of(int dt, bool greatest) {
  int bits = nx_loop::bits_of(dt);
  held h = {0, 0};
  if (nx_loop::is_bool(dt)) h.lo = greatest;
  else if (!nx_loop::is_signed(dt))
    h.lo = !greatest ? 0 : bits == 64 ? ~0ull : (1ull << bits) - 1;
  else if (bits == 64) h.lo = greatest ? (uint64_t)INT64_MAX : (uint64_t)INT64_MIN;
  else {
    int32_t m = (int32_t)((1u << (bits - 1)) - 1);
    h.lo = (uint32_t)(greatest ? m : -m - 1);
  }
  return h;
}

/* Load [k]'s least or greatest element over the box of the unit's rows
   and the keys [j0, j0 + nk): over the positions its strides reach, or
   its dtype's end past NX_ATT_BOX of them. */
__device__ held load_end(const u64 *w, int k, const unit &c, int64_t j0,
                         int nk, bool greatest) {
  const u64 *d = w + w[w[NX_ATT_LOADS] + k];
  int nl = (int)w[NX_ATT_NLEAD], dt = dtype(d);
  int nh = stride(d, nl) ? c.nh : 1, nt = stride(d, nl + 1) ? c.nt : 1;
  int nj = stride(d, nl + 2) ? nk : 1;
  if ((int64_t)nh * nt * nj > NX_ATT_BOX) return end_of(dt, greatest);
  held best = end_of(dt, !greatest);
  for (int a = 0; a < nh; a++)
    for (int b = 0; b < nt; b++)
      for (int e = 0; e < nj; e++) {
        held x = load_at(w, k, c, c.h0 + a, c.i0 + b, j0 + e);
        if (greatest ? before(dt, best, x) : before(dt, x, best)) best = x;
      }
  return best;
}

/* The ranges' planes at element [e] for the box of the unit's rows and
   the keys [j0, j0 + nk). */
__device__ __noinline__ void give_box(const u64 *w, int E, int e,
                                      const unit &c, int64_t j0, int nk) {
  int g = (int)w[NX_ATT_RANGES_GIVEN];
  int nl = (int)w[NX_ATT_NLEAD], n = (int)w[g];
  for (int k = 1; k <= n; k++) {
    u64 t = w[g + k];
    int kind = (int)(t & 0xff), x8 = (int)((t >> 8) & 0xffffff);
    int pl = (int)(t >> 32);
    bool hi = kind == NX_ATT_GIVE_COORD_HI || kind == NX_ATT_GIVE_LOAD_HI;
    if (kind == NX_ATT_GIVE_COORD_LO || kind == NX_ATT_GIVE_COORD_HI) {
      int64_t y = x8 < nl       ? lead_coord(w, c.b, x8)
                  : x8 == nl     ? c.h0 + (hi ? c.nh - 1 : 0)
                  : x8 == nl + 1 ? c.i0 + (hi ? c.nt - 1 : 0)
                                 : j0 + (hi ? nk - 1 : 0);
      nx_loop::put<int64_t>(E, pl, e, y);
    } else {
      const u64 *d = w + w[w[NX_ATT_LOADS] + x8];
      nx_loop::put_held(E, pl, e, nx_loop::words_of(dtype(d)),
                        load_end(w, x8, c, j0, nk, hi));
    }
  }
}

/* The classes of the unit's blocks [c0, c1), at most NX_ATT_CLASS_RUN of
   them, into CLS: output 0's ranges over each block's box, a block an
   element of the tape. Every thread calls it. */
__device__ __noinline__ void classes(const u64 *w, const unit &c,
                                     int64_t c0, int64_t c1, uint8_t *CLS) {
  const int E = NX_ATT_CLASS_RUN, e = threadIdx.x;
  __syncthreads(); /* the previous chunk's readers of CLS are done, and the
                      planes overlap the FFMA staging the previous block
                      read */
  if (e < E) {
    int64_t kb = c0 + e, j0 = kb * KEYS;
    if (kb < c1)
      give_box(w, E, e, c, j0, (int)(c.len - j0 < KEYS ? c.len - j0 : KEYS));
    tape(w, E, 1, (int)w[NX_ATT_RANGES_AT], (int)w[NX_ATT_RANGES_N]);
    int lo = (int)w[NX_ATT_RANGES_LO], hi = (int)w[NX_ATT_RANGES_HI];
    if (kb < c1) CLS[e] = word(E, lo, e) ? SHOWN : word(E, hi, e) ? MIXED : HIDDEN;
  }
  __syncthreads();
}

/* Block [kb]'s class: its ranges' from CLS, computed NX_ATT_CLASS_RUN
   blocks at a time from [first], or, without ranges, mixed where output 0
   runs and shown where it is true everywhere. */
__device__ __forceinline__ int class_of(const u64 *w, const unit &c,
                                        int64_t first, int64_t last,
                                        int64_t kb, uint8_t *CLS) {
  u64 f = w[NX_ATT_FLAGS];
  if (!(f & NX_ATT_RANGES)) return f & NX_ATT_MASK ? MIXED : SHOWN;
  const int64_t E = NX_ATT_CLASS_RUN;
  int64_t at = kb - first, c0 = kb - at % E;
  if (at % E == 0) classes(w, c, c0, c0 + E < last ? c0 + E : last, CLS);
  return CLS[at % E];
}

/* Output 0 on the pairs of the block [j0, j0 + nk) of the unit whose keys
   are present, into VIS. Every thread of the group calls it. */
template <int RT>
__device__ __noinline__ void mask_vis(const u64 *w, const unit &c,
                                      int64_t j0, int nk,
                                      unsigned long long *VIS,
                                      const int64_t *KO) {
  int V = (int)w[NX_ATT_V], E = V * AT, out = (int)w[NX_ATT_MASK_OUT];
  int g = (int)w[NX_ATT_MASK_GIVEN];
  for (int e0 = 0; e0 < RT * nk; e0 += E) {
    for (int v = 0; v < V; v++) {
      int e = e0 + slot(v), r = e % RT, j = e / RT;
      if (r >= c.rows || j >= nk || KO[j] < 0) continue;
      give<float>(w, g, E, slot(v), c, c.h0 + r / c.nt, c.i0 + r % c.nt,
                  j0 + j, 0.f);
    }
    tape(w, E, V, (int)w[NX_ATT_MASK_AT], (int)w[NX_ATT_MASK_N]);
    for (int v = 0; v < V; v++) {
      int e = e0 + slot(v), r = e % RT, j = e / RT;
      if (r >= c.rows || j >= nk || KO[j] < 0) continue;
      if (word(E, out, slot(v))) atomicOr(&VIS[r], 1ull << j);
    }
  }
}

/* Block [j0, j0 + nk) of the unit for its rows: KO, VO its keys'
   offsets (-1 where absent), VIS each row's visible keys, by output 0
   where the block is mixed. False where no row sees a key: the caller
   skips the block. Every thread of the group calls it. In line, so that
   a shown block makes no call around which its caller's registers would
   go to the stack. Where [fresh], no thread reads KO, VO or VIS until a
   barrier past the call. */
template <int RT, bool fresh>
__device__ __forceinline__ bool block_vis(const u64 *w, const unit &c,
                                          int cls, int64_t j0, int nk,
                                          unsigned long long *VIS,
                                          int64_t *KO, int64_t *VO) {
  const int tid = threadIdx.x, lane = tid % 32;
  if (cls == HIDDEN) return false;
  if (cls == SHOWN) {
    /* Each warp finds the keys present: a row sees each of them. */
    int64_t k0 = -1, v0 = -1, k1 = -1, v1 = -1;
    if (lane < nk) key_at(w, c, j0 + lane, &k0, &v0);
    if (32 + lane < nk) key_at(w, c, j0 + 32 + lane, &k1, &v1);
    unsigned long long m = __ballot_sync(0xffffffff, k0 >= 0) |
                           (unsigned long long)__ballot_sync(0xffffffff, k1 >= 0) << 32;
    if (!fresh) __syncthreads(); /* the previous block's readers are done */
    if (tid < 32) {
      KO[lane] = k0, VO[lane] = v0;
      KO[32 + lane] = k1, VO[32 + lane] = v1;
    }
    if (tid < RT) VIS[tid] = tid < c.rows ? m : 0;
    __syncthreads();
    return m != 0;
  }
  if (!fresh) __syncthreads(); /* the previous block's readers are done */
  if (tid < KEYS) {
    int64_t ko = -1, vo = -1;
    if (tid < nk) key_at(w, c, j0 + tid, &ko, &vo);
    KO[tid] = ko;
    VO[tid] = vo;
  }
  if (tid < RT) VIS[tid] = 0;
  __syncthreads();
  mask_vis<RT>(w, c, j0, nk, VIS, KO);
  __syncthreads();
  int any = tid < RT && VIS[tid] != 0;
  return __syncthreads_or(any);
}

/* Output 1 of the pairs of a thread's [n] scores x[], at most
   NX_ATT_MOST_V, at element slots 0 to n - 1 of rows rr[] and keys jj[],
   in place: one run of the tape. A pair of key -1 takes no part. In line,
   its slots unrolled, so that x[] stays in its caller's registers. */
template <class A>
__device__ __forceinline__ void rescore(const u64 *w, const unit &c, int n,
                                        A *x, const int *rr,
                                        const int64_t *jj) {
  int V = (int)w[NX_ATT_V], E = V * AT, out = (int)w[NX_ATT_SCORE_OUT];
  int g = (int)w[NX_ATT_SCORE_GIVEN];
#pragma unroll
  for (int v = 0; v < NX_ATT_MOST_V; v++)
    if (v < n && jj[v] >= 0)
      give<A>(w, g, E, slot(v), c, c.h0 + rr[v] / c.nt, c.i0 + rr[v] % c.nt,
              jj[v], x[v]);
  tape(w, E, n, (int)w[NX_ATT_SCORE_AT], (int)w[NX_ATT_SCORE_N]);
#pragma unroll
  for (int v = 0; v < NX_ATT_MOST_V; v++)
    if (v < n && jj[v] >= 0) x[v] = nx_loop::get<A>(E, out, slot(v));
}

/* The FFMA kernel: nx.cpu's arithmetic, bit for bit, on tiles.

   Thread (ty, tx) holds rows ty RPT .. + RPT of both S and o, keys
   4 tx .. 4 tx + 3 of S, and columns tx DPT .. + DPT of o. Each pair's
   arithmetic is nx.cpu's, so a row's bits are nx.cpu's where the keys are
   one part; the tile changes no bit. Staging goes in two halves: raw
   loads into registers, issued a chunk ahead so that they are in flight
   while the group computes, then the widening stores into shared memory.
   R holds an element's bits: 32 bits under a float32 accumulator, 64
   under float64. */

template <class R, int W>
__device__ __forceinline__ R raw_at(const void *x, int64_t i) {
  if (W == 2) return ((const uint16_t *)x)[i];
  if (W == 4) return ((const uint32_t *)x)[i];
  return (R)((const unsigned long long *)x)[i];
}

template <class A, int DT, class R> __device__ __forceinline__ A widen(R r) {
  if (DT == NX_FLOAT16) return (A)nx_f16_to_float((uint16_t)r);
  if (DT == NX_BFLOAT16) return (A)nx_bf16_to_float((uint16_t)r);
  if (DT == NX_FLOAT32) return (A)__uint_as_float((uint32_t)r);
  return (A)__longlong_as_double((long long)r);
}

template <class R, int DT> __device__ __forceinline__ bool finite_raw(R r) {
  if (DT == NX_FLOAT16) return (r & 0x7c00) != 0x7c00;
  if (DT == NX_BFLOAT16) return (r & 0x7f80) != 0x7f80;
  if (DT == NX_FLOAT32) return (r & 0x7f800000u) != 0x7f800000u;
  return ((unsigned long long)r & 0x7ff0000000000000ull) !=
         0x7ff0000000000000ull;
}

__device__ __forceinline__ bool narrow(int dt) {
  return dt == NX_FLOAT16 || dt == NX_BFLOAT16;
}

/* A chunk of rows ro[] (element offsets, -1 absent) of x, dimensions
   [dc, dc + nd) of stride sd: element it of this thread is dimension
   8 (y % 4) + lane % 8 of row 4 (y / 4) + lane / 8, y = warp + 8 it, so
   that a warp's shared stores fall in 32 banks: the thread's dimension
   t_dim and its rows t_row + 8 it. */
__device__ __forceinline__ int t_dim() {
  return (threadIdx.x / 32 % 4) * 8 + threadIdx.x % 8;
}

__device__ __forceinline__ int t_row() {
  return (threadIdx.x / 128) * 4 + threadIdx.x % 32 / 8;
}

template <class R, int W, int NI>
__device__ __forceinline__ void ld_t(R (&raw)[NI], const void *x,
                                     const int64_t *ro, int64_t sd, int64_t dc,
                                     int nd) {
  const int d = t_dim(), r = t_row();
  const bool in = d < nd;
  const int64_t at = (dc + d) * sd;
#pragma unroll
  for (int it = 0; it < NI; it++) {
    int64_t a = ro[r + 8 * it];
    raw[it] = a >= 0 && in ? raw_at<R, W>(x, a + at) : (R)0;
  }
}

template <class R, int NI>
__device__ __forceinline__ void ld_tw(R (&raw)[NI], const void *x,
                                      const int64_t *ro, int64_t sd,
                                      int64_t dc, int nd, int dt) {
  if (narrow(dt)) ld_t<R, 2, NI>(raw, x, ro, sd, dc, nd);
  else if (dt == NX_FLOAT32 || sizeof(R) == 4) ld_t<R, 4, NI>(raw, x, ro, sd, dc, nd);
  else ld_t<R, 8, NI>(raw, x, ro, sd, dc, nd);
}

template <class A, int DT, class R, int NI>
__device__ __forceinline__ void st_t(A *T, int TS, const R (&raw)[NI]) {
  A *t = T + t_dim() * TS + t_row();
#pragma unroll
  for (int it = 0; it < NI; it++) t[8 * it] = widen<A, DT>(raw[it]);
}

template <class A, class R, int NI>
__device__ __forceinline__ void st_tw(A *T, int TS, const R (&raw)[NI],
                                      int dt) {
  switch (dt) {
  case NX_FLOAT16: st_t<A, NX_FLOAT16>(T, TS, raw); break;
  case NX_BFLOAT16: st_t<A, NX_BFLOAT16>(T, TS, raw); break;
  case NX_FLOAT32: st_t<A, NX_FLOAT32>(T, TS, raw); break;
  default:
    if constexpr (sizeof(A) == 8) st_t<A, NX_FLOAT64>(T, TS, raw);
  }
}

/* A chunk of keys vo[0..nv) of v, columns [c0, c0 + DVC) below dv:
   element it of this thread is e = tid + 256 it, key e / DVC, column
   e % DVC, the thread's column for each; zeros elsewhere. */
template <class R, int W, int NI, int DVC>
__device__ __forceinline__ void ld_v(R (&raw)[NI], const void *x,
                                     const int64_t *vo, int nv, int64_t sd,
                                     int64_t c0, int64_t dv) {
  static_assert(AT % DVC == 0, "a thread's column is one");
  const int c = threadIdx.x % DVC, j = threadIdx.x / DVC;
  const bool in = c0 + c < dv;
  const int64_t at = (c0 + c) * sd;
#pragma unroll
  for (int it = 0; it < NI; it++) {
    int jj = j + AT / DVC * it;
    int64_t a = jj < nv ? vo[jj] : -1;
    raw[it] = a >= 0 && in ? raw_at<R, W>(x, a + at) : (R)0;
  }
}

template <class R, int NI, int DVC>
__device__ __forceinline__ void ld_vw(R (&raw)[NI], const void *x,
                                      const int64_t *vo, int nv, int64_t sd,
                                      int64_t c0, int64_t dv, int dt) {
  if (narrow(dt)) ld_v<R, 2, NI, DVC>(raw, x, vo, nv, sd, c0, dv);
  else if (dt == NX_FLOAT32 || sizeof(R) == 4) ld_v<R, 4, NI, DVC>(raw, x, vo, nv, sd, c0, dv);
  else ld_v<R, 8, NI, DVC>(raw, x, vo, nv, sd, c0, dv);
}

/* The stores; whether an element is not finite. */
template <class A, int DT, class R, int NI>
__device__ __forceinline__ int st_v(A *V, const R (&raw)[NI]) {
  int bad = 0;
#pragma unroll
  for (int it = 0; it < NI; it++) {
    bad |= !finite_raw<R, DT>(raw[it]);
    V[threadIdx.x + AT * it] = widen<A, DT>(raw[it]);
  }
  return bad;
}

template <class A, class R, int NI>
__device__ __forceinline__ int st_vw(A *V, const R (&raw)[NI], int dt) {
  switch (dt) {
  case NX_FLOAT16: return st_v<A, NX_FLOAT16>(V, raw);
  case NX_BFLOAT16: return st_v<A, NX_BFLOAT16>(V, raw);
  case NX_FLOAT32: return st_v<A, NX_FLOAT32>(V, raw);
  default:
    if constexpr (sizeof(A) == 8) return st_v<A, NX_FLOAT64>(V, raw);
    return 0;
  }
}

/* N consecutive A's from shared memory, by 16-byte loads where N fills
   them (the callers align them). */
template <class A, int N>
__device__ __forceinline__ void ld(const A *x, A (&y)[N]) {
  if (sizeof(A) * N % 16 == 0) {
#pragma unroll
    for (int i = 0; i < N; i += 16 / (int)sizeof(A)) {
      if (sizeof(A) == 4) {
        float4 v = *(const float4 *)(x + i);
        ((float *)y)[i] = v.x, ((float *)y)[i + 1] = v.y;
        ((float *)y)[i + 2] = v.z, ((float *)y)[i + 3] = v.w;
      } else {
        double2 v = *(const double2 *)(x + i);
        ((double *)y)[i] = v.x, ((double *)y)[i + 1] = v.y;
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < N; i++) y[i] = x[i];
  }
}

/* One dimension of a thread's scores: RPT rows by 4 keys, fused. */
template <class A, int RPT>
__device__ __forceinline__ void dot_step(const A *q, const A *k,
                                         A (&s)[RPT][4]) {
  A qv[RPT], kv[4];
  ld<A, RPT>(q, qv);
  ld<A, 4>(k, kv);
#pragma unroll
  for (int r = 0; r < RPT; r++)
#pragma unroll
    for (int x = 0; x < 4; x++) s[r][x] = fma_(qv[r], kv[x], s[r][x]);
}

/* One key of a thread's o: RPT rows by DPT columns, fused. */
template <class A, int RPT, int DPT>
__device__ __forceinline__ void pv_step(const A *pr, const A *vr,
                                        A (&o)[RPT][DPT]) {
  A pv[RPT], vv[DPT];
  ld<A, RPT>(pr, pv);
  ld<A, DPT>(vr, vv);
#pragma unroll
  for (int r = 0; r < RPT; r++)
#pragma unroll
    for (int x = 0; x < DPT; x++) o[r][x] = fma_(pv[r], vv[x], o[r][x]);
}

template <class A, int RT, int DVC> struct Simt {
  static constexpr int RPT = RT / 16, DPT = DVC / 16, DKC = 32, VKC = 32;
  static constexpr int PAD = sizeof(A) == 4 ? 4 : 2;
  static constexpr int QS = RT + PAD, KS = KEYS + PAD, PS = RT + PAD;
  static constexpr int G = AT / RT; /* threads a row in the softmax */
  /* Shared memory, in A's: the staging region (Qc and Kc, or Vc), P,
     the rows' m, l, alpha, the greatest per thread of a row, the lanes. */
  static constexpr int STAGE = (DKC * QS + DKC * KS) > (VKC * DVC)
                                   ? (DKC * QS + DKC * KS)
                                   : (VKC * DVC);
  static constexpr int OFF_P = STAGE, OFF_M = OFF_P + KEYS * PS;
  static constexpr int OFF_L = OFF_M + RT, OFF_AL = OFF_L + RT;
  static constexpr int OFF_MX = OFF_AL + RT, OFF_RED = OFF_MX + RT;
  static constexpr int OFF_LANE = OFF_RED + G * RT, N_A = OFF_LANE + 8 * RT;
};

/* Where the unit's row [r] puts its columns of o: the array, the element
   of column 0 and the step between columns, and o's dtype, or -1 for this
   part's state in the workspace, the row's a. Once a row, so that its
   columns cost a store each. */
typedef struct {
  void *x;
  int64_t at, step;
  int dt;
} row_dst;

template <class A>
__device__ __noinline__ row_dst dst_of(const u64 *w, const unit &c, int r) {
  int64_t h = c.h0 + r / c.nt, i = c.i0 + r % c.nt;
  if (w[NX_ATT_PARTS] > 1) {
    int64_t hq = hw(w, NX_ATT_HQ), tq = hw(w, NX_ATT_TQ);
    int64_t rows = hw(w, NX_ATT_B) * hq * tq;
    int64_t R = (int64_t)blockIdx.z * rows + (c.b * hq + h) * tq + i;
    return {(void *)w[NX_ATT_HEADER + w[NX_ATT_WORK]],
            R * (hw(w, NX_ATT_DV) + 2) + 2, 1, -1};
  }
  const u64 *o = desc(w, NX_ATT_O);
  int nl = (int)w[NX_ATT_NLEAD];
  return {(void *)base(w, o),
          c.oa + h * stride(o, nl) + i * stride(o, nl + 1), stride(o, nl + 2),
          (int)((w[NX_ATT_DT] >> 8) & 0xff)};
}

/* The row's result at column [col] of o, from a and l, or a. */
template <class A>
__device__ __forceinline__ void put_o(const row_dst &d, int64_t col, A a,
                                      A l) {
  if (d.dt < 0) ((A *)d.x)[d.at + col] = a;
  else wr(d.x, d.at + col * d.step, d.dt, l == 0 ? (A)0 : a / l + (A)0);
}

template <class A>
__device__ __forceinline__ void put_lse(const u64 *w, const unit &c,
                                        int r, A m, A l) {
  int64_t R = (c.h0 + r / c.nt) * hw(w, NX_ATT_TQ) + c.i0 + r % c.nt;
  if (w[NX_ATT_PARTS] > 1) {
    int64_t rows = hw(w, NX_ATT_B) * hw(w, NX_ATT_HQ) * hw(w, NX_ATT_TQ);
    int64_t dv = hw(w, NX_ATT_DV);
    A *ws = (A *)w[NX_ATT_HEADER + w[NX_ATT_WORK]] +
            ((int64_t)blockIdx.z * rows + c.b * hw(w, NX_ATT_HQ) * hw(w, NX_ATT_TQ) + R) * (dv + 2);
    ws[0] = m;
    ws[1] = l;
    return;
  }
  const u64 *d = desc(w, NX_ATT_LSE);
  int nl = (int)w[NX_ATT_NLEAD];
  int64_t at = c.la + (c.h0 + r / c.nt) * stride(d, nl) +
               (c.i0 + r % c.nt) * stride(d, nl + 1);
  ((A *)base(w, d))[at] = l == 0 ? (A)-INFINITY : m == (A)INFINITY ? m : m + log_(l);
}

/* The element of the unit's row [r] and key [j] in the scores
   (NX_ATT_SCORES). */
__device__ __forceinline__ int64_t score_at(const u64 *w, const unit &c,
                                            int r, int64_t j) {
  int64_t hq = hw(w, NX_ATT_HQ), tq = hw(w, NX_ATT_TQ);
  return ((c.b * hq + c.h0 + r / c.nt) * tq + c.i0 + r % c.nt) *
             hw(w, NX_ATT_TK) + j;
}

/* The keys' blocks of the group's part: [*kb0, *kb1). */
__device__ __forceinline__ void part_blocks(const u64 *w, int64_t *kb0,
                                            int64_t *kb1) {
  int64_t nblk = (hw(w, NX_ATT_TK) + KEYS - 1) / KEYS;
  *kb0 = w[NX_ATT_PARTS] > 1 ? blockIdx.z * hw(w, NX_ATT_PBLOCKS) : 0;
  *kb1 = w[NX_ATT_PARTS] > 1 ? *kb0 + hw(w, NX_ATT_PBLOCKS) : nblk;
}

template <class A, int RT, int DVC>
__device__ __noinline__ void simt(const u64 *w) {
  typedef Simt<A, RT, DVC> S;
  typedef typename std::conditional<sizeof(A) == 8, unsigned long long,
                                    uint32_t>::type R;
  constexpr int RPT = S::RPT, DPT = S::DPT, DKC = S::DKC, VKC = S::VKC;
  constexpr int QS = S::QS, KS = S::KS, PS = S::PS, G = S::G;
  extern __shared__ __align__(16) unsigned char smem[];
  int V = (int)w[NX_ATT_V];
  A *sa = (A *)(smem + (size_t)w[NX_ATT_SHARED]);
  A *Qc = sa, *Kc = sa + DKC * QS, *Vc = sa;
  A *P = sa + S::OFF_P, *M = sa + S::OFF_M, *L = sa + S::OFF_L;
  A *AL = sa + S::OFF_AL, *MX = sa + S::OFF_MX, *RED = sa + S::OFF_RED;
  A *LANE = sa + S::OFF_LANE;
  unsigned long long *VIS = (unsigned long long *)(sa + S::N_A);
  int64_t *KO = (int64_t *)(VIS + RT), *VO = KO + KEYS;
  int64_t *ROWQ = VO + KEYS;
  uint8_t *CLS = (uint8_t *)(ROWQ + RT);

  const int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
  const int dt = (int)(w[NX_ATT_DT] & 0xff);
  const int64_t dk = hw(w, NX_ATT_DK), dv = hw(w, NX_ATT_DV);
  const int64_t ksd = dim_stride(w, NX_ATT_K), vsd = dim_stride(w, NX_ATT_VALUES);
  const u64 *qd = desc(w, NX_ATT_Q);
  const int nl = (int)w[NX_ATT_NLEAD];
  const void *qx = base(w, qd), *kx = base(w, desc(w, NX_ATT_K));
  const void *vx = base(w, desc(w, NX_ATT_VALUES));
  const int64_t qsd = stride(qd, nl + 2);
  const bool rescores = w[NX_ATT_FLAGS] & NX_ATT_SCORE;
  const bool scores = w[NX_ATT_FLAGS] & NX_ATT_SCORES;
  A *const SC = (A *)w[NX_ATT_HEADER + w[NX_ATT_WORK]];
  int64_t kb0, kb1;
  part_blocks(w, &kb0, &kb1);
  const A scale = (A)__longlong_as_double((long long)w[NX_ATT_SCALE]);
  int64_t ncols = (dv + DVC - 1) / DVC;
  ncols = ncols > 0 ? ncols : 1;

  for (int64_t cb = blockIdx.y; cb < ncols; cb += gridDim.y)
  for (int64_t u = blockIdx.x; u < hw(w, NX_ATT_UNITS); u += gridDim.x) {
    const int64_t cbase = cb * DVC;
    unit c = unit_of(w, u);
    __syncthreads(); /* the previous unit's readers are done */
    if (tid < RT) {
      M[tid] = -INFINITY;
      L[tid] = 0;
      ROWQ[tid] = -1;
      if (tid < c.rows)
        ROWQ[tid] = c.qa + (c.h0 + tid / c.nt) * stride(qd, nl) +
                    (c.i0 + tid % c.nt) * stride(qd, nl + 1);
    }
    A o[RPT][DPT];
#pragma unroll
    for (int rr = 0; rr < RPT; rr++)
#pragma unroll
      for (int x = 0; x < DPT; x++) o[rr][x] = 0;

    int64_t nb = (c.len + KEYS - 1) / KEYS;
    int64_t e1 = kb1 < nb ? kb1 : nb;
    for (int64_t kb = kb0; kb < e1; kb++) {
      int64_t j0 = kb * KEYS;
      int nk = (int)(c.len - j0 < KEYS ? c.len - j0 : KEYS);
      int cls = class_of(w, c, kb0, e1, kb, CLS);
      if (!block_vis<RT, false>(w, c, cls, j0, nk, VIS, KO, VO)) continue;

      /* S = q k^T: chunks of DKC dimensions, staged widened and
         transposed. */
      A s[RPT][4];
#pragma unroll
      for (int rr = 0; rr < RPT; rr++)
#pragma unroll
        for (int x = 0; x < 4; x++) s[rr][x] = 0;
      R qr[RT / 8], kr[KEYS / 8];
      if (dk > 0) {
        int nd = (int)(dk < DKC ? dk : DKC);
        ld_tw(qr, qx, ROWQ, qsd, 0, nd, dt);
        ld_tw(kr, kx, KO, ksd, 0, nd, dt);
      }
      for (int64_t dc = 0; dc < dk; dc += DKC) {
        int nd = (int)(dk - dc < DKC ? dk - dc : DKC);
        __syncthreads();
        st_tw(Qc, QS, qr, dt);
        st_tw(Kc, KS, kr, dt);
        __syncthreads();
        if (dc + DKC < dk) {
          int nn = (int)(dk - dc - DKC < DKC ? dk - dc - DKC : DKC);
          ld_tw(qr, qx, ROWQ, qsd, dc + DKC, nn, dt);
          ld_tw(kr, kx, KO, ksd, dc + DKC, nn, dt);
        }
        if (nd == DKC) {
#pragma unroll 2
          for (int d = 0; d < DKC; d++)
            dot_step<A, RPT>(Qc + d * QS + ty * RPT, Kc + d * KS + tx * 4, s);
        } else {
#pragma unroll 1
          for (int d = 0; d < nd; d++)
            dot_step<A, RPT>(Qc + d * QS + ty * RPT, Kc + d * KS + tx * 4, s);
        }
      }
#pragma unroll
      for (int rr = 0; rr < RPT; rr++)
#pragma unroll
        for (int x = 0; x < 4; x++)
          P[(tx * 4 + x) * PS + ty * RPT + rr] = s[rr][x] * scale;
      __syncthreads();
      if (scores) {
        /* The block's present keys' scores, a row's keys by consecutive
           threads. */
        for (int e = tid; e < RT * KEYS; e += AT) {
          int r = e / KEYS, j = e % KEYS;
          if (r < c.rows && (VIS[r] >> j) & 1)
            SC[score_at(w, c, r, j0 + j)] = P[j * PS + r];
        }
        continue;
      }

      /* Phase A: output 1 on the visible pairs, V of a thread's at once,
         hidden pairs -inf, each thread's greatest. */
      {
        int r = tid % RT, k = tid / RT;
        unsigned long long vis = r < c.rows ? VIS[r] : 0;
        if (rescores)
          for (int j1 = k; j1 < KEYS; j1 += G * V) {
            A x[NX_ATT_MOST_V] = {};
            int rr[NX_ATT_MOST_V] = {};
            int64_t jj[NX_ATT_MOST_V];
#pragma unroll
            for (int v = 0; v < NX_ATT_MOST_V; v++) {
              int j = j1 + G * v;
              bool on = v < V && j < nk && ((vis >> j) & 1);
              x[v] = on ? P[j * PS + r] : (A)0;
              rr[v] = r;
              jj[v] = on ? j0 + j : -1;
            }
            rescore<A>(w, c, V, x, rr, jj);
#pragma unroll
            for (int v = 0; v < NX_ATT_MOST_V; v++)
              if (jj[v] >= 0) P[(j1 + G * v) * PS + r] = x[v];
          }
        A mx = -INFINITY;
        if (vis)
          for (int j = k; j < nk; j += G) {
            A x = (vis >> j) & 1 ? P[j * PS + r] : (A)-INFINITY;
            P[j * PS + r] = x;
            mx = greatest(mx, x);
          }
        RED[k * RT + r] = mx;
      }
      /* The first chunk of v in flight through the softmax, issued past
         output 1, whose run holds the most registers. */
      R vr[VKC * DVC / AT];
      ld_vw<R, VKC * DVC / AT, DVC>(vr, vx, VO, nk < VKC ? nk : VKC, vsd, cbase, dv, dt);
      __syncthreads();
      if (tid < RT && tid < c.rows && VIS[tid]) {
        A mx = M[tid];
        for (int k = 0; k < G; k++) mx = greatest(mx, RED[k * RT + tid]);
        MX[tid] = mx;
      }
      __syncthreads();
      /* Phase B: the weights, 0 where the greatest is -inf or the row sees
         no key. */
      {
        int r = tid % RT, k = tid / RT;
        bool on = r < c.rows && VIS[r] != 0 && MX[r] != (A)-INFINITY;
        A mx = on ? MX[r] : (A)0;
        for (int j = k; j < nk; j += G)
          P[j * PS + r] = on ? exp_(P[j * PS + r] - mx) : (A)0;
      }
      __syncthreads();
      /* Phase C: lane i sums terms i, i + 8, ... from +0. */
      {
        int r = tid % RT, k = tid / RT;
        if (r < c.rows && VIS[r] != 0)
          for (int i = k; i < 8; i += G) {
            A t = 0;
            for (int j = i; j < nk; j += 8) t = t + P[j * PS + r];
            LANE[r * 8 + i] = t;
          }
      }
      __syncthreads();
      /* Phase D: the lanes' tree, l, m and alpha. */
      if (tid < RT) {
        A al = 1;
        if (tid < c.rows && VIS[tid] != 0 && MX[tid] != (A)-INFINITY) {
          A *ln = LANE + tid * 8, mx = MX[tid];
          for (int hh = 4; hh > 0; hh /= 2)
            for (int i = 0; i < hh; i++) ln[i] = ln[i] + ln[i + hh];
          al = exp_(M[tid] - mx);
          A t = L[tid] * al;
          L[tid] = t + ln[0];
          M[tid] = mx;
        }
        AL[tid] = al;
      }
      __syncthreads();

      /* o = alpha o, then the products by chunks of VKC keys. A chunk
         holding a value not finite takes the pairs its rows see alone. */
#pragma unroll
      for (int rr = 0; rr < RPT; rr++) {
        A al = AL[ty * RPT + rr];
        if (al != 1)
#pragma unroll
          for (int x = 0; x < DPT; x++) o[rr][x] = o[rr][x] * al;
      }
      for (int vk = 0; vk < nk; vk += VKC) {
        int nv = nk - vk < VKC ? nk - vk : VKC;
        __syncthreads();
        int bad = st_vw(Vc, vr, dt);
        bad = __syncthreads_or(bad);
        if (vk + VKC < nk) {
          int nn = nk - vk - VKC < VKC ? nk - vk - VKC : VKC;
          ld_vw<R, VKC * DVC / AT, DVC>(vr, vx, VO + vk + VKC, nn, vsd, cbase, dv, dt);
        }
        if (!bad && nv == VKC) {
#pragma unroll 2
          for (int jj = 0; jj < VKC; jj++)
            pv_step<A, RPT, DPT>(P + (vk + jj) * PS + ty * RPT, Vc + jj * DVC + tx * DPT, o);
        } else if (!bad) {
#pragma unroll 1
          for (int jj = 0; jj < nv; jj++)
            pv_step<A, RPT, DPT>(P + (vk + jj) * PS + ty * RPT, Vc + jj * DVC + tx * DPT, o);
        } else {
          for (int jj = 0; jj < nv; jj++) {
            const A *pr = P + (vk + jj) * PS + ty * RPT, *vv = Vc + jj * DVC + tx * DPT;
#pragma unroll
            for (int rr = 0; rr < RPT; rr++) {
              if (!((VIS[ty * RPT + rr] >> (vk + jj)) & 1)) continue;
#pragma unroll
              for (int x = 0; x < DPT; x++) o[rr][x] = fma_(pr[rr], vv[x], o[rr][x]);
            }
          }
        }
      }
    }

    if (scores) continue;
    /* The rows' results: o and lse, or this part's state. One loop over
       the thread's elements, out of registers. */
    __syncthreads();
    A ol[RPT * DPT];
#pragma unroll
    for (int rr = 0; rr < RPT; rr++)
#pragma unroll
      for (int x = 0; x < DPT; x++) ol[rr * DPT + x] = o[rr][x];
#pragma unroll 1
    for (int rr = 0; rr < RPT; rr++) {
      int r = ty * RPT + rr;
      if (r >= c.rows) continue;
      row_dst d = dst_of<A>(w, c, r);
#pragma unroll 1
      for (int x = 0; x < DPT; x++) {
        int64_t col = cbase + tx * DPT + x;
        if (col < dv) put_o<A>(d, col, ol[rr * DPT + x], L[r]);
      }
    }
    if (tid < c.rows && cb == 0) put_lse<A>(w, c, tid, M[tid], L[tid]);
  }
}

/* The parts' states merged in order, a thread an element of o (and the
   row's lse at column dv): mx = max, α = exp (m - mx), β = exp (m_p - mx),
   l = l α + l_p β, a = fma (β, a_p, α a), a sum where mx = -inf. */
template <class A> __device__ void merge(const u64 *w) {
  int64_t hq = hw(w, NX_ATT_HQ), tq = hw(w, NX_ATT_TQ), dv = hw(w, NX_ATT_DV);
  int64_t rows = hw(w, NX_ATT_B) * hq * tq, width = dv + 1;
  int parts = (int)w[NX_ATT_PARTS], nl = (int)w[NX_ATT_NLEAD];
  int odt = (int)((w[NX_ATT_DT] >> 8) & 0xff);
  const A *ws = (const A *)w[NX_ATT_HEADER + w[NX_ATT_WORK]];
  const u64 *od = desc(w, NX_ATT_O), *ld = desc(w, NX_ATT_LSE);
  int64_t step = (int64_t)gridDim.x * blockDim.x;
  for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
       t < rows * width; t += step) {
    int64_t R = t / width, col = t % width;
    int64_t part = rows * (dv + 2);
    A m = ws[R * (dv + 2)], l = ws[R * (dv + 2) + 1];
    A a = col < dv ? ws[R * (dv + 2) + 2 + col] : (A)0;
    for (int z = 1; z < parts; z++) {
      const A *y = ws + z * part + R * (dv + 2);
      A ym = y[0], yl = y[1], ya = col < dv ? y[2 + col] : (A)0;
      A mx = greatest(m, ym);
      if (mx == (A)-INFINITY) {
        a = a + ya;
        continue;
      }
      A al = exp_(m - mx), be = exp_(ym - mx);
      A t1 = l * al, t2 = yl * be;
      l = t1 + t2;
      m = mx;
      A aa = al != 1 ? a * al : a;
      a = fma_(be, ya, aa);
    }
    int64_t i = R % tq, h = R / tq % hq, b = R / tq / hq;
    if (col < dv) {
      int64_t at = lead_at(w, od, b) + h * stride(od, nl) + i * stride(od, nl + 1) +
                   col * stride(od, nl + 2);
      wr((void *)base(w, od), at, odt, l == 0 ? (A)0 : a / l + (A)0);
    } else {
      int64_t at = lead_at(w, ld, b) + h * stride(ld, nl) + i * stride(ld, nl + 1);
      ((A *)base(w, ld))[at] =
          l == 0 ? (A)-INFINITY : m == (A)INFINITY ? m : m + log_(l);
    }
  }
}

/* The matrix-unit kernel: float16 or bfloat16 q, k and v into a float32
   accumulator. Eight warps take 128 rows, a warp 16, against blocks of
   64 keys:

     q tile --> registers (ldmatrix), once a tile
     per block: k, v --> shared (raw 16-bit), the next block's in flight
       (cp.async); S = q k^T (8 key tiles x dk/16 steps); scale, output 1,
       hidden -inf; the row's greatest, the weights, lanes 2t and 2t + 1
       in the thread, the tree across the quad (xor 2, xor 1): nx.cpu's
       lane order; o = alpha o; p rounded into A fragments; o += p v
       (dv/8 tiles x 4 steps), v by ldmatrix.trans

   dk at most NX_ATT_MMA_DK (registers), any dv by chunks of 128
   columns. */

constexpr int MMA_RT = 128, MMA_DVC = NX_ATT_COLUMNS, MMA_DK = NX_ATT_MMA_DK;
constexpr int MMA_QS = MMA_DK + 8;  /* q and k rows, in 16-bit elements */
constexpr int MMA_VS = MMA_DVC + 8; /* v rows */

/* Shared memory: the q tile, whose region then holds the second buffer
   of k and v; the first buffer; then VIS, KO, VO for each buffer, and
   CLS. */
constexpr size_t MMA_KV = 2 * KEYS * MMA_QS + 2 * KEYS * MMA_VS; /* bytes */
constexpr size_t MMA_OFF_B =
    2 * MMA_RT * MMA_QS > MMA_KV ? 2 * MMA_RT * MMA_QS : MMA_KV;
constexpr size_t MMA_OFF_X = MMA_OFF_B + MMA_KV;

/* ldmatrix x4 at the shared address [s], plain and transposed. */
__device__ __forceinline__ unsigned smem_at(const void *a) {
  return (unsigned)__cvta_generic_to_shared(a);
}

__device__ __forceinline__ void ldsm4(unsigned (&r)[4], unsigned s) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(s));
}

__device__ __forceinline__ void ldsm4t(unsigned (&r)[4], unsigned s) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(s));
}

/* d = a b + d, m16n8k16, float32 accumulators; BF selects bfloat16. */
template <bool BF>
__device__ __forceinline__ void mma16816(float (&d)[4], const unsigned (&a)[4],
                                         unsigned b0, unsigned b1) {
  if (BF)
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
  else
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* Two float32 weights rounded once to nearest into v's 16-bit dtype,
   packed, x low: nx_float_to_f16's and nx_float_to_bf16's bits but a NaN's
   payload. */
template <bool BF>
__device__ __forceinline__ unsigned pack2(float x, float y) {
  unsigned d;
  if (BF) asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(y), "f"(x));
  else asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(d) : "f"(y), "f"(x));
  return d;
}

/* n 16-bit elements of row a (stride st) from position d0 into s, zeros
   past the row's end or for an absent row: by 16 bytes where the row steps
   one element and its address lies on 16 bytes, else one by one. */
__device__ __forceinline__ void row8(uint16_t *s, const uint16_t *x,
                                     int64_t a, int64_t st, int64_t d0,
                                     int64_t n) {
  if (a < 0 || d0 >= n) {
    *(uint4 *)s = make_uint4(0, 0, 0, 0);
    return;
  }
  const uint16_t *at = x + a + d0 * st;
  if (st == 1 && d0 + 8 <= n && (uintptr_t)at % 16 == 0) {
    *(uint4 *)s = *(const uint4 *)at;
    return;
  }
#pragma unroll
  for (int e = 0; e < 8; e++) s[e] = d0 + e < n ? x[a + (d0 + e) * st] : 0;
}

/* row8 by cp.async where it would load 16 bytes, so that a block's k and
   v are in flight while the previous block computes. */
__device__ __forceinline__ void row8a(uint16_t *s, const uint16_t *x,
                                      int64_t a, int64_t st, int64_t d0,
                                      int64_t n) {
  const uint16_t *at = x + a + d0 * st;
  if (a >= 0 && st == 1 && d0 + 8 <= n && (uintptr_t)at % 16 == 0) {
    unsigned d = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d),
                 "l"(at)
                 : "memory");
    return;
  }
  row8(s, x, a, st, d0, n);
}

/* A block's k and v into a buffer: the issue half of the pipeline. */
__device__ __forceinline__ void issue_kv(const u64 *w, uint16_t *Ks,
                                         uint16_t *Vs, const int64_t *KO,
                                         const int64_t *VO, int64_t cbase) {
  const uint16_t *kx = (const uint16_t *)base(w, desc(w, NX_ATT_K));
  const uint16_t *vx = (const uint16_t *)base(w, desc(w, NX_ATT_VALUES));
  int64_t ksd = dim_stride(w, NX_ATT_K), vsd = dim_stride(w, NX_ATT_VALUES);
  int64_t dk = hw(w, NX_ATT_DK), dv = hw(w, NX_ATT_DV);
  for (int e = threadIdx.x; e < KEYS * (MMA_DK / 8); e += AT) {
    int j = e / (MMA_DK / 8), d8 = e % (MMA_DK / 8);
    row8a(Ks + j * MMA_QS + d8 * 8, kx, KO[j], ksd, d8 * 8, dk);
  }
  for (int e = threadIdx.x; e < KEYS * (MMA_DVC / 8); e += AT) {
    int j = e / (MMA_DVC / 8), d8 = e % (MMA_DVC / 8);
    row8a(Vs + j * MMA_VS + d8 * 8, vx, VO[j], vsd, cbase + d8 * 8, dv);
  }
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

/* Output 1 on the visible pairs among an mma thread's 32 scores s[n][e]
   (rows r0 + 8 (e / 2), keys 8 n + 2 t4 + e % 2), a fragment's four at a
   time, VV of them a run of the tape: 32 / VV runs a block. The
   fragment's index stays a loop's, its scores moved by selects, so that s
   stays in registers. */
template <int VV>
__device__ __forceinline__ void rescore_units(const u64 *w, const unit &c,
                                              float (&s)[8][4], int r0,
                                              int t4, int64_t j0, int nk,
                                              const unsigned long long *VIS) {
#pragma unroll 1
  for (int n = 0; n < 8; n++) {
    float x[4];
    int rr[4];
    int64_t jj[4];
#pragma unroll
    for (int e = 0; e < 4; e++) {
      x[e] = s[0][e];
#pragma unroll
      for (int m = 1; m < 8; m++)
        if (m == n) x[e] = s[m][e];
      int r = r0 + 8 * (e / 2), j = 8 * n + 2 * t4 + e % 2;
      rr[e] = r;
      jj[e] = r < c.rows && j < nk && ((VIS[r] >> j) & 1) ? j0 + j : -1;
    }
#pragma unroll
    for (int h = 0; h < 4; h += VV) rescore<float>(w, c, VV, x + h, rr + h, jj + h);
#pragma unroll
    for (int e = 0; e < 4; e++)
#pragma unroll
      for (int m = 0; m < 8; m++)
        if (m == n && jj[e] >= 0) s[m][e] = x[e];
  }
}

__device__ __noinline__ void mma(const u64 *w) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned char *sm = smem + (size_t)w[NX_ATT_SHARED];
  uint16_t *Qs = (uint16_t *)sm;
  uint16_t *KB[2] = {(uint16_t *)(sm + MMA_OFF_B), (uint16_t *)sm};
  unsigned long long *VISB = (unsigned long long *)(sm + MMA_OFF_X);
  int64_t *KOB = (int64_t *)(VISB + 2 * MMA_RT), *VOB = KOB + 2 * KEYS;
  uint8_t *CLS = (uint8_t *)(VOB + 2 * KEYS);

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int t4 = lane % 4, r0 = warp * 16 + lane / 4; /* rows r0, r0 + 8 */
  const bool bf = (w[NX_ATT_DT] & 0xff) == NX_BFLOAT16;
  const u64 *qd = desc(w, NX_ATT_Q);
  const int nl = (int)w[NX_ATT_NLEAD];
  const uint16_t *qx = (const uint16_t *)base(w, qd);
  const int64_t dk = hw(w, NX_ATT_DK), dv = hw(w, NX_ATT_DV);
  const int nks = (int)((dk + 15) / 16);
  const bool rescores = w[NX_ATT_FLAGS] & NX_ATT_SCORE;
  const bool scores = w[NX_ATT_FLAGS] & NX_ATT_SCORES;
  float *const SC = (float *)w[NX_ATT_HEADER + w[NX_ATT_WORK]];
  int64_t kb0, kb1;
  part_blocks(w, &kb0, &kb1);
  const float scale = (float)__longlong_as_double((long long)w[NX_ATT_SCALE]);
  int64_t ncols = (dv + MMA_DVC - 1) / MMA_DVC;
  ncols = ncols > 0 ? ncols : 1;

  for (int64_t cb = blockIdx.y; cb < ncols; cb += gridDim.y)
  for (int64_t u = blockIdx.x; u < hw(w, NX_ATT_UNITS); u += gridDim.x) {
    const int64_t cbase = cb * MMA_DVC;
    const int ndv = (int)(dv - cbase < MMA_DVC ? (dv - cbase > 0 ? dv - cbase : 0) : MMA_DVC);
    unit c = unit_of(w, u);

    /* The q tile into registers. */
    __syncthreads();
    for (int e = tid; e < MMA_RT * (MMA_DK / 8); e += AT) {
      int r = e / (MMA_DK / 8), d8 = e % (MMA_DK / 8);
      int64_t a = -1;
      if (r < c.rows)
        a = c.qa + (c.h0 + r / c.nt) * stride(qd, nl) +
            (c.i0 + r % c.nt) * stride(qd, nl + 1);
      row8(Qs + r * MMA_QS + d8 * 8, qx, a, stride(qd, nl + 2), d8 * 8, dk);
    }
    __syncthreads();
    unsigned qf[MMA_DK / 16][4];
#pragma unroll
    for (int ks = 0; ks < MMA_DK / 16; ks++)
      if (ks < nks)
        ldsm4(qf[ks], smem_at(Qs + (warp * 16 + lane % 16) * MMA_QS + ks * 16 + (lane / 16) * 8));

    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0, 0};
    float o[MMA_DVC / 8][4];
#pragma unroll
    for (int n = 0; n < MMA_DVC / 8; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0;

    int64_t nb = (c.len + KEYS - 1) / KEYS;
    int64_t e1 = kb1 < nb ? kb1 : nb;
    /* The pipeline: the next block some row sees, its keys' offsets and
       visibility into slot s ^ 1 and its k and v in flight, while slot s
       computes. */
    __syncthreads(); /* the q tile is in registers: its region is buffer 1 */
    int sl = 0;
    int64_t nk_next = 0;
    auto next = [&](int64_t from, int slt) -> int64_t {
      for (int64_t x = from; x < e1; x++) {
        int64_t j0 = x * KEYS;
        int nk = (int)(c.len - j0 < KEYS ? c.len - j0 : KEYS);
        int cls = class_of(w, c, kb0, e1, x, CLS);
        if (!block_vis<MMA_RT, true>(w, c, cls, j0, nk, VISB + slt * MMA_RT,
                               KOB + slt * KEYS, VOB + slt * KEYS))
          continue;
        nk_next = nk;
        issue_kv(w, KB[slt], KB[slt] + KEYS * MMA_QS, KOB + slt * KEYS,
                 VOB + slt * KEYS, cbase);
        return x;
      }
      return -1;
    };
    int64_t kb = next(kb0, 0);
    while (kb >= 0) {
      int nk = (int)nk_next;
      int64_t kn = next(kb + 1, sl ^ 1);
      if (kn >= 0) asm volatile("cp.async.wait_group 1;\n" ::: "memory");
      else asm volatile("cp.async.wait_group 0;\n" ::: "memory");
      __syncthreads();
      uint16_t *Ks = KB[sl], *Vs = KB[sl] + KEYS * MMA_QS;
      const unsigned long long *VIS = VISB + sl * MMA_RT;
      int64_t j0 = kb * KEYS;
      /* whether a value of the block is not finite */
      int bad = 0;
      {
        uint16_t ex = bf ? 0x7f80 : 0x7c00;
        for (int e = tid; e < KEYS * (MMA_DVC / 8); e += AT) {
          uint4 x = *(const uint4 *)(Vs + (e / (MMA_DVC / 8)) * MMA_VS + (e % (MMA_DVC / 8)) * 8);
          unsigned y[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
          for (int i = 0; i < 4; i++)
            bad |= ((y[i] & ex) == ex) | (((y[i] >> 16) & ex) == ex);
        }
      }
      bad = __syncthreads_or(bad);

      /* S = q k^T. The lane's addresses in k and v, the fragments' offsets
         from them constants. */
      const int mi = lane / 8;
      const unsigned ka = smem_at(Ks + (8 * (mi / 2) + lane % 8) * MMA_QS + (mi % 2) * 8);
      const unsigned va = smem_at(Vs + ((mi % 2) * 8 + lane % 8) * MMA_VS + 8 * (mi / 2));
      float s[8][4];
#pragma unroll
      for (int n = 0; n < 8; n++) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0;
      /* Each dtype's loop of its own, so that no mma issues predicated
         off. */
      auto qk = [&](auto bfc) {
        constexpr bool BF = decltype(bfc)::value;
#pragma unroll
        for (int ks = 0; ks < MMA_DK / 16; ks++) {
          if (ks >= nks) break;
#pragma unroll
          for (int n = 0; n < 8; n += 2) {
            unsigned kf[4];
            ldsm4(kf, ka + 2 * (8 * n * MMA_QS + ks * 16));
            mma16816<BF>(s[n], qf[ks], kf[0], kf[1]);
            mma16816<BF>(s[n + 1], qf[ks], kf[2], kf[3]);
          }
        }
      };
      if (bf) qk(std::true_type());
      else qk(std::false_type());
#pragma unroll
      for (int n = 0; n < 8; n++)
#pragma unroll
        for (int e = 0; e < 4; e++) s[n][e] = s[n][e] * scale;

      if (scores) {
        /* The thread's pairs' scores where their keys are present. */
#pragma unroll
        for (int q = 0; q < 32; q++) {
          int r = r0 + 8 * ((q % 4) / 2), j = 8 * (q / 4) + 2 * t4 + q % 2;
          if (r < c.rows && (VIS[r] >> j) & 1)
            SC[score_at(w, c, r, j0 + j)] = s[q / 4][q % 4];
        }
        __syncthreads(); /* buffer sl is free for the block after next */
        sl ^= 1;
        kb = kn;
        continue;
      }

      /* Output 1 on the thread's visible pairs. */
      if (rescores) {
        int V = (int)w[NX_ATT_V];
        if (V >= 4) rescore_units<4>(w, c, s, r0, t4, j0, nk, VIS);
        else if (V == 2) rescore_units<2>(w, c, s, r0, t4, j0, nk, VIS);
        else rescore_units<1>(w, c, s, r0, t4, j0, nk, VIS);
      }

      /* The softmax of rows r0 (x = 0) and r0 + 8 (x = 1), in registers. */
      float alpha[2];
#pragma unroll
      for (int x = 0; x < 2; x++) {
        int r = r0 + 8 * x;
        unsigned long long vis = r < c.rows ? VIS[r] : 0;
        float mx = m[x];
#pragma unroll
        for (int n = 0; n < 8; n++)
#pragma unroll
          for (int e = 0; e < 2; e++) {
            int j = 8 * n + 2 * t4 + e;
            float y = (vis >> j) & 1 ? s[n][2 * x + e] : -INFINITY;
            s[n][2 * x + e] = y;
            mx = greatest(mx, y);
          }
        mx = greatest(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        mx = greatest(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        bool on = vis != 0 && mx != -INFINITY;
        float l0 = 0, l1 = 0;
#pragma unroll
        for (int n = 0; n < 8; n++) {
          float w0 = on ? exp_(s[n][2 * x] - mx) : 0.f;
          float w1 = on ? exp_(s[n][2 * x + 1] - mx) : 0.f;
          s[n][2 * x] = w0, s[n][2 * x + 1] = w1;
          l0 = l0 + w0, l1 = l1 + w1;
        }
        /* lanes i and i + 4 (xor 2), then i + 2 (xor 1), then i + 1 */
        l0 = l0 + __shfl_xor_sync(0xffffffff, l0, 2);
        l1 = l1 + __shfl_xor_sync(0xffffffff, l1, 2);
        l0 = l0 + __shfl_xor_sync(0xffffffff, l0, 1);
        l1 = l1 + __shfl_xor_sync(0xffffffff, l1, 1);
        float sum = l0 + l1, al = 1;
        if (on) {
          al = exp_(m[x] - mx);
          float t = l[x] * al;
          l[x] = t + sum;
          m[x] = mx;
        }
        alpha[x] = al;
      }
#pragma unroll
      for (int n = 0; n < MMA_DVC / 8; n++) {
        if (alpha[0] != 1) o[n][0] = o[n][0] * alpha[0], o[n][1] = o[n][1] * alpha[0];
        if (alpha[1] != 1) o[n][2] = o[n][2] * alpha[1], o[n][3] = o[n][3] * alpha[1];
      }

      if (!bad) {
        /* o += p v: p from the score fragments, rounded into v's dtype. */
        auto pv = [&](auto bfc) {
          constexpr bool BF = decltype(bfc)::value;
#pragma unroll
          for (int kk = 0; kk < 4; kk++) {
            unsigned pa[4] = {pack2<BF>(s[2 * kk][0], s[2 * kk][1]),
                              pack2<BF>(s[2 * kk][2], s[2 * kk][3]),
                              pack2<BF>(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                              pack2<BF>(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
            for (int n = 0; n < MMA_DVC / 8; n += 2) {
              if (8 * n >= ndv) break;
              unsigned vf[4];
              ldsm4t(vf, va + 2 * (16 * kk * MMA_VS + 8 * n));
              mma16816<BF>(o[n], pa, vf[0], vf[1]);
              mma16816<BF>(o[n + 1], pa, vf[2], vf[3]);
            }
          }
        };
        if (bf) pv(std::true_type());
        else pv(std::false_type());
      } else {
        /* The pairs each row sees, fused in increasing key: each key's
           weights from the quad's thread that holds them. */
        unsigned long long v0 = r0 < c.rows ? VIS[r0] : 0;
        unsigned long long v1 = r0 + 8 < c.rows ? VIS[r0 + 8] : 0;
        for (int j = 0; j < nk; j++) {
          int n = j / 8, e = j % 2, from = (lane & ~3) | (j % 8) / 2;
          float w0 = 0, w1 = 0;
#pragma unroll
          for (int x = 0; x < 8; x++)
            if (x == n) {
              w0 = e ? s[x][1] : s[x][0];
              w1 = e ? s[x][3] : s[x][2];
            }
          float p0 = __shfl_sync(0xffffffff, w0, from);
          float p1 = __shfl_sync(0xffffffff, w1, from);
          bool s0 = (v0 >> j) & 1, s1 = (v1 >> j) & 1;
#pragma unroll
          for (int n2 = 0; n2 < MMA_DVC / 8; n2++)
#pragma unroll
            for (int e2 = 0; e2 < 2; e2++) {
              int col = 8 * n2 + 2 * t4 + e2;
              uint16_t x = Vs[j * MMA_VS + col];
              float y = bf ? nx_bf16_to_float(x) : nx_f16_to_float(x);
              if (s0) o[n2][e2] = fmaf(p0, y, o[n2][e2]);
              if (s1) o[n2][2 + e2] = fmaf(p1, y, o[n2][2 + e2]);
            }
        }
      }
      __syncthreads(); /* buffer sl is free for the block after next */
      sl ^= 1;
      kb = kn;
    }

    if (scores) continue;
    /* o and lse, or this part's state: one loop over the thread's
       elements, out of registers. */
    float ol[MMA_DVC / 8][4];
#pragma unroll
    for (int n = 0; n < MMA_DVC / 8; n++)
#pragma unroll
      for (int e = 0; e < 4; e++) ol[n][e] = o[n][e];
#pragma unroll 1
    for (int x = 0; x < 2; x++) {
      int r = r0 + 8 * x;
      if (r >= c.rows) continue;
      row_dst d = dst_of<float>(w, c, r);
#pragma unroll 1
      for (int e = 0; e < MMA_DVC / 4; e++) {
        int n = e / 2, col = 8 * n + 2 * t4 + e % 2;
        if (col < ndv) put_o<float>(d, cbase + col, ol[n][2 * x + e % 2], l[x]);
      }
    }
#pragma unroll
    for (int x = 0; x < 2; x++) {
      int r = r0 + 8 * x;
      if (r >= c.rows || t4 != 0 || cb != 0) continue;
      put_lse<float>(w, c, r, m[x], l[x]);
    }
  }
}

} // namespace att

#undef KEYS
#undef AT

/* The kernels' entries: one for every path, which holds the code the
   paths share once (kernels.h). */

#define ATTENTION(name, ...)                                                   \
  extern "C" __global__ void __launch_bounds__(NX_ATT_THREADS, 1)              \
      name(const __grid_constant__ attention_params p) {                       \
    const u64 *w = (const u64 *)p.w;                                           \
    switch (w[NX_ATT_KERNEL]) {                                                \
    case NX_ATT_F32: att::simt<float, 128, NX_ATT_COLUMNS>(w); return;         \
    case NX_ATT_F32_R16: att::simt<float, 16, NX_ATT_COLUMNS>(w); return;      \
    case NX_ATT_F64: att::simt<double, 64, 64>(w); return;                     \
    default: att::mma(w);                                                      \
    }                                                                          \
  }

#define ATTENTION_MERGE(name, ...)                                             \
  extern "C" __global__ void __launch_bounds__(NX_ATT_THREADS)                 \
      name(const __grid_constant__ attention_params p) {                       \
    const u64 *w = (const u64 *)p.w;                                           \
    if (w[NX_ATT_KERNEL] == NX_ATT_F64) att::merge<double>(w);                 \
    else att::merge<float>(w);                                                 \
  }

#endif
