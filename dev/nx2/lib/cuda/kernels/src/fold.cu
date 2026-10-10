/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Reductions and scans of one operand by Sum, Prod, Max or Min
   (kernels.h's fold_params), in nx.cpu's association, so that the two
   libraries agree bit for bit.

   An output's terms are t = 0, 1, ... in C order of the reduced axes. They
   fall into blocks of NX_FOLD_BLOCK consecutive terms and, within a block,
   into NX_FOLD_LANES lanes from the identity, term t into lane t mod 16,
   each lane folding its terms in increasing t. Lane i takes lane i + 8 for
   i < 8, then lane i + 4, i + 2 and i + 1: lane 0 is the block's value.
   The blocks combine by the binary tree whose left part holds the largest
   power of two of blocks below their count.

   The tree composes from aligned ranges. A range of 2^s blocks from a
   multiple of 2^s whose blocks all exist is a subtree. The last range of
   an output, cut at c blocks, folds into a tail: the subtrees of c's set
   bits, largest first, each combined with the fold of those after it. The
   tree over the whole ranges with the tail innermost is the tree over the
   blocks. A counter folds a run of values either way in one pass
   (Counter). So each kernel folds an aligned range, and fold_tree folds
   the ranges' values of an output:

     fold_rows   a group of 16 threads a lane each, along a run of terms
     fold_cols   a thread an output, its 16 lanes in registers, the
                 threads of a warp on neighbouring outputs
     fold_tree   a block an output, over its ranges' values

   A scan cuts each slice along its axis into chunks of NX_SCAN_CHUNK
   terms from its start. scan_totals folds each whole chunk as above;
   scan_rescan starts chunk c from the carry, the identity combined with
   the totals of chunks 0 to c - 1 in order, and combines it with the
   chunk's terms left to right.

   NaN. Float folds track the first NaN term of what they fold, as an
   int64 position, combined by the least. A result that is NaN where a
   term is NaN is that first term's bits; a scan's results from its first
   NaN term on are that term's bits.

   Part of kernels.cu's translation unit, after nx_kinds.h. */

#ifndef NX_CUDA_FOLD_CU
#define NX_CUDA_FOLD_CU

#define BLOCK NX_FOLD_BLOCK
#define LANES NX_FOLD_LANES
#define NONE INT64_MAX

/* A range's value as fold_tree and scan_rescan read it: its bits, then
   its first NaN term or NONE. */
typedef struct {
  uint64_t bits;
  int64_t first;
} part;

/* Monoids */

/* The monoid [m], an NX_SUM to NX_MIN of nx_spec.h: a parameter of the
   launch, so its switch takes the same branch in every thread. */
template <typename T> __device__ __forceinline__ T op(int m, T a, T b) {
  switch (m) {
  case NX_SUM: return k_add(a, b);
  case NX_PROD: return k_mul(a, b);
  case NX_MAX: return k_maximum(a, b);
  default: return k_minimum(a, b);
  }
}

/* The least and greatest values of T: the infinities of a float. */
template <typename T> __device__ __forceinline__ T least() {
  if constexpr (real<T>) return -T(INFINITY);
  else if constexpr (T(-1) < T(0)) return T(T(1) << (8 * sizeof(T) - 1));
  else return T(0);
}

template <typename T> __device__ __forceinline__ T greatest() {
  if constexpr (real<T>) return T(INFINITY);
  else return T(~least<T>());
}

template <typename T> __device__ __forceinline__ T identity(int m) {
  switch (m) {
  case NX_SUM: return T(0);
  case NX_PROD: return T(1);
  case NX_MAX: return least<T>();
  default: return greatest<T>();
  }
}

/* Whether [v] is a NaN: never for an integer. */
template <typename T> __device__ __forceinline__ bool is_nan(T v) {
  if constexpr (real<T>) return v != v;
  else return false;
}

/* Elements */

/* Element [i] of [x], of the dtype [dt], in the compute type T: a float
   in its own type, an integer widened by its own sign, a bool as 0 or 1. */
/* CR: Decode NX_BOOL as byte != 0 here. A valid one-element Bool byte 2
   makes Max/Min reductions and scans store 2; nx.cpu produces 1, observable
   through a Uint8 bitcast. Normalizing at this shared load boundary
   preserves the promised CPU-identical bits for every path. */
template <typename T> __device__ __forceinline__ T load(const void *x, int64_t i, int dt) {
  if constexpr (real<T> || sizeof(T) == 8) return ((const T *)x)[i];
  else
    switch (dt) {
    case NX_INT8: return (T)((const int8_t *)x)[i];
    case NX_UINT8:
    case NX_BOOL: return (T)((const uint8_t *)x)[i];
    case NX_INT16: return (T)((const int16_t *)x)[i];
    case NX_UINT16: return (T)((const uint16_t *)x)[i];
    default: return ((const T *)x)[i];
    }
}

/* Stores [v] at element [i] of [y], of the dtype [dt]: an integer's low
   bits. */
template <typename T> __device__ __forceinline__ void store(void *y, int64_t i, int dt, T v) {
  if constexpr (real<T> || sizeof(T) == 8) ((T *)y)[i] = v;
  else
    switch (dt) {
    case NX_INT8:
    case NX_UINT8:
    case NX_BOOL: ((uint8_t *)y)[i] = (uint8_t)v; return;
    case NX_INT16:
    case NX_UINT16: ((uint16_t *)y)[i] = (uint16_t)v; return;
    default: ((T *)y)[i] = v;
    }
}

template <typename T> __device__ __forceinline__ uint64_t to_bits(T v) {
  uint64_t b = 0;
  memcpy(&b, &v, sizeof(T));
  return b;
}

template <typename T> __device__ __forceinline__ T of_bits(uint64_t b) {
  T v;
  memcpy(&v, &b, sizeof(T));
  return v;
}

/* Addresses */

/* The offset in elements of term [t] along the reduced axes, from the
   output's first term: one multiplication along one axis, out of line
   along several. */
__device__ __noinline__ int64_t terms_at(const fold_params &p, int64_t t) {
  int64_t at = 0;
  for (int d = p.nred - 1; d >= 0; d--) {
    const int64_t e = p.red[d][0];
    at += t % e * p.red[d][1];
    t /= e;
  }
  return at;
}

__device__ __forceinline__ int64_t term_at(const fold_params &p, int64_t t) {
  return p.nred == 1 ? t * p.red[0][1] : terms_at(p, t);
}

/* The offset in elements of output (or slice) [o] in x ([side] 1) or y
   ([side] 2), along the kept axes. */
__device__ __noinline__ int64_t kept_at(const fold_params &p, int64_t o, int side) {
  int64_t at = 0;
  for (int d = p.nkept - 1; d >= 0; d--) {
    const int64_t e = p.kept[d][0];
    at += o % e * p.kept[d][side];
    o /= e;
  }
  return at;
}

/* The counter */

/* Folds a run of values in the blocks' tree: after the k-th push, it holds
   the subtrees of k's set bits, largest at the bottom; a push combines the
   two newest once per trailing zero of k. [finish] folds them, newest
   first, into the tail [t] where given. It holds DEPTH values: the plan
   gives no fold more than 2^DEPTH - 1 of them. */
#define DEPTH 24
template <typename T> struct Counter {
  int m;
  T v[DEPTH];
  int64_t f[DEPTH];
  int top = 0;
  uint64_t pushed = 0;

  __device__ void push(T x, int64_t first) {
    v[top] = x, f[top] = first, top++, pushed++;
    for (uint64_t k = pushed; (k & 1) == 0; k >>= 1, top--) {
      v[top - 2] = op<T>(m, v[top - 2], v[top - 1]);
      f[top - 2] = min(f[top - 2], f[top - 1]);
    }
  }

  /* The fold of the run into [*x] and [*first]: the identity and NONE for
     no value and no tail. */
  __device__ void finish(bool tail, T t, int64_t tf, T *x, int64_t *first) {
    int i = top - 1;
    T r = identity<T>(m);
    int64_t rf = NONE;
    if (tail) r = t, rf = tf;
    else if (i >= 0) r = v[i], rf = f[i], i--;
    for (; i >= 0; i--) {
      r = op<T>(m, v[i], r);
      rf = min(f[i], rf);
    }
    *x = r, *first = rf;
  }
};

/* The 16 lanes [a] combined by the lanes' tree into a[0]. */
template <typename T> __device__ __forceinline__ void lane_tree(int m, T (&a)[LANES]) {
#pragma unroll
  for (int d = LANES / 2; d > 0; d /= 2)
#pragma unroll
    for (int i = 0; i < d; i++) a[i] = op<T>(m, a[i], a[i + d]);
}

/* Output [o]'s result [v], whose first NaN term is [first], stored: that
   term's bits where [v] is a NaN. */
template <typename T>
__device__ void put_output(const fold_params &p, int64_t o, T v, int64_t first) {
  if (is_nan(v) && first != NONE) v = load<T>(p.x, kept_at(p, o, 1) + term_at(p, first), p.dtype);
  store<T>(p.y, o, p.dtype, v);
}

/* A unit's value: the output's result where one unit folds it, else its
   range's value for fold_tree. */
template <typename T>
__device__ void put_unit(const fold_params &p, int64_t o, int64_t g, T v, int64_t first) {
  if (p.groups == 1) put_output<T>(p, o, v, first);
  else ((part *)p.partials)[o * p.groups + g] = part{to_bits(v), first};
}

/* Kernels */

/* A block's values, first NaN terms and counts, one of each a thread:
   declared once, outside the templates, so that the instances of a kernel
   share them. */
__device__ __forceinline__ uint64_t *shared_values() {
  __shared__ uint64_t v[256];
  return v;
}

__device__ __forceinline__ int64_t *shared_firsts() {
  __shared__ int64_t f[256];
  return f;
}

__device__ __forceinline__ int64_t *shared_counts() {
  __shared__ int64_t n[256];
  return n;
}

/* A block of 16 Q threads per unit (o, g): the range of Q x span blocks
   from block g Q span of output o. Lane group q (16 threads, lane l each)
   folds the span blocks from (g Q + q) span by a counter, its lanes summed
   by shuffles; thread 0 folds the groups' values. Every group runs every
   iteration, so the shuffles see the whole warp. */
template <typename T> __device__ void fold_rows(const fold_params &p) {
  const int m = p.monoid;
  T *sv = (T *)shared_values();
  int64_t *sf = shared_firsts(), *sn = shared_counts();
  const int64_t u = blockIdx.x, o = u / p.groups, g = u % p.groups;
  const int Q = blockDim.x / LANES, q = threadIdx.x / LANES, l = threadIdx.x % LANES;
  const int64_t x0 = kept_at(p, o, 1), b0 = (g * Q + q) * p.span;
  Counter<T> c{p.monoid};
  for (int64_t i = 0; i < p.span; i++) {
    const int64_t b = b0 + i;
    T a = identity<T>(m);
    int64_t first = NONE;
    if (b < p.blocks) {
      const int64_t end = min(p.terms, (b + 1) * BLOCK);
      for (int64_t t = b * BLOCK + l; t < end; t += LANES) {
        const T x = load<T>(p.x, x0 + term_at(p, t), p.dtype);
        if (is_nan(x) && first == NONE) first = t;
        a = op<T>(m, a, x);
      }
    }
#pragma unroll
    for (int d = LANES / 2; d > 0; d /= 2) {
      const T w = __shfl_down_sync(0xFFFFFFFFu, a, d, LANES);
      const int64_t wf = __shfl_down_sync(0xFFFFFFFFu, first, d, LANES);
      if (l < d) a = op<T>(m, a, w), first = min(first, wf);
    }
    if (l == 0 && b < p.blocks) c.push(a, first);
  }
  if (l == 0) {
    T v;
    int64_t f;
    c.finish(false, T(0), NONE, &v, &f);
    sv[q] = v, sf[q] = f, sn[q] = max((int64_t)0, min(p.span, p.blocks - b0));
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  Counter<T> d{p.monoid};
  int k = 0;
  for (; k < Q && sn[k] == p.span; k++) d.push(sv[k], sf[k]);
  T v;
  int64_t f;
  d.finish(k < Q && sn[k] > 0, k < Q ? sv[k] : T(0), k < Q ? sf[k] : NONE, &v, &f);
  put_unit<T>(p, o, g, v, f);
}

/* A thread per unit (o, g), u = g outputs + o: the span blocks from block
   g span of output o, each block's 16 lanes in registers. */
template <typename T> __device__ void fold_cols(const fold_params &p) {
  const int m = p.monoid;
  const int64_t u = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (u >= p.outputs * p.groups) return;
  const int64_t o = u % p.outputs, g = u / p.outputs;
  const int64_t x0 = kept_at(p, o, 1), b0 = g * p.span;
  const int64_t b1 = min(b0 + p.span, p.blocks);
  Counter<T> c{p.monoid};
  for (int64_t b = b0; b < b1; b++) {
    T a[LANES];
#pragma unroll
    for (int l = 0; l < LANES; l++) a[l] = identity<T>(m);
    int64_t first = NONE;
    const int64_t t0 = b * BLOCK, n = min((int64_t)BLOCK, p.terms - t0);
#pragma unroll 1
    for (int64_t j = 0; j < n; j += LANES)
#pragma unroll
      for (int l = 0; l < LANES; l++)
        if (j + l < n) {
          const T x = load<T>(p.x, x0 + term_at(p, t0 + j + l), p.dtype);
          if (is_nan(x) && first == NONE) first = t0 + j + l;
          a[l] = op<T>(m, a[l], x);
        }
    lane_tree<T>(m, a);
    c.push(a[0], first);
  }
  T v;
  int64_t f;
  c.finish(false, T(0), NONE, &v, &f);
  put_unit<T>(p, o, g, v, f);
}

/* A block of 256 threads per output: its [full] whole ranges' values, then
   the tail range's, where [groups] says there is one. Thread j folds the
   span values from j span; the thread at [full] / span takes the tail
   range's value as its tail. */
template <typename T> __device__ void fold_tree(const fold_params &p) {
  T *sv = (T *)shared_values();
  int64_t *sf = shared_firsts(), *sn = shared_counts();
  const int64_t o = blockIdx.x;
  const part *in = (const part *)p.partials + o * p.groups;
  const int j = threadIdx.x;
  const int64_t a0 = j * p.span, a1 = min(a0 + p.span, p.full);
  const bool tail = p.groups > p.full, last = j == p.full / p.span;
  Counter<T> c{p.monoid};
  for (int64_t k = a0; k < a1; k++) c.push(of_bits<T>(in[k].bits), in[k].first);
  T v;
  int64_t f;
  const bool tails = last && tail;
  c.finish(tails, tails ? of_bits<T>(in[p.full].bits) : T(0),
           tails ? in[p.full].first : NONE, &v, &f);
  sv[j] = v, sf[j] = f, sn[j] = a0 < a1 || tails;
  __syncthreads();
  if (j != 0) return;
  const int whole = (int)(p.full / p.span);
  Counter<T> d{p.monoid};
  for (int k = 0; k < whole; k++) d.push(sv[k], sf[k]);
  const bool rest = whole < 256 && sn[whole];
  d.finish(rest, rest ? sv[whole] : T(0), rest ? sf[whole] : NONE, &v, &f);
  put_output<T>(p, o, v, f);
}

/* Scans: [outputs] slices along the axis red[0] of [terms] terms, in
   [blocks] chunks; unit u = c outputs + s is chunk c of slice s. */

/* A thread per whole chunk: its total, as a reduction of its terms. */
template <typename T> __device__ void scan_totals(const fold_params &p) {
  const int m = p.monoid;
  const int64_t u = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (u >= p.outputs * (p.blocks - 1)) return;
  const int64_t s = u % p.outputs, ch = u / p.outputs;
  const int64_t x0 = kept_at(p, s, 1), xs = p.red[0][1];
  Counter<T> c{p.monoid};
#pragma unroll 1
  for (int b = 0; b < NX_SCAN_CHUNK / BLOCK; b++) {
    T a[LANES];
#pragma unroll
    for (int l = 0; l < LANES; l++) a[l] = identity<T>(m);
    int64_t first = NONE;
    const int64_t t0 = ch * NX_SCAN_CHUNK + b * BLOCK;
#pragma unroll 1
    for (int j = 0; j < BLOCK; j += LANES)
#pragma unroll
      for (int l = 0; l < LANES; l++) {
        const T x = load<T>(p.x, x0 + (t0 + j + l) * xs, p.dtype);
        if (is_nan(x) && first == NONE) first = t0 + j + l;
        a[l] = op<T>(m, a[l], x);
      }
    lane_tree<T>(m, a);
    c.push(a[0], first);
  }
  T v;
  int64_t f;
  c.finish(false, T(0), NONE, &v, &f);
  ((part *)p.partials)[u] = part{to_bits(v), f};
}

/* A thread per chunk: its carry from the totals before it, then its
   results left to right. */
template <typename T> __device__ void scan_rescan(const fold_params &p) {
  const int m = p.monoid;
  const int64_t u = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (u >= p.outputs * p.blocks) return;
  const int64_t s = u % p.outputs, ch = u / p.outputs;
  const int64_t x0 = kept_at(p, s, 1), y0 = kept_at(p, s, 2);
  const int64_t xs = p.red[0][1], ys = p.red[0][2];
  T r = identity<T>(m);
  int64_t first = NONE;
  const part *totals = (const part *)p.partials;
  for (int64_t k = 0; k < ch; k++) {
    const part t = totals[k * p.outputs + s];
    r = op<T>(m, r, of_bits<T>(t.bits));
    if (first == NONE) first = t.first;
  }
  const int64_t t0 = ch * NX_SCAN_CHUNK, t1 = min(p.terms, t0 + NX_SCAN_CHUNK);
  bool nan = first != NONE;
  T held = nan ? load<T>(p.x, x0 + first * xs, p.dtype) : T(0);
  for (int64_t t = t0; t < t1; t++) {
    if (nan) {
      store<T>(p.y, y0 + t * ys, p.dtype, held);
      continue;
    }
    const T x = load<T>(p.x, x0 + t * xs, p.dtype);
    if (is_nan(x)) nan = true, held = x;
    r = op<T>(m, r, x);
    store<T>(p.y, y0 + t * ys, p.dtype, nan ? held : r);
  }
}

/* Instances: one kernel per body, its monoid and dtype switched once. */

#define FOLD_BODY(name)                                                        \
  template <typename T> struct body_##name {                                   \
    static __device__ void run(const fold_params &p) { name<T>(p); }          \
  };
FOLD_BODY(fold_rows)
FOLD_BODY(fold_cols)
FOLD_BODY(fold_tree)
FOLD_BODY(scan_totals)
FOLD_BODY(scan_rescan)
#undef FOLD_BODY

/* The compute type of a dtype: a float its own, 8- to 32-bit integers the
   32-bit type of their signedness, bool uint32. */
template <template <typename> class B> __device__ void by_dtype(const fold_params &p) {
  switch (p.dtype) {
  case NX_FLOAT32: return B<float>::run(p);
  case NX_FLOAT64: return B<double>::run(p);
  case NX_INT64: return B<int64_t>::run(p);
  case NX_UINT64: return B<uint64_t>::run(p);
  case NX_INT32:
  case NX_INT16:
  case NX_INT8: return B<int32_t>::run(p);
  default: return B<uint32_t>::run(p);
  }
}

#define FOLD(name, ...)                                                        \
  extern "C" __global__ void __launch_bounds__(256)                            \
      name(const __grid_constant__ fold_params p) {                            \
    by_dtype<body_##name>(p);                                                  \
  }

#undef BLOCK
#undef LANES
#undef NONE
#undef DEPTH

#endif
