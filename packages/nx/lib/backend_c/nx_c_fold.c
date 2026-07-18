/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_fold.c — the fold kernel family: axis reductions (sum/prod/max/min),
   argmax/argmin, and inclusive associative scan (cumsum/cumprod/cummax/cummin).

   Every per-dtype init/step is generated from the one dtype table in nx_c.h
   via NX_C_FOR_EACH_COMPUTE_DTYPE; the vtables (nx_c_fold_table / nx_c_arg_table /
   nx_c_scan_table) are designated initializers whose unsupported slots stay NULL
   and are turned into an NX_C_ERR_UNSUPPORTED_DTYPE / NX_C_ERR_PACKED status by
   the engine — kernel code never null-checks a slot. The stubs reach the
   drivers only through the engine funnels (nx_c_engine.h).

   Accumulator widths follow nx_c.h's dtype table and "Dtype semantics": small
   ints and bool accumulate in int64, u32/u64
   in uint64, f16/bf16/fp8/f32 in float, f64 in double, complex in native
   complex. The f32 (and every float) sum step carries ≥2 partial accumulators
   so the contiguous run autovectorizes and a 2^25-ones reduction does not stall
   a single float at 2^24; f16/bf16 sums accumulate in float so a 4096-/512-ones
   run does not stall at the half/bfloat integer ceiling. max/min propagate NaN
   for floats (any NaN in the extent wins); argmax/argmin agree (NaN wins, first
   index). Complex has no ordered comparison, so max/min/argmax/argmin/cummax/
   cummin leave complex NULL; bool has no arithmetic, so sum/prod/cumsum/cumprod
   leave bool NULL — bool max/min are or/and on 0/1.

   The backend contract pins these semantics. This file only states scalar
   behavior over a run. It includes neither caml/fail.h nor caml/threads.h: kernels never
   raise and never touch the runtime lock — only the funnels (nx_c_engine.c) do,
   and the CAMLprim stubs at the foot call exactly one funnel each. */

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h"

/* ── Accumulator field for a compute type ─────────────────────────────────────
   nx_c_acc is a union keyed by C type; a kernel carries its reduced value in the
   slot matching its compute type. bool (compute uint8_t) folds through the int64
   slot: its 0/1 values fit and its max/min reuse the integer machinery. */
#define NX_C_ACCF_float f
#define NX_C_ACCF_double d
#define NX_C_ACCF_int64_t i
#define NX_C_ACCF_uint64_t u
#define NX_C_ACCF_uint8_t i
#define NX_C_ACCF_nx_c_complex32 c32
#define NX_C_ACCF_nx_c_complex64 c64

/* Sentinels for max/min init, keyed by compute type so the first real element
   always wins (the empty axis is rejected by the caller — nx_c.h fold ABI).
   For max the seed is the type minimum, for min the type maximum; unsigned u8/
   u16 compute in int64 (non-negative), so INT64_MIN / INT64_MAX bound them. */
#define NX_C_MAXSENT_float (-INFINITY)
#define NX_C_MAXSENT_double (-INFINITY)
#define NX_C_MAXSENT_int64_t INT64_MIN
#define NX_C_MAXSENT_uint64_t 0
#define NX_C_MINSENT_float INFINITY
#define NX_C_MINSENT_double INFINITY
#define NX_C_MINSENT_int64_t INT64_MAX
#define NX_C_MINSENT_uint64_t UINT64_MAX

/* ── Per-element combine, folding value V into accumulator lvalue M ───────────
   The float max/min forms propagate NaN: once M is NaN, (V > M) and (V < M) are
   both false and only a NaN V re-arms the second clause, so the first NaN sticks
   and later numbers cannot displace it — matching reduce_max/min. bool max/min
   are or/and on 0/1. V must be a plain variable (evaluated more than once). */
#define NX_C_CMB_SUM(M, V) (M) += (V)
#define NX_C_CMB_PROD(M, V) (M) *= (V)
/* Signed sum/prod combine in the unsigned width: the contract is modular wrap
   (i64 reaches the 64-bit boundary; narrower ints widen into the accumulator),
   defined without -fwrapv, which stays in the flags as belt and suspenders.
   SINT compute is always int64_t (dtype table), so the widths are exact. */
#define NX_C_CMB_SUM_WRAP(M, V) (M) = (int64_t)((uint64_t)(M) + (uint64_t)(V))
#define NX_C_CMB_PROD_WRAP(M, V) (M) = (int64_t)((uint64_t)(M) * (uint64_t)(V))
#define NX_C_CMB_MAXI(M, V) \
  if ((V) > (M)) (M) = (V)
#define NX_C_CMB_MINI(M, V) \
  if ((V) < (M)) (M) = (V)
#define NX_C_CMB_MAXF(M, V)   \
  if ((V) > (M))             \
    (M) = (V);               \
  else if ((V) != (V))       \
  (M) = (V)
#define NX_C_CMB_MINF(M, V)   \
  if ((V) < (M))             \
    (M) = (V);               \
  else if ((V) != (V))       \
  (M) = (V)
#define NX_C_CMB_MAXB(M, V) (M) |= (V)
#define NX_C_CMB_MINB(M, V) (M) &= (V)

/* ── Comparison for argmax/argmin: does V displace the running best B? ────────
   Strict, so ties keep the earlier index; the float forms also take V when it is
   NaN and B is not, so the first NaN wins the index (agreeing with reduce). */
#define NX_C_ACMP_MAXI(V, B) ((V) > (B))
#define NX_C_ACMP_MINI(V, B) ((V) < (B))
#define NX_C_ACMP_MAXF(V, B) \
  (((V) > (B)) || (((V) != (V)) && !((B) != (B))))
#define NX_C_ACMP_MINF(V, B) \
  (((V) < (B)) || (((V) != (V)) && !((B) != (B))))

/* ── Kernel templates ────────────────────────────────────────────────────────
   One reduction/scan/argreduce body, specialized by the combine/compare macro.
   Each branches on a unit-stride run so the contiguous case walks a typed
   pointer and autovectorizes; the strided branch handles arbitrary (incl.
   negative) byte strides. */

/* Single-accumulator reduction: prod (all), integer/complex sum, max/min. */
#define NX_C_REDUCE_STEP(opname, sfx, storage, compute, CMB)                    \
  static void nx_c_##opname##_step_##sfx(nx_c_acc *acc, const char *in,          \
                                        int64_t in_step, int64_t n,            \
                                        void *ctx) {                           \
    (void)ctx;                                                                 \
    compute m = acc->NX_C_ACCF_##compute;                                       \
    if (in_step == (int64_t)sizeof(storage)) {                                 \
      const storage *p = (const storage *)in;                                  \
      for (int64_t k = 0; k < n; k++) {                                        \
        compute v = nx_c_ld_##sfx(&p[k]);                                       \
        CMB(m, v);                                                             \
      }                                                                        \
    } else {                                                                   \
      for (int64_t k = 0; k < n; k++) {                                        \
        compute v = nx_c_ld_##sfx(in + k * in_step);                            \
        CMB(m, v);                                                             \
      }                                                                        \
    }                                                                          \
    acc->NX_C_ACCF_##compute = m;                                               \
  }

/* Float sum: sixteen partial accumulators. The contiguous run vectorizes into
   four independent NEON accumulators (clang packs four partials per vector) —
   the instruction-level parallelism a reduction needs to stream at single-core
   read bandwidth, where two accumulators (an 8-wide unroll) stall latency-bound
   at ~half the rate. The partials also keep each lane below the compute type's
   integer ceiling, so a run of ones sums exactly instead of stalling a single
   accumulator. The combine is a fixed balanced tree, so the grouping — hence the
   rounding — does not depend on n. */
#define NX_C_SUM_STEP_FLOAT_DEF(sfx, storage, compute)                          \
  static void nx_c_sum_step_##sfx(nx_c_acc *acc, const char *in, int64_t in_step,\
                                 int64_t n, void *ctx) {                        \
    (void)ctx;                                                                 \
    compute s[16];                                                             \
    for (int i = 0; i < 16; i++) s[i] = 0;                                     \
    int64_t k = 0;                                                             \
    if (in_step == (int64_t)sizeof(storage)) {                                 \
      const storage *p = (const storage *)in;                                  \
      for (; k + 16 <= n; k += 16)                                             \
        for (int i = 0; i < 16; i++) s[i] += nx_c_ld_##sfx(&p[k + i]);          \
      for (; k < n; k++) s[0] += nx_c_ld_##sfx(&p[k]);                          \
    } else {                                                                   \
      for (; k < n; k++) s[0] += nx_c_ld_##sfx(in + k * in_step);              \
    }                                                                          \
    compute lo = ((s[0] + s[1]) + (s[2] + s[3])) +                            \
                 ((s[4] + s[5]) + (s[6] + s[7]));                             \
    compute hi = ((s[8] + s[9]) + (s[10] + s[11])) +                          \
                 ((s[12] + s[13]) + (s[14] + s[15]));                         \
    acc->NX_C_ACCF_##compute += lo + hi;                                        \
  }

/* Inclusive scan: sequential within a slice; the running value is stored back
   per element (so a float scan stores rounded copies while carrying full
   precision). state carries across calls in case a slice is handed over split. */
#define NX_C_SCAN_STEP(opname, sfx, storage, compute, CMB)                      \
  static void nx_c_##opname##_step_##sfx(                                       \
      char *out, int64_t out_step, const char *in, int64_t in_step, int64_t n, \
      nx_c_acc *state, void *ctx) {                                             \
    (void)ctx;                                                                 \
    compute s = state->NX_C_ACCF_##compute;                                     \
    for (int64_t k = 0; k < n; k++) {                                          \
      compute v = nx_c_ld_##sfx(in + k * in_step);                             \
      CMB(s, v);                                                               \
      nx_c_st_##sfx(out + k * out_step, s);                                     \
    }                                                                          \
    state->NX_C_ACCF_##compute = s;                                             \
  }

/* Argreduce over exactly one axis: k is the axis index directly (one run per
   output, nx_c.h argreduce ABI). index < 0 seeds on the first element. */
#define NX_C_ARG_STEP(opname, sfx, storage, compute, ACMP)                      \
  static void nx_c_##opname##_step_##sfx(nx_c_arg_acc *acc, const char *in,      \
                                        int64_t in_step, int64_t n,            \
                                        void *ctx) {                           \
    (void)ctx;                                                                 \
    for (int64_t k = 0; k < n; k++) {                                          \
      compute v = nx_c_ld_##sfx(in + k * in_step);                             \
      if (acc->index < 0) {                                                    \
        acc->value.NX_C_ACCF_##compute = v;                                     \
        acc->index = k;                                                        \
        continue;                                                              \
      }                                                                        \
      compute best = acc->value.NX_C_ACCF_##compute;                            \
      if (ACMP(v, best)) {                                                     \
        acc->value.NX_C_ACCF_##compute = v;                                     \
        acc->index = k;                                                        \
      }                                                                        \
    }                                                                          \
  }

/* Streaming reduction step (nx_c_engine.h streaming path): fold one reduced row
   of n lane elements into a per-lane accumulator array. `first` seeds the array
   from this row; otherwise it combines with CMB. The lane loop has independent
   accumulators so it vectorizes; the contiguous branch (unit lane step) lets the
   compiler pack the loads. Accumulators are the compute type — wide accumulation
   matches the per-output step, but per lane there is a single accumulator, since
   the vectorization comes from the lanes, not from unrolling the reduced axis. */
#define NX_C_STREAM_STEP(opname, sfx, storage, compute, CMB)                    \
  static void nx_c_##opname##_stream_##sfx(void *accs, const char *in,          \
                                          int64_t in_step, int64_t n,          \
                                          int first, void *ctx) {              \
    (void)ctx;                                                                 \
    compute *a = (compute *)accs;                                              \
    if (in_step == (int64_t)sizeof(storage)) {                                 \
      const storage *p = (const storage *)in;                                  \
      if (first) {                                                             \
        for (int64_t j = 0; j < n; j++) a[j] = nx_c_ld_##sfx(&p[j]);           \
      } else {                                                                 \
        for (int64_t j = 0; j < n; j++) {                                      \
          compute v = nx_c_ld_##sfx(&p[j]);                                     \
          CMB(a[j], v);                                                        \
        }                                                                      \
      }                                                                        \
    } else {                                                                   \
      if (first) {                                                             \
        for (int64_t j = 0; j < n; j++) a[j] = nx_c_ld_##sfx(in + j * in_step); \
      } else {                                                                 \
        for (int64_t j = 0; j < n; j++) {                                      \
          compute v = nx_c_ld_##sfx(in + j * in_step);                         \
          CMB(a[j], v);                                                        \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  }

/* ── Shared init and fini ─────────────────────────────────────────────────────
   init seeds the op identity; fini narrows the accumulator to the output dtype
   (modular for ints, converting for f16/bf16/fp8, normalizing for bool). fini
   depends only on the dtype, so one instance per compute dtype serves every
   reduction that supports it. */
#define NX_C_DEF_FINI(sfx, kind, storage, compute, ld, st, cat)                 \
  static void nx_c_fini_##sfx(char *out, const nx_c_acc *acc, void *ctx) {       \
    (void)ctx;                                                                 \
    nx_c_st_##sfx(out, acc->NX_C_ACCF_##compute);                                \
  }
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_DEF_FINI)
#undef NX_C_DEF_FINI

/* Streaming scatter: convert the n compute-type accumulators to storage and
   write the strided output slice. Depends only on the dtype (not the op), so one
   instance per compute dtype is shared by every reduction's stream table. */
#define NX_C_DEF_SCATTER(sfx, kind, storage, compute, ld, st, cat)              \
  static void nx_c_scatter_##sfx(char *out, int64_t out_step, const void *accs, \
                                int64_t n, void *ctx) {                        \
    (void)ctx;                                                                 \
    const compute *a = (const compute *)accs;                                  \
    for (int64_t j = 0; j < n; j++) nx_c_st_##sfx(out + j * out_step, a[j]);   \
  }
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_DEF_SCATTER)
#undef NX_C_DEF_SCATTER

/* zero/one init for sum/prod (and cumsum/cumprod); arith dtypes only. */
#define NX_C_INIT_ZO(sfx, compute)                                              \
  static void nx_c_init_zero_##sfx(nx_c_acc *a, void *c) {                       \
    (void)c;                                                                   \
    a->NX_C_ACCF_##compute = 0;                                                 \
  }                                                                            \
  static void nx_c_init_one_##sfx(nx_c_acc *a, void *c) {                        \
    (void)c;                                                                   \
    a->NX_C_ACCF_##compute = 1;                                                 \
  }
#define NX_C_INIT_ZO_NX_C_CAT_FLOAT(sfx, compute) NX_C_INIT_ZO(sfx, compute)
#define NX_C_INIT_ZO_NX_C_CAT_SINT(sfx, compute) NX_C_INIT_ZO(sfx, compute)
#define NX_C_INIT_ZO_NX_C_CAT_UINT(sfx, compute) NX_C_INIT_ZO(sfx, compute)
#define NX_C_INIT_ZO_NX_C_CAT_COMPLEX(sfx, compute) NX_C_INIT_ZO(sfx, compute)
#define NX_C_INIT_ZO_NX_C_CAT_BOOL(sfx, compute)
#define NX_C_INIT_ZO_ROW(sfx, kind, storage, compute, ld, st, cat)              \
  NX_C_INIT_ZO_##cat(sfx, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_INIT_ZO_ROW)
#undef NX_C_INIT_ZO_ROW
#undef NX_C_INIT_ZO_NX_C_CAT_FLOAT
#undef NX_C_INIT_ZO_NX_C_CAT_SINT
#undef NX_C_INIT_ZO_NX_C_CAT_UINT
#undef NX_C_INIT_ZO_NX_C_CAT_COMPLEX
#undef NX_C_INIT_ZO_NX_C_CAT_BOOL
#undef NX_C_INIT_ZO

/* max/min init for max/min (and cummax/cummin); ordered dtypes only. */
#define NX_C_INIT_MM(sfx, compute, maxseed, minseed)                            \
  static void nx_c_init_max_##sfx(nx_c_acc *a, void *c) {                        \
    (void)c;                                                                   \
    a->NX_C_ACCF_##compute = (maxseed);                                         \
  }                                                                            \
  static void nx_c_init_min_##sfx(nx_c_acc *a, void *c) {                        \
    (void)c;                                                                   \
    a->NX_C_ACCF_##compute = (minseed);                                         \
  }
#define NX_C_INIT_MM_NX_C_CAT_FLOAT(sfx, compute)                                \
  NX_C_INIT_MM(sfx, compute, NX_C_MAXSENT_##compute, NX_C_MINSENT_##compute)
#define NX_C_INIT_MM_NX_C_CAT_SINT(sfx, compute)                                 \
  NX_C_INIT_MM(sfx, compute, NX_C_MAXSENT_##compute, NX_C_MINSENT_##compute)
#define NX_C_INIT_MM_NX_C_CAT_UINT(sfx, compute)                                 \
  NX_C_INIT_MM(sfx, compute, NX_C_MAXSENT_##compute, NX_C_MINSENT_##compute)
#define NX_C_INIT_MM_NX_C_CAT_BOOL(sfx, compute) NX_C_INIT_MM(sfx, compute, 0, 1)
#define NX_C_INIT_MM_NX_C_CAT_COMPLEX(sfx, compute)
#define NX_C_INIT_MM_ROW(sfx, kind, storage, compute, ld, st, cat)              \
  NX_C_INIT_MM_##cat(sfx, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_INIT_MM_ROW)
#undef NX_C_INIT_MM_ROW
#undef NX_C_INIT_MM_NX_C_CAT_FLOAT
#undef NX_C_INIT_MM_NX_C_CAT_SINT
#undef NX_C_INIT_MM_NX_C_CAT_UINT
#undef NX_C_INIT_MM_NX_C_CAT_BOOL
#undef NX_C_INIT_MM_NX_C_CAT_COMPLEX
#undef NX_C_INIT_MM

/* ── Step definitions ────────────────────────────────────────────────────────
   Each op emits one step per supported dtype; the category dispatcher picks the
   combine (and leaves unsupported categories empty). */

/* sum: float multi-accumulator, int/complex single accumulator, bool none. */
#define NX_C_SUM_STEP_NX_C_CAT_FLOAT(sfx, storage, compute)                      \
  NX_C_SUM_STEP_FLOAT_DEF(sfx, storage, compute)
#define NX_C_SUM_STEP_NX_C_CAT_SINT(sfx, storage, compute)                       \
  NX_C_REDUCE_STEP(sum, sfx, storage, compute, NX_C_CMB_SUM_WRAP)
#define NX_C_SUM_STEP_NX_C_CAT_UINT(sfx, storage, compute)                       \
  NX_C_REDUCE_STEP(sum, sfx, storage, compute, NX_C_CMB_SUM)
#define NX_C_SUM_STEP_NX_C_CAT_COMPLEX(sfx, storage, compute)                    \
  NX_C_REDUCE_STEP(sum, sfx, storage, compute, NX_C_CMB_SUM)
#define NX_C_SUM_STEP_NX_C_CAT_BOOL(sfx, storage, compute)
#define NX_C_SUM_STEP_ROW(sfx, kind, storage, compute, ld, st, cat)             \
  NX_C_SUM_STEP_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_SUM_STEP_ROW)
#undef NX_C_SUM_STEP_ROW
#undef NX_C_SUM_STEP_NX_C_CAT_FLOAT
#undef NX_C_SUM_STEP_NX_C_CAT_SINT
#undef NX_C_SUM_STEP_NX_C_CAT_UINT
#undef NX_C_SUM_STEP_NX_C_CAT_COMPLEX
#undef NX_C_SUM_STEP_NX_C_CAT_BOOL

/* prod: single accumulator for every arith dtype, bool none. */
#define NX_C_PROD_STEP_NX_C_CAT_FLOAT(sfx, storage, compute)                     \
  NX_C_REDUCE_STEP(prod, sfx, storage, compute, NX_C_CMB_PROD)
#define NX_C_PROD_STEP_NX_C_CAT_SINT(sfx, storage, compute)                      \
  NX_C_REDUCE_STEP(prod, sfx, storage, compute, NX_C_CMB_PROD_WRAP)
#define NX_C_PROD_STEP_NX_C_CAT_UINT(sfx, storage, compute)                      \
  NX_C_REDUCE_STEP(prod, sfx, storage, compute, NX_C_CMB_PROD)
#define NX_C_PROD_STEP_NX_C_CAT_COMPLEX(sfx, storage, compute)                   \
  NX_C_REDUCE_STEP(prod, sfx, storage, compute, NX_C_CMB_PROD)
#define NX_C_PROD_STEP_NX_C_CAT_BOOL(sfx, storage, compute)
#define NX_C_PROD_STEP_ROW(sfx, kind, storage, compute, ld, st, cat)            \
  NX_C_PROD_STEP_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_PROD_STEP_ROW)
#undef NX_C_PROD_STEP_ROW
#undef NX_C_PROD_STEP_NX_C_CAT_FLOAT
#undef NX_C_PROD_STEP_NX_C_CAT_SINT
#undef NX_C_PROD_STEP_NX_C_CAT_UINT
#undef NX_C_PROD_STEP_NX_C_CAT_COMPLEX
#undef NX_C_PROD_STEP_NX_C_CAT_BOOL

/* max: NaN-propagating for float, plain for int, or for bool; complex none. */
#define NX_C_MAX_STEP_NX_C_CAT_FLOAT(sfx, storage, compute)                      \
  NX_C_REDUCE_STEP(max, sfx, storage, compute, NX_C_CMB_MAXF)
#define NX_C_MAX_STEP_NX_C_CAT_SINT(sfx, storage, compute)                       \
  NX_C_REDUCE_STEP(max, sfx, storage, compute, NX_C_CMB_MAXI)
#define NX_C_MAX_STEP_NX_C_CAT_UINT(sfx, storage, compute)                       \
  NX_C_REDUCE_STEP(max, sfx, storage, compute, NX_C_CMB_MAXI)
#define NX_C_MAX_STEP_NX_C_CAT_BOOL(sfx, storage, compute)                       \
  NX_C_REDUCE_STEP(max, sfx, storage, compute, NX_C_CMB_MAXB)
#define NX_C_MAX_STEP_NX_C_CAT_COMPLEX(sfx, storage, compute)
#define NX_C_MAX_STEP_ROW(sfx, kind, storage, compute, ld, st, cat)             \
  NX_C_MAX_STEP_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_MAX_STEP_ROW)
#undef NX_C_MAX_STEP_ROW
#undef NX_C_MAX_STEP_NX_C_CAT_FLOAT
#undef NX_C_MAX_STEP_NX_C_CAT_SINT
#undef NX_C_MAX_STEP_NX_C_CAT_UINT
#undef NX_C_MAX_STEP_NX_C_CAT_BOOL
#undef NX_C_MAX_STEP_NX_C_CAT_COMPLEX

/* min: symmetric to max. */
#define NX_C_MIN_STEP_NX_C_CAT_FLOAT(sfx, storage, compute)                      \
  NX_C_REDUCE_STEP(min, sfx, storage, compute, NX_C_CMB_MINF)
#define NX_C_MIN_STEP_NX_C_CAT_SINT(sfx, storage, compute)                       \
  NX_C_REDUCE_STEP(min, sfx, storage, compute, NX_C_CMB_MINI)
#define NX_C_MIN_STEP_NX_C_CAT_UINT(sfx, storage, compute)                       \
  NX_C_REDUCE_STEP(min, sfx, storage, compute, NX_C_CMB_MINI)
#define NX_C_MIN_STEP_NX_C_CAT_BOOL(sfx, storage, compute)                       \
  NX_C_REDUCE_STEP(min, sfx, storage, compute, NX_C_CMB_MINB)
#define NX_C_MIN_STEP_NX_C_CAT_COMPLEX(sfx, storage, compute)
#define NX_C_MIN_STEP_ROW(sfx, kind, storage, compute, ld, st, cat)             \
  NX_C_MIN_STEP_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_MIN_STEP_ROW)
#undef NX_C_MIN_STEP_ROW
#undef NX_C_MIN_STEP_NX_C_CAT_FLOAT
#undef NX_C_MIN_STEP_NX_C_CAT_SINT
#undef NX_C_MIN_STEP_NX_C_CAT_UINT
#undef NX_C_MIN_STEP_NX_C_CAT_BOOL
#undef NX_C_MIN_STEP_NX_C_CAT_COMPLEX

/* Streaming steps, one per op over its supported categories. sum uses a single
   per-lane accumulator (not the per-output float multi-accumulator: the lanes
   provide the vectorization here); the rest reuse the same combine macros as
   their per-output steps, so the scalar semantics — including NaN propagation
   for float max/min — are identical. */
#define NX_C_SUM_STREAM_NX_C_CAT_FLOAT(sfx, storage, compute)                    \
  NX_C_STREAM_STEP(sum, sfx, storage, compute, NX_C_CMB_SUM)
#define NX_C_SUM_STREAM_NX_C_CAT_SINT(sfx, storage, compute)                     \
  NX_C_STREAM_STEP(sum, sfx, storage, compute, NX_C_CMB_SUM_WRAP)
#define NX_C_SUM_STREAM_NX_C_CAT_UINT(sfx, storage, compute)                     \
  NX_C_STREAM_STEP(sum, sfx, storage, compute, NX_C_CMB_SUM)
#define NX_C_SUM_STREAM_NX_C_CAT_COMPLEX(sfx, storage, compute)                  \
  NX_C_STREAM_STEP(sum, sfx, storage, compute, NX_C_CMB_SUM)
#define NX_C_SUM_STREAM_NX_C_CAT_BOOL(sfx, storage, compute)
#define NX_C_SUM_STREAM_ROW(sfx, kind, storage, compute, ld, st, cat)           \
  NX_C_SUM_STREAM_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_SUM_STREAM_ROW)
#undef NX_C_SUM_STREAM_ROW
#undef NX_C_SUM_STREAM_NX_C_CAT_FLOAT
#undef NX_C_SUM_STREAM_NX_C_CAT_SINT
#undef NX_C_SUM_STREAM_NX_C_CAT_UINT
#undef NX_C_SUM_STREAM_NX_C_CAT_COMPLEX
#undef NX_C_SUM_STREAM_NX_C_CAT_BOOL

#define NX_C_PROD_STREAM_NX_C_CAT_FLOAT(sfx, storage, compute)                   \
  NX_C_STREAM_STEP(prod, sfx, storage, compute, NX_C_CMB_PROD)
#define NX_C_PROD_STREAM_NX_C_CAT_SINT(sfx, storage, compute)                    \
  NX_C_STREAM_STEP(prod, sfx, storage, compute, NX_C_CMB_PROD_WRAP)
#define NX_C_PROD_STREAM_NX_C_CAT_UINT(sfx, storage, compute)                    \
  NX_C_STREAM_STEP(prod, sfx, storage, compute, NX_C_CMB_PROD)
#define NX_C_PROD_STREAM_NX_C_CAT_COMPLEX(sfx, storage, compute)                 \
  NX_C_STREAM_STEP(prod, sfx, storage, compute, NX_C_CMB_PROD)
#define NX_C_PROD_STREAM_NX_C_CAT_BOOL(sfx, storage, compute)
#define NX_C_PROD_STREAM_ROW(sfx, kind, storage, compute, ld, st, cat)          \
  NX_C_PROD_STREAM_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_PROD_STREAM_ROW)
#undef NX_C_PROD_STREAM_ROW
#undef NX_C_PROD_STREAM_NX_C_CAT_FLOAT
#undef NX_C_PROD_STREAM_NX_C_CAT_SINT
#undef NX_C_PROD_STREAM_NX_C_CAT_UINT
#undef NX_C_PROD_STREAM_NX_C_CAT_COMPLEX
#undef NX_C_PROD_STREAM_NX_C_CAT_BOOL

#define NX_C_MAX_STREAM_NX_C_CAT_FLOAT(sfx, storage, compute)                    \
  NX_C_STREAM_STEP(max, sfx, storage, compute, NX_C_CMB_MAXF)
#define NX_C_MAX_STREAM_NX_C_CAT_SINT(sfx, storage, compute)                     \
  NX_C_STREAM_STEP(max, sfx, storage, compute, NX_C_CMB_MAXI)
#define NX_C_MAX_STREAM_NX_C_CAT_UINT(sfx, storage, compute)                     \
  NX_C_STREAM_STEP(max, sfx, storage, compute, NX_C_CMB_MAXI)
#define NX_C_MAX_STREAM_NX_C_CAT_BOOL(sfx, storage, compute)                     \
  NX_C_STREAM_STEP(max, sfx, storage, compute, NX_C_CMB_MAXB)
#define NX_C_MAX_STREAM_NX_C_CAT_COMPLEX(sfx, storage, compute)
#define NX_C_MAX_STREAM_ROW(sfx, kind, storage, compute, ld, st, cat)           \
  NX_C_MAX_STREAM_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_MAX_STREAM_ROW)
#undef NX_C_MAX_STREAM_ROW
#undef NX_C_MAX_STREAM_NX_C_CAT_FLOAT
#undef NX_C_MAX_STREAM_NX_C_CAT_SINT
#undef NX_C_MAX_STREAM_NX_C_CAT_UINT
#undef NX_C_MAX_STREAM_NX_C_CAT_BOOL
#undef NX_C_MAX_STREAM_NX_C_CAT_COMPLEX

#define NX_C_MIN_STREAM_NX_C_CAT_FLOAT(sfx, storage, compute)                    \
  NX_C_STREAM_STEP(min, sfx, storage, compute, NX_C_CMB_MINF)
#define NX_C_MIN_STREAM_NX_C_CAT_SINT(sfx, storage, compute)                     \
  NX_C_STREAM_STEP(min, sfx, storage, compute, NX_C_CMB_MINI)
#define NX_C_MIN_STREAM_NX_C_CAT_UINT(sfx, storage, compute)                     \
  NX_C_STREAM_STEP(min, sfx, storage, compute, NX_C_CMB_MINI)
#define NX_C_MIN_STREAM_NX_C_CAT_BOOL(sfx, storage, compute)                     \
  NX_C_STREAM_STEP(min, sfx, storage, compute, NX_C_CMB_MINB)
#define NX_C_MIN_STREAM_NX_C_CAT_COMPLEX(sfx, storage, compute)
#define NX_C_MIN_STREAM_ROW(sfx, kind, storage, compute, ld, st, cat)           \
  NX_C_MIN_STREAM_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_MIN_STREAM_ROW)
#undef NX_C_MIN_STREAM_ROW
#undef NX_C_MIN_STREAM_NX_C_CAT_FLOAT
#undef NX_C_MIN_STREAM_NX_C_CAT_SINT
#undef NX_C_MIN_STREAM_NX_C_CAT_UINT
#undef NX_C_MIN_STREAM_NX_C_CAT_BOOL
#undef NX_C_MIN_STREAM_NX_C_CAT_COMPLEX

/* argmax / argmin: float first-index-with-NaN, int/bool first-index. */
#define NX_C_ARGMAX_STEP_NX_C_CAT_FLOAT(sfx, storage, compute)                   \
  NX_C_ARG_STEP(argmax, sfx, storage, compute, NX_C_ACMP_MAXF)
#define NX_C_ARGMAX_STEP_NX_C_CAT_SINT(sfx, storage, compute)                    \
  NX_C_ARG_STEP(argmax, sfx, storage, compute, NX_C_ACMP_MAXI)
#define NX_C_ARGMAX_STEP_NX_C_CAT_UINT(sfx, storage, compute)                    \
  NX_C_ARG_STEP(argmax, sfx, storage, compute, NX_C_ACMP_MAXI)
#define NX_C_ARGMAX_STEP_NX_C_CAT_BOOL(sfx, storage, compute)                    \
  NX_C_ARG_STEP(argmax, sfx, storage, compute, NX_C_ACMP_MAXI)
#define NX_C_ARGMAX_STEP_NX_C_CAT_COMPLEX(sfx, storage, compute)
#define NX_C_ARGMAX_STEP_ROW(sfx, kind, storage, compute, ld, st, cat)          \
  NX_C_ARGMAX_STEP_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_ARGMAX_STEP_ROW)
#undef NX_C_ARGMAX_STEP_ROW
#undef NX_C_ARGMAX_STEP_NX_C_CAT_FLOAT
#undef NX_C_ARGMAX_STEP_NX_C_CAT_SINT
#undef NX_C_ARGMAX_STEP_NX_C_CAT_UINT
#undef NX_C_ARGMAX_STEP_NX_C_CAT_BOOL
#undef NX_C_ARGMAX_STEP_NX_C_CAT_COMPLEX

#define NX_C_ARGMIN_STEP_NX_C_CAT_FLOAT(sfx, storage, compute)                   \
  NX_C_ARG_STEP(argmin, sfx, storage, compute, NX_C_ACMP_MINF)
#define NX_C_ARGMIN_STEP_NX_C_CAT_SINT(sfx, storage, compute)                    \
  NX_C_ARG_STEP(argmin, sfx, storage, compute, NX_C_ACMP_MINI)
#define NX_C_ARGMIN_STEP_NX_C_CAT_UINT(sfx, storage, compute)                    \
  NX_C_ARG_STEP(argmin, sfx, storage, compute, NX_C_ACMP_MINI)
#define NX_C_ARGMIN_STEP_NX_C_CAT_BOOL(sfx, storage, compute)                    \
  NX_C_ARG_STEP(argmin, sfx, storage, compute, NX_C_ACMP_MINI)
#define NX_C_ARGMIN_STEP_NX_C_CAT_COMPLEX(sfx, storage, compute)
#define NX_C_ARGMIN_STEP_ROW(sfx, kind, storage, compute, ld, st, cat)          \
  NX_C_ARGMIN_STEP_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_ARGMIN_STEP_ROW)
#undef NX_C_ARGMIN_STEP_ROW
#undef NX_C_ARGMIN_STEP_NX_C_CAT_FLOAT
#undef NX_C_ARGMIN_STEP_NX_C_CAT_SINT
#undef NX_C_ARGMIN_STEP_NX_C_CAT_UINT
#undef NX_C_ARGMIN_STEP_NX_C_CAT_BOOL
#undef NX_C_ARGMIN_STEP_NX_C_CAT_COMPLEX

/* cumsum / cumprod: arith dtypes; cummax / cummin: ordered dtypes. */
#define NX_C_CUMSUM_STEP_NX_C_CAT_FLOAT(sfx, storage, compute)                   \
  NX_C_SCAN_STEP(cumsum, sfx, storage, compute, NX_C_CMB_SUM)
#define NX_C_CUMSUM_STEP_NX_C_CAT_SINT(sfx, storage, compute)                    \
  NX_C_SCAN_STEP(cumsum, sfx, storage, compute, NX_C_CMB_SUM_WRAP)
#define NX_C_CUMSUM_STEP_NX_C_CAT_UINT(sfx, storage, compute)                    \
  NX_C_SCAN_STEP(cumsum, sfx, storage, compute, NX_C_CMB_SUM)
#define NX_C_CUMSUM_STEP_NX_C_CAT_COMPLEX(sfx, storage, compute)                 \
  NX_C_SCAN_STEP(cumsum, sfx, storage, compute, NX_C_CMB_SUM)
#define NX_C_CUMSUM_STEP_NX_C_CAT_BOOL(sfx, storage, compute)
#define NX_C_CUMSUM_STEP_ROW(sfx, kind, storage, compute, ld, st, cat)          \
  NX_C_CUMSUM_STEP_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CUMSUM_STEP_ROW)
#undef NX_C_CUMSUM_STEP_ROW
#undef NX_C_CUMSUM_STEP_NX_C_CAT_FLOAT
#undef NX_C_CUMSUM_STEP_NX_C_CAT_SINT
#undef NX_C_CUMSUM_STEP_NX_C_CAT_UINT
#undef NX_C_CUMSUM_STEP_NX_C_CAT_COMPLEX
#undef NX_C_CUMSUM_STEP_NX_C_CAT_BOOL

#define NX_C_CUMPROD_STEP_NX_C_CAT_FLOAT(sfx, storage, compute)                  \
  NX_C_SCAN_STEP(cumprod, sfx, storage, compute, NX_C_CMB_PROD)
#define NX_C_CUMPROD_STEP_NX_C_CAT_SINT(sfx, storage, compute)                   \
  NX_C_SCAN_STEP(cumprod, sfx, storage, compute, NX_C_CMB_PROD_WRAP)
#define NX_C_CUMPROD_STEP_NX_C_CAT_UINT(sfx, storage, compute)                   \
  NX_C_SCAN_STEP(cumprod, sfx, storage, compute, NX_C_CMB_PROD)
#define NX_C_CUMPROD_STEP_NX_C_CAT_COMPLEX(sfx, storage, compute)                \
  NX_C_SCAN_STEP(cumprod, sfx, storage, compute, NX_C_CMB_PROD)
#define NX_C_CUMPROD_STEP_NX_C_CAT_BOOL(sfx, storage, compute)
#define NX_C_CUMPROD_STEP_ROW(sfx, kind, storage, compute, ld, st, cat)         \
  NX_C_CUMPROD_STEP_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CUMPROD_STEP_ROW)
#undef NX_C_CUMPROD_STEP_ROW
#undef NX_C_CUMPROD_STEP_NX_C_CAT_FLOAT
#undef NX_C_CUMPROD_STEP_NX_C_CAT_SINT
#undef NX_C_CUMPROD_STEP_NX_C_CAT_UINT
#undef NX_C_CUMPROD_STEP_NX_C_CAT_COMPLEX
#undef NX_C_CUMPROD_STEP_NX_C_CAT_BOOL

#define NX_C_CUMMAX_STEP_NX_C_CAT_FLOAT(sfx, storage, compute)                   \
  NX_C_SCAN_STEP(cummax, sfx, storage, compute, NX_C_CMB_MAXF)
#define NX_C_CUMMAX_STEP_NX_C_CAT_SINT(sfx, storage, compute)                    \
  NX_C_SCAN_STEP(cummax, sfx, storage, compute, NX_C_CMB_MAXI)
#define NX_C_CUMMAX_STEP_NX_C_CAT_UINT(sfx, storage, compute)                    \
  NX_C_SCAN_STEP(cummax, sfx, storage, compute, NX_C_CMB_MAXI)
#define NX_C_CUMMAX_STEP_NX_C_CAT_BOOL(sfx, storage, compute)                    \
  NX_C_SCAN_STEP(cummax, sfx, storage, compute, NX_C_CMB_MAXB)
#define NX_C_CUMMAX_STEP_NX_C_CAT_COMPLEX(sfx, storage, compute)
#define NX_C_CUMMAX_STEP_ROW(sfx, kind, storage, compute, ld, st, cat)          \
  NX_C_CUMMAX_STEP_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CUMMAX_STEP_ROW)
#undef NX_C_CUMMAX_STEP_ROW
#undef NX_C_CUMMAX_STEP_NX_C_CAT_FLOAT
#undef NX_C_CUMMAX_STEP_NX_C_CAT_SINT
#undef NX_C_CUMMAX_STEP_NX_C_CAT_UINT
#undef NX_C_CUMMAX_STEP_NX_C_CAT_BOOL
#undef NX_C_CUMMAX_STEP_NX_C_CAT_COMPLEX

#define NX_C_CUMMIN_STEP_NX_C_CAT_FLOAT(sfx, storage, compute)                   \
  NX_C_SCAN_STEP(cummin, sfx, storage, compute, NX_C_CMB_MINF)
#define NX_C_CUMMIN_STEP_NX_C_CAT_SINT(sfx, storage, compute)                    \
  NX_C_SCAN_STEP(cummin, sfx, storage, compute, NX_C_CMB_MINI)
#define NX_C_CUMMIN_STEP_NX_C_CAT_UINT(sfx, storage, compute)                    \
  NX_C_SCAN_STEP(cummin, sfx, storage, compute, NX_C_CMB_MINI)
#define NX_C_CUMMIN_STEP_NX_C_CAT_BOOL(sfx, storage, compute)                    \
  NX_C_SCAN_STEP(cummin, sfx, storage, compute, NX_C_CMB_MINB)
#define NX_C_CUMMIN_STEP_NX_C_CAT_COMPLEX(sfx, storage, compute)
#define NX_C_CUMMIN_STEP_ROW(sfx, kind, storage, compute, ld, st, cat)          \
  NX_C_CUMMIN_STEP_##cat(sfx, storage, compute)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CUMMIN_STEP_ROW)
#undef NX_C_CUMMIN_STEP_ROW
#undef NX_C_CUMMIN_STEP_NX_C_CAT_FLOAT
#undef NX_C_CUMMIN_STEP_NX_C_CAT_SINT
#undef NX_C_CUMMIN_STEP_NX_C_CAT_UINT
#undef NX_C_CUMMIN_STEP_NX_C_CAT_BOOL
#undef NX_C_CUMMIN_STEP_NX_C_CAT_COMPLEX

/* ── Dispatch tables ─────────────────────────────────────────────────────────
   A slot is filled only for the categories an op supports (arith = numeric
   except bool; ord = float/int/bool, i.e. no complex); the rest stay NULL and
   the engine reports them. NX_C_SLOT emits `[dt] = prefix_sfx,` when the support
   flag is 1 and nothing when 0. */
#define NX_C_PASTE(a, b) NX_C_PASTE_(a, b)
#define NX_C_PASTE_(a, b) a##b
#define NX_C_SLOT_0(sfx, prefix)
#define NX_C_SLOT_1(sfx, prefix) [NX_C_DTYPE_##sfx] = prefix##_##sfx,
#define NX_C_SLOT(sup, sfx, prefix) NX_C_PASTE(NX_C_SLOT_, sup)(sfx, prefix)

#define NX_C_SUP_ARITH_NX_C_CAT_FLOAT 1
#define NX_C_SUP_ARITH_NX_C_CAT_SINT 1
#define NX_C_SUP_ARITH_NX_C_CAT_UINT 1
#define NX_C_SUP_ARITH_NX_C_CAT_COMPLEX 1
#define NX_C_SUP_ARITH_NX_C_CAT_BOOL 0
#define NX_C_SUP_ORD_NX_C_CAT_FLOAT 1
#define NX_C_SUP_ORD_NX_C_CAT_SINT 1
#define NX_C_SUP_ORD_NX_C_CAT_UINT 1
#define NX_C_SUP_ORD_NX_C_CAT_COMPLEX 0
#define NX_C_SUP_ORD_NX_C_CAT_BOOL 1

static const nx_c_fold_table nx_c_sum_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_init_zero)
    .init = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_sum_step)
    .step = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_fini)
    .fini = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_fold_table nx_c_prod_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_init_one)
    .init = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_prod_step)
    .step = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_fini)
    .fini = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_fold_table nx_c_max_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_init_max)
    .init = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_max_step)
    .step = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_fini)
    .fini = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_fold_table nx_c_min_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_init_min)
    .init = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_min_step)
    .step = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_fini)
    .fini = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

/* Streaming tables mirror the fold tables' support masks: `stream` per op,
   `scatter` per dtype (the shared instances). A slot is filled only where the
   op streams that dtype; NULL elsewhere makes the driver fall back to the
   per-output path. */
static const nx_c_stream_table nx_c_sum_stream_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_sum_stream)
    .stream = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_scatter)
    .scatter = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_stream_table nx_c_prod_stream_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_prod_stream)
    .stream = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_scatter)
    .scatter = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_stream_table nx_c_max_stream_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_max_stream)
    .stream = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_scatter)
    .scatter = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_stream_table nx_c_min_stream_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_min_stream)
    .stream = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_scatter)
    .scatter = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_arg_table nx_c_argmax_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_argmax_step)
    .step = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_arg_table nx_c_argmin_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_argmin_step)
    .step = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_scan_table nx_c_cumsum_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_init_zero)
    .init = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_cumsum_step)
    .step = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_scan_table nx_c_cumprod_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_init_one)
    .init = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ARITH_##cat, sfx, nx_c_cumprod_step)
    .step = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_scan_table nx_c_cummax_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_init_max)
    .init = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_cummax_step)
    .step = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

static const nx_c_scan_table nx_c_cummin_table = {
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_init_min)
    .init = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
#define NX_C_G(sfx, kind, storage, compute, ld, st, cat)                        \
  NX_C_SLOT(NX_C_SUP_ORD_##cat, sfx, nx_c_cummin_step)
    .step = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_G)},
#undef NX_C_G
};

/* ── FFI stubs ────────────────────────────────────────────────────────────────
   Reductions/scans are memory-bound (read the whole extent, trivial per-element
   work), so all carry NX_C_COST_BANDWIDTH. Reductions dispatch on the input
   dtype and preserve it (backend_intf reduce_* returns the input dtype);
   argmax/argmin write int32 (the engine's fini). Each stub calls exactly one
   funnel, which validates, dispatches, runs, and raises with the op name.

   `no_identity` is the op's empty-axis policy (nx_c_engine.h): max/min have no
   identity for an empty reduced extent, so they pass true and the fold driver
   returns NX_C_ERR_EMPTY_REDUCE before any kernel runs (never leaking the init
   sentinel) — but only when there are outputs to fill; an empty result reduces
   nothing and is a no-op. sum/prod have identities 0/1, so they pass false and
   an empty extent stores the identity. The argreduce funnel rejects an empty or
   oversized axis itself. */

#define NX_C_FOLD_STUB(cname, opname, table, stream_table, no_identity)         \
  CAMLprim value caml_nx_c_##cname(value vout, value vin, value vaxes) {        \
    CAMLparam3(vout, vin, vaxes);                                              \
    nx_c_fold_funnel((opname), &(table), &(stream_table), NX_C_COST_BANDWIDTH,   \
                    vout, vin, vaxes, (no_identity), NULL);                    \
    CAMLreturn(Val_unit);                                                      \
  }

#define NX_C_ARG_STUB(cname, opname, table)                                     \
  CAMLprim value caml_nx_c_##cname(value vout, value vin, value vaxis) {        \
    CAMLparam3(vout, vin, vaxis);                                              \
    nx_c_argreduce_funnel((opname), &(table), NX_C_COST_BANDWIDTH, vout, vin,    \
                         Int_val(vaxis), NULL);                                \
    CAMLreturn(Val_unit);                                                      \
  }

#define NX_C_SCAN_STUB(cname, opname, table)                                    \
  CAMLprim value caml_nx_c_##cname(value vout, value vin, value vaxis) {        \
    CAMLparam3(vout, vin, vaxis);                                              \
    nx_c_scan_funnel((opname), &(table), NX_C_COST_BANDWIDTH, vout, vin,         \
                    Int_val(vaxis), NULL);                                     \
    CAMLreturn(Val_unit);                                                      \
  }

NX_C_FOLD_STUB(reduce_sum, "reduce_sum", nx_c_sum_table, nx_c_sum_stream_table,
              false)
NX_C_FOLD_STUB(reduce_prod, "reduce_prod", nx_c_prod_table, nx_c_prod_stream_table,
              false)
NX_C_FOLD_STUB(reduce_max, "reduce_max", nx_c_max_table, nx_c_max_stream_table,
              true)
NX_C_FOLD_STUB(reduce_min, "reduce_min", nx_c_min_table, nx_c_min_stream_table,
              true)
NX_C_ARG_STUB(argmax, "argmax", nx_c_argmax_table)
NX_C_ARG_STUB(argmin, "argmin", nx_c_argmin_table)
NX_C_SCAN_STUB(cumsum, "cumsum", nx_c_cumsum_table)
NX_C_SCAN_STUB(cumprod, "cumprod", nx_c_cumprod_table)
NX_C_SCAN_STUB(cummax, "cummax", nx_c_cummax_table)
NX_C_SCAN_STUB(cummin, "cummin", nx_c_cummin_table)
