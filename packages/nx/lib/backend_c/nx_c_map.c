/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_map.c — the map kernel family: elementwise unary, binary, comparison,
   where, and cast. Every kernel is a 5-line inner loop over one 1-D run honoring
   the nx_c_map_loop ABI (nx_c.h); the engine (nx_c_engine.c) owns coalescing,
   strategy, threading, dispatch null-checks, and the funnel. Kernels state scalar
   semantics only, never touch the OCaml runtime, and never fail — the backend
   contract encodes the normative behavior.

   Generation is table-driven: NX_C_FOR_EACH_COMPUTE_DTYPE walks the compute rows
   of the one dtype table (nx_c.h), so a per-op dispatch table is a set of
   designated initializers whose absent slots are NULL — the engine turns NULL
   into a clean NX_C_ERR_UNSUPPORTED_DTYPE / NX_C_ERR_PACKED status. Cast is the one
   pair-indexed op: a src×dst matrix of specialized converters (src LOAD ->
   intermediate -> dst STORE), with int4/uint4 handled on a separate contiguous
   serial nibble path (they are storage-only; a strided int4 cast is rejected).

   Only the family stubs (bottom) reach the OCaml runtime, via the engine funnel
   or the sanctioned nx_c_raise / nx_c_raise_status raisers — never a kernel. */

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h"

/* ── Fast/strided contiguous fast path ─────────────────────────────────────

   The engine hands byte steps; for a contiguous run every step equals the
   element size. A kernel branches on that once and, on the contiguous branch,
   walks typed pointers (compile-time unit stride) so clang autovectorizes the
   f32/f64/int arithmetic ops; the strided branch keeps the generic byte walk.
   Both branches share one scalar expression. `es` folds to a constant, so only
   the compare uses it — the hot loop is plain array indexing. */

/* Correctly libm-suffixed function for a compute type: NX_C_MFN(sin, float) ->
   sinf, NX_C_MFN(sin, double) -> sin, NX_C_MFN(cexp, nx_c_complex32) -> cexpf. The
   suffix lives with the compute type, not per op, so a new float dtype needs no
   change here. */
#define NX_C_PASTE_(a, b) a##b
#define NX_C_PASTE(a, b) NX_C_PASTE_(a, b)
#define NX_C_MSUF_float f
#define NX_C_MSUF_double
#define NX_C_MSUF_nx_c_complex32 f
#define NX_C_MSUF_nx_c_complex64
#define NX_C_MFN(fn, compute) NX_C_PASTE(fn, NX_C_MSUF_##compute)

/* Map-table entry for a generated kernel. */
#define NX_C_TE(op, sfx) [NX_C_DTYPE_##sfx] = nx_c_##op##_##sfx,

/* Unary kernel over one dtype. The _I layer forces `op` (which may be a macro
   like NX_C_CUROP) to expand before it is pasted into the kernel name. */
#define NX_C_UK(op, sfx, storage, compute, ld, st, EXPR)                        \
  NX_C_UK_I(op, sfx, storage, compute, ld, st, EXPR)
#define NX_C_UK_I(op, sfx, storage, compute, ld, st, EXPR)                      \
  static void nx_c_##op##_##sfx(char *const *pp, const int64_t *ssx,            \
                               int64_t nn, void *ctx) {                        \
    (void)ctx;                                                                 \
    char *out = pp[0];                                                         \
    const char *in0 = pp[1];                                                   \
    const int64_t so = ssx[0], sa = ssx[1];                                    \
    const int64_t es = (int64_t)sizeof(storage);                              \
    if (so == es && sa == es) {                                               \
      storage *pO = (storage *)out;                                           \
      const storage *pA = (const storage *)in0;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute vx = (compute)ld(pA[i]);                                       \
        pO[i] = (storage)st(EXPR);                                             \
      }                                                                        \
    } else {                                                                   \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute vx = nx_c_ld_##sfx(in0 + i * sa);                              \
        nx_c_st_##sfx(out + i * so, (EXPR));                                    \
      }                                                                        \
    }                                                                          \
  }

/* Binary kernel over one dtype. */
#define NX_C_BK(op, sfx, storage, compute, ld, st, EXPR)                        \
  NX_C_BK_I(op, sfx, storage, compute, ld, st, EXPR)
#define NX_C_BK_I(op, sfx, storage, compute, ld, st, EXPR)                      \
  static void nx_c_##op##_##sfx(char *const *pp, const int64_t *ssx,            \
                               int64_t nn, void *ctx) {                        \
    (void)ctx;                                                                 \
    char *out = pp[0];                                                         \
    const char *in0 = pp[1];                                                   \
    const char *in1 = pp[2];                                                   \
    const int64_t so = ssx[0], sa = ssx[1], sb = ssx[2];                       \
    const int64_t es = (int64_t)sizeof(storage);                              \
    if (so == es && sa == es && sb == es) {                                   \
      storage *pO = (storage *)out;                                           \
      const storage *pA = (const storage *)in0;                               \
      const storage *pB = (const storage *)in1;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = (compute)ld(pA[i]);                                       \
        compute vb = (compute)ld(pB[i]);                                       \
        pO[i] = (storage)st(EXPR);                                             \
      }                                                                        \
    } else {                                                                   \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = nx_c_ld_##sfx(in0 + i * sa);                              \
        compute vb = nx_c_ld_##sfx(in1 + i * sb);                              \
        nx_c_st_##sfx(out + i * so, (EXPR));                                    \
      }                                                                        \
    }                                                                          \
  }

/* Comparison kernel: value inputs, bool (uint8 0/1) output. Dispatched on the
   INPUT dtype (the stub calls nx_c_map_run with the input dtype), so `storage`
   here is the input storage and the output step is 1 byte. */
#define NX_C_CMPK(op, sfx, storage, compute, ld, EXPR)                          \
  NX_C_CMPK_I(op, sfx, storage, compute, ld, EXPR)
#define NX_C_CMPK_I(op, sfx, storage, compute, ld, EXPR)                        \
  static void nx_c_##op##_##sfx(char *const *pp, const int64_t *ssx,            \
                               int64_t nn, void *ctx) {                        \
    (void)ctx;                                                                 \
    char *out = pp[0];                                                         \
    const char *in0 = pp[1];                                                   \
    const char *in1 = pp[2];                                                   \
    const int64_t so = ssx[0], sa = ssx[1], sb = ssx[2];                       \
    const int64_t es = (int64_t)sizeof(storage);                              \
    if (so == 1 && sa == es && sb == es) {                                    \
      uint8_t *pO = (uint8_t *)out;                                           \
      const storage *pA = (const storage *)in0;                               \
      const storage *pB = (const storage *)in1;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = (compute)ld(pA[i]);                                       \
        compute vb = (compute)ld(pB[i]);                                       \
        pO[i] = (uint8_t)(EXPR);                                               \
      }                                                                        \
    } else {                                                                   \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = nx_c_ld_##sfx(in0 + i * sa);                              \
        compute vb = nx_c_ld_##sfx(in1 + i * sb);                              \
        *(uint8_t *)(out + i * so) = (uint8_t)(EXPR);                          \
      }                                                                        \
    }                                                                          \
  }

/* ── Integer power and 4-bit saturating narrow (helpers a kernel body calls) ─ */

/* base^e in the compute width, wrapping on overflow (unsigned arithmetic avoids
   signed-overflow UB; the cast back is modular). A negative exponent has no
   integer value except for |base| == 1, matching a total, never-trapping op. */
static int64_t nx_c_ipow_s(int64_t base, int64_t e) {
  if (e < 0) return (base == 1) ? 1 : (base == -1) ? ((e & 1) ? -1 : 1) : 0;
  uint64_t b = (uint64_t)base, r = 1;
  while (e > 0) {
    if (e & 1) r *= b;
    e >>= 1;
    if (e) b *= b;
  }
  return (int64_t)r;
}
static uint64_t nx_c_ipow_u(uint64_t b, uint64_t e) {
  uint64_t r = 1;
  while (e > 0) {
    if (e & 1) r *= b;
    e >>= 1;
    if (e) b *= b;
  }
  return r;
}

/* Saturating double -> signed/unsigned 4-bit (NaN -> 0), the int4/uint4 analogue
   of nx_c.h's nx_c_f2i for the storage-only packed cast. */
static inline int nx_c_f2i4_s(double v) {
  if (isnan(v)) return 0;
  if (v <= -8.0) return -8;
  if (v >= 7.0) return 7;
  return (int)v;
}
static inline int nx_c_f2i4_u(double v) {
  if (isnan(v)) return 0;
  if (v <= 0.0) return 0;
  if (v >= 15.0) return 15;
  return (int)v;
}

/* ── Cast conversion policy (normative precision rules) ─────────────────────

   NX_C_CASTVAL(dcat, dcompute, dsfx, scat, v) turns a loaded src compute value v
   (category scat) into the dst compute value to store (category dcat). One
   expression per (dcat, scat) family:
     * real dst from complex src takes the real part (NX_C_SREAL);
     * int dst from a float/complex src saturates through nx_c_f2i (NX_C_TOINT),
       from an int/bool src wraps on store (modular);
     * int<->int rides the 64-bit compute widths (int64/uint64), so u64<->i64
       and u64->float go direct with no signed detour;
     * complex dst takes real src as re+0i;
     * bool dst is (v != 0), so NaN -> true. */
#define NX_C_SREAL_NX_C_CAT_SINT(v) (v)
#define NX_C_SREAL_NX_C_CAT_UINT(v) (v)
#define NX_C_SREAL_NX_C_CAT_FLOAT(v) (v)
#define NX_C_SREAL_NX_C_CAT_BOOL(v) (v)
#define NX_C_SREAL_NX_C_CAT_COMPLEX(v) creal(v)

#define NX_C_TOINT_NX_C_CAT_SINT(dsfx, v) (v)
#define NX_C_TOINT_NX_C_CAT_UINT(dsfx, v) (v)
#define NX_C_TOINT_NX_C_CAT_BOOL(dsfx, v) (v)
#define NX_C_TOINT_NX_C_CAT_FLOAT(dsfx, v) nx_c_f2i_##dsfx((double)(v))
#define NX_C_TOINT_NX_C_CAT_COMPLEX(dsfx, v) nx_c_f2i_##dsfx((double)creal(v))

/* A float dst narrower than its float compute type double-rounds: e.g. f64->f16
   is (float)(double) then float_to_half, since float_to_half is the only f16
   converter. The extra rounding is sub-ulp and within the conformance tolerance;
   noted for honesty, not correctness. */
#define NX_C_CASTVAL_NX_C_CAT_FLOAT(dcompute, dsfx, scat, v)                     \
  ((dcompute)(NX_C_SREAL_##scat(v)))
#define NX_C_CASTVAL_NX_C_CAT_COMPLEX(dcompute, dsfx, scat, v) ((dcompute)(v))
#define NX_C_CASTVAL_NX_C_CAT_BOOL(dcompute, dsfx, scat, v)                      \
  ((dcompute)((v) != 0))
#define NX_C_CASTVAL_NX_C_CAT_SINT(dcompute, dsfx, scat, v)                      \
  ((dcompute)(NX_C_TOINT_##scat(dsfx, v)))
#define NX_C_CASTVAL_NX_C_CAT_UINT(dcompute, dsfx, scat, v)                      \
  ((dcompute)(NX_C_TOINT_##scat(dsfx, v)))

/* ── Shared per-op table row generators ────────────────────────────────────
   Each maps a dtype row to a designated initializer for the categories the op
   supports (NX_C_CUROP is the op being built), and to nothing for the rest —
   yielding the NULL slots the engine reads as unsupported. */
#define NX_C_TN_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TN_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TN_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TN_NX_C_CAT_COMPLEX(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TN_NX_C_CAT_BOOL(op, sfx)
#define NX_C_TROW_NUM(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_TN_##cat(NX_C_CUROP, sfx) /* sint,uint,float,complex */

#define NX_C_TFC_NX_C_CAT_SINT(op, sfx)
#define NX_C_TFC_NX_C_CAT_UINT(op, sfx)
#define NX_C_TFC_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TFC_NX_C_CAT_COMPLEX(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TFC_NX_C_CAT_BOOL(op, sfx)
#define NX_C_TROW_FC(sfx, kind, storage, compute, ld, st, cat)                  \
  NX_C_TFC_##cat(NX_C_CUROP, sfx) /* float,complex */

#define NX_C_TFO_NX_C_CAT_SINT(op, sfx)
#define NX_C_TFO_NX_C_CAT_UINT(op, sfx)
#define NX_C_TFO_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TFO_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TFO_NX_C_CAT_BOOL(op, sfx)
#define NX_C_TROW_FLOAT(sfx, kind, storage, compute, ld, st, cat)               \
  NX_C_TFO_##cat(NX_C_CUROP, sfx) /* float only */

#define NX_C_TIF_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TIF_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TIF_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TIF_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TIF_NX_C_CAT_BOOL(op, sfx)
#define NX_C_TROW_INTF(sfx, kind, storage, compute, ld, st, cat)                \
  NX_C_TIF_##cat(NX_C_CUROP, sfx) /* sint,uint,float */

#define NX_C_TMM_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TMM_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TMM_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TMM_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TMM_NX_C_CAT_BOOL(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TROW_MINMAX(sfx, kind, storage, compute, ld, st, cat)              \
  NX_C_TMM_##cat(NX_C_CUROP, sfx) /* sint,uint,float,bool */

#define NX_C_TBW_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TBW_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TBW_NX_C_CAT_FLOAT(op, sfx)
#define NX_C_TBW_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TBW_NX_C_CAT_BOOL(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TROW_BITWISE(sfx, kind, storage, compute, ld, st, cat)             \
  NX_C_TBW_##cat(NX_C_CUROP, sfx) /* sint,uint,bool */

#define NX_C_TSH_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TSH_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TSH_NX_C_CAT_FLOAT(op, sfx)
#define NX_C_TSH_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TSH_NX_C_CAT_BOOL(op, sfx)
#define NX_C_TROW_SHIFT(sfx, kind, storage, compute, ld, st, cat)               \
  NX_C_TSH_##cat(NX_C_CUROP, sfx) /* sint,uint */

#define NX_C_TAL_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TAL_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TAL_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TAL_NX_C_CAT_COMPLEX(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TAL_NX_C_CAT_BOOL(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TROW_ALL(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_TAL_##cat(NX_C_CUROP, sfx) /* every compute dtype */

#define NX_C_TOR_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TOR_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TOR_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TOR_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TOR_NX_C_CAT_BOOL(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TROW_ORD(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_TOR_##cat(NX_C_CUROP, sfx) /* sint,uint,float,bool (no ordered complex) */

/* ══════════════════════════════════════════════════════════════════════════
   Unary ops (21)
   ═════════════════════════════════════════════════════════════════════════ */

/* neg / recip / abs / sign: int + float + complex (never bool). */

/* Signed negate runs in the unsigned width: -(vx) on int64_t is UB at INT64_MIN
   (only i64 reaches it; narrower signed dtypes widen safely). Modular, matching
   the wrap the references expect — same trick as idiv's -1 guard. */
#define NX_C_NEG_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_UK(neg, sfx, storage, compute, ld, st, (compute)(-(uint64_t)(vx)))
#define NX_C_NEG_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_UK(neg, sfx, storage, compute, ld, st, (-(vx)))
#define NX_C_NEG_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_UK(neg, sfx, storage, compute, ld, st, (-(vx)))
#define NX_C_NEG_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)                 \
  NX_C_UK(neg, sfx, storage, compute, ld, st, (-(vx)))
#define NX_C_NEG_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_NEG_KROW(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_NEG_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_NEG_KROW)

#define NX_C_RECIP_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                  \
  NX_C_UK(recip, sfx, storage, compute, ld, st, ((vx) == 0 ? 0 : 1 / (vx)))
#define NX_C_RECIP_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                  \
  NX_C_UK(recip, sfx, storage, compute, ld, st, ((vx) == 0 ? 0 : 1 / (vx)))
#define NX_C_RECIP_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                 \
  NX_C_UK(recip, sfx, storage, compute, ld, st, ((compute)1 / (vx)))
#define NX_C_RECIP_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)               \
  NX_C_UK(recip, sfx, storage, compute, ld, st, ((compute)1 / (vx)))
#define NX_C_RECIP_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_RECIP_KROW(sfx, kind, storage, compute, ld, st, cat)               \
  NX_C_RECIP_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_RECIP_KROW)

/* Signed abs negates in the unsigned width for the same INT64_MIN reason as neg;
   abs(INT64_MIN) wraps to INT64_MIN (matching numpy), never traps. */
#define NX_C_ABS_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_UK(abs, sfx, storage, compute, ld, st,                                   \
         ((vx) < 0 ? (compute)(-(uint64_t)(vx)) : (vx)))
#define NX_C_ABS_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_UK(abs, sfx, storage, compute, ld, st, (vx))
#define NX_C_ABS_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_UK(abs, sfx, storage, compute, ld, st, NX_C_MFN(fabs, compute)(vx))
#define NX_C_ABS_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)                 \
  NX_C_UK(abs, sfx, storage, compute, ld, st,                                   \
         (compute)NX_C_MFN(cabs, compute)(vx))
#define NX_C_ABS_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_ABS_KROW(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_ABS_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_ABS_KROW)

/* sign: -1/0/1 for signed, 0/1 for unsigned, NaN-preserving for float,
   z/|z| (0 -> 0) for complex. */
#define NX_C_SIGN_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                   \
  NX_C_UK(sign, sfx, storage, compute, ld, st,                                  \
         (compute)(((vx) > 0) - ((vx) < 0)))
#define NX_C_SIGN_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                   \
  NX_C_UK(sign, sfx, storage, compute, ld, st, (compute)((vx) != 0))
#define NX_C_SIGN_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                  \
  NX_C_UK(sign, sfx, storage, compute, ld, st,                                  \
         (isnan(vx) ? (vx) : (compute)(((vx) > 0) - ((vx) < 0))))
#define NX_C_SIGN_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)                \
  NX_C_UK(sign, sfx, storage, compute, ld, st,                                  \
         (NX_C_MFN(cabs, compute)(vx) == 0                                      \
              ? (compute)0                                                     \
              : (vx) / NX_C_MFN(cabs, compute)(vx)))
#define NX_C_SIGN_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_SIGN_KROW(sfx, kind, storage, compute, ld, st, cat)                \
  NX_C_SIGN_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_SIGN_KROW)

#define NX_C_CUROP neg
static const nx_c_map_table nx_c_neg_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_NUM)}};
#undef NX_C_CUROP
#define NX_C_CUROP recip
static const nx_c_map_table nx_c_recip_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_NUM)}};
#undef NX_C_CUROP
#define NX_C_CUROP abs
static const nx_c_map_table nx_c_abs_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_NUM)}};
#undef NX_C_CUROP
#define NX_C_CUROP sign
static const nx_c_map_table nx_c_sign_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_NUM)}};
#undef NX_C_CUROP

/* Transcendentals: float (fn) + complex (c-fn). The float fn is the op name and
   the complex fn is c<op>, both libm-suffixed by compute type. Integer dtypes
   are promoted to float by the frontend, so their slots stay NULL. */
#define NX_C_TRK_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_UK(NX_C_CUROP, sfx, storage, compute, ld, st,                             \
         NX_C_MFN(NX_C_CUROP, compute)(vx))
#define NX_C_TRK_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)                 \
  NX_C_UK(NX_C_CUROP, sfx, storage, compute, ld, st,                             \
         NX_C_MFN(NX_C_PASTE(c, NX_C_CUROP), compute)(vx))
#define NX_C_TRK_NX_C_CAT_SINT(sfx, storage, compute, ld, st)
#define NX_C_TRK_NX_C_CAT_UINT(sfx, storage, compute, ld, st)
#define NX_C_TRK_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_TRANS_KROW(sfx, kind, storage, compute, ld, st, cat)               \
  NX_C_TRK_##cat(sfx, storage, compute, ld, st)

#define NX_C_TRANS(op)                                                          \
  NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TRANS_KROW)                                   \
  static const nx_c_map_table nx_c_##op##_table = {                             \
      .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_FC)}};

#define NX_C_CUROP exp
NX_C_TRANS(exp)
#undef NX_C_CUROP
#define NX_C_CUROP log
NX_C_TRANS(log)
#undef NX_C_CUROP
#define NX_C_CUROP sin
NX_C_TRANS(sin)
#undef NX_C_CUROP
#define NX_C_CUROP cos
NX_C_TRANS(cos)
#undef NX_C_CUROP
#define NX_C_CUROP tan
NX_C_TRANS(tan)
#undef NX_C_CUROP
#define NX_C_CUROP asin
NX_C_TRANS(asin)
#undef NX_C_CUROP
#define NX_C_CUROP acos
NX_C_TRANS(acos)
#undef NX_C_CUROP
#define NX_C_CUROP atan
NX_C_TRANS(atan)
#undef NX_C_CUROP
#define NX_C_CUROP sinh
NX_C_TRANS(sinh)
#undef NX_C_CUROP
#define NX_C_CUROP cosh
NX_C_TRANS(cosh)
#undef NX_C_CUROP
#define NX_C_CUROP tanh
NX_C_TRANS(tanh)
#undef NX_C_CUROP
#define NX_C_CUROP sqrt
NX_C_TRANS(sqrt)
#undef NX_C_CUROP

/* erf: float only (no standard complex error function). */
#define NX_C_FOK_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_UK(NX_C_CUROP, sfx, storage, compute, ld, st,                             \
         NX_C_MFN(NX_C_CUROP, compute)(vx))
#define NX_C_FOK_NX_C_CAT_SINT(sfx, storage, compute, ld, st)
#define NX_C_FOK_NX_C_CAT_UINT(sfx, storage, compute, ld, st)
#define NX_C_FOK_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_FOK_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_FO_KROW(sfx, kind, storage, compute, ld, st, cat)                  \
  NX_C_FOK_##cat(sfx, storage, compute, ld, st)
#define NX_C_CUROP erf
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_FO_KROW)
static const nx_c_map_table nx_c_erf_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_FLOAT)}};
#undef NX_C_CUROP

/* Rounding: identity on integers, the libm rounder on floats, rejected on
   complex (NULL). */
#define NX_C_RNK_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_UK(NX_C_CUROP, sfx, storage, compute, ld, st, (vx))
#define NX_C_RNK_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_UK(NX_C_CUROP, sfx, storage, compute, ld, st, (vx))
#define NX_C_RNK_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_UK(NX_C_CUROP, sfx, storage, compute, ld, st,                             \
         NX_C_MFN(NX_C_CUROP, compute)(vx))
#define NX_C_RNK_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_RNK_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_RND_KROW(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_RNK_##cat(sfx, storage, compute, ld, st)

#define NX_C_ROUNDOP(op)                                                        \
  NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_RND_KROW)                                     \
  static const nx_c_map_table nx_c_##op##_table = {                             \
      .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_INTF)}};

#define NX_C_CUROP trunc
NX_C_ROUNDOP(trunc)
#undef NX_C_CUROP
#define NX_C_CUROP ceil
NX_C_ROUNDOP(ceil)
#undef NX_C_CUROP
#define NX_C_CUROP floor
NX_C_ROUNDOP(floor)
#undef NX_C_CUROP
#define NX_C_CUROP round
NX_C_ROUNDOP(round)
#undef NX_C_CUROP

/* ══════════════════════════════════════════════════════════════════════════
   Binary ops (14 + shl/shr)
   ═════════════════════════════════════════════════════════════════════════ */

/* add / sub / mul: int + float + complex, differing only in the operator.
   Signed forms run in the unsigned width: the contract is modular wrap, and
   only i64 reaches the 64-bit boundary (narrower signed dtypes widen safely
   and wrap at the store narrowing). Defined without -fwrapv, which stays in
   the flags as belt and suspenders — same idiom as neg/abs/idiv. */
#define NX_C_ARK_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st,                             \
         (compute)((uint64_t)(va)NX_C_CURSYM(uint64_t)(vb)))
#define NX_C_ARK_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st, ((va)NX_C_CURSYM(vb)))
#define NX_C_ARK_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st, ((va)NX_C_CURSYM(vb)))
#define NX_C_ARK_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)                 \
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st, ((va)NX_C_CURSYM(vb)))
#define NX_C_ARK_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_ARITH_KROW(sfx, kind, storage, compute, ld, st, cat)               \
  NX_C_ARK_##cat(sfx, storage, compute, ld, st)

#define NX_C_ARITH(op, sym)                                                     \
  NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_ARITH_KROW)                                   \
  static const nx_c_map_table nx_c_##op##_table = {                             \
      .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_NUM)}};

#define NX_C_CUROP add
#define NX_C_CURSYM +
NX_C_ARITH(add, +)
#undef NX_C_CURSYM
#undef NX_C_CUROP
#define NX_C_CUROP sub
#define NX_C_CURSYM -
NX_C_ARITH(sub, -)
#undef NX_C_CURSYM
#undef NX_C_CUROP
#define NX_C_CUROP mul
#define NX_C_CURSYM *
NX_C_ARITH(mul, *)
#undef NX_C_CURSYM
#undef NX_C_CUROP

/* idiv: integer truncating division (by-zero -> 0), signed -1 guarded against
   INT_MIN overflow; float idiv truncates the quotient. Complex uses fdiv. */
#define NX_C_IDIV_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                   \
  NX_C_BK(idiv, sfx, storage, compute, ld, st,                                  \
         ((vb) == 0 ? (compute)0                                               \
                    : (vb) == -1 ? (compute)(-(uint64_t)(va)) : (va) / (vb)))
#define NX_C_IDIV_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                   \
  NX_C_BK(idiv, sfx, storage, compute, ld, st,                                  \
         ((vb) == 0 ? (compute)0 : (va) / (vb)))
#define NX_C_IDIV_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                  \
  NX_C_BK(idiv, sfx, storage, compute, ld, st,                                  \
         NX_C_MFN(trunc, compute)((va) / (vb)))
#define NX_C_IDIV_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_IDIV_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_IDIV_KROW(sfx, kind, storage, compute, ld, st, cat)                \
  NX_C_IDIV_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_IDIV_KROW)
#define NX_C_CUROP idiv
static const nx_c_map_table nx_c_idiv_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_INTF)}};
#undef NX_C_CUROP

/* fdiv: true division for float and complex (int div routes to idiv). */
#define NX_C_FDIV_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                  \
  NX_C_BK(fdiv, sfx, storage, compute, ld, st, ((va) / (vb)))
#define NX_C_FDIV_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)                \
  NX_C_BK(fdiv, sfx, storage, compute, ld, st, ((va) / (vb)))
#define NX_C_FDIV_NX_C_CAT_SINT(sfx, storage, compute, ld, st)
#define NX_C_FDIV_NX_C_CAT_UINT(sfx, storage, compute, ld, st)
#define NX_C_FDIV_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_FDIV_KROW(sfx, kind, storage, compute, ld, st, cat)                \
  NX_C_FDIV_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_FDIV_KROW)
#define NX_C_CUROP fdiv
static const nx_c_map_table nx_c_fdiv_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_FC)}};
#undef NX_C_CUROP

/* mod: integer remainder (by-zero -> 0, sign follows dividend), fmod on floats.
   No complex remainder. */
#define NX_C_MOD_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(mod, sfx, storage, compute, ld, st,                                   \
         ((vb) == 0 ? (compute)0 : (vb) == -1 ? (compute)0 : (va) % (vb)))
#define NX_C_MOD_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(mod, sfx, storage, compute, ld, st,                                   \
         ((vb) == 0 ? (compute)0 : (va) % (vb)))
#define NX_C_MOD_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_BK(mod, sfx, storage, compute, ld, st, NX_C_MFN(fmod, compute)(va, vb))
#define NX_C_MOD_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_MOD_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_MOD_KROW(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_MOD_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_MOD_KROW)
#define NX_C_CUROP mod
static const nx_c_map_table nx_c_mod_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_INTF)}};
#undef NX_C_CUROP

/* max / min: NaN-propagating on floats, ordered on int/bool, rejected on
   complex. The compute width carries signedness, so > / < are the right
   signed/unsigned comparison per dtype. */
#define NX_C_MMK_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st,                             \
         ((va)NX_C_CURSYM(vb) ? (va) : (vb)))
#define NX_C_MMK_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st,                             \
         ((va)NX_C_CURSYM(vb) ? (va) : (vb)))
#define NX_C_MMK_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)                    \
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st,                             \
         ((va)NX_C_CURSYM(vb) ? (va) : (vb)))
#define NX_C_MMK_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st,                             \
         ((isnan(va) || isnan(vb)) ? (compute)NAN                              \
                                   : ((va)NX_C_CURSYM(vb) ? (va) : (vb))))
#define NX_C_MMK_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_MINMAX_KROW(sfx, kind, storage, compute, ld, st, cat)              \
  NX_C_MMK_##cat(sfx, storage, compute, ld, st)

#define NX_C_MINMAX(op)                                                         \
  NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_MINMAX_KROW)                                  \
  static const nx_c_map_table nx_c_##op##_table = {                             \
      .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_MINMAX)}};

#define NX_C_CUROP max
#define NX_C_CURSYM >
NX_C_MINMAX(max)
#undef NX_C_CURSYM
#undef NX_C_CUROP
#define NX_C_CUROP min
#define NX_C_CURSYM <
NX_C_MINMAX(min)
#undef NX_C_CURSYM
#undef NX_C_CUROP

/* pow: integer power by squaring in the compute width (wrap on store), powf/pow
   for floats, cpow for complex. */
#define NX_C_POW_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(pow, sfx, storage, compute, ld, st, nx_c_ipow_s((va), (vb)))
#define NX_C_POW_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(pow, sfx, storage, compute, ld, st, nx_c_ipow_u((va), (vb)))
#define NX_C_POW_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_BK(pow, sfx, storage, compute, ld, st, NX_C_MFN(pow, compute)((va), (vb)))
#define NX_C_POW_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)                 \
  NX_C_BK(pow, sfx, storage, compute, ld, st,                                   \
         NX_C_MFN(cpow, compute)((va), (vb)))
#define NX_C_POW_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_POW_KROW(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_POW_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_POW_KROW)
#define NX_C_CUROP pow
static const nx_c_map_table nx_c_pow_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_NUM)}};
#undef NX_C_CUROP

/* atan2: float only (int inputs are promoted by the frontend). */
#define NX_C_ATAN2_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                 \
  NX_C_BK(atan2, sfx, storage, compute, ld, st,                                 \
         NX_C_MFN(atan2, compute)((va), (vb)))
#define NX_C_ATAN2_NX_C_CAT_SINT(sfx, storage, compute, ld, st)
#define NX_C_ATAN2_NX_C_CAT_UINT(sfx, storage, compute, ld, st)
#define NX_C_ATAN2_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_ATAN2_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_ATAN2_KROW(sfx, kind, storage, compute, ld, st, cat)               \
  NX_C_ATAN2_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_ATAN2_KROW)
#define NX_C_CUROP atan2
static const nx_c_map_table nx_c_atan2_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_FLOAT)}};
#undef NX_C_CUROP

/* xor / or / and: integer + bool (logical on bool), differing only in the
   operator. */
#define NX_C_BWK_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st, ((va)NX_C_CURSYM(vb)))
#define NX_C_BWK_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st, ((va)NX_C_CURSYM(vb)))
#define NX_C_BWK_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)                    \
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st, ((va)NX_C_CURSYM(vb)))
#define NX_C_BWK_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)
#define NX_C_BWK_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_BITWISE_KROW(sfx, kind, storage, compute, ld, st, cat)             \
  NX_C_BWK_##cat(sfx, storage, compute, ld, st)

#define NX_C_BITWISE(op)                                                        \
  NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_BITWISE_KROW)                                 \
  static const nx_c_map_table nx_c_##op##_table = {                             \
      .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_BITWISE)}};

#define NX_C_CUROP xor
#define NX_C_CURSYM ^
NX_C_BITWISE(xor)
#undef NX_C_CURSYM
#undef NX_C_CUROP
#define NX_C_CUROP or
#define NX_C_CURSYM |
NX_C_BITWISE(or)
#undef NX_C_CURSYM
#undef NX_C_CUROP
#define NX_C_CUROP and
#define NX_C_CURSYM &
NX_C_BITWISE(and)
#undef NX_C_CURSYM
#undef NX_C_CUROP

/* shl / shr: integer only. A count negative or >= the dtype width yields 0
   (documented, total). shl runs in the unsigned width to avoid signed-overflow
   UB; shr keeps the signed/unsigned compute type so it is arithmetic on signed
   and logical on unsigned dtypes. */
#define NX_C_SHL_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(shl, sfx, storage, compute, ld, st,                                   \
         (((vb) < 0 || (vb) >= (compute)(sizeof(storage) * 8))                 \
              ? (compute)0                                                     \
              : (compute)((uint64_t)(va) << (vb))))
#define NX_C_SHL_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(shl, sfx, storage, compute, ld, st,                                   \
         (((vb) >= (compute)(sizeof(storage) * 8))                             \
              ? (compute)0                                                     \
              : (compute)((uint64_t)(va) << (vb))))
#define NX_C_SHL_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)
#define NX_C_SHL_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_SHL_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_SHL_KROW(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_SHL_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_SHL_KROW)
#define NX_C_CUROP shl
static const nx_c_map_table nx_c_shl_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_SHIFT)}};
#undef NX_C_CUROP

#define NX_C_SHR_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(shr, sfx, storage, compute, ld, st,                                   \
         (((vb) < 0 || (vb) >= (compute)(sizeof(storage) * 8))                 \
              ? (compute)0                                                     \
              : ((va) >> (vb))))
#define NX_C_SHR_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(shr, sfx, storage, compute, ld, st,                                   \
         (((vb) >= (compute)(sizeof(storage) * 8)) ? (compute)0                \
                                                   : ((va) >> (vb))))
#define NX_C_SHR_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)
#define NX_C_SHR_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_SHR_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_SHR_KROW(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_SHR_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_SHR_KROW)
#define NX_C_CUROP shr
static const nx_c_map_table nx_c_shr_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_SHIFT)}};
#undef NX_C_CUROP

/* ══════════════════════════════════════════════════════════════════════════
   Comparisons (4) — bool output, dispatched on the INPUT dtype
   ═════════════════════════════════════════════════════════════════════════ */

/* cmpeq / cmpne: every compute dtype (== and != are defined for complex). */
#define NX_C_CEQK_NX_C_CAT_SINT(sfx, storage, compute, ld)                       \
  NX_C_CMPK(NX_C_CUROP, sfx, storage, compute, ld, ((va)NX_C_CURSYM(vb)))
#define NX_C_CEQK_NX_C_CAT_UINT(sfx, storage, compute, ld)                       \
  NX_C_CMPK(NX_C_CUROP, sfx, storage, compute, ld, ((va)NX_C_CURSYM(vb)))
#define NX_C_CEQK_NX_C_CAT_FLOAT(sfx, storage, compute, ld)                      \
  NX_C_CMPK(NX_C_CUROP, sfx, storage, compute, ld, ((va)NX_C_CURSYM(vb)))
#define NX_C_CEQK_NX_C_CAT_COMPLEX(sfx, storage, compute, ld)                    \
  NX_C_CMPK(NX_C_CUROP, sfx, storage, compute, ld, ((va)NX_C_CURSYM(vb)))
#define NX_C_CEQK_NX_C_CAT_BOOL(sfx, storage, compute, ld)                       \
  NX_C_CMPK(NX_C_CUROP, sfx, storage, compute, ld, ((va)NX_C_CURSYM(vb)))
#define NX_C_CEQ_KROW(sfx, kind, storage, compute, ld, st, cat)                 \
  NX_C_CEQK_##cat(sfx, storage, compute, ld)

#define NX_C_CMPEQ(op)                                                          \
  NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CEQ_KROW)                                     \
  static const nx_c_map_table nx_c_##op##_table = {                             \
      .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_ALL)}};

#define NX_C_CUROP cmpeq
#define NX_C_CURSYM ==
NX_C_CMPEQ(cmpeq)
#undef NX_C_CURSYM
#undef NX_C_CUROP
#define NX_C_CUROP cmpne
#define NX_C_CURSYM !=
NX_C_CMPEQ(cmpne)
#undef NX_C_CURSYM
#undef NX_C_CUROP

/* cmplt / cmple: every compute dtype except complex (no ordered comparison). */
#define NX_C_CORDK_NX_C_CAT_SINT(sfx, storage, compute, ld)                      \
  NX_C_CMPK(NX_C_CUROP, sfx, storage, compute, ld, ((va)NX_C_CURSYM(vb)))
#define NX_C_CORDK_NX_C_CAT_UINT(sfx, storage, compute, ld)                      \
  NX_C_CMPK(NX_C_CUROP, sfx, storage, compute, ld, ((va)NX_C_CURSYM(vb)))
#define NX_C_CORDK_NX_C_CAT_FLOAT(sfx, storage, compute, ld)                     \
  NX_C_CMPK(NX_C_CUROP, sfx, storage, compute, ld, ((va)NX_C_CURSYM(vb)))
#define NX_C_CORDK_NX_C_CAT_BOOL(sfx, storage, compute, ld)                      \
  NX_C_CMPK(NX_C_CUROP, sfx, storage, compute, ld, ((va)NX_C_CURSYM(vb)))
#define NX_C_CORDK_NX_C_CAT_COMPLEX(sfx, storage, compute, ld)
#define NX_C_CORD_KROW(sfx, kind, storage, compute, ld, st, cat)                \
  NX_C_CORDK_##cat(sfx, storage, compute, ld)

#define NX_C_CMPORD(op)                                                         \
  NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CORD_KROW)                                    \
  static const nx_c_map_table nx_c_##op##_table = {                             \
      .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_ORD)}};

#define NX_C_CUROP cmplt
#define NX_C_CURSYM <
NX_C_CMPORD(cmplt)
#undef NX_C_CURSYM
#undef NX_C_CUROP
#define NX_C_CUROP cmple
#define NX_C_CURSYM <=
NX_C_CMPORD(cmple)
#undef NX_C_CURSYM
#undef NX_C_CUROP

/* ══════════════════════════════════════════════════════════════════════════
   where (map3) — bool condition, two value operands, pure bit-select
   ═════════════════════════════════════════════════════════════════════════ */

#define NX_C_WK(sfx, storage)                                                   \
  static void nx_c_where_##sfx(char *const *pp, const int64_t *ssx, int64_t nn, \
                              void *ctx) {                                     \
    (void)ctx;                                                                 \
    char *out = pp[0];                                                         \
    const char *cnd = pp[1];                                                   \
    const char *in0 = pp[2];                                                   \
    const char *in1 = pp[3];                                                   \
    const int64_t so = ssx[0], sc = ssx[1], sa = ssx[2], sb = ssx[3];          \
    const int64_t es = (int64_t)sizeof(storage);                              \
    if (so == es && sa == es && sb == es && sc == 1) {                        \
      storage *pO = (storage *)out;                                           \
      const uint8_t *pC = (const uint8_t *)cnd;                               \
      const storage *pA = (const storage *)in0;                               \
      const storage *pB = (const storage *)in1;                               \
      for (int64_t i = 0; i < nn; i++) pO[i] = pC[i] ? pA[i] : pB[i];          \
    } else {                                                                   \
      for (int64_t i = 0; i < nn; i++) {                                       \
        uint8_t c = *(const uint8_t *)(cnd + i * sc);                          \
        *(storage *)(out + i * so) = c ? *(const storage *)(in0 + i * sa)      \
                                       : *(const storage *)(in1 + i * sb);     \
      }                                                                        \
    }                                                                          \
  }
#define NX_C_WHERE_KROW(sfx, kind, storage, compute, ld, st, cat)               \
  NX_C_WK(sfx, storage)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_WHERE_KROW)
#define NX_C_CUROP where
static const nx_c_map_table nx_c_where_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_ALL)}};
#undef NX_C_CUROP

/* ══════════════════════════════════════════════════════════════════════════
   cast — the pair matrix (compute src × compute dst) plus a packed nibble path
   ═════════════════════════════════════════════════════════════════════════ */

/* Local float->f16/bf16 converters for the contiguous cast fast path.

   The buffer layer's float_to_half / float_to_bfloat16 (nx_buffer_stubs.h) are
   correct but branchy: a call inside the store loop blocks auto-vectorization
   (the loop cannot become SIMD across a branchy body), so a contiguous f32->f16
   cast ran at scalar speed. These are branchless/hardware forms that the
   vectorizer turns into packed converts. They are a DELIBERATE, TESTED copy of
   the buffer converters: they MUST produce bit-identical results for EVERY input
   (rounding mode, NaN quieting, subnormals, overflow) — pinned by the
   nx_c_cast_convert_selfcheck gate, which sweeps all 65536 f16 patterns plus
   NaN/inf/subnormal/overflow f32 edges against the canonical converters. If a
   future edit here diverges, that test fails; do not "fix" it by loosening the
   gate. The buffer layer stays the single owner of the storage format; this is
   only a vectorizable restatement used nowhere but the cast fast path. */

/* Branchless f32 -> IEEE binary16, round-to-nearest-even. All three exponent
   regimes are computed and the result selected; the subnormal shift is masked so
   a discarded lane never triggers undefined shift behaviour. */
static inline uint16_t nx_c_f32_to_f16_sw(float f) {
  union { float f; uint32_t i; } u = {.f = f};
  uint32_t b = u.i;
  uint32_t sgn = (b & 0x80000000u) >> 16; /* half sign in bit 15 */
  uint32_t exp = b & 0x7F800000u;         /* biased exponent field, in place */
  uint32_t sig = b & 0x007FFFFFu;         /* mantissa */

  /* Large: finite overflow / inf / NaN. NaN keeps the top payload bits and is
     bumped to stay a NaN when they are all zero. */
  uint32_t is_nan = (exp == 0x7F800000u) & (sig != 0u);
  uint16_t nan_ret = (uint16_t)(0x7C00u + (sig >> 13));
  nan_ret += (nan_ret == 0x7C00u);
  uint16_t large = (uint16_t)(sgn + (is_nan ? nan_ret : 0x7C00u));

  /* Subnormal / zero. e in [102,112] over the live range; the shift mask keeps
     a smaller exponent (a discarded lane) free of UB. */
  uint32_t e = exp >> 23;
  uint32_t ssig = sig + 0x00800000u; /* implicit one */
  uint32_t sh = (113u - e) & 31u;
  ssig >>= sh;
  uint32_t sround =
      ((ssig & 0x00003FFFu) != 0x00001000u) | ((b & 0x000007FFu) != 0u);
  ssig += sround ? 0x00001000u : 0u;
  uint16_t sub = (uint16_t)(sgn + (ssig >> 13));
  uint16_t small = (uint16_t)((exp < 0x33000000u) ? sgn : sub);

  /* Regular. */
  uint32_t rsig = sig + (((sig & 0x00003FFFu) != 0x00001000u) ? 0x00001000u : 0u);
  uint16_t reg = (uint16_t)(sgn + ((exp - 0x38000000u) >> 13) + (rsig >> 13));

  if (exp >= 0x47800000u) return large;
  if (exp <= 0x38000000u) return small;
  return reg;
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
/* Same result via the hardware narrowing convert (FCVTN, round-to-nearest-even)
   plus a branchless NaN-payload fixup: the hardware convert force-quiets NaNs by
   setting the mantissa MSB, whereas float_to_half preserves the raw payload, so
   only NaN lanes are recomputed to stay bit-identical. Finite/inf/subnormal all
   already match the canonical converter exactly. The whole loop stays SIMD. */
static inline uint16_t nx_c_f32_to_f16_hw(float f) {
  _Float16 h = (_Float16)f;
  uint16_t o;
  __builtin_memcpy(&o, &h, sizeof o);
  union { float f; uint32_t i; } u = {.f = f};
  uint32_t b = u.i;
  uint32_t is_nan = ((b & 0x7F800000u) == 0x7F800000u) & ((b & 0x007FFFFFu) != 0u);
  uint16_t sgn = (uint16_t)((b & 0x80000000u) >> 16);
  uint16_t nan_ret = (uint16_t)(0x7C00u + ((b & 0x007FFFFFu) >> 13));
  nan_ret += (nan_ret == 0x7C00u);
  return is_nan ? (uint16_t)(sgn + nan_ret) : o;
}
static inline uint16_t nx_c_f32_to_f16(float f) { return nx_c_f32_to_f16_hw(f); }
#else
static inline uint16_t nx_c_f32_to_f16(float f) { return nx_c_f32_to_f16_sw(f); }
#endif

/* Branchless f32 -> bfloat16, round-to-nearest-even (truncation of the top 16
   bits with an RNE bias; NaN is quieted while keeping the sign). Already cheap
   enough to vectorize on its own. */
static inline uint16_t nx_c_f32_to_bf16(float f) {
  union { float f; uint32_t i; } u = {.f = f};
  uint32_t b = u.i;
  uint32_t is_nan = (b & 0x7FFFFFFFu) > 0x7F800000u;
  uint16_t nan_ret = (uint16_t)((b >> 16) | 0x0040u);
  uint32_t bias = ((b >> 16) & 1u) + 0x7FFFu;
  uint16_t norm = (uint16_t)((b + bias) >> 16);
  return is_nan ? nan_ret : norm;
}

/* Fast-path (contiguous) store per dst dtype. f16/bf16 route through the local
   vectorizable converters above; every other dst uses the canonical typed store
   (nx_c_st_<dsfx>). The value handed in is already the dst compute type, so the
   f16/bf16 rows only re-narrow the float. */
#define NX_C_CFSTORE_f16(pO, i, v) ((pO)[i] = nx_c_f32_to_f16((float)(v)))
#define NX_C_CFSTORE_bf16(pO, i, v) ((pO)[i] = nx_c_f32_to_bf16((float)(v)))
#define NX_C_CFSTORE_f32(pO, i, v) nx_c_st_f32(&(pO)[i], (v))
#define NX_C_CFSTORE_f64(pO, i, v) nx_c_st_f64(&(pO)[i], (v))
#define NX_C_CFSTORE_f8e4m3(pO, i, v) nx_c_st_f8e4m3(&(pO)[i], (v))
#define NX_C_CFSTORE_f8e5m2(pO, i, v) nx_c_st_f8e5m2(&(pO)[i], (v))
#define NX_C_CFSTORE_i8(pO, i, v) nx_c_st_i8(&(pO)[i], (v))
#define NX_C_CFSTORE_u8(pO, i, v) nx_c_st_u8(&(pO)[i], (v))
#define NX_C_CFSTORE_i16(pO, i, v) nx_c_st_i16(&(pO)[i], (v))
#define NX_C_CFSTORE_u16(pO, i, v) nx_c_st_u16(&(pO)[i], (v))
#define NX_C_CFSTORE_i32(pO, i, v) nx_c_st_i32(&(pO)[i], (v))
#define NX_C_CFSTORE_u32(pO, i, v) nx_c_st_u32(&(pO)[i], (v))
#define NX_C_CFSTORE_i64(pO, i, v) nx_c_st_i64(&(pO)[i], (v))
#define NX_C_CFSTORE_u64(pO, i, v) nx_c_st_u64(&(pO)[i], (v))
#define NX_C_CFSTORE_c32(pO, i, v) nx_c_st_c32(&(pO)[i], (v))
#define NX_C_CFSTORE_c64(pO, i, v) nx_c_st_c64(&(pO)[i], (v))
#define NX_C_CFSTORE_bool_(pO, i, v) nx_c_st_bool_(&(pO)[i], (v))

/* The inner (dst) dimension of the src×dst cast matrix needs a dtype iterator
   distinct from the one walking the outer (src) dimension: the C preprocessor
   cannot recurse the header's NX_C_DTYPE_TABLE into itself. So the OUTER src walk
   reuses the header's NX_C_FOR_EACH_COMPUTE_DTYPE and only the INNER dst walk is a
   local copy (suffix, compute type, category) — a differently named macro that
   nests cleanly. Any drift (a missing row) surfaces at once: the corresponding
   cast slot stays NULL and every conformance/map test that casts through it
   fails loudly. Order mirrors nx_c.h's compute rows. */
#define NX_C_CAST_DST_LIST(X, A)                                                \
  X(A, f16, uint16_t, float, NX_C_CAT_FLOAT)                                    \
  X(A, f32, float, float, NX_C_CAT_FLOAT)                                       \
  X(A, f64, double, double, NX_C_CAT_FLOAT)                                     \
  X(A, bf16, uint16_t, float, NX_C_CAT_FLOAT)                                   \
  X(A, f8e4m3, caml_ba_fp8_e4m3, float, NX_C_CAT_FLOAT)                         \
  X(A, f8e5m2, caml_ba_fp8_e5m2, float, NX_C_CAT_FLOAT)                         \
  X(A, i8, int8_t, int64_t, NX_C_CAT_SINT) X(A, u8, uint8_t, int64_t, NX_C_CAT_UINT) \
  X(A, i16, int16_t, int64_t, NX_C_CAT_SINT)                                    \
  X(A, u16, uint16_t, int64_t, NX_C_CAT_UINT)                                   \
  X(A, i32, int32_t, int64_t, NX_C_CAT_SINT)                                    \
  X(A, u32, caml_ba_uint32, uint64_t, NX_C_CAT_UINT)                            \
  X(A, i64, int64_t, int64_t, NX_C_CAT_SINT)                                    \
  X(A, u64, caml_ba_uint64, uint64_t, NX_C_CAT_UINT)                            \
  X(A, c32, nx_c_complex32, nx_c_complex32, NX_C_CAT_COMPLEX)                     \
  X(A, c64, nx_c_complex64, nx_c_complex64, NX_C_CAT_COMPLEX)                     \
  X(A, bool_, caml_ba_bool, uint8_t, NX_C_CAT_BOOL)

/* The header's _Static_assert pins the dtype enum but not this local list; pin
   its length to the table's compute-row count so a dtype added to the table but
   forgotten here fails to compile rather than only at runtime on the one
   untested pair. (Counts, not identity — a wrong row still shows up as a NULL
   cast slot and a failing test.) */
#define NX_C_CAST_DST_CNT(A, dsfx, dstorage, dcompute, dcat) +1
#define NX_C_COMPUTE_CNT(sfx, kind, storage, compute, ld, st, cat) +1
_Static_assert((0 NX_C_CAST_DST_LIST(NX_C_CAST_DST_CNT, _)) ==
                   (0 NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_COMPUTE_CNT)),
               "cast dst list drifted from the dtype table");
#undef NX_C_CAST_DST_CNT
#undef NX_C_COMPUTE_CNT

#define NX_C_UN5(a, b, c, d, e) a, b, c, d, e

/* One converter per (src, dst). Contiguous runs take a typed fast path (unit-
   stride src/dst pointers so the compiler knows the stride is constant and can
   vectorize); the f16/bf16 stores route through the local branchless/hardware
   converters, so an f32->f16 cast becomes packed SIMD instead of a scalar walk
   over a branchy converter. Strided runs keep the generic byte walk. A carries
   the src suffix / storage / compute / load / category. */
#define NX_C_CAST_KI(A, dsfx, dstorage, dcompute, dcat)                         \
  NX_C_CAST_KI_X(NX_C_UN5 A, dsfx, dstorage, dcompute, dcat)
#define NX_C_CAST_KI_X(...) NX_C_CAST_KI3(__VA_ARGS__)
#define NX_C_CAST_KI3(ssfx, sstorage, scompute, sld, scat, dsfx, dstorage,      \
                     dcompute, dcat)                                           \
  static void nx_c_cast_##ssfx##_to_##dsfx(char *const *pp, const int64_t *ssx, \
                                          int64_t nn, void *ctx) {             \
    (void)ctx;                                                                 \
    char *out = pp[0];                                                         \
    const char *in0 = pp[1];                                                   \
    const int64_t so = ssx[0], sa = ssx[1];                                    \
    if (so == (int64_t)sizeof(dstorage) && sa == (int64_t)sizeof(sstorage)) {  \
      dstorage *pO = (dstorage *)out;                                         \
      const sstorage *pA = (const sstorage *)in0;                            \
      for (int64_t i = 0; i < nn; i++) {                                       \
        scompute vs = (scompute)sld(pA[i]);                                   \
        NX_C_CFSTORE_##dsfx(pO, i, NX_C_CASTVAL_##dcat(dcompute, dsfx, scat, vs)); \
      }                                                                        \
    } else {                                                                   \
      for (int64_t i = 0; i < nn; i++) {                                       \
        scompute vs = nx_c_ld_##ssfx(in0 + i * sa);                            \
        nx_c_st_##dsfx(out + i * so,                                           \
                      NX_C_CASTVAL_##dcat(dcompute, dsfx, scat, vs));          \
      }                                                                        \
    }                                                                          \
  }
#define NX_C_CAST_KGEN_SRC(ssfx, kind, sstorage, scompute, sld, sst, scat)      \
  NX_C_CAST_DST_LIST(NX_C_CAST_KI, (ssfx, sstorage, scompute, sld, scat))
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CAST_KGEN_SRC)

/* Per-src dispatch table indexed by dst dtype; the top-level array is indexed by
   src dtype. A carries the src suffix. */
#define NX_C_UN1(a) a
#define NX_C_CAST_TE(A, dsfx, dstorage, dcompute, dcat)                         \
  NX_C_CAST_TE2(NX_C_UN1 A, dsfx)
#define NX_C_CAST_TE2(ssfx, dsfx) NX_C_CAST_TE3(ssfx, dsfx)
#define NX_C_CAST_TE3(ssfx, dsfx)                                               \
  [NX_C_DTYPE_##dsfx] = nx_c_cast_##ssfx##_to_##dsfx,
#define NX_C_CAST_SRCTBL(ssfx, kind, sstorage, scompute, sld, sst, scat)        \
  [NX_C_DTYPE_##ssfx] = {.fn = {NX_C_CAST_DST_LIST(NX_C_CAST_TE, (ssfx))}},
static const nx_c_map_table nx_c_cast_tables[NX_C_DTYPE_COUNT] = {
    NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CAST_SRCTBL)};

/* Packed (int4/uint4) nibble path — serial and contiguous only. Element i of a
   packed operand is nibble (offset + i): byte (offset+i)>>1, low nibble on even,
   high on odd. A signed int4 nibble sign-extends; both wrap the low nibble on
   store. */

/* compute src -> packed dst (i4/u4). The nibble value wraps for int/bool sources
   and saturates for float/complex (F2I4). */
#define NX_C_NIB_NX_C_CAT_SINT(F2I4, v) ((int)(v))
#define NX_C_NIB_NX_C_CAT_UINT(F2I4, v) ((int)(v))
#define NX_C_NIB_NX_C_CAT_BOOL(F2I4, v) ((int)(v))
#define NX_C_NIB_NX_C_CAT_FLOAT(F2I4, v) F2I4((double)(v))
#define NX_C_NIB_NX_C_CAT_COMPLEX(F2I4, v) F2I4((double)creal(v))

typedef void nx_c_castp_to(uint8_t *dbytes, int64_t doff, const char *sbase,
                          int64_t n);
typedef void nx_c_castp_from(char *dbase, const uint8_t *sbytes, int64_t soff,
                            int64_t n);

#define NX_C_CASTP_TO_KERN(PK, F2I4, sfx, storage, compute, ld, cat)            \
  static void nx_c_castp_##sfx##_to_##PK(uint8_t *db, int64_t doff,             \
                                        const char *sbase, int64_t n) {        \
    const storage *pA = (const storage *)sbase;                               \
    for (int64_t i = 0; i < n; i++) {                                          \
      uint8_t nib = (uint8_t)NX_C_NIB_##cat(F2I4, (compute)ld(pA[i])) & 0x0F;   \
      int64_t di = doff + i;                                                   \
      uint8_t *bp = &db[di >> 1];                                             \
      *bp = (di & 1) ? (uint8_t)((*bp & 0x0F) | (nib << 4))                    \
                     : (uint8_t)((*bp & 0xF0) | nib);                          \
    }                                                                          \
  }
#define NX_C_CASTP_TO_ROW(sfx, kind, storage, compute, ld, st, cat)             \
  NX_C_CASTP_TO_KERN(i4, nx_c_f2i4_s, sfx, storage, compute, ld, cat)            \
  NX_C_CASTP_TO_KERN(u4, nx_c_f2i4_u, sfx, storage, compute, ld, cat)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CASTP_TO_ROW)

/* packed src (i4/u4) -> compute dst, reusing the cast precision policy with the
   nibble value as a small signed (i4) / unsigned (u4) integer. */
#define NX_C_CASTP_FROM_I4(sfx, storage, compute, st, dcat)                     \
  static void nx_c_castp_i4_to_##sfx(char *dbase, const uint8_t *sb,            \
                                    int64_t soff, int64_t n) {                 \
    storage *pO = (storage *)dbase;                                           \
    for (int64_t i = 0; i < n; i++) {                                          \
      int64_t si = soff + i;                                                   \
      uint8_t by = sb[si >> 1];                                               \
      int v = (si & 1) ? ((int8_t)by >> 4) : ((int8_t)((by & 0x0F) << 4) >> 4);\
      pO[i] = (storage)st(NX_C_CASTVAL_##dcat(compute, sfx, NX_C_CAT_SINT, v));  \
    }                                                                          \
  }
#define NX_C_CASTP_FROM_U4(sfx, storage, compute, st, dcat)                     \
  static void nx_c_castp_u4_to_##sfx(char *dbase, const uint8_t *sb,            \
                                    int64_t soff, int64_t n) {                 \
    storage *pO = (storage *)dbase;                                           \
    for (int64_t i = 0; i < n; i++) {                                          \
      int64_t si = soff + i;                                                   \
      uint8_t by = sb[si >> 1];                                               \
      int v = (si & 1) ? (by >> 4) : (by & 0x0F);                             \
      pO[i] = (storage)st(NX_C_CASTVAL_##dcat(compute, sfx, NX_C_CAT_UINT, v));  \
    }                                                                          \
  }
#define NX_C_CASTP_FROM_ROW(sfx, kind, storage, compute, ld, st, cat)           \
  NX_C_CASTP_FROM_I4(sfx, storage, compute, st, cat)                            \
  NX_C_CASTP_FROM_U4(sfx, storage, compute, st, cat)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CASTP_FROM_ROW)

#define NX_C_CASTP_TO_I4_TE(sfx, kind, storage, compute, ld, st, cat)           \
  [NX_C_DTYPE_##sfx] = nx_c_castp_##sfx##_to_i4,
#define NX_C_CASTP_TO_U4_TE(sfx, kind, storage, compute, ld, st, cat)           \
  [NX_C_DTYPE_##sfx] = nx_c_castp_##sfx##_to_u4,
#define NX_C_CASTP_FROM_I4_TE(sfx, kind, storage, compute, ld, st, cat)         \
  [NX_C_DTYPE_##sfx] = nx_c_castp_i4_to_##sfx,
#define NX_C_CASTP_FROM_U4_TE(sfx, kind, storage, compute, ld, st, cat)         \
  [NX_C_DTYPE_##sfx] = nx_c_castp_u4_to_##sfx,
static nx_c_castp_to *const nx_c_castp_to_i4[NX_C_DTYPE_COUNT] = {
    NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CASTP_TO_I4_TE)};
static nx_c_castp_to *const nx_c_castp_to_u4[NX_C_DTYPE_COUNT] = {
    NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CASTP_TO_U4_TE)};
static nx_c_castp_from *const nx_c_castp_from_i4[NX_C_DTYPE_COUNT] = {
    NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CASTP_FROM_I4_TE)};
static nx_c_castp_from *const nx_c_castp_from_u4[NX_C_DTYPE_COUNT] = {
    NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CASTP_FROM_U4_TE)};

/* Contiguous (ignoring size-1 dims) — the only layout the packed path accepts,
   so logical element i maps to storage element offset + i. */
static bool nx_c_cast_dense(const nx_c_ndarray *a) {
  int64_t expect = 1;
  for (int i = a->ndim - 1; i >= 0; i--) {
    if (a->shape[i] == 1) continue;
    if (a->strides[i] != expect) return false;
    expect *= a->shape[i];
  }
  return true;
}
static int64_t nx_c_cast_count(const nx_c_ndarray *a) {
  int64_t t = 1;
  for (int i = 0; i < a->ndim; i++) t *= a->shape[i];
  return t;
}

static nx_c_status nx_c_cast_packed(nx_c_dtype src, nx_c_dtype dst,
                                  const nx_c_ndarray *o, const nx_c_ndarray *in) {
  if (!nx_c_cast_dense(o) || !nx_c_cast_dense(in)) return NX_C_ERR_PACKED;
  int64_t n = nx_c_cast_count(o);
  if (n == 0) return NX_C_OK;
  bool sp = nx_c_dtype_is_packed(src), dp = nx_c_dtype_is_packed(dst);
  if (sp && dp) { /* packed -> packed is a nibble copy (low 4 bits carry over) */
    const uint8_t *S = (const uint8_t *)in->data;
    uint8_t *D = (uint8_t *)o->data;
    for (int64_t i = 0; i < n; i++) {
      int64_t si = in->offset + i, di = o->offset + i;
      uint8_t nib = (uint8_t)((S[si >> 1] >> ((si & 1) * 4)) & 0x0F);
      uint8_t *bp = &D[di >> 1];
      *bp = (di & 1) ? (uint8_t)((*bp & 0x0F) | (nib << 4))
                     : (uint8_t)((*bp & 0xF0) | nib);
    }
    return NX_C_OK;
  }
  if (dp) {
    nx_c_castp_to *fn =
        (dst == NX_C_DTYPE_u4) ? nx_c_castp_to_u4[src] : nx_c_castp_to_i4[src];
    if (fn == NULL) return NX_C_ERR_UNSUPPORTED_DTYPE;
    const char *sbase =
        (const char *)in->data + in->offset * nx_c_elem_size(src);
    fn((uint8_t *)o->data, o->offset, sbase, n);
    return NX_C_OK;
  }
  nx_c_castp_from *fn =
      (src == NX_C_DTYPE_u4) ? nx_c_castp_from_u4[dst] : nx_c_castp_from_i4[dst];
  if (fn == NULL) return NX_C_ERR_UNSUPPORTED_DTYPE;
  char *dbase = (char *)o->data + o->offset * nx_c_elem_size(dst);
  fn(dbase, (const uint8_t *)in->data, in->offset, n);
  return NX_C_OK;
}

/* ══════════════════════════════════════════════════════════════════════════
   Family stubs — the one place this file touches the OCaml runtime
   ═════════════════════════════════════════════════════════════════════════ */

/* Unary. */
NX_C_MAP1_STUB(neg, "neg", nx_c_neg_table, NX_C_COST_BANDWIDTH)
NX_C_MAP1_STUB(recip, "recip", nx_c_recip_table, NX_C_COST_BANDWIDTH)
NX_C_MAP1_STUB(abs, "abs", nx_c_abs_table, NX_C_COST_BANDWIDTH)
NX_C_MAP1_STUB(sign, "sign", nx_c_sign_table, NX_C_COST_BANDWIDTH)
NX_C_MAP1_STUB(sqrt, "sqrt", nx_c_sqrt_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(exp, "exp", nx_c_exp_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(log, "log", nx_c_log_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(sin, "sin", nx_c_sin_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(cos, "cos", nx_c_cos_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(tan, "tan", nx_c_tan_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(asin, "asin", nx_c_asin_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(acos, "acos", nx_c_acos_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(atan, "atan", nx_c_atan_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(sinh, "sinh", nx_c_sinh_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(cosh, "cosh", nx_c_cosh_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(tanh, "tanh", nx_c_tanh_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(trunc, "trunc", nx_c_trunc_table, NX_C_COST_BANDWIDTH)
NX_C_MAP1_STUB(ceil, "ceil", nx_c_ceil_table, NX_C_COST_BANDWIDTH)
NX_C_MAP1_STUB(floor, "floor", nx_c_floor_table, NX_C_COST_BANDWIDTH)
NX_C_MAP1_STUB(round, "round", nx_c_round_table, NX_C_COST_BANDWIDTH)
NX_C_MAP1_STUB(erf, "erf", nx_c_erf_table, NX_C_COST_COMPUTE)

/* Binary. */
NX_C_MAP2_STUB(add, "add", nx_c_add_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(sub, "sub", nx_c_sub_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(mul, "mul", nx_c_mul_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(idiv, "idiv", nx_c_idiv_table, NX_C_COST_COMPUTE)
NX_C_MAP2_STUB(fdiv, "fdiv", nx_c_fdiv_table, NX_C_COST_COMPUTE)
NX_C_MAP2_STUB(mod, "mod", nx_c_mod_table, NX_C_COST_COMPUTE)
NX_C_MAP2_STUB(max, "max", nx_c_max_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(min, "min", nx_c_min_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(pow, "pow", nx_c_pow_table, NX_C_COST_COMPUTE)
NX_C_MAP2_STUB(atan2, "atan2", nx_c_atan2_table, NX_C_COST_COMPUTE)
NX_C_MAP2_STUB(xor, "xor", nx_c_xor_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(or, "or", nx_c_or_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(and, "and", nx_c_and_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(shl, "shl", nx_c_shl_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(shr, "shr", nx_c_shr_table, NX_C_COST_BANDWIDTH)

/* where. */
NX_C_MAP3_STUB(where, "where", nx_c_where_table, NX_C_COST_BANDWIDTH)

/* Comparisons dispatch on the INPUT dtype (bool output), so they call the map
   driver by hand rather than through the output-dispatching funnel, then hand any
   non-NULL status to the engine's own classifier (nx_c_raise_status) — the single
   owner of the status→exception mapping, so a hand-assembled stub and a funnel
   raise the same exception kind for the same status. */
static void nx_c_cmp_run(const char *op, const nx_c_map_table *tbl, value vout,
                        value va, value vb) {
  nx_c_ndarray ops[3];
  nx_c_status s;
  if ((s = nx_c_ndarray_of_value(vout, &ops[0])) != NX_C_OK) nx_c_raise(op, s);
  if ((s = nx_c_ndarray_of_value(va, &ops[1])) != NX_C_OK) nx_c_raise(op, s);
  if ((s = nx_c_ndarray_of_value(vb, &ops[2])) != NX_C_OK) nx_c_raise(op, s);
  nx_c_dtype in = nx_c_dtype_of_value(va);
  nx_c_dtype out = nx_c_dtype_of_value(vout);
  if (in == NX_C_DTYPE_COUNT || out == NX_C_DTYPE_COUNT)
    nx_c_raise(op, NX_C_ERR_BAD_KIND);
  int64_t elem[3] = {nx_c_elem_size(out), nx_c_elem_size(in), nx_c_elem_size(in)};
  s = nx_c_map_run(tbl, in, 2, ops, elem, NX_C_COST_BANDWIDTH, NULL);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
}

#define NX_C_CMP_STUB(cname, opname, table)                                     \
  CAMLprim value caml_nx_c_##cname(value vout, value va, value vb) {            \
    CAMLparam3(vout, va, vb);                                                  \
    nx_c_cmp_run((opname), &(table), vout, va, vb);                            \
    CAMLreturn(Val_unit);                                                      \
  }
NX_C_CMP_STUB(cmpeq, "cmpeq", nx_c_cmpeq_table)
NX_C_CMP_STUB(cmpne, "cmpne", nx_c_cmpne_table)
NX_C_CMP_STUB(cmplt, "cmplt", nx_c_cmplt_table)
NX_C_CMP_STUB(cmple, "cmple", nx_c_cmple_table)

/* cast keys on the (src, dst) pair. Compute pairs run the pair matrix through
   the map driver (dispatched on the dst dtype); anything touching int4/uint4
   takes the serial nibble path. */
CAMLprim value caml_nx_c_cast(value vout, value va) {
  CAMLparam2(vout, va);
  nx_c_ndarray ops[2];
  nx_c_status s;
  if ((s = nx_c_ndarray_of_value(vout, &ops[0])) != NX_C_OK) nx_c_raise("cast", s);
  if ((s = nx_c_ndarray_of_value(va, &ops[1])) != NX_C_OK) nx_c_raise("cast", s);
  nx_c_dtype dst = nx_c_dtype_of_value(vout);
  nx_c_dtype src = nx_c_dtype_of_value(va);
  if (dst == NX_C_DTYPE_COUNT || src == NX_C_DTYPE_COUNT)
    nx_c_raise("cast", NX_C_ERR_BAD_KIND);
  if (nx_c_dtype_is_packed(src) || nx_c_dtype_is_packed(dst)) {
    s = nx_c_cast_packed(src, dst, &ops[0], &ops[1]);
  } else {
    int64_t elem[2] = {nx_c_elem_size(dst), nx_c_elem_size(src)};
    s = nx_c_map_run(&nx_c_cast_tables[src], dst, 1, ops, elem, NX_C_COST_BANDWIDTH,
                    NULL);
  }
  if (s != NX_C_OK) nx_c_raise_status("cast", s);
  CAMLreturn(Val_unit);
}

/* Equivalence gate for the local cast-fast-path converters. Returns the number
   of inputs where nx_c_f32_to_f16 (both the portable and, where compiled, the
   active hardware form) or nx_c_f32_to_bf16 disagree by even one bit with the
   canonical nx_buffer_stubs.h converters — MUST be 0. Covers all 65536 f16 bit
   patterns round-tripped through half_to_float, a dense sweep of the rounding
   band (every 13-bit round/sticky decision at representative magnitudes and both
   signs), and the NaN/inf/subnormal/overflow f32 edges. Pure C, no allocation;
   the backend-local maintenance stub asserts the result is 0. This is a plain
   C hook rather than a shipping OCaml primitive. */
int64_t nx_c_cast_convert_selfcheck(void) {
  int64_t mism = 0;

#define NX_C_CHK_F16(f)                                                         \
  do {                                                                         \
    float f_ = (f);                                                            \
    uint16_t want = float_to_half(f_);                                        \
    mism += (nx_c_f32_to_f16_sw(f_) != want);                                  \
    mism += (nx_c_f32_to_f16(f_) != want);                                     \
  } while (0)
#define NX_C_CHK_BF16(f)                                                        \
  do {                                                                         \
    float f_ = (f);                                                            \
    mism += (nx_c_f32_to_bf16(f_) != float_to_bfloat16(f_));                   \
  } while (0)
#define NX_C_CHK(f)                                                             \
  do {                                                                         \
    NX_C_CHK_F16(f);                                                            \
    NX_C_CHK_BF16(f);                                                           \
  } while (0)

  /* All 65536 f16 patterns via the inverse converter (exact grid points, all
     subnormals, inf, and the f16 NaN payloads). */
  for (uint32_t h = 0; h < 65536u; h++) NX_C_CHK(half_to_float((uint16_t)h));

  /* Rounding band: for each biased exponent from below the smallest subnormal to
     past overflow, sweep the low 13 mantissa bits (the round/sticky decision)
     across a spread of high mantissa bits, both signs. */
  for (uint32_t ef = 0x30000000u; ef <= 0x49000000u; ef += 0x00800000u) {
    for (uint32_t mhi = 0; mhi < 0x00800000u; mhi += 0x00100000u) {
      for (uint32_t mlo = 0; mlo < 0x00002000u; mlo++) {
        uint32_t bits = ef | (mhi & 0x007FE000u) | mlo;
        union { uint32_t u; float f; } p = {.u = bits};
        NX_C_CHK(p.f);
        union { uint32_t u; float f; } n = {.u = bits | 0x80000000u};
        NX_C_CHK(n.f);
      }
    }
  }

  /* Explicit edges: signed zeros, subnormal/overflow boundaries, inf, and a
     spread of NaN payloads (both signs). */
  static const uint32_t edges[] = {
      0x00000000u, 0x80000000u, /* +/-0 */
      0x33000000u, 0x33800000u, 0x38000000u, 0x38800000u, /* subnormal edges */
      0x477FE000u, 0x47800000u, 0x47FFE000u, 0x48000000u, /* overflow edges */
      0x7F7FFFFFu, 0xFF7FFFFFu, /* +/-max finite f32 */
      0x7F800000u, 0xFF800000u, /* +/-inf */
      0x7F800001u, 0x7FABCDEFu, 0x7FC00000u, 0x7FFFFFFFu, /* NaN payloads */
      0xFF800001u, 0xFFABCDEFu, 0xFFC00000u, 0xFFFFFFFFu,
  };
  for (size_t i = 0; i < sizeof(edges) / sizeof(edges[0]); i++) {
    union { uint32_t u; float f; } p = {.u = edges[i]};
    NX_C_CHK(p.f);
  }

#undef NX_C_CHK
#undef NX_C_CHK_F16
#undef NX_C_CHK_BF16
  return mism;
}
