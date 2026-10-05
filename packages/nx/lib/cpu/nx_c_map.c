/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_map.c — the map kernel family: elementwise unary, binary, comparison,
   where, fma and cast. Every kernel is a 5-line inner loop over one 1-D run honoring
   the nx_c_map_loop ABI (nx_c.h); the engine (nx_c_engine.c) owns coalescing,
   strategy, threading, dispatch null-checks, and the funnel. Kernels state scalar
   semantics only, never touch the OCaml runtime, and never fail — the backend
   contract encodes the normative behavior.

   Generation is table-driven: NX_C_FOR_EACH_COMPUTE_DTYPE walks the compute rows
   of the one dtype table (nx_c.h), so a per-op dispatch table is a set of
   designated initializers whose absent slots are NULL — the engine turns NULL
   into a clean NX_C_ERR_UNSUPPORTED_DTYPE / NX_C_ERR_PACKED status. Cast is the one
   pair-indexed op: a src×dst matrix of specialized converters (src LOAD ->
   intermediate -> dst STORE), and the sub-byte dtypes (bit, int4, uint4) cast
   through their byte dtypes.

   Only the family stubs (bottom) reach the OCaml runtime, via the engine funnel
   or the sanctioned nx_c_raise / nx_c_raise_status raisers — never a kernel. */

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_engine.h"
#include "nx_c_packed.h"

/* ── Fast/strided contiguous fast path ─────────────────────────────────────

   The engine hands byte steps; for a contiguous run every step equals the
   element size. A kernel branches on that once and, on the contiguous branch,
   walks typed pointers (compile-time unit stride) so clang autovectorizes the
   f32/f64/int arithmetic ops; the strided branch keeps the generic byte walk.
   Both branches share one scalar expression. `es` folds to a constant, so only
   the compare uses it — the hot loop is plain array indexing.

   A step of 0 is the third shape worth naming: it is how a 0-d operand
   broadcast against a full tensor arrives (the `_s` scalar family, and every
   `binop x (zeros_like x)`), and the generic walk would re-read that one cell
   per element and stay scalar. A multi-input kernel therefore also branches on
   a 0 step, hoisting that operand's single load out of the loop and leaving the
   remaining pointers unit-stride — so the loop still vectorizes and reads half
   the memory. Splat branches restate the SAME scalar expression, so results
   are bit-identical to the contiguous branch, NaNs included (NX_C_NAN2). Only
   the innermost coalesced step is 0 for a 0-d broadcast; a partially broadcast
   operand (say [1,k] against [n,k]) keeps an innermost step of `es` and stays
   on the contiguous branch. */

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
    } else if (so == es && sa == es && sb == 0) {                             \
      storage *pO = (storage *)out;                                           \
      const storage *pA = (const storage *)in0;                               \
      const compute vb = nx_c_ld_##sfx(in1);                                  \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = (compute)ld(pA[i]);                                       \
        pO[i] = (storage)st(EXPR);                                             \
      }                                                                        \
    } else if (so == es && sa == 0 && sb == es) {                             \
      storage *pO = (storage *)out;                                           \
      const compute va = nx_c_ld_##sfx(in0);                                  \
      const storage *pB = (const storage *)in1;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
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
    } else if (so == 1 && sa == es && sb == 0) {                              \
      uint8_t *pO = (uint8_t *)out;                                           \
      const storage *pA = (const storage *)in0;                               \
      const compute vb = nx_c_ld_##sfx(in1);                                  \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = (compute)ld(pA[i]);                                       \
        pO[i] = (uint8_t)(EXPR);                                               \
      }                                                                        \
    } else if (so == 1 && sa == 0 && sb == es) {                              \
      uint8_t *pO = (uint8_t *)out;                                           \
      const compute va = nx_c_ld_##sfx(in0);                                  \
      const storage *pB = (const storage *)in1;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
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

/* An f16, bf16 or float8 dst computes in float. A wider src reaches its
   encoder through double_to_float_odd, so the value rounds once; a 64-bit
   integer reaches it as a double rounded to odd from its exact value. */
#define NX_C_NARROW_ODD(v)                                                      \
  _Generic((v),                                                                 \
      float: (v),                                                               \
      int64_t: double_to_float_odd(i64_to_double_odd((int64_t)(v))),            \
      uint64_t: double_to_float_odd(u64_to_double_odd((uint64_t)(v))),          \
      default: double_to_float_odd((double)(v)))
#define NX_C_FNARROW_f16(dcompute, v) NX_C_NARROW_ODD(v)
#define NX_C_FNARROW_f32(dcompute, v) ((dcompute)(v))
#define NX_C_FNARROW_f64(dcompute, v) ((dcompute)(v))
#define NX_C_FNARROW_bf16(dcompute, v) NX_C_NARROW_ODD(v)
#define NX_C_FNARROW_f8e4m3(dcompute, v) NX_C_NARROW_ODD(v)
#define NX_C_FNARROW_f8e5m2(dcompute, v) NX_C_NARROW_ODD(v)
#define NX_C_CASTVAL_NX_C_CAT_FLOAT(dcompute, dsfx, scat, v)                     \
  NX_C_FNARROW_##dsfx(dcompute, NX_C_SREAL_##scat(v))
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
#define NX_C_TROW_NUM(sfx, storage, compute, ld, st, cat)                       \
  NX_C_TN_##cat(NX_C_CUROP, sfx) /* sint,uint,float,complex */

#define NX_C_TFC_NX_C_CAT_SINT(op, sfx)
#define NX_C_TFC_NX_C_CAT_UINT(op, sfx)
#define NX_C_TFC_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TFC_NX_C_CAT_COMPLEX(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TFC_NX_C_CAT_BOOL(op, sfx)
#define NX_C_TROW_FC(sfx, storage, compute, ld, st, cat)                        \
  NX_C_TFC_##cat(NX_C_CUROP, sfx) /* float,complex */

#define NX_C_TFO_NX_C_CAT_SINT(op, sfx)
#define NX_C_TFO_NX_C_CAT_UINT(op, sfx)
#define NX_C_TFO_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TFO_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TFO_NX_C_CAT_BOOL(op, sfx)
#define NX_C_TROW_FLOAT(sfx, storage, compute, ld, st, cat)                     \
  NX_C_TFO_##cat(NX_C_CUROP, sfx) /* float only */

#define NX_C_TIF_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TIF_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TIF_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TIF_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TIF_NX_C_CAT_BOOL(op, sfx)
#define NX_C_TROW_INTF(sfx, storage, compute, ld, st, cat)                      \
  NX_C_TIF_##cat(NX_C_CUROP, sfx) /* sint,uint,float */

#define NX_C_TMM_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TMM_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TMM_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TMM_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TMM_NX_C_CAT_BOOL(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TROW_MINMAX(sfx, storage, compute, ld, st, cat)                    \
  NX_C_TMM_##cat(NX_C_CUROP, sfx) /* sint,uint,float,bool */

#define NX_C_TBW_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TBW_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TBW_NX_C_CAT_FLOAT(op, sfx)
#define NX_C_TBW_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TBW_NX_C_CAT_BOOL(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TROW_BITWISE(sfx, storage, compute, ld, st, cat)                   \
  NX_C_TBW_##cat(NX_C_CUROP, sfx) /* sint,uint,bool */

#define NX_C_TSH_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TSH_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TSH_NX_C_CAT_FLOAT(op, sfx)
#define NX_C_TSH_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TSH_NX_C_CAT_BOOL(op, sfx)
#define NX_C_TROW_SHIFT(sfx, storage, compute, ld, st, cat)                     \
  NX_C_TSH_##cat(NX_C_CUROP, sfx) /* sint,uint */

#define NX_C_TAL_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TAL_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TAL_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TAL_NX_C_CAT_COMPLEX(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TAL_NX_C_CAT_BOOL(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TROW_ALL(sfx, storage, compute, ld, st, cat)                       \
  NX_C_TAL_##cat(NX_C_CUROP, sfx) /* every compute dtype */

#define NX_C_TOR_NX_C_CAT_SINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TOR_NX_C_CAT_UINT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TOR_NX_C_CAT_FLOAT(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TOR_NX_C_CAT_COMPLEX(op, sfx)
#define NX_C_TOR_NX_C_CAT_BOOL(op, sfx) NX_C_TE(op, sfx)
#define NX_C_TROW_ORD(sfx, storage, compute, ld, st, cat)                       \
  NX_C_TOR_##cat(NX_C_CUROP, sfx) /* sint,uint,float,bool (no ordered complex) */

/* ══════════════════════════════════════════════════════════════════════════
   Unary ops (23)
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
#define NX_C_NEG_KROW(sfx, storage, compute, ld, st, cat)                       \
  NX_C_NEG_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_NEG_KROW)

/* Complex division by Smith's algorithm, as OCaml's Complex.div computes it:
   the divisor's larger part divides its smaller, so that no intermediate
   overflows or underflows where the quotient's parts do not. C's operator calls
   the platform's runtime (__divdc3), whose scaling gives a NaN part where the
   quotient's is 0: (0 + 2^-50 i) / (0 - 2^-1074 i) is -inf + NaN i with
   glibc's libgcc. */
static inline nx_c_complex32 nx_c_cdiv32(nx_c_complex32 a, nx_c_complex32 b) {
  float ar = crealf(a), ai = cimagf(a), br = crealf(b), bi = cimagf(b);
  if (fabsf(br) >= fabsf(bi)) {
    float r = bi / br, d = br + r * bi;
    return CMPLXF((ar + r * ai) / d, (ai - r * ar) / d);
  }
  float r = br / bi, d = bi + r * br;
  return CMPLXF((r * ar + ai) / d, (r * ai - ar) / d);
}

static inline nx_c_complex64 nx_c_cdiv64(nx_c_complex64 a, nx_c_complex64 b) {
  double ar = creal(a), ai = cimag(a), br = creal(b), bi = cimag(b);
  if (fabs(br) >= fabs(bi)) {
    double r = bi / br, d = br + r * bi;
    return CMPLX((ar + r * ai) / d, (ai - r * ar) / d);
  }
  double r = br / bi, d = bi + r * br;
  return CMPLX((r * ar + ai) / d, (r * ai - ar) / d);
}

#define NX_C_CDIV(a, b)                                                        \
  _Generic((a), nx_c_complex32: nx_c_cdiv32, nx_c_complex64: nx_c_cdiv64)(a, b)

/* NaN operands. Given two NaNs, an arm64 or x86 instruction returns its first
   register's, and clang orders the operands of a commutative operation (add,
   mul, fma's product) per loop: a vector loop over a splat operand puts the
   splat second, the scalar loop after it may put it first. So float arithmetic
   picks the NaN in the source, as nx_c_fmax does: a NaN result is the first
   NaN operand, its bits unchanged, or the operation's own NaN when no operand
   is NaN. A complex result's NaN part is the first NaN among the operands'
   parts, real before imaginary, or else the positive quiet NaN: a complex
   operation is several float operations, and the sign of their own NaN
   depends on how the compiler fused and negated them. float16, bfloat16 and
   float8 compute in float, and widening quiets a signaling NaN.

   Each NaN test selects over the computed result. Under its default
   -ftrapping-math, gcc moves an operation that a test can skip onto the
   branch that uses it, will not evaluate it unconditionally there, and leaves
   the loop scalar. Beside a splat operand the compiler makes that operand's
   test once.

   Where every operand is an array, the two tests cost more than the
   operation. There a float or complex kernel (NX_C_BKN, NX_C_FMAKN) computes
   blocks of NX_C_NAN_BLOCK elements by the plain operation, noting whether a
   result is NaN, and computes again by the rule only a block that has one.
   The rule changes nothing but NaN results, and the second pass reads the
   operands as the first did, since a destination shares no memory with
   them. */
#define NX_C_DEFINE_NAN(T, sfx)                                                \
  static inline T nx_c_nan2_##sfx(T a, T b, T r) {                             \
    r = isnan(b) ? b : r;                                                      \
    return isnan(a) ? a : r;                                                   \
  }                                                                            \
  static inline T nx_c_nan3_##sfx(T a, T b, T c, T r) {                        \
    r = isnan(c) ? c : r;                                                      \
    return nx_c_nan2_##sfx(a, b, r);                                           \
  }
NX_C_DEFINE_NAN(float, f)
NX_C_DEFINE_NAN(double, d)
#undef NX_C_DEFINE_NAN
#define NX_C_NAN2(a, b, r)                                                     \
  _Generic((a), float: nx_c_nan2_f, double: nx_c_nan2_d)(a, b, r)
#define NX_C_NAN3(a, b, c, r)                                                  \
  _Generic((a), float: nx_c_nan3_f, double: nx_c_nan3_d)(a, b, c, r)

#define NX_C_DEFINE_CNAN(C, T, re, im, make)                                   \
  static inline C nx_c_cnan_##C(C a, C b, C r) {                               \
    T n = NX_C_NAN2(re(a), im(a), NX_C_NAN2(re(b), im(b), (T)NAN));            \
    T rr = re(r), ri = im(r);                                                  \
    return make(rr != rr ? n : rr, ri != ri ? n : ri);                         \
  }
NX_C_DEFINE_CNAN(nx_c_complex32, float, crealf, cimagf, CMPLXF)
NX_C_DEFINE_CNAN(nx_c_complex64, double, creal, cimag, CMPLX)
#undef NX_C_DEFINE_CNAN
#define NX_C_CNAN(a, b, r)                                                     \
  _Generic((a), nx_c_complex32: nx_c_cnan_nx_c_complex32,                       \
      nx_c_complex64: nx_c_cnan_nx_c_complex64)(a, b, r)

/* Whether a result is NaN, or has a NaN part. An int, as is the flag that a
   plain loop ors it into: gcc leaves a loop scalar whose bool flag is
   narrower than its floats. A complex test is a bitwise or, which evaluates
   both parts and leaves the loop no branch. */
static inline int nx_c_isnan_f(float r) { return isnan(r); }
static inline int nx_c_isnan_d(double r) { return isnan(r); }
static inline int nx_c_isnan_c32(nx_c_complex32 r) {
  return isnan(crealf(r)) | isnan(cimagf(r));
}
static inline int nx_c_isnan_c64(nx_c_complex64 r) {
  return isnan(creal(r)) | isnan(cimag(r));
}
#define NX_C_ISNAN(r)                                                          \
  _Generic((r), float: nx_c_isnan_f, double: nx_c_isnan_d,                     \
      nx_c_complex32: nx_c_isnan_c32, nx_c_complex64: nx_c_isnan_c64)(r)

#define NX_C_NAN_BLOCK 1024

/* Kept out of line: merged into the kernel below, the block loop's frame was
   built on every call, and a run beside a splat operand, which never enters
   the loop, took 1.5 times as long (1024-element runs on an M1 Max). */
#if defined(__GNUC__) || defined(__clang__)
#define NX_C_NOINLINE __attribute__((noinline))
#else
#define NX_C_NOINLINE
#endif

/* The kernel [op]. A run beside a splat operand, whose NaN test the compiler
   makes once, takes [op]_nan whole. Any other run goes by blocks: [op]_plain
   computes a block and says whether a result is NaN, and [op]_nan computes
   such a block again. */
#define NX_C_NANK(op, sfx, nin) NX_C_NANK_I(op, sfx, nin)
#define NX_C_NANK_I(op, sfx, nin)                                              \
  NX_C_NOINLINE static void nx_c_##op##_blocks_##sfx(                          \
      char *const *pp, const int64_t *ssx, int64_t nn, void *ctx) {            \
    for (int64_t s = 0; s < nn; s += NX_C_NAN_BLOCK) {                         \
      const int64_t n = nn - s < NX_C_NAN_BLOCK ? nn - s : NX_C_NAN_BLOCK;    \
      char *q[nin + 1];                                                        \
      for (int k = 0; k <= nin; k++) q[k] = pp[k] + s * ssx[k];               \
      if (nx_c_##op##_plain_##sfx(q, ssx, n))                                  \
        nx_c_##op##_nan_##sfx(q, ssx, n, ctx);                                 \
    }                                                                          \
  }                                                                            \
  static void nx_c_##op##_##sfx(char *const *pp, const int64_t *ssx,           \
                               int64_t nn, void *ctx) {                        \
    for (int k = 1; k <= nin; k++)                                             \
      if (ssx[k] == 0) {                                                       \
        nx_c_##op##_nan_##sfx(pp, ssx, nn, ctx);                               \
        return;                                                                \
      }                                                                        \
    nx_c_##op##_blocks_##sfx(pp, ssx, nn, ctx);                                \
  }

/* Binary float or complex arithmetic: EXPR is the plain operation, RULE the
   same with the NaN rule. */
#define NX_C_BKN(op, sfx, storage, compute, ld, st, EXPR, RULE)               \
  NX_C_BKN_I(op, sfx, storage, compute, ld, st, EXPR, RULE)
#define NX_C_BKN_I(op, sfx, storage, compute, ld, st, EXPR, RULE)             \
  NX_C_BK_I(op##_nan, sfx, storage, compute, ld, st, RULE)                    \
  static int nx_c_##op##_plain_##sfx(char *const *pp, const int64_t *ssx,     \
                                     int64_t nn) {                             \
    char *out = pp[0];                                                         \
    const char *in0 = pp[1];                                                   \
    const char *in1 = pp[2];                                                   \
    const int64_t so = ssx[0], sa = ssx[1], sb = ssx[2];                       \
    const int64_t es = (int64_t)sizeof(storage);                              \
    int has_nan = 0;                                                           \
    if (so == es && sa == es && sb == es) {                                   \
      storage *pO = (storage *)out;                                           \
      const storage *pA = (const storage *)in0;                               \
      const storage *pB = (const storage *)in1;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = (compute)ld(pA[i]);                                       \
        compute vb = (compute)ld(pB[i]);                                       \
        compute r = EXPR;                                                      \
        has_nan |= NX_C_ISNAN(r);                                              \
        pO[i] = (storage)st(r);                                                \
      }                                                                        \
    } else {                                                                   \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = nx_c_ld_##sfx(in0 + i * sa);                              \
        compute vb = nx_c_ld_##sfx(in1 + i * sb);                              \
        compute r = EXPR;                                                      \
        has_nan |= NX_C_ISNAN(r);                                              \
        nx_c_st_##sfx(out + i * so, r);                                        \
      }                                                                        \
    }                                                                          \
    return has_nan;                                                            \
  }                                                                            \
  NX_C_NANK_I(op, sfx, 2)

#define NX_C_RECIP_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                  \
  NX_C_UK(recip, sfx, storage, compute, ld, st, ((vx) == 0 ? 0 : 1 / (vx)))
#define NX_C_RECIP_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                  \
  NX_C_UK(recip, sfx, storage, compute, ld, st, ((vx) == 0 ? 0 : 1 / (vx)))
#define NX_C_RECIP_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                 \
  NX_C_UK(recip, sfx, storage, compute, ld, st, ((compute)1 / (vx)))
#define NX_C_RECIP_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)               \
  NX_C_UK(recip, sfx, storage, compute, ld, st, NX_C_CDIV((compute)1, (vx)))
#define NX_C_RECIP_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_RECIP_KROW(sfx, storage, compute, ld, st, cat)                     \
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
#define NX_C_ABS_KROW(sfx, storage, compute, ld, st, cat)                       \
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
#define NX_C_SIGN_KROW(sfx, storage, compute, ld, st, cat)                      \
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
#define NX_C_TRANS_KROW(sfx, storage, compute, ld, st, cat)                     \
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

/* erf, log1p and expm1: float only (C has no complex forms). */
#define NX_C_FOK_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_UK(NX_C_CUROP, sfx, storage, compute, ld, st,                             \
         NX_C_MFN(NX_C_CUROP, compute)(vx))
#define NX_C_FOK_NX_C_CAT_SINT(sfx, storage, compute, ld, st)
#define NX_C_FOK_NX_C_CAT_UINT(sfx, storage, compute, ld, st)
#define NX_C_FOK_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_FOK_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_FO_KROW(sfx, storage, compute, ld, st, cat)                        \
  NX_C_FOK_##cat(sfx, storage, compute, ld, st)
#define NX_C_FLOATOP(op)                                                        \
  NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_FO_KROW)                                      \
  static const nx_c_map_table nx_c_##op##_table = {                             \
      .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_FLOAT)}};

#define NX_C_CUROP erf
NX_C_FLOATOP(erf)
#undef NX_C_CUROP
#define NX_C_CUROP log1p
NX_C_FLOATOP(log1p)
#undef NX_C_CUROP
#define NX_C_CUROP expm1
NX_C_FLOATOP(expm1)
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
#define NX_C_RND_KROW(sfx, storage, compute, ld, st, cat)                       \
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
  NX_C_BKN(NX_C_CUROP, sfx, storage, compute, ld, st, ((va)NX_C_CURSYM(vb)),     \
          NX_C_NAN2(va, vb, (va)NX_C_CURSYM(vb)))
#define NX_C_ARK_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)                 \
  NX_C_BKN(NX_C_CUROP, sfx, storage, compute, ld, st, ((va)NX_C_CURSYM(vb)),     \
          NX_C_CNAN(va, vb, (va)NX_C_CURSYM(vb)))
#define NX_C_ARK_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_ARITH_KROW(sfx, storage, compute, ld, st, cat)                     \
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
#define NX_C_IDIV_KROW(sfx, storage, compute, ld, st, cat)                      \
  NX_C_IDIV_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_IDIV_KROW)
#define NX_C_CUROP idiv
static const nx_c_map_table nx_c_idiv_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_INTF)}};
#undef NX_C_CUROP

/* fdiv: true division for float and complex (int div routes to idiv). */
#define NX_C_FDIV_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                  \
  NX_C_BKN(fdiv, sfx, storage, compute, ld, st, ((va) / (vb)),                  \
          NX_C_NAN2(va, vb, (va) / (vb)))
#define NX_C_FDIV_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)                \
  NX_C_BKN(fdiv, sfx, storage, compute, ld, st, NX_C_CDIV(va, vb),              \
          NX_C_CNAN(va, vb, NX_C_CDIV(va, vb)))
#define NX_C_FDIV_NX_C_CAT_SINT(sfx, storage, compute, ld, st)
#define NX_C_FDIV_NX_C_CAT_UINT(sfx, storage, compute, ld, st)
#define NX_C_FDIV_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_FDIV_KROW(sfx, storage, compute, ld, st, cat)                      \
  NX_C_FDIV_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_FDIV_KROW)
#define NX_C_CUROP fdiv
static const nx_c_map_table nx_c_fdiv_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_FC)}};
#undef NX_C_CUROP

/* mod: integer remainder of the sign of the dividend, by-zero -> the dividend
   so that a = b * idiv a b + mod a b holds for every b, and by -1 -> 0 guarded
   against INT_MIN overflow; fmod on floats. No complex remainder. */
#define NX_C_MOD_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(mod, sfx, storage, compute, ld, st,                                   \
         ((vb) == 0 ? (va) : (vb) == -1 ? (compute)0 : (va) % (vb)))
#define NX_C_MOD_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_BK(mod, sfx, storage, compute, ld, st,                                   \
         ((vb) == 0 ? (va) : (va) % (vb)))
#define NX_C_MOD_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_BK(mod, sfx, storage, compute, ld, st, NX_C_MFN(fmod, compute)(va, vb))
#define NX_C_MOD_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_MOD_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_MOD_KROW(sfx, storage, compute, ld, st, cat)                       \
  NX_C_MOD_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_MOD_KROW)
#define NX_C_CUROP mod
static const nx_c_map_table nx_c_mod_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_INTF)}};
#undef NX_C_CUROP

/* max / min: IEEE 754-2019 maximum and minimum on floats (nx_c_fmax and
   nx_c_fmin: NaN propagates, -0 orders below +0), ordered on int/bool,
   rejected on complex. The compute width carries signedness, so > / < are the
   right signed/unsigned comparison per dtype. */
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
  NX_C_BK(NX_C_CUROP, sfx, storage, compute, ld, st, NX_C_CURFLOAT(va, vb))
#define NX_C_MMK_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_MINMAX_KROW(sfx, storage, compute, ld, st, cat)                    \
  NX_C_MMK_##cat(sfx, storage, compute, ld, st)

#define NX_C_MINMAX(op)                                                         \
  NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_MINMAX_KROW)                                  \
  static const nx_c_map_table nx_c_##op##_table = {                             \
      .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_MINMAX)}};

#define NX_C_CUROP max
#define NX_C_CURSYM >
#define NX_C_CURFLOAT nx_c_fmax
NX_C_MINMAX(max)
#undef NX_C_CURFLOAT
#undef NX_C_CURSYM
#undef NX_C_CUROP
#define NX_C_CUROP min
#define NX_C_CURSYM <
#define NX_C_CURFLOAT nx_c_fmin
NX_C_MINMAX(min)
#undef NX_C_CURFLOAT
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
#define NX_C_POW_KROW(sfx, storage, compute, ld, st, cat)                       \
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
#define NX_C_ATAN2_KROW(sfx, storage, compute, ld, st, cat)                     \
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
#define NX_C_BITWISE_KROW(sfx, storage, compute, ld, st, cat)                   \
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
#define NX_C_SHL_KROW(sfx, storage, compute, ld, st, cat)                       \
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
#define NX_C_SHR_KROW(sfx, storage, compute, ld, st, cat)                       \
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
#define NX_C_CEQ_KROW(sfx, storage, compute, ld, st, cat)                       \
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
#define NX_C_CORD_KROW(sfx, storage, compute, ld, st, cat)                      \
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

/* The splat branches read the varying arm into a local BEFORE the select rather
   than indexing inside it. Written as `pC[i] ? pA[i] : vb` the surviving arm is
   a conditional load, which clang keeps as a real branch — and on a data-driven
   mask that mispredicts every other element, making the splat form ~6x slower
   than the byte walk it replaces. Loading unconditionally leaves a plain select
   the vectorizer turns into a blend. Not cosmetic; do not fold the load back
   into the ternary. */

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
    } else if (so == es && sa == es && sb == 0 && sc == 1) {                  \
      storage *pO = (storage *)out;                                           \
      const uint8_t *pC = (const uint8_t *)cnd;                               \
      const storage *pA = (const storage *)in0;                               \
      const storage vb = *(const storage *)in1;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
        storage va = pA[i];                                                   \
        pO[i] = pC[i] ? va : vb;                                              \
      }                                                                        \
    } else if (so == es && sa == 0 && sb == es && sc == 1) {                  \
      storage *pO = (storage *)out;                                           \
      const uint8_t *pC = (const uint8_t *)cnd;                               \
      const storage va = *(const storage *)in0;                               \
      const storage *pB = (const storage *)in1;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
        storage vb = pB[i];                                                   \
        pO[i] = pC[i] ? va : vb;                                              \
      }                                                                        \
    } else if (so == es && sa == 0 && sb == 0 && sc == 1) {                   \
      storage *pO = (storage *)out;                                           \
      const uint8_t *pC = (const uint8_t *)cnd;                               \
      const storage va = *(const storage *)in0;                               \
      const storage vb = *(const storage *)in1;                               \
      for (int64_t i = 0; i < nn; i++) pO[i] = pC[i] ? va : vb;                \
    } else {                                                                   \
      for (int64_t i = 0; i < nn; i++) {                                       \
        uint8_t c = *(const uint8_t *)(cnd + i * sc);                          \
        *(storage *)(out + i * so) = c ? *(const storage *)(in0 + i * sa)      \
                                       : *(const storage *)(in1 + i * sb);     \
      }                                                                        \
    }                                                                          \
  }
#define NX_C_WHERE_KROW(sfx, storage, compute, ld, st, cat)                     \
  NX_C_WK(sfx, storage)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_WHERE_KROW)
#define NX_C_CUROP where
static const nx_c_map_table nx_c_where_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_ALL)}};
#undef NX_C_CUROP

/* ══════════════════════════════════════════════════════════════════════════
   fma (map3) — a * b + c
   ═════════════════════════════════════════════════════════════════════════ */

/* A float takes the libm multiply-add of its compute type, rounded once: float16,
   bfloat16 and float8 take float32's and round again on the store. An integer
   wraps, its signed forms running in the unsigned width as add and mul do. As
   in NX_C_BK, a contiguous run with one operand broadcast (a 0 step) loads that
   operand once, outside a loop that still vectorizes. */
#define NX_C_FMAK(op, sfx, storage, compute, ld, st, EXPR)                      \
  static void nx_c_##op##_##sfx(char *const *pp, const int64_t *ssx,            \
                               int64_t nn, void *ctx) {                        \
    (void)ctx;                                                                 \
    char *out = pp[0];                                                         \
    const char *in0 = pp[1], *in1 = pp[2], *in2 = pp[3];                       \
    const int64_t so = ssx[0], sa = ssx[1], sb = ssx[2], sc = ssx[3];          \
    const int64_t es = (int64_t)sizeof(storage);                              \
    if (so == es && sa == es && sb == es && sc == es) {                       \
      storage *pO = (storage *)out;                                           \
      const storage *pA = (const storage *)in0;                               \
      const storage *pB = (const storage *)in1;                               \
      const storage *pC = (const storage *)in2;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = (compute)ld(pA[i]);                                       \
        compute vb = (compute)ld(pB[i]);                                       \
        compute vc = (compute)ld(pC[i]);                                       \
        pO[i] = (storage)st(EXPR);                                             \
      }                                                                        \
    } else if (so == es && sa == 0 && sb == es && sc == es) {                 \
      storage *pO = (storage *)out;                                           \
      const compute va = nx_c_ld_##sfx(in0);                                  \
      const storage *pB = (const storage *)in1;                               \
      const storage *pC = (const storage *)in2;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute vb = (compute)ld(pB[i]);                                       \
        compute vc = (compute)ld(pC[i]);                                       \
        pO[i] = (storage)st(EXPR);                                             \
      }                                                                        \
    } else if (so == es && sa == es && sb == 0 && sc == es) {                 \
      storage *pO = (storage *)out;                                           \
      const storage *pA = (const storage *)in0;                               \
      const compute vb = nx_c_ld_##sfx(in1);                                  \
      const storage *pC = (const storage *)in2;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = (compute)ld(pA[i]);                                       \
        compute vc = (compute)ld(pC[i]);                                       \
        pO[i] = (storage)st(EXPR);                                             \
      }                                                                        \
    } else if (so == es && sa == es && sb == es && sc == 0) {                 \
      storage *pO = (storage *)out;                                           \
      const storage *pA = (const storage *)in0;                               \
      const storage *pB = (const storage *)in1;                               \
      const compute vc = nx_c_ld_##sfx(in2);                                  \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = (compute)ld(pA[i]);                                       \
        compute vb = (compute)ld(pB[i]);                                       \
        pO[i] = (storage)st(EXPR);                                             \
      }                                                                        \
    } else {                                                                   \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute va = nx_c_ld_##sfx(in0 + i * sa);                              \
        compute vb = nx_c_ld_##sfx(in1 + i * sb);                              \
        compute vc = nx_c_ld_##sfx(in2 + i * sc);                              \
        nx_c_st_##sfx(out + i * so, (EXPR));                                    \
      }                                                                        \
    }                                                                          \
  }

/* fma over floats, as NX_C_BKN. */
#define NX_C_FMAKN(sfx, storage, compute, ld, st)                               \
  NX_C_FMAK(fma_nan, sfx, storage, compute, ld, st,                             \
           NX_C_NAN3(va, vb, vc, NX_C_MFN(fma, compute)(va, vb, vc)))            \
  static int nx_c_fma_plain_##sfx(char *const *pp, const int64_t *ssx,         \
                                  int64_t nn) {                                \
    char *out = pp[0];                                                         \
    const char *in0 = pp[1], *in1 = pp[2], *in2 = pp[3];                       \
    const int64_t so = ssx[0], sa = ssx[1], sb = ssx[2], sc = ssx[3];          \
    const int64_t es = (int64_t)sizeof(storage);                              \
    int has_nan = 0;                                                           \
    if (so == es && sa == es && sb == es && sc == es) {                       \
      storage *pO = (storage *)out;                                           \
      const storage *pA = (const storage *)in0;                               \
      const storage *pB = (const storage *)in1;                               \
      const storage *pC = (const storage *)in2;                               \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute r = NX_C_MFN(fma, compute)((compute)ld(pA[i]),                 \
                                           (compute)ld(pB[i]),                 \
                                           (compute)ld(pC[i]));                \
        has_nan |= NX_C_ISNAN(r);                                              \
        pO[i] = (storage)st(r);                                                \
      }                                                                        \
    } else {                                                                   \
      for (int64_t i = 0; i < nn; i++) {                                       \
        compute r = NX_C_MFN(fma, compute)(nx_c_ld_##sfx(in0 + i * sa),        \
                                           nx_c_ld_##sfx(in1 + i * sb),        \
                                           nx_c_ld_##sfx(in2 + i * sc));       \
        has_nan |= NX_C_ISNAN(r);                                              \
        nx_c_st_##sfx(out + i * so, r);                                        \
      }                                                                        \
    }                                                                          \
    return has_nan;                                                            \
  }                                                                            \
  NX_C_NANK(fma, sfx, 3)
#define NX_C_FMA_NX_C_CAT_SINT(sfx, storage, compute, ld, st)                    \
  NX_C_FMAK(fma, sfx, storage, compute, ld, st,                                 \
           (compute)((uint64_t)(va) * (uint64_t)(vb) + (uint64_t)(vc)))
#define NX_C_FMA_NX_C_CAT_UINT(sfx, storage, compute, ld, st)                    \
  NX_C_FMAK(fma, sfx, storage, compute, ld, st, ((va) * (vb) + (vc)))
#define NX_C_FMA_NX_C_CAT_FLOAT(sfx, storage, compute, ld, st)                   \
  NX_C_FMAKN(sfx, storage, compute, ld, st)
#define NX_C_FMA_NX_C_CAT_COMPLEX(sfx, storage, compute, ld, st)
#define NX_C_FMA_NX_C_CAT_BOOL(sfx, storage, compute, ld, st)
#define NX_C_FMA_KROW(sfx, storage, compute, ld, st, cat)                       \
  NX_C_FMA_##cat(sfx, storage, compute, ld, st)
NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_FMA_KROW)
#define NX_C_CUROP fma
static const nx_c_map_table nx_c_fma_table = {
    .fn = {NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_TROW_INTF)}};
#undef NX_C_CUROP

/* ══════════════════════════════════════════════════════════════════════════
   cast — the pair matrix (compute src × compute dst) and the sub-byte casts
   ═════════════════════════════════════════════════════════════════════════ */

/* Local float->f16/bf16 converters for the contiguous cast fast path.

   nx_dtype.h's float_to_half / float_to_bfloat16 are correct but branchy: a
   call inside the store loop blocks auto-vectorization (the loop cannot become
   SIMD across a branchy body), so a contiguous f32->f16 cast ran at scalar
   speed. These are branchless/hardware forms that the vectorizer turns into
   packed converts. They are a DELIBERATE, TESTED copy of the nx_dtype.h
   converters: they MUST produce bit-identical results for EVERY input
   (rounding mode, NaN quieting, subnormals, overflow) — pinned by the cast sweep of
   packages/nx/test/dtype/test_float_codecs.ml, which checks Nx.cast from
   float32 against an exact reference over every bit pattern around each
   format's rounding bit. If a future edit here diverges, that test fails; do not
   "fix" it by loosening the test. nx_dtype.h stays the single owner of the storage format; this is only
   a vectorizable restatement used nowhere but the cast fast path. */

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
/* The portable converter's result (the #else branch) via the hardware
   narrowing convert (FCVTN, round-to-nearest-even) plus a branchless
   NaN-payload fixup: the hardware convert force-quiets NaNs by setting the
   mantissa MSB, whereas float_to_half preserves the raw payload, so only NaN
   lanes are recomputed to stay bit-identical. Finite/inf/subnormal all already
   match the canonical converter exactly. The whole loop stays SIMD. */
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
  X(A, f8e4m3, uint8_t, float, NX_C_CAT_FLOAT)                                  \
  X(A, f8e5m2, uint8_t, float, NX_C_CAT_FLOAT)                                  \
  X(A, i8, int8_t, int64_t, NX_C_CAT_SINT) X(A, u8, uint8_t, int64_t, NX_C_CAT_UINT) \
  X(A, i16, int16_t, int64_t, NX_C_CAT_SINT)                                    \
  X(A, u16, uint16_t, int64_t, NX_C_CAT_UINT)                                   \
  X(A, i32, int32_t, int64_t, NX_C_CAT_SINT)                                    \
  X(A, u32, uint32_t, uint64_t, NX_C_CAT_UINT)                                  \
  X(A, i64, int64_t, int64_t, NX_C_CAT_SINT)                                    \
  X(A, u64, uint64_t, uint64_t, NX_C_CAT_UINT)                                  \
  X(A, c32, nx_c_complex32, nx_c_complex32, NX_C_CAT_COMPLEX)                     \
  X(A, c64, nx_c_complex64, nx_c_complex64, NX_C_CAT_COMPLEX)                     \
  X(A, bool_, uint8_t, uint8_t, NX_C_CAT_BOOL)

/* The header's _Static_assert pins the dtype enum but not this local list; pin
   its length to the table's compute-row count so a dtype added to the table but
   forgotten here fails to compile rather than only at runtime on the one
   untested pair. (Counts, not identity — a wrong row still shows up as a NULL
   cast slot and a failing test.) */
#define NX_C_CAST_DST_CNT(A, dsfx, dstorage, dcompute, dcat) +1
#define NX_C_COMPUTE_CNT(sfx, storage, compute, ld, st, cat) +1
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
#define NX_C_CAST_KGEN_SRC(ssfx, sstorage, scompute, sld, sst, scat)            \
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
#define NX_C_CAST_SRCTBL(ssfx, sstorage, scompute, sld, sst, scat)              \
  [NX_C_DTYPE_##ssfx] = {.fn = {NX_C_CAST_DST_LIST(NX_C_CAST_TE, (ssfx))}},
static const nx_c_map_table nx_c_cast_tables[NX_C_DTYPE_COUNT] = {
    NX_C_FOR_EACH_COMPUTE_DTYPE(NX_C_CAST_SRCTBL)};

/* Sub-byte casts. An element of bit, int4 or uint4 is a value of its byte
   dtype (nx_c_packed_via), so a cast from one is the cast from that dtype,
   and a cast to one is the cast to that dtype followed by the narrowing that
   keeps the byte's low bits, or for bit whether it is non-zero. A float cast
   to int4 or uint4 is then held at the ends of the 4-bit range: 9. becomes 7,
   as the float cast to int8 holds 300. at 127. Blocks of 64 elements in C
   order go through 64 bytes on the stack: the cast table's kernels convert
   them, and the family's widening and narrowing move them to and from their
   elements. */

/* A compute operand read or written by its elements in C order, as
   nx_c_packed_src reads a packed one: its view with size-1 axes dropped and
   runs merged. */
typedef struct {
  char *base;
  int64_t esize;
  int ndim; /* >= 1 */
  int64_t offset;
  int64_t shape[NX_C_MAX_NDIM];
  int64_t strides[NX_C_MAX_NDIM];
} nx_c_cast_side;

static void nx_c_cast_side_init(nx_c_cast_side *c, const nx_c_ndarray *a,
                                int64_t esize) {
  c->base = (char *)a->data;
  c->esize = esize;
  c->offset = a->offset;
  int nd = 0;
  for (int d = 0; d < a->ndim; d++) {
    if (a->shape[d] == 1) continue;
    if (nd > 0 && c->strides[nd - 1] == a->strides[d] * a->shape[d]) {
      c->shape[nd - 1] *= a->shape[d];
      c->strides[nd - 1] = a->strides[d];
      continue;
    }
    c->shape[nd] = a->shape[d];
    c->strides[nd] = a->strides[d];
    nd++;
  }
  if (nd == 0) {
    c->shape[0] = 1;
    c->strides[0] = 1;
    nd = 1;
  }
  c->ndim = nd;
}

static bool nx_c_cast_side_dense(const nx_c_cast_side *c) {
  return c->ndim == 1 && c->strides[0] == 1;
}

/* Calls f on the runs of elements [e, e + k) of c in C order: the first
   element's address, the byte step and the run's length, and the run's place
   among the k. */
typedef void nx_c_cast_piece(const void *ctx, char *p, int64_t step,
                             int64_t n, int64_t j);

static void nx_c_cast_runs(const nx_c_cast_side *c, int64_t e, int64_t k,
                           nx_c_cast_piece *f, const void *ctx) {
  int last = c->ndim - 1;
  int64_t n_in = c->shape[last], s_in = c->strides[last];
  for (int64_t j = 0; j < k;) {
    int64_t col = (e + j) % n_in, r = (e + j) / n_in;
    int64_t pos = c->offset + col * s_in;
    for (int d = last - 1; d >= 0; d--) {
      pos += (r % c->shape[d]) * c->strides[d];
      r /= c->shape[d];
    }
    int64_t n = k - j;
    if (n_in - col < n) n = n_in - col;
    f(ctx, c->base + pos * c->esize, s_in * c->esize, n, j);
    j += n;
  }
}

/* Elements [e, e + k) of the packed s, of dtype dt, as bytes of its byte
   dtype. */
static void nx_c_cast_widen(const nx_c_packed_src *s, nx_c_dtype dt,
                            int64_t e, int k, uint8_t *bytes) {
  int per = 64 / s->bits;
  for (int j = 0; j < k; j += per) {
    int n = k - j < per ? k - j : per;
    nx_c_packed_widen(bytes + j, nx_c_packed_read(s, e + j, n), n, dt);
  }
}

/* A byte block converted by kernel f, or copied when f is NULL. */
static void nx_c_cast_block(nx_c_map_loop *f, char *dst, int64_t dstep,
                            const char *src, int64_t sstep, int64_t n) {
  if (f) {
    char *ptrs[2] = {dst, (char *)src};
    int64_t steps[2] = {dstep, sstep};
    f(ptrs, steps, n, NULL);
    return;
  }
  for (int64_t i = 0; i < n; i++) dst[i * dstep] = src[i * sstep];
}

/* To a sub-byte dtype. */

typedef struct {
  nx_c_dtype src, dst;
  bool packed;           /* src is sub-byte: read through pin */
  nx_c_cast_side in;     /* src otherwise */
  nx_c_packed_src pin;
  nx_c_map_loop *to_via; /* src, or its byte dtype, to dst's; NULL if equal */
  int clamp_lo, clamp_hi; /* a float source's range, else 0 and -1 */
} nx_c_cast_pack_ctx;

typedef struct {
  const nx_c_cast_pack_ctx *c;
  uint8_t *bytes;
} nx_c_cast_pack_piece_ctx;

static void nx_c_cast_pack_piece(const void *vctx, char *p, int64_t step,
                                 int64_t n, int64_t j) {
  const nx_c_cast_pack_piece_ctx *pc = vctx;
  nx_c_cast_block(pc->c->to_via, (char *)pc->bytes + j, 1, p, step, n);
}

static uint64_t nx_c_cast_pack_fill(const void *vctx, int64_t e, int k) {
  const nx_c_cast_pack_ctx *c = vctx;
  const nx_c_cast_side *in = &c->in;
  bool clamp = c->clamp_lo <= c->clamp_hi;
  if (!c->packed && !c->to_via && !clamp && nx_c_cast_side_dense(in))
    return nx_c_packed_narrow((const uint8_t *)in->base + in->offset + e, k,
                              c->dst);
  uint8_t bytes[64];
  if (c->packed) {
    uint8_t wide[64];
    nx_c_cast_widen(&c->pin, c->src, e, k, c->to_via ? wide : bytes);
    if (c->to_via)
      nx_c_cast_block(c->to_via, (char *)bytes, 1, (const char *)wide, 1, k);
  } else {
    nx_c_cast_pack_piece_ctx pc = {c, bytes};
    nx_c_cast_runs(in, e, k, nx_c_cast_pack_piece, &pc);
  }
  if (clamp)
    for (int j = 0; j < k; j++) {
      int v = c->dst == NX_C_DTYPE_i4 ? (int8_t)bytes[j] : bytes[j];
      v = v < c->clamp_lo ? c->clamp_lo : v > c->clamp_hi ? c->clamp_hi : v;
      bytes[j] = (uint8_t)v;
    }
  return nx_c_packed_narrow(bytes, k, c->dst);
}

static void nx_c_cast_pack_words(const void *vctx, int64_t e, int64_t n,
                                 uint8_t *dst) {
  const nx_c_cast_pack_ctx *c = vctx;
  const nx_c_cast_side *in = &c->in;
  if (!c->packed && !c->to_via && c->clamp_lo > c->clamp_hi &&
      nx_c_cast_side_dense(in)) {
    nx_c_packed_narrow_words(dst, (const uint8_t *)in->base + in->offset + e,
                             n, c->dst);
    return;
  }
  int per = 64 / nx_c_packed_bits(c->dst);
  for (int64_t j = 0; j < n; j++)
    nx_c_st64(dst + 8 * j, nx_c_cast_pack_fill(c, e + per * j, per));
}

/* From a sub-byte dtype to a compute one. */

typedef struct {
  nx_c_dtype src;
  nx_c_packed_src in;
  nx_c_cast_side out;
  nx_c_map_loop *from_via; /* src's byte dtype to dst; NULL if equal */
} nx_c_cast_unpack_ctx;

typedef struct {
  const nx_c_cast_unpack_ctx *c;
  const uint8_t *bytes;
} nx_c_cast_unpack_piece_ctx;

static void nx_c_cast_unpack_piece(const void *vctx, char *p, int64_t step,
                                   int64_t n, int64_t j) {
  const nx_c_cast_unpack_piece_ctx *pc = vctx;
  nx_c_cast_block(pc->c->from_via, p, step, (const char *)pc->bytes + j, 1,
                  n);
}

/* Blocks [lo, hi) of 64 elements of the destination. */
static void nx_c_cast_unpack_body(int64_t lo, int64_t hi, int worker,
                                  void *vctx) {
  (void)worker;
  const nx_c_cast_unpack_ctx *c = vctx;
  const nx_c_cast_side *out = &c->out;
  int64_t total = 1;
  for (int d = 0; d < out->ndim; d++) total *= out->shape[d];
  int64_t end = hi * 64 < total ? hi * 64 : total;
  if (!c->from_via && nx_c_cast_side_dense(out) && nx_c_packed_dense(&c->in)) {
    nx_c_packed_widen_run((uint8_t *)out->base + out->offset + lo * 64,
                          c->in.base, c->in.offset + lo * 64, end - lo * 64,
                          c->src);
    return;
  }
  bool direct = !c->from_via && nx_c_cast_side_dense(out);
  uint8_t bytes[64];
  for (int64_t blk = lo; blk < hi; blk++) {
    int64_t e = blk * 64;
    int k = total - e < 64 ? (int)(total - e) : 64;
    if (direct) {
      nx_c_cast_widen(&c->in, c->src, e, k,
                      (uint8_t *)out->base + out->offset + e);
      continue;
    }
    nx_c_cast_widen(&c->in, c->src, e, k, bytes);
    nx_c_cast_unpack_piece_ctx pc = {c, bytes};
    nx_c_cast_runs(out, e, k, nx_c_cast_unpack_piece, &pc);
  }
}

/* The cast kernel from a to b, NULL when they are one dtype. */
static nx_c_map_loop *nx_c_cast_kernel(nx_c_dtype a, nx_c_dtype b) {
  return a == b ? NULL : nx_c_cast_tables[a].fn[b];
}

static nx_c_status nx_c_cast_sub(nx_c_dtype src, nx_c_dtype dst,
                                 const nx_c_ndarray *o,
                                 const nx_c_ndarray *in) {
  if (o->ndim != in->ndim) return NX_C_ERR_RANK_MISMATCH;
  int64_t total = 1;
  for (int d = 0; d < o->ndim; d++) {
    if (o->shape[d] != in->shape[d]) return NX_C_ERR_SHAPE;
    total *= o->shape[d];
  }
  bool sp = nx_c_dtype_is_packed(src);
  nx_c_dtype from = sp ? nx_c_packed_via(src) : src;
  if (nx_c_dtype_is_packed(dst)) {
    nx_c_cast_pack_ctx c;
    c.src = src;
    c.dst = dst;
    c.packed = sp;
    if (sp)
      nx_c_packed_src_init(&c.pin, in, nx_c_packed_bits(src));
    else
      nx_c_cast_side_init(&c.in, in, nx_c_elem_size(src));
    c.to_via = nx_c_cast_kernel(from, nx_c_packed_via(dst));
    c.clamp_lo = 0;
    c.clamp_hi = -1;
    if (nx_c_dtype_is_float(src) || nx_c_dtype_is_complex(src)) {
      if (dst == NX_C_DTYPE_i4) c.clamp_lo = -8, c.clamp_hi = 7;
      if (dst == NX_C_DTYPE_u4) c.clamp_hi = 15;
    }
    int64_t bytes = nx_c_dtype_bytes(src, total) + nx_c_dtype_bytes(dst, total);
    nx_c_packed_filler f = {nx_c_cast_pack_fill, nx_c_cast_pack_words, &c};
    return nx_c_packed_write(o, nx_c_packed_bits(dst), &f, bytes);
  }
  if (total == 0) return NX_C_OK;
  for (int d = 0; d < o->ndim; d++)
    if (o->strides[d] == 0 && o->shape[d] > 1) return NX_C_ERR_OUT_ALIASED;
  nx_c_cast_unpack_ctx c;
  c.src = src;
  nx_c_packed_src_init(&c.in, in, nx_c_packed_bits(src));
  nx_c_cast_side_init(&c.out, o, nx_c_elem_size(dst));
  c.from_via = nx_c_cast_kernel(from, dst);
  int64_t blocks = (total + 63) / 64;
  int64_t bytes = nx_c_dtype_bytes(src, total) + nx_c_dtype_bytes(dst, total);
  int nth = nx_c_threads_for(NX_C_COST_BANDWIDTH, total, 1, bytes);
  if (nth > blocks) nth = (int)blocks;
  nx_c_parallel_for(nth, blocks, bytes, nx_c_cast_unpack_body, &c, NULL);
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
NX_C_MAP1_STUB(log1p, "log1p", nx_c_log1p_table, NX_C_COST_COMPUTE)
NX_C_MAP1_STUB(expm1, "expm1", nx_c_expm1_table, NX_C_COST_COMPUTE)

/* Binary. */
NX_C_MAP2_STUB(add, "add", nx_c_add_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(sub, "sub", nx_c_sub_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(mul, "mul", nx_c_mul_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(idiv, "idiv", nx_c_idiv_table, NX_C_COST_COMPUTE)
NX_C_MAP2_STUB(fdiv, "fdiv", nx_c_fdiv_table, NX_C_COST_COMPUTE)
NX_C_MAP2_STUB(mod, "mod", nx_c_mod_table, NX_C_COST_COMPUTE)
NX_C_MAP2_STUB(pow, "pow", nx_c_pow_table, NX_C_COST_COMPUTE)
NX_C_MAP2_STUB(atan2, "atan2", nx_c_atan2_table, NX_C_COST_COMPUTE)

/* The logical operations take bit operands too, word by word: on bit, max is
   or and min is and, as on bool. */
static void nx_c_logic_run(const char *op, const nx_c_map_table *tbl,
                           nx_c_bit_op bop, value vout, value va, value vb) {
  if (nx_c_dtype_of_value(vout) != NX_C_DTYPE_bit) {
    value vals[3] = {vout, va, vb};
    nx_c_map_funnel(op, tbl, NX_C_COST_BANDWIDTH, 2, vals, NULL);
    return;
  }
  nx_c_ndarray ops[3];
  nx_c_status s;
  if ((s = nx_c_ndarray_of_value(vout, &ops[0])) != NX_C_OK) nx_c_raise(op, s);
  if ((s = nx_c_ndarray_of_value(va, &ops[1])) != NX_C_OK) nx_c_raise(op, s);
  if ((s = nx_c_ndarray_of_value(vb, &ops[2])) != NX_C_OK) nx_c_raise(op, s);
  s = nx_c_bit_logic(bop, &ops[0], &ops[1], &ops[2]);
  if (s != NX_C_OK) nx_c_raise_status(op, s);
}

#define NX_C_LOGIC_STUB(cname, opname, table, bop)                              \
  CAMLprim value caml_nx_c_##cname(value vout, value va, value vb) {            \
    CAMLparam3(vout, va, vb);                                                  \
    nx_c_logic_run((opname), &(table), (bop), vout, va, vb);                   \
    CAMLreturn(Val_unit);                                                      \
  }
NX_C_LOGIC_STUB(max, "max", nx_c_max_table, NX_C_BIT_OR)
NX_C_LOGIC_STUB(min, "min", nx_c_min_table, NX_C_BIT_AND)
NX_C_LOGIC_STUB(xor, "xor", nx_c_xor_table, NX_C_BIT_XOR)
NX_C_LOGIC_STUB(or, "or", nx_c_or_table, NX_C_BIT_OR)
NX_C_LOGIC_STUB(and, "and", nx_c_and_table, NX_C_BIT_AND)
NX_C_MAP2_STUB(shl, "shl", nx_c_shl_table, NX_C_COST_BANDWIDTH)
NX_C_MAP2_STUB(shr, "shr", nx_c_shr_table, NX_C_COST_BANDWIDTH)

/* where and fma. */
NX_C_MAP3_STUB(where, "where", nx_c_where_table, NX_C_COST_BANDWIDTH)
NX_C_MAP3_STUB(fma, "fma", nx_c_fma_table, NX_C_COST_BANDWIDTH)

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
   the map driver (dispatched on the dst dtype); a pair with a sub-byte dtype
   goes through its byte dtype. */
CAMLprim value caml_nx_c_cast(value vout, value va) {
  CAMLparam2(vout, va);
  nx_c_ndarray ops[2];
  nx_c_status s;
  if ((s = nx_c_ndarray_of_value(vout, &ops[0])) != NX_C_OK) nx_c_raise("cast", s);
  if ((s = nx_c_ndarray_of_value(va, &ops[1])) != NX_C_OK) nx_c_raise("cast", s);
  nx_c_dtype dst = nx_c_dtype_of_value(vout);
  nx_c_dtype src = nx_c_dtype_of_value(va);
  if (nx_c_dtype_is_packed(src) || nx_c_dtype_is_packed(dst)) {
    s = nx_c_cast_sub(src, dst, &ops[0], &ops[1]);
  } else {
    int64_t elem[2] = {nx_c_elem_size(dst), nx_c_elem_size(src)};
    s = nx_c_map_run(&nx_c_cast_tables[src], dst, 1, ops, elem, NX_C_COST_BANDWIDTH,
                    NULL);
  }
  if (s != NX_C_OK) nx_c_raise_status("cast", s);
  CAMLreturn(Val_unit);
}

