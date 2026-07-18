/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* nx_c_matmul.h — the one GEMM internal the numerics families (linalg) build on.

   nx_c_linalg.c's blocked factorizations are BLAS-3-bound: their trailing updates
   are GEMMs. Rather than duplicate a kernel, they call nx_c_gemm2d_ct — the owned
   GEMM's blocked/microkernel path, specialized to one 2-D matrix in a compute
   dtype. Everything else in nx_c_matmul.c stays private. */

#ifndef NX_C_MATMUL_H
#define NX_C_MATMUL_H

#include "nx_c.h"

/* C = A·B for a single 2-D matrix, all operands COMPUTE-typed (dt one of
   NX_C_DTYPE_{f32,f64,c32,c64}) with arbitrary element strides — transpose and
   conjugate-free-transpose are the caller's job via strides on materialized
   panels. m×k times k×n into m×n. Serial (no pool); the caller owns any batch
   parallelism. Returns NX_C_ERR_ALLOC on scratch failure, NX_C_ERR_UNSUPPORTED_DTYPE
   for a non-compute dt, else NX_C_OK. Strides are in ELEMENTS of the compute type;
   pointers are byte bases at the matrix's first element.

   DEVIATION (adjudicated): the blocked path here mallocs its packing panels per
   call. When a linalg factorization calls this from INSIDE a pooled worker body
   (cholesky's trailing update under the released lock), that technically bends
   nx_c_engine.h's "bodies never allocate" rule. It is safe — the alloc/free is
   self-contained within the call, thread-safe (each worker's call is
   independent), and a failure surfaces as NX_C_ERR_ALLOC that the driver reports
   as a status — so the rule's PURPOSES (no leak on any path, no persistent
   hot-loop allocation) hold even though the letter does not. The fix, a
   caller-provided-workspace variant (nullable Ap/Bp + a scratch-bytes helper so
   the driver's per-worker block absorbs the pack panels), lands WITH the first
   additional in-body caller — the blocked tridiagonalization — alongside blocked
   TRSM/QR; it is deliberately not added speculatively to this just-reviewed
   surface. Callers on the frontend path (matmul, non-pooled) are unaffected. */
nx_c_status nx_c_gemm2d_ct(nx_c_dtype dt, int64_t m, int64_t n, int64_t k,
                         const char *A, int64_t a_rs, int64_t a_cs,
                         const char *B, int64_t b_rs, int64_t b_cs, char *C,
                         int64_t c_rs, int64_t c_cs);

/* Caller-workspace variant of nx_c_gemm2d_ct — the fix the DEVIATION note above
   promised, landing with the blocked linalg. Identical result, but the packing
   panels (and the KC accumulator when the contraction sub-blocks) come from
   caller-provided `scratch` rather than mm_alloc, so a factorization calling
   GEMM from inside a pooled worker body allocates nothing. nx_c_gemm2d_ct_scratch
   returns the byte size a driver must reserve per worker (64-byte aligned); it is
   0 when the direct path is taken, and then `scratch` may be NULL. nx_c_gemm2d_ct
   is unchanged and remains the frontend/non-pooled entry. */
int64_t nx_c_gemm2d_ct_scratch(nx_c_dtype dt, int64_t m, int64_t n, int64_t k);
nx_c_status nx_c_gemm2d_ct_ws(nx_c_dtype dt, int64_t m, int64_t n, int64_t k,
                            const char *A, int64_t a_rs, int64_t a_cs,
                            const char *B, int64_t b_rs, int64_t b_cs, char *C,
                            int64_t c_rs, int64_t c_cs, char *scratch);

/* Internal maintenance hooks. Only the backend-local test and benchmark stubs
   bind these; they are not part of the installed OCaml API. */
void nx_c_matmul_maintenance(value vout, value va, value vb, int mode);
int nx_c_matmul_accelerate_available(void);
int nx_c_matmul_accelerate_enabled(void);
void nx_c_matmul_accelerate_override(int mode);

#endif /* NX_C_MATMUL_H */
