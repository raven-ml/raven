/*---------------------------------------------------------------------------
   Copyright (c) 2026 The Raven authors. All rights reserved.
   SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*/

/* Test-only driver for nx_c_gemm2d_ct_ws (nx_c_matmul.h) — linalg's caller-
   workspace GEMM, which has no frontend FFI path. Extracts a single 2-D matmul
   from the maintenance-test FFI records (buffer/shape/strides/offset), sizes the workspace
   with nx_c_gemm2d_ct_scratch, allocates it here (the real linalg driver sub-slots
   one per pool worker), and runs the GEMM. The point is to exercise the
   k > MM_KC_FULLK_MAX Cacc-alloc branch of _scratch/_ws that no other test
   reaches; the caller cross-checks the result against the direct oracle, so a
   mis-sized/mis-offset scratch surfaces as a wrong answer, not a silent pass.
   nx_c.h (via nx_c_matmul.h) supplies nx_c_ndarray_of_value / nx_c_dtype_of_value /
   nx_c_elem_size; the _ws/_scratch symbols come from the nx_c dep. */

#include <stdlib.h>

#include <caml/memory.h>
#include <caml/mlvalues.h>

#include "nx_c_matmul.h"

CAMLprim value caml_nx_c_matmul_ex(value vout, value va, value vb,
                                    value vmode) {
  CAMLparam4(vout, va, vb, vmode);
  nx_c_matmul_maintenance(vout, va, vb, Int_val(vmode));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_c_matmul_accel_available(value unit) {
  CAMLparam1(unit);
  CAMLreturn(Val_int(nx_c_matmul_accelerate_available()));
}

CAMLprim value caml_nx_c_matmul_accel_enabled(value unit) {
  CAMLparam1(unit);
  CAMLreturn(Val_int(nx_c_matmul_accelerate_enabled()));
}

CAMLprim value caml_nx_c_matmul_accel_set_override(value vmode) {
  CAMLparam1(vmode);
  nx_c_matmul_accelerate_override(Int_val(vmode));
  CAMLreturn(Val_unit);
}

CAMLprim value caml_nx_c_gemm2d_ct_ws_test(value vc, value va, value vb) {
  CAMLparam3(vc, va, vb);
  nx_c_ndarray A, B, C;
  nx_c_ndarray_of_value(va, &A);
  nx_c_ndarray_of_value(vb, &B);
  nx_c_ndarray_of_value(vc, &C);
  nx_c_dtype dt = nx_c_dtype_of_value(va);
  int64_t esz = nx_c_elem_size(dt);
  int64_t m = A.shape[0], k = A.shape[1], n = B.shape[1];
  const char *a = (const char *)A.data + A.offset * esz;
  const char *b = (const char *)B.data + B.offset * esz;
  char *c = (char *)C.data + C.offset * esz;
  int64_t sz = nx_c_gemm2d_ct_scratch(dt, m, n, k);
  /* _scratch returns a multiple of 64 (sum of 64-aligned slots), the exact size
     aligned_alloc wants; NULL is the contract for a 0 query (direct path). */
  char *scratch = sz > 0 ? aligned_alloc(64, (size_t)sz) : NULL;
  nx_c_gemm2d_ct_ws(dt, m, n, k, a, A.strides[0], A.strides[1], b, B.strides[0],
                   B.strides[1], c, C.strides[0], C.strides[1], scratch);
  free(scratch);
  CAMLreturn(Val_unit);
}
