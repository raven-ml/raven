(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

include Ocaml_simd_sse.Float64x2
module Raw = Load_store.Vec128.Raw_Float64x2

external permute
  :  (Ocaml_simd.Permute2.t[@untagged])
  -> t
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_avx_vec128_permute_64"
[@@noalloc] [@@builtin]

external permute_by
  :  t
  -> idx:int64x2#
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_avx_vec128_permutev_64"
[@@noalloc] [@@builtin]

external mul_add
  :  t
  -> t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_fma_float64x2_mul_add"
[@@noalloc] [@@builtin]

external mul_sub
  :  t
  -> t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_fma_float64x2_mul_sub"
[@@noalloc] [@@builtin]

external mul_add_sub
  :  t
  -> t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_fma_float64x2_mul_addsub"
[@@noalloc] [@@builtin]

external mul_sub_add
  :  t
  -> t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_fma_float64x2_mul_subadd"
[@@noalloc] [@@builtin]

external neg_mul_add
  :  t
  -> t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_fma_float64x2_neg_mul_add"
[@@noalloc] [@@builtin]

external neg_mul_sub
  :  t
  -> t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_fma_float64x2_neg_mul_sub"
[@@noalloc] [@@builtin]

let[@inline] of_float64x4 x = Float64x4_internal.low_to_f64x2 x