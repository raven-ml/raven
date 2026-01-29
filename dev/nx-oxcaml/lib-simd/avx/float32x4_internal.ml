(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

  include Ocaml_simd_sse.Float32x4
  module Raw = Load_store.Vec128.Raw_Float32x4
  
  external set1
    :  float32#
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx_vec128_broadcast_32"
  [@@noalloc] [@@builtin]
  
  external permute
    :  (Ocaml_simd.Permute4.t[@untagged])
    -> t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx_vec128_permute_32"
  [@@noalloc] [@@builtin]
  
  external permute_by
    :  t
    -> idx:int32x4#
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx_vec128_permutev_32"
  [@@noalloc] [@@builtin]
  
  external mul_add
    :  t
    -> t
    -> t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_fma_float32x4_mul_add"
  [@@noalloc] [@@builtin]
  
  external mul_sub
    :  t
    -> t
    -> t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_fma_float32x4_mul_sub"
  [@@noalloc] [@@builtin]
  
  external mul_add_sub
    :  t
    -> t
    -> t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_fma_float32x4_mul_addsub"
  [@@noalloc] [@@builtin]
  
  external mul_sub_add
    :  t
    -> t
    -> t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_fma_float32x4_mul_subadd"
  [@@noalloc] [@@builtin]
  
  external neg_mul_add
    :  t
    -> t
    -> t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_fma_float32x4_neg_mul_add"
  [@@noalloc] [@@builtin]
  
  external neg_mul_sub
    :  t
    -> t
    -> t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_fma_float32x4_neg_mul_sub"
  [@@noalloc] [@@builtin]
  
  external of_float16x8
    :  float16x8#
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_f16c_cvt_float16x8_float32x4"
  [@@noalloc] [@@builtin]
  
  let[@inline] of_float64x4 x = Float64x4_internal.cvt_f32 x
  let[@inline] of_float32x8 x = Float32x8_internal.low_to_f32x4 x