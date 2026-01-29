(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)
  include Ocaml_simd_sse.Int32x4
  module Raw = Load_store.Vec128.Raw_Int32x4
  
  external low_of
    :  int32#
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_int32x4_low_of_int32"
  [@@noalloc] [@@builtin]
  
  external broadcast
    :  t
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
    -> idx:t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx_vec128_permutev_32"
  [@@noalloc] [@@builtin]
  
  external shift_left_logical_by
    :  t
    -> shift:t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx2_int32x4_sllv"
  [@@noalloc] [@@builtin]
  
  external shift_right_logical_by
    :  t
    -> shift:t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx2_int32x4_srlv"
  [@@noalloc] [@@builtin]
  
  external shift_right_arithmetic_by
    :  t
    -> shift:t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx2_int32x4_srav"
  [@@noalloc] [@@builtin]
  
  let[@inline] set1 x = broadcast (low_of x)
  let[@inline] of_float64x4 x = Float64x4_internal.cvt_i32 x
  let[@inline] of_float64x4_trunc x = Float64x4_internal.cvtt_i32 x
  let[@inline] of_int32x8 x = Int32x8_internal.low_to_i32x4 x