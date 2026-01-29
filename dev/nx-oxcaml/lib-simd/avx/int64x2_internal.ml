(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)
  include Ocaml_simd_sse.Int64x2
  module Raw = Load_store.Vec128.Raw_Int64x2
  
  external permute
    :  (Ocaml_simd.Permute2.t[@untagged])
    -> t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx_vec128_permute_64"
  [@@noalloc] [@@builtin]
  
  external permute_by
    :  t
    -> idx:t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx_vec128_permutev_64"
  [@@noalloc] [@@builtin]
  
  external shift_left_logical_by
    :  t
    -> shift:t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx2_int64x2_sllv"
  [@@noalloc] [@@builtin]
  
  external shift_right_logical_by
    :  t
    -> shift:t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx2_int64x2_srlv"
  [@@noalloc] [@@builtin]
  
  let[@inline] of_int64x4 x = Int64x4_internal.low_to_i64x2 x