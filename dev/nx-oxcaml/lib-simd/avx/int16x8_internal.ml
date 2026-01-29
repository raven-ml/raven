(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)
  include Ocaml_simd_sse.Int16x8

  external low_of
    :  int16#
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_int16x8_low_of_int16"
  [@@noalloc] [@@builtin]
  
  external broadcast
    :  t
    -> t
    @@ portable
    = "ocaml_simd_avx_unreachable" "caml_avx2_vec128_broadcast_16"
  [@@noalloc] [@@builtin]
  
  let[@inline] set1 x = broadcast (low_of x)
  let[@inline] of_int16x16 x = Int16x16_internal.low_to_i16x8 x