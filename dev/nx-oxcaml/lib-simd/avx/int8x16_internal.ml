(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  Distributed under the ISC license, see terms at the end of the file.

  Based on ocaml_simd (https://github.com/janestreet/ocaml_simd)
  Copyright (c) 2025-2026 Jane Street Group, LLC
  Released under the MIT license.
  ---------------------------------------------------------------------------*)

include Ocaml_simd_sse.Int8x16

external low_of
  :  int8#
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_int8x16_low_of_int8"
[@@noalloc] [@@builtin]

external broadcast
  :  t
  -> t
  @@ portable
  = "ocaml_simd_avx_unreachable" "caml_avx2_vec128_broadcast_8"
[@@noalloc] [@@builtin]

let[@inline] set1 x = broadcast (low_of x)
let[@inline] of_int8x32 x = Int8x32_internal.low_to_i8x16 x