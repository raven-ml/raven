type t = int64x2#

external low_of
  :  int64#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_int64x2_low_of_int64"
[@@noalloc] [@@builtin]

external low_to
  :  t
  -> int64#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_int64x2_low_to_int64"
[@@noalloc] [@@builtin]

external insert
  :  idx:int64#
  -> t
  -> int64#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int64x2_insert"
[@@noalloc] [@@builtin]

external extract
  :  idx:int64#
  -> t
  -> int64#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int64x2_extract"
[@@noalloc] [@@builtin]

external of_int8x16
  :  int8x16#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external of_int16x8
  :  int16x8#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external of_int32x4
  :  int32x4#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external of_float16x8
  :  float16x8#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external of_float32x4
  :  float32x4#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external of_float64x2
  :  float64x2#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_vec128_cast"
[@@noalloc] [@@builtin]

external const1
  :  int64#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_int64x2_const1"
[@@noalloc] [@@builtin]

external add
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int64x2_add"
[@@noalloc] [@@builtin]

external sub
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int64x2_sub"
[@@noalloc] [@@builtin]

external cmpeq
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_int64x2_cmpeq"
[@@noalloc] [@@builtin]

external cmpgt
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse42_int64x2_cmpgt"
[@@noalloc] [@@builtin]

external sll
  :  t
  -> int64x2#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int64x2_sll"
[@@noalloc] [@@builtin]

external srl
  :  t
  -> int64x2#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int64x2_srl"
[@@noalloc] [@@builtin]

external and_
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse_vec128_and"
[@@noalloc] [@@builtin]

external andnot
  :  not:t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse_vec128_andnot"
[@@noalloc] [@@builtin]

external or_ : t -> t -> t @@ portable = "ocaml_simd_sse_unreachable" "caml_sse_vec128_or"
[@@noalloc] [@@builtin]

external xor
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse_vec128_xor"
[@@noalloc] [@@builtin]

external slli
  :  int64#
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int64x2_slli"
[@@noalloc] [@@builtin]

external srli
  :  int64#
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_int64x2_srli"
[@@noalloc] [@@builtin]

external high_64_to_low_64
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse_vec128_high_64_to_low_64"
[@@noalloc] [@@builtin]

external low_64_to_high_64
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse_vec128_low_64_to_high_64"
[@@noalloc] [@@builtin]

external interleave_high_64
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_vec128_interleave_high_64"
[@@noalloc] [@@builtin]

external interleave_low_64
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_vec128_interleave_low_64"
[@@noalloc] [@@builtin]

external blendv_64
  :  t
  -> t
  -> int64x2#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_vec128_blendv_64"
[@@noalloc] [@@builtin]

external dup_low_64
  :  t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse3_vec128_dup_low_64"
[@@noalloc] [@@builtin]

external movemask_64
  :  t
  -> int64#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_vec128_movemask_64"
[@@noalloc] [@@builtin]
