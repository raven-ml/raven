type t = float64x2#

external low_of
  :  float#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_float64x2_low_of_float"
[@@noalloc] [@@builtin]

external low_to
  :  t
  -> float#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_float64x2_low_to_float"
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

external of_int64x2
  :  int64x2#
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

external cmp
  :  (Ocaml_simd.Float.Comparison.t[@untagged])
  -> t
  -> t
  -> int64x2#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_float64x2_cmp"
[@@noalloc] [@@builtin]

external add
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_float64x2_add"
[@@noalloc] [@@builtin]

external sub
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_float64x2_sub"
[@@noalloc] [@@builtin]

external mul
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_float64x2_mul"
[@@noalloc] [@@builtin]

external div
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_float64x2_div"
[@@noalloc] [@@builtin]

external max
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_float64x2_max"
[@@noalloc] [@@builtin]

external min
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_float64x2_min"
[@@noalloc] [@@builtin]

external sqrt
  :  t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_float64x2_sqrt"
[@@noalloc] [@@builtin]

external addsub
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse3_float64x2_addsub"
[@@noalloc] [@@builtin]

external horizontal_add
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse3_float64x2_hadd"
[@@noalloc] [@@builtin]

external horizontal_sub
  :  t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse3_float64x2_hsub"
[@@noalloc] [@@builtin]

external dp
  :  int64#
  -> t
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_float64x2_dp"
[@@noalloc] [@@builtin]

external round
  :  (Ocaml_simd.Float.Rounding.t[@untagged])
  -> t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_float64x2_round"
[@@noalloc] [@@builtin]

external blendv_64
  :  t
  -> t
  -> int64x2#
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse41_vec128_blendv_64"
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

external dup_low_64
  :  t
  -> t
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse3_vec128_dup_low_64"
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

external cvt_f32
  :  t
  -> float32x4#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_cvt_float64x2_float32x2"
[@@noalloc] [@@builtin]

external cvt_i32
  :  t
  -> int32x4#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_cvt_float64x2_int32x2"
[@@noalloc] [@@builtin]

external cvtt_i32
  :  t
  -> int32x4#
  @@ portable
  = "ocaml_simd_sse_unreachable" "caml_sse2_cvtt_float64x2_int32x2"
[@@noalloc] [@@builtin]
