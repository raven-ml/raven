(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Nx_core.Dtype
module View = Nx_core.View
module Shape = Nx_core.Shape
module Array1 = Bigarray.Array1
module Parallel = Parallel
module Float_u = Stdlib_upstream_compatible.Float_u
module Float32_u = Stdlib_stable.Float32_u
module Int32_u = Stdlib_upstream_compatible.Int32_u
module Int64_u = Stdlib_upstream_compatible.Int64_u
module Int8_u = Stdlib_stable.Int8_u
module Int16_u = Stdlib_stable.Int16_u
module Float32x4 = Simd.Float32x4
module Float64x2 = Simd.Float64x2
module Int32x4 = Simd.Int32x4
module Int64x2 = Simd.Int64x2
module Array = struct
  include Stdlib.Array

  external get : ('a : any mod non_null separable). 'a array -> int -> 'a
    = "%array_safe_get"
  [@@layout_poly]

  external set :
    ('a : any mod non_null separable). 'a array -> int -> 'a -> unit
    = "%array_safe_set"
  [@@layout_poly]

  external unsafe_get : ('a : any mod non_null separable). 'a array -> int -> 'a
    = "%array_unsafe_get"
  [@@layout_poly]

  external unsafe_set :
    ('a : any mod non_null separable). 'a array -> int -> 'a -> unit
    = "%array_unsafe_set"
  [@@layout_poly]

  external length : ('a : any mod non_null separable). 'a array -> int
    = "%array_length"
  [@@layout_poly]

  external make_float64 : int -> float# array = "caml_make_unboxed_float64_vect"

  external make_float32 : int -> float32# array
    = "caml_make_unboxed_float32_vect"

  external make_int32 : int -> int32# array = "caml_make_unboxed_int32_vect"
  external make_int64 : int -> int64# array = "caml_make_unboxed_int64_vect"
  external make_int8 : int -> int8# array = "caml_make_untagged_int8_vect"
  external make_int16 : int -> int16# array
    = "caml_make_untagged_int16_vect"

  external ba_to_unboxed_float_array
  : (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> float# array
  = "caml_ba_to_unboxed_float64_array"

  external ba_to_unboxed_float32_array
  : (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> float32# array
  = "caml_ba_to_unboxed_float32_array"

  external ba_to_unboxed_int64_array
  : (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> int64# array
  = "caml_ba_to_unboxed_int64_array"

  external ba_to_unboxed_int32_array
  : (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> int32# array
  = "caml_ba_to_unboxed_int32_array"

  external ba_to_unboxed_int8_array
  : (int, Bigarray.int8_signed_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> int8# array
  = "caml_ba_to_unboxed_int8_array"

  external ba_to_unboxed_int16_array
  : (int, Bigarray.int16_signed_elt, Bigarray.c_layout) Bigarray.Array1.t
  -> int16# array
  = "caml_ba_to_unboxed_int16_array"

  external unboxed_float64_to_ba
  : float# array -> int
  -> (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_unboxed_float64_array_to_ba"

  external unboxed_float32_to_ba
  : float32# array -> int
  -> (float, Bigarray.float32_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_unboxed_float32_array_to_ba"

  external unboxed_int64_to_ba
  : int64# array -> int
  -> (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_unboxed_int64_array_to_ba"

  external unboxed_int32_to_ba
  : int32# array -> int
  -> (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_unboxed_int32_array_to_ba"

  external unboxed_int8_to_ba
  : int8# array -> int
  -> (int, Bigarray.int8_signed_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_unboxed_int8_array_to_ba"

  external unboxed_int16_to_ba
  : int16# array -> int
  -> (int, Bigarray.int16_signed_elt, Bigarray.c_layout) Bigarray.Array1.t
  = "caml_unboxed_int16_array_to_ba"
end

let shape (v : View.t) : int array = View.shape v
let numel (v : View.t) : int = View.numel v
