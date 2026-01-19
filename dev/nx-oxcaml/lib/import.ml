(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Dtype = Nx_core.Dtype
module View = Nx_core.View
module Shape = Nx_core.Shape
module Symbolic_shape = Nx_core.Symbolic_shape
module Error = Nx_core.Error
module Parallel = Parallel
module Float_u = Stdlib_upstream_compatible.Float_u
module Float32_u = Stdlib_stable.Float32_u
module Int32_u = Stdlib_upstream_compatible.Int32_u
module Int64_u = Stdlib_upstream_compatible.Int64_u
module Int8_u = Stdlib_stable.Int8_u
module Int16_u = Stdlib_stable.Int16_u

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
end

let shape (v : View.t) : int array =
  match Symbolic_shape.eval (View.shape v) with
  | Some arr -> arr
  | None -> Error.failed ~op:"shape" ~what:"symbolic shape not evaluable" ()

let numel (v : View.t) : int =
  match Symbolic_shape.eval_dim (View.numel v) with
  | Some n -> n
  | None -> Error.failed ~op:"numel" ~what:"symbolic numel not evaluable" ()
