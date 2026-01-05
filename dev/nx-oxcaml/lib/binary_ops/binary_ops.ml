module Dtype = Nx_core.Dtype
module View = Nx_core.View
module Shape = Nx_core.Shape
module Symbolic_shape = Nx_core.Symbolic_shape
module Error = Nx_core.Error
module Parallel = Parallel_pool
(* open Stdlib_upstream_compatible *)
(* *)
module Float_u = Stdlib_upstream_compatible.Float_u
module Float32_u = Stdlib_stable.Float32_u
module Int32_u = Stdlib_upstream_compatible.Int32_u
module Int64_u = Stdlib_upstream_compatible.Int64_u

(* *)
type context = { pool : Parallel.pool }

let create_function_lookup _dtype _op_name = failwith "Not implemented"

let shape (v : View.t) : int array =
  match Symbolic_shape.eval (View.shape v) with
  | Some arr -> arr
  | None -> Error.failed ~op:"shape" ~what:"symbolic shape not evaluable" ()

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
end

