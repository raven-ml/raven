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
end

let shape (v : View.t) : int array =
  match Symbolic_shape.eval (View.shape v) with
  | Some arr -> arr
  | None -> Error.failed ~op:"shape" ~what:"symbolic shape not evaluable" ()

let numel (v : View.t) : int =
  match Symbolic_shape.eval_dim (View.numel v) with
  | Some n -> n
  | None -> Error.failed ~op:"numel" ~what:"symbolic numel not evaluable" ()

let view_dim (dim : int) (v : View.t) : int =
  match Symbolic_shape.eval_dim (View.dim dim v) with
  | Some n -> n
  | None -> Error.failed ~op:"numel" ~what:"symbolic numel not evaluable" ()

type context = { pool : Parallel.pool }

let create_context () = { pool = Parallel.get_or_setup_pool () }

type 'b buffer =
  | Float64 : float# array -> Dtype.float64_elt buffer
  | Float32 : float32# array -> Dtype.float32_elt buffer
  | Int32 : int32# array -> Dtype.int32_elt buffer
  | Int64 : int64# array -> Dtype.int64_elt buffer

type ('a, 'b) t = {
  dtype : ('a, 'b) Dtype.t;
  buffer : 'b buffer;
  view : View.t;
  context : context;
}

let view t = t.view
let dtype t = t.dtype
let context t = t.context

let blit : type a b. (a, b) t -> (a, b) t -> unit =
 fun src dst ->
  let src_view = view src in
  let dst_view = view dst in

  if View.ndim src_view <> View.ndim dst_view then
    invalid_arg "blit: tensors must have the same number of dimensions";
  if not (Shape.equal (shape src_view) (shape dst_view)) then
    invalid_arg "blit: tensors must have the same shape";

  let total_elements = numel src_view in
  if total_elements = 0 then () (* Nothing to blit *)
  else
    let src_buffer = src.buffer in
    let dst_buffer = dst.buffer in
    let n_dims = View.ndim src_view in

    (* TODO: Handle overlapping bigarrays correctly. Currently, when src and dst
       are views of the same underlying buffer with overlapping regions, the
       copy may produce incorrect results as source data can be overwritten
       before being read.

       Consider using https://github.com/dinosaure/overlap which provides a
       library for checking if bigarrays overlap. If overlap is detected, we
       should either: 1. Make a copy of the source data first 2. Copy in the
       appropriate order (backward if dst > src) 3. Use memmove-like semantics

       See test_blit_overlapping_views for expected behavior. *)
    (* Scalar case *)
    if n_dims = 0 then
      match (src_buffer, dst_buffer) with
      | Float64 src, Float64 dst ->
          Array.unsafe_set dst (View.offset dst_view)
            (Array.unsafe_get src (View.offset src_view))
      | Float32 src, Float32 dst ->
          Array.unsafe_set dst (View.offset dst_view)
            (Array.unsafe_get src (View.offset src_view))
      | Int32 src, Int32 dst ->
          Array.unsafe_set dst (View.offset dst_view)
            (Array.unsafe_get src (View.offset src_view))
      | Int64 src, Int64 dst ->
          Array.unsafe_set dst (View.offset dst_view)
            (Array.unsafe_get src (View.offset src_view))
    else
      (* Iterate through logical elements based on common shape *)
      let current_md_idx = Array.make n_dims 0 in
      let rec blit_slice dim =
        if dim = n_dims then (
          let src_physical_offset =
            View.offset src_view
            + Shape.ravel_index current_md_idx (View.strides src_view)
          in
          let dst_physical_offset =
            View.offset dst_view
            + Shape.ravel_index current_md_idx (View.strides dst_view)
          in
          (* Debug output *)
          if false then
            Printf.printf "Copying from src[%d] to dst[%d]\n"
              src_physical_offset dst_physical_offset;
          (* dst_buffer.{dst_physical_offset} <-
             src_buffer.{src_physical_offset}) *)
          match (src_buffer, dst_buffer) with
          | Float64 src, Float64 dst ->
              Array.unsafe_set dst dst_physical_offset
                (Array.unsafe_get src src_physical_offset)
          | Float32 src, Float32 dst ->
              Array.unsafe_set dst dst_physical_offset
                (Array.unsafe_get src src_physical_offset)
          | Int32 src, Int32 dst ->
              Array.unsafe_set dst dst_physical_offset
                (Array.unsafe_get src src_physical_offset)
          | Int64 src, Int64 dst ->
              Array.unsafe_set dst dst_physical_offset
                (Array.unsafe_get src src_physical_offset))
        else
          for i = 0 to view_dim dim src_view - 1 do
            (* Use src_view's shape, same as dst_view's *)
            current_md_idx.(dim) <- i;
            blit_slice (dim + 1)
          done
      in
      blit_slice 0
