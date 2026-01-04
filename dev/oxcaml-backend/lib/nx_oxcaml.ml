(* open Nx_core open Bigarray_ext *)
open Nx_oxcaml_binary
module Dtype = Nx_core.Dtype
module View = Nx_core.View
module Shape = Nx_core.Shape
module Symbolic_shape = Nx_core.Symbolic_shape
module Error = Nx_core.Error
module Parallel = Parallel_pool
(* *)

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

type ('a, 'b) oxcaml_tensor = {
  data : 'b buffer;
  shape : int array;
  strides : int array;
  offset : int;
}
[@@warning "-69"]

(* Helper functions *)
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

external make_float64_array : int -> float# array
  = "caml_make_unboxed_float64_vect"

external make_float32_array : int -> float32# array
  = "caml_make_unboxed_float32_vect"

external make_int32_array : int -> int32# array = "caml_make_unboxed_int32_vect"
external make_int64_array : int -> int64# array = "caml_make_unboxed_int64_vect"

let view t = t.view
let dtype t = t.dtype
let context t = t.context
let data t = t.buffer

let strides t =
  match View.strides_opt t.view with
  | Some s -> s
  | None -> Error.failed ~op:"strides" ~what:"cannot get strides for view" ()

let offset t = View.offset t.view
let is_contiguous t = View.is_c_contiguous t.view

(* Check if a tensor can be efficiently operated on *)
let can_get_strides t = View.can_get_strides t.view

(* Buffer allocation *)
let op_buffer (type a b) context (dtype : (a, b) Dtype.t) (size : int) :
    (a, b) t =
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  match dtype with
  | Dtype.Float64 ->
      let buffer = make_float64_array size in
      { dtype; buffer = Float64 buffer; view; context }
  | Dtype.Float32 ->
      let buffer = make_float32_array size in
      { dtype; buffer = Float32 buffer; view; context }
  | Dtype.Int32 ->
      let buffer = make_int32_array size in
      { dtype; buffer = Int32 buffer; view; context }
  | Dtype.Int64 ->
      let buffer = make_int64_array size in
      { dtype; buffer = Int64 buffer; view; context }
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

(* Create a new tensor with given shape *)
let create_tensor (type a b) ctx (dtype : (a, b) Dtype.t) shape_arr : (a, b) t =
  let size = Array.fold_left ( * ) 1 shape_arr in
  let shape = Symbolic_shape.of_ints shape_arr in
  let view = View.create shape in
  match dtype with
  | Dtype.Float64 ->
      let buffer = make_float64_array size in
      { dtype; buffer = Float64 buffer; view; context = ctx }
  | Dtype.Float32 ->
      let buffer = make_float32_array size in
      { dtype; buffer = Float32 buffer; view; context = ctx }
  | Dtype.Int32 ->
      let buffer = make_int32_array size in
      { dtype; buffer = Int32 buffer; view; context = ctx }
  | Dtype.Int64 ->
      let buffer = make_int64_array size in
      { dtype; buffer = Int64 buffer; view; context = ctx }
  | _ -> Error.invalid ~op:"create_tensor" ~what:"unsupported dtype" ()

let caml_add (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = volume vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            add_float64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else add_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            add_float32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else add_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            add_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else add_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            add_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else add_int64 a_arr b_arr out_arr va vb vout 0 vol

let caml_sub (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = volume vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            sub_float64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else sub_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            sub_float32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else sub_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            sub_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else sub_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            sub_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else sub_int64 a_arr b_arr out_arr va vb vout 0 vol

(* Binary operations *)
let op_add ~out x y = binary_op caml_add ~out x y
let op_sub ~out x y = binary_op caml_sub ~out x y
