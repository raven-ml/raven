module Dtype = Nx_core.Dtype
module View = Nx_core.View
module Shape = Nx_core.Shape
module Symbolic_shape = Nx_core.Symbolic_shape
module Error = Nx_core.Error
module Parallel = Parallel_internal

module Array = struct
  include Stdlib.Array
  external[@layout_poly] get : ('a : any mod non_null separable). 'a array -> int -> 'a = "%array_safe_get"
  external[@layout_poly] set : ('a : any mod non_null separable). 'a array -> int -> 'a -> unit = "%array_safe_set"
  external[@layout_poly] unsafe_get : ('a : any mod non_null separable). 'a array -> int -> 'a = "%array_unsafe_get"
  external[@layout_poly] unsafe_set : ('a : any mod non_null separable). 'a array -> int -> 'a -> unit = "%array_unsafe_set"
end

external make_float64_array : int -> float# array = "caml_make_unboxed_float64_vect"
external make_float32_array : int -> float32# array = "caml_make_unboxed_float32_vect"
external make_int32_array : int -> int32# array = "caml_make_unboxed_int32_vect"
external make_int64_array : int -> int64# array = "caml_make_unboxed_int64_vect"

type 'b buffer =
  | Float64 : float#   array -> Dtype.float64_elt buffer
  | Float32 : float32# array -> Dtype.float32_elt buffer
  | Int32   : int32#   array -> Dtype.int32_elt   buffer
  | Int64   : int64#   array -> Dtype.int64_elt   buffer

type context = { pool : Parallel.pool }

type ('a, 'b) t = {
  dtype : ('a, 'b) Dtype.t;
  buffer : ('b) buffer;
  view : View.t;
  context : context;
}

let view t = t.view
let dtype t = t.dtype
let context t = t.context

let shape (v : View.t) : int array =
  match Symbolic_shape.eval (View.shape v) with
  | Some arr -> arr
  | None -> Error.failed ~op:"shape" ~what:"symbolic shape not evaluable" ()

let volume (v : View.t) : int =
  Array.fold_left (fun acc s -> acc * s) 1 (shape v)

(*  *)

let create_context () = { pool = Parallel.get_or_setup_pool () }

(*  *)

module Float_u = Stdlib_upstream_compatible.Float_u
module Float32_u = Stdlib_stable.Float32_u
module Int32_u = Stdlib_upstream_compatible.Int32_u
module Int64_u = Stdlib_upstream_compatible.Int64_u

let op_buffer (type a b) context (dtype : (a, b) Dtype.t) (size : int) : (a, b) t =
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

(*  *)

let add_float64 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va && View.is_c_contiguous vb then begin
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let b0 = Array.unsafe_get b_arr (b_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let b1 = Array.unsafe_get b_arr (b_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let b2 = Array.unsafe_get b_arr (b_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      let b3 = Array.unsafe_get b_arr (b_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Float_u.add a0 b0);
      Array.unsafe_set out_arr (out_base + i1) (Float_u.add a1 b1);
      Array.unsafe_set out_arr (out_base + i2) (Float_u.add a2 b2);
      Array.unsafe_set out_arr (out_base + i3) (Float_u.add a3 b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Float_u.add a_val b_val);
      incr i
    done
  end
  else begin
    let out_shape = shape vout in
    let a_shape = shape va in
    let b_shape = shape vb in
    let a_strides = View.strides va in
    let b_strides = View.strides vb in
    let a_offset = View.offset va in
    let b_offset = View.offset vb in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      let b_val = Array.unsafe_get b_arr (b_offset + b_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Float_u.add a_val b_val)
    done
  end

let add_float32 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va && View.is_c_contiguous vb then begin
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let b0 = Array.unsafe_get b_arr (b_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let b1 = Array.unsafe_get b_arr (b_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let b2 = Array.unsafe_get b_arr (b_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      let b3 = Array.unsafe_get b_arr (b_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Float32_u.add a0 b0);
      Array.unsafe_set out_arr (out_base + i1) (Float32_u.add a1 b1);
      Array.unsafe_set out_arr (out_base + i2) (Float32_u.add a2 b2);
      Array.unsafe_set out_arr (out_base + i3) (Float32_u.add a3 b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Float32_u.add a_val b_val);
      incr i
    done
  end
  else begin
    let out_shape = shape vout in
    let a_shape = shape va in
    let b_shape = shape vb in
    let a_strides = View.strides va in
    let b_strides = View.strides vb in
    let a_offset = View.offset va in
    let b_offset = View.offset vb in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      let b_val = Array.unsafe_get b_arr (b_offset + b_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Float32_u.add a_val b_val)
    done
  end

let add_int32 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va && View.is_c_contiguous vb then begin
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let b0 = Array.unsafe_get b_arr (b_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let b1 = Array.unsafe_get b_arr (b_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let b2 = Array.unsafe_get b_arr (b_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      let b3 = Array.unsafe_get b_arr (b_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Int32_u.add a0 b0);
      Array.unsafe_set out_arr (out_base + i1) (Int32_u.add a1 b1);
      Array.unsafe_set out_arr (out_base + i2) (Int32_u.add a2 b2);
      Array.unsafe_set out_arr (out_base + i3) (Int32_u.add a3 b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int32_u.add a_val b_val);
      incr i
    done
  end
  else begin
    let out_shape = shape vout in
    let a_shape = shape va in
    let b_shape = shape vb in
    let a_strides = View.strides va in
    let b_strides = View.strides vb in
    let a_offset = View.offset va in
    let b_offset = View.offset vb in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      let b_val = Array.unsafe_get b_arr (b_offset + b_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Int32_u.add a_val b_val)
    done
  end

let add_int64 a_arr b_arr out_arr va vb vout start_idx end_idx =
  let out_base = View.offset vout + start_idx in
  let a_base = View.offset va + start_idx in
  let b_base = View.offset vb + start_idx in
  if View.is_c_contiguous vout && View.is_c_contiguous va && View.is_c_contiguous vb then begin
    let i = ref 0 in
    let n = end_idx - start_idx in
    let n4 = n - 3 in
    while !i < n4 do
      let i0 = !i in
      let i1 = i0 + 1 in
      let i2 = i0 + 2 in
      let i3 = i0 + 3 in
      let a0 = Array.unsafe_get a_arr (a_base + i0) in
      let b0 = Array.unsafe_get b_arr (b_base + i0) in
      let a1 = Array.unsafe_get a_arr (a_base + i1) in
      let b1 = Array.unsafe_get b_arr (b_base + i1) in
      let a2 = Array.unsafe_get a_arr (a_base + i2) in
      let b2 = Array.unsafe_get b_arr (b_base + i2) in
      let a3 = Array.unsafe_get a_arr (a_base + i3) in
      let b3 = Array.unsafe_get b_arr (b_base + i3) in
      Array.unsafe_set out_arr (out_base + i0) (Int64_u.add a0 b0);
      Array.unsafe_set out_arr (out_base + i1) (Int64_u.add a1 b1);
      Array.unsafe_set out_arr (out_base + i2) (Int64_u.add a2 b2);
      Array.unsafe_set out_arr (out_base + i3) (Int64_u.add a3 b3);
      i := i0 + 4
    done;
    while !i < n do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int64_u.add a_val b_val);
      incr i
    done
  end
  else begin
    let out_shape = shape vout in
    let a_shape = shape va in
    let b_shape = shape vb in
    let a_strides = View.strides va in
    let b_strides = View.strides vb in
    let a_offset = View.offset va in
    let b_offset = View.offset vb in
    let out_offset = View.offset vout in
    let md_idx = Array.make (Array.length out_shape) 0 in
    let a_idx = Array.make (Array.length a_shape) 0 in
    let b_idx = Array.make (Array.length b_shape) 0 in
    for k = start_idx to end_idx - 1 do
      Shape.unravel_index_into k out_shape md_idx;
      Shape.broadcast_index_into md_idx a_shape a_idx;
      let a_lin = Shape.ravel_index a_idx a_strides in
      Shape.broadcast_index_into md_idx b_shape b_idx;
      let b_lin = Shape.ravel_index b_idx b_strides in
      let a_val = Array.unsafe_get a_arr (a_offset + a_lin) in
      let b_val = Array.unsafe_get b_arr (b_offset + b_lin) in
      Array.unsafe_set out_arr (out_offset + k) (Int64_u.add a_val b_val)
    done
  end

let op_add (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = volume vout in
  match out.buffer, a.buffer, b.buffer with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
    if vol > parallel_threshold then
      Parallel.parallel_for out.context.pool 0 (vol - 1) (fun start_idx end_idx ->
        add_float64 a_arr b_arr out_arr va vb vout start_idx end_idx)
    else
      add_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
    if vol > parallel_threshold then
      Parallel.parallel_for out.context.pool 0 (vol - 1) (fun start_idx end_idx ->
        add_float32 a_arr b_arr out_arr va vb vout start_idx end_idx)
    else
      add_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
    if vol > parallel_threshold then
      Parallel.parallel_for out.context.pool 0 (vol - 1) (fun start_idx end_idx ->
        add_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
    else
      add_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
    if vol > parallel_threshold then
      Parallel.parallel_for out.context.pool 0 (vol - 1) (fun start_idx end_idx ->
        add_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
    else
      add_int64 a_arr b_arr out_arr va vb vout 0 vol