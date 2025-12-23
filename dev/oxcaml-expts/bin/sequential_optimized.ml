module Dtype = Nx_core.Dtype
module View = Nx_core.View
module Symbolic_shape = Nx_core.Symbolic_shape
module Error = Nx_core.Error

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

type 'b buffer =
  | Float64 : float# array -> Dtype.float64_elt buffer
  | Float32 : float32# array -> Dtype.float32_elt buffer
  | Int32 : int32# array -> Dtype.int32_elt buffer
  | Int64 : int64# array -> Dtype.int64_elt buffer

type ('a, 'b) t = {
  dtype : ('a, 'b) Dtype.t;
  buffer : 'b buffer;
  view : View.t;
}

let view t = t.view
let dtype t = t.dtype
let context _t = ()

let shape (v : View.t) : int array =
  match Symbolic_shape.eval (View.shape v) with
  | Some arr -> arr
  | None -> Error.failed ~op:"shape" ~what:"symbolic shape not evaluable" ()

let volume (v : View.t) : int =
  Array.fold_left (fun acc s -> acc * s) 1 (shape v)

let physical_index (v : View.t) (idx : int array) : int =
  let strides = View.strides v in
  let offset = View.offset v in
  let ndim = Array.length strides in
  let rec loop i acc =
    if i = ndim then acc + offset
    else loop (i + 1) (acc + (strides.(i) * idx.(i)))
  in
  loop 0 0

let increment_index (idx : int array) (shape : int array) : unit =
  let ndim = Array.length shape in
  let carry = ref true in
  let pos = ref (ndim - 1) in
  while !carry && !pos >= 0 do
    idx.(!pos) <- idx.(!pos) + 1;
    if idx.(!pos) = shape.(!pos) then idx.(!pos) <- 0 else carry := false;
    decr pos
  done

(* *)

module Float_u = Stdlib_upstream_compatible.Float_u
module Float32_u = Stdlib_stable.Float32_u
module Int32_u = Stdlib_upstream_compatible.Int32_u
module Int64_u = Stdlib_upstream_compatible.Int64_u

let op_buffer (type a b) () (dtype : (a, b) Dtype.t) (size : int) : (a, b) t =
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  match dtype with
  | Dtype.Float64 ->
      let buffer = make_float64_array size in
      { dtype; buffer = Float64 buffer; view }
  | Dtype.Float32 ->
      let buffer = make_float32_array size in
      { dtype; buffer = Float32 buffer; view }
  | Dtype.Int32 ->
      let buffer = make_int32_array size in
      { dtype; buffer = Int32 buffer; view }
  | Dtype.Int64 ->
      let buffer = make_int64_array size in
      { dtype; buffer = Int64 buffer; view }
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

(* *)

let parallel_fork_join (par : Parallel.t @ local) a0 b0 a1 b1 a2 b2 a3 b3 =
  let #(o0, o1, o2, o3) =
    Parallel.fork_join2 par
      (fun _par -> a0 + b0)
      (fun _par -> a1 + b1)
      (fun _par -> a2 + b2)
      (fun _par -> a3 + b3)
  in
  (o0, o1, o2, o3)

let run_one_test ~(f : Parallel.t @ local -> 'a) : 'a =
  let module Scheduler = Parallel_scheduler in
  let scheduler = Scheduler.create () in
  let result = Scheduler.parallel scheduler ~f in
  Scheduler.stop scheduler;
  result

let add_float64 out_arr a_arr b_arr vout va vb =
  let vol = volume vout in
  let out_base = View.offset vout in
  let a_base = View.offset va in
  let b_base = View.offset vb in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let vol4 = vol - 3 in
    while !i < vol4 do
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

      let test_sum_oxcaml_parallel (par @ local) =
        parallel_fork_join par a0 b0 a1 b1 a2 b2 a3 b3
      in

      let o0, o1, o2, o3 = run_one_test ~f:test_sum_oxcaml_parallel in

      Array.unsafe_set out_arr (out_base + i0) o0;
      Array.unsafe_set out_arr (out_base + i1) o1;
      Array.unsafe_set out_arr (out_base + i2) o2;
      Array.unsafe_set out_arr (out_base + i3) o3;
      i := i0 + 4
    done;
    while !i < vol do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Float_u.add a_val b_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let ndim = Array.length out_shape in
    let idx = Array.make ndim 0 in
    for _ = 0 to vol - 1 do
      let out_p = physical_index vout idx in
      let a_p = physical_index va idx in
      let b_p = physical_index vb idx in
      Array.unsafe_set out_arr out_p
        (Float_u.add (Array.unsafe_get a_arr a_p) (Array.unsafe_get b_arr b_p));
      increment_index idx out_shape
    done

let add_float32 out_arr a_arr b_arr vout va vb =
  let vol = volume vout in
  let out_base = View.offset vout in
  let a_base = View.offset va in
  let b_base = View.offset vb in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let vol4 = vol - 3 in
    while !i < vol4 do
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
    while !i < vol do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Float32_u.add a_val b_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let ndim = Array.length out_shape in
    let idx = Array.make ndim 0 in
    for _ = 0 to vol - 1 do
      let out_p = physical_index vout idx in
      let a_p = physical_index va idx in
      let b_p = physical_index vb idx in
      Array.unsafe_set out_arr out_p
        (Float32_u.add
           (Array.unsafe_get a_arr a_p)
           (Array.unsafe_get b_arr b_p));
      increment_index idx out_shape
    done

let add_int32 out_arr a_arr b_arr vout va vb =
  let vol = volume vout in
  let out_base = View.offset vout in
  let a_base = View.offset va in
  let b_base = View.offset vb in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let vol4 = vol - 3 in
    while !i < vol4 do
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
    while !i < vol do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int32_u.add a_val b_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let ndim = Array.length out_shape in
    let idx = Array.make ndim 0 in
    for _ = 0 to vol - 1 do
      let out_p = physical_index vout idx in
      let a_p = physical_index va idx in
      let b_p = physical_index vb idx in
      Array.unsafe_set out_arr out_p
        (Int32_u.add (Array.unsafe_get a_arr a_p) (Array.unsafe_get b_arr b_p));
      increment_index idx out_shape
    done

let add_int64 out_arr a_arr b_arr vout va vb =
  let vol = volume vout in
  let out_base = View.offset vout in
  let a_base = View.offset va in
  let b_base = View.offset vb in
  if
    View.is_c_contiguous vout && View.is_c_contiguous va
    && View.is_c_contiguous vb
  then (
    let i = ref 0 in
    let vol4 = vol - 3 in
    while !i < vol4 do
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
    while !i < vol do
      let idx = !i in
      let a_val = Array.unsafe_get a_arr (a_base + idx) in
      let b_val = Array.unsafe_get b_arr (b_base + idx) in
      Array.unsafe_set out_arr (out_base + idx) (Int64_u.add a_val b_val);
      incr i
    done)
  else
    let out_shape = shape vout in
    let ndim = Array.length out_shape in
    let idx = Array.make ndim 0 in
    for _ = 0 to vol - 1 do
      let out_p = physical_index vout idx in
      let a_p = physical_index va idx in
      let b_p = physical_index vb idx in
      Array.unsafe_set out_arr out_p
        (Int64_u.add (Array.unsafe_get a_arr a_p) (Array.unsafe_get b_arr b_p));
      increment_index idx out_shape
    done

let op_add (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      add_float64 out_arr a_arr b_arr vout va vb
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      add_float32 out_arr a_arr b_arr vout va vb
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      add_int32 out_arr a_arr b_arr vout va vb
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      add_int64 out_arr a_arr b_arr vout va vb
