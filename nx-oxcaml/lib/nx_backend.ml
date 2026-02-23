(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import
open Nx_buffer

type context = { pool : Parallel.pool }

let create_context () = { pool = Parallel.get_or_setup_pool () }

type 'b buffer =
  | Float64 : float# array -> Dtype.float64_elt buffer
  | Float32 : float32# array -> Dtype.float32_elt buffer
  | Int8 : int8# array -> Dtype.int8_elt buffer
  | Int16 : int16# array -> Dtype.int16_elt buffer
  | Int32 : int32# array -> Dtype.int32_elt buffer
  | Int64 : int64# array -> Dtype.int64_elt buffer
  | Bool : bool array -> Dtype.bool_elt buffer

type ('a, 'b) t = {
  dtype : ('a, 'b) Dtype.t;
  buffer : 'b buffer;
  view : View.t;
  context : context;
}

let view t = t.view
let dtype t = t.dtype
let context t = t.context

let to_host (type a b) (t : (a, b) t) :
    (a, b, Nx_buffer.c_layout) Nx_buffer.Array1.t =
  let n = numel t.view in
  match t.dtype with
  | Dtype.Float64 ->
    let (Float64 arr) = t.buffer in
    Array.unboxed_float64_to_ba arr n
  | Dtype.Float32 ->
    let (Float32 arr) = t.buffer in
    Array.unboxed_float32_to_ba arr n
  | Dtype.Int64 ->
    let (Int64 arr) = t.buffer in
    Array.unboxed_int64_to_ba arr n
  | Dtype.Int32 ->
    let (Int32 arr) = t.buffer in
    Array.unboxed_int32_to_ba arr n
  | _ -> Error.invalid ~op:"to_host" ~what:"unsupported dtype" ()

let buffer (type a b) context (dtype : (a, b) Dtype.t) (shape_arr : int array) :
    (a, b) t =
  let size = Stdlib.Array.fold_left ( * ) 1 shape_arr in
  let sym_shape = Symbolic_shape.of_ints shape_arr in
  let view = View.create sym_shape in
  match dtype with
  | Dtype.Float64 ->
      let buffer = Array.make_float64 size in
      { dtype; buffer = Float64 buffer; view; context }
  | Dtype.Float32 ->
      let buffer = Array.make_float32 size in
      { dtype; buffer = Float32 buffer; view; context }
  | Dtype.Int8 ->
      let buffer = Array.make_int8 size in
      { dtype; buffer = Int8 buffer; view; context }
  | Dtype.Int16 ->
      let buffer = Array.make_int16 size in
      { dtype; buffer = Int16 buffer; view; context }
  | Dtype.Int32 ->
      let buffer = Array.make_int32 size in
      { dtype; buffer = Int32 buffer; view; context }
  | Dtype.Int64 ->
      let buffer = Array.make_int64 size in
      { dtype; buffer = Int64 buffer; view; context }
  | Dtype.Bool ->
      let buffer = Array.make size false in
      { dtype; buffer = Bool buffer; view; context }
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let full (type a b) context (dtype : (a, b) Dtype.t) (shape_arr : int array)
    (value : a) : (a, b) t =
  let t = buffer context dtype shape_arr in
  let size = Stdlib.Array.fold_left ( * ) 1 shape_arr in
  (match (dtype : (a, b) Dtype.t) with
  | Dtype.Float64 ->
      let Float64 arr = t.buffer in
      let v = Float_u.of_float value in
      for i = 0 to size - 1 do
        Array.unsafe_set arr i v
      done
  | Dtype.Float32 ->
      let Float32 arr = t.buffer in
      let v = Float32_u.of_float (Float_u.of_float value) in
      for i = 0 to size - 1 do
        Array.unsafe_set arr i v
      done
  | Dtype.Int8 ->
      let Int8 arr = t.buffer in
      let v = Int8_u.of_int value in
      for i = 0 to size - 1 do
        Array.unsafe_set arr i v
      done
  | Dtype.Int16 ->
      let Int16 arr = t.buffer in
      let v = Int16_u.of_int value in
      for i = 0 to size - 1 do
        Array.unsafe_set arr i v
      done
  | Dtype.Int32 ->
      let Int32 arr = t.buffer in
      let v = Int32_u.of_int32 value in
      for i = 0 to size - 1 do
        Array.unsafe_set arr i v
      done
  | Dtype.Int64 ->
      let Int64 arr = t.buffer in
      let v = Int64_u.of_int64 value in
      for i = 0 to size - 1 do
        Array.unsafe_set arr i v
      done
  | Dtype.Bool ->
      let Bool arr = t.buffer in
      for i = 0 to size - 1 do
        Stdlib.Array.unsafe_set arr i value
      done
  | _ -> Error.invalid ~op:"full" ~what:"unsupported dtype" ());
  t

let add (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_add.add_float64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_add.add_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_add.add_float32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_add.add_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_add.add_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_add.add_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_add.add_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_add.add_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let sub (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sub.sub_float64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_sub.sub_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sub.sub_float32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_sub.sub_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sub.sub_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_sub.sub_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sub.sub_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_sub.sub_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let mul (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_mul.mul_float64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_mul.mul_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_mul.mul_float32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_mul.mul_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_mul.mul_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_mul.mul_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_mul.mul_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_mul.mul_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let idiv (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_idiv.idiv_float64 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_idiv.idiv_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_idiv.idiv_float32 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_idiv.idiv_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_idiv.idiv_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_idiv.idiv_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_idiv.idiv_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_idiv.idiv_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let fdiv (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_fdiv.fdiv_float64 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_fdiv.fdiv_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_fdiv.fdiv_float32 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_fdiv.fdiv_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_fdiv.fdiv_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_fdiv.fdiv_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_fdiv.fdiv_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_fdiv.fdiv_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"fdiv" ~what:"unsupported dtype" ()

let div ~out x y =
  let dt = dtype out in
  if Dtype.is_int dt || Dtype.is_uint dt then idiv ~out x y
  else fdiv ~out x y

let mod_ (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_mod.mod_float64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_mod.mod_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_mod.mod_float32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_mod.mod_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_mod.mod_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_mod.mod_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_mod.mod_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_mod.mod_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let pow (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_pow.pow_float64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_pow.pow_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_pow.pow_float32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_pow.pow_float32 a_arr b_arr out_arr va vb vout 0 vol
  | _ ->
      Error.invalid ~op:"pow" ~what:"not implemented for unboxed ints" ()

let cmpeq (type a b) ~(out : (bool, Nx_buffer.bool_elt) t) (a : (a, b) t)
    (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Bool out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmpeq.cmpeq_float64 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmpeq.cmpeq_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmpeq.cmpeq_float32 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmpeq.cmpeq_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmpeq.cmpeq_int32 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmpeq.cmpeq_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmpeq.cmpeq_int64 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmpeq.cmpeq_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let cmpne (type a b) ~(out : (bool, Nx_buffer.bool_elt) t) (a : (a, b) t)
    (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Bool out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmpne.cmpne_float64 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmpne.cmpne_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmpne.cmpne_float32 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmpne.cmpne_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmpne.cmpne_int32 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmpne.cmpne_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmpne.cmpne_int64 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmpne.cmpne_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let cmplt (type a b) ~(out : (bool, Nx_buffer.bool_elt) t) (a : (a, b) t)
    (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Bool out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmplt.cmplt_float64 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmplt.cmplt_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmplt.cmplt_float32 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmplt.cmplt_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmplt.cmplt_int32 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmplt.cmplt_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmplt.cmplt_int64 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmplt.cmplt_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let cmple (type a b) ~(out : (bool, Nx_buffer.bool_elt) t) (a : (a, b) t)
    (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Bool out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmple.cmple_float64 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmple.cmple_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmple.cmple_float32 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmple.cmple_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmple.cmple_int32 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmple.cmple_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Bool out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cmple.cmple_int64 a_arr b_arr out_arr va vb vout start_idx
              end_idx)
      else Op_cmple.cmple_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let max (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_max.max_float64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_max.max_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_max.max_float32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_max.max_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_max.max_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_max.max_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_max.max_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_max.max_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"max" ~what:"unsupported dtype" ()

let min (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 out_arr, Float64 a_arr, Float64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_min.min_float64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_min.min_float64 a_arr b_arr out_arr va vb vout 0 vol
  | Float32 out_arr, Float32 a_arr, Float32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_min.min_float32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_min.min_float32 a_arr b_arr out_arr va vb vout 0 vol
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_min.min_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_min.min_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_min.min_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_min.min_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"min" ~what:"unsupported dtype" ()

let xor (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_xor.xor_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_xor.xor_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_xor.xor_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_xor.xor_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"or_" ~what:"not implemented for unboxed ints" ()

let or_ (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_or.or_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_or.or_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_or.or_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_or.or_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"or_" ~what:"not implemented for unboxed ints" ()

let and_ (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vb = b.view in
  let vol = numel vout in
  match (out.buffer, a.buffer, b.buffer) with
  | Int32 out_arr, Int32 a_arr, Int32 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_and.and_int32 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_and.and_int32 a_arr b_arr out_arr va vb vout 0 vol
  | Int64 out_arr, Int64 a_arr, Int64 b_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_and.and_int64 a_arr b_arr out_arr va vb vout start_idx end_idx)
      else Op_and.and_int64 a_arr b_arr out_arr va vb vout 0 vol
  | _ -> Error.invalid ~op:"and_" ~what:"not implemented for unboxed ints" ()

let neg (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_neg.neg_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_neg.neg_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_neg.neg_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_neg.neg_float32 a_arr out_arr va vout 0 vol
  | Int8 out_arr, Int8 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_neg.neg_int8 a_arr out_arr va vout start_idx end_idx)
      else Op_neg.neg_int8 a_arr out_arr va vout 0 vol
  | Int16 out_arr, Int16 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_neg.neg_int16 a_arr out_arr va vout start_idx end_idx)
      else Op_neg.neg_int16 a_arr out_arr va vout 0 vol
  | Int32 out_arr, Int32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_neg.neg_int32 a_arr out_arr va vout start_idx end_idx)
      else Op_neg.neg_int32 a_arr out_arr va vout 0 vol
  | Int64 out_arr, Int64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_neg.neg_int64 a_arr out_arr va vout start_idx end_idx)
      else Op_neg.neg_int64 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let recip (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_recip.recip_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_recip.recip_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_recip.recip_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_recip.recip_float32 a_arr out_arr va vout 0 vol
  | Int8 out_arr, Int8 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_recip.recip_int8 a_arr out_arr va vout start_idx end_idx)
      else Op_recip.recip_int8 a_arr out_arr va vout 0 vol
  | Int16 out_arr, Int16 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_recip.recip_int16 a_arr out_arr va vout start_idx end_idx)
      else Op_recip.recip_int16 a_arr out_arr va vout 0 vol
  | Int32 out_arr, Int32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_recip.recip_int32 a_arr out_arr va vout start_idx end_idx)
      else Op_recip.recip_int32 a_arr out_arr va vout 0 vol
  | Int64 out_arr, Int64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_recip.recip_int64 a_arr out_arr va vout start_idx end_idx)
      else Op_recip.recip_int64 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let abs (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_abs.abs_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_abs.abs_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_abs.abs_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_abs.abs_float32 a_arr out_arr va vout 0 vol
  | Int8 out_arr, Int8 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_abs.abs_int8 a_arr out_arr va vout start_idx end_idx)
      else Op_abs.abs_int8 a_arr out_arr va vout 0 vol
  | Int16 out_arr, Int16 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_abs.abs_int16 a_arr out_arr va vout start_idx end_idx)
      else Op_abs.abs_int16 a_arr out_arr va vout 0 vol
  | Int32 out_arr, Int32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_abs.abs_int32 a_arr out_arr va vout start_idx end_idx)
      else Op_abs.abs_int32 a_arr out_arr va vout 0 vol
  | Int64 out_arr, Int64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_abs.abs_int64 a_arr out_arr va vout start_idx end_idx)
      else Op_abs.abs_int64 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let sqrt (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sqrt.sqrt_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_sqrt.sqrt_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sqrt.sqrt_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_sqrt.sqrt_float32 a_arr out_arr va vout 0 vol
  | _ ->
      Error.invalid ~op:"sqrt" ~what:"not implemented for unboxed ints" ()

let exp (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_exp.exp_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_exp.exp_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_exp.exp_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_exp.exp_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"exp" ~what:"not implemented for unboxed ints" ()

let log (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_log.log_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_log.log_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_log.log_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_log.log_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"log" ~what:"not implemented for unboxed ints" ()

let sin (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sin.sin_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_sin.sin_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sin.sin_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_sin.sin_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"sin" ~what:"not implemented for unboxed ints" ()

let cos (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cos.cos_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_cos.cos_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cos.cos_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_cos.cos_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"cos" ~what:"not implemented for unboxed ints" ()

let sign (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sign.sign_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_sign.sign_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sign.sign_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_sign.sign_float32 a_arr out_arr va vout 0 vol
  | Int8 out_arr, Int8 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sign.sign_int8 a_arr out_arr va vout start_idx end_idx)
      else Op_sign.sign_int8 a_arr out_arr va vout 0 vol
  | Int16 out_arr, Int16 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sign.sign_int16 a_arr out_arr va vout start_idx end_idx)
      else Op_sign.sign_int16 a_arr out_arr va vout 0 vol
  | Int32 out_arr, Int32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sign.sign_int32 a_arr out_arr va vout start_idx end_idx)
      else Op_sign.sign_int32 a_arr out_arr va vout 0 vol
  | Int64 out_arr, Int64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sign.sign_int64 a_arr out_arr va vout start_idx end_idx)
      else Op_sign.sign_int64 a_arr out_arr va vout 0 vol
  | Bool out_arr, Bool a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sign.sign_bool a_arr out_arr va vout start_idx end_idx)
      else Op_sign.sign_bool a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"sign" ~what:"unsupported dtype" ()

let tan (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_tan.tan_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_tan.tan_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_tan.tan_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_tan.tan_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"tan" ~what:"not implemented for unboxed ints" ()

let asin (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_asin.asin_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_asin.asin_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_asin.asin_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_asin.asin_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"asin" ~what:"not implemented for unboxed ints" ()

let acos (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_acos.acos_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_acos.acos_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_acos.acos_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_acos.acos_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"acos" ~what:"not implemented for unboxed ints" ()

let atan (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_atan.atan_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_atan.atan_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_atan.atan_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_atan.atan_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"atan" ~what:"not implemented for unboxed ints" ()

let atan2 ~out:_ _ _ = Error.invalid ~op:"atan2" ~what:"not implemented" ()

let sinh (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sinh.sinh_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_sinh.sinh_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_sinh.sinh_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_sinh.sinh_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"sinh" ~what:"not implemented for unboxed ints" ()

let cosh (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cosh.cosh_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_cosh.cosh_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_cosh.cosh_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_cosh.cosh_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"cosh" ~what:"not implemented for unboxed ints" ()

let tanh (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_tanh.tanh_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_tanh.tanh_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_tanh.tanh_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_tanh.tanh_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"tanh" ~what:"not implemented for unboxed ints" ()

let trunc (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_trunc.trunc_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_trunc.trunc_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_trunc.trunc_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_trunc.trunc_float32 a_arr out_arr va vout 0 vol
  | Int8 out_arr, Int8 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_trunc.trunc_int8 a_arr out_arr va vout start_idx end_idx)
      else Op_trunc.trunc_int8 a_arr out_arr va vout 0 vol
  | Int16 out_arr, Int16 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_trunc.trunc_int16 a_arr out_arr va vout start_idx end_idx)
      else Op_trunc.trunc_int16 a_arr out_arr va vout 0 vol
  | Int32 out_arr, Int32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_trunc.trunc_int32 a_arr out_arr va vout start_idx end_idx)
      else Op_trunc.trunc_int32 a_arr out_arr va vout 0 vol
  | Int64 out_arr, Int64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_trunc.trunc_int64 a_arr out_arr va vout start_idx end_idx)
      else Op_trunc.trunc_int64 a_arr out_arr va vout 0 vol
  | Bool out_arr, Bool a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_trunc.trunc_bool a_arr out_arr va vout start_idx end_idx)
      else Op_trunc.trunc_bool a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"trunc" ~what:"unsupported dtype" ()

let ceil (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_ceil.ceil_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_ceil.ceil_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_ceil.ceil_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_ceil.ceil_float32 a_arr out_arr va vout 0 vol
  | Int8 out_arr, Int8 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_ceil.ceil_int8 a_arr out_arr va vout start_idx end_idx)
      else Op_ceil.ceil_int8 a_arr out_arr va vout 0 vol
  | Int16 out_arr, Int16 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_ceil.ceil_int16 a_arr out_arr va vout start_idx end_idx)
      else Op_ceil.ceil_int16 a_arr out_arr va vout 0 vol
  | Int32 out_arr, Int32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_ceil.ceil_int32 a_arr out_arr va vout start_idx end_idx)
      else Op_ceil.ceil_int32 a_arr out_arr va vout 0 vol
  | Int64 out_arr, Int64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_ceil.ceil_int64 a_arr out_arr va vout start_idx end_idx)
      else Op_ceil.ceil_int64 a_arr out_arr va vout 0 vol
  | Bool out_arr, Bool a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_ceil.ceil_bool a_arr out_arr va vout start_idx end_idx)
      else Op_ceil.ceil_bool a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"ceil" ~what:"unsupported dtype" ()

let floor (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_floor.floor_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_floor.floor_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_floor.floor_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_floor.floor_float32 a_arr out_arr va vout 0 vol
  | Int8 out_arr, Int8 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_floor.floor_int8 a_arr out_arr va vout start_idx end_idx)
      else Op_floor.floor_int8 a_arr out_arr va vout 0 vol
  | Int16 out_arr, Int16 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_floor.floor_int16 a_arr out_arr va vout start_idx end_idx)
      else Op_floor.floor_int16 a_arr out_arr va vout 0 vol
  | Int32 out_arr, Int32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_floor.floor_int32 a_arr out_arr va vout start_idx end_idx)
      else Op_floor.floor_int32 a_arr out_arr va vout 0 vol
  | Int64 out_arr, Int64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_floor.floor_int64 a_arr out_arr va vout start_idx end_idx)
      else Op_floor.floor_int64 a_arr out_arr va vout 0 vol
  | Bool out_arr, Bool a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_floor.floor_bool a_arr out_arr va vout start_idx end_idx)
      else Op_floor.floor_bool a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"floor" ~what:"unsupported dtype" ()

let round (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_round.round_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_round.round_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_round.round_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_round.round_float32 a_arr out_arr va vout 0 vol
  | Int8 out_arr, Int8 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_round.round_int8 a_arr out_arr va vout start_idx end_idx)
      else Op_round.round_int8 a_arr out_arr va vout 0 vol
  | Int16 out_arr, Int16 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_round.round_int16 a_arr out_arr va vout start_idx end_idx)
      else Op_round.round_int16 a_arr out_arr va vout 0 vol
  | Int32 out_arr, Int32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_round.round_int32 a_arr out_arr va vout start_idx end_idx)
      else Op_round.round_int32 a_arr out_arr va vout 0 vol
  | Int64 out_arr, Int64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_round.round_int64 a_arr out_arr va vout start_idx end_idx)
      else Op_round.round_int64 a_arr out_arr va vout 0 vol
  | Bool out_arr, Bool a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_round.round_bool a_arr out_arr va vout start_idx end_idx)
      else Op_round.round_bool a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"round" ~what:"unsupported dtype" ()

let erf (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let va = a.view in
  let vol = numel vout in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_erf.erf_float64 a_arr out_arr va vout start_idx end_idx)
      else Op_erf.erf_float64 a_arr out_arr va vout 0 vol
  | Float32 out_arr, Float32 a_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_erf.erf_float32 a_arr out_arr va vout start_idx end_idx)
      else Op_erf.erf_float32 a_arr out_arr va vout 0 vol
  | _ -> Error.invalid ~op:"erf" ~what:"not implemented for unboxed ints" ()

let where (type a b) ~(out : (a, b) t) (cond : (bool, Nx_buffer.bool_elt) t)
    (if_true : (a, b) t) (if_false : (a, b) t) : unit =
  let parallel_threshold = 62500 in
  let vout = out.view in
  let vtrue = if_true.view in
  let vfalse = if_false.view in
  let vcond = cond.view in
  let vol = numel vout in
  match (out.buffer, cond.buffer, if_true.buffer, if_false.buffer) with
  | Float64 out_arr, Bool cond_arr, Float64 true_arr, Float64 false_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_where.where_float64 cond_arr true_arr false_arr out_arr vcond
              vtrue vfalse vout start_idx end_idx)
      else
        Op_where.where_float64 cond_arr true_arr false_arr out_arr vcond vtrue
          vfalse vout 0 vol
  | Float32 out_arr, Bool cond_arr, Float32 true_arr, Float32 false_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_where.where_float32 cond_arr true_arr false_arr out_arr vcond
              vtrue vfalse vout start_idx end_idx)
      else
        Op_where.where_float32 cond_arr true_arr false_arr out_arr vcond vtrue
          vfalse vout 0 vol
  | Int64 out_arr, Bool cond_arr, Int64 true_arr, Int64 false_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_where.where_int64 cond_arr true_arr false_arr out_arr vcond vtrue
              vfalse vout start_idx end_idx)
      else
        Op_where.where_int64 cond_arr true_arr false_arr out_arr vcond vtrue
          vfalse vout 0 vol
  | Int32 out_arr, Bool cond_arr, Int32 true_arr, Int32 false_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_where.where_int32 cond_arr true_arr false_arr out_arr vcond vtrue
              vfalse vout start_idx end_idx)
      else
        Op_where.where_int32 cond_arr true_arr false_arr out_arr vcond vtrue
          vfalse vout 0 vol
  | Int8 out_arr, Bool cond_arr, Int8 true_arr, Int8 false_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_where.where_int8 cond_arr true_arr false_arr out_arr vcond vtrue
              vfalse vout start_idx end_idx)
      else
        Op_where.where_int8 cond_arr true_arr false_arr out_arr vcond vtrue
          vfalse vout 0 vol
  | Int16 out_arr, Bool cond_arr, Int16 true_arr, Int16 false_arr ->
      if vol > parallel_threshold then
        Parallel.parallel_for out.context.pool 0 (vol - 1)
          (fun start_idx end_idx ->
            Op_where.where_int16 cond_arr true_arr false_arr out_arr vcond vtrue
              vfalse vout start_idx end_idx)
      else
        Op_where.where_int16 cond_arr true_arr false_arr out_arr vcond vtrue
          vfalse vout 0 vol
  | _ -> Error.invalid ~op:"where" ~what:"not implemented for this dtype" ()

let reduce_sum (type a b) ~(out : (a, b) t) ~axes ~keepdims (a : (a, b) t) :
    unit =
  let vout = out.view in
  let va = a.view in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      Reduce_ops.reduce_sum_float64 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Float32 out_arr, Float32 a_arr ->
      Reduce_ops.reduce_sum_float32 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Int32 out_arr, Int32 a_arr ->
      Reduce_ops.reduce_sum_int32 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Int64 out_arr, Int64 a_arr ->
      Reduce_ops.reduce_sum_int64 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let reduce_prod (type a b) ~(out : (a, b) t) ~axes ~keepdims (a : (a, b) t) :
    unit =
  let vout = out.view in
  let va = a.view in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      Reduce_ops.reduce_prod_float64 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Float32 out_arr, Float32 a_arr ->
      Reduce_ops.reduce_prod_float32 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Int32 out_arr, Int32 a_arr ->
      Reduce_ops.reduce_prod_int32 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Int64 out_arr, Int64 a_arr ->
      Reduce_ops.reduce_prod_int64 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let reduce_max (type a b) ~(out : (a, b) t) ~axes ~keepdims (a : (a, b) t) :
    unit =
  let vout = out.view in
  let va = a.view in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      Reduce_ops.reduce_max_float64 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Float32 out_arr, Float32 a_arr ->
      Reduce_ops.reduce_max_float32 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Int32 out_arr, Int32 a_arr ->
      Reduce_ops.reduce_max_int32 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Int64 out_arr, Int64 a_arr ->
      Reduce_ops.reduce_max_int64 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let reduce_min (type a b) ~(out : (a, b) t) ~axes ~keepdims (a : (a, b) t) :
    unit =
  let vout = out.view in
  let va = a.view in
  match (out.buffer, a.buffer) with
  | Float64 out_arr, Float64 a_arr ->
      Reduce_ops.reduce_min_float64 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Float32 out_arr, Float32 a_arr ->
      Reduce_ops.reduce_min_float32 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Int32 out_arr, Int32 a_arr ->
      Reduce_ops.reduce_min_int32 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | Int64 out_arr, Int64 a_arr ->
      Reduce_ops.reduce_min_int64 out.context.pool ~out_arr ~a_arr ~va ~vout
        ~axes ~keepdims
  | _ -> Error.invalid ~op:"buffer" ~what:"unsupported dtype" ()

let associative_scan ~out:_ ~axis:_ ~op:_ _ =
  Error.invalid ~op:"associative_scan" ~what:"not implemented" ()

let argmax ~out:_ ~axis:_ ~keepdims:_ _ =
  Error.invalid ~op:"argmax" ~what:"not implemented" ()

let argmin ~out:_ ~axis:_ ~keepdims:_ _ =
  Error.invalid ~op:"argmin" ~what:"not implemented" ()

let sort ~out:_ ~axis:_ ~descending:_ _ =
  Error.invalid ~op:"sort" ~what:"not implemented" ()

let argsort ~out:_ ~axis:_ ~descending:_ _ =
  Error.invalid ~op:"argsort" ~what:"not implemented" ()

let from_host (type a b) ctx (array : (a, b, c_layout) Bigarray.Array1.t) :
    (a, b) t =
  let dtype = Dtype.of_buffer_kind (Array1.kind array) in
  let size = Array1.dim array in
  let shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create shape in
  match dtype with
  | Dtype.Float64 ->
    let unboxed_array = Array.ba_to_unboxed_float_array array in
    { context = ctx; dtype; buffer = Float64 unboxed_array; view }
  | Dtype.Float32 ->
    let unboxed_array = Array.ba_to_unboxed_float32_array array in
    { context = ctx; dtype; buffer = Float32 unboxed_array; view }
  | Dtype.Int64 ->
    let unboxed_array = Array.ba_to_unboxed_int64_array array in
    { context = ctx; dtype; buffer = Int64 unboxed_array; view }
  | Dtype.Int32 ->
    let unboxed_array = Array.ba_to_unboxed_int32_array array in
    { context = ctx; dtype; buffer = Int32 unboxed_array; view }
  | _ -> Error.invalid ~op:"from_host" ~what:"unsupported dtype" ()
let expand x shape = { x with view = View.expand x.view shape }
let reshape x shape = { x with view = View.reshape x.view shape }
let permute x axes = { x with view = View.permute x.view axes }
let shrink x bounds = { x with view = View.shrink x.view bounds }
let flip x axes = { x with view = View.flip x.view axes }

let pad (type a b) (x : (a, b) t) (padding : (int * int) array)
    (fill_value : a) : (a, b) t =
  let in_view = x.view in
  let in_shape = shape in_view in
  let ndim = Array.length in_shape in
  if Array.length padding <> ndim then
    Error.invalid ~op:"pad" ~what:"padding rank mismatch" ();
  let out_shape =
    Array.init ndim (fun i ->
        let before, after = padding.(i) in
        if before < 0 || after < 0 then
          Error.invalid ~op:"pad" ~what:"padding values must be non-negative" ();
        in_shape.(i) + before + after)
  in
  let out_view = View.create (Symbolic_shape.of_ints out_shape) in
  let in_numel = numel in_view in
  let out_numel = numel out_view in
  let in_offset = View.offset in_view in
  let out_offset = View.offset out_view in
  let in_strides = View.strides in_view in
  let out_strides = View.strides out_view in
  match x with
  | { dtype = Dtype.Float64; buffer = Float64 in_arr; context; _ } ->
    let fill_value = Float_u.of_float fill_value in
    let out_arr = Array.make_float64 out_numel in
    for i = 0 to out_numel - 1 do
      Array.unsafe_set out_arr i fill_value
    done;
    Op_pad.pad_float64 in_arr out_arr in_shape padding in_offset out_offset
      in_strides out_strides in_numel;
    { dtype = Dtype.Float64; buffer = Float64 out_arr; view = out_view; context }
  | { dtype = Dtype.Float32; buffer = Float32 in_arr; context; _ } ->
    let fill_value = Float32_u.of_float (Float_u.of_float fill_value) in
    let out_arr = Array.make_float32 out_numel in
    for i = 0 to out_numel - 1 do
      Array.unsafe_set out_arr i fill_value
    done;
    Op_pad.pad_float32 in_arr out_arr in_shape padding in_offset out_offset
      in_strides out_strides in_numel;
    { dtype = Dtype.Float32; buffer = Float32 out_arr; view = out_view; context }
  | { dtype = Dtype.Int8; buffer = Int8 in_arr; context; _ } ->
    let fill_value = Int8_u.of_int fill_value in
    let out_arr = Array.make_int8 out_numel in
    for i = 0 to out_numel - 1 do
      Array.unsafe_set out_arr i fill_value
    done;
    Op_pad.pad_int8 in_arr out_arr in_shape padding in_offset out_offset
      in_strides out_strides in_numel;
    { dtype = Dtype.Int8; buffer = Int8 out_arr; view = out_view; context }
  | { dtype = Dtype.Int16; buffer = Int16 in_arr; context; _ } ->
    let fill_value = Int16_u.of_int fill_value in
    let out_arr = Array.make_int16 out_numel in
    for i = 0 to out_numel - 1 do
      Array.unsafe_set out_arr i fill_value
    done;
    Op_pad.pad_int16 in_arr out_arr in_shape padding in_offset out_offset
      in_strides out_strides in_numel;
    { dtype = Dtype.Int16; buffer = Int16 out_arr; view = out_view; context }
  | { dtype = Dtype.Int32; buffer = Int32 in_arr; context; _ } ->
    let fill_value = Int32_u.of_int32 fill_value in
    let out_arr = Array.make_int32 out_numel in
    for i = 0 to out_numel - 1 do
      Array.unsafe_set out_arr i fill_value
    done;
    Op_pad.pad_int32 in_arr out_arr in_shape padding in_offset out_offset
      in_strides out_strides in_numel;
    { dtype = Dtype.Int32; buffer = Int32 out_arr; view = out_view; context }
  | { dtype = Dtype.Int64; buffer = Int64 in_arr; context; _ } ->
    let fill_value = Int64_u.of_int64 fill_value in
    let out_arr = Array.make_int64 out_numel in
    for i = 0 to out_numel - 1 do
      Array.unsafe_set out_arr i fill_value
    done;
    Op_pad.pad_int64 in_arr out_arr in_shape padding in_offset out_offset
      in_strides out_strides in_numel;
    { dtype = Dtype.Int64; buffer = Int64 out_arr; view = out_view; context }
  | { dtype = Dtype.Bool; buffer = Bool in_arr; context; _ } ->
    let out_arr = Array.make out_numel fill_value in
    Op_pad.pad_bool in_arr out_arr in_shape padding in_offset out_offset
      in_strides out_strides in_numel;
    { dtype = Dtype.Bool; buffer = Bool out_arr; view = out_view; context }
  | _ -> .

let cat (type a b) ~(out : (a, b) t) (xs : (a, b) t list) ~(axis : int) : unit =
  match xs with
  | [] -> Error.invalid ~op:"cat" ~what:"empty input list" ()
  | x0 :: _ ->
    let rank = Array.length (shape x0.view) in
    let axis = if axis < 0 then rank + axis else axis in
    if axis < 0 || axis >= rank then
      Error.axis_out_of_bounds ~op:"cat" ~axis ~ndim:rank ();
    let out_offset = View.offset out.view in
    let out_strides = View.strides out.view in
    (match (x0, out) with
    | { buffer = Float64 _; _ }, { buffer = Float64 out_arr; _ } ->
      let srcs =
        List.map
          (fun x -> match x.buffer with Float64 a -> (a, x.view) | _ -> .)
          xs
      in
      Op_cat.cat_float64 srcs out_arr rank axis out_offset out_strides
    | { buffer = Float32 _; _ }, { buffer = Float32 out_arr; _ } ->
      let srcs =
        List.map
          (fun x -> match x.buffer with Float32 a -> (a, x.view) | _ -> .)
          xs
      in
      Op_cat.cat_float32 srcs out_arr rank axis out_offset out_strides
    | { buffer = Int8 _; _ }, { buffer = Int8 out_arr; _ } ->
      let srcs =
        List.map
          (fun x -> match x.buffer with Int8 a -> (a, x.view) | _ -> .)
          xs
      in
      Op_cat.cat_int8 srcs out_arr rank axis out_offset out_strides
    | { buffer = Int16 _; _ }, { buffer = Int16 out_arr; _ } ->
      let srcs =
        List.map
          (fun x -> match x.buffer with Int16 a -> (a, x.view) | _ -> .)
          xs
      in
      Op_cat.cat_int16 srcs out_arr rank axis out_offset out_strides
    | { buffer = Int32 _; _ }, { buffer = Int32 out_arr; _ } ->
      let srcs =
        List.map
          (fun x -> match x.buffer with Int32 a -> (a, x.view) | _ -> .)
          xs
      in
      Op_cat.cat_int32 srcs out_arr rank axis out_offset out_strides
    | { buffer = Int64 _; _ }, { buffer = Int64 out_arr; _ } ->
      let srcs =
        List.map
          (fun x -> match x.buffer with Int64 a -> (a, x.view) | _ -> .)
          xs
      in
      Op_cat.cat_int64 srcs out_arr rank axis out_offset out_strides
    | { buffer = Bool _; _ }, { buffer = Bool out_arr; _ } ->
      let srcs =
        List.map
          (fun x -> match x.buffer with Bool a -> (a, x.view) | _ -> .)
          xs
      in
      Op_cat.cat_bool srcs out_arr rank axis out_offset out_strides
    | _ -> .)
let cast ~out:_ _ = Error.invalid ~op:"cast" ~what:"not implemented" ()

let contiguous _ =
  Error.invalid ~op:"contiguous" ~what:"not implemented" ()

let copy _ = Error.invalid ~op:"copy" ~what:"not implemented" ()
let assign _ _ = Error.invalid ~op:"assign" ~what:"not implemented" ()

let threefry ~out:_ _ _ = Error.invalid ~op:"threefry" ~what:"not implemented" ()
let gather ~out:_ _ _ ~axis:_ = Error.invalid ~op:"gather" ~what:"not implemented" ()

let scatter ?mode:_ ?unique_indices:_ _ ~indices:_ ~updates:_ ~axis:_ =
  Error.invalid ~op:"scatter" ~what:"not implemented" ()

let unfold ?out:_ _ ~kernel_size:_ ~stride:_ ~dilation:_ ~padding:_ =
  Error.invalid ~op:"unfold" ~what:"not implemented" ()

let fold ?out:_ _ ~output_size:_ ~kernel_size:_ ~stride:_ ~dilation:_
    ~padding:_ =
  Error.invalid ~op:"fold" ~what:"not implemented" ()

let matmul (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit
    =
  let va = a.view and vb = b.view and vout = out.view in
  let m = (shape vout).(0) in
  let nd_out = Array.length (shape vout) in
  let batch_shape = Array.sub (shape vout) 0 (Stdlib.max 0 (nd_out - 2)) in
  let batch_sz =
    if Array.length batch_shape = 0 then 1 else Shape.numel batch_shape
  in
  let total_units = batch_sz * m in
  match (out.buffer, a.buffer, b.buffer) with
  | Float64 c, Float64 a, Float64 b ->
      if
        View.is_c_contiguous va && View.is_c_contiguous vb
        && Array.length (shape va) = 2
        && Array.length (shape vb) = 2
      then
        let n = (shape vout).(nd_out - 1) in
        let k = (shape va).(Array.length (shape va) - 1) in
        Op_matmul.Gemm_f64.gemm ~pool:out.context.pool a b c ~m ~n ~k
          ~a_off:(View.offset va) ~b_off:(View.offset vb)
          ~c_off:(View.offset vout) ~ldc:n ()
      else
        Parallel.parallel_for out.context.pool 0 (total_units - 1) (fun s e ->
            Op_matmul.matmul_float64_slow a b c va vb vout s e)
  | Float32 c, Float32 a, Float32 b ->
      if
        View.is_c_contiguous va && View.is_c_contiguous vb
        && Array.length (shape va) = 2
        && Array.length (shape vb) = 2
      then
        let n = (shape vout).(nd_out - 1) in
        let k = (shape va).(Array.length (shape va) - 1) in
        Op_matmul.Gemm_f32.gemm ~pool:out.context.pool a b c ~m ~n ~k
          ~a_off:(View.offset va) ~b_off:(View.offset vb)
          ~c_off:(View.offset vout) ~ldc:n ()
      else
        Parallel.parallel_for out.context.pool 0 (total_units - 1) (fun s e ->
            Op_matmul.matmul_float32_slow a b c va vb vout s e)
  | Int64 c, Int64 a, Int64 b ->
      if
        View.is_c_contiguous va && View.is_c_contiguous vb
        && View.offset va = 0
        && View.offset vb = 0
        && Array.length (shape va) = 2
        && Array.length (shape vb) = 2
      then
        Parallel.parallel_for out.context.pool 0 (m - 1) (fun s e ->
            Op_matmul.matmul_int64_fast a b c va vb vout s e)
      else
        Parallel.parallel_for out.context.pool 0 (total_units - 1) (fun s e ->
            Op_matmul.matmul_int64_slow a b c va vb vout s e)
  | Int32 c, Int32 a, Int32 b ->
      if
        View.is_c_contiguous va && View.is_c_contiguous vb
        && View.offset va = 0
        && View.offset vb = 0
        && Array.length (shape va) = 2
        && Array.length (shape vb) = 2
      then
        Parallel.parallel_for out.context.pool 0 (m - 1) (fun s e ->
            Op_matmul.matmul_int32_fast a b c va vb vout s e)
      else
        Parallel.parallel_for out.context.pool 0 (total_units - 1) (fun s e ->
            Op_matmul.matmul_int32_slow a b c va vb vout s e)
  | _ ->
      Error.invalid ~op:"matmul" ~what:"not implemented for small unboxed ints" ()

let fft ?out:_ _ ~axes:_ =
  Error.invalid ~op:"fft" ~what:"not implemented" ()

let ifft ?out:_ _ ~axes:_ =
  Error.invalid ~op:"ifft" ~what:"not implemented" ()

let rfft ?out:_ _ ~dtype:_ ~axes:_ =
  Error.invalid ~op:"rfft" ~what:"not implemented" ()

let irfft ?out:_ ?s:_ _ ~dtype:_ ~axes:_ =
  Error.invalid ~op:"irfft" ~what:"not implemented" ()

let cholesky ~upper:_ _ =
  Error.invalid ~op:"cholesky" ~what:"not implemented" ()

let qr ~reduced:_ _ = Error.invalid ~op:"qr" ~what:"not implemented" ()

let svd ~full_matrices:_ _ =
  Error.invalid ~op:"svd" ~what:"not implemented" ()

let eig ~vectors:_ _ = Error.invalid ~op:"eig" ~what:"not implemented" ()

let eigh ~vectors:_ _ =
  Error.invalid ~op:"eigh" ~what:"not implemented" ()

let triangular_solve ~upper:_ ~transpose:_ ~unit_diag:_ _ _ =
  Error.invalid ~op:"triangular_solve" ~what:"not implemented" ()
