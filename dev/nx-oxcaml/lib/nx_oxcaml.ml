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

(* [to_host] returns a Bigarray, but Bigarrays cannot point to OCaml heap
   memory. Unboxed arrays are GC-managed, so we cannot create a Bigarray view of
   them without risking memory safety. Use [data_array] to access the raw
   buffer. *)
let to_host _ =
  failwith
    "Nx_oxcaml.to_host is not supported. Bigarrays cannot point to OCaml heap \
     memory. Use Nx_oxcaml.data_array instead."

let data_array t = t.buffer

let op_buffer (type a b) context (dtype : (a, b) Dtype.t) (size : int) :
    (a, b) t =
  let sym_shape = Symbolic_shape.of_ints [| size |] in
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let of_float64_multidim context (arr : float# array) (shape : int array) :
    (float, Dtype.float64_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  let op_reshape x shape = { x with view = View.reshape x.view shape } in
  op_reshape
    { dtype = Dtype.Float64; buffer = Float64 arr; view; context }
    (Symbolic_shape.of_ints shape)

let of_float32_multidim context (arr : float32# array) (shape : int array) :
    (float, Dtype.float32_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  let op_reshape x shape = { x with view = View.reshape x.view shape } in
  op_reshape
    { dtype = Dtype.Float32; buffer = Float32 arr; view; context }
    (Symbolic_shape.of_ints shape)

let of_float64 context (arr : float# array) : (float, Dtype.float64_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  { dtype = Dtype.Float64; buffer = Float64 arr; view; context }

let of_float32 context (arr : float32# array) : (float, Dtype.float32_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  { dtype = Dtype.Float32; buffer = Float32 arr; view; context }

let of_int8 context (arr : int8# array) : (int, Dtype.int8_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  { dtype = Dtype.Int8; buffer = Int8 arr; view; context }

let of_int16 context (arr : int16# array) : (int, Dtype.int16_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  { dtype = Dtype.Int16; buffer = Int16 arr; view; context }

let of_int32 context (arr : int32# array) : (int32, Dtype.int32_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  { dtype = Dtype.Int32; buffer = Int32 arr; view; context }

let of_int64 context (arr : int64# array) : (int64, Dtype.int64_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  { dtype = Dtype.Int64; buffer = Int64 arr; view; context }

let of_bool context (arr : bool array) : (bool, Dtype.bool_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  { dtype = Dtype.Bool; buffer = Bool arr; view; context }

let op_add (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_sub (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_mul (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_idiv (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_fdiv (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_mod (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_pow (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
      Error.invalid ~op:"op_cmpow" ~what:"not implemented for unboxed ints" ()

let op_cmpeq (type a b) ~(out : (bool, Nx_buffer.bool_elt) t) (a : (a, b) t)
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_cmpne (type a b) ~(out : (bool, Nx_buffer.bool_elt) t) (a : (a, b) t)
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_cmplt (type a b) ~(out : (bool, Nx_buffer.bool_elt) t) (a : (a, b) t)
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_cmple (type a b) ~(out : (bool, Nx_buffer.bool_elt) t) (a : (a, b) t)
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_max (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_max" ~what:"unsupported dtype" ()

let op_min (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_min" ~what:"unsupported dtype" ()

let op_xor (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_or" ~what:"not implemented for unboxed ints" ()

let op_or (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_or" ~what:"not implemented for unboxed ints" ()

let op_and (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_and" ~what:"not implemented for unboxed ints" ()

let op_neg (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_recip (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_abs (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_sqrt (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
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
      Error.invalid ~op:"op_sqrt " ~what:"not implemented for unboxed ints" ()

let op_exp (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_exp " ~what:"not implemented for unboxed ints" ()

let op_log (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_log " ~what:"not implemented for unboxed ints" ()

let op_sin (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_sin " ~what:"not implemented for unboxed ints" ()

let op_cos (type a b) ~(out : (a, b) t) (a : (a, b) t) : unit =
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
  | _ -> Error.invalid ~op:"op_cos " ~what:"not implemented for unboxed ints" ()

let op_where (type a b) ~(out : (a, b) t) (cond : (bool, Nx_buffer.bool_elt) t)
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
  | _ -> Error.invalid ~op:"op_where " ~what:"not implemented for this dtype" ()

let op_reduce_sum (type a b) ~(out : (a, b) t) ~axes ~keepdims (a : (a, b) t) :
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_reduce_prod (type a b) ~(out : (a, b) t) ~axes ~keepdims (a : (a, b) t) :
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_reduce_max (type a b) ~(out : (a, b) t) ~axes ~keepdims (a : (a, b) t) :
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_reduce_min (type a b) ~(out : (a, b) t) ~axes ~keepdims (a : (a, b) t) :
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
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let op_associative_scan ~axis:_ ~op:_ _ =
  Error.invalid ~op:"op_associative_scan" ~what:"not implemented" ()
  
  
  let op_const_scalar  _ _ =
  Error.invalid ~op:"op_const_scalar" ~what:"not implemented" ()

  
let from_host (type a b) ctx (array : (a, b, c_layout) Bigarray.Array1.t) : (a, b) t =
  let dtype = Dtype.of_buffer_kind (Array1.kind array) in
  let size = Array1.dim array in
  (* Create a view for the 1D array *)
  let shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create shape in
  match dtype with
  | Dtype.Float64 ->
      let unboxed_array = (Array.ba_to_unboxed_float_array array) in
      { context = ctx; dtype; buffer = Float64 unboxed_array; view }
  | Dtype.Float32 ->
    let unboxed_array = (Array.ba_to_unboxed_float32_array array) in
    { context = ctx; dtype; buffer = Float32 unboxed_array; view }
  | Dtype.Int64 ->
    let unboxed_array = (Array.ba_to_unboxed_int64_array array) in
    { context = ctx; dtype; buffer = Int64 unboxed_array; view }
  | Dtype.Int32 ->
    let unboxed_array = (Array.ba_to_unboxed_int32_array array) in
    { context = ctx; dtype; buffer = Int32 unboxed_array; view }
  | _ -> Error.invalid ~op:"from_host" ~what:"unsupported dtype" ()

let op_expand x shape = { x with view = View.expand x.view shape }
let op_reshape x shape = { x with view = View.reshape x.view shape }
let op_permute x axes = { x with view = View.permute x.view axes }
let op_shrink x bounds = { x with view = View.shrink x.view bounds }
let op_flip x axes = { x with view = View.flip x.view axes }
let op_pad (type a b) (x : (a, b) t) (padding : (int * int) array)
    (fill_value : a) : (a, b) t =
  let in_view = x.view in
  let in_shape = shape in_view in
  let ndim = Array.length in_shape in
  if Array.length padding <> ndim then
    Error.invalid ~op:"op_pad" ~what:"padding rank mismatch" ();
  let out_shape =
    Array.init ndim (fun i ->
        let before, after = padding.(i) in
        if before < 0 || after < 0 then
          Error.invalid ~op:"op_pad" ~what:"padding values must be non-negative"
            ();
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
let op_cat (type a b) (xs : (a, b) t list) (axis : int) : (a, b) t =
  match xs with
  | [] -> Error.invalid ~op:"op_cat" ~what:"empty input list" ()
  | x0 :: _ ->
      let rank = Array.length (shape x0.view) in
      let axis = if axis < 0 then rank + axis else axis in
      if axis < 0 || axis >= rank then
        Error.axis_out_of_bounds ~op:"op_cat" ~axis ~ndim:rank ();

      let out_shape = Array.copy (shape x0.view) in
      out_shape.(axis) <- 0;
      List.iter
        (fun x ->
          let s = shape x.view in
          if Array.length s <> rank then
            Error.invalid ~op:"op_cat" ~what:"rank mismatch" ();
          for i = 0 to rank - 1 do
            if i <> axis && s.(i) <> (shape x0.view).(i) then
              Error.invalid ~op:"op_cat" ~what:"shape mismatch" ()
          done;
          out_shape.(axis) <- out_shape.(axis) + s.(axis))
        xs;

      let out_numel = Shape.numel out_shape in
      let out =
        let t = op_buffer x0.context x0.dtype out_numel in
        { t with view = View.reshape t.view (Symbolic_shape.of_ints out_shape) }
      in

      let out_offset = View.offset out.view in
      let out_strides = View.strides out.view in

      (match (x0, out) with
      | { buffer = Float64 _; _ }, { buffer = Float64 out_arr; _ } ->
          let srcs =
            List.map
              (fun x ->
                match x.buffer with Float64 a -> (a, x.view) | _ -> .)
              xs
          in
          Op_cat.cat_float64 srcs out_arr rank axis out_offset out_strides
      | { buffer = Float32 _; _ }, { buffer = Float32 out_arr; _ } ->
          let srcs =
            List.map
              (fun x ->
                match x.buffer with Float32 a -> (a, x.view) | _ -> .)
              xs
          in
          Op_cat.cat_float32 srcs out_arr rank axis out_offset out_strides
      | { buffer = Int8 _; _ }, { buffer = Int8 out_arr; _ } ->
          let srcs =
            List.map (fun x -> match x.buffer with Int8 a -> (a, x.view) | _ -> .) xs
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
            List.map (fun x -> match x.buffer with Bool a -> (a, x.view) | _ -> .) xs
          in
          Op_cat.cat_bool srcs out_arr rank axis out_offset out_strides
      | _ -> .);
      out

let op_cast (type a b c d) (x : (a, b) t) (target_dtype : (c, d) Dtype.t) :
    (c, d) t =
  let in_view = x.view in
  let in_shape = shape in_view in
  let n = numel in_view in
  let out =
    let t = op_buffer x.context target_dtype n in
    { t with view = View.reshape t.view (Symbolic_shape.of_ints in_shape) }
  in
  let in_offset = View.offset in_view in
  let in_strides = View.strides in_view in
  let out_offset = View.offset out.view in
  let out_strides = View.strides out.view in
  match (x.buffer, out.buffer) with
  | Float64 src, Float64 dst ->
      Op_cast.cast_float64_float64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float64 src, Float32 dst ->
      Op_cast.cast_float64_float32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float64 src, Int8 dst ->
      Op_cast.cast_float64_int8 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float64 src, Int16 dst ->
      Op_cast.cast_float64_int16 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float64 src, Int32 dst ->
      Op_cast.cast_float64_int32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float64 src, Int64 dst ->
      Op_cast.cast_float64_int64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float64 src, Bool dst ->
      Op_cast.cast_float64_bool src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float32 src, Float64 dst ->
      Op_cast.cast_float32_float64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float32 src, Float32 dst ->
      Op_cast.cast_float32_float32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float32 src, Int8 dst ->
      Op_cast.cast_float32_int8 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float32 src, Int16 dst ->
      Op_cast.cast_float32_int16 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float32 src, Int32 dst ->
      Op_cast.cast_float32_int32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float32 src, Int64 dst ->
      Op_cast.cast_float32_int64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float32 src, Bool dst ->
      Op_cast.cast_float32_bool src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int8 src, Float64 dst ->
      Op_cast.cast_int8_float64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int8 src, Float32 dst ->
      Op_cast.cast_int8_float32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int8 src, Int8 dst ->
      Op_cast.cast_int8_int8 src dst n in_shape in_offset in_strides out_offset
        out_strides;
      out
  | Int8 src, Int16 dst ->
      Op_cast.cast_int8_int16 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int8 src, Int32 dst ->
      Op_cast.cast_int8_int32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int8 src, Int64 dst ->
      Op_cast.cast_int8_int64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int8 src, Bool dst ->
      Op_cast.cast_int8_bool src dst n in_shape in_offset in_strides out_offset
        out_strides;
      out
  | Int16 src, Float64 dst ->
      Op_cast.cast_int16_float64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int16 src, Float32 dst ->
      Op_cast.cast_int16_float32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int16 src, Int8 dst ->
      Op_cast.cast_int16_int8 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int16 src, Int16 dst ->
      Op_cast.cast_int16_int16 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int16 src, Int32 dst ->
      Op_cast.cast_int16_int32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int16 src, Int64 dst ->
      Op_cast.cast_int16_int64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int16 src, Bool dst ->
      Op_cast.cast_int16_bool src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int32 src, Float64 dst ->
      Op_cast.cast_int32_float64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int32 src, Float32 dst ->
      Op_cast.cast_int32_float32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int32 src, Int8 dst ->
      Op_cast.cast_int32_int8 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int32 src, Int16 dst ->
      Op_cast.cast_int32_int16 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int32 src, Int32 dst ->
      Op_cast.cast_int32_int32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int32 src, Int64 dst ->
      Op_cast.cast_int32_int64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int32 src, Bool dst ->
      Op_cast.cast_int32_bool src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int64 src, Float64 dst ->
      Op_cast.cast_int64_float64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int64 src, Float32 dst ->
      Op_cast.cast_int64_float32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int64 src, Int8 dst ->
      Op_cast.cast_int64_int8 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int64 src, Int16 dst ->
      Op_cast.cast_int64_int16 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int64 src, Int32 dst ->
      Op_cast.cast_int64_int32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int64 src, Int64 dst ->
      Op_cast.cast_int64_int64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int64 src, Bool dst ->
      Op_cast.cast_int64_bool src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Bool src, Float64 dst ->
      Op_cast.cast_bool_float64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Bool src, Float32 dst ->
      Op_cast.cast_bool_float32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Bool src, Int8 dst ->
      Op_cast.cast_bool_int8 src dst n in_shape in_offset in_strides out_offset
        out_strides;
      out
  | Bool src, Int16 dst ->
      Op_cast.cast_bool_int16 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Bool src, Int32 dst ->
      Op_cast.cast_bool_int32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Bool src, Int64 dst ->
      Op_cast.cast_bool_int64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Bool src, Bool dst ->
      Op_cast.cast_bool_bool src dst n in_shape in_offset in_strides out_offset
        out_strides;
      out
  | _ -> .

let materialize_contiguous (type a b) (x : (a, b) t) : (a, b) t =
  let in_view = x.view in
  let in_shape = shape in_view in
  let n = numel in_view in
  let out =
    let t = op_buffer x.context x.dtype n in
    { t with view = View.reshape t.view (Symbolic_shape.of_ints in_shape) }
  in
  let in_offset = View.offset in_view in
  let in_strides = View.strides in_view in
  let out_offset = View.offset out.view in
  let out_strides = View.strides out.view in
  match (x.buffer, out.buffer) with
  | Float64 src, Float64 dst ->
      Op_contiguous.materialize_float64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Float32 src, Float32 dst ->
      Op_contiguous.materialize_float32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int8 src, Int8 dst ->
      Op_contiguous.materialize_int8 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int16 src, Int16 dst ->
      Op_contiguous.materialize_int16 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int32 src, Int32 dst ->
      Op_contiguous.materialize_int32 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Int64 src, Int64 dst ->
      Op_contiguous.materialize_int64 src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | Bool src, Bool dst ->
      Op_contiguous.materialize_bool src dst n in_shape in_offset in_strides
        out_offset out_strides;
      out
  | _ -> .

let op_contiguous (type a b) (x : (a, b) t) : (a, b) t =
  if View.is_c_contiguous x.view && View.offset x.view = 0 then x
  else materialize_contiguous x

let op_copy (type a b) (x : (a, b) t) : (a, b) t = materialize_contiguous x

let op_assign (type a b) (dst : (a, b) t) (src : (a, b) t) : unit =
  let dst_shape = shape dst.view in
  let src_shape = shape src.view in
  if dst_shape <> src_shape then
    Error.invalid ~op:"op_assign" ~what:"shape mismatch" ();
  let n = numel dst.view in
  let dst_offset = View.offset dst.view in
  let dst_strides = View.strides dst.view in
  let src_offset = View.offset src.view in
  let src_strides = View.strides src.view in
  match (dst.buffer, src.buffer) with
  | Float64 dst_arr, Float64 src_arr ->
      Op_assign.assign_float64 dst_arr src_arr n dst_shape dst_offset dst_strides
        src_offset src_strides
  | Float32 dst_arr, Float32 src_arr ->
      Op_assign.assign_float32 dst_arr src_arr n dst_shape dst_offset dst_strides
        src_offset src_strides
  | Int8 dst_arr, Int8 src_arr ->
      Op_assign.assign_int8 dst_arr src_arr n dst_shape dst_offset dst_strides
        src_offset src_strides
  | Int16 dst_arr, Int16 src_arr ->
      Op_assign.assign_int16 dst_arr src_arr n dst_shape dst_offset dst_strides
        src_offset src_strides
  | Int32 dst_arr, Int32 src_arr ->
      Op_assign.assign_int32 dst_arr src_arr n dst_shape dst_offset dst_strides
        src_offset src_strides
  | Int64 dst_arr, Int64 src_arr ->
      Op_assign.assign_int64 dst_arr src_arr n dst_shape dst_offset dst_strides
        src_offset src_strides
  | Bool dst_arr, Bool src_arr ->
      Op_assign.assign_bool dst_arr src_arr n dst_shape dst_offset dst_strides
        src_offset src_strides
  | _ -> .

let op_as_strided (type a b) (x : (a, b) t) shape_sym strides offset : (a, b) t
    =
  let shape_arr =
    match Symbolic_shape.eval shape_sym with
    | Some s -> s
    | None ->
        Error.invalid ~op:"op_as_strided" ~what:"shape"
          ~reason:"symbolic shapes are not supported in nx_oxcaml.as_strided"
          ()
  in
  if Array.length strides <> Array.length shape_arr then
    Error.invalid ~op:"op_as_strided" ~what:"strides rank mismatch" ();
  let base_offset = View.offset x.view + offset in
  let buffer_len =
    match x.buffer with
    | Float64 a -> Array.length a
    | Float32 a -> Array.length a
    | Int8 a -> Array.length a
    | Int16 a -> Array.length a
    | Int32 a -> Array.length a
    | Int64 a -> Array.length a
    | Bool a -> Array.length a
  in
  if Array.exists (( = ) 0) shape_arr then
    { x with view = View.create ~offset:base_offset ~strides shape_sym }
  else (
    let min_delta = ref 0 in
    let max_delta = ref 0 in
    for i = 0 to Array.length shape_arr - 1 do
      let span = (shape_arr.(i) - 1) * strides.(i) in
      if span >= 0 then max_delta := !max_delta + span
      else min_delta := !min_delta + span
    done;
    let min_idx = base_offset + !min_delta in
    let max_idx = base_offset + !max_delta in
    if min_idx < 0 || max_idx >= buffer_len then
      Error.invalid ~op:"op_as_strided" ~what:"out-of-bounds view" ();
    { x with view = View.create ~offset:base_offset ~strides shape_sym })

let op_threefry _ _ = Error.invalid ~op:"op_threefry" ~what:"not implemented" ()
let op_gather (type a b) (data : (a, b) t)
    (indices : (int32, Dtype.int32_elt) t) axis : (a, b) t =
  let dshape = shape data.view in
  let ishape = shape indices.view in
  if Array.length dshape <> Array.length ishape then
    Error.invalid ~op:"op_gather" ~what:"rank mismatch" ();
  let rank = Array.length dshape in
  let axis = if axis < 0 then rank + axis else axis in
  if axis < 0 || axis >= rank then
    Error.axis_out_of_bounds ~op:"op_gather" ~axis ~ndim:rank ();
  let n = numel indices.view in
  let out =
    let t = op_buffer data.context data.dtype n in
    { t with view = View.reshape t.view (Symbolic_shape.of_ints ishape) }
  in
  let data_offset = View.offset data.view in
  let data_strides = View.strides data.view in
  let idx_offset = View.offset indices.view in
  let idx_strides = View.strides indices.view in
  let out_offset = View.offset out.view in
  let out_strides = View.strides out.view in
  let idx_arr = match indices.buffer with Int32 a -> a | _ -> . in
  match (data.buffer, out.buffer) with
  | Float64 src, Float64 dst ->
      Op_gather.gather_float64 src dst n ishape dshape axis idx_arr data_offset
        data_strides idx_offset idx_strides out_offset out_strides;
      out
  | Float32 src, Float32 dst ->
      Op_gather.gather_float32 src dst n ishape dshape axis idx_arr data_offset
        data_strides idx_offset idx_strides out_offset out_strides;
      out
  | Int8 src, Int8 dst ->
      Op_gather.gather_int8 src dst n ishape dshape axis idx_arr data_offset
        data_strides idx_offset idx_strides out_offset out_strides;
      out
  | Int16 src, Int16 dst ->
      Op_gather.gather_int16 src dst n ishape dshape axis idx_arr data_offset
        data_strides idx_offset idx_strides out_offset out_strides;
      out
  | Int32 src, Int32 dst ->
      Op_gather.gather_int32 src dst n ishape dshape axis idx_arr data_offset
        data_strides idx_offset idx_strides out_offset out_strides;
      out
  | Int64 src, Int64 dst ->
      Op_gather.gather_int64 src dst n ishape dshape axis idx_arr data_offset
        data_strides idx_offset idx_strides out_offset out_strides;
      out
  | Bool src, Bool dst ->
      Op_gather.gather_bool src dst n ishape dshape axis idx_arr data_offset
        data_strides idx_offset idx_strides out_offset out_strides;
      out
  | _ -> .

let op_scatter ?(mode = `Set) ?unique_indices:_ (type a b)
    (data_template : (a, b) t) (indices : (int32, Dtype.int32_elt) t)
    (updates : (a, b) t) axis : (a, b) t =
  let tshape = shape data_template.view in
  let ishape = shape indices.view in
  let ushape = shape updates.view in
  if Array.length tshape <> Array.length ishape then
    Error.invalid ~op:"op_scatter" ~what:"rank mismatch" ();
  if ishape <> ushape then
    Error.invalid ~op:"op_scatter" ~what:"indices/updates shape mismatch" ();
  let rank = Array.length tshape in
  let axis = if axis < 0 then rank + axis else axis in
  if axis < 0 || axis >= rank then
    Error.axis_out_of_bounds ~op:"op_scatter" ~axis ~ndim:rank ();
  let out =
    match mode with
    | `Set -> op_copy data_template
    | `Add ->
        let n = numel data_template.view in
        let t = op_buffer data_template.context data_template.dtype n in
        { t with view = View.reshape t.view (Symbolic_shape.of_ints tshape) }
  in
  let n = numel indices.view in
  let idx_offset = View.offset indices.view in
  let idx_strides = View.strides indices.view in
  let upd_offset = View.offset updates.view in
  let upd_strides = View.strides updates.view in
  let out_offset = View.offset out.view in
  let out_strides = View.strides out.view in
  let idx_arr = match indices.buffer with Int32 a -> a | _ -> . in
  match (updates.buffer, out.buffer) with
  | Float64 src_arr, Float64 out_arr ->
      Op_scatter.scatter_float64 mode src_arr out_arr n ishape tshape axis
        idx_arr upd_offset upd_strides idx_offset idx_strides out_offset
        out_strides;
      out
  | Float32 src_arr, Float32 out_arr ->
      Op_scatter.scatter_float32 mode src_arr out_arr n ishape tshape axis
        idx_arr upd_offset upd_strides idx_offset idx_strides out_offset
        out_strides;
      out
  | Int8 src_arr, Int8 out_arr ->
      Op_scatter.scatter_int8 mode src_arr out_arr n ishape tshape axis idx_arr
        upd_offset upd_strides idx_offset idx_strides out_offset out_strides;
      out
  | Int16 src_arr, Int16 out_arr ->
      Op_scatter.scatter_int16 mode src_arr out_arr n ishape tshape axis idx_arr
        upd_offset upd_strides idx_offset idx_strides out_offset out_strides;
      out
  | Int32 src_arr, Int32 out_arr ->
      Op_scatter.scatter_int32 mode src_arr out_arr n ishape tshape axis idx_arr
        upd_offset upd_strides idx_offset idx_strides out_offset out_strides;
      out
  | Int64 src_arr, Int64 out_arr ->
      Op_scatter.scatter_int64 mode src_arr out_arr n ishape tshape axis idx_arr
        upd_offset upd_strides idx_offset idx_strides out_offset out_strides;
      out
  | Bool src_arr, Bool out_arr ->
      Op_scatter.scatter_bool mode src_arr out_arr n ishape tshape axis idx_arr
        upd_offset upd_strides idx_offset idx_strides out_offset out_strides;
      out
  | _ -> .

let op_unfold :
    type a b.
    ?out:(a, b) t ->
    (a, b) t ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    (a, b) t =
 fun ?out (x : (a, b) t) ~kernel_size ~stride ~dilation ~padding ->
  let in_view = x.view in
  let in_shape = shape in_view in
  if Array.length in_shape < 3 then
    Error.invalid ~op:"op_unfold" ~what:"input rank"
      ~reason:"expected input shape [N, C, ...spatial_dims]" ();

  let spatial_ndim = Array.length in_shape - 2 in
  if
    not
      (Array.length kernel_size = spatial_ndim
      && Array.length stride = spatial_ndim
      && Array.length dilation = spatial_ndim
      && Array.length padding = spatial_ndim)
  then
    Error.invalid ~op:"op_unfold" ~what:"parameter lengths"
      ~reason:"kernel_size/stride/dilation/padding must match spatial rank" ();

  let n = in_shape.(0) in
  let channels = in_shape.(1) in
  let input_spatial = Array.sub in_shape 2 spatial_ndim in

  let kernel_elems = ref 1 in
  for i = 0 to spatial_ndim - 1 do
    if kernel_size.(i) <= 0 then
      Error.invalid ~op:"op_unfold" ~what:"kernel_size"
        ~reason:"all kernel dimensions must be positive" ();
    if stride.(i) <= 0 then
      Error.invalid ~op:"op_unfold" ~what:"stride"
        ~reason:"all stride dimensions must be positive" ();
    if dilation.(i) <= 0 then
      Error.invalid ~op:"op_unfold" ~what:"dilation"
        ~reason:"all dilation dimensions must be positive" ();
    let pad_before, pad_after = padding.(i) in
    if pad_before < 0 || pad_after < 0 then
      Error.invalid ~op:"op_unfold" ~what:"padding"
        ~reason:"padding must be non-negative" ();
    kernel_elems := !kernel_elems * kernel_size.(i)
  done;

  let out_spatial = Array.make spatial_ndim 0 in
  for i = 0 to spatial_ndim - 1 do
    let pad_before, pad_after = padding.(i) in
    let padded = input_spatial.(i) + pad_before + pad_after in
    let kernel_extent = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
    let diff = padded - kernel_extent in
    if diff < 0 then
      Error.invalid ~op:"op_unfold"
        ~what:"kernel size larger than padded input" ();
    out_spatial.(i) <- (diff / stride.(i)) + 1
  done;

  let num_blocks = Shape.numel out_spatial in
  let out_shape = [| n; channels * !kernel_elems; num_blocks |] in
  let out_numel = Shape.numel out_shape in
  let out_t : (a, b) t =
    match out with
    | Some out_t ->
        if shape out_t.view <> out_shape then
          Error.invalid ~op:"op_unfold" ~what:"out shape mismatch" ();
        out_t
    | None ->
        let out_t : (a, b) t = op_buffer x.context x.dtype out_numel in
        { out_t with view = View.reshape out_t.view (Symbolic_shape.of_ints out_shape) }
  in

  let in_offset = View.offset in_view in
  let in_strides = View.strides in_view in
  let out_view = out_t.view in
  let out_offset = View.offset out_view in
  let out_strides = View.strides out_view in
  let kernel_elems = !kernel_elems in
  match (x.buffer, out_t.buffer) with
  | Float64 in_arr, Float64 out_arr ->
      Op_unfold.unfold_float64 in_arr out_arr ~n ~channels ~input_spatial
        ~kernel_elems ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride
        ~dilation ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Float32 in_arr, Float32 out_arr ->
      Op_unfold.unfold_float32 in_arr out_arr ~n ~channels ~input_spatial
        ~kernel_elems ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride
        ~dilation ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Int8 in_arr, Int8 out_arr ->
      Op_unfold.unfold_int8 in_arr out_arr ~n ~channels ~input_spatial
        ~kernel_elems ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride
        ~dilation ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Int16 in_arr, Int16 out_arr ->
      Op_unfold.unfold_int16 in_arr out_arr ~n ~channels ~input_spatial
        ~kernel_elems ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride
        ~dilation ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Int32 in_arr, Int32 out_arr ->
      Op_unfold.unfold_int32 in_arr out_arr ~n ~channels ~input_spatial
        ~kernel_elems ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride
        ~dilation ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Int64 in_arr, Int64 out_arr ->
      Op_unfold.unfold_int64 in_arr out_arr ~n ~channels ~input_spatial
        ~kernel_elems ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride
        ~dilation ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Bool in_arr, Bool out_arr ->
      Op_unfold.unfold_bool in_arr out_arr ~n ~channels ~input_spatial
        ~kernel_elems ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride
        ~dilation ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | _ -> .

let op_fold :
    type a b.
    ?out:(a, b) t ->
    (a, b) t ->
    output_size:int array ->
    kernel_size:int array ->
    stride:int array ->
    dilation:int array ->
    padding:(int * int) array ->
    (a, b) t =
 fun ?out (x : (a, b) t) ~output_size ~kernel_size ~stride ~dilation ~padding ->
  let in_view = x.view in
  let in_shape = shape in_view in
  if Array.length in_shape <> 3 then
    Error.invalid ~op:"op_fold" ~what:"input rank"
      ~reason:"expected input shape [N, C * prod(kernel_size), L]" ();

  let spatial_ndim = Array.length output_size in
  if spatial_ndim = 0 then
    Error.invalid ~op:"op_fold" ~what:"output_size"
      ~reason:"must contain at least one spatial dimension" ();
  if
    not
      (Array.length kernel_size = spatial_ndim
      && Array.length stride = spatial_ndim
      && Array.length dilation = spatial_ndim
      && Array.length padding = spatial_ndim)
  then
    Error.invalid ~op:"op_fold" ~what:"parameter lengths"
      ~reason:"output_size/kernel_size/stride/dilation/padding must match" ();

  let n = in_shape.(0) in
  let c_times_k = in_shape.(1) in
  let num_blocks = in_shape.(2) in

  let kernel_elems = ref 1 in
  for i = 0 to spatial_ndim - 1 do
    if output_size.(i) <= 0 then
      Error.invalid ~op:"op_fold" ~what:"output_size"
        ~reason:"all output dimensions must be positive" ();
    if kernel_size.(i) <= 0 then
      Error.invalid ~op:"op_fold" ~what:"kernel_size"
        ~reason:"all kernel dimensions must be positive" ();
    if stride.(i) <= 0 then
      Error.invalid ~op:"op_fold" ~what:"stride"
        ~reason:"all stride dimensions must be positive" ();
    if dilation.(i) <= 0 then
      Error.invalid ~op:"op_fold" ~what:"dilation"
        ~reason:"all dilation dimensions must be positive" ();
    let pad_before, pad_after = padding.(i) in
    if pad_before < 0 || pad_after < 0 then
      Error.invalid ~op:"op_fold" ~what:"padding"
        ~reason:"padding must be non-negative" ();
    kernel_elems := !kernel_elems * kernel_size.(i)
  done;

  if c_times_k mod !kernel_elems <> 0 then
    Error.invalid ~op:"op_fold" ~what:"input shape"
      ~reason:"C * prod(kernel_size) dimension mismatch" ();
  let channels = c_times_k / !kernel_elems in

  let blocks_shape = Array.make spatial_ndim 0 in
  for i = 0 to spatial_ndim - 1 do
    let pad_before, pad_after = padding.(i) in
    let eff_kernel = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
    let numer = output_size.(i) + pad_before + pad_after - eff_kernel in
    if numer < 0 then
      Error.invalid ~op:"op_fold" ~what:"output_size/padding/kernel_size"
        ~reason:"effective kernel does not fit output spatial dimension" ();
    blocks_shape.(i) <- (numer / stride.(i)) + 1
  done;
  let expected_blocks = Shape.numel blocks_shape in
  if expected_blocks <> num_blocks then
    Error.invalid ~op:"op_fold" ~what:"input shape"
      ~reason:"L dimension does not match computed number of sliding blocks" ();

  let out_shape = Array.append [| n; channels |] output_size in
  let out_numel = Shape.numel out_shape in
  let out_t : (a, b) t =
    match out with
    | Some out_t ->
        if shape out_t.view <> out_shape then
          Error.invalid ~op:"op_fold" ~what:"out shape mismatch" ();
        out_t
    | None ->
        let out_t : (a, b) t = op_buffer x.context x.dtype out_numel in
        { out_t with view = View.reshape out_t.view (Symbolic_shape.of_ints out_shape) }
  in

  let in_offset = View.offset in_view in
  let in_strides = View.strides in_view in
  let out_view = out_t.view in
  let out_offset = View.offset out_view in
  let out_strides = View.strides out_view in


  let kernel_elems = !kernel_elems in
  match (x.buffer, out_t.buffer) with
  | Float64 in_arr, Float64 out_arr ->
      Op_fold.fold_float64 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
        ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
        ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Float32 in_arr, Float32 out_arr ->
      Op_fold.fold_float32 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
        ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
        ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Int8 in_arr, Int8 out_arr ->
      Op_fold.fold_int8 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
        ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
        ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Int16 in_arr, Int16 out_arr ->
      Op_fold.fold_int16 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
        ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
        ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Int32 in_arr, Int32 out_arr ->
      Op_fold.fold_int32 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
        ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
        ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Int64 in_arr, Int64 out_arr ->
      Op_fold.fold_int64 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
        ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
        ~padding ~in_offset ~in_strides ~out_offset ~out_strides;
      out_t
  | Bool _, _ ->
      Error.invalid ~op:"op_fold" ~what:"unsupported dtype"
        ~reason:"bool fold is undefined because overlaps are summed" ()
  | _ -> .

let op_matmul (type a b) ~(out : (a, b) t) (a : (a, b) t) (b : (a, b) t) : unit
    =
  let va = a.view and vb = b.view and vout = out.view in
  let m = (shape vout).(0) in
  let nd_out = Array.length (shape vout) in
  let batch_shape = Array.sub (shape vout) 0 (max 0 (nd_out - 2)) in
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
      Error.invalid ~op:"op_matmul" ~what:"not implemented for small unboxed ints" ()

let op_fft ?out:_ _ ~axes:_ =
  Error.invalid ~op:"op_fft" ~what:"not implemented" ()

let op_ifft ?out:_ _ ~axes:_ =
  Error.invalid ~op:"op_ifft" ~what:"not implemented" ()

let op_rfft ?out:_ _ ~dtype:_ ~axes:_ =
  Error.invalid ~op:"op_rfft" ~what:"not implemented" ()

let op_irfft ?out:_ _ ~dtype:_ ~axes:_ ~s:_ =
  Error.invalid ~op:"op_irfft" ~what:"not implemented" ()

let op_cholesky ~upper:_ _ =
  Error.invalid ~op:"op_cholesky" ~what:"not implemented" ()

let op_qr ~reduced:_ _ = Error.invalid ~op:"op_qr" ~what:"not implemented" ()

let op_svd ~full_matrices:_ _ =
  Error.invalid ~op:"op_svd" ~what:"not implemented" ()

let op_eig ~vectors:_ _ = Error.invalid ~op:"op_eig" ~what:"not implemented" ()

let op_eigh ~vectors:_ _ =
  Error.invalid ~op:"op_eigh" ~what:"not implemented" ()

let op_triangular_solve ~upper:_ ~transpose:_ ~unit_diag:_ _ _ =
  Error.invalid ~op:"op_triangular_solve" ~what:"not implemented" ()
