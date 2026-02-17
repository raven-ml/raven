(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

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
    
let op_const_scalar _ _ _ =
    Error.invalid ~op:"op_const_scalar" ~what:"not implemented" ()
  
let from_host _ _ = Error.invalid ~op:"from_host" ~what:"not implemented" ()
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

let op_cast _ _ = Error.invalid ~op:"op_cast" ~what:"not implemented" ()

let op_contiguous _ =
Error.invalid ~op:"op_contiguous" ~what:"not implemented" ()

let op_copy _ = Error.invalid ~op:"op_copy" ~what:"not implemented" ()
let op_assign _ _ = Error.invalid ~op:"op_assign" ~what:"not implemented" ()

let op_as_strided _ _ _ _ =
Error.invalid ~op:"op_as_strided" ~what:"not implemented" ()

let op_threefry _ _ = Error.invalid ~op:"op_threefry" ~what:"not implemented" ()
let op_gather _ _ _ = Error.invalid ~op:"op_gather" ~what:"not implemented" ()

let op_scatter ?mode:_ ?unique_indices:_ _ _ _ _ =
Error.invalid ~op:"op_scatter" ~what:"not implemented" ()

let op_unfold ?out:_ _ ~kernel_size:_ ~stride:_ ~dilation:_ ~padding:_ =
Error.invalid ~op:"op_unfold" ~what:"not implemented" ()

let op_fold ?out:_ _ ~output_size:_ ~kernel_size:_ ~stride:_ ~dilation:_
  ~padding:_ =
Error.invalid ~op:"op_fold" ~what:"not implemented" ()

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
