open Import

(* [data] returns a Bigarray, but Bigarrays cannot point to OCaml heap memory.
   Unboxed arrays are GC-managed, so we cannot create a Bigarray view of them
   without risking memory safety. Use [data_array] to access the raw buffer. *)
let data _ =
  failwith
    "Nx_oxcaml.data is not supported. Bigarrays cannot point to OCaml heap \
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
  | Dtype.Int32 ->
      let buffer = Array.make_int32 size in
      { dtype; buffer = Int32 buffer; view; context }
  | Dtype.Int64 ->
      let buffer = Array.make_int64 size in
      { dtype; buffer = Int64 buffer; view; context }
  | _ -> Error.invalid ~op:"op_buffer" ~what:"unsupported dtype" ()

let of_float64 context (arr : float #array) : (float, Dtype.float64_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  { dtype = Dtype.Float64; buffer = Float64 arr; view; context }

let of_float32 context (arr : float32 #array) : (float, Dtype.float32_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  { dtype = Dtype.Float32; buffer = Float32 arr; view; context }

let of_int32 context (arr : int32 #array) : (int32, Dtype.int32_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  { dtype = Dtype.Int32; buffer = Int32 arr; view; context }

let of_int64 context (arr : int64 #array) : (int64, Dtype.int64_elt) t =
  let size = Array.length arr in
  let sym_shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create sym_shape in
  { dtype = Dtype.Int64; buffer = Int64 arr; view; context }

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

let op_cmpeq ~out:_ _ _ =
  Error.invalid ~op:"op_cmpeq" ~what:"not implemented" ()

let op_cmpne ~out:_ _ _ =
  Error.invalid ~op:"op_cmpne" ~what:"not implemented" ()

let op_cmplt ~out:_ _ _ =
  Error.invalid ~op:"op_cmplt" ~what:"not implemented" ()

let op_cmple ~out:_ _ _ =
  Error.invalid ~op:"op_cmple" ~what:"not implemented" ()

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

let op_recip ~out:_ _ = Error.invalid ~op:"op_recip" ~what:"not implemented" ()

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

let op_sqrt ~out:_ _ = Error.invalid ~op:"op_sqrt" ~what:"not implemented" ()
let op_exp ~out:_ _ = Error.invalid ~op:"op_exp" ~what:"not implemented" ()
let op_log ~out:_ _ = Error.invalid ~op:"op_log" ~what:"not implemented" ()
let op_sin ~out:_ _ = Error.invalid ~op:"op_sin" ~what:"not implemented" ()
let op_cos ~out:_ _ = Error.invalid ~op:"op_cos" ~what:"not implemented" ()

let op_where ~out:_ _ _ _ =
  Error.invalid ~op:"op_where" ~what:"not implemented" ()

let op_reduce_sum (type a b) ~axes ~keepdims (a : (a, b) t) (out : (a, b) t) =
  let vout = out.view in
  let va = a.view in
  let in_shape = shape a.view in
  let rank = Array.length in_shape in
  let axes_to_reduce =
    List.sort_uniq compare
      (List.map
         (fun ax ->
           let ax' = if ax < 0 then ax + rank else ax in
           if ax' < 0 || ax' >= rank then
             invalid_arg
               (Printf.sprintf "sum: invalid axis %d for tensor with rank %d" ax
                  rank)
           else ax')
         (Array.to_list axes))
  in
  if axes_to_reduce = [] then blit a out
  else if rank = 0 then blit a out
  else
    let out_shape =
      Reduce_ops.reduction_output_shape in_shape axes_to_reduce keepdims
    in
    let num_output_elements = Array.fold_left ( * ) 1 out_shape in
    let num_input_elements = Array.fold_left ( * ) 1 in_shape in
    if num_output_elements = 1 && num_input_elements > 0 then
      let total_sum_value =
        match a.dtype with
        | Float64 -> parallel_sum_all_f64 out.context a
        | Float32 -> parallel_sum_all_f32 out.context a
        | Int64 -> parallel_sum_all_i64 out.context a
        | Int32 -> parallel_sum_all_i32 out.context a
      in
      let out_buf = out.buffer in
      Array.unsafe_set out_buf (offset out) total_sum_value
    else if num_output_elements > 0 && num_input_elements > 0 then
      match (out.buffer, a.buffer) with
      | Float64 out_arr, Float64 a_arr ->
          if vol > parallel_threshold then
            Parallel.parallel_for context.pool 0 (num_output_elements - 1)
              (fun start_idx end_idx ->
                kernel_sum_axis a_arr out_arr va vout axes_to_reduce start_idx
                  end_idx)
          else Op_abs.abs_float64 a_arr out_arr va vout 0 vol
      | Float32 out_arr, Float32 a_arr ->
          if vol > parallel_threshold then
            Parallel.parallel_for context.pool 0 (num_output_elements - 1)
              (fun start_idx end_idx ->
                kernel_sum_axis a_arr out_arr va vout axes_to_reduce start_idx
                  end_idx)
          else Op_abs.abs_float32 a_arr out_arr va vout 0 vol
      | Int32 out_arr, Int32 a_arr ->
          if vol > parallel_threshold then
            Parallel.parallel_for context.pool 0 (num_output_elements - 1)
              (fun start_idx end_idx ->
                kernel_sum_axis a_arr out_arr va vout axes_to_reduce start_idx
                  end_idx)
          else Op_abs.abs_int32 a_arr out_arr va vout 0 vol
      | Int64 out_arr, Int64 a_arr ->
          if vol > parallel_threshold then
            Parallel.parallel_for context.pool 0 (num_output_elements - 1)
              (fun start_idx end_idx ->
                kernel_sum_axis a_arr out_arr va vout axes_to_reduce start_idx
                  end_idx)
          else Op_abs.abs_int64 a_arr out_arr va vout 0 vol

let op_reduce_prod ~out:_ ~axes:_ ~keepdims:_ _ =
  Error.invalid ~op:"op_reduce_prod" ~what:"not implemented" ()

let op_reduce_max ~out:_ ~axes:_ ~keepdims:_ _ =
  Error.invalid ~op:"op_reduce_max" ~what:"not implemented" ()

let op_reduce_min ~out:_ ~axes:_ ~keepdims:_ _ =
  Error.invalid ~op:"op_reduce_min" ~what:"not implemented" ()

let op_associative_scan ~axis:_ ~op:_ _ =
  Error.invalid ~op:"op_associative_scan" ~what:"not implemented" ()

let op_const_scalar _ _ _ =
  Error.invalid ~op:"op_const_scalar" ~what:"not implemented" ()

let op_const_array _ _ =
  Error.invalid ~op:"op_const_array" ~what:"not implemented" ()

let op_expand x shape = { x with view = View.expand x.view shape }
let op_reshape x shape = { x with view = View.reshape x.view shape }
let op_permute _ _ = Error.invalid ~op:"op_permute" ~what:"not implemented" ()
let op_shrink _ _ = Error.invalid ~op:"op_shrink" ~what:"not implemented" ()
let op_flip _ _ = Error.invalid ~op:"op_flip" ~what:"not implemented" ()
let op_pad _ _ _ = Error.invalid ~op:"op_pad" ~what:"not implemented" ()
let op_cat _ _ = Error.invalid ~op:"op_cat" ~what:"not implemented" ()
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

let op_matmul ~out:_ _ _ =
  Error.invalid ~op:"op_matmul" ~what:"not implemented" ()

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
