open Import

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
let data t = t.buffer

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

let op_mul _out _a _b =
  Error.invalid ~op:"op_mul" ~what:"not implemented yet" ()

let op_div _out _a _b =
  Error.invalid ~op:"op_div" ~what:"not implemented yet" ()
