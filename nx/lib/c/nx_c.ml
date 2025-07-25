open Nx_core

external assign :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_assign_bc" "caml_nx_assign"

external copy :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_copy_bc" "caml_nx_copy"

external cast :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('d, 'e, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_cast_bc" "caml_nx_cast"

external neg :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_neg_bc" "caml_nx_neg"

external sqrt :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_sqrt_bc" "caml_nx_sqrt"

external sin :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_sin_bc" "caml_nx_sin"

external exp2 :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_exp2_bc" "caml_nx_exp2"

external log2 :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_log2_bc" "caml_nx_log2"

external recip :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_recip_bc" "caml_nx_recip"

external add :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_add_bc" "caml_nx_add"

external sub :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_sub_bc" "caml_nx_sub"

external mul :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_mul_bc" "caml_nx_mul"

external fdiv :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_fdiv_bc" "caml_nx_fdiv"

external max :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_max_bc" "caml_nx_max"

external mod_ :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_mod_bc" "caml_nx_mod"

external pow :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_pow_bc" "caml_nx_pow"

external idiv :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_idiv_bc" "caml_nx_idiv"

external xor :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_xor_bc" "caml_nx_xor"

external or_ :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_or_bc" "caml_nx_or"

external and_ :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_and_bc" "caml_nx_and"

external cmplt :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  (int, Bigarray.int8_unsigned_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_cmplt_bc" "caml_nx_cmplt"

external cmpne :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  (int, Bigarray.int8_unsigned_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_cmpne_bc" "caml_nx_cmpne"

external reduce_sum :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  unit = "caml_nx_reduce_sum_bc" "caml_nx_reduce_sum"

external reduce_max :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  unit = "caml_nx_reduce_max_bc" "caml_nx_reduce_max"

external reduce_prod :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  unit = "caml_nx_reduce_prod_bc" "caml_nx_reduce_prod"

external where :
  int ->
  int array ->
  (int, Bigarray.int8_unsigned_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_where_bc" "caml_nx_where"

external pad :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  'a ->
  unit = "caml_nx_pad_bc" "caml_nx_pad"

external cat :
  (('a, 'b, 'c) Bigarray.Array1.t * View.t) array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  unit = "caml_nx_cat_bc" "caml_nx_cat"

external threefry :
  int ->
  int array ->
  (int32, Bigarray.int32_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  (int32, Bigarray.int32_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  (int32, Bigarray.int32_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_threefry_bc" "caml_nx_threefry"

external gather :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  (int32, Bigarray.int32_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_gather_bc" "caml_nx_gather"

external scatter :
  int ->
  (* template_ndim *)
  int array ->
  (* template_shape *)
  int ->
  (* indices_ndim *)
  int array ->
  (* indices_shape *)
  ('a, 'b, 'c) Bigarray.Array1.t ->
  (* template buffer *)
  int array ->
  (* template strides *)
  int ->
  (* template offset *)
  (int32, Bigarray.int32_elt, 'c) Bigarray.Array1.t ->
  (* indices buffer *)
  int array ->
  (* indices strides *)
  int ->
  (* indices offset *)
  ('a, 'b, 'c) Bigarray.Array1.t ->
  (* updates buffer *)
  int array ->
  (* updates strides *)
  int ->
  (* updates offset *)
  int ->
  (* axis *)
  ('a, 'b, 'c) Bigarray.Array1.t ->
  (* output buffer *)
  int array ->
  (* output strides *)
  int ->
  (* output offset *)
  int ->
  (* computation mode *)
  unit = "caml_nx_scatter_bc" "caml_nx_scatter"

type ('a, 'b) buffer = ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
type context = unit

let create_context () = ()

type ('a, 'b) t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : View.t;
}

let view t = t.view
let dtype t = t.dtype
let context t = t.context
let data t = t.buffer
let create ctx dtype buffer view = { context = ctx; dtype; buffer; view }

let make_buffer (type a b) (dtype : (a, b) Dtype.t) size =
  Bigarray.Array1.create (Dtype.to_bigarray_kind dtype) Bigarray.c_layout size

let make_tensor x shape =
  let numel = Array.fold_left ( * ) 1 shape in
  create x.context x.dtype (make_buffer x.dtype numel) (View.create shape)

let op_buffer ctx dtype size =
  create ctx dtype (make_buffer dtype size) (View.create [| size |])

let op_const_scalar ctx value dtype =
  let buffer = make_buffer dtype 1 in
  Bigarray.Array1.set buffer 0 value;
  create ctx dtype buffer (View.create [||])

let op_const_array ctx array =
  let dtype = Dtype.of_bigarray_kind (Bigarray.Array1.kind array) in
  let size = Bigarray.Array1.dim array in
  let buffer = make_buffer dtype size in
  Bigarray.Array1.blit array buffer;
  create ctx dtype buffer (View.create [| size |])

(* op_copy always creates a fresh tensor with a new buffer and copies the
   data. *)
let op_copy x =
  let result = make_tensor x (View.shape x.view) in
  copy (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) result.buffer (View.strides result.view)
    (View.offset result.view);
  let _ = x.view in
  (* FIX: Keep input tensor alive during C call. *)
  result

(* op_contiguous is a smart copy. It returns the original tensor if it's already
   contiguous, otherwise it behaves like op_copy. *)
let op_contiguous x = if View.is_c_contiguous x.view then x else op_copy x

(* op_assign copies data from a source tensor into an existing destination
   tensor. *)
let op_assign dst src =
  if View.shape dst.view <> View.shape src.view then
    failwith "op_assign: source and destination shapes must match";

  assign (View.ndim src.view) (View.shape src.view) src.buffer
    (View.strides src.view) (View.offset src.view) dst.buffer
    (View.strides dst.view) (View.offset dst.view);
  let _ = (dst.view, src.view) in
  (* FIX: Keep input tensors alive during C call. *)
  ()

(* op_cast creates a new tensor of a different dtype and copies the data,
   converting type. *)
let op_cast (type a b c d) (x : (a, b) t) (target_dtype : (c, d) Dtype.t) :
    (c, d) t =
  match Dtype.equal_witness x.dtype target_dtype with
  | Some Equal -> op_copy x
  | None ->
      let result =
        create x.context target_dtype
          (make_buffer target_dtype (View.numel x.view))
          x.view
      in
      cast (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
        (View.offset x.view) result.buffer (View.strides result.view)
        (View.offset result.view);
      let _ = x.view in
      (* FIX: Keep input tensor alive during C call. *)
      result

let unop op x =
  let result = make_tensor x (View.shape x.view) in
  op (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) result.buffer (View.strides result.view)
    (View.offset result.view);
  let _ = x.view in
  (* FIX: Keep input tensor alive during C call. *)
  result

let binop op x y =
  let result = make_tensor x (View.shape x.view) in
  op (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) y.buffer (View.strides y.view) (View.offset y.view)
    result.buffer (View.strides result.view) (View.offset result.view);
  let _ = (x.view, y.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

let op_neg x = unop neg x
let op_sqrt x = unop sqrt x
let op_sin x = unop sin x
let op_exp2 x = unop exp2 x
let op_log2 x = unop log2 x
let op_recip x = unop recip x
let op_add a b = binop add a b
let op_sub a b = binop sub a b
let op_mul a b = binop mul a b
let op_fdiv a b = binop fdiv a b
let op_max a b = binop max a b
let op_mod a b = binop mod_ a b
let op_pow a b = binop pow a b
let op_idiv a b = binop idiv a b
let op_xor a b = binop xor a b
let op_or a b = binop or_ a b
let op_and a b = binop and_ a b

let binop_cmp op x y =
  (* Comparison ops return uint8 *)
  let result_dtype = Dtype.uint8 in
  let result =
    create x.context result_dtype
      (make_buffer result_dtype (View.numel x.view))
      (View.create (View.shape x.view))
  in
  op (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) y.buffer (View.strides y.view) (View.offset y.view)
    result.buffer (View.strides result.view) (View.offset result.view);
  let _ = (x.view, y.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

let op_cmplt a b = binop_cmp cmplt a b
let op_cmpne a b = binop_cmp cmpne a b

let reduce_op op ~axes ~keepdims x =
  let input_shape = View.shape x.view in
  let ndim = Array.length input_shape in

  (* Special case: if input is already a scalar (0-dimensional), just return
     it *)
  if ndim = 0 then op_copy x (* FIX: Return a copy to maintain semantics. *)
  else
    (* Normalize axes *)
    let normalized_axes =
      Array.map (fun ax -> if ax < 0 then ax + ndim else ax) axes
    in

    (* Compute output shape *)
    let output_shape =
      if keepdims then
        Array.mapi
          (fun i dim -> if Array.mem i normalized_axes then 1 else dim)
          input_shape
      else
        let filtered = ref [] in
        Array.iteri
          (fun i dim ->
            if not (Array.mem i normalized_axes) then
              filtered := dim :: !filtered)
          input_shape;
        Array.of_list (List.rev !filtered)
    in

    (* Create result tensor *)
    let result_numel = Array.fold_left ( * ) 1 output_shape in
    let result =
      create x.context x.dtype
        (make_buffer x.dtype result_numel)
        (View.create output_shape)
    in

    (* Call the C implementation *)
    op (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
      (View.offset x.view) result.buffer (View.strides result.view)
      (View.offset result.view) normalized_axes
      (if keepdims then 1 else 0);
    let _ = x.view in
    (* FIX: Keep input tensor alive during C call. *)
    result

let op_reduce_sum ~axes ~keepdims x = reduce_op reduce_sum ~axes ~keepdims x
let op_reduce_max ~axes ~keepdims x = reduce_op reduce_max ~axes ~keepdims x
let op_reduce_prod ~axes ~keepdims x = reduce_op reduce_prod ~axes ~keepdims x
let op_reshape x shape = { x with view = View.reshape x.view shape }
let op_expand x shape = { x with view = View.expand x.view shape }
let op_permute x axes = { x with view = View.permute x.view axes }
let op_shrink x bounds = { x with view = View.shrink x.view bounds }
let op_flip x axes = { x with view = View.flip x.view axes }

let op_where cond x y =
  (* All inputs must have the same shape *)
  if
    View.shape cond.view <> View.shape x.view
    || View.shape x.view <> View.shape y.view
  then failwith "op_where: all inputs must have the same shape";

  let result = make_tensor x (View.shape x.view) in
  where (View.ndim x.view) (View.shape x.view) cond.buffer
    (View.strides cond.view) (View.offset cond.view) x.buffer
    (View.strides x.view) (View.offset x.view) y.buffer (View.strides y.view)
    (View.offset y.view) result.buffer (View.strides result.view)
    (View.offset result.view);
  let _ = (cond.view, x.view, y.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

let op_cat inputs axis =
  if List.length inputs = 0 then failwith "op_cat: need at least one input";

  let first = List.hd inputs in
  let ndim = View.ndim first.view in
  let axis = if axis < 0 then axis + ndim else axis in

  (* Verify all inputs have same shape except along concat axis *)
  let first_shape = View.shape first.view in
  List.iter
    (fun input ->
      let shape = View.shape input.view in
      if Array.length shape <> ndim then
        failwith "op_cat: all inputs must have same number of dimensions";
      Array.iteri
        (fun i dim ->
          if i <> axis && dim <> first_shape.(i) then
            failwith
              "op_cat: all inputs must have same shape except along concat axis")
        shape)
    inputs;

  (* Compute output shape *)
  let output_shape = Array.copy first_shape in
  output_shape.(axis) <-
    List.fold_left
      (fun sum input -> sum + (View.shape input.view).(axis))
      0 inputs;

  (* Create result tensor *)
  let result = make_tensor first output_shape in

  (* Prepare inputs for C function *)
  let input_pairs =
    Array.of_list (List.map (fun input -> (input.buffer, input.view)) inputs)
  in

  (* Call C implementation *)
  cat input_pairs axis result.buffer (View.strides result.view)
    (View.offset result.view) output_shape;

  (* FIX: Keep all input tensors alive during C call. *)
  let _ = List.map (fun t -> t.view) inputs in

  result

let op_threefry data seed =
  (* Inputs must have same shape *)
  if View.shape data.view <> View.shape seed.view then
    failwith "op_threefry: data and seed must have same shape";

  let result = make_tensor data (View.shape data.view) in
  threefry (View.ndim data.view) (View.shape data.view) data.buffer
    (View.strides data.view) (View.offset data.view) seed.buffer
    (View.strides seed.view) (View.offset seed.view) result.buffer
    (View.strides result.view) (View.offset result.view);
  let _ = (data.view, seed.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

let op_gather data indices axis =
  (* Validate axis *)
  let data_ndim = View.ndim data.view in
  let axis = if axis < 0 then axis + data_ndim else axis in
  if axis < 0 || axis >= data_ndim then failwith "op_gather: axis out of bounds";

  (* Check rank compatibility *)
  let indices_ndim = View.ndim indices.view in
  if data_ndim <> indices_ndim then
    failwith "op_gather: data and indices must have same rank";

  (* Check shape compatibility *)
  let data_shape = View.shape data.view in
  let indices_shape = View.shape indices.view in
  Array.iteri
    (fun i dim ->
      if i <> axis && dim > data_shape.(i) then
        failwith "op_gather: indices shape incompatible with data shape")
    indices_shape;

  (* Output has shape of indices *)
  let result = make_tensor data indices_shape in
  gather data_ndim data_shape data.buffer (View.strides data.view)
    (View.offset data.view) indices.buffer
    (View.strides indices.view)
    (View.offset indices.view) axis result.buffer (View.strides result.view)
    (View.offset result.view);
  let _ = (data.view, indices.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

let op_scatter ?(mode = `Set) ?(unique_indices = false) data_template indices
    updates axis =
  let _ = unique_indices in
  (* TODO: use this hint for optimization *)
  (* Validate axis *)
  let template_ndim = View.ndim data_template.view in
  let axis = if axis < 0 then axis + template_ndim else axis in
  if axis < 0 || axis >= template_ndim then
    failwith "op_scatter: axis out of bounds";

  (* Shape checks *)
  if View.shape indices.view <> View.shape updates.view then
    failwith "op_scatter: indices and updates must have same shape";

  let template_shape = View.shape data_template.view in
  let updates_shape = View.shape updates.view in
  Array.iteri
    (fun i dim ->
      if i <> axis && dim > template_shape.(i) then
        failwith "op_scatter: updates shape incompatible with template shape")
    updates_shape;

  (* Convert mode to integer *)
  let computation_mode =
    match mode with
    | `Set -> 0 (* SCATTER_REPLACE *)
    | `Add -> 1 (* SCATTER_ADD *)
  in

  (* Create output as copy of template *)
  let result = op_copy data_template in
  let indices_ndim = View.ndim indices.view in
  let indices_shape = View.shape indices.view in
  scatter template_ndim template_shape indices_ndim indices_shape
    data_template.buffer
    (View.strides data_template.view)
    (View.offset data_template.view)
    indices.buffer
    (View.strides indices.view)
    (View.offset indices.view) updates.buffer
    (View.strides updates.view)
    (View.offset updates.view) axis result.buffer (View.strides result.view)
    (View.offset result.view) computation_mode;
  let _ = (data_template.view, indices.view, updates.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

external matmul :
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int array ->
  int ->
  unit = "caml_nx_matmul_bc" "caml_nx_matmul"

external unfold :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  int array ->
  int array ->
  int array ->
  int array ->
  unit = "caml_nx_unfold_bc" "caml_nx_unfold"

external fold :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  int array ->
  int array ->
  int array ->
  int array ->
  unit = "caml_nx_fold_bc" "caml_nx_fold"

let op_matmul a b =
  (* Check dimensions compatibility *)
  let a_shape = View.shape a.view in
  let b_shape = View.shape b.view in
  let a_ndim = Array.length a_shape in
  let b_ndim = Array.length b_shape in

  if a_ndim < 2 || b_ndim < 2 then
    failwith "op_matmul: inputs must have at least 2 dimensions";

  (* Check inner dimensions match *)
  (if a_shape.(a_ndim - 1) <> b_shape.(b_ndim - 2) then
     let a_last = a_shape.(a_ndim - 1) in
     let b_first = b_shape.(b_ndim - 2) in
     let msg =
       Printf.sprintf
         "dot: cannot contract %s (last axis: %d) to %s (axis 0: %d) (size \
          %d≠%d)"
         (Shape.to_string a_shape) a_last (Shape.to_string b_shape) b_first
         a_last b_first
     in
     invalid_arg msg);

  (* Check dtypes match *)
  if a.dtype <> b.dtype then failwith "op_matmul: inputs must have same dtype";

  (* Compute output shape *)
  let output_shape =
    if a_ndim = 2 && b_ndim = 2 then [| a_shape.(0); b_shape.(1) |]
    else
      (* Handle broadcasting for batch dimensions *)
      let max_ndim = Stdlib.max a_ndim b_ndim in
      let result_shape = Array.make max_ndim 1 in
      (* Copy batch dimensions *)
      for i = 0 to max_ndim - 3 do
        let a_idx = if i < a_ndim - 2 then i else -1 in
        let b_idx = if i < b_ndim - 2 then i else -1 in
        let a_dim = if a_idx >= 0 then a_shape.(a_idx) else 1 in
        let b_dim = if b_idx >= 0 then b_shape.(b_idx) else 1 in
        if a_dim <> b_dim && a_dim <> 1 && b_dim <> 1 then
          failwith "op_matmul: batch dimensions are not compatible";
        result_shape.(i) <- Stdlib.max a_dim b_dim
      done;
      result_shape.(max_ndim - 2) <- a_shape.(a_ndim - 2);
      result_shape.(max_ndim - 1) <- b_shape.(b_ndim - 1);
      result_shape
  in

  let result = make_tensor a output_shape in

  matmul a.buffer (View.shape a.view) (View.strides a.view) (View.offset a.view)
    b.buffer (View.shape b.view) (View.strides b.view) (View.offset b.view)
    result.buffer (View.shape result.view) (View.strides result.view)
    (View.offset result.view);

  let _ = (a.view, b.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

let op_unfold x ~kernel_size ~stride ~dilation ~padding =
  let x_shape = View.shape x.view in
  let ndim = Array.length x_shape in
  let num_spatial_dims = Array.length kernel_size in

  (* Batch dimensions are all dimensions before channels and spatial dims *)
  let batch_dims = ndim - num_spatial_dims - 1 in
  if batch_dims < 0 then
    failwith
      "op_unfold: input must have at least one channel and one spatial \
       dimension";

  (* Validate parameters have correct number of dimensions *)
  if Array.length stride <> num_spatial_dims then
    failwith "op_unfold: stride must match number of spatial dimensions";
  if Array.length dilation <> num_spatial_dims then
    failwith "op_unfold: dilation must match number of spatial dimensions";
  if Array.length padding <> num_spatial_dims then
    failwith "op_unfold: padding must match number of spatial dimensions";

  (* Extract batch shape and channels *)
  let batch_shape = Array.sub x_shape 0 batch_dims in
  let channels = x_shape.(batch_dims) in

  (* Compute the shape of the grid of patches (output_spatial_shape) *)
  let output_spatial_shape = Array.make num_spatial_dims 0 in
  for i = 0 to num_spatial_dims - 1 do
    let pad_before, pad_after = padding.(i) in
    let dilated_kernel_size = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
    let input_spatial_dim = x_shape.(batch_dims + 1 + i) in
    let padded_size = input_spatial_dim + pad_before + pad_after in
    output_spatial_shape.(i) <-
      ((padded_size - dilated_kernel_size) / stride.(i)) + 1;
    if output_spatial_shape.(i) <= 0 then
      failwith
        (Printf.sprintf
           "op_unfold: output spatial dimension %d is non-positive (%d)" i
           output_spatial_shape.(i))
  done;

  (* Calculate total number of patches and the size of each patch column *)
  let num_patches = Array.fold_left ( * ) 1 output_spatial_shape in
  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let patch_size = channels * kernel_elements in

  (* Create the output tensor with shape [batch..., patch_size, num_patches] *)
  let output_shape =
    Array.concat [ batch_shape; [| patch_size; num_patches |] ]
  in
  let result = make_tensor x output_shape in

  (* The C function needs the 'lower' (before) padding values as a simple
     array *)
  let padding_lower = Array.map fst padding in

  (* Call the C implementation with all required parameters *)
  unfold (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) (View.ndim result.view) (View.shape result.view)
    result.buffer (View.strides result.view) (View.offset result.view)
    output_spatial_shape kernel_size stride padding_lower dilation;
  let _ = x.view in
  (* FIX: Keep input tensor alive during C call. *)
  result

let op_fold x ~output_size ~kernel_size ~stride ~dilation ~padding =
  let x_shape = View.shape x.view in
  let ndim = Array.length x_shape in

  (* Input has shape [...batch, patch_size, num_patches] *)
  if ndim < 2 then failwith "op_fold: input must have at least 2 dimensions";

  let num_spatial_dims = Array.length output_size in
  let batch_dims = ndim - 2 in
  let batch_shape = Array.sub x_shape 0 batch_dims in

  let patch_size = x_shape.(batch_dims) in
  let num_patches_in = x_shape.(batch_dims + 1) in

  (* Infer channels from patch_size and kernel_size *)
  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  if patch_size mod kernel_elements <> 0 then
    failwith "op_fold: patch_size must be divisible by product of kernel_size";
  let channels = patch_size / kernel_elements in

  (* Validate parameters *)
  if Array.length kernel_size <> num_spatial_dims then
    failwith "op_fold: kernel_size must match number of spatial dimensions";
  if Array.length stride <> num_spatial_dims then
    failwith "op_fold: stride must match number of spatial dimensions";
  if Array.length dilation <> num_spatial_dims then
    failwith "op_fold: dilation must match number of spatial dimensions";
  if Array.length padding <> num_spatial_dims then
    failwith "op_fold: padding must match number of spatial dimensions";

  (* Calculate the expected number of patches from the output geometry *)
  let output_spatial_shape = Array.make num_spatial_dims 0 in
  for i = 0 to num_spatial_dims - 1 do
    let pad_before, pad_after = padding.(i) in
    let dilated_kernel_size = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
    let padded_size = output_size.(i) + pad_before + pad_after in
    output_spatial_shape.(i) <-
      ((padded_size - dilated_kernel_size) / stride.(i)) + 1
  done;

  (* Verify that the input number of patches matches the expectation *)
  let num_patches_expected = Array.fold_left ( * ) 1 output_spatial_shape in
  if num_patches_in <> num_patches_expected then
    failwith
      (Printf.sprintf
         "op_fold: input has %d patches but output geometry implies %d"
         num_patches_in num_patches_expected);

  (* Create the output tensor with shape [batch..., channels, ...output_size] *)
  let full_output_shape =
    Array.concat [ batch_shape; [| channels |]; output_size ]
  in
  let result = make_tensor x full_output_shape in

  let padding_lower = Array.map fst padding in

  (* Call the C implementation *)
  fold (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) (View.ndim result.view) (View.shape result.view)
    result.buffer (View.strides result.view) (View.offset result.view)
    output_spatial_shape kernel_size stride padding_lower dilation;
  let _ = x.view in
  (* FIX: Keep input tensor alive during C call. *)
  result

let op_pad x padding value =
  (* padding is a list of (before, after) pairs for each dimension *)
  let ndim = View.ndim x.view in
  if Array.length padding <> ndim then
    failwith "op_pad: padding list must have one pair per dimension";

  (* Compute output shape *)
  let input_shape = View.shape x.view in
  let output_shape =
    Array.mapi
      (fun i dim ->
        let before, after = Array.get padding i in
        dim + before + after)
      input_shape
  in

  (* Create flat padding array for C function *)
  let padding_array = Array.make (ndim * 2) 0 in
  Array.iteri
    (fun i (before, after) ->
      padding_array.(i * 2) <- before;
      padding_array.((i * 2) + 1) <- after)
    padding;

  (* Create result tensor *)
  let result =
    create x.context x.dtype
      (make_buffer x.dtype (Array.fold_left ( * ) 1 output_shape))
      (View.create output_shape)
  in

  (* Call C implementation *)
  pad ndim input_shape x.buffer (View.strides x.view) (View.offset x.view)
    output_shape result.buffer (View.strides result.view)
    (View.offset result.view) padding_array value;
  let _ = x.view in
  (* FIX: Keep input tensor alive during C call. *)
  result

(* FFT operations *)
external fft_complex64 :
  int ->
  int array ->
  (Complex.t, Bigarray.complex64_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  (Complex.t, Bigarray.complex64_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  bool ->
  unit = "caml_nx_fft_complex64_bc" "caml_nx_fft_complex64"

external fft_complex32 :
  int ->
  int array ->
  (Complex.t, Bigarray.complex32_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  (Complex.t, Bigarray.complex32_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  bool ->
  unit = "caml_nx_fft_complex32_bc" "caml_nx_fft_complex32"

external rfft_float64 :
  int ->
  int array ->
  (float, Bigarray.float64_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  (Complex.t, Bigarray.complex64_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  unit = "caml_nx_rfft_float64_bc" "caml_nx_rfft_float64"

external irfft_complex64 :
  int ->
  int array ->
  (Complex.t, Bigarray.complex64_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  (float, Bigarray.float64_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  int ->
  unit = "caml_nx_irfft_complex64_bc" "caml_nx_irfft_complex64"

let op_fft (type a b) (x : (a, b) t) ~axes ~s : (a, b) t =
  let input_shape = View.shape x.view in
  let ndim = Array.length input_shape in

  (* Compute output shape *)
  let output_shape =
    match s with
    | None -> Array.copy input_shape
    | Some sizes ->
        let out_shape = Array.copy input_shape in
        Array.iteri
          (fun i axis ->
            let axis = if axis < 0 then ndim + axis else axis in
            out_shape.(axis) <- sizes.(i))
          axes;
        out_shape
  in

  let result = make_tensor x output_shape in

  (* Copy input to output first *)
  copy (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) result.buffer (View.strides result.view)
    (View.offset result.view);

  (* Call appropriate FFT function based on dtype *)
  (match x.dtype with
  | Dtype.Complex64 ->
      fft_complex64 ndim output_shape result.buffer (View.strides result.view)
        (View.offset result.view) result.buffer (View.strides result.view)
        (View.offset result.view) axes (Array.length axes) false
  | Dtype.Complex32 ->
      fft_complex32 ndim output_shape result.buffer (View.strides result.view)
        (View.offset result.view) result.buffer (View.strides result.view)
        (View.offset result.view) axes (Array.length axes) false
  | _ -> failwith "op_fft: input must be complex");

  let _ = x.view in
  result

let op_ifft (type a b) (x : (a, b) t) ~axes ~s : (a, b) t =
  let input_shape = View.shape x.view in
  let ndim = Array.length input_shape in

  (* Compute output shape *)
  let output_shape =
    match s with
    | None -> Array.copy input_shape
    | Some sizes ->
        let out_shape = Array.copy input_shape in
        Array.iteri
          (fun i axis ->
            let axis = if axis < 0 then ndim + axis else axis in
            out_shape.(axis) <- sizes.(i))
          axes;
        out_shape
  in

  let result = make_tensor x output_shape in

  (* Copy input to output first *)
  copy (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) result.buffer (View.strides result.view)
    (View.offset result.view);

  (* Call appropriate IFFT function based on dtype *)
  (match x.dtype with
  | Dtype.Complex64 ->
      fft_complex64 ndim output_shape result.buffer (View.strides result.view)
        (View.offset result.view) result.buffer (View.strides result.view)
        (View.offset result.view) axes (Array.length axes) true
  | Dtype.Complex32 ->
      fft_complex32 ndim output_shape result.buffer (View.strides result.view)
        (View.offset result.view) result.buffer (View.strides result.view)
        (View.offset result.view) axes (Array.length axes) true
  | _ -> failwith "op_ifft: input must be complex");

  let _ = x.view in
  result

let op_rfft (type a b) (x : (a, b) t) ~axes ~s :
    (Complex.t, Dtype.complex64_elt) t =
  let input_shape = View.shape x.view in
  let ndim = Array.length input_shape in

  (* For rfft, the last axis in the transform is halved + 1 *)
  let output_shape =
    let shape =
      match s with
      | None -> Array.copy input_shape
      | Some sizes ->
          let out_shape = Array.copy input_shape in
          Array.iteri
            (fun i axis ->
              let axis = if axis < 0 then ndim + axis else axis in
              out_shape.(axis) <- sizes.(i))
            axes;
          out_shape
    in
    (* Adjust last axis for rfft *)
    let last_axis_idx = Array.length axes - 1 in
    let last_axis = axes.(last_axis_idx) in
    let last_axis = if last_axis < 0 then ndim + last_axis else last_axis in
    shape.(last_axis) <- (shape.(last_axis) / 2) + 1;
    shape
  in

  (* Create complex output tensor *)
  let result =
    create x.context Dtype.Complex64
      (make_buffer Dtype.Complex64 (Array.fold_left ( * ) 1 output_shape))
      (View.create output_shape)
  in

  (* We need to handle different float types separately due to OCaml's type system *)
  (* For simplicity, always work with float64 internally *)
  let float64_x : (float, Dtype.float64_elt) t =
    match Dtype.equal_witness x.dtype Dtype.Float64 with
    | Some Equal -> x
    | None -> (
        (* Cast to float64 *)
        match x.dtype with
        | Dtype.Float32 | Dtype.Float16 ->
            let result =
              create x.context Dtype.Float64
                (make_buffer Dtype.Float64 (View.numel x.view))
                x.view
            in
            cast (View.ndim x.view) (View.shape x.view) x.buffer
              (View.strides x.view) (View.offset x.view) result.buffer
              (View.strides result.view) (View.offset result.view);
            result
        | _ -> failwith "op_rfft: input must be real")
  in

  rfft_float64 ndim
    (View.shape float64_x.view)
    float64_x.buffer
    (View.strides float64_x.view)
    (View.offset float64_x.view)
    result.buffer (View.strides result.view) (View.offset result.view) axes
    (Array.length axes);

  let _ = x.view in
  result

let op_irfft (type a b) (x : (a, b) t) ~axes ~s : (float, Dtype.float64_elt) t =
  let input_shape = View.shape x.view in
  let ndim = Array.length input_shape in

  (* For irfft, restore full size for last axis *)
  let last_axis_idx = Array.length axes - 1 in
  let last_axis = axes.(last_axis_idx) in
  let last_axis = if last_axis < 0 then ndim + last_axis else last_axis in

  let output_shape =
    let shape = Array.copy input_shape in
    (* Restore full size for last axis *)
    shape.(last_axis) <-
      (match s with
      | None -> (shape.(last_axis) - 1) * 2
      | Some sizes -> sizes.(Array.length sizes - 1));

    match s with
    | None -> shape
    | Some sizes ->
        Array.iteri
          (fun i axis ->
            let axis = if axis < 0 then ndim + axis else axis in
            shape.(axis) <- sizes.(i))
          axes;
        shape
  in

  (* Create real output tensor *)
  let result =
    create x.context Dtype.Float64
      (make_buffer Dtype.Float64 (Array.fold_left ( * ) 1 output_shape))
      (View.create output_shape)
  in

  (* For simplicity, always work with complex64 internally *)
  let complex64_x : (Complex.t, Dtype.complex64_elt) t =
    match Dtype.equal_witness x.dtype Dtype.Complex64 with
    | Some Equal -> x
    | None -> (
        (* Cast to complex64 *)
        match x.dtype with
        | Dtype.Complex32 ->
            let result =
              create x.context Dtype.Complex64
                (make_buffer Dtype.Complex64 (View.numel x.view))
                x.view
            in
            cast (View.ndim x.view) (View.shape x.view) x.buffer
              (View.strides x.view) (View.offset x.view) result.buffer
              (View.strides result.view) (View.offset result.view);
            result
        | _ -> failwith "op_irfft: input must be complex")
  in

  irfft_complex64 ndim input_shape complex64_x.buffer
    (View.strides complex64_x.view)
    (View.offset complex64_x.view)
    result.buffer (View.strides result.view) (View.offset result.view) axes
    (Array.length axes) output_shape.(last_axis);

  let _ = x.view in
  result
