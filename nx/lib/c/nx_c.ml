open Nx_core
open Bigarray_ext

external assign :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_assign_bc" "caml_nx_assign"

external copy :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_copy_bc" "caml_nx_copy"

external cast :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('d, 'e, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_cast_bc" "caml_nx_cast"

external neg :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_neg_bc" "caml_nx_neg"

external sqrt :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_sqrt_bc" "caml_nx_sqrt"

external sin :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_sin_bc" "caml_nx_sin"

external exp2 :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_exp2_bc" "caml_nx_exp2"

external log2 :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_log2_bc" "caml_nx_log2"

external recip :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_recip_bc" "caml_nx_recip"

external add :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_add_bc" "caml_nx_add"

external sub :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_sub_bc" "caml_nx_sub"

external mul :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_mul_bc" "caml_nx_mul"

external fdiv :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_fdiv_bc" "caml_nx_fdiv"

external max :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_max_bc" "caml_nx_max"

external mod_ :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_mod_bc" "caml_nx_mod"

external pow :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_pow_bc" "caml_nx_pow"

external idiv :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_idiv_bc" "caml_nx_idiv"

external xor :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_xor_bc" "caml_nx_xor"

external or_ :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_or_bc" "caml_nx_or"

external and_ :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_and_bc" "caml_nx_and"

external cmplt :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  (int, int8_unsigned_elt, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_cmplt_bc" "caml_nx_cmplt"

external cmpne :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  (int, int8_unsigned_elt, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_cmpne_bc" "caml_nx_cmpne"

external reduce_sum :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  unit = "caml_nx_reduce_sum_bc" "caml_nx_reduce_sum"

external reduce_max :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  unit = "caml_nx_reduce_max_bc" "caml_nx_reduce_max"

external reduce_prod :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  unit = "caml_nx_reduce_prod_bc" "caml_nx_reduce_prod"

external where :
  int ->
  int array ->
  (int, int8_unsigned_elt, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_where_bc" "caml_nx_where"

external pad :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  int array ->
  'a ->
  unit = "caml_nx_pad_bc" "caml_nx_pad"

external cat :
  (('a, 'b, 'c) Array1.t * View.t) array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  int array ->
  unit = "caml_nx_cat_bc" "caml_nx_cat"

external threefry :
  int ->
  int array ->
  (int32, int32_elt, 'c) Array1.t ->
  int array ->
  int ->
  (int32, int32_elt, 'c) Array1.t ->
  int array ->
  int ->
  (int32, int32_elt, 'c) Array1.t ->
  int array ->
  int ->
  unit = "caml_nx_threefry_bc" "caml_nx_threefry"

external gather :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  int array ->
  (int32, int32_elt, 'c) Array1.t ->
  int array ->
  int ->
  int ->
  ('a, 'b, 'c) Array1.t ->
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
  ('a, 'b, 'c) Array1.t ->
  (* template buffer *)
  int array ->
  (* template strides *)
  int ->
  (* template offset *)
  (int32, int32_elt, 'c) Array1.t ->
  (* indices buffer *)
  int array ->
  (* indices strides *)
  int ->
  (* indices offset *)
  ('a, 'b, 'c) Array1.t ->
  (* updates buffer *)
  int array ->
  (* updates strides *)
  int ->
  (* updates offset *)
  int ->
  (* axis *)
  ('a, 'b, 'c) Array1.t ->
  (* output buffer *)
  int array ->
  (* output strides *)
  int ->
  (* output offset *)
  int ->
  (* computation mode *)
  unit = "caml_nx_scatter_bc" "caml_nx_scatter"

type ('a, 'b) buffer = ('a, 'b, c_layout) Array1.t
type context = unit

let create_context () = ()

type ('a, 'b) t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : Lazy_view.t;
}

let view t = t.view
let dtype t = t.dtype
let context t = t.context
let data t = t.buffer
let create ctx dtype buffer view = { context = ctx; dtype; buffer; view }

let make_buffer (type a b) (dtype : (a, b) Dtype.t) size =
  Array1.create (Dtype.to_bigarray_ext_kind dtype) c_layout size

let make_tensor x shape =
  let numel = Array.fold_left ( * ) 1 shape in
  create x.context x.dtype
    (make_buffer x.dtype numel)
    (Lazy_view.create (Symbolic_shape.of_ints shape))

let op_buffer ctx dtype size =
  create ctx dtype (make_buffer dtype size)
    (Lazy_view.create (Symbolic_shape.of_ints [| size |]))

let op_const_scalar ctx value dtype =
  let buffer = make_buffer dtype 1 in
  Array1.set buffer 0 value;
  create ctx dtype buffer (Lazy_view.create (Symbolic_shape.of_ints [||]))

let op_const_array ctx array =
  let dtype = Dtype.of_bigarray_ext_kind (Array1.kind array) in
  let size = Array1.dim array in
  let buffer = make_buffer dtype size in
  Array1.blit array buffer;
  create ctx dtype buffer (Lazy_view.create (Symbolic_shape.of_ints [| size |]))

(* op_copy always creates a fresh tensor with a new buffer and copies the
   data. *)
(* Helper to get concrete shape from view *)
let get_shape view =
  match Symbolic_shape.eval (Lazy_view.shape view) with
  | Some arr -> arr
  | None ->
      Error.failed ~op:"get_shape" ~what:"cannot evaluate symbolic shape" ()

(* Helper to get strides from view *)
let get_strides view =
  match Lazy_view.strides view with
  | Some s -> s
  | None ->
      Error.failed ~op:"get_strides"
        ~what:"cannot get strides for non-contiguous view" ()

(* Helper to get offset from view *)
let get_offset view =
  match Symbolic_shape.eval_dim (Lazy_view.offset view) with
  | Some n -> n
  | None ->
      Error.failed ~op:"get_offset" ~what:"cannot evaluate symbolic offset" ()

(* Helper to get numel from view *)
let get_numel view =
  match Symbolic_shape.eval_dim (Lazy_view.numel view) with
  | Some n -> n
  | None ->
      Error.failed ~op:"get_numel" ~what:"cannot evaluate symbolic numel" ()

let op_copy x =
  let shape = get_shape x.view in
  let result = make_tensor x shape in
  copy (Lazy_view.ndim x.view) shape x.buffer (get_strides x.view)
    (get_offset x.view) result.buffer (get_strides result.view)
    (get_offset result.view);
  let _ = x.view in
  (* FIX: Keep input tensor alive during C call. *)
  result

(* op_contiguous is a smart copy. It returns the original tensor if it's already
   contiguous, otherwise it behaves like op_copy. *)
let op_contiguous x = if Lazy_view.is_contiguous x.view then x else op_copy x

(* op_assign copies data from a source tensor into an existing destination
   tensor. *)
let op_assign dst src =
  let dst_shape = get_shape dst.view in
  let src_shape = get_shape src.view in
  if dst_shape <> src_shape then
    failwith "op_assign: source and destination shapes must match";

  assign (Lazy_view.ndim src.view) src_shape src.buffer (get_strides src.view)
    (get_offset src.view) dst.buffer (get_strides dst.view)
    (get_offset dst.view);
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
          (make_buffer target_dtype (get_numel x.view))
          x.view
      in
      let shape = get_shape x.view in
      cast (Lazy_view.ndim x.view) shape x.buffer (get_strides x.view)
        (get_offset x.view) result.buffer (get_strides result.view)
        (get_offset result.view);
      let _ = x.view in
      (* FIX: Keep input tensor alive during C call. *)
      result

let unop op x =
  let shape = get_shape x.view in
  let result = make_tensor x shape in
  op (Lazy_view.ndim x.view) shape x.buffer (get_strides x.view)
    (get_offset x.view) result.buffer (get_strides result.view)
    (get_offset result.view);
  let _ = x.view in
  (* FIX: Keep input tensor alive during C call. *)
  result

let binop op x y =
  let shape = get_shape x.view in
  let result = make_tensor x shape in
  op (Lazy_view.ndim x.view) shape x.buffer (get_strides x.view)
    (get_offset x.view) y.buffer (get_strides y.view) (get_offset y.view)
    result.buffer (get_strides result.view) (get_offset result.view);
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
  let shape = get_shape x.view in
  let result =
    create x.context result_dtype
      (make_buffer result_dtype (get_numel x.view))
      (Lazy_view.create (Symbolic_shape.of_ints shape))
  in
  op (Lazy_view.ndim x.view) shape x.buffer (get_strides x.view)
    (get_offset x.view) y.buffer (get_strides y.view) (get_offset y.view)
    result.buffer (get_strides result.view) (get_offset result.view);
  let _ = (x.view, y.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

let op_cmplt a b = binop_cmp cmplt a b
let op_cmpne a b = binop_cmp cmpne a b

let reduce_op op ~axes ~keepdims x =
  let input_shape = get_shape x.view in
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
        (Lazy_view.create (Symbolic_shape.of_ints output_shape))
    in

    (* Call the C implementation *)
    op (Lazy_view.ndim x.view) (get_shape x.view) x.buffer (get_strides x.view)
      (get_offset x.view) result.buffer (get_strides result.view)
      (get_offset result.view) normalized_axes
      (if keepdims then 1 else 0);
    let _ = x.view in
    (* FIX: Keep input tensor alive during C call. *)
    result

let op_reduce_sum ~axes ~keepdims x = reduce_op reduce_sum ~axes ~keepdims x
let op_reduce_max ~axes ~keepdims x = reduce_op reduce_max ~axes ~keepdims x
let op_reduce_prod ~axes ~keepdims x = reduce_op reduce_prod ~axes ~keepdims x
let op_reshape x shape = { x with view = Lazy_view.reshape shape x.view }
let op_expand x shape = { x with view = Lazy_view.expand shape x.view }
let op_permute x axes = { x with view = Lazy_view.permute axes x.view }
let op_shrink x bounds = { x with view = Lazy_view.shrink bounds x.view }
let op_flip x axes = { x with view = Lazy_view.flip axes x.view }

let op_where cond x y =
  (* All inputs must have the same shape *)
  let cond_shape = get_shape cond.view in
  let x_shape = get_shape x.view in
  let y_shape = get_shape y.view in
  if cond_shape <> x_shape || x_shape <> y_shape then
    failwith "op_where: all inputs must have the same shape";

  let result = make_tensor x x_shape in
  where (Lazy_view.ndim x.view) x_shape cond.buffer (get_strides cond.view)
    (get_offset cond.view) x.buffer (get_strides x.view) (get_offset x.view)
    y.buffer (get_strides y.view) (get_offset y.view) result.buffer
    (get_strides result.view) (get_offset result.view);
  let _ = (cond.view, x.view, y.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

let op_cat inputs axis =
  if List.length inputs = 0 then failwith "op_cat: need at least one input";

  let first = List.hd inputs in
  let ndim = Lazy_view.ndim first.view in
  let axis = if axis < 0 then axis + ndim else axis in

  (* Verify all inputs have same shape except along concat axis *)
  let first_shape = get_shape first.view in
  List.iter
    (fun input ->
      let shape = get_shape input.view in
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
      (fun sum input -> sum + (get_shape input.view).(axis))
      0 inputs;

  (* Create result tensor *)
  let result = make_tensor first output_shape in

  (* TODO: Fix this - need to convert Lazy_view to View for C function *)
  let _ = failwith "op_cat: not yet implemented with Lazy_view" in
  (* The following is unreachable but prevents type errors *)
  result

let op_threefry data seed =
  (* Inputs must have same shape *)
  let data_shape = get_shape data.view in
  let seed_shape = get_shape seed.view in
  if data_shape <> seed_shape then
    failwith "op_threefry: data and seed must have same shape";

  let result = make_tensor data data_shape in
  threefry (Lazy_view.ndim data.view) data_shape data.buffer
    (get_strides data.view) (get_offset data.view) seed.buffer
    (get_strides seed.view) (get_offset seed.view) result.buffer
    (get_strides result.view) (get_offset result.view);
  let _ = (data.view, seed.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

let op_gather data indices axis =
  (* Validate axis *)
  let data_ndim = Lazy_view.ndim data.view in
  let axis = if axis < 0 then axis + data_ndim else axis in
  if axis < 0 || axis >= data_ndim then failwith "op_gather: axis out of bounds";

  (* Check rank compatibility *)
  let indices_ndim = Lazy_view.ndim indices.view in
  if data_ndim <> indices_ndim then
    failwith "op_gather: data and indices must have same rank";

  (* Check shape compatibility *)
  let data_shape = get_shape data.view in
  let indices_shape = get_shape indices.view in
  Array.iteri
    (fun i dim ->
      if i <> axis && dim > data_shape.(i) then
        failwith "op_gather: indices shape incompatible with data shape")
    indices_shape;

  (* Output has shape of indices *)
  let result = make_tensor data indices_shape in

  gather data_ndim data_shape data.buffer (get_strides data.view)
    (get_offset data.view) indices_shape indices.buffer (get_strides indices.view)
    (get_offset indices.view) axis result.buffer (get_strides result.view)
    (get_offset result.view);
  let _ = (data.view, indices.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

let op_scatter ?(mode = `Set) ?(unique_indices = false) data_template indices
    updates axis =
  let _ = unique_indices in
  (* TODO: use this hint for optimization *)
  (* Validate axis *)
  let template_ndim = Lazy_view.ndim data_template.view in
  let axis = if axis < 0 then axis + template_ndim else axis in
  if axis < 0 || axis >= template_ndim then
    failwith "op_scatter: axis out of bounds";

  (* Shape checks *)
  let indices_shape = get_shape indices.view in
  let updates_shape = get_shape updates.view in
  if indices_shape <> updates_shape then
    failwith "op_scatter: indices and updates must have same shape";

  let template_shape = get_shape data_template.view in
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
  let indices_ndim = Lazy_view.ndim indices.view in
  scatter template_ndim template_shape indices_ndim indices_shape
    data_template.buffer
    (get_strides data_template.view)
    (get_offset data_template.view)
    indices.buffer (get_strides indices.view) (get_offset indices.view)
    updates.buffer (get_strides updates.view) (get_offset updates.view) axis
    result.buffer (get_strides result.view) (get_offset result.view)
    computation_mode;
  let _ = (data_template.view, indices.view, updates.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

external matmul :
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int array ->
  int ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int array ->
  int ->
  unit = "caml_nx_matmul_bc" "caml_nx_matmul"

external unfold :
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
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
  ('a, 'b, 'c) Array1.t ->
  int array ->
  int ->
  int ->
  int array ->
  ('a, 'b, 'c) Array1.t ->
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
  let a_shape = get_shape a.view in
  let b_shape = get_shape b.view in
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

  matmul a.buffer a_shape (get_strides a.view) (get_offset a.view) b.buffer
    b_shape (get_strides b.view) (get_offset b.view) result.buffer output_shape
    (get_strides result.view) (get_offset result.view);

  let _ = (a.view, b.view) in
  (* FIX: Keep input tensors alive during C call. *)
  result

let op_unfold x ~kernel_size ~stride ~dilation ~padding =
  let x_shape = get_shape x.view in
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
  unfold (Lazy_view.ndim x.view) x_shape x.buffer (get_strides x.view)
    (get_offset x.view)
    (Lazy_view.ndim result.view)
    output_shape result.buffer (get_strides result.view)
    (get_offset result.view) output_spatial_shape kernel_size stride
    padding_lower dilation;
  let _ = x.view in
  (* FIX: Keep input tensor alive during C call. *)
  result

let op_fold x ~output_size ~kernel_size ~stride ~dilation ~padding =
  let x_shape = get_shape x.view in
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
  fold (Lazy_view.ndim x.view) x_shape x.buffer (get_strides x.view)
    (get_offset x.view)
    (Lazy_view.ndim result.view)
    full_output_shape result.buffer (get_strides result.view)
    (get_offset result.view) output_spatial_shape kernel_size stride
    padding_lower dilation;
  let _ = x.view in
  (* FIX: Keep input tensor alive during C call. *)
  result

let op_pad x padding value =
  (* padding is a list of (before, after) pairs for each dimension *)
  let ndim = Lazy_view.ndim x.view in
  if Array.length padding <> ndim then
    failwith "op_pad: padding list must have one pair per dimension";

  (* Compute output shape *)
  let input_shape = get_shape x.view in
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
      (Lazy_view.create (Symbolic_shape.of_ints output_shape))
  in

  (* Call C implementation *)
  pad ndim input_shape x.buffer (get_strides x.view) (get_offset x.view)
    output_shape result.buffer (get_strides result.view)
    (get_offset result.view) padding_array value;
  let _ = x.view in
  (* FIX: Keep input tensor alive during C call. *)
  result

(* FFT operations *)
external fft_complex64 :
  int ->
  int array ->
  (Complex.t, complex64_elt, 'c) Array1.t ->
  int array ->
  int ->
  (Complex.t, complex64_elt, 'c) Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  bool ->
  unit = "caml_nx_fft_complex64_bc" "caml_nx_fft_complex64"

external fft_complex32 :
  int ->
  int array ->
  (Complex.t, complex32_elt, 'c) Array1.t ->
  int array ->
  int ->
  (Complex.t, complex32_elt, 'c) Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  bool ->
  unit = "caml_nx_fft_complex32_bc" "caml_nx_fft_complex32"

external rfft_float64 :
  int ->
  int array ->
  (float, float64_elt, 'c) Array1.t ->
  int array ->
  int ->
  (Complex.t, complex64_elt, 'c) Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  unit = "caml_nx_rfft_float64_bc" "caml_nx_rfft_float64"

external irfft_complex64 :
  int ->
  int array ->
  (Complex.t, complex64_elt, 'c) Array1.t ->
  int array ->
  int ->
  (float, float64_elt, 'c) Array1.t ->
  int array ->
  int ->
  int array ->
  int ->
  int ->
  unit = "caml_nx_irfft_complex64_bc" "caml_nx_irfft_complex64"

let op_fft (type a b) (x : (a, b) t) ~axes ~s : (a, b) t =
  let input_shape = get_shape x.view in
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
  copy (Lazy_view.ndim x.view) input_shape x.buffer (get_strides x.view)
    (get_offset x.view) result.buffer (get_strides result.view)
    (get_offset result.view);

  (* Call appropriate FFT function based on dtype *)
  (match x.dtype with
  | Dtype.Complex64 ->
      fft_complex64 ndim output_shape result.buffer (get_strides result.view)
        (get_offset result.view) result.buffer (get_strides result.view)
        (get_offset result.view) axes (Array.length axes) false
  | Dtype.Complex32 ->
      fft_complex32 ndim output_shape result.buffer (get_strides result.view)
        (get_offset result.view) result.buffer (get_strides result.view)
        (get_offset result.view) axes (Array.length axes) false
  | _ -> failwith "op_fft: input must be complex");

  let _ = x.view in
  result

let op_ifft (type a b) (x : (a, b) t) ~axes ~s : (a, b) t =
  let input_shape = get_shape x.view in
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
  copy (Lazy_view.ndim x.view) input_shape x.buffer (get_strides x.view)
    (get_offset x.view) result.buffer (get_strides result.view)
    (get_offset result.view);

  (* Call appropriate IFFT function based on dtype *)
  (match x.dtype with
  | Dtype.Complex64 ->
      fft_complex64 ndim output_shape result.buffer (get_strides result.view)
        (get_offset result.view) result.buffer (get_strides result.view)
        (get_offset result.view) axes (Array.length axes) true
  | Dtype.Complex32 ->
      fft_complex32 ndim output_shape result.buffer (get_strides result.view)
        (get_offset result.view) result.buffer (get_strides result.view)
        (get_offset result.view) axes (Array.length axes) true
  | _ -> failwith "op_ifft: input must be complex");

  let _ = x.view in
  result

let op_rfft (type a b c) (x : (a, b) t) ~(dtype : (Complex.t, c) Dtype.t) ~axes
    ~s : (Complex.t, c) t =
  let input_shape = get_shape x.view in
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

  (* We need to handle different float types separately due to OCaml's type system *)
  (* For simplicity, always work with float64 internally *)
  let float64_x : (float, Dtype.float64_elt) t =
    match Dtype.equal_witness x.dtype Dtype.Float64 with
    | Some Equal -> x
    | None -> (
        (* Cast to float64 *)
        match x.dtype with
        | Dtype.Float32 | Dtype.Float16 | Dtype.BFloat16 | Dtype.Float8_e4m3
        | Dtype.Float8_e5m2 ->
            let result =
              create x.context Dtype.Float64
                (make_buffer Dtype.Float64 (get_numel x.view))
                x.view
            in
            cast (Lazy_view.ndim x.view) (get_shape x.view) x.buffer
              (get_strides x.view) (get_offset x.view) result.buffer
              (get_strides result.view) (get_offset result.view);
            result
        | _ -> failwith "op_rfft: input must be real")
  in

  (* Always compute FFT as Complex64 internally *)
  let complex64_result =
    create x.context Dtype.Complex64
      (make_buffer Dtype.Complex64 (Array.fold_left ( * ) 1 output_shape))
      (Lazy_view.create (Symbolic_shape.of_ints output_shape))
  in

  rfft_float64 ndim (get_shape float64_x.view) float64_x.buffer
    (get_strides float64_x.view)
    (get_offset float64_x.view)
    complex64_result.buffer
    (get_strides complex64_result.view)
    (get_offset complex64_result.view)
    axes (Array.length axes);

  (* Cast to requested output dtype if needed *)
  match dtype with
  | Dtype.Complex64 ->
      let _ = x.view in
      complex64_result
  | Dtype.Complex32 ->
      let result =
        create x.context Dtype.Complex32
          (make_buffer Dtype.Complex32 (Array.fold_left ( * ) 1 output_shape))
          (Lazy_view.create (Symbolic_shape.of_ints output_shape))
      in
      cast ndim output_shape complex64_result.buffer
        (get_strides complex64_result.view)
        (get_offset complex64_result.view)
        result.buffer (get_strides result.view) (get_offset result.view);
      let _ = x.view in
      result
  | Dtype.Complex16 ->
      (* Cast from Complex64 to Complex16 *)
      let result =
        create x.context Dtype.Complex16
          (make_buffer Dtype.Complex16 (Array.fold_left ( * ) 1 output_shape))
          (Lazy_view.create (Symbolic_shape.of_ints output_shape))
      in
      cast ndim output_shape complex64_result.buffer
        (get_strides complex64_result.view)
        (get_offset complex64_result.view)
        result.buffer (get_strides result.view) (get_offset result.view);
      let _ = x.view in
      result

let op_irfft (type a b c) (x : (a, b) t) ~(dtype : (float, c) Dtype.t) ~axes ~s
    : (float, c) t =
  let input_shape = get_shape x.view in
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

  (* For simplicity, always work with complex64 internally *)
  let complex64_x : (Complex.t, Dtype.complex64_elt) t =
    match Dtype.equal_witness x.dtype Dtype.Complex64 with
    | Some Equal -> x
    | None -> (
        (* Cast to complex64 *)
        match x.dtype with
        | Dtype.Complex32 | Dtype.Complex16 ->
            let result =
              create x.context Dtype.Complex64
                (make_buffer Dtype.Complex64 (get_numel x.view))
                x.view
            in
            cast (Lazy_view.ndim x.view) (get_shape x.view) x.buffer
              (get_strides x.view) (get_offset x.view) result.buffer
              (get_strides result.view) (get_offset result.view);
            result
        | _ -> failwith "op_irfft: input must be complex")
  in

  (* Always compute IRFFT as Float64 internally *)
  let float64_result =
    create x.context Dtype.Float64
      (make_buffer Dtype.Float64 (Array.fold_left ( * ) 1 output_shape))
      (Lazy_view.create (Symbolic_shape.of_ints output_shape))
  in

  irfft_complex64 ndim input_shape complex64_x.buffer
    (get_strides complex64_x.view)
    (get_offset complex64_x.view)
    float64_result.buffer
    (get_strides float64_result.view)
    (get_offset float64_result.view)
    axes (Array.length axes) output_shape.(last_axis);

  (* Cast to requested output dtype if needed *)
  match dtype with
  | Dtype.Float64 ->
      let _ = x.view in
      float64_result
  | Dtype.Float32 ->
      let result =
        create x.context Dtype.Float32
          (make_buffer Dtype.Float32 (Array.fold_left ( * ) 1 output_shape))
          (Lazy_view.create (Symbolic_shape.of_ints output_shape))
      in
      cast ndim output_shape float64_result.buffer
        (get_strides float64_result.view)
        (get_offset float64_result.view)
        result.buffer (get_strides result.view) (get_offset result.view);
      let _ = x.view in
      result
  | Dtype.Float16 ->
      (* Cast from Float64 to Float16 *)
      let result =
        create x.context Dtype.Float16
          (make_buffer Dtype.Float16 (Array.fold_left ( * ) 1 output_shape))
          (Lazy_view.create (Symbolic_shape.of_ints output_shape))
      in
      cast ndim output_shape float64_result.buffer
        (get_strides float64_result.view)
        (get_offset float64_result.view)
        result.buffer (get_strides result.view) (get_offset result.view);
      let _ = x.view in
      result
  | Dtype.BFloat16 ->
      (* Cast from Float64 to BFloat16 *)
      let result =
        create x.context Dtype.BFloat16
          (make_buffer Dtype.BFloat16 (Array.fold_left ( * ) 1 output_shape))
          (Lazy_view.create (Symbolic_shape.of_ints output_shape))
      in
      cast ndim output_shape float64_result.buffer
        (get_strides float64_result.view)
        (get_offset float64_result.view)
        result.buffer (get_strides result.view) (get_offset result.view);
      let _ = x.view in
      result
  | Dtype.Float8_e4m3 ->
      (* Cast from Float64 to Float8_e4m3 *)
      let result =
        create x.context Dtype.Float8_e4m3
          (make_buffer Dtype.Float8_e4m3 (Array.fold_left ( * ) 1 output_shape))
          (Lazy_view.create (Symbolic_shape.of_ints output_shape))
      in
      cast ndim output_shape float64_result.buffer
        (get_strides float64_result.view)
        (get_offset float64_result.view)
        result.buffer (get_strides result.view) (get_offset result.view);
      let _ = x.view in
      result
  | Dtype.Float8_e5m2 ->
      (* Cast from Float64 to Float8_e5m2 *)
      let result =
        create x.context Dtype.Float8_e5m2
          (make_buffer Dtype.Float8_e5m2 (Array.fold_left ( * ) 1 output_shape))
          (Lazy_view.create (Symbolic_shape.of_ints output_shape))
      in
      cast ndim output_shape float64_result.buffer
        (get_strides float64_result.view)
        (get_offset float64_result.view)
        result.buffer (get_strides result.view) (get_offset result.view);
      let _ = x.view in
      result

(* Linear algebra operations *)

external cholesky :
  int ->
  (* upper *)
  ('a, 'b, 'c) Array1.t ->
  (* input buffer *)
  int array ->
  (* input shape *)
  int array ->
  (* input strides *)
  int ->
  (* input offset *)
  ('a, 'b, 'c) Array1.t ->
  (* output buffer *)
  int array ->
  (* output strides *)
  int ->
  (* output offset *)
  unit = "caml_nx_cholesky_bc" "caml_nx_cholesky"

external triangular_solve :
  int ->
  (* upper *)
  int ->
  (* transpose *)
  int ->
  (* unit_diag *)
  ('a, 'b, 'c) Array1.t ->
  (* A buffer *)
  int array ->
  (* A shape *)
  int array ->
  (* A strides *)
  int ->
  (* A offset *)
  ('a, 'b, 'c) Array1.t ->
  (* B buffer *)
  int array ->
  (* B shape *)
  int array ->
  (* B strides *)
  int ->
  (* B offset *)
  ('a, 'b, 'c) Array1.t ->
  (* output buffer *)
  int array ->
  (* output strides *)
  int ->
  (* output offset *)
  unit = "caml_nx_triangular_solve_bc" "caml_nx_triangular_solve"

let op_cholesky ~upper x =
  let result = make_tensor x (get_shape x.view) in
  cholesky
    (if upper then 1 else 0)
    x.buffer (get_shape x.view) (get_strides x.view) (get_offset x.view)
    result.buffer (get_strides result.view) (get_offset result.view);
  let _ = x.view in
  (* Keep input tensor alive during C call *)
  result

let op_triangular_solve ~upper ~transpose ~unit_diag a b =
  let result = make_tensor b (get_shape b.view) in
  triangular_solve
    (if upper then 1 else 0)
    (if transpose then 1 else 0)
    (if unit_diag then 1 else 0)
    a.buffer (get_shape a.view) (get_strides a.view) (get_offset a.view)
    b.buffer (get_shape b.view) (get_strides b.view) (get_offset b.view)
    result.buffer (get_strides result.view) (get_offset result.view);
  let _ = (a.view, b.view) in
  (* Keep input tensors alive during C call *)
  result

external qr :
  int ->
  (* reduced *)
  ('a, 'b, 'c) Array1.t ->
  (* input buffer *)
  int array ->
  (* input shape *)
  int array ->
  (* input strides *)
  int ->
  (* input offset *)
  ('a, 'b, 'c) Array1.t ->
  (* Q buffer *)
  int array ->
  (* Q shape *)
  int array ->
  (* Q strides *)
  int ->
  (* Q offset *)
  ('a, 'b, 'c) Array1.t ->
  (* R buffer *)
  int array ->
  (* R shape *)
  int array ->
  (* R strides *)
  int ->
  (* R offset *)
  unit = "caml_nx_qr_bc" "caml_nx_qr"

let op_qr ~reduced x =
  let shape_x = get_shape x.view in
  let ndim = Array.length shape_x in
  if ndim < 2 then failwith "op_qr: input must have at least 2 dimensions";

  let m = shape_x.(ndim - 2) in
  let n = shape_x.(ndim - 1) in

  (* Determine output shapes *)
  let shape_q =
    let s = Array.copy shape_x in
    s.(ndim - 1) <- (if reduced then min m n else m);
    s
  in
  let shape_r =
    let s = Array.copy shape_x in
    s.(ndim - 2) <- (if reduced then min m n else m);
    s
  in

  let q = make_tensor x shape_q in
  let r = make_tensor x shape_r in

  qr
    (if reduced then 1 else 0)
    x.buffer (get_shape x.view) (get_strides x.view) (get_offset x.view)
    q.buffer (get_shape q.view) (get_strides q.view) (get_offset q.view)
    r.buffer (get_shape r.view) (get_strides r.view) (get_offset r.view);

  let _ = x.view in
  (q, r)

external svd :
  int ->
  (* full_matrices *)
  ('a, 'b, 'c) Array1.t ->
  (* input buffer *)
  int array ->
  (* input shape *)
  int array ->
  (* input strides *)
  int ->
  (* input offset *)
  ('a, 'b, 'c) Array1.t ->
  (* U buffer *)
  int array ->
  (* U shape *)
  int array ->
  (* U strides *)
  int ->
  (* U offset *)
  ('d, 'e, 'c) Array1.t ->
  (* S buffer *)
  int array ->
  (* S shape *)
  int array ->
  (* S strides *)
  int ->
  (* S offset *)
  ('a, 'b, 'c) Array1.t ->
  (* VT buffer *)
  int array ->
  (* VT shape *)
  int array ->
  (* VT strides *)
  int ->
  (* VT offset *)
  unit = "caml_nx_svd_bc" "caml_nx_svd"

let op_svd (type a b) ~(full_matrices : bool) (x : (a, b) t) :
    (a, b) t * (float, Dtype.float64_elt) t * (a, b) t =
  let shape_x = get_shape x.view in
  let ndim = Array.length shape_x in
  if ndim < 2 then failwith "op_svd: input must have at least 2 dimensions";

  let m = shape_x.(ndim - 2) in
  let n = shape_x.(ndim - 1) in
  let min_mn = min m n in

  (* Determine output shapes *)
  let shape_u =
    let s = Array.copy shape_x in
    s.(ndim - 1) <- (if full_matrices then m else min_mn);
    s
  in
  let shape_s =
    let s = Array.sub shape_x 0 (ndim - 1) in
    s.(ndim - 2) <- min_mn;
    s
  in
  let shape_vt =
    let s = Array.copy shape_x in
    s.(ndim - 2) <- (if full_matrices then n else min_mn);
    s
  in

  let u = make_tensor x shape_u in
  let vt = make_tensor x shape_vt in

  (* S is always float64 *)
  let s =
    create x.context Dtype.Float64
      (make_buffer Dtype.Float64 (Array.fold_left ( * ) 1 shape_s))
      (Lazy_view.create (Symbolic_shape.of_ints shape_s))
  in

  svd
    (if full_matrices then 1 else 0)
    x.buffer (get_shape x.view) (get_strides x.view) (get_offset x.view)
    u.buffer (get_shape u.view) (get_strides u.view) (get_offset u.view)
    s.buffer (get_shape s.view) (get_strides s.view) (get_offset s.view)
    vt.buffer (get_shape vt.view) (get_strides vt.view) (get_offset vt.view);
  let _ = x.view in
  (u, s, vt)

external eig :
  int ->
  (* symmetric *)
  int ->
  (* compute_vectors *)
  ('a, 'b, 'c) Array1.t ->
  (* input buffer *)
  int array ->
  (* input shape *)
  int array ->
  (* input strides *)
  int ->
  (* input offset *)
  ('d, 'e, 'c) Array1.t ->
  (* eigenvalues buffer *)
  int array ->
  (* eigenvalues shape *)
  int array ->
  (* eigenvalues strides *)
  int ->
  (* eigenvalues offset *)
  ('f, 'g, 'c) Array1.t ->
  (* eigenvectors buffer *)
  int array ->
  (* eigenvectors shape *)
  int array ->
  (* eigenvectors strides *)
  int ->
  (* eigenvectors offset *)
  unit = "caml_nx_eig_bc" "caml_nx_eig"

let op_eig (type a b) ~vectors (x : (a, b) t) :
    (Complex.t, Dtype.complex64_elt) t
    * (Complex.t, Dtype.complex64_elt) t option =
  let shape_x = get_shape x.view in
  let ndim = Array.length shape_x in
  if ndim < 2 then failwith "op_eig: input must have at least 2 dimensions";

  let n = shape_x.(ndim - 1) in
  let m = shape_x.(ndim - 2) in
  if n != m then failwith "op_eig: input must be square matrix";

  let shape_vals =
    let s = Array.sub shape_x 0 (ndim - 1) in
    s.(ndim - 2) <- n;
    s
  in

  (* Always output complex64 *)
  let vals =
    create x.context Dtype.Complex64
      (make_buffer Dtype.Complex64 (Array.fold_left ( * ) 1 shape_vals))
      (Lazy_view.create (Symbolic_shape.of_ints shape_vals))
  in

  let vecs_opt =
    if vectors then
      Some
        (create x.context Dtype.Complex64
           (make_buffer Dtype.Complex64 (Array.fold_left ( * ) 1 shape_x))
           (Lazy_view.create (Symbolic_shape.of_ints shape_x)))
    else None
  in

  let dummy =
    if vectors then None
    else
      Some
        (create x.context Dtype.Complex64
           (make_buffer Dtype.Complex64 1)
           (Lazy_view.create (Symbolic_shape.of_ints [| 1 |])))
  in

  eig 0
    (if vectors then 1 else 0)
    x.buffer (get_shape x.view) (get_strides x.view) (get_offset x.view)
    vals.buffer (get_shape vals.view) (get_strides vals.view)
    (get_offset vals.view)
    (match vecs_opt with
    | Some v -> v.buffer
    | None -> (Option.get dummy).buffer)
    (match vecs_opt with Some v -> get_shape v.view | None -> [| 1 |])
    (match vecs_opt with Some v -> get_strides v.view | None -> [| 1 |])
    (match vecs_opt with Some v -> get_offset v.view | None -> 0);
  let _ = x.view in
  (vals, vecs_opt)

let op_eigh (type a b) ~vectors (x : (a, b) t) :
    (float, Dtype.float64_elt) t * (a, b) t option =
  let shape_x = get_shape x.view in
  let ndim = Array.length shape_x in
  if ndim < 2 then failwith "op_eigh: input must have at least 2 dimensions";

  let n = shape_x.(ndim - 1) in
  let m = shape_x.(ndim - 2) in
  if n != m then failwith "op_eigh: input must be square matrix";

  let shape_vals =
    let s = Array.sub shape_x 0 (ndim - 1) in
    s.(ndim - 2) <- n;
    s
  in

  (* Eigenvalues are always float64 *)
  let vals =
    create x.context Dtype.Float64
      (make_buffer Dtype.Float64 (Array.fold_left ( * ) 1 shape_vals))
      (Lazy_view.create (Symbolic_shape.of_ints shape_vals))
  in

  let vecs_opt = if vectors then Some (make_tensor x shape_x) else None in
  let dummy = if vectors then None else Some (make_tensor x [| 1 |]) in

  eig 1
    (if vectors then 1 else 0)
    x.buffer (get_shape x.view) (get_strides x.view) (get_offset x.view)
    vals.buffer (get_shape vals.view) (get_strides vals.view)
    (get_offset vals.view)
    (match vecs_opt with
    | Some v -> v.buffer
    | None -> (Option.get dummy).buffer)
    (match vecs_opt with Some v -> get_shape v.view | None -> [| 1 |])
    (match vecs_opt with Some v -> get_strides v.view | None -> [| 1 |])
    (match vecs_opt with Some v -> get_offset v.view | None -> 0);
  let _ = x.view in
  (vals, vecs_opt)
