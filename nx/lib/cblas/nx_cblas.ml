(* nx_cblas.ml - Minimal BLAS backend *)

open Nx_core

(* External functions - dtype dispatch handled in C via Bigarray kind *)
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
  unit = "nx_add_bc" "nx_add"

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
  unit = "nx_sub_bc" "nx_sub"

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
  unit = "nx_mul_bc" "nx_mul"

external div :
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
  unit = "nx_div_bc" "nx_div"

external max_ :
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
  unit = "nx_max_bc" "nx_max"

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
  unit = "nx_pow_bc" "nx_pow"

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
  unit = "nx_mod_bc" "nx_mod"

external neg :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "nx_neg_bc" "nx_neg"

external sqrt :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "nx_sqrt_bc" "nx_sqrt"

external sin :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "nx_sin_bc" "nx_sin"

external exp2 :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "nx_exp2_bc" "nx_exp2"

external log2 :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "nx_log2_bc" "nx_log2"

external recip :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "nx_recip_bc" "nx_recip"

external copy :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "nx_copy_bc" "nx_copy"

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
  unit = "nx_cmplt_bc" "nx_cmplt"

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
  unit = "nx_cmpne_bc" "nx_cmpne"

external reduce_sum :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  bool ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  unit = "nx_reduce_sum_bc" "nx_reduce_sum"

(* Types *)
type ('a, 'b) buffer = ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t
type context = unit

(* Create context *)
let create_context () = ()

type ('a, 'b) t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : View.t;
}

(* Accessors *)
let view t = t.view
let dtype t = t.dtype
let context t = t.context
let data t = t.buffer

(* Helpers *)
let create ctx dtype buffer view = { context = ctx; dtype; buffer; view }

let make_buffer (type a b) (dtype : (a, b) Dtype.t) size =
  Bigarray.Array1.create (Dtype.to_bigarray_kind dtype) Bigarray.c_layout size

let make_tensor x shape =
  let numel = Array.fold_left ( * ) 1 shape in
  create x.context x.dtype (make_buffer x.dtype numel) (View.create shape)

(* Buffer operations *)
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

(* Generic wrappers *)
let binop op x y =
  let result = make_tensor x (View.shape x.view) in
  op (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) y.buffer (View.strides y.view) (View.offset y.view)
    result.buffer (View.strides result.view) (View.offset result.view);
  result

let unop op x =
  let result = make_tensor x (View.shape x.view) in
  op (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) result.buffer (View.strides result.view)
    (View.offset result.view);
  result

(* Binary operations *)
let op_add x y = binop add x y
let op_sub x y = binop sub x y

let op_mul x y =
  let xs = View.shape x.view and ys = View.shape y.view in
  let xn = Array.length xs and yn = Array.length ys in

  let result_shape =
    if xn >= 2 && yn >= 2 && xs.(xn - 1) = ys.(yn - 2) then (
      let rs = Array.copy xs in
      rs.(xn - 1) <- ys.(yn - 1);
      rs)
    else if Shape.equal xs ys then xs
    else failwith "incompatible shapes for mul"
  in

  let result = make_tensor x result_shape in
  mul (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) y.buffer (View.strides y.view) (View.offset y.view)
    result.buffer (View.strides result.view) (View.offset result.view);
  result

let op_fdiv x y = binop div x y
let op_max x y = binop max_ x y
let op_mod x y = binop mod_ x y
let op_pow x y = binop pow x y

(* Unary operations *)
let op_neg x = unop neg x
let op_sqrt x = unop sqrt x
let op_sin x = unop sin x
let op_exp2 x = unop exp2 x
let op_log2 x = unop log2 x
let op_recip x = unop recip x

(* Copy/contiguous *)
let op_contiguous x =
  if View.is_c_contiguous x.view then x
  else
    let result = make_tensor x (View.shape x.view) in
    copy (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
      (View.offset x.view) result.buffer (View.strides result.view)
      (View.offset result.view);
    result

let op_copy = op_contiguous

(* Comparisons *)
let cmp_op op x y =
  let result =
    create x.context Dtype.UInt8
      (make_buffer Dtype.UInt8 (View.numel x.view))
      (View.create (View.shape x.view))
  in
  op (View.ndim x.view) (View.shape x.view) x.buffer (View.strides x.view)
    (View.offset x.view) y.buffer (View.strides y.view) (View.offset y.view)
    result.buffer (View.strides result.view) (View.offset result.view);
  result

let op_cmplt x y = cmp_op cmplt x y
let op_cmpne x y = cmp_op cmpne x y

(* Reductions *)
let op_reduce_sum ~axes ~keepdims x =
  if Array.length axes = 0 then (
    (* Full reduction *)
    let result_shape =
      if keepdims then Array.make (View.ndim x.view) 1 else [||]
    in
    let result = make_tensor x result_shape in
    reduce_sum (View.ndim x.view) (View.shape x.view) x.buffer
      (View.strides x.view) (View.offset x.view) keepdims result.buffer;
    result)
  else failwith "partial reduction not implemented"

(* Movement operations *)
let op_expand x shape = { x with view = View.expand x.view shape }

let op_reshape x shape =
  if View.is_c_contiguous x.view then { x with view = View.create shape }
  else failwith "reshape needs contiguous"

let op_permute x axes = { x with view = View.permute x.view axes }
let op_shrink x bounds = { x with view = View.shrink x.view bounds }
let op_flip x axes = { x with view = View.flip x.view axes }

(* Additional external functions *)
external reduce_max :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  bool ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  unit = "nx_reduce_max_bc" "nx_reduce_max"

external reduce_min :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  bool ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  unit = "nx_reduce_min_bc" "nx_reduce_min"

external reduce_prod :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  bool ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  unit = "nx_reduce_prod_bc" "nx_reduce_prod"

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
  unit = "nx_idiv_bc" "nx_idiv"

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
  unit = "nx_xor_bc" "nx_xor"

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
  unit = "nx_or_bc" "nx_or"

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
  unit = "nx_and_bc" "nx_and"

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
  unit = "nx_where_bc" "nx_where"

(* Integer division *)
let op_idiv x y = binop idiv x y

(* Bitwise operations *)
let op_xor x y = binop xor x y
let op_or x y = binop or_ x y
let op_and x y = binop and_ x y

(* WHERE operation *)
let op_where cond x y =
  let result = make_tensor x (View.shape x.view) in
  where (View.ndim cond.view) (View.shape cond.view) cond.buffer
    (View.strides cond.view) (View.offset cond.view) x.buffer
    (View.strides x.view) (View.offset x.view) y.buffer (View.strides y.view)
    (View.offset y.view) result.buffer (View.strides result.view)
    (View.offset result.view);
  result

(* Reductions *)
let op_reduce_max ~axes ~keepdims x =
  if Array.length axes = 0 then (
    (* Full reduction *)
    let result_shape =
      if keepdims then Array.make (View.ndim x.view) 1 else [||]
    in
    let result = make_tensor x result_shape in
    reduce_max (View.ndim x.view) (View.shape x.view) x.buffer
      (View.strides x.view) (View.offset x.view) keepdims result.buffer;
    result)
  else failwith "partial reduction not implemented"

let op_reduce_prod ~axes ~keepdims x =
  if Array.length axes = 0 then (
    (* Full reduction *)
    let result_shape =
      if keepdims then Array.make (View.ndim x.view) 1 else [||]
    in
    let result = make_tensor x result_shape in
    reduce_prod (View.ndim x.view) (View.shape x.view) x.buffer
      (View.strides x.view) (View.offset x.view) keepdims result.buffer;
    result)
  else failwith "partial reduction not implemented"

(* Additional external functions for pad and cast *)
external pad :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  (int * int) array ->
  float ->
  unit = "nx_pad_bc" "nx_pad"

external cast :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('d, 'e, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int ->
  int ->
  unit = "nx_cast_bc" "nx_cast"

(* PAD operation *)
let op_pad x pad_config fill_value =
  (* Check if padding is needed *)
  let needs_padding =
    Array.exists (fun (pb, pa) -> pb > 0 || pa > 0) pad_config
  in

  if not needs_padding then
    (* Just update the view *)
    { x with view = View.pad x.view pad_config }
  else
    (* Need to create new buffer *)
    let ndim = View.ndim x.view in
    let shape = View.shape x.view in

    (* Calculate output shape *)
    let out_shape =
      Array.init ndim (fun i ->
          let before, after = pad_config.(i) in
          shape.(i) + before + after)
    in

    let result = make_tensor x out_shape in

    (* Fill entire output with fill value *)
    Bigarray.Array1.fill result.buffer fill_value;

    (* Copy input data to padded positions *)
    (* For now, use a simple nested loop approach *)
    let rec copy_data indices dim =
      if dim = ndim then (
        (* Calculate source and destination indices *)
        let src_idx = ref (View.offset x.view) in
        let dst_idx = ref (View.offset result.view) in
        for d = 0 to ndim - 1 do
          src_idx := !src_idx + (indices.(d) * (View.strides x.view).(d));
          let before, _ = pad_config.(d) in
          dst_idx :=
            !dst_idx + ((indices.(d) + before) * (View.strides result.view).(d))
        done;
        (* Copy the value *)
        let value = Bigarray.Array1.get x.buffer !src_idx in
        Bigarray.Array1.set result.buffer !dst_idx value)
      else
        for i = 0 to shape.(dim) - 1 do
          indices.(dim) <- i;
          copy_data indices (dim + 1)
        done
    in

    (if ndim > 0 then
       let indices = Array.make ndim 0 in
       copy_data indices 0);

    result

(* CAST operation *)
let op_cast (type a b c d) (x : (a, b) t) (target_dtype : (c, d) Dtype.t) :
    (c, d) t =
  (* Create result buffer *)
  let result_buffer = make_buffer target_dtype (View.numel x.view) in
  let result = create x.context target_dtype result_buffer x.view in

  (* For contiguous arrays, we can use a simple loop *)
  if View.is_c_contiguous x.view && View.is_c_contiguous result.view then
    let total = View.numel x.view in
    (* We need to handle each conversion case separately due to OCaml's type
       system *)
    match (x.dtype, target_dtype) with
    | Dtype.Float32, Dtype.Float64 ->
        for i = 0 to total - 1 do
          let v : float = Bigarray.Array1.get x.buffer i in
          Bigarray.Array1.set result_buffer i v
        done
    | Dtype.Float64, Dtype.Float32 ->
        for i = 0 to total - 1 do
          let v : float = Bigarray.Array1.get x.buffer i in
          Bigarray.Array1.set result_buffer i v
        done
    | _, _ when Dtype.equal x.dtype target_dtype ->
        (* Same type, just copy - but we can't do this due to type
           constraints *)
        failwith "cast: same type casting should not reach backend"
    | _ -> failwith "cast: unsupported dtype conversion"
  else
    (* For non-contiguous arrays, we need to handle strides *)
    failwith "cast: non-contiguous arrays not yet supported";

  result

(* CAT operation - simple implementation *)
let op_cat inputs axis =
  match inputs with
  | [] -> failwith "cat: empty input list"
  | [ x ] -> x
  | first :: rest ->
      (* Check all inputs have same dtype and dimensions except along axis *)
      let ndim = View.ndim first.view in
      let dtype = first.dtype in
      let base_shape = View.shape first.view in

      (* Verify compatibility and calculate output shape *)
      let total_axis_size = ref base_shape.(axis) in
      List.iter
        (fun x ->
          if x.dtype <> dtype then failwith "cat: incompatible dtypes";
          if View.ndim x.view <> ndim then
            failwith "cat: incompatible dimensions";
          let shape = View.shape x.view in
          for i = 0 to ndim - 1 do
            if i <> axis && shape.(i) <> base_shape.(i) then
              failwith "cat: incompatible shapes"
          done;
          total_axis_size := !total_axis_size + shape.(axis))
        rest;

      (* Create output shape *)
      let out_shape = Array.copy base_shape in
      out_shape.(axis) <- !total_axis_size;

      (* Create output tensor *)
      let result = make_tensor first out_shape in

      (* Copy data from each input *)
      let axis_offset = ref 0 in
      List.iter
        (fun input ->
          let input_shape = View.shape input.view in
          let input_axis_size = input_shape.(axis) in

          (* Copy this input's data to the output *)
          (* For simplicity, we'll do element-by-element copy *)
          let rec copy_data indices dim =
            if dim = ndim then (
              (* Calculate source index *)
              let src_idx = ref (View.offset input.view) in
              for d = 0 to ndim - 1 do
                src_idx :=
                  !src_idx + (indices.(d) * (View.strides input.view).(d))
              done;

              (* Calculate destination index *)
              let dst_indices = Array.copy indices in
              dst_indices.(axis) <- dst_indices.(axis) + !axis_offset;
              let dst_idx = ref (View.offset result.view) in
              for d = 0 to ndim - 1 do
                dst_idx :=
                  !dst_idx + (dst_indices.(d) * (View.strides result.view).(d))
              done;

              (* Copy value *)
              let value = Bigarray.Array1.get input.buffer !src_idx in
              Bigarray.Array1.set result.buffer !dst_idx value)
            else
              let limit =
                if dim = axis then input_axis_size else input_shape.(dim)
              in
              for i = 0 to limit - 1 do
                indices.(dim) <- i;
                copy_data indices (dim + 1)
              done
          in

          (if ndim > 0 then
             let indices = Array.make ndim 0 in
             copy_data indices 0);

          axis_offset := !axis_offset + input_axis_size)
        inputs;

      result

(* ASSIGN operation - simple implementation *)
let op_assign dst src =
  (* Copy src data to dst, respecting views *)
  if View.shape dst.view <> View.shape src.view then
    failwith "assign: shapes must match";

  let ndim = View.ndim src.view in
  let shape = View.shape src.view in

  (* Element-by-element copy *)
  let rec copy_data indices dim =
    if dim = ndim then (
      (* Calculate indices *)
      let src_idx = ref (View.offset src.view) in
      let dst_idx = ref (View.offset dst.view) in
      for d = 0 to ndim - 1 do
        src_idx := !src_idx + (indices.(d) * (View.strides src.view).(d));
        dst_idx := !dst_idx + (indices.(d) * (View.strides dst.view).(d))
      done;

      (* Copy value *)
      let value = Bigarray.Array1.get src.buffer !src_idx in
      Bigarray.Array1.set dst.buffer !dst_idx value)
    else
      for i = 0 to shape.(dim) - 1 do
        indices.(dim) <- i;
        copy_data indices (dim + 1)
      done
  in

  if ndim > 0 then
    let indices = Array.make ndim 0 in
    copy_data indices 0
  else
    (* Scalar case *)
    let value = Bigarray.Array1.get src.buffer (View.offset src.view) in
    Bigarray.Array1.set dst.buffer (View.offset dst.view) value

(* External functions for remaining operations *)
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
  unit = "nx_threefry_bc" "nx_threefry"

external gather :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  (int32, Bigarray.int32_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "nx_gather_bc" "nx_gather"

external scatter :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  (int32, Bigarray.int32_elt, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "nx_scatter_bc" "nx_scatter"

(* THREEFRY operation *)
let op_threefry key counter =
  (* Validate inputs *)
  if key.dtype <> Dtype.Int32 then failwith "threefry: key must be int32";
  if counter.dtype <> Dtype.Int32 then
    failwith "threefry: counter must be int32";

  (* Result has same shape as counter *)
  let result = make_tensor counter (View.shape counter.view) in

  threefry (View.ndim key.view) (View.shape key.view) key.buffer
    (View.strides key.view) (View.offset key.view) counter.buffer
    (View.strides counter.view)
    (View.offset counter.view) result.buffer (View.strides result.view)
    (View.offset result.view);

  result

(* GATHER operation *)
let op_gather data indices axis =
  (* Validate inputs *)
  if indices.dtype <> Dtype.Int32 then failwith "gather: indices must be int32";

  let data_shape = View.shape data.view in
  let indices_shape = View.shape indices.view in
  let ndim = View.ndim data.view in

  (* Validate ranks match *)
  if View.ndim indices.view <> ndim then
    failwith "gather: data and indices must have same rank";

  (* Normalize axis *)
  let axis = if axis < 0 then axis + ndim else axis in
  if axis < 0 || axis >= ndim then failwith "gather: axis out of bounds";

  (* Result has shape of indices *)
  let result =
    create data.context data.dtype
      (make_buffer data.dtype (View.numel indices.view))
      (View.create indices_shape)
  in

  gather ndim data_shape data.buffer (View.strides data.view)
    (View.offset data.view) indices_shape indices.buffer
    (View.strides indices.view)
    (View.offset indices.view) axis result.buffer (View.strides result.view)
    (View.offset result.view);

  result

(* SCATTER operation *)
let op_scatter data_template indices updates axis =
  (* Validate inputs *)
  if indices.dtype <> Dtype.Int32 then failwith "scatter: indices must be int32";
  if data_template.dtype <> updates.dtype then
    failwith "scatter: data_template and updates must have same dtype";

  let template_shape = View.shape data_template.view in
  let indices_shape = View.shape indices.view in
  let updates_shape = View.shape updates.view in
  let ndim = View.ndim data_template.view in

  (* Validate shapes *)
  if View.ndim indices.view <> ndim || View.ndim updates.view <> ndim then
    failwith "scatter: all inputs must have same rank";

  if indices_shape <> updates_shape then
    failwith "scatter: indices and updates must have same shape";

  (* Normalize axis *)
  let axis = if axis < 0 then axis + ndim else axis in
  if axis < 0 || axis >= ndim then failwith "scatter: axis out of bounds";

  (* Result has shape of data_template *)
  let result = make_tensor data_template template_shape in

  scatter ndim template_shape data_template.buffer
    (View.strides data_template.view)
    (View.offset data_template.view)
    indices_shape indices.buffer
    (View.strides indices.view)
    (View.offset indices.view) updates.buffer
    (View.strides updates.view)
    (View.offset updates.view) axis result.buffer (View.strides result.view)
    (View.offset result.view);

  result

(* External functions for new operations *)
external matmul :
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
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  unit = "nx_matmul_bc" "nx_matmul"

external unfold :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int ->
  int array ->
  int array ->
  int array ->
  (int * int) array ->
  unit = "nx_unfold_bc" "nx_unfold"

external fold :
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int array ->
  ('a, 'b, 'c) Bigarray.Array1.t ->
  int array ->
  int ->
  int ->
  int array ->
  int array ->
  int array ->
  int array ->
  (int * int) array ->
  unit = "nx_fold_bc" "nx_fold"

(* MATMUL operation *)
let op_matmul a b =
  let shape_a = View.shape a.view in
  let shape_b = View.shape b.view in
  let ndim_a = Array.length shape_a in
  let ndim_b = Array.length shape_b in

  if ndim_a < 2 || ndim_b < 2 then failwith "matmul: inputs must be at least 2D";

  (* Check matrix dimensions compatibility *)
  let m = shape_a.(ndim_a - 2) in
  let k_a = shape_a.(ndim_a - 1) in
  let k_b = shape_b.(ndim_b - 2) in
  let n = shape_b.(ndim_b - 1) in

  if k_a <> k_b then
    invalid_arg
      (Printf.sprintf
         "matmul: cannot contract %s (last axis: %d) to %s (axis %d: %d) (size \
          %d≠%d)"
         (Shape.to_string shape_a) k_a (Shape.to_string shape_b) (ndim_b - 2)
         k_b k_a k_b);

  (* Extract batch dimensions *)
  let batch_a = Array.sub shape_a 0 (ndim_a - 2) in
  let batch_b = Array.sub shape_b 0 (ndim_b - 2) in

  (* Broadcast batch dimensions *)
  let max_batch_ndim = max (Array.length batch_a) (Array.length batch_b) in
  let batch_shape = Array.make max_batch_ndim 1 in

  (* Fill from the right *)
  for i = 0 to Array.length batch_a - 1 do
    batch_shape.(max_batch_ndim - Array.length batch_a + i) <- batch_a.(i)
  done;

  for i = 0 to Array.length batch_b - 1 do
    let idx = max_batch_ndim - Array.length batch_b + i in
    if batch_shape.(idx) = 1 then batch_shape.(idx) <- batch_b.(i)
    else if batch_b.(i) <> 1 && batch_b.(i) <> batch_shape.(idx) then
      failwith
        (Printf.sprintf "matmul: cannot broadcast shapes %s and %s"
           (Shape.to_string shape_a) (Shape.to_string shape_b))
  done;

  (* Output shape is batch_shape + [m; n] *)
  let out_shape = Array.concat [ batch_shape; [| m; n |] ] in

  (* Create output *)
  let result =
    create a.context a.dtype
      (make_buffer a.dtype (Array.fold_left ( * ) 1 out_shape))
      (View.create out_shape)
  in

  (* For CBLAS, we need to ensure inputs are contiguous *)
  let a_contig = if View.is_c_contiguous a.view then a else op_contiguous a in
  let b_contig = if View.is_c_contiguous b.view then b else op_contiguous b in

  (* Call the C implementation *)
  matmul (View.ndim a_contig.view) (View.shape a_contig.view) a_contig.buffer
    (View.strides a_contig.view)
    (View.offset a_contig.view)
    (View.shape b_contig.view) b_contig.buffer
    (View.strides b_contig.view)
    (View.offset b_contig.view)
    (View.shape result.view) result.buffer (View.strides result.view)
    (View.offset result.view);

  result

(* UNFOLD operation *)
let op_unfold t ~kernel_size ~stride ~dilation ~padding =
  let t_shape = View.shape t.view in
  let ndim = Array.length t_shape in
  let n_spatial = Array.length kernel_size in

  if ndim < n_spatial + 2 then
    invalid_arg
      "op_unfold: input must have at least batch and channel dimensions";

  let batch_size = t_shape.(0) in
  let channels = t_shape.(1) in
  let _spatial_dims = Array.sub t_shape 2 n_spatial in

  (* Apply padding if needed *)
  let t_padded =
    if Array.for_all (fun (before, after) -> before = 0 && after = 0) padding
    then t
    else
      let pad_config = Array.concat [ Array.make 2 (0, 0); padding ] in
      let fill_value : type a b. (a, b) t -> a =
       fun t ->
        match t.dtype with
        | Dtype.Float16 -> 0.0
        | Dtype.Float32 -> 0.0
        | Dtype.Float64 -> 0.0
        | Dtype.Int8 -> 0
        | Dtype.UInt8 -> 0
        | Dtype.Int16 -> 0
        | Dtype.UInt16 -> 0
        | Dtype.Int32 -> Int32.zero
        | Dtype.Int64 -> Int64.zero
        | Dtype.Int -> 0
        | Dtype.NativeInt -> 0n
        | Dtype.Complex32 -> Complex.zero
        | Dtype.Complex64 -> Complex.zero
      in
      op_pad t pad_config (fill_value t)
  in

  let padded_shape = View.shape t_padded.view in
  let padded_spatial = Array.sub padded_shape 2 n_spatial in

  (* Calculate output spatial dimensions *)
  let out_spatial =
    Array.init n_spatial (fun i ->
        let input_size = padded_spatial.(i) in
        let kernel = kernel_size.(i) in
        let s = stride.(i) in
        let d = dilation.(i) in
        let effective_kernel = 1 + ((kernel - 1) * d) in
        ((input_size - effective_kernel) / s) + 1)
  in

  (* Calculate number of patches *)
  let num_patches = Array.fold_left ( * ) 1 out_spatial in
  let patch_size = channels * Array.fold_left ( * ) 1 kernel_size in

  (* Output shape is [batch_size; patch_size; num_patches] *)
  let out_shape = [| batch_size; patch_size; num_patches |] in

  (* Create output *)
  let result =
    create t.context t.dtype
      (make_buffer t.dtype (Array.fold_left ( * ) 1 out_shape))
      (View.create out_shape)
  in

  (* Ensure input is contiguous *)
  let t_contig =
    if View.is_c_contiguous t_padded.view then t_padded
    else op_contiguous t_padded
  in

  (* Call the C implementation *)
  unfold (View.ndim t_contig.view) (View.shape t_contig.view) t_contig.buffer
    (View.strides t_contig.view)
    (View.offset t_contig.view)
    (View.shape result.view) result.buffer (View.strides result.view)
    (View.offset result.view) n_spatial kernel_size stride dilation padding;

  result

(* FOLD operation *)
let op_fold t ~output_size ~kernel_size ~stride ~dilation ~padding =
  let t_shape = View.shape t.view in
  let n_spatial = Array.length kernel_size in

  if Array.length t_shape <> 3 then
    invalid_arg "op_fold: input must be 3D [batch; patch_size; num_patches]";

  let batch_size = t_shape.(0) in
  let patch_size = t_shape.(1) in
  let _num_patches = t_shape.(2) in

  (* Calculate expected patch size *)
  let channels = patch_size / Array.fold_left ( * ) 1 kernel_size in
  if channels * Array.fold_left ( * ) 1 kernel_size <> patch_size then
    invalid_arg
      "op_fold: patch_size must be divisible by product of kernel_size";

  (* Calculate padded output size *)
  let padded_size =
    Array.init n_spatial (fun i ->
        let pad_before, pad_after = padding.(i) in
        output_size.(i) + pad_before + pad_after)
  in

  (* Output shape is [batch_size; channels; *padded_size] *)
  let out_shape = Array.concat [ [| batch_size; channels |]; padded_size ] in

  (* Create output *)
  let result =
    create t.context t.dtype
      (make_buffer t.dtype (Array.fold_left ( * ) 1 out_shape))
      (View.create out_shape)
  in

  (* Ensure input is contiguous *)
  let t_contig = if View.is_c_contiguous t.view then t else op_contiguous t in

  (* Call the C implementation *)
  fold (View.ndim t_contig.view) (View.shape t_contig.view) t_contig.buffer
    (View.strides t_contig.view)
    (View.offset t_contig.view)
    (View.shape result.view) result.buffer (View.strides result.view)
    (View.offset result.view) n_spatial output_size kernel_size stride dilation
    padding;

  (* Remove padding if needed *)
  if Array.for_all (fun (before, after) -> before = 0 && after = 0) padding then
    result
  else
    let shrink_bounds =
      Array.concat
        [
          [| (0, batch_size); (0, channels) |];
          Array.mapi
            (fun i _ ->
              let pad_before, _ = padding.(i) in
              (pad_before, pad_before + output_size.(i)))
            output_size;
        ]
    in
    op_shrink result shrink_bounds
