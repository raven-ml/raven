(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core
open Nx_buffer

type ('a, 'b) buffer = ('a, 'b, c_layout) Array1.t
type context = unit

let create_context () = ()

type ('a, 'b) t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : View.t;
}

(* We define an FFI tensor type for easy access to the view fields in C.

   XXX: probably more efficient to inline those in our [t] type and have the
   view function create a view when called. *)
type ('a, 'b) ffi_tensor = {
  data : ('a, 'b) buffer;
  shape : int array;
  strides : int array;
  offset : int;
}
[@@warning "-69"]

(* ───── External FFI Declarations ───── *)

external caml_add :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_add"

external caml_mul :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_mul"

external caml_idiv :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_idiv"

external caml_fdiv :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_fdiv"

external caml_max :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_max"

external caml_min :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_min"

external caml_sub :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_sub"

external caml_mod :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_mod"

external caml_pow :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_pow"

external caml_cmpeq :
  ('a, 'b) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  (bool, Dtype.bool_elt) ffi_tensor ->
  unit = "caml_nx_cmpeq"

external caml_cmpne :
  ('a, 'b) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  (bool, Dtype.bool_elt) ffi_tensor ->
  unit = "caml_nx_cmpne"

external caml_cmplt :
  ('a, 'b) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  (bool, Dtype.bool_elt) ffi_tensor ->
  unit = "caml_nx_cmplt"

external caml_cmple :
  ('a, 'b) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  (bool, Dtype.bool_elt) ffi_tensor ->
  unit = "caml_nx_cmple"

external caml_xor :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_xor"

external caml_or :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_or"

external caml_and :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_and"

(* ───── Unary Operation FFI Declarations ───── *)

external caml_neg : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_neg"

external caml_sin : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_sin"

external caml_cos : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_cos"

external caml_sqrt : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_sqrt"

external caml_abs : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_abs"

external caml_log : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_log"

external caml_exp : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_exp"

external caml_recip : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_recip"

(* ───── Ternary Operation FFI Declarations ───── *)

external caml_where :
  (bool, Dtype.bool_elt) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  unit = "caml_nx_where"

(* ───── Reduction Operation FFI Declarations ───── *)

external caml_reduce_sum :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> int array -> bool -> unit
  = "caml_nx_reduce_sum"

external caml_reduce_max :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> int array -> bool -> unit
  = "caml_nx_reduce_max"

external caml_reduce_prod :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> int array -> bool -> unit
  = "caml_nx_reduce_prod"

external caml_reduce_min :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> int array -> bool -> unit
  = "caml_nx_reduce_min"

external caml_associative_scan :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> int -> int -> unit
  = "caml_nx_associative_scan"

(* Cast operation FFI declaration *)
external caml_cast : ('a, 'b) ffi_tensor -> ('c, 'd) ffi_tensor -> unit
  = "caml_nx_cast"

(* ───── Memory Operation FFI Declarations ───── *)

external caml_copy : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor = "caml_nx_copy"

external caml_contiguous : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor
  = "caml_nx_contiguous"

external caml_assign : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_assign"

(* ───── Index Operation FFI Declarations ───── *)

external caml_gather :
  ('a, 'b) ffi_tensor ->
  (int32, Dtype.int32_elt) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  int ->
  unit = "caml_nx_op_gather"

external caml_scatter :
  ('a, 'b) ffi_tensor ->
  (int32, Dtype.int32_elt) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  int ->
  ('a, 'b) ffi_tensor ->
  int ->
  bool ->
  unit = "caml_nx_op_scatter_bc" "caml_nx_op_scatter"

(* ───── Linear Algebra Operation FFI Declarations ───── *)

external caml_cholesky :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> bool -> unit
  = "caml_nx_op_cholesky"

external caml_matmul :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_matmul"

external caml_triangular_solve :
  ('a, 'b) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  bool ->
  bool ->
  bool ->
  unit = "caml_nx_op_triangular_solve_bc" "caml_nx_op_triangular_solve"

external caml_qr :
  ('a, 'b) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  bool ->
  unit = "caml_nx_op_qr"

external caml_eig :
  ('a, 'b) ffi_tensor ->
  ('c, 'd) ffi_tensor ->
  ('e, 'f) ffi_tensor ->
  bool ->
  bool ->
  unit = "caml_nx_op_eig"

external caml_svd :
  ('a, 'b) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  ('c, 'd) ffi_tensor ->
  ('a, 'b) ffi_tensor ->
  bool ->
  unit = "caml_nx_op_svd"

(* ───── Shape Operation FFI Declarations ───── *)

external caml_cat :
  ('a, 'b) ffi_tensor list -> int -> ('a, 'b) ffi_tensor -> unit = "caml_nx_cat"

external caml_pad :
  ('a, 'b) ffi_tensor -> int array -> 'a -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_pad"

(* ───── Window Operation FFI Declarations ───── *)

external caml_unfold :
  ('a, 'b) ffi_tensor ->
  int array ->
  int array ->
  int array ->
  int array ->
  ('a, 'b) ffi_tensor ->
  unit = "caml_nx_op_unfold_bc" "caml_nx_op_unfold"

external caml_fold :
  ('a, 'b) ffi_tensor ->
  int array ->
  int array ->
  int array ->
  int array ->
  int array ->
  ('a, 'b) ffi_tensor ->
  unit = "caml_nx_op_fold_bc" "caml_nx_op_fold"

(* ───── Random Operation FFI Declarations ───── *)

external caml_threefry :
  (int32, Dtype.int32_elt) ffi_tensor ->
  (int32, Dtype.int32_elt) ffi_tensor ->
  (int32, Dtype.int32_elt) ffi_tensor ->
  unit = "caml_nx_threefry"

(* ───── Helper Functions ───── *)

let view t = t.view
let dtype t = t.dtype
let to_host t = t.buffer
let context t = t.context

let shape t =
  let s = View.shape t.view in
  match Symbolic_shape.eval s with
  | Some arr -> arr
  | None -> Error.failed ~op:"shape" ~what:"symbolic shape not evaluable" ()

let strides t =
  match View.strides_opt t.view with
  | Some s -> s
  | None -> Error.failed ~op:"strides" ~what:"cannot get strides for view" ()

let offset t = View.offset t.view
let is_contiguous t = View.is_c_contiguous t.view

(* Check if a tensor can be efficiently operated on *)
let can_get_strides t = View.can_get_strides t.view

(* Convert tensor to FFI representation if possible *)
let to_ffi_tensor t =
  if not (can_get_strides t) then
    Error.failed ~op:"to_ffi_tensor" ~what:"tensor has non-materializable view"
      ()
  else
    { data = t.buffer; shape = shape t; strides = strides t; offset = offset t }

(* Create a new tensor with given shape *)
let create_tensor ctx dtype shape_arr =
  let size = Array.fold_left ( * ) 1 shape_arr in
  let kind = Dtype.to_buffer_kind dtype in
  let buffer = Array1.create kind c_layout size in
  let shape = Symbolic_shape.of_ints shape_arr in
  let view = View.create shape in
  { context = ctx; dtype; buffer; view }

(* Materialize a tensor to contiguous layout if needed *)
let materialize t =
  (* Check if it has broadcast dimensions (zero strides) *)
  let strides_arr = strides t in
  let has_broadcast = Array.exists (( = ) 0) strides_arr in

  if is_contiguous t && offset t = 0 && not has_broadcast then t
  else
    (* Create a contiguous copy *)
    let out_shape = shape t in
    let out = create_tensor t.context t.dtype out_shape in
    let t_ffi = to_ffi_tensor t in
    let out_ffi = to_ffi_tensor out in
    caml_assign t_ffi out_ffi;
    out

(* Ensure tensor is materializable for C operations *)
let ensure_materializable t =
  if not (can_get_strides t) then
    (* Broadcast views or complex chains need materialization *)
    materialize t
  else
    (* Check for zero strides (broadcast dimensions) *)
    let strides_arr = strides t in
    if Array.exists (( = ) 0) strides_arr then
      (* Has broadcast dimensions - need to materialize *)
      materialize t
    else t

(* Generic binary operation - writes to caller-provided out buffer *)
let binary_op op_name ffi_op ~out x y =
  (* Ensure both inputs have the same shape *)
  let x_shape = shape x in
  let y_shape = shape y in
  if x_shape <> y_shape then
    Error.invalid ~op:op_name ~what:"shape mismatch"
      ~reason:
        (Printf.sprintf "x: %s, y: %s"
           (Array.to_list x_shape |> List.map string_of_int |> String.concat "x")
           (Array.to_list y_shape |> List.map string_of_int |> String.concat "x"))
      ()
  else
    (* Ensure inputs are materializable *)
    let x' = ensure_materializable x in
    let y' = ensure_materializable y in

    (* Convert to FFI tensors *)
    let x_ffi = to_ffi_tensor x' in
    let y_ffi = to_ffi_tensor y' in
    let out_ffi = to_ffi_tensor out in

    (* Call C implementation *)
    ffi_op x_ffi y_ffi out_ffi

(* Comparison operation - writes bool result to caller-provided out buffer *)
let comparison_op op_name ffi_op ~out x y =
  (* Ensure both inputs have the same shape *)
  let x_shape = shape x in
  let y_shape = shape y in
  if x_shape <> y_shape then
    Error.invalid ~op:op_name ~what:"shape mismatch"
      ~reason:
        (Printf.sprintf "x: %s, y: %s"
           (Array.to_list x_shape |> List.map string_of_int |> String.concat "x")
           (Array.to_list y_shape |> List.map string_of_int |> String.concat "x"))
      ()
  else
    (* Ensure inputs are materializable *)
    let x' = ensure_materializable x in
    let y' = ensure_materializable y in

    (* Convert to FFI tensors *)
    let x_ffi = to_ffi_tensor x' in
    let y_ffi = to_ffi_tensor y' in
    let out_ffi = to_ffi_tensor out in

    (* Call C implementation *)
    ffi_op x_ffi y_ffi out_ffi

(* ───── Buffer Allocation ───── *)

let op_buffer ctx dtype size_in_elements =
  let kind = Dtype.to_buffer_kind dtype in
  let buffer = Array1.create kind c_layout size_in_elements in
  (* Create a flat view for the buffer *)
  let shape = Symbolic_shape.of_ints [| size_in_elements |] in
  let view = View.create shape in
  { context = ctx; dtype; buffer; view }

(* ───── Constant Creation Ops ───── *)

let op_const_scalar ctx value dtype =
  let kind = Dtype.to_buffer_kind dtype in
  let buffer = Array1.create kind c_layout 1 in
  Array1.set buffer 0 value;
  (* Create a scalar view (0-dimensional) *)
  let shape = Symbolic_shape.of_ints [||] in
  let view = View.create shape in
  { context = ctx; dtype; buffer; view }

let from_host ctx array =
  let dtype = Dtype.of_buffer_kind (Array1.kind array) in
  let size = Array1.dim array in
  (* Create a view for the 1D array *)
  let shape = Symbolic_shape.of_ints [| size |] in
  let view = View.create shape in
  (* Note: We're sharing the buffer directly, assuming it's contiguous *)
  { context = ctx; dtype; buffer = array; view }

(* Generic unary operation - writes to caller-provided out buffer *)
let unary_op _op_name ffi_op ~out x =
  (* Ensure input is materializable *)
  let x' = ensure_materializable x in

  (* Convert to FFI tensors *)
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in

  (* Call C implementation *)
  ffi_op x_ffi out_ffi

(* ───── Binary Operations ───── *)

let op_add ~out x y = binary_op "add" caml_add ~out x y
let op_sub ~out x y = binary_op "sub" caml_sub ~out x y
let op_mul ~out x y = binary_op "mul" caml_mul ~out x y
let op_idiv ~out x y = binary_op "idiv" caml_idiv ~out x y
let op_fdiv ~out x y = binary_op "fdiv" caml_fdiv ~out x y
let op_max ~out x y = binary_op "max" caml_max ~out x y
let op_min ~out x y = binary_op "min" caml_min ~out x y
let op_mod ~out x y = binary_op "mod" caml_mod ~out x y
let op_pow ~out x y = binary_op "pow" caml_pow ~out x y
let op_xor ~out x y = binary_op "xor" caml_xor ~out x y
let op_or ~out x y = binary_op "or" caml_or ~out x y
let op_and ~out x y = binary_op "and" caml_and ~out x y

(* ───── Comparison Operations ───── *)

let op_cmpeq ~out x y = comparison_op "cmpeq" caml_cmpeq ~out x y
let op_cmpne ~out x y = comparison_op "cmpne" caml_cmpne ~out x y
let op_cmplt ~out x y = comparison_op "cmplt" caml_cmplt ~out x y
let op_cmple ~out x y = comparison_op "cmple" caml_cmple ~out x y

(* ───── Unary Operations ───── *)

let op_neg ~out x = unary_op "neg" caml_neg ~out x
let op_log ~out x = unary_op "log" caml_log ~out x
let op_exp ~out x = unary_op "exp" caml_exp ~out x
let op_sin ~out x = unary_op "sin" caml_sin ~out x
let op_cos ~out x = unary_op "cos" caml_cos ~out x
let op_sqrt ~out x = unary_op "sqrt" caml_sqrt ~out x
let op_abs ~out x = unary_op "abs" caml_abs ~out x
let op_recip ~out x = unary_op "recip" caml_recip ~out x

(* Ternary Op - writes to caller-provided out buffer *)
let op_where ~out cond if_true if_false =
  (* Ensure all inputs have the same shape *)
  let cond_shape = shape cond in
  let if_true_shape = shape if_true in
  let if_false_shape = shape if_false in

  if cond_shape <> if_true_shape || if_true_shape <> if_false_shape then
    Error.invalid ~op:"op_where" ~what:"shape mismatch"
      ~reason:
        (Printf.sprintf "cond: %s, if_true: %s, if_false: %s"
           (Array.to_list cond_shape |> List.map string_of_int
          |> String.concat "x")
           (Array.to_list if_true_shape
           |> List.map string_of_int |> String.concat "x")
           (Array.to_list if_false_shape
           |> List.map string_of_int |> String.concat "x"))
      ()
  else
    (* Ensure inputs are materializable *)
    let cond' = ensure_materializable cond in
    let if_true' = ensure_materializable if_true in
    let if_false' = ensure_materializable if_false in

    (* Convert to FFI tensors *)
    let cond_ffi = to_ffi_tensor cond' in
    let if_true_ffi = to_ffi_tensor if_true' in
    let if_false_ffi = to_ffi_tensor if_false' in
    let out_ffi = to_ffi_tensor out in

    (* Call C implementation *)
    caml_where cond_ffi if_true_ffi if_false_ffi out_ffi

(* Reduction Ops - writes to caller-provided out buffer *)
let reduce_op _op_name ffi_op ~out ~axes ~keepdims x =
  let input_shape = shape x in
  let ndim = Array.length input_shape in

  (* Special case: if input is already a scalar (0-dimensional), just copy *)
  if ndim = 0 then Array1.set out.buffer 0 (Array1.get x.buffer 0)
  else
    (* Normalize axes *)
    let normalized_axes =
      Array.map (fun ax -> if ax < 0 then ax + ndim else ax) axes
    in

    (* Ensure input is materializable *)
    let x' = ensure_materializable x in

    (* Convert to FFI tensors *)
    let x_ffi = to_ffi_tensor x' in
    let out_ffi = to_ffi_tensor out in

    (* Call C implementation *)
    ffi_op x_ffi out_ffi normalized_axes keepdims

let op_reduce_sum ~out ~axes ~keepdims x =
  reduce_op "reduce_sum" caml_reduce_sum ~out ~axes ~keepdims x

let op_reduce_max ~out ~axes ~keepdims x =
  reduce_op "reduce_max" caml_reduce_max ~out ~axes ~keepdims x

let op_reduce_prod ~out ~axes ~keepdims x =
  reduce_op "reduce_prod" caml_reduce_prod ~out ~axes ~keepdims x

let op_reduce_min ~out ~axes ~keepdims x =
  reduce_op "reduce_min" caml_reduce_min ~out ~axes ~keepdims x

let op_associative_scan ~axis ~op x =
  let x_shape = shape x in
  let rank = Array.length x_shape in
  if rank = 0 then
    Error.invalid ~op:"associative_scan" ~what:"tensor"
      ~reason:"requires rank >= 1" ()
  else
    let axis = if axis < 0 then axis + rank else axis in
    if axis < 0 || axis >= rank then
      Error.invalid ~op:"associative_scan" ~what:"axis"
        ~reason:(Printf.sprintf "axis %d out of bounds for rank %d" axis rank)
        ()
    else
      let x' = ensure_materializable x in
      let out = create_tensor x.context x.dtype x_shape in
      let x_ffi = to_ffi_tensor x' in
      let out_ffi = to_ffi_tensor out in
      let op_tag =
        match op with `Sum -> 0 | `Prod -> 1 | `Max -> 2 | `Min -> 3
      in
      caml_associative_scan x_ffi out_ffi axis op_tag;
      out

(* Movement Ops - These are view-only operations *)
let op_expand x shape = { x with view = View.expand x.view shape }
let op_reshape x shape = { x with view = View.reshape x.view shape }
let op_permute x axes = { x with view = View.permute x.view axes }

let op_pad x padding fill_value =
  let x' = ensure_materializable x in

  (* Calculate output shape *)
  let in_shape = shape x in
  let ndim = Array.length in_shape in

  (* Convert pairs to flat array for C interface *)
  let padding_flat =
    Array.init (2 * ndim) (fun i ->
        let dim = i / 2 in
        if i mod 2 = 0 then fst padding.(dim) else snd padding.(dim))
  in

  (* Calculate output shape *)
  let out_shape =
    Array.init ndim (fun i ->
        let before, after = padding.(i) in
        in_shape.(i) + before + after)
  in

  let out = create_tensor x.context x.dtype out_shape in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in
  caml_pad x_ffi padding_flat fill_value out_ffi;
  out

let op_shrink x bounds = { x with view = View.shrink x.view bounds }
let op_flip x axes = { x with view = View.flip x.view axes }

let op_cat tensors axis =
  match tensors with
  | [] -> Error.failed ~op:"op_cat" ~what:"empty tensor list" ()
  | first :: _ ->
      let tensors' = List.map ensure_materializable tensors in

      (* Calculate output shape *)
      let first_shape = shape first in
      let ndim = Array.length first_shape in
      let norm_axis = if axis < 0 then ndim + axis else axis in

      (* Sum up dimensions along concatenation axis *)
      let total_axis_size =
        List.fold_left
          (fun acc t ->
            let s = shape t in
            acc + s.(norm_axis))
          0 tensors
      in

      let out_shape =
        Array.mapi
          (fun i dim -> if i = norm_axis then total_axis_size else dim)
          first_shape
      in

      let out = create_tensor first.context first.dtype out_shape in
      let tensors_ffi = List.map to_ffi_tensor tensors' in
      let out_ffi = to_ffi_tensor out in
      caml_cat tensors_ffi norm_axis out_ffi;
      out

(* ───── Other Ops ───── *)

let op_cast (type a b c d) (x : (a, b) t) (target_dtype : (c, d) Dtype.t) :
    (c, d) t =
  (* Ensure input is materializable *)
  let x' = ensure_materializable x in

  (* Get input shape *)
  let x_shape = shape x in

  (* Create output tensor with target dtype *)
  let out = create_tensor x.context target_dtype x_shape in

  (* Convert to FFI tensors *)
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in

  (* Call C implementation *)
  caml_cast x_ffi out_ffi;

  out

let op_contiguous x =
  (* Check if already contiguous with no offset and no broadcast dimensions *)
  let strides_arr = strides x in
  let has_broadcast = Array.exists (( = ) 0) strides_arr in
  if is_contiguous x && offset x = 0 && not has_broadcast then x
  else
    let x' = ensure_materializable x in
    let x_ffi = to_ffi_tensor x' in
    let out_ffi = caml_contiguous x_ffi in
    (* Create tensor from FFI result - it's contiguous so simple view *)
    let shape_sym = Symbolic_shape.of_ints out_ffi.shape in
    let view = View.create shape_sym in
    { context = x.context; dtype = x.dtype; buffer = out_ffi.data; view }

let op_copy x =
  let x' = ensure_materializable x in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = caml_copy x_ffi in
  (* Create tensor from FFI result - it's contiguous so simple view *)
  let shape_sym = Symbolic_shape.of_ints out_ffi.shape in
  let view = View.create shape_sym in
  { context = x.context; dtype = x.dtype; buffer = out_ffi.data; view }

let op_assign dst src =
  let src' = ensure_materializable src in
  (* dst doesn't need materialization - we're writing to it *)
  let src_ffi = to_ffi_tensor src' in
  let dst_ffi = to_ffi_tensor dst in
  caml_assign src_ffi dst_ffi

let op_threefry key counter =
  let key' = ensure_materializable key in
  let counter' = ensure_materializable counter in

  (* Output has same shape as counter *)
  let out_shape = shape counter in
  let out = create_tensor counter.context counter.dtype out_shape in

  let key_ffi = to_ffi_tensor key' in
  let counter_ffi = to_ffi_tensor counter' in
  let out_ffi = to_ffi_tensor out in
  caml_threefry key_ffi counter_ffi out_ffi;
  out

(* ───── Element Access Ops ───── *)

let op_gather data indices axis =
  (* Ensure inputs are materializable. Preserve broadcasted strides for indices
     to enable C fast paths (e.g., memcpy row gather). *)
  let data' = ensure_materializable data in
  (* Do not materialize indices unless we cannot get strides *)
  let indices' =
    if can_get_strides indices then indices else ensure_materializable indices
  in

  (* Get shapes *)
  let indices_shape = shape indices in

  (* Calculate output shape: indices shape *)
  let out_shape = indices_shape in

  (* Create output tensor with same dtype as data *)
  let out = create_tensor data.context data.dtype out_shape in

  (* Convert to FFI tensors *)
  let data_ffi = to_ffi_tensor data' in
  let indices_ffi = to_ffi_tensor indices' in
  let out_ffi = to_ffi_tensor out in

  (* Call FFI function *)
  caml_gather data_ffi indices_ffi out_ffi axis;

  out

let op_scatter ?(mode = `Set) ?(unique_indices = false) data_template indices
    updates axis =
  (* Ensure inputs are materializable *)
  let template' = ensure_materializable data_template in
  let indices' = ensure_materializable indices in
  let updates' = ensure_materializable updates in

  (* Create output tensor - for Set mode, start with a copy of template *)
  let out =
    if mode = `Set then op_copy data_template (* Start with copy of template *)
    else
      create_tensor data_template.context data_template.dtype
        (shape data_template)
  in

  (* Convert to FFI tensors *)
  let template_ffi = to_ffi_tensor template' in
  let indices_ffi = to_ffi_tensor indices' in
  let updates_ffi = to_ffi_tensor updates' in
  let out_ffi = to_ffi_tensor out in

  (* Convert mode to integer: 0 for Set, 1 for Add *)
  let mode_int = match mode with `Set -> 0 | `Add -> 1 in

  (* Call FFI function *)
  caml_scatter template_ffi indices_ffi updates_ffi axis out_ffi mode_int
    unique_indices;

  out

let op_unfold ?out x ~kernel_size ~stride ~dilation ~padding =
  let x' = ensure_materializable x in

  let in_shape = shape x in
  let batch_size = in_shape.(0) in
  let in_channels = in_shape.(1) in
  let spatial_dims = Array.sub in_shape 2 (Array.length in_shape - 2) in

  let padding_flat =
    Array.init
      (Array.length padding * 2)
      (fun i ->
        let dim = i / 2 in
        if i mod 2 = 0 then fst padding.(dim) else snd padding.(dim))
  in

  let out_spatial =
    Array.init (Array.length spatial_dims) (fun i ->
        let pad_before, pad_after = padding.(i) in
        let padded = spatial_dims.(i) + pad_before + pad_after in
        let kernel_extent = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
        let diff = padded - kernel_extent in
        if diff < 0 then
          Error.invalid ~op:"unfold"
            ~what:"kernel size larger than padded input" ()
        else (diff / stride.(i)) + 1)
  in

  let kernel_prod = Array.fold_left ( * ) 1 kernel_size in
  let spatial_prod = Array.fold_left ( * ) 1 out_spatial in
  let out_shape = [| batch_size; in_channels * kernel_prod; spatial_prod |] in

  let out =
    match out with
    | Some o -> o
    | None -> create_tensor x.context x.dtype out_shape
  in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in
  caml_unfold x_ffi kernel_size stride dilation padding_flat out_ffi;
  out

let op_fold ?out x ~output_size ~kernel_size ~stride ~dilation ~padding =
  let x' = ensure_materializable x in

  let in_shape = shape x in
  let batch_size = in_shape.(0) in
  let kernel_prod = Array.fold_left ( * ) 1 kernel_size in
  let channels = in_shape.(1) / kernel_prod in

  let padding_flat =
    Array.init
      (Array.length padding * 2)
      (fun i ->
        let dim = i / 2 in
        if i mod 2 = 0 then fst padding.(dim) else snd padding.(dim))
  in

  let _ =
    Array.init (Array.length output_size) (fun i ->
        let pad_before, pad_after = padding.(i) in
        let padded = output_size.(i) + pad_before + pad_after in
        let kernel_extent = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
        let diff = padded - kernel_extent in
        if diff < 0 then
          Error.invalid ~op:"fold" ~what:"kernel size larger than padded output"
            ()
        else (diff / stride.(i)) + 1)
  in

  let out_shape = Array.concat [ [| batch_size; channels |]; output_size ] in

  let out =
    match out with
    | Some o -> o
    | None -> create_tensor x.context x.dtype out_shape
  in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in
  caml_fold x_ffi output_size kernel_size stride dilation padding_flat out_ffi;
  out

let op_matmul ~out x y =
  let x' = if is_contiguous x then x else ensure_materializable x in
  let y' = if is_contiguous y then y else ensure_materializable y in
  let x_ffi = to_ffi_tensor x' in
  let y_ffi = to_ffi_tensor y' in
  let out_ffi = to_ffi_tensor out in
  caml_matmul x_ffi y_ffi out_ffi

(* Helper to compute contiguous strides in bytes *)
let contiguous_strides shape elem_size =
  let ndim = Array.length shape in
  if ndim = 0 then [||]
  else
    let strides = Array.make ndim 1 in
    for i = ndim - 2 downto 0 do
      strides.(i) <- strides.(i + 1) * shape.(i + 1)
    done;
    Array.map (fun s -> s * elem_size) strides

(* ───── Fourier Transforms Using PocketFFT ───── *)

let op_fft (type a b) ?out (x : (a, b) t) ~axes : (a, b) t =
  let x' = materialize x in
  let out_shape = shape x' in
  let out =
    match out with
    | Some o -> o
    | None -> create_tensor x.context x.dtype out_shape
  in

  let shape_arr = out_shape in
  let elem_size = Dtype.itemsize x.dtype in
  let strides_in = contiguous_strides out_shape elem_size in
  let strides_out = contiguous_strides out_shape elem_size in
  (* Normalize negative axes *)
  let ndim = Array.length out_shape in
  let axes_arr = Array.map (fun ax -> if ax < 0 then ndim + ax else ax) axes in

  (match (x.dtype : (a, b) Dtype.t) with
  | Dtype.Complex64 ->
      Pocketfft.c2c_f32 ~shape:shape_arr ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_arr ~forward:true ~fct:1.0
        ~data_in:x'.buffer ~data_out:out.buffer ~nthreads:1
  | Dtype.Complex128 ->
      Pocketfft.c2c_f64 ~shape:shape_arr ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_arr ~forward:true ~fct:1.0
        ~data_in:x'.buffer ~data_out:out.buffer ~nthreads:1
  | _ -> Error.failed ~op:"op_fft" ~what:"unsupported dtype" ());

  out

let op_ifft (type a b) ?out (x : (a, b) t) ~axes : (a, b) t =
  let x' = materialize x in
  let out_shape = shape x' in
  let out =
    match out with
    | Some o -> o
    | None -> create_tensor x.context x.dtype out_shape
  in

  let shape_arr = out_shape in
  let elem_size = Dtype.itemsize x.dtype in
  let strides_in = contiguous_strides out_shape elem_size in
  let strides_out = contiguous_strides out_shape elem_size in
  (* Normalize negative axes *)
  let ndim = Array.length out_shape in
  let axes_arr = Array.map (fun ax -> if ax < 0 then ndim + ax else ax) axes in

  (match (x.dtype : (a, b) Dtype.t) with
  | Dtype.Complex64 ->
      Pocketfft.c2c_f32 ~shape:shape_arr ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_arr ~forward:false ~fct:1.0
        ~data_in:x'.buffer ~data_out:out.buffer ~nthreads:1
  | Dtype.Complex128 ->
      Pocketfft.c2c_f64 ~shape:shape_arr ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_arr ~forward:false ~fct:1.0
        ~data_in:x'.buffer ~data_out:out.buffer ~nthreads:1
  | _ -> Error.failed ~op:"op_ifft" ~what:"unsupported dtype" ());

  out

let op_rfft (type a b c d) ?out (x : (a, b) t) ~(dtype : (c, d) Dtype.t) ~axes :
    (c, d) t =
  let x' = materialize x in

  (* Calculate output shape for rfft *)
  let in_shape = shape x' in
  let out_shape = Array.copy in_shape in
  let last_axis = Array.length axes - 1 in
  (if last_axis >= 0 then
     let axis_idx =
       if axes.(last_axis) < 0 then Array.length in_shape + axes.(last_axis)
       else axes.(last_axis)
     in
     out_shape.(axis_idx) <- (in_shape.(axis_idx) / 2) + 1);

  let out =
    match out with
    | Some o -> o
    | None -> create_tensor x.context dtype out_shape
  in

  let strides_in = contiguous_strides in_shape (Dtype.itemsize x.dtype) in
  let strides_out = contiguous_strides out_shape (Dtype.itemsize dtype) in

  (* Normalize negative axes *)
  let ndim = Array.length in_shape in
  let axes_normalized =
    Array.map (fun ax -> if ax < 0 then ndim + ax else ax) axes
  in

  (match ((x.dtype : (a, b) Dtype.t), (dtype : (c, d) Dtype.t)) with
  | Dtype.Float32, Dtype.Complex64 ->
      let data_in : (float, Dtype.float32_elt, c_layout) Array1.t = x'.buffer in
      let data_out : (Complex.t, Dtype.complex32_elt, c_layout) Array1.t =
        out.buffer
      in
      Pocketfft.r2c_f32 ~shape_in:in_shape ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_normalized ~forward:true ~fct:1.0
        ~data_in ~data_out ~nthreads:1
  | Dtype.Float64, Dtype.Complex128 ->
      let data_in : (float, Dtype.float64_elt, c_layout) Array1.t = x'.buffer in
      let data_out : (Complex.t, Dtype.complex64_elt, c_layout) Array1.t =
        out.buffer
      in
      Pocketfft.r2c_f64 ~shape_in:in_shape ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_normalized ~forward:true ~fct:1.0
        ~data_in ~data_out ~nthreads:1
  | _ -> Error.failed ~op:"op_rfft" ~what:"unsupported dtype combination" ());

  out

let op_irfft (type a b c d) ?out (x : (a, b) t) ~(dtype : (c, d) Dtype.t) ~axes
    ~s : (c, d) t =
  let x' = materialize x in

  (* Calculate output shape for irfft *)
  let in_shape = shape x' in
  let out_shape = Array.copy in_shape in
  let last_axis = Array.length axes - 1 in

  (if last_axis >= 0 then
     let axis_idx =
       if axes.(last_axis) < 0 then Array.length in_shape + axes.(last_axis)
       else axes.(last_axis)
     in
     let size =
       match s with
       | None -> (in_shape.(axis_idx) - 1) * 2
       | Some sizes -> sizes.(last_axis)
     in
     out_shape.(axis_idx) <- size);

  let out =
    match out with
    | Some o -> o
    | None -> create_tensor x.context dtype out_shape
  in

  let strides_in = contiguous_strides in_shape (Dtype.itemsize x.dtype) in
  let strides_out = contiguous_strides out_shape (Dtype.itemsize dtype) in

  (* Normalize negative axes *)
  let ndim = Array.length in_shape in
  let axes_normalized =
    Array.map (fun ax -> if ax < 0 then ndim + ax else ax) axes
  in

  (match ((x.dtype : (a, b) Dtype.t), (dtype : (c, d) Dtype.t)) with
  | Dtype.Complex64, Dtype.Float32 ->
      let data_in : (Complex.t, Dtype.complex32_elt, c_layout) Array1.t =
        x'.buffer
      in
      let data_out : (float, Dtype.float32_elt, c_layout) Array1.t =
        out.buffer
      in
      Pocketfft.c2r_f32 ~shape_out:out_shape ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_normalized ~forward:false ~fct:1.0
        ~data_in ~data_out ~nthreads:1
  | Dtype.Complex128, Dtype.Float64 ->
      let data_in : (Complex.t, Dtype.complex64_elt, c_layout) Array1.t =
        x'.buffer
      in
      let data_out : (float, Dtype.float64_elt, c_layout) Array1.t =
        out.buffer
      in
      Pocketfft.c2r_f64 ~shape_out:out_shape ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_normalized ~forward:false ~fct:1.0
        ~data_in ~data_out ~nthreads:1
  | _ -> Error.failed ~op:"op_irfft" ~what:"unsupported dtype combination" ());

  out

(* ───── Linear Algebra Operations ───── *)

let op_cholesky ~upper x =
  (* Ensure input is materializable *)
  let x' = ensure_materializable x in

  (* Create output tensor with same shape and dtype *)
  let out_shape = shape x in
  let out = create_tensor x.context x.dtype out_shape in

  (* Convert to FFI tensors *)
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in

  (* Call FFI function *)
  caml_cholesky x_ffi out_ffi upper;

  out

let op_qr ~reduced x =
  let x' = ensure_materializable x in
  let x_shape = shape x in
  let m = x_shape.(Array.length x_shape - 2) in
  let n = x_shape.(Array.length x_shape - 1) in
  let k = min m n in

  (* Calculate Q and R shapes *)
  let q_shape = Array.copy x_shape in
  let r_shape = Array.copy x_shape in

  if reduced then (
    (* Reduced QR: Q is m×k, R is k×n *)
    q_shape.(Array.length q_shape - 1) <- k;
    r_shape.(Array.length r_shape - 2) <- k)
  else (
    (* Complete QR: Q is m×m, R is m×n *)
    q_shape.(Array.length q_shape - 1) <- m;
    (* R shape is already m×n from the copy *)
    ());

  let q = create_tensor x.context x.dtype q_shape in
  let r = create_tensor x.context x.dtype r_shape in

  let x_ffi = to_ffi_tensor x' in
  let q_ffi = to_ffi_tensor q in
  let r_ffi = to_ffi_tensor r in

  caml_qr x_ffi q_ffi r_ffi reduced;
  (q, r)

let op_svd (type a b) ~full_matrices (x : (a, b) t) :
    (a, b) t * (float, Dtype.float64_elt) t * (a, b) t =
  let x' = ensure_materializable x in
  let x_shape = shape x in
  let m = x_shape.(Array.length x_shape - 2) in
  let n = x_shape.(Array.length x_shape - 1) in
  let k = min m n in

  (* Calculate U, S, Vt shapes *)
  let batch_shape = Array.sub x_shape 0 (Array.length x_shape - 2) in

  let u_shape =
    Array.append batch_shape (if full_matrices then [| m; m |] else [| m; k |])
  in
  let s_shape = Array.append batch_shape [| k |] in
  let vt_shape =
    Array.append batch_shape (if full_matrices then [| n; n |] else [| k; n |])
  in

  let u = create_tensor x.context x.dtype u_shape in
  let s = create_tensor x.context Dtype.Float64 s_shape in
  let vt = create_tensor x.context x.dtype vt_shape in

  let x_ffi = to_ffi_tensor x' in
  let u_ffi = to_ffi_tensor u in
  let s_ffi = to_ffi_tensor s in
  let vt_ffi = to_ffi_tensor vt in

  caml_svd x_ffi u_ffi s_ffi vt_ffi full_matrices;
  (u, s, vt)

let op_eig (type a b) ~vectors (x : (a, b) t) :
    (Complex.t, Dtype.complex64_elt) t
    * (Complex.t, Dtype.complex64_elt) t option =
  let x' = ensure_materializable x in
  let x_shape = shape x in
  let n = x_shape.(Array.length x_shape - 1) in

  (* Eigenvalues and eigenvectors are always complex128 *)
  let batch_shape = Array.sub x_shape 0 (Array.length x_shape - 2) in
  let vals_shape = Array.append batch_shape [| n |] in
  let vecs_shape = x_shape in

  let vals = create_tensor x.context Dtype.Complex128 vals_shape in
  let vecs =
    if vectors then create_tensor x.context Dtype.Complex128 vecs_shape
    else
      (* Create dummy tensor for C interface *)
      create_tensor x.context Dtype.Complex128 [| 1 |]
  in

  let x_ffi = to_ffi_tensor x' in
  let vals_ffi = to_ffi_tensor vals in
  let vecs_ffi = to_ffi_tensor vecs in

  caml_eig x_ffi vals_ffi vecs_ffi false vectors;
  if vectors then (vals, Some vecs) else (vals, None)

let op_eigh (type a b) ~vectors (x : (a, b) t) :
    (float, Dtype.float64_elt) t * (a, b) t option =
  let x' = ensure_materializable x in
  let x_shape = shape x in

  (* For symmetric/hermitian matrices, eigenvalues are always float64 *)
  let batch_shape = Array.sub x_shape 0 (Array.length x_shape - 2) in
  let n = x_shape.(Array.length x_shape - 1) in
  let vals_shape = Array.append batch_shape [| n |] in

  let vals = create_tensor x.context Dtype.Float64 vals_shape in
  let vecs =
    if vectors then create_tensor x.context x.dtype x_shape
    else
      (* Create dummy tensor for C interface *)
      create_tensor x.context x.dtype [| 1 |]
  in

  let x_ffi = to_ffi_tensor x' in
  let vals_ffi = to_ffi_tensor vals in
  let vecs_ffi = to_ffi_tensor vecs in

  caml_eig x_ffi vals_ffi vecs_ffi true vectors;
  if vectors then (vals, Some vecs) else (vals, None)

let op_triangular_solve ~upper ~transpose ~unit_diag a b =
  let a' = ensure_materializable a in

  (* Handle 1D input b by expanding to 2D *)
  let b_shape = shape b in
  let b_ndim = Array.length b_shape in
  let b_is_1d = b_ndim = 1 in

  let b_expanded, out_shape =
    if b_is_1d then
      (* Expand 1D to 2D by adding a trailing dimension *)
      let new_shape = [| b_shape.(0); 1 |] in
      let b_reshaped = op_reshape b (Symbolic_shape.of_ints new_shape) in
      (b_reshaped, b_shape) (* Keep original shape for output *)
    else (b, shape b)
  in

  let b' = ensure_materializable b_expanded in

  (* Create output with appropriate shape *)
  let out_shape_expanded = shape b_expanded in
  let out_expanded = create_tensor b.context b.dtype out_shape_expanded in

  let a_ffi = to_ffi_tensor a' in
  let b_ffi = to_ffi_tensor b' in
  let out_ffi = to_ffi_tensor out_expanded in

  caml_triangular_solve a_ffi b_ffi out_ffi upper transpose unit_diag;

  (* Squeeze output back to 1D if input was 1D *)
  if b_is_1d then op_reshape out_expanded (Symbolic_shape.of_ints out_shape)
  else out_expanded

let op_as_strided t new_shape new_strides_in_elements offset_in_elements =
  (* C backend implementation: production-grade with zero-copy support *)

  (* Validate that the new view doesn't access out-of-bounds memory *)
  let buffer_size = Array1.dim t.buffer in
  let new_shape_arr =
    match Symbolic_shape.eval new_shape with
    | Some arr -> arr
    | None ->
        Error.failed ~op:"op_as_strided" ~what:"symbolic shapes not supported"
          ()
  in

  (* Calculate the maximum element accessed to ensure we stay within bounds *)
  let max_element_accessed = ref offset_in_elements in
  Array.iteri
    (fun i dim ->
      if dim > 0 then
        max_element_accessed :=
          max !max_element_accessed
            (offset_in_elements + ((dim - 1) * new_strides_in_elements.(i))))
    new_shape_arr;

  if !max_element_accessed >= buffer_size then
    Error.failed ~op:"op_as_strided"
      ~what:"view would access out-of-bounds memory" ();

  (* Create a new view with custom strides and offset - zero-copy operation *)
  let new_view =
    View.create new_shape ~strides:new_strides_in_elements
      ~offset:offset_in_elements
  in

  (* Return tensor with the same buffer but new view - no data copying *)
  { t with view = new_view }
