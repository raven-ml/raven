(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core

let err op fmt = Printf.ksprintf (fun msg -> invalid_arg (op ^ ": " ^ msg)) fmt

type ('a, 'b) buffer = ('a, 'b) Nx_buffer.t
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

external caml_atan2 :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_atan2"

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

external caml_sign : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_sign"

external caml_tan : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_tan"

external caml_asin : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_asin"

external caml_acos : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_acos"

external caml_atan : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_atan"

external caml_sinh : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_sinh"

external caml_cosh : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_cosh"

external caml_tanh : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_tanh"

external caml_trunc : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_trunc"

external caml_ceil : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_ceil"

external caml_floor : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_floor"

external caml_round : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_round"

external caml_erf : ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> unit
  = "caml_nx_erf"

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

external caml_argmax :
  ('a, 'b) ffi_tensor ->
  (int32, Dtype.int32_elt) ffi_tensor ->
  int ->
  bool ->
  unit = "caml_nx_argmax"

external caml_argmin :
  ('a, 'b) ffi_tensor ->
  (int32, Dtype.int32_elt) ffi_tensor ->
  int ->
  bool ->
  unit = "caml_nx_argmin"

external caml_sort :
  ('a, 'b) ffi_tensor -> ('a, 'b) ffi_tensor -> int -> bool -> unit
  = "caml_nx_sort"

external caml_argsort :
  ('a, 'b) ffi_tensor ->
  (int32, Dtype.int32_elt) ffi_tensor ->
  int ->
  bool ->
  unit = "caml_nx_argsort"

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
let shape t = View.shape t.view

let strides t =
  match View.strides_opt t.view with
  | Some s -> s
  | None -> invalid_arg "strides: cannot get strides for view"

let offset t = View.offset t.view
let is_contiguous t = View.is_c_contiguous t.view

(* Check if a tensor can be efficiently operated on *)
let can_get_strides t = View.can_get_strides t.view

(* Convert tensor to FFI representation if possible *)
let to_ffi_tensor t =
  if not (can_get_strides t) then
    invalid_arg "to_ffi_tensor: tensor has non-materializable view"
  else
    { data = t.buffer; shape = shape t; strides = strides t; offset = offset t }

(* Create a new tensor with given shape *)
let create_tensor ctx dtype shape_arr =
  let size = Array.fold_left ( * ) 1 shape_arr in
  let kind = Dtype.to_buffer_kind dtype in
  let buffer = Nx_buffer.create kind size in
  let view = View.create shape_arr in
  { context = ctx; dtype; buffer; view }

let buffer ctx dtype shape_arr =
  let kind = Dtype.to_buffer_kind dtype in
  let size = Array.fold_left ( * ) 1 shape_arr in
  let buffer = Nx_buffer.create kind size in
  let view = View.create shape_arr in
  { context = ctx; dtype; buffer; view }

let full ctx dtype shape_arr value =
  let t = buffer ctx dtype shape_arr in
  Nx_buffer.fill t.buffer value;
  t

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

(* Generic binary operation - allocates output and returns it *)
let binary_op op_name ffi_op x y =
  let x_shape = shape x in
  let y_shape = shape y in
  if x_shape <> y_shape then
    err op_name "shape mismatch: x %s, y %s" (Shape.to_string x_shape)
      (Shape.to_string y_shape)
  else
    let x' = ensure_materializable x in
    let y' = ensure_materializable y in
    let out = buffer () (dtype x) x_shape in
    let x_ffi = to_ffi_tensor x' in
    let y_ffi = to_ffi_tensor y' in
    let out_ffi = to_ffi_tensor out in
    ffi_op x_ffi y_ffi out_ffi;
    out

(* Comparison operation - allocates bool output and returns it *)
let comparison_op op_name ffi_op x y =
  let x_shape = shape x in
  let y_shape = shape y in
  if x_shape <> y_shape then
    err op_name "shape mismatch: x %s, y %s" (Shape.to_string x_shape)
      (Shape.to_string y_shape)
  else
    let x' = ensure_materializable x in
    let y' = ensure_materializable y in
    let out = buffer () Dtype.Bool x_shape in
    let x_ffi = to_ffi_tensor x' in
    let y_ffi = to_ffi_tensor y' in
    let out_ffi = to_ffi_tensor out in
    ffi_op x_ffi y_ffi out_ffi;
    out

(* ───── Buffer Allocation ───── *)

let from_host ctx array =
  let dtype = Dtype.of_buffer_kind (Nx_buffer.kind array) in
  let size = Nx_buffer.length array in
  (* Create a view for the 1D array *)
  let view = View.create [| size |] in
  (* Note: We're sharing the buffer directly, assuming it's contiguous *)
  { context = ctx; dtype; buffer = array; view }

(* Generic unary operation - allocates output and returns it *)
let unary_op _op_name ffi_op x =
  let x' = ensure_materializable x in
  let out = buffer () (dtype x) (shape x) in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in
  ffi_op x_ffi out_ffi;
  out

(* ───── Binary Operations ───── *)

let add x y = binary_op "add" caml_add x y
let sub x y = binary_op "sub" caml_sub x y
let mul x y = binary_op "mul" caml_mul x y
let max x y = binary_op "max" caml_max x y
let min x y = binary_op "min" caml_min x y
let mod_ x y = binary_op "mod" caml_mod x y
let pow x y = binary_op "pow" caml_pow x y
let xor x y = binary_op "xor" caml_xor x y
let or_ x y = binary_op "or" caml_or x y
let and_ x y = binary_op "and" caml_and x y
let atan2 y x = binary_op "atan2" caml_atan2 y x

(* ───── Comparison Operations ───── *)

let cmpeq x y = comparison_op "cmpeq" caml_cmpeq x y
let cmpne x y = comparison_op "cmpne" caml_cmpne x y
let cmplt x y = comparison_op "cmplt" caml_cmplt x y
let cmple x y = comparison_op "cmple" caml_cmple x y

(* ───── Unary Operations ───── *)

let neg x = unary_op "neg" caml_neg x
let log x = unary_op "log" caml_log x
let exp x = unary_op "exp" caml_exp x
let sin x = unary_op "sin" caml_sin x
let cos x = unary_op "cos" caml_cos x
let sqrt x = unary_op "sqrt" caml_sqrt x
let abs x = unary_op "abs" caml_abs x
let recip x = unary_op "recip" caml_recip x
let sign x = unary_op "sign" caml_sign x
let tan x = unary_op "tan" caml_tan x
let asin x = unary_op "asin" caml_asin x
let acos x = unary_op "acos" caml_acos x
let atan x = unary_op "atan" caml_atan x
let sinh x = unary_op "sinh" caml_sinh x
let cosh x = unary_op "cosh" caml_cosh x
let tanh x = unary_op "tanh" caml_tanh x
let trunc x = unary_op "trunc" caml_trunc x
let ceil x = unary_op "ceil" caml_ceil x
let floor x = unary_op "floor" caml_floor x
let round x = unary_op "round" caml_round x
let erf x = unary_op "erf" caml_erf x

(* Ternary Op - allocates output and returns it *)
let where cond if_true if_false =
  let cond_shape = shape cond in
  let if_true_shape = shape if_true in
  let if_false_shape = shape if_false in

  if cond_shape <> if_true_shape || if_true_shape <> if_false_shape then
    err "where" "shape mismatch: cond %s, if_true %s, if_false %s"
      (Shape.to_string cond_shape)
      (Shape.to_string if_true_shape)
      (Shape.to_string if_false_shape)
  else
    let cond' = ensure_materializable cond in
    let if_true' = ensure_materializable if_true in
    let if_false' = ensure_materializable if_false in
    let out = buffer () (dtype if_true) if_true_shape in
    let cond_ffi = to_ffi_tensor cond' in
    let if_true_ffi = to_ffi_tensor if_true' in
    let if_false_ffi = to_ffi_tensor if_false' in
    let out_ffi = to_ffi_tensor out in
    caml_where cond_ffi if_true_ffi if_false_ffi out_ffi;
    out

(* Reduction Ops - allocates output and returns it *)
let reduce_op _op_name ffi_op ~axes ~keepdims x =
  let input_shape = shape x in
  let ndim = Array.length input_shape in

  if ndim = 0 then begin
    let out = buffer () (dtype x) [||] in
    Nx_buffer.set out.buffer 0 (Nx_buffer.get x.buffer 0);
    out
  end
  else
    let normalized_axes =
      Array.map (fun ax -> if ax < 0 then ax + ndim else ax) axes
    in
    let out_shape =
      Shape.reduce_output_shape input_shape normalized_axes keepdims
    in
    let out = buffer () (dtype x) out_shape in
    let x' = ensure_materializable x in
    let x_ffi = to_ffi_tensor x' in
    let out_ffi = to_ffi_tensor out in
    ffi_op x_ffi out_ffi normalized_axes keepdims;
    out

let reduce_sum ~axes ~keepdims x =
  reduce_op "reduce_sum" caml_reduce_sum ~axes ~keepdims x

let reduce_max ~axes ~keepdims x =
  reduce_op "reduce_max" caml_reduce_max ~axes ~keepdims x

let reduce_prod ~axes ~keepdims x =
  reduce_op "reduce_prod" caml_reduce_prod ~axes ~keepdims x

let reduce_min ~axes ~keepdims x =
  reduce_op "reduce_min" caml_reduce_min ~axes ~keepdims x

let associative_scan ~axis ~op x =
  let x_shape = shape x in
  let rank = Array.length x_shape in
  if rank = 0 then invalid_arg "associative_scan: requires rank >= 1"
  else
    let axis = if axis < 0 then axis + rank else axis in
    if axis < 0 || axis >= rank then
      err "associative_scan" "axis %d out of bounds for rank %d" axis rank
    else
      let x' = ensure_materializable x in
      let out = buffer () (dtype x) x_shape in
      let x_ffi = to_ffi_tensor x' in
      let out_ffi = to_ffi_tensor out in
      let op_tag =
        match op with `Sum -> 0 | `Prod -> 1 | `Max -> 2 | `Min -> 3
      in
      caml_associative_scan x_ffi out_ffi axis op_tag;
      out

(* Movement Ops - These are view-only operations *)
let expand x shape = { x with view = View.expand x.view shape }
let reshape x shape = { x with view = View.reshape x.view shape }
let permute x axes = { x with view = View.permute x.view axes }

let pad x padding fill_value =
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

let shrink x bounds = { x with view = View.shrink x.view bounds }
let flip x axes = { x with view = View.flip x.view axes }

let cat tensors ~axis =
  match tensors with
  | [] -> invalid_arg "cat: empty tensor list"
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
      let out = buffer () (dtype first) out_shape in
      let tensors_ffi = List.map to_ffi_tensor tensors' in
      let out_ffi = to_ffi_tensor out in
      caml_cat tensors_ffi norm_axis out_ffi;
      out

(* ───── Other Ops ───── *)

let cast (type a b c d) ~(dtype : (c, d) Dtype.t) (x : (a, b) t) : (c, d) t =
  let x' = ensure_materializable x in
  let out = buffer () dtype (shape x) in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in
  caml_cast x_ffi out_ffi;
  out

let contiguous x =
  (* Check if already contiguous with no offset and no broadcast dimensions *)
  let strides_arr = strides x in
  let has_broadcast = Array.exists (( = ) 0) strides_arr in
  if is_contiguous x && offset x = 0 && not has_broadcast then x
  else
    let x' = ensure_materializable x in
    let x_ffi = to_ffi_tensor x' in
    let out_ffi = caml_contiguous x_ffi in
    (* Create tensor from FFI result - it's contiguous so simple view *)
    let view = View.create out_ffi.shape in
    { context = x.context; dtype = x.dtype; buffer = out_ffi.data; view }

let copy x =
  let x' = ensure_materializable x in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = caml_copy x_ffi in
  (* Create tensor from FFI result - it's contiguous so simple view *)
  let view = View.create out_ffi.shape in
  { context = x.context; dtype = x.dtype; buffer = out_ffi.data; view }

let assign dst src =
  let src' = ensure_materializable src in
  (* dst doesn't need materialization - we're writing to it *)
  let src_ffi = to_ffi_tensor src' in
  let dst_ffi = to_ffi_tensor dst in
  caml_assign src_ffi dst_ffi

let threefry key counter =
  let key' = ensure_materializable key in
  let counter' = ensure_materializable counter in
  let out = buffer () Dtype.Int32 (shape counter) in
  let key_ffi = to_ffi_tensor key' in
  let counter_ffi = to_ffi_tensor counter' in
  let out_ffi = to_ffi_tensor out in
  caml_threefry key_ffi counter_ffi out_ffi;
  out

(* ───── Element Access Ops ───── *)

let gather data indices ~axis =
  (* Ensure inputs are materializable. Preserve broadcasted strides for indices
     to enable C fast paths (e.g., memcpy row gather). *)
  let data' = ensure_materializable data in
  (* Do not materialize indices unless we cannot get strides *)
  let indices' =
    if can_get_strides indices then indices else ensure_materializable indices
  in
  let out = buffer () (dtype data) (shape indices) in
  let data_ffi = to_ffi_tensor data' in
  let indices_ffi = to_ffi_tensor indices' in
  let out_ffi = to_ffi_tensor out in
  caml_gather data_ffi indices_ffi out_ffi axis;
  out

let scatter ?(mode = `Set) ?(unique_indices = false) data_template ~indices
    ~updates ~axis =
  (* Ensure inputs are materializable *)
  let template' = ensure_materializable data_template in
  let indices' = ensure_materializable indices in
  let updates' = ensure_materializable updates in

  (* Create output tensor - for Set mode, start with a copy of template *)
  let out =
    if mode = `Set then copy data_template (* Start with copy of template *)
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

let unfold x ~kernel_size ~stride ~dilation ~padding =
  let x' = ensure_materializable x in
  let in_shape = shape x in
  let k = Array.length kernel_size in
  let leading_ndim = Array.length in_shape - k in
  let leading_shape = Array.sub in_shape 0 leading_ndim in
  let spatial_dims = Array.sub in_shape leading_ndim k in

  let padding_flat =
    Array.init
      (Array.length padding * 2)
      (fun i ->
        let dim = i / 2 in
        if i mod 2 = 0 then fst padding.(dim) else snd padding.(dim))
  in

  let out_spatial =
    Array.init k (fun i ->
        let pad_before, pad_after = padding.(i) in
        let padded = spatial_dims.(i) + pad_before + pad_after in
        let kernel_extent = (dilation.(i) * (kernel_size.(i) - 1)) + 1 in
        let diff = padded - kernel_extent in
        if diff < 0 then
          invalid_arg "unfold: kernel size larger than padded input"
        else (diff / stride.(i)) + 1)
  in

  let kernel_prod = Array.fold_left ( * ) 1 kernel_size in
  let spatial_prod = Array.fold_left ( * ) 1 out_spatial in
  let out_shape =
    Array.concat [ leading_shape; [| kernel_prod; spatial_prod |] ]
  in

  let out = create_tensor x.context x.dtype out_shape in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in
  caml_unfold x_ffi kernel_size stride dilation padding_flat out_ffi;
  out

let fold x ~output_size ~kernel_size ~stride ~dilation ~padding =
  let x' = ensure_materializable x in
  let in_shape = shape x in
  let leading_ndim = Array.length in_shape - 2 in
  let leading_shape = Array.sub in_shape 0 leading_ndim in

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
          invalid_arg "fold: kernel size larger than padded output"
        else (diff / stride.(i)) + 1)
  in

  let out_shape = Array.concat [ leading_shape; output_size ] in

  let out = create_tensor x.context x.dtype out_shape in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in
  caml_fold x_ffi output_size kernel_size stride dilation padding_flat out_ffi;
  out

let matmul x y =
  let x' = if is_contiguous x then x else ensure_materializable x in
  let y' = if is_contiguous y then y else ensure_materializable y in
  let x_shape = shape x in
  let y_shape = shape y in
  let x_ndim = Array.length x_shape in
  let y_ndim = Array.length y_shape in
  let m = x_shape.(x_ndim - 2) in
  let n = y_shape.(y_ndim - 1) in
  let max_ndim = Int.max x_ndim y_ndim in
  let batch_nd = max_ndim - 2 in
  let batch_dims =
    Array.init batch_nd (fun i ->
        let a_idx = i - (max_ndim - x_ndim) in
        let b_idx = i - (max_ndim - y_ndim) in
        let sa = if a_idx >= 0 then x_shape.(a_idx) else 1 in
        let sb = if b_idx >= 0 then y_shape.(b_idx) else 1 in
        Int.max sa sb)
  in
  let out_shape = Array.append batch_dims [| m; n |] in
  let out = buffer () (dtype x) out_shape in
  let x_ffi = to_ffi_tensor x' in
  let y_ffi = to_ffi_tensor y' in
  let out_ffi = to_ffi_tensor out in
  caml_matmul x_ffi y_ffi out_ffi;
  out

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

let fft (type a b) ?out (x : (a, b) t) ~axes : (a, b) t =
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
        ~data_in:(Nx_buffer.to_bigarray1 x'.buffer)
        ~data_out:(Nx_buffer.to_bigarray1 out.buffer)
        ~nthreads:1
  | Dtype.Complex128 ->
      Pocketfft.c2c_f64 ~shape:shape_arr ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_arr ~forward:true ~fct:1.0
        ~data_in:(Nx_buffer.to_bigarray1 x'.buffer)
        ~data_out:(Nx_buffer.to_bigarray1 out.buffer)
        ~nthreads:1
  | _ -> invalid_arg "fft: unsupported dtype");

  out

let ifft (type a b) ?out (x : (a, b) t) ~axes : (a, b) t =
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
        ~data_in:(Nx_buffer.to_bigarray1 x'.buffer)
        ~data_out:(Nx_buffer.to_bigarray1 out.buffer)
        ~nthreads:1
  | Dtype.Complex128 ->
      Pocketfft.c2c_f64 ~shape:shape_arr ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_arr ~forward:false ~fct:1.0
        ~data_in:(Nx_buffer.to_bigarray1 x'.buffer)
        ~data_out:(Nx_buffer.to_bigarray1 out.buffer)
        ~nthreads:1
  | _ -> invalid_arg "ifft: unsupported dtype");

  out

let rfft (type a b c d) ?out (x : (a, b) t) ~(dtype : (c, d) Dtype.t) ~axes :
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
      Pocketfft.r2c_f32 ~shape_in:in_shape ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_normalized ~forward:true ~fct:1.0
        ~data_in:(Nx_buffer.to_bigarray1 x'.buffer)
        ~data_out:(Nx_buffer.to_bigarray1 out.buffer)
        ~nthreads:1
  | Dtype.Float64, Dtype.Complex128 ->
      Pocketfft.r2c_f64 ~shape_in:in_shape ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_normalized ~forward:true ~fct:1.0
        ~data_in:(Nx_buffer.to_bigarray1 x'.buffer)
        ~data_out:(Nx_buffer.to_bigarray1 out.buffer)
        ~nthreads:1
  | _ -> invalid_arg "rfft: unsupported dtype combination");

  out

let irfft (type a b c d) ?out ?s (x : (a, b) t) ~(dtype : (c, d) Dtype.t) ~axes
    : (c, d) t =
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
      Pocketfft.c2r_f32 ~shape_out:out_shape ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_normalized ~forward:false ~fct:1.0
        ~data_in:(Nx_buffer.to_bigarray1 x'.buffer)
        ~data_out:(Nx_buffer.to_bigarray1 out.buffer)
        ~nthreads:1
  | Dtype.Complex128, Dtype.Float64 ->
      Pocketfft.c2r_f64 ~shape_out:out_shape ~stride_in:strides_in
        ~stride_out:strides_out ~axes:axes_normalized ~forward:false ~fct:1.0
        ~data_in:(Nx_buffer.to_bigarray1 x'.buffer)
        ~data_out:(Nx_buffer.to_bigarray1 out.buffer)
        ~nthreads:1
  | _ -> invalid_arg "irfft: unsupported dtype combination");

  out

(* ───── Linear Algebra Operations ───── *)

let cholesky ~upper x =
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

let qr ~reduced x =
  let x' = ensure_materializable x in
  let x_shape = shape x in
  let m = x_shape.(Array.length x_shape - 2) in
  let n = x_shape.(Array.length x_shape - 1) in
  let k = Stdlib.min m n in

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

let svd (type a b) ~full_matrices (x : (a, b) t) :
    (a, b) t * (float, Dtype.float64_elt) t * (a, b) t =
  let x' = ensure_materializable x in
  let x_shape = shape x in
  let m = x_shape.(Array.length x_shape - 2) in
  let n = x_shape.(Array.length x_shape - 1) in
  let k = Stdlib.min m n in

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

let eig (type a b) ~vectors (x : (a, b) t) :
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

let eigh (type a b) ~vectors (x : (a, b) t) :
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

let triangular_solve ~upper ~transpose ~unit_diag a b =
  let a' = ensure_materializable a in

  (* Handle 1D input b by expanding to 2D *)
  let b_shape = shape b in
  let b_ndim = Array.length b_shape in
  let b_is_1d = b_ndim = 1 in

  let b_expanded, out_shape =
    if b_is_1d then
      (* Expand 1D to 2D by adding a trailing dimension *)
      let new_shape = [| b_shape.(0); 1 |] in
      let b_reshaped = reshape b new_shape in
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
  if b_is_1d then reshape out_expanded out_shape else out_expanded

let div x y =
  let dt = dtype x in
  if Dtype.is_int dt || Dtype.is_uint dt then binary_op "idiv" caml_idiv x y
  else binary_op "fdiv" caml_fdiv x y

let argmax ~axis ~keepdims x =
  let x' = ensure_materializable x in
  let out_shape = Shape.reduce_output_shape (shape x) [| axis |] keepdims in
  let out = buffer () Dtype.Int32 out_shape in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in
  caml_argmax x_ffi out_ffi axis keepdims;
  out

let argmin ~axis ~keepdims x =
  let x' = ensure_materializable x in
  let out_shape = Shape.reduce_output_shape (shape x) [| axis |] keepdims in
  let out = buffer () Dtype.Int32 out_shape in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in
  caml_argmin x_ffi out_ffi axis keepdims;
  out

let sort ~axis ~descending x =
  let x' = ensure_materializable x in
  let out = buffer () (dtype x) (shape x) in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in
  caml_sort x_ffi out_ffi axis descending;
  out

let argsort ~axis ~descending x =
  let x' = ensure_materializable x in
  let out = buffer () Dtype.Int32 (shape x) in
  let x_ffi = to_ffi_tensor x' in
  let out_ffi = to_ffi_tensor out in
  caml_argsort x_ffi out_ffi axis descending;
  out
