open Nx_core
module Shape_expr = Rune_jit.Shape_expr

module Symbolic_id = struct
  type t = int

  let counter = ref 0

  let fresh () =
    incr counter;
    !counter

  let compare = Int.compare
  let equal = Int.equal
  let hash = Hashtbl.hash
  let pp fmt v = Format.fprintf fmt "sym%d" v
end

type context = Native_context : Nx_c.context -> context
type device_type = Ocaml | C | Metal

(* Backend interface requires type ('a, 'b) t *)
type ('a, 'b) t =
  | Native_tensor : ('a, 'b) Nx_c.t -> ('a, 'b) t
  | Symbolic_tensor : {
      id : Symbolic_id.t;
      dtype : ('a, 'b) Dtype.t;
      shape : Symbolic_shape.t;
    }
      -> ('a, 'b) t

(* Effects - no context in most operations per new backend interface *)
type _ Effect.t +=
  | E_view : ('a, 'b) t -> View.t Effect.t
  | E_buffer : {
      context : context;
      dtype : ('a, 'b) Dtype.t;
      size_in_elements : int;
    }
      -> ('a, 'b) t Effect.t
  | E_const_scalar : {
      context : context;
      value : 'a;
      dtype : ('a, 'b) Dtype.t;
    }
      -> ('a, 'b) t Effect.t
  | E_const_array : {
      context : context;
      array : ('a, 'b, Bigarray_ext.c_layout) Bigarray_ext.Array1.t;
    }
      -> ('a, 'b) t Effect.t
  | E_add : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_mul : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_idiv : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_fdiv : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_max : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_mod : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_pow : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_cmplt : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> (bool, Dtype.bool_elt) t Effect.t
  | E_cmpne : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> (bool, Dtype.bool_elt) t Effect.t
  | E_xor : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_or : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_and : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_neg : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_log2 : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_exp2 : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sin : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sqrt : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_recip : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_where : {
      condition : (bool, Dtype.bool_elt) t;
      if_true : ('a, 'b) t;
      if_false : ('a, 'b) t;
    }
      -> ('a, 'b) t Effect.t
  | E_reduce_sum : {
      t_in : ('a, 'b) t;
      axes : int array;
      keepdims : bool;
    }
      -> ('a, 'b) t Effect.t
  | E_reduce_max : {
      t_in : ('a, 'b) t;
      axes : int array;
      keepdims : bool;
    }
      -> ('a, 'b) t Effect.t
  | E_reduce_prod : {
      t_in : ('a, 'b) t;
      axes : int array;
      keepdims : bool;
    }
      -> ('a, 'b) t Effect.t
  | E_associative_scan : {
      t_in : ('a, 'b) t;
      axis : int;
      op : [ `Sum | `Prod | `Max | `Min ];
    }
      -> ('a, 'b) t Effect.t
  | E_permute : { t_in : ('a, 'b) t; axes : int array } -> ('a, 'b) t Effect.t
  | E_reshape : {
      t_in : ('a, 'b) t;
      new_shape : Symbolic_shape.t;
    }
      -> ('a, 'b) t Effect.t
  | E_expand : {
      t_in : ('a, 'b) t;
      new_target_shape : Symbolic_shape.t;
    }
      -> ('a, 'b) t Effect.t
  | E_pad : {
      t_in : ('a, 'b) t;
      padding_config : (int * int) array;
      fill_value : 'a;
    }
      -> ('a, 'b) t Effect.t
  | E_shrink : {
      t_in : ('a, 'b) t;
      limits : (int * int) array;
    }
      -> ('a, 'b) t Effect.t
  | E_as_strided : {
      t_in : ('a, 'b) t;
      new_shape : int array;
      new_strides : int array;
      offset : int;
    }
      -> ('a, 'b) t Effect.t
  | E_flip : {
      t_in : ('a, 'b) t;
      dims_to_flip : bool array;
    }
      -> ('a, 'b) t Effect.t
  | E_cat : { t_list : ('a, 'b) t list; axis : int } -> ('a, 'b) t Effect.t
  | E_cast : {
      t_in : ('a, 'b) t;
      target_dtype : ('c, 'd) Dtype.t;
    }
      -> ('c, 'd) t Effect.t
  | E_contiguous : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_copy : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_assign : { dst : ('a, 'b) t; src : ('a, 'b) t } -> unit Effect.t
  | E_threefry : {
      key : (int32, Dtype.int32_elt) t;
      ctr : (int32, Dtype.int32_elt) t;
    }
      -> (int32, Dtype.int32_elt) t Effect.t
  | E_gather : {
      data : ('a, 'b) t;
      indices : (int32, Dtype.int32_elt) t;
      axis : int;
    }
      -> ('a, 'b) t Effect.t
  | E_scatter : {
      data_template : ('a, 'b) t;
      indices : (int32, Dtype.int32_elt) t;
      updates : ('a, 'b) t;
      axis : int;
    }
      -> ('a, 'b) t Effect.t
  | E_to_device : {
      context : context;
      t_in : ('a, 'b) t;
    }
      -> ('a, 'b) t Effect.t
  | E_unfold : {
      t_in : ('a, 'b) t;
      kernel_size : int array;
      stride : int array;
      dilation : int array;
      padding : (int * int) array;
    }
      -> ('a, 'b) t Effect.t
  | E_fold : {
      t_in : ('a, 'b) t;
      output_size : int array;
      kernel_size : int array;
      stride : int array;
      dilation : int array;
      padding : (int * int) array;
    }
      -> ('a, 'b) t Effect.t
  | E_matmul : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_fft : {
      t : (Complex.t, 'b) t;
      axes : int array;
    }
      -> (Complex.t, 'b) t Effect.t
  | E_ifft : {
      t : (Complex.t, 'b) t;
      axes : int array;
    }
      -> (Complex.t, 'b) t Effect.t
  | E_rfft : {
      t : (float, 'b) t;
      axes : int array;
    }
      -> (Complex.t, Dtype.complex64_elt) t Effect.t
  | E_irfft : {
      t : (Complex.t, 'b) t;
      axes : int array;
      s : int array option;
    }
      -> (float, Dtype.float64_elt) t Effect.t
  | E_psum : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_qr : {
      t_in : ('a, 'b) t;
      reduced : bool;
    }
    -> (('a, 'b) t * ('a, 'b) t) Effect.t

(* Native_context creation *)
let create_context () : context = Native_context (Nx_c.create_context ())

(* Extract context from tensor *)
let context : type a b. (a, b) t -> context = function
  | Native_tensor cpu_t -> Native_context (Nx_c.context cpu_t)
  | Symbolic_tensor _ -> failwith "Symbolic tensors do not have a context"

(* Device transfer operations *)
let to_device (target_ctx : context) (t : ('a, 'b) t) : ('a, 'b) t =
  match (target_ctx, t) with
  (* Already on correct device *)
  | Native_context _, Native_tensor _ -> t
  (* Symbolic tensors update their context *)
  | _, Symbolic_tensor _ -> failwith "Cannot transfer symbolic tensor to device"

(* Lenses *)
let view (type a b) (x : (a, b) t) : View.t =
  try Effect.perform (E_view x)
  with Effect.Unhandled _ -> (
    match x with
    | Native_tensor t -> Nx_c.view t
    | Symbolic_tensor { shape; _ } -> View.create shape)

let dtype : type a b. (a, b) t -> (a, b) Dtype.t = function
  | Native_tensor t -> Nx_c.dtype t
  | Symbolic_tensor { dtype; _ } -> dtype

let is_symbolic = function Symbolic_tensor _ -> true | _ -> false

let shape_expr_of_symbolic (sym_shape : Symbolic_shape.t) : Shape_expr.shape =
  let cache = Hashtbl.create 8 in
  let rec convert = function
    | Symbolic_shape.Const n -> Shape_expr.const n
    | Symbolic_shape.Var v ->
        let id = Symbolic_shape.var_id v in
        let var =
          match Hashtbl.find_opt cache id with
          | Some var -> var
          | None ->
              let name = Symbolic_shape.var_name v in
              let min, max = Symbolic_shape.var_bounds v in
              let var = Shape_expr.Var.create ~id ~name ~min ~max in
              Hashtbl.add cache id var;
              var
        in
        Shape_expr.var var
    | Symbolic_shape.Add (a, b) -> Shape_expr.add (convert a) (convert b)
    | Symbolic_shape.Mul (a, b) -> Shape_expr.mul (convert a) (convert b)
    | Symbolic_shape.Neg e -> Shape_expr.neg (convert e)
  in
  Array.map convert sym_shape

let view_shape_expr view = shape_expr_of_symbolic (View.shape view)
let view_shape_eval view = Symbolic_shape.eval (View.shape view)
let shape_upper_bound shape_expr = Shape_expr.upper_bounds shape_expr

let data : type a b.
    (a, b) t -> (a, b, Bigarray_ext.c_layout) Bigarray_ext.Array1.t = function
  | Native_tensor t -> Nx_c.data t
  | Symbolic_tensor { id; _ } ->
      failwith (Printf.sprintf "Cannot extract data from symbolic tensor %d" id)

(* Helper functions for different operation types *)

let ensure_same_device a b =
  match (a, b) with
  | Native_tensor _, Native_tensor _ -> (a, b)
  | _ ->
      (* Convert b to a's device *)
      let ctx = context a in
      (a, to_device ctx b)

let binary_op eff cpu_op a b =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    let a', b' = ensure_same_device a b in
    match (a', b') with
    | Native_tensor t1, Native_tensor t2 -> Native_tensor (cpu_op t1 t2)
    | _ -> assert false)

let unary_op eff cpu_op t_in =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t -> Native_tensor (cpu_op t)
    | Symbolic_tensor _ ->
        failwith "Cannot perform operation on symbolic tensor")

let comparison_op eff cpu_op a b =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    let a', b' = ensure_same_device a b in
    match (a', b') with
    | Native_tensor t1, Native_tensor t2 -> Native_tensor (cpu_op t1 t2)
    | _ -> assert false)

let reduce_op eff cpu_op ~axes ~keepdims t_in =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t -> Native_tensor (cpu_op ~axes ~keepdims t)
    | Symbolic_tensor _ ->
        failwith "Cannot perform reduction on symbolic tensor")

let shape_op1 eff cpu_op symbolic_shape t_in =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t -> Native_tensor (cpu_op t symbolic_shape)
    | Symbolic_tensor _ ->
        failwith "Cannot perform shape operation on symbolic tensor")

let axes_op1 eff cpu_op t_in axes_arg =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t -> Native_tensor (cpu_op t axes_arg)
    | Symbolic_tensor _ ->
        failwith "Cannot perform axes operation on symbolic tensor")

let limits_op1 eff cpu_op t_in limits_arg =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t -> Native_tensor (cpu_op t limits_arg)
    | Symbolic_tensor _ ->
        failwith "Cannot perform limits operation on symbolic tensor")

let bool_array_op1 eff cpu_op t_in bool_arg =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t -> Native_tensor (cpu_op t bool_arg)
    | Symbolic_tensor _ ->
        failwith "Cannot perform bool array operation on symbolic tensor")

let ternary_op eff cpu_op cond if_true if_false =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    (* Ensure all three tensors are on the same device *)
    let ctx = context cond in
    let cond' = to_device ctx cond in
    let if_true' = to_device ctx if_true in
    let if_false' = to_device ctx if_false in
    match (cond', if_true', if_false') with
    | Native_tensor t1, Native_tensor t2, Native_tensor t3 ->
        Native_tensor (cpu_op t1 t2 t3)
    | _ -> assert false)

(* Binary operations *)
let op_add a b = binary_op (fun () -> E_add { a; b }) Nx_c.op_add a b
let op_mul a b = binary_op (fun () -> E_mul { a; b }) Nx_c.op_mul a b
let op_idiv a b = binary_op (fun () -> E_idiv { a; b }) Nx_c.op_idiv a b
let op_fdiv a b = binary_op (fun () -> E_fdiv { a; b }) Nx_c.op_fdiv a b
let op_max a b = binary_op (fun () -> E_max { a; b }) Nx_c.op_max a b
let op_mod a b = binary_op (fun () -> E_mod { a; b }) Nx_c.op_mod a b
let op_pow a b = binary_op (fun () -> E_pow { a; b }) Nx_c.op_pow a b
let op_xor a b = binary_op (fun () -> E_xor { a; b }) Nx_c.op_xor a b
let op_or a b = binary_op (fun () -> E_or { a; b }) Nx_c.op_or a b
let op_and a b = binary_op (fun () -> E_and { a; b }) Nx_c.op_and a b

(* Comparison operations *)
let op_cmplt a b = comparison_op (fun () -> E_cmplt { a; b }) Nx_c.op_cmplt a b
let op_cmpne a b = comparison_op (fun () -> E_cmpne { a; b }) Nx_c.op_cmpne a b

(* Unary operations *)
let op_neg t_in = unary_op (fun () -> E_neg { t_in }) Nx_c.op_neg t_in
let op_log2 t_in = unary_op (fun () -> E_log2 { t_in }) Nx_c.op_log2 t_in
let op_exp2 t_in = unary_op (fun () -> E_exp2 { t_in }) Nx_c.op_exp2 t_in
let op_sin t_in = unary_op (fun () -> E_sin { t_in }) Nx_c.op_sin t_in
let op_sqrt t_in = unary_op (fun () -> E_sqrt { t_in }) Nx_c.op_sqrt t_in
let op_recip t_in = unary_op (fun () -> E_recip { t_in }) Nx_c.op_recip t_in

(* Collective primitive: parallel sum across mapped axis, to be handled by
   vmap. *)
let op_psum t_in =
  try Effect.perform (E_psum { t_in })
  with Effect.Unhandled _ -> failwith "psum must be used under vmap"

(* Reduction operations *)
let op_reduce_sum ~axes ~keepdims t_in =
  reduce_op
    (fun () -> E_reduce_sum { t_in; axes; keepdims })
    Nx_c.op_reduce_sum ~axes ~keepdims t_in

let op_reduce_max ~axes ~keepdims t_in =
  reduce_op
    (fun () -> E_reduce_max { t_in; axes; keepdims })
    Nx_c.op_reduce_max ~axes ~keepdims t_in

let op_reduce_prod ~axes ~keepdims t_in =
  reduce_op
    (fun () -> E_reduce_prod { t_in; axes; keepdims })
    Nx_c.op_reduce_prod ~axes ~keepdims t_in

let op_associative_scan ~axis ~op t_in =
  try Effect.perform (E_associative_scan { t_in; axis; op })
  with Effect.Unhandled _ -> (
    match to_device (context t_in) t_in with
    | Native_tensor t -> Native_tensor (Nx_c.op_associative_scan ~axis ~op t)
    | Symbolic_tensor _ ->
        failwith "Cannot perform associative_scan on symbolic tensor")

(* Shape operations *)
let op_reshape t_in new_shape =
  shape_op1
    (fun () -> E_reshape { t_in; new_shape })
    Nx_c.op_reshape new_shape t_in

let op_expand t_in new_target_shape =
  shape_op1
    (fun () -> E_expand { t_in; new_target_shape })
    Nx_c.op_expand new_target_shape t_in

let op_permute t_in axes =
  axes_op1 (fun () -> E_permute { t_in; axes }) Nx_c.op_permute t_in axes

let op_shrink t_in limits =
  limits_op1 (fun () -> E_shrink { t_in; limits }) Nx_c.op_shrink t_in limits

let op_flip t_in dims_to_flip =
  bool_array_op1
    (fun () -> E_flip { t_in; dims_to_flip })
    Nx_c.op_flip t_in dims_to_flip

(* Pad operation (needs special handling for fill_value) *)
let op_pad t_in padding_config fill_value =
  try Effect.perform (E_pad { t_in; padding_config; fill_value })
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t -> Native_tensor (Nx_c.op_pad t padding_config fill_value)
    | Symbolic_tensor _ -> failwith "Cannot pad symbolic tensor")

(* Creation operations *)
let op_buffer ctx dtype size_in_elements =
  try Effect.perform (E_buffer { context = ctx; dtype; size_in_elements })
  with Effect.Unhandled _ -> (
    match ctx with
    | Native_context ctx ->
        Native_tensor (Nx_c.op_buffer ctx dtype size_in_elements))

let op_const_scalar ctx value dtype =
  try Effect.perform (E_const_scalar { context = ctx; value; dtype })
  with Effect.Unhandled _ -> (
    match ctx with
    | Native_context ctx -> Native_tensor (Nx_c.op_const_scalar ctx value dtype))

let op_const_array ctx array =
  try Effect.perform (E_const_array { context = ctx; array })
  with Effect.Unhandled _ -> (
    match ctx with
    | Native_context ctx -> Native_tensor (Nx_c.op_const_array ctx array))

(* Copy operations *)
let op_contiguous t_in =
  unary_op (fun () -> E_contiguous { t_in }) Nx_c.op_contiguous t_in

let op_copy t_in = unary_op (fun () -> E_copy { t_in }) Nx_c.op_copy t_in

(* Where operation *)
let op_where condition if_true if_false =
  ternary_op
    (fun () -> E_where { condition; if_true; if_false })
    Nx_c.op_where condition if_true if_false

(* Cat operation (special handling for lists) *)
let op_cat t_list axis =
  try Effect.perform (E_cat { t_list; axis })
  with Effect.Unhandled _ -> (
    if List.length t_list = 0 then failwith "op_cat: empty list"
    else
      let first = List.hd t_list in
      let ctx = context first in
      let converted = List.map (to_device ctx) t_list in
      match ctx with
      | Native_context _ ->
          let cpu_list =
            List.map
              (function Native_tensor t -> t | _ -> assert false)
              converted
          in
          Native_tensor (Nx_c.op_cat cpu_list axis))

(* Cast operation *)
let op_cast : type a b c d. (a, b) t -> (c, d) Dtype.t -> (c, d) t =
 fun t_in target_dtype ->
  try Effect.perform (E_cast { t_in; target_dtype })
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t -> Native_tensor (Nx_c.op_cast t target_dtype)
    | Symbolic_tensor _ -> failwith "Cannot cast symbolic tensor")

(* Assign operation *)
let op_assign dst src =
  try Effect.perform (E_assign { dst; src })
  with Effect.Unhandled _ -> (
    let dst', src' = ensure_same_device dst src in
    match (dst', src') with
    | Native_tensor d, Native_tensor s -> Nx_c.op_assign d s
    | _ -> assert false)

(* Gather operation *)
let op_gather data indices axis =
  try Effect.perform (E_gather { data; indices; axis })
  with Effect.Unhandled _ -> (
    let data', indices' = ensure_same_device data indices in
    match (data', indices') with
    | Native_tensor d, Native_tensor i ->
        Native_tensor (Nx_c.op_gather d i axis)
    | _ -> assert false)

(* Scatter operation *)
let op_scatter ?(mode = `Set) ?(unique_indices = false) data_template indices
    updates axis =
  try Effect.perform (E_scatter { data_template; indices; updates; axis })
  with Effect.Unhandled _ -> (
    (* Ensure all three tensors are on the same device *)
    let ctx = context data_template in
    let tmpl = to_device ctx data_template in
    let idx = to_device ctx indices in
    let upd = to_device ctx updates in
    match (tmpl, idx, upd) with
    | Native_tensor t, Native_tensor i, Native_tensor u ->
        Native_tensor (Nx_c.op_scatter ~mode ~unique_indices t i u axis)
    | _ -> assert false)

(* Threefry operation *)
let op_threefry key ctr =
  binary_op (fun () -> E_threefry { key; ctr }) Nx_c.op_threefry key ctr

(* Unfold operation *)
let op_unfold t_in ~kernel_size ~stride ~dilation ~padding =
  try Effect.perform (E_unfold { t_in; kernel_size; stride; dilation; padding })
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t ->
        Native_tensor (Nx_c.op_unfold t ~kernel_size ~stride ~dilation ~padding)
    | Symbolic_tensor _ -> failwith "todo: op_unfold for symbolic tensors")

(* Fold operation *)
let op_fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding =
  try
    Effect.perform
      (E_fold { t_in; output_size; kernel_size; stride; dilation; padding })
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t ->
        Native_tensor
          (Nx_c.op_fold t ~output_size ~kernel_size ~stride ~dilation ~padding)
    | Symbolic_tensor _ -> failwith "todo: op_fold for symbolic tensors")

(* Matrix multiplication *)
let op_matmul a b =
  try Effect.perform (E_matmul { a; b })
  with Effect.Unhandled _ -> (
    let a', b' = ensure_same_device a b in
    match (a', b') with
    | Native_tensor a_t, Native_tensor b_t ->
        Native_tensor (Nx_c.op_matmul a_t b_t)
    | Symbolic_tensor _, _ | _, Symbolic_tensor _ ->
        failwith "todo: op_matmul for symbolic tensors")

(* FFT operations *)
let op_fft t ~axes =
  try Effect.perform (E_fft { t; axes })
  with Effect.Unhandled _ -> (
    match t with
    | Native_tensor t -> Native_tensor (Nx_c.op_fft t ~axes)
    | Symbolic_tensor _ -> failwith "todo: op_fft for symbolic tensors")

let op_ifft t ~axes =
  try Effect.perform (E_ifft { t; axes })
  with Effect.Unhandled _ -> (
    match t with
    | Native_tensor t -> Native_tensor (Nx_c.op_ifft t ~axes)
    | Symbolic_tensor _ -> failwith "todo: op_ifft for symbolic tensors")

let op_rfft (type a c) (t : (float, a) t) ~(dtype : (Complex.t, c) Dtype.t)
    ~axes : (Complex.t, c) t =
  match t with
  | Native_tensor t ->
      let result = Nx_c.op_rfft t ~dtype ~axes in
      (Native_tensor result : (Complex.t, c) t)
  | Symbolic_tensor _ -> failwith "todo: op_rfft for symbolic tensors"

let op_irfft (type a c) (t : (Complex.t, a) t) ~(dtype : (float, c) Dtype.t)
    ~axes ~s : (float, c) t =
  match t with
  | Native_tensor t ->
      let result = Nx_c.op_irfft t ~dtype ~axes ~s in
      (Native_tensor result : (float, c) t)
  | Symbolic_tensor _ -> failwith "todo: op_irfft for symbolic tensors"

(* Linear algebra operations *)

let op_cholesky ~upper:_ _ =
  failwith "op_cholesky: not implemented in Rune backend"

let op_qr ~reduced t_in =
  try Effect.perform (E_qr { t_in; reduced })
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t ->
        let q, r = Nx_c.op_qr ~reduced t in
        (Native_tensor q, Native_tensor r)
    | Symbolic_tensor _ ->
        failwith "op_qr: symbolic tensors not supported in Rune backend")


let op_svd ~full_matrices:_ _ =
  failwith "op_svd: not implemented in Rune backend"

let op_eig ~vectors:_ _ = failwith "op_eig: not implemented in Rune backend"
let op_eigh ~vectors:_ _ = failwith "op_eigh: not implemented in Rune backend"

let op_triangular_solve ~upper:_ ~transpose:_ ~unit_diag:_ _ _ =
  failwith "op_triangular_solve: not implemented in Rune backend"

let op_as_strided t_in new_shape new_strides_in_elements offset_in_elements =
  (* Rune backend implementation of as_strided *)
  let new_shape_arr =
    match Symbolic_shape.eval new_shape with
    | Some arr -> arr
    | None ->
        failwith "op_as_strided: symbolic shapes not supported in Rune backend"
  in

  (* Use the as_strided effect for autodiff tracking *)
  try
    Effect.perform
      (E_as_strided
         {
           t_in;
           new_shape = new_shape_arr;
           new_strides = new_strides_in_elements;
           offset = offset_in_elements;
         })
  with Effect.Unhandled _ -> (
    match t_in with
    | Native_tensor t ->
        Native_tensor
          (Nx_c.op_as_strided t new_shape new_strides_in_elements
             offset_in_elements)
    | Symbolic_tensor _ ->
        failwith "op_as_strided: cannot operate on symbolic tensor")
