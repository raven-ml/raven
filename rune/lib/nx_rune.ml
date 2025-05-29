open Nx_core

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

type context =
  | Cpu_context : Nx_native.context -> context
  | Metal_context : Rune_metal.context -> context

type device_type = Cpu | Metal

(* Backend interface requires type ('a, 'b) t *)
type ('a, 'b) t =
  | Cpu_tensor : ('a, 'b) Nx_native.t -> ('a, 'b) t
  | Metal_tensor : ('a, 'b) Rune_metal.t -> ('a, 'b) t
  | Symbolic_tensor : {
      id : Symbolic_id.t;
      dtype : ('a, 'b) Dtype.t;
      shape : int array;
    }
      -> ('a, 'b) t

(* Context creation *)
let create_context ?(device = Cpu) () : context =
  match device with
  | Cpu -> Cpu_context (Nx_native.create_context ())
  | Metal ->
      if not Rune_metal.is_available then
        failwith "Metal backend is not available on this platform"
      else Metal_context (Rune_metal.create_context ())

let default_device () : device_type =
  if Rune_metal.is_available && Rune_metal.is_available then Metal else Cpu

let create_default_context () : context =
  create_context ~device:(default_device ()) ()

(* Extract context from tensor *)
let context : type a b. (a, b) t -> context = function
  | Cpu_tensor cpu_t -> Cpu_context (Nx_native.context cpu_t)
  | Metal_tensor metal_t -> Metal_context (Rune_metal.context metal_t)
  | Symbolic_tensor _ -> failwith "Symbolic tensors do not have a context"

(* Device transfer operations *)
let to_device (target_ctx : context) (t : ('a, 'b) t) : ('a, 'b) t =
  match (target_ctx, t) with
  (* Already on correct device *)
  | Cpu_context _, Cpu_tensor _ | Metal_context _, Metal_tensor _ -> t
  (* CPU to Metal *)
  | Metal_context metal_ctx, Cpu_tensor cpu_t ->
      let data = Nx_native.data cpu_t in
      Metal_tensor (Rune_metal.op_const_array metal_ctx data)
  (* Metal to CPU *)
  | Cpu_context cpu_ctx, Metal_tensor metal_t ->
      let data = Rune_metal.data metal_t in
      Cpu_tensor (Nx_native.op_const_array cpu_ctx data)
  (* Symbolic tensors update their context *)
  | _, Symbolic_tensor _ -> failwith "Cannot transfer symbolic tensor to device"

(* Lenses *)
let view : type a b. (a, b) t -> View.t = function
  | Cpu_tensor t -> Nx_native.view t
  | Metal_tensor t -> Rune_metal.view t
  | Symbolic_tensor { shape; _ } -> View.create shape

let dtype : type a b. (a, b) t -> (a, b) Dtype.t = function
  | Cpu_tensor t -> Nx_native.dtype t
  | Metal_tensor t -> Rune_metal.dtype t
  | Symbolic_tensor { dtype; _ } -> dtype

let is_symbolic = function Symbolic_tensor _ -> true | _ -> false

let data : type a b. (a, b) t -> (a, b, Bigarray.c_layout) Bigarray.Array1.t =
  function
  | Cpu_tensor t -> Nx_native.data t
  | Metal_tensor _ ->
      failwith
        "Cannot extract raw data from Metal tensor. Transfer to CPU first."
  | Symbolic_tensor { id; _ } ->
      failwith (Printf.sprintf "Cannot extract data from symbolic tensor %d" id)

(* Effects - no context in most operations per new backend interface *)
type _ Effect.t +=
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
      array : ('a, 'b, Bigarray.c_layout) Bigarray.Array1.t;
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
      -> (int, Dtype.uint8_elt) t Effect.t
  | E_cmpne : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> (int, Dtype.uint8_elt) t Effect.t
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
      condition : (int, Dtype.uint8_elt) t;
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
  | E_permute : { t_in : ('a, 'b) t; axes : int array } -> ('a, 'b) t Effect.t
  | E_reshape : {
      t_in : ('a, 'b) t;
      new_shape : int array;
    }
      -> ('a, 'b) t Effect.t
  | E_expand : {
      t_in : ('a, 'b) t;
      new_target_shape : int array;
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

(* Helper functions for different operation types *)

let ensure_same_device a b =
  match (a, b) with
  | Cpu_tensor _, Cpu_tensor _ | Metal_tensor _, Metal_tensor _ -> (a, b)
  | _ ->
      (* Convert b to a's device *)
      let ctx = context a in
      (a, to_device ctx b)

let binary_op eff cpu_op metal_op a b =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    let a', b' = ensure_same_device a b in
    match (a', b') with
    | Cpu_tensor t1, Cpu_tensor t2 -> Cpu_tensor (cpu_op t1 t2)
    | Metal_tensor t1, Metal_tensor t2 -> Metal_tensor (metal_op t1 t2)
    | _ -> assert false)

let unary_op eff cpu_op metal_op t_in =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Cpu_tensor t -> Cpu_tensor (cpu_op t)
    | Metal_tensor t -> Metal_tensor (metal_op t)
    | Symbolic_tensor _ ->
        failwith "Cannot perform operation on symbolic tensor")

let comparison_op eff cpu_op metal_op a b =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    let a', b' = ensure_same_device a b in
    match (a', b') with
    | Cpu_tensor t1, Cpu_tensor t2 -> Cpu_tensor (cpu_op t1 t2)
    | Metal_tensor t1, Metal_tensor t2 -> Metal_tensor (metal_op t1 t2)
    | _ -> assert false)

let reduce_op eff cpu_op metal_op ~axes ~keepdims t_in =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Cpu_tensor t -> Cpu_tensor (cpu_op ~axes ~keepdims t)
    | Metal_tensor t -> Metal_tensor (metal_op ~axes ~keepdims t)
    | Symbolic_tensor _ ->
        failwith "Cannot perform reduction on symbolic tensor")

let shape_op1 eff cpu_op metal_op t_in shape_arg =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Cpu_tensor t -> Cpu_tensor (cpu_op t shape_arg)
    | Metal_tensor t -> Metal_tensor (metal_op t shape_arg)
    | Symbolic_tensor _ ->
        failwith "Cannot perform shape operation on symbolic tensor")

let ternary_op eff cpu_op metal_op cond if_true if_false =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    (* Ensure all three tensors are on the same device *)
    let ctx = context cond in
    let cond' = to_device ctx cond in
    let if_true' = to_device ctx if_true in
    let if_false' = to_device ctx if_false in
    match (cond', if_true', if_false') with
    | Cpu_tensor t1, Cpu_tensor t2, Cpu_tensor t3 ->
        Cpu_tensor (cpu_op t1 t2 t3)
    | Metal_tensor t1, Metal_tensor t2, Metal_tensor t3 ->
        Metal_tensor (metal_op t1 t2 t3)
    | _ -> assert false)

(* Binary operations *)
let op_add a b =
  binary_op (fun () -> E_add { a; b }) Nx_native.op_add Rune_metal.op_add a b

let op_mul a b =
  binary_op (fun () -> E_mul { a; b }) Nx_native.op_mul Rune_metal.op_mul a b

let op_idiv a b =
  binary_op (fun () -> E_idiv { a; b }) Nx_native.op_idiv Rune_metal.op_idiv a b

let op_fdiv a b =
  binary_op (fun () -> E_fdiv { a; b }) Nx_native.op_fdiv Rune_metal.op_fdiv a b

let op_max a b =
  binary_op (fun () -> E_max { a; b }) Nx_native.op_max Rune_metal.op_max a b

let op_mod a b =
  binary_op (fun () -> E_mod { a; b }) Nx_native.op_mod Rune_metal.op_mod a b

let op_pow a b =
  binary_op (fun () -> E_pow { a; b }) Nx_native.op_pow Rune_metal.op_pow a b

let op_xor a b =
  binary_op (fun () -> E_xor { a; b }) Nx_native.op_xor Rune_metal.op_xor a b

let op_or a b =
  binary_op (fun () -> E_or { a; b }) Nx_native.op_or Rune_metal.op_or a b

let op_and a b =
  binary_op (fun () -> E_and { a; b }) Nx_native.op_and Rune_metal.op_and a b

(* Comparison operations *)
let op_cmplt a b =
  comparison_op
    (fun () -> E_cmplt { a; b })
    Nx_native.op_cmplt Rune_metal.op_cmplt a b

let op_cmpne a b =
  comparison_op
    (fun () -> E_cmpne { a; b })
    Nx_native.op_cmpne Rune_metal.op_cmpne a b

(* Unary operations *)
let op_neg t_in =
  unary_op (fun () -> E_neg { t_in }) Nx_native.op_neg Rune_metal.op_neg t_in

let op_log2 t_in =
  unary_op (fun () -> E_log2 { t_in }) Nx_native.op_log2 Rune_metal.op_log2 t_in

let op_exp2 t_in =
  unary_op (fun () -> E_exp2 { t_in }) Nx_native.op_exp2 Rune_metal.op_exp2 t_in

let op_sin t_in =
  unary_op (fun () -> E_sin { t_in }) Nx_native.op_sin Rune_metal.op_sin t_in

let op_sqrt t_in =
  unary_op (fun () -> E_sqrt { t_in }) Nx_native.op_sqrt Rune_metal.op_sqrt t_in

let op_recip t_in =
  unary_op
    (fun () -> E_recip { t_in })
    Nx_native.op_recip Rune_metal.op_recip t_in

(* Reduction operations *)
let op_reduce_sum ~axes ~keepdims t_in =
  reduce_op
    (fun () -> E_reduce_sum { t_in; axes; keepdims })
    Nx_native.op_reduce_sum Rune_metal.op_reduce_sum ~axes ~keepdims t_in

let op_reduce_max ~axes ~keepdims t_in =
  reduce_op
    (fun () -> E_reduce_max { t_in; axes; keepdims })
    Nx_native.op_reduce_max Rune_metal.op_reduce_max ~axes ~keepdims t_in

let op_reduce_prod ~axes ~keepdims t_in =
  reduce_op
    (fun () -> E_reduce_prod { t_in; axes; keepdims })
    Nx_native.op_reduce_prod Rune_metal.op_reduce_prod ~axes ~keepdims t_in

(* Shape operations *)
let op_reshape t_in new_shape =
  shape_op1
    (fun () -> E_reshape { t_in; new_shape })
    Nx_native.op_reshape Rune_metal.op_reshape t_in new_shape

let op_expand t_in new_target_shape =
  shape_op1
    (fun () -> E_expand { t_in; new_target_shape })
    Nx_native.op_expand Rune_metal.op_expand t_in new_target_shape

let op_permute t_in axes =
  shape_op1
    (fun () -> E_permute { t_in; axes })
    Nx_native.op_permute Rune_metal.op_permute t_in axes

let op_shrink t_in limits =
  shape_op1
    (fun () -> E_shrink { t_in; limits })
    Nx_native.op_shrink Rune_metal.op_shrink t_in limits

let op_flip t_in dims_to_flip =
  shape_op1
    (fun () -> E_flip { t_in; dims_to_flip })
    Nx_native.op_flip Rune_metal.op_flip t_in dims_to_flip

(* Pad operation (needs special handling for fill_value) *)
let op_pad t_in padding_config fill_value =
  try Effect.perform (E_pad { t_in; padding_config; fill_value })
  with Effect.Unhandled _ -> (
    match t_in with
    | Cpu_tensor t -> Cpu_tensor (Nx_native.op_pad t padding_config fill_value)
    | Metal_tensor t ->
        Metal_tensor (Rune_metal.op_pad t padding_config fill_value)
    | Symbolic_tensor _ -> failwith "Cannot pad symbolic tensor")

(* Creation operations *)
let op_buffer ctx dtype size_in_elements =
  try Effect.perform (E_buffer { context = ctx; dtype; size_in_elements })
  with Effect.Unhandled _ -> (
    match ctx with
    | Cpu_context cpu_ctx ->
        Cpu_tensor (Nx_native.op_buffer cpu_ctx dtype size_in_elements)
    | Metal_context metal_ctx ->
        Metal_tensor (Rune_metal.op_buffer metal_ctx dtype size_in_elements))

let op_const_scalar ctx value dtype =
  try Effect.perform (E_const_scalar { context = ctx; value; dtype })
  with Effect.Unhandled _ -> (
    match ctx with
    | Cpu_context cpu_ctx ->
        Cpu_tensor (Nx_native.op_const_scalar cpu_ctx value dtype)
    | Metal_context metal_ctx ->
        Metal_tensor (Rune_metal.op_const_scalar metal_ctx value dtype))

let op_const_array ctx array =
  try Effect.perform (E_const_array { context = ctx; array })
  with Effect.Unhandled _ -> (
    match ctx with
    | Cpu_context cpu_ctx -> Cpu_tensor (Nx_native.op_const_array cpu_ctx array)
    | Metal_context metal_ctx ->
        Metal_tensor (Rune_metal.op_const_array metal_ctx array))

(* Copy operations *)
let op_contiguous t_in =
  unary_op
    (fun () -> E_contiguous { t_in })
    Nx_native.op_contiguous Rune_metal.op_contiguous t_in

let op_copy t_in =
  unary_op (fun () -> E_copy { t_in }) Nx_native.op_copy Rune_metal.op_copy t_in

(* Where operation *)
let op_where condition if_true if_false =
  ternary_op
    (fun () -> E_where { condition; if_true; if_false })
    Nx_native.op_where Rune_metal.op_where condition if_true if_false

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
      | Cpu_context _ ->
          let cpu_list =
            List.map
              (function Cpu_tensor t -> t | _ -> assert false)
              converted
          in
          Cpu_tensor (Nx_native.op_cat cpu_list axis)
      | Metal_context _ ->
          let metal_list =
            List.map
              (function Metal_tensor t -> t | _ -> assert false)
              converted
          in
          Metal_tensor (Rune_metal.op_cat metal_list axis))

(* Cast operation *)
let op_cast : type a b c d. (a, b) t -> (c, d) Dtype.t -> (c, d) t =
 fun t_in target_dtype ->
  try Effect.perform (E_cast { t_in; target_dtype })
  with Effect.Unhandled _ -> (
    match t_in with
    | Cpu_tensor t -> Cpu_tensor (Nx_native.op_cast t target_dtype)
    | Metal_tensor t -> Metal_tensor (Rune_metal.op_cast t target_dtype)
    | Symbolic_tensor _ -> failwith "Cannot cast symbolic tensor")

(* Assign operation *)
let op_assign dst src =
  try Effect.perform (E_assign { dst; src })
  with Effect.Unhandled _ -> (
    let dst', src' = ensure_same_device dst src in
    match (dst', src') with
    | Cpu_tensor d, Cpu_tensor s -> Nx_native.op_assign d s
    | Metal_tensor d, Metal_tensor s -> Rune_metal.op_assign d s
    | _ -> assert false)

(* Gather operation *)
let op_gather data indices axis =
  try Effect.perform (E_gather { data; indices; axis })
  with Effect.Unhandled _ -> (
    let data', indices' = ensure_same_device data indices in
    match (data', indices') with
    | Cpu_tensor d, Cpu_tensor i -> Cpu_tensor (Nx_native.op_gather d i axis)
    | Metal_tensor d, Metal_tensor i ->
        Metal_tensor (Rune_metal.op_gather d i axis)
    | _ -> assert false)

(* Scatter operation *)
let op_scatter data_template indices updates axis =
  try Effect.perform (E_scatter { data_template; indices; updates; axis })
  with Effect.Unhandled _ -> (
    (* Ensure all three tensors are on the same device *)
    let ctx = context data_template in
    let tmpl = to_device ctx data_template in
    let idx = to_device ctx indices in
    let upd = to_device ctx updates in
    match (tmpl, idx, upd) with
    | Cpu_tensor t, Cpu_tensor i, Cpu_tensor u ->
        Cpu_tensor (Nx_native.op_scatter t i u axis)
    | Metal_tensor t, Metal_tensor i, Metal_tensor u ->
        Metal_tensor (Rune_metal.op_scatter t i u axis)
    | _ -> assert false)

(* Threefry operation *)
let op_threefry key ctr =
  binary_op
    (fun () -> E_threefry { key; ctr })
    Nx_native.op_threefry Rune_metal.op_threefry key ctr
