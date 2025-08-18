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
  | Ocaml_context : Nx_native.context -> context
  | C_context : Nx_c.context -> context
  | Metal_context : Rune_metal.context -> context

type device_type = Ocaml | C | Metal

(* Backend interface requires type ('a, 'b) t *)
type ('a, 'b) t =
  | Ocaml_tensor : ('a, 'b) Nx_native.t -> ('a, 'b) t
  | C_tensor : ('a, 'b) Nx_c.t -> ('a, 'b) t
  | Metal_tensor : ('a, 'b) Rune_metal.t -> ('a, 'b) t
  | Symbolic_tensor : {
      id : Symbolic_id.t;
      dtype : ('a, 'b) Dtype.t;
      shape : int array;
    }
      -> ('a, 'b) t

let is_device_available = function
  | Ocaml -> true
  | Metal -> Rune_metal.is_available
  | C -> (
      match Sys.backend_type with
      | Sys.(Native | Bytecode) -> true
      | Sys.Other "js_of_ocaml" -> false
      | _ -> false)

(* Ocaml_context creation *)
let create_context ?(device = Ocaml) () : context =
  match device with
  | Ocaml -> Ocaml_context (Nx_native.create_context ())
  | C ->
      if not (is_device_available C) then
        failwith "C backend is not available on this platform"
      else C_context (Nx_c.create_context ())
  | Metal ->
      if not (is_device_available Metal) then
        failwith "Metal backend is not available on this platform"
      else Metal_context (Rune_metal.create_context ())

let default_device () : device_type =
  if is_device_available Metal then Metal
  else if is_device_available C then C
  else Ocaml

let create_default_context () : context =
  create_context ~device:(default_device ()) ()

(* Extract context from tensor *)
let context : type a b. (a, b) t -> context = function
  | Ocaml_tensor cpu_t -> Ocaml_context (Nx_native.context cpu_t)
  | C_tensor c_t -> C_context (Nx_c.context c_t)
  | Metal_tensor metal_t -> Metal_context (Rune_metal.context metal_t)
  | Symbolic_tensor _ -> failwith "Symbolic tensors do not have a context"

(* Device transfer operations *)
let to_device (target_ctx : context) (t : ('a, 'b) t) : ('a, 'b) t =
  match (target_ctx, t) with
  (* Already on correct device *)
  | Ocaml_context _, Ocaml_tensor _
  | Metal_context _, Metal_tensor _
  | C_context _, C_tensor _ ->
      t
  (* CPU to Metal *)
  | Metal_context metal_ctx, Ocaml_tensor cpu_t ->
      let data = Nx_native.data cpu_t in
      Metal_tensor (Rune_metal.op_const_array metal_ctx data)
  (* Metal to CPU *)
  | Ocaml_context ctx, Metal_tensor metal_t ->
      let data = Rune_metal.data metal_t in
      Ocaml_tensor (Nx_native.op_const_array ctx data)
  (* CPU to C *)
  | C_context c_ctx, Ocaml_tensor cpu_t ->
      let data = Nx_native.data cpu_t in
      C_tensor (Nx_c.op_const_array c_ctx data)
  (* C to CPU *)
  | Ocaml_context ctx, C_tensor c_t ->
      let data = Nx_c.data c_t in
      Ocaml_tensor (Nx_native.op_const_array ctx data)
  (* Metal to C *)
  | C_context c_ctx, Metal_tensor metal_t ->
      let data = Rune_metal.data metal_t in
      C_tensor (Nx_c.op_const_array c_ctx data)
  (* C to Metal *)
  | Metal_context metal_ctx, C_tensor c_t ->
      let data = Nx_c.data c_t in
      Metal_tensor (Rune_metal.op_const_array metal_ctx data)
  (* Symbolic tensors update their context *)
  | _, Symbolic_tensor _ -> failwith "Cannot transfer symbolic tensor to device"

(* Lenses *)
let view : type a b. (a, b) t -> Lazy_view.t = function
  | Ocaml_tensor t -> Nx_native.view t
  | Metal_tensor t -> Rune_metal.view t
  | C_tensor t -> Nx_c.view t
  | Symbolic_tensor { shape; _ } ->
      Lazy_view.create (Symbolic_shape.of_ints shape)

let dtype : type a b. (a, b) t -> (a, b) Dtype.t = function
  | Ocaml_tensor t -> Nx_native.dtype t
  | Metal_tensor t -> Rune_metal.dtype t
  | C_tensor t -> Nx_c.dtype t
  | Symbolic_tensor { dtype; _ } -> dtype

let is_symbolic = function Symbolic_tensor _ -> true | _ -> false

let data : type a b.
    (a, b) t -> (a, b, Bigarray_ext.c_layout) Bigarray_ext.Array1.t = function
  | Ocaml_tensor t -> Nx_native.data t
  | Metal_tensor t -> Rune_metal.data t
  | C_tensor t -> Nx_c.data t
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
      s : int array option;
    }
      -> (Complex.t, 'b) t Effect.t
  | E_ifft : {
      t : (Complex.t, 'b) t;
      axes : int array;
      s : int array option;
    }
      -> (Complex.t, 'b) t Effect.t
  | E_rfft : {
      t : (float, 'b) t;
      axes : int array;
      s : int array option;
    }
      -> (Complex.t, Dtype.complex64_elt) t Effect.t
  | E_irfft : {
      t : (Complex.t, 'b) t;
      axes : int array;
      s : int array option;
    }
      -> (float, Dtype.float64_elt) t Effect.t

(* Helper functions for different operation types *)

let ensure_same_device a b =
  match (a, b) with
  | Ocaml_tensor _, Ocaml_tensor _
  | C_tensor _, C_tensor _
  | Metal_tensor _, Metal_tensor _ ->
      (a, b)
  | _ ->
      (* Convert b to a's device *)
      let ctx = context a in
      (a, to_device ctx b)

let binary_op eff cpu_op metal_op c_op a b =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    let a', b' = ensure_same_device a b in
    match (a', b') with
    | Ocaml_tensor t1, Ocaml_tensor t2 -> Ocaml_tensor (cpu_op t1 t2)
    | C_tensor t1, C_tensor t2 -> C_tensor (c_op t1 t2)
    | Metal_tensor t1, Metal_tensor t2 -> Metal_tensor (metal_op t1 t2)
    | _ -> assert false)

let unary_op eff cpu_op metal_op c_op t_in =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Ocaml_tensor t -> Ocaml_tensor (cpu_op t)
    | C_tensor t -> C_tensor (c_op t)
    | Metal_tensor t -> Metal_tensor (metal_op t)
    | Symbolic_tensor _ ->
        failwith "Cannot perform operation on symbolic tensor")

let comparison_op eff cpu_op metal_op c_op a b =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    let a', b' = ensure_same_device a b in
    match (a', b') with
    | Ocaml_tensor t1, Ocaml_tensor t2 -> Ocaml_tensor (cpu_op t1 t2)
    | C_tensor t1, C_tensor t2 -> C_tensor (c_op t1 t2)
    | Metal_tensor t1, Metal_tensor t2 -> Metal_tensor (metal_op t1 t2)
    | _ -> assert false)

let reduce_op eff cpu_op metal_op c_op ~axes ~keepdims t_in =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Ocaml_tensor t -> Ocaml_tensor (cpu_op ~axes ~keepdims t)
    | C_tensor t -> C_tensor (c_op ~axes ~keepdims t)
    | Metal_tensor t -> Metal_tensor (metal_op ~axes ~keepdims t)
    | Symbolic_tensor _ ->
        failwith "Cannot perform reduction on symbolic tensor")

let shape_op1 eff cpu_op metal_op c_op t_in shape_arg =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Ocaml_tensor t ->
        Ocaml_tensor (cpu_op t (Symbolic_shape.of_ints shape_arg))
    | C_tensor t -> C_tensor (c_op t (Symbolic_shape.of_ints shape_arg))
    | Metal_tensor t ->
        Metal_tensor (metal_op t (Symbolic_shape.of_ints shape_arg))
    | Symbolic_tensor _ ->
        failwith "Cannot perform shape operation on symbolic tensor")

let axes_op1 eff cpu_op metal_op c_op t_in axes_arg =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Ocaml_tensor t -> Ocaml_tensor (cpu_op t axes_arg)
    | C_tensor t -> C_tensor (c_op t axes_arg)
    | Metal_tensor t -> Metal_tensor (metal_op t axes_arg)
    | Symbolic_tensor _ ->
        failwith "Cannot perform axes operation on symbolic tensor")

let limits_op1 eff cpu_op metal_op c_op t_in limits_arg =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Ocaml_tensor t -> Ocaml_tensor (cpu_op t limits_arg)
    | C_tensor t -> C_tensor (c_op t limits_arg)
    | Metal_tensor t -> Metal_tensor (metal_op t limits_arg)
    | Symbolic_tensor _ ->
        failwith "Cannot perform limits operation on symbolic tensor")

let bool_array_op1 eff cpu_op metal_op c_op t_in bool_arg =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Ocaml_tensor t -> Ocaml_tensor (cpu_op t bool_arg)
    | C_tensor t -> C_tensor (c_op t bool_arg)
    | Metal_tensor t -> Metal_tensor (metal_op t bool_arg)
    | Symbolic_tensor _ ->
        failwith "Cannot perform bool array operation on symbolic tensor")

let ternary_op eff cpu_op metal_op c_op cond if_true if_false =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    (* Ensure all three tensors are on the same device *)
    let ctx = context cond in
    let cond' = to_device ctx cond in
    let if_true' = to_device ctx if_true in
    let if_false' = to_device ctx if_false in
    match (cond', if_true', if_false') with
    | Ocaml_tensor t1, Ocaml_tensor t2, Ocaml_tensor t3 ->
        Ocaml_tensor (cpu_op t1 t2 t3)
    | C_tensor t1, C_tensor t2, C_tensor t3 -> C_tensor (c_op t1 t2 t3)
    | Metal_tensor t1, Metal_tensor t2, Metal_tensor t3 ->
        Metal_tensor (metal_op t1 t2 t3)
    | _ -> assert false)

(* Binary operations *)
let op_add a b =
  binary_op
    (fun () -> E_add { a; b })
    Nx_native.op_add Rune_metal.op_add Nx_c.op_add a b

let op_mul a b =
  binary_op
    (fun () -> E_mul { a; b })
    Nx_native.op_mul Rune_metal.op_mul Nx_c.op_mul a b

let op_idiv a b =
  binary_op
    (fun () -> E_idiv { a; b })
    Nx_native.op_idiv Rune_metal.op_idiv Nx_c.op_idiv a b

let op_fdiv a b =
  binary_op
    (fun () -> E_fdiv { a; b })
    Nx_native.op_fdiv Rune_metal.op_fdiv Nx_c.op_fdiv a b

let op_max a b =
  binary_op
    (fun () -> E_max { a; b })
    Nx_native.op_max Rune_metal.op_max Nx_c.op_max a b

let op_mod a b =
  binary_op
    (fun () -> E_mod { a; b })
    Nx_native.op_mod Rune_metal.op_mod Nx_c.op_mod a b

let op_pow a b =
  binary_op
    (fun () -> E_pow { a; b })
    Nx_native.op_pow Rune_metal.op_pow Nx_c.op_pow a b

let op_xor a b =
  binary_op
    (fun () -> E_xor { a; b })
    Nx_native.op_xor Rune_metal.op_xor Nx_c.op_xor a b

let op_or a b =
  binary_op
    (fun () -> E_or { a; b })
    Nx_native.op_or Rune_metal.op_or Nx_c.op_or a b

let op_and a b =
  binary_op
    (fun () -> E_and { a; b })
    Nx_native.op_and Rune_metal.op_and Nx_c.op_and a b

(* Comparison operations *)
let op_cmplt a b =
  comparison_op
    (fun () -> E_cmplt { a; b })
    Nx_native.op_cmplt Rune_metal.op_cmplt Nx_c.op_cmplt a b

let op_cmpne a b =
  comparison_op
    (fun () -> E_cmpne { a; b })
    Nx_native.op_cmpne Rune_metal.op_cmpne Nx_c.op_cmpne a b

(* Unary operations *)
let op_neg t_in =
  unary_op
    (fun () -> E_neg { t_in })
    Nx_native.op_neg Rune_metal.op_neg Nx_c.op_neg t_in

let op_log2 t_in =
  unary_op
    (fun () -> E_log2 { t_in })
    Nx_native.op_log2 Rune_metal.op_log2 Nx_c.op_log2 t_in

let op_exp2 t_in =
  unary_op
    (fun () -> E_exp2 { t_in })
    Nx_native.op_exp2 Rune_metal.op_exp2 Nx_c.op_exp2 t_in

let op_sin t_in =
  unary_op
    (fun () -> E_sin { t_in })
    Nx_native.op_sin Rune_metal.op_sin Nx_c.op_sin t_in

let op_sqrt t_in =
  unary_op
    (fun () -> E_sqrt { t_in })
    Nx_native.op_sqrt Rune_metal.op_sqrt Nx_c.op_sqrt t_in

let op_recip t_in =
  unary_op
    (fun () -> E_recip { t_in })
    Nx_native.op_recip Rune_metal.op_recip Nx_c.op_recip t_in

(* Reduction operations *)
let op_reduce_sum ~axes ~keepdims t_in =
  reduce_op
    (fun () -> E_reduce_sum { t_in; axes; keepdims })
    Nx_native.op_reduce_sum Rune_metal.op_reduce_sum Nx_c.op_reduce_sum ~axes
    ~keepdims t_in

let op_reduce_max ~axes ~keepdims t_in =
  reduce_op
    (fun () -> E_reduce_max { t_in; axes; keepdims })
    Nx_native.op_reduce_max Rune_metal.op_reduce_max Nx_c.op_reduce_max ~axes
    ~keepdims t_in

let op_reduce_prod ~axes ~keepdims t_in =
  reduce_op
    (fun () -> E_reduce_prod { t_in; axes; keepdims })
    Nx_native.op_reduce_prod Rune_metal.op_reduce_prod Nx_c.op_reduce_prod ~axes
    ~keepdims t_in

(* Shape operations *)
let op_reshape t_in new_shape =
  let new_shape_array =
    match Symbolic_shape.eval new_shape with
    | Some arr -> arr
    | None -> failwith "Cannot reshape with symbolic shape"
  in
  shape_op1
    (fun () -> E_reshape { t_in; new_shape = new_shape_array })
    Nx_native.op_reshape Rune_metal.op_reshape Nx_c.op_reshape t_in
    new_shape_array

let op_expand t_in new_target_shape =
  let new_target_shape_array =
    match Symbolic_shape.eval new_target_shape with
    | Some arr -> arr
    | None -> failwith "Cannot expand with symbolic shape"
  in
  shape_op1
    (fun () -> E_expand { t_in; new_target_shape = new_target_shape_array })
    Nx_native.op_expand Rune_metal.op_expand Nx_c.op_expand t_in
    new_target_shape_array

let op_permute t_in axes =
  axes_op1
    (fun () -> E_permute { t_in; axes })
    Nx_native.op_permute Rune_metal.op_permute Nx_c.op_permute t_in axes

let op_shrink t_in limits =
  limits_op1
    (fun () -> E_shrink { t_in; limits })
    Nx_native.op_shrink Rune_metal.op_shrink Nx_c.op_shrink t_in limits

let op_flip t_in dims_to_flip =
  bool_array_op1
    (fun () -> E_flip { t_in; dims_to_flip })
    Nx_native.op_flip Rune_metal.op_flip Nx_c.op_flip t_in dims_to_flip

(* Pad operation (needs special handling for fill_value) *)
let op_pad t_in padding_config fill_value =
  try Effect.perform (E_pad { t_in; padding_config; fill_value })
  with Effect.Unhandled _ -> (
    match t_in with
    | Ocaml_tensor t ->
        Ocaml_tensor (Nx_native.op_pad t padding_config fill_value)
    | Metal_tensor t ->
        Metal_tensor (Rune_metal.op_pad t padding_config fill_value)
    | C_tensor t -> C_tensor (Nx_c.op_pad t padding_config fill_value)
    | Symbolic_tensor _ -> failwith "Cannot pad symbolic tensor")

(* Creation operations *)
let op_buffer ctx dtype size_in_elements =
  try Effect.perform (E_buffer { context = ctx; dtype; size_in_elements })
  with Effect.Unhandled _ -> (
    match ctx with
    | Ocaml_context ctx ->
        Ocaml_tensor (Nx_native.op_buffer ctx dtype size_in_elements)
    | Metal_context metal_ctx ->
        Metal_tensor (Rune_metal.op_buffer metal_ctx dtype size_in_elements)
    | C_context c_ctx -> C_tensor (Nx_c.op_buffer c_ctx dtype size_in_elements))

let op_const_scalar ctx value dtype =
  try Effect.perform (E_const_scalar { context = ctx; value; dtype })
  with Effect.Unhandled _ -> (
    match ctx with
    | Ocaml_context ctx ->
        Ocaml_tensor (Nx_native.op_const_scalar ctx value dtype)
    | Metal_context metal_ctx ->
        Metal_tensor (Rune_metal.op_const_scalar metal_ctx value dtype)
    | C_context c_ctx -> C_tensor (Nx_c.op_const_scalar c_ctx value dtype))

let op_const_array ctx array =
  try Effect.perform (E_const_array { context = ctx; array })
  with Effect.Unhandled _ -> (
    match ctx with
    | Ocaml_context ctx -> Ocaml_tensor (Nx_native.op_const_array ctx array)
    | Metal_context metal_ctx ->
        Metal_tensor (Rune_metal.op_const_array metal_ctx array)
    | C_context c_ctx -> C_tensor (Nx_c.op_const_array c_ctx array))

(* Copy operations *)
let op_contiguous t_in =
  unary_op
    (fun () -> E_contiguous { t_in })
    Nx_native.op_contiguous Rune_metal.op_contiguous Nx_c.op_contiguous t_in

let op_copy t_in =
  unary_op
    (fun () -> E_copy { t_in })
    Nx_native.op_copy Rune_metal.op_copy Nx_c.op_copy t_in

(* Where operation *)
let op_where condition if_true if_false =
  ternary_op
    (fun () -> E_where { condition; if_true; if_false })
    Nx_native.op_where Rune_metal.op_where Nx_c.op_where condition if_true
    if_false

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
      | Ocaml_context _ ->
          let cpu_list =
            List.map
              (function Ocaml_tensor t -> t | _ -> assert false)
              converted
          in
          Ocaml_tensor (Nx_native.op_cat cpu_list axis)
      | C_context _ ->
          let c_list =
            List.map (function C_tensor t -> t | _ -> assert false) converted
          in
          C_tensor (Nx_c.op_cat c_list axis)
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
    | Ocaml_tensor t -> Ocaml_tensor (Nx_native.op_cast t target_dtype)
    | C_tensor t -> C_tensor (Nx_c.op_cast t target_dtype)
    | Metal_tensor t -> Metal_tensor (Rune_metal.op_cast t target_dtype)
    | Symbolic_tensor _ -> failwith "Cannot cast symbolic tensor")

(* Assign operation *)
let op_assign dst src =
  try Effect.perform (E_assign { dst; src })
  with Effect.Unhandled _ -> (
    let dst', src' = ensure_same_device dst src in
    match (dst', src') with
    | Ocaml_tensor d, Ocaml_tensor s -> Nx_native.op_assign d s
    | C_tensor d, C_tensor s -> Nx_c.op_assign d s
    | Metal_tensor d, Metal_tensor s -> Rune_metal.op_assign d s
    | _ -> assert false)

(* Gather operation *)
let op_gather data indices axis =
  try Effect.perform (E_gather { data; indices; axis })
  with Effect.Unhandled _ -> (
    let data', indices' = ensure_same_device data indices in
    match (data', indices') with
    | Ocaml_tensor d, Ocaml_tensor i ->
        Ocaml_tensor (Nx_native.op_gather d i axis)
    | C_tensor d, C_tensor i -> C_tensor (Nx_c.op_gather d i axis)
    | Metal_tensor d, Metal_tensor i ->
        Metal_tensor (Rune_metal.op_gather d i axis)
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
    | Ocaml_tensor t, Ocaml_tensor i, Ocaml_tensor u ->
        Ocaml_tensor (Nx_native.op_scatter ~mode ~unique_indices t i u axis)
    | C_tensor t, C_tensor i, C_tensor u ->
        C_tensor (Nx_c.op_scatter ~mode ~unique_indices t i u axis)
    | Metal_tensor t, Metal_tensor i, Metal_tensor u ->
        Metal_tensor (Rune_metal.op_scatter ~mode ~unique_indices t i u axis)
    | _ -> assert false)

(* Threefry operation *)
let op_threefry key ctr =
  binary_op
    (fun () -> E_threefry { key; ctr })
    Nx_native.op_threefry Rune_metal.op_threefry Nx_c.op_threefry key ctr

(* Unfold operation *)
let op_unfold t_in ~kernel_size ~stride ~dilation ~padding =
  try Effect.perform (E_unfold { t_in; kernel_size; stride; dilation; padding })
  with Effect.Unhandled _ -> (
    match t_in with
    | Ocaml_tensor t ->
        Ocaml_tensor
          (Nx_native.op_unfold t ~kernel_size ~stride ~dilation ~padding)
    | C_tensor t ->
        C_tensor (Nx_c.op_unfold t ~kernel_size ~stride ~dilation ~padding)
    | Metal_tensor t ->
        Metal_tensor
          (Rune_metal.op_unfold t ~kernel_size ~stride ~dilation ~padding)
    | Symbolic_tensor _ -> failwith "todo: op_unfold for symbolic tensors")

(* Fold operation *)
let op_fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding =
  try
    Effect.perform
      (E_fold { t_in; output_size; kernel_size; stride; dilation; padding })
  with Effect.Unhandled _ -> (
    match t_in with
    | Ocaml_tensor t ->
        Ocaml_tensor
          (Nx_native.op_fold t ~output_size ~kernel_size ~stride ~dilation
             ~padding)
    | C_tensor t ->
        C_tensor
          (Nx_c.op_fold t ~output_size ~kernel_size ~stride ~dilation ~padding)
    | Metal_tensor t ->
        Metal_tensor
          (Rune_metal.op_fold t ~output_size ~kernel_size ~stride ~dilation
             ~padding)
    | Symbolic_tensor _ -> failwith "todo: op_fold for symbolic tensors")

(* Matrix multiplication *)
let op_matmul a b =
  try Effect.perform (E_matmul { a; b })
  with Effect.Unhandled _ -> (
    let a', b' = ensure_same_device a b in
    match (a', b') with
    | Ocaml_tensor a_t, Ocaml_tensor b_t ->
        Ocaml_tensor (Nx_native.op_matmul a_t b_t)
    | C_tensor a_t, C_tensor b_t -> C_tensor (Nx_c.op_matmul a_t b_t)
    | Metal_tensor a_t, Metal_tensor b_t ->
        Metal_tensor (Rune_metal.op_matmul a_t b_t)
    | Symbolic_tensor _, _ | _, Symbolic_tensor _ ->
        failwith "todo: op_matmul for symbolic tensors"
    | _ -> assert false)

(* FFT operations *)
let op_fft t ~axes ~s =
  try Effect.perform (E_fft { t; axes; s })
  with Effect.Unhandled _ -> (
    match t with
    | Ocaml_tensor t -> Ocaml_tensor (Nx_native.op_fft t ~axes ~s)
    | C_tensor t -> C_tensor (Nx_c.op_fft t ~axes ~s)
    | Metal_tensor t -> Metal_tensor (Rune_metal.op_fft t ~axes ~s)
    | Symbolic_tensor _ -> failwith "todo: op_fft for symbolic tensors")

let op_ifft t ~axes ~s =
  try Effect.perform (E_ifft { t; axes; s })
  with Effect.Unhandled _ -> (
    match t with
    | Ocaml_tensor t -> Ocaml_tensor (Nx_native.op_ifft t ~axes ~s)
    | C_tensor t -> C_tensor (Nx_c.op_ifft t ~axes ~s)
    | Metal_tensor t -> Metal_tensor (Rune_metal.op_ifft t ~axes ~s)
    | Symbolic_tensor _ -> failwith "todo: op_ifft for symbolic tensors")

let op_rfft (type a c) (t : (float, a) t) ~(dtype : (Complex.t, c) Dtype.t)
    ~axes ~s : (Complex.t, c) t =
  match t with
  | Ocaml_tensor t ->
      let result = Nx_native.op_rfft t ~dtype ~axes ~s in
      (Ocaml_tensor result : (Complex.t, c) t)
  | C_tensor t ->
      let result = Nx_c.op_rfft t ~dtype ~axes ~s in
      (C_tensor result : (Complex.t, c) t)
  | Metal_tensor t ->
      let result = Rune_metal.op_rfft t ~dtype ~axes ~s in
      (Metal_tensor result : (Complex.t, c) t)
  | Symbolic_tensor _ -> failwith "todo: op_rfft for symbolic tensors"

let op_irfft (type a c) (t : (Complex.t, a) t) ~(dtype : (float, c) Dtype.t)
    ~axes ~s : (float, c) t =
  match t with
  | Ocaml_tensor t ->
      let result = Nx_native.op_irfft t ~dtype ~axes ~s in
      (Ocaml_tensor result : (float, c) t)
  | C_tensor t ->
      let result = Nx_c.op_irfft t ~dtype ~axes ~s in
      (C_tensor result : (float, c) t)
  | Metal_tensor t ->
      let result = Rune_metal.op_irfft t ~dtype ~axes ~s in
      (Metal_tensor result : (float, c) t)
  | Symbolic_tensor _ -> failwith "todo: op_irfft for symbolic tensors"

(* Linear algebra operations *)

let op_cholesky ~upper:_ _ =
  failwith "op_cholesky: not implemented in Rune backend"

let op_qr ~reduced:_ _ = failwith "op_qr: not implemented in Rune backend"

let op_svd ~full_matrices:_ _ =
  failwith "op_svd: not implemented in Rune backend"

let op_eig ~vectors:_ _ = failwith "op_eig: not implemented in Rune backend"
let op_eigh ~vectors:_ _ = failwith "op_eigh: not implemented in Rune backend"

let op_triangular_solve ~upper:_ ~transpose:_ ~unit_diag:_ _ _ =
  failwith "op_triangular_solve: not implemented in Rune backend"
