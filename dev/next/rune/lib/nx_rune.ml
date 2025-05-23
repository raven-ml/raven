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

type ('a, 'b) buffer =
  | Cpu_buffer : ('a, 'b) Nx_native.buffer -> ('a, 'b) buffer
  | Symbolic_buffer : {
      id : Symbolic_id.t;
      dtype_repr : ('a, 'b) Dtype.t; (* To reconstruct type if needed *)
      shape_repr : int array;
    }
      -> ('a, 'b) buffer

type context = Cpu_context : Nx_native.context -> context

type ('a, 'b) t = {
  dtype : ('a, 'b) Dtype.t;
  buffer : ('a, 'b) buffer;
  view : View.t;
}

(* These effects correspond one-to-one with the operations in
   Nx_core.Backend_intf.S. They carry the original Rune context and Nx_rune.t
   arguments. Effect handlers (JIT, Autodiff) will catch these. *)
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
  | E_load : {
      context : context;
      buffer_source : ('a, 'b) t;
      logical_indices : (int, Dtype.int32_elt) t list;
      valid_mask : (int, Dtype.uint8_t) t option;
    }
      -> ('a, 'b) t Effect.t
  | E_store : {
      context : context;
      buffer_target : ('a, 'b) t;
      logical_indices : (int, Dtype.int32_elt) t list;
      scalar_value_to_store : ('a, 'b) t;
      valid_mask : (int, Dtype.uint8_t) t option;
    }
      -> unit Effect.t
  | E_add : {
      context : context;
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> ('a, 'b) t Effect.t
  | E_mul : {
      context : context;
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> ('a, 'b) t Effect.t
  | E_sum : {
      context : context;
      t_in : ('a, 'b) t;
      axes : int array;
      keepdims : bool;
    }
      -> ('a, 'b) t Effect.t
  | E_expand : {
      context : context;
      t_in : ('a, 'b) t;
      new_target_shape : int array;
    }
      -> ('a, 'b) t Effect.t
  | E_reshape : {
      context : context;
      t_in : ('a, 'b) t;
      new_shape : int array;
    }
      -> ('a, 'b) t Effect.t

let unwrap_to_nx_native (rune_t : ('a, 'b) t) : ('a, 'b) Nx_native.t =
  match rune_t.buffer with
  | Cpu_buffer native_buf ->
      Nx_native.
        { dtype = rune_t.dtype; buffer = native_buf; view = rune_t.view }
  | Symbolic_buffer sb ->
      (* Construct a descriptive error message *)
      let sym_id_str =
        Symbolic_id.pp Format.str_formatter sb.id;
        Format.flush_str_formatter ()
      in
      let dtype_str = Dtype.to_string sb.dtype_repr in
      let shape_str = View.pp_int_array sb.shape_repr in
      failwith
        (Printf.sprintf
           "Nx_rune.unwrap_to_nx_native: Attempted to unwrap a symbolic tensor \
            (ID: %s, Dtype: %s, Shape: %s) to a concrete Nx_native tensor. \
            Symbolic tensors cannot be used in eager/concrete backend \
            execution without a handler."
           sym_id_str dtype_str shape_str)

(* wrap_from_nx_native remains the same *)
let wrap_from_nx_native (native_t : ('a, 'b) Nx_native.t) : ('a, 'b) t =
  {
    dtype = Nx_native.dtype native_t;
    buffer = Cpu_buffer (Nx_native.data native_t);
    view = Nx_native.view native_t;
  }

(* unwrap_to_nx_native_context remains the same *)
let unwrap_to_nx_native_context (rune_ctx : context) : Nx_native.context =
  match rune_ctx with Cpu_context native_ctx -> native_ctx

(** [get_symbolic_info t] returns (SymbolicId, Dtype, Shape) if [t] is symbolic,
    else None. The Dtype and Shape returned are those stored with the symbolic
    buffer, which should represent its abstract properties. The view on the
    tensor [t] might be different due to subsequent lazy movement operations. *)
let get_symbolic_info (type a b) (t : (a, b) t) :
    (Symbolic_id.t * (a, b) Dtype.t * int array) option =
  match t.buffer with
  | Symbolic_buffer sb ->
      (* Ensure type consistency. This check is more for sanity; GADT should
         enforce. *)
      if Dtype.eq sb.dtype_repr t.dtype then
        Some (sb.id, sb.dtype_repr, sb.shape_repr)
      else
        (* This case should ideally not happen if Symbolic_buffer is constructed
           correctly relative to the outer t.dtype. It's a potential internal
           inconsistency. *)
        failwith
          (Printf.sprintf
             "Nx_rune.get_symbolic_info: Dtype mismatch between tensor shell \
              (%s) and symbolic buffer metadata (%s) for ID %s."
             (Dtype.to_string t.dtype)
             (Dtype.to_string sb.dtype_repr)
             (Symbolic_id.pp Format.str_formatter sb.id;
              Format.flush_str_formatter ()))
  | _ -> None

(* Lenses *)
let view (t : ('a, 'b) t) : View.t = t.view
let dtype (t : ('a, 'b) t) : ('a, 'b) Dtype.t = t.dtype

let is_symbolic (t : ('a, 'b) t) : bool =
  match t.buffer with Symbolic_buffer _ -> true | _ -> false

(* Context creation *)
let create_context () : context =
  (* Default to CPU context. In a multi-device setup, this might involve device
     selection based on environment variables or explicit API calls. *)
  Cpu_context (Nx_native.create_context ())

(* UOps *)
let op_buffer (rune_ctx : context) (dt : ('a, 'b) Dtype.t)
    (size_in_elements : int) : ('a, 'b) t =
  try
    Effect.perform
      (E_buffer { context = rune_ctx; dtype = dt; size_in_elements })
  with Effect.Unhandled _ ->
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    let native_result = Nx_native.op_buffer native_ctx dt size_in_elements in
    wrap_from_nx_native native_result

let op_const_scalar (rune_ctx : context) (value : 'a) (dt : ('a, 'b) Dtype.t) :
    ('a, 'b) t =
  try Effect.perform (E_const_scalar { context = rune_ctx; value; dtype = dt })
  with Effect.Unhandled _ ->
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    let native_result = Nx_native.op_const_scalar native_ctx value dt in
    wrap_from_nx_native native_result

let op_add (rune_ctx : context) (a : ('a, 'b) t) (b : ('a, 'b) t) : ('a, 'b) t =
  try Effect.perform (E_add { context = rune_ctx; a; b })
  with Effect.Unhandled _ ->
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    let native_a = unwrap_to_nx_native a in
    let native_b = unwrap_to_nx_native b in
    let native_result = Nx_native.op_add native_ctx native_a native_b in
    wrap_from_nx_native native_result

let op_mul (rune_ctx : context) (a : ('a, 'b) t) (b : ('a, 'b) t) : ('a, 'b) t =
  try Effect.perform (E_mul { context = rune_ctx; a; b })
  with Effect.Unhandled _ ->
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    let native_a = unwrap_to_nx_native a in
    let native_b = unwrap_to_nx_native b in
    let native_result = Nx_native.op_mul native_ctx native_a native_b in
    wrap_from_nx_native native_result

let op_sum (rune_ctx : context) ~(axes : int array) ~keepdims
    (t_in : ('a, 'b) t) : ('a, 'b) t =
  try Effect.perform (E_sum { context = rune_ctx; t_in; axes; keepdims })
  with Effect.Unhandled _ ->
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    let native_t_in = unwrap_to_nx_native t_in in
    let native_result =
      Nx_native.op_reduce_sum native_ctx ~axes ~keepdims native_t_in
    in
    wrap_from_nx_native native_result

let op_expand (rune_ctx : context) (t_in : ('a, 'b) t)
    (new_target_shape : int array) : ('a, 'b) t =
  try Effect.perform (E_expand { context = rune_ctx; t_in; new_target_shape })
  with Effect.Unhandled _ ->
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    let native_t_in = unwrap_to_nx_native t_in in
    let native_result =
      Nx_native.op_expand native_ctx native_t_in new_target_shape
    in
    wrap_from_nx_native native_result

let op_reshape (rune_ctx : context) (t_in : ('a, 'b) t) (new_shape : int array)
    : ('a, 'b) t =
  try Effect.perform (E_reshape { context = rune_ctx; t_in; new_shape })
  with Effect.Unhandled _ ->
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    let native_t_in = unwrap_to_nx_native t_in in
    let native_result = Nx_native.op_reshape native_ctx native_t_in new_shape in
    wrap_from_nx_native native_result
