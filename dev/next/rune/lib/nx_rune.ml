open Nx_core

type ('a, 'b) buffer =
  | Cpu_buffer : ('a, 'b) Nx_native.buffer -> ('a, 'b) buffer

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
  | E_define_global : {
      context : context;
      name : string;
      t_in : ('a, 'b) t;
    }
      -> ('a, 'b) t Effect.t
  | E_range : {
      context : context;
      name_hint : string;
      bound : (int, 'c) t;
    }
      -> (int, 'c) t Effect.t
  | E_special : {
      context : context;
      name_hint : string;
      kind : Backend_intf.special_kind;
    }
      -> (int, 'c) t Effect.t

let unwrap_to_nx_native (rune_t : ('a, 'b) t) : ('a, 'b) Nx_native.t =
  match rune_t.buffer with
  | Cpu_buffer native_buf ->
      (* Construct the Nx_native.t record directly using its internal
         structure *)
      Nx_native.
        { dtype = rune_t.dtype; buffer = native_buf; view = rune_t.view }
(* Example for a future GPU backend: | B_Gpu _ -> failwith "unwrap_to_nx_native:
   Expected B_Cpu buffer for this Nx_native operation" *)

(** Wraps an Nx_native.t into an Nx_rune.t. *)
let wrap_from_nx_native (native_t : ('a, 'b) Nx_native.t) : ('a, 'b) t =
  {
    dtype = Nx_native.dtype native_t;
    buffer = Cpu_buffer (Nx_native.buffer native_t);
    view = Nx_native.view native_t;
  }

(** Unwraps an Nx_rune.context to its corresponding Nx_native.context. Raises
    Failure if the context is not for Nx_native. *)
let unwrap_to_nx_native_context (rune_ctx : context) : Nx_native.context =
  match rune_ctx with Cpu_context native_ctx -> native_ctx
(* Example for a future GPU backend: | Ctx_Gpu _ -> failwith
   "unwrap_to_nx_native_context: Expected Ctx_Cpu for this Nx_native
   operation" *)

(* Lenses *)
let view (t : ('a, 'b) t) : View.t = t.view
let dtype (t : ('a, 'b) t) : ('a, 'b) Dtype.t = t.dtype

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

let op_load (rune_ctx : context) ?valid_mask (buffer_source : ('a, 'b) t)
    (logical_indices : (int, Dtype.int32_elt) t list) : ('a, 'b) t =
  try
    Effect.perform
      (E_load { context = rune_ctx; buffer_source; logical_indices; valid_mask })
  with Effect.Unhandled _ ->
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    let native_source = unwrap_to_nx_native buffer_source in
    let native_indices = List.map unwrap_to_nx_native logical_indices in
    let native_valid_mask = Option.map unwrap_to_nx_native valid_mask in
    let native_result =
      Nx_native.op_load native_ctx ?valid_mask:native_valid_mask native_source
        native_indices
    in
    wrap_from_nx_native native_result

let op_store (rune_ctx : context) ?valid_mask (buffer_target : ('a, 'b) t)
    (logical_indices : (int, Dtype.int32_elt) t list)
    (scalar_value_to_store : ('a, 'b) t) : unit =
  try
    Effect.perform
      (E_store
         {
           context = rune_ctx;
           buffer_target;
           logical_indices;
           scalar_value_to_store;
           valid_mask;
         })
  with Effect.Unhandled _ ->
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    let native_target = unwrap_to_nx_native buffer_target in
    let native_indices = List.map unwrap_to_nx_native logical_indices in
    let native_value_to_store = unwrap_to_nx_native scalar_value_to_store in
    let native_valid_mask = Option.map unwrap_to_nx_native valid_mask in
    Nx_native.op_store native_ctx ?valid_mask:native_valid_mask native_target
      native_indices native_value_to_store

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
      Nx_native.op_sum native_ctx ~axes ~keepdims native_t_in
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

(* JIT specific ops *)
let op_define_global (rune_ctx : context) (name : string) (t_in : ('a, 'b) t) :
    ('a, 'b) t =
  try Effect.perform (E_define_global { context = rune_ctx; name; t_in })
  with Effect.Unhandled _ ->
    (* In eager mode, Nx_native.op_define_global is identity. We still call it
       for consistency and to allow Nx_native to do any (future) eager-mode
       bookkeeping. *)
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    let native_t_in = unwrap_to_nx_native t_in in
    let native_result =
      Nx_native.op_define_global native_ctx name native_t_in
    in
    if native_result == native_t_in then t_in
      (* Physical equality optimization if Nx_native returns same tensor *)
    else wrap_from_nx_native native_result

let op_range (rune_ctx : context) (name_hint : string) (bound : (int, 'b) t) :
    (int, 'b) t =
  try Effect.perform (E_range { context = rune_ctx; name_hint; bound })
  with Effect.Unhandled _ ->
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    let native_bound = unwrap_to_nx_native bound in
    (* Nx_native.op_range correctly raises an error in eager mode. *)
    let native_result = Nx_native.op_range native_ctx name_hint native_bound in
    wrap_from_nx_native native_result

let op_special (rune_ctx : context) (name_hint : string)
    (kind : Backend_intf.special_kind) : (int, 'b) t =
  try Effect.perform (E_special { context = rune_ctx; name_hint; kind })
  with Effect.Unhandled _ ->
    let native_ctx = unwrap_to_nx_native_context rune_ctx in
    (* Nx_native.op_special correctly raises an error or returns a default in
       eager mode. *)
    let native_result = Nx_native.op_special native_ctx name_hint kind in
    wrap_from_nx_native native_result
