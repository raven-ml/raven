(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core

(* Types *)

type context = Nx_backend.context

(* OCaml extensible GADT constructors (the [E_add], [E_mul], ... below) require
   that type variables in the payload be deducible from the return type. With a
   transparent alias [type ('a,'b) t = ('a,'b) Nx_backend.t], the compiler sees
   through to the concrete record and concludes that ['a] and ['b] are not
   injective — so every effect definition fails with "type variable cannot be
   deduced". Wrapping in a single-constructor GADT restores injectivity: [T] is
   a fresh constructor whose parameters are, by definition, determined by the
   return type. At runtime this is a zero-cost box (single-field
   constructor).

   [Deferred] is a host tensor whose bytes have not arrived yet: it carries its
   metadata (context, dtype, shape) plus a fill thunk that produces the host
   buffer on first data access. Rune's jit uses it to keep compiled outputs
   resident on non-host-sharing devices (CUDA, Metal): metadata reads answer
   from the record, and only a data access runs the thunk — which copies the
   device buffer out — memoizing the resulting backend tensor. A deferred
   tensor is never a trace placeholder, so its metadata never needs an
   effect. *)
type ('a, 'b) t =
  | T : ('a, 'b) Nx_backend.t -> ('a, 'b) t
  | Deferred : ('a, 'b) deferred -> ('a, 'b) t

and ('a, 'b) deferred = {
  d_id : int; (* fresh; lets creators key side tables by handle *)
  d_context : Nx_backend.context;
  d_dtype : ('a, 'b) Dtype.t;
  d_view : View.t; (* C-contiguous over the tensor's shape *)
  d_fill : unit -> ('a, 'b) Nx_buffer.t; (* runs at most once *)
  mutable d_forced : ('a, 'b) Nx_backend.t option;
}

(* [Nx_buffer.t] is not injective in its parameters either (it abbreviates a
   bigarray), so a host buffer cannot be an effect's result directly; the same
   boxing trick restores deducibility for [E_to_host]. *)
type ('a, 'b) host_buffer =
  | Host_buffer : ('a, 'b) Nx_buffer.t -> ('a, 'b) host_buffer

(* Effects *)

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
  | E_from_host : {
      context : context;
      array : ('a, 'b) Nx_buffer.t;
    }
      -> ('a, 'b) t Effect.t
  | E_add : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sub : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_mul : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_idiv : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_fdiv : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_max : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_min : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_mod : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_pow : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_xor : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_or : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_and : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_atan2 : { a : ('a, 'b) t; b : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_cmpeq : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> (bool, Dtype.bool_elt) t Effect.t
  | E_cmpne : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> (bool, Dtype.bool_elt) t Effect.t
  | E_cmplt : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> (bool, Dtype.bool_elt) t Effect.t
  | E_cmple : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
    }
      -> (bool, Dtype.bool_elt) t Effect.t
  | E_neg : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sin : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sqrt : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_recip : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_log : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_exp : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_cos : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_abs : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sign : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_tan : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_asin : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_acos : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_atan : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_sinh : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_cosh : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_tanh : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_trunc : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_ceil : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_floor : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_round : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_erf : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
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
  | E_reduce_min : {
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
  | E_argmax : {
      t_in : ('a, 'b) t;
      axis : int;
      keepdims : bool;
    }
      -> (int32, Dtype.int32_elt) t Effect.t
  | E_argmin : {
      t_in : ('a, 'b) t;
      axis : int;
      keepdims : bool;
    }
      -> (int32, Dtype.int32_elt) t Effect.t
  | E_sort : {
      t_in : ('a, 'b) t;
      axis : int;
      descending : bool;
    }
      -> ('a, 'b) t Effect.t
  | E_argsort : {
      t_in : ('a, 'b) t;
      axis : int;
      descending : bool;
    }
      -> (int32, Dtype.int32_elt) t Effect.t
  | E_associative_scan : {
      t_in : ('a, 'b) t;
      axis : int;
      op : [ `Sum | `Prod | `Max | `Min ];
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
      mode : [ `Set | `Add ];
      unique_indices : bool;
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
      dtype : (Complex.t, 'c) Dtype.t;
      axes : int array;
    }
      -> (Complex.t, 'c) t Effect.t
  | E_irfft : {
      t : (Complex.t, 'b) t;
      dtype : (float, 'c) Dtype.t;
      axes : int array;
      s : int array option;
    }
      -> (float, 'c) t Effect.t
  | E_psum : { t_in : ('a, 'b) t } -> ('a, 'b) t Effect.t
  | E_cholesky : { t_in : ('a, 'b) t; upper : bool } -> ('a, 'b) t Effect.t
  | E_qr : {
      t_in : ('a, 'b) t;
      reduced : bool;
    }
      -> (('a, 'b) t * ('a, 'b) t) Effect.t
  | E_svd : {
      t_in : ('a, 'b) t;
      full_matrices : bool;
    }
      -> (('a, 'b) t * (float, Dtype.float64_elt) t * ('a, 'b) t) Effect.t
  | E_eig : {
      t_in : ('a, 'b) t;
      vectors : bool;
    }
      -> ((Complex.t, Dtype.complex64_elt) t
         * (Complex.t, Dtype.complex64_elt) t option)
         Effect.t
  | E_eigh : {
      t_in : ('a, 'b) t;
      vectors : bool;
    }
      -> ((float, Dtype.float64_elt) t * ('a, 'b) t option) Effect.t
  | E_triangular_solve : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
      upper : bool;
      transpose : bool;
      unit_diag : bool;
    }
      -> ('a, 'b) t Effect.t
  | E_to_host : ('a, 'b) t -> ('a, 'b) host_buffer Effect.t

(* Unwrap. Forcing a deferred tensor runs its fill thunk once, wraps the
   resulting host buffer as a backend tensor of the recorded shape, and
   memoizes it: after that the handle behaves as a plain host tensor, and later
   mutations of the memoized tensor are observed through the handle. *)

let force (type a b) (d : (a, b) deferred) : (a, b) Nx_backend.t =
  match d.d_forced with
  | Some t -> t
  | None ->
      let host = d.d_fill () in
      let t =
        Nx_backend.reshape
          (Nx_backend.from_host d.d_context host)
          (View.shape d.d_view)
      in
      d.d_forced <- Some t;
      t

let unwrap : type a b. (a, b) t -> (a, b) Nx_backend.t = function
  | T t -> t
  | Deferred d -> force d

(* Deferred constructors *)

let deferred_id_counter = ref 0

let deferred (type a b) (ctx : context) (dtype : (a, b) Dtype.t)
    (shape : int array) (fill : unit -> (a, b) Nx_buffer.t) : (a, b) t =
  incr deferred_id_counter;
  Deferred
    {
      d_id = !deferred_id_counter;
      d_context = ctx;
      d_dtype = dtype;
      d_view = View.create shape;
      d_fill = fill;
      d_forced = None;
    }

(* [None] once forced: the handle is then a plain host tensor and its creator's
   side state (for example a resident device buffer) is gone. *)
let deferred_id : type a b. (a, b) t -> int option = function
  | T _ -> None
  | Deferred d -> ( match d.d_forced with Some _ -> None | None -> Some d.d_id)

(* Lenses. Metadata reads on a deferred tensor answer from its record without
   running the fill thunk. The [E_view] effect is still performed first: a
   handler may present a transformed view (vmap shows batched tensors without
   their batch axis); only the unhandled fallback answers from the record. *)

let create_context () : context = Nx_backend.create_context ()

let context : type a b. (a, b) t -> context = function
  | T t -> Nx_backend.context t
  | Deferred d -> d.d_context

let to_device (_ctx : context) (t : ('a, 'b) t) : ('a, 'b) t = t

let view (type a b) (x : (a, b) t) : View.t =
  try Effect.perform (E_view x)
  with Effect.Unhandled _ -> (
    match x with
    | T t -> Nx_backend.view t
    | Deferred d -> (
        match d.d_forced with Some t -> Nx_backend.view t | None -> d.d_view))

let dtype : type a b. (a, b) t -> (a, b) Dtype.t = function
  | T t -> Nx_backend.dtype t
  | Deferred d -> d.d_dtype

let to_host (type a b) (x : (a, b) t) : (a, b) Nx_buffer.t =
  try
    let (Host_buffer buf) = Effect.perform (E_to_host x) in
    buf
  with Effect.Unhandled _ -> Nx_backend.to_host (unwrap x)

(* Fallback dispatch helpers.

   Each helper performs an effect. When no handler is installed, it falls back
   to the C backend. The pattern is uniform: try the effect, on [Unhandled]
   unwrap the [T] and call [Nx_backend]. *)

let binary_op eff cpu_op a b =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> T (cpu_op (unwrap a) (unwrap b))

let unary_op eff cpu_op t_in =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> T (cpu_op (unwrap t_in))

let movement_op eff cpu_op t_in arg =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> T (cpu_op (unwrap t_in) arg)

let assign dst src =
  try Effect.perform (E_assign { dst; src })
  with Effect.Unhandled _ -> Nx_backend.assign (unwrap dst) (unwrap src)

(* Binary operations *)

let add a b = binary_op (fun () -> E_add { a; b }) Nx_backend.add a b
let sub a b = binary_op (fun () -> E_sub { a; b }) Nx_backend.sub a b
let mul a b = binary_op (fun () -> E_mul { a; b }) Nx_backend.mul a b
let max a b = binary_op (fun () -> E_max { a; b }) Nx_backend.max a b
let min a b = binary_op (fun () -> E_min { a; b }) Nx_backend.min a b
let mod_ a b = binary_op (fun () -> E_mod { a; b }) Nx_backend.mod_ a b
let pow a b = binary_op (fun () -> E_pow { a; b }) Nx_backend.pow a b
let xor a b = binary_op (fun () -> E_xor { a; b }) Nx_backend.xor a b
let or_ a b = binary_op (fun () -> E_or { a; b }) Nx_backend.or_ a b
let and_ a b = binary_op (fun () -> E_and { a; b }) Nx_backend.and_ a b
let atan2 a b = binary_op (fun () -> E_atan2 { a; b }) Nx_backend.atan2 a b

let fdiv a b = binary_op (fun () -> E_fdiv { a; b }) Nx_backend.fdiv a b
let idiv a b = binary_op (fun () -> E_idiv { a; b }) Nx_backend.idiv a b

(* Comparison operations *)

let cmpeq a b =
  try Effect.perform (E_cmpeq { a; b })
  with Effect.Unhandled _ -> T (Nx_backend.cmpeq (unwrap a) (unwrap b))

let cmpne a b =
  try Effect.perform (E_cmpne { a; b })
  with Effect.Unhandled _ -> T (Nx_backend.cmpne (unwrap a) (unwrap b))

let cmplt a b =
  try Effect.perform (E_cmplt { a; b })
  with Effect.Unhandled _ -> T (Nx_backend.cmplt (unwrap a) (unwrap b))

let cmple a b =
  try Effect.perform (E_cmple { a; b })
  with Effect.Unhandled _ -> T (Nx_backend.cmple (unwrap a) (unwrap b))

(* Unary operations *)

let neg t = unary_op (fun () -> E_neg { t_in = t }) Nx_backend.neg t
let sin t = unary_op (fun () -> E_sin { t_in = t }) Nx_backend.sin t
let sqrt t = unary_op (fun () -> E_sqrt { t_in = t }) Nx_backend.sqrt t
let recip t = unary_op (fun () -> E_recip { t_in = t }) Nx_backend.recip t
let log t = unary_op (fun () -> E_log { t_in = t }) Nx_backend.log t
let exp t = unary_op (fun () -> E_exp { t_in = t }) Nx_backend.exp t
let cos t = unary_op (fun () -> E_cos { t_in = t }) Nx_backend.cos t
let abs t = unary_op (fun () -> E_abs { t_in = t }) Nx_backend.abs t
let sign t = unary_op (fun () -> E_sign { t_in = t }) Nx_backend.sign t
let tan t = unary_op (fun () -> E_tan { t_in = t }) Nx_backend.tan t
let asin t = unary_op (fun () -> E_asin { t_in = t }) Nx_backend.asin t
let acos t = unary_op (fun () -> E_acos { t_in = t }) Nx_backend.acos t
let atan t = unary_op (fun () -> E_atan { t_in = t }) Nx_backend.atan t
let sinh t = unary_op (fun () -> E_sinh { t_in = t }) Nx_backend.sinh t
let cosh t = unary_op (fun () -> E_cosh { t_in = t }) Nx_backend.cosh t
let tanh t = unary_op (fun () -> E_tanh { t_in = t }) Nx_backend.tanh t
let trunc t = unary_op (fun () -> E_trunc { t_in = t }) Nx_backend.trunc t
let ceil t = unary_op (fun () -> E_ceil { t_in = t }) Nx_backend.ceil t
let floor t = unary_op (fun () -> E_floor { t_in = t }) Nx_backend.floor t
let round t = unary_op (fun () -> E_round { t_in = t }) Nx_backend.round t
let erf t = unary_op (fun () -> E_erf { t_in = t }) Nx_backend.erf t

let op_psum t_in =
  try Effect.perform (E_psum { t_in })
  with Effect.Unhandled _ -> failwith "psum must be used under vmap"

(* Reduction operations *)

let reduce ~op ~axes t_in =
  let eff =
    match op with
    | `Sum -> E_reduce_sum { t_in; axes; keepdims = false }
    | `Prod -> E_reduce_prod { t_in; axes; keepdims = false }
    | `Max -> E_reduce_max { t_in; axes; keepdims = false }
    | `Min -> E_reduce_min { t_in; axes; keepdims = false }
  in
  try Effect.perform eff
  with Effect.Unhandled _ -> T (Nx_backend.reduce ~op ~axes (unwrap t_in))

let argmax ~axis ~keepdims t_in =
  try Effect.perform (E_argmax { t_in; axis; keepdims })
  with Effect.Unhandled _ ->
    T (Nx_backend.argmax ~axis ~keepdims (unwrap t_in))

let argmin ~axis ~keepdims t_in =
  try Effect.perform (E_argmin { t_in; axis; keepdims })
  with Effect.Unhandled _ ->
    T (Nx_backend.argmin ~axis ~keepdims (unwrap t_in))

let associative_scan ~axis ~op t_in =
  try Effect.perform (E_associative_scan { t_in; axis; op })
  with Effect.Unhandled _ ->
    T (Nx_backend.associative_scan ~axis ~op (unwrap t_in))

let sort ~axis ~descending t_in =
  try Effect.perform (E_sort { t_in; axis; descending })
  with Effect.Unhandled _ ->
    T (Nx_backend.sort ~axis ~descending (unwrap t_in))

let argsort ~axis ~descending t_in =
  try Effect.perform (E_argsort { t_in; axis; descending })
  with Effect.Unhandled _ ->
    T (Nx_backend.argsort ~axis ~descending (unwrap t_in))

(* Movement operations *)

let reshape t_in new_shape =
  movement_op
    (fun () -> E_reshape { t_in; new_shape })
    Nx_backend.reshape t_in new_shape

let expand t_in new_target_shape =
  movement_op
    (fun () -> E_expand { t_in; new_target_shape })
    Nx_backend.expand t_in new_target_shape

let permute t_in axes =
  movement_op (fun () -> E_permute { t_in; axes }) Nx_backend.permute t_in axes

let shrink t_in limits =
  movement_op
    (fun () -> E_shrink { t_in; limits })
    Nx_backend.shrink t_in limits

let flip t_in dims_to_flip =
  movement_op
    (fun () -> E_flip { t_in; dims_to_flip })
    Nx_backend.flip t_in dims_to_flip

let pad t_in padding_config fill_value =
  try Effect.perform (E_pad { t_in; padding_config; fill_value })
  with Effect.Unhandled _ ->
    T (Nx_backend.pad (unwrap t_in) padding_config fill_value)

(* Creation operations *)

let buffer ctx dtype shape_arr =
  let size_in_elements = Array.fold_left ( * ) 1 shape_arr in
  let flat =
    try Effect.perform (E_buffer { context = ctx; dtype; size_in_elements })
    with Effect.Unhandled _ -> T (Nx_backend.buffer ctx dtype shape_arr)
  in
  reshape flat shape_arr

let const_scalar ctx value dtype =
  try Effect.perform (E_const_scalar { context = ctx; value; dtype })
  with Effect.Unhandled _ -> T (Nx_backend.full ctx dtype [||] value)

let full ctx dtype shape_arr value =
  (* Under an effect handler (jit tracing), a filled tensor is a broadcast
     scalar constant: no bytes are materialized and none enter the trace.
     Without a handler it stays a concrete, mutable backend tensor. *)
  match Effect.perform (E_const_scalar { context = ctx; value; dtype }) with
  | scalar ->
      if Array.length shape_arr = 0 then scalar
      else expand (reshape scalar (Array.map (fun _ -> 1) shape_arr)) shape_arr
  | exception Effect.Unhandled _ ->
      T (Nx_backend.full ctx dtype shape_arr value)

let from_host ctx array =
  try Effect.perform (E_from_host { context = ctx; array })
  with Effect.Unhandled _ -> T (Nx_backend.from_host ctx array)

(* Copy operations *)

let contiguous t_in =
  try Effect.perform (E_contiguous { t_in })
  with Effect.Unhandled _ -> T (Nx_backend.contiguous (unwrap t_in))

let copy t_in =
  try Effect.perform (E_copy { t_in })
  with Effect.Unhandled _ -> T (Nx_backend.copy (unwrap t_in))

(* Ternary operations *)

let where condition if_true if_false =
  try Effect.perform (E_where { condition; if_true; if_false })
  with Effect.Unhandled _ ->
    T (Nx_backend.where (unwrap condition) (unwrap if_true) (unwrap if_false))

(* Cat *)

let cat t_list ~axis =
  try Effect.perform (E_cat { t_list; axis })
  with Effect.Unhandled _ -> T (Nx_backend.cat (List.map unwrap t_list) ~axis)

(* Cast *)

let cast (type a b c d) ~(dtype : (c, d) Dtype.t) (t_in : (a, b) t) : (c, d) t =
  let target_dtype = dtype in
  try Effect.perform (E_cast { t_in; target_dtype })
  with Effect.Unhandled _ ->
    T (Nx_backend.cast ~dtype:target_dtype (unwrap t_in))

(* Indexed access *)

let gather data indices ~axis =
  try Effect.perform (E_gather { data; indices; axis })
  with Effect.Unhandled _ ->
    T (Nx_backend.gather (unwrap data) (unwrap indices) ~axis)

let scatter ?(mode = `Set) ?(unique_indices = false) data_template ~indices
    ~updates ~axis =
  try
    Effect.perform
      (E_scatter { data_template; indices; updates; axis; mode; unique_indices })
  with Effect.Unhandled _ ->
    T
      (Nx_backend.scatter ~mode ~unique_indices (unwrap data_template)
         ~indices:(unwrap indices) ~updates:(unwrap updates) ~axis)

(* Random *)

let threefry key ctr =
  try Effect.perform (E_threefry { key; ctr })
  with Effect.Unhandled _ -> T (Nx_backend.threefry (unwrap key) (unwrap ctr))

(* Window operations *)

let unfold t_in ~kernel_size ~stride ~dilation ~padding =
  try Effect.perform (E_unfold { t_in; kernel_size; stride; dilation; padding })
  with Effect.Unhandled _ ->
    T (Nx_backend.unfold (unwrap t_in) ~kernel_size ~stride ~dilation ~padding)

let fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding =
  try
    Effect.perform
      (E_fold { t_in; output_size; kernel_size; stride; dilation; padding })
  with Effect.Unhandled _ ->
    T
      (Nx_backend.fold (unwrap t_in) ~output_size ~kernel_size ~stride ~dilation
         ~padding)

(* Matrix operations *)

let matmul a b =
  try Effect.perform (E_matmul { a; b })
  with Effect.Unhandled _ -> T (Nx_backend.matmul (unwrap a) (unwrap b))

(* FFT operations *)

let fft t ~axes =
  try Effect.perform (E_fft { t; axes })
  with Effect.Unhandled _ -> T (Nx_backend.fft (unwrap t) ~axes)

let ifft t ~axes =
  try Effect.perform (E_ifft { t; axes })
  with Effect.Unhandled _ -> T (Nx_backend.ifft (unwrap t) ~axes)

let rfft (type a c) (t : (float, a) t) ~(dtype : (Complex.t, c) Dtype.t) ~axes
    : (Complex.t, c) t =
  try Effect.perform (E_rfft { t; dtype; axes })
  with Effect.Unhandled _ -> T (Nx_backend.rfft (unwrap t) ~dtype ~axes)

let irfft (type a c) ?s (t : (Complex.t, a) t) ~(dtype : (float, c) Dtype.t)
    ~axes : (float, c) t =
  try Effect.perform (E_irfft { t; dtype; axes; s })
  with Effect.Unhandled _ -> T (Nx_backend.irfft ?s (unwrap t) ~dtype ~axes)

(* Linear algebra *)

let cholesky ~upper t_in =
  try Effect.perform (E_cholesky { t_in; upper })
  with Effect.Unhandled _ -> T (Nx_backend.cholesky ~upper (unwrap t_in))

let qr ~reduced t_in =
  try Effect.perform (E_qr { t_in; reduced })
  with Effect.Unhandled _ ->
    let q, r = Nx_backend.qr ~reduced (unwrap t_in) in
    (T q, T r)

let svd ~full_matrices t_in =
  try Effect.perform (E_svd { t_in; full_matrices })
  with Effect.Unhandled _ ->
    let u, s, vt = Nx_backend.svd ~full_matrices (unwrap t_in) in
    (T u, T s, T vt)

let eig ~vectors t_in =
  try Effect.perform (E_eig { t_in; vectors })
  with Effect.Unhandled _ ->
    let vals, vecs_opt = Nx_backend.eig ~vectors (unwrap t_in) in
    (T vals, Option.map (fun v -> T v) vecs_opt)

let eigh ~vectors t_in =
  try Effect.perform (E_eigh { t_in; vectors })
  with Effect.Unhandled _ ->
    let vals, vecs_opt = Nx_backend.eigh ~vectors (unwrap t_in) in
    (T vals, Option.map (fun v -> T v) vecs_opt)

let triangular_solve ~upper ~transpose ~unit_diag a b =
  try Effect.perform (E_triangular_solve { a; b; upper; transpose; unit_diag })
  with Effect.Unhandled _ ->
    T
      (Nx_backend.triangular_solve ~upper ~transpose ~unit_diag (unwrap a)
         (unwrap b))
