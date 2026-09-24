(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_core

(* Types

   OCaml extensible GADT constructors (the [E_add], [E_mul], ... below) require
   that type variables in the payload be deducible from the return type. A
   transparent alias of [Nx_backend.t] would not be injective, so the tensor is
   a GADT of its own, whose parameters the return type determines.

   A tensor is one of three things. [Host] is a tensor of the link-time engine,
   the host's. [Placed] is a value that an engine carried by a device holds on a
   device list other than the host's: nx knows its placement, dtype and view,
   and the engine alone knows its storage. [Traced] is a node of a trace: it has
   no bytes and never will, and the tracer that made it keeps its payload in
   [t_node].

   The views of one placed storage share one cell, which holds what belongs to
   the storage rather than to a view: whether it is live or was donated to a
   compiled call, and how many reachable programs bind it.

   A context says where a creation effect makes its value. It is declared over
   any device type, ahead of the recursive definition, so that its [Host] stays
   apart from the tensor's. *)

type 'device context_of = Host of Nx_backend.context | On of 'device list

type ('a, 'b) t =
  | Host : ('a, 'b) Nx_backend.t -> ('a, 'b) t
  | Placed : ('a, 'b) resident -> ('a, 'b) t
  | Traced : ('a, 'b) traced -> ('a, 'b) t

and ('a, 'b) resident = {
  r_id : int; (* fresh per value; identity tables key by it *)
  r_placement : placement; (* never the host *)
  r_dtype : ('a, 'b) Dtype.t;
  r_view : View.t; (* per shard, the same on every shard *)
  r_cell : cell; (* one per storage, shared by all its views *)
}

and cell = {
  engine : engine;
  length : int; (* elements of the storage, per shard *)
  mutable state : state;
  mutable bound : int; (* reachable programs that bind the storage *)
}

and state = Live of storage | Donated

and ('a, 'b) traced = {
  t_id : int; (* fresh; identity tables key by it *)
  t_context : device context_of;
  t_dtype : ('a, 'b) Dtype.t;
  t_view : View.t; (* C-contiguous over the tensor's shape *)
  t_node : node; (* the tracer's payload *)
}

and engine = {
  read : 'a 'b. ('a, 'b) resident -> ('a, 'b) Nx_buffer.t;
      (* the view's elements, in C order, in a buffer the caller owns *)
  place : 'a 'b. placement -> ('a, 'b) t -> ('a, 'b) t;
      (* the value on a placement of this engine; its source stays. Placing an
         empty value allocates nothing, and raises [Invalid_argument] if the
         placement cannot hold the dtype: nx checks held values this way. *)
}

and device = { d_id : int; d_name : string; d_engine : engine }

and placement =
  | Device of device
  | Replicated of device list
  | Sharded of { axis : int; devices : device list }

and storage = ..
and node = ..

type context = device context_of

(* A value of one element that nx holds itself: a scalar created in a device
   context, or a one-element result. It allocates nothing on the device; an
   engine passes it to a program as it passes a host value. *)
type storage += Held : ('a, 'b) Dtype.t * 'a -> storage

let id_counter = ref 0

let fresh_id () =
  incr id_counter;
  !id_counter

(* Ids are handed out in increasing order, so a tracer can tell the traced
   tensors made before a point of its trace from those made after it. *)
let next_traced_id () = !id_counter + 1
let host_context = Nx_backend.create_context ()

let outside_trace () =
  invalid_arg
    "a traced tensor has no bytes; it was used outside the trace that made it"

let donated () =
  invalid_arg
    "this value was donated to a compiled call and no longer exists; read or \
     copy it before the call"

(* Reading placed values *)

(* The elements of a placed value's view. A held value's are its one element,
   broadcast. *)
let read_elements (type a b) (r : (a, b) resident) : (a, b) Nx_buffer.t =
  match r.r_cell.state with
  | Donated -> donated ()
  | Live (Held (dt, v)) -> (
      let buf = Nx_buffer.create r.r_dtype (View.numel r.r_view) in
      match Dtype.equal_witness dt r.r_dtype with
      | Some Type.Equal ->
          Nx_buffer.fill buf v;
          buf
      | None -> assert false)
  | Live _ -> r.r_cell.engine.read r

(* A split value's view is each shard's, and its shape the whole's. *)
let whole_view r =
  match r.r_placement with
  | Sharded { axis; devices } ->
      let v = r.r_view in
      let shape = Array.copy (View.shape v) in
      shape.(axis) <- shape.(axis) * List.length devices;
      View.create ~offset:(View.offset v) ~strides:(View.strides v) shape
  | Device _ | Replicated _ -> r.r_view

let read_host (type a b) (r : (a, b) resident) : (a, b) Nx_backend.t =
  Nx_backend.reshape
    (Nx_backend.from_host host_context (read_elements r))
    (View.shape (whole_view r))

(* [host_of x] is [x]'s value as a host tensor: [x] itself on the host, a copy
   of its view's elements when it is placed. *)
let host_of : type a b. (a, b) t -> (a, b) Nx_backend.t = function
  | Host t -> t
  | Placed r -> read_host r
  | Traced _ -> outside_trace ()

(* Devices *)

module Device = struct
  type t = device

  exception Out_of_memory of t * int

  let make name engine =
    { d_id = fresh_id (); d_name = name; d_engine = engine }

  let name d = d.d_name
  let engine d = d.d_engine
  let equal = ( == )
  let compare a b = Int.compare a.d_id b.d_id
  let pp ppf d = Format.pp_print_string ppf d.d_name

  (* The host holds no placed value: a value moves to it by a read. *)
  let rec host_engine =
    {
      read = (fun _ -> invalid_arg "the host holds no placed value");
      place =
        (fun p x ->
          match p with
          | Device d when d.d_engine == host_engine -> Host (host_of x)
          | _ -> invalid_arg "the host engine places values on the host only");
    }

  let host = make "CPU" host_engine

  let () =
    Printexc.register_printer (function
      | Out_of_memory (d, n) ->
          Some
            (Printf.sprintf "Nx.Device.Out_of_memory(%s, %d bytes)" d.d_name n)
      | _ -> None)
end

(* Placements *)

module Placement = struct
  type t = placement =
    | Device of device
    | Replicated of device list
    | Sharded of { axis : int; devices : device list }

  let host = Device Device.host
  let device d = Device d

  let devices = function
    | Device d -> [ d ]
    | Replicated ds | Sharded { devices = ds; _ } -> ds

  let engine p = (List.hd (devices p)).d_engine
  let is_host = function Device d -> d == Device.host | _ -> false

  let check what ds =
    let fail fmt =
      Printf.ksprintf invalid_arg ("Nx.Placement.%s: " ^^ fmt) what
    in
    let rec distinct = function
      | [] -> ()
      | d :: rest ->
          if List.memq d rest then fail "%s appears twice" d.d_name;
          distinct rest
    in
    match ds with
    | [] -> fail "no device"
    | d :: rest ->
        distinct ds;
        List.iter
          (fun d' ->
            if d'.d_engine != d.d_engine then
              fail "%s and %s belong to different engines" d.d_name d'.d_name)
          rest

  let replicated ds =
    check "replicated" ds;
    match ds with [ d ] -> Device d | ds -> Replicated ds

  let sharded ~axis ds =
    if axis < 0 then
      invalid_arg (Printf.sprintf "Nx.Placement.sharded: axis %d < 0" axis);
    check "sharded" ds;
    match ds with [ d ] -> Device d | ds -> Sharded { axis; devices = ds }

  let equal a b =
    match (a, b) with
    | Device a, Device b -> a == b
    | Replicated a, Replicated b -> List.equal ( == ) a b
    | Sharded a, Sharded b ->
        a.axis = b.axis && List.equal ( == ) a.devices b.devices
    | _ -> false

  let pp ppf p =
    let list ppf ds =
      Format.pp_print_list
        ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
        Device.pp ppf ds
    in
    match p with
    | Device d -> Device.pp ppf d
    | Replicated ds -> Format.fprintf ppf "replicated [%a]" list ds
    | Sharded { axis; devices } ->
        Format.fprintf ppf "sharded ~axis:%d [%a]" axis list devices
end

(* Placed constructors, for engines *)

(* A cell over [storage] of [length] elements, which [engine] owns. The engine
   attaches the finaliser that releases the storage. *)
let cell engine ~length storage =
  { engine; length; state = Live storage; bound = 0 }

let placed placement dtype view cell =
  if Placement.is_host placement then
    invalid_arg "Nx_effect.placed: a placed value is never on the host";
  Placed
    {
      r_id = fresh_id ();
      r_placement = placement;
      r_dtype = dtype;
      r_view = view;
      r_cell = cell;
    }

(* Whether [r]'s view covers its storage: C order, offset 0, every element. *)
let covers r =
  View.is_c_contiguous r.r_view
  && View.offset r.r_view = 0
  && View.numel r.r_view = r.r_cell.length

(* A held value of shape [shape] on [p]. The engine is asked to place an empty
   value of the dtype first, which allocates nothing and raises if [p] cannot
   hold the dtype. *)
let held p dtype value shape =
  let engine = Placement.engine p in
  ignore (engine.place p (Host (Nx_backend.buffer host_context dtype [| 0 |])));
  placed p dtype (View.create shape)
    (cell engine ~length:1 (Held (dtype, value)))

(* A hash for identity tables. A placed or traced value hashes by its id, which
   never changes; a host tensor by its structure, which no table sees change,
   since tensors are values. *)
let identity_hash : type a b. (a, b) t -> int = function
  | Host _ as x -> Hashtbl.hash x
  | Placed r -> r.r_id
  | Traced t -> t.t_id

(* Traced constructor *)

let traced (type a b) (ctx : context) (dtype : (a, b) Dtype.t)
    (shape : int array) (node : node) : (a, b) t =
  Traced
    {
      t_id = fresh_id ();
      t_context = ctx;
      t_dtype = dtype;
      t_view = View.create shape;
      t_node = node;
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
    }
      -> ('a, 'b) t Effect.t
  | E_reduce_max : {
      t_in : ('a, 'b) t;
      axes : int array;
    }
      -> ('a, 'b) t Effect.t
  | E_reduce_min : {
      t_in : ('a, 'b) t;
      axes : int array;
    }
      -> ('a, 'b) t Effect.t
  | E_reduce_prod : {
      t_in : ('a, 'b) t;
      axes : int array;
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
  | E_sliding_window : {
      t_in : ('a, 'b) t;
      axis : int;
      window : int;
      step : int;
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
  | E_update : {
      t_in : ('a, 'b) t;
      starts : (int32, Dtype.int32_elt) t;
      v : ('a, 'b) t;
    }
      -> ('a, 'b) t Effect.t
  | E_place : {
      placement : placement;
      t_in : ('a, 'b) t;
    }
      -> ('a, 'b) t Effect.t
  | E_placement : ('a, 'b) t -> placement Effect.t
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
  | E_axis_index : (int32, Dtype.int32_elt) t Effect.t
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
  | E_eigvals : {
      t_in : ('a, 'b) t;
    }
      -> (Complex.t, Dtype.complex64_elt) t Effect.t
  | E_eig : {
      t_in : ('a, 'b) t;
    }
      -> ((Complex.t, Dtype.complex64_elt) t
         * (Complex.t, Dtype.complex64_elt) t)
         Effect.t
  | E_eigvalsh : { t_in : ('a, 'b) t } -> (float, Dtype.float64_elt) t Effect.t
  | E_eigh : {
      t_in : ('a, 'b) t;
    }
      -> ((float, Dtype.float64_elt) t * ('a, 'b) t) Effect.t
  | E_solve_triangular : {
      a : ('a, 'b) t;
      b : ('a, 'b) t;
      upper : bool;
      transpose : bool;
      unit_diag : bool;
    }
      -> ('a, 'b) t Effect.t
  | E_to_host : ('a, 'b) t -> ('a, 'b) host_buffer Effect.t

(* Lenses. The effect is performed first: a handler may present a transformed
   view (vmap shows batched tensors without their batch axis) or placement; only
   the unhandled fallback answers from the tensor. *)

let create_context () : context = Host (Nx_backend.create_context ())

(* The context of host tensors, allocated once: the frontend asks for a context
   each time it builds a constant beside an operand. *)
let host_tensor_context : context = Host host_context

let context : type a b. (a, b) t -> context = function
  | Host t ->
      let c = Nx_backend.context t in
      if c == host_context then host_tensor_context else Host c
  | Placed r -> On (Placement.devices r.r_placement)
  | Traced t -> t.t_context

let view (type a b) (x : (a, b) t) : View.t =
  try Effect.perform (E_view x)
  with Effect.Unhandled _ -> (
    match x with
    | Host t -> Nx_backend.view t
    | Placed r -> whole_view r
    | Traced t -> t.t_view)

let dtype : type a b. (a, b) t -> (a, b) Dtype.t = function
  | Host t -> Nx_backend.dtype t
  | Placed r -> r.r_dtype
  | Traced t -> t.t_dtype

let placement (type a b) (x : (a, b) t) : placement =
  try Effect.perform (E_placement x)
  with Effect.Unhandled _ -> (
    match x with
    | Host _ -> Placement.host
    | Placed r -> r.r_placement
    | Traced _ -> outside_trace ())

(* The host engine's storage of a host value, and the view's elements of a
   placed one: readers take [contiguous] first, so the view is the storage. *)
let to_host (type a b) (x : (a, b) t) : (a, b) Nx_buffer.t =
  try
    let (Host_buffer buf) = Effect.perform (E_to_host x) in
    buf
  with Effect.Unhandled _ -> (
    match x with
    | Host t -> Nx_backend.to_host t
    | Placed r -> read_elements r
    | Traced _ -> outside_trace ())

(* Moving *)

let move (type a b) p (x : (a, b) t) : (a, b) t =
  match x with
  | Traced _ -> outside_trace ()
  | Placed { r_cell = { state = Donated; _ }; _ } -> donated ()
  | Placed { r_placement; _ } when Placement.equal r_placement p -> x
  | Host _ when Placement.is_host p -> x
  | Placed r when Placement.is_host p -> Host (read_host r)
  | Host _ | Placed _ -> (
      match p with
      | Sharded { axis; devices } ->
          let shape = View.shape (view x) in
          let n = List.length devices in
          if axis >= Array.length shape || shape.(axis) mod n <> 0 then
            invalid_arg
              (Printf.sprintf
                 "Nx.place: axis %d of shape %s does not split evenly over %d \
                  devices"
                 axis (Shape.to_string shape) n);
          (Placement.engine p).place p x
      | Device _ | Replicated _ -> (Placement.engine p).place p x)

let place p x =
  try Effect.perform (E_place { placement = p; t_in = x })
  with Effect.Unhandled _ -> move p x

(* Routing

   Every fallback runs where its operands live. Operands all on the host run on
   the link-time engine. Placed operands must share one device, which host
   operands join: until devices compute, the operation reads the placed
   operands' windows, runs on the host engine and places its result there. A
   result of one element is held by nx instead, so reading it back moves
   nothing. The route is decided before anything is read, so operands on two
   device lists raise before any work.

   A value on several devices is read to the host by an operation, whose result
   is a host value, until operations over device lists exist. Its devices still
   count: it does not mix with a value on other devices. *)

type route =
  | On_host
  | At of placement (* one device: the result is placed there *)
  | Read_from of placement (* several devices: the result is a host value *)

let join : type a b. string -> route -> (a, b) t -> route =
 fun op r x ->
  match x with
  | Host _ -> r
  | Traced _ -> outside_trace ()
  | Placed { r_placement = p; _ } -> (
      match r with
      | On_host -> ( match p with Device _ -> At p | _ -> Read_from p)
      | At q | Read_from q ->
          if List.equal ( == ) (Placement.devices q) (Placement.devices p) then
            r
          else
            invalid_arg
              (Format.asprintf "Nx.%s: operands on %a and %a; place one of them"
                 op Placement.pp q Placement.pp p))

let route1 op a = join op On_host a
let route2 op a b = join op (join op On_host a) b
let route3 op a b c = join op (join op (join op On_host a) b) c

let settle : type a b. route -> (a, b) Nx_backend.t -> (a, b) t =
 fun r h ->
  match r with
  | On_host | Read_from _ -> Host h
  | At p ->
      let shape = View.shape (Nx_backend.view h) in
      if Array.fold_left ( * ) 1 shape = 1 then
        let v =
          Nx_buffer.get (Nx_backend.to_host h) (View.offset (Nx_backend.view h))
        in
        held p (Nx_backend.dtype h) v shape
      else (Placement.engine p).place p (Host h)

(* [routed1 op x f] runs [f] where [x] lives, [x] being no host tensor. *)
let routed1 op x f =
  let r = route1 op x in
  settle r (f (host_of x))

let unary_op op eff host_op t_in =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with Host t -> Host (host_op t) | _ -> routed1 op t_in host_op)

let binary_op op eff host_op a b =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (host_op a b)
    | _ ->
        let r = route2 op a b in
        settle r (host_op (host_of a) (host_of b)))

(* A movement of a placed value is view arithmetic over the same storage, the
   same on every replica. *)
let movement_op eff host_op view_op t_in arg =
  try Effect.perform (eff ())
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (host_op t arg)
    | Placed ({ r_placement = Device _ | Replicated _; _ } as r) ->
        Placed { r with r_id = fresh_id (); r_view = view_op r.r_view arg }
    | Placed ({ r_placement = Sharded _; _ } as r) ->
        Host (host_op (read_host r) arg)
    | Traced _ -> outside_trace ())

(* Binary operations *)

let add a b = binary_op "add" (fun () -> E_add { a; b }) Nx_backend.add a b
let sub a b = binary_op "sub" (fun () -> E_sub { a; b }) Nx_backend.sub a b
let mul a b = binary_op "mul" (fun () -> E_mul { a; b }) Nx_backend.mul a b
let max a b = binary_op "max" (fun () -> E_max { a; b }) Nx_backend.max a b
let min a b = binary_op "min" (fun () -> E_min { a; b }) Nx_backend.min a b
let mod_ a b = binary_op "mod" (fun () -> E_mod { a; b }) Nx_backend.mod_ a b
let pow a b = binary_op "pow" (fun () -> E_pow { a; b }) Nx_backend.pow a b
let xor a b = binary_op "xor" (fun () -> E_xor { a; b }) Nx_backend.xor a b
let or_ a b = binary_op "or" (fun () -> E_or { a; b }) Nx_backend.or_ a b
let and_ a b = binary_op "and" (fun () -> E_and { a; b }) Nx_backend.and_ a b

let atan2 a b =
  binary_op "atan2" (fun () -> E_atan2 { a; b }) Nx_backend.atan2 a b

let fdiv a b = binary_op "div" (fun () -> E_fdiv { a; b }) Nx_backend.fdiv a b
let idiv a b = binary_op "div" (fun () -> E_idiv { a; b }) Nx_backend.idiv a b

(* Comparison operations *)

(* [routed2 op a b f] runs [f] where [a] and [b] live, one being no host
   tensor. *)
let routed2 op a b f =
  let r = route2 op a b in
  settle r (f (host_of a) (host_of b))

let cmpeq a b =
  try Effect.perform (E_cmpeq { a; b })
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (Nx_backend.cmpeq a b)
    | _ -> routed2 "equal" a b Nx_backend.cmpeq)

let cmpne a b =
  try Effect.perform (E_cmpne { a; b })
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (Nx_backend.cmpne a b)
    | _ -> routed2 "not_equal" a b Nx_backend.cmpne)

let cmplt a b =
  try Effect.perform (E_cmplt { a; b })
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (Nx_backend.cmplt a b)
    | _ -> routed2 "less" a b Nx_backend.cmplt)

let cmple a b =
  try Effect.perform (E_cmple { a; b })
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (Nx_backend.cmple a b)
    | _ -> routed2 "less_equal" a b Nx_backend.cmple)

(* Unary operations *)

let neg t = unary_op "neg" (fun () -> E_neg { t_in = t }) Nx_backend.neg t
let sin t = unary_op "sin" (fun () -> E_sin { t_in = t }) Nx_backend.sin t
let sqrt t = unary_op "sqrt" (fun () -> E_sqrt { t_in = t }) Nx_backend.sqrt t

let recip t =
  unary_op "recip" (fun () -> E_recip { t_in = t }) Nx_backend.recip t

let log t = unary_op "log" (fun () -> E_log { t_in = t }) Nx_backend.log t
let exp t = unary_op "exp" (fun () -> E_exp { t_in = t }) Nx_backend.exp t
let cos t = unary_op "cos" (fun () -> E_cos { t_in = t }) Nx_backend.cos t
let abs t = unary_op "abs" (fun () -> E_abs { t_in = t }) Nx_backend.abs t
let sign t = unary_op "sign" (fun () -> E_sign { t_in = t }) Nx_backend.sign t
let tan t = unary_op "tan" (fun () -> E_tan { t_in = t }) Nx_backend.tan t
let asin t = unary_op "asin" (fun () -> E_asin { t_in = t }) Nx_backend.asin t
let acos t = unary_op "acos" (fun () -> E_acos { t_in = t }) Nx_backend.acos t
let atan t = unary_op "atan" (fun () -> E_atan { t_in = t }) Nx_backend.atan t
let sinh t = unary_op "sinh" (fun () -> E_sinh { t_in = t }) Nx_backend.sinh t
let cosh t = unary_op "cosh" (fun () -> E_cosh { t_in = t }) Nx_backend.cosh t
let tanh t = unary_op "tanh" (fun () -> E_tanh { t_in = t }) Nx_backend.tanh t

let trunc t =
  unary_op "trunc" (fun () -> E_trunc { t_in = t }) Nx_backend.trunc t

let ceil t = unary_op "ceil" (fun () -> E_ceil { t_in = t }) Nx_backend.ceil t

let floor t =
  unary_op "floor" (fun () -> E_floor { t_in = t }) Nx_backend.floor t

let round t =
  unary_op "round" (fun () -> E_round { t_in = t }) Nx_backend.round t

let erf t = unary_op "erf" (fun () -> E_erf { t_in = t }) Nx_backend.erf t

let op_psum t_in =
  try Effect.perform (E_psum { t_in })
  with Effect.Unhandled _ -> failwith "psum must be used under vmap"

(* Reduction operations. The host case of each operation below calls its backend
   directly, allocating nothing beyond the effect and the result. *)

let reduce ~op ~axes t_in =
  let eff =
    match op with
    | `Sum -> E_reduce_sum { t_in; axes }
    | `Prod -> E_reduce_prod { t_in; axes }
    | `Max -> E_reduce_max { t_in; axes }
    | `Min -> E_reduce_min { t_in; axes }
  in
  try Effect.perform eff
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.reduce ~op ~axes t)
    | _ -> routed1 "reduce" t_in (Nx_backend.reduce ~op ~axes))

let argmax ~axis ~keepdims t_in =
  try Effect.perform (E_argmax { t_in; axis; keepdims })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.argmax ~axis ~keepdims t)
    | _ -> routed1 "argmax" t_in (Nx_backend.argmax ~axis ~keepdims))

let argmin ~axis ~keepdims t_in =
  try Effect.perform (E_argmin { t_in; axis; keepdims })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.argmin ~axis ~keepdims t)
    | _ -> routed1 "argmin" t_in (Nx_backend.argmin ~axis ~keepdims))

let associative_scan ~axis ~op t_in =
  try Effect.perform (E_associative_scan { t_in; axis; op })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.associative_scan ~axis ~op t)
    | _ ->
        routed1 "associative_scan" t_in (Nx_backend.associative_scan ~axis ~op))

let sort ~axis ~descending t_in =
  try Effect.perform (E_sort { t_in; axis; descending })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.sort ~axis ~descending t)
    | _ -> routed1 "sort" t_in (Nx_backend.sort ~axis ~descending))

let argsort ~axis ~descending t_in =
  try Effect.perform (E_argsort { t_in; axis; descending })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.argsort ~axis ~descending t)
    | _ -> routed1 "argsort" t_in (Nx_backend.argsort ~axis ~descending))

(* Movement operations *)

let reshape t_in new_shape =
  movement_op
    (fun () -> E_reshape { t_in; new_shape })
    Nx_backend.reshape View.reshape t_in new_shape

let expand t_in new_target_shape =
  movement_op
    (fun () -> E_expand { t_in; new_target_shape })
    Nx_backend.expand View.expand t_in new_target_shape

let permute t_in axes =
  movement_op
    (fun () -> E_permute { t_in; axes })
    Nx_backend.permute View.permute t_in axes

let shrink t_in limits =
  movement_op
    (fun () -> E_shrink { t_in; limits })
    Nx_backend.shrink View.shrink t_in limits

let flip t_in dims_to_flip =
  movement_op
    (fun () -> E_flip { t_in; dims_to_flip })
    Nx_backend.flip View.flip t_in dims_to_flip

let sliding_window t_in ~axis ~window ~step =
  movement_op
    (fun () -> E_sliding_window { t_in; axis; window; step })
    (fun t () -> Nx_backend.sliding_window t ~axis ~window ~step)
    (fun v () -> View.sliding_window v ~axis ~window ~step)
    t_in ()

let pad t_in padding_config fill_value =
  try Effect.perform (E_pad { t_in; padding_config; fill_value })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.pad t padding_config fill_value)
    | _ ->
        routed1 "pad" t_in (fun t -> Nx_backend.pad t padding_config fill_value))

(* Creation operations. A value created in the context of one device lives
   there, and a scalar there is held by nx and allocates nothing. A value
   created in the context of several devices is a host value, as an operation
   over them gives. *)

let at_devices = function [ d ] -> At (Device d) | _ -> On_host

let buffer (ctx : context) dtype shape_arr =
  let size_in_elements = Array.fold_left ( * ) 1 shape_arr in
  let flat =
    try Effect.perform (E_buffer { context = ctx; dtype; size_in_elements })
    with Effect.Unhandled _ -> (
      match ctx with
      | Host c -> Host (Nx_backend.buffer c dtype shape_arr)
      | On ds ->
          settle (at_devices ds)
            (Nx_backend.buffer host_context dtype shape_arr))
  in
  reshape flat shape_arr

let const_scalar (ctx : context) value dtype =
  try Effect.perform (E_const_scalar { context = ctx; value; dtype })
  with Effect.Unhandled _ -> (
    match ctx with
    | Host c -> Host (Nx_backend.full c dtype [||] value)
    | On [ d ] -> held (Device d) dtype value [||]
    | On _ -> Host (Nx_backend.full host_context dtype [||] value))

let broadcast scalar shape_arr =
  if Array.length shape_arr = 0 then scalar
  else expand (reshape scalar (Array.map (fun _ -> 1) shape_arr)) shape_arr

let full (ctx : context) dtype shape_arr value =
  (* Under an effect handler (jit tracing), and in a device context, a filled
     tensor is a broadcast scalar constant: no bytes are materialized. On the
     host it stays a concrete, mutable backend tensor. *)
  match Effect.perform (E_const_scalar { context = ctx; value; dtype }) with
  | scalar -> broadcast scalar shape_arr
  | exception Effect.Unhandled _ -> (
      match ctx with
      | Host c -> Host (Nx_backend.full c dtype shape_arr value)
      | On [ d ] -> broadcast (held (Device d) dtype value [||]) shape_arr
      | On _ -> Host (Nx_backend.full host_context dtype shape_arr value))

let from_host (ctx : context) array =
  try Effect.perform (E_from_host { context = ctx; array })
  with Effect.Unhandled _ -> (
    match ctx with
    | Host c -> Host (Nx_backend.from_host c array)
    | On ds -> settle (at_devices ds) (Nx_backend.from_host host_context array))

(* Copy operations. A placed value whose view covers its storage is already
   contiguous. *)

let contiguous t_in =
  try Effect.perform (E_contiguous { t_in })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.contiguous t)
    | Placed r when covers r -> t_in
    | _ -> routed1 "contiguous" t_in Fun.id)

let copy t_in =
  try Effect.perform (E_copy { t_in })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.copy t)
    | _ -> routed1 "copy" t_in Nx_backend.copy)

(* Ternary operations *)

let where condition if_true if_false =
  try Effect.perform (E_where { condition; if_true; if_false })
  with Effect.Unhandled _ ->
    let r = route3 "where" condition if_true if_false in
    settle r
      (Nx_backend.where (host_of condition) (host_of if_true) (host_of if_false))

(* Cat *)

let cat t_list ~axis =
  try Effect.perform (E_cat { t_list; axis })
  with Effect.Unhandled _ ->
    if List.for_all (function Host _ -> true | _ -> false) t_list then
      Host (Nx_backend.cat (List.map host_of t_list) ~axis)
    else
      let r = List.fold_left (join "concatenate") On_host t_list in
      settle r (Nx_backend.cat (List.map host_of t_list) ~axis)

(* Cast *)

let cast (type a b c d) ~(dtype : (c, d) Dtype.t) (t_in : (a, b) t) : (c, d) t =
  let target_dtype = dtype in
  try Effect.perform (E_cast { t_in; target_dtype })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.cast ~dtype:target_dtype t)
    | _ -> routed1 "cast" t_in (Nx_backend.cast ~dtype:target_dtype))

(* Indexed access *)

let gather data indices ~axis =
  try Effect.perform (E_gather { data; indices; axis })
  with Effect.Unhandled _ ->
    let r = route2 "take" data indices in
    settle r (Nx_backend.gather (host_of data) (host_of indices) ~axis)

let update t_in ~starts v =
  try Effect.perform (E_update { t_in; starts; v })
  with Effect.Unhandled _ ->
    let r = route3 "set" t_in starts v in
    settle r
      (Nx_backend.update (host_of t_in) ~starts:(host_of starts) (host_of v))

let scatter ~mode ~unique_indices data_template ~indices ~updates ~axis =
  try
    Effect.perform
      (E_scatter { data_template; indices; updates; axis; mode; unique_indices })
  with Effect.Unhandled _ ->
    let r = route3 "scatter" data_template indices updates in
    settle r
      (Nx_backend.scatter ~mode ~unique_indices (host_of data_template)
         ~indices:(host_of indices) ~updates:(host_of updates) ~axis)

(* Random *)

let threefry key ctr =
  try Effect.perform (E_threefry { key; ctr })
  with Effect.Unhandled _ ->
    let r = route2 "threefry" key ctr in
    settle r (Nx_backend.threefry (host_of key) (host_of ctr))

(* The index of the current lane along the innermost mapped axis. The vmap/pmap
   handler answers with a per-lane (batched) or per-device index; with no
   handler there is a single lane, index 0. [Nx.Rng.fold_in_axis] folds it into
   a key to decorrelate lanes. *)
let axis_index ctx =
  try Effect.perform E_axis_index
  with Effect.Unhandled _ -> const_scalar ctx 0l Dtype.int32

(* Window operations *)

let unfold t_in ~kernel_size ~stride ~dilation ~padding =
  try Effect.perform (E_unfold { t_in; kernel_size; stride; dilation; padding })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t ->
        Host (Nx_backend.unfold t ~kernel_size ~stride ~dilation ~padding)
    | _ ->
        routed1 "unfold" t_in (fun t ->
            Nx_backend.unfold t ~kernel_size ~stride ~dilation ~padding))

let fold t_in ~output_size ~kernel_size ~stride ~dilation ~padding =
  try
    Effect.perform
      (E_fold { t_in; output_size; kernel_size; stride; dilation; padding })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t ->
        Host
          (Nx_backend.fold t ~output_size ~kernel_size ~stride ~dilation
             ~padding)
    | _ ->
        routed1 "fold" t_in (fun t ->
            Nx_backend.fold t ~output_size ~kernel_size ~stride ~dilation
              ~padding))

(* Matrix operations *)

let matmul a b =
  try Effect.perform (E_matmul { a; b })
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b -> Host (Nx_backend.matmul a b)
    | _ -> routed2 "matmul" a b Nx_backend.matmul)

(* FFT operations *)

let fft t ~axes =
  try Effect.perform (E_fft { t; axes })
  with Effect.Unhandled _ -> (
    match t with
    | Host h -> Host (Nx_backend.fft h ~axes)
    | _ -> routed1 "fft" t (Nx_backend.fft ~axes))

let ifft t ~axes =
  try Effect.perform (E_ifft { t; axes })
  with Effect.Unhandled _ -> (
    match t with
    | Host h -> Host (Nx_backend.ifft h ~axes)
    | _ -> routed1 "ifft" t (Nx_backend.ifft ~axes))

let rfft (type a c) (t : (float, a) t) ~(dtype : (Complex.t, c) Dtype.t) ~axes :
    (Complex.t, c) t =
  try Effect.perform (E_rfft { t; dtype; axes })
  with Effect.Unhandled _ -> (
    match t with
    | Host h -> Host (Nx_backend.rfft h ~dtype ~axes)
    | _ -> routed1 "rfft" t (Nx_backend.rfft ~dtype ~axes))

let irfft (type a c) ?s (t : (Complex.t, a) t) ~(dtype : (float, c) Dtype.t)
    ~axes : (float, c) t =
  try Effect.perform (E_irfft { t; dtype; axes; s })
  with Effect.Unhandled _ -> (
    match t with
    | Host h -> Host (Nx_backend.irfft ?s h ~dtype ~axes)
    | _ -> routed1 "irfft" t (Nx_backend.irfft ?s ~dtype ~axes))

(* Linear algebra *)

let cholesky ~upper t_in =
  try Effect.perform (E_cholesky { t_in; upper })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.cholesky ~upper t)
    | _ -> routed1 "cholesky" t_in (Nx_backend.cholesky ~upper))

let qr ~reduced t_in =
  try Effect.perform (E_qr { t_in; reduced })
  with Effect.Unhandled _ ->
    let r = route1 "qr" t_in in
    let q, rr = Nx_backend.qr ~reduced (host_of t_in) in
    (settle r q, settle r rr)

let svd ~full_matrices t_in =
  try Effect.perform (E_svd { t_in; full_matrices })
  with Effect.Unhandled _ ->
    let r = route1 "svd" t_in in
    let u, s, vt = Nx_backend.svd ~full_matrices (host_of t_in) in
    (settle r u, settle r s, settle r vt)

let eigvals t_in =
  try Effect.perform (E_eigvals { t_in })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.eigvals t)
    | _ -> routed1 "eigvals" t_in Nx_backend.eigvals)

let eig t_in =
  try Effect.perform (E_eig { t_in })
  with Effect.Unhandled _ ->
    let r = route1 "eig" t_in in
    let vals, vecs = Nx_backend.eig (host_of t_in) in
    (settle r vals, settle r vecs)

let eigvalsh t_in =
  try Effect.perform (E_eigvalsh { t_in })
  with Effect.Unhandled _ -> (
    match t_in with
    | Host t -> Host (Nx_backend.eigvalsh t)
    | _ -> routed1 "eigvalsh" t_in Nx_backend.eigvalsh)

let eigh t_in =
  try Effect.perform (E_eigh { t_in })
  with Effect.Unhandled _ ->
    let r = route1 "eigh" t_in in
    let vals, vecs = Nx_backend.eigh (host_of t_in) in
    (settle r vals, settle r vecs)

let solve_triangular ~upper ~transpose ~unit_diag a b =
  try Effect.perform (E_solve_triangular { a; b; upper; transpose; unit_diag })
  with Effect.Unhandled _ -> (
    match (a, b) with
    | Host a, Host b ->
        Host (Nx_backend.solve_triangular ~upper ~transpose ~unit_diag a b)
    | _ ->
        let r = route2 "solve_triangular" a b in
        settle r
          (Nx_backend.solve_triangular ~upper ~transpose ~unit_diag (host_of a)
             (host_of b)))
