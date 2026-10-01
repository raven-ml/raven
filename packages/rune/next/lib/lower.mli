(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Lowering nx operations to tolk's intermediate representation.

    A trace ({!scope}) answers each nx operation it is given ({!op}) with a
    traced value: a value with a shape, a dtype and a placement but no bytes,
    whose node is the UOp graph that computes it. The graph is a tensor graph of
    tolk: movements, arithmetic and storage, over the whole value however its
    placement splits it.

    An operand that is not traced is a {e capture}: a value the traced function
    closes over. A trace binds each capture once, the first time it meets it,
    and the program holds what it binds ({!captures}).

    {b Numerics.} The graph computes what nx documents for the operation, the
    result that nx.cpu gives for the same operands, to the operation's class:
    exactly, or within a bound its tests state. *)

open Tolk_next

exception Jit_error of string
(** [Jit_error why] is raised when a traced function cannot be compiled. [why]
    names the operation and the reason. *)

type (_, _) Nx.Repr.node +=
  | Uop : Ops.t -> ('a, 'b) Nx.Repr.node
        (** [Uop u] is the payload of a traced value whose elements [u]
            computes. *)

(** {1:dtypes Dtypes} *)

val dtype : ('a, 'b) Nx_dtype.t -> Dtype.t option
(** [dtype dt] is the data type of [dt]'s elements in a graph: floats, integers
    and booleans one to one, and [float8_e4m3] and [float8_e5m2] as
    {!Dtype.Fp8e4m3} and {!Dtype.Fp8e5m2}. It is [None] for [int4], [uint4],
    [complex64] and [complex128], which no graph holds. *)

val const : ('a, 'b) Nx_dtype.t -> 'a -> Dtype.const
(** [const dt v] is the element [v] of [dt] as a constant.

    Raises [Invalid_argument] on a complex element. *)

(** {1:storage Views over storage} *)

val span : Dtype.t -> Nx_array.View.t -> int * int
(** [span dt v] is the run of elements of [dt] that the non-empty view [v]
    reaches, from the element at or below the first one it reaches whose offset
    is a multiple of 16 bytes, through the last one: [(start, length)]. *)

val within : Dtype.t -> Nx_array.View.t -> Nx_array.View.t
(** [within dt v] is the non-empty view [v] over the run {!span} gives: of [v]'s
    shape and strides, offset from the run's start, with the stride of each axis
    of one element [0], which no read steps along. Views that differ only in
    those strides, or in where their runs start, give equal views. *)

val run : Dtype.t -> Nx_array.View.t -> Nx_device.Buffer.t -> Nx_device.Buffer.t
(** [run dt v b] is the run of [b]'s elements of [dt] that {!span} gives for the
    view [v]: the buffer a parameter of a value of view [v] over [b] binds. *)

val storage : ('a, 'b) Nx.t -> Nx_device.Buffer.t list * Nx_array.View.t
(** [storage x] is the buffer of [x]'s storage on each device of [x]'s
    placement, in order, and [x]'s view of it.

    Raises [Invalid_argument] if [x] is traced, a value used outside the trace
    that made it, and as {!Nx.Repr.Storage.buffers} does. *)

val phase : Dtype.t -> Nx_device.Buffer.t -> int -> int
(** [phase dt b start] is the bytes by which element [start] of [dt] in [b] lies
    past a 16-byte boundary of [b]'s memory: the phase of storage that starts
    there ({!Tolk_next.Ops.param_arg}). It is [0] on the disk, whose files are
    read at any byte.

    Raises [Invalid_argument] if [b] is dead. *)

val broadcast : Ops.t -> int array -> Ops.t
(** [broadcast c shape] is the scalar node [c] at every position of [shape]. *)

val strided : Ops.t -> Nx_array.View.t -> int -> Ops.t
(** [strided flat v start] is the view [v] over the node [flat] of its storage's
    elements from element [start] on: movements that reach the elements [v]
    reaches, in [v]'s shape. *)

(** {1:traces Traces} *)

type scope
(** The type for traces in progress: the renderer of each device, the device of
    every name its nodes carry, and the captures it has bound. *)

val scope : renderer:(Nx.Device.t -> Renderer.t) -> scope
(** [scope ~renderer] is an empty trace in which the programs of a device [d]
    are rendered by [renderer d], which decides the dtypes [d] computes
    ({!Decomp_dtype.computes}). [renderer] is called once per device of the
    trace. *)

val op : scope -> 'r Nx.Op.t -> 'r
(** [op s o] is [o] in the trace [s]. Each result is a traced value at the
    placement nx gives [o]'s result, whose node computes it. An operand at
    another placement is copied to the result's devices, as nx places a host
    operand of an operation on a device, on every call; a traced operand that
    reads only captures is instead computed on those devices, its captures
    copied there once, so its values carry their arithmetic and may differ from
    eager's in the last bits.

    An operand that is not traced is a capture, bound the first time [s] meets
    its storage and view at a placement:
    - if it reaches a single element, it is that element: a constant read from
      its storage when the host owns that storage, read once from its device
      otherwise;
    - otherwise it is storage [s] holds ({!captures}): the value itself if it
      lies where the operation computes, and a copy placed there once otherwise.

    [Read] of a value that is not traced is answered in the enclosing
    interpretation.

    Raises [Jit_error] for [Read] of a traced value, with a message that starts
    with the name of the function that reads ([Nx.Op.Read]'s [by]), for an
    operation or a dtype that a device of the result cannot compute, and for a
    placement over a grid that is not one axis of devices. Raises
    [Invalid_argument] where nx does for operands that cannot meet, and for two
    devices of one name. *)

val param : scope -> slot:int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [param s ~slot x] is the traced value that stands for [x] in [s]: at [x]'s
    placement, of its dtype and shape, whose node views the storage of slot
    [slot] ({!Tolk_next.Ops.new_buffer}) as [x]'s view views its storage. The
    storage is the run of [x]'s storage that {!captures} describes for a
    capture; a program binds it to that run of each argument it is called with.
    Its phase is where that run of [x]'s storage starts within 16 bytes
    ({!Tolk_next.Ops.param_arg}), and the program accesses memory as aligned to
    it: an argument whose run starts elsewhere within 16 bytes needs a program
    traced from it.

    Raises [Jit_error] if a device of [x]'s placement cannot compute its dtype
    or [x]'s buffers start at different places within 16 bytes, and
    [Invalid_argument] if [x] is traced. *)

val output :
  scope ->
  slot:int ->
  Nx.Placement.t ->
  ('a, 'b) Nx_dtype.t ->
  int array ->
  Ops.t
(** [output s ~slot p dt shape] is the node that views the storage of slot
    [slot] as a value of [dt] and [shape] at [p], in C order: each device's
    window of the value, starting on 16 bytes. A program stores a result into
    it.

    Raises as {!param} does. *)

val parameter :
  scope ->
  slot:int ->
  Nx.Placement.t ->
  ('a, 'b) Nx_dtype.t ->
  int array ->
  ('a, 'b) Nx.t
(** [parameter s ~slot p dt shape] is the traced value of [dt] and [shape] at
    [p] that the parameter [slot] of a called body ({!Tolk_next.Ops.call}) holds
    in C order: each device's window, starting on 16 bytes.

    Raises as {!param} does. *)

val engine : scope -> string -> Tolk_next_engine.device
(** [engine s] is the engine's device of each name [s]'s nodes carry, and of the
    host of each, which submits its work ({!Tolk_next_engine.device}). *)

val value : scope -> ('a, 'b) Nx.t -> Ops.t
(** [value s x] is the node of [x] in [s] at [x]'s placement: its own if [x] is
    traced, and its capture ({!op}) otherwise. *)

val traced : Nx.Placement.t -> ('a, 'b) Nx_dtype.t -> Ops.t -> ('a, 'b) Nx.t
(** [traced p dt u] is the traced value at [p] of [dt] whose elements [u]
    computes, of [u]'s shape. *)

val uop : ('a, 'b) Nx.t -> Ops.t
(** [uop x] is the node of the traced value [x].

    Raises [Invalid_argument] if [x] is not a value traced by a scope. *)

val devices : scope -> (string * Nx.Device.t) list
(** [devices s] is the device of each name that [s]'s nodes carry, in the order
    [s] met them. *)

val captures : scope -> (Ops.t * Nx_device.Buffer.t list) list
(** [captures s] is the storage node of each capture [s] holds and the buffers
    it binds, one per device of the node, in the order [s] met them. A node
    holds the run of its capture's storage from the element at or below the
    first element its view reaches whose offset is a multiple of 16 bytes,
    through the last one it reaches; its phase is where that run starts within
    16 bytes ({!Tolk_next.Ops.param_arg}). *)

val held : scope -> Nx.Repr.Storage.t list
(** [held s] is the placed storage that [s]'s captures bind: what a program of
    [s] pins ({!Nx.Repr.Storage.pin}). Host storage records no binding. *)

type write = {
  result : Ops.t;  (** The node of the write's result. *)
  into : Ops.t;  (** The node of the value it writes into. *)
  regions : Lower_index.region list;
      (** The regions it writes, when it has a region form: storing them into
          the storage of [into] makes it hold [result]. Empty otherwise. *)
}
(** The type for indexed writes ([Nx.Op.Update], [Nx.Op.Scatter]). *)

val writes : scope -> write list
(** [writes s] is each indexed write that [s] lowered, the last first. *)
