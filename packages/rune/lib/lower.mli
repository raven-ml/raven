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

open Tolk

exception Jit_error of string
(** [Jit_error why] is raised when a traced function cannot be compiled. [why]
    names the operation and the reason. *)

(** {1:dtypes Dtypes} *)

val dtype : ('a, 'b) Nx_dtype.t -> Dtype.t option
(** [dtype dt] is the data type of [dt]'s elements in a graph: floats, integers
    and booleans one to one, [float8_e4m3] and [float8_e5m2] as {!Dtype.Fp8e4m3}
    and {!Dtype.Fp8e5m2}, and the packed dtypes a byte each: [bit] as
    {!Dtype.Bool}, [int4] as {!Dtype.Int8} and [uint4] as {!Dtype.Uint8},
    holding their representatives. It is [None] for [complex64] and
    [complex128], which no graph holds.

    The storage of a packed dtype is its bytes as {!Dtype.Uint8}: a graph
    unpacks the elements where it reads storage that nx binds, an argument or a
    capture, and packs a result where it stores it ({!output}). *)

val const : ('a, 'b) Nx_dtype.t -> 'a -> Dtype.const
(** [const dt v] is the element [v] of [dt] as a constant.

    Raises [Invalid_argument] on a complex element. *)

(** {1:storage Views over storage} *)

val span : ('a, 'b) Nx_dtype.t -> Nx_array.View.t -> int * int
(** [span dt v] is the run of elements of [dt] that the non-empty view [v]
    reaches, from the element at or below the first one it reaches whose bits
    start at a multiple of 16 bytes, through the last one: [(start, length)]. *)

val within : ('a, 'b) Nx_dtype.t -> Nx_array.View.t -> Nx_array.View.t
(** [within dt v] is the non-empty view [v] over the run {!span} gives: of [v]'s
    shape and strides, offset from the run's start, with the stride of each axis
    of one element [0], which no read steps along. Views that differ only in
    those strides, or in where their runs start, give equal views. *)

val run :
  ('a, 'b) Nx_dtype.t ->
  Nx_array.View.t ->
  Nx_device.Buffer.t ->
  Nx_device.Buffer.t
(** [run dt v b] is the run of [b]'s elements of [dt] that {!span} gives for the
    view [v]: the buffer a parameter of a value of view [v] over [b] binds. *)

val phase : ('a, 'b) Nx_dtype.t -> Nx_device.Buffer.t -> int -> int
(** [phase dt b start] is the bytes by which the byte holding the first bit of
    element [start] of [dt] in [b] lies past a 16-byte boundary of [b]'s memory:
    the phase of storage that starts there ({!Tolk.Ops.param_arg}). [start] is a
    run's start ({!span}), whose bits start a byte. It is [0] on the disk, whose
    files are read at any byte.

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

val scope : renderer:(Nx_device.t -> Renderer.t) -> scope
(** [scope ~renderer] is an empty trace in which the programs of a device [d]
    are rendered by [renderer d], which decides the dtypes [d] computes
    ({!Decomp_dtype.computes}). [renderer] is called once per device of the
    trace. The trace is live until {!finish}. *)

val finish : scope -> unit
(** [finish s] ends the trace [s]: a value it traced that another trace meets
    has escaped it ({!uop}). *)

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
      lies where the operation computes; a copy of its storage on the
      operation's one device once otherwise, a value on the disk read through
      the host; and, for a placement over several devices, the value where it
      lies, which the program moves as it moves a traced operand.

    The lowering reads and places a capture through its storage, never through
    an operation of nx: it does so only when it traces, which its caller's cache
    decides, so no interpretation of operations sees it.

    [Read] of a value that is not traced, and [Check] of operands none of which
    is, are answered in the enclosing interpretation. [Check] of a traced
    operand reads nothing: [s] records it ({!checks}).

    Raises [Jit_error] for [Read] of a traced value, with a message that starts
    with the name of the function that reads ([Nx.Op.Read]'s [by]), for an
    operation or a dtype that a device of the result cannot compute, and for a
    placement over a grid that is not one axis of devices. Raises
    [Invalid_argument] where nx does for operands that cannot meet, and for two
    devices of one name. *)

val param : scope -> slot:int -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t
(** [param s ~slot x] is the traced value that stands for [x] in [s]: at [x]'s
    placement, of its dtype and shape, whose node views the storage of slot
    [slot] ({!Tolk.Ops.new_buffer}) as [x]'s view views its storage. The storage
    is the run of [x]'s storage that {!captures} describes for a capture; a
    program binds it to that run of each argument it is called with. Its phase
    is where that run of [x]'s storage starts within 16 bytes
    ({!Tolk.Ops.param_arg}), and the program accesses memory as aligned to it:
    an argument whose run starts elsewhere within 16 bytes needs a program
    traced from it.

    Raises [Jit_error] if a device of [x]'s placement cannot compute its dtype
    or [x]'s buffers start at different places within 16 bytes, and
    [Invalid_argument] if [x] is traced. *)

type target
(** The type for the storage a program stores a result into. *)

val output :
  scope ->
  slot:int ->
  Nx.Placement.t ->
  ('a, 'b) Nx_dtype.t ->
  int array ->
  target
(** [output s ~slot p dt shape] is the storage of slot [slot] for a value of
    [dt] and [shape] at [p], in C order: each device's window of the value,
    starting on 16 bytes.

    Raises as {!param} does. *)

val view : target -> Ops.t
(** [view t] is the node that views [t]'s storage as its value. *)

val store : target -> Ops.t -> Ops.t
(** [store t u] stores the value [u], of [t]'s dtype and shape, into [t]'s
    storage: packed, for a packed dtype, each device's window by that device. *)

val stored : target -> Ops.t list -> Ops.t
(** [stored t stores] is {!view}[ t] once [stores], stores into [t]'s storage,
    have run. *)

val scratch :
  scope -> Nx.Placement.t -> ('a, 'b) Nx_dtype.t -> int array -> target
(** [scratch s p dt shape] is storage that the program alone holds for a value
    of [dt] and [shape] at [p], in C order: a byte for each element of a packed
    dtype, as a graph computes it.

    Raises as {!param} does. *)

(** {2:rows Rows of a loop's input} *)

val stacked : scope -> ('a, 'b) Nx_dtype.t -> int array -> Ops.t -> Ops.t
(** [stacked s dt row u] is the storage a loop reads the rows of shape [row] of
    [u], a node of [s] of a value of [dt], from: [u] itself, or for a packed
    [dt] whose rows are whole bytes, [u]'s bytes in C order, those of its
    storage where [u] views it C-contiguously from a byte and packed otherwise.
*)

val row :
  scope ->
  slot:int ->
  Nx.Placement.t ->
  ('a, 'b) Nx_dtype.t ->
  int array ->
  ('a, 'b) Nx.t
(** [row s ~slot p dt shape] is {!parameter} for a row of a loop's input, which
    holds the storage {!stacked} gives for one row: unpacked by the body for a
    packed [dt] whose rows are whole bytes. *)

val parameter :
  scope ->
  slot:int ->
  Nx.Placement.t ->
  ('a, 'b) Nx_dtype.t ->
  int array ->
  ('a, 'b) Nx.t
(** [parameter s ~slot p dt shape] is the traced value of [dt] and [shape] at
    [p] that the parameter [slot] of a called body ({!Tolk.Ops.call}) holds in C
    order: each device's window, starting on 16 bytes.

    Raises as {!param} does. *)

val scalar :
  scope -> slot:int -> Nx.Placement.t -> ('a, 'b) Nx_dtype.t -> ('a, 'b) Nx.t
(** [scalar s ~slot p dt] is the traced scalar of [dt] at [p] that the scalar
    parameter [slot] of a called body ({!Tolk.Ops.call}) holds: a value its call
    passes, such as an expression of the loop around the call.

    Raises as {!param} does. *)

val argument :
  scope ->
  slot:int ->
  Nx.Placement.t ->
  ('a, 'b) Nx_dtype.t ->
  int array ->
  ('a, 'b) Nx.t
(** [argument s ~slot p dt shape] is a traced value that stands for an argument
    of [dt] and [shape] at [p] in C order over the storage of slot [slot], which
    [s] counts among its arguments: a value with no element is zeros.

    Raises as {!param} does. *)

val engine : scope -> string -> Tolk_engine.device
(** [engine s] is the engine's device of each name [s]'s nodes carry, and of the
    host of each, which submits its work ({!Tolk_engine.device}). *)

val value : scope -> ('a, 'b) Nx.t -> Ops.t
(** [value s x] is the node of [x] in [s] at [x]'s placement: its own if [x] is
    traced, and its capture ({!op}) otherwise. *)

val traced :
  scope -> Nx.Placement.t -> ('a, 'b) Nx_dtype.t -> Ops.t -> ('a, 'b) Nx.t
(** [traced s p dt u] is the value of [s] at [p] of [dt] whose elements [u]
    computes, of [u]'s shape. *)

val traces : scope -> ('a, 'b) Nx.t -> bool
(** [traces s x] is [true] iff [x] is a value of [s]. *)

val is_traced : ('a, 'b) Nx.t -> bool
(** [is_traced x] is [true] iff [x] is a value of a trace. *)

val is_constant : ('a, 'b) Nx.t -> bool
(** [is_constant x] is [true] iff [x] is a value of a trace whose node reads no
    storage: one every trace reads as it is, finished or not ({!uop}). *)

val uop : scope -> ('a, 'b) Nx.t -> Ops.t
(** [uop s x] is the node in [s] of the traced value [x], its own: [x] is a
    value of [s], of another trace that is live, an input of [s], or a constant
    ({!is_constant}).

    Raises [Invalid_argument] if [x] is a value of a trace that is finished,
    naming [Rune.jit] and saying a traced value escaped the function that traced
    it; if [x] is a value another transformation tracks, naming [Rune.jit] and
    saying the function reads it through its closure; and if [x] is not traced.
*)

val devices : scope -> (string * Nx_device.t) list
(** [devices s] is the device of each name that [s]'s nodes carry, in the order
    [s] met them. *)

val captures : scope -> (Ops.t * Nx_device.Buffer.t list) list
(** [captures s] is the storage node of each capture [s] holds and the buffers
    it binds, one per device of the node, in the order [s] met them. A node
    holds the run of its capture's storage from the element at or below the
    first element its view reaches whose offset is a multiple of 16 bytes,
    through the last one it reaches; its phase is where that run starts within
    16 bytes ({!Tolk.Ops.param_arg}). *)

type write = {
  result : Ops.t;  (** The node of the write's result. *)
  into : Ops.t;  (** The node of the value it writes into. *)
  regions : Lower_index.region list;
      (** The regions it writes, when it has a region form: storing them into
          the storage of [into] makes it hold [result]. Empty otherwise. A
          packed dtype's regions are runs of bytes, each written by one store,
          and a packed value whose bytes [s] does not know has none. *)
}
(** The type for indexed writes ([Nx.Op.Update], [Nx.Op.Scatter]). *)

val writes : scope -> write list
(** [writes s] is each indexed write that [s] lowered, the last first. *)

val regions : target -> write -> (Ops.t * Ops.t) list
(** [regions t w] is each region of [w] as a destination in [t] and its value:
    storing each value into its destination leaves [t]'s storage holding
    [w.result], for [t] the storage of [w.into]. *)

type check = {
  first : (int64, Nx_dtype.int64_elt) Nx.t;
      (** The traced scalar index, in C order, of the first false element of the
          checked value, or its element count where every one holds. *)
  data : Nx.packed list;
      (** The traced scalar element of each of the check's data at [first], a
          zero of its dtype where every element holds. *)
  shape : int array;  (** The checked value's shape. *)
  fail : int array -> Nx.packed list -> exn;
      (** The exception of a failure at an index, from [data]'s elements. *)
}
(** The type for checks ([Nx.Op.Check]) of traced values, which a program
    answers when it has run. *)

val checks : scope -> check list
(** [checks s] is each check of a non-empty traced value that [s] lowered, in
    the order it lowered them. *)

val checking : scope -> (unit -> 'a) -> 'a * check list
(** [checking s f] is [f ()] and the checks it lowered in [s], which {!checks}
    then leaves out. *)
