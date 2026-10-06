(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Storage, sharding, variables and calls.

    The nodes a schedule passes to calls and the calls themselves: storage of a
    shape ({!param}, {!alloc}, {!placeholder}), values sharded over devices
    ({!shard}, {!axis}), variables bound to values ({!bind}), calls that write
    new storage ({!call_with_outputs}) and the launch information of a
    program ({!program_info_of_sink}). They read shapes ({!Shape}).

    {b Errors.} A broken precondition raises [Invalid_argument]. *)

(** {1:multi Several devices} *)

val axis : Ops.t -> int option
(** [axis u] is the axis [u] is sharded on, if it is sharded on one.

    Raises [Invalid_argument] if [u] is sharded on several axes, or if a reshape
    moves elements between shards. *)

val bounds : Ops.t -> (Ops.sint * Ops.sint) list
(** [bounds u] is the start and end of each shard along [u]'s axis.

    Raises [Invalid_argument] if [u] is not sharded. *)

val shard_shape : Ops.t -> Ops.sint list
(** [shard_shape u] is the shape of one of [u]'s shards. *)

val max_shard_shape : Ops.t -> int list
(** [max_shard_shape u] is [to_max_shape (shard_shape u)]. *)

val unshard : ?ranges:Ops.t list -> Ops.t -> int list -> Ops.t
(** [unshard ~ranges u axes] reassembles [u], sharded on [axes] over [ranges]
    (default one {!Ops.Axis_type.Device} range over [u]'s devices).

    Raises [Invalid_argument] if the lengths differ or an axis repeats. *)

val shard_slice : Ops.t -> int -> Ops.t -> Ops.t
(** [shard_slice u axis r] is the part of [u] along [axis] that the range [r]
    selects: the [n] elements of [axis] from [r * n], where [n] is the size of
    [axis] divided by [r]'s count. It is [u] if [u] is a scalar.

    Raises [Invalid_argument] if [r]'s count does not divide the axis. *)

val shard : ?axis:int -> Ops.t -> string list -> Ops.t
(** [shard ~axis u devices] is [u] copied to [devices], split along [axis] if
    given. *)

val store_call : Ops.t -> Ops.t -> Ops.t
(** [store_call dst src] is the call that copies [src] into [dst]. *)

(** {1:storage Storage}

    Sizes of storage are [int]s: a shape of more than [max_int] elements raises
    [Invalid_argument]. *)

val empty : ?device:Ops.device -> Ops.sint list -> Dtype.t -> Ops.t
(** [empty ~device shape dt] is uninitialised storage of [shape] for whoever
    realizes the graph to bind.

    Raises [Invalid_argument] if [dt] is weak. *)

val empty_like : ?dtype:Dtype.t -> ?device:Ops.device -> Ops.t -> Ops.t
(** [empty_like u] is uninitialised storage of [u]'s shape, type and placement,
    sharded as [u]. *)

val clone : ?device:Ops.device -> Ops.t -> Ops.t
(** [clone ~device u] is a copy of [u] into new storage on [device] (default
    [u]'s).

    Raises [Invalid_argument] if [device] is a disk. *)

val alloc :
  ?slot:int ->
  ?addrspace:Dtype.addr_space ->
  ?device:Ops.device ->
  ?axis:int ->
  Ops.sint list ->
  Dtype.t ->
  Ops.t
(** [alloc shape dt] is call-local storage of [shape], sharded on [axis]. *)

val alloc_like : ?slot:int -> ?addrspace:Dtype.addr_space -> Ops.t -> Ops.t
(** [alloc_like u] is [alloc] of [u]'s shard shape and type. *)

val placeholder :
  ?slot:int ->
  ?addrspace:Dtype.addr_space ->
  ?device:Ops.device ->
  ?volatile:bool ->
  ?tag:Ops.Tag.t ->
  int list ->
  Dtype.t ->
  Ops.t
(** [placeholder shape dt] is a parameter of [shape] and [dt] ([dt] committed,
    {!Dtype.strong}), or workgroup or register storage for those address spaces.
    A [String] tag also names it.

    Raises [Invalid_argument] if local storage gets a device. *)

val placeholder_like : ?addrspace:Dtype.addr_space -> Ops.t -> int -> Ops.t
(** [placeholder_like u slot] is a placeholder of [u]'s shard shape and type. *)

val param :
  ?shape:Ops.sint list ->
  ?device:Ops.device ->
  ?vmin_vmax:Dtype.value * Dtype.value ->
  ?multiple_of:int ->
  ?name:string ->
  ?addrspace:Dtype.addr_space option ->
  ?volatile:bool ->
  ?phase:int ->
  ?align:int ->
  int ->
  Dtype.t ->
  Ops.t
(** [param ~shape slot dt] is the parameter [slot] of type [dt]: a scalar
    without [shape], flat storage of its greatest size viewed as [shape]
    otherwise, whose first element lies [phase] bytes past a multiple of [align]
    (defaults [0] and [16]; see {!Ops.param_arg}).

    Raises [Invalid_argument] if [dt] is weak. *)

val param_like : Ops.t -> int -> Ops.t
(** [param_like u slot] is a parameter in [slot] that [u] can be passed to: a
    scalar variable without its name and value, one shard of a sharded value, or
    storage of [u]'s shape. Storage has the phase and alignment of the storage
    [u] views ({!storage_phase}).

    Raises [Invalid_argument] if the phase does not fit [u]'s type. *)

val storage_phase : Ops.t -> int * int
(** [storage_phase u] is the alignment and phase [(align, phase)]
    ({!Ops.param_arg}) of the storage [u] views: its storage's, moved by the
    bytes a shrink skips when it shrinks the storage seen whole, a buffer or a
    stage the schedule allocates, in row-major order. A shrink by a symbolic
    start known only to multiples of fewer bytes than the storage's alignment,
    such as a window that moves with a range, lowers the alignment to the
    largest power of two those bytes are a multiple of, and a symbolic start
    into a view that reorders or pads the storage keeps only the element's size.
    Any other view keeps its storage's: a reordering, a bitcast, an ordering or
    a shard selection. A stage of a view of a buffer through movements and
    bitcasts is the view when scheduling finds it contiguous, and storage of its
    own on a 16-byte boundary otherwise, so it has what both hold: phase [0] and
    the largest power of two up to the buffer's alignment that the view's first
    byte is a multiple of; [(1, 0)] when a size is symbolic or the buffer
    sharded. A view whose first or last element is padding, or whose last
    element does not lie as many elements past its first as a run of its size,
    is no run, and its stage has [(16, 0)]. Storage on a disk, which no vector
    access reads, and anything else that is not storage or a view of it have
    [(16, 0)]. *)

val view_as : ?axis:int -> Ops.t -> Ops.sint list -> Ops.t
(** [view_as ~axis u shape] views the flat storage [u] as [shape], sharded on
    [axis]. *)

(** {1:variables Variables} *)

val is_bound_var : Ops.t -> bool
(** [is_bound_var u] is [true] iff [u] is a variable bound to a value. *)

val bind : Ops.t -> Dtype.value -> Ops.t
(** [bind v x] is [v] bound to [x].

    Raises [Invalid_argument] if [v] is not an unbound variable, or [x] is out
    of its range or not a multiple of its [multiple_of]. *)

val unbound : Ops.t -> Ops.t
(** [unbound v] is [v] without its value and tag. *)

val unbind : Ops.t -> Ops.t * Dtype.value
(** [unbind v] is [(unbound v, x)] for [v] bound to [x].

    Raises [Invalid_argument] if [v] is not bound. *)

val unbind_all : Ops.t -> Ops.t * (Ops.t * Dtype.value) list
(** [unbind_all u] is [u] with each bound variable unbound, and each variable
    with its value. *)

val variables : Ops.t -> Ops.t list
(** [variables u] is the unbound scalar variables [u] reads, with a
    ["_device_num"] variable for each device range, sorted by name and slot. *)

(** {1:calls Calls} *)

val call_with_outputs :
  ?name:string ->
  ?precompile:bool ->
  ?aux:Ops.hcq_info ->
  ?output_pos:int list ->
  Ops.t list ->
  Ops.t list ->
  Ops.t list
(** [call_with_outputs values args] calls a body that computes [values] from
    [args], and is the outputs, each ordered after the call. Each output is new
    storage passed to the call at its position in [output_pos] (default after
    [args]); [args] fill the other positions in order.

    Raises [Invalid_argument] if [output_pos] is not strictly ascending within
    the argument list. *)

val call_with_output :
  ?name:string -> ?precompile:bool -> Ops.t -> Ops.t list -> Ops.t
(** [call_with_output value args] is the one output of
    [call_with_outputs [value] args]. *)

val custom_kernel : Ops.t list -> (Ops.t list -> Ops.t) -> Ops.t list
(** [custom_kernel args f] calls the kernel [f] builds on placeholders of
    [args], and is [args], each ordered after it. *)

(** {1:programs Programs} *)

val program_info_of_sink : ?target:Helpers.Target.t -> Ops.t -> Ops.program_info
(** [program_info_of_sink ~target sink] reads the program's launch dimensions
    from its {!Op.Special} nodes, its variables and buffers from its {!Op.Param}
    nodes, and which buffers it reads and writes from its loads and stores. When
    none is found reading or writing, every buffer is taken to do both. [target]
    defaults to the empty target. *)

val launch_dims : Ops.program_info -> (string * int) list -> int list * int list
(** [launch_dims p vars] is [p]'s global and local sizes with each variable
    named in [vars] replaced by its value ({!Shape.sym_infer}). *)
