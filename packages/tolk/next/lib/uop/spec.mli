(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Specifications: the nodes each stage of compilation may hold.

    A specification is a matcher from nodes to verdicts. A node passes when the
    first rule that decides it gives [true]: a rule decides with [Some b] and
    declines with [None], and a node that no rule decides fails. A specification
    judges one node at a time, from its operation, type, argument and sources;
    {!type_verify} judges a graph.

    {!shared} holds at every stage. {!tensor} and {!program} add the operations
    of tensor graphs and of programs, and {!hcq} those of command-queue
    programs. {!full} accepts the nodes of every stage and of the forms passes
    produce in between; linking this module makes it the check that {!Ops.v}
    runs on the nodes it builds when {!Helpers.spec} is 2 or more.
    {!kernel_graph} is the graph of kernel calls that scheduling produces. *)

type t = (unit, bool) Ops.Pattern_matcher.t
(** The type for specifications. *)

(** {1:specs Specifications} *)

val shared : t
(** [shared] accepts the operations of every stage:
    - sinks, and constants and stacks of constants;
    - arithmetic whose operands have its type: a {!Op.Where}'s condition is
      boolean and comparisons compare operands of one type; a shift's count may
      also be [uint32]; bitwise operations take no floats, and divisions and
      remainders only integers;
    - casts, ranges, indexes with integer indexes, ends of bounded loops, and
      loop back-edges on a scalar boolean condition;
    - parameters, local and register buffers, binaries, groups of effects,
      orderings of storage, custom code, calls of opaque bodies, barriers and
      instructions;
    - loads and stores through an index or a shrink, a gated load's alternative
      being of the load's type, and stores into storage;
    - matrix multiply-accumulates.

    A weak type ({!Dtype.weaks}) and {!Ops.invalid} match any type.

    When {!Helpers.check_oob} holds, a load or store through an index into
    storage must be proved in bounds: the bounds of the index ({!Ops.vmin},
    {!Ops.vmax}) must lie within the storage's {!Ops.max_numel}. An index they
    do not prove fails, whatever its gate, and its bounds, the index and the
    gate are printed on standard error, saying that the bound cannot be proven
    without a solver. *)

val tensor : t
(** [tensor] is {!shared} with the operations of tensor graphs: float-only unary
    math; global buffers with a size and a device, and storage declared without
    a buffer ({!Op.Alloc}); variables without a device; unlowered
    {!Op.Special}s; movement; reductions; copies to a device other than a disk
    and all-reductions; sharding ({!Op.Unshard}, {!Op.Mselect}, {!Op.Mstack});
    {!Op.Detach}, {!Op.Contiguous_backward} and {!Op.Stage}; and programs as
    compilation fills them in: a sink, then its linear form, its source and its
    binary. A multi-device buffer or copy carries one {!Ops.Axis_type.Device}
    range over its devices. *)

val program : t
(** [program] is {!shared} as programs restrict it: every width is stated, so a
    constant appears only under the cast that types it and nothing else is weak;
    there is no movement but a shrink of storage by a constant length, no global
    buffer and no {!Ops.invalid}. It adds conditionals ({!Op.If}, {!Op.Endif})
    and lowered, [int32] {!Op.Special}s. *)

val hcq : t
(** [hcq] is {!shared} with the operations of command-queue programs: the
    address of storage on a device ({!Op.Getaddr}), and programs over a buffer
    or parameter. *)

val full : t
(** [full] is the rules of the forms between stages (ends of loops over any
    integers, any ordering, any load or store), then {!tensor}'s, {!program}'s
    and {!hcq}'s rules, in that order. *)

val kernel_graph : t
(** [kernel_graph] accepts the graph of kernel calls: a sink of calls of opaque
    bodies over storage and parameters, with the constants, stacks, casts and
    bitcasts that make their arguments, sharding ({!Op.Mstack}, {!Op.Mselect}),
    {!Ops.Axis_type.Device} ranges, and orderings of storage. *)

(** {1:verify Verifying} *)

val type_verify : ?enter_calls:bool -> t -> Ops.t -> unit
(** [type_verify ~enter_calls spec u] checks every node of [u]'s graph against
    [spec], sources first ({!Ops.toposort}[ ~enter_calls u]; [enter_calls]
    defaults to [true]). When {!Helpers.debug} is 3 or more, a failure first
    prints the graph's nodes on standard error ({!Render.pp_uops}).

    Raises [Invalid_argument] on the first node that fails, naming its position
    in that order, its operation, type and number of sources, each source's
    operation, type and argument, and its own argument. *)
