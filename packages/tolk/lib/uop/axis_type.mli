(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Kernel loop axis classification.

    An axis type labels each range (loop) in a kernel schedule with the
    role it plays after lowering: a hardware execution dimension
    (threads, warps, workgroups, global grid), a compiler transform
    (unroll, upcast, reduce), or a plain software loop. The linearizer
    and renderer consult these labels to decide how each range is
    materialised in the emitted source.

    {b Invariant.} The variant declaration order is load-bearing.
    {!compare} delegates to {!Stdlib.compare}, which compares by
    constructor index, and the schedule optimiser relies on this order
    when sorting ranges by [(to_pos, axis_id)]. Reordering variants
    changes observable sort order. *)

(** {1:types Types} *)

(** The axis kinds assignable to a kernel range.

    Exactly one kind is attached to each range; most transforms fix a
    kind at creation and never mutate it afterwards. *)
type t =
  | Global
      (** Global work-grid dimension. Materialises as a workgroup index
          on the target (CUDA block, Metal threadgroup, OpenCL group). *)
  | Warp
      (** Warp / subgroup lane dimension. Used by tensor-core lowerings
          and cross-lane reductions. *)
  | Local
      (** Workgroup-local thread dimension. Materialises as the in-group
          thread index and is the natural companion of shared-memory
          staging. *)
  | Loop
      (** Plain software loop. Emitted as a [for] in the rendered
          source; no implicit parallelism. *)
  | Group_reduce
      (** Reduction across a {!Local} workgroup, typically staged
          through shared memory and a tree reduction. *)
  | Reduce
      (** Sequential reduction axis. Emits an accumulator loop around
          the reduced body. *)
  | Upcast
      (** Fully unrolled axis that becomes a vector lane. Enables
          packed loads/stores and SIMD arithmetic in the renderer. *)
  | Unroll
      (** Fully unrolled axis over a reduction, producing an explicit
          sum of per-iteration expressions. *)
  | Thread
      (** Per-thread private dimension, used by tensor-core lowerings to
          describe register-held tiles. *)
  | Placeholder
      (** Unassigned axis. Temporary kind used between schedule passes;
          no range with this kind may reach the renderer. *)

(** {1:predicates Predicates and ordering} *)

val equal : t -> t -> bool
(** [equal a b] is structural equality on variants. *)

val compare : t -> t -> int
(** [compare a b] orders by variant declaration order (see the module
    invariant). *)

(** {1:fmt Formatting} *)

val to_string : t -> string
(** [to_string t] is the lowercase constructor name (e.g. ["global"],
    ["group_reduce"]). Stable identifier used in serialised kernel
    info and debugging output. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf t] prints [t] using {!to_string}. *)

(** {1:scheduling Scheduling} *)

val to_pos : t -> int
(** [to_pos t] is the schedule priority of [t]. The postrange pass
    orders ranges by [(to_pos kind, axis_id)], which groups them as:
    loops wrap hardware dimensions, hardware dimensions wrap
    reductions, reductions wrap unrolls, and placeholders sink to the
    bottom.

    Priorities:

    {ul
    {- {!Loop} [-> -1]}
    {- {!Thread}, {!Global} [-> 0]}
    {- {!Warp} [-> 1]}
    {- {!Local}, {!Group_reduce} [-> 2]}
    {- {!Upcast} [-> 3]}
    {- {!Reduce} [-> 4]}
    {- {!Unroll} [-> 5]}
    {- {!Placeholder} [-> 6]}} *)

val letter : t -> string
(** [letter t] is the single-character tag used when building axis
    names in the rendered source; the renderer prefixes it to form
    indices like ["gidx0"], ["lidx1"], or ["Ridx0"].

    Mapping:

    {ul
    {- {!Global} [-> "g"]}
    {- {!Thread} [-> "t"]}
    {- {!Local} [-> "l"]}
    {- {!Warp} [-> "w"]}
    {- {!Loop} [-> "L"]}
    {- {!Upcast} [-> "u"]}
    {- {!Group_reduce} [-> "G"]}
    {- {!Reduce} [-> "R"]}
    {- {!Unroll} [-> "r"]}
    {- {!Placeholder} [-> "?"]}} *)
