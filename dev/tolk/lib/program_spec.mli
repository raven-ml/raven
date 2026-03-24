(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Compile-time kernel descriptions extracted from {!Ir.Program.t}.

    A {!t} is the runtime-facing description of a lowered kernel before
    device-specific preparation. It captures the lowered program, kernel name,
    launch metadata, scalar variables, buffer reads and writes, and cost
    estimates.

    Scalar {e variables} are runtime parameters defined by
    {!Ir.Program.instr.Define_var} instructions. Buffer {e reads} and {e writes}
    are parameter indices traced from {!Ir.Program.instr.Load} and
    {!Ir.Program.instr.Store} instructions respectively. *)

(** {1:types Types} *)

type var = {
  name : string;  (** Variable name matching the IR definition. *)
  lo : int;  (** Inclusive lower bound. *)
  hi : int;  (** Inclusive upper bound. *)
  dtype : Tolk_ir.Dtype.t;  (** Scalar data type. *)
}
(** The type for scalar kernel parameters with runtime bounds.

    Each [var] corresponds to one {!Ir.Program.instr.Define_var} instruction in
    the lowered program. *)

type core_id = {
  var_index : int;  (** Index into {!vars} identifying this variable. *)
  lo : int;  (** Inclusive lower bound. Always [0]. *)
  hi : int;  (** Inclusive upper bound. *)
}
(** The type for the runtime-managed ["core_id"] variable.

    When present, ["core_id"] enables multi-core dispatch. The runtime assigns
    each core a value in \[[lo];[hi]\]. *)

val thread_count : core_id -> int
(** [thread_count cid] is [cid.hi - cid.lo + 1]. *)

(** The type for kernel launch models. A kernel uses exactly one model. *)
type launch_kind =
  | Serial  (** No parallelism. Global and local sizes are all [1]. *)
  | Thread_groups
      (** Thread-group model (e.g. Metal, CUDA blocks). Both global and local
          dimensions are meaningful. *)
  | Threads
      (** Flat-thread model (e.g. OpenCL global work). Only global dimensions
          are meaningful; local size is [None]. *)

(** {1:estimates Cost estimates} *)

module Estimates : sig
  (** Estimated kernel costs for scheduling and profiling.

      Each cost component is either an exact integer or a symbolic expression
      preserved from the upstream {!Ir.Kernel.estimates}. *)

  (** The type for a single cost component. *)
  type estimate =
    | Int of int  (** Exact integer count. *)
    | Symbolic of string  (** Symbolic expression from the IR. *)

  type t = {
    ops : estimate;  (** Arithmetic operation count. *)
    lds : estimate;  (** Local data share (shared memory) access count. *)
    mem : estimate;  (** Global memory access count. *)
  }
  (** The type for kernel cost estimates. *)

  val zero : t
  (** [zero] is [{ops = Int 0; lds = Int 0; mem = Int 0}]. *)

  val ( + ) : t -> t -> t
  (** [a + b] is the component-wise sum of [a] and [b]. Two [Int] values produce
      an [Int]. When either side is [Symbolic], the result is [Symbolic] with
      the expressions concatenated. *)

  val of_kernel : Tolk_ir.Kernel.estimates -> t
  (** [of_kernel e] is the lossless conversion of {!Tolk_ir.Kernel.estimates} [e]. *)

  val of_program : Tolk_ir.Program.t -> t
  (** [of_program p] computes estimates by walking [p]. Counts FLOPs (excluding
      index arithmetic), load/store bytes, and total memory accessed (capped at
      buffer size for re-reads). Loop multipliers are stacked through
      {!Tolk_ir.Program.view.Range}/{!Tolk_ir.Program.view.End_range} and
      {!Tolk_ir.Program.view.Special} nodes. *)
end

(** {1:spec Kernel specifications} *)

type t
(** The type for compile-time kernel descriptions.

    Invariants:
    - Variable order is stable and sorted by [(name, lo, hi)].
    - Read and write parameter indices are sorted and deduplicated.
    - Launch metadata uses exactly one model: {!Serial}, {!Thread_groups}, or
      {!Threads}.
    - ["core_id"], when present, is unique and has [lo = 0]. *)

(** {2:constructors Constructors} *)

val of_program : ?estimates:Estimates.t -> name:string -> Tolk_ir.Program.t -> t
(** [of_program ~name program] extracts a kernel description from [program].

    [estimates] defaults to {!Estimates.zero}.

    Raises [Invalid_argument] if:
    - launch metadata depends on an unsupported scalar instruction,
    - a launch axis is outside [0..2],
    - a launch axis is repeated,
    - launch metadata mixes flat-thread and thread-group models,
    - ["core_id"] is defined more than once, or
    - ["core_id"] has a lower bound different from [0]. *)

val with_estimates : Estimates.t -> t -> t
(** [with_estimates e spec] is [spec] with estimates replaced by [e]. *)

val with_global_dims : int array -> t -> t
(** [with_global_dims dims spec] is [spec] with the global launch dimensions
    replaced by constant values [dims]. Used by beam search to scale down
    kernel size during timing. *)

(** {2:accessors Accessors} *)

val name : t -> string
(** [name spec] is the kernel entry-point name. *)

val program : t -> Tolk_ir.Program.t
(** [program spec] is the lowered IR program consumed by renderers. *)

val vars : t -> var list
(** [vars spec] is the scalar variable definitions in stable argument order. *)

val outs : t -> int list
(** [outs spec] is the sorted, deduplicated parameter indices written by the
    kernel. *)

val ins : t -> int list
(** [ins spec] is the sorted, deduplicated parameter indices read by the kernel.
*)

val globals : t -> int list
(** [globals spec] is the sorted, deduplicated union of {!outs} and {!ins}. *)

val core_id : t -> core_id option
(** [core_id spec] is the runtime-managed ["core_id"] variable, if any. *)

val launch_kind : t -> launch_kind
(** [launch_kind spec] is the kernel launch model. *)

val estimates : t -> Estimates.t
(** [estimates spec] is the kernel cost estimates. *)

(** {2:launch Launch dimensions} *)

val launch_dims : t -> int list -> int array * int array option
(** [launch_dims spec args] evaluates the launch dimensions of [spec] using
    runtime scalar values [args].

    The first element of the pair is the global dimensions (length [3]). The
    second is the local dimensions when meaningful:
    - {!Serial} produces [([|1; 1; 1|], Some [|1; 1; 1|])].
    - {!Thread_groups} produces [(global, Some local)].
    - {!Threads} produces [(global, None)].

    [args] must contain one value per {!var} in the order returned by {!vars}.
*)
