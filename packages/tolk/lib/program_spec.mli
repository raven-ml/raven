(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Compile-time kernel descriptions extracted from a linearized uop
    program.

    A {!t} is the runtime-facing description of a lowered kernel before
    device-specific preparation. It captures the lowered program, kernel
    name, launch metadata, scalar variables, buffer reads and writes,
    and cost estimates. *)

(** {1:types Types} *)

type program = Tolk_uop.Uop.t list
(** The linearized uop program produced by the linearizer. *)

type var = {
  name : string;  (** Variable name matching the IR definition. *)
  lo : int;  (** Inclusive lower bound. *)
  hi : int;  (** Inclusive upper bound. *)
  dtype : Tolk_uop.Dtype.t;  (** Scalar data type. *)
}
(** Bounded scalar {!Tolk_uop.Uop.Param_arg} kernel parameter. *)

type core_id = {
  var_index : int;
  lo : int;
  hi : int;
}
(** Runtime-managed ["core_id"] variable for multi-core dispatch. *)

val thread_count : core_id -> int
(** [thread_count cid] is [cid.hi - cid.lo + 1]. *)

type launch_kind =
  | Serial
  | Thread_groups
  | Threads
(** Kernel launch model. *)

(** {1:estimates Cost estimates} *)

module Estimates : sig
  type estimate =
    | Int of int
    | Symbolic of Tolk_uop.Uop.t

  type t = {
    ops : estimate;
    lds : estimate;
    mem : estimate;
  }

  val zero : t
  (** [zero] is [{ops = Int 0; lds = Int 0; mem = Int 0}]. *)

  val ( + ) : t -> t -> t
  (** [a + b] is the component-wise sum of [a] and [b]. *)

  val of_uop : Tolk_uop.Uop.estimates -> t
  (** [of_uop e] converts a uop estimates record. *)

  val of_program : program -> t
  (** [of_program p] computes estimates by walking [p]. *)
end

(** {1:spec Kernel specifications} *)

type t
(** Compile-time kernel description. *)

val of_program :
  name:string ->
  src:string ->
  device:string ->
  ?lib:bytes ->
  ?applied_opts:Tolk_uop.Uop.Opt.t list ->
  ?estimates:Estimates.t ->
  ?aux:string list ->
  program ->
  t
(** [of_program ~name ~src ~device ?lib ?applied_opts ?estimates ?aux program]
    extracts a kernel description from [program]. If [estimates] is omitted,
    estimates are computed from [program]. [aux] is copied to
    {!Tolk_uop.Uop.program_info}. *)

val with_lib : bytes -> t -> t
(** [with_lib lib spec] is [spec] with [lib] set to [Some lib]. *)

val with_estimates : Estimates.t -> t -> t
(** [with_estimates e spec] is [spec] with estimates replaced by [e]. *)

val with_global_dims : int array -> t -> t
(** [with_global_dims dims spec] is [spec] with the global launch
    dimensions replaced by constant values [dims]. *)

val name : t -> string
val src : t -> string
val device : t -> string
val program : t -> program
val lib : t -> bytes option
val applied_opts : t -> Tolk_uop.Uop.Opt.t list
val vars : t -> var list
val outs : t -> int list
val ins : t -> int list
val globals : t -> int list
val core_id : t -> core_id option
val launch_kind : t -> launch_kind
val estimates : t -> Estimates.t
val global_size : t -> Tolk_uop.Uop.t array
val local_size : t -> Tolk_uop.Uop.t array option
val program_info : t -> Tolk_uop.Uop.program_info
(** [program_info spec] is the tinygrad-shaped program metadata carried by
    [spec]. Symbolic global dimensions are preserved as launch expressions;
    local dimensions are present only when all local dimensions are fixed
    integers, and backend auxiliary metadata is preserved. *)

val launch_dims : t -> (string * int) list -> int array * int array option
(** [launch_dims spec var_vals] evaluates launch dimensions by replacing
    symbolic variables with the values in [var_vals]. *)
