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

  val to_uop : t -> Tolk_uop.Uop.estimates
  (** [to_uop t] is the inverse of {!of_uop}. *)

  val of_program : program -> t
  (** [of_program p] computes estimates by walking [p]. A loop counts its
      body once per iteration; a loop whose trip count reads memory counts at
      the trip count's upper bound. *)
end

(** {1:spec Kernel specifications} *)

type t
(** Compile-time kernel description. *)

val of_program :
  name:string ->
  src:string ->
  device:string ->
  ?target:Tolk_uop.Target.t ->
  ?lib:bytes ->
  ?applied_opts:Tolk_uop.Uop.Opt.t list ->
  ?estimates:Estimates.t ->
  program ->
  t
(** [of_program ~name ~src ~device ?target ?lib ?applied_opts ?estimates program]
    extracts a kernel description from [program]. If [estimates] is omitted,
    estimates are computed from [program]. [target] records the compilation
    target and defaults to an unspecified target. *)

val with_lib : bytes -> t -> t
(** [with_lib lib spec] is [spec] with [lib] set to [Some lib]. *)

val to_elf : t -> Tolk_uop.Tiny_elf.t
(** [to_elf spec] is [spec]'s binary and argument signature.
    Raises [Invalid_argument] if [spec] has no compiled binary. *)

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
val launch_kind : t -> launch_kind
val estimates : t -> Estimates.t
val global_size : t -> Tolk_uop.Uop.t array
val local_size : t -> Tolk_uop.Uop.t array option
val program_info : t -> Tolk_uop.Uop.program_info
(** [program_info spec] is the runtime-facing program metadata carried by
    [spec]. Symbolic global dimensions are preserved as launch expressions;
    local dimensions are present only when every local dimension is a fixed
    integer. *)

val launch_dims : t -> (string * int) list -> int array * int array option
(** [launch_dims spec var_vals] evaluates launch dimensions by replacing
    symbolic variables with the values in [var_vals].

    Raises [Invalid_argument], naming the program and variable, if a symbolic
    dimension references a missing variable. *)
