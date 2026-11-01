(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** The calls of a schedule, and the compilation of their kernels.

    A schedule is an {!Op.Linear} of calls ({!Schedule}). This module reads what
    a call does (its buffers, its variables, what it writes, its name and its
    cost) and compiles every kernel a schedule calls, in parallel. Running them
    is the engine's. *)

(** {1:calls Calls} *)

val get_call_arg_uops : Ops.t -> Ops.t list
(** [get_call_arg_uops call] is the buffer arguments of [call]: its arguments
    without its scalar ones, the variables it binds and the values it passes. *)

val get_call_kernel_args : Ops.t -> Ops.t -> Ops.t list
(** [get_call_kernel_args call prg] is what [call] passes for each parameter
    of the program [prg] ({!Shape.kernel_params}), in the order the program
    takes them: for a buffer, [call]'s argument in its slot; for a scalar
    parameter that a slot names, [call]'s argument in that slot, an expression
    of the loops around [call]; for a named variable, the constant [call] binds
    it to, or else the variable, at slot [-1], that a run binds by name. The
    bounds of a split program's block ({!Ops.kernel_info.split}) are those of
    its whole loop.

    Raises [Invalid_argument] if [prg] is not a linearized program. *)

val get_call_outs_ins : Ops.t -> int list * int list
(** [get_call_outs_ins call] is the positions among [call]'s arguments of the
    buffers it writes and of those it reads: a program's
    ({!Ops.program_info.outs} and {!Ops.program_info.ins}), [([0], [1])] for a
    copy, and [([], [])] for any other call. *)

val get_call_name :
  ?var_vals:(string * int) list -> Ops.t -> Ops.t list -> string
(** [get_call_name ~var_vals call bufs] is how diagnostics name [call], whose
    buffers are [bufs]: a program's kernel name, as its kernel gives it, or, for
    a copy, ["copy S, D <- E"] in yellow, where [S] is the size copied with the
    variables of [var_vals] (default [[]]) replaced by their values, and [D] and
    [E] the first seven characters of the destination's and the source's
    devices.

    Raises [Invalid_argument] for any other call. *)

val estimate_uop : Ops.t -> Ops.estimates
(** [estimate_uop call] is the cost of [call], seen through its {!Op.After}s: a
    program's kernel estimates and, for a copy, its bytes as both memory
    touched and bytes loaded and stored. Any other call costs nothing. *)

(** {1:compiling Compiling} *)

val lower_and_compile :
  ?search:(int -> Postrange.Scheduler.t -> Postrange.Scheduler.t) ->
  targets:(string -> Helpers.Target.t) ->
  Ops.t ->
  Ops.t
(** [lower_and_compile ~search ~targets linear] is [linear] with the body of
    each call that is a kernel ({!Op.Sink} with {!Ops.type-kernel_info}),
    outside the bodies of the calls, replaced by its compiled program
    ({!Codegen.to_program}), for the renderer
    {!Device.renderer} picks for the target [targets d] of the call's device [d]
    (its first, on several).

    Each kernel is compiled once. A kernel whose program this process has made
    ({!Codegen.made}) takes it as it is; the others are compiled in parallel on
    {!Worker.map}, except when there is only one, or when a kernel of [linear]
    asks for a beam search: the search times candidates ([search]), which a
    concurrent compilation would disturb, so the kernels are then compiled in
    order. [search] is {!Codegen.to_program}'s [beam].

    Raises as {!Codegen.to_program} does, and [Invalid_argument] if
    {!Device.renderer} finds no renderer for a target. *)

val compile_linear :
  ?search:(int -> Postrange.Scheduler.t -> Postrange.Scheduler.t) ->
  targets:(string -> Helpers.Target.t) ->
  Ops.t ->
  Ops.t
(** [compile_linear ~search ~targets linear] is the schedule [linear], whose
    devices compile for the targets [targets] gives, ready to link and run:
    + when the setting {!Setting.beam} is [1] or more, each kernel that asks for
      no beam search asks for one of that width;
    + its kernels are compiled ({!lower_and_compile}, with [search]).

    Each call of the result is a call of a compiled {!Op.Program} (a kernel), or
    of an {!Op.Store} of its second argument's storage into its first's (a
    copy); an {!Op.End} of ranges around such calls, which run once for each
    combination of the ranges' values, the last range varying fastest; or an
    {!Op.Backedge} of a range and a flag around them, which run once per trip of
    the range while the flag's boolean holds.

    Raises as {!lower_and_compile} does. *)
