(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A jit program compiled through tolk.

    A program compiles when it is straight-line, all its values lie on one
    device and every operation lowers ({!Lower.op}): its parameters bind the
    storage of the arguments a call passes, its constants are bound once as
    captures, and each result is stored into memory of its own. The program is
    lowered and linked once per {e variant}: each argument's layout over the
    run of storage it reads ({!within}), where that run starts within 16 bytes
    of memory (its {e phase}), and the settings of tolk that shape what it
    compiles. *)

type t
(** The type for compiled programs. *)

val within : ('v, 's) Nx.Dtype.t -> Nx_array.Layout.t -> Nx_array.Layout.t
(** [within dt l] is [l] over the run of storage a parameter of [dt] laid out
    by [l] binds: [l]'s shape and strides, offset from the run's start, the
    element at or below [l]'s first position whose bits start on a multiple of
    16 bytes from its buffer's start. A layout with no element is itself. *)

val report : ('a, out_channel, unit) format -> 'a
(** [report fmt] prints on standard error a line starting [rune.jit: ] where
    the environment held [RUNE_JIT_DEBUG=1] when the module was initialised,
    and nothing otherwise. *)

val make : name:string -> Program.t -> Program.form array -> (t, string) result
(** [make ~name p forms] is [p], traced over arguments of [forms], ready to
    link once per variant at its first call ({!run}), or [Error why] where the
    lowering declines it, [why] saying which value, construct or failure: a
    map's construct ({!Program.Lanes}, {!Program.Lane_index}), a check in a
    construct inside a loop's step, an operation that {!Lower.op} does not
    lower, a value on another device than the others or over several, or a
    check off the host. [name] is the entry point messages name.

    A construct's programs lower with [p]'s: a call's, a remat's, a custom
    rule's primal and a linear call's function as their equations over the
    construct's operands, and a scan or an iterate as one loop of [p], a range
    of its trips around one call of its step, the parts of the step that read
    no trip's parameter computed before the loop. A loop until a stop tests a
    flag the engine reads before each trip, and its failure after [max] trips
    is a check of [p]. A step's checks are carried: the first trip that fails
    keeps each check's first failure and data, which are checks of [p]. *)

val run : t -> Program.value array -> Program.value array
(** [run t args] is the outputs of [t]'s program applied to [args], values of
    the forms [t] was made for that no interpretation traces: each result
    fresh, C-contiguous, at its output's form's placement. The first call at a
    new variant links the program for it.

    The call is a consumer of [args] ({!Nx.Prim.Consumer}): it claims the
    donated handles among them, whose donors die, and reads each argument as
    it was then. A result takes a donated argument's memory where rune2.mli's
    lending rules allow it and {!Nx.Prim.Consumer.take} gives it; otherwise
    it computes over a copy of that argument. Each check, on the host, is
    computed with the program, and the first that failed raises its
    exception once the program ran. A variant the lowering or tolk declines is evaluated
    over the consumer's arrays ({!evaluate}). Raises [Invalid_argument] as
    {!Nx.Prim.Consumer.with_} does, and what linking a variant raises when it
    does not decline it. *)

val evaluate :
  name:string -> Program.t -> Program.value array -> Program.value array
(** [evaluate ~name p args] is [p]'s outputs evaluated by nx over [args], none
    of them traced: an output that is an argument is a copy of it. Where
    [args] hold donated handles, it claims every argument first, as {!run}
    does, and evaluates [p] over the arrays they hold, copying each
    result. *)
