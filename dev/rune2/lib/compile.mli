(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A jit program compiled through tolk.

    A program compiles when it is straight-line, all its values lie on one
    device and every operation lowers ({!Lower.op}): its parameters bind the
    storage of the arguments a call passes, its constants are bound once as
    captures, and each result is stored into memory of its own. The schedule is
    compiled and linked once per argument's {e phase}, where its storage starts
    within 16 bytes of memory. *)

type t
(** The type for compiled programs. *)

val layout : Program.form -> Nx_array.Layout.t
(** [layout f] is the layout a parameter of form [f] binds: [f]'s, or C order
    for a value of every set, which a call computes afresh. *)

val within : ('v, 's) Nx.Dtype.t -> Nx_array.Layout.t -> Nx_array.Layout.t
(** [within dt l] is [l] over the run of storage a parameter of [dt] laid out
    by [l] binds: [l]'s shape and strides, offset from the run's start, the
    element at or below [l]'s first position whose bits start on 16 bytes.
    A layout with no element is itself. Two arguments whose layouts give
    equal ones read their runs alike. *)

val report : ('a, out_channel, unit) format -> 'a
(** [report fmt] prints on standard error a line starting [rune.jit: ] where
    the environment held [RUNE_JIT_DEBUG=1] when the module was initialised,
    and nothing otherwise. *)

val make : name:string -> Program.t -> Program.form array -> (t, string) result
(** [make ~name p forms] is [p], traced over arguments of [forms], lowered and
    ready to run, or [Error why] where the compiler declines it, [why] saying
    which value or construct: a loop or a map's construct ({!Program.Scan},
    {!Program.Iterate}, {!Program.Lanes}, {!Program.Lane_index}), an operation
    that {!Lower.op} does not lower, a value on another device than the
    others or over several, or a result of every set computed from the
    arguments, which has no storage. [name] is the entry point messages name.
*)

val run : t -> Program.value array -> Program.value array
(** [run t args] is the outputs of [t]'s program applied to [args], values of
    the forms [t] was made for that no interpretation traces: each result
    fresh, C-contiguous, at its output's form's placement. A result that is a
    constant of every set is that constant. The first call with arguments at
    new phases compiles and links the program for them.

    The call is a consumer of [args] ({!Nx.Prim.Consumer}): it claims the
    donated handles among them, whose donors die, and reads each argument as
    it was then. Raises [Invalid_argument] as {!Nx.Prim.Consumer.with_} does,
    and what tolk raises compiling or linking. *)
