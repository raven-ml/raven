(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A compiled function's programs, one per key.

    A call's {e key} is what its program depends on of the call: its
    arguments' skeleton ({!Nx.Ptree.Skeleton}); for each leaf, its dtype, its
    layout over the run of storage it reads (shape, strides, and its offset
    from the run's start), its placement, whether it is a donated handle
    ({!Nx.donate}) and whether an interpretation traces it; and every setting
    of tolk that shapes what it compiles ({!Tolk.Setting.shaping}), the search
    widths [BEAM] and [JITBEAM] among them, and whether the engine links each
    kernel as a submission of its own ({!Tolk.Setting.debug} at [2] or more).
    Two calls share a program exactly when their keys are equal.

    With [RUNE_JIT_DEBUG=1] in the environment, a call prints on standard
    error each retrace, with the first difference between its key and the
    previous call's, and why a program evaluates uncompiled. *)

type 'y entry = {
  body : Program.t;  (** The function traced. *)
  like : 'y;  (** The result the trace returned, for its structure. *)
  compiled : (Program.value array -> Program.value array) option;
      (** [body] compiled ({!Compile.run}), where the leaves are untraced and
          the compiler takes it. *)
}
(** The type for a key's program. *)

type 'y t
(** The type for the programs of one compiled function whose results are of
    type ['y]. *)

val create : name:string -> unit -> 'y t
(** [create ~name ()] is an empty table, for the entry point [name]. *)

val find :
  'y t ->
  'd Nx.packed list ->
  Nx.Ptree.Skeleton.t ->
  paths:(unit -> string list) ->
  (unit -> Program.t * 'y) ->
  'y entry
(** [find t leaves skeleton ~paths trace] is the program of the call whose
    arguments' leaves are [leaves], in walk order, and skeleton [skeleton],
    made by [trace ()] the first time [t] meets its key, then compiled. A key
    being made makes the other calls with that key wait; a [trace] that raises
    installs nothing, and the next call with the key traces again. [paths ()]
    names the leaves in reports. *)
