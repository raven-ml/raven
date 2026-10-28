(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A compiled function's programs, one per key.

    A call's {e key} is the abstract value of its arguments: their skeleton
    ({!Nx.Ptree.Skeleton}), and each leaf's dtype, shape and placement, and
    the lanes of each map its program names, where they differ from those it
    was first traced at. A key is traced once, over stand-ins in C order; its
    program is compiled the first time it runs compiled, then once per variant
    of its arguments' layouts, phases and tolk's settings ({!Compile}).

    With [RUNE_JIT_DEBUG=1] in the environment, a call prints on standard
    error each retrace, with the first difference between its key and the
    previous call's, and why a program evaluates uncompiled. *)

type 'y entry = {
  body : Program.t;  (** The function traced. *)
  like : 'y;  (** The result the trace returned, for its structure. *)
  site : Program.site;
      (** [body] compiled ({!Compile.run}), or evaluated where the compiler
          declines it ({!Compile.evaluate}), over untraced arguments then its
          totals' running values in, and the programs derived from it, compiled
          alike. *)
  lanes : int list;
      (** The lanes of each map [body] names ({!Program.axes}) at the trace,
          [-1] for one not around it. *)
}
(** The type for a key's program. *)

type 'y t
(** The type for the programs of one compiled function whose results are of
    type ['y]. *)

val create : name:string -> unit -> 'y t
(** [create ~name ()] is an empty table, for the entry point [name]. *)

val stand_ins : 'd Nx.packed list -> Program.form array
(** [stand_ins leaves] is the forms a call traces [leaves] at: their dtypes,
    shapes and placements, in C order. *)

val find :
  ?totals:Program.addition array ->
  'y t ->
  'd Nx.packed list ->
  Nx.Ptree.Skeleton.t ->
  paths:(unit -> string list) ->
  (Program.form array -> Program.addition array -> Program.t * 'y) ->
  'y entry
(** [find ~totals t leaves skeleton ~paths trace] is the program of the call
    whose arguments' leaves are [leaves], in walk order, and skeleton
    [skeleton], its totals' running values in placed as [totals] places them,
    made by [trace forms totals] the first time [t] meets its key, [forms] the
    leaves' stand-ins. A key being made makes the other calls with that key
    wait, but on the domain making it ({!Cache}); a [trace] that raises installs
    nothing, and the next call with the key traces again. [paths ()] names the
    leaves in reports. *)
