(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A compiled function's programs, by key. This module implements the keys of
    {!Rune.section-jit}.

    A compiled function is traced over stand-ins ({!stand_ins}). Each program
    compiles the first time a call runs it compiled, then once per variant
    ({!Compile}). A key's programs differ in the answers their traces got to the
    questions they asked of the call's context: the number of lanes of a map of
    a name ({!Program.bound}), and whether a collect of a probe is open
    ({!Program.opened}). A call runs the first of its key's programs whose
    questions it answers as that program's trace was answered, and traces
    another where none does, so it traces when the answers to the questions
    its trace would ask are new to its key. A call that traces holds its key's
    next program's latch, so a call of another domain waits for it; a call of
    the same key made inside that trace traces again, and its program is kept
    beside the others.

    With [RUNE_JIT_DEBUG] a nonzero integer in the environment
    ({!Compile.report}), a call reports each retrace, with the first difference
    between its key and that of the latest call that returned, on any domain, or
    the line [a map around the call differs] or
    [a probe's collect is open or closed], and, the first time, why its program
    evaluates uncompiled. *)

type probe =
  | Probe : 'a Program.probe -> probe  (** A probe a trace asked about. *)

type 'y entry = {
  body : Program.t;  (** The function traced. *)
  like : 'y;  (** The result the trace returned, for its structure. *)
  site : Program.site;
      (** The site of [body] ({!Program.site}), over its arguments then its
          totals' running values in. It runs [body] compiled ({!Compile.run}),
          or evaluated where the compiler declines it ({!Compile.evaluate}), and
          compiles the programs derived from it alike. *)
  axes : Program.axis list;
      (** Each map the trace asked about ({!Program.Bound_asked}) or [body]
          names ({!Program.axes}), once. *)
  lanes : int list;
      (** The lanes of each map of [axes] at the trace, [-1] for one not around
          it. *)
  asked : probe list;
      (** Each probe a site of the trace asked about ({!Program.Open}), once. *)
  probes : (int * bool) list;
      (** The number of each probe of [asked] and whether a collect of it was
          open. *)
}
(** The type for a key's program. *)

type 'y t
(** The type for the programs of one compiled function whose results are of
    type ['y]. *)

val create : name:string -> ?beam:int -> ?parallel:int -> unit -> 'y t
(** [create ~name ~beam ~parallel ()] is an empty table, for the entry point
    [name], whose programs, and those derived from them, compile with
    {!Compile.make}'s [beam] and [parallel]. *)

val stand_ins : 'd Nx.packed list -> Program.form array
(** [stand_ins leaves] is the forms a call traces [leaves] at. Each has its
    leaf's dtype, shape and placement, laid out in C order. *)

val footprint :
  name:string ->
  Program.t ->
  Program.form array ->
  Program.value array ->
  (Rig.t * int) list
(** [footprint ~name body forms args] is {!Compile.footprint} of [body] over
    [args], of [forms], as a compiled function's first call over [args]
    would make it, its emissions as outputs.

    Raises [Invalid_argument] where the compiler declines the program. *)

val find :
  ?totals:Program.addition array ->
  'y t ->
  'd Nx.packed list ->
  Nx.Ptree.Skeleton.t ->
  paths:(unit -> string list) ->
  (Program.form array ->
  Program.addition array ->
  Program.t * 'y * Program.asked list) ->
  'y entry
(** [find ~totals t leaves skeleton ~paths trace] is the program of the call
    whose arguments' leaves are [leaves], in walk order, with skeleton
    [skeleton] and its totals' running values in lying at the placements
    [totals] gives. The first time [t] meets its key, [trace forms totals] makes
    it, where [forms] is the leaves' stand-ins, and gives the questions it
    asked. A call that answers one of those otherwise than at the trace has a
    program of its own. Calls on several domains wait for each other as {!Cache}
    states. A [trace] that raises installs nothing, and the next call with the
    key traces again. [paths ()] names the leaves in reports. *)
