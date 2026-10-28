(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A compiled function's programs, one per key. This module implements the keys
    of {!Rune2.section-jit}.

    A compiled function is traced once per key, over stand-ins ({!stand_ins}).
    Its program compiles the first time a call runs it compiled, then once per
    variant ({!Compile}). A call first finds the program of its key without the
    number of lanes of any map. Where the number of lanes of a map the program
    names differs from those at its trace, the call then finds the program of
    its key with them, tracing it the first time. A call whose key equals the
    previous call's, where that call found its program without lanes, reads that
    program without the table's lock.

    With [RUNE_JIT_DEBUG=1] in the environment ({!Compile.report}), a call
    reports each retrace, with the first difference between its key and the
    previous call's or the line [a map's lanes differ], and why its program
    evaluates uncompiled. *)

type 'y entry = {
  body : Program.t;  (** The function traced. *)
  like : 'y;  (** The result the trace returned, for its structure. *)
  site : Program.site;
      (** The site of [body] ({!Program.site}), over its arguments then its
          totals' running values in. It runs [body] compiled ({!Compile.run}),
          or evaluated where the compiler declines it ({!Compile.evaluate}), and
          compiles the programs derived from it alike. *)
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
(** [stand_ins leaves] is the forms a call traces [leaves] at. Each has its
    leaf's dtype, shape and placement, laid out in C order. *)

val find :
  ?totals:Program.addition array ->
  'y t ->
  'd Nx.packed list ->
  Nx.Ptree.Skeleton.t ->
  paths:(unit -> string list) ->
  (Program.form array -> Program.addition array -> Program.t * 'y) ->
  'y entry
(** [find ~totals t leaves skeleton ~paths trace] is the program of the call
    whose arguments' leaves are [leaves], in walk order, with skeleton
    [skeleton] and its totals' running values in lying at the placements
    [totals] gives. The first time [t] meets its key, [trace forms totals] makes
    it, where [forms] is the leaves' stand-ins. Calls on several domains wait
    for each other as {!Cache} states. A [trace] that raises installs nothing,
    and the next call with the key traces again. [paths ()] names the leaves in
    reports. *)
