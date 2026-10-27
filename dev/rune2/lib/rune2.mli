(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Transformations of functions over nx values: compilation.

    A transformation takes the structure ({!Nx.Ptree.t}) of each value whose
    arrays it enumerates; everything else the function uses is captured and is a
    constant of the transformation.

    {b Traced functions.} The function {!jit} receives is {e traced} at the
    call: its OCaml code runs once per call over stand-ins of its arguments'
    arrays, values with a dtype, a shape and a placement but no elements, and
    what it computes is recorded as a program, which the call then evaluates.
    Inside it {!Nx.item} of a stand-in, or of a value computed in it, raises
    ([Nx.item: the value is traced by Rune.jit]), and a draw without [~key]
    raises naming {!jit}.

    {b Errors.} Every message starts with the entry point the caller applied, as
    ["Rune.jit"] for {!jit}. *)

(** {1:jit Compilation} *)

val jit : ('a -> 'b, 'd) Nx.Ptree.fn -> ('a -> 'b) -> 'a -> 'b
(** [jit s f] is [f] traced into a program, [s] its signature, one structure per
    argument and one for the result. Each call traces [f] over its arguments'
    forms and evaluates the program. Every operation [f] applies is staged,
    those over values [f] captures included.

    Raises [Invalid_argument] when applied to [s] if [s] has no argument, and at
    a call if [f] captures a value that another transformation traces. *)

(** {1:one Functions of one array} *)

val jit' :
  (('v, 's, 'd) Nx.t -> ('w, 'r, 'd) Nx.t) ->
  ('v, 's, 'd) Nx.t ->
  ('w, 'r, 'd) Nx.t
(** [jit' f] is {!jit} of a function of one array. *)
