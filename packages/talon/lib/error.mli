(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Errors.

    [Talon.Error] documents errors. This interface adds the plan step that found
    a failure in data. *)

type t

val v :
  ?file:string ->
  ?line:int ->
  ?column:int ->
  ?row_group:int ->
  ?bytes:int * int ->
  ?text:string ->
  string ->
  t

val pp : Format.formatter -> t -> unit
val get_ok : ('a, t) result -> 'a

val in_step : string -> row:int -> t -> t
(** [in_step step ~row e] is [e] found by the plan step [step], as [Query.pp]
    writes it on one line, at the row [row] of its input. {!pp} writes them
    before [e]'s places:
    [filter (Str.parse int32 s > 0): row 3: "x1": not an integer.] The run adds
    them to the failures it finds in data, never to a source's errors, which
    locate themselves. *)
