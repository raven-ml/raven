(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Plan problems.

    A problem is one thing wrong with a plan that its input schema reveals: a
    missing name, a kind that does not bind, types that do not meet, a literal
    its type does not hold. Selectors ([Sel.check]), expressions
    ([Expr.bind_out], [Expr.bind_predicate]) and verbs produce them. A verb
    collects every problem it finds and raises one [Invalid_argument] that lists
    them, so problems never escape the library.

    This module is internal: [Talon] does not export it. *)

type t
(** The type for problems. A problem is a message for people, without the
    location that the verb's report puts around it. *)

val v : ('a, Format.formatter, unit, t) format4 -> 'a
(** [v fmt …] is the problem whose message [fmt] formats, as in
    [v "%a and %a do not meet" Type.pp t0 Type.pp t1]. The message is one
    sentence, without a trailing newline. *)

val missing : string -> Schema.t -> t
(** [missing name s] is the problem that [s] has no column [name]. Its message
    suggests the names of [s] nearest to [name] by edit distance, in schema
    order, when that distance is at most 2, as in
    [no column "carier". Did you mean "carrier"?], and lists [s]'s names when
    none is that close. *)

val repeated : string list -> string list
(** [repeated ns] is the names that [ns] holds more than once, each once, in the
    order of their second occurrence. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf p] formats [p]'s message. *)
