(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Column selectors.

    [Talon.Sel] documents the selectors that users write. This interface adds
    their resolution and their formatting, which serve the verbs and the plan
    printer. *)

type t

val all : t
val names : string list -> t
val prefix : string -> t
val suffix : string -> t
val of_kind : 'a Kind.t -> t
val where : (string -> Type.any -> bool) -> t
val ( + ) : t -> t -> t
val ( - ) : t -> t -> t
val inter : t -> t -> t

(** {1:resolving Resolving} *)

val check : t -> Schema.t -> string list * Problem.t list
(** [check sel s] is [(ns, ps)]: [ns] the names that [sel] selects in [s],
    distinct and in the order the constructors state, and [ps] one
    {!Problem.missing} per name of a {!names} that [s] lacks, in the order they
    appear in [sel]. A selector that matches nothing selects [[]], which is no
    problem. *)

(** {1:fmt Formatting} *)

val pp_arg : Format.formatter -> t -> unit
(** [pp_arg ppf s] formats [s] as it is written inside [Sel.( … )], as a
    function's argument: in parentheses unless it is [all], as in
    [(prefix "wk" - names ["wk76"])]. A {!where} formats as [where <fn>]. *)
