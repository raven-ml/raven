(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The implementation of {!Norn.Summary}. *)

type t

type finding =
  | Rhat_high of { path : Nx.Ptree.Path.t; index : int array; rhat : float }
  | Ess_low of {
      path : Nx.Ptree.Path.t;
      index : int array;
      bulk : float;
      tail : float;
    }
  | Not_finite of { path : Nx.Ptree.Path.t; index : int array; count : int }
  | Constant of { path : Nx.Ptree.Path.t; index : int array }
  | Divergent of { count : int; total : int; regions : region list }
  | Saturated of { count : int; total : int }
  | Ebfmi_low of { chain : int; ebfmi : float }

and region = { path : Nx.Ptree.Path.t; index : int array; shift : float }

val v :
  'u Nx.Ptree.t ->
  ?stats:'f Stats.t Draws.t ->
  ?superchains:int ->
  'u Draws.t ->
  t

val concat : t list -> t

val findings : t -> finding list

val labels : t -> string array

val columns : t -> (string * Nx.float64_t) list

val pp_finding : Format.formatter -> finding -> unit

val pp : Format.formatter -> t -> unit
