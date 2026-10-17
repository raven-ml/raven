(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Answers of solves: the implementation of {!Jera.Solution}, with the
    constructor the solves build their answers with and the codes of the
    statuses in a status tensor.

    An answer keeps what its report needs: the entry point, the solve's
    settings, the budget it counts and the facts of each lane. {!get} and
    {!pp} build the report of a lane from these. *)

type 'a t

type status = Converged | Budget_spent | Not_bracketed | Not_finite | Stalled

type fact =
  | Fact : string * (float, 'b) Nx.t -> fact
      (** A named float of each lane, which a report prints. It must
          broadcast to the lanes' shape. *)

type spent = {
  used : (int32, Nx.int32_elt) Nx.t;  (** Each lane's count of the budget. *)
  unit : string;  (** What the budget counts, plural: ["pieces"]. *)
  budget : int;
}
(** The type for what a solve's budget counts. *)

val code : status -> int32
(** [code s] is [s]'s code in a status tensor: [0l] for [Converged], then
    [1l] to [4l] in the order of the constructors. *)

val v :
  fn:string ->
  settings:string ->
  ?spent:spent ->
  fix:(status -> (string * float) list -> string) ->
  value:'a ->
  error:'a ->
  status:(int32, Nx.int32_elt) Nx.t ->
  evaluations:(int32, Nx.int32_elt) Nx.t ->
  facts:fact list ->
  unit ->
  'a t
(** [v ~fn ~settings ?spent ~fix ~value ~error ~status ~evaluations ~facts ()]
    is an answer of the solve [fn] run with [settings], such as
    ["tol rel 1e-06 abs 0"]. [fix st facts] is what to change for a lane that
    ended [st], given its facts' values. Without [spent], the report leaves out
    the budget and each lane's budget used is [0]. *)

val get : 'a t -> 'a
val best : 'a t -> 'a
val ok : 'a t -> (bool, Nx.bool_elt) Nx.t
val is : status -> 'a t -> (bool, Nx.bool_elt) Nx.t
val error : 'a t -> 'a
val evaluations : 'a t -> (int32, Nx.int32_elt) Nx.t
val ptree : 'a Nx.Ptree.t -> 'a t Nx.Ptree.t
val pp : Format.formatter -> 'a t -> unit
