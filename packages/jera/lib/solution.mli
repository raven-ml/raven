(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Answers of solves: the implementation of {!Jera.Solution}, with the
    constructor the solves build their answers with and the codes of the
    statuses in a status tensor.

    An answer keeps what its report needs: the entry point, the solve's
    settings, the budget it counts and the facts of each lane. {!get} and {!pp}
    build the report of a lane from these. *)

type ('a, 'd) t
type status = Converged | Budget_spent | Not_bracketed | Not_finite | Stalled

type 'd fact =
  | Fact : string * (float, 'b, 'd) Nx.t -> 'd fact
      (** A named float of each lane, which a report prints. It must broadcast
          to the lanes' shape. *)

type 'd spent = {
  used : 'd Nx.int32_t;  (** Each lane's count of the budget. *)
  unit : string;  (** What the budget counts, plural: ["pieces"]. *)
  budget : int;
}
(** The type for what a solve's budget counts. *)

val code : status -> int32
(** [code s] is [s]'s code in a status tensor: [0l] for [Converged], then [1l]
    to [4l] in the order of the constructors. *)

val v :
  fn:string ->
  settings:string ->
  ?spent:'d spent ->
  fix:(status -> (string * float) list -> string) ->
  value:'a ->
  error:'a ->
  status:'d Nx.int32_t ->
  evaluations:'d Nx.int32_t ->
  facts:'d fact list ->
  unit ->
  ('a, 'd) t
(** [v ~fn ~settings ?spent ~fix ~value ~error ~status ~evaluations ~facts ()]
    is an answer of the solve [fn] run with [settings], such as
    ["tol rel 1e-06 abs 0"]. [fix st facts] is what to change for a lane that
    ended [st], given its facts' values. Without [spent], the report leaves out
    the budget and each lane's budget used is [0]. *)

val get : ('a, 'd) t -> 'a
val best : ('a, 'd) t -> 'a
val ok : ('a, 'd) t -> 'd Nx.bool_t
val is : status -> ('a, 'd) t -> 'd Nx.bool_t
val error : ('a, 'd) t -> 'a
val evaluations : ('a, 'd) t -> 'd Nx.int32_t
val ptree : ('a, 'd) Nx.Ptree.t -> (('a, 'd) t, 'd) Nx.Ptree.t
val pp : Format.formatter -> ('a, 'd) t -> unit
