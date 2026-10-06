(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Answers of solves.

    A solve returns its answer with a status per lane: one lane per element for
    an elementwise family ({!Root}, {!Quad}'s one-dimensional solves), one for a
    structured family, and one per lane of each {!Rune.val-vmap} around it.
    {!get} reads an answer that converged everywhere and raises otherwise;
    {!best} and {!ok} read every lane.

    {[
    let s = Root.bracket ~tol f ~lo ~hi in
    let ok = Solution.ok s in
    Nx.where ok (Solution.best s) (Nx.zeros_like lo)
    ]} *)

type 'a t = 'a Answer.t
(** The type for answers of type ['a]. *)

(** The type for a lane's outcome. *)
type status = Answer.status =
  | Converged  (** The error estimate met the tolerance. *)
  | Budget_spent  (** The budget ran out first. *)
  | Not_bracketed  (** The ends of a bracket have one sign. *)
  | Not_finite  (** A value of the problem's function is not finite. *)
  | Stalled
      (** The search stopped moving without meeting the tolerance: its steps
          fell below the resolution of the floats, or the problem has no answer
          the method can reach, such as a pole in a bracket. *)

val get : 'a t -> 'a
(** [get s] is the answer if every lane converged.

    Raises [Failure] with the report of the first lane in C order that did not
    converge: the entry point, the lane, its status, the solve's settings, the
    lane's data, the budget it used, what to change, and how many other elements
    of its problem converged. Inside {!Rune.val-jit}, [get] returns {!best} and
    the compiled call raises when it returns. *)

val best : 'a t -> 'a
(** [best s] is the estimate in every lane. A lane that did not converge holds
    its last finite estimate, and its derivative is zero. *)

val ok : 'a t -> (bool, Nx.bool_elt) Nx.t
(** [ok s] is [is Converged s]. *)

val is : status -> 'a t -> (bool, Nx.bool_elt) Nx.t
(** [is st s] is [true] in each lane whose status is [st], of the lanes' shape.
*)

val error : 'a t -> 'a
(** [error s] is each lane's error estimate, of the answer's structure, as its
    solve defines it: an estimate, never a bound. *)

val evaluations : 'a t -> (int32, Nx.int32_elt) Nx.t
(** [evaluations s] is each lane's count of the points at which the problem's
    function was evaluated, of the lanes' shape. *)

val ptree : 'a Nx.Ptree.t -> 'a t Nx.Ptree.t
(** [ptree s] is the structure of answers of structure [s], so a compiled
    function returns one: the answer at [value], the error at [error], the
    statuses at [status], the counts at [evaluations], the budget used at [used]
    and the report's data under [report]. *)

val pp : Format.formatter -> 'a t -> unit
(** [pp ppf s] formats the count of lanes in each status and the report {!get}
    would raise for the first lane that did not converge. It reads the statuses,
    so it runs eagerly. *)
