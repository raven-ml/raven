(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Linear systems: the implementation of {!Jera.Linear}, with the solvers' runs
    on vectors that the derivatives and the Newton-type steps of {!System},
    {!Minimize} and {!Ode} solve with.

    A run solves [apply u = r] on vectors of [n] unknowns. On a system it cannot
    solve, its residual misses its bound or one of its outcomes holds, which
    {!failed} reports. *)

type 'x t

val dense : 'x t
val banded : width:int -> 'x t
val cg : rel:float -> budget:int -> precondition:('x -> 'x) -> 'x t

val gmres :
  restart:int -> rel:float -> budget:int -> precondition:('x -> 'x) -> 'x t

val solve :
  ('x, 'd) Nx.Ptree.t -> 'x t -> ('x -> 'x) -> 'x -> ('x, 'd) Solution.t

val name : 'x t -> string
(** [name s] is [s] in a report's settings: ["dense"], ["banded, width w"],
    ["cg"] or ["gmres"]. *)

(** {1:vectors Vectors} *)

(** The type for the float tensors of values of a structure as one vector of
    their dtype ['q]. [ravel v] is [v]'s vector, of [size] elements, and
    [unravel u] is the value whose float tensors are read from [u] in walk order
    and whose other tensors are those of the value the space was made from. *)
type ('x, 'd) space =
  | Space : {
      dtype : (float, 'q) Nx.dtype;
      size : int;
      ravel : 'x -> (float, 'q, 'd) Nx.t;
      unravel : (float, 'q, 'd) Nx.t -> 'x;
    }
      -> ('x, 'd) space

val is_float : 'd Nx.packed -> bool
(** [is_float p] is [true] iff [p] is a float tensor. *)

val space : string -> string -> ('x, 'd) Nx.Ptree.t -> 'x -> ('x, 'd) space
(** [space fn what x like] is the space of the float tensors of [like], of
    structure [x]. A value with no float tensor has a space of float64 and size
    [0].

    Raises [Invalid_argument] naming [fn] and [what] if the float tensors of
    [like] differ in dtype. *)

val checked : string -> string -> ('x, 'd) Nx.Ptree.t -> ('x -> 'x) -> 'x -> 'x
(** [checked fn what x a u] is [a u].

    Raises [Invalid_argument] naming [fn] and [what] if [a u] has another
    structure, or a leaf of another dtype or shape, than [u]. *)

(** {1:runs Runs} *)

type ('q, 'd) run = {
  u : (float, 'q, 'd) Nx.t;  (** The solution. *)
  residual : (float, 'q, 'd) Nx.t;
      (** [apply u − r]. [cg] and the direct solvers apply [apply] once more to
          compute it; [gmres] keeps it from its last cycle, and is [−r] when no
          cycle ran, as when [r] is zero or not finite. *)
  bound : (float, 'q, 'd) Nx.t;  (** The bound the residual's norm must meet. *)
  outcomes : ('d Nx.bool_t * Solution.status) list;
      (** The conditions that end a lane before the check, in order, each with
          the status it ends the lane with. *)
  spent : 'd Solution.spent option;
      (** The budget spent, for a solver with a budget. *)
  applications : 'd Nx.int32_t;  (** The count of applications of [apply]. *)
  facts : 'd Solution.fact list;  (** The report's facts. *)
}
(** The type for a solver's run on vectors of dtype ['q]. *)

val run :
  'x t ->
  (float, 'q) Nx.dtype ->
  int ->
  ((float, 'q, 'd) Nx.t -> (float, 'q, 'd) Nx.t) ->
  ((float, 'q, 'd) Nx.t -> (float, 'q, 'd) Nx.t) ->
  (float, 'q, 'd) Nx.t ->
  ('q, 'd) run
(** [run s dtype n apply precondition r] is the run of [s] on [apply u = r], [n]
    unknowns, with [precondition] on vectors. *)

val prepare :
  'x t ->
  'd Nx.Placement.t ->
  (float, 'q) Nx.dtype ->
  int ->
  ((float, 'q, 'd) Nx.t -> (float, 'q, 'd) Nx.t) ->
  ((float, 'q, 'd) Nx.t -> (float, 'q, 'd) Nx.t) ->
  (float, 'q, 'd) Nx.t ->
  ('q, 'd) run
(** [prepare s at dtype n apply precondition] is
    [run s dtype n apply precondition], prepared for many right-hand sides at
    [at]. Under {!dense} it materialises and factors the matrix once, so each
    right-hand side costs the substitutions and the check. *)

val preconditioner :
  string -> ('x, 'd) Nx.Ptree.t -> 'x t -> ('x -> 'v) -> ('v -> 'x) -> 'v -> 'v
(** [preconditioner fn x s ravel unravel] is [s]'s preconditioner on the vectors
    of [ravel] and [unravel]: the identity under {!dense} and {!banded}. The
    preconditioner raises as {!checked} does, naming [fn] and ["precondition"].
*)

val failed : ('q, 'd) run -> 'd Nx.bool_t
(** [failed r] is [true] if the norm of [r]'s residual does not meet its bound,
    as a NaN norm does not, or if a condition of its outcomes holds. *)

val derivative : string -> ('x, 'd) Nx.Ptree.t -> 'x t -> ('x -> 'x) -> 'x -> 'x
(** [derivative fn x s op b] is the [v] with [op v = b], solved by [s]: the
    linear solve that {!Rune.root} takes. It is [b] for a [b] of no float
    element, and every element is NaN where [s] fails ({!failed}).

    Raises [Invalid_argument] naming [fn] as {!space} and {!checked} do. *)
