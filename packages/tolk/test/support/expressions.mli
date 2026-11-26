(** Random expressions for the laws of simplification ({!Simplification}).

    Integer expressions are index arithmetic as kernels compute it, weak or at a
    committed width, over three bounded variables [a], [b] and [c], with
    divisions by constants and by two variables [d] and [e] whose bounds exclude
    [0]. Float expressions are float32 arithmetic over the scalar parameters of
    slots [0] to [2] and {!Simplification.specials}. *)

open Windtrap
open Tolk

(** {1:integers Integer expressions} *)

(** The type for integer expressions. *)
type t =
  | Var of int  (** The [i]th variable. *)
  | Divisor of int  (** The [i]th divisor variable. *)
  | Const of int
  | Neg of t
  | Add of t * t
  | Mul of t * int
  | Mul_var of t * t
  | Div of t * int
  | Mod of t * int
  | Div_var of t * int
  | Mod_var of t * int
  | Max of t * t
  | Select of t * t * t  (** [Select (a, c, b)] is [where (a < c) b a]. *)

type scenario = {
  dtype : Dtype.t;  (** The type of the variables and divisors. *)
  vars : (int * int) list;  (** The bounds of [a], [b] and [c]. *)
  divisors : (int * int) list;  (** The bounds of [d] and [e]. *)
  expr : t;
}
(** The type for an expression with the type and bounds of its leaves. *)

val pp_scenario : Format.formatter -> scenario -> unit
(** [pp_scenario] formats a scenario as its type, its bounds and its expression.
*)

val node : scenario -> Ops.t
(** [node s] is [s]'s expression over variables of [s.dtype] and [s]'s bounds.
*)

val constants : scenario -> Ops.t
(** [constants s] is [s]'s expression over committed constants of [s.dtype] in
    place of its leaves: a variable's least value, a divisor's greatest. *)

val scenario_of : Dtype.t Gen.t -> scenario Gen.t
(** [scenario_of dtype] draws an expression of depth at most [4] over leaves of
    a type [dtype] draws, variables within \[[-10];[30]\] (\[[0];[30]\]
    unsigned) and divisors within \[[1];[12]\] or, signed, \[[-12];[-1]\]. *)

val scenario : scenario Gen.t
(** [scenario] is {!scenario_of} over {!Dtype.Weak_int} three times in five, and
    otherwise over {!Dtype.Int8}, {!Dtype.Uint8}, {!Dtype.Int16}, {!Dtype.Int32}
    or {!Dtype.Uint32}, each as likely. *)

val weak_scenario : scenario Gen.t
(** [weak_scenario] is {!scenario_of} over {!Dtype.Weak_int}. *)

val committed_scenario : scenario Gen.t
(** [committed_scenario] is {!scenario_of} over the committed types of
    {!scenario}. *)

(** {1:floats Float expressions} *)

(** The type for float32 expressions. *)
type float_expr =
  | Param of int  (** The scalar float32 parameter of slot [i]. *)
  | Special of int
      (** The [i]th of {!Simplification.specials}, as a constant. *)
  | Fneg of float_expr
  | Fadd of float_expr * float_expr
  | Fsub of float_expr * float_expr
  | Fmul of float_expr * float_expr
  | Fdiv of float_expr * float_expr
  | Fmax of float_expr * float_expr
  | Fselect of float_expr * float_expr * float_expr
      (** [Fselect (a, c, b)] is [where (a < c) b a]. *)

val float_node : float_expr -> Ops.t
(** [float_node e] is [e]'s graph. *)

val float_scenario : float_expr Gen.t
(** [float_scenario] draws a float expression of depth at most [4]. *)
