(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Symbolic dimensions for shape-polymorphic tensor programs.

    A symbolic shape is an array of dimension expressions. Expressions can be
    constants, symbolic variables, or arithmetic combinations. *)

(** {1:types Types} *)

type var
(** The type for symbolic variables.

    Variables have unique identities, names, inclusive bounds, and mutable
    optional runtime values. *)

(** The type for dimension expressions. *)
type expr =
  | Const of int  (** Constant dimension value. *)
  | Var of var  (** Variable dimension. *)
  | Add of expr * expr  (** Sum of dimensions. *)
  | Mul of expr * expr  (** Product of dimensions. *)
  | Neg of expr  (** Negated dimension. *)

type dim = expr
(** The type for dimensions. *)

type t = dim array
(** The type for symbolic shapes.

    [t] is represented as an array. By convention, shape operations treat inputs
    as persistent values and return new arrays unless documented otherwise. *)

(** {1:constructors Construction} *)

val static : int -> dim
(** [static n] is a constant dimension [n].

    Raises [Invalid_argument] if [n < 0]. *)

val dynamic : string -> min:int -> max:int -> dim
(** [dynamic name ~min ~max] is [dim_of_var (var name ~min ~max)].

    Raises [Invalid_argument] if [min < 0] or [min > max]. *)

val var : string -> min:int -> max:int -> var
(** [var name ~min ~max] is a fresh variable.

    Variables are compared by identity; two calls are always distinct, even with
    identical [name], [min], and [max].

    Raises [Invalid_argument] if [min < 0] or [min > max]. *)

val dim_of_var : var -> dim
(** [dim_of_var v] is [Var v]. *)

val of_ints : int array -> t
(** [of_ints a] maps {!static} over [a].

    Raises [Invalid_argument] if any element of [a] is negative. *)

val of_list : int list -> t
(** [of_list l] is [of_ints (Array.of_list l)].

    Raises [Invalid_argument] if any element of [l] is negative. *)

(** {1:expr Expression combinators} *)

val add : dim -> dim -> dim
(** [add d0 d1] is [Add (d0, d1)]. *)

val mul : dim -> dim -> dim
(** [mul d0 d1] is [Mul (d0, d1)]. *)

val neg : dim -> dim
(** [neg d] is [Neg d]. *)

(** {1:binding Runtime binding and evaluation} *)

val bind : var -> int -> t -> unit
(** [bind v value shape] mutates [v] to [Some value] and propagates the same
    binding to all variables in [shape] that share [v]'s identity.

    Raises [Invalid_argument] if [value] is outside [v]'s bounds. *)

val eval : t -> int array option
(** [eval shape] evaluates all dimensions in [shape].

    Returns [Some dims] iff all variables referenced by [shape] are bound.
    Returns [None] if at least one variable is unbound. *)

val eval_dim : dim -> int option
(** [eval_dim d] evaluates [d] to [Some n] iff all variables in [d] are bound;
    otherwise [None]. *)

val partial_eval : t -> int option array
(** [partial_eval shape] evaluates each dimension independently.

    A result element is [Some n] for evaluable dimensions and [None] otherwise.
*)

val is_fully_bound : t -> bool
(** [is_fully_bound shape] is [true] iff every variable in [shape] is bound. *)

val numel : t -> int option
(** [numel shape] is the product of evaluated dimensions in [shape], if all
    dimensions are evaluable.

    [numel [||]] is [Some 1]. *)

(** {1:infer Inferred dimensions} *)

val infer : dim
(** [infer] is the distinguished dimension [Const (-1)] used by
    [resolve_reshape]. *)

val is_infer : dim -> bool
(** [is_infer d] is [true] iff [eval_dim d = Some (-1)]. *)

(** {1:reshape Reshape resolution} *)

val resolve_reshape : from_shape:t -> to_shape:t -> t option
(** [resolve_reshape ~from_shape ~to_shape] resolves [infer] dimensions in
    [to_shape] using [numel from_shape].

    Returns [None] if [from_shape] cannot be fully evaluated, or if a single
    inferred dimension cannot be computed as an integer size.

    Raises [Invalid_argument] if:
    - More than one [infer] appears in [to_shape].
    - A non-infer dimension in [to_shape] evaluates to [<= 0]. *)

(** {1:substitute Substitution} *)

val substitute : (var * int) list -> t -> t
(** [substitute bindings shape] replaces variables in [shape] by [Const] values
    according to [bindings].

    Variables not present in [bindings] are preserved. This does not mutate
    variable state. *)

(** {1:analysis Analysis} *)

val vars : t -> var list
(** [vars shape] is the list of distinct variables occurring in [shape].

    Distinctness is by variable identity. *)

val var_id : var -> int
(** [var_id v] is [v]'s unique identity. *)

val var_name : var -> string
(** [var_name v] is [v]'s display name. *)

val var_bounds : var -> int * int
(** [var_bounds v] is [v]'s inclusive bounds [(min, max)]. *)

val is_static : t -> bool
(** [is_static shape] is [true] iff [shape] has no variables. *)

val rank : t -> int
(** [rank shape] is [Array.length shape]. *)

(** {1:format Formatting and comparison} *)

val to_string : t -> string
(** [to_string shape] formats [shape] for inspection.

    Variables include their identity, and bound variables include their value.
*)

val equal : t -> t -> bool
(** [equal s0 s1] is structural equality on expressions.

    Variable nodes are equal iff they have the same variable identity.
    Algebraically equivalent but structurally different expressions are not
    equal. *)
