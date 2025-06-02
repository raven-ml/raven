(** Symbolic shapes for shape-polymorphic tensors.

    This module provides symbolic dimensions that can be bound at runtime,
    enabling dynamic shapes in compiled kernels and graph optimization. *)

type var
(** A symbolic variable representing a dimension *)

(** Expression type for dimension arithmetic *)
type expr =
  | Const of int
  | Var of var
  | Add of expr * expr
  | Mul of expr * expr
  | Neg of expr

type dim = expr
(** A dimension is an expression *)

type t = dim array
(** A shape is an array of dimensions *)

(** {2 Creation} *)

val static : int -> dim
(** [static n] creates a static dimension with value [n].

    @raise Invalid_argument if [n < 0] *)

val dynamic : string -> min:int -> max:int -> dim
(** [dynamic name ~min ~max] creates a dynamic dimension with bounds.

    @raise Invalid_argument if [min < 0] or [min > max] *)

val of_ints : int array -> t
(** [of_ints arr] creates a shape with all static dimensions. *)

val of_list : int list -> t
(** [of_list lst] creates a shape with all static dimensions. *)

(** {2 Dimension Operations} *)

val add : dim -> dim -> dim
(** [add d1 d2] creates a dimension expression [d1 + d2]. *)

val mul : dim -> dim -> dim
(** [mul d1 d2] creates a dimension expression [d1 * d2]. *)

val neg : dim -> dim
(** [neg d] creates a dimension expression [-d]. *)

(** {2 Runtime Binding} *)

val bind : string -> int -> t -> unit
(** [bind name value shape] binds a concrete value to all variables with [name]
    in [shape].

    @raise Invalid_argument if [value] outside variable's bounds *)

val eval : t -> int array option
(** [eval shape] returns concrete shape if all dimensions are bound. *)

val eval_dim : dim -> int option
(** [eval_dim dim] evaluates a single dimension. *)

val partial_eval : t -> int option array
(** [partial_eval shape] returns an array of evaluated dimensions, with None for
    unbound. *)

val is_fully_bound : t -> bool
(** [is_fully_bound shape] returns true if all symbolic dimensions are bound. *)

val numel : t -> int option
(** [numel shape] returns the total number of elements if shape is fully bound.
*)

(** {2 Special Values} *)

val infer : dim
(** Special dimension value representing "infer from context" (like -1 in NumPy
    reshape). *)

val is_infer : dim -> bool
(** [is_infer dim] returns true if dimension should be inferred. *)

(** {2 Shape Resolution} *)

val resolve_reshape : from_shape:t -> to_shape:t -> t option
(** [resolve_reshape ~from_shape ~to_shape] resolves a reshape operation,
    computing any [infer] dimensions in [to_shape] based on [from_shape]'s
    numel. Returns [None] if resolution is impossible. *)

val substitute : (string * int) list -> t -> t
(** [substitute bindings shape] substitutes variable bindings into a shape,
    replacing variables with their bound values. *)

(** {2 Analysis} *)

val vars : t -> var list
(** [vars shape] returns all unique symbolic variables in shape. *)

val is_static : t -> bool
(** [is_static shape] returns true if all dimensions are static. *)

val rank : t -> int
(** [rank shape] returns number of dimensions. *)

(** {2 Utilities} *)

val to_string : t -> string
(** [to_string shape] returns human-readable representation. *)

val equal : t -> t -> bool
(** [equal s1 s2] compares shapes structurally. *)
