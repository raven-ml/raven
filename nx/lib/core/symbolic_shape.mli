(** Symbolic shapes for shape-polymorphic tensors.

    This module provides symbolic dimensions that can be bound at runtime,
    enabling dynamic shapes in compiled kernels and graph optimization. *)

type var
(** A symbolic variable representing a dimension *)

(** A dimension can be either static (known) or dynamic (unknown) *)
type dim = Static of int | Dynamic of var

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

(** {2 Runtime Binding} *)

val bind : var -> int -> unit
(** [bind var value] binds a concrete value to a dynamic dimension.

    @raise Invalid_argument if [value] outside variable's bounds *)

val eval : t -> int array option
(** [eval shape] returns concrete shape if all dimensions are bound. *)

val eval_dim : dim -> int option
(** [eval_dim dim] evaluates a single dimension. *)

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
