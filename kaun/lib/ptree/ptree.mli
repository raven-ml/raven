(** Parameter tree data structures and operations *)

type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t
(** Convenience type alias for float tensors *)

(** Parameter tree type - recursive structure for model parameters *)
type ('layout, 'dev) t =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) t list
  | Record of (string * ('layout, 'dev) t) list

(** {2 Tree Operations} *)

val map :
  (('layout, 'dev) tensor -> ('layout, 'dev) tensor) ->
  ('layout, 'dev) t ->
  ('layout, 'dev) t
(** [map f tree] applies function [f] to all tensors in the tree *)

val map2 :
  (('layout, 'dev) tensor -> ('layout, 'dev) tensor -> ('layout, 'dev) tensor) ->
  ('layout, 'dev) t ->
  ('layout, 'dev) t ->
  ('layout, 'dev) t
(** [map2 f tree1 tree2] applies binary function [f] to corresponding tensors in
    both trees. Raises [Invalid_argument] if trees have different structures. *)

val iter : (('layout, 'dev) tensor -> unit) -> ('layout, 'dev) t -> unit
(** [iter f tree] applies function [f] to all tensors in the tree for side
    effects *)

val fold : ('a -> ('layout, 'dev) tensor -> 'a) -> 'a -> ('layout, 'dev) t -> 'a
(** [fold f init tree] folds function [f] over all tensors in the tree *)

val equal_structure : ('layout, 'dev) t -> ('layout, 'dev) t -> bool
(** [equal_structure tree1 tree2] returns true if both trees have the same
    structure (ignoring tensor values) *)

(** {2 Tree Construction} *)

val zeros_like : ('layout, 'dev) t -> ('layout, 'dev) t
(** [zeros_like tree] creates a new tree with same structure but all tensors
    filled with zeros *)

val ones_like : ('layout, 'dev) t -> ('layout, 'dev) t
(** [ones_like tree] creates a new tree with same structure but all tensors
    filled with ones *)

val copy : ('layout, 'dev) t -> ('layout, 'dev) t
(** [copy tree] creates a deep copy of the tree *)

(** {2 Tree Inspection} *)

val count_tensors : ('layout, 'dev) t -> int
(** [count_tensors tree] returns the number of tensors in the tree *)

val count_parameters : ('layout, 'dev) t -> int
(** [count_parameters tree] returns the total number of scalar parameters across
    all tensors *)

val to_flat_list : ('layout, 'dev) t -> ('layout, 'dev) tensor list
(** [to_flat_list tree] extracts all tensors from the tree into a flat list *)

val from_flat_list :
  ('layout, 'dev) t -> ('layout, 'dev) tensor list -> ('layout, 'dev) t
(** [from_flat_list template tensors] reconstructs a tree using [template]
    structure and values from [tensors] list. Raises [Invalid_argument] if list
    length doesn't match. *)

(** {2 Arithmetic Operations} *)

val add : ('layout, 'dev) t -> ('layout, 'dev) t -> ('layout, 'dev) t
(** [add tree1 tree2] performs element-wise addition of corresponding tensors *)

val sub : ('layout, 'dev) t -> ('layout, 'dev) t -> ('layout, 'dev) t
(** [sub tree1 tree2] performs element-wise subtraction of corresponding tensors
*)

val mul : ('layout, 'dev) t -> ('layout, 'dev) t -> ('layout, 'dev) t
(** [mul tree1 tree2] performs element-wise multiplication of corresponding
    tensors *)

val div : ('layout, 'dev) t -> ('layout, 'dev) t -> ('layout, 'dev) t
(** [div tree1 tree2] performs element-wise division of corresponding tensors *)

val scale : float -> ('layout, 'dev) t -> ('layout, 'dev) t
(** [scale alpha tree] multiplies all tensors in the tree by scalar [alpha] *)

val neg : ('layout, 'dev) t -> ('layout, 'dev) t
(** [neg tree] negates all tensors in the tree *)

(** {2 Utility Functions} *)

val pp : Format.formatter -> ('layout, 'dev) t -> unit
(** Pretty printer for parameter trees *)

val to_string : ('layout, 'dev) t -> string
(** [to_string tree] returns a string representation of the tree structure *)
