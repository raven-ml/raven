(** Parameter tree data structures and operations *)

type ('layout, 'dev) tensor = (float, 'layout, 'dev) Rune.t
(** Convenience type alias for float tensors *)

module Record : module type of Map.Make (String)

(** Parameter tree type - recursive structure for model parameters *)
type ('layout, 'dev) t =
  | Tensor of ('layout, 'dev) tensor
  | List of ('layout, 'dev) t list
  | Record of ('layout, 'dev) t Record.t

type mask_tree =
  | Mask_tensor of bool
  | Mask_list of mask_tree list
  | Mask_record of mask_tree Record.t

(** {2 Builders} *)

val tensor : ('layout, 'dev) tensor -> ('layout, 'dev) t
(** Create a leaf tensor node *)

val list_of : ('layout, 'dev) t list -> ('layout, 'dev) t
(** Create a list node *)

val record_of : (string * ('layout, 'dev) t) list -> ('layout, 'dev) t
(** Create a record node from bindings. Raises if duplicate keys. *)

(** {2 Accessors} *)

val get_tensor : ('layout, 'dev) t -> ('layout, 'dev) tensor option
(** [get_tensor tree] returns [Some tensor] if tree is a Tensor, [None]
    otherwise *)

val get_list : ('layout, 'dev) t -> ('layout, 'dev) t list option
(** [get_list tree] returns [Some list] if tree is a List, [None] otherwise *)

val get_record : ('layout, 'dev) t -> ('layout, 'dev) t Record.t option
(** [get_record tree] returns [Some record] if tree is a Record, [None]
    otherwise *)

val find_in_record : string -> ('layout, 'dev) t -> ('layout, 'dev) t option
(** [find_in_record key tree] returns [Some value] if tree is a Record
    containing key, [None] otherwise *)

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

val zip :
  (('layout, 'dev) tensor -> ('layout, 'dev) tensor -> 'a) ->
  ('layout, 'dev) t ->
  ('layout, 'dev) t ->
  'a list
(** [zip f tree1 tree2] applies [f] to pairs of corresponding tensors, returning
    a flat list of results. Useful for pairing without building a new tree. *)

val iter : (('layout, 'dev) tensor -> unit) -> ('layout, 'dev) t -> unit
(** [iter f tree] applies function [f] to all tensors in the tree for side
    effects *)

val fold : ('a -> ('layout, 'dev) tensor -> 'a) -> 'a -> ('layout, 'dev) t -> 'a
(** [fold f init tree] folds function [f] over all tensors in the tree *)

val equal_structure : ('layout, 'dev) t -> ('layout, 'dev) t -> bool
(** [equal_structure tree1 tree2] returns true if both trees have the same
    structure (ignoring tensor values) *)

val filter :
  (('layout, 'dev) tensor -> bool) -> ('layout, 'dev) t -> ('layout, 'dev) t
(** [filter pred tree] replaces tensors where [pred] is false with zeros_like *)

val apply_mask : mask_tree -> ('layout, 'dev) t -> ('layout, 'dev) t
(** [apply_mask mask tree] zeros out tensors where mask is false. Raises if
    structures differ. *)

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

val flatten_with_rebuild :
  ('layout, 'dev) t ->
  ('layout, 'dev) tensor list
  * (('layout, 'dev) tensor list -> ('layout, 'dev) t)
(** [flatten_with_rebuild tree] returns flat tensor list and a rebuild function
*)

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

(** {2 Path-based Flattening} *)

val flatten_with_paths :
  ('layout, 'dev) t -> (string * ('layout, 'dev) tensor) list
(** [flatten_with_paths tree] returns a list of (path, tensor) pairs where paths
    use dot notation for records (e.g., "layer1.weight") and bracket notation
    for lists (e.g., "layers[0]"). *)

val unflatten_from_paths :
  (string * ('layout, 'dev) tensor) list -> ('layout, 'dev) t
(** [unflatten_from_paths pairs] reconstructs a parameter tree from path-tensor
    pairs. Raises [Invalid_argument] if paths are malformed or inconsistent. *)
