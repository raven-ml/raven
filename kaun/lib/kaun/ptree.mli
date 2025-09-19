(** Parameter tree data structures and operations *)

module Record : module type of Map.Make (String)

(** Parameter tree type - recursive structure for model parameters *)
type 'layout t =
  | Tensor of (float, 'layout) Rune.t
  | List of 'layout t list
  | Record of 'layout t Record.t

type mask_tree =
  | Mask_tensor of bool
  | Mask_list of mask_tree list
  | Mask_record of mask_tree Record.t

(** {2 Builders} *)

val tensor : (float, 'layout) Rune.t -> 'layout t
(** Create a leaf tensor node *)

val list_of : 'layout t list -> 'layout t
(** Create a list node *)

val record_of : (string * 'layout t) list -> 'layout t
(** Create a record node from bindings. Raises if duplicate keys. *)

(** {2 Accessors} *)

val get_tensor : 'layout t -> (float, 'layout) Rune.t option
(** [get_tensor tree] returns [Some tensor] if tree is a Tensor, [None]
    otherwise *)

val get_list : 'layout t -> 'layout t list option
(** [get_list tree] returns [Some list] if tree is a List, [None] otherwise *)

val get_record : 'layout t -> 'layout t Record.t option
(** [get_record tree] returns [Some record] if tree is a Record, [None]
    otherwise *)

val find_in_record : string -> 'layout t -> 'layout t option
(** [find_in_record key tree] returns [Some value] if tree is a Record
    containing key, [None] otherwise *)

(** {2 Tree Operations} *)

val map :
  ((float, 'layout) Rune.t -> (float, 'layout) Rune.t) -> 'layout t -> 'layout t
(** [map f tree] applies function [f] to all tensors in the tree *)

val map2 :
  ((float, 'layout) Rune.t ->
  (float, 'layout) Rune.t ->
  (float, 'layout) Rune.t) ->
  'layout t ->
  'layout t ->
  'layout t
(** [map2 f tree1 tree2] applies binary function [f] to corresponding tensors in
    both trees. Raises [Invalid_argument] if trees have different structures. *)

val zip :
  ((float, 'layout) Rune.t -> (float, 'layout) Rune.t -> 'a) ->
  'layout t ->
  'layout t ->
  'a list
(** [zip f tree1 tree2] applies [f] to pairs of corresponding tensors, returning
    a flat list of results. Useful for pairing without building a new tree. *)

val iter : ((float, 'layout) Rune.t -> unit) -> 'layout t -> unit
(** [iter f tree] applies function [f] to all tensors in the tree for side
    effects *)

val fold : ('a -> (float, 'layout) Rune.t -> 'a) -> 'a -> 'layout t -> 'a
(** [fold f init tree] folds function [f] over all tensors in the tree *)

val equal_structure : 'layout t -> 'layout t -> bool
(** [equal_structure tree1 tree2] returns true if both trees have the same
    structure (ignoring tensor values) *)

val filter : ((float, 'layout) Rune.t -> bool) -> 'layout t -> 'layout t
(** [filter pred tree] replaces tensors where [pred] is false with zeros_like *)

val apply_mask : mask_tree -> 'layout t -> 'layout t
(** [apply_mask mask tree] zeros out tensors where mask is false. Raises if
    structures differ. *)

(** {2 Tree Construction} *)

val zeros_like : 'layout t -> 'layout t
(** [zeros_like tree] creates a new tree with same structure but all tensors
    filled with zeros *)

val ones_like : 'layout t -> 'layout t
(** [ones_like tree] creates a new tree with same structure but all tensors
    filled with ones *)

val copy : 'layout t -> 'layout t
(** [copy tree] creates a deep copy of the tree *)

(** {2 Tree Inspection} *)

val count_tensors : 'layout t -> int
(** [count_tensors tree] returns the number of tensors in the tree *)

val count_parameters : 'layout t -> int
(** [count_parameters tree] returns the total number of scalar parameters across
    all tensors *)

val flatten :
  'layout t ->
  (float, 'layout) Rune.t list * ((float, 'layout) Rune.t list -> 'layout t)
(** [flatten tree] returns flat tensor list and a rebuild function *)

(** {2 Arithmetic Operations} *)

val add : 'layout t -> 'layout t -> 'layout t
(** [add tree1 tree2] performs element-wise addition of corresponding tensors *)

val sub : 'layout t -> 'layout t -> 'layout t
(** [sub tree1 tree2] performs element-wise subtraction of corresponding tensors
*)

val mul : 'layout t -> 'layout t -> 'layout t
(** [mul tree1 tree2] performs element-wise multiplication of corresponding
    tensors *)

val div : 'layout t -> 'layout t -> 'layout t
(** [div tree1 tree2] performs element-wise division of corresponding tensors *)

val scale : float -> 'layout t -> 'layout t
(** [scale alpha tree] multiplies all tensors in the tree by scalar [alpha] *)

val neg : 'layout t -> 'layout t
(** [neg tree] negates all tensors in the tree *)

(** {2 Utility Functions} *)

val pp : Format.formatter -> 'layout t -> unit
(** Pretty printer for parameter trees *)

val to_string : 'layout t -> string
(** [to_string tree] returns a string representation of the tree structure *)

(** {2 Path-based Flattening} *)

val flatten_with_paths : 'layout t -> (string * (float, 'layout) Rune.t) list
(** [flatten_with_paths tree] returns a list of (path, tensor) pairs where paths
    use dot notation for records (e.g., "layer1.weight") and bracket notation
    for lists (e.g., "layers[0]"). *)

val unflatten_from_paths : (string * (float, 'layout) Rune.t) list -> 'layout t
(** [unflatten_from_paths pairs] reconstructs a parameter tree from path-tensor
    pairs. Raises [Invalid_argument] if paths are malformed or inconsistent. *)

(** {2 Path-based Access} *)

val get_by_path : string -> 'layout t -> 'layout t
(** [get_by_path path tree] retrieves the subtree at the given path. Path uses
    dot notation for records and bracket notation for lists. Examples:
    "encoder.weight", "layers[0].attention.q_proj"
    @raise Invalid_argument if the path doesn't exist or is malformed *)

val set_by_path : string -> 'layout t -> 'layout t -> 'layout t
(** [set_by_path path value tree] returns a new tree with the value at path
    replaced. Creates intermediate records if they don't exist.
    @raise Invalid_argument
      if the path is malformed or incompatible with tree structure *)

val validate_tree : ?path:string -> 'layout t -> unit
(** [validate_tree ?path tree] checks for structural issues:
    - Empty keys
    - Duplicate keys within records
    - Invalid characters in keys (. [ ])
    - Warnings for empty lists/records

    @param path Starting path for error messages (default: "root")
    @raise Failure if validation fails *)

(** {2 Enhanced Introspection} *)

val list_named_params : 'layout t -> (string * string * int) list
(** [list_named_params tree] returns a list of (path, shape_string,
    num_elements). Shape strings are formatted as "2×3×4" for tensors or
    "scalar" for 0-d tensors. Useful for inspecting model architecture. *)

val find_params_by_pattern :
  string -> 'layout t -> (string * (float, 'layout) Rune.t) list
(** [find_params_by_pattern pattern tree] returns all params whose paths match
    the regex pattern. Example: find_params_by_pattern ".*weight$" finds all
    weight tensors. *)

val get_param_stats : 'layout t -> int * (string * int) list
(** [get_param_stats tree] returns (total_params, [(group_name, count), ...]).
    Groups parameters by top-level key for a summary view. Example: (1000000,
    [("encoder", 800000); ("decoder", 200000)]) *)
