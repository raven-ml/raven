(** Action and observation spaces for environment interfaces.

    Spaces define the valid observations and actions for an environment. They
    specify shapes, constraints, and provide methods to validate, sample, and
    serialize values. Each space type corresponds to common RL scenarios:
    discrete choices, continuous vectors, multi-dimensional arrays, and
    composite structures.

    {1 Space Types}

    - {!Discrete}: Integer choices from a finite set
    - {!Box}: Continuous vectors with bounded ranges
    - {!Multi_binary}: Binary vectors for multi-label scenarios
    - {!Multi_discrete}: Multiple discrete choices
    - {!Tuple}: Fixed-length heterogeneous sequences
    - {!Dict}: Named fields with different space types
    - {!Sequence}: Variable-length homogeneous sequences
    - {!Text}: String spaces for textual observations or actions

    {1 Usage}

    Create a discrete action space and sample from it:
    {[
      let action_space = Space.Discrete.create 4 in
      let action = Space.sample action_space
    ]}

    Create a continuous observation space:
    {[
      let obs_space = Space.Box.create
        ~low:[|-1.0; -1.0|]
        ~high:[|1.0; 1.0|]
      in
      let is_valid = Space.contains obs_space observation
    ]}

    Composite spaces for structured data:
    {[
      let space =
        Space.Dict.create
          [
            ( "position",
              Space.Pack (Box.create ~low:[| -10.0 |] ~high:[| 10.0 |]) );
            ("velocity", Space.Pack (Box.create ~low:[| -1.0 |] ~high:[| 1.0 |]));
          ]
    ]} *)

module Value : sig
  type t =
    | Int of int
    | Float of float
    | Bool of bool
    | Int_array of int array
    | Float_array of float array
    | Bool_array of bool array
    | List of t list
    | Tuple of t list
    | Dict of (string * t) list
    | String of string
        (** Universal value type for packing/unpacking space elements.

            Provides a common representation for serialization and type-erased
            manipulation of space values. *)

  val pp : Format.formatter -> t -> unit
  (** [pp formatter value] pretty-prints [value]. *)

  val to_string : t -> string
  (** [to_string value] converts [value] to a string representation. *)
end

type 'a t = {
  shape : int array option;  (** Dimensionality, if applicable *)
  contains : 'a -> bool;  (** Validates whether a value belongs to this space *)
  sample : ?rng:Rune.Rng.key -> unit -> 'a;
      (** Generates a random valid value *)
  pack : 'a -> Value.t;  (** Converts to universal value representation *)
  unpack : Value.t -> ('a, string) result;
      (** Parses from universal representation *)
}
(** Typed space representing valid values of type ['a].

    Spaces encapsulate validation, sampling, and serialization logic for a type.
*)

type packed =
  | Pack : 'a t -> packed
      (** Type-erased space for heterogeneous collections. *)

val shape : 'a t -> int array option
(** [shape space] returns the shape of [space], if defined.

    Shape represents dimensionality for array-like spaces. Returns [None] for
    scalar or variable-length spaces. *)

val contains : 'a t -> 'a -> bool
(** [contains space value] checks whether [value] is valid in [space].

    Returns [true] if [value] satisfies all constraints of [space]. *)

val sample : ?rng:Rune.Rng.key -> 'a t -> 'a
(** [sample ~rng space] generates a random valid value from [space].

    If [rng] is not provided, uses a default RNG. *)

val pack : 'a t -> 'a -> Value.t
(** [pack space value] converts [value] to a universal representation. *)

val unpack : 'a t -> Value.t -> ('a, string) result
(** [unpack space value] parses [value] from universal representation.

    Returns [Ok v] if [value] can be converted to a valid element of [space],
    [Error msg] otherwise. *)

module Discrete : sig
  type element = (int32, Rune.int32_elt) Rune.t
  (** Discrete action represented as a scalar tensor. *)

  val create : ?start:int -> int -> element t
  (** [create ~start n] creates a discrete space with [n] choices.

      Valid values are integers in the range \[start, start + n). If [start] is
      omitted, defaults to 0, producing the range \[0, n).

      Common use: Action spaces for environments with discrete actions (e.g.,
      move left, move right, jump).

      @raise Invalid_argument if [n <= 0]. *)
end

module Box : sig
  type element = (float, Rune.float32_elt) Rune.t
  (** Continuous vector represented as a float tensor. *)

  val create : low:float array -> high:float array -> element t
  (** [create ~low ~high] creates a continuous space with bounded ranges.

      Valid values are tensors where each element [i] satisfies
      [low.(i) <= x.(i) <= high.(i)]. Arrays [low] and [high] must have the same
      length, defining the space dimensionality.

      Common use: Observation spaces for continuous state (e.g., position,
      velocity) or continuous action spaces (e.g., torque, steering angle).

      @raise Invalid_argument
        if [low] and [high] have different lengths or if any
        [low.(i) > high.(i)]. *)
end

module Multi_binary : sig
  type element = (int32, Rune.int32_elt) Rune.t
  (** Binary vector for multi-label scenarios. *)

  val create : int -> element t
  (** [create n] creates a binary vector space of length [n].

      Valid values are tensors with [n] elements, each 0 or 1. Represents
      independent binary choices (e.g., which objects are present in an image).

      @raise Invalid_argument if [n <= 0]. *)
end

module Multi_discrete : sig
  type element = (int32, Rune.int32_elt) Rune.t
  (** Multiple discrete choices, each with different cardinality. *)

  val create : int array -> element t
  (** [create nvec] creates a multi-discrete space.

      Valid values are tensors where element [i] is in \[0, nvec.(i)). Each
      dimension represents an independent discrete choice with its own number of
      options.

      Common use: Environments with multiple independent discrete actions (e.g.,
      [character_move, weapon_select, jump_or_not]).

      @raise Invalid_argument if any [nvec.(i) <= 0]. *)
end

module Tuple : sig
  type element = Value.t list
  (** Fixed-length heterogeneous sequence. *)

  val create : packed list -> element t
  (** [create spaces] creates a tuple space from a list of subspaces.

      Valid values are lists where element [i] belongs to [spaces.(i)]. All
      tuples have fixed length equal to the number of subspaces.

      Common use: Observations combining different data types (e.g.,
      [image, scalar_speed]). *)
end

module Dict : sig
  type element = (string * Value.t) list
  (** Named fields with different space types. *)

  val create : (string * packed) list -> element t
  (** [create fields] creates a dictionary space with named fields.

      Valid values are association lists where each key-value pair [(k, v)] has
      [v] belonging to the space associated with key [k] in [fields].

      Common use: Structured observations with named components (e.g.,
      [{"position": box, "inventory": multi_binary}]). *)
end

module Sequence : sig
  type 'a element = 'a list
  (** Variable-length homogeneous sequence. *)

  val create : ?min_length:int -> ?max_length:int -> 'a t -> 'a element t
  (** [create ~min_length ~max_length subspace] creates a sequence space.

      Valid values are lists of elements from [subspace], with length
      constraints:
      - If [min_length] is provided, list length must be >= [min_length]
      - If [max_length] is provided, list length must be <= [max_length]

      Common use: Variable-length observations (e.g., lists of detected objects,
      sequences of variable horizon length). *)
end

module Text : sig
  type element = string
  (** String space for textual observations or actions. *)

  val create : ?charset:string -> ?max_length:int -> unit -> element t
  (** [create ~charset ~max_length ()] creates a text space.

      Valid values are strings satisfying:
      - All characters appear in [charset] (if provided)
      - Length <= [max_length] (if provided)

      Common use: Text-based environments, language model actions, or string
      commands. *)
end
