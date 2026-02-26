(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Observation and action spaces.

    Spaces define valid observations and actions for reinforcement learning
    environments. They specify shapes, constraints, and provide methods to
    validate, sample, and serialize values.

    Each space type corresponds to a common RL scenario: discrete choices,
    continuous vectors, binary indicators, composite structures, and
    variable-length sequences. *)

(** {1:spec Structural description} *)

(** Structural description of a space. Two spaces are compatible when their
    specs are equal. *)
type spec =
  | Discrete of { start : int; n : int }
      (** Integer choices in \[[start]; [start + n - 1]\]. *)
  | Box of { low : float array; high : float array }
      (** Continuous vector bounded per dimension. *)
  | Multi_binary of { n : int }  (** Binary vector of length [n]. *)
  | Multi_discrete of { nvec : int array }
      (** Multiple discrete axes with per-axis cardinalities. *)
  | Tuple of spec list  (** Fixed-length heterogeneous sequence. *)
  | Dict of (string * spec) list  (** Named fields with different types. *)
  | Sequence of { min_length : int; max_length : int option; base : spec }
      (** Variable-length homogeneous sequence. *)
  | Text of { charset : string; max_length : int }
      (** Character strings from a fixed alphabet. *)

val equal_spec : spec -> spec -> bool
(** [equal_spec a b] is [true] iff [a] and [b] describe structurally identical
    spaces. *)

(** {1:spaces Spaces} *)

type 'a t
(** The type for spaces over values of type ['a]. A space is self-contained: all
    bounds, constraints, and serialization logic are stored in the value itself.
*)

type packed =
  | Pack : 'a t -> packed
      (** Type-erased space for heterogeneous collections. *)

(** {1:ops Operations} *)

val spec : 'a t -> spec
(** [spec s] is the structural description of [s]. *)

val shape : 'a t -> int array option
(** [shape s] is the dimensionality of [s], if defined. [None] for scalar or
    variable-length spaces. *)

val contains : 'a t -> 'a -> bool
(** [contains s v] is [true] iff [v] is valid in [s]. *)

val sample : 'a t -> 'a
(** [sample s] is a uniformly sampled value from [s].

    Random keys are drawn from the implicit RNG scope. *)

val pack : 'a t -> 'a -> Value.t
(** [pack s v] is [v] converted to the universal {!Value.t} representation. *)

val unpack : 'a t -> Value.t -> ('a, string) result
(** [unpack s v] is [Ok x] if [v] can be converted to a valid element of [s], or
    [Error msg] otherwise. *)

val boundary_values : 'a t -> Value.t list
(** [boundary_values s] is a list of representative edge-case values for [s].
    Includes lower/upper bounds or canonical sentinels when known. The empty
    list when no boundary values apply. *)

(** {1:space_types Space types} *)

module Discrete : sig
  type element = (int32, Rune.int32_elt) Rune.t
  (** Discrete action represented as a scalar int32 tensor. *)

  val create : ?start:int -> int -> element t
  (** [create ?start n] is a discrete space with [n] choices in the range
      \[[start]; [start + n - 1]\]. [start] defaults to [0].

      Raises [Invalid_argument] if [n <= 0]. *)

  val n : element t -> int
  (** [n s] is the number of choices in [s].

      Raises [Invalid_argument] if [s] is not a discrete space. *)

  val start : element t -> int
  (** [start s] is the starting value of [s].

      Raises [Invalid_argument] if [s] is not a discrete space. *)

  val to_int : element -> int
  (** [to_int e] is the integer value of the discrete element [e]. *)

  val of_int : int -> element
  (** [of_int v] is a discrete element with value [v]. *)
end

module Box : sig
  type element = (float, Rune.float32_elt) Rune.t
  (** Continuous vector represented as a float32 tensor. *)

  val create : low:float array -> high:float array -> element t
  (** [create ~low ~high] is a continuous space where element [i] satisfies
      [low.(i) <= x.(i) <= high.(i)]. Both arrays must have the same positive
      length.

      When the range of a dimension is not finite (e.g. bounds set to
      [Float.max_float]), sampling falls back to a uniform draw in \[[-1e6];
      [1e6]\] clamped to bounds.

      Raises [Invalid_argument] if [low] is empty, if [low] and [high] differ in
      length, or if any [low.(i) > high.(i)]. *)

  val bounds : element t -> float array * float array
  (** [bounds s] is [(low, high)] copies of the bound vectors.

      Raises [Invalid_argument] if [s] is not a box space. *)
end

module Multi_binary : sig
  type element = (int32, Rune.int32_elt) Rune.t
  (** Binary vector for multi-label scenarios. *)

  val create : int -> element t
  (** [create n] is a binary vector space of length [n]. Valid values are int32
      tensors with [n] elements, each 0 or 1.

      Raises [Invalid_argument] if [n <= 0]. *)
end

module Multi_discrete : sig
  type element = (int32, Rune.int32_elt) Rune.t
  (** Multiple discrete choices with independent cardinalities. *)

  val create : int array -> element t
  (** [create nvec] is a multi-discrete space where element [i] is in \[[0];
      [nvec.(i) - 1]\].

      Raises [Invalid_argument] if [nvec] is empty or any [nvec.(i) <= 0]. *)
end

module Tuple : sig
  type element = Value.t list
  (** Fixed-length heterogeneous sequence in {!Value.t} form. *)

  val create : packed list -> element t
  (** [create spaces] is a tuple space. Valid values are lists where element [i]
      belongs to [spaces.(i)]. {!unpack} validates each element against its
      subspace. *)
end

module Dict : sig
  type element = (string * Value.t) list
  (** Named fields with different space types. *)

  val create : (string * packed) list -> element t
  (** [create fields] is a dictionary space with named fields. Valid values are
      association lists matching the keys and subspaces of [fields].

      Raises [Invalid_argument] if [fields] contains duplicate keys. *)
end

module Sequence : sig
  type 'a element = 'a list
  (** Variable-length homogeneous sequence. *)

  val create : ?min_length:int -> ?max_length:int -> 'a t -> 'a element t
  (** [create ?min_length ?max_length s] is a sequence space over [s].
      [min_length] defaults to [0]. When [max_length] is provided, sampling
      draws a uniform length in \[[min_length]; [max_length]\]; otherwise the
      sampler returns sequences of length [min_length].

      Raises [Invalid_argument] if [min_length < 0] or
      [max_length < min_length]. *)
end

module Text : sig
  type element = string
  (** String space for textual observations or actions. *)

  val create : ?charset:string -> ?max_length:int -> unit -> element t
  (** [create ?charset ?max_length ()] is a text space. [charset] defaults to
      alphanumeric plus space. [max_length] defaults to [64]. Valid strings
      contain only characters from [charset] and have length at most
      [max_length].

      Raises [Invalid_argument] if [max_length <= 0] or [charset] is empty. *)
end
