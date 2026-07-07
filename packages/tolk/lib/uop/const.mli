(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Typed compile-time constants.

    A constant pairs a scalar payload with its {!Dtype.Val.t}. Direct integer
    and floating-point constructors validate that the dtype is scalar (vector
    width [1]) and matches the payload kind. Dtype-directed construction can
    keep vector dtypes for scalar-broadcast constants. Integer payloads are
    always stored as [int64] regardless of the integer width; floating-point
    payloads are stored as native [float] and are only truncated to [dtype]'s
    precision on emission, not on construction.

    Constants participate in pattern matching and constant folding through
    {!view}. The {!Invalid} payload is the absorbing element of ALU folding
    and represents masked-out lanes (see {!invalid}). *)

(** {1:types Types} *)

type t
(** The type for typed constants. *)

(** Read-only constant payload. Obtain via {!view}. *)
type view =
  | Bool of bool  (** Boolean payload. *)
  | Int of int64
      (** Integer payload. Both signed and unsigned dtypes are stored in a
          single [int64] slot; unsigned values are held as their raw 64-bit
          bit pattern (e.g. [uint64] max is [Int64.minus_one]). *)
  | Float of float  (** Floating-point payload in native double precision. *)
  | Invalid
      (** Sentinel for masked-out or undefined values. Absorbing under ALU
          folding: any ALU op with an [Invalid] operand folds to [Invalid]. *)

(** {1:access Accessors} *)

val view : t -> view
(** [view c] is the payload of [c]. *)

val dtype : t -> Dtype.Val.t
(** [dtype c] is the dtype of [c]. It may be a vector dtype for
    scalar-broadcast or invalid vector constants. *)

(** {1:constructors Constructors} *)

val bool : bool -> t
(** [bool b] is the boolean constant [b] with dtype {!Dtype.Val.bool}. *)

val int : Dtype.Val.t -> int -> t
(** [int dtype n] is [int64 dtype (Int64.of_int n)]. See {!int64}.

    Raises [Invalid_argument] if [dtype] is not a scalar integer dtype. *)

val int64 : Dtype.Val.t -> int64 -> t
(** [int64 dtype n] is the integer constant [n] tagged with [dtype]. The
    value is stored verbatim; no range-checking or truncation is performed
    against [dtype]'s width.

    Raises [Invalid_argument]
      if [dtype] is not a scalar integer dtype (as per
      {!Dtype.Val.is_int}, which accepts {!Dtype.Val.weakint} and all signed
      and unsigned integer types). *)

val float : Dtype.Val.t -> float -> t
(** [float dtype x] is the floating-point constant [x] tagged with [dtype].
    The value is stored verbatim; narrowing to [dtype]'s precision is
    deferred to emission via {!Dtype.truncate_float}.

    Raises [Invalid_argument] if [dtype] is not a scalar floating-point dtype. *)

val invalid : ?dtype:Dtype.Val.t -> unit -> t
(** [invalid ?dtype ()] is the [Invalid] sentinel. [dtype] defaults to
    {!Dtype.Val.weakint}. The dtype is unchecked: any scalar or vector
    dtype is accepted so that [Invalid] can propagate through typed IR
    positions. *)

val of_scalar : Dtype.Val.t -> Dtype.storage_scalar -> t
(** [of_scalar dtype x] coerces storage scalar [x] according to [dtype], like
    tinygrad's [DType.const]: floating-point dtypes produce canonicalized
    {!Float} payloads, bool dtypes produce {!Bool}, and all other dtypes
    produce integer payloads. Vector dtypes keep their vector dtype while
    carrying the scalar payload, matching tinygrad scalar broadcasts.
    Float-to-integer conversion follows {!Int64.of_float}. *)

val of_view : Dtype.Val.t -> view -> t
(** [of_view dtype v] is [invalid ~dtype ()] for {!Invalid}; otherwise it
    coerces [v] with {!of_scalar}. *)

(** {1:predicates Predicates and comparisons} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] carry the same dtype and the
    same payload. Floating-point payloads are compared by their [int64]
    bit pattern, so [NaN] equals [NaN] and [-0.0] differs from [0.0]. *)

val compare : t -> t -> int
(** [compare a b] is a total order, keyed first by {!Dtype.Val.compare}
    then by payload. Float payloads are ordered by their [int64] bit
    pattern (same caveat as {!equal}); [Bool < Int < Float < Invalid]
    within a given dtype. *)

(** {1:fmt Formatting} *)

val to_string : t -> string
(** [to_string c] is a compact [value:dtype] representation using
    {!Dtype.Val.to_string} for the tag (e.g. ["42:i32"], ["3.14:f32"],
    ["true:bool"], ["Invalid:weakint"]). *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a constant with {!to_string}. *)

(** {1:helpers Dtype-aware helpers}

    All helpers take a scalar [dtype] and raise [Invalid_argument] via the
    underlying constructor if [dtype] is not scalar. *)

val zero : Dtype.Val.t -> t
(** [zero dtype] is the additive identity: [0.0] for floats, [false] for
    {!Dtype.Val.bool}, [0] for integers. *)

val one : Dtype.Val.t -> t
(** [one dtype] is the multiplicative identity: [1.0] for floats, [true]
    for {!Dtype.Val.bool}, [1] for integers. *)

val min_value : Dtype.Val.t -> t
(** [min_value dtype] is the smallest representable value of [dtype]:
    [-infinity] for floats, [false] for bools, {!Dtype.min} for integers.
    It is the identity element of a max-reduction over [dtype]. *)

val max_value : Dtype.Val.t -> t
(** [max_value dtype] is the largest representable value of [dtype]:
    [infinity] for floats, [true] for bools, {!Dtype.max} for integers.
    It is the identity element of a min-reduction over [dtype]. *)
