(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Typed compile-time constants.

    A constant pairs a scalar value with its {!Dtype.t}. Constructors validate
    that the dtype matches the value kind (boolean, integer, or float) and that
    the dtype is scalar. *)

(** {1:types Types} *)

type t
(** The type for typed constants. *)

type view =
  | Bool of bool  (** Boolean payload. *)
  | Int of int64  (** Integer payload (signed or unsigned, stored as [int64]). *)
  | Float of float  (** Floating-point payload. *)
(** Read-only constant payload. Pattern-match via {!view}. *)

(** {1:access Accessors} *)

val view : t -> view
(** [view c] is the payload of [c]. *)

val dtype : t -> Dtype.Val.t
(** [dtype c] is the dtype of [c]. *)

(** {1:constructors Constructors} *)

val bool : bool -> t
(** [bool b] is a boolean constant with dtype {!Dtype.bool}. *)

val int : Dtype.Val.t -> int -> t
(** [int dtype n] is an integer constant.

    Raises [Invalid_argument] if [dtype] is not a scalar integer or boolean
    dtype. *)

val int64 : Dtype.Val.t -> int64 -> t
(** [int64 dtype n] is an integer constant.

    Raises [Invalid_argument] if [dtype] is not a scalar integer or boolean
    dtype. *)

val float : Dtype.Val.t -> float -> t
(** [float dtype x] is a floating-point constant.

    Raises [Invalid_argument] if [dtype] is not a scalar float dtype. *)

(** {1:predicates Predicates and comparisons} *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [a] and [b] carry the same dtype and payload. *)

val compare : t -> t -> int
(** [compare a b] totally orders constants, first by dtype then by payload. *)

(** {1:fmt Formatting} *)

val to_string : t -> string
(** [to_string c] is a compact [value:dtype] representation (e.g. ["42:int32"],
    ["3.14:float32"], ["true:bool"]). *)

val pp : Format.formatter -> t -> unit
(** [pp] formats a constant with {!to_string}. *)

(** {1:helpers Dtype-aware helpers} *)

val zero : Dtype.Val.t -> t
(** [zero dtype] is the zero constant for [dtype]: [0.0] for floats,
    [false] for bools, [0] for integers. *)

val identity_element : Op.reduce -> Dtype.Val.t -> t
(** [identity_element op dtype] is the identity element for reduction [op]
    on [dtype]: [0] for [`Add], [1] for [`Mul], [dtype.min] for [`Max]. *)
