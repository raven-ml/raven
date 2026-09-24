(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.
  ---------------------------------------------------------------------------*)

(** Numeric interval endpoints. Integers retain arbitrary precision; floating
    bounds may be infinite but must not be NaN. Comparisons between integer
    and floating bounds do not round the integer. *)

type t = Dtype.bound

val int : int -> t
(** [int n] is the exact integer endpoint [n]. *)
val zero : t
(** [zero] is the integer endpoint zero. *)
val one : t
(** [one] is the integer endpoint one. *)
val integer : t -> Z.t
(** [integer b] is the integer value of [b], interpreting booleans as zero or
    one. Raises [Invalid_argument] for floating-point endpoints. *)
val to_int : t -> int
(** [to_int b] is [integer b] as a host integer. Raises [Invalid_argument] if
    [b] is floating-point or exceeds the host integer range. *)
val to_float : t -> float
(** [to_float b] converts [b] to a host float, rounding integers if necessary. *)
val compare : t -> t -> int
(** [compare a b] compares numeric values exactly across kinds. Raises
    [Invalid_argument] if either endpoint is NaN. *)
val equal : t -> t -> bool
(** [equal a b] is numeric equality, including both signs of floating zero. *)
val lt : t -> t -> bool
(** [lt a b] is [a < b]. *)
val le : t -> t -> bool
(** [le a b] is [a <= b]. *)
val min : t -> t -> t
(** [min a b] is the smaller endpoint. *)
val max : t -> t -> t
(** [max a b] is the larger endpoint. *)
val add : t -> t -> t
(** [add a b] adds endpoints, using floating arithmetic if either is floating. *)
val sub : t -> t -> t
(** [sub a b] subtracts endpoints with the same promotion as {!add}. *)
val mul : t -> t -> t
(** [mul a b] multiplies endpoints with the same promotion as {!add}. *)
val neg : t -> t
(** [neg b] negates [b]. *)
val succ : t -> t
(** [succ b] adds one to [b]. *)
val pred : t -> t
(** [pred b] subtracts one from [b]. *)
val cdiv : t -> t -> t
(** [cdiv a b] divides integer endpoints, truncating towards zero. *)
val floordiv : t -> t -> t
(** [floordiv a b] divides integer endpoints, rounding towards minus infinity. *)
val floormod : t -> t -> t
(** [floormod a b] is [a - floordiv a b * b]. *)
val lognot : t -> t
(** [lognot b] complements an integer endpoint. *)
val shift_left : t -> t -> t
(** [shift_left a b] shifts integer [a] left by the non-negative host-sized
    integer [b]. *)
val shift_right : t -> t -> t
(** [shift_right a b] arithmetically shifts integer [a] right by [b]. *)
val round : Dtype.t -> t -> t
(** [round dtype b] applies monotone rounding at a cast: float precision or
    truncation towards zero for integer destinations. It does not wrap or
    clamp integers. Infinite endpoints remain infinite. *)
val const : Dtype.t -> t -> Const.t
(** [const dtype b] converts [b] to a constant at [dtype]. *)
