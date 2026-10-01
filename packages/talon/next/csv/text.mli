(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The text forms of values.

    Each function reads the [len] bytes at [pos] of a byte sequence as the text
    form of its type that {!Talon_next_csv} documents, and raises {!Invalid}
    when they are not that form or their value is outside the type.

    A valid text allocates nothing, except a float that the exact fast path does
    not cover (a significand above 2{^ 53}, its digits read as an integer, or a
    decimal exponent outside \[[-22];[22]\]), which [float_of_string] reads,
    correctly rounded. Values that an OCaml [int] may not hold are stored into
    an array at an index, never returned, so that they are never boxed. *)

type int64s = (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for arrays of [int64] storage. *)

type float64s =
  (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
(** The type for arrays of [float64] storage. *)

exception Invalid of string
(** [Invalid why] is raised for a text that is not a value of the type, [why]
    saying what is wrong: [not an integer], [out of range]. *)

(** {1:numbers Booleans and numbers} *)

val bool : Bytes.t -> int -> int -> bool
(** [bool b pos len] reads [true] and [false]. *)

val int : min:int -> max:int -> Bytes.t -> int -> int -> int
(** [int ~min ~max b pos len] reads a decimal integer with an optional sign, in
    \[[min];[max]\], a range of at most 2{^ 32} values. *)

val int64 : Bytes.t -> int -> int -> int64s -> int -> unit
(** [int64 b pos len a i] reads a decimal integer with an optional sign that
    [int64] holds, and stores it at [a.{i}]. *)

val uint64 : Bytes.t -> int -> int -> int64s -> int -> unit
(** [uint64 b pos len a i] reads a decimal integer with an optional sign that
    [uint64] holds, a minus only before zero, and stores its 64 bits at [a.{i}].
*)

val float : Bytes.t -> int -> int -> float64s -> int -> unit
(** [float b pos len a i] reads a decimal number, [inf], [infinity] or [nan] as
    the nearest [float64], and stores it at [a.{i}]. *)

val float32 : Bytes.t -> int -> int -> float64s -> int -> unit
(** [float32 b pos len a i] reads the text as {!float} does, and stores at
    [a.{i}] a float64 that rounds to the float32 nearest to the text, ties to
    even. It allocates when the float64 nearest to the text is halfway between
    two float32 values, whose digits it then compares exactly. *)

val float16 : Bytes.t -> int -> int -> float64s -> int -> unit
(** [float16 b pos len a i] is {!float32} for float16. *)

val decimal : precision:int -> scale:int -> Bytes.t -> int -> int -> int
(** [decimal ~precision ~scale b pos len] reads a decimal number without an
    exponent, exact at [scale] digits after the point and of at most [precision]
    digits, as its unscaled value. [precision <= 18]. *)

(** {1:time Dates and datetimes} *)

val date : Bytes.t -> int -> int -> int
(** [date b pos len] reads [YYYY-MM-DD] as its days since 1970-01-01. *)

val datetime :
  Talon_next.Type.unit_ ->
  zoned:bool ->
  Bytes.t ->
  int ->
  int ->
  int64s ->
  int ->
  unit
(** [datetime u ~zoned b pos len a i] reads a datetime, with an offset iff
    [zoned], as ticks of [u] since 1970-01-01 00:00:00, in UTC when it has an
    offset, and stores them at [a.{i}]. The value must be a whole number of [u]
    whose ticks [int64] holds. *)

(** {1:text Text} *)

val utf_8 : Bytes.t -> int -> int -> unit
(** [utf_8 b pos len] is [()] if the text is valid UTF-8. *)

val leading_zero : Bytes.t -> int -> int -> bool
(** [leading_zero b pos len] is [true] iff the text, after an optional sign,
    starts with a zero followed by a digit, as [007] and [-01.5] do. *)
