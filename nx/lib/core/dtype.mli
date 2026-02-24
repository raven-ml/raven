(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tensor element types.

    A dtype value describes both the OCaml value representation and the
    underlying buffer element kind used by [nx]. *)

(** {1:elements Element kinds} *)

type float16_elt = Nx_buffer.float16_elt
(** The element kind for IEEE 754 binary16 values. *)

type float32_elt = Nx_buffer.float32_elt
(** The element kind for IEEE 754 binary32 values. *)

type float64_elt = Nx_buffer.float64_elt
(** The element kind for IEEE 754 binary64 values. *)

type bfloat16_elt = Nx_buffer.bfloat16_elt
(** The element kind for bfloat16 values. *)

type float8_e4m3_elt = Nx_buffer.float8_e4m3_elt
(** The element kind for float8 e4m3 values. *)

type float8_e5m2_elt = Nx_buffer.float8_e5m2_elt
(** The element kind for float8 e5m2 values. *)

type int4_elt = Nx_buffer.int4_signed_elt
(** The element kind for signed 4-bit integers. *)

type uint4_elt = Nx_buffer.int4_unsigned_elt
(** The element kind for unsigned 4-bit integers. *)

type int8_elt = Nx_buffer.int8_signed_elt
(** The element kind for signed 8-bit integers. *)

type uint8_elt = Nx_buffer.int8_unsigned_elt
(** The element kind for unsigned 8-bit integers. *)

type int16_elt = Nx_buffer.int16_signed_elt
(** The element kind for signed 16-bit integers. *)

type uint16_elt = Nx_buffer.int16_unsigned_elt
(** The element kind for unsigned 16-bit integers. *)

type int32_elt = Nx_buffer.int32_elt
(** The element kind for signed 32-bit integers. *)

type uint32_elt = Nx_buffer.uint32_elt
(** The element kind for unsigned 32-bit integers. *)

type int64_elt = Nx_buffer.int64_elt
(** The element kind for signed 64-bit integers. *)

type uint64_elt = Nx_buffer.uint64_elt
(** The element kind for unsigned 64-bit integers. *)

type complex32_elt = Nx_buffer.complex32_elt
(** The element kind for complex values with float32 components. *)

type complex64_elt = Nx_buffer.complex64_elt
(** The element kind for complex values with float64 components. *)

type bool_elt = Nx_buffer.bool_elt
(** The element kind for boolean values. *)

(** {1:types Dtypes} *)

(** The type for dtypes.

    The first parameter is the OCaml value type and the second parameter is the
    buffer element kind. *)
type ('a, 'b) t =
  | Float16 : (float, float16_elt) t  (** 16-bit float. *)
  | Float32 : (float, float32_elt) t  (** 32-bit float. *)
  | Float64 : (float, float64_elt) t  (** 64-bit float. *)
  | BFloat16 : (float, bfloat16_elt) t  (** bfloat16. *)
  | Float8_e4m3 : (float, float8_e4m3_elt) t  (** float8 e4m3. *)
  | Float8_e5m2 : (float, float8_e5m2_elt) t  (** float8 e5m2. *)
  | Int4 : (int, int4_elt) t  (** Signed 4-bit integer carried in [int]. *)
  | UInt4 : (int, uint4_elt) t  (** Unsigned 4-bit integer carried in [int]. *)
  | Int8 : (int, int8_elt) t  (** Signed 8-bit integer carried in [int]. *)
  | UInt8 : (int, uint8_elt) t  (** Unsigned 8-bit integer carried in [int]. *)
  | Int16 : (int, int16_elt) t  (** Signed 16-bit integer carried in [int]. *)
  | UInt16 : (int, uint16_elt) t
      (** Unsigned 16-bit integer carried in [int]. *)
  | Int32 : (int32, int32_elt) t  (** Signed 32-bit integer. *)
  | UInt32 : (int32, uint32_elt) t
      (** Unsigned 32-bit integer carried in [int32]. *)
  | Int64 : (int64, int64_elt) t  (** Signed 64-bit integer. *)
  | UInt64 : (int64, uint64_elt) t
      (** Unsigned 64-bit integer carried in [int64]. *)
  | Complex64 : (Complex.t, complex32_elt) t
      (** Complex values with float32 components. *)
  | Complex128 : (Complex.t, complex64_elt) t
      (** Complex values with float64 components. *)
  | Bool : (bool, bool_elt) t  (** Boolean values. *)

(** {1:constructors Constructor values} *)

val float16 : (float, float16_elt) t
(** [float16] is {!Float16}. *)

val float32 : (float, float32_elt) t
(** [float32] is {!Float32}. *)

val float64 : (float, float64_elt) t
(** [float64] is {!Float64}. *)

val bfloat16 : (float, bfloat16_elt) t
(** [bfloat16] is {!BFloat16}. *)

val float8_e4m3 : (float, float8_e4m3_elt) t
(** [float8_e4m3] is {!Float8_e4m3}. *)

val float8_e5m2 : (float, float8_e5m2_elt) t
(** [float8_e5m2] is {!Float8_e5m2}. *)

val int4 : (int, int4_elt) t
(** [int4] is {!Int4}. *)

val uint4 : (int, uint4_elt) t
(** [uint4] is {!UInt4}. *)

val int8 : (int, int8_elt) t
(** [int8] is {!Int8}. *)

val uint8 : (int, uint8_elt) t
(** [uint8] is {!UInt8}. *)

val int16 : (int, int16_elt) t
(** [int16] is {!Int16}. *)

val uint16 : (int, uint16_elt) t
(** [uint16] is {!UInt16}. *)

val int32 : (int32, int32_elt) t
(** [int32] is {!Int32}. *)

val uint32 : (int32, uint32_elt) t
(** [uint32] is {!UInt32}. *)

val int64 : (int64, int64_elt) t
(** [int64] is {!Int64}. *)

val uint64 : (int64, uint64_elt) t
(** [uint64] is {!UInt64}. *)

val complex64 : (Complex.t, complex32_elt) t
(** [complex64] is {!Complex64}. *)

val complex128 : (Complex.t, complex64_elt) t
(** [complex128] is {!Complex128}. *)

val bool : (bool, bool_elt) t
(** [bool] is {!Bool}. *)

(** {1:preds Predicates and properties} *)

val to_string : ('a, 'b) t -> string
(** [to_string d] is the stable lowercase name of [d]. *)

val pp : Format.formatter -> ('a, 'b) t -> unit
(** [pp] formats dtypes with [to_string]. *)

val itemsize : ('a, 'b) t -> int
(** [itemsize d] is the storage quantum in bytes for [d].

    For [Int4] and [UInt4], this is [1] even though values are 4-bit. Use
    {!bits} to get logical bit width. *)

val bits : ('a, 'b) t -> int
(** [bits d] is the logical bit width of elements of [d]. *)

val is_float : ('a, 'b) t -> bool
(** [is_float d] is [true] iff [d] is one of the float dtypes. *)

val is_complex : ('a, 'b) t -> bool
(** [is_complex d] is [true] iff [d] is one of the complex dtypes. *)

val is_int : ('a, 'b) t -> bool
(** [is_int d] is [true] iff [d] is an integer dtype, signed or unsigned. *)

val is_uint : ('a, 'b) t -> bool
(** [is_uint d] is [true] iff [d] is an unsigned integer dtype. *)

(** {1:constants Constants} *)

val zero : ('a, 'b) t -> 'a
(** [zero d] is the additive identity value for [d]. *)

val one : ('a, 'b) t -> 'a
(** [one d] is the multiplicative identity value for [d]. *)

val two : ('a, 'b) t -> 'a
(** [two d] is the value [2] represented in [d].

    For [Bool], [two Bool] is [true]. *)

val minus_one : ('a, 'b) t -> 'a
(** [minus_one d] is the value [-1] represented in [d].

    For unsigned integer and bool dtypes this is the all-ones bit pattern. *)

val min_value : ('a, 'b) t -> 'a
(** [min_value d] is the minimum value used by [d].

    For floating dtypes this is [-infinity].

    Raises [Invalid_argument] for complex dtypes. *)

val max_value : ('a, 'b) t -> 'a
(** [max_value d] is the maximum value used by [d].

    For floating dtypes this is [+infinity].

    Raises [Invalid_argument] for complex dtypes. *)

(** {1:converting Converting} *)

val of_float : ('a, 'b) t -> float -> 'a
(** [of_float d x] converts [x] to dtype [d].

    Unsigned integer conversions clamp to their representable range. *)

val of_buffer_kind : ('a, 'b) Nx_buffer.kind -> ('a, 'b) t
(** [of_buffer_kind k] is the dtype corresponding to [k].

    Raises [Invalid_argument] if [k] is unsupported. *)

val to_buffer_kind : ('a, 'b) t -> ('a, 'b) Nx_buffer.kind
(** [to_buffer_kind d] is the [Nx_buffer] kind corresponding to [d]. *)

val of_bigarray_kind : ('a, 'b) Bigarray.kind -> ('a, 'b) t
(** [of_bigarray_kind k] is the dtype corresponding to [k].

    Raises [Invalid_argument] if [k] is unsupported. *)

val to_bigarray_kind : ('a, 'b) t -> ('a, 'b) Bigarray.kind
(** [to_bigarray_kind d] is the standard [Bigarray] kind for [d].

    Raises [Invalid_argument] for extended dtypes that standard [Bigarray]
    cannot represent ([BFloat16], [Float8_e4m3], [Float8_e5m2], [Int4], [UInt4],
    [UInt32], [UInt64], [Bool]). *)

(** {1:equality Equality} *)

val equal : ('a, 'b) t -> ('c, 'd) t -> bool
(** [equal d0 d1] is [true] iff [d0] and [d1] denote the same dtype constructor.
*)

val equal_witness :
  ('a, 'b) t -> ('c, 'd) t -> (('a, 'b) t, ('c, 'd) t) Type.eq option
(** [equal_witness d0 d1] is [Some Type.Equal] iff [equal d0 d1] is [true], and
    [None] otherwise. *)

(** {1:packed Packed dtypes} *)

type packed =
  | Pack : ('a, 'b) t -> packed  (** Existential wrapper over dtypes. *)

val pack : ('a, 'b) t -> packed
(** [pack d] is [Pack d]. *)

module Packed : sig
  (** Operations on [packed]. *)

  type t = packed
  (** The type for packed dtypes. *)

  val all : t list
  (** [all] lists all supported dtypes. *)

  val of_string : string -> t option
  (** [of_string s] is the dtype named [s], if any. *)

  val to_string : t -> string
  (** [to_string t] is the lowercase name of [t]. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats packed dtypes with [to_string]. *)

  val equal : t -> t -> bool
  (** [equal d0 d1] is [true] iff [d0] and [d1] are the same dtype. *)

  val compare : t -> t -> int
  (** [compare] orders dtypes by a stable internal tag. *)

  val hash : t -> int
  (** [hash t] is a hash derived from [tag]. *)

  val tag : t -> int
  (** [tag t] is the stable integer tag used by [compare] and [hash]. *)
end

(** {1:ops Scalar operations} *)

val add : ('a, 'b) t -> 'a -> 'a -> 'a
(** [add d x y] adds [x] and [y] with dtype semantics of [d].

    Narrow integer dtypes wrap to their bit width. For [Bool], this is boolean
    disjunction. *)

val sub : ('a, 'b) t -> 'a -> 'a -> 'a
(** [sub d x y] subtracts [y] from [x] with dtype semantics of [d].

    Narrow integer dtypes wrap to their bit width.

    Raises [Invalid_argument] for [Bool]. *)

val mul : ('a, 'b) t -> 'a -> 'a -> 'a
(** [mul d x y] multiplies [x] and [y] with dtype semantics of [d].

    Narrow integer dtypes wrap to their bit width. For [Bool], this is boolean
    conjunction. *)

val div : ('a, 'b) t -> 'a -> 'a -> 'a
(** [div d x y] divides [x] by [y] with dtype semantics of [d].

    Narrow integer dtypes wrap to their bit width.

    Raises [Division_by_zero] for integer division by zero. Raises
    [Invalid_argument] for [Bool]. *)
