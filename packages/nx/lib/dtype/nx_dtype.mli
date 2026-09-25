(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dtypes and storage formats.

    A dtype {!type:t} names the element type of an array: the OCaml type its
    values are read and written as, and the element type of its storage. Typed
    code takes a [('a, 'b) t] and learns the value type ['a] from it.

    A {!Scalar.t} names a storage format with no type parameters, for code that
    allocates, moves or compiles bytes without reading them as OCaml values. It
    has two formats with no dtype.

    [Nx] re-exports the dtypes as [Nx.dtype]. *)

(** {1:elt Element types}

    The second parameter of a dtype. The standard ones are {!Bigarray}'s; the
    others have no values. *)

type float16_elt = Bigarray.float16_elt
(** The element type of IEEE 754 binary16 values. *)

type float32_elt = Bigarray.float32_elt
(** The element type of IEEE 754 binary32 values. *)

type float64_elt = Bigarray.float64_elt
(** The element type of IEEE 754 binary64 values. *)

type bfloat16_elt
(** The element type of bfloat16 values: 8 exponent and 7 fraction bits. *)

type float8_e4m3_elt
(** The element type of float8 e4m3 values: 4 exponent and 3 fraction bits, no
    infinities. *)

type float8_e5m2_elt
(** The element type of float8 e5m2 values: 5 exponent and 2 fraction bits. *)

type int4_elt
(** The element type of signed 4-bit integers, two to a byte. *)

type uint4_elt
(** The element type of unsigned 4-bit integers, two to a byte. *)

type int8_elt = Bigarray.int8_signed_elt
(** The element type of signed 8-bit integers. *)

type uint8_elt = Bigarray.int8_unsigned_elt
(** The element type of unsigned 8-bit integers. *)

type int16_elt = Bigarray.int16_signed_elt
(** The element type of signed 16-bit integers. *)

type uint16_elt = Bigarray.int16_unsigned_elt
(** The element type of unsigned 16-bit integers. *)

type int32_elt = Bigarray.int32_elt
(** The element type of signed 32-bit integers. *)

type uint32_elt
(** The element type of unsigned 32-bit integers. *)

type int64_elt = Bigarray.int64_elt
(** The element type of signed 64-bit integers. *)

type uint64_elt
(** The element type of unsigned 64-bit integers. *)

type complex32_elt = Bigarray.complex32_elt
(** The element type of complex values with binary32 components. *)

type complex64_elt = Bigarray.complex64_elt
(** The element type of complex values with binary64 components. *)

type bool_elt
(** The element type of booleans, one to a byte. *)

(** {1:types Dtypes} *)

(** The type for dtypes. ['a] is the OCaml type of values and ['b] the element
    type. Unsigned 32-bit and 64-bit integers are carried in [int32] and [int64]
    with their bits unchanged. *)
type ('a, 'b) t =
  | Float16 : (float, float16_elt) t  (** IEEE 754 binary16. *)
  | Float32 : (float, float32_elt) t  (** IEEE 754 binary32. *)
  | Float64 : (float, float64_elt) t  (** IEEE 754 binary64. *)
  | BFloat16 : (float, bfloat16_elt) t  (** bfloat16. *)
  | Float8_e4m3 : (float, float8_e4m3_elt) t  (** float8 e4m3. *)
  | Float8_e5m2 : (float, float8_e5m2_elt) t  (** float8 e5m2. *)
  | Int4 : (int, int4_elt) t  (** Signed 4-bit integers. *)
  | UInt4 : (int, uint4_elt) t  (** Unsigned 4-bit integers. *)
  | Int8 : (int, int8_elt) t  (** Signed 8-bit integers. *)
  | UInt8 : (int, uint8_elt) t  (** Unsigned 8-bit integers. *)
  | Int16 : (int, int16_elt) t  (** Signed 16-bit integers. *)
  | UInt16 : (int, uint16_elt) t  (** Unsigned 16-bit integers. *)
  | Int32 : (int32, int32_elt) t  (** Signed 32-bit integers. *)
  | UInt32 : (int32, uint32_elt) t  (** Unsigned 32-bit integers. *)
  | Int64 : (int64, int64_elt) t  (** Signed 64-bit integers. *)
  | UInt64 : (int64, uint64_elt) t  (** Unsigned 64-bit integers. *)
  | Complex64 : (Complex.t, complex32_elt) t
      (** Complex values with binary32 components. *)
  | Complex128 : (Complex.t, complex64_elt) t
      (** Complex values with binary64 components. *)
  | Bool : (bool, bool_elt) t  (** Booleans. *)

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

(** {1:queries Queries} *)

val to_string : ('a, 'b) t -> string
(** [to_string dt] is the lowercase name of [dt], its constructor's name, such
    as ["float32"] or ["bfloat16"]. *)

val pp : Format.formatter -> ('a, 'b) t -> unit
(** [pp ppf dt] formats [dt] with {!to_string}. *)

val itemsize : ('a, 'b) t -> int
(** [itemsize dt] is the size in bytes of one element of [dt]. It is [1] for
    {!Int4} and {!UInt4}, whose elements are 4 bits. *)

val is_float : ('a, 'b) t -> bool
(** [is_float dt] is [true] iff [dt] is a real floating-point dtype. *)

val is_complex : ('a, 'b) t -> bool
(** [is_complex dt] is [true] iff [dt] is {!Complex64} or {!Complex128}. *)

val is_int : ('a, 'b) t -> bool
(** [is_int dt] is [true] iff [dt] is an integer dtype, signed or unsigned. *)

val is_uint : ('a, 'b) t -> bool
(** [is_uint dt] is [true] iff [dt] is an unsigned integer dtype. *)

(** {1:constants Constants} *)

val zero : ('a, 'b) t -> 'a
(** [zero dt] is [0] as a value of [dt], and [false] for {!Bool}. *)

val one : ('a, 'b) t -> 'a
(** [one dt] is [1] as a value of [dt], and [true] for {!Bool}. *)

val two : ('a, 'b) t -> 'a
(** [two dt] is [2] as a value of [dt], and [true] for {!Bool}. *)

val minus_one : ('a, 'b) t -> 'a
(** [minus_one dt] is [-1] as a value of [dt]. For unsigned dtypes it is the
    value with every bit set, and for {!Bool} it is [true]. *)

val min_value : ('a, 'b) t -> 'a
(** [min_value dt] is the least value of [dt]: [neg_infinity] for the float
    dtypes, except [-448.] for {!Float8_e4m3}, which has no infinity.

    Raises [Invalid_argument] if [dt] is complex. *)

val max_value : ('a, 'b) t -> 'a
(** [max_value dt] is the greatest value of [dt]: [infinity] for the float
    dtypes, except [448.] for {!Float8_e4m3}, which has no infinity. For
    unsigned 32-bit and 64-bit integers it is the value with every bit set.

    Raises [Invalid_argument] if [dt] is complex. *)

(** {1:converting Converting} *)

val of_float : ('a, 'b) t -> float -> 'a
(** [of_float dt x] is [x] as a value of [dt]. Float dtypes keep [x]; storing it
    rounds it. Signed integers truncate [x] toward zero, unsigned integers also
    clamp it to their range, complex dtypes give [x] a zero imaginary part and
    {!Bool} is [x <> 0.]. *)

val of_bigarray_kind : ('a, 'b) Bigarray.kind -> ('a, 'b) t
(** [of_bigarray_kind k] is the dtype of [k]'s elements.

    Raises [Invalid_argument] if [k] is [Char], [Int] or [Nativeint]. *)

val to_bigarray_kind : ('a, 'b) t -> ('a, 'b) Bigarray.kind option
(** [to_bigarray_kind dt] is the {!Bigarray.kind} of [dt]'s elements, if
    {!Bigarray} has one. It is [None] for {!BFloat16}, the float8 dtypes,
    {!Int4}, {!UInt4}, {!UInt32}, {!UInt64} and {!Bool}. *)

(** {1:predicates Equality} *)

val equal : ('a, 'b) t -> ('c, 'd) t -> bool
(** [equal dt0 dt1] is [true] iff [dt0] and [dt1] are the same constructor. *)

val equal_witness :
  ('a, 'b) t -> ('c, 'd) t -> (('a, 'b) t, ('c, 'd) t) Type.eq option
(** [equal_witness dt0 dt1] is [Some Type.Equal] iff [equal dt0 dt1], and [None]
    otherwise. *)

(** {1:scalar Scalar formats} *)

(** Storage formats with no type parameters. {!of_dtype} gives a dtype's. *)
module Scalar : sig
  type ('a, 'b) dtype := ('a, 'b) t

  (** The type for storage formats: one per dtype, of the same name, and the two
      float8 fnuz formats, which have no dtype. *)
  type t =
    | Float16
    | Float32
    | Float64
    | BFloat16
    | Float8_e4m3
    | Float8_e5m2
    | Float8_e4m3fnuz
        (** float8 with 4 exponent and 3 fraction bits, exponent bias 8, no
            infinities, no negative zero and one NaN, [0x80]. *)
    | Float8_e5m2fnuz
        (** float8 with 5 exponent and 2 fraction bits, exponent bias 16, no
            infinities, no negative zero and one NaN, [0x80]. *)
    | Int4
    | UInt4
    | Int8
    | UInt8
    | Int16
    | UInt16
    | Int32
    | UInt32
    | Int64
    | UInt64
    | Complex64
    | Complex128
    | Bool

  val of_dtype : ('a, 'b) dtype -> t
  (** [of_dtype dt] is the format of [dt]'s elements. *)

  val bitsize : t -> int
  (** [bitsize s] is the size in bits of one element of [s]: [4] for {!Int4} and
      {!UInt4}, and [8] for {!Bool}. *)

  val to_string : t -> string
  (** [to_string s] is the lowercase name of [s], its constructor's name. For
      the formats of a dtype it is that dtype's {!Nx_dtype.to_string}. *)

  val equal : t -> t -> bool
  (** [equal s0 s1] is [true] iff [s0] and [s1] are the same format. *)

  (** {1:encoding Encoding}

      The float formats narrower than binary32, {!Float16}, {!BFloat16} and the
      four float8 formats, have no OCaml type. Their values are read as [float]s
      and stored as the bits these functions give. The C header [nx_dtype.h],
      which nx.dtype installs, holds the same encoders, and [Nx]'s element
      stores and casts use them, except a store of a [float] into float16, which
      rounds to binary32 first. *)

  val encode : t -> float -> int
  (** [encode s x] is the bits of [x] in format [s], rounded once to nearest,
      ties to even. A finite [x] past the largest finite value of [s] encodes as
      the infinity of its sign, or as NaN in the formats with no infinity:
      {!Float8_e4m3} and the fnuz formats. A NaN keeps its sign, except in the
      fnuz formats, whose one NaN is [0x80]; which payload bits it keeps is
      unspecified.

      Raises [Invalid_argument] if [s] is not one of the formats above. *)

  val decode : t -> int -> float
  (** [decode s c] is the value of the bits [c] in format [s]. Every finite
      value is exact.

      Raises [Invalid_argument] if [s] is not one of the formats above, or if
      [c] is not in \[[0];[2{^n} - 1]\] where [n] is [bitsize s]. *)
end
