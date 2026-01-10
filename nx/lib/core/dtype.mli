(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Data types for tensor elements. *)

(** {2 Element Types} *)

type float16_elt = Nx_buffer.float16_elt
type float32_elt = Nx_buffer.float32_elt
type float64_elt = Nx_buffer.float64_elt
type bfloat16_elt = Nx_buffer.bfloat16_elt
type float8_e4m3_elt = Nx_buffer.float8_e4m3_elt
type float8_e5m2_elt = Nx_buffer.float8_e5m2_elt
type int4_elt = Nx_buffer.int4_signed_elt
type uint4_elt = Nx_buffer.int4_unsigned_elt
type int8_elt = Nx_buffer.int8_signed_elt
type uint8_elt = Nx_buffer.int8_unsigned_elt
type int16_elt = Nx_buffer.int16_signed_elt
type uint16_elt = Nx_buffer.int16_unsigned_elt
type int32_elt = Nx_buffer.int32_elt
type uint32_elt = Nx_buffer.uint32_elt
type int64_elt = Nx_buffer.int64_elt
type uint64_elt = Nx_buffer.uint64_elt
type complex32_elt = Nx_buffer.complex32_elt
type complex64_elt = Nx_buffer.complex64_elt
type bool_elt = Nx_buffer.bool_elt

(** {2 Data Type} *)

(** GADT representing tensor element types. First type parameter is the OCaml
    type, second is the Bigarray element type. *)
type ('a, 'b) t =
  | Float16 : (float, float16_elt) t
  | Float32 : (float, float32_elt) t
  | Float64 : (float, float64_elt) t
  | BFloat16 : (float, bfloat16_elt) t
  | Float8_e4m3 : (float, float8_e4m3_elt) t
  | Float8_e5m2 : (float, float8_e5m2_elt) t
  | Int4 : (int, int4_elt) t
  | UInt4 : (int, uint4_elt) t
  | Int8 : (int, int8_elt) t
  | UInt8 : (int, uint8_elt) t
  | Int16 : (int, int16_elt) t
  | UInt16 : (int, uint16_elt) t
  | Int32 : (int32, int32_elt) t
  | UInt32 : (int32, uint32_elt) t
  | Int64 : (int64, int64_elt) t
  | UInt64 : (int64, uint64_elt) t
  | Complex64 : (Complex.t, complex32_elt) t
  | Complex128 : (Complex.t, complex64_elt) t
  | Bool : (bool, bool_elt) t

(** {2 Constructors} *)

val float16 : (float, float16_elt) t
val float32 : (float, float32_elt) t
val float64 : (float, float64_elt) t
val bfloat16 : (float, bfloat16_elt) t
val float8_e4m3 : (float, float8_e4m3_elt) t
val float8_e5m2 : (float, float8_e5m2_elt) t
val int4 : (int, int4_elt) t
val uint4 : (int, uint4_elt) t
val int8 : (int, int8_elt) t
val uint8 : (int, uint8_elt) t
val int16 : (int, int16_elt) t
val uint16 : (int, uint16_elt) t
val int32 : (int32, int32_elt) t
val uint32 : (int32, uint32_elt) t
val int64 : (int64, int64_elt) t
val uint64 : (int64, uint64_elt) t
val complex64 : (Complex.t, complex32_elt) t
val complex128 : (Complex.t, complex64_elt) t
val bool : (bool, bool_elt) t

(** {2 Properties} *)

val to_string : ('a, 'b) t -> string
(** [to_string dtype] returns string representation. *)

val pp : Format.formatter -> ('a, 'b) t -> unit
(** [pp fmt dtype] pretty-prints the dtype. *)

val itemsize : ('a, 'b) t -> int
(** [itemsize dtype] returns storage size in bytes per element. For packed types
    like Int4/UInt4, this returns 1 (the storage quantum) even though two
    elements can be packed per byte. Use {!bits} for the actual bit width. *)

val bits : ('a, 'b) t -> int
(** [bits dtype] returns the bit width of the dtype. *)

(** {2 Type Classes} *)

val is_float : ('a, 'b) t -> bool
(** [is_float dtype] returns true for floating-point types. *)

val is_complex : ('a, 'b) t -> bool
(** [is_complex dtype] returns true for complex types. *)

val is_int : ('a, 'b) t -> bool
(** [is_int dtype] returns true for integer types (signed and unsigned). *)

val is_uint : ('a, 'b) t -> bool
(** [is_uint dtype] returns true for unsigned integer types. *)

(** {2 Constants} *)

val zero : ('a, 'b) t -> 'a
(** [zero dtype] returns zero value. *)

val one : ('a, 'b) t -> 'a
(** [one dtype] returns one value. *)

val two : ('a, 'b) t -> 'a
(** [two dtype] returns two value. *)

val minus_one : ('a, 'b) t -> 'a
(** [minus_one dtype] returns negative one value. *)

val min_value : ('a, 'b) t -> 'a
(** [min_value dtype] returns minimum representable value. *)

val max_value : ('a, 'b) t -> 'a
(** [max_value dtype] returns maximum representable value. *)

(** {2 Conversions} *)

val of_float : ('a, 'b) t -> float -> 'a
(** [of_float dtype f] converts float to dtype value. *)

val of_buffer_kind : ('a, 'b) Nx_buffer.kind -> ('a, 'b) t
(** [of_buffer_kind kind] returns corresponding dtype from Nx_buffer kind. *)

val to_bigarray_kind : ('a, 'b) t -> ('a, 'b) Bigarray.kind
(** [to_bigarray_kind dtype] returns corresponding standard Bigarray kind.

    @raise Invalid_argument
      if dtype is an extended type not supported by standard Bigarray *)

val to_buffer_kind : ('a, 'b) t -> ('a, 'b) Nx_buffer.kind
(** [to_buffer_kind dtype] returns corresponding Nx_buffer kind. Works for all
    types including extended ones. *)

val of_bigarray_kind : ('a, 'b) Bigarray.kind -> ('a, 'b) t
(** [of_bigarray_kind kind] returns corresponding dtype from standard Bigarray
    kind. *)

(** {2 Equality} *)

val equal : ('a, 'b) t -> ('c, 'd) t -> bool
(** [equal dtype2 dtype2] tests dtype equality. *)

val equal_witness :
  ('a, 'b) t -> ('c, 'd) t -> (('a, 'b) t, ('c, 'd) t) Type.eq option
(** [equal_witness dtype2 dtype2] returns equality proof if dtypes match. *)

(** {2 Packed Types} *)

type packed =
  | Pack : ('a, 'b) t -> packed
      (** Existentially packed dtype that hides type parameters, allowing
          heterogeneous collections and runtime dtype manipulation. *)

val pack : ('a, 'b) t -> packed
(** [pack dtype] wraps a dtype in a packed container. *)

(** Operations on packed dtypes. *)
module Packed : sig
  type t = packed

  val all : t list
  (** [all] is a list of all available dtypes. *)

  val of_string : string -> t option
  (** [of_string s] returns the dtype matching string [s], if it exists. *)

  val to_string : t -> string
  (** [to_string t] returns the string representation. *)

  val pp : Format.formatter -> t -> unit
  (** [pp fmt t] pretty-prints the packed dtype. *)

  val equal : t -> t -> bool
  (** [equal t1 t2] tests equality. *)

  val compare : t -> t -> int
  (** [compare t1 t2] total ordering on packed dtypes. *)

  val hash : t -> int
  (** [hash t] returns a hash value suitable for hash tables. *)

  val tag : t -> int
  (** [tag t] returns the integer tag for this dtype. *)
end

(** {2 Operations} *)

val add : ('a, 'b) t -> 'a -> 'a -> 'a
val sub : ('a, 'b) t -> 'a -> 'a -> 'a
val mul : ('a, 'b) t -> 'a -> 'a -> 'a
val div : ('a, 'b) t -> 'a -> 'a -> 'a
