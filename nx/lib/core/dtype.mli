(** Data types for tensor elements. *)

(** {2 Element Types} *)

type float16_elt = Bigarray_ext.float16_elt
type float32_elt = Bigarray_ext.float32_elt
type float64_elt = Bigarray_ext.float64_elt
type int8_elt = Bigarray_ext.int8_signed_elt
type uint8_elt = Bigarray_ext.int8_unsigned_elt
type int16_elt = Bigarray_ext.int16_signed_elt
type uint16_elt = Bigarray_ext.int16_unsigned_elt
type int32_elt = Bigarray_ext.int32_elt
type int64_elt = Bigarray_ext.int64_elt
type complex32_elt = Bigarray_ext.complex32_elt
type complex64_elt = Bigarray_ext.complex64_elt

(* Extended types from Bigarray_ext *)
type uint32_elt = Bigarray_ext.uint32_elt
type uint64_elt = Bigarray_ext.uint64_elt
type bfloat16_elt = Bigarray_ext.bfloat16_elt
type bool_elt = Bigarray_ext.bool_elt
type int4_elt = Bigarray_ext.int4_signed_elt
type uint4_elt = Bigarray_ext.int4_unsigned_elt
type float8_e4m3_elt = Bigarray_ext.float8_e4m3_elt
type float8_e5m2_elt = Bigarray_ext.float8_e5m2_elt

(** {2 Data Type} *)

(** GADT representing tensor element types. First type parameter is the OCaml
    type, second is the Bigarray element type. *)
type ('a, 'b) t =
  | Float16 : (float, float16_elt) t
  | Float32 : (float, float32_elt) t
  | Float64 : (float, float64_elt) t
  | Int8 : (int, int8_elt) t
  | UInt8 : (int, uint8_elt) t
  | Int16 : (int, int16_elt) t
  | UInt16 : (int, uint16_elt) t
  | Int32 : (int32, int32_elt) t
  | Int64 : (int64, int64_elt) t
  | UInt32 : (int32, uint32_elt) t
  | UInt64 : (int64, uint64_elt) t
  | Complex64 : (Complex.t, complex32_elt) t
  | Complex128 : (Complex.t, complex64_elt) t
  (* Extended types *)
  | BFloat16 : (float, bfloat16_elt) t
  | Bool : (bool, bool_elt) t
  | Int4 : (int, int4_elt) t
  | UInt4 : (int, uint4_elt) t
  | Float8_e4m3 : (float, float8_e4m3_elt) t
  | Float8_e5m2 : (float, float8_e5m2_elt) t

(** {2 Constructors} *)

val float16 : (float, float16_elt) t
val float32 : (float, float32_elt) t
val float64 : (float, float64_elt) t
val int8 : (int, int8_elt) t
val uint8 : (int, uint8_elt) t
val int16 : (int, int16_elt) t
val uint16 : (int, uint16_elt) t
val int32 : (int32, int32_elt) t
val int64 : (int64, int64_elt) t
val uint32 : (int32, uint32_elt) t
val uint64 : (int64, uint64_elt) t
val complex64 : (Complex.t, complex32_elt) t
val complex128 : (Complex.t, complex64_elt) t

(* Extended types *)
val bfloat16 : (float, bfloat16_elt) t
val bool : (bool, bool_elt) t
val int4 : (int, int4_elt) t
val uint4 : (int, uint4_elt) t
val float8_e4m3 : (float, float8_e4m3_elt) t
val float8_e5m2 : (float, float8_e5m2_elt) t

(** {2 Properties} *)

val to_string : ('a, 'b) t -> string
(** [to_string dtype] returns string representation. *)

val pp : Format.formatter -> ('a, 'b) t -> unit
(** [pp fmt dtype] pretty-prints the dtype. *)

val itemsize : ('a, 'b) t -> int
(** [itemsize dtype] returns size in bytes. *)

(** {2 Type Classes} *)

val is_float : ('a, 'b) t -> bool
(** [is_float dtype] returns true for floating-point types. *)

val is_complex : ('a, 'b) t -> bool
(** [is_complex dtype] returns true for complex types. *)

val is_int : ('a, 'b) t -> bool
(** [is_int dtype] returns true for signed integer types. *)

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

val of_bigarray_ext_kind : ('a, 'b) Bigarray_ext.kind -> ('a, 'b) t
(** [of_bigarray_ext_kind kind] returns corresponding dtype from Bigarray_ext
    kind. *)

val to_bigarray_kind : ('a, 'b) t -> ('a, 'b) Bigarray.kind
(** [to_bigarray_kind dtype] returns corresponding standard Bigarray kind.

    @raise Failure
      if dtype is an extended type not supported by standard Bigarray *)

val to_bigarray_ext_kind : ('a, 'b) t -> ('a, 'b) Bigarray_ext.kind
(** [to_bigarray_ext_kind dtype] returns corresponding Bigarray_ext kind. Works
    for all types including extended ones. *)

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
      (** Packed dtype that hides type parameters, allowing heterogeneous
          collections. *)

val pack : ('a, 'b) t -> packed
(** [pack dtype] wraps a dtype in a packed container. *)

val all_dtypes : packed list
(** [all_dtypes] is a list of all available dtypes. *)

val of_string : string -> packed option
(** [of_string s] returns the dtype with string representation [s], if it
    exists. *)

val equal_packed : packed -> packed -> bool
(** [equal_packed p1 p2] tests equality of packed dtypes. *)

val pp_packed : Format.formatter -> packed -> unit
(** [pp_packed fmt packed] pretty-prints a packed dtype. *)

val packed_to_string : packed -> string
(** [packed_to_string packed] returns string representation of packed dtype. *)

(** {2 Operations} *)

val add : ('a, 'b) t -> 'a -> 'a -> 'a
val sub : ('a, 'b) t -> 'a -> 'a -> 'a
val mul : ('a, 'b) t -> 'a -> 'a -> 'a
val div : ('a, 'b) t -> 'a -> 'a -> 'a
