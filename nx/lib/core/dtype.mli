(** Data types for tensor elements. *)

(** {2 Element Types} *)

type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt
type int8_elt = Bigarray.int8_signed_elt
type uint8_elt = Bigarray.int8_unsigned_elt
type int16_elt = Bigarray.int16_signed_elt
type uint16_elt = Bigarray.int16_unsigned_elt
type int32_elt = Bigarray.int32_elt
type int64_elt = Bigarray.int64_elt
type int_elt = Bigarray.int_elt
type nativeint_elt = Bigarray.nativeint_elt
type complex32_elt = Bigarray.complex32_elt
type complex64_elt = Bigarray.complex64_elt

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
  | Int : (int, int_elt) t
  | NativeInt : (nativeint, nativeint_elt) t
  | Complex32 : (Complex.t, complex32_elt) t
  | Complex64 : (Complex.t, complex64_elt) t

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
val int : (int, int_elt) t
val nativeint : (nativeint, nativeint_elt) t
val complex32 : (Complex.t, complex32_elt) t
val complex64 : (Complex.t, complex64_elt) t

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

val to_bigarray_kind : ('a, 'b) t -> ('a, 'b) Bigarray.kind
(** [to_bigarray_kind dtype] returns corresponding Bigarray kind. *)

val of_bigarray_kind : ('a, 'b) Bigarray.kind -> ('a, 'b) t
(** [of_bigarray_kind kind] returns corresponding dtype.

    @raise Failure if kind is unsupported *)

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
