open Bigarray

module Bigstring : sig
  (** Represents a string outside the heap *)
  type t

  external of_string : string -> t = "hdf5_caml_struct_bigstring_of_string"
  external to_string : t -> string = "hdf5_caml_struct_bigstring_to_string"

  (** The given [Array1] must be [NULL] terminated *)
  val of_array1 : (char, int8_unsigned_elt, c_layout) Array1.t -> t
  val to_array1 : t -> (char, int8_unsigned_elt, c_layout) Array1.t
end

module Array_float32 : sig
  type t = (float, float32_elt, c_layout) Array1.t
end

module Array_float64 : sig
  type t = (float, float64_elt, c_layout) Array1.t
end

module Array_sint8 : sig
  type t = (int, int8_signed_elt, c_layout) Array1.t
end

module Array_uint8 : sig
  type t = (int, int8_unsigned_elt, c_layout) Array1.t
end

module Array_sint16 : sig
  type t = (int, int16_signed_elt, c_layout) Array1.t
end

module Array_uint16 : sig
  type t = (int, int16_unsigned_elt, c_layout) Array1.t
end

module Array_int32 : sig
  type t = (int32, int32_elt, c_layout) Array1.t
end

module Array_int64 : sig
  type t = (int64, int64_elt, c_layout) Array1.t
end

module Array_int : sig
  type t = (int, int_elt, c_layout) Array1.t
end

module Array_nativeint : sig
  type t = (nativeint, nativeint_elt, c_layout) Array1.t
end

module Array_char : sig
  type t = (char, int8_unsigned_elt, c_layout) Array1.t

  external of_string : string -> t = "hdf5_caml_struct_array_char_of_string"
  external to_string : t -> string = "hdf5_caml_struct_array_char_to_string"
end

module Unpacked : sig
  type 'a t =
  | Int             : int t
  | Int64           : int64 t
  | Float64         : float t
  | String          : int -> string t
  | Bigstring       : Bigstring.t t
  | Array_float32   : Array_float32.t t
  | Array_float64   : Array_float64.t t
  | Array_sint8     : Array_sint8.t t
  | Array_uint8     : Array_uint8.t t
  | Array_sint16    : Array_sint16.t t
  | Array_uint16    : Array_uint16.t t
  | Array_int32     : Array_int32.t t
  | Array_int64     : Array_int64.t t
  | Array_int       : Array_int.t t
  | Array_nativeint : Array_nativeint.t t
  | Array_char      : Array_char.t t

  val size : _ t -> int
end

type t = T : _ Unpacked.t -> t
