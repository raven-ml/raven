open Bigarray

module Bigstring = struct
  type t = (char, int8_unsigned_elt, c_layout) Array1.t

  external of_string : string -> t = "hdf5_caml_struct_bigstring_of_string"
  external to_string : t -> string = "hdf5_caml_struct_bigstring_to_string"

  let of_array1 a =
    if a.{Array1.dim a - 1} <> '\000' then
      invalid_arg "The array must by NULL terminated";
    a

  let to_array1 t = t
end

module Array_float32 = struct
  type t = (float, float32_elt, c_layout) Array1.t
end

module Array_float64 = struct
  type t = (float, float64_elt, c_layout) Array1.t
end

module Array_sint8 = struct
  type t = (int, int8_signed_elt, c_layout) Array1.t
end

module Array_uint8 = struct
  type t = (int, int8_unsigned_elt, c_layout) Array1.t
end

module Array_sint16 = struct
  type t = (int, int16_signed_elt, c_layout) Array1.t
end

module Array_uint16 = struct
  type t = (int, int16_unsigned_elt, c_layout) Array1.t
end

module Array_int32 = struct
  type t = (int32, int32_elt, c_layout) Array1.t
end

module Array_int64 = struct
  type t = (int64, int64_elt, c_layout) Array1.t
end

module Array_int = struct
  type t = (int, int_elt, c_layout) Array1.t
end

module Array_nativeint = struct
  type t = (nativeint, nativeint_elt, c_layout) Array1.t
end

module Array_char = struct
  type t = (char, int8_unsigned_elt, c_layout) Array1.t

  external of_string : string -> t = "hdf5_caml_struct_array_char_of_string"
  external to_string : t -> string = "hdf5_caml_struct_array_char_to_string"
end

module Unpacked = struct
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

  let size : type a . a t -> int = function
    | Int             -> 8
    | Int64           -> 8
    | Float64         -> 8
    | Bigstring       -> 8
    | String l        -> l
    | Array_float32   -> 16
    | Array_float64   -> 16
    | Array_sint8     -> 16
    | Array_uint8     -> 16
    | Array_sint16    -> 16
    | Array_uint16    -> 16
    | Array_int32     -> 16
    | Array_int64     -> 16
    | Array_int       -> 16
    | Array_nativeint -> 16
    | Array_char      -> 16
end

type t = T : _ Unpacked.t -> t
