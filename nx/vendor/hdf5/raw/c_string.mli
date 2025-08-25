open Bigarray

type t

val null : t
external of_string : string -> t = "hdf5_c_string_of_string"
external to_string : t -> string = "hdf5_c_string_to_string"
external to_bigstring : t -> (char, int8_unsigned_elt, c_layout) Array1.t
  = "hdf5_c_string_to_bigstring"
external free : t -> unit = "free"
