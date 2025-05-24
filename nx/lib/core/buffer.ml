open Bigarray

type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt
type int8_elt = Bigarray.int8_signed_elt
type uint8_elt = Bigarray.int8_unsigned_elt
type int16_elt = Bigarray.int16_signed_elt
type uint16_elt = Bigarray.int16_unsigned_elt
type int32_elt = Bigarray.int32_elt
type int64_elt = Bigarray.int64_elt
type complex32_elt = Bigarray.complex32_elt
type complex64_elt = Bigarray.complex64_elt
type ('a, 'b) buffer = ('a, 'b, c_layout) Array1.t

let kind_of_dtype : type a b. (a, b) Descriptor.dtype -> (a, b) kind =
 fun dtype ->
  match dtype with
  | Descriptor.Float16 -> Float16
  | Descriptor.Float32 -> Float32
  | Descriptor.Float64 -> Float64
  | Descriptor.Int8 -> Int8_signed
  | Descriptor.UInt8 -> Int8_unsigned
  | Descriptor.Int16 -> Int16_signed
  | Descriptor.UInt16 -> Int16_unsigned
  | Descriptor.Int32 -> Int32
  | Descriptor.Int64 -> Int64
  | Descriptor.Complex32 -> Complex32
  | Descriptor.Complex64 -> Complex64

let dtype_of_kind : type a b. (a, b) kind -> (a, b) Descriptor.dtype = function
  | Bigarray.Float32 -> Float32
  | Bigarray.Float64 -> Float64
  | Bigarray.Int8_signed -> Int8
  | Bigarray.Int8_unsigned -> UInt8
  | Bigarray.Int16_signed -> Int16
  | Bigarray.Int16_unsigned -> UInt16
  | Bigarray.Int32 -> Int32
  | Bigarray.Int64 -> Int64
  | Bigarray.Complex32 -> Complex32
  | Bigarray.Complex64 -> Complex64
  | Bigarray.Float16 -> Float16
  | _ -> failwith "dtype_of_kind: Unsupported Bigarray kind"

let create_buffer : type a b. (a, b) Descriptor.dtype -> int -> (a, b) buffer =
 fun dtype size ->
  let kind = kind_of_dtype dtype in
  Array1.create kind c_layout size

let fill : type a b. a -> (a, b) buffer -> unit =
 fun value buffer -> Array1.fill buffer value

let blit : type a b. (a, b) buffer -> (a, b) buffer -> unit =
 fun src dst -> Array1.blit src dst

let length : type a b. (a, b) buffer -> int = fun buf -> Array1.dim buf

let size_in_bytes : type a b. (a, b) buffer -> int =
 fun buf -> Array1.size_in_bytes buf

let of_array : type a b. (a, b) Descriptor.dtype -> a array -> (a, b) buffer =
 fun dtype arr ->
  let kind = kind_of_dtype dtype in
  Array1.of_array kind c_layout arr
