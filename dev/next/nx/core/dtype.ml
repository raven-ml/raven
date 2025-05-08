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

type ('a, 'b) dtype =
  | Float16 : (float, float16_elt) dtype
  | Float32 : (float, float32_elt) dtype
  | Float64 : (float, float64_elt) dtype
  | Int8 : (int, int8_elt) dtype
  | Int16 : (int, int16_elt) dtype
  | Int32 : (int32, int32_elt) dtype
  | Int64 : (int64, int64_elt) dtype
  | UInt8 : (int, uint8_elt) dtype
  | UInt16 : (int, uint16_elt) dtype
  | Complex32 : (Complex.t, complex32_elt) dtype
  | Complex64 : (Complex.t, complex64_elt) dtype

let float16 = Float16
let float32 = Float32
let float64 = Float64
let int8 = Int8
let uint8 = UInt8
let int16 = Int16
let uint16 = UInt16
let int32 = Int32
let int64 = Int64
let complex32 = Complex32
let complex64 = Complex64

let zero : type a b. (a, b) dtype -> a =
 fun dtype ->
  match dtype with
  | Float16 -> 0.0
  | Float32 -> 0.0
  | Float64 -> 0.0
  | Int8 -> 0
  | UInt8 -> 0
  | Int16 -> 0
  | UInt16 -> 0
  | Int32 -> 0l
  | Int64 -> 0L
  | Complex32 -> Complex.zero
  | Complex64 -> Complex.zero

let one : type a b. (a, b) dtype -> a =
 fun dtype ->
  match dtype with
  | Float16 -> 1.0
  | Float32 -> 1.0
  | Float64 -> 1.0
  | Int8 -> 1
  | UInt8 -> 1
  | Int16 -> 1
  | UInt16 -> 1
  | Int32 -> 1l
  | Int64 -> 1L
  | Complex32 -> Complex.one
  | Complex64 -> Complex.one

let itemsize : type a b. (a, b) dtype -> int = function
  | Float16 -> 2
  | Float32 -> 4
  | Float64 -> 8
  | Int8 -> 1
  | UInt8 -> 1
  | Int16 -> 2
  | UInt16 -> 2
  | Int32 -> 4
  | Int64 -> 8
  | Complex32 -> 8
  | Complex64 -> 16

let kind_of_dtype : type a b. (a, b) dtype -> (a, b) Bigarray.kind =
 fun dtype ->
  match dtype with
  | Float16 -> Bigarray.Float16
  | Float32 -> Bigarray.Float32
  | Float64 -> Bigarray.Float64
  | Int8 -> Bigarray.Int8_signed
  | UInt8 -> Bigarray.Int8_unsigned
  | Int16 -> Bigarray.Int16_signed
  | UInt16 -> Bigarray.Int16_unsigned
  | Int32 -> Bigarray.Int32
  | Int64 -> Bigarray.Int64
  | Complex32 -> Bigarray.Complex32
  | Complex64 -> Bigarray.Complex64

let dtype_of_kind : type a b. (a, b) Bigarray.kind -> (a, b) dtype = function
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
