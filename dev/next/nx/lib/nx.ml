module B = Nx_core.Make_frontend (Nx_native)

type ('a, 'b) t = ('a, 'b) B.t
type layout = Nx_core.View.layout = C_contiguous | Strided
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

type ('a, 'b) dtype = ('a, 'b) Nx_core.Dtype.t =
  | Float16 : (float, float16_elt) dtype
  | Float32 : (float, float32_elt) dtype
  | Float64 : (float, float64_elt) dtype
  | Int8 : (int, int8_elt) dtype
  | UInt8 : (int, uint8_elt) dtype
  | Int16 : (int, int16_elt) dtype
  | UInt16 : (int, uint16_elt) dtype
  | Int32 : (int32, int32_elt) dtype
  | Int64 : (int64, int64_elt) dtype
  | Int : (int, int_elt) dtype
  | NativeInt : (nativeint, nativeint_elt) dtype
  | Complex32 : (Complex.t, complex32_elt) dtype
  | Complex64 : (Complex.t, complex64_elt) dtype

type float16_t = (float, float16_elt) dtype
type float32_t = (float, float32_elt) dtype
type float64_t = (float, float64_elt) dtype
type int8_t = (int, int8_elt) dtype
type uint8_t = (int, uint8_elt) dtype
type int16_t = (int, int16_elt) dtype
type uint16_t = (int, uint16_elt) dtype
type int32_t = (int32, int32_elt) dtype
type int64_t = (int64, int64_elt) dtype
type std_int_t = (int, int_elt) dtype
type std_nativeint_t = (nativeint, nativeint_elt) dtype
type complex32_t = (Complex.t, complex32_elt) dtype
type complex64_t = (Complex.t, complex64_elt) dtype

let float16 = Float16
let float32 = Float32
let float64 = Float64
let int8 = Int8
let uint8 = UInt8
let int16 = Int16
let uint16 = UInt16
let int32 = Int32
let int64 = Int64
let int = Int
let nativeint = NativeInt
let complex32 = Complex32
let complex64 = Complex64

(* context *)

let context = B.create_context ()

(* creation functions *)

let create dtype shape arr = B.create context dtype shape arr
let init dtype shape f = B.init context dtype shape f
let full dtype shape value = B.full context dtype shape value
let ones dtype shape = B.ones context dtype shape
let zeros dtype shape = B.zeros context dtype shape
let ones_like t = B.ones_like context t
let zeros_like t = B.zeros_like context t
let empty_like t = B.empty_like context t
let full_like value t = B.full_like context value t
let scalar dtype v = B.scalar context dtype v
let eye ?m ?k dtype n = B.eye context ?m ?k dtype n
let identity dtype n = B.identity context dtype n
let empty dtype shape = B.empty context dtype shape
let copy t = B.copy context t
let blit src dst = B.blit context src dst
let fill value t = B.fill context value t

(* range *)

let arange dtype start stop step = B.arange context dtype start stop step
let arange_f dtype start stop step = B.arange_f context dtype start stop step

let linspace dtype ?endpoint start stop num =
  B.linspace context dtype ?endpoint start stop num

let logspace dtype ?endpoint ?base start stop num =
  B.logspace context dtype ?endpoint ?base start stop num

let geomspace dtype ?endpoint start stop num =
  B.geomspace context dtype ?endpoint start stop num

(* accessors *)

let data t = B.buffer t
let shape t = B.shape t
let dtype t = B.dtype t
let strides t = B.strides t
let stride i t = B.stride i t
let dims t = B.dims t
let dim i t = B.dim i t
let ndim t = B.ndim t
let itemsize t = B.itemsize t
let size t = B.size t
let nbytes t = B.nbytes t
let offset t = B.offset t
let layout t = B.layout t

(* element-wise *)

let add a b = B.add context a b
let mul a b = B.mul context a b

(* reductions *)

let sum ?axes ?(keepdims = false) x = B.sum context ?axes ~keepdims x

(* transformations *)

let broadcast_to x target_shape = B.broadcast_to context x target_shape
let reshape t new_shape = B.reshape context t new_shape
let expand t new_shape = B.expand context t new_shape
