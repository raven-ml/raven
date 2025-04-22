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

type ('a, 'b) dtype = ('a, 'b) Ndarray_core.dtype =
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
let int16 = Int16
let int32 = Int32
let int64 = Int64
let uint8 = UInt8
let uint16 = UInt16
let complex32 = Complex32
let complex64 = Complex64

type 'dev device = 'dev Internal.device
type ('a, 'b, 'dev) t = ('a, 'b, 'dev) Internal.t

include Tensor
include Neural_net

let eval = Eval.eval
let grad = Autodiff.grad
let grads = Autodiff.grads
let value_and_grad = Autodiff.value_and_grad
let value_and_grads = Autodiff.value_and_grads
