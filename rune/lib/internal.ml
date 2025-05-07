module Backend_cpu = Ndarray_core.Make (Ndarray_native)

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
type ('a, 'b) dtype = ('a, 'b) Ndarray_core.dtype

type ('a, 'b, 'c) data =
  | Cpu_data :
      ('a, 'b) Backend_cpu.t * Backend_cpu.context
      -> ('a, 'b, [ `cpu ]) data

type 'dev device = Cpu : Backend_cpu.context -> [ `cpu ] device

let cpu = Cpu (Backend_cpu.create_context ())

type ('a, 'b, 'dev) t = { id : int; data : ('a, 'b, 'dev) data }

type _ Effect.t +=
  | Const : ('a, 'b, 'dev) data -> ('a, 'b, 'dev) t Effect.t
  | Neg : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Exp : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Log : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Abs : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Sqrt : (float, 'b, 'dev) t -> (float, 'b, 'dev) t Effect.t
  | Sin : (float, 'b, 'dev) t -> (float, 'b, 'dev) t Effect.t
  | Cos : (float, 'b, 'dev) t -> (float, 'b, 'dev) t Effect.t
  | Add : ('a, 'b, 'dev) t * ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Sub : ('a, 'b, 'dev) t * ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Mul : ('a, 'b, 'dev) t * ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Div : ('a, 'b, 'dev) t * ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Maximum : ('a, 'b, 'dev) t * ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Minimum : ('a, 'b, 'dev) t * ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Matmul : ('a, 'b, 'dev) t * ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Transpose : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Sum :
      ('a, 'b, 'dev) t * int array option * bool option
      -> ('a, 'b, 'dev) t Effect.t
  | Mean :
      (float, 'b, 'dev) t * int array option * bool option
      -> (float, 'b, 'dev) t Effect.t
  | Max :
      ('a, 'b, 'dev) t * int array option * bool option
      -> ('a, 'b, 'dev) t Effect.t
  | Min :
      ('a, 'b, 'dev) t * int array option * bool option
      -> ('a, 'b, 'dev) t Effect.t
  | Reshape : ('a, 'b, 'dev) t * int array -> ('a, 'b, 'dev) t Effect.t
  | Slice :
      ('a, 'b, 'dev) t * int array * int array * int array option
      -> ('a, 'b, 'dev) t Effect.t
  | Cast : ('a, 'b) dtype * ('c, 'd, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Move :
      'dev_to device * ('a, 'b, 'dev_from) t
      -> ('a, 'b, 'dev_to) t Effect.t

let next_id = ref 0

let create_internal data =
  let id = !next_id in
  incr next_id;
  { id; data }

let const x =
  try Effect.perform (Const x) with Effect.Unhandled _ -> create_internal x

let cast_float (type a b) (dtype : (a, b) dtype) (v : float) : a =
  match dtype with
  | Float16 -> v
  | Float32 -> v
  | Float64 -> v
  | Int8 -> int_of_float v
  | Int16 -> int_of_float v
  | Int32 -> Int32.of_float v
  | Int64 -> Int64.of_float v
  | UInt8 -> int_of_float v
  | UInt16 -> int_of_float v
  | Complex32 -> Complex.{ re = v; im = 0.0 }
  | Complex64 -> Complex.{ re = v; im = 0.0 }
