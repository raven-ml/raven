[@@@ocaml.warning "-32"]

module Backend_cpu = Ndarray_core.Make (Ndarray_cpu)

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

type ('a, 'b, 'c) data =
  | Cpu_data :
      ('a, 'b) Backend_cpu.t * Backend_cpu.context
      -> ('a, 'b, [ `cpu ]) data

type 'dev device = Cpu : Backend_cpu.context -> [ `cpu ] device
type ('a, 'b, 'dev) t = { id : int; data : ('a, 'b, 'dev) data }

type _ Effect.t +=
  | Const : ('a, 'b, 'dev) data -> ('a, 'b, 'dev) t Effect.t
  | Neg : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Exp : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
  | Log : ('a, 'b, 'dev) t -> ('a, 'b, 'dev) t Effect.t
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
  | Sum : ('a, 'b, 'dev) t * int array option -> ('a, 'b, 'dev) t Effect.t
  | Mean :
      (float, 'b, 'dev) t * int array option
      -> (float, 'b, 'dev) t Effect.t
  | Reshape : ('a, 'b, 'dev) t * int array -> ('a, 'b, 'dev) t Effect.t

module Dispatch = struct
  let unary_op (type a b dev) ~cpu_op (x : (a, b, dev) data) : (a, b, dev) data
      =
    match x with Cpu_data (x, ctx) -> Cpu_data (cpu_op ctx x, ctx)

  let unary_no_context_op (type a b dev) ~cpu_op (x : (a, b, dev) data) :
      (a, b, dev) data =
    match x with Cpu_data (x, ctx) -> Cpu_data (cpu_op x, ctx)

  let binary_op (type a b dev) ~cpu_op (x : (a, b, dev) data)
      (y : (a, b, dev) data) : (a, b, dev) data =
    match (x, y) with
    | Cpu_data (x, ctx_x), Cpu_data (y, ctx_y) when ctx_x = ctx_y ->
        Cpu_data (cpu_op ctx_x x y, ctx_x)
    | _ -> failwith "The two tensors must be on the same device"

  let binary_op_scalar (type a b dev) ~cpu_op (x : (a, b, dev) data) (y : a) :
      (a, b, dev) data =
    match x with Cpu_data (x, ctx_x) -> Cpu_data (cpu_op ctx_x x y, ctx_x)

  let compare_op (type a b dev) ~cpu_op (x : (a, b, dev) data)
      (y : (a, b, dev) data) : (int, Ndarray_core.uint8_elt, dev) data =
    match (x, y) with
    | Cpu_data (x, ctx_x), Cpu_data (y, ctx_y) when ctx_x = ctx_y ->
        Cpu_data (cpu_op ctx_x x y, ctx_x)
    | _ -> failwith "The two tensors must be on the same device"

  let reduce_op (type a b dev) ~cpu_op ?axes ?keepdims (x : (a, b, dev) data) :
      (a, b, dev) data =
    match x with
    | Cpu_data (x, ctx) -> Cpu_data (cpu_op ctx ?axes ?keepdims x, ctx)

  let logic_op (type a b dev) ~cpu_op (x : (a, b, dev) data)
      (y : (a, b, dev) data) : bool =
    match (x, y) with
    | Cpu_data (x, ctx_x), Cpu_data (y, ctx_y) when ctx_x = ctx_y ->
        cpu_op ctx_x x y
    | _ -> failwith "The two tensors must be on the same device"

  let property (type a b dev) ~cpu_op (x : (a, b, dev) data) =
    match x with Cpu_data (x, _ctx) -> cpu_op x

  let property_with_context (type a b dev) ~cpu_op (x : (a, b, dev) data) =
    match x with Cpu_data (x, ctx) -> cpu_op ctx x

  let empty_like (type a b dev) (x : (a, b, dev) data) : (a, b, dev) data =
    match x with
    | Cpu_data (x, ctx) -> Cpu_data (Backend_cpu.empty_like ctx x, ctx)

  let zeros_like (type a b dev) (x : (a, b, dev) data) : (a, b, dev) data =
    match x with
    | Cpu_data (x, ctx) -> Cpu_data (Backend_cpu.zeros_like ctx x, ctx)

  let ones_like (type a b dev) (x : (a, b, dev) data) : (a, b, dev) data =
    match x with
    | Cpu_data (x, ctx) -> Cpu_data (Backend_cpu.ones_like ctx x, ctx)

  let full_like (type a b dev) value (x : (a, b, dev) data) : (a, b, dev) data =
    match x with
    | Cpu_data (x, ctx) -> Cpu_data (Backend_cpu.full_like ctx value x, ctx)

  let data x = property ~cpu_op:Backend_cpu.data x
  let ndim x = property ~cpu_op:Backend_cpu.ndim x
  let shape x = property ~cpu_op:Backend_cpu.shape x
  let dim i x = property ~cpu_op:(Backend_cpu.dim i) x
  let dims x = property ~cpu_op:Backend_cpu.dims x
  let dtype x = property ~cpu_op:Backend_cpu.dtype x
  let nbytes x = property ~cpu_op:Backend_cpu.nbytes x
  let size x = property ~cpu_op:Backend_cpu.size x
  let strides x = property ~cpu_op:Backend_cpu.strides x
  let stride i x = property ~cpu_op:(Backend_cpu.stride i) x
  let itemsize x = property ~cpu_op:Backend_cpu.itemsize x
  let offset x = property ~cpu_op:Backend_cpu.offset x

  let get_item indices x =
    property_with_context
      ~cpu_op:(fun ctx -> Backend_cpu.get_item ctx indices)
      x

  let set_item indices value x =
    property_with_context
      ~cpu_op:(fun ctx -> Backend_cpu.set_item ctx indices value)
      x

  let get indices x =
    property_with_context ~cpu_op:(fun ctx -> Backend_cpu.get ctx indices) x

  let set indices value x =
    property_with_context
      ~cpu_op:(fun ctx -> Backend_cpu.set ctx indices value)
      x

  let add x y = binary_op ~cpu_op:Backend_cpu.add x y
  let add_inplace x y = binary_op ~cpu_op:Backend_cpu.add_inplace x y
  let add_scalar x y = binary_op_scalar ~cpu_op:Backend_cpu.add_scalar x y
  let sub x y = binary_op ~cpu_op:Backend_cpu.sub x y
  let sub_inplace x y = binary_op ~cpu_op:Backend_cpu.sub_inplace x y
  let sub_scalar x y = binary_op_scalar ~cpu_op:Backend_cpu.sub_scalar x y
  let mul x y = binary_op ~cpu_op:Backend_cpu.mul x y
  let mul_inplace x y = binary_op ~cpu_op:Backend_cpu.mul_inplace x y
  let mul_scalar x y = binary_op_scalar ~cpu_op:Backend_cpu.mul_scalar x y
  let div x y = binary_op ~cpu_op:Backend_cpu.div x y
  let div_inplace x y = binary_op ~cpu_op:Backend_cpu.div_inplace x y
  let div_scalar x y = binary_op_scalar ~cpu_op:Backend_cpu.div_scalar x y
  let rem x y = binary_op ~cpu_op:Backend_cpu.rem x y
  let rem_inplace x y = binary_op ~cpu_op:Backend_cpu.rem_inplace x y
  let rem_scalar x y = binary_op_scalar ~cpu_op:Backend_cpu.rem_scalar x y
  let pow x y = binary_op ~cpu_op:Backend_cpu.pow x y
  let pow_scalar x y = binary_op_scalar ~cpu_op:Backend_cpu.pow_scalar x y
  let pow_scalar x y = binary_op_scalar ~cpu_op:Backend_cpu.pow_scalar x y
  let maximum x y = binary_op ~cpu_op:Backend_cpu.maximum x y
  let maximum_inplace x y = binary_op ~cpu_op:Backend_cpu.maximum_inplace x y

  let maximum_scalar x y =
    binary_op_scalar ~cpu_op:Backend_cpu.maximum_scalar x y

  let minimum x y = binary_op ~cpu_op:Backend_cpu.minimum x y
  let minimum_inplace x y = binary_op ~cpu_op:Backend_cpu.minimum_inplace x y

  let minimum_scalar x y =
    binary_op_scalar ~cpu_op:Backend_cpu.minimum_scalar x y

  let exp x = unary_op ~cpu_op:Backend_cpu.exp x
  let log x = unary_op ~cpu_op:Backend_cpu.log x
  let abs x = unary_op ~cpu_op:Backend_cpu.abs x
  let neg x = unary_op ~cpu_op:Backend_cpu.neg x
  let sign x = unary_op ~cpu_op:Backend_cpu.sign x
  let sqrt x = unary_op ~cpu_op:Backend_cpu.sqrt x
  let square x = binary_op ~cpu_op:Backend_cpu.mul x x
  let sin x = unary_op ~cpu_op:Backend_cpu.sin x
  let cos x = unary_op ~cpu_op:Backend_cpu.cos x
  let tan x = unary_op ~cpu_op:Backend_cpu.tan x
  let asin x = unary_op ~cpu_op:Backend_cpu.asin x
  let acos x = unary_op ~cpu_op:Backend_cpu.acos x
  let atan x = unary_op ~cpu_op:Backend_cpu.atan x
  let sinh x = unary_op ~cpu_op:Backend_cpu.sinh x
  let cosh x = unary_op ~cpu_op:Backend_cpu.cosh x
  let tanh x = unary_op ~cpu_op:Backend_cpu.tanh x
  let asinh x = unary_op ~cpu_op:Backend_cpu.asinh x
  let acosh x = unary_op ~cpu_op:Backend_cpu.acosh x
  let atanh x = unary_op ~cpu_op:Backend_cpu.atanh x
  let equal x y = compare_op ~cpu_op:Backend_cpu.equal x y
  let greater x y = compare_op ~cpu_op:Backend_cpu.greater x y
  let greater_equal x y = compare_op ~cpu_op:Backend_cpu.greater_equal x y
  let less x y = compare_op ~cpu_op:Backend_cpu.less x y
  let less_equal x y = compare_op ~cpu_op:Backend_cpu.less_equal x y

  let sum ?axes ?keepdims x =
    reduce_op ~cpu_op:Backend_cpu.sum ?axes ?keepdims x

  let prod ?axes ?keepdims x =
    reduce_op ~cpu_op:Backend_cpu.prod ?axes ?keepdims x

  let mean ?axes ?keepdims x =
    reduce_op ~cpu_op:Backend_cpu.mean ?axes ?keepdims x

  let max ?axes ?keepdims x =
    reduce_op ~cpu_op:Backend_cpu.max ?axes ?keepdims x

  let min ?axes ?keepdims x =
    reduce_op ~cpu_op:Backend_cpu.min ?axes ?keepdims x

  let var ?axes ?keepdims x =
    reduce_op ~cpu_op:Backend_cpu.var ?axes ?keepdims x

  let std ?axes ?keepdims x =
    reduce_op ~cpu_op:Backend_cpu.std ?axes ?keepdims x

  let dot x y = binary_op ~cpu_op:Backend_cpu.dot x y
  let matmul x y = binary_op ~cpu_op:Backend_cpu.matmul x y

  let convolve ?mode x y =
    binary_op ~cpu_op:(fun ctx -> Backend_cpu.convolve1d ctx ?mode) x y

  let flatten x = unary_op ~cpu_op:Backend_cpu.flatten x
  let ravel x = unary_op ~cpu_op:Backend_cpu.ravel x

  let reshape shape x =
    unary_op ~cpu_op:(fun ctx x -> Backend_cpu.reshape ctx shape x) x

  let transpose ?axes x = unary_op ~cpu_op:(Backend_cpu.transpose ?axes) x
  let squeeze ?axes x = unary_op ~cpu_op:(Backend_cpu.squeeze ?axes) x

  let slice ?steps starts stops x =
    unary_op ~cpu_op:(fun ctx -> Backend_cpu.slice ctx ?steps starts stops) x

  let pad pad_width value x =
    unary_op ~cpu_op:(fun ctx x -> Backend_cpu.pad ctx pad_width value x) x

  let expand_dims axis x =
    unary_op ~cpu_op:(fun ctx -> Backend_cpu.expand_dims ctx axis) x

  let broadcast_to new_shape x =
    unary_op ~cpu_op:(fun ctx -> Backend_cpu.broadcast_to ctx new_shape) x

  let where (type a b dev) (cond : (int, Ndarray_core.uint8_elt, dev) data)
      (x : (a, b, dev) data) (y : (a, b, dev) data) : (a, b, dev) data =
    match (cond, x, y) with
    | Cpu_data (cond, ctx_cond), Cpu_data (x, ctx_x), Cpu_data (y, ctx_y)
      when ctx_cond = ctx_x && ctx_cond = ctx_y ->
        Cpu_data (Backend_cpu.where ctx_x cond x y, ctx_x)
    | _ -> failwith "The three tensors must be on the same device"

  let pp fmt x =
    property_with_context ~cpu_op:(fun ctx -> Backend_cpu.pp ctx fmt) x

  let to_string x =
    property_with_context ~cpu_op:(fun ctx -> Backend_cpu.to_string ctx) x

  let print x =
    property_with_context ~cpu_op:(fun ctx -> Backend_cpu.print ctx) x
end

let next_id = ref 0

let create_internal data =
  let id = !next_id in
  incr next_id;
  { id; data }

let eval_handler : ('a, 'a) Effect.Deep.handler =
  let open Effect.Deep in
  let handle_ap2 op x y k =
    let data = op x y in
    continue k (create_internal data)
  in

  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option = function
    | Const v -> Some (fun k -> continue k (create_internal v))
    | Neg x ->
        Some (fun k -> continue k (create_internal (Dispatch.neg x.data)))
    | Exp x ->
        Some (fun k -> continue k (create_internal (Dispatch.exp x.data)))
    | Log x ->
        Some (fun k -> continue k (create_internal (Dispatch.log x.data)))
    | Sin x ->
        Some (fun k -> continue k (create_internal (Dispatch.sin x.data)))
    | Cos x ->
        Some (fun k -> continue k (create_internal (Dispatch.cos x.data)))
    | Add (x, y) -> Some (handle_ap2 Dispatch.add x.data y.data)
    | Sub (x, y) -> Some (handle_ap2 Dispatch.sub x.data y.data)
    | Mul (x, y) -> Some (handle_ap2 Dispatch.mul x.data y.data)
    | Div (x, y) -> Some (handle_ap2 Dispatch.div x.data y.data)
    | Maximum (x, y) -> Some (handle_ap2 Dispatch.maximum x.data y.data)
    | Minimum (x, y) -> Some (handle_ap2 Dispatch.minimum x.data y.data)
    | Matmul (x, y) -> Some (handle_ap2 Dispatch.matmul x.data y.data)
    | Transpose x ->
        Some (fun k -> continue k (create_internal (Dispatch.transpose x.data)))
    | Sum (x, axes) ->
        Some (fun k -> continue k (create_internal (Dispatch.sum ?axes x.data)))
    | Mean (x, axes) ->
        Some
          (fun k -> continue k (create_internal (Dispatch.mean ?axes x.data)))
    | Reshape (x, shape) ->
        Some
          (fun k ->
            continue k (create_internal (Dispatch.reshape shape x.data)))
    | _ -> None
  in
  { retc = (fun x -> x); exnc = raise; effc }

let eval (type a b) (f : a -> b) (x : a) : b =
  Effect.Deep.match_with f x eval_handler

(* device *)

let device (type dev) (x : ('a, 'b, dev) t) : dev device =
  match x.data with Cpu_data (_, ctx) -> Cpu ctx

(* properties *)

let shape x = Dispatch.shape x.data
let dim i x = Dispatch.dim i x.data
let size x = Dispatch.size x.data
let dtype x = Dispatch.dtype x.data
let itemsize x = Dispatch.itemsize x.data
let nbytes x = Dispatch.nbytes x.data
let strides x = Dispatch.strides x.data
let stride i x = Dispatch.stride i x.data
let offset x = Dispatch.offset x.data
let ndim x = Dispatch.ndim x.data
let dims x = Dispatch.dims x.data

(* creation *)

let const x =
  try Effect.perform (Const x) with Effect.Unhandled _ -> create_internal x

let ndarray (type a b) (x : (a, b) Ndarray.t) =
  let context = Backend_cpu.create_context () in
  let buffer = Ndarray.data x in
  let core_dtype : (a, b) dtype =
    match Ndarray.dtype x with
    | Ndarray.Float16 -> float16
    | Ndarray.Float32 -> float32
    | Ndarray.Float64 -> float64
    | Ndarray.Int8 -> int8
    | Ndarray.Int16 -> int16
    | Ndarray.Int32 -> int32
    | Ndarray.Int64 -> int64
    | Ndarray.UInt8 -> uint8
    | Ndarray.UInt16 -> uint16
    | Ndarray.Complex32 -> complex32
    | Ndarray.Complex64 -> complex64
  in
  let core_layout =
    match Ndarray.layout x with
    | Ndarray.C_contiguous -> Ndarray_core.C_contiguous
    | Ndarray.Strided -> Ndarray_core.Strided
  in
  let descriptor =
    Ndarray_core.
      {
        dtype = core_dtype;
        shape = Ndarray.shape x;
        layout = core_layout;
        strides = Ndarray.strides x;
        offset = Ndarray.offset x;
      }
  in
  let data = Backend_cpu.from_buffer context descriptor buffer in
  const (Cpu_data (data, context))

let create dtype shape x =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.create context dtype shape x, context))

let create_on_device (type a b c) (device : c device) (dtype : (a, b) dtype)
    shape x : (a, b, c) t =
  match device with
  | Cpu ctx -> const (Cpu_data (Backend_cpu.create ctx dtype shape x, ctx))

let zeros dtype shape =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.zeros context dtype shape, context))

let ones dtype shape =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.ones context dtype shape, context))

let rand dtype shape =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.rand context dtype shape, context))

let scalar dtype x =
  let context = Backend_cpu.create_context () in
  const (Cpu_data (Backend_cpu.scalar context dtype x, context))

(* access *)

let get indices x = Dispatch.get_item indices x.data
let set indices value x = Dispatch.set_item indices value x.data

(* ops *)

let neg x =
  try Effect.perform (Neg x)
  with Effect.Unhandled _ -> create_internal (Dispatch.neg x.data)

let sin x =
  try Effect.perform (Sin x)
  with Effect.Unhandled _ -> create_internal (Dispatch.sin x.data)

let cos x =
  try Effect.perform (Cos x)
  with Effect.Unhandled _ -> create_internal (Dispatch.cos x.data)

let exp x =
  try Effect.perform (Exp x)
  with Effect.Unhandled _ -> create_internal (Dispatch.exp x.data)

let log x =
  try Effect.perform (Log x)
  with Effect.Unhandled _ -> create_internal (Dispatch.log x.data)

let add x y =
  try Effect.perform (Add (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.add x.data y.data)

let add_inplace x y =
  try Effect.perform (Add (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.add_inplace x.data y.data)

let sub x y =
  try Effect.perform (Sub (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.sub x.data y.data)

let sub_inplace x y =
  try Effect.perform (Sub (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.sub_inplace x.data y.data)

let mul x y =
  try Effect.perform (Mul (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.mul x.data y.data)

let mul_inplace x y =
  try Effect.perform (Mul (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.mul_inplace x.data y.data)

let div x y =
  try Effect.perform (Div (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.div x.data y.data)

let div_inplace x y =
  try Effect.perform (Div (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.div_inplace x.data y.data)

let maximum x y =
  try Effect.perform (Maximum (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.maximum x.data y.data)

let maximum_inplace x y =
  try Effect.perform (Maximum (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.maximum_inplace x.data y.data)

let minimum x y =
  try Effect.perform (Minimum (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.minimum x.data y.data)

let minimum_inplace x y =
  try Effect.perform (Minimum (x, y))
  with Effect.Unhandled _ ->
    create_internal (Dispatch.minimum_inplace x.data y.data)

let sum ?axes x =
  try Effect.perform (Sum (x, axes))
  with Effect.Unhandled _ -> create_internal (Dispatch.sum ?axes x.data)

let mean ?axes x =
  try Effect.perform (Mean (x, axes))
  with Effect.Unhandled _ -> create_internal (Dispatch.mean ?axes x.data)

let matmul x y =
  try Effect.perform (Matmul (x, y))
  with Effect.Unhandled _ -> create_internal (Dispatch.matmul x.data y.data)

let transpose x =
  try Effect.perform (Transpose x)
  with Effect.Unhandled _ -> create_internal (Dispatch.transpose x.data)

let reshape x shape =
  try Effect.perform (Reshape (x, shape))
  with Effect.Unhandled _ -> create_internal (Dispatch.reshape shape x.data)

(* utils *)

let pp fmt x = Dispatch.pp fmt x.data
let to_string x = Dispatch.to_string x.data
let print x = Dispatch.print x.data

(* nn *)

(* one-hot encoding: converts integer labels to one-hot vectors *)
let one_hot (type a b c d) (dtype : (a, b) dtype) (labels : (c, d, [ `cpu ]) t)
    depth =
  let input_shape = shape labels in
  let n = size labels in
  let labels_flat = reshape labels [| n |] in
  let oh_flat = zeros dtype [| n; depth |] in
  let lbl_dtype = Dispatch.dtype labels.data in
  for i = 0 to n - 1 do
    let idx : int =
      match lbl_dtype with
      | Int8 -> get [| i |] labels_flat
      | UInt8 -> get [| i |] labels_flat
      | Int16 -> get [| i |] labels_flat
      | UInt16 -> get [| i |] labels_flat
      | Int32 -> Int32.to_int (get [| i |] labels_flat)
      | Int64 -> Int64.to_int (get [| i |] labels_flat)
      | _ -> failwith "one_hot: labels must have integer dtype"
    in
    let one : a =
      match dtype with
      | Float16 -> 1.
      | Float32 -> 1.
      | Float64 -> 1.
      | Int8 -> 1
      | UInt8 -> 1
      | Int16 -> 1
      | UInt16 -> 1
      | Int32 -> Int32.of_int 1
      | Int64 -> Int64.of_int 1
      | Complex32 -> Complex.{ re = 1.; im = 0. }
      | Complex64 -> Complex.{ re = 1.; im = 0. }
    in
    set [| i; idx |] one oh_flat
  done;
  reshape oh_flat (Array.append input_shape [| depth |])

(* autodiff *)

type arg = L | R

let deriv_neg x =
  let ones = Dispatch.ones_like x.data in
  neg (const ones)

let deriv_sin x = cos x
let deriv_cos x = neg (sin x)
let deriv_exp x = exp x
let deriv_log x = div (const (Dispatch.ones_like x.data)) x

let deriv_add _ x _ =
  let ones = Dispatch.ones_like x.data in
  const ones

let deriv_sub arg x _ =
  let ones = Dispatch.ones_like x.data in
  match arg with L -> const ones | R -> neg (const ones)

let deriv_mul arg x y = match arg with L -> y | R -> x

let deriv_div arg x y =
  let ones = Dispatch.ones_like x.data in
  match arg with L -> div (const ones) y | R -> neg (div x (mul y y))

module Set_int = Set.Make (struct
  type t = int

  let compare = compare
end)

let compute_new_shape input_shape reduced_axes =
  let ndim = Array.length input_shape in
  let reduced_set =
    Array.fold_left
      (fun set ax -> Set_int.add ax set)
      Set_int.empty reduced_axes
  in
  Array.init ndim (fun i ->
      if Set_int.mem i reduced_set then 1 else input_shape.(i))

let deriv_sum x axes =
  let input_shape = Dispatch.shape x.data in
  fun twg_bv ->
    if axes = None then const (Dispatch.broadcast_to input_shape twg_bv.data)
    else
      let reduced_axes = Option.get axes in
      let new_shape = compute_new_shape input_shape reduced_axes in
      let reshaped_grad = Dispatch.reshape new_shape twg_bv.data in
      const (Dispatch.broadcast_to input_shape reshaped_grad)

let deriv_mean x axes =
  let input_shape = Dispatch.shape x.data in
  fun twg_bv ->
    let grad =
      if axes = None then Dispatch.broadcast_to input_shape twg_bv.data
      else
        let reduced_axes = Option.get axes in
        let new_shape = compute_new_shape input_shape reduced_axes in
        let reshaped_grad = Dispatch.reshape new_shape twg_bv.data in
        Dispatch.broadcast_to input_shape reshaped_grad
    in
    if axes = None then
      let n = Array.fold_left ( * ) 1 input_shape in
      const (Dispatch.div_scalar grad (float_of_int n))
    else
      let reduced_axes = Option.get axes in
      let reduced_sizes = Array.map (fun ax -> input_shape.(ax)) reduced_axes in
      let n = Array.fold_left ( * ) 1 reduced_sizes in
      const (Dispatch.div_scalar grad (float_of_int n))

let deriv_maximum arg x y =
  let t_ones = const (Dispatch.ones_like x.data) in
  let t_zeros = const (Dispatch.zeros_like x.data) in
  match arg with
  | L ->
      const
        (Dispatch.where
           (Dispatch.greater_equal x.data y.data)
           t_ones.data t_zeros.data)
  | R ->
      const
        (Dispatch.where
           (Dispatch.greater y.data x.data)
           t_ones.data t_zeros.data)

let deriv_minimum arg x y =
  let t_ones = const (Dispatch.ones_like x.data) in
  let t_zeros = const (Dispatch.zeros_like x.data) in
  match arg with
  | L ->
      const
        (Dispatch.where
           (Dispatch.less_equal x.data y.data)
           t_ones.data t_zeros.data)
  | R ->
      const
        (Dispatch.where (Dispatch.less y.data x.data) t_ones.data t_zeros.data)

type ('a, 'b, 'dev) t_with_grad = {
  v : ('a, 'b, 'dev) t;
  mutable bv : ('a, 'b, 'dev) t;
}

type any_t_with_grad =
  | Any_t_with_grad : ('a, 'b, 'dev) t_with_grad -> any_t_with_grad

let unwrap_t_with_grad (type a b dev) (_ : (a, b, dev) t)
    (any : any_t_with_grad) : (a, b, dev) t_with_grad =
  match any with Any_t_with_grad m -> Obj.magic m

let reduce_gradient grad_output input_shape output_shape =
  let ndim_output = Array.length output_shape in
  let ndim_input = Array.length input_shape in
  let padded_input_shape =
    Array.append (Array.make (ndim_output - ndim_input) 1) input_shape
  in
  let axes_to_sum =
    List.filter
      (fun i -> padded_input_shape.(i) = 1 && output_shape.(i) > 1)
      (List.init ndim_output (fun i -> i))
  in
  if axes_to_sum = [] then grad_output
  else
    let axes_array = Array.of_list axes_to_sum in
    const (Dispatch.sum grad_output.data ~axes:axes_array)

let make_reverse_handler tape =
  let open Effect.Deep in
  let effc : type a. a Effect.t -> ((a, _) continuation -> _) option =
    let handle_ap0 n k =
      let tensor = const n in
      let zeros = Dispatch.zeros_like tensor.data in
      let t_with_grad = { v = tensor; bv = const zeros } in
      Hashtbl.add tape tensor.id (Any_t_with_grad t_with_grad);
      continue k tensor
    in

    let handle_ap1 ~deriv ~op x k =
      let r = op x in
      let zeros = Dispatch.zeros_like r.data in
      let twg = { v = r; bv = const zeros } in
      Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
      (if not (Hashtbl.mem tape x.id) then
         let zeros_x = Dispatch.zeros_like x.data in
         let twg_x = { v = x; bv = const zeros_x } in
         Hashtbl.add tape x.id (Any_t_with_grad twg_x));
      let any_twg_x = Hashtbl.find tape x.id in
      let twg_x = unwrap_t_with_grad x any_twg_x in
      let t = continue k r in
      twg_x.bv <- add twg_x.bv (mul (deriv twg_x.v) twg.bv);
      t
    in

    let handle_ap2 ~deriv ~op x y k =
      let r = op x y in
      let zeros = Dispatch.zeros_like r.data in
      let twg = { v = r; bv = const zeros } in
      Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
      (if not (Hashtbl.mem tape x.id) then
         let zeros_x = Dispatch.zeros_like x.data in
         let twg_x = { v = x; bv = const zeros_x } in
         Hashtbl.add tape x.id (Any_t_with_grad twg_x));
      let any_twg_x = Hashtbl.find tape x.id in
      let twg_x = unwrap_t_with_grad x any_twg_x in
      (if not (Hashtbl.mem tape y.id) then
         let zeros_y = Dispatch.zeros_like y.data in
         let twg_y = { v = y; bv = const zeros_y } in
         Hashtbl.add tape y.id (Any_t_with_grad twg_y));
      let any_twg_y = Hashtbl.find tape y.id in
      let twg_y = unwrap_t_with_grad y any_twg_y in
      let t = continue k r in
      let grad_x = mul (deriv L x y) twg.bv in
      let grad_y = mul (deriv R x y) twg.bv in
      let reduced_grad_x =
        reduce_gradient grad_x (Dispatch.shape x.data) (Dispatch.shape r.data)
      in
      let reduced_grad_y =
        reduce_gradient grad_y (Dispatch.shape y.data) (Dispatch.shape r.data)
      in
      twg_x.bv <- add twg_x.bv reduced_grad_x;
      twg_y.bv <- add twg_y.bv reduced_grad_y;
      t
    in

    function
    | Const v -> Some (handle_ap0 v)
    | Neg x -> Some (handle_ap1 ~deriv:deriv_neg ~op:neg x)
    | Exp x -> Some (handle_ap1 ~deriv:deriv_exp ~op:exp x)
    | Log x -> Some (handle_ap1 ~deriv:deriv_log ~op:log x)
    | Sin x -> Some (handle_ap1 ~deriv:deriv_sin ~op:sin x)
    | Cos x -> Some (handle_ap1 ~deriv:deriv_cos ~op:cos x)
    | Add (x, y) -> Some (handle_ap2 ~deriv:deriv_add ~op:add x y)
    | Sub (x, y) -> Some (handle_ap2 ~deriv:deriv_sub ~op:sub x y)
    | Mul (x, y) -> Some (handle_ap2 ~deriv:deriv_mul ~op:mul x y)
    | Div (x, y) -> Some (handle_ap2 ~deriv:deriv_div ~op:div x y)
    | Maximum (x, y) -> Some (handle_ap2 ~deriv:deriv_maximum ~op:maximum x y)
    | Minimum (x, y) -> Some (handle_ap2 ~deriv:deriv_minimum ~op:minimum x y)
    | Sum (x, axes) ->
        Some
          (fun k ->
            let r = sum ?axes x in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let t = continue k r in
            let deriv_fn = deriv_sum x axes in
            twg_x.bv <- add twg_x.bv (deriv_fn twg.bv);
            t)
    | Mean (x, axes) ->
        Some
          (fun k ->
            let r = mean ?axes x in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let t = continue k r in
            let deriv_fn = deriv_mean x axes in
            twg_x.bv <- add twg_x.bv (deriv_fn twg.bv);
            t)
    | Matmul (x, y) ->
        Some
          (fun k ->
            let r = matmul x y in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            (if not (Hashtbl.mem tape y.id) then
               let zeros_y = Dispatch.zeros_like y.data in
               let twg_y = { v = y; bv = const zeros_y } in
               Hashtbl.add tape y.id (Any_t_with_grad twg_y));
            let any_twg_y = Hashtbl.find tape y.id in
            let twg_y = unwrap_t_with_grad y any_twg_y in
            let t = continue k r in
            twg_x.bv <- add twg_x.bv (matmul twg.bv (transpose twg_y.v));
            twg_y.bv <- add twg_y.bv (matmul (transpose twg_x.v) twg.bv);
            t)
    | Transpose x ->
        Some
          (fun k ->
            let r = transpose x in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            (if not (Hashtbl.mem tape x.id) then
               let zeros_x = Dispatch.zeros_like x.data in
               let twg_x = { v = x; bv = const zeros_x } in
               Hashtbl.add tape x.id (Any_t_with_grad twg_x));
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let t = continue k r in
            twg_x.bv <- add twg_x.bv (transpose twg.bv);
            t)
    | Reshape (x, shape) ->
        Some
          (fun k ->
            let original_shape = Dispatch.shape x.data in
            let r = reshape x shape in
            let zeros = Dispatch.zeros_like r.data in
            let twg = { v = r; bv = const zeros } in
            Hashtbl.add tape twg.v.id (Any_t_with_grad twg);
            let any_twg_x = Hashtbl.find tape x.id in
            let twg_x = unwrap_t_with_grad x any_twg_x in
            let t = continue k r in
            twg_x.bv <- add twg_x.bv (reshape twg.bv original_shape);
            t)
    | _ -> None
  in
  {
    retc =
      (fun x ->
        match Hashtbl.find_opt tape x.id with
        | Some any_m ->
            let m = unwrap_t_with_grad x any_m in
            let ones = Dispatch.ones_like x.data in
            m.bv <- const ones;
            x
        | None ->
            Printf.eprintf "Result tensor not found in tape\n";
            x);
    exnc = raise;
    effc;
  }

let grad f input_tensor =
  let tape = Hashtbl.create 10 in
  let zeros = Dispatch.zeros_like input_tensor.data in
  let m_input = { v = input_tensor; bv = const zeros } in
  Hashtbl.add tape input_tensor.id (Any_t_with_grad m_input);
  let handler = make_reverse_handler tape in
  let _ = Effect.Deep.match_with f input_tensor handler in
  let any_final_m_input = Hashtbl.find tape input_tensor.id in
  let final_m_input = unwrap_t_with_grad input_tensor any_final_m_input in
  final_m_input.bv

let grads f input_tensors =
  let tape = Hashtbl.create 10 in
  let input_twgs =
    List.map
      (fun input_tensor ->
        let zeros = Dispatch.zeros_like input_tensor.data in
        let twg = { v = input_tensor; bv = const zeros } in
        Hashtbl.add tape input_tensor.id (Any_t_with_grad twg);
        twg)
      input_tensors
  in
  let handler = make_reverse_handler tape in
  let _ = Effect.Deep.match_with f input_tensors handler in
  List.map (fun twg -> twg.bv) input_twgs

let value_and_grad f input_tensor =
  let tape = Hashtbl.create 10 in
  let zeros = Dispatch.zeros_like input_tensor.data in
  let m_input = { v = input_tensor; bv = const zeros } in
  Hashtbl.add tape input_tensor.id (Any_t_with_grad m_input);
  let handler = make_reverse_handler tape in
  let result_tensor = Effect.Deep.match_with f input_tensor handler in
  let any_final_m_input = Hashtbl.find tape input_tensor.id in
  let final_m_input = unwrap_t_with_grad input_tensor any_final_m_input in
  (result_tensor, final_m_input.bv)

let value_and_grads f input_tensors =
  let tape = Hashtbl.create 10 in
  let input_twgs =
    List.map
      (fun input_tensor ->
        let zeros = Dispatch.zeros_like input_tensor.data in
        let twg = { v = input_tensor; bv = const zeros } in
        Hashtbl.add tape input_tensor.id (Any_t_with_grad twg);
        twg)
      input_tensors
  in
  let handler = make_reverse_handler tape in
  let result_tensor = Effect.Deep.match_with f input_tensors handler in
  let grads = List.map (fun twg -> twg.bv) input_twgs in
  (result_tensor, grads)
