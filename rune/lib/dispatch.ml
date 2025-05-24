open Internal

let unary_op (type a b dev) ~cpu_op (x : (a, b, dev) data) : (a, b, dev) data =
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
    (y : (a, b, dev) data) : (int, Nx_core.uint8_elt, dev) data =
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
  match x with Cpu_data (x, ctx) -> Cpu_data (Backend_cpu.ones_like ctx x, ctx)

let full_like (type a b dev) value (x : (a, b, dev) data) : (a, b, dev) data =
  match x with
  | Cpu_data (x, ctx) -> Cpu_data (Backend_cpu.full_like ctx value x, ctx)

let fill (type a b dev) (value : a) (x : (a, b, dev) data) =
  match x with Cpu_data (x, ctx) -> Backend_cpu.fill ctx value x

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
  property_with_context ~cpu_op:(fun ctx -> Backend_cpu.get_item ctx indices) x

let set_item indices value x =
  property_with_context
    ~cpu_op:(fun ctx -> Backend_cpu.set_item ctx indices value)
    x

let get indices x =
  property_with_context ~cpu_op:(fun ctx -> Backend_cpu.get ctx indices) x

let set indices value x =
  property_with_context ~cpu_op:(fun ctx -> Backend_cpu.set ctx indices value) x

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
let pow_inplace x y = binary_op ~cpu_op:Backend_cpu.pow_inplace x y
let maximum x y = binary_op ~cpu_op:Backend_cpu.maximum x y
let maximum_inplace x y = binary_op ~cpu_op:Backend_cpu.maximum_inplace x y
let minimum x y = binary_op ~cpu_op:Backend_cpu.minimum x y
let minimum_inplace x y = binary_op ~cpu_op:Backend_cpu.minimum_inplace x y
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
let sum ?axes ?keepdims x = reduce_op ~cpu_op:Backend_cpu.sum ?axes ?keepdims x

let prod ?axes ?keepdims x =
  reduce_op ~cpu_op:Backend_cpu.prod ?axes ?keepdims x

let mean ?axes ?keepdims x =
  reduce_op ~cpu_op:Backend_cpu.mean ?axes ?keepdims x

let max ?axes ?keepdims x = reduce_op ~cpu_op:Backend_cpu.max ?axes ?keepdims x
let min ?axes ?keepdims x = reduce_op ~cpu_op:Backend_cpu.min ?axes ?keepdims x
let var ?axes ?keepdims x = reduce_op ~cpu_op:Backend_cpu.var ?axes ?keepdims x
let std ?axes ?keepdims x = reduce_op ~cpu_op:Backend_cpu.std ?axes ?keepdims x
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

let set_slice ?steps starts stops value x =
  ignore
    (binary_op
       ~cpu_op:(fun ctx value x ->
         Backend_cpu.set_slice ctx ?steps starts stops value x;
         x)
       value x)

let pad pad_width value x =
  unary_op ~cpu_op:(fun ctx x -> Backend_cpu.pad ctx pad_width value x) x

let expand_dims axis x =
  unary_op ~cpu_op:(fun ctx -> Backend_cpu.expand_dims ctx axis) x

let broadcast_to new_shape x =
  unary_op ~cpu_op:(fun ctx -> Backend_cpu.broadcast_to ctx new_shape) x

let astype (type a b c d dev) (dtype : (a, b) dtype) (x : (c, d, dev) data) :
    (a, b, dev) data =
  match x with
  | Cpu_data (v, ctx) -> Cpu_data (Backend_cpu.astype ctx dtype v, ctx)

let move (type a b dev_to dev_from) (device : dev_to device)
    (x : (a, b, dev_from) data) : (a, b, dev_to) data =
  match (x, device) with Cpu_data (v, _), Cpu ctx -> Cpu_data (v, ctx)

let where (type a b dev) (cond : (int, Nx_core.uint8_elt, dev) data)
    (x : (a, b, dev) data) (y : (a, b, dev) data) : (a, b, dev) data =
  match (cond, x, y) with
  | Cpu_data (cond, ctx_cond), Cpu_data (x, ctx_x), Cpu_data (y, ctx_y)
    when ctx_cond = ctx_x && ctx_cond = ctx_y ->
      Cpu_data (Backend_cpu.where ctx_x cond x y, ctx_x)
  | _ -> failwith "The three tensors must be on the same device"

let stack (type a b dev) ~axis (tensors : (a, b, dev) data list) :
    (a, b, dev) data =
  match tensors with
  | [] -> failwith "Cannot stack empty list"
  | hd :: _ -> (
      let data_list =
        List.map
          (fun (t : (a, b, dev) data) -> match t with Cpu_data (d, _) -> d)
          tensors
      in
      match hd with
      | Cpu_data (_, ctx) ->
          Cpu_data (Backend_cpu.stack ctx ~axis data_list, ctx))

let array_equal (type a b dev) (x : (a, b, dev) data) (y : (a, b, dev) data) :
    bool =
  match (x, y) with
  | Cpu_data (x, ctx_x), Cpu_data (y, ctx_y) when ctx_x = ctx_y ->
      Backend_cpu.array_equal ctx_x x y
  | _ -> failwith "The two tensors must be on the same device"

let pp fmt x =
  property_with_context ~cpu_op:(fun ctx -> Backend_cpu.pp ctx fmt) x

let to_string x =
  property_with_context ~cpu_op:(fun ctx -> Backend_cpu.to_string ctx) x

let print x = property_with_context ~cpu_op:(fun ctx -> Backend_cpu.print ctx) x
