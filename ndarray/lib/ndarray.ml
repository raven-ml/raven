module B = Ndarray_core.Make (Ndarray_cpu)

type ('a, 'b) t = ('a, 'b) B.t
type layout = Ndarray_core.layout = C_contiguous | Strided

let context = Ndarray_cpu.create_context ()

(* Concrete types for dtypes *)

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

type float16_t = (float, float16_elt) t
type float32_t = (float, float32_elt) t
type float64_t = (float, float64_elt) t
type int8_t = (int, int8_elt) t
type int16_t = (int, int16_elt) t
type int32_t = (int32, int32_elt) t
type int64_t = (int64, int64_elt) t
type uint8_t = (int, uint8_elt) t
type uint16_t = (int, uint16_elt) t
type complex32_t = (Complex.t, complex32_elt) t
type complex64_t = (Complex.t, complex64_elt) t

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

(* Creation Function *)

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

(* Native Creation / Basic Ops *)

let empty dtype shape = B.empty context dtype shape
let copy t = B.copy context t
let blit src dst = B.blit context src dst
let fill value t = B.fill context value t

(* Range Generation *)

let arange dtype start stop step = B.arange context dtype start stop step
let arange_f dtype start stop step = B.arange_f context dtype start stop step

let linspace dtype ?endpoint start stop num =
  B.linspace context dtype ?endpoint start stop num

let logspace dtype ?endpoint ?base start stop num =
  B.logspace context dtype ?endpoint ?base start stop num

let geomspace dtype ?endpoint start stop num =
  B.geomspace context dtype ?endpoint start stop num

(* Property Access *)

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

(* Element-wise Binary Operations *)

let add t1 t2 = B.add context t1 t2

let add_inplace t1 t2 =
  let _ = B.add_inplace context t1 t2 in
  t1

let add_scalar s t = B.add_scalar context s t
let sub t1 t2 = B.sub context t1 t2

let sub_inplace t1 t2 =
  let _ = B.sub_inplace context t1 t2 in
  t1

let sub_scalar s t = B.sub_scalar context s t
let mul t1 t2 = B.mul context t1 t2

let mul_inplace t1 t2 =
  let _ = B.mul_inplace context t1 t2 in
  t1

let mul_scalar s t = B.mul_scalar context s t
let div t1 t2 = B.div context t1 t2

let div_inplace t1 t2 =
  let _ = B.div_inplace context t1 t2 in
  t1

let div_scalar s t = B.div_scalar context s t
let pow t1 t2 = B.pow context t1 t2

let pow_inplace t1 t2 =
  let _ = B.pow_inplace context t1 t2 in
  t1

let pow_scalar s t = B.pow_scalar context s t
let rem t1 t2 = B.rem context t1 t2

let rem_inplace t1 t2 =
  let _ = B.rem_inplace context t1 t2 in
  t1

let rem_scalar s t = B.rem_scalar context s t
let maximum t1 t2 = B.maximum context t1 t2

let maximum_inplace t1 t2 =
  let _ = B.maximum_inplace context t1 t2 in
  t1

let maximum_scalar s t = B.maximum_scalar context s t
let minimum t1 t2 = B.minimum context t1 t2

let minimum_inplace t1 t2 =
  let _ = B.minimum_inplace context t1 t2 in
  t1

let minimum_scalar s t = B.minimum_scalar context s t

(* Fused and special math *)

let fma t1 t2 t3 = B.fma context t1 t2 t3
let fma_inplace t1 t2 t3 = B.fma_inplace context t1 t2 t3

(* Comparison Operations *)

let equal t1 t2 = B.equal context t1 t2
let greater t1 t2 = B.greater context t1 t2
let greater_equal t1 t2 = B.greater_equal context t1 t2
let less t1 t2 = B.less context t1 t2
let less_equal t1 t2 = B.less_equal context t1 t2

(* Element-wise Unary Operations *)

let square t = B.square context t
let neg t = B.neg context t
let abs t = B.abs context t
let sign t = B.sign context t
let sqrt t = B.sqrt context t
let exp t = B.exp context t
let log t = B.log context t
let sin t = B.sin context t
let cos t = B.cos context t
let tan t = B.tan context t
let asin t = B.asin context t
let acos t = B.acos context t
let atan t = B.atan context t
let sinh t = B.sinh context t
let cosh t = B.cosh context t
let tanh t = B.tanh context t
let asinh t = B.asinh context t
let acosh t = B.acosh context t
let atanh t = B.atanh context t
let round t = B.round context t
let floor t = B.floor context t
let ceil t = B.ceil context t
let clip ~min ~max t = B.clip context ~min ~max t

(* Indexing *)

let where cond t2 t3 = B.where context cond t2 t3

(* Reductions *)

let sum ?axes ?keepdims t = B.sum context ?axes ?keepdims t
let prod ?axes ?keepdims t = B.prod context ?axes ?keepdims t
let max ?axes ?keepdims t = B.max context ?axes ?keepdims t
let min ?axes ?keepdims t = B.min context ?axes ?keepdims t

(* Linear Algebra *)

let matmul t1 t2 = B.matmul context t1 t2
let convolve1d ?mode t1 t2 = B.convolve1d context ?mode t1 t2
let dot t1 t2 = B.dot context t1 t2

(* Logic functions *)

let array_equal t1 t2 = B.array_equal context t1 t2

(* Interoperability *)

let to_bigarray t = B.to_bigarray context t
let of_bigarray ba = B.of_bigarray context ba
let to_array t = B.to_array context t

(* Random Generation *)

let rand dtype ?seed shape = B.rand context dtype ?seed shape
let randn dtype ?seed shape = B.randn context dtype ?seed shape

let randint dtype ?seed ?high shape low =
  B.randint context dtype ?seed ?high shape low

(* Element Access *)

let get_item indices t = B.get_item context indices t
let set_item indices v t = B.set_item context indices v t
let get indices t = B.get context indices t
let set indices t1 t2 = B.set context indices t1 t2

(* Higher-Order Function *)

let map f t = B.map context f t
let iter f t = B.iter context f t
let fold f acc t = B.fold context f acc t

(* Transformation *)

let reshape new_shape t = B.reshape context new_shape t
let flatten t = B.flatten context t
let ravel t = B.ravel context t
let pad padding value t = B.pad context padding value t
let transpose ?axes t = B.transpose context ?axes t
let broadcast_to new_shape t = B.broadcast_to context new_shape t
let squeeze ?axes t = B.squeeze context ?axes t
let expand_dims axis t = B.expand_dims context axis t
let slice ?steps starts stops t = B.slice context ?steps starts stops t

let set_slice ?steps starts stops value t =
  B.set_slice context ?steps starts stops value t

let astype dtype t = B.astype context dtype t
let array_split ?(axis = 0) sections t = B.array_split context ~axis sections t
let split ?(axis = 0) sections t = B.split context ~axis sections t

(* Stacking and concatenation *)

let concatenate ?axis ts = B.concatenate context ?axis ts
let stack ?axis ts = B.stack context ?axis ts
let vstack ts = B.vstack context ts
let hstack ts = B.hstack context ts
let dstack ts = B.dstack context ts

(* Shape and broadcast helpers *)

let broadcast_arrays ts = B.broadcast_arrays context ts
let tile reps t = B.tile context reps t
let repeat ?axis reps t = B.repeat context ?axis reps t

(* Reordering *)

let flip ?axes t = B.flip context ?axes t
let roll ?axis shift t = B.roll context ?axis shift t
let moveaxis src dst t = B.moveaxis context src dst t
let swapaxes i j t = B.swapaxes context i j t

(* Bitwise and logical *)

let bitwise_and t1 t2 = B.bitwise_and context t1 t2
let bitwise_or t1 t2 = B.bitwise_or context t1 t2
let bitwise_xor t1 t2 = B.bitwise_xor context t1 t2
let invert t = B.invert context t

(* Statistical and histogram *)

let mean ?axes ?keepdims t = B.mean context ?axes ?keepdims t
let var ?axes ?keepdims t = B.var context ?axes ?keepdims t
let std ?axes ?keepdims t = B.std context ?axes ?keepdims t

(* Linear algebra extras *)

let inv t = B.inv context t
let solve a b = B.solve context a b
let svd t = B.svd context t
let eig t = B.eig context t
let eigh t = B.eigh context t

(* Sorting and selection *)

let sort ?axis t = B.sort context ?axis t
let argsort ?axis t = B.argsort context ?axis t
let argmax ?axis t = B.argmax context ?axis t
let argmin ?axis t = B.argmin context ?axis t
let unique t = B.unique context t
let nonzero t = B.nonzero context t

(* Logical functions *)

let logical_and t1 t2 = B.logical_and context t1 t2
let logical_or t1 t2 = B.logical_or context t1 t2
let logical_not t = B.logical_not context t
let logical_xor t1 t2 = B.logical_xor context t1 t2

(* NaN/Inf handling *)

let isnan t = B.isnan context t
let isinf t = B.isinf context t
let isfinite t = B.isfinite context t

(* Formatting *)

let pp_dtype fmt dtype = B.pp_dtype context fmt dtype
let dtype_to_string dtype = B.dtype_to_string context dtype
let pp_shape fmt shape = B.pp_shape context fmt shape
let shape_to_string shape = B.shape_to_string context shape
let pp fmt t = B.pp context fmt t
let pp_info fmt t = B.pp_info context fmt t
let print t = B.print context t
let print_info t = B.print_info context t
let to_string t = B.to_string context t
let to_string_info t = B.to_string_info context t
