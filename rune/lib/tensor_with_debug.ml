module T = Tensor

let ctx = Nx_rune.create_context ()

(* Creation operations *)

let full dt target_shape fill_value =
  Debug.with_context "full" (fun () -> T.full ctx dt target_shape fill_value)

let zeros dtype shape_arr =
  Debug.with_context "zeros" (fun () -> T.zeros ctx dtype shape_arr)

let ones dtype shape_arr =
  Debug.with_context "ones" (fun () -> T.ones ctx dtype shape_arr)

let empty dtype shape_arr =
  Debug.with_context "empty" (fun () -> T.empty ctx dtype shape_arr)

let scalar dtype value =
  Debug.with_context "scalar" (fun () -> T.scalar ctx dtype value)

let create dtype shape data =
  Debug.with_context "create" (fun () -> T.create ctx dtype shape data)

let init dtype shape f =
  Debug.with_context "init" (fun () -> T.init ctx dtype shape f)

(* Random operations *)

let rand dtype ?seed shape =
  Debug.with_context "rand" (fun () -> T.rand ctx dtype ?seed shape)

let randn dtype ?seed shape =
  Debug.with_context "randn" (fun () -> T.randn ctx dtype ?seed shape)

(* Binary operations *)

let add a b = Debug.with_context "add" (fun () -> T.add a b)
let sub a b = Debug.with_context "sub" (fun () -> T.sub a b)
let mul a b = Debug.with_context "mul" (fun () -> T.mul a b)
let div a b = Debug.with_context "div" (fun () -> T.div a b)
let pow a b = Debug.with_context "pow" (fun () -> T.pow a b)
let mod_ a b = Debug.with_context "mod" (fun () -> T.mod_ a b)
let maximum a b = Debug.with_context "maximum" (fun () -> T.maximum a b)
let minimum a b = Debug.with_context "minimum" (fun () -> T.minimum a b)

(* Unary operations *)

let neg x = Debug.with_context "neg" (fun () -> T.neg x)
let abs x = Debug.with_context "abs" (fun () -> T.abs x)
let sign x = Debug.with_context "sign" (fun () -> T.sign x)
let square x = Debug.with_context "square" (fun () -> T.square x)
let sqrt x = Debug.with_context "sqrt" (fun () -> T.sqrt x)
let recip x = Debug.with_context "recip" (fun () -> T.recip x)
let exp x = Debug.with_context "exp" (fun () -> T.exp x)
let log x = Debug.with_context "log" (fun () -> T.log x)
let sin x = Debug.with_context "sin" (fun () -> T.sin x)
let cos x = Debug.with_context "cos" (fun () -> T.cos x)
let tan x = Debug.with_context "tan" (fun () -> T.tan x)

(* Shape operations *)

let reshape shape x = Debug.with_context "reshape" (fun () -> T.reshape shape x)
let expand shape x = Debug.with_context "expand" (fun () -> T.expand shape x)
let squeeze ?axes x = Debug.with_context "squeeze" (fun () -> T.squeeze ?axes x)

let unsqueeze ?axes x =
  Debug.with_context "unsqueeze" (fun () -> T.unsqueeze ?axes x)

let transpose ?axes x =
  Debug.with_context "transpose" (fun () -> T.transpose ?axes x)

let flip ?axes x = Debug.with_context "flip" (fun () -> T.flip ?axes x)

(* Reduction operations *)

let sum ?axes ?keepdims x =
  Debug.with_context "sum" (fun () -> T.sum ?axes ?keepdims x)

let max ?axes ?keepdims x =
  Debug.with_context "max" (fun () -> T.max ?axes ?keepdims x)

let min ?axes ?keepdims x =
  Debug.with_context "min" (fun () -> T.min ?axes ?keepdims x)

let mean ?axes ?keepdims x =
  Debug.with_context "mean" (fun () -> T.mean ?axes ?keepdims x)

let prod ?axes ?keepdims x =
  Debug.with_context "prod" (fun () -> T.prod ?axes ?keepdims x)

(* Linear algebra *)

let matmul a b = Debug.with_context "matmul" (fun () -> T.matmul a b)
let dot a b = Debug.with_context "dot" (fun () -> T.dot a b)

(* Additional Linear Algebra Operations *)

let diagonal ?offset ?axis1 ?axis2 a =
  Debug.with_context "diagonal" (fun () -> T.diagonal ?offset ?axis1 ?axis2 a)

let matrix_transpose a =
  Debug.with_context "matrix_transpose" (fun () -> T.matrix_transpose a)

let vdot a b = Debug.with_context "vdot" (fun () -> T.vdot a b)

let vecdot ?axis a b =
  Debug.with_context "vecdot" (fun () -> T.vecdot ?axis a b)

let inner a b = Debug.with_context "inner" (fun () -> T.inner a b)
let outer a b = Debug.with_context "outer" (fun () -> T.outer a b)

let tensordot ?axes a b =
  Debug.with_context "tensordot" (fun () -> T.tensordot ?axes a b)

let einsum subscripts operands =
  Debug.with_context "einsum" (fun () -> T.einsum subscripts operands)

let kron a b = Debug.with_context "kron" (fun () -> T.kron a b)

let multi_dot arrays =
  Debug.with_context "multi_dot" (fun () -> T.multi_dot arrays)

let matrix_power a n =
  Debug.with_context "matrix_power" (fun () -> T.matrix_power a n)

let cross ?axis a b = Debug.with_context "cross" (fun () -> T.cross ?axis a b)

(* Matrix Decompositions *)

let cholesky ?upper a =
  Debug.with_context "cholesky" (fun () -> T.cholesky ?upper a)

let qr ?mode a = Debug.with_context "qr" (fun () -> T.qr ?mode a)

let svd ?full_matrices a =
  Debug.with_context "svd" (fun () -> T.svd ?full_matrices a)

let svdvals a = Debug.with_context "svdvals" (fun () -> T.svdvals a)

(* Eigenvalues and Eigenvectors *)

let eig a = Debug.with_context "eig" (fun () -> T.eig a)
let eigh ?uplo a = Debug.with_context "eigh" (fun () -> T.eigh ?uplo a)
let eigvals a = Debug.with_context "eigvals" (fun () -> T.eigvals a)

let eigvalsh ?uplo a =
  Debug.with_context "eigvalsh" (fun () -> T.eigvalsh ?uplo a)

(* Norms and Condition Numbers *)

let norm ?ord ?axes ?keepdims x =
  Debug.with_context "norm" (fun () -> T.norm ?ord ?axes ?keepdims x)

let cond ?p x = Debug.with_context "cond" (fun () -> T.cond ?p x)
let det a = Debug.with_context "det" (fun () -> T.det a)
let slogdet a = Debug.with_context "slogdet" (fun () -> T.slogdet a)

let matrix_rank ?tol ?rtol ?hermitian a =
  Debug.with_context "matrix_rank" (fun () ->
      T.matrix_rank ?tol ?rtol ?hermitian a)

let trace ?offset a = Debug.with_context "trace" (fun () -> T.trace ?offset a)

(* Solving Linear Systems *)

let solve a b = Debug.with_context "solve" (fun () -> T.solve a b)
let lstsq ?rcond a b = Debug.with_context "lstsq" (fun () -> T.lstsq ?rcond a b)
let inv a = Debug.with_context "inv" (fun () -> T.inv a)

let pinv ?rtol ?hermitian a =
  Debug.with_context "pinv" (fun () -> T.pinv ?rtol ?hermitian a)

let tensorsolve ?axes a b =
  Debug.with_context "tensorsolve" (fun () -> T.tensorsolve ?axes a b)

let tensorinv ?ind a =
  Debug.with_context "tensorinv" (fun () -> T.tensorinv ?ind a)

(* FFT *)

let fft ?axis ?n ?(norm = `Backward) x =
  Debug.with_context "fft" (fun () -> T.fft ?axis ?n ~norm x)

let ifft ?axis ?n ?(norm = `Backward) x =
  Debug.with_context "ifft" (fun () -> T.ifft ?axis ?n ~norm x)

let fft2 ?axes ?s ?(norm = `Backward) x =
  Debug.with_context "fft2" (fun () -> T.fft2 ?axes ?s ~norm x)

let ifft2 ?axes ?s ?(norm = `Backward) x =
  Debug.with_context "ifft2" (fun () -> T.ifft2 ?axes ?s ~norm x)

let fftn ?axes ?s ?(norm = `Backward) x =
  Debug.with_context "fftn" (fun () -> T.fftn ?axes ?s ~norm x)

let ifftn ?axes ?s ?(norm = `Backward) x =
  Debug.with_context "ifftn" (fun () -> T.ifftn ?axes ?s ~norm x)

let rfft ?axis ?n ?(norm = `Backward) x =
  Debug.with_context "rfft" (fun () -> T.rfft ?axis ?n ~norm x)

let irfft ?axis ?n ?(norm = `Backward) x =
  Debug.with_context "irfft" (fun () -> T.irfft ?axis ?n ~norm x)

let rfft2 ?axes ?s ?(norm = `Backward) x =
  Debug.with_context "rfft2" (fun () -> T.rfft2 ?axes ?s ~norm x)

let irfft2 ?axes ?s ?(norm = `Backward) x =
  Debug.with_context "irfft2" (fun () -> T.irfft2 ?axes ?s ~norm x)

let rfftn ?axes ?s ?(norm = `Backward) x =
  Debug.with_context "rfftn" (fun () -> T.rfftn ?axes ?s ~norm x)

let irfftn ?axes ?s ?(norm = `Backward) x =
  Debug.with_context "irfftn" (fun () -> T.irfftn ?axes ?s ~norm x)

let hfft ?axis ?n ?norm x =
  Debug.with_context "hfft" (fun () -> T.hfft ?axis ?n ?norm x)

let ihfft ?axis ?n ?norm x =
  Debug.with_context "ihfft" (fun () -> T.ihfft ?axis ?n ?norm x)

let fftfreq ?d n = Debug.with_context "fftfreq" (fun () -> T.fftfreq ctx ?d n)

let rfftfreq ?d n =
  Debug.with_context "rfftfreq" (fun () -> T.rfftfreq ctx ?d n)

let fftshift ?axes x =
  Debug.with_context "fftshift" (fun () -> T.fftshift ?axes x)

let ifftshift ?axes x =
  Debug.with_context "ifftshift" (fun () -> T.ifftshift ?axes x)

(* Other operations *)

let cast dtype x = Debug.with_context "cast" (fun () -> T.cast dtype x)
let contiguous x = Debug.with_context "contiguous" (fun () -> T.contiguous x)
let copy x = Debug.with_context "copy" (fun () -> T.copy x)

let where cond if_true if_false =
  Debug.with_context "where" (fun () -> T.where cond if_true if_false)

let concatenate ?axis ts =
  Debug.with_context "concatenate" (fun () -> T.concatenate ?axis ts)

let stack ?axis ts = Debug.with_context "stack" (fun () -> T.stack ?axis ts)

(* Data access operations *)

let data t = T.data t
let strides t = T.strides t
let stride t axis = T.stride t axis
let dims t = T.dims t
let array_prod t = T.array_prod t

(* Utility operations *)

let ensure_int_dtype t =
  Debug.with_context "ensure_int_dtype" (fun () -> T.ensure_int_dtype t)

let resolve_axis len axis = T.resolve_axis len axis
let resolve_single_axis ndim axis = T.resolve_single_axis ndim axis

(* Already defined: reshape, cast *)
let astype dtype x = Debug.with_context "astype" (fun () -> T.astype dtype x)
(* Already defined: contiguous *)

let blit src dst = Debug.with_context "blit" (fun () -> T.blit src dst)
(* Already defined: create, init, scalar, empty, full, zeros, ones *)

let scalar_like x value =
  Debug.with_context "scalar_like" (fun () -> T.scalar_like x value)

let empty_like x = Debug.with_context "empty_like" (fun () -> T.empty_like x)

let full_like x value =
  Debug.with_context "full_like" (fun () -> T.full_like x value)

let zeros_like x = Debug.with_context "zeros_like" (fun () -> T.zeros_like x)
let ones_like x = Debug.with_context "ones_like" (fun () -> T.ones_like x)
let fill x value = Debug.with_context "fill" (fun () -> T.fill x value)

(* Conversion operations *)

let to_bigarray t = T.to_bigarray t

let of_bigarray ba =
  Debug.with_context "of_bigarray" (fun () -> T.of_bigarray ctx ba)

let to_bigarray_ext t = T.to_bigarray_ext t

let of_bigarray_ext ba =
  Debug.with_context "of_bigarray_ext" (fun () -> T.of_bigarray_ext ctx ba)

let to_array t = T.to_array t

(* Binary operations with scalars *)

let binop op a b = Debug.with_context "binop" (fun () -> T.binop op a b)

let scalar_op op a b =
  Debug.with_context "scalar_op" (fun () -> T.scalar_op op a b)

let inplace_op op a b =
  Debug.with_context "inplace_op" (fun () -> T.inplace_op op a b)

(* Already defined: add, sub, mul, div, pow, mod_, maximum, minimum *)
let sub_s a s = Debug.with_context "sub_s" (fun () -> T.sub_s a s)
let isub a b = Debug.with_context "isub" (fun () -> T.isub a b)
let isub_s a s = Debug.with_context "isub_s" (fun () -> T.isub_s a s)
let mul_s a s = Debug.with_context "mul_s" (fun () -> T.mul_s a s)
let imul a b = Debug.with_context "imul" (fun () -> T.imul a b)
let imul_s a s = Debug.with_context "imul_s" (fun () -> T.imul_s a s)
let div_s a s = Debug.with_context "div_s" (fun () -> T.div_s a s)
let idiv a b = Debug.with_context "idiv" (fun () -> T.idiv a b)
let idiv_s a s = Debug.with_context "idiv_s" (fun () -> T.idiv_s a s)
let pow_s a s = Debug.with_context "pow_s" (fun () -> T.pow_s a s)
let ipow a b = Debug.with_context "ipow" (fun () -> T.ipow a b)
let ipow_s a s = Debug.with_context "ipow_s" (fun () -> T.ipow_s a s)
let maximum_s a s = Debug.with_context "maximum_s" (fun () -> T.maximum_s a s)

let rmaximum_s s a =
  Debug.with_context "rmaximum_s" (fun () -> T.rmaximum_s s a)

let imaximum a b = Debug.with_context "imaximum" (fun () -> T.imaximum a b)

let imaximum_s a s =
  Debug.with_context "imaximum_s" (fun () -> T.imaximum_s a s)

let minimum_s a s = Debug.with_context "minimum_s" (fun () -> T.minimum_s a s)

let rminimum_s s a =
  Debug.with_context "rminimum_s" (fun () -> T.rminimum_s s a)

let iminimum a b = Debug.with_context "iminimum" (fun () -> T.iminimum a b)

let iminimum_s a s =
  Debug.with_context "iminimum_s" (fun () -> T.iminimum_s a s)

let mod_s a s = Debug.with_context "mod_s" (fun () -> T.mod_s a s)

(* Bitwise operations *)

let bitwise_xor a b =
  Debug.with_context "bitwise_xor" (fun () -> T.bitwise_xor a b)

let bitwise_or a b =
  Debug.with_context "bitwise_or" (fun () -> T.bitwise_or a b)

let bitwise_and a b =
  Debug.with_context "bitwise_and" (fun () -> T.bitwise_and a b)

(* Logical operations *)

let logical_and a b =
  Debug.with_context "logical_and" (fun () -> T.logical_and a b)

let logical_or a b =
  Debug.with_context "logical_or" (fun () -> T.logical_or a b)

let logical_xor a b =
  Debug.with_context "logical_xor" (fun () -> T.logical_xor a b)

let logical_not x = Debug.with_context "logical_not" (fun () -> T.logical_not x)

(* Comparison operations *)

let cmplt a b = Debug.with_context "cmplt" (fun () -> T.cmplt a b)
let less a b = Debug.with_context "less" (fun () -> T.less a b)
let cmpne a b = Debug.with_context "cmpne" (fun () -> T.cmpne a b)
let not_equal a b = Debug.with_context "not_equal" (fun () -> T.not_equal a b)
let cmpeq a b = Debug.with_context "cmpeq" (fun () -> T.cmpeq a b)
let equal a b = Debug.with_context "equal" (fun () -> T.equal a b)
let greater a b = Debug.with_context "greater" (fun () -> T.greater a b)

let greater_equal a b =
  Debug.with_context "greater_equal" (fun () -> T.greater_equal a b)

let less_equal a b =
  Debug.with_context "less_equal" (fun () -> T.less_equal a b)

(* Already defined: neg, abs, sign, square, sqrt, recip, exp, log, sin, cos,
   tan *)
let bitwise_not x = Debug.with_context "bitwise_not" (fun () -> T.bitwise_not x)
let invert x = Debug.with_context "invert" (fun () -> T.invert x)
let sigmoid x = Debug.with_context "sigmoid" (fun () -> T.sigmoid x)
let rsqrt x = Debug.with_context "rsqrt" (fun () -> T.rsqrt x)
let asin x = Debug.with_context "asin" (fun () -> T.asin x)
let acos x = Debug.with_context "acos" (fun () -> T.acos x)
let atan x = Debug.with_context "atan" (fun () -> T.atan x)
let sinh x = Debug.with_context "sinh" (fun () -> T.sinh x)
let cosh x = Debug.with_context "cosh" (fun () -> T.cosh x)
let tanh x = Debug.with_context "tanh" (fun () -> T.tanh x)
let asinh x = Debug.with_context "asinh" (fun () -> T.asinh x)
let acosh x = Debug.with_context "acosh" (fun () -> T.acosh x)
let atanh x = Debug.with_context "atanh" (fun () -> T.atanh x)
let ceil x = Debug.with_context "ceil" (fun () -> T.ceil x)
let floor x = Debug.with_context "floor" (fun () -> T.floor x)
let round x = Debug.with_context "round" (fun () -> T.round x)
let isinf x = Debug.with_context "isinf" (fun () -> T.isinf x)
let isnan x = Debug.with_context "isnan" (fun () -> T.isnan x)
let isfinite x = Debug.with_context "isfinite" (fun () -> T.isfinite x)
let relu x = Debug.with_context "relu" (fun () -> T.relu x)
let log_sigmoid x = Debug.with_context "log_sigmoid" (fun () -> T.log_sigmoid x)
let exp2 x = Debug.with_context "exp2" (fun () -> T.exp2 x)
let log2 x = Debug.with_context "log2" (fun () -> T.log2 x)

(* More advanced operations *)

let lerp a b weight = Debug.with_context "lerp" (fun () -> T.lerp a b weight)
let lshift a n = Debug.with_context "lshift" (fun () -> T.lshift a n)
let rshift a n = Debug.with_context "rshift" (fun () -> T.rshift a n)

let clamp ?min ?max x =
  Debug.with_context "clamp" (fun () -> T.clamp ?min ?max x)

let clip ?min ?max x = Debug.with_context "clip" (fun () -> T.clip ?min ?max x)

(* Already defined: sum, max, min, prod, mean *)
let var ?axes ?keepdims ?ddof x =
  Debug.with_context "var" (fun () -> T.var ?axes ?keepdims ?ddof x)

let std ?axes ?keepdims ?ddof x =
  Debug.with_context "std" (fun () -> T.std ?axes ?keepdims ?ddof x)

(* Shape manipulation *)

let pad padding_config fill_value x =
  Debug.with_context "pad" (fun () -> T.pad padding_config fill_value x)

let shrink limits x = Debug.with_context "shrink" (fun () -> T.shrink limits x)

let unflatten axis sizes x =
  Debug.with_context "unflatten" (fun () -> T.unflatten axis sizes x)

let ravel x = Debug.with_context "ravel" (fun () -> T.ravel x)

(* Already defined: transpose, flip *)
let moveaxis src dst x =
  Debug.with_context "moveaxis" (fun () -> T.moveaxis src dst x)

let swapaxes axis1 axis2 x =
  Debug.with_context "swapaxes" (fun () -> T.swapaxes axis1 axis2 x)

let roll ?axis shift x =
  Debug.with_context "roll" (fun () -> T.roll ?axis shift x)

let tile x reps = Debug.with_context "tile" (fun () -> T.tile x reps)

let repeat ?axis x repeats =
  Debug.with_context "repeat" (fun () -> T.repeat ?axis x repeats)

(* Already defined: concatenate, stack *)
let vstack ts = Debug.with_context "vstack" (fun () -> T.vstack ts)
let hstack ts = Debug.with_context "hstack" (fun () -> T.hstack ts)
let dstack ts = Debug.with_context "dstack" (fun () -> T.dstack ts)

let broadcast_arrays ts =
  Debug.with_context "broadcast_arrays" (fun () -> T.broadcast_arrays ts)

let broadcast_to shape x =
  Debug.with_context "broadcast_to" (fun () -> T.broadcast_to shape x)

let expand_dims axes x =
  Debug.with_context "expand_dims" (fun () -> T.expand_dims axes x)

(* Creation operations *)

let eye ?m ?k dtype n =
  Debug.with_context "eye" (fun () -> T.eye ctx ?m ?k dtype n)

let identity dtype n =
  Debug.with_context "identity" (fun () -> T.identity ctx dtype n)

let arange dtype start stop step =
  Debug.with_context "arange" (fun () -> T.arange ctx dtype start stop step)

let arange_f dtype start stop step =
  Debug.with_context "arange_f" (fun () -> T.arange_f ctx dtype start stop step)

let linspace dtype ?endpoint start stop count =
  Debug.with_context "linspace" (fun () ->
      T.linspace ctx dtype ?endpoint start stop count)

let logspace dtype ?endpoint ?base start stop count =
  Debug.with_context "logspace" (fun () ->
      T.logspace ctx dtype ?base ?endpoint start stop count)

let geomspace dtype ?endpoint start stop count =
  Debug.with_context "geomspace" (fun () ->
      T.geomspace ctx dtype ?endpoint start stop count)

let meshgrid ?indexing xs =
  Debug.with_context "meshgrid" (fun () -> T.meshgrid ?indexing xs)

(* Indexing operations *)

let slice specs t = Debug.with_context "slice" (fun () -> T.slice specs t)

let set_slice specs t value =
  Debug.with_context "set_slice" (fun () -> T.set_slice specs t value)

let item indices t = T.unsafe_get indices t
let set_item indices t value = T.unsafe_set indices t value

let one_hot ~num_classes indices =
  Debug.with_context "one_hot" (fun () -> T.one_hot ~num_classes indices)

(* Missing operations for BERT *)

let softmax ?axes x = Debug.with_context "softmax" (fun () -> T.softmax ?axes x)
let cumsum ?axis x = Debug.with_context "cumsum" (fun () -> T.cumsum ?axis x)
let numel x = T.numel x
let get indices x = Debug.with_context "get" (fun () -> T.get indices x)

let set indices x value =
  Debug.with_context "set" (fun () -> T.set indices x value)

let take ?axis ?mode x indices =
  Debug.with_context "take" (fun () -> T.take ?axis ?mode x indices)

let take_along_axis ~axis x indices =
  Debug.with_context "take_along_axis" (fun () ->
      T.take_along_axis ~axis x indices)

let add_s a s = Debug.with_context "add_s" (fun () -> T.add_s a s)

(* Splitting operations *)

let array_split ~axis t indices_or_sections =
  Debug.with_context "array_split" (fun () ->
      T.array_split ~axis t indices_or_sections)

let split ~axis t indices_or_sections =
  Debug.with_context "split" (fun () -> T.split ~axis t indices_or_sections)

let randint dtype ?seed ?high shape low =
  Debug.with_context "randint" (fun () ->
      T.randint ctx dtype ?seed ?high shape low)

(* Sorting operations *)

let sort ?descending ?axis x =
  Debug.with_context "sort" (fun () -> T.sort ?descending ?axis x)

let argsort ?descending ?axis x =
  Debug.with_context "argsort" (fun () -> T.argsort ?descending ?axis x)

let argmax ?axis ?keepdims x =
  Debug.with_context "argmax" (fun () -> T.argmax ?axis ?keepdims x)

let argmin ?axis ?keepdims x =
  Debug.with_context "argmin" (fun () -> T.argmin ?axis ?keepdims x)

(* Convolution operations *)

let convolve1d ?groups ?stride ?padding_mode ?dilation ?fillvalue ?bias x w =
  Debug.with_context "convolve1d" (fun () ->
      T.convolve1d ?groups ?stride ?padding_mode ?dilation ?fillvalue ?bias x w)

let convolve2d ?groups ?stride ?padding_mode ?dilation ?fillvalue ?bias x w =
  Debug.with_context "convolve2d" (fun () ->
      T.convolve2d ?groups ?stride ?padding_mode ?dilation ?fillvalue ?bias x w)

(* Pooling operations *)

let pool_setup = T.pool_setup

let avg_pool1d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
    ?count_include_pad x =
  Debug.with_context "avg_pool1d" (fun () ->
      T.avg_pool1d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
        ?count_include_pad x)

let avg_pool2d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
    ?count_include_pad x =
  Debug.with_context "avg_pool2d" (fun () ->
      T.avg_pool2d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
        ?count_include_pad x)

let max_pool1d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
    ?return_indices x =
  Debug.with_context "max_pool1d" (fun () ->
      T.max_pool1d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
        ?return_indices x)

let max_pool2d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
    ?return_indices x =
  Debug.with_context "max_pool2d" (fun () ->
      T.max_pool2d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
        ?return_indices x)

let min_pool1d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
    ?return_indices x =
  Debug.with_context "min_pool1d" (fun () ->
      T.min_pool1d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
        ?return_indices x)

let min_pool2d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
    ?return_indices x =
  Debug.with_context "min_pool2d" (fun () ->
      T.min_pool2d ~kernel_size ?stride ?dilation ?padding_spec ?ceil_mode
        ?return_indices x)

let max_unpool1d x indices ~kernel_size =
  Debug.with_context "max_unpool1d" (fun () ->
      T.max_unpool1d x indices ~kernel_size)

let max_unpool2d x indices ~kernel_size =
  Debug.with_context "max_unpool2d" (fun () ->
      T.max_unpool2d x indices ~kernel_size)

let im2col ~kernel_size ~stride ~dilation ~padding x =
  Debug.with_context "im2col" (fun () ->
      T.im2col ~kernel_size ~stride ~dilation ~padding x)

let col2im ~output_size ~kernel_size ~stride ~dilation ~padding x =
  Debug.with_context "col2im" (fun () ->
      T.col2im ~output_size ~kernel_size ~stride ~dilation ~padding x)
