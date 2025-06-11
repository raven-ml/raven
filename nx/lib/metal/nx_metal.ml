open Nx_core

type buffer = Internal.metal_buffer
type context = Internal.context

type ('a, 'b) t = ('a, 'b) Internal.t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : buffer;
  view : View.t;
}

let view t = t.view
let dtype t = t.dtype
let context t = t.context

let create_context () =
  let device = Metal.Device.create_system_default () in
  let queue = Metal.CommandQueue.on_device device in
  let library = Kernels.compile_library device in
  let kernels = Internal.create_kernel_cache () in
  let pool = Buffer_pool.create device in
  { Internal.device; queue; library; kernels; pool }

let data : type a b. (a, b) t -> (a, b, Bigarray.c_layout) Bigarray.Array1.t =
 fun t ->
  (* For compatibility with the frontend's unsafe_get, we need to return a
     bigarray that preserves the view's offset. This means returning the entire
     buffer, not just the viewed portion. *)
  let contents = Metal.Buffer.contents t.buffer.buffer in
  let kind = Dtype.to_bigarray_kind t.dtype in
  let elem_size = Internal.sizeof_dtype t.dtype in
  let buffer_size = t.buffer.size_bytes / elem_size in

  (* Create a bigarray view of the entire Metal buffer *)
  Ctypes.bigarray_of_ptr Ctypes.array1 buffer_size kind (Obj.magic contents)

let op_contiguous t =
  (* Check if view is contiguous AND buffer has the expected size *)
  let view_size = View.numel t.view in
  let expected_bytes = view_size * Internal.sizeof_dtype t.dtype in
  let actual_bytes = t.buffer.size_bytes in

  if View.is_c_contiguous t.view && actual_bytes >= expected_bytes then t
  else Ops_movement.make_contiguous t.context t

(* Buffer operations *)
let op_buffer ctx dtype size_in_elements =
  (* Check if dtype is supported by Metal *)
  let _ = Internal.dtype_to_metal_type dtype in
  let size_bytes = size_in_elements * Internal.sizeof_dtype dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let view =
    if size_in_elements = 0 then View.create [| 0 |]
    else View.create [| size_in_elements |]
  in
  { context = ctx; dtype; buffer = metal_buffer; view }

let op_const_scalar : type a b. context -> a -> (a, b) Dtype.t -> (a, b) t =
 fun ctx value dtype ->
  (* Check if dtype is supported by Metal *)
  let _ = Internal.dtype_to_metal_type dtype in
  let size_bytes = Internal.sizeof_dtype dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in

  (* Create temporary bigarray to hold the value *)
  let kind = Dtype.to_bigarray_kind dtype in
  let ba = Bigarray.Array1.create kind Bigarray.c_layout 1 in
  Bigarray.Array1.set ba 0 value;

  (* Copy to Metal buffer *)
  let metal_buffer = { Internal.buffer; size_bytes } in
  let t =
    { context = ctx; dtype; buffer = metal_buffer; view = View.create [||] }
  in
  Internal.copy_from_bigarray ctx metal_buffer ba;
  t

let op_const_array : type a b.
    context -> (a, b, Bigarray.c_layout) Bigarray.Array1.t -> (a, b) t =
 fun ctx bigarray ->
  let dtype = Dtype.of_bigarray_kind (Bigarray.Array1.kind bigarray) in
  (* Check if dtype is supported by Metal *)
  let _ = Internal.dtype_to_metal_type dtype in
  let size = Bigarray.Array1.dim bigarray in
  let size_bytes = size * Internal.sizeof_dtype dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let view = View.create [| size |] in
  let t = { context = ctx; dtype; buffer = metal_buffer; view } in
  Internal.copy_from_bigarray ctx metal_buffer bigarray;
  t

(* Movement operations *)
let op_expand t new_shape =
  (* Expand changes the view metadata, setting strides to 0 for broadcast
     dimensions *)
  { t with view = View.expand t.view new_shape }

let op_reshape t new_shape =
  (* Reshape changes the view metadata *)
  { t with view = View.reshape t.view new_shape }

let op_permute t axes =
  (* Permute changes the view metadata *)
  { t with view = View.permute t.view axes }

let op_shrink t bounds =
  (* Shrink changes the view metadata *)
  { t with view = View.shrink t.view bounds }

let op_flip t axes_to_flip =
  (* Flip changes the view metadata *)
  { t with view = View.flip t.view axes_to_flip }

let op_pad t padding fill_value =
  (* Validate padding values *)
  Array.iter
    (fun (before, after) ->
      if before < 0 || after < 0 then
        Nx_core.Error.invalid ~op:"pad" ~what:"padding values"
          ~reason:"negative values not allowed"
          ~hint:"use shrink or slice to remove elements" ())
    padding;
  
  (* Padding requires actual computation *)
  let ctx = t.context in
  let old_shape = View.shape t.view in
  let new_shape =
    Array.mapi
      (fun i dim ->
        if i < Array.length padding then
          let before, after = padding.(i) in
          dim + before + after
        else dim)
      old_shape
  in
  let out = op_buffer ctx t.dtype (Array.fold_left ( * ) 1 new_shape) in
  let out = { out with view = View.create new_shape } in

  (* Convert fill_value to float for the kernel - handle by checking dtype
     size *)
  let is_float_dtype : type a b. (a, b) Dtype.t -> bool = function
    | Dtype.Float32 | Dtype.Float64 -> true
    | _ -> false
  in
  let dtype_size = Internal.sizeof_dtype t.dtype in
  let fill_value_float =
    if is_float_dtype t.dtype then Obj.magic fill_value
    else if dtype_size = 8 then
      (* 64-bit integer *)
      Int64.to_float (Obj.magic fill_value)
    else if dtype_size = 4 then
      (* 32-bit integer *)
      Int32.to_float (Obj.magic fill_value)
    else
      (* 8/16-bit integer *)
      float_of_int (Obj.magic fill_value)
  in
  Ops_movement.pad ctx t out padding fill_value_float;
  out

let rec op_cat tensors axis =
  (* Concatenation requires actual computation *)
  match tensors with
  | [] -> failwith "op_cat: empty list"
  | [ t ] -> op_copy t
  | first :: _ ->
      let ctx = first.context in
      Ops_movement.cat ctx tensors axis

and op_copy t =
  let ctx = t.context in
  let out = op_buffer ctx t.dtype (Internal.numel t) in
  let out = { out with view = View.create (View.shape t.view) } in
  Ops_movement.copy ctx t out;
  out

(* Binary operations *)
let op_add a b = Ops_binary.add a.context a b
let op_mul a b = Ops_binary.mul a.context a b
let op_idiv a b = Ops_binary.idiv a.context a b
let op_fdiv a b = Ops_binary.fdiv a.context a b
let op_max a b = Ops_binary.max a.context a b
let op_mod a b = Ops_binary.mod_ a.context a b
let op_pow a b = Ops_binary.pow a.context a b
let op_cmplt a b = Ops_binary.cmplt a.context a b
let op_cmpne a b = Ops_binary.cmpne a.context a b
let op_xor a b = Ops_binary.xor a.context a b
let op_or a b = Ops_binary.or_ a.context a b
let op_and a b = Ops_binary.and_ a.context a b

(* Unary operations *)
let op_neg t = Ops_unary.neg t.context t
let op_log2 t = Ops_unary.log2 t.context t
let op_exp2 t = Ops_unary.exp2 t.context t
let op_sin t = Ops_unary.sin t.context t
let op_sqrt t = Ops_unary.sqrt t.context t
let op_recip t = Ops_unary.recip t.context t

(* Ternary operations *)
let op_where cond if_true if_false =
  Ops_special.where cond.context cond if_true if_false

(* Reduction operations *)
let op_reduce_sum ~axes ~keepdims t =
  Ops_reduce.reduce_sum t.context ~axes ~keepdims t

let op_reduce_max ~axes ~keepdims t =
  Ops_reduce.reduce_max t.context ~axes ~keepdims t

let op_reduce_prod ~axes ~keepdims t =
  Ops_reduce.reduce_prod t.context ~axes ~keepdims t

(* Special operations *)
let op_cast : type a b c d. (a, b) t -> (c, d) Dtype.t -> (c, d) t =
 fun t target_dtype -> Ops_special.cast t.context t target_dtype

let op_assign dst src = Ops_special.assign dst.context dst src

let op_gather data indices axis =
  Ops_special.gather data.context data indices axis

let op_scatter data_template indices updates axis =
  Ops_special.scatter data_template.context data_template indices updates axis

let op_threefry key counter = Ops_special.threefry key.context key counter

let op_unfold t ~kernel_size ~stride ~dilation ~padding =
  Ops_im2col.op_unfold t.context t ~kernel_size ~stride ~dilation ~padding

let op_fold t ~output_size ~kernel_size ~stride ~dilation ~padding =
  Ops_im2col.op_fold t.context t ~output_size ~kernel_size ~stride ~dilation
    ~padding

let op_matmul a b = Ops_matmul.op_matmul a.context a b
