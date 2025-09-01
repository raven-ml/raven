open Nx_core
open Bigarray_ext

type buffer = Internal.metal_buffer
type context = Internal.context

type ('a, 'b) t = ('a, 'b) Internal.t = {
  context : context;
  dtype : ('a, 'b) Dtype.t;
  buffer : buffer;
  view : Lazy_view.t;
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

let data : type a b. (a, b) t -> (a, b, c_layout) Array1.t =
 fun t ->
  (* For compatibility with the frontend's unsafe_get, we need to return a
     bigarray that preserves the view's offset. This means returning the entire
     buffer, not just the viewed portion. *)
  let contents = Metal.Buffer.contents t.buffer.buffer in

  match t.dtype with
  | Dtype.BFloat16 | Dtype.Bool | Dtype.Complex16 ->
      (* For extended types, use the extended kind and our special function *)
      let kind = Dtype.to_bigarray_ext_kind t.dtype in
      let elem_size = Internal.sizeof_dtype t.dtype in
      let buffer_size = t.buffer.size_bytes / elem_size in
      (* Use the external function to create bigarray from pointer *)
      let ptr_as_nativeint = Ctypes.raw_address_of_ptr contents in
      let genarray =
        Internal.ba_from_ptr
          (Internal.kind_to_int kind)
          (Internal.layout_to_int Bigarray_ext.c_layout)
          buffer_size ptr_as_nativeint
      in
      Bigarray_ext.array1_of_genarray genarray
  | _ ->
      (* Standard bigarray types *)
      let kind = Dtype.to_bigarray_kind t.dtype in
      let elem_size = Internal.sizeof_dtype t.dtype in
      let buffer_size = t.buffer.size_bytes / elem_size in
      (* Create a bigarray view of the entire Metal buffer *)
      Ctypes.bigarray_of_ptr Ctypes.array1 buffer_size kind (Obj.magic contents)

let op_contiguous t =
  (* Check if view is contiguous AND buffer has the expected size *)
  let view_size = Internal.numel t in
  let expected_bytes = view_size * Internal.sizeof_dtype t.dtype in
  let actual_bytes = t.buffer.size_bytes in

  if Lazy_view.is_contiguous t.view && actual_bytes >= expected_bytes then t
  else Ops_movement.make_contiguous t.context t

(* Buffer operations *)
let op_buffer ctx dtype size_in_elements =
  (* Check if dtype is supported by Metal *)
  let _ = Internal.dtype_to_metal_type dtype in
  let size_bytes = size_in_elements * Internal.sizeof_dtype dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let view =
    if size_in_elements = 0 then
      Lazy_view.create (Symbolic_shape.of_ints [| 0 |])
    else Lazy_view.create (Symbolic_shape.of_ints [| size_in_elements |])
  in
  { context = ctx; dtype; buffer = metal_buffer; view }

let op_const_scalar : type a b. context -> a -> (a, b) Dtype.t -> (a, b) t =
 fun ctx value dtype ->
  (* Check if dtype is supported by Metal *)
  let _ = Internal.dtype_to_metal_type dtype in
  let size_bytes = Internal.sizeof_dtype dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in

  (* Create temporary bigarray to hold the value *)
  let kind = Dtype.to_bigarray_ext_kind dtype in
  let ba = Array1.create kind c_layout 1 in
  Array1.set ba 0 value;

  (* Copy to Metal buffer *)
  let metal_buffer = { Internal.buffer; size_bytes } in
  let t =
    {
      context = ctx;
      dtype;
      buffer = metal_buffer;
      view = Lazy_view.create (Symbolic_shape.of_ints [||]);
    }
  in
  Internal.copy_from_bigarray ctx metal_buffer ba;
  t

let op_const_array : type a b. context -> (a, b, c_layout) Array1.t -> (a, b) t
    =
 fun ctx bigarray ->
  let dtype = Dtype.of_bigarray_ext_kind (Array1.kind bigarray) in
  (* Check if dtype is supported by Metal *)
  let _ = Internal.dtype_to_metal_type dtype in
  let size = Array1.dim bigarray in
  let size_bytes = size * Internal.sizeof_dtype dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let view = Lazy_view.create (Symbolic_shape.of_ints [| size |]) in
  let t = { context = ctx; dtype; buffer = metal_buffer; view } in
  Internal.copy_from_bigarray ctx metal_buffer bigarray;
  t

(* Movement operations *)
let op_expand t new_shape =
  (* Expand changes the view metadata, setting strides to 0 for broadcast
     dimensions *)
  let current_shape = Lazy_view.shape t.view in
  let current_rank = Symbolic_shape.rank current_shape in
  let new_rank = Symbolic_shape.rank new_shape in

  if current_rank = 0 && new_rank > 0 then
    (* Special case: expanding a scalar to a higher rank *)
    (* First reshape scalar to have the right number of dimensions with size 1 *)
    let ones_shape = Array.make new_rank 1 in
    let reshaped =
      {
        t with
        view = Lazy_view.reshape (Symbolic_shape.of_ints ones_shape) t.view;
      }
    in
    { reshaped with view = Lazy_view.expand new_shape reshaped.view }
  else { t with view = Lazy_view.expand new_shape t.view }

let op_reshape t new_shape =
  (* Reshape changes the view metadata *)
  { t with view = Lazy_view.reshape new_shape t.view }

let op_permute t axes =
  (* Permute changes the view metadata *)
  { t with view = Lazy_view.permute axes t.view }

let op_shrink t bounds =
  (* Shrink changes the view metadata *)
  { t with view = Lazy_view.shrink bounds t.view }

let op_flip t axes_to_flip =
  (* Flip changes the view metadata *)
  { t with view = Lazy_view.flip axes_to_flip t.view }

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
  let old_shape = Internal.shape t in
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
  let out =
    { out with view = Lazy_view.create (Symbolic_shape.of_ints new_shape) }
  in

  (* Convert fill_value to float for the kernel - handle by checking dtype
     size *)
  let is_float_dtype : type a b. (a, b) Dtype.t -> bool = function
    | Dtype.Float32 | Dtype.Float64 -> true
    | _ -> false
  in
  let dtype_size = Internal.sizeof_dtype t.dtype in
  (* TODO: why are we converting to float here? *)
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
  let out = { out with view = Lazy_view.create (Lazy_view.shape t.view) } in
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
let op_cmpeq a b = Ops_binary.cmpeq a.context a b
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
let op_reduce_sum ~axes ~keepdims t = Ops_reduce.sum t.context t ~axes ~keepdims
let op_reduce_max ~axes ~keepdims t = Ops_reduce.max t.context t ~axes ~keepdims

let op_reduce_prod ~axes ~keepdims t =
  Ops_reduce.prod t.context t ~axes ~keepdims

(* Special operations *)
let op_cast : type a b c d. (a, b) t -> (c, d) Dtype.t -> (c, d) t =
 fun t target_dtype -> Ops_special.cast t.context t target_dtype

let op_assign dst src = Ops_special.assign dst.context dst src

let op_gather data indices axis =
  Ops_special.gather data.context data indices axis

let op_scatter ?(mode = `Set) ?(unique_indices = false) data_template indices
    updates axis =
  Ops_special.scatter ~mode ~unique_indices data_template.context data_template
    indices updates axis

let op_threefry key counter = Ops_special.threefry key.context key counter

let op_unfold t ~kernel_size ~stride ~dilation ~padding =
  Ops_im2col.op_unfold t.context t ~kernel_size ~stride ~dilation ~padding

let op_fold t ~output_size ~kernel_size ~stride ~dilation ~padding =
  Ops_im2col.op_fold t.context t ~output_size ~kernel_size ~stride ~dilation
    ~padding

let op_matmul a b = Ops_matmul.op_matmul a.context a b

(* FFT operations *)
let op_fft t ~axes ~s = Ops_fft.op_fft t.context t ~axes ~s
let op_ifft t ~axes ~s = Ops_fft.op_ifft t.context t ~axes ~s
let op_rfft t ~dtype ~axes ~s = Ops_fft.op_rfft t.context t ~dtype ~axes ~s
let op_irfft t ~dtype ~axes ~s = Ops_fft.op_irfft t.context t ~dtype ~axes ~s

(* Linear algebra operations *)
let op_cholesky ~upper t = Ops_linalg.op_cholesky t.context ~upper t
let op_qr ~reduced t = Ops_linalg.op_qr t.context ~reduced t
let op_svd ~full_matrices t = Ops_linalg.op_svd t.context ~full_matrices t
let op_eig ~vectors t = Ops_linalg.op_eig t.context ~vectors t
let op_eigh ~vectors t = Ops_linalg.op_eigh t.context ~vectors t

let op_triangular_solve ~upper ~transpose ~unit_diag a b =
  Ops_linalg.op_triangular_solve a.context ~upper ~transpose ~unit_diag a b

let op_as_strided t new_shape new_strides_in_elements offset_in_elements =
  (* Metal backend may need to copy for non-trivial strides *)

  (* First check if this is essentially a contiguous view - in that case we can
     do zero-copy *)
  let new_shape_arr =
    match Symbolic_shape.eval new_shape with
    | Some arr -> arr
    | None ->
        Error.failed ~op:"op_as_strided" ~what:"symbolic shapes not supported"
          ()
  in
  (* Compute C-contiguous strides for the shape *)
  let compute_strides shape_array =
    let n = Array.length shape_array in
    if n = 0 then [||]
    else
      let strides = Array.make n 0 in
      strides.(n - 1) <- (if shape_array.(n - 1) = 0 then 0 else 1);
      for i = n - 2 downto 0 do
        strides.(i) <-
          (if shape_array.(i) = 0 then 0
           else strides.(i + 1) * max 1 shape_array.(i + 1))
      done;
      strides
  in
  let expected_strides = compute_strides new_shape_arr in
  let is_contiguous =
    offset_in_elements = 0
    && Array.length new_strides_in_elements = Array.length expected_strides
    && Array.for_all2 ( = ) new_strides_in_elements expected_strides
  in

  if is_contiguous then
    (* Zero-copy case: just update the view *)
    let new_view =
      Lazy_view.create_strided new_shape ~strides:new_strides_in_elements
        ~offset:offset_in_elements
    in
    { t with view = new_view }
  else
    (* Non-contiguous case: need to materialize *)
    (* Validate bounds first *)
    let buffer_size = t.buffer.size_bytes / Internal.sizeof_dtype t.dtype in
    let max_element_accessed = ref offset_in_elements in
    Array.iteri
      (fun i dim ->
        if dim > 0 then
          max_element_accessed :=
            max !max_element_accessed
              (offset_in_elements + ((dim - 1) * new_strides_in_elements.(i))))
      new_shape_arr;

    if !max_element_accessed >= buffer_size then
      Error.failed ~op:"op_as_strided"
        ~what:"view would access out-of-bounds memory" ();

    (* Create new contiguous buffer and copy strided data *)
    let numel = Array.fold_left ( * ) 1 new_shape_arr in
    let result = op_buffer t.context t.dtype numel in

    (* Copy elements according to the strided view *)
    (* This requires CPU-side copying for now *)
    let src_data = data t in
    let dst_data = data result in

    let rec copy_recursive indices dim_idx flat_idx =
      if dim_idx = Array.length new_shape_arr then (
        (* Compute source offset from indices and strides *)
        let src_offset = ref offset_in_elements in
        for i = 0 to Array.length indices - 1 do
          src_offset := !src_offset + (indices.(i) * new_strides_in_elements.(i))
        done;
        (* Copy the element *)
        let value = Bigarray_ext.Array1.get src_data !src_offset in
        Bigarray_ext.Array1.set dst_data flat_idx value;
        flat_idx + 1)
      else
        let dim_size = new_shape_arr.(dim_idx) in
        let next_flat_idx = ref flat_idx in
        for i = 0 to dim_size - 1 do
          indices.(dim_idx) <- i;
          next_flat_idx := copy_recursive indices (dim_idx + 1) !next_flat_idx
        done;
        !next_flat_idx
    in

    (if numel > 0 then
       let indices = Array.make (Array.length new_shape_arr) 0 in
       let _ = copy_recursive indices 0 0 in
       ());

    (* Update result's view to match the requested shape *)
    let result_view = Lazy_view.reshape new_shape result.view in
    { result with view = result_view }

let op_associative_scan = Obj.magic ()
