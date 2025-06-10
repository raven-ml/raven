open Nx_core
open Metal

let dispatch_binary ctx op_name a b =
  let out_shape = View.shape a.Internal.view in
  let out_size = View.numel a.Internal.view in
  let size_bytes = out_size * Internal.sizeof_dtype a.dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let view = View.create out_shape in
  let out =
    { context = ctx; Internal.dtype = a.dtype; buffer = metal_buffer; view }
  in

  let func = Kernels.get_binary_kernel ctx a.dtype op_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        a.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:2
        b.Internal.buffer.buffer;

      (* Set shape and stride information *)
      let ndim = Array.length out_shape in
      let out_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      let a_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
      let b_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in

      let a_strides = View.strides a.view in
      let b_strides = View.strides b.view in

      for i = 0 to ndim - 1 do
        Ctypes.(out_shape_arr +@ i <-@ Unsigned.UInt32.of_int out_shape.(i));
        Ctypes.(a_strides_arr +@ i <-@ Int32.of_int a_strides.(i));
        Ctypes.(b_strides_arr +@ i <-@ Int32.of_int b_strides.(i))
      done;

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp out_shape_arr)
        ~length:(ndim * 4) ~index:3;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp a_strides_arr)
        ~length:(ndim * 4) ~index:4;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp b_strides_arr)
        ~length:(ndim * 4) ~index:5;

      let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp ndim_val)
        ~length:4 ~index:6;

      (* Pass view offsets *)
      let a_offset_val =
        Ctypes.(allocate int32_t (Int32.of_int (View.offset a.Internal.view)))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp a_offset_val)
        ~length:4 ~index:7;

      let b_offset_val =
        Ctypes.(allocate int32_t (Int32.of_int (View.offset b.Internal.view)))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp b_offset_val)
        ~length:4 ~index:8;

      (* Dispatch threads *)
      let threads_per_group, num_groups =
        Internal.compute_thread_groups out_size
      in
      let grid_size =
        { Metal.Size.width = num_groups; height = 1; depth = 1 }
      in
      let group_size =
        { Metal.Size.width = threads_per_group; height = 1; depth = 1 }
      in

      ComputeCommandEncoder.dispatch_threadgroups encoder
        ~threadgroups_per_grid:grid_size ~threads_per_threadgroup:group_size;
      ComputeCommandEncoder.end_encoding encoder);

  out

let add ctx a b = dispatch_binary ctx "add" a b
let mul ctx a b = dispatch_binary ctx "mul" a b
let idiv ctx a b = dispatch_binary ctx "idiv" a b
let fdiv ctx a b = dispatch_binary ctx "fdiv" a b
let max ctx a b = dispatch_binary ctx "max" a b
let mod_ ctx a b = dispatch_binary ctx "mod" a b
let pow ctx a b = dispatch_binary ctx "pow" a b

let dispatch_comparison ctx op_name a b =
  let out_shape = View.shape a.Internal.view in
  let out_size = View.numel a.Internal.view in
  let size_bytes = out_size * Internal.sizeof_dtype Dtype.UInt8 in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let view = View.create out_shape in
  let out =
    { context = ctx; Internal.dtype = Dtype.UInt8; buffer = metal_buffer; view }
  in

  let func = Kernels.get_binary_kernel ctx a.dtype op_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        a.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:2
        b.Internal.buffer.buffer;

      (* Set shape and stride information *)
      let ndim = Array.length out_shape in
      let out_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      let a_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
      let b_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in

      let a_strides = View.strides a.view in
      let b_strides = View.strides b.view in

      for i = 0 to ndim - 1 do
        Ctypes.(out_shape_arr +@ i <-@ Unsigned.UInt32.of_int out_shape.(i));
        Ctypes.(a_strides_arr +@ i <-@ Int32.of_int a_strides.(i));
        Ctypes.(b_strides_arr +@ i <-@ Int32.of_int b_strides.(i))
      done;

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp out_shape_arr)
        ~length:(ndim * 4) ~index:3;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp a_strides_arr)
        ~length:(ndim * 4) ~index:4;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp b_strides_arr)
        ~length:(ndim * 4) ~index:5;

      let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp ndim_val)
        ~length:4 ~index:6;

      (* Pass view offsets *)
      let a_offset_val =
        Ctypes.(allocate int32_t (Int32.of_int (View.offset a.Internal.view)))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp a_offset_val)
        ~length:4 ~index:7;

      let b_offset_val =
        Ctypes.(allocate int32_t (Int32.of_int (View.offset b.Internal.view)))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp b_offset_val)
        ~length:4 ~index:8;

      (* Dispatch threads *)
      let threads_per_group, num_groups =
        Internal.compute_thread_groups out_size
      in
      let grid_size =
        { Metal.Size.width = num_groups; height = 1; depth = 1 }
      in
      let group_size =
        { Metal.Size.width = threads_per_group; height = 1; depth = 1 }
      in

      ComputeCommandEncoder.dispatch_threadgroups encoder
        ~threadgroups_per_grid:grid_size ~threads_per_threadgroup:group_size;
      ComputeCommandEncoder.end_encoding encoder);

  out

let cmplt ctx a b = dispatch_comparison ctx "cmplt" a b
let cmpne ctx a b = dispatch_comparison ctx "cmpne" a b

let is_integer_dtype : type a b. (a, b) Dtype.t -> bool = function
  | Dtype.Int32 | Dtype.Int64 | Dtype.UInt8 | Dtype.UInt16 | Dtype.Int8
  | Dtype.Int16 | Dtype.Int | Dtype.NativeInt ->
      true
  | _ -> false

let dispatch_bitwise ctx op_name a b =
  (* Bitwise operations only work on integer types *)
  if is_integer_dtype a.Internal.dtype then dispatch_binary ctx op_name a b
  else
    failwith
      (Printf.sprintf "Bitwise operation %s not supported for dtype" op_name)

let xor ctx a b = dispatch_bitwise ctx "xor" a b
let or_ ctx a b = dispatch_bitwise ctx "or" a b
let and_ ctx a b = dispatch_bitwise ctx "and" a b
