open Nx_core
open Metal

let dispatch_unary ctx op_name t =
  let out_shape = View.shape t.Internal.view in
  let out_size = View.numel t.Internal.view in
  let size_bytes = out_size * Internal.sizeof_dtype t.Internal.dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let view = View.create out_shape in
  let out =
    { context = ctx; Internal.dtype = t.dtype; buffer = metal_buffer; view }
  in

  let func = Kernels.get_unary_kernel ctx t.dtype op_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        t.Internal.buffer.buffer;

      (* Pass shape and stride information *)
      let ndim = Array.length out_shape in
      let shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      let strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in

      let strides = View.strides t.Internal.view in

      for i = 0 to ndim - 1 do
        Ctypes.(shape_arr +@ i <-@ Unsigned.UInt32.of_int out_shape.(i));
        Ctypes.(strides_arr +@ i <-@ Int32.of_int strides.(i))
      done;

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp shape_arr)
        ~length:(ndim * 4) ~index:2;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp strides_arr)
        ~length:(ndim * 4) ~index:3;

      let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp ndim_val)
        ~length:4 ~index:4;

      (* Pass view offset *)
      let offset_val =
        Ctypes.(allocate int32_t (Int32.of_int (View.offset t.Internal.view)))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp offset_val)
        ~length:4 ~index:5;

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

let neg ctx t = dispatch_unary ctx "neg" t
let log2 ctx t = dispatch_unary ctx "log2" t
let exp2 ctx t = dispatch_unary ctx "exp2" t
let sin ctx t = dispatch_unary ctx "sin" t
let sqrt ctx t = dispatch_unary ctx "sqrt" t
let recip ctx t = dispatch_unary ctx "recip" t
