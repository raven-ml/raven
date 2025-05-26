open Nx_core
open Metal

let dispatch_unary ctx op_name t =
  let out_size = Internal.numel t in
  let size_bytes = out_size * Internal.sizeof_dtype t.Internal.dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in

  let out =
    {
      context = ctx;
      Internal.dtype = t.dtype;
      buffer = metal_buffer;
      view = t.view;
    }
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

      let size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int out_size))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp size_val)
        ~length:4 ~index:2;

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
