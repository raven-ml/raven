open Nx_core
open Metal

(* Cumsum operation dispatcher for Metal backend *)
let cumsum (ctx : Internal.context) ~(axis : int) (t : ('a, 'b) Internal.t) : ('a, 'b) Internal.t =
  let shape = Internal.shape t in
  let ndim = Array.length shape in
  
  (* Validate axis *)
  let axis = if axis < 0 then ndim + axis else axis in
  if axis < 0 || axis >= ndim then
    invalid_arg (Printf.sprintf "cumsum: axis %d out of bounds for tensor with rank %d" axis ndim);
  
  (* Output has same shape as input *)
  let out_shape = Array.copy shape in
  let out_numel = Array.fold_left ( * ) 1 out_shape in
  
  (* Allocate output buffer *)
  let size_bytes = out_numel * Internal.sizeof_dtype t.dtype in
  let buffer = Buffer_pool.allocate ctx.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out = {
    context = ctx;
    Internal.dtype = t.dtype;
    buffer = metal_buffer;
    view = View.create out_shape;
  } in
  
  (* Get kernel and create pipeline *)
  let dtype_suffix = Internal.dtype_to_metal_type t.dtype in
  let kernel_name = Printf.sprintf "cumsum_%s" dtype_suffix in
  let func = Kernels.get_special_kernel ctx kernel_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in
  
  (* Calculate number of slices (all dimensions except the cumsum axis) *)
  let total_slices = ref 1 in
  for i = 0 to ndim - 1 do
    if i <> axis then
      total_slices := !total_slices * shape.(i)
  done;
  
  (* Dispatch kernel *)
  Internal.with_command_buffer ctx (fun cmd_buffer ->
    let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in
    ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
    
    (* Set buffers *)
    ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0 out.buffer.buffer;
    ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1 t.buffer.buffer;
    
    (* Set scalar parameters *)
    let in_size = Internal.numel t in
    let in_size_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int in_size)) in
    let axis_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int axis)) in
    
    ComputeCommandEncoder.set_bytes encoder
      ~bytes:Ctypes.(to_voidp in_size_val)
      ~length:4 ~index:2;
    ComputeCommandEncoder.set_bytes encoder
      ~bytes:Ctypes.(to_voidp axis_val)
      ~length:4 ~index:3;
    
    (* Pass shape array *)
    let shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
    for i = 0 to ndim - 1 do
      Ctypes.(shape_arr +@ i <-@ Unsigned.UInt32.of_int shape.(i))
    done;
    ComputeCommandEncoder.set_bytes encoder
      ~bytes:Ctypes.(to_voidp shape_arr)
      ~length:(ndim * 4) ~index:4;
    
    let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
    ComputeCommandEncoder.set_bytes encoder
      ~bytes:Ctypes.(to_voidp ndim_val)
      ~length:4 ~index:5;
    
    (* Pass input strides and offset *)
    let in_strides = View.strides t.view in
    let in_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
    for i = 0 to ndim - 1 do
      Ctypes.(in_strides_arr +@ i <-@ Int32.of_int in_strides.(i))
    done;
    ComputeCommandEncoder.set_bytes encoder
      ~bytes:Ctypes.(to_voidp in_strides_arr)
      ~length:(ndim * 4) ~index:6;
    
    let in_offset_val = Ctypes.(allocate int32_t (Int32.of_int (View.offset t.view))) in
    ComputeCommandEncoder.set_bytes encoder
      ~bytes:Ctypes.(to_voidp in_offset_val)
      ~length:4 ~index:7;
    
    (* Pass output strides and offset *)
    let out_strides = View.strides out.view in
    let out_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
    for i = 0 to ndim - 1 do
      Ctypes.(out_strides_arr +@ i <-@ Int32.of_int out_strides.(i))
    done;
    ComputeCommandEncoder.set_bytes encoder
      ~bytes:Ctypes.(to_voidp out_strides_arr)
      ~length:(ndim * 4) ~index:8;
    
    let out_offset_val = Ctypes.(allocate int32_t (Int32.of_int (View.offset out.view))) in
    ComputeCommandEncoder.set_bytes encoder
      ~bytes:Ctypes.(to_voidp out_offset_val)
      ~length:4 ~index:9;
    
    (* Dispatch threads - one thread per slice *)
    let threads_per_group, num_groups = Internal.compute_thread_groups !total_slices in
    ComputeCommandEncoder.dispatch_threadgroups encoder
      ~threadgroups_per_grid:{ Metal.Size.width = num_groups; height = 1; depth = 1 }
      ~threads_per_threadgroup:{ Metal.Size.width = threads_per_group; height = 1; depth = 1 };
    
    ComputeCommandEncoder.end_encoding encoder
  );
  
  out