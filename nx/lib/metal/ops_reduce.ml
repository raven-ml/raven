open Nx_core
open Metal

let compute_reduction_params shape axes =
  let ndim = Array.length shape in
  let normalized_axes =
    Array.map (fun ax -> if ax < 0 then ndim + ax else ax) axes
  in
  let sorted_axes = Array.copy normalized_axes in
  Array.sort compare sorted_axes;

  (* Compute output shape *)
  let out_shape =
    Array.mapi (fun i dim -> if Array.mem i sorted_axes then 1 else dim) shape
  in

  (* Compute reduction size and number of reductions *)
  let reduction_size =
    Array.fold_left (fun acc ax -> acc * shape.(ax)) 1 sorted_axes
  in
  let num_reductions = Array.fold_left ( * ) 1 shape / reduction_size in

  (out_shape, reduction_size, num_reductions)

let dispatch_reduce ctx op_name t ~axes ~keepdims =
  let shape = Internal.shape t in
  let out_shape, reduction_size, num_reductions =
    compute_reduction_params shape axes
  in

  (* Create output tensor *)
  let out_size = Array.fold_left ( * ) 1 out_shape in
  let size_bytes = out_size * Internal.sizeof_dtype t.Internal.dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let final_shape =
    if keepdims then out_shape
    else Array.of_list (List.filter (( <> ) 1) (Array.to_list out_shape))
  in
  let view = View.create final_shape in
  let out =
    { context = ctx; Internal.dtype = t.dtype; buffer = metal_buffer; view }
  in

  let dtype_suffix = Internal.dtype_to_metal_type t.dtype in
  let func = Kernels.get_reduce_kernel ctx op_name dtype_suffix in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        t.Internal.buffer.buffer;

      let in_size = Internal.numel t in
      let in_size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int in_size))
      in
      let reduction_size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int reduction_size))
      in
      let num_reductions_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int num_reductions))
      in

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp in_size_val)
        ~length:4 ~index:2;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp reduction_size_val)
        ~length:4 ~index:3;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp num_reductions_val)
        ~length:4 ~index:4;

      (* Pass shape and axes arrays *)
      let ndim = Array.length shape in
      let naxes = Array.length axes in
      let shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      let axes_arr = Ctypes.(allocate_n uint32_t ~count:naxes) in

      for i = 0 to ndim - 1 do
        Ctypes.(shape_arr +@ i <-@ Unsigned.UInt32.of_int shape.(i))
      done;

      for i = 0 to naxes - 1 do
        let ax = if axes.(i) < 0 then ndim + axes.(i) else axes.(i) in
        Ctypes.(axes_arr +@ i <-@ Unsigned.UInt32.of_int ax)
      done;

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp shape_arr)
        ~length:(ndim * 4) ~index:5;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp axes_arr)
        ~length:(naxes * 4) ~index:6;

      let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
      let naxes_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int naxes))
      in

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp ndim_val)
        ~length:4 ~index:7;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp naxes_val)
        ~length:4 ~index:8;

      (* Pass input strides and offset *)
      let in_strides = View.strides t.Internal.view in
      let in_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
      for i = 0 to ndim - 1 do
        Ctypes.(in_strides_arr +@ i <-@ Int32.of_int in_strides.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp in_strides_arr)
        ~length:(ndim * 4) ~index:9;

      let in_offset_val =
        Ctypes.(allocate int32_t (Int32.of_int (View.offset t.Internal.view)))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp in_offset_val)
        ~length:4 ~index:10;

      (* Pass input strides and offset for non-contiguous views *)
      let in_strides = View.strides t.view in
      let in_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
      for i = 0 to ndim - 1 do
        Ctypes.(in_strides_arr +@ i <-@ Int32.of_int in_strides.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp in_strides_arr)
        ~length:(ndim * 4) ~index:9;

      let in_offset_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int (View.offset t.view)))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp in_offset_val)
        ~length:4 ~index:10;

      (* Set threadgroup memory size based on dtype *)
      let elem_size = Internal.sizeof_dtype t.dtype in
      let threadgroup_memory_size = 256 * elem_size in
      ComputeCommandEncoder.set_threadgroup_memory_length encoder
        ~length:threadgroup_memory_size ~index:0;

      (* Dispatch one threadgroup per reduction *)
      let grid_size =
        { Metal.Size.width = num_reductions; height = 1; depth = 1 }
      in
      let group_size = { Metal.Size.width = 256; height = 1; depth = 1 } in

      ComputeCommandEncoder.dispatch_threadgroups encoder
        ~threadgroups_per_grid:grid_size ~threads_per_threadgroup:group_size;
      ComputeCommandEncoder.end_encoding encoder);

  out

let reduce_sum ctx ~axes ~keepdims t =
  dispatch_reduce ctx "reduce_sum" t ~axes ~keepdims

let reduce_max ctx ~axes ~keepdims t =
  dispatch_reduce ctx "reduce_max" t ~axes ~keepdims

let reduce_prod ctx ~axes ~keepdims t =
  dispatch_reduce ctx "reduce_prod" t ~axes ~keepdims
