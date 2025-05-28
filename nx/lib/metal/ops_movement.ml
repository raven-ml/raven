open Nx_core
open Metal

let copy ctx src dst =
  let size = Internal.numel src in
  let dtype_suffix = Internal.dtype_to_metal_type src.Internal.dtype in
  let kernel_name = Printf.sprintf "copy_%s" dtype_suffix in
  let func = Kernels.get_special_kernel ctx kernel_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        dst.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        src.Internal.buffer.buffer;

      let size_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int size)) in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp size_val)
        ~length:4 ~index:2;

      let threads_per_group, num_groups = Internal.compute_thread_groups size in
      let grid_size =
        { Metal.Size.width = num_groups; height = 1; depth = 1 }
      in
      let group_size =
        { Metal.Size.width = threads_per_group; height = 1; depth = 1 }
      in

      ComputeCommandEncoder.dispatch_threadgroups encoder
        ~threadgroups_per_grid:grid_size ~threads_per_threadgroup:group_size;
      ComputeCommandEncoder.end_encoding encoder)

let make_contiguous ctx t =
  let out_size = Internal.numel t in
  let size_bytes = out_size * Internal.sizeof_dtype t.Internal.dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = t.dtype;
      buffer = metal_buffer;
      view = View.create (Internal.shape t);
    }
  in
  let out = { out with view = View.create (Internal.shape t) } in

  let dtype_suffix = Internal.dtype_to_metal_type t.dtype in
  let kernel_name = Printf.sprintf "strided_copy_%s" dtype_suffix in
  let func = Kernels.get_special_kernel ctx kernel_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        t.Internal.buffer.buffer;

      (* Set shape and strides *)
      let shape = Internal.shape t in
      let strides = View.strides t.view in
      let ndim = Array.length shape in

      let shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      let strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in

      for i = 0 to ndim - 1 do
        Ctypes.(shape_arr +@ i <-@ Unsigned.UInt32.of_int shape.(i));
        Ctypes.(strides_arr +@ i <-@ Int32.of_int strides.(i))
      done;

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp shape_arr)
        ~length:(ndim * 4) ~index:2;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp strides_arr)
        ~length:(ndim * 4) ~index:3;

      let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
      let size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int (Internal.numel t)))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp ndim_val)
        ~length:4 ~index:4;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp size_val)
        ~length:4 ~index:5;
      
      (* Pass the offset *)
      let offset_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int (View.offset t.view))) in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp offset_val)
        ~length:4 ~index:6;

      let threads_per_group, num_groups =
        Internal.compute_thread_groups (Internal.numel t)
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

let pad ctx t out padding fill_value =
  let dtype_suffix = Internal.dtype_to_metal_type t.Internal.dtype in
  let kernel_name = Printf.sprintf "pad_%s" dtype_suffix in
  let func = Kernels.get_special_kernel ctx kernel_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        t.Internal.buffer.buffer;

      (* Set shapes and padding info *)
      let out_shape = Internal.shape out in
      let in_shape = Internal.shape t in
      let ndim = Array.length out_shape in

      let out_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      let in_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      let pad_before_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in

      for i = 0 to ndim - 1 do
        Ctypes.(out_shape_arr +@ i <-@ Unsigned.UInt32.of_int out_shape.(i));
        Ctypes.(in_shape_arr +@ i <-@ Unsigned.UInt32.of_int in_shape.(i));
        let before, _ =
          if i < Array.length padding then padding.(i) else (0, 0)
        in
        Ctypes.(pad_before_arr +@ i <-@ Unsigned.UInt32.of_int before)
      done;

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp out_shape_arr)
        ~length:(ndim * 4) ~index:2;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp in_shape_arr)
        ~length:(ndim * 4) ~index:3;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp pad_before_arr)
        ~length:(ndim * 4) ~index:4;

      (* Set fill value based on dtype *)
      let set_fill_value : type a b. (a, b) Dtype.t -> unit = function
        | Dtype.Float32 ->
            let v = Ctypes.(allocate float fill_value) in
            ComputeCommandEncoder.set_bytes encoder
              ~bytes:Ctypes.(to_voidp v)
              ~length:4 ~index:5
        | Dtype.Float64 ->
            let v = Ctypes.(allocate double fill_value) in
            ComputeCommandEncoder.set_bytes encoder
              ~bytes:Ctypes.(to_voidp v)
              ~length:8 ~index:5
        | Dtype.Int32 ->
            let v = Ctypes.(allocate int32_t (Int32.of_float fill_value)) in
            ComputeCommandEncoder.set_bytes encoder
              ~bytes:Ctypes.(to_voidp v)
              ~length:4 ~index:5
        | Dtype.Int64 ->
            let v = Ctypes.(allocate int64_t (Int64.of_float fill_value)) in
            ComputeCommandEncoder.set_bytes encoder
              ~bytes:Ctypes.(to_voidp v)
              ~length:8 ~index:5
        | _ -> failwith "pad: unsupported dtype"
      in
      set_fill_value t.Internal.dtype;

      let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp ndim_val)
        ~length:4 ~index:6;

      let out_size = Internal.numel out in
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
      ComputeCommandEncoder.end_encoding encoder)

let cat ctx tensors axis =
  match tensors with
  | [] -> failwith "cat: empty list"
  | [ t ] ->
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
      copy ctx t out;
      out
  | _ ->
      (* Compute output shape *)
      let first = List.hd tensors in
      let shape = Array.copy (Internal.shape first) in
      let axis = if axis < 0 then Array.length shape + axis else axis in

      (* Sum sizes along concatenation axis *)
      shape.(axis) <-
        List.fold_left (fun acc t -> acc + (Internal.shape t).(axis)) 0 tensors;

      let out_size = Array.fold_left ( * ) 1 shape in
      let size_bytes = out_size * Internal.sizeof_dtype first.Internal.dtype in
      let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
      let metal_buffer = { Internal.buffer; size_bytes } in
      let out =
        {
          context = ctx;
          Internal.dtype = first.dtype;
          buffer = metal_buffer;
          view = View.create shape;
        }
      in
      let out = { out with view = View.create shape } in

      (* Use the concat_axis kernel that handles all axes *)
      let dtype_suffix = Internal.dtype_to_metal_type first.dtype in
      let kernel_name = Printf.sprintf "concat_axis_%s" dtype_suffix in
      let func = Kernels.get_special_kernel ctx kernel_name in
      let pipeline = Kernels.create_compute_pipeline ctx.device func in
      
      let ndim = Array.length shape in
      let axis_offset = ref 0 in
      
      List.iter
        (fun t ->
          (* Make input contiguous if needed *)
          let t_contig = 
            if View.is_contiguous t.Internal.view then t 
            else make_contiguous ctx t 
          in
          
          let in_shape = Internal.shape t_contig in
          let in_size = Internal.numel t_contig in

          Internal.with_command_buffer ctx (fun cmd_buffer ->
              let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

              ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
              ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
                out.Internal.buffer.buffer;
              ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
                t_contig.Internal.buffer.buffer;

              (* Set shape arrays *)
              let out_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
              let in_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
              for i = 0 to ndim - 1 do
                Ctypes.(out_shape_arr +@ i <-@ Unsigned.UInt32.of_int shape.(i));
                Ctypes.(in_shape_arr +@ i <-@ Unsigned.UInt32.of_int in_shape.(i))
              done;
              
              ComputeCommandEncoder.set_bytes encoder
                ~bytes:Ctypes.(to_voidp out_shape_arr)
                ~length:(ndim * 4) ~index:2;
              ComputeCommandEncoder.set_bytes encoder
                ~bytes:Ctypes.(to_voidp in_shape_arr)
                ~length:(ndim * 4) ~index:3;
              
              (* Set axis, axis_offset, and ndim *)
              let axis_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int axis)) in
              let axis_offset_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int !axis_offset)) in
              let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
              
              ComputeCommandEncoder.set_bytes encoder
                ~bytes:Ctypes.(to_voidp axis_val)
                ~length:4 ~index:4;
              ComputeCommandEncoder.set_bytes encoder
                ~bytes:Ctypes.(to_voidp axis_offset_val)
                ~length:4 ~index:5;
              ComputeCommandEncoder.set_bytes encoder
                ~bytes:Ctypes.(to_voidp ndim_val)
                ~length:4 ~index:6;

              let threads_per_group, num_groups =
                Internal.compute_thread_groups in_size
              in
              let grid_size =
                { Metal.Size.width = num_groups; height = 1; depth = 1 }
              in
              let group_size =
                { Metal.Size.width = threads_per_group; height = 1; depth = 1 }
              in

              ComputeCommandEncoder.dispatch_threadgroups encoder
                ~threadgroups_per_grid:grid_size
                ~threads_per_threadgroup:group_size;
              ComputeCommandEncoder.end_encoding encoder);

          axis_offset := !axis_offset + in_shape.(axis))
        tensors;

      out
