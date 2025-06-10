open Nx_core
open Metal

let where ctx cond if_true if_false =
  (* All inputs should have the same shape *)
  let shape = Internal.shape if_true in
  let ndim = Array.length shape in
  let out_size = Internal.numel if_true in
  let size_bytes = out_size * Internal.sizeof_dtype if_true.Internal.dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = if_true.dtype;
      buffer = metal_buffer;
      view = View.create shape;
    }
  in

  let dtype_suffix = Internal.dtype_to_metal_type if_true.dtype in
  let kernel_name = Printf.sprintf "where_%s" dtype_suffix in
  let func = Kernels.get_special_kernel ctx kernel_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        cond.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:2
        if_true.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:3
        if_false.Internal.buffer.buffer;

      (* Set shape array *)
      let shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      for i = 0 to ndim - 1 do
        Ctypes.(shape_arr +@ i <-@ Unsigned.UInt32.of_int shape.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp shape_arr)
        ~length:(ndim * 4) ~index:4;

      (* Set strides for each input *)
      let set_strides view index =
        let strides = View.strides view in
        let strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
        for i = 0 to ndim - 1 do
          Ctypes.(strides_arr +@ i <-@ Int32.of_int strides.(i))
        done;
        ComputeCommandEncoder.set_bytes encoder
          ~bytes:Ctypes.(to_voidp strides_arr)
          ~length:(ndim * 4) ~index
      in
      set_strides cond.Internal.view 5;
      set_strides if_true.Internal.view 6;
      set_strides if_false.Internal.view 7;

      (* Set ndim *)
      let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp ndim_val)
        ~length:4 ~index:8;

      (* Set offsets *)
      let cond_offset =
        Ctypes.(
          allocate uint32_t
            (Unsigned.UInt32.of_int (View.offset cond.Internal.view)))
      in
      let true_offset =
        Ctypes.(
          allocate uint32_t
            (Unsigned.UInt32.of_int (View.offset if_true.Internal.view)))
      in
      let false_offset =
        Ctypes.(
          allocate uint32_t
            (Unsigned.UInt32.of_int (View.offset if_false.Internal.view)))
      in

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp cond_offset)
        ~length:4 ~index:9;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp true_offset)
        ~length:4 ~index:10;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp false_offset)
        ~length:4 ~index:11;

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

let cast : type a b c d.
    Internal.context -> (a, b) Internal.t -> (c, d) Dtype.t -> (c, d) Internal.t
    =
 fun ctx t target_dtype ->
  match Dtype.equal_witness t.Internal.dtype target_dtype with
  | Some Equal ->
      (* Same dtype - just copy *)
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
      Ops_movement.copy ctx t out;
      out
  | None ->
      (* Check if source or target types are unsupported *)
      let is_complex_type : type x y. (x, y) Dtype.t -> bool = function
        | Dtype.Complex32 -> true
        | Dtype.Complex64 -> true
        | _ -> false
      in
      if is_complex_type t.Internal.dtype || is_complex_type target_dtype then
        failwith "Metal backend does not support complex types"
      else
        let out_size = Internal.numel t in
        let size_bytes = out_size * Internal.sizeof_dtype target_dtype in
        let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
        let metal_buffer = { Internal.buffer; size_bytes } in
        let out =
          {
            context = ctx;
            Internal.dtype = target_dtype;
            buffer = metal_buffer;
            view = t.view;
          }
        in

        (* Construct kernel name based on source and target types *)
        let src_type = Internal.dtype_to_metal_type t.dtype in
        let dst_type = Internal.dtype_to_metal_type target_dtype in
        let kernel_name = Printf.sprintf "cast_%s_to_%s" src_type dst_type in

        let func = Kernels.get_special_kernel ctx kernel_name in
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
              ~threadgroups_per_grid:grid_size
              ~threads_per_threadgroup:group_size;
            ComputeCommandEncoder.end_encoding encoder);

        out

let assign ctx dst src =
  (* For now, just copy the data *)
  Ops_movement.copy ctx src dst

let gather ctx data indices axis =
  let data_shape = Internal.shape data in
  let indices_shape = Internal.shape indices in
  let ndim = Array.length data_shape in
  let axis = if axis < 0 then ndim + axis else axis in

  (* Output shape: data.shape[:axis] + indices.shape + data.shape[axis+1:] *)
  let out_shape =
    Array.concat
      [
        Array.sub data_shape 0 axis;
        indices_shape;
        Array.sub data_shape (axis + 1) (ndim - axis - 1);
      ]
  in
  let out_size = Array.fold_left ( * ) 1 out_shape in
  let size_bytes = out_size * Internal.sizeof_dtype data.Internal.dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = data.dtype;
      buffer = metal_buffer;
      view = View.create out_shape;
    }
  in
  let out = { out with view = View.create out_shape } in

  (* Compute dimensions for gather *)
  let axis_size = data_shape.(axis) in
  let inner_size =
    let rec prod i = if i >= ndim then 1 else data_shape.(i) * prod (i + 1) in
    prod (axis + 1)
  in
  let outer_size =
    let rec prod i = if i >= axis then 1 else data_shape.(i) * prod (i + 1) in
    prod 0
  in
  let indices_size = Internal.numel indices in

  let dtype_suffix = Internal.dtype_to_metal_type data.dtype in
  let kernel_name = Printf.sprintf "gather_%s" dtype_suffix in
  let func = Kernels.get_special_kernel ctx kernel_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        data.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:2
        indices.Internal.buffer.buffer;

      let axis_size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int axis_size))
      in
      let inner_size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int inner_size))
      in
      let indices_size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int indices_size))
      in
      let outer_size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int outer_size))
      in

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp axis_size_val)
        ~length:4 ~index:3;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp inner_size_val)
        ~length:4 ~index:4;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp indices_size_val)
        ~length:4 ~index:5;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp outer_size_val)
        ~length:4 ~index:6;

      let total_work = outer_size * indices_size * inner_size in
      let threads_per_group, num_groups =
        Internal.compute_thread_groups total_work
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

let scatter ctx data_template indices updates axis =
  (* Create output initialized with data_template *)
  (* Create output initialized with data_template *)
  let out_size = Internal.numel data_template in
  let size_bytes =
    out_size * Internal.sizeof_dtype data_template.Internal.dtype
  in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = data_template.dtype;
      buffer = metal_buffer;
      view = data_template.view;
    }
  in
  Ops_movement.copy ctx data_template out;

  let data_shape = Internal.shape data_template in
  let ndim = Array.length data_shape in
  let axis = if axis < 0 then ndim + axis else axis in

  (* Compute dimensions for scatter *)
  let axis_size = data_shape.(axis) in
  let inner_size =
    let rec prod i = if i >= ndim then 1 else data_shape.(i) * prod (i + 1) in
    prod (axis + 1)
  in
  let indices_size = Internal.numel indices in

  let dtype_suffix = Internal.dtype_to_metal_type data_template.dtype in
  let kernel_name = Printf.sprintf "scatter_%s" dtype_suffix in
  let func = Kernels.get_special_kernel ctx kernel_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        indices.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:2
        updates.Internal.buffer.buffer;

      let axis_size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int axis_size))
      in
      let inner_size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int inner_size))
      in
      let indices_size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int indices_size))
      in

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp axis_size_val)
        ~length:4 ~index:3;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp inner_size_val)
        ~length:4 ~index:4;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp indices_size_val)
        ~length:4 ~index:5;

      let total_work = indices_size * inner_size in
      let threads_per_group, num_groups =
        Internal.compute_thread_groups total_work
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

let threefry ctx key counter =
  let out_size = Internal.numel counter in
  let size_bytes = out_size * Internal.sizeof_dtype Dtype.Int32 in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = Dtype.Int32;
      buffer = metal_buffer;
      view = counter.view;
    }
  in
  let out = { out with view = counter.view } in

  let func = Kernels.get_special_kernel ctx "threefry_int32" in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        key.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:2
        counter.Internal.buffer.buffer;

      let size_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int out_size))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp size_val)
        ~length:4 ~index:3;

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
