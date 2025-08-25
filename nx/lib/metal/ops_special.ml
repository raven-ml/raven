open Nx_core
open Bigarray_ext
open Metal

(* Helper to check if view is C-contiguous *)
let is_c_contiguous view = Lazy_view.is_contiguous view

(* Helper to get offset from view *)
let get_offset view =
  match Symbolic_shape.eval_dim (Lazy_view.offset view) with
  | Some n -> n
  | None ->
      Error.failed ~op:"get_offset" ~what:"cannot evaluate symbolic offset" ()

(* Helper to get strides from view *)
let get_strides view =
  match Lazy_view.strides view with
  | Some s -> s
  | None ->
      Error.failed ~op:"get_strides"
        ~what:"cannot get strides for non-contiguous view" ()

(* Helper to get linear index from view *)
let linear_index view indices =
  (* Since Lazy_view doesn't have linear_index, we need to compute it *)
  let ndim = Symbolic_shape.rank (Lazy_view.shape view) in
  if Array.length indices <> ndim then
    Error.invalid ~op:"linear_index" ~what:"indices"
      ~reason:
        (Printf.sprintf "rank mismatch: %d vs %d" (Array.length indices) ndim)
      ();
  let strides = get_strides view in
  let offset = get_offset view in
  let physical_offset = ref offset in
  Array.iteri
    (fun i idx -> physical_offset := !physical_offset + (idx * strides.(i)))
    indices;
  !physical_offset

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
      view = Lazy_view.create (Symbolic_shape.of_ints shape);
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
        let strides =
          match Lazy_view.strides view with
          | Some s -> s
          | None ->
              Error.failed ~op:"where"
                ~what:"cannot get strides for non-contiguous view" ()
        in
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
            (Unsigned.UInt32.of_int
               (match
                  Symbolic_shape.eval_dim (Lazy_view.offset cond.Internal.view)
                with
               | Some n -> n
               | None ->
                   Error.failed ~op:"where"
                     ~what:"cannot get offset with symbolic value" ())))
      in
      let true_offset =
        Ctypes.(
          allocate uint32_t
            (Unsigned.UInt32.of_int
               (match
                  Symbolic_shape.eval_dim
                    (Lazy_view.offset if_true.Internal.view)
                with
               | Some n -> n
               | None ->
                   Error.failed ~op:"where"
                     ~what:"cannot get offset with symbolic value" ())))
      in
      let false_offset =
        Ctypes.(
          allocate uint32_t
            (Unsigned.UInt32.of_int
               (match
                  Symbolic_shape.eval_dim
                    (Lazy_view.offset if_false.Internal.view)
                with
               | Some n -> n
               | None ->
                   Error.failed ~op:"where"
                     ~what:"cannot get offset with symbolic value" ())))
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
            ~threadgroups_per_grid:grid_size ~threads_per_threadgroup:group_size;
          ComputeCommandEncoder.end_encoding encoder);

      out

let assign ctx dst src =
  (* Check if destination is contiguous - if not, fall back to CPU *)
  if not (is_c_contiguous dst.Internal.view && get_offset dst.Internal.view = 0)
  then
    (* Destination is strided - use CPU fallback for now *)
    Internal.with_command_buffer ctx (fun _cmd_buffer ->
        (* Get CPU-accessible arrays *)
        let get_data : type a b. (a, b) Internal.t -> (a, b, c_layout) Array1.t
            =
         fun t ->
          let contents = Metal.Buffer.contents t.buffer.buffer in
          let elem_size = Internal.sizeof_dtype t.dtype in
          let buffer_size = t.buffer.size_bytes / elem_size in
          match t.dtype with
          | Dtype.BFloat16 | Dtype.Bool | Dtype.Complex16 ->
              (* For extended types, use the extended kind and our special
                 function *)
              let kind = Dtype.to_bigarray_ext_kind t.dtype in
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
              Ctypes.bigarray_of_ptr Ctypes.array1 buffer_size kind
                (Obj.magic contents)
        in

        let src_arr = get_data src in
        let dst_arr = get_data dst in

        let shape = Internal.shape src in
        let size = Internal.numel src in

        (* Copy element by element, respecting views *)
        let md_idx = Array.make (Array.length shape) 0 in
        for linear_idx = 0 to size - 1 do
          (* Get multi-dimensional index *)
          Shape.unravel_index_into linear_idx shape md_idx;

          (* Get offsets in source and destination *)
          let src_offset = linear_index src.view md_idx in
          let dst_offset = linear_index dst.view md_idx in

          (* Copy value *)
          let value = Array1.unsafe_get src_arr src_offset in
          Array1.unsafe_set dst_arr dst_offset value
        done)
  else
    (* Destination is contiguous - use Metal kernel *)
    let shape = Internal.shape src in
    let ndim = Array.length shape in
    let out_size = Internal.numel src in

    let dtype_suffix = Internal.dtype_to_metal_type src.dtype in
    let kernel_name = Printf.sprintf "assign_strided_%s" dtype_suffix in
    let func = Kernels.get_special_kernel ctx kernel_name in
    let pipeline = Kernels.create_compute_pipeline ctx.device func in

    Internal.with_command_buffer ctx (fun cmd_buffer ->
        let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

        ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
        ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
          dst.Internal.buffer.buffer;
        ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
          src.Internal.buffer.buffer;

        (* Set shape array *)
        let shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
        for i = 0 to ndim - 1 do
          Ctypes.(shape_arr +@ i <-@ Unsigned.UInt32.of_int shape.(i))
        done;
        ComputeCommandEncoder.set_bytes encoder
          ~bytes:Ctypes.(to_voidp shape_arr)
          ~length:(ndim * 4) ~index:2;

        (* Set source strides *)
        let src_strides = get_strides src.Internal.view in
        let strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
        for i = 0 to ndim - 1 do
          Ctypes.(strides_arr +@ i <-@ Int32.of_int src_strides.(i))
        done;
        ComputeCommandEncoder.set_bytes encoder
          ~bytes:Ctypes.(to_voidp strides_arr)
          ~length:(ndim * 4) ~index:3;

        (* Set ndim *)
        let ndim_val =
          Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim))
        in
        ComputeCommandEncoder.set_bytes encoder
          ~bytes:Ctypes.(to_voidp ndim_val)
          ~length:4 ~index:4;

        (* Set source offset *)
        let src_offset =
          Ctypes.(
            allocate uint32_t
              (Unsigned.UInt32.of_int (get_offset src.Internal.view)))
        in
        ComputeCommandEncoder.set_bytes encoder
          ~bytes:Ctypes.(to_voidp src_offset)
          ~length:4 ~index:5;

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

let gather ctx data indices axis =
  let data_shape = Internal.shape data in
  let indices_shape = Internal.shape indices in
  let ndim = Array.length data_shape in
  let indices_ndim = Array.length indices_shape in
  let axis = if axis < 0 then ndim + axis else axis in

  (* Validate inputs - must have same rank *)
  if ndim <> indices_ndim then
    invalid_arg
      (Printf.sprintf "gather: data rank (%d) and indices rank (%d) must match"
         ndim indices_ndim);

  (* For NumPy-style gather, output shape is same as indices shape *)
  let out_shape = indices_shape in
  let out_size = Array.fold_left ( * ) 1 out_shape in
  let size_bytes = out_size * Internal.sizeof_dtype data.Internal.dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = data.dtype;
      buffer = metal_buffer;
      view = Lazy_view.create (Symbolic_shape.of_ints out_shape);
    }
  in
  let out =
    { out with view = Lazy_view.create (Symbolic_shape.of_ints out_shape) }
  in

  (* Use Metal kernel for NumPy-style gather *)
  let dtype_suffix = Internal.dtype_to_metal_type data.dtype in
  let kernel_name = Printf.sprintf "gather_strided_%s" dtype_suffix in
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

      (* Set output shape *)
      let shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      for i = 0 to ndim - 1 do
        Ctypes.(shape_arr +@ i <-@ Unsigned.UInt32.of_int out_shape.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp shape_arr)
        ~length:(ndim * 4) ~index:3;

      (* Set data shape *)
      let data_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      for i = 0 to ndim - 1 do
        Ctypes.(data_shape_arr +@ i <-@ Unsigned.UInt32.of_int data_shape.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp data_shape_arr)
        ~length:(ndim * 4) ~index:4;

      (* Set data strides *)
      let data_strides = get_strides data.Internal.view in
      let strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
      for i = 0 to ndim - 1 do
        Ctypes.(strides_arr +@ i <-@ Int32.of_int data_strides.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp strides_arr)
        ~length:(ndim * 4) ~index:5;

      (* Set indices strides *)
      let indices_strides = get_strides indices.Internal.view in
      let indices_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
      for i = 0 to ndim - 1 do
        Ctypes.(indices_strides_arr +@ i <-@ Int32.of_int indices_strides.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp indices_strides_arr)
        ~length:(ndim * 4) ~index:6;

      (* Set ndim *)
      let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp ndim_val)
        ~length:4 ~index:7;

      (* Set axis *)
      let axis_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int axis)) in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp axis_val)
        ~length:4 ~index:8;

      (* Set offsets *)
      let data_offset =
        Ctypes.(
          allocate uint32_t
            (Unsigned.UInt32.of_int (get_offset data.Internal.view)))
      in
      let indices_offset =
        Ctypes.(
          allocate uint32_t
            (Unsigned.UInt32.of_int (get_offset indices.Internal.view)))
      in

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp data_offset)
        ~length:4 ~index:9;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp indices_offset)
        ~length:4 ~index:10;

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

let scatter ?(mode = `Set) ?(unique_indices = false) ctx data_template indices
    updates axis =
  let _ = unique_indices in
  (* TODO: use this hint for optimization *)
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
  let kernel_name =
    match mode with
    | `Set -> Printf.sprintf "scatter_%s" dtype_suffix
    | `Add -> Printf.sprintf "scatter_add_%s" dtype_suffix
  in
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
