open Nx_core
open Metal

(* im2col/col2im operations for efficient convolution *)

let op_unfold ctx t ~kernel_size ~stride ~dilation ~padding =
  let t_shape = Internal.shape t in
  let ndim = Array.length t_shape in
  let n_spatial = Array.length kernel_size in

  if ndim < n_spatial + 2 then
    invalid_arg
      "op_unfold: input must have at least batch and channel dimensions";

  let batch_size = t_shape.(0) in
  let channels = t_shape.(1) in
  let _spatial_dims = Array.sub t_shape 2 n_spatial in

  (* Apply padding *)
  let pad_config = Array.concat [ Array.make 2 (0, 0); padding ] in
  let t_padded =
    if Array.for_all (fun (before, after) -> before = 0 && after = 0) padding
    then t
    else
      let old_shape =
        match Symbolic_shape.eval (Lazy_view.shape t.Internal.view) with
        | Some arr -> arr
        | None ->
            Error.failed ~op:"ensure_3d_for_conv"
              ~what:"cannot evaluate symbolic shape" ()
      in
      let new_shape =
        Array.mapi
          (fun i dim ->
            if i < Array.length pad_config then
              let before, after = pad_config.(i) in
              dim + before + after
            else dim)
          old_shape
      in
      let out_size = Array.fold_left ( * ) 1 new_shape in
      let size_bytes = out_size * Internal.sizeof_dtype t.Internal.dtype in
      let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
      let metal_buffer = { Internal.buffer; size_bytes } in
      let out =
        {
          context = ctx;
          Internal.dtype = t.dtype;
          buffer = metal_buffer;
          view = Lazy_view.create (Symbolic_shape.of_ints new_shape);
        }
      in

      (* Use the pad kernel *)
      let dtype_suffix = Internal.dtype_to_metal_type t.dtype in
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
              if i < Array.length pad_config then pad_config.(i) else (0, 0)
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

          (* Set fill value to 0 *)
          let set_fill_value : type a b. (a, b) Dtype.t -> unit = function
            | Dtype.Float32 ->
                let v = Ctypes.(allocate float 0.0) in
                ComputeCommandEncoder.set_bytes encoder
                  ~bytes:Ctypes.(to_voidp v)
                  ~length:4 ~index:5
            | Dtype.Float64 ->
                let v = Ctypes.(allocate double 0.0) in
                ComputeCommandEncoder.set_bytes encoder
                  ~bytes:Ctypes.(to_voidp v)
                  ~length:8 ~index:5
            | Dtype.Int32 ->
                let v = Ctypes.(allocate int32_t Int32.zero) in
                ComputeCommandEncoder.set_bytes encoder
                  ~bytes:Ctypes.(to_voidp v)
                  ~length:4 ~index:5
            | Dtype.Int64 ->
                let v = Ctypes.(allocate int64_t Int64.zero) in
                ComputeCommandEncoder.set_bytes encoder
                  ~bytes:Ctypes.(to_voidp v)
                  ~length:8 ~index:5
            | _ -> failwith "pad: unsupported dtype"
          in
          set_fill_value t.Internal.dtype;

          let ndim_val =
            Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim))
          in
          ComputeCommandEncoder.set_bytes encoder
            ~bytes:Ctypes.(to_voidp ndim_val)
            ~length:4 ~index:6;

          (* Set input strides *)
          let in_strides =
            match Lazy_view.strides t.Internal.view with
            | Some s -> s
            | None ->
                Error.failed ~op:"unfold"
                  ~what:"cannot get strides for non-contiguous view" ()
          in
          let in_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
          for i = 0 to ndim - 1 do
            Ctypes.(in_strides_arr +@ i <-@ Int32.of_int in_strides.(i))
          done;
          ComputeCommandEncoder.set_bytes encoder
            ~bytes:Ctypes.(to_voidp in_strides_arr)
            ~length:(ndim * 4) ~index:7;

          (* Set input offset *)
          let in_offset =
            match
              Symbolic_shape.eval_dim (Lazy_view.offset t.Internal.view)
            with
            | Some n -> n
            | None ->
                Error.failed ~op:"unfold"
                  ~what:"cannot evaluate symbolic offset" ()
          in
          let in_offset_val =
            Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int in_offset))
          in
          ComputeCommandEncoder.set_bytes encoder
            ~bytes:Ctypes.(to_voidp in_offset_val)
            ~length:4 ~index:8;

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
          ComputeCommandEncoder.end_encoding encoder);
      out
  in
  let padded_shape = Internal.shape t_padded in
  let padded_spatial = Array.sub padded_shape 2 n_spatial in

  (* Calculate output dimensions *)
  let output_spatial_dims =
    Array.init n_spatial (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  let num_blocks = Array.fold_left ( * ) 1 output_spatial_dims in
  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let col_shape = [| batch_size; channels * kernel_elements; num_blocks |] in

  (* Create output buffer *)
  let out_size = Array.fold_left ( * ) 1 col_shape in
  let size_bytes = out_size * Internal.sizeof_dtype t.dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = t.dtype;
      buffer = metal_buffer;
      view = Lazy_view.create (Symbolic_shape.of_ints col_shape);
    }
  in

  (* Prepare parameters for kernel *)
  let dtype_suffix = Internal.dtype_to_metal_type t.dtype in
  let kernel_name = Printf.sprintf "unfold_%s" dtype_suffix in
  let func = Kernels.get_special_kernel ctx kernel_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        t_padded.Internal.buffer.buffer;

      (* Set shape arrays - in_shape includes batch, channels, and spatial
         dims *)
      let in_shape =
        Array.concat [ [| batch_size; channels |]; padded_spatial ]
      in
      let in_shape_arr =
        Ctypes.(allocate_n uint32_t ~count:(Array.length in_shape))
      in
      for i = 0 to Array.length in_shape - 1 do
        Ctypes.(in_shape_arr +@ i <-@ Unsigned.UInt32.of_int in_shape.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp in_shape_arr)
        ~length:(Array.length in_shape * 4)
        ~index:2;

      (* Set kernel_size array *)
      let kernel_size_arr = Ctypes.(allocate_n uint32_t ~count:n_spatial) in
      for i = 0 to n_spatial - 1 do
        Ctypes.(kernel_size_arr +@ i <-@ Unsigned.UInt32.of_int kernel_size.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp kernel_size_arr)
        ~length:(n_spatial * 4) ~index:3;

      (* Set stride array *)
      let stride_arr = Ctypes.(allocate_n uint32_t ~count:n_spatial) in
      for i = 0 to n_spatial - 1 do
        Ctypes.(stride_arr +@ i <-@ Unsigned.UInt32.of_int stride.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp stride_arr)
        ~length:(n_spatial * 4) ~index:4;

      (* Set dilation array *)
      let dilation_arr = Ctypes.(allocate_n uint32_t ~count:n_spatial) in
      for i = 0 to n_spatial - 1 do
        Ctypes.(dilation_arr +@ i <-@ Unsigned.UInt32.of_int dilation.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp dilation_arr)
        ~length:(n_spatial * 4) ~index:5;

      (* Set padding array - pad_before and pad_after for each dim *)
      let padding_arr = Ctypes.(allocate_n uint32_t ~count:(n_spatial * 2)) in
      for i = 0 to n_spatial - 1 do
        Ctypes.(
          padding_arr +@ (i * 2) <-@ Unsigned.UInt32.of_int (fst padding.(i)));
        Ctypes.(
          padding_arr +@ ((i * 2) + 1)
          <-@ Unsigned.UInt32.of_int (snd padding.(i)))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp padding_arr)
        ~length:(n_spatial * 2 * 4)
        ~index:6;

      (* Set output spatial dimensions *)
      let out_spatial_arr = Ctypes.(allocate_n uint32_t ~count:n_spatial) in
      for i = 0 to n_spatial - 1 do
        Ctypes.(
          out_spatial_arr +@ i
          <-@ Unsigned.UInt32.of_int output_spatial_dims.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp out_spatial_arr)
        ~length:(n_spatial * 4) ~index:7;

      (* Set scalar values *)
      let n_spatial_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int n_spatial))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp n_spatial_val)
        ~length:4 ~index:8;

      let channels_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int channels))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp channels_val)
        ~length:4 ~index:9;

      let kernel_elements_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int kernel_elements))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp kernel_elements_val)
        ~length:4 ~index:10;

      let num_blocks_val =
        Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int num_blocks))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp num_blocks_val)
        ~length:4 ~index:11;

      (* Dispatch kernel - using 3D grid for better thread organization *)
      let threads_per_group = 256 in
      let col_row_size = channels * kernel_elements in
      let group_size =
        {
          Metal.Size.width = min threads_per_group col_row_size;
          height = 1;
          depth = 1;
        }
      in

      ComputeCommandEncoder.dispatch_threadgroups encoder
        ~threadgroups_per_grid:
          {
            Metal.Size.width =
              (col_row_size + threads_per_group - 1) / threads_per_group;
            height = num_blocks;
            depth = batch_size;
          }
        ~threads_per_threadgroup:group_size;
      ComputeCommandEncoder.end_encoding encoder);

  out

let op_fold ctx t ~output_size ~kernel_size ~stride ~dilation ~padding =
  let t_shape = Internal.shape t in
  if Array.length t_shape <> 3 then
    invalid_arg
      "op_fold: input must be 3D (batch, channels * kernel_elements, blocks)";

  let batch_size = t_shape.(0) in
  let col_channels = t_shape.(1) in
  let num_blocks = t_shape.(2) in

  let n_spatial = Array.length output_size in
  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let channels = col_channels / kernel_elements in

  if channels * kernel_elements <> col_channels then
    invalid_arg "op_fold: col_channels must be divisible by kernel_elements";

  (* Calculate padded output size *)
  let padded_size =
    Array.init n_spatial (fun i ->
        output_size.(i) + fst padding.(i) + snd padding.(i))
  in

  (* Create padded output buffer *)
  let padded_shape = Array.concat [ [| batch_size; channels |]; padded_size ] in
  let out_size = Array.fold_left ( * ) 1 padded_shape in
  let size_bytes = out_size * Internal.sizeof_dtype t.dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out_padded =
    {
      context = ctx;
      Internal.dtype = t.dtype;
      buffer = metal_buffer;
      view = Lazy_view.create (Symbolic_shape.of_ints padded_shape);
    }
  in

  (* Initialize to zero *)
  let fill_func = Kernels.get_unary_kernel ctx t.dtype "fill" in
  let fill_pipeline = Kernels.create_compute_pipeline ctx.device fill_func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder fill_pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out_padded.Internal.buffer.buffer;

      (* Set fill value to 0 *)
      let set_fill_value : type a b. (a, b) Dtype.t -> unit = function
        | Dtype.Float32 ->
            let v = Ctypes.(allocate float 0.0) in
            ComputeCommandEncoder.set_bytes encoder
              ~bytes:Ctypes.(to_voidp v)
              ~length:4 ~index:1
        | Dtype.Float64 ->
            let v = Ctypes.(allocate double 0.0) in
            ComputeCommandEncoder.set_bytes encoder
              ~bytes:Ctypes.(to_voidp v)
              ~length:8 ~index:1
        | Dtype.Int32 ->
            let v = Ctypes.(allocate int32_t Int32.zero) in
            ComputeCommandEncoder.set_bytes encoder
              ~bytes:Ctypes.(to_voidp v)
              ~length:4 ~index:1
        | Dtype.Int64 ->
            let v = Ctypes.(allocate int64_t Int64.zero) in
            ComputeCommandEncoder.set_bytes encoder
              ~bytes:Ctypes.(to_voidp v)
              ~length:8 ~index:1
        | _ -> failwith "fold: unsupported dtype"
      in
      set_fill_value t.dtype;

      let out_size = Internal.numel out_padded in
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

  (* Calculate output block dimensions *)
  let output_spatial_dims =
    Array.init n_spatial (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_size.(i) - effective_kernel) / stride.(i)) + 1)
  in

  (* Run fold kernel *)
  let dtype_suffix = Internal.dtype_to_metal_type t.dtype in
  let kernel_name = Printf.sprintf "fold_%s" dtype_suffix in
  let func = Kernels.get_special_kernel ctx kernel_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in

      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out_padded.Internal.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        t.Internal.buffer.buffer;

      (* Set dimensions and parameters *)
      let params =
        Ctypes.(allocate_n uint32_t ~count:(6 + (6 * n_spatial) + 2))
      in

      (* Basic dimensions *)
      Ctypes.(params +@ 0 <-@ Unsigned.UInt32.of_int batch_size);
      Ctypes.(params +@ 1 <-@ Unsigned.UInt32.of_int channels);
      Ctypes.(params +@ 2 <-@ Unsigned.UInt32.of_int n_spatial);

      (* Padded spatial dimensions *)
      for i = 0 to n_spatial - 1 do
        Ctypes.(params +@ (3 + i) <-@ Unsigned.UInt32.of_int padded_size.(i))
      done;

      (* Kernel sizes *)
      for i = 0 to n_spatial - 1 do
        Ctypes.(
          params +@ (3 + n_spatial + i)
          <-@ Unsigned.UInt32.of_int kernel_size.(i))
      done;

      (* Strides *)
      for i = 0 to n_spatial - 1 do
        Ctypes.(
          params +@ (3 + (2 * n_spatial) + i)
          <-@ Unsigned.UInt32.of_int stride.(i))
      done;

      (* Dilations *)
      for i = 0 to n_spatial - 1 do
        Ctypes.(
          params +@ (3 + (3 * n_spatial) + i)
          <-@ Unsigned.UInt32.of_int dilation.(i))
      done;

      (* Padding offsets *)
      for i = 0 to n_spatial - 1 do
        Ctypes.(
          params +@ (3 + (4 * n_spatial) + i)
          <-@ Unsigned.UInt32.of_int (fst padding.(i)))
      done;

      (* Output spatial dimensions *)
      for i = 0 to n_spatial - 1 do
        Ctypes.(
          params +@ (3 + (5 * n_spatial) + i)
          <-@ Unsigned.UInt32.of_int output_spatial_dims.(i))
      done;

      Ctypes.(
        params +@ (3 + (6 * n_spatial)) <-@ Unsigned.UInt32.of_int num_blocks);
      Ctypes.(
        params +@ (3 + (6 * n_spatial) + 1)
        <-@ Unsigned.UInt32.of_int kernel_elements);

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp params)
        ~length:((6 + (6 * n_spatial) + 2) * 4)
        ~index:2;

      (* Dispatch kernel *)
      let total_threads =
        batch_size * num_blocks * channels * kernel_elements
      in
      let threads_per_group, num_groups =
        Internal.compute_thread_groups total_threads
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

  (* Remove padding if needed *)
  if Array.for_all2 (fun p q -> p = q) padding (Array.make n_spatial (0, 0))
  then out_padded
  else
    let bounds =
      Array.concat
        [
          [| (0, batch_size); (0, channels) |];
          Array.mapi
            (fun i _ -> (fst padding.(i), padded_size.(i) - snd padding.(i)))
            output_size;
        ]
    in
    { out_padded with view = Lazy_view.shrink bounds out_padded.view }
