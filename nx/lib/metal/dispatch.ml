open Nx_core
open Metal

(* Generic dispatch helpers to reduce boilerplate in ops_*.ml files *)

(* Helper to set shape, strides, and offset parameters *)
let set_view_params encoder view ~shape_index ~strides_index ~offset_index
    ~ndim_index =
  let shape = View.shape view in
  let strides = View.strides view in
  let offset = View.offset view in
  let ndim = Array.length shape in

  (* Allocate arrays *)
  let shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
  let strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in

  (* Fill arrays *)
  for i = 0 to ndim - 1 do
    Ctypes.(shape_arr +@ i <-@ Unsigned.UInt32.of_int shape.(i));
    Ctypes.(strides_arr +@ i <-@ Int32.of_int strides.(i))
  done;

  (* Set parameters *)
  ComputeCommandEncoder.set_bytes encoder
    ~bytes:Ctypes.(to_voidp shape_arr)
    ~length:(ndim * 4) ~index:shape_index;
  ComputeCommandEncoder.set_bytes encoder
    ~bytes:Ctypes.(to_voidp strides_arr)
    ~length:(ndim * 4) ~index:strides_index;

  let offset_val = Ctypes.(allocate int32_t (Int32.of_int offset)) in
  ComputeCommandEncoder.set_bytes encoder
    ~bytes:Ctypes.(to_voidp offset_val)
    ~length:4 ~index:offset_index;

  let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
  ComputeCommandEncoder.set_bytes encoder
    ~bytes:Ctypes.(to_voidp ndim_val)
    ~length:4 ~index:ndim_index

(* Generic unary operation dispatcher *)
let dispatch_unary_op (ctx : Internal.context) (op_name : string)
    (t : ('a, 'b) Internal.t) : ('a, 'b) Internal.t =
  let shape = Internal.shape t in
  let numel = View.numel t.view in

  (* Allocate output buffer *)
  let size_bytes = numel * Internal.sizeof_dtype t.dtype in
  let buffer = Buffer_pool.allocate ctx.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = t.dtype;
      buffer = metal_buffer;
      view = View.create shape;
    }
  in

  (* Get kernel and create pipeline *)
  let func = Kernels.get_unary_kernel ctx t.dtype op_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  (* Dispatch kernel *)
  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in
      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;

      (* Set buffers *)
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        t.buffer.buffer;

      (* Set view parameters - ndim at index 4, offset at index 5 *)
      set_view_params encoder t.view ~shape_index:2 ~strides_index:3
        ~offset_index:5 ~ndim_index:4;

      (* Dispatch threads *)
      let threads_per_group, num_groups =
        Internal.compute_thread_groups numel
      in
      ComputeCommandEncoder.dispatch_threadgroups encoder
        ~threadgroups_per_grid:
          { Metal.Size.width = num_groups; height = 1; depth = 1 }
        ~threads_per_threadgroup:
          { Metal.Size.width = threads_per_group; height = 1; depth = 1 };

      ComputeCommandEncoder.end_encoding encoder);

  out

(* Generic binary operation dispatcher *)
let dispatch_binary_op (ctx : Internal.context) (op_name : string)
    (a : ('a, 'b) Internal.t) (b : ('a, 'b) Internal.t) : ('a, 'b) Internal.t =
  (* Broadcast shapes *)
  let shape_a = Internal.shape a in
  let shape_b = Internal.shape b in
  let out_shape = Shape.broadcast shape_a shape_b in
  let numel = Array.fold_left ( * ) 1 out_shape in

  (* Check broadcast compatibility *)
  if numel = 0 then
    failwith
      (Printf.sprintf "binary op: cannot broadcast shapes %s and %s"
         (Shape.to_string shape_a) (Shape.to_string shape_b));

  (* Allocate output buffer *)
  let size_bytes = numel * Internal.sizeof_dtype a.dtype in
  let buffer = Buffer_pool.allocate ctx.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = a.dtype;
      buffer = metal_buffer;
      view = View.create out_shape;
    }
  in

  (* Get kernel and create pipeline *)
  let func = Kernels.get_binary_kernel ctx a.dtype op_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  (* Helper to compute broadcast strides *)
  let ndim = Array.length out_shape in

  let get_broadcast_strides tensor =
    let t_shape = Internal.shape tensor in
    let t_strides = View.strides tensor.view in
    let t_ndim = Array.length t_shape in
    let broadcast_strides = Array.make ndim 0 in
    let shape_offset = ndim - t_ndim in
    for i = 0 to t_ndim - 1 do
      if t_shape.(i) = 1 then broadcast_strides.(i + shape_offset) <- 0
      else broadcast_strides.(i + shape_offset) <- t_strides.(i)
    done;
    broadcast_strides
  in

  let a_strides = get_broadcast_strides a in
  let b_strides = get_broadcast_strides b in

  (* Dispatch kernel *)
  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in
      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;

      (* Set buffers *)
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        a.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:2
        b.buffer.buffer;

      (* Set output shape first at index 3 *)
      let out_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      for i = 0 to ndim - 1 do
        Ctypes.(out_shape_arr +@ i <-@ Unsigned.UInt32.of_int out_shape.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp out_shape_arr)
        ~length:(ndim * 4) ~index:3;

      (* Set stride arrays *)
      let a_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
      let b_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in

      for i = 0 to ndim - 1 do
        Ctypes.(a_strides_arr +@ i <-@ Int32.of_int a_strides.(i));
        Ctypes.(b_strides_arr +@ i <-@ Int32.of_int b_strides.(i))
      done;

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp a_strides_arr)
        ~length:(ndim * 4) ~index:4;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp b_strides_arr)
        ~length:(ndim * 4) ~index:5;

      (* Set ndim *)
      let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp ndim_val)
        ~length:4 ~index:6;

      (* Set offsets *)
      let a_offset_val =
        Ctypes.(allocate int32_t (Int32.of_int (View.offset a.view)))
      in
      let b_offset_val =
        Ctypes.(allocate int32_t (Int32.of_int (View.offset b.view)))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp a_offset_val)
        ~length:4 ~index:7;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp b_offset_val)
        ~length:4 ~index:8;

      (* Dispatch threads *)
      let threads_per_group, num_groups =
        Internal.compute_thread_groups numel
      in
      ComputeCommandEncoder.dispatch_threadgroups encoder
        ~threadgroups_per_grid:
          { Metal.Size.width = num_groups; height = 1; depth = 1 }
        ~threads_per_threadgroup:
          { Metal.Size.width = threads_per_group; height = 1; depth = 1 };

      ComputeCommandEncoder.end_encoding encoder);

  out

(* Generic comparison operation dispatcher - outputs uint8 *)
let dispatch_comparison_op (ctx : Internal.context) (op_name : string)
    (a : ('a, 'b) Internal.t) (b : ('a, 'b) Internal.t) :
    (int, Dtype.uint8_elt) Internal.t =
  (* Broadcast shapes *)
  let shape_a = Internal.shape a in
  let shape_b = Internal.shape b in
  let out_shape = Shape.broadcast shape_a shape_b in
  let numel = Array.fold_left ( * ) 1 out_shape in

  (* Check broadcast compatibility *)
  if numel = 0 then
    failwith
      (Printf.sprintf "comparison op: cannot broadcast shapes %s and %s"
         (Shape.to_string shape_a) (Shape.to_string shape_b));

  (* Allocate output buffer - always uint8 *)
  let size_bytes = numel * Internal.sizeof_dtype Dtype.UInt8 in
  let buffer = Buffer_pool.allocate ctx.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = Dtype.UInt8;
      buffer = metal_buffer;
      view = View.create out_shape;
    }
  in

  (* Get kernel and create pipeline *)
  let func = Kernels.get_binary_kernel ctx a.dtype op_name in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  (* Helper to compute broadcast strides *)
  let ndim = Array.length out_shape in

  let get_broadcast_strides tensor =
    let t_shape = Internal.shape tensor in
    let t_strides = View.strides tensor.view in
    let t_ndim = Array.length t_shape in
    let broadcast_strides = Array.make ndim 0 in
    let shape_offset = ndim - t_ndim in
    for i = 0 to t_ndim - 1 do
      if t_shape.(i) = 1 then broadcast_strides.(i + shape_offset) <- 0
      else broadcast_strides.(i + shape_offset) <- t_strides.(i)
    done;
    broadcast_strides
  in

  let a_strides = get_broadcast_strides a in
  let b_strides = get_broadcast_strides b in

  (* Dispatch kernel *)
  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in
      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;

      (* Set buffers *)
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        a.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:2
        b.buffer.buffer;

      (* Set output shape first at index 3 *)
      let out_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim) in
      for i = 0 to ndim - 1 do
        Ctypes.(out_shape_arr +@ i <-@ Unsigned.UInt32.of_int out_shape.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp out_shape_arr)
        ~length:(ndim * 4) ~index:3;

      (* Set stride arrays *)
      let a_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
      let b_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in

      for i = 0 to ndim - 1 do
        Ctypes.(a_strides_arr +@ i <-@ Int32.of_int a_strides.(i));
        Ctypes.(b_strides_arr +@ i <-@ Int32.of_int b_strides.(i))
      done;

      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp a_strides_arr)
        ~length:(ndim * 4) ~index:4;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp b_strides_arr)
        ~length:(ndim * 4) ~index:5;

      (* Set ndim *)
      let ndim_val = Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim)) in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp ndim_val)
        ~length:4 ~index:6;

      (* Set offsets *)
      let a_offset_val =
        Ctypes.(allocate int32_t (Int32.of_int (View.offset a.view)))
      in
      let b_offset_val =
        Ctypes.(allocate int32_t (Int32.of_int (View.offset b.view)))
      in
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp a_offset_val)
        ~length:4 ~index:7;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp b_offset_val)
        ~length:4 ~index:8;

      (* Dispatch threads *)
      let threads_per_group, num_groups =
        Internal.compute_thread_groups numel
      in
      ComputeCommandEncoder.dispatch_threadgroups encoder
        ~threadgroups_per_grid:
          { Metal.Size.width = num_groups; height = 1; depth = 1 }
        ~threads_per_threadgroup:
          { Metal.Size.width = threads_per_group; height = 1; depth = 1 };

      ComputeCommandEncoder.end_encoding encoder);

  out

(* Generic reduce operation dispatcher *)
let dispatch_reduce_op (ctx : Internal.context) (op_name : string)
    ~(axes : int array) ?(keepdims = false) (t : ('a, 'b) Internal.t) :
    ('a, 'b) Internal.t =
  let shape = Internal.shape t in
  let ndim = Array.length shape in

  (* Handle axes - normalize negative indices *)
  let axes = Array.map (fun ax -> if ax < 0 then ndim + ax else ax) axes in

  (* Calculate output shape *)
  let out_shape =
    if keepdims then
      Array.mapi (fun i dim -> if Array.mem i axes then 1 else dim) shape
    else
      Array.of_list
        (List.mapi
           (fun i dim -> if Array.mem i axes then None else Some dim)
           (Array.to_list shape)
        |> List.filter_map (fun x -> x))
  in

  let out_numel = Array.fold_left ( * ) 1 out_shape in

  (* Allocate output buffer *)
  let size_bytes = out_numel * Internal.sizeof_dtype t.dtype in
  let buffer = Buffer_pool.allocate ctx.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = t.dtype;
      buffer = metal_buffer;
      view = View.create out_shape;
    }
  in

  (* Get kernel and create pipeline *)
  let dtype_suffix = Internal.dtype_to_metal_type t.dtype in
  let func = Kernels.get_reduce_kernel ctx op_name dtype_suffix in
  let pipeline = Kernels.create_compute_pipeline ctx.device func in

  (* Calculate reduction parameters *)
  let sorted_axes = Array.copy axes in
  Array.sort compare sorted_axes;

  let reduction_size =
    Array.fold_left (fun acc ax -> acc * shape.(ax)) 1 sorted_axes
  in
  let num_reductions = Array.fold_left ( * ) 1 shape / reduction_size in

  (* Dispatch kernel *)
  Internal.with_command_buffer ctx (fun cmd_buffer ->
      let encoder = ComputeCommandEncoder.on_buffer cmd_buffer in
      ComputeCommandEncoder.set_compute_pipeline_state encoder pipeline;

      (* Set buffers *)
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:0
        out.buffer.buffer;
      ComputeCommandEncoder.set_buffer encoder ~offset:0 ~index:1
        t.buffer.buffer;

      (* Set scalar parameters *)
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
      let in_strides = View.strides t.view in
      let in_strides_arr = Ctypes.(allocate_n int32_t ~count:ndim) in
      for i = 0 to ndim - 1 do
        Ctypes.(in_strides_arr +@ i <-@ Int32.of_int in_strides.(i))
      done;
      ComputeCommandEncoder.set_bytes encoder
        ~bytes:Ctypes.(to_voidp in_strides_arr)
        ~length:(ndim * 4) ~index:9;

      let in_offset_val =
        Ctypes.(allocate int32_t (Int32.of_int (View.offset t.view)))
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
