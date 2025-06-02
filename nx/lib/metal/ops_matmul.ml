open Nx_core
open Metal

(* Helper to check if view is C-contiguous *)
let is_c_contiguous view = Lazy_view.is_contiguous view

(* Matrix multiplication operation *)

let op_matmul ctx a b =
  let shape_a = Internal.shape a in
  let shape_b = Internal.shape b in
  let ndim_a = Array.length shape_a in
  let ndim_b = Array.length shape_b in

  if ndim_a < 2 || ndim_b < 2 then failwith "matmul: inputs must be at least 2D";

  (* Check matrix dimensions compatibility *)
  let m = shape_a.(ndim_a - 2) in
  let k_a = shape_a.(ndim_a - 1) in
  let k_b = shape_b.(ndim_b - 2) in
  let n = shape_b.(ndim_b - 1) in

  if k_a <> k_b then
    invalid_arg
      (Printf.sprintf
         "dot: cannot contract %s (last axis: %d) to %s (axis %d: %d) (size \
          %d≠%d)"
         (Shape.to_string shape_a) k_a (Shape.to_string shape_b) (ndim_b - 2)
         k_b k_a k_b);

  (* Extract batch dimensions *)
  let batch_a = Array.sub shape_a 0 (ndim_a - 2) in
  let batch_b = Array.sub shape_b 0 (ndim_b - 2) in

  (* Broadcast batch dimensions *)
  let max_batch_ndim = max (Array.length batch_a) (Array.length batch_b) in
  let batch_shape = Array.make max_batch_ndim 1 in

  (* Fill from the right *)
  for i = 0 to Array.length batch_a - 1 do
    batch_shape.(max_batch_ndim - Array.length batch_a + i) <- batch_a.(i)
  done;

  for i = 0 to Array.length batch_b - 1 do
    let idx = max_batch_ndim - Array.length batch_b + i in
    if batch_shape.(idx) = 1 then batch_shape.(idx) <- batch_b.(i)
    else if batch_b.(i) <> 1 && batch_b.(i) <> batch_shape.(idx) then
      failwith
        (Printf.sprintf "matmul: cannot broadcast shapes %s and %s"
           (Shape.to_string shape_a) (Shape.to_string shape_b))
  done;

  (* Output shape is batch_shape + [m; n] *)
  let out_shape = Array.concat [ batch_shape; [| m; n |] ] in

  (* Create output buffer *)
  let out_size = Array.fold_left ( * ) 1 out_shape in
  let size_bytes = out_size * Internal.sizeof_dtype a.dtype in
  let buffer = Buffer_pool.allocate ctx.Internal.pool size_bytes in
  let metal_buffer = { Internal.buffer; size_bytes } in
  let out =
    {
      context = ctx;
      Internal.dtype = a.dtype;
      buffer = metal_buffer;
      view = Lazy_view.create (Symbolic_shape.of_ints out_shape);
    }
  in

  (* Handle empty matrices - if any dimension is 0, return empty output *)
  if m = 0 || n = 0 || k_a = 0 then out
  else
    (* Ensure inputs are contiguous - Metal kernel doesn't support strided
       views *)
    let a =
      if is_c_contiguous a.view then a else Ops_movement.make_contiguous ctx a
    in
    let b =
      if is_c_contiguous b.view then b else Ops_movement.make_contiguous ctx b
    in

    (* Get kernel - use tiled version for better performance *)
    let dtype_suffix = Internal.dtype_to_metal_type a.dtype in
    let kernel_name = Printf.sprintf "matmul_tiled_%s" dtype_suffix in
    let func = Kernels.get_special_kernel ctx kernel_name in
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

        (* Set shape arrays *)
        let ndim_out = Array.length out_shape in
        let out_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim_out) in
        let a_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim_a) in
        let b_shape_arr = Ctypes.(allocate_n uint32_t ~count:ndim_b) in

        for i = 0 to ndim_out - 1 do
          Ctypes.(out_shape_arr +@ i <-@ Unsigned.UInt32.of_int out_shape.(i))
        done;
        for i = 0 to ndim_a - 1 do
          Ctypes.(a_shape_arr +@ i <-@ Unsigned.UInt32.of_int shape_a.(i))
        done;
        for i = 0 to ndim_b - 1 do
          Ctypes.(b_shape_arr +@ i <-@ Unsigned.UInt32.of_int shape_b.(i))
        done;

        ComputeCommandEncoder.set_bytes encoder
          ~bytes:Ctypes.(to_voidp out_shape_arr)
          ~length:(ndim_out * 4) ~index:3;
        ComputeCommandEncoder.set_bytes encoder
          ~bytes:Ctypes.(to_voidp a_shape_arr)
          ~length:(ndim_a * 4) ~index:4;
        ComputeCommandEncoder.set_bytes encoder
          ~bytes:Ctypes.(to_voidp b_shape_arr)
          ~length:(ndim_b * 4) ~index:5;

        (* Set dimensions *)
        let ndim_out_val =
          Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim_out))
        in
        let ndim_a_val =
          Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim_a))
        in
        let ndim_b_val =
          Ctypes.(allocate uint32_t (Unsigned.UInt32.of_int ndim_b))
        in

        ComputeCommandEncoder.set_bytes encoder
          ~bytes:Ctypes.(to_voidp ndim_out_val)
          ~length:4 ~index:6;
        ComputeCommandEncoder.set_bytes encoder
          ~bytes:Ctypes.(to_voidp ndim_a_val)
          ~length:4 ~index:7;
        ComputeCommandEncoder.set_bytes encoder
          ~bytes:Ctypes.(to_voidp ndim_b_val)
          ~length:4 ~index:8;

        (* Calculate thread configuration for tiled kernel *)
        let batch_size = ref 1 in
        for i = 0 to ndim_out - 3 do
          batch_size := !batch_size * out_shape.(i)
        done;

        (* For tiled kernel, we use TILE_SIZE x TILE_SIZE thread groups *)
        let tile_size = 16 in
        let group_size =
          { Metal.Size.width = tile_size; height = tile_size; depth = 1 }
        in

        ComputeCommandEncoder.dispatch_threadgroups encoder
          ~threadgroups_per_grid:
            {
              Metal.Size.width = (n + tile_size - 1) / tile_size;
              height = (m + tile_size - 1) / tile_size;
              depth = !batch_size;
            }
          ~threads_per_threadgroup:group_size;
        ComputeCommandEncoder.end_encoding encoder);

    out
