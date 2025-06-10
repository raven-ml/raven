open Bigarray
module Dtype = Nx_core.Dtype
module Shape = Nx_core.Shape
open Internal

(* Matrix multiplication operations *)

(* Helper to broadcast shapes for matmul *)
let broadcast_matmul_shapes shape_a shape_b =
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
  Array.concat [ batch_shape; [| m; n |] ]

(* Specialized float32 matrix multiplication kernel *)
let kernel_matmul_float32 context (a : (float, float32_elt) t)
    (b : (float, float32_elt) t) (out : (float, float32_elt) t) =
  let a_buf, b_buf, out_buf = (buffer a, buffer b, buffer out) in
  let shape_a = shape a in
  let shape_b = shape b in
  let out_shape = shape out in

  let ndim_a = Array.length shape_a in
  let ndim_b = Array.length shape_b in
  let ndim_out = Array.length out_shape in

  (* Matrix dimensions *)
  let m = shape_a.(ndim_a - 2) in
  let k = shape_a.(ndim_a - 1) in
  let n = shape_b.(ndim_b - 1) in

  (* Calculate batch size *)
  let batch_size = ref 1 in
  for i = 0 to ndim_out - 3 do
    batch_size := !batch_size * out_shape.(i)
  done;

  (* Strides for the last two dimensions *)
  let stride_a_batch = m * k in
  let stride_b_batch = k * n in
  let stride_out_batch = m * n in

  (* Pre-allocate work array for batch coordinates *)
  let max_batch_dims = max (ndim_a - 2) (ndim_b - 2) in
  let batch_coord = Array.make max_batch_dims 0 in

  (* Helper to calculate batch index with broadcasting *)
  let get_batch_offset batch_idx shape ndim stride_batch =
    let idx = ref batch_idx in
    let batch_dims = ndim - 2 in

    (* Calculate multi-dimensional batch index *)
    for i = batch_dims - 1 downto 0 do
      let dim_size = out_shape.(i) in
      batch_coord.(i) <- !idx mod dim_size;
      idx := !idx / dim_size
    done;

    (* Map to actual tensor considering broadcasting *)
    idx := 0;
    let stride = ref 1 in
    for i = batch_dims - 1 downto 0 do
      let actual_dim = if i < ndim - 2 then shape.(i) else 1 in
      if actual_dim > 1 then idx := !idx + (batch_coord.(i) * !stride);
      stride := !stride * actual_dim
    done;

    !idx * stride_batch
  in

  (* Initialize output to zero *)
  let out_size = Array.fold_left ( * ) 1 out_shape in
  for i = 0 to out_size - 1 do
    Array1.unsafe_set out_buf i 0.0
  done;

  (* Perform batched matrix multiplication with parallelization *)
  if !batch_size > 1 then
    (* Parallelize across batches only to avoid any potential race conditions *)
    Parallel.parallel_for context.pool 0 (!batch_size - 1)
      (fun start_batch end_batch ->
        for batch = start_batch to end_batch do
          let a_batch_offset =
            get_batch_offset batch shape_a ndim_a stride_a_batch
          in
          let b_batch_offset =
            get_batch_offset batch shape_b ndim_b stride_b_batch
          in
          let out_batch_offset = batch * stride_out_batch in

          (* Matrix multiplication for this batch *)
          for i = 0 to m - 1 do
            for j = 0 to n - 1 do
              let sum = ref 0.0 in
              for l = 0 to k - 1 do
                let a_idx = offset a + a_batch_offset + (i * k) + l in
                let b_idx = offset b + b_batch_offset + (l * n) + j in
                sum :=
                  !sum
                  +. Array1.unsafe_get a_buf a_idx
                     *. Array1.unsafe_get b_buf b_idx
              done;
              let out_idx = out_batch_offset + (i * n) + j in
              Array1.unsafe_set out_buf out_idx !sum
            done
          done
        done)
  else
    (* Small matrices - run sequentially *)
    for batch = 0 to !batch_size - 1 do
      let a_batch_offset =
        get_batch_offset batch shape_a ndim_a stride_a_batch
      in
      let b_batch_offset =
        get_batch_offset batch shape_b ndim_b stride_b_batch
      in
      let out_batch_offset = batch * stride_out_batch in

      (* Matrix multiplication for this batch *)
      for i = 0 to m - 1 do
        for j = 0 to n - 1 do
          let sum = ref 0.0 in
          for l = 0 to k - 1 do
            let a_idx = offset a + a_batch_offset + (i * k) + l in
            let b_idx = offset b + b_batch_offset + (l * n) + j in
            sum :=
              !sum
              +. (Array1.unsafe_get a_buf a_idx *. Array1.unsafe_get b_buf b_idx)
          done;
          let out_idx = out_batch_offset + (i * n) + j in
          Array1.unsafe_set out_buf out_idx !sum
        done
      done
    done

(* Generic matmul dispatcher *)
let matmul (type a b) context (a : (a, b) t) (b : (a, b) t) : (a, b) t =
  let out_shape = broadcast_matmul_shapes (shape a) (shape b) in
  let out = empty context a.dtype out_shape in

  (match dtype a with
  | Dtype.Float32 -> kernel_matmul_float32 context a b out
  | Dtype.Float64 -> failwith "matmul: not yet implemented for float64"
  | Dtype.Int32 -> failwith "matmul: not yet implemented for int32"
  | Dtype.Int64 -> failwith "matmul: not yet implemented for int64"
  | Dtype.UInt8 -> failwith "matmul: not yet implemented for uint8"
  | Dtype.UInt16 -> failwith "matmul: not yet implemented for uint16"
  | Dtype.Int8 -> failwith "matmul: not yet implemented for int8"
  | Dtype.Int16 -> failwith "matmul: not yet implemented for int16"
  | Dtype.Float16 -> failwith "matmul: not yet implemented for float16"
  | Dtype.Complex32 -> failwith "matmul: not yet implemented for complex32"
  | Dtype.Complex64 -> failwith "matmul: not yet implemented for complex64"
  | Dtype.Int -> failwith "matmul: not yet implemented for int"
  | Dtype.NativeInt -> failwith "matmul: not yet implemented for nativeint");

  out
