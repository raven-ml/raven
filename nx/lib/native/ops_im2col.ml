open Bigarray
module Dtype = Nx_core.Dtype
module Shape = Nx_core.Shape
open Internal

(* im2col/col2im operations for efficient convolution *)

(* Helper to calculate output shape for unfold (im2col) operation *)
let calculate_unfold_output_shape input_shape ~kernel_size ~stride ~dilation
    ~padding =
  let num_spatial_dims = Array.length kernel_size in
  let batch_dims = Array.length input_shape - num_spatial_dims - 1 in

  (* Extract batch and channel dimensions *)
  let batch_shape = Array.sub input_shape 0 batch_dims in
  let channels = input_shape.(batch_dims) in

  (* Calculate spatial dimensions after padding *)
  let padded_spatial =
    Array.init num_spatial_dims (fun i ->
        let dim_idx = batch_dims + 1 + i in
        input_shape.(dim_idx) + fst padding.(i) + snd padding.(i))
  in

  (* Calculate output spatial dimensions *)
  let output_spatial =
    Array.init num_spatial_dims (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  (* Calculate total number of kernel elements *)
  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in

  (* Output shape: [batch..., channels * kernel_elements, num_blocks] *)
  let num_blocks = Array.fold_left ( * ) 1 output_spatial in
  Array.concat
    [ batch_shape; [| channels * kernel_elements |]; [| num_blocks |] ]

(* Specialized unfold (im2col) for float32 - worker function *)
let kernel_unfold_float32 (input : (float, float32_elt) t)
    (out : (float, float32_elt) t) ~kernel_size ~stride ~dilation ~padding
    start_block end_block =
  let input_buf, out_buf = (buffer input, buffer out) in
  let input_shape = shape input in

  (* Extract dimensions *)
  let num_spatial_dims = Array.length kernel_size in
  let batch_dims = Array.length input_shape - num_spatial_dims - 1 in
  let batch_size =
    Array.fold_left ( * ) 1 (Array.sub input_shape 0 batch_dims)
  in
  let channels = input_shape.(batch_dims) in
  let input_spatial = Array.sub input_shape (batch_dims + 1) num_spatial_dims in

  (* Calculate padded dimensions *)
  let padded_spatial =
    Array.init num_spatial_dims (fun i ->
        input_spatial.(i) + fst padding.(i) + snd padding.(i))
  in

  (* Calculate output spatial dimensions *)
  let output_spatial =
    Array.init num_spatial_dims (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let num_blocks = Array.fold_left ( * ) 1 output_spatial in

  (* Pre-allocate work arrays to avoid allocations in loops *)
  let out_spatial_coords = Array.make num_spatial_dims 0 in
  let kernel_coords = Array.make num_spatial_dims 0 in
  let input_coords = Array.make num_spatial_dims 0 in

  (* Helper to convert flat index to multi-dimensional index (in-place) *)
  let unravel_spatial_inplace flat_idx dims result =
    let idx = ref flat_idx in
    for i = Array.length dims - 1 downto 0 do
      result.(i) <- !idx mod dims.(i);
      idx := !idx / dims.(i)
    done
  in

  (* Helper to convert multi-dimensional index to flat index *)
  let ravel_spatial indices dims =
    let idx = ref 0 in
    let stride = ref 1 in
    for i = Array.length dims - 1 downto 0 do
      idx := !idx + (indices.(i) * !stride);
      stride := !stride * dims.(i)
    done;
    !idx
  in

  (* Main im2col loop - process assigned block range *)
  for block_idx = start_block to end_block - 1 do
    for batch_idx = 0 to batch_size - 1 do
      (* Convert block index to output spatial coordinates *)
      unravel_spatial_inplace block_idx output_spatial out_spatial_coords;

      (* Iterate through all kernel positions *)
      for kernel_idx = 0 to kernel_elements - 1 do
        (* Convert kernel index to kernel coordinates *)
        unravel_spatial_inplace kernel_idx kernel_size kernel_coords;

        (* Calculate corresponding input coordinates *)
        let valid = ref true in
        for i = 0 to num_spatial_dims - 1 do
          let coord =
            Int64.(
              to_int
                (sub
                   (add
                      (mul (of_int out_spatial_coords.(i)) (of_int stride.(i)))
                      (mul (of_int kernel_coords.(i)) (of_int dilation.(i))))
                   (of_int (fst padding.(i)))))
          in
          if coord < 0 || coord >= input_spatial.(i) then valid := false;
          input_coords.(i) <- coord
        done;

        (* Iterate through all channels *)
        for c = 0 to channels - 1 do
          let out_idx =
            (batch_idx * (channels * kernel_elements * num_blocks))
            + (((c * kernel_elements) + kernel_idx) * num_blocks)
            + block_idx
          in

          if !valid then
            (* Compute input index *)
            let input_flat_spatial = ravel_spatial input_coords input_spatial in
            let input_idx =
              (batch_idx * (channels * Array.fold_left ( * ) 1 input_spatial))
              + (c * Array.fold_left ( * ) 1 input_spatial)
              + input_flat_spatial
            in
            Array1.unsafe_set out_buf out_idx
              (Array1.unsafe_get input_buf (offset input + input_idx))
          else
            (* Padding region - set to 0 *)
            Array1.unsafe_set out_buf out_idx 0.0
        done
      done
    done
  done

(* Specialized unfold (im2col) for uint8 - worker function *)
let kernel_unfold_uint8 (input : (int, int8_unsigned_elt) t)
    (out : (int, int8_unsigned_elt) t) ~kernel_size ~stride ~dilation ~padding
    start_block end_block =
  let input_buf, out_buf = (buffer input, buffer out) in
  let input_shape = shape input in

  (* Extract dimensions *)
  let num_spatial_dims = Array.length kernel_size in
  let batch_dims = Array.length input_shape - num_spatial_dims - 1 in
  let batch_size =
    Array.fold_left ( * ) 1 (Array.sub input_shape 0 batch_dims)
  in
  let channels = input_shape.(batch_dims) in
  let input_spatial = Array.sub input_shape (batch_dims + 1) num_spatial_dims in

  (* Calculate padded dimensions *)
  let padded_spatial =
    Array.init num_spatial_dims (fun i ->
        input_spatial.(i) + fst padding.(i) + snd padding.(i))
  in

  (* Calculate output spatial dimensions *)
  let output_spatial =
    Array.init num_spatial_dims (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let num_blocks = Array.fold_left ( * ) 1 output_spatial in

  (* Pre-allocate work arrays to avoid allocations in loops *)
  let out_spatial_coords = Array.make num_spatial_dims 0 in
  let kernel_coords = Array.make num_spatial_dims 0 in
  let input_coords = Array.make num_spatial_dims 0 in

  (* Helper to convert flat index to multi-dimensional index (in-place) *)
  let unravel_spatial_inplace flat_idx dims result =
    let idx = ref flat_idx in
    for i = Array.length dims - 1 downto 0 do
      result.(i) <- !idx mod dims.(i);
      idx := !idx / dims.(i)
    done
  in

  (* Helper to convert multi-dimensional index to flat index *)
  let ravel_spatial indices dims =
    let idx = ref 0 in
    let stride = ref 1 in
    for i = Array.length dims - 1 downto 0 do
      idx := !idx + (indices.(i) * !stride);
      stride := !stride * dims.(i)
    done;
    !idx
  in

  (* Main im2col loop - process assigned block range *)
  for block_idx = start_block to end_block - 1 do
    for batch_idx = 0 to batch_size - 1 do
      (* Convert block index to output spatial coordinates *)
      unravel_spatial_inplace block_idx output_spatial out_spatial_coords;

      (* Iterate through all kernel positions *)
      for kernel_idx = 0 to kernel_elements - 1 do
        (* Convert kernel index to kernel coordinates *)
        unravel_spatial_inplace kernel_idx kernel_size kernel_coords;

        (* Calculate corresponding input coordinates *)
        let valid = ref true in
        for i = 0 to num_spatial_dims - 1 do
          let coord =
            Int64.(
              to_int
                (sub
                   (add
                      (mul (of_int out_spatial_coords.(i)) (of_int stride.(i)))
                      (mul (of_int kernel_coords.(i)) (of_int dilation.(i))))
                   (of_int (fst padding.(i)))))
          in
          if coord < 0 || coord >= input_spatial.(i) then valid := false;
          input_coords.(i) <- coord
        done;

        (* Iterate through all channels *)
        for c = 0 to channels - 1 do
          let out_idx =
            (batch_idx * (channels * kernel_elements * num_blocks))
            + (((c * kernel_elements) + kernel_idx) * num_blocks)
            + block_idx
          in

          if !valid then
            (* Compute input index *)
            let input_flat_spatial = ravel_spatial input_coords input_spatial in
            let input_idx =
              (batch_idx * (channels * Array.fold_left ( * ) 1 input_spatial))
              + (c * Array.fold_left ( * ) 1 input_spatial)
              + input_flat_spatial
            in
            Array1.unsafe_set out_buf out_idx
              (Array1.unsafe_get input_buf (offset input + input_idx))
          else
            (* Padding region - set to 0 *)
            Array1.unsafe_set out_buf out_idx 0
        done
      done
    done
  done

(* Specialized unfold (im2col) for float64 - worker function *)
let kernel_unfold_float64 (input : (float, float64_elt) t)
    (out : (float, float64_elt) t) ~kernel_size ~stride ~dilation ~padding
    start_block end_block =
  let input_buf, out_buf = (buffer input, buffer out) in
  let input_shape = shape input in

  (* Extract dimensions *)
  let num_spatial_dims = Array.length kernel_size in
  let batch_dims = Array.length input_shape - num_spatial_dims - 1 in
  let batch_size =
    Array.fold_left ( * ) 1 (Array.sub input_shape 0 batch_dims)
  in
  let channels = input_shape.(batch_dims) in
  let input_spatial = Array.sub input_shape (batch_dims + 1) num_spatial_dims in

  (* Calculate padded dimensions *)
  let padded_spatial =
    Array.init num_spatial_dims (fun i ->
        input_spatial.(i) + fst padding.(i) + snd padding.(i))
  in

  (* Calculate output spatial dimensions *)
  let output_spatial =
    Array.init num_spatial_dims (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let num_blocks = Array.fold_left ( * ) 1 output_spatial in

  (* Pre-allocate work arrays to avoid allocations in loops *)
  let out_spatial_coords = Array.make num_spatial_dims 0 in
  let kernel_coords = Array.make num_spatial_dims 0 in
  let input_coords = Array.make num_spatial_dims 0 in

  (* Helper to convert flat index to multi-dimensional index (in-place) *)
  let unravel_spatial_inplace flat_idx dims result =
    let idx = ref flat_idx in
    for i = Array.length dims - 1 downto 0 do
      result.(i) <- !idx mod dims.(i);
      idx := !idx / dims.(i)
    done
  in

  (* Helper to convert multi-dimensional index to flat index *)
  let ravel_spatial indices dims =
    let idx = ref 0 in
    let stride = ref 1 in
    for i = Array.length dims - 1 downto 0 do
      idx := !idx + (indices.(i) * !stride);
      stride := !stride * dims.(i)
    done;
    !idx
  in

  (* Main im2col loop - process assigned block range *)
  for block_idx = start_block to end_block - 1 do
    for batch_idx = 0 to batch_size - 1 do
      (* Convert block index to output spatial coordinates *)
      unravel_spatial_inplace block_idx output_spatial out_spatial_coords;

      (* Iterate through all kernel positions *)
      for kernel_idx = 0 to kernel_elements - 1 do
        (* Convert kernel index to kernel coordinates *)
        unravel_spatial_inplace kernel_idx kernel_size kernel_coords;

        (* Calculate corresponding input coordinates *)
        let valid = ref true in
        for i = 0 to num_spatial_dims - 1 do
          let coord =
            Int64.(
              to_int
                (sub
                   (add
                      (mul (of_int out_spatial_coords.(i)) (of_int stride.(i)))
                      (mul (of_int kernel_coords.(i)) (of_int dilation.(i))))
                   (of_int (fst padding.(i)))))
          in
          if coord < 0 || coord >= input_spatial.(i) then valid := false;
          input_coords.(i) <- coord
        done;

        (* Iterate through all channels *)
        for c = 0 to channels - 1 do
          let out_idx =
            (batch_idx * (channels * kernel_elements * num_blocks))
            + (((c * kernel_elements) + kernel_idx) * num_blocks)
            + block_idx
          in

          if !valid then
            (* Compute input index *)
            let input_flat_spatial = ravel_spatial input_coords input_spatial in
            let input_idx =
              (batch_idx * (channels * Array.fold_left ( * ) 1 input_spatial))
              + (c * Array.fold_left ( * ) 1 input_spatial)
              + input_flat_spatial
            in
            Array1.unsafe_set out_buf out_idx
              (Array1.unsafe_get input_buf (offset input + input_idx))
          else
            (* Padding region - set to 0 *)
            Array1.unsafe_set out_buf out_idx 0.0
        done
      done
    done
  done

(* Specialized unfold (im2col) for int32 - worker function *)
let kernel_unfold_int32 (input : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) ~kernel_size ~stride ~dilation ~padding
    start_block end_block =
  let input_buf, out_buf = (buffer input, buffer out) in
  let input_shape = shape input in

  (* Extract dimensions *)
  let num_spatial_dims = Array.length kernel_size in
  let batch_dims = Array.length input_shape - num_spatial_dims - 1 in
  let batch_size =
    Array.fold_left ( * ) 1 (Array.sub input_shape 0 batch_dims)
  in
  let channels = input_shape.(batch_dims) in
  let input_spatial = Array.sub input_shape (batch_dims + 1) num_spatial_dims in

  (* Calculate padded dimensions *)
  let padded_spatial =
    Array.init num_spatial_dims (fun i ->
        input_spatial.(i) + fst padding.(i) + snd padding.(i))
  in

  (* Calculate output spatial dimensions *)
  let output_spatial =
    Array.init num_spatial_dims (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let num_blocks = Array.fold_left ( * ) 1 output_spatial in

  (* Pre-allocate work arrays to avoid allocations in loops *)
  let out_spatial_coords = Array.make num_spatial_dims 0 in
  let kernel_coords = Array.make num_spatial_dims 0 in
  let input_coords = Array.make num_spatial_dims 0 in

  (* Helper to convert flat index to multi-dimensional index (in-place) *)
  let unravel_spatial_inplace flat_idx dims result =
    let idx = ref flat_idx in
    for i = Array.length dims - 1 downto 0 do
      result.(i) <- !idx mod dims.(i);
      idx := !idx / dims.(i)
    done
  in

  (* Helper to convert multi-dimensional index to flat index *)
  let ravel_spatial indices dims =
    let idx = ref 0 in
    let stride = ref 1 in
    for i = Array.length dims - 1 downto 0 do
      idx := !idx + (indices.(i) * !stride);
      stride := !stride * dims.(i)
    done;
    !idx
  in

  (* Main im2col loop - process assigned block range *)
  for block_idx = start_block to end_block - 1 do
    for batch_idx = 0 to batch_size - 1 do
      (* Convert block index to output spatial coordinates *)
      unravel_spatial_inplace block_idx output_spatial out_spatial_coords;

      (* Iterate through all kernel positions *)
      for kernel_idx = 0 to kernel_elements - 1 do
        (* Convert kernel index to kernel coordinates *)
        unravel_spatial_inplace kernel_idx kernel_size kernel_coords;

        (* Calculate corresponding input coordinates *)
        let valid = ref true in
        for i = 0 to num_spatial_dims - 1 do
          let coord =
            Int64.(
              to_int
                (sub
                   (add
                      (mul (of_int out_spatial_coords.(i)) (of_int stride.(i)))
                      (mul (of_int kernel_coords.(i)) (of_int dilation.(i))))
                   (of_int (fst padding.(i)))))
          in
          if coord < 0 || coord >= input_spatial.(i) then valid := false;
          input_coords.(i) <- coord
        done;

        (* Iterate through all channels *)
        for c = 0 to channels - 1 do
          let out_idx =
            (batch_idx * (channels * kernel_elements * num_blocks))
            + (((c * kernel_elements) + kernel_idx) * num_blocks)
            + block_idx
          in

          if !valid then
            (* Compute input index *)
            let input_flat_spatial = ravel_spatial input_coords input_spatial in
            let input_idx =
              (batch_idx * (channels * Array.fold_left ( * ) 1 input_spatial))
              + (c * Array.fold_left ( * ) 1 input_spatial)
              + input_flat_spatial
            in
            Array1.unsafe_set out_buf out_idx
              (Array1.unsafe_get input_buf (offset input + input_idx))
          else
            (* Padding region - set to 0 *)
            Array1.unsafe_set out_buf out_idx 0l
        done
      done
    done
  done

(* Specialized unfold (im2col) for int64 - worker function *)
let kernel_unfold_int64 (input : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) ~kernel_size ~stride ~dilation ~padding
    start_block end_block =
  let input_buf, out_buf = (buffer input, buffer out) in
  let input_shape = shape input in

  (* Extract dimensions *)
  let num_spatial_dims = Array.length kernel_size in
  let batch_dims = Array.length input_shape - num_spatial_dims - 1 in
  let batch_size =
    Array.fold_left ( * ) 1 (Array.sub input_shape 0 batch_dims)
  in
  let channels = input_shape.(batch_dims) in
  let input_spatial = Array.sub input_shape (batch_dims + 1) num_spatial_dims in

  (* Calculate padded dimensions *)
  let padded_spatial =
    Array.init num_spatial_dims (fun i ->
        input_spatial.(i) + fst padding.(i) + snd padding.(i))
  in

  (* Calculate output spatial dimensions *)
  let output_spatial =
    Array.init num_spatial_dims (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let num_blocks = Array.fold_left ( * ) 1 output_spatial in

  (* Pre-allocate work arrays to avoid allocations in loops *)
  let out_spatial_coords = Array.make num_spatial_dims 0 in
  let kernel_coords = Array.make num_spatial_dims 0 in
  let input_coords = Array.make num_spatial_dims 0 in

  (* Helper to convert flat index to multi-dimensional index (in-place) *)
  let unravel_spatial_inplace flat_idx dims result =
    let idx = ref flat_idx in
    for i = Array.length dims - 1 downto 0 do
      result.(i) <- !idx mod dims.(i);
      idx := !idx / dims.(i)
    done
  in

  (* Helper to convert multi-dimensional index to flat index *)
  let ravel_spatial indices dims =
    let idx = ref 0 in
    let stride = ref 1 in
    for i = Array.length dims - 1 downto 0 do
      idx := !idx + (indices.(i) * !stride);
      stride := !stride * dims.(i)
    done;
    !idx
  in

  (* Main im2col loop - process assigned block range *)
  for block_idx = start_block to end_block - 1 do
    for batch_idx = 0 to batch_size - 1 do
      (* Convert block index to output spatial coordinates *)
      unravel_spatial_inplace block_idx output_spatial out_spatial_coords;

      (* Iterate through all kernel positions *)
      for kernel_idx = 0 to kernel_elements - 1 do
        (* Convert kernel index to kernel coordinates *)
        unravel_spatial_inplace kernel_idx kernel_size kernel_coords;

        (* Calculate corresponding input coordinates *)
        let valid = ref true in
        for i = 0 to num_spatial_dims - 1 do
          let coord =
            Int64.(
              to_int
                (sub
                   (add
                      (mul (of_int out_spatial_coords.(i)) (of_int stride.(i)))
                      (mul (of_int kernel_coords.(i)) (of_int dilation.(i))))
                   (of_int (fst padding.(i)))))
          in
          if coord < 0 || coord >= input_spatial.(i) then valid := false;
          input_coords.(i) <- coord
        done;

        (* Iterate through all channels *)
        for c = 0 to channels - 1 do
          let out_idx =
            (batch_idx * (channels * kernel_elements * num_blocks))
            + (((c * kernel_elements) + kernel_idx) * num_blocks)
            + block_idx
          in

          if !valid then
            (* Compute input index *)
            let input_flat_spatial = ravel_spatial input_coords input_spatial in
            let input_idx =
              (batch_idx * (channels * Array.fold_left ( * ) 1 input_spatial))
              + (c * Array.fold_left ( * ) 1 input_spatial)
              + input_flat_spatial
            in
            Array1.unsafe_set out_buf out_idx
              (Array1.unsafe_get input_buf (offset input + input_idx))
          else
            (* Padding region - set to 0 *)
            Array1.unsafe_set out_buf out_idx 0L
        done
      done
    done
  done

(* Specialized fold (col2im) for float32 - worker function *)
let kernel_fold_float32 (input : (float, float32_elt) t)
    (out : (float, float32_elt) t) ~output_size ~kernel_size ~stride ~dilation
    ~padding start_block end_block =
  let input_buf, out_buf = (buffer input, buffer out) in
  let out_shape = shape out in

  (* Extract dimensions *)
  let num_spatial_dims = Array.length kernel_size in
  let batch_dims = Array.length out_shape - num_spatial_dims - 1 in
  let batch_size = Array.fold_left ( * ) 1 (Array.sub out_shape 0 batch_dims) in
  let channels = out_shape.(batch_dims) in

  (* Output spatial dimensions from output_size *)
  let output_spatial = output_size in

  (* Calculate padded dimensions *)
  let padded_spatial =
    Array.init num_spatial_dims (fun i ->
        output_spatial.(i) + fst padding.(i) + snd padding.(i))
  in

  (* Calculate unfolded spatial dimensions *)
  let unfolded_spatial =
    Array.init num_spatial_dims (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let num_blocks = Array.fold_left ( * ) 1 unfolded_spatial in

  (* Note: Output buffer must be initialized to zero before calling this
     function *)

  (* Pre-allocate work arrays to avoid allocations in loops *)
  let unfolded_coords = Array.make num_spatial_dims 0 in
  let kernel_coords = Array.make num_spatial_dims 0 in
  let output_coords = Array.make num_spatial_dims 0 in

  (* Helper to convert flat index to multi-dimensional index (in-place) *)
  let unravel_spatial_inplace flat_idx dims result =
    let idx = ref flat_idx in
    for i = Array.length dims - 1 downto 0 do
      result.(i) <- !idx mod dims.(i);
      idx := !idx / dims.(i)
    done
  in

  (* Helper to convert multi-dimensional index to flat index *)
  let ravel_spatial indices dims =
    let idx = ref 0 in
    let stride = ref 1 in
    for i = Array.length dims - 1 downto 0 do
      idx := !idx + (indices.(i) * !stride);
      stride := !stride * dims.(i)
    done;
    !idx
  in

  (* Main col2im loop - process assigned block range *)
  for block_idx = start_block to end_block - 1 do
    for batch_idx = 0 to batch_size - 1 do
      (* Convert block index to unfolded spatial coordinates *)
      unravel_spatial_inplace block_idx unfolded_spatial unfolded_coords;

      (* Iterate through all kernel positions *)
      for kernel_idx = 0 to kernel_elements - 1 do
        (* Convert kernel index to kernel coordinates *)
        unravel_spatial_inplace kernel_idx kernel_size kernel_coords;

        (* Calculate corresponding output coordinates *)
        let valid = ref true in
        for i = 0 to num_spatial_dims - 1 do
          let coord =
            Int64.(
              to_int
                (sub
                   (add
                      (mul (of_int unfolded_coords.(i)) (of_int stride.(i)))
                      (mul (of_int kernel_coords.(i)) (of_int dilation.(i))))
                   (of_int (fst padding.(i)))))
          in
          if coord < 0 || coord >= output_spatial.(i) then valid := false;
          output_coords.(i) <- coord
        done;

        (* Iterate through all channels *)
        for c = 0 to channels - 1 do
          let input_idx =
            (batch_idx * (channels * kernel_elements * num_blocks))
            + (((c * kernel_elements) + kernel_idx) * num_blocks)
            + block_idx
          in

          if !valid then
            (* Compute output index *)
            let output_flat_spatial =
              ravel_spatial output_coords output_spatial
            in
            let output_idx =
              (batch_idx * (channels * Array.fold_left ( * ) 1 output_spatial))
              + (c * Array.fold_left ( * ) 1 output_spatial)
              + output_flat_spatial
            in
            (* Accumulate value *)
            let current_val = Array1.unsafe_get out_buf output_idx in
            let input_val =
              Array1.unsafe_get input_buf (offset input + input_idx)
            in
            Array1.unsafe_set out_buf output_idx (current_val +. input_val)
        done
      done
    done
  done

(* Specialized fold (col2im) for float64 - worker function *)
let kernel_fold_float64 (input : (float, float64_elt) t)
    (out : (float, float64_elt) t) ~output_size ~kernel_size ~stride ~dilation
    ~padding start_block end_block =
  let input_buf, out_buf = (buffer input, buffer out) in
  let out_shape = shape out in

  (* Extract dimensions *)
  let num_spatial_dims = Array.length kernel_size in
  let batch_dims = Array.length out_shape - num_spatial_dims - 1 in
  let batch_size = Array.fold_left ( * ) 1 (Array.sub out_shape 0 batch_dims) in
  let channels = out_shape.(batch_dims) in

  (* Output spatial dimensions from output_size *)
  let output_spatial = output_size in

  (* Calculate padded dimensions *)
  let padded_spatial =
    Array.init num_spatial_dims (fun i ->
        output_spatial.(i) + fst padding.(i) + snd padding.(i))
  in

  (* Calculate unfolded spatial dimensions *)
  let unfolded_spatial =
    Array.init num_spatial_dims (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let num_blocks = Array.fold_left ( * ) 1 unfolded_spatial in

  (* Pre-allocate work arrays *)
  let block_coords = Array.make num_spatial_dims 0 in
  let kernel_coords = Array.make num_spatial_dims 0 in
  let out_coords = Array.make num_spatial_dims 0 in

  (* Helper to convert flat index to multi-dimensional index (in-place) *)
  let unravel_spatial_inplace flat_idx dims result =
    let idx = ref flat_idx in
    for i = Array.length dims - 1 downto 0 do
      result.(i) <- !idx mod dims.(i);
      idx := !idx / dims.(i)
    done
  in

  (* Helper to convert multi-dimensional index to flat index *)
  let ravel_spatial indices dims =
    let idx = ref 0 in
    let stride = ref 1 in
    for i = Array.length dims - 1 downto 0 do
      idx := !idx + (indices.(i) * !stride);
      stride := !stride * dims.(i)
    done;
    !idx
  in

  (* Main col2im loop - process assigned block range *)
  for block_idx = start_block to end_block - 1 do
    for batch_idx = 0 to batch_size - 1 do
      (* Convert block index to spatial coordinates *)
      unravel_spatial_inplace block_idx unfolded_spatial block_coords;

      (* Iterate through all kernel positions *)
      for kernel_idx = 0 to kernel_elements - 1 do
        (* Convert kernel index to kernel coordinates *)
        unravel_spatial_inplace kernel_idx kernel_size kernel_coords;

        (* Calculate corresponding output coordinates *)
        let valid = ref true in
        for i = 0 to num_spatial_dims - 1 do
          let coord =
            block_coords.(i) * stride.(i)
            + kernel_coords.(i) * dilation.(i)
            - fst padding.(i)
          in
          if coord < 0 || coord >= output_spatial.(i) then valid := false;
          out_coords.(i) <- coord
        done;

        if !valid then
          (* Iterate through all channels *)
          for c = 0 to channels - 1 do
            let input_idx =
              (batch_idx * (channels * kernel_elements * num_blocks))
              + (((c * kernel_elements) + kernel_idx) * num_blocks)
              + block_idx
            in

            (* Compute output index *)
            let output_flat_spatial = ravel_spatial out_coords output_spatial in
            let output_idx =
              (batch_idx * (channels * Array.fold_left ( * ) 1 output_spatial))
              + (c * Array.fold_left ( * ) 1 output_spatial)
              + output_flat_spatial
            in

            (* Accumulate value *)
            let current_val = Array1.unsafe_get out_buf output_idx in
            let input_val =
              Array1.unsafe_get input_buf (offset input + input_idx)
            in
            Array1.unsafe_set out_buf output_idx (current_val +. input_val)
        done
      done
    done
  done

(* Specialized fold (col2im) for uint8 - worker function *)
let kernel_fold_uint8 (input : (int, int8_unsigned_elt) t)
    (out : (int, int8_unsigned_elt) t) ~output_size ~kernel_size ~stride ~dilation
    ~padding start_block end_block =
  let input_buf, out_buf = (buffer input, buffer out) in
  let out_shape = shape out in

  (* Extract dimensions *)
  let num_spatial_dims = Array.length kernel_size in
  let batch_dims = Array.length out_shape - num_spatial_dims - 1 in
  let batch_size = Array.fold_left ( * ) 1 (Array.sub out_shape 0 batch_dims) in
  let channels = out_shape.(batch_dims) in

  (* Output spatial dimensions from output_size *)
  let output_spatial = output_size in

  (* Calculate padded dimensions *)
  let padded_spatial =
    Array.init num_spatial_dims (fun i ->
        output_spatial.(i) + fst padding.(i) + snd padding.(i))
  in

  (* Calculate unfolded spatial dimensions *)
  let unfolded_spatial =
    Array.init num_spatial_dims (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let num_blocks = Array.fold_left ( * ) 1 unfolded_spatial in

  (* Pre-allocate work arrays *)
  let block_coords = Array.make num_spatial_dims 0 in
  let kernel_coords = Array.make num_spatial_dims 0 in
  let out_coords = Array.make num_spatial_dims 0 in

  (* Helper to convert flat index to multi-dimensional index (in-place) *)
  let unravel_spatial_inplace flat_idx dims result =
    let idx = ref flat_idx in
    for i = Array.length dims - 1 downto 0 do
      result.(i) <- !idx mod dims.(i);
      idx := !idx / dims.(i)
    done
  in

  (* Helper to convert multi-dimensional index to flat index *)
  let ravel_spatial indices dims =
    let idx = ref 0 in
    let stride = ref 1 in
    for i = Array.length dims - 1 downto 0 do
      idx := !idx + (indices.(i) * !stride);
      stride := !stride * dims.(i)
    done;
    !idx
  in

  (* Main col2im loop - process assigned block range *)
  for block_idx = start_block to end_block - 1 do
    for batch_idx = 0 to batch_size - 1 do
      (* Convert block index to spatial coordinates *)
      unravel_spatial_inplace block_idx unfolded_spatial block_coords;

      (* Iterate through all kernel positions *)
      for kernel_idx = 0 to kernel_elements - 1 do
        (* Convert kernel index to kernel coordinates *)
        unravel_spatial_inplace kernel_idx kernel_size kernel_coords;

        (* Calculate corresponding output coordinates *)
        let valid = ref true in
        for i = 0 to num_spatial_dims - 1 do
          let coord =
            block_coords.(i) * stride.(i)
            + kernel_coords.(i) * dilation.(i)
            - fst padding.(i)
          in
          if coord < 0 || coord >= output_spatial.(i) then valid := false;
          out_coords.(i) <- coord
        done;

        if !valid then
          (* Iterate through all channels *)
          for c = 0 to channels - 1 do
            let input_idx =
              (batch_idx * (channels * kernel_elements * num_blocks))
              + (((c * kernel_elements) + kernel_idx) * num_blocks)
              + block_idx
            in

            (* Compute output index *)
            let output_flat_spatial = ravel_spatial out_coords output_spatial in
            let output_idx =
              (batch_idx * (channels * Array.fold_left ( * ) 1 output_spatial))
              + (c * Array.fold_left ( * ) 1 output_spatial)
              + output_flat_spatial
            in

            (* Accumulate value - saturating add for uint8 *)
            let current_val = Array1.unsafe_get out_buf output_idx in
            let input_val =
              Array1.unsafe_get input_buf (offset input + input_idx)
            in
            let sum = current_val + input_val in
            let result = if sum > 255 then 255 else sum in
            Array1.unsafe_set out_buf output_idx result
        done
      done
    done
  done

(* Specialized fold (col2im) for int32 - worker function *)
let kernel_fold_int32 (input : (int32, int32_elt) t)
    (out : (int32, int32_elt) t) ~output_size ~kernel_size ~stride ~dilation
    ~padding start_block end_block =
  let input_buf, out_buf = (buffer input, buffer out) in
  let out_shape = shape out in

  (* Extract dimensions *)
  let num_spatial_dims = Array.length kernel_size in
  let batch_dims = Array.length out_shape - num_spatial_dims - 1 in
  let batch_size = Array.fold_left ( * ) 1 (Array.sub out_shape 0 batch_dims) in
  let channels = out_shape.(batch_dims) in

  (* Output spatial dimensions from output_size *)
  let output_spatial = output_size in

  (* Calculate padded dimensions *)
  let padded_spatial =
    Array.init num_spatial_dims (fun i ->
        output_spatial.(i) + fst padding.(i) + snd padding.(i))
  in

  (* Calculate unfolded spatial dimensions *)
  let unfolded_spatial =
    Array.init num_spatial_dims (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let num_blocks = Array.fold_left ( * ) 1 unfolded_spatial in

  (* Pre-allocate work arrays *)
  let block_coords = Array.make num_spatial_dims 0 in
  let kernel_coords = Array.make num_spatial_dims 0 in
  let out_coords = Array.make num_spatial_dims 0 in

  (* Helper to convert flat index to multi-dimensional index (in-place) *)
  let unravel_spatial_inplace flat_idx dims result =
    let idx = ref flat_idx in
    for i = Array.length dims - 1 downto 0 do
      result.(i) <- !idx mod dims.(i);
      idx := !idx / dims.(i)
    done
  in

  (* Helper to convert multi-dimensional index to flat index *)
  let ravel_spatial indices dims =
    let idx = ref 0 in
    let stride = ref 1 in
    for i = Array.length dims - 1 downto 0 do
      idx := !idx + (indices.(i) * !stride);
      stride := !stride * dims.(i)
    done;
    !idx
  in

  (* Main col2im loop - process assigned block range *)
  for block_idx = start_block to end_block - 1 do
    for batch_idx = 0 to batch_size - 1 do
      (* Convert block index to spatial coordinates *)
      unravel_spatial_inplace block_idx unfolded_spatial block_coords;

      (* Iterate through all kernel positions *)
      for kernel_idx = 0 to kernel_elements - 1 do
        (* Convert kernel index to kernel coordinates *)
        unravel_spatial_inplace kernel_idx kernel_size kernel_coords;

        (* Calculate corresponding output coordinates *)
        let valid = ref true in
        for i = 0 to num_spatial_dims - 1 do
          let coord =
            block_coords.(i) * stride.(i)
            + kernel_coords.(i) * dilation.(i)
            - fst padding.(i)
          in
          if coord < 0 || coord >= output_spatial.(i) then valid := false;
          out_coords.(i) <- coord
        done;

        if !valid then
          (* Iterate through all channels *)
          for c = 0 to channels - 1 do
            let input_idx =
              (batch_idx * (channels * kernel_elements * num_blocks))
              + (((c * kernel_elements) + kernel_idx) * num_blocks)
              + block_idx
            in

            (* Compute output index *)
            let output_flat_spatial = ravel_spatial out_coords output_spatial in
            let output_idx =
              (batch_idx * (channels * Array.fold_left ( * ) 1 output_spatial))
              + (c * Array.fold_left ( * ) 1 output_spatial)
              + output_flat_spatial
            in

            (* Accumulate value *)
            let current_val = Array1.unsafe_get out_buf output_idx in
            let input_val =
              Array1.unsafe_get input_buf (offset input + input_idx)
            in
            Array1.unsafe_set out_buf output_idx (Int32.add current_val input_val)
        done
      done
    done
  done

(* Specialized fold (col2im) for int64 - worker function *)
let kernel_fold_int64 (input : (int64, int64_elt) t)
    (out : (int64, int64_elt) t) ~output_size ~kernel_size ~stride ~dilation
    ~padding start_block end_block =
  let input_buf, out_buf = (buffer input, buffer out) in
  let out_shape = shape out in

  (* Extract dimensions *)
  let num_spatial_dims = Array.length kernel_size in
  let batch_dims = Array.length out_shape - num_spatial_dims - 1 in
  let batch_size = Array.fold_left ( * ) 1 (Array.sub out_shape 0 batch_dims) in
  let channels = out_shape.(batch_dims) in

  (* Output spatial dimensions from output_size *)
  let output_spatial = output_size in

  (* Calculate padded dimensions *)
  let padded_spatial =
    Array.init num_spatial_dims (fun i ->
        output_spatial.(i) + fst padding.(i) + snd padding.(i))
  in

  (* Calculate unfolded spatial dimensions *)
  let unfolded_spatial =
    Array.init num_spatial_dims (fun i ->
        let effective_kernel = ((kernel_size.(i) - 1) * dilation.(i)) + 1 in
        ((padded_spatial.(i) - effective_kernel) / stride.(i)) + 1)
  in

  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let num_blocks = Array.fold_left ( * ) 1 unfolded_spatial in

  (* Pre-allocate work arrays *)
  let block_coords = Array.make num_spatial_dims 0 in
  let kernel_coords = Array.make num_spatial_dims 0 in
  let out_coords = Array.make num_spatial_dims 0 in

  (* Helper to convert flat index to multi-dimensional index (in-place) *)
  let unravel_spatial_inplace flat_idx dims result =
    let idx = ref flat_idx in
    for i = Array.length dims - 1 downto 0 do
      result.(i) <- !idx mod dims.(i);
      idx := !idx / dims.(i)
    done
  in

  (* Helper to convert multi-dimensional index to flat index *)
  let ravel_spatial indices dims =
    let idx = ref 0 in
    let stride = ref 1 in
    for i = Array.length dims - 1 downto 0 do
      idx := !idx + (indices.(i) * !stride);
      stride := !stride * dims.(i)
    done;
    !idx
  in

  (* Main col2im loop - process assigned block range *)
  for block_idx = start_block to end_block - 1 do
    for batch_idx = 0 to batch_size - 1 do
      (* Convert block index to spatial coordinates *)
      unravel_spatial_inplace block_idx unfolded_spatial block_coords;

      (* Iterate through all kernel positions *)
      for kernel_idx = 0 to kernel_elements - 1 do
        (* Convert kernel index to kernel coordinates *)
        unravel_spatial_inplace kernel_idx kernel_size kernel_coords;

        (* Calculate corresponding output coordinates *)
        let valid = ref true in
        for i = 0 to num_spatial_dims - 1 do
          let coord =
            block_coords.(i) * stride.(i)
            + kernel_coords.(i) * dilation.(i)
            - fst padding.(i)
          in
          if coord < 0 || coord >= output_spatial.(i) then valid := false;
          out_coords.(i) <- coord
        done;

        if !valid then
          (* Iterate through all channels *)
          for c = 0 to channels - 1 do
            let input_idx =
              (batch_idx * (channels * kernel_elements * num_blocks))
              + (((c * kernel_elements) + kernel_idx) * num_blocks)
              + block_idx
            in

            (* Compute output index *)
            let output_flat_spatial = ravel_spatial out_coords output_spatial in
            let output_idx =
              (batch_idx * (channels * Array.fold_left ( * ) 1 output_spatial))
              + (c * Array.fold_left ( * ) 1 output_spatial)
              + output_flat_spatial
            in

            (* Accumulate value *)
            let current_val = Array1.unsafe_get out_buf output_idx in
            let input_val =
              Array1.unsafe_get input_buf (offset input + input_idx)
            in
            Array1.unsafe_set out_buf output_idx (Int64.add current_val input_val)
        done
      done
    done
  done

(* Generic kernel dispatch for unfold *)
let kernel_unfold (type a b) (context : context) (input : (a, b) t)
    (out : (a, b) t) ~kernel_size ~stride ~dilation ~padding =
  (* Calculate number of output blocks for parallelization *)
  let out_shape = shape out in
  let num_blocks = out_shape.(Array.length out_shape - 1) in
  (* last dimension is num_blocks *)

  if num_blocks > 1 then
    (* Parallelize across output blocks *)
    (* Note: parallel_for's end_idx is exclusive in the worker function *)
    Parallel.parallel_for context.pool 0 (num_blocks - 1)
      (fun start_block end_block ->
        match Array1.kind (buffer input) with
        | Float32 ->
            kernel_unfold_float32 input out ~kernel_size ~stride ~dilation
              ~padding start_block end_block
        | Float64 ->
            kernel_unfold_float64 input out ~kernel_size ~stride ~dilation
              ~padding start_block end_block
        | Int8_unsigned ->
            kernel_unfold_uint8 input out ~kernel_size ~stride ~dilation
              ~padding start_block end_block
        | Int32 ->
            kernel_unfold_int32 input out ~kernel_size ~stride ~dilation
              ~padding start_block end_block
        | Int64 ->
            kernel_unfold_int64 input out ~kernel_size ~stride ~dilation
              ~padding start_block end_block
        | _ -> failwith "Unfold not yet implemented for this dtype")
  else
    (* Single block - run directly *)
    match Array1.kind (buffer input) with
    | Float32 ->
        kernel_unfold_float32 input out ~kernel_size ~stride ~dilation ~padding
          0 num_blocks
    | Float64 ->
        kernel_unfold_float64 input out ~kernel_size ~stride ~dilation ~padding
          0 num_blocks
    | Int8_unsigned ->
        kernel_unfold_uint8 input out ~kernel_size ~stride ~dilation ~padding
          0 num_blocks
    | Int32 ->
        kernel_unfold_int32 input out ~kernel_size ~stride ~dilation ~padding
          0 num_blocks
    | Int64 ->
        kernel_unfold_int64 input out ~kernel_size ~stride ~dilation ~padding
          0 num_blocks
    | _ -> failwith "Unfold not yet implemented for this dtype"

(* Generic unfold (im2col) operation dispatcher *)
let unfold context input ~kernel_size ~stride ~dilation ~padding =
  let output_shape =
    calculate_unfold_output_shape (shape input) ~kernel_size ~stride ~dilation
      ~padding
  in
  let out = empty input.context input.dtype output_shape in
  kernel_unfold context input out ~kernel_size ~stride ~dilation ~padding;
  out

(* Generic kernel dispatch for fold *)
let kernel_fold (type a b) (context : context) (input : (a, b) t)
    (out : (a, b) t) ~output_size ~kernel_size ~stride ~dilation ~padding =
  (* For fold, we need to be careful about parallelization. When stride=1 and
     kernel_size>1, multiple input blocks write to overlapping output locations.
     For now, let's try parallel blocks but be aware of potential race
     conditions. *)
  let input_shape = shape input in
  let num_blocks = input_shape.(Array.length input_shape - 1) in

  if num_blocks > 1 && false then
    (* Disabled for now due to potential race conditions *)
    Parallel.parallel_for context.pool 0 (num_blocks - 1)
      (fun start_block end_block ->
        match Array1.kind (buffer input) with
        | Float32 ->
            kernel_fold_float32 input out ~output_size ~kernel_size ~stride
              ~dilation ~padding start_block end_block
        | Float64 ->
            kernel_fold_float64 input out ~output_size ~kernel_size ~stride
              ~dilation ~padding start_block end_block
        | Int8_unsigned ->
            kernel_fold_uint8 input out ~output_size ~kernel_size ~stride
              ~dilation ~padding start_block end_block
        | Int32 ->
            kernel_fold_int32 input out ~output_size ~kernel_size ~stride
              ~dilation ~padding start_block end_block
        | Int64 ->
            kernel_fold_int64 input out ~output_size ~kernel_size ~stride
              ~dilation ~padding start_block end_block
        | _ -> failwith "Fold not yet implemented for this dtype")
  else
    (* Run sequentially *)
    match Array1.kind (buffer input) with
    | Float32 ->
        kernel_fold_float32 input out ~output_size ~kernel_size ~stride
          ~dilation ~padding 0 num_blocks
    | Float64 ->
        kernel_fold_float64 input out ~output_size ~kernel_size ~stride
          ~dilation ~padding 0 num_blocks
    | Int8_unsigned ->
        kernel_fold_uint8 input out ~output_size ~kernel_size ~stride
          ~dilation ~padding 0 num_blocks
    | Int32 ->
        kernel_fold_int32 input out ~output_size ~kernel_size ~stride
          ~dilation ~padding 0 num_blocks
    | Int64 ->
        kernel_fold_int64 input out ~output_size ~kernel_size ~stride
          ~dilation ~padding 0 num_blocks
    | _ -> failwith "Fold not yet implemented for this dtype"

(* Generic fold (col2im) operation dispatcher *)
let fold context input ~output_size ~kernel_size ~stride ~dilation ~padding =
  (* Calculate full output shape including batch and channel dimensions *)
  let input_shape = shape input in
  let batch_dims = Array.length input_shape - 2 in
  (* -2 for channel*kernel and num_blocks dims *)
  let batch_shape = Array.sub input_shape 0 batch_dims in

  (* Extract channels from the folded dimension *)
  let kernel_elements = Array.fold_left ( * ) 1 kernel_size in
  let channels = input_shape.(batch_dims) / kernel_elements in

  let output_shape =
    Array.concat [ batch_shape; [| channels |]; output_size ]
  in

  let out = empty input.context input.dtype output_shape in

  (* Initialize output to zero *)
  let out_buf = buffer out in
  let out_size = Array.fold_left ( * ) 1 output_shape in
  (* Initialize output to zero using generic approach *)
  let zero_val = Dtype.zero input.dtype in
  for i = 0 to out_size - 1 do
    Array1.unsafe_set out_buf i zero_val
  done;

  kernel_fold context input out ~output_size ~kernel_size ~stride ~dilation
    ~padding;
  out
