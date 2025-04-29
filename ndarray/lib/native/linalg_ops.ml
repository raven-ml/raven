open Bigarray
open Ndarray_core
open Internal

let block_size = 32
let bm = block_size
let bn = block_size
let bk = block_size

let get_slice_info tensor batch_multi_idx =
  let ndim = Array.length (shape tensor) in
  if ndim < 2 then invalid_arg "get_slice_info: tensor must be at least 2D";
  let num_batch_dims = max 0 (ndim - 2) in
  let slice_offset = ref (offset tensor) in
  let source_batch_shape = Array.sub (shape tensor) 0 num_batch_dims in

  let source_batch_multi_idx =
    compute_broadcast_index batch_multi_idx source_batch_shape
  in
  for i = 0 to num_batch_dims - 1 do
    (* Check bounds for safety, though broadcasting should ensure
       compatibility *)
    if i < Array.length (strides tensor) then
      slice_offset :=
        !slice_offset + (source_batch_multi_idx.(i) * (strides tensor).(i))
  done;
  let stride0 = if ndim >= 2 then (strides tensor).(ndim - 2) else 0 in
  let stride1 = if ndim >= 1 then (strides tensor).(ndim - 1) else 0 in
  (!slice_offset, stride0, stride1)

let kernel_matmul_block_slice a_buf a_dtype a_base_offset a_stride_m a_stride_k
    b_buf b_base_offset b_stride_k b_stride_n c_buf c_base_offset c_stride_m
    c_stride_n m n k_dim block_row_start block_col_start =
  let i_end = min (block_row_start + bm) m in
  let j_end = min (block_col_start + bn) n in

  let l_block_start = ref 0 in
  while !l_block_start < k_dim do
    let current_l_start = !l_block_start in
    let l_end = min (current_l_start + bk) k_dim in

    for i = block_row_start to i_end - 1 do
      for j = block_col_start to j_end - 1 do
        let c_linear_idx =
          c_base_offset + (i * c_stride_m) + (j * c_stride_n)
        in
        let sum = ref (Array1.unsafe_get c_buf c_linear_idx) in

        for l = current_l_start to l_end - 1 do
          let a_val =
            Array1.unsafe_get a_buf
              (a_base_offset + (i * a_stride_m) + (l * a_stride_k))
          in
          let b_val =
            Array1.unsafe_get b_buf
              (b_base_offset + (l * b_stride_k) + (j * b_stride_n))
          in
          sum := fma_dtype a_dtype a_val b_val !sum
        done;

        Array1.unsafe_set c_buf c_linear_idx !sum
      done
    done;
    l_block_start := !l_block_start + bk
  done

let matmul context a_op b_op c_op =
  let a_op_shape = shape a_op in
  let b_op_shape = shape b_op in
  let a_op_ndim = Array.length a_op_shape in
  let b_op_ndim = Array.length b_op_shape in

  let m = a_op_shape.(a_op_ndim - 2) in
  let k_dim = a_op_shape.(a_op_ndim - 1) in
  let n = b_op_shape.(b_op_ndim - 1) in

  let out_shape = shape c_op in
  (* Calculate batch size based on the already computed output shape *)
  let num_batch_dims = max 0 (Array.length out_shape - 2) in
  let out_batch_shape_arr = Array.sub out_shape 0 num_batch_dims in
  let total_batch_ops =
    if num_batch_dims = 0 then 1
    else Array.fold_left ( * ) 1 out_batch_shape_arr
  in

  let a_op_buf = buffer a_op in
  let b_op_buf = buffer b_op in
  let c_buf = buffer c_op in
  let dt = dtype a_op in

  let process_batch_item batch_linear_idx =
    let batch_multi_idx =
      multi_index_from_linear batch_linear_idx out_batch_shape_arr
    in

    (* get_slice_info correctly handles broadcasting based on target index *)
    let a_offset, a_stride_m, a_stride_k =
      get_slice_info a_op batch_multi_idx
    in
    let b_offset, b_stride_k, b_stride_n =
      get_slice_info b_op batch_multi_idx
    in
    let c_offset, c_stride_m, c_stride_n =
      get_slice_info c_op batch_multi_idx
    in

    let num_blocks_m = (m + bm - 1) / bm in
    let num_blocks_n = (n + bn - 1) / bn in
    for ii = 0 to num_blocks_m - 1 do
      for jj = 0 to num_blocks_n - 1 do
        let block_row_start = ii * bm in
        let block_col_start = jj * bn in
        kernel_matmul_block_slice a_op_buf dt a_offset a_stride_m a_stride_k
          b_op_buf b_offset b_stride_k b_stride_n c_buf c_offset c_stride_m
          c_stride_n m n k_dim block_row_start block_col_start
      done
    done
  in

  let parallel_batch_threshold = 4 in
  (* Example threshold *)

  if total_batch_ops < parallel_batch_threshold || total_batch_ops = 0 then
    (* Handle case where size is 0 explicitly to avoid loop range issues *)
    for batch_linear_idx = 0 to total_batch_ops - 1 do
      process_batch_item batch_linear_idx
    done
  else
    Parallel.parallel_for context.pool 0 (total_batch_ops - 1)
      (fun batch_idx_start batch_idx_end ->
        for batch_linear_idx = batch_idx_start to batch_idx_end do
          process_batch_item batch_linear_idx
        done)
