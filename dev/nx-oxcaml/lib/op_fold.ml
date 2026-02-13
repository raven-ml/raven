(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let fold_float64 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Float_u.of_float 0.0 in
  let out_numel = Array.fold_left ( * ) (n * channels) output_size in
  for i = 0 to out_numel - 1 do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx blocks_shape block_coords;
      for c_idx = 0 to channels - 1 do
        for k_idx = 0 to kernel_elems - 1 do
          Shape.unravel_index_into k_idx kernel_size kernel_coords;
          let valid = ref true in
          for d = 0 to spatial_ndim - 1 do
            let pad_before, _ = padding.(d) in
            let pos =
              (block_coords.(d) * stride.(d))
              - pad_before
              + (kernel_coords.(d) * dilation.(d))
            in
            out_spatial.(d) <- pos;
            if pos < 0 || pos >= output_size.(d) then valid := false
          done;
          if !valid then (
            let src_ch = (c_idx * kernel_elems) + k_idx in
            let src_lin =
              in_offset
              + (n_idx * in_strides.(0))
              + (src_ch * in_strides.(1))
              + (b_idx * in_strides.(2))
            in
            let dst_lin =
              ref (out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1)))
            in
            for d = 0 to spatial_ndim - 1 do
              dst_lin := !dst_lin + (out_spatial.(d) * out_strides.(d + 2))
            done;
            let prev = Array.unsafe_get out_arr !dst_lin in
            let v = Array.unsafe_get in_arr src_lin in
            Array.unsafe_set out_arr !dst_lin (Float_u.add prev v))
        done
      done
    done
  done

let fold_float32 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Float32_u.of_int 0 in
  let out_numel = Array.fold_left ( * ) (n * channels) output_size in
  for i = 0 to out_numel - 1 do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx blocks_shape block_coords;
      for c_idx = 0 to channels - 1 do
        for k_idx = 0 to kernel_elems - 1 do
          Shape.unravel_index_into k_idx kernel_size kernel_coords;
          let valid = ref true in
          for d = 0 to spatial_ndim - 1 do
            let pad_before, _ = padding.(d) in
            let pos =
              (block_coords.(d) * stride.(d))
              - pad_before
              + (kernel_coords.(d) * dilation.(d))
            in
            out_spatial.(d) <- pos;
            if pos < 0 || pos >= output_size.(d) then valid := false
          done;
          if !valid then (
            let src_ch = (c_idx * kernel_elems) + k_idx in
            let src_lin =
              in_offset
              + (n_idx * in_strides.(0))
              + (src_ch * in_strides.(1))
              + (b_idx * in_strides.(2))
            in
            let dst_lin =
              ref (out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1)))
            in
            for d = 0 to spatial_ndim - 1 do
              dst_lin := !dst_lin + (out_spatial.(d) * out_strides.(d + 2))
            done;
            let prev = Array.unsafe_get out_arr !dst_lin in
            let v = Array.unsafe_get in_arr src_lin in
            Array.unsafe_set out_arr !dst_lin (Float32_u.add prev v))
        done
      done
    done
  done

let fold_int8 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Int8_u.of_int 0 in
  let out_numel = Array.fold_left ( * ) (n * channels) output_size in
  for i = 0 to out_numel - 1 do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx blocks_shape block_coords;
      for c_idx = 0 to channels - 1 do
        for k_idx = 0 to kernel_elems - 1 do
          Shape.unravel_index_into k_idx kernel_size kernel_coords;
          let valid = ref true in
          for d = 0 to spatial_ndim - 1 do
            let pad_before, _ = padding.(d) in
            let pos =
              (block_coords.(d) * stride.(d))
              - pad_before
              + (kernel_coords.(d) * dilation.(d))
            in
            out_spatial.(d) <- pos;
            if pos < 0 || pos >= output_size.(d) then valid := false
          done;
          if !valid then (
            let src_ch = (c_idx * kernel_elems) + k_idx in
            let src_lin =
              in_offset
              + (n_idx * in_strides.(0))
              + (src_ch * in_strides.(1))
              + (b_idx * in_strides.(2))
            in
            let dst_lin =
              ref (out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1)))
            in
            for d = 0 to spatial_ndim - 1 do
              dst_lin := !dst_lin + (out_spatial.(d) * out_strides.(d + 2))
            done;
            let prev = Array.unsafe_get out_arr !dst_lin in
            let v = Array.unsafe_get in_arr src_lin in
            Array.unsafe_set out_arr !dst_lin (Int8_u.add prev v))
        done
      done
    done
  done

let fold_int16 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Int16_u.of_int 0 in
  let out_numel = Array.fold_left ( * ) (n * channels) output_size in
  for i = 0 to out_numel - 1 do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx blocks_shape block_coords;
      for c_idx = 0 to channels - 1 do
        for k_idx = 0 to kernel_elems - 1 do
          Shape.unravel_index_into k_idx kernel_size kernel_coords;
          let valid = ref true in
          for d = 0 to spatial_ndim - 1 do
            let pad_before, _ = padding.(d) in
            let pos =
              (block_coords.(d) * stride.(d))
              - pad_before
              + (kernel_coords.(d) * dilation.(d))
            in
            out_spatial.(d) <- pos;
            if pos < 0 || pos >= output_size.(d) then valid := false
          done;
          if !valid then (
            let src_ch = (c_idx * kernel_elems) + k_idx in
            let src_lin =
              in_offset
              + (n_idx * in_strides.(0))
              + (src_ch * in_strides.(1))
              + (b_idx * in_strides.(2))
            in
            let dst_lin =
              ref (out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1)))
            in
            for d = 0 to spatial_ndim - 1 do
              dst_lin := !dst_lin + (out_spatial.(d) * out_strides.(d + 2))
            done;
            let prev = Array.unsafe_get out_arr !dst_lin in
            let v = Array.unsafe_get in_arr src_lin in
            Array.unsafe_set out_arr !dst_lin (Int16_u.add prev v))
        done
      done
    done
  done

let fold_int32 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Int32_u.of_int32 0l in
  let out_numel = Array.fold_left ( * ) (n * channels) output_size in
  for i = 0 to out_numel - 1 do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx blocks_shape block_coords;
      for c_idx = 0 to channels - 1 do
        for k_idx = 0 to kernel_elems - 1 do
          Shape.unravel_index_into k_idx kernel_size kernel_coords;
          let valid = ref true in
          for d = 0 to spatial_ndim - 1 do
            let pad_before, _ = padding.(d) in
            let pos =
              (block_coords.(d) * stride.(d))
              - pad_before
              + (kernel_coords.(d) * dilation.(d))
            in
            out_spatial.(d) <- pos;
            if pos < 0 || pos >= output_size.(d) then valid := false
          done;
          if !valid then (
            let src_ch = (c_idx * kernel_elems) + k_idx in
            let src_lin =
              in_offset
              + (n_idx * in_strides.(0))
              + (src_ch * in_strides.(1))
              + (b_idx * in_strides.(2))
            in
            let dst_lin =
              ref (out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1)))
            in
            for d = 0 to spatial_ndim - 1 do
              dst_lin := !dst_lin + (out_spatial.(d) * out_strides.(d + 2))
            done;
            let prev = Array.unsafe_get out_arr !dst_lin in
            let v = Array.unsafe_get in_arr src_lin in
            Array.unsafe_set out_arr !dst_lin (Int32_u.add prev v))
        done
      done
    done
  done

let fold_int64 in_arr out_arr ~n ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Int64_u.of_int64 0L in
  let out_numel = Array.fold_left ( * ) (n * channels) output_size in
  for i = 0 to out_numel - 1 do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx blocks_shape block_coords;
      for c_idx = 0 to channels - 1 do
        for k_idx = 0 to kernel_elems - 1 do
          Shape.unravel_index_into k_idx kernel_size kernel_coords;
          let valid = ref true in
          for d = 0 to spatial_ndim - 1 do
            let pad_before, _ = padding.(d) in
            let pos =
              (block_coords.(d) * stride.(d))
              - pad_before
              + (kernel_coords.(d) * dilation.(d))
            in
            out_spatial.(d) <- pos;
            if pos < 0 || pos >= output_size.(d) then valid := false
          done;
          if !valid then (
            let src_ch = (c_idx * kernel_elems) + k_idx in
            let src_lin =
              in_offset
              + (n_idx * in_strides.(0))
              + (src_ch * in_strides.(1))
              + (b_idx * in_strides.(2))
            in
            let dst_lin =
              ref (out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1)))
            in
            for d = 0 to spatial_ndim - 1 do
              dst_lin := !dst_lin + (out_spatial.(d) * out_strides.(d + 2))
            done;
            let prev = Array.unsafe_get out_arr !dst_lin in
            let v = Array.unsafe_get in_arr src_lin in
            Array.unsafe_set out_arr !dst_lin (Int64_u.add prev v))
        done
      done
    done
  done
