(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let unfold_float64 in_arr out_arr ~n ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Float_u.of_float 0.0 in
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx out_spatial block_coords;
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
            in_spatial.(d) <- pos;
            if pos < 0 || pos >= input_spatial.(d) then valid := false
          done;
          let v =
            if !valid then
              let src_lin =
                ref (in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1)))
              in
              for d = 0 to spatial_ndim - 1 do
                src_lin := !src_lin + (in_spatial.(d) * in_strides.(d + 2))
              done;
              Array.unsafe_get in_arr !src_lin
            else zero
          in
          let dst_ch = (c_idx * kernel_elems) + k_idx in
          let dst_lin =
            out_offset
            + (n_idx * out_strides.(0))
            + (dst_ch * out_strides.(1))
            + (b_idx * out_strides.(2))
          in
          Array.unsafe_set out_arr dst_lin v
        done
      done
    done
  done

let unfold_float32 in_arr out_arr ~n ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Float32_u.of_int 0 in
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx out_spatial block_coords;
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
            in_spatial.(d) <- pos;
            if pos < 0 || pos >= input_spatial.(d) then valid := false
          done;
          let v =
            if !valid then
              let src_lin =
                ref (in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1)))
              in
              for d = 0 to spatial_ndim - 1 do
                src_lin := !src_lin + (in_spatial.(d) * in_strides.(d + 2))
              done;
              Array.unsafe_get in_arr !src_lin
            else zero
          in
          let dst_ch = (c_idx * kernel_elems) + k_idx in
          let dst_lin =
            out_offset
            + (n_idx * out_strides.(0))
            + (dst_ch * out_strides.(1))
            + (b_idx * out_strides.(2))
          in
          Array.unsafe_set out_arr dst_lin v
        done
      done
    done
  done

let unfold_int8 in_arr out_arr ~n ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Int8_u.of_int 0 in
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx out_spatial block_coords;
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
            in_spatial.(d) <- pos;
            if pos < 0 || pos >= input_spatial.(d) then valid := false
          done;
          let v =
            if !valid then
              let src_lin =
                ref (in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1)))
              in
              for d = 0 to spatial_ndim - 1 do
                src_lin := !src_lin + (in_spatial.(d) * in_strides.(d + 2))
              done;
              Array.unsafe_get in_arr !src_lin
            else zero
          in
          let dst_ch = (c_idx * kernel_elems) + k_idx in
          let dst_lin =
            out_offset
            + (n_idx * out_strides.(0))
            + (dst_ch * out_strides.(1))
            + (b_idx * out_strides.(2))
          in
          Array.unsafe_set out_arr dst_lin v
        done
      done
    done
  done

let unfold_int16 in_arr out_arr ~n ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Int16_u.of_int 0 in
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx out_spatial block_coords;
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
            in_spatial.(d) <- pos;
            if pos < 0 || pos >= input_spatial.(d) then valid := false
          done;
          let v =
            if !valid then
              let src_lin =
                ref (in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1)))
              in
              for d = 0 to spatial_ndim - 1 do
                src_lin := !src_lin + (in_spatial.(d) * in_strides.(d + 2))
              done;
              Array.unsafe_get in_arr !src_lin
            else zero
          in
          let dst_ch = (c_idx * kernel_elems) + k_idx in
          let dst_lin =
            out_offset
            + (n_idx * out_strides.(0))
            + (dst_ch * out_strides.(1))
            + (b_idx * out_strides.(2))
          in
          Array.unsafe_set out_arr dst_lin v
        done
      done
    done
  done

let unfold_int32 in_arr out_arr ~n ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Int32_u.of_int32 0l in
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx out_spatial block_coords;
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
            in_spatial.(d) <- pos;
            if pos < 0 || pos >= input_spatial.(d) then valid := false
          done;
          let v =
            if !valid then
              let src_lin =
                ref (in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1)))
              in
              for d = 0 to spatial_ndim - 1 do
                src_lin := !src_lin + (in_spatial.(d) * in_strides.(d + 2))
              done;
              Array.unsafe_get in_arr !src_lin
            else zero
          in
          let dst_ch = (c_idx * kernel_elems) + k_idx in
          let dst_lin =
            out_offset
            + (n_idx * out_strides.(0))
            + (dst_ch * out_strides.(1))
            + (b_idx * out_strides.(2))
          in
          Array.unsafe_set out_arr dst_lin v
        done
      done
    done
  done

let unfold_int64 in_arr out_arr ~n ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Int64_u.of_int64 0L in
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx out_spatial block_coords;
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
            in_spatial.(d) <- pos;
            if pos < 0 || pos >= input_spatial.(d) then valid := false
          done;
          let v =
            if !valid then
              let src_lin =
                ref (in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1)))
              in
              for d = 0 to spatial_ndim - 1 do
                src_lin := !src_lin + (in_spatial.(d) * in_strides.(d + 2))
              done;
              Array.unsafe_get in_arr !src_lin
            else zero
          in
          let dst_ch = (c_idx * kernel_elems) + k_idx in
          let dst_lin =
            out_offset
            + (n_idx * out_strides.(0))
            + (dst_ch * out_strides.(1))
            + (b_idx * out_strides.(2))
          in
          Array.unsafe_set out_arr dst_lin v
        done
      done
    done
  done

let unfold_bool in_arr out_arr ~n ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = false in
  for n_idx = 0 to n - 1 do
    for b_idx = 0 to num_blocks - 1 do
      Shape.unravel_index_into b_idx out_spatial block_coords;
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
            in_spatial.(d) <- pos;
            if pos < 0 || pos >= input_spatial.(d) then valid := false
          done;
          let v =
            if !valid then
              let src_lin =
                ref (in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1)))
              in
              for d = 0 to spatial_ndim - 1 do
                src_lin := !src_lin + (in_spatial.(d) * in_strides.(d + 2))
              done;
              Array.unsafe_get in_arr !src_lin
            else zero
          in
          let dst_ch = (c_idx * kernel_elems) + k_idx in
          let dst_lin =
            out_offset
            + (n_idx * out_strides.(0))
            + (dst_ch * out_strides.(1))
            + (b_idx * out_strides.(2))
          in
          Array.unsafe_set out_arr dst_lin v
        done
      done
    done
  done
