(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let is_identity_window ~spatial_ndim ~kernel_elems ~kernel_size ~stride ~dilation
    ~padding =
  if kernel_elems <> 1 then false
  else
    let ok = ref true in
    for d = 0 to spatial_ndim - 1 do
      let pad_before, pad_after = padding.(d) in
      if
        kernel_size.(d) <> 1 || stride.(d) <> 1 || dilation.(d) <> 1
        || pad_before <> 0 || pad_after <> 0
      then ok := false
    done;
    !ok

let is_c_contiguous_spatial_tail spatial strides =
  let expected = ref 1 in
  let ok = ref true in
  for d = Array.length spatial - 1 downto 0 do
    if strides.(d + 2) <> !expected then ok := false;
    expected := !expected * spatial.(d)
  done;
  !ok

let fold_float64 in_arr out_arr ~n_start ~n_end ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  if
    is_identity_window ~spatial_ndim ~kernel_elems ~kernel_size ~stride
      ~dilation ~padding
    && num_blocks = Shape.numel output_size
    && in_strides.(2) = 1
    && is_c_contiguous_spatial_tail output_size out_strides
  then (
    for n_idx = n_start to n_end - 1 do
      for c_idx = 0 to channels - 1 do
        let src_base =
          in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1))
        in
        let dst_base =
          out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1))
        in
        if in_strides.(2) = 1 then (
          let i = ref 0 in
          let n = num_blocks in
          let n8 = n - 7 in
          while !i < n8 do
            let idx = !i in
            let a0 = Float64x2.Array.unsafe_get in_arr ~idx:(src_base + idx) in
            let a1 =
              Float64x2.Array.unsafe_get in_arr ~idx:(src_base + idx + 2)
            in
            let a2 =
              Float64x2.Array.unsafe_get in_arr ~idx:(src_base + idx + 4)
            in
            let a3 =
              Float64x2.Array.unsafe_get in_arr ~idx:(src_base + idx + 6)
            in
            Float64x2.Array.unsafe_set out_arr ~idx:(dst_base + idx) a0;
            Float64x2.Array.unsafe_set out_arr ~idx:(dst_base + idx + 2) a1;
            Float64x2.Array.unsafe_set out_arr ~idx:(dst_base + idx + 4) a2;
            Float64x2.Array.unsafe_set out_arr ~idx:(dst_base + idx + 6) a3;
            i := idx + 8
          done;
          let n2 = n - 1 in
          while !i < n2 do
            let idx = !i in
            let a = Float64x2.Array.unsafe_get in_arr ~idx:(src_base + idx) in
            Float64x2.Array.unsafe_set out_arr ~idx:(dst_base + idx) a;
            i := idx + 2
          done;
          while !i < n do
            let idx = !i in
            Array.unsafe_set out_arr (dst_base + idx)
              (Array.unsafe_get in_arr (src_base + idx));
            incr i
          done)
        else
          for b_idx = 0 to num_blocks - 1 do
            let src_lin = src_base + (b_idx * in_strides.(2)) in
            let dst_lin = dst_base + (b_idx * out_strides.(2)) in
            Array.unsafe_set out_arr dst_lin (Array.unsafe_get in_arr src_lin)
          done
      done
    done)
  else (
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Float_u.of_float 0.0 in
  let batch_numel = Array.fold_left ( * ) channels output_size in
  let zero_start = n_start * batch_numel in
  let zero_end = (n_end * batch_numel) - 1 in
  for i = zero_start to zero_end do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = n_start to n_end - 1 do
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
  done)

let fold_float32 in_arr out_arr ~n_start ~n_end ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  if
    is_identity_window ~spatial_ndim ~kernel_elems ~kernel_size ~stride
      ~dilation ~padding
    && num_blocks = Shape.numel output_size
    && in_strides.(2) = 1
    && is_c_contiguous_spatial_tail output_size out_strides
  then (
    for n_idx = n_start to n_end - 1 do
      for c_idx = 0 to channels - 1 do
        let src_base =
          in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1))
        in
        let dst_base =
          out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1))
        in
        if in_strides.(2) = 1 then (
          let i = ref 0 in
          let n = num_blocks in
          let n16 = n - 15 in
          while !i < n16 do
            let idx = !i in
            let a0 = Float32x4.Array.unsafe_get in_arr ~idx:(src_base + idx) in
            let a1 =
              Float32x4.Array.unsafe_get in_arr ~idx:(src_base + idx + 4)
            in
            let a2 =
              Float32x4.Array.unsafe_get in_arr ~idx:(src_base + idx + 8)
            in
            let a3 =
              Float32x4.Array.unsafe_get in_arr ~idx:(src_base + idx + 12)
            in
            Float32x4.Array.unsafe_set out_arr ~idx:(dst_base + idx) a0;
            Float32x4.Array.unsafe_set out_arr ~idx:(dst_base + idx + 4) a1;
            Float32x4.Array.unsafe_set out_arr ~idx:(dst_base + idx + 8) a2;
            Float32x4.Array.unsafe_set out_arr ~idx:(dst_base + idx + 12) a3;
            i := idx + 16
          done;
          let n4 = n - 3 in
          while !i < n4 do
            let idx = !i in
            let a = Float32x4.Array.unsafe_get in_arr ~idx:(src_base + idx) in
            Float32x4.Array.unsafe_set out_arr ~idx:(dst_base + idx) a;
            i := idx + 4
          done;
          while !i < n do
            let idx = !i in
            Array.unsafe_set out_arr (dst_base + idx)
              (Array.unsafe_get in_arr (src_base + idx));
            incr i
          done)
        else
          for b_idx = 0 to num_blocks - 1 do
            let src_lin = src_base + (b_idx * in_strides.(2)) in
            let dst_lin = dst_base + (b_idx * out_strides.(2)) in
            Array.unsafe_set out_arr dst_lin (Array.unsafe_get in_arr src_lin)
          done
      done
    done)
  else (
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Float32_u.of_int 0 in
  let batch_numel = Array.fold_left ( * ) channels output_size in
  let zero_start = n_start * batch_numel in
  let zero_end = (n_end * batch_numel) - 1 in
  for i = zero_start to zero_end do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = n_start to n_end - 1 do
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
  done)

let fold_int8 in_arr out_arr ~n_start ~n_end ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Int8_u.of_int 0 in
  let batch_numel = Array.fold_left ( * ) channels output_size in
  let zero_start = n_start * batch_numel in
  let zero_end = (n_end * batch_numel) - 1 in
  for i = zero_start to zero_end do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = n_start to n_end - 1 do
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

let fold_int16 in_arr out_arr ~n_start ~n_end ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Int16_u.of_int 0 in
  let batch_numel = Array.fold_left ( * ) channels output_size in
  let zero_start = n_start * batch_numel in
  let zero_end = (n_end * batch_numel) - 1 in
  for i = zero_start to zero_end do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = n_start to n_end - 1 do
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

let fold_int32 in_arr out_arr ~n_start ~n_end ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  if
    is_identity_window ~spatial_ndim ~kernel_elems ~kernel_size ~stride
      ~dilation ~padding
    && num_blocks = Shape.numel output_size
    && in_strides.(2) = 1
    && is_c_contiguous_spatial_tail output_size out_strides
  then (
    for n_idx = n_start to n_end - 1 do
      for c_idx = 0 to channels - 1 do
        let src_base =
          in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1))
        in
        let dst_base =
          out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1))
        in
        let i = ref 0 in
        let n = num_blocks in
        let n16 = n - 15 in
        while !i < n16 do
          let idx = !i in
          let a0 = Int32x4.Array.unsafe_get in_arr ~idx:(src_base + idx) in
          let a1 = Int32x4.Array.unsafe_get in_arr ~idx:(src_base + idx + 4) in
          let a2 = Int32x4.Array.unsafe_get in_arr ~idx:(src_base + idx + 8) in
          let a3 = Int32x4.Array.unsafe_get in_arr ~idx:(src_base + idx + 12) in
          Int32x4.Array.unsafe_set out_arr ~idx:(dst_base + idx) a0;
          Int32x4.Array.unsafe_set out_arr ~idx:(dst_base + idx + 4) a1;
          Int32x4.Array.unsafe_set out_arr ~idx:(dst_base + idx + 8) a2;
          Int32x4.Array.unsafe_set out_arr ~idx:(dst_base + idx + 12) a3;
          i := idx + 16
        done;
        let n4 = n - 3 in
        while !i < n4 do
          let idx = !i in
          let a = Int32x4.Array.unsafe_get in_arr ~idx:(src_base + idx) in
          Int32x4.Array.unsafe_set out_arr ~idx:(dst_base + idx) a;
          i := idx + 4
        done;
        while !i < n do
          let idx = !i in
          Array.unsafe_set out_arr (dst_base + idx)
            (Array.unsafe_get in_arr (src_base + idx));
          incr i
        done
      done
    done)
  else (
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Int32_u.of_int32 0l in
  let batch_numel = Array.fold_left ( * ) channels output_size in
  let zero_start = n_start * batch_numel in
  let zero_end = (n_end * batch_numel) - 1 in
  for i = zero_start to zero_end do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = n_start to n_end - 1 do
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
  done)

let fold_int64 in_arr out_arr ~n_start ~n_end ~channels ~num_blocks ~kernel_elems
    ~spatial_ndim ~blocks_shape ~kernel_size ~output_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  if
    is_identity_window ~spatial_ndim ~kernel_elems ~kernel_size ~stride
      ~dilation ~padding
    && num_blocks = Shape.numel output_size
    && in_strides.(2) = 1
    && is_c_contiguous_spatial_tail output_size out_strides
  then (
    for n_idx = n_start to n_end - 1 do
      for c_idx = 0 to channels - 1 do
        let src_base =
          in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1))
        in
        let dst_base =
          out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1))
        in
        let i = ref 0 in
        let n = num_blocks in
        let n8 = n - 7 in
        while !i < n8 do
          let idx = !i in
          let a0 = Int64x2.Array.unsafe_get in_arr ~idx:(src_base + idx) in
          let a1 = Int64x2.Array.unsafe_get in_arr ~idx:(src_base + idx + 2) in
          let a2 = Int64x2.Array.unsafe_get in_arr ~idx:(src_base + idx + 4) in
          let a3 = Int64x2.Array.unsafe_get in_arr ~idx:(src_base + idx + 6) in
          Int64x2.Array.unsafe_set out_arr ~idx:(dst_base + idx) a0;
          Int64x2.Array.unsafe_set out_arr ~idx:(dst_base + idx + 2) a1;
          Int64x2.Array.unsafe_set out_arr ~idx:(dst_base + idx + 4) a2;
          Int64x2.Array.unsafe_set out_arr ~idx:(dst_base + idx + 6) a3;
          i := idx + 8
        done;
        let n2 = n - 1 in
        while !i < n2 do
          let idx = !i in
          let a = Int64x2.Array.unsafe_get in_arr ~idx:(src_base + idx) in
          Int64x2.Array.unsafe_set out_arr ~idx:(dst_base + idx) a;
          i := idx + 2
        done;
        while !i < n do
          let idx = !i in
          Array.unsafe_set out_arr (dst_base + idx)
            (Array.unsafe_get in_arr (src_base + idx));
          incr i
        done
      done
    done)
  else (
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let out_spatial = Array.make spatial_ndim 0 in
  let zero = Int64_u.of_int64 0L in
  let batch_numel = Array.fold_left ( * ) channels output_size in
  let zero_start = n_start * batch_numel in
  let zero_end = (n_end * batch_numel) - 1 in
  for i = zero_start to zero_end do
    Array.unsafe_set out_arr i zero
  done;
  for n_idx = n_start to n_end - 1 do
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
  done)
