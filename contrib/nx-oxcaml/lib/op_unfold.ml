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

let unfold_float64 in_arr out_arr ~n_start ~n_end ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  if
    is_identity_window ~spatial_ndim ~kernel_elems ~kernel_size ~stride
      ~dilation ~padding
    && num_blocks = Shape.numel input_spatial
    && is_c_contiguous_spatial_tail input_spatial in_strides
    && out_strides.(2) = 1
  then (
    for n_idx = n_start to n_end - 1 do
      for c_idx = 0 to channels - 1 do
        let src_base =
          in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1))
        in
        let dst_base =
          out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1))
        in
        if out_strides.(2) = 1 then (
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
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Float_u.of_float 0.0 in
  for n_idx = n_start to n_end - 1 do
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
  done)

let unfold_float32 in_arr out_arr ~n_start ~n_end ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  if
    is_identity_window ~spatial_ndim ~kernel_elems ~kernel_size ~stride
      ~dilation ~padding
    && num_blocks = Shape.numel input_spatial
    && is_c_contiguous_spatial_tail input_spatial in_strides
    && out_strides.(2) = 1
  then (
    for n_idx = n_start to n_end - 1 do
      for c_idx = 0 to channels - 1 do
        let src_base =
          in_offset + (n_idx * in_strides.(0)) + (c_idx * in_strides.(1))
        in
        let dst_base =
          out_offset + (n_idx * out_strides.(0)) + (c_idx * out_strides.(1))
        in
        if out_strides.(2) = 1 then (
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
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Float32_u.of_int 0 in
  for n_idx = n_start to n_end - 1 do
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
  done)

let unfold_int8 in_arr out_arr ~n_start ~n_end ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Int8_u.of_int 0 in
  for n_idx = n_start to n_end - 1 do
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

let unfold_int16 in_arr out_arr ~n_start ~n_end ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Int16_u.of_int 0 in
  for n_idx = n_start to n_end - 1 do
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

let unfold_int32 in_arr out_arr ~n_start ~n_end ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  if
    is_identity_window ~spatial_ndim ~kernel_elems ~kernel_size ~stride
      ~dilation ~padding
    && num_blocks = Shape.numel input_spatial
    && is_c_contiguous_spatial_tail input_spatial in_strides
    && out_strides.(2) = 1
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
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Int32_u.of_int32 0l in
  for n_idx = n_start to n_end - 1 do
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
  done)

let unfold_int64 in_arr out_arr ~n_start ~n_end ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  if
    is_identity_window ~spatial_ndim ~kernel_elems ~kernel_size ~stride
      ~dilation ~padding
    && num_blocks = Shape.numel input_spatial
    && is_c_contiguous_spatial_tail input_spatial in_strides
    && out_strides.(2) = 1
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
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = Int64_u.of_int64 0L in
  for n_idx = n_start to n_end - 1 do
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
  done)

let unfold_bool in_arr out_arr ~n_start ~n_end ~channels ~input_spatial ~kernel_elems
    ~num_blocks ~spatial_ndim ~out_spatial ~kernel_size ~stride ~dilation
    ~padding ~in_offset ~in_strides ~out_offset ~out_strides =
  let block_coords = Array.make spatial_ndim 0 in
  let kernel_coords = Array.make spatial_ndim 0 in
  let in_spatial = Array.make spatial_ndim 0 in
  let zero = false in
  for n_idx = n_start to n_end - 1 do
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
