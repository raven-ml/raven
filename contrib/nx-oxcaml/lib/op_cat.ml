(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let copy_block_float64 (src : float# array) (dst : float# array) src_base
    dst_base len =
  let i = ref 0 in
  let n8 = len - 7 in
  while !i < n8 do
    let idx = !i in
    let a0 = Float64x2.Array.unsafe_get src ~idx:(src_base + idx) in
    let a1 = Float64x2.Array.unsafe_get src ~idx:(src_base + idx + 2) in
    let a2 = Float64x2.Array.unsafe_get src ~idx:(src_base + idx + 4) in
    let a3 = Float64x2.Array.unsafe_get src ~idx:(src_base + idx + 6) in
    Float64x2.Array.unsafe_set dst ~idx:(dst_base + idx) a0;
    Float64x2.Array.unsafe_set dst ~idx:(dst_base + idx + 2) a1;
    Float64x2.Array.unsafe_set dst ~idx:(dst_base + idx + 4) a2;
    Float64x2.Array.unsafe_set dst ~idx:(dst_base + idx + 6) a3;
    i := idx + 8
  done;
  let n2 = len - 1 in
  while !i < n2 do
    let idx = !i in
    let a = Float64x2.Array.unsafe_get src ~idx:(src_base + idx) in
    Float64x2.Array.unsafe_set dst ~idx:(dst_base + idx) a;
    i := idx + 2
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_float32 (src : float32# array) (dst : float32# array) src_base
    dst_base len =
  let i = ref 0 in
  let n16 = len - 15 in
  while !i < n16 do
    let idx = !i in
    let a0 = Float32x4.Array.unsafe_get src ~idx:(src_base + idx) in
    let a1 = Float32x4.Array.unsafe_get src ~idx:(src_base + idx + 4) in
    let a2 = Float32x4.Array.unsafe_get src ~idx:(src_base + idx + 8) in
    let a3 = Float32x4.Array.unsafe_get src ~idx:(src_base + idx + 12) in
    Float32x4.Array.unsafe_set dst ~idx:(dst_base + idx) a0;
    Float32x4.Array.unsafe_set dst ~idx:(dst_base + idx + 4) a1;
    Float32x4.Array.unsafe_set dst ~idx:(dst_base + idx + 8) a2;
    Float32x4.Array.unsafe_set dst ~idx:(dst_base + idx + 12) a3;
    i := idx + 16
  done;
  let n4 = len - 3 in
  while !i < n4 do
    let idx = !i in
    let a = Float32x4.Array.unsafe_get src ~idx:(src_base + idx) in
    Float32x4.Array.unsafe_set dst ~idx:(dst_base + idx) a;
    i := idx + 4
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_int8 (src : int8# array) (dst : int8# array) src_base dst_base
    len =
  let i = ref 0 in
  let n8 = len - 7 in
  while !i < n8 do
    let idx = !i in
    let src_idx = src_base + idx in
    let dst_idx = dst_base + idx in
    Array.unsafe_set dst dst_idx (Array.unsafe_get src src_idx);
    Array.unsafe_set dst (dst_idx + 1) (Array.unsafe_get src (src_idx + 1));
    Array.unsafe_set dst (dst_idx + 2) (Array.unsafe_get src (src_idx + 2));
    Array.unsafe_set dst (dst_idx + 3) (Array.unsafe_get src (src_idx + 3));
    Array.unsafe_set dst (dst_idx + 4) (Array.unsafe_get src (src_idx + 4));
    Array.unsafe_set dst (dst_idx + 5) (Array.unsafe_get src (src_idx + 5));
    Array.unsafe_set dst (dst_idx + 6) (Array.unsafe_get src (src_idx + 6));
    Array.unsafe_set dst (dst_idx + 7) (Array.unsafe_get src (src_idx + 7));
    i := idx + 8
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_int16 (src : int16# array) (dst : int16# array) src_base dst_base
    len =
  let i = ref 0 in
  let n8 = len - 7 in
  while !i < n8 do
    let idx = !i in
    let src_idx = src_base + idx in
    let dst_idx = dst_base + idx in
    Array.unsafe_set dst dst_idx (Array.unsafe_get src src_idx);
    Array.unsafe_set dst (dst_idx + 1) (Array.unsafe_get src (src_idx + 1));
    Array.unsafe_set dst (dst_idx + 2) (Array.unsafe_get src (src_idx + 2));
    Array.unsafe_set dst (dst_idx + 3) (Array.unsafe_get src (src_idx + 3));
    Array.unsafe_set dst (dst_idx + 4) (Array.unsafe_get src (src_idx + 4));
    Array.unsafe_set dst (dst_idx + 5) (Array.unsafe_get src (src_idx + 5));
    Array.unsafe_set dst (dst_idx + 6) (Array.unsafe_get src (src_idx + 6));
    Array.unsafe_set dst (dst_idx + 7) (Array.unsafe_get src (src_idx + 7));
    i := idx + 8
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_int32 (src : int32# array) (dst : int32# array) src_base dst_base
    len =
  let i = ref 0 in
  let n16 = len - 15 in
  while !i < n16 do
    let idx = !i in
    let a0 = Int32x4.Array.unsafe_get src ~idx:(src_base + idx) in
    let a1 = Int32x4.Array.unsafe_get src ~idx:(src_base + idx + 4) in
    let a2 = Int32x4.Array.unsafe_get src ~idx:(src_base + idx + 8) in
    let a3 = Int32x4.Array.unsafe_get src ~idx:(src_base + idx + 12) in
    Int32x4.Array.unsafe_set dst ~idx:(dst_base + idx) a0;
    Int32x4.Array.unsafe_set dst ~idx:(dst_base + idx + 4) a1;
    Int32x4.Array.unsafe_set dst ~idx:(dst_base + idx + 8) a2;
    Int32x4.Array.unsafe_set dst ~idx:(dst_base + idx + 12) a3;
    i := idx + 16
  done;
  let n4 = len - 3 in
  while !i < n4 do
    let idx = !i in
    let a = Int32x4.Array.unsafe_get src ~idx:(src_base + idx) in
    Int32x4.Array.unsafe_set dst ~idx:(dst_base + idx) a;
    i := idx + 4
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_int64 (src : int64# array) (dst : int64# array) src_base dst_base
    len =
  let i = ref 0 in
  let n8 = len - 7 in
  while !i < n8 do
    let idx = !i in
    let a0 = Int64x2.Array.unsafe_get src ~idx:(src_base + idx) in
    let a1 = Int64x2.Array.unsafe_get src ~idx:(src_base + idx + 2) in
    let a2 = Int64x2.Array.unsafe_get src ~idx:(src_base + idx + 4) in
    let a3 = Int64x2.Array.unsafe_get src ~idx:(src_base + idx + 6) in
    Int64x2.Array.unsafe_set dst ~idx:(dst_base + idx) a0;
    Int64x2.Array.unsafe_set dst ~idx:(dst_base + idx + 2) a1;
    Int64x2.Array.unsafe_set dst ~idx:(dst_base + idx + 4) a2;
    Int64x2.Array.unsafe_set dst ~idx:(dst_base + idx + 6) a3;
    i := idx + 8
  done;
  let n2 = len - 1 in
  while !i < n2 do
    let idx = !i in
    let a = Int64x2.Array.unsafe_get src ~idx:(src_base + idx) in
    Int64x2.Array.unsafe_set dst ~idx:(dst_base + idx) a;
    i := idx + 2
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_bool (src : bool array) (dst : bool array) src_base dst_base len =
  let i = ref 0 in
  let n8 = len - 7 in
  while !i < n8 do
    let idx = !i in
    let src_idx = src_base + idx in
    let dst_idx = dst_base + idx in
    Array.unsafe_set dst dst_idx (Array.unsafe_get src src_idx);
    Array.unsafe_set dst (dst_idx + 1) (Array.unsafe_get src (src_idx + 1));
    Array.unsafe_set dst (dst_idx + 2) (Array.unsafe_get src (src_idx + 2));
    Array.unsafe_set dst (dst_idx + 3) (Array.unsafe_get src (src_idx + 3));
    Array.unsafe_set dst (dst_idx + 4) (Array.unsafe_get src (src_idx + 4));
    Array.unsafe_set dst (dst_idx + 5) (Array.unsafe_get src (src_idx + 5));
    Array.unsafe_set dst (dst_idx + 6) (Array.unsafe_get src (src_idx + 6));
    Array.unsafe_set dst (dst_idx + 7) (Array.unsafe_get src (src_idx + 7));
    i := idx + 8
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_strided_float64 (src : float# array) (dst : float# array) in_shape
    in_offset in_strides out_offset out_strides rank =
  let n = Shape.numel in_shape in
  let index = Array.make rank 0 in
  let src_pos = ref in_offset in
  let dst_pos = ref out_offset in
  for element = 0 to n - 1 do
    Array.unsafe_set dst !dst_pos (Array.unsafe_get src !src_pos);
    if element < n - 1 then (
      let dim = ref (rank - 1) in
      while !dim >= 0 && index.(!dim) = in_shape.(!dim) - 1 do
        index.(!dim) <- 0;
        src_pos := !src_pos - ((in_shape.(!dim) - 1) * in_strides.(!dim));
        dst_pos := !dst_pos - ((in_shape.(!dim) - 1) * out_strides.(!dim));
        decr dim
      done;
      if !dim >= 0 then (
        index.(!dim) <- index.(!dim) + 1;
        src_pos := !src_pos + in_strides.(!dim);
        dst_pos := !dst_pos + out_strides.(!dim)))
  done

let copy_strided_float32 (src : float32# array) (dst : float32# array) in_shape
    in_offset in_strides out_offset out_strides rank =
  let n = Shape.numel in_shape in
  let index = Array.make rank 0 in
  let src_pos = ref in_offset in
  let dst_pos = ref out_offset in
  for element = 0 to n - 1 do
    Array.unsafe_set dst !dst_pos (Array.unsafe_get src !src_pos);
    if element < n - 1 then (
      let dim = ref (rank - 1) in
      while !dim >= 0 && index.(!dim) = in_shape.(!dim) - 1 do
        index.(!dim) <- 0;
        src_pos := !src_pos - ((in_shape.(!dim) - 1) * in_strides.(!dim));
        dst_pos := !dst_pos - ((in_shape.(!dim) - 1) * out_strides.(!dim));
        decr dim
      done;
      if !dim >= 0 then (
        index.(!dim) <- index.(!dim) + 1;
        src_pos := !src_pos + in_strides.(!dim);
        dst_pos := !dst_pos + out_strides.(!dim)))
  done

let copy_strided_int8 (src : int8# array) (dst : int8# array) in_shape in_offset
    in_strides out_offset out_strides rank =
  let n = Shape.numel in_shape in
  let index = Array.make rank 0 in
  let src_pos = ref in_offset in
  let dst_pos = ref out_offset in
  for element = 0 to n - 1 do
    Array.unsafe_set dst !dst_pos (Array.unsafe_get src !src_pos);
    if element < n - 1 then (
      let dim = ref (rank - 1) in
      while !dim >= 0 && index.(!dim) = in_shape.(!dim) - 1 do
        index.(!dim) <- 0;
        src_pos := !src_pos - ((in_shape.(!dim) - 1) * in_strides.(!dim));
        dst_pos := !dst_pos - ((in_shape.(!dim) - 1) * out_strides.(!dim));
        decr dim
      done;
      if !dim >= 0 then (
        index.(!dim) <- index.(!dim) + 1;
        src_pos := !src_pos + in_strides.(!dim);
        dst_pos := !dst_pos + out_strides.(!dim)))
  done

let copy_strided_int16 (src : int16# array) (dst : int16# array) in_shape
    in_offset in_strides out_offset out_strides rank =
  let n = Shape.numel in_shape in
  let index = Array.make rank 0 in
  let src_pos = ref in_offset in
  let dst_pos = ref out_offset in
  for element = 0 to n - 1 do
    Array.unsafe_set dst !dst_pos (Array.unsafe_get src !src_pos);
    if element < n - 1 then (
      let dim = ref (rank - 1) in
      while !dim >= 0 && index.(!dim) = in_shape.(!dim) - 1 do
        index.(!dim) <- 0;
        src_pos := !src_pos - ((in_shape.(!dim) - 1) * in_strides.(!dim));
        dst_pos := !dst_pos - ((in_shape.(!dim) - 1) * out_strides.(!dim));
        decr dim
      done;
      if !dim >= 0 then (
        index.(!dim) <- index.(!dim) + 1;
        src_pos := !src_pos + in_strides.(!dim);
        dst_pos := !dst_pos + out_strides.(!dim)))
  done

let copy_strided_int32 (src : int32# array) (dst : int32# array) in_shape
    in_offset in_strides out_offset out_strides rank =
  let n = Shape.numel in_shape in
  let index = Array.make rank 0 in
  let src_pos = ref in_offset in
  let dst_pos = ref out_offset in
  for element = 0 to n - 1 do
    Array.unsafe_set dst !dst_pos (Array.unsafe_get src !src_pos);
    if element < n - 1 then (
      let dim = ref (rank - 1) in
      while !dim >= 0 && index.(!dim) = in_shape.(!dim) - 1 do
        index.(!dim) <- 0;
        src_pos := !src_pos - ((in_shape.(!dim) - 1) * in_strides.(!dim));
        dst_pos := !dst_pos - ((in_shape.(!dim) - 1) * out_strides.(!dim));
        decr dim
      done;
      if !dim >= 0 then (
        index.(!dim) <- index.(!dim) + 1;
        src_pos := !src_pos + in_strides.(!dim);
        dst_pos := !dst_pos + out_strides.(!dim)))
  done

let copy_strided_int64 (src : int64# array) (dst : int64# array) in_shape
    in_offset in_strides out_offset out_strides rank =
  let n = Shape.numel in_shape in
  let index = Array.make rank 0 in
  let src_pos = ref in_offset in
  let dst_pos = ref out_offset in
  for element = 0 to n - 1 do
    Array.unsafe_set dst !dst_pos (Array.unsafe_get src !src_pos);
    if element < n - 1 then (
      let dim = ref (rank - 1) in
      while !dim >= 0 && index.(!dim) = in_shape.(!dim) - 1 do
        index.(!dim) <- 0;
        src_pos := !src_pos - ((in_shape.(!dim) - 1) * in_strides.(!dim));
        dst_pos := !dst_pos - ((in_shape.(!dim) - 1) * out_strides.(!dim));
        decr dim
      done;
      if !dim >= 0 then (
        index.(!dim) <- index.(!dim) + 1;
        src_pos := !src_pos + in_strides.(!dim);
        dst_pos := !dst_pos + out_strides.(!dim)))
  done

let copy_strided_bool (src : bool array) (dst : bool array) in_shape in_offset
    in_strides out_offset out_strides rank =
  let n = Shape.numel in_shape in
  let index = Array.make rank 0 in
  let src_pos = ref in_offset in
  let dst_pos = ref out_offset in
  for element = 0 to n - 1 do
    Array.unsafe_set dst !dst_pos (Array.unsafe_get src !src_pos);
    if element < n - 1 then (
      let dim = ref (rank - 1) in
      while !dim >= 0 && index.(!dim) = in_shape.(!dim) - 1 do
        index.(!dim) <- 0;
        src_pos := !src_pos - ((in_shape.(!dim) - 1) * in_strides.(!dim));
        dst_pos := !dst_pos - ((in_shape.(!dim) - 1) * out_strides.(!dim));
        decr dim
      done;
      if !dim >= 0 then (
        index.(!dim) <- index.(!dim) + 1;
        src_pos := !src_pos + in_strides.(!dim);
        dst_pos := !dst_pos + out_strides.(!dim)))
  done

let cat_float64 (srcs : (float# array * View.t) list) (dst : float# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let out_shape =
    match srcs with
    | [] -> invalid_arg "cat_float64: empty input list"
    | (_, first_view) :: rest ->
        let out_shape = Array.copy (shape first_view) in
        let axis_size = ref out_shape.(axis) in
        List.iter
          (fun (_, view) -> axis_size := !axis_size + (shape view).(axis))
          rest;
        out_shape.(axis) <- !axis_size;
        out_shape
  in
  let out_contiguous = out_strides = Shape.c_contiguous_strides out_shape in
  let all_contiguous =
    out_contiguous
    && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  let axis_base = ref 0 in
  if all_contiguous then (
    let inner = ref 1 in
    for dim = axis + 1 to rank - 1 do
      inner := !inner * out_shape.(dim)
    done;
    let outer = ref 1 in
    for dim = 0 to axis - 1 do
      outer := !outer * out_shape.(dim)
    done;
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let axis_size = in_shape.(axis) in
        let block_size = axis_size * !inner in
        for block = 0 to !outer - 1 do
          let src_base = View.offset view + (block * block_size) in
          let dst_base =
            out_offset
            + (((block * out_shape.(axis)) + !axis_base) * !inner)
          in
          copy_block_float64 src dst src_base dst_base block_size
        done;
        axis_base := !axis_base + axis_size)
      srcs)
  else
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        copy_strided_float64 src dst in_shape (View.offset view)
          (View.strides view) dst_base out_strides rank;
        axis_base := !axis_base + in_shape.(axis))
      srcs

let cat_float32 (srcs : (float32# array * View.t) list) (dst : float32# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let out_shape =
    match srcs with
    | [] -> invalid_arg "cat_float32: empty input list"
    | (_, first_view) :: rest ->
        let out_shape = Array.copy (shape first_view) in
        let axis_size = ref out_shape.(axis) in
        List.iter
          (fun (_, view) -> axis_size := !axis_size + (shape view).(axis))
          rest;
        out_shape.(axis) <- !axis_size;
        out_shape
  in
  let out_contiguous = out_strides = Shape.c_contiguous_strides out_shape in
  let all_contiguous =
    out_contiguous
    && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  let axis_base = ref 0 in
  if all_contiguous then (
    let inner = ref 1 in
    for dim = axis + 1 to rank - 1 do
      inner := !inner * out_shape.(dim)
    done;
    let outer = ref 1 in
    for dim = 0 to axis - 1 do
      outer := !outer * out_shape.(dim)
    done;
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let axis_size = in_shape.(axis) in
        let block_size = axis_size * !inner in
        for block = 0 to !outer - 1 do
          let src_base = View.offset view + (block * block_size) in
          let dst_base =
            out_offset
            + (((block * out_shape.(axis)) + !axis_base) * !inner)
          in
          copy_block_float32 src dst src_base dst_base block_size
        done;
        axis_base := !axis_base + axis_size)
      srcs)
  else
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        copy_strided_float32 src dst in_shape (View.offset view)
          (View.strides view) dst_base out_strides rank;
        axis_base := !axis_base + in_shape.(axis))
      srcs

let cat_int8 (srcs : (int8# array * View.t) list) (dst : int8# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let out_shape =
    match srcs with
    | [] -> invalid_arg "cat_int8: empty input list"
    | (_, first_view) :: rest ->
        let out_shape = Array.copy (shape first_view) in
        let axis_size = ref out_shape.(axis) in
        List.iter
          (fun (_, view) -> axis_size := !axis_size + (shape view).(axis))
          rest;
        out_shape.(axis) <- !axis_size;
        out_shape
  in
  let out_contiguous = out_strides = Shape.c_contiguous_strides out_shape in
  let all_contiguous =
    out_contiguous
    && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  let axis_base = ref 0 in
  if all_contiguous then (
    let inner = ref 1 in
    for dim = axis + 1 to rank - 1 do
      inner := !inner * out_shape.(dim)
    done;
    let outer = ref 1 in
    for dim = 0 to axis - 1 do
      outer := !outer * out_shape.(dim)
    done;
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let axis_size = in_shape.(axis) in
        let block_size = axis_size * !inner in
        for block = 0 to !outer - 1 do
          let src_base = View.offset view + (block * block_size) in
          let dst_base =
            out_offset
            + (((block * out_shape.(axis)) + !axis_base) * !inner)
          in
          copy_block_int8 src dst src_base dst_base block_size
        done;
        axis_base := !axis_base + axis_size)
      srcs)
  else
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        copy_strided_int8 src dst in_shape (View.offset view)
          (View.strides view) dst_base out_strides rank;
        axis_base := !axis_base + in_shape.(axis))
      srcs

let cat_int16 (srcs : (int16# array * View.t) list) (dst : int16# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let out_shape =
    match srcs with
    | [] -> invalid_arg "cat_int16: empty input list"
    | (_, first_view) :: rest ->
        let out_shape = Array.copy (shape first_view) in
        let axis_size = ref out_shape.(axis) in
        List.iter
          (fun (_, view) -> axis_size := !axis_size + (shape view).(axis))
          rest;
        out_shape.(axis) <- !axis_size;
        out_shape
  in
  let out_contiguous = out_strides = Shape.c_contiguous_strides out_shape in
  let all_contiguous =
    out_contiguous
    && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  let axis_base = ref 0 in
  if all_contiguous then (
    let inner = ref 1 in
    for dim = axis + 1 to rank - 1 do
      inner := !inner * out_shape.(dim)
    done;
    let outer = ref 1 in
    for dim = 0 to axis - 1 do
      outer := !outer * out_shape.(dim)
    done;
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let axis_size = in_shape.(axis) in
        let block_size = axis_size * !inner in
        for block = 0 to !outer - 1 do
          let src_base = View.offset view + (block * block_size) in
          let dst_base =
            out_offset
            + (((block * out_shape.(axis)) + !axis_base) * !inner)
          in
          copy_block_int16 src dst src_base dst_base block_size
        done;
        axis_base := !axis_base + axis_size)
      srcs)
  else
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        copy_strided_int16 src dst in_shape (View.offset view)
          (View.strides view) dst_base out_strides rank;
        axis_base := !axis_base + in_shape.(axis))
      srcs

let cat_int32 (srcs : (int32# array * View.t) list) (dst : int32# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let out_shape =
    match srcs with
    | [] -> invalid_arg "cat_int32: empty input list"
    | (_, first_view) :: rest ->
        let out_shape = Array.copy (shape first_view) in
        let axis_size = ref out_shape.(axis) in
        List.iter
          (fun (_, view) -> axis_size := !axis_size + (shape view).(axis))
          rest;
        out_shape.(axis) <- !axis_size;
        out_shape
  in
  let out_contiguous = out_strides = Shape.c_contiguous_strides out_shape in
  let all_contiguous =
    out_contiguous
    && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  let axis_base = ref 0 in
  if all_contiguous then (
    let inner = ref 1 in
    for dim = axis + 1 to rank - 1 do
      inner := !inner * out_shape.(dim)
    done;
    let outer = ref 1 in
    for dim = 0 to axis - 1 do
      outer := !outer * out_shape.(dim)
    done;
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let axis_size = in_shape.(axis) in
        let block_size = axis_size * !inner in
        for block = 0 to !outer - 1 do
          let src_base = View.offset view + (block * block_size) in
          let dst_base =
            out_offset
            + (((block * out_shape.(axis)) + !axis_base) * !inner)
          in
          copy_block_int32 src dst src_base dst_base block_size
        done;
        axis_base := !axis_base + axis_size)
      srcs)
  else
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        copy_strided_int32 src dst in_shape (View.offset view)
          (View.strides view) dst_base out_strides rank;
        axis_base := !axis_base + in_shape.(axis))
      srcs

let cat_int64 (srcs : (int64# array * View.t) list) (dst : int64# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let out_shape =
    match srcs with
    | [] -> invalid_arg "cat_int64: empty input list"
    | (_, first_view) :: rest ->
        let out_shape = Array.copy (shape first_view) in
        let axis_size = ref out_shape.(axis) in
        List.iter
          (fun (_, view) -> axis_size := !axis_size + (shape view).(axis))
          rest;
        out_shape.(axis) <- !axis_size;
        out_shape
  in
  let out_contiguous = out_strides = Shape.c_contiguous_strides out_shape in
  let all_contiguous =
    out_contiguous
    && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  let axis_base = ref 0 in
  if all_contiguous then (
    let inner = ref 1 in
    for dim = axis + 1 to rank - 1 do
      inner := !inner * out_shape.(dim)
    done;
    let outer = ref 1 in
    for dim = 0 to axis - 1 do
      outer := !outer * out_shape.(dim)
    done;
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let axis_size = in_shape.(axis) in
        let block_size = axis_size * !inner in
        for block = 0 to !outer - 1 do
          let src_base = View.offset view + (block * block_size) in
          let dst_base =
            out_offset
            + (((block * out_shape.(axis)) + !axis_base) * !inner)
          in
          copy_block_int64 src dst src_base dst_base block_size
        done;
        axis_base := !axis_base + axis_size)
      srcs)
  else
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        copy_strided_int64 src dst in_shape (View.offset view)
          (View.strides view) dst_base out_strides rank;
        axis_base := !axis_base + in_shape.(axis))
      srcs

let cat_bool (srcs : (bool array * View.t) list) (dst : bool array) (rank : int)
    (axis : int) (out_offset : int) (out_strides : int array) =
  let out_shape =
    match srcs with
    | [] -> invalid_arg "cat_bool: empty input list"
    | (_, first_view) :: rest ->
        let out_shape = Array.copy (shape first_view) in
        let axis_size = ref out_shape.(axis) in
        List.iter
          (fun (_, view) -> axis_size := !axis_size + (shape view).(axis))
          rest;
        out_shape.(axis) <- !axis_size;
        out_shape
  in
  let out_contiguous = out_strides = Shape.c_contiguous_strides out_shape in
  let all_contiguous =
    out_contiguous
    && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  let axis_base = ref 0 in
  if all_contiguous then (
    let inner = ref 1 in
    for dim = axis + 1 to rank - 1 do
      inner := !inner * out_shape.(dim)
    done;
    let outer = ref 1 in
    for dim = 0 to axis - 1 do
      outer := !outer * out_shape.(dim)
    done;
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let axis_size = in_shape.(axis) in
        let block_size = axis_size * !inner in
        for block = 0 to !outer - 1 do
          let src_base = View.offset view + (block * block_size) in
          let dst_base =
            out_offset
            + (((block * out_shape.(axis)) + !axis_base) * !inner)
          in
          copy_block_bool src dst src_base dst_base block_size
        done;
        axis_base := !axis_base + axis_size)
      srcs)
  else
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        copy_strided_bool src dst in_shape (View.offset view)
          (View.strides view) dst_base out_strides rank;
        axis_base := !axis_base + in_shape.(axis))
      srcs
