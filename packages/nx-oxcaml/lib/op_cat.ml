(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let range_prod shape start_idx end_idx_excl =
  let acc = ref 1 in
  for i = start_idx to end_idx_excl - 1 do
    acc := !acc * shape.(i)
  done;
  !acc

let out_shape_from_srcs srcs axis =
  match srcs with
  | [] -> [||]
  | (_, first_view) :: rest ->
      let out_shape = Array.copy (shape first_view) in
      let axis_total = ref out_shape.(axis) in
      List.iter
        (fun (_, view) -> axis_total := !axis_total + (shape view).(axis))
        rest;
      out_shape.(axis) <- !axis_total;
      out_shape

let is_c_contiguous_strides shape_arr strides =
  let expected = Shape.c_contiguous_strides shape_arr in
  let ok = ref (Array.length expected = Array.length strides) in
  let i = ref 0 in
  while !ok && !i < Array.length expected do
    if expected.(!i) <> strides.(!i) then ok := false;
    incr i
  done;
  !ok

let copy_block_float64 src dst src_base dst_base len =
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
    let v = Float64x2.Array.unsafe_get src ~idx:(src_base + idx) in
    Float64x2.Array.unsafe_set dst ~idx:(dst_base + idx) v;
    i := idx + 2
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_float32 src dst src_base dst_base len =
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
    let v = Float32x4.Array.unsafe_get src ~idx:(src_base + idx) in
    Float32x4.Array.unsafe_set dst ~idx:(dst_base + idx) v;
    i := idx + 4
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_int32 src dst src_base dst_base len =
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
    let v = Int32x4.Array.unsafe_get src ~idx:(src_base + idx) in
    Int32x4.Array.unsafe_set dst ~idx:(dst_base + idx) v;
    i := idx + 4
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_int64 src dst src_base dst_base len =
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
    let v = Int64x2.Array.unsafe_get src ~idx:(src_base + idx) in
    Int64x2.Array.unsafe_set dst ~idx:(dst_base + idx) v;
    i := idx + 2
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_int8 src dst src_base dst_base len =
  let i = ref 0 in
  let n8 = len - 7 in
  while !i < n8 do
    let idx = !i in
    let s0 = src_base + idx in
    let d0 = dst_base + idx in
    Array.unsafe_set dst d0 (Array.unsafe_get src s0);
    Array.unsafe_set dst (d0 + 1) (Array.unsafe_get src (s0 + 1));
    Array.unsafe_set dst (d0 + 2) (Array.unsafe_get src (s0 + 2));
    Array.unsafe_set dst (d0 + 3) (Array.unsafe_get src (s0 + 3));
    Array.unsafe_set dst (d0 + 4) (Array.unsafe_get src (s0 + 4));
    Array.unsafe_set dst (d0 + 5) (Array.unsafe_get src (s0 + 5));
    Array.unsafe_set dst (d0 + 6) (Array.unsafe_get src (s0 + 6));
    Array.unsafe_set dst (d0 + 7) (Array.unsafe_get src (s0 + 7));
    i := idx + 8
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_int16 src dst src_base dst_base len =
  let i = ref 0 in
  let n8 = len - 7 in
  while !i < n8 do
    let idx = !i in
    let s0 = src_base + idx in
    let d0 = dst_base + idx in
    Array.unsafe_set dst d0 (Array.unsafe_get src s0);
    Array.unsafe_set dst (d0 + 1) (Array.unsafe_get src (s0 + 1));
    Array.unsafe_set dst (d0 + 2) (Array.unsafe_get src (s0 + 2));
    Array.unsafe_set dst (d0 + 3) (Array.unsafe_get src (s0 + 3));
    Array.unsafe_set dst (d0 + 4) (Array.unsafe_get src (s0 + 4));
    Array.unsafe_set dst (d0 + 5) (Array.unsafe_get src (s0 + 5));
    Array.unsafe_set dst (d0 + 6) (Array.unsafe_get src (s0 + 6));
    Array.unsafe_set dst (d0 + 7) (Array.unsafe_get src (s0 + 7));
    i := idx + 8
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let copy_block_bool src dst src_base dst_base len =
  let i = ref 0 in
  let n8 = len - 7 in
  while !i < n8 do
    let idx = !i in
    let s0 = src_base + idx in
    let d0 = dst_base + idx in
    Array.unsafe_set dst d0 (Array.unsafe_get src s0);
    Array.unsafe_set dst (d0 + 1) (Array.unsafe_get src (s0 + 1));
    Array.unsafe_set dst (d0 + 2) (Array.unsafe_get src (s0 + 2));
    Array.unsafe_set dst (d0 + 3) (Array.unsafe_get src (s0 + 3));
    Array.unsafe_set dst (d0 + 4) (Array.unsafe_get src (s0 + 4));
    Array.unsafe_set dst (d0 + 5) (Array.unsafe_get src (s0 + 5));
    Array.unsafe_set dst (d0 + 6) (Array.unsafe_get src (s0 + 6));
    Array.unsafe_set dst (d0 + 7) (Array.unsafe_get src (s0 + 7));
    i := idx + 8
  done;
  while !i < len do
    let idx = !i in
    Array.unsafe_set dst (dst_base + idx) (Array.unsafe_get src (src_base + idx));
    incr i
  done

let cat_strided_copy src dst in_shape in_offset in_strides out_offset out_strides
    set_elem =
  let n = Shape.numel in_shape in
  let dims = Array.length in_shape in
  let coord = Array.make dims 0 in
  let in_i = ref in_offset in
  let out_i = ref out_offset in
  for _ = 0 to n - 1 do
    set_elem src dst !in_i !out_i;
    let d = ref (dims - 1) in
    while !d >= 0 do
      coord.(!d) <- coord.(!d) + 1;
      in_i := !in_i + in_strides.(!d);
      out_i := !out_i + out_strides.(!d);
      if coord.(!d) < in_shape.(!d) then d := -1
      else (
        in_i := !in_i - (coord.(!d) * in_strides.(!d));
        out_i := !out_i - (coord.(!d) * out_strides.(!d));
        coord.(!d) <- 0;
        decr d)
    done
  done

let cat_float64 (srcs : (float# array * View.t) list) (dst : float# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  if Array.length out_strides <> rank then invalid_arg "cat_float64: invalid rank";
  let out_shape = out_shape_from_srcs srcs axis in
  let out_contiguous = is_c_contiguous_strides out_shape out_strides in
  let all_src_contiguous =
    out_contiguous && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  if all_src_contiguous then (
    let inner = range_prod out_shape (axis + 1) rank in
    let outer = range_prod out_shape 0 axis in
    let out_axis = out_shape.(axis) in
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let axis_len = in_shape.(axis) in
        let block = axis_len * inner in
        for o = 0 to outer - 1 do
          let src_base = in_offset + (o * block) in
          let dst_base = out_offset + (((o * out_axis) + !axis_base) * inner) in
          copy_block_float64 src dst src_base dst_base block
        done;
        axis_base := !axis_base + axis_len)
      srcs
  ) else (
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let in_strides = View.strides view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        cat_strided_copy src dst in_shape in_offset in_strides dst_base out_strides
          (fun src_arr dst_arr in_i out_i ->
            Array.unsafe_set dst_arr out_i (Array.unsafe_get src_arr in_i));
        axis_base := !axis_base + in_shape.(axis))
      srcs)

let cat_float32 (srcs : (float32# array * View.t) list) (dst : float32# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  if Array.length out_strides <> rank then invalid_arg "cat_float32: invalid rank";
  let out_shape = out_shape_from_srcs srcs axis in
  let out_contiguous = is_c_contiguous_strides out_shape out_strides in
  let all_src_contiguous =
    out_contiguous && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  if all_src_contiguous then (
    let inner = range_prod out_shape (axis + 1) rank in
    let outer = range_prod out_shape 0 axis in
    let out_axis = out_shape.(axis) in
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let axis_len = in_shape.(axis) in
        let block = axis_len * inner in
        for o = 0 to outer - 1 do
          let src_base = in_offset + (o * block) in
          let dst_base = out_offset + (((o * out_axis) + !axis_base) * inner) in
          copy_block_float32 src dst src_base dst_base block
        done;
        axis_base := !axis_base + axis_len)
      srcs
  ) else (
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let in_strides = View.strides view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        cat_strided_copy src dst in_shape in_offset in_strides dst_base out_strides
          (fun src_arr dst_arr in_i out_i ->
            Array.unsafe_set dst_arr out_i (Array.unsafe_get src_arr in_i));
        axis_base := !axis_base + in_shape.(axis))
      srcs)

let cat_int8 (srcs : (int8# array * View.t) list) (dst : int8# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  if Array.length out_strides <> rank then invalid_arg "cat_int8: invalid rank";
  let out_shape = out_shape_from_srcs srcs axis in
  let out_contiguous = is_c_contiguous_strides out_shape out_strides in
  let all_src_contiguous =
    out_contiguous && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  if all_src_contiguous then (
    let inner = range_prod out_shape (axis + 1) rank in
    let outer = range_prod out_shape 0 axis in
    let out_axis = out_shape.(axis) in
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let axis_len = in_shape.(axis) in
        let block = axis_len * inner in
        for o = 0 to outer - 1 do
          let src_base = in_offset + (o * block) in
          let dst_base = out_offset + (((o * out_axis) + !axis_base) * inner) in
          copy_block_int8 src dst src_base dst_base block
        done;
        axis_base := !axis_base + axis_len)
      srcs
  ) else (
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let in_strides = View.strides view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        cat_strided_copy src dst in_shape in_offset in_strides dst_base out_strides
          (fun src_arr dst_arr in_i out_i ->
            Array.unsafe_set dst_arr out_i (Array.unsafe_get src_arr in_i));
        axis_base := !axis_base + in_shape.(axis))
      srcs)

let cat_int16 (srcs : (int16# array * View.t) list) (dst : int16# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  if Array.length out_strides <> rank then invalid_arg "cat_int16: invalid rank";
  let out_shape = out_shape_from_srcs srcs axis in
  let out_contiguous = is_c_contiguous_strides out_shape out_strides in
  let all_src_contiguous =
    out_contiguous && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  if all_src_contiguous then (
    let inner = range_prod out_shape (axis + 1) rank in
    let outer = range_prod out_shape 0 axis in
    let out_axis = out_shape.(axis) in
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let axis_len = in_shape.(axis) in
        let block = axis_len * inner in
        for o = 0 to outer - 1 do
          let src_base = in_offset + (o * block) in
          let dst_base = out_offset + (((o * out_axis) + !axis_base) * inner) in
          copy_block_int16 src dst src_base dst_base block
        done;
        axis_base := !axis_base + axis_len)
      srcs
  ) else (
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let in_strides = View.strides view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        cat_strided_copy src dst in_shape in_offset in_strides dst_base out_strides
          (fun src_arr dst_arr in_i out_i ->
            Array.unsafe_set dst_arr out_i (Array.unsafe_get src_arr in_i));
        axis_base := !axis_base + in_shape.(axis))
      srcs)

let cat_int32 (srcs : (int32# array * View.t) list) (dst : int32# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  if Array.length out_strides <> rank then invalid_arg "cat_int32: invalid rank";
  let out_shape = out_shape_from_srcs srcs axis in
  let out_contiguous = is_c_contiguous_strides out_shape out_strides in
  let all_src_contiguous =
    out_contiguous && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  if all_src_contiguous then (
    let inner = range_prod out_shape (axis + 1) rank in
    let outer = range_prod out_shape 0 axis in
    let out_axis = out_shape.(axis) in
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let axis_len = in_shape.(axis) in
        let block = axis_len * inner in
        for o = 0 to outer - 1 do
          let src_base = in_offset + (o * block) in
          let dst_base = out_offset + (((o * out_axis) + !axis_base) * inner) in
          copy_block_int32 src dst src_base dst_base block
        done;
        axis_base := !axis_base + axis_len)
      srcs
  ) else (
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let in_strides = View.strides view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        cat_strided_copy src dst in_shape in_offset in_strides dst_base out_strides
          (fun src_arr dst_arr in_i out_i ->
            Array.unsafe_set dst_arr out_i (Array.unsafe_get src_arr in_i));
        axis_base := !axis_base + in_shape.(axis))
      srcs)

let cat_int64 (srcs : (int64# array * View.t) list) (dst : int64# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  if Array.length out_strides <> rank then invalid_arg "cat_int64: invalid rank";
  let out_shape = out_shape_from_srcs srcs axis in
  let out_contiguous = is_c_contiguous_strides out_shape out_strides in
  let all_src_contiguous =
    out_contiguous && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  if all_src_contiguous then (
    let inner = range_prod out_shape (axis + 1) rank in
    let outer = range_prod out_shape 0 axis in
    let out_axis = out_shape.(axis) in
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let axis_len = in_shape.(axis) in
        let block = axis_len * inner in
        for o = 0 to outer - 1 do
          let src_base = in_offset + (o * block) in
          let dst_base = out_offset + (((o * out_axis) + !axis_base) * inner) in
          copy_block_int64 src dst src_base dst_base block
        done;
        axis_base := !axis_base + axis_len)
      srcs
  ) else (
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let in_strides = View.strides view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        cat_strided_copy src dst in_shape in_offset in_strides dst_base out_strides
          (fun src_arr dst_arr in_i out_i ->
            Array.unsafe_set dst_arr out_i (Array.unsafe_get src_arr in_i));
        axis_base := !axis_base + in_shape.(axis))
      srcs)

let cat_bool (srcs : (bool array * View.t) list) (dst : bool array) (rank : int)
    (axis : int) (out_offset : int) (out_strides : int array) =
  if Array.length out_strides <> rank then invalid_arg "cat_bool: invalid rank";
  let out_shape = out_shape_from_srcs srcs axis in
  let out_contiguous = is_c_contiguous_strides out_shape out_strides in
  let all_src_contiguous =
    out_contiguous && List.for_all (fun (_, view) -> View.is_c_contiguous view) srcs
  in
  if all_src_contiguous then (
    let inner = range_prod out_shape (axis + 1) rank in
    let outer = range_prod out_shape 0 axis in
    let out_axis = out_shape.(axis) in
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let axis_len = in_shape.(axis) in
        let block = axis_len * inner in
        for o = 0 to outer - 1 do
          let src_base = in_offset + (o * block) in
          let dst_base = out_offset + (((o * out_axis) + !axis_base) * inner) in
          copy_block_bool src dst src_base dst_base block
        done;
        axis_base := !axis_base + axis_len)
      srcs
  ) else (
    let axis_base = ref 0 in
    List.iter
      (fun (src, view) ->
        let in_shape = shape view in
        let in_offset = View.offset view in
        let in_strides = View.strides view in
        let dst_base = out_offset + (!axis_base * out_strides.(axis)) in
        cat_strided_copy src dst in_shape in_offset in_strides dst_base out_strides
          (fun src_arr dst_arr in_i out_i ->
            Array.unsafe_set dst_arr out_i (Array.unsafe_get src_arr in_i));
        axis_base := !axis_base + in_shape.(axis))
      srcs)
