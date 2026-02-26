(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let pad_float64 (in_arr : float# array) (out_arr : float# array) in_shape padding
    in_offset out_offset in_strides out_strides in_numel =
  let ndim = Array.length in_shape in
  let md_index = Array.make ndim 0 in
  for k = 0 to in_numel - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let v = Array.unsafe_get in_arr src_lin in
    for d = 0 to ndim - 1 do
      let before, _ = padding.(d) in
      md_index.(d) <- md_index.(d) + before
    done;
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set out_arr dst_lin v
  done

let pad_float32 (in_arr : float32# array) (out_arr : float32# array) in_shape
    padding in_offset out_offset in_strides out_strides in_numel =
  let ndim = Array.length in_shape in
  let md_index = Array.make ndim 0 in
  for k = 0 to in_numel - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let v = Array.unsafe_get in_arr src_lin in
    for d = 0 to ndim - 1 do
      let before, _ = padding.(d) in
      md_index.(d) <- md_index.(d) + before
    done;
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set out_arr dst_lin v
  done

let pad_int8 (in_arr : int8# array) (out_arr : int8# array) in_shape padding
    in_offset out_offset in_strides out_strides in_numel =
  let ndim = Array.length in_shape in
  let md_index = Array.make ndim 0 in
  for k = 0 to in_numel - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let v = Array.unsafe_get in_arr src_lin in
    for d = 0 to ndim - 1 do
      let before, _ = padding.(d) in
      md_index.(d) <- md_index.(d) + before
    done;
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set out_arr dst_lin v
  done

let pad_int16 (in_arr : int16# array) (out_arr : int16# array) in_shape padding
    in_offset out_offset in_strides out_strides in_numel =
  let ndim = Array.length in_shape in
  let md_index = Array.make ndim 0 in
  for k = 0 to in_numel - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let v = Array.unsafe_get in_arr src_lin in
    for d = 0 to ndim - 1 do
      let before, _ = padding.(d) in
      md_index.(d) <- md_index.(d) + before
    done;
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set out_arr dst_lin v
  done

let pad_int32 (in_arr : int32# array) (out_arr : int32# array) in_shape padding
    in_offset out_offset in_strides out_strides in_numel =
  let ndim = Array.length in_shape in
  let md_index = Array.make ndim 0 in
  for k = 0 to in_numel - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let v = Array.unsafe_get in_arr src_lin in
    for d = 0 to ndim - 1 do
      let before, _ = padding.(d) in
      md_index.(d) <- md_index.(d) + before
    done;
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set out_arr dst_lin v
  done

let pad_int64 (in_arr : int64# array) (out_arr : int64# array) in_shape padding
    in_offset out_offset in_strides out_strides in_numel =
  let ndim = Array.length in_shape in
  let md_index = Array.make ndim 0 in
  for k = 0 to in_numel - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let v = Array.unsafe_get in_arr src_lin in
    for d = 0 to ndim - 1 do
      let before, _ = padding.(d) in
      md_index.(d) <- md_index.(d) + before
    done;
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set out_arr dst_lin v
  done

let pad_bool (in_arr : bool array) (out_arr : bool array) in_shape padding
    in_offset out_offset in_strides out_strides in_numel =
  let ndim = Array.length in_shape in
  let md_index = Array.make ndim 0 in
  for k = 0 to in_numel - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let v = Array.unsafe_get in_arr src_lin in
    for d = 0 to ndim - 1 do
      let before, _ = padding.(d) in
      md_index.(d) <- md_index.(d) + before
    done;
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set out_arr dst_lin v
  done
