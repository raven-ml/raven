(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let assign_float64 (dst_arr : float# array) (src_arr : float# array) n dst_shape
    dst_offset dst_strides src_offset src_strides =
  let md_index = Array.make (Array.length dst_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k dst_shape md_index;
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let dst_lin = dst_offset + Shape.ravel_index md_index dst_strides in
    Array.unsafe_set dst_arr dst_lin (Array.unsafe_get src_arr src_lin)
  done

let assign_float32 (dst_arr : float32# array) (src_arr : float32# array) n
    dst_shape dst_offset dst_strides src_offset src_strides =
  let md_index = Array.make (Array.length dst_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k dst_shape md_index;
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let dst_lin = dst_offset + Shape.ravel_index md_index dst_strides in
    Array.unsafe_set dst_arr dst_lin (Array.unsafe_get src_arr src_lin)
  done

let assign_int8 (dst_arr : int8# array) (src_arr : int8# array) n dst_shape
    dst_offset dst_strides src_offset src_strides =
  let md_index = Array.make (Array.length dst_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k dst_shape md_index;
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let dst_lin = dst_offset + Shape.ravel_index md_index dst_strides in
    Array.unsafe_set dst_arr dst_lin (Array.unsafe_get src_arr src_lin)
  done

let assign_int16 (dst_arr : int16# array) (src_arr : int16# array) n dst_shape
    dst_offset dst_strides src_offset src_strides =
  let md_index = Array.make (Array.length dst_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k dst_shape md_index;
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let dst_lin = dst_offset + Shape.ravel_index md_index dst_strides in
    Array.unsafe_set dst_arr dst_lin (Array.unsafe_get src_arr src_lin)
  done

let assign_int32 (dst_arr : int32# array) (src_arr : int32# array) n dst_shape
    dst_offset dst_strides src_offset src_strides =
  let md_index = Array.make (Array.length dst_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k dst_shape md_index;
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let dst_lin = dst_offset + Shape.ravel_index md_index dst_strides in
    Array.unsafe_set dst_arr dst_lin (Array.unsafe_get src_arr src_lin)
  done

let assign_int64 (dst_arr : int64# array) (src_arr : int64# array) n dst_shape
    dst_offset dst_strides src_offset src_strides =
  let md_index = Array.make (Array.length dst_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k dst_shape md_index;
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let dst_lin = dst_offset + Shape.ravel_index md_index dst_strides in
    Array.unsafe_set dst_arr dst_lin (Array.unsafe_get src_arr src_lin)
  done

let assign_bool (dst_arr : bool array) (src_arr : bool array) n dst_shape
    dst_offset dst_strides src_offset src_strides =
  let md_index = Array.make (Array.length dst_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k dst_shape md_index;
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let dst_lin = dst_offset + Shape.ravel_index md_index dst_strides in
    Array.unsafe_set dst_arr dst_lin (Array.unsafe_get src_arr src_lin)
  done
