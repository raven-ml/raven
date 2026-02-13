(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let materialize_float64 (src : float# array) (dst : float# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let materialize_float32 (src : float32# array) (dst : float32# array) n ishape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length ishape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let materialize_int8 (src : int8# array) (dst : int8# array) n ishape in_offset
    in_strides out_offset out_strides =
  let md_index = Array.make (Array.length ishape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let materialize_int16 (src : int16# array) (dst : int16# array) n ishape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length ishape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let materialize_int32 (src : int32# array) (dst : int32# array) n ishape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length ishape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let materialize_int64 (src : int64# array) (dst : int64# array) n ishape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length ishape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let materialize_bool (src : bool array) (dst : bool array) n ishape in_offset
    in_strides out_offset out_strides =
  let md_index = Array.make (Array.length ishape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done
