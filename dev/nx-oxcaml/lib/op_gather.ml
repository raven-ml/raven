(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let gather_float64 (src : float# array) (dst : float# array) n ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let src_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_str in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_gather" ~what:"index out of bounds" ();
    Array.blit md_index 0 src_index 0 rank;
    src_index.(axis) <- idx;
    let src_lin = data_offset + Shape.ravel_index src_index data_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let gather_float32 (src : float32# array) (dst : float32# array) n ishape dshape
    axis (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let src_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_str in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_gather" ~what:"index out of bounds" ();
    Array.blit md_index 0 src_index 0 rank;
    src_index.(axis) <- idx;
    let src_lin = data_offset + Shape.ravel_index src_index data_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let gather_int8 (src : int8# array) (dst : int8# array) n ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let src_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_str in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_gather" ~what:"index out of bounds" ();
    Array.blit md_index 0 src_index 0 rank;
    src_index.(axis) <- idx;
    let src_lin = data_offset + Shape.ravel_index src_index data_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let gather_int16 (src : int16# array) (dst : int16# array) n ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let src_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_str in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_gather" ~what:"index out of bounds" ();
    Array.blit md_index 0 src_index 0 rank;
    src_index.(axis) <- idx;
    let src_lin = data_offset + Shape.ravel_index src_index data_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let gather_int32 (src : int32# array) (dst : int32# array) n ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let src_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_str in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_gather" ~what:"index out of bounds" ();
    Array.blit md_index 0 src_index 0 rank;
    src_index.(axis) <- idx;
    let src_lin = data_offset + Shape.ravel_index src_index data_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let gather_int64 (src : int64# array) (dst : int64# array) n ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let src_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_str in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_gather" ~what:"index out of bounds" ();
    Array.blit md_index 0 src_index 0 rank;
    src_index.(axis) <- idx;
    let src_lin = data_offset + Shape.ravel_index src_index data_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done

let gather_bool (src : bool array) (dst : bool array) n ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let src_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_str in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_gather" ~what:"index out of bounds" ();
    Array.blit md_index 0 src_index 0 rank;
    src_index.(axis) <- idx;
    let src_lin = data_offset + Shape.ravel_index src_index data_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
  done
