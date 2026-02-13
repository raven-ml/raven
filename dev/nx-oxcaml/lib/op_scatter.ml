(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let scatter_float64 mode (src : float# array) (dst : float# array) n ishape
    dshape axis (idx_arr : int32# array) src_offset src_strides idx_offset
    idx_strides out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let dst_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_strides in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_scatter" ~what:"index out of bounds" ();
    Array.blit md_index 0 dst_index 0 rank;
    dst_index.(axis) <- idx;
    let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let v = Array.unsafe_get src src_lin in
    match mode with
    | `Set -> Array.unsafe_set dst dst_lin v
    | `Add -> Array.unsafe_set dst dst_lin (Float_u.add (Array.unsafe_get dst dst_lin) v)
  done

let scatter_float32 mode (src : float32# array) (dst : float32# array) n ishape
    dshape axis (idx_arr : int32# array) src_offset src_strides idx_offset
    idx_strides out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let dst_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_strides in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_scatter" ~what:"index out of bounds" ();
    Array.blit md_index 0 dst_index 0 rank;
    dst_index.(axis) <- idx;
    let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let v = Array.unsafe_get src src_lin in
    match mode with
    | `Set -> Array.unsafe_set dst dst_lin v
    | `Add -> Array.unsafe_set dst dst_lin (Float32_u.add (Array.unsafe_get dst dst_lin) v)
  done

let scatter_int8 mode (src : int8# array) (dst : int8# array) n ishape dshape
    axis (idx_arr : int32# array) src_offset src_strides idx_offset idx_strides
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let dst_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_strides in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_scatter" ~what:"index out of bounds" ();
    Array.blit md_index 0 dst_index 0 rank;
    dst_index.(axis) <- idx;
    let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let v = Array.unsafe_get src src_lin in
    match mode with
    | `Set -> Array.unsafe_set dst dst_lin v
    | `Add -> Array.unsafe_set dst dst_lin (Int8_u.add (Array.unsafe_get dst dst_lin) v)
  done

let scatter_int16 mode (src : int16# array) (dst : int16# array) n ishape dshape
    axis (idx_arr : int32# array) src_offset src_strides idx_offset idx_strides
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let dst_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_strides in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_scatter" ~what:"index out of bounds" ();
    Array.blit md_index 0 dst_index 0 rank;
    dst_index.(axis) <- idx;
    let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let v = Array.unsafe_get src src_lin in
    match mode with
    | `Set -> Array.unsafe_set dst dst_lin v
    | `Add -> Array.unsafe_set dst dst_lin (Int16_u.add (Array.unsafe_get dst dst_lin) v)
  done

let scatter_int32 mode (src : int32# array) (dst : int32# array) n ishape dshape
    axis (idx_arr : int32# array) src_offset src_strides idx_offset idx_strides
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let dst_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_strides in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_scatter" ~what:"index out of bounds" ();
    Array.blit md_index 0 dst_index 0 rank;
    dst_index.(axis) <- idx;
    let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let v = Array.unsafe_get src src_lin in
    match mode with
    | `Set -> Array.unsafe_set dst dst_lin v
    | `Add -> Array.unsafe_set dst dst_lin (Int32_u.add (Array.unsafe_get dst dst_lin) v)
  done

let scatter_int64 mode (src : int64# array) (dst : int64# array) n ishape dshape
    axis (idx_arr : int32# array) src_offset src_strides idx_offset idx_strides
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let dst_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_strides in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_scatter" ~what:"index out of bounds" ();
    Array.blit md_index 0 dst_index 0 rank;
    dst_index.(axis) <- idx;
    let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let v = Array.unsafe_get src src_lin in
    match mode with
    | `Set -> Array.unsafe_set dst dst_lin v
    | `Add -> Array.unsafe_set dst dst_lin (Int64_u.add (Array.unsafe_get dst dst_lin) v)
  done

let scatter_bool mode (src : bool array) (dst : bool array) n ishape dshape axis
    (idx_arr : int32# array) src_offset src_strides idx_offset idx_strides
    out_offset out_strides =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  let dst_index = Array.make rank 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k ishape md_index;
    let idx_lin = idx_offset + Shape.ravel_index md_index idx_strides in
    let idx = Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr idx_lin)) in
    if idx < 0 || idx >= dshape.(axis) then
      Error.invalid ~op:"op_scatter" ~what:"index out of bounds" ();
    Array.blit md_index 0 dst_index 0 rank;
    dst_index.(axis) <- idx;
    let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
    let src_lin = src_offset + Shape.ravel_index md_index src_strides in
    let v = Array.unsafe_get src src_lin in
    match mode with
    | `Set -> Array.unsafe_set dst dst_lin v
    | `Add -> Array.unsafe_set dst dst_lin (Array.unsafe_get dst dst_lin || v)
  done
