(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let cat_float64 (srcs : (float# array * View.t) list) (dst : float# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let axis_base = ref 0 in
  List.iter
    (fun (src, view) ->
      let in_shape = shape view in
      let in_offset = View.offset view in
      let in_strides = View.strides view in
      let n = numel view in
      let md_index = Array.make rank 0 in
      let dst_index = Array.make rank 0 in
      for k = 0 to n - 1 do
        Shape.unravel_index_into k in_shape md_index;
        Array.blit md_index 0 dst_index 0 rank;
        dst_index.(axis) <- dst_index.(axis) + !axis_base;
        let src_lin = in_offset + Shape.ravel_index md_index in_strides in
        let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
        Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
      done;
      axis_base := !axis_base + in_shape.(axis))
    srcs

let cat_float32 (srcs : (float32# array * View.t) list) (dst : float32# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let axis_base = ref 0 in
  List.iter
    (fun (src, view) ->
      let in_shape = shape view in
      let in_offset = View.offset view in
      let in_strides = View.strides view in
      let n = numel view in
      let md_index = Array.make rank 0 in
      let dst_index = Array.make rank 0 in
      for k = 0 to n - 1 do
        Shape.unravel_index_into k in_shape md_index;
        Array.blit md_index 0 dst_index 0 rank;
        dst_index.(axis) <- dst_index.(axis) + !axis_base;
        let src_lin = in_offset + Shape.ravel_index md_index in_strides in
        let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
        Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
      done;
      axis_base := !axis_base + in_shape.(axis))
    srcs

let cat_int8 (srcs : (int8# array * View.t) list) (dst : int8# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let axis_base = ref 0 in
  List.iter
    (fun (src, view) ->
      let in_shape = shape view in
      let in_offset = View.offset view in
      let in_strides = View.strides view in
      let n = numel view in
      let md_index = Array.make rank 0 in
      let dst_index = Array.make rank 0 in
      for k = 0 to n - 1 do
        Shape.unravel_index_into k in_shape md_index;
        Array.blit md_index 0 dst_index 0 rank;
        dst_index.(axis) <- dst_index.(axis) + !axis_base;
        let src_lin = in_offset + Shape.ravel_index md_index in_strides in
        let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
        Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
      done;
      axis_base := !axis_base + in_shape.(axis))
    srcs

let cat_int16 (srcs : (int16# array * View.t) list) (dst : int16# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let axis_base = ref 0 in
  List.iter
    (fun (src, view) ->
      let in_shape = shape view in
      let in_offset = View.offset view in
      let in_strides = View.strides view in
      let n = numel view in
      let md_index = Array.make rank 0 in
      let dst_index = Array.make rank 0 in
      for k = 0 to n - 1 do
        Shape.unravel_index_into k in_shape md_index;
        Array.blit md_index 0 dst_index 0 rank;
        dst_index.(axis) <- dst_index.(axis) + !axis_base;
        let src_lin = in_offset + Shape.ravel_index md_index in_strides in
        let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
        Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
      done;
      axis_base := !axis_base + in_shape.(axis))
    srcs

let cat_int32 (srcs : (int32# array * View.t) list) (dst : int32# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let axis_base = ref 0 in
  List.iter
    (fun (src, view) ->
      let in_shape = shape view in
      let in_offset = View.offset view in
      let in_strides = View.strides view in
      let n = numel view in
      let md_index = Array.make rank 0 in
      let dst_index = Array.make rank 0 in
      for k = 0 to n - 1 do
        Shape.unravel_index_into k in_shape md_index;
        Array.blit md_index 0 dst_index 0 rank;
        dst_index.(axis) <- dst_index.(axis) + !axis_base;
        let src_lin = in_offset + Shape.ravel_index md_index in_strides in
        let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
        Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
      done;
      axis_base := !axis_base + in_shape.(axis))
    srcs

let cat_int64 (srcs : (int64# array * View.t) list) (dst : int64# array)
    (rank : int) (axis : int) (out_offset : int) (out_strides : int array) =
  let axis_base = ref 0 in
  List.iter
    (fun (src, view) ->
      let in_shape = shape view in
      let in_offset = View.offset view in
      let in_strides = View.strides view in
      let n = numel view in
      let md_index = Array.make rank 0 in
      let dst_index = Array.make rank 0 in
      for k = 0 to n - 1 do
        Shape.unravel_index_into k in_shape md_index;
        Array.blit md_index 0 dst_index 0 rank;
        dst_index.(axis) <- dst_index.(axis) + !axis_base;
        let src_lin = in_offset + Shape.ravel_index md_index in_strides in
        let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
        Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
      done;
      axis_base := !axis_base + in_shape.(axis))
    srcs

let cat_bool (srcs : (bool array * View.t) list) (dst : bool array) (rank : int)
    (axis : int) (out_offset : int) (out_strides : int array) =
  let axis_base = ref 0 in
  List.iter
    (fun (src, view) ->
      let in_shape = shape view in
      let in_offset = View.offset view in
      let in_strides = View.strides view in
      let n = numel view in
      let md_index = Array.make rank 0 in
      let dst_index = Array.make rank 0 in
      for k = 0 to n - 1 do
        Shape.unravel_index_into k in_shape md_index;
        Array.blit md_index 0 dst_index 0 rank;
        dst_index.(axis) <- dst_index.(axis) + !axis_base;
        let src_lin = in_offset + Shape.ravel_index md_index in_strides in
        let dst_lin = out_offset + Shape.ravel_index dst_index out_strides in
        Array.unsafe_set dst dst_lin (Array.unsafe_get src src_lin)
      done;
      axis_base := !axis_base + in_shape.(axis))
    srcs
