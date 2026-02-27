(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let init_state ishape idx_strides src_strides out_strides dshape axis start_idx
    idx_offset src_offset out_offset =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  if start_idx <> 0 then Shape.unravel_index_into start_idx ishape md_index;
  let idx_lin = ref (idx_offset + Shape.ravel_index md_index idx_strides) in
  let src_lin = ref (src_offset + Shape.ravel_index md_index src_strides) in
  let dst_base = ref out_offset in
  for d = 0 to rank - 1 do
    if d <> axis then
      dst_base :=
        !dst_base + (Array.unsafe_get md_index d * Array.unsafe_get out_strides d)
  done;
  (md_index, idx_lin, src_lin, dst_base)

let advance_state md_index ishape idx_strides src_strides out_strides axis idx_lin
    src_lin dst_base =
  let d = ref (Array.length ishape - 1) in
  while !d >= 0 do
    let dim = !d in
    let cur = Array.unsafe_get md_index dim in
    let next = cur + 1 in
    if next < Array.unsafe_get ishape dim then (
      Array.unsafe_set md_index dim next;
      idx_lin := !idx_lin + Array.unsafe_get idx_strides dim;
      src_lin := !src_lin + Array.unsafe_get src_strides dim;
      if dim <> axis then
        dst_base := !dst_base + Array.unsafe_get out_strides dim;
      d := -1)
    else (
      Array.unsafe_set md_index dim 0;
      idx_lin := !idx_lin - (cur * Array.unsafe_get idx_strides dim);
      src_lin := !src_lin - (cur * Array.unsafe_get src_strides dim);
      if dim <> axis then
        dst_base := !dst_base - (cur * Array.unsafe_get out_strides dim);
      d := dim - 1)
  done

let scatter_float64 mode (src : float# array) (dst : float# array) ishape dshape
    axis (idx_arr : int32# array) src_offset src_strides idx_offset idx_strides
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else
    let axis_stride = Array.unsafe_get out_strides axis in
    let md_index, idx_lin, src_lin, dst_base =
      init_state ishape idx_strides src_strides out_strides dshape axis start_idx
        idx_offset src_offset out_offset
    in
    let step () =
      let idx =
        Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
      in
      if idx < 0 || idx >= Array.unsafe_get dshape axis then
        invalid_arg "scatter: index out of bounds";
      let dst_lin = !dst_base + (idx * axis_stride) in
      let v = Array.unsafe_get src !src_lin in
      (match mode with
      | `Set -> Array.unsafe_set dst dst_lin v
      | `Add ->
          Array.unsafe_set dst dst_lin
            (Float_u.add (Array.unsafe_get dst dst_lin) v))
    in
    let i = ref start_idx in
    let n4 = end_idx - 3 in
    while !i < n4 do
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      i := !i + 4;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done;
    while !i < end_idx do
      step ();
      incr i;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done

let scatter_float32 mode (src : float32# array) (dst : float32# array) ishape
    dshape axis (idx_arr : int32# array) src_offset src_strides idx_offset
    idx_strides out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else
    let axis_stride = Array.unsafe_get out_strides axis in
    let md_index, idx_lin, src_lin, dst_base =
      init_state ishape idx_strides src_strides out_strides dshape axis start_idx
        idx_offset src_offset out_offset
    in
    let step () =
      let idx =
        Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
      in
      if idx < 0 || idx >= Array.unsafe_get dshape axis then
        invalid_arg "scatter: index out of bounds";
      let dst_lin = !dst_base + (idx * axis_stride) in
      let v = Array.unsafe_get src !src_lin in
      (match mode with
      | `Set -> Array.unsafe_set dst dst_lin v
      | `Add ->
          Array.unsafe_set dst dst_lin
            (Float32_u.add (Array.unsafe_get dst dst_lin) v))
    in
    let i = ref start_idx in
    let n4 = end_idx - 3 in
    while !i < n4 do
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      i := !i + 4;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done;
    while !i < end_idx do
      step ();
      incr i;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done

let scatter_int8 mode (src : int8# array) (dst : int8# array) ishape dshape axis
    (idx_arr : int32# array) src_offset src_strides idx_offset idx_strides
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else
    let axis_stride = Array.unsafe_get out_strides axis in
    let md_index, idx_lin, src_lin, dst_base =
      init_state ishape idx_strides src_strides out_strides dshape axis start_idx
        idx_offset src_offset out_offset
    in
    let step () =
      let idx =
        Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
      in
      if idx < 0 || idx >= Array.unsafe_get dshape axis then
        invalid_arg "scatter: index out of bounds";
      let dst_lin = !dst_base + (idx * axis_stride) in
      let v = Array.unsafe_get src !src_lin in
      (match mode with
      | `Set -> Array.unsafe_set dst dst_lin v
      | `Add ->
          Array.unsafe_set dst dst_lin
            (Int8_u.add (Array.unsafe_get dst dst_lin) v))
    in
    let i = ref start_idx in
    let n4 = end_idx - 3 in
    while !i < n4 do
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      i := !i + 4;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done;
    while !i < end_idx do
      step ();
      incr i;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done

let scatter_int16 mode (src : int16# array) (dst : int16# array) ishape dshape
    axis (idx_arr : int32# array) src_offset src_strides idx_offset idx_strides
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else
    let axis_stride = Array.unsafe_get out_strides axis in
    let md_index, idx_lin, src_lin, dst_base =
      init_state ishape idx_strides src_strides out_strides dshape axis start_idx
        idx_offset src_offset out_offset
    in
    let step () =
      let idx =
        Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
      in
      if idx < 0 || idx >= Array.unsafe_get dshape axis then
        invalid_arg "scatter: index out of bounds";
      let dst_lin = !dst_base + (idx * axis_stride) in
      let v = Array.unsafe_get src !src_lin in
      (match mode with
      | `Set -> Array.unsafe_set dst dst_lin v
      | `Add ->
          Array.unsafe_set dst dst_lin
            (Int16_u.add (Array.unsafe_get dst dst_lin) v))
    in
    let i = ref start_idx in
    let n4 = end_idx - 3 in
    while !i < n4 do
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      i := !i + 4;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done;
    while !i < end_idx do
      step ();
      incr i;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done

let scatter_int32 mode (src : int32# array) (dst : int32# array) ishape dshape
    axis (idx_arr : int32# array) src_offset src_strides idx_offset idx_strides
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else
    let axis_stride = Array.unsafe_get out_strides axis in
    let md_index, idx_lin, src_lin, dst_base =
      init_state ishape idx_strides src_strides out_strides dshape axis start_idx
        idx_offset src_offset out_offset
    in
    let step () =
      let idx =
        Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
      in
      if idx < 0 || idx >= Array.unsafe_get dshape axis then
        invalid_arg "scatter: index out of bounds";
      let dst_lin = !dst_base + (idx * axis_stride) in
      let v = Array.unsafe_get src !src_lin in
      (match mode with
      | `Set -> Array.unsafe_set dst dst_lin v
      | `Add ->
          Array.unsafe_set dst dst_lin
            (Int32_u.add (Array.unsafe_get dst dst_lin) v))
    in
    let i = ref start_idx in
    let n4 = end_idx - 3 in
    while !i < n4 do
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      i := !i + 4;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done;
    while !i < end_idx do
      step ();
      incr i;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done

let scatter_int64 mode (src : int64# array) (dst : int64# array) ishape dshape
    axis (idx_arr : int32# array) src_offset src_strides idx_offset idx_strides
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else
    let axis_stride = Array.unsafe_get out_strides axis in
    let md_index, idx_lin, src_lin, dst_base =
      init_state ishape idx_strides src_strides out_strides dshape axis start_idx
        idx_offset src_offset out_offset
    in
    let step () =
      let idx =
        Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
      in
      if idx < 0 || idx >= Array.unsafe_get dshape axis then
        invalid_arg "scatter: index out of bounds";
      let dst_lin = !dst_base + (idx * axis_stride) in
      let v = Array.unsafe_get src !src_lin in
      (match mode with
      | `Set -> Array.unsafe_set dst dst_lin v
      | `Add ->
          Array.unsafe_set dst dst_lin
            (Int64_u.add (Array.unsafe_get dst dst_lin) v))
    in
    let i = ref start_idx in
    let n4 = end_idx - 3 in
    while !i < n4 do
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      i := !i + 4;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done;
    while !i < end_idx do
      step ();
      incr i;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done

let scatter_bool mode (src : bool array) (dst : bool array) ishape dshape axis
    (idx_arr : int32# array) src_offset src_strides idx_offset idx_strides
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else
    let axis_stride = Array.unsafe_get out_strides axis in
    let md_index, idx_lin, src_lin, dst_base =
      init_state ishape idx_strides src_strides out_strides dshape axis start_idx
        idx_offset src_offset out_offset
    in
    let step () =
      let idx =
        Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
      in
      if idx < 0 || idx >= Array.unsafe_get dshape axis then
        invalid_arg "scatter: index out of bounds";
      let dst_lin = !dst_base + (idx * axis_stride) in
      let v = Array.unsafe_get src !src_lin in
      (match mode with
      | `Set -> Array.unsafe_set dst dst_lin v
      | `Add -> Array.unsafe_set dst dst_lin (Array.unsafe_get dst dst_lin || v))
    in
    let i = ref start_idx in
    let n4 = end_idx - 3 in
    while !i < n4 do
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      advance_state md_index ishape idx_strides src_strides out_strides axis
        idx_lin src_lin dst_base;
      step ();
      i := !i + 4;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done;
    while !i < end_idx do
      step ();
      incr i;
      if !i < end_idx then
        advance_state md_index ishape idx_strides src_strides out_strides axis
          idx_lin src_lin dst_base
    done
