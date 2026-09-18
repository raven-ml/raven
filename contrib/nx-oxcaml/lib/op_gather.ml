(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let init_state ishape idx_str out_str dshape data_strides axis start_idx
    idx_offset out_offset data_offset =
  let rank = Array.length dshape in
  let md_index = Array.make rank 0 in
  if start_idx <> 0 then Shape.unravel_index_into start_idx ishape md_index;
  let idx_lin = ref (idx_offset + Shape.ravel_index md_index idx_str) in
  let out_lin = ref (out_offset + Shape.ravel_index md_index out_str) in
  let src_base = ref data_offset in
  for d = 0 to rank - 1 do
    if d <> axis then
      src_base :=
        !src_base
        + (Array.unsafe_get md_index d * Array.unsafe_get data_strides d)
  done;
  (md_index, idx_lin, out_lin, src_base)

let advance_state md_index ishape idx_str out_str data_strides axis idx_lin
    out_lin src_base =
  let d = ref (Array.length ishape - 1) in
  while !d >= 0 do
    let dim = !d in
    let cur = Array.unsafe_get md_index dim in
    let next = cur + 1 in
    if next < Array.unsafe_get ishape dim then (
      Array.unsafe_set md_index dim next;
      idx_lin := !idx_lin + Array.unsafe_get idx_str dim;
      out_lin := !out_lin + Array.unsafe_get out_str dim;
      if dim <> axis then
        src_base := !src_base + Array.unsafe_get data_strides dim;
      d := -1)
    else (
      Array.unsafe_set md_index dim 0;
      idx_lin := !idx_lin - (cur * Array.unsafe_get idx_str dim);
      out_lin := !out_lin - (cur * Array.unsafe_get out_str dim);
      if dim <> axis then
        src_base := !src_base - (cur * Array.unsafe_get data_strides dim);
      d := dim - 1)
  done

let gather_float64 (src : float# array) (dst : float# array) ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else (
    let rank = Array.length dshape in
    let axis_stride = Array.unsafe_get data_strides axis in
    if
      rank = 1 && axis = 0
      && Array.unsafe_get data_strides 0 = 1
      && Array.unsafe_get idx_str 0 = 1
      && Array.unsafe_get out_strides 0 = 1
    then (
      let i = ref start_idx in
      let n2 = end_idx - 1 in
      while !i < n2 do
        let k0 = !i in
        let k1 = k0 + 1 in
        let idx0 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k0)))
        in
        let idx1 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k1)))
        in
        if idx0 < 0 || idx0 >= Array.unsafe_get dshape 0 || idx1 < 0
           || idx1 >= Array.unsafe_get dshape 0
        then invalid_arg "gather: index out of bounds";
        let v0 = Array.unsafe_get src (data_offset + idx0) in
        let v1 = Array.unsafe_get src (data_offset + idx1) in
        let vec = Float64x2.set v0 v1 in
        Float64x2.Array.unsafe_set dst ~idx:(out_offset + k0) vec;
        i := k0 + 2
      done;
      while !i < end_idx do
        let k = !i in
        let idx =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k)))
        in
        if idx < 0 || idx >= Array.unsafe_get dshape 0 then
          invalid_arg "gather: index out of bounds";
        Array.unsafe_set dst (out_offset + k) (Array.unsafe_get src (data_offset + idx));
        incr i
      done)
    else
      let md_index, idx_lin, out_lin, src_base =
        init_state ishape idx_str out_strides dshape data_strides axis start_idx
          idx_offset out_offset data_offset
      in
      for k = start_idx to end_idx - 1 do
        let idx =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
        in
        if idx < 0 || idx >= Array.unsafe_get dshape axis then
          invalid_arg "gather: index out of bounds";
        let src_lin = !src_base + (idx * axis_stride) in
        Array.unsafe_set dst !out_lin (Array.unsafe_get src src_lin);
        if k + 1 < end_idx then
          advance_state md_index ishape idx_str out_strides data_strides axis
            idx_lin out_lin src_base
      done)

let gather_float32 (src : float32# array) (dst : float32# array) ishape dshape
    axis (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else (
    let rank = Array.length dshape in
    let axis_stride = Array.unsafe_get data_strides axis in
    if
      rank = 1 && axis = 0
      && Array.unsafe_get data_strides 0 = 1
      && Array.unsafe_get idx_str 0 = 1
      && Array.unsafe_get out_strides 0 = 1
    then (
      let i = ref start_idx in
      let n4 = end_idx - 3 in
      while !i < n4 do
        let k0 = !i in
        let k1 = k0 + 1 in
        let k2 = k0 + 2 in
        let k3 = k0 + 3 in
        let idx0 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k0)))
        in
        let idx1 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k1)))
        in
        let idx2 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k2)))
        in
        let idx3 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k3)))
        in
        if idx0 < 0 || idx0 >= Array.unsafe_get dshape 0 || idx1 < 0
           || idx1 >= Array.unsafe_get dshape 0 || idx2 < 0
           || idx2 >= Array.unsafe_get dshape 0 || idx3 < 0
           || idx3 >= Array.unsafe_get dshape 0
        then invalid_arg "gather: index out of bounds";
        let v0 = Array.unsafe_get src (data_offset + idx0) in
        let v1 = Array.unsafe_get src (data_offset + idx1) in
        let v2 = Array.unsafe_get src (data_offset + idx2) in
        let v3 = Array.unsafe_get src (data_offset + idx3) in
        let vec = Float32x4.set v0 v1 v2 v3 in
        Float32x4.Array.unsafe_set dst ~idx:(out_offset + k0) vec;
        i := k0 + 4
      done;
      while !i < end_idx do
        let k = !i in
        let idx =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k)))
        in
        if idx < 0 || idx >= Array.unsafe_get dshape 0 then
          invalid_arg "gather: index out of bounds";
        Array.unsafe_set dst (out_offset + k) (Array.unsafe_get src (data_offset + idx));
        incr i
      done)
    else
      let md_index, idx_lin, out_lin, src_base =
        init_state ishape idx_str out_strides dshape data_strides axis start_idx
          idx_offset out_offset data_offset
      in
      for k = start_idx to end_idx - 1 do
        let idx =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
        in
        if idx < 0 || idx >= Array.unsafe_get dshape axis then
          invalid_arg "gather: index out of bounds";
        let src_lin = !src_base + (idx * axis_stride) in
        Array.unsafe_set dst !out_lin (Array.unsafe_get src src_lin);
        if k + 1 < end_idx then
          advance_state md_index ishape idx_str out_strides data_strides axis
            idx_lin out_lin src_base
      done)

let gather_int8 (src : int8# array) (dst : int8# array) ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else
    let axis_stride = Array.unsafe_get data_strides axis in
    let md_index, idx_lin, out_lin, src_base =
      init_state ishape idx_str out_strides dshape data_strides axis start_idx
        idx_offset out_offset data_offset
    in
    for k = start_idx to end_idx - 1 do
      let idx =
        Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
      in
      if idx < 0 || idx >= Array.unsafe_get dshape axis then
        invalid_arg "gather: index out of bounds";
      let src_lin = !src_base + (idx * axis_stride) in
      Array.unsafe_set dst !out_lin (Array.unsafe_get src src_lin);
      if k + 1 < end_idx then
        advance_state md_index ishape idx_str out_strides data_strides axis
          idx_lin out_lin src_base
    done

let gather_int16 (src : int16# array) (dst : int16# array) ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else
    let axis_stride = Array.unsafe_get data_strides axis in
    let md_index, idx_lin, out_lin, src_base =
      init_state ishape idx_str out_strides dshape data_strides axis start_idx
        idx_offset out_offset data_offset
    in
    for k = start_idx to end_idx - 1 do
      let idx =
        Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
      in
      if idx < 0 || idx >= Array.unsafe_get dshape axis then
        invalid_arg "gather: index out of bounds";
      let src_lin = !src_base + (idx * axis_stride) in
      Array.unsafe_set dst !out_lin (Array.unsafe_get src src_lin);
      if k + 1 < end_idx then
        advance_state md_index ishape idx_str out_strides data_strides axis
          idx_lin out_lin src_base
    done

let gather_int32 (src : int32# array) (dst : int32# array) ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else (
    let rank = Array.length dshape in
    let axis_stride = Array.unsafe_get data_strides axis in
    if
      rank = 1 && axis = 0
      && Array.unsafe_get data_strides 0 = 1
      && Array.unsafe_get idx_str 0 = 1
      && Array.unsafe_get out_strides 0 = 1
    then (
      let i = ref start_idx in
      let n4 = end_idx - 3 in
      while !i < n4 do
        let k0 = !i in
        let k1 = k0 + 1 in
        let k2 = k0 + 2 in
        let k3 = k0 + 3 in
        let idx0 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k0)))
        in
        let idx1 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k1)))
        in
        let idx2 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k2)))
        in
        let idx3 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k3)))
        in
        if idx0 < 0 || idx0 >= Array.unsafe_get dshape 0 || idx1 < 0
           || idx1 >= Array.unsafe_get dshape 0 || idx2 < 0
           || idx2 >= Array.unsafe_get dshape 0 || idx3 < 0
           || idx3 >= Array.unsafe_get dshape 0
        then invalid_arg "gather: index out of bounds";
        let v0 = Array.unsafe_get src (data_offset + idx0) in
        let v1 = Array.unsafe_get src (data_offset + idx1) in
        let v2 = Array.unsafe_get src (data_offset + idx2) in
        let v3 = Array.unsafe_get src (data_offset + idx3) in
        let vec = Int32x4.set v0 v1 v2 v3 in
        Int32x4.Array.unsafe_set dst ~idx:(out_offset + k0) vec;
        i := k0 + 4
      done;
      while !i < end_idx do
        let k = !i in
        let idx =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k)))
        in
        if idx < 0 || idx >= Array.unsafe_get dshape 0 then
          invalid_arg "gather: index out of bounds";
        Array.unsafe_set dst (out_offset + k) (Array.unsafe_get src (data_offset + idx));
        incr i
      done)
    else
      let md_index, idx_lin, out_lin, src_base =
        init_state ishape idx_str out_strides dshape data_strides axis start_idx
          idx_offset out_offset data_offset
      in
      for k = start_idx to end_idx - 1 do
        let idx =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
        in
        if idx < 0 || idx >= Array.unsafe_get dshape axis then
          invalid_arg "gather: index out of bounds";
        let src_lin = !src_base + (idx * axis_stride) in
        Array.unsafe_set dst !out_lin (Array.unsafe_get src src_lin);
        if k + 1 < end_idx then
          advance_state md_index ishape idx_str out_strides data_strides axis
            idx_lin out_lin src_base
      done)

let gather_int64 (src : int64# array) (dst : int64# array) ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else (
    let rank = Array.length dshape in
    let axis_stride = Array.unsafe_get data_strides axis in
    if
      rank = 1 && axis = 0
      && Array.unsafe_get data_strides 0 = 1
      && Array.unsafe_get idx_str 0 = 1
      && Array.unsafe_get out_strides 0 = 1
    then (
      let i = ref start_idx in
      let n2 = end_idx - 1 in
      while !i < n2 do
        let k0 = !i in
        let k1 = k0 + 1 in
        let idx0 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k0)))
        in
        let idx1 =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k1)))
        in
        if idx0 < 0 || idx0 >= Array.unsafe_get dshape 0 || idx1 < 0
           || idx1 >= Array.unsafe_get dshape 0
        then invalid_arg "gather: index out of bounds";
        let v0 = Array.unsafe_get src (data_offset + idx0) in
        let v1 = Array.unsafe_get src (data_offset + idx1) in
        let vec = Int64x2.set v0 v1 in
        Int64x2.Array.unsafe_set dst ~idx:(out_offset + k0) vec;
        i := k0 + 2
      done;
      while !i < end_idx do
        let k = !i in
        let idx =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr (idx_offset + k)))
        in
        if idx < 0 || idx >= Array.unsafe_get dshape 0 then
          invalid_arg "gather: index out of bounds";
        Array.unsafe_set dst (out_offset + k) (Array.unsafe_get src (data_offset + idx));
        incr i
      done)
    else
      let md_index, idx_lin, out_lin, src_base =
        init_state ishape idx_str out_strides dshape data_strides axis start_idx
          idx_offset out_offset data_offset
      in
      for k = start_idx to end_idx - 1 do
        let idx =
          Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
        in
        if idx < 0 || idx >= Array.unsafe_get dshape axis then
          invalid_arg "gather: index out of bounds";
        let src_lin = !src_base + (idx * axis_stride) in
        Array.unsafe_set dst !out_lin (Array.unsafe_get src src_lin);
        if k + 1 < end_idx then
          advance_state md_index ishape idx_str out_strides data_strides axis
            idx_lin out_lin src_base
      done)

let gather_bool (src : bool array) (dst : bool array) ishape dshape axis
    (idx_arr : int32# array) data_offset data_strides idx_offset idx_str
    out_offset out_strides start_idx end_idx =
  if start_idx >= end_idx then ()
  else
    let axis_stride = Array.unsafe_get data_strides axis in
    let md_index, idx_lin, out_lin, src_base =
      init_state ishape idx_str out_strides dshape data_strides axis start_idx
        idx_offset out_offset data_offset
    in
    for k = start_idx to end_idx - 1 do
      let idx =
        Int32.to_int (Int32_u.to_int32 (Array.unsafe_get idx_arr !idx_lin))
      in
      if idx < 0 || idx >= Array.unsafe_get dshape axis then
        invalid_arg "gather: index out of bounds";
      let src_lin = !src_base + (idx * axis_stride) in
      Array.unsafe_set dst !out_lin (Array.unsafe_get src src_lin);
      if k + 1 < end_idx then
        advance_state md_index ishape idx_str out_strides data_strides axis
          idx_lin out_lin src_base
    done
