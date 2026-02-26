(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

let parallel_threshold = 64

(* Stable merge sort on indices. Sorts [indices[0..n-1]] by comparing
   values fetched via [get_val]. [tmp] is a pre-allocated scratch buffer. *)
let merge_sort_indices (indices : int array) (tmp : int array) n
    (cmp : int -> int -> int) =
  (* Bottom-up merge sort *)
  let width = ref 1 in
  while !width < n do
    let w = !width in
    let i = ref 0 in
    while !i < n do
      let left = !i in
      let mid = min (left + w) n in
      let right = min (left + 2 * w) n in
      (* Merge [left..mid) and [mid..right) into tmp *)
      let l = ref left in
      let r = ref mid in
      for k = left to right - 1 do
        if !l < mid && (!r >= right || cmp indices.(!l) indices.(!r) <= 0) then (
          tmp.(k) <- indices.(!l);
          incr l)
        else (
          tmp.(k) <- indices.(!r);
          incr r)
      done;
      (* Copy back *)
      Array.blit tmp left indices left (right - left);
      i := !i + 2 * w
    done;
    width := w * 2
  done

(* --- sort --- *)

let sort_float64 pool ~(out_arr : float# array) ~a_arr ~va ~vout ~axis
    ~descending =
  let in_shape = shape va in
  let rank = Array.length in_shape in
  let axis_size = in_shape.(axis) in
  if axis_size <= 1 then (
    (* Nothing to sort, just copy *)
    let n = numel vout in
    let out_offset = View.offset vout in
    let a_offset = View.offset va in
    if View.is_c_contiguous vout && View.is_c_contiguous va then
      for i = 0 to n - 1 do
        Array.unsafe_set out_arr (out_offset + i)
          (Array.unsafe_get a_arr (a_offset + i))
      done
    else
      let out_shape = shape vout in
      let out_strides = View.strides vout in
      let a_strides = View.strides va in
      let md_idx = Array.make rank 0 in
      for i = 0 to n - 1 do
        Shape.unravel_index_into i out_shape md_idx;
        let a_lin = Shape.ravel_index md_idx a_strides in
        let o_lin = Shape.ravel_index md_idx out_strides in
        Array.unsafe_set out_arr (out_offset + o_lin)
          (Array.unsafe_get a_arr (a_offset + a_lin))
      done)
  else
    let outer =
      let p = ref 1 in
      for d = 0 to axis - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let inner =
      let p = ref 1 in
      for d = axis + 1 to rank - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let groups = outer * inner in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_strides = View.strides vout in
    let out_offset = View.offset vout in
    let a_axis_stride = a_strides.(axis) in
    let out_axis_stride = out_strides.(axis) in
    let work_on_group g =
      let o = g / inner in
      let i = g mod inner in
      (* Compute base offset for this lane in input *)
      let a_base =
        let off = ref a_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * a_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * a_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let out_base =
        let off = ref out_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * out_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * out_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let indices = Array.init axis_size Fun.id in
      let tmp = Array.make axis_size 0 in
      let acc = Array.make_float64 2 in
      let cmp a_idx b_idx =
        Array.unsafe_set acc 0
          (Array.unsafe_get a_arr (a_base + (a_idx * a_axis_stride)));
        Array.unsafe_set acc 1
          (Array.unsafe_get a_arr (a_base + (b_idx * a_axis_stride)));
        let c = Float_u.compare (Array.unsafe_get acc 0) (Array.unsafe_get acc 1) in
        if descending then -c else c
      in
      merge_sort_indices indices tmp axis_size cmp;
      for j = 0 to axis_size - 1 do
        let src_idx = indices.(j) in
        Array.unsafe_set out_arr
          (out_base + (j * out_axis_stride))
          (Array.unsafe_get a_arr (a_base + (src_idx * a_axis_stride)))
      done
    in
    if groups > parallel_threshold then
      Parallel.parallel_for pool 0 (groups - 1) (fun s e ->
          for g = s to e - 1 do
            work_on_group g
          done)
    else
      for g = 0 to groups - 1 do
        work_on_group g
      done

let sort_float32 pool ~(out_arr : float32# array) ~a_arr ~va ~vout ~axis
    ~descending =
  let in_shape = shape va in
  let rank = Array.length in_shape in
  let axis_size = in_shape.(axis) in
  if axis_size <= 1 then (
    let n = numel vout in
    let out_offset = View.offset vout in
    let a_offset = View.offset va in
    if View.is_c_contiguous vout && View.is_c_contiguous va then
      for i = 0 to n - 1 do
        Array.unsafe_set out_arr (out_offset + i)
          (Array.unsafe_get a_arr (a_offset + i))
      done
    else
      let out_shape = shape vout in
      let out_strides = View.strides vout in
      let a_strides = View.strides va in
      let md_idx = Array.make rank 0 in
      for i = 0 to n - 1 do
        Shape.unravel_index_into i out_shape md_idx;
        let a_lin = Shape.ravel_index md_idx a_strides in
        let o_lin = Shape.ravel_index md_idx out_strides in
        Array.unsafe_set out_arr (out_offset + o_lin)
          (Array.unsafe_get a_arr (a_offset + a_lin))
      done)
  else
    let outer =
      let p = ref 1 in
      for d = 0 to axis - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let inner =
      let p = ref 1 in
      for d = axis + 1 to rank - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let groups = outer * inner in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_strides = View.strides vout in
    let out_offset = View.offset vout in
    let a_axis_stride = a_strides.(axis) in
    let out_axis_stride = out_strides.(axis) in
    let work_on_group g =
      let o = g / inner in
      let i = g mod inner in
      let a_base =
        let off = ref a_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * a_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * a_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let out_base =
        let off = ref out_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * out_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * out_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let indices = Array.init axis_size Fun.id in
      let tmp = Array.make axis_size 0 in
      let acc = Array.make_float32 2 in
      let cmp a_idx b_idx =
        Array.unsafe_set acc 0
          (Array.unsafe_get a_arr (a_base + (a_idx * a_axis_stride)));
        Array.unsafe_set acc 1
          (Array.unsafe_get a_arr (a_base + (b_idx * a_axis_stride)));
        let c =
          Float32_u.compare (Array.unsafe_get acc 0) (Array.unsafe_get acc 1)
        in
        if descending then -c else c
      in
      merge_sort_indices indices tmp axis_size cmp;
      for j = 0 to axis_size - 1 do
        let src_idx = indices.(j) in
        Array.unsafe_set out_arr
          (out_base + (j * out_axis_stride))
          (Array.unsafe_get a_arr (a_base + (src_idx * a_axis_stride)))
      done
    in
    if groups > parallel_threshold then
      Parallel.parallel_for pool 0 (groups - 1) (fun s e ->
          for g = s to e - 1 do
            work_on_group g
          done)
    else
      for g = 0 to groups - 1 do
        work_on_group g
      done

let sort_int32 pool ~(out_arr : int32# array) ~(a_arr : int32# array) ~va ~vout
    ~axis ~descending =
  let in_shape = shape va in
  let rank = Array.length in_shape in
  let axis_size = in_shape.(axis) in
  if axis_size <= 1 then (
    let n = numel vout in
    let out_offset = View.offset vout in
    let a_offset = View.offset va in
    if View.is_c_contiguous vout && View.is_c_contiguous va then
      for i = 0 to n - 1 do
        Array.unsafe_set out_arr (out_offset + i)
          (Array.unsafe_get a_arr (a_offset + i))
      done
    else
      let out_shape = shape vout in
      let out_strides = View.strides vout in
      let a_strides = View.strides va in
      let md_idx = Array.make rank 0 in
      for i = 0 to n - 1 do
        Shape.unravel_index_into i out_shape md_idx;
        let a_lin = Shape.ravel_index md_idx a_strides in
        let o_lin = Shape.ravel_index md_idx out_strides in
        Array.unsafe_set out_arr (out_offset + o_lin)
          (Array.unsafe_get a_arr (a_offset + a_lin))
      done)
  else
    let outer =
      let p = ref 1 in
      for d = 0 to axis - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let inner =
      let p = ref 1 in
      for d = axis + 1 to rank - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let groups = outer * inner in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_strides = View.strides vout in
    let out_offset = View.offset vout in
    let a_axis_stride = a_strides.(axis) in
    let out_axis_stride = out_strides.(axis) in
    let work_on_group g =
      let o = g / inner in
      let i = g mod inner in
      let a_base =
        let off = ref a_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * a_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * a_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let out_base =
        let off = ref out_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * out_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * out_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let indices = Array.init axis_size Fun.id in
      let tmp = Array.make axis_size 0 in
      let acc = Array.make_int32 2 in
      let cmp a_idx b_idx =
        Array.unsafe_set acc 0
          (Array.unsafe_get a_arr (a_base + (a_idx * a_axis_stride)));
        Array.unsafe_set acc 1
          (Array.unsafe_get a_arr (a_base + (b_idx * a_axis_stride)));
        let c =
          Int32_u.compare (Array.unsafe_get acc 0) (Array.unsafe_get acc 1)
        in
        if descending then -c else c
      in
      merge_sort_indices indices tmp axis_size cmp;
      for j = 0 to axis_size - 1 do
        let src_idx = indices.(j) in
        Array.unsafe_set out_arr
          (out_base + (j * out_axis_stride))
          (Array.unsafe_get a_arr (a_base + (src_idx * a_axis_stride)))
      done
    in
    if groups > parallel_threshold then
      Parallel.parallel_for pool 0 (groups - 1) (fun s e ->
          for g = s to e - 1 do
            work_on_group g
          done)
    else
      for g = 0 to groups - 1 do
        work_on_group g
      done

let sort_int64 pool ~(out_arr : int64# array) ~(a_arr : int64# array) ~va ~vout
    ~axis ~descending =
  let in_shape = shape va in
  let rank = Array.length in_shape in
  let axis_size = in_shape.(axis) in
  if axis_size <= 1 then (
    let n = numel vout in
    let out_offset = View.offset vout in
    let a_offset = View.offset va in
    if View.is_c_contiguous vout && View.is_c_contiguous va then
      for i = 0 to n - 1 do
        Array.unsafe_set out_arr (out_offset + i)
          (Array.unsafe_get a_arr (a_offset + i))
      done
    else
      let out_shape = shape vout in
      let out_strides = View.strides vout in
      let a_strides = View.strides va in
      let md_idx = Array.make rank 0 in
      for i = 0 to n - 1 do
        Shape.unravel_index_into i out_shape md_idx;
        let a_lin = Shape.ravel_index md_idx a_strides in
        let o_lin = Shape.ravel_index md_idx out_strides in
        Array.unsafe_set out_arr (out_offset + o_lin)
          (Array.unsafe_get a_arr (a_offset + a_lin))
      done)
  else
    let outer =
      let p = ref 1 in
      for d = 0 to axis - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let inner =
      let p = ref 1 in
      for d = axis + 1 to rank - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let groups = outer * inner in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_strides = View.strides vout in
    let out_offset = View.offset vout in
    let a_axis_stride = a_strides.(axis) in
    let out_axis_stride = out_strides.(axis) in
    let work_on_group g =
      let o = g / inner in
      let i = g mod inner in
      let a_base =
        let off = ref a_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * a_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * a_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let out_base =
        let off = ref out_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * out_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * out_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let indices = Array.init axis_size Fun.id in
      let tmp = Array.make axis_size 0 in
      let acc = Array.make_int64 2 in
      let cmp a_idx b_idx =
        Array.unsafe_set acc 0
          (Array.unsafe_get a_arr (a_base + (a_idx * a_axis_stride)));
        Array.unsafe_set acc 1
          (Array.unsafe_get a_arr (a_base + (b_idx * a_axis_stride)));
        let c =
          Int64_u.compare (Array.unsafe_get acc 0) (Array.unsafe_get acc 1)
        in
        if descending then -c else c
      in
      merge_sort_indices indices tmp axis_size cmp;
      for j = 0 to axis_size - 1 do
        let src_idx = indices.(j) in
        Array.unsafe_set out_arr
          (out_base + (j * out_axis_stride))
          (Array.unsafe_get a_arr (a_base + (src_idx * a_axis_stride)))
      done
    in
    if groups > parallel_threshold then
      Parallel.parallel_for pool 0 (groups - 1) (fun s e ->
          for g = s to e - 1 do
            work_on_group g
          done)
    else
      for g = 0 to groups - 1 do
        work_on_group g
      done

(* --- argsort --- *)

let argsort_float64 pool ~(out_arr : int32# array) ~a_arr ~va ~vout ~axis
    ~descending =
  let in_shape = shape va in
  let rank = Array.length in_shape in
  let axis_size = in_shape.(axis) in
  if axis_size <= 1 then (
    let n = numel vout in
    let out_offset = View.offset vout in
    for i = 0 to n - 1 do
      Array.unsafe_set out_arr (out_offset + i) (Int32_u.of_int 0)
    done)
  else
    let outer =
      let p = ref 1 in
      for d = 0 to axis - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let inner =
      let p = ref 1 in
      for d = axis + 1 to rank - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let groups = outer * inner in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_strides = View.strides vout in
    let out_offset = View.offset vout in
    let a_axis_stride = a_strides.(axis) in
    let out_axis_stride = out_strides.(axis) in
    let work_on_group g =
      let o = g / inner in
      let i = g mod inner in
      let a_base =
        let off = ref a_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * a_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * a_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let out_base =
        let off = ref out_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * out_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * out_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let indices = Array.init axis_size Fun.id in
      let tmp = Array.make axis_size 0 in
      let acc = Array.make_float64 2 in
      let cmp a_idx b_idx =
        Array.unsafe_set acc 0
          (Array.unsafe_get a_arr (a_base + (a_idx * a_axis_stride)));
        Array.unsafe_set acc 1
          (Array.unsafe_get a_arr (a_base + (b_idx * a_axis_stride)));
        let c =
          Float_u.compare (Array.unsafe_get acc 0) (Array.unsafe_get acc 1)
        in
        if descending then -c else c
      in
      merge_sort_indices indices tmp axis_size cmp;
      for j = 0 to axis_size - 1 do
        Array.unsafe_set out_arr
          (out_base + (j * out_axis_stride))
          (Int32_u.of_int indices.(j))
      done
    in
    if groups > parallel_threshold then
      Parallel.parallel_for pool 0 (groups - 1) (fun s e ->
          for g = s to e - 1 do
            work_on_group g
          done)
    else
      for g = 0 to groups - 1 do
        work_on_group g
      done

let argsort_float32 pool ~(out_arr : int32# array) ~a_arr ~va ~vout ~axis
    ~descending =
  let in_shape = shape va in
  let rank = Array.length in_shape in
  let axis_size = in_shape.(axis) in
  if axis_size <= 1 then (
    let n = numel vout in
    let out_offset = View.offset vout in
    for i = 0 to n - 1 do
      Array.unsafe_set out_arr (out_offset + i) (Int32_u.of_int 0)
    done)
  else
    let outer =
      let p = ref 1 in
      for d = 0 to axis - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let inner =
      let p = ref 1 in
      for d = axis + 1 to rank - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let groups = outer * inner in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_strides = View.strides vout in
    let out_offset = View.offset vout in
    let a_axis_stride = a_strides.(axis) in
    let out_axis_stride = out_strides.(axis) in
    let work_on_group g =
      let o = g / inner in
      let i = g mod inner in
      let a_base =
        let off = ref a_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * a_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * a_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let out_base =
        let off = ref out_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * out_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * out_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let indices = Array.init axis_size Fun.id in
      let tmp = Array.make axis_size 0 in
      let acc = Array.make_float32 2 in
      let cmp a_idx b_idx =
        Array.unsafe_set acc 0
          (Array.unsafe_get a_arr (a_base + (a_idx * a_axis_stride)));
        Array.unsafe_set acc 1
          (Array.unsafe_get a_arr (a_base + (b_idx * a_axis_stride)));
        let c =
          Float32_u.compare (Array.unsafe_get acc 0) (Array.unsafe_get acc 1)
        in
        if descending then -c else c
      in
      merge_sort_indices indices tmp axis_size cmp;
      for j = 0 to axis_size - 1 do
        Array.unsafe_set out_arr
          (out_base + (j * out_axis_stride))
          (Int32_u.of_int indices.(j))
      done
    in
    if groups > parallel_threshold then
      Parallel.parallel_for pool 0 (groups - 1) (fun s e ->
          for g = s to e - 1 do
            work_on_group g
          done)
    else
      for g = 0 to groups - 1 do
        work_on_group g
      done

let argsort_int32 pool ~(out_arr : int32# array) ~(a_arr : int32# array) ~va
    ~vout ~axis ~descending =
  let in_shape = shape va in
  let rank = Array.length in_shape in
  let axis_size = in_shape.(axis) in
  if axis_size <= 1 then (
    let n = numel vout in
    let out_offset = View.offset vout in
    for i = 0 to n - 1 do
      Array.unsafe_set out_arr (out_offset + i) (Int32_u.of_int 0)
    done)
  else
    let outer =
      let p = ref 1 in
      for d = 0 to axis - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let inner =
      let p = ref 1 in
      for d = axis + 1 to rank - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let groups = outer * inner in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_strides = View.strides vout in
    let out_offset = View.offset vout in
    let a_axis_stride = a_strides.(axis) in
    let out_axis_stride = out_strides.(axis) in
    let work_on_group g =
      let o = g / inner in
      let i = g mod inner in
      let a_base =
        let off = ref a_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * a_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * a_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let out_base =
        let off = ref out_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * out_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * out_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let indices = Array.init axis_size Fun.id in
      let tmp = Array.make axis_size 0 in
      let acc = Array.make_int32 2 in
      let cmp a_idx b_idx =
        Array.unsafe_set acc 0
          (Array.unsafe_get a_arr (a_base + (a_idx * a_axis_stride)));
        Array.unsafe_set acc 1
          (Array.unsafe_get a_arr (a_base + (b_idx * a_axis_stride)));
        let c =
          Int32_u.compare (Array.unsafe_get acc 0) (Array.unsafe_get acc 1)
        in
        if descending then -c else c
      in
      merge_sort_indices indices tmp axis_size cmp;
      for j = 0 to axis_size - 1 do
        Array.unsafe_set out_arr
          (out_base + (j * out_axis_stride))
          (Int32_u.of_int indices.(j))
      done
    in
    if groups > parallel_threshold then
      Parallel.parallel_for pool 0 (groups - 1) (fun s e ->
          for g = s to e - 1 do
            work_on_group g
          done)
    else
      for g = 0 to groups - 1 do
        work_on_group g
      done

let argsort_int64 pool ~(out_arr : int32# array) ~(a_arr : int64# array) ~va
    ~vout ~axis ~descending =
  let in_shape = shape va in
  let rank = Array.length in_shape in
  let axis_size = in_shape.(axis) in
  if axis_size <= 1 then (
    let n = numel vout in
    let out_offset = View.offset vout in
    for i = 0 to n - 1 do
      Array.unsafe_set out_arr (out_offset + i) (Int32_u.of_int 0)
    done)
  else
    let outer =
      let p = ref 1 in
      for d = 0 to axis - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let inner =
      let p = ref 1 in
      for d = axis + 1 to rank - 1 do
        p := !p * in_shape.(d)
      done;
      !p
    in
    let groups = outer * inner in
    let a_strides = View.strides va in
    let a_offset = View.offset va in
    let out_strides = View.strides vout in
    let out_offset = View.offset vout in
    let a_axis_stride = a_strides.(axis) in
    let out_axis_stride = out_strides.(axis) in
    let work_on_group g =
      let o = g / inner in
      let i = g mod inner in
      let a_base =
        let off = ref a_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * a_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * a_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let out_base =
        let off = ref out_offset in
        let rem = ref o in
        for d = axis - 1 downto 0 do
          let s = in_shape.(d) in
          off := !off + (!rem mod s) * out_strides.(d);
          rem := !rem / s
        done;
        let rem2 = ref i in
        for d = rank - 1 downto axis + 1 do
          let s = in_shape.(d) in
          off := !off + (!rem2 mod s) * out_strides.(d);
          rem2 := !rem2 / s
        done;
        !off
      in
      let indices = Array.init axis_size Fun.id in
      let tmp = Array.make axis_size 0 in
      let acc = Array.make_int64 2 in
      let cmp a_idx b_idx =
        Array.unsafe_set acc 0
          (Array.unsafe_get a_arr (a_base + (a_idx * a_axis_stride)));
        Array.unsafe_set acc 1
          (Array.unsafe_get a_arr (a_base + (b_idx * a_axis_stride)));
        let c =
          Int64_u.compare (Array.unsafe_get acc 0) (Array.unsafe_get acc 1)
        in
        if descending then -c else c
      in
      merge_sort_indices indices tmp axis_size cmp;
      for j = 0 to axis_size - 1 do
        Array.unsafe_set out_arr
          (out_base + (j * out_axis_stride))
          (Int32_u.of_int indices.(j))
      done
    in
    if groups > parallel_threshold then
      Parallel.parallel_for pool 0 (groups - 1) (fun s e ->
          for g = s to e - 1 do
            work_on_group g
          done)
    else
      for g = 0 to groups - 1 do
        work_on_group g
      done
