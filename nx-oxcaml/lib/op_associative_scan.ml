(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

type op = [ `Sum | `Prod | `Max | `Min ]

let product_range arr start_idx end_idx =
  let p = ref 1 in
  for i = start_idx to end_idx - 1 do
    p := !p * arr.(i)
  done;
  !p

let run_scan ~pool ~shape ~axis ~in_view ~out_view ~scan_slice =
  let rank = Array.length shape in
  let axis_len = shape.(axis) in
  if axis_len = 0 then ()
  else
    let inner_size = product_range shape (axis + 1) rank in
    let outer_size = product_range shape 0 axis in
    let total_slices = outer_size * inner_size in
    if total_slices = 0 then ()
    else
      let in_strides = View.strides in_view in
      let out_strides = View.strides out_view in
      let in_offset = View.offset in_view in
      let out_offset = View.offset out_view in
      let in_axis_stride = in_strides.(axis) in
      let out_axis_stride = out_strides.(axis) in

      let contiguous =
        View.is_c_contiguous in_view
        && View.is_c_contiguous out_view
        && axis = rank - 1
      in

      if contiguous then (
        let process_rows start_row end_row =
          for row = start_row to end_row - 1 do
            let base_in = in_offset + (row * axis_len) in
            let base_out = out_offset + (row * axis_len) in
            scan_slice base_in base_out 1 1 axis_len
          done
        in
        if outer_size > 8192 then
          Parallel.parallel_for pool 0 (outer_size - 1) process_rows
        else process_rows 0 outer_size
      )

      else (
        (* Build dims/strides excluding axis *)
        let slice_rank = rank - 1 in
        let dims = Array.make slice_rank 0 in
        let in_str = Array.make slice_rank 0 in
        let out_str = Array.make slice_rank 0 in

        let idx = ref 0 in
        for d = 0 to rank - 1 do
          if d <> axis then (
            dims.(!idx) <- shape.(d);
            in_str.(!idx) <- in_strides.(d);
            out_str.(!idx) <- out_strides.(d);
            incr idx)
        done;

        let process_chunk start_slice end_slice =
          (* Initialize coordinate state for this chunk *)
          let coords = Array.make slice_rank 0 in
          let in_base = ref in_offset in
          let out_base = ref out_offset in

          (* Advance to start_slice by incremental carry *)
          for _ = 0 to start_slice - 1 do
            let rec carry d =
              if d < 0 then ()
              else
                let next = coords.(d) + 1 in
                if next < dims.(d) then (
                  coords.(d) <- next;
                  in_base := !in_base + in_str.(d);
                  out_base := !out_base + out_str.(d)
                ) else (
                  coords.(d) <- 0;
                  in_base := !in_base - (dims.(d) - 1) * in_str.(d);
                  out_base := !out_base - (dims.(d) - 1) * out_str.(d);
                  carry (d - 1)
                )
            in
            carry (slice_rank - 1)
          done;

          for _ = start_slice to end_slice - 1 do
            scan_slice !in_base !out_base
              in_axis_stride out_axis_stride axis_len;

            (* Increment slice coordinates *)
            let rec carry d =
              if d >= 0 then
                let next = coords.(d) + 1 in
                if next < dims.(d) then (
                  coords.(d) <- next;
                  in_base := !in_base + in_str.(d);
                  out_base := !out_base + out_str.(d)
                ) else (
                  coords.(d) <- 0;
                  in_base := !in_base - (dims.(d) - 1) * in_str.(d);
                  out_base := !out_base - (dims.(d) - 1) * out_str.(d);
                  carry (d - 1)
                )
            in
            carry (slice_rank - 1)
          done
        in

        let parallel_threshold = 62500 in
        if total_slices > parallel_threshold then
          Parallel.parallel_for pool 0 (total_slices - 1) process_chunk
        else process_chunk 0 total_slices
      )

let scan_float64_sum_last_contiguous pool ~(out_arr : float# array)
    ~(in_arr : float# array) ~shape ~axis ~in_view ~out_view =
  let axis_len = shape.(axis) in
  let rows = product_range shape 0 axis in
  let in_offset = View.offset in_view in
  let out_offset = View.offset out_view in
  let process_rows start_row end_row =
    for row = start_row to end_row - 1 do
      let base_in = in_offset + (row * axis_len) in
      let base_out = out_offset + (row * axis_len) in
      let first = Array.unsafe_get in_arr base_in in
      Array.unsafe_set out_arr base_out first;
      let rec loop i acc =
        if i + 1 < axis_len then (
          let v = Float64x2.Array.unsafe_get in_arr ~idx:(base_in + i) in
          let #(v0, v1) = Float64x2.splat v in
          let p0 = Float_u.add acc v0 in
          let p1 = Float_u.add p0 v1 in
          Float64x2.Array.unsafe_set out_arr ~idx:(base_out + i)
            (Float64x2.set p0 p1);
          loop (i + 2) p1)
        else if i < axis_len then
          let next = Float_u.add acc (Array.unsafe_get in_arr (base_in + i)) in
          Array.unsafe_set out_arr (base_out + i) next;
          loop (i + 1) next
      in
      loop 1 first
    done
  in
  if rows > 8192 then Parallel.parallel_for pool 0 (rows - 1) process_rows
  else process_rows 0 rows

let scan_float64_prod_last_contiguous pool ~(out_arr : float# array)
    ~(in_arr : float# array) ~shape ~axis ~in_view ~out_view =
  let axis_len = shape.(axis) in
  let rows = product_range shape 0 axis in
  let in_offset = View.offset in_view in
  let out_offset = View.offset out_view in
  let process_rows start_row end_row =
    for row = start_row to end_row - 1 do
      let base_in = in_offset + (row * axis_len) in
      let base_out = out_offset + (row * axis_len) in
      let first = Array.unsafe_get in_arr base_in in
      Array.unsafe_set out_arr base_out first;
      let rec loop i acc =
        if i + 1 < axis_len then (
          let v = Float64x2.Array.unsafe_get in_arr ~idx:(base_in + i) in
          let #(v0, v1) = Float64x2.splat v in
          let p0 = Float_u.mul acc v0 in
          let p1 = Float_u.mul p0 v1 in
          Float64x2.Array.unsafe_set out_arr ~idx:(base_out + i)
            (Float64x2.set p0 p1);
          loop (i + 2) p1)
        else if i < axis_len then
          let next = Float_u.mul acc (Array.unsafe_get in_arr (base_in + i)) in
          Array.unsafe_set out_arr (base_out + i) next;
          loop (i + 1) next
      in
      loop 1 first
    done
  in
  if rows > 8192 then Parallel.parallel_for pool 0 (rows - 1) process_rows
  else process_rows 0 rows

let scan_float32_sum_last_contiguous pool ~(out_arr : float32# array)
    ~(in_arr : float32# array) ~shape ~axis ~in_view ~out_view =
  let axis_len = shape.(axis) in
  let rows = product_range shape 0 axis in
  let in_offset = View.offset in_view in
  let out_offset = View.offset out_view in
  let process_rows start_row end_row =
    for row = start_row to end_row - 1 do
      let base_in = in_offset + (row * axis_len) in
      let base_out = out_offset + (row * axis_len) in
      let first = Array.unsafe_get in_arr base_in in
      Array.unsafe_set out_arr base_out first;
      let rec loop i acc =
        if i + 3 < axis_len then (
          let v = Float32x4.Array.unsafe_get in_arr ~idx:(base_in + i) in
          let #(v0, v1, v2, v3) = Float32x4.splat v in
          let p0 = Float32_u.add acc v0 in
          let p1 = Float32_u.add p0 v1 in
          let p2 = Float32_u.add p1 v2 in
          let p3 = Float32_u.add p2 v3 in
          Float32x4.Array.unsafe_set out_arr ~idx:(base_out + i)
            (Float32x4.set p0 p1 p2 p3);
          loop (i + 4) p3)
        else if i < axis_len then
          let next = Float32_u.add acc (Array.unsafe_get in_arr (base_in + i)) in
          Array.unsafe_set out_arr (base_out + i) next;
          loop (i + 1) next
      in
      loop 1 first
    done
  in
  if rows > 8192 then Parallel.parallel_for pool 0 (rows - 1) process_rows
  else process_rows 0 rows

let scan_float32_prod_last_contiguous pool ~(out_arr : float32# array)
    ~(in_arr : float32# array) ~shape ~axis ~in_view ~out_view =
  let axis_len = shape.(axis) in
  let rows = product_range shape 0 axis in
  let in_offset = View.offset in_view in
  let out_offset = View.offset out_view in
  let process_rows start_row end_row =
    for row = start_row to end_row - 1 do
      let base_in = in_offset + (row * axis_len) in
      let base_out = out_offset + (row * axis_len) in
      let first = Array.unsafe_get in_arr base_in in
      Array.unsafe_set out_arr base_out first;
      let rec loop i acc =
        if i + 3 < axis_len then (
          let v = Float32x4.Array.unsafe_get in_arr ~idx:(base_in + i) in
          let #(v0, v1, v2, v3) = Float32x4.splat v in
          let p0 = Float32_u.mul acc v0 in
          let p1 = Float32_u.mul p0 v1 in
          let p2 = Float32_u.mul p1 v2 in
          let p3 = Float32_u.mul p2 v3 in
          Float32x4.Array.unsafe_set out_arr ~idx:(base_out + i)
            (Float32x4.set p0 p1 p2 p3);
          loop (i + 4) p3)
        else if i < axis_len then
          let next = Float32_u.mul acc (Array.unsafe_get in_arr (base_in + i)) in
          Array.unsafe_set out_arr (base_out + i) next;
          loop (i + 1) next
      in
      loop 1 first
    done
  in
  if rows > 8192 then Parallel.parallel_for pool 0 (rows - 1) process_rows
  else process_rows 0 rows

let scan_float64 pool ~(out_arr : float# array) ~(in_arr : float# array) ~shape
    ~axis ~in_view ~out_view ~op =
  if
    axis = Array.length shape - 1
    && View.is_c_contiguous in_view
    && View.is_c_contiguous out_view
  then
    match op with
    | `Sum ->
        scan_float64_sum_last_contiguous pool ~out_arr ~in_arr ~shape ~axis
          ~in_view ~out_view
    | `Prod ->
        scan_float64_prod_last_contiguous pool ~out_arr ~in_arr ~shape ~axis
          ~in_view ~out_view
    | `Max | `Min ->
        let combine =
          match op with
          | `Max -> Float_u.max
          | `Min -> Float_u.min
          | _ -> assert false
        in
        let scan_slice in_base out_base in_step out_step axis_len =
          let first = Array.unsafe_get in_arr in_base in
          Array.unsafe_set out_arr out_base first;
          let rec loop i acc in_idx out_idx =
            if i < axis_len then
              let next = combine acc (Array.unsafe_get in_arr in_idx) in
              Array.unsafe_set out_arr out_idx next;
              loop (i + 1) next (in_idx + in_step) (out_idx + out_step)
          in
          loop 1 first (in_base + in_step) (out_base + out_step)
        in
        run_scan ~pool ~shape ~axis ~in_view ~out_view ~scan_slice
  else
    let combine =
      match op with
      | `Sum -> Float_u.add
      | `Prod -> Float_u.mul
      | `Max -> Float_u.max
      | `Min -> Float_u.min
    in
    let scan_slice in_base out_base in_step out_step axis_len =
      let first = Array.unsafe_get in_arr in_base in
      Array.unsafe_set out_arr out_base first;
      let rec loop i acc in_idx out_idx =
        if i < axis_len then
          let next = combine acc (Array.unsafe_get in_arr in_idx) in
          Array.unsafe_set out_arr out_idx next;
          loop (i + 1) next (in_idx + in_step) (out_idx + out_step)
      in
      loop 1 first (in_base + in_step) (out_base + out_step)
    in
    run_scan ~pool ~shape ~axis ~in_view ~out_view ~scan_slice

let scan_float32 pool ~(out_arr : float32# array) ~(in_arr : float32# array)
    ~shape ~axis ~in_view ~out_view ~op =
  if
    axis = Array.length shape - 1
    && View.is_c_contiguous in_view
    && View.is_c_contiguous out_view
  then
    match op with
    | `Sum ->
        scan_float32_sum_last_contiguous pool ~out_arr ~in_arr ~shape ~axis
          ~in_view ~out_view
    | `Prod ->
        scan_float32_prod_last_contiguous pool ~out_arr ~in_arr ~shape ~axis
          ~in_view ~out_view
    | `Max | `Min ->
        let combine =
          match op with
          | `Max -> Float32_u.max
          | `Min -> Float32_u.min
          | _ -> assert false
        in
        let scan_slice in_base out_base in_step out_step axis_len =
          let first = Array.unsafe_get in_arr in_base in
          Array.unsafe_set out_arr out_base first;
          let rec loop i acc in_idx out_idx =
            if i < axis_len then
              let next = combine acc (Array.unsafe_get in_arr in_idx) in
              Array.unsafe_set out_arr out_idx next;
              loop (i + 1) next (in_idx + in_step) (out_idx + out_step)
          in
          loop 1 first (in_base + in_step) (out_base + out_step)
        in
        run_scan ~pool ~shape ~axis ~in_view ~out_view ~scan_slice
  else
    let combine =
      match op with
      | `Sum -> Float32_u.add
      | `Prod -> Float32_u.mul
      | `Max -> Float32_u.max
      | `Min -> Float32_u.min
    in
    let scan_slice in_base out_base in_step out_step axis_len =
      let first = Array.unsafe_get in_arr in_base in
      Array.unsafe_set out_arr out_base first;
      let rec loop i acc in_idx out_idx =
        if i < axis_len then
          let next = combine acc (Array.unsafe_get in_arr in_idx) in
          Array.unsafe_set out_arr out_idx next;
          loop (i + 1) next (in_idx + in_step) (out_idx + out_step)
      in
      loop 1 first (in_base + in_step) (out_base + out_step)
    in
    run_scan ~pool ~shape ~axis ~in_view ~out_view ~scan_slice

let scan_int8 pool ~(out_arr : int8# array) ~(in_arr : int8# array) ~shape ~axis
    ~in_view ~out_view ~op =
  let combine =
    match op with
    | `Sum -> Int8_u.add
    | `Prod -> Int8_u.mul
    | `Max -> Int8_u.max
    | `Min -> Int8_u.min
  in
  let scan_slice in_base out_base in_step out_step axis_len =
    let first = Array.unsafe_get in_arr in_base in
    Array.unsafe_set out_arr out_base first;
    let rec loop i acc in_idx out_idx =
      if i < axis_len then
        let next = combine acc (Array.unsafe_get in_arr in_idx) in
        Array.unsafe_set out_arr out_idx next;
        loop (i + 1) next (in_idx + in_step) (out_idx + out_step)
    in
    loop 1 first (in_base + in_step) (out_base + out_step)
  in
  run_scan ~pool ~shape ~axis ~in_view ~out_view ~scan_slice

let scan_int16 pool ~(out_arr : int16# array) ~(in_arr : int16# array) ~shape
    ~axis ~in_view ~out_view ~op =
  let combine =
    match op with
    | `Sum -> Int16_u.add
    | `Prod -> Int16_u.mul
    | `Max -> Int16_u.max
    | `Min -> Int16_u.min
  in
  let scan_slice in_base out_base in_step out_step axis_len =
    let first = Array.unsafe_get in_arr in_base in
    Array.unsafe_set out_arr out_base first;
    let rec loop i acc in_idx out_idx =
      if i < axis_len then
        let next = combine acc (Array.unsafe_get in_arr in_idx) in
        Array.unsafe_set out_arr out_idx next;
        loop (i + 1) next (in_idx + in_step) (out_idx + out_step)
    in
    loop 1 first (in_base + in_step) (out_base + out_step)
  in
  run_scan ~pool ~shape ~axis ~in_view ~out_view ~scan_slice

let scan_int32 pool ~(out_arr : int32# array) ~(in_arr : int32# array) ~shape
    ~axis ~in_view ~out_view ~op =
  let combine =
    match op with
    | `Sum -> Int32_u.add
    | `Prod -> Int32_u.mul
    | `Max -> Int32_u.max
    | `Min -> Int32_u.min
  in
  let scan_slice in_base out_base in_step out_step axis_len =
    let first = Array.unsafe_get in_arr in_base in
    Array.unsafe_set out_arr out_base first;
    let rec loop i acc in_idx out_idx =
      if i < axis_len then
        let next = combine acc (Array.unsafe_get in_arr in_idx) in
        Array.unsafe_set out_arr out_idx next;
        loop (i + 1) next (in_idx + in_step) (out_idx + out_step)
    in
    loop 1 first (in_base + in_step) (out_base + out_step)
  in
  run_scan ~pool ~shape ~axis ~in_view ~out_view ~scan_slice

let scan_int64 pool ~(out_arr : int64# array) ~(in_arr : int64# array) ~shape
    ~axis ~in_view ~out_view ~op =
  let combine =
    match op with
    | `Sum -> Int64_u.add
    | `Prod -> Int64_u.mul
    | `Max -> Int64_u.max
    | `Min -> Int64_u.min
  in
  let scan_slice in_base out_base in_step out_step axis_len =
    let first = Array.unsafe_get in_arr in_base in
    Array.unsafe_set out_arr out_base first;
    let rec loop i acc in_idx out_idx =
      if i < axis_len then
        let next = combine acc (Array.unsafe_get in_arr in_idx) in
        Array.unsafe_set out_arr out_idx next;
        loop (i + 1) next (in_idx + in_step) (out_idx + out_step)
    in
    loop 1 first (in_base + in_step) (out_base + out_step)
  in
  run_scan ~pool ~shape ~axis ~in_view ~out_view ~scan_slice
