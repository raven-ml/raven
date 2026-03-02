(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

(* --- Float64 --- *)

let cast_with_views src_view dst_view f =
  let n = View.numel src_view in
  let shape = View.shape src_view in
  let in_offset = View.offset src_view in
  let in_strides = View.strides src_view in
  let out_offset = View.offset dst_view in
  let out_strides = View.strides dst_view in
  if n <> View.numel dst_view then invalid_arg "cast views must have equal numel";
  let contiguous =
    Array.for_all (( = ) 1) in_strides &&
    Array.for_all (( = ) 1) out_strides
  in
  if contiguous then (
    let i = ref 0 in
    let in_i = ref in_offset in
    let out_i = ref out_offset in
    while !i + 3 < n do
      f !in_i !out_i;
      f (!in_i + 1) (!out_i + 1);
      f (!in_i + 2) (!out_i + 2);
      f (!in_i + 3) (!out_i + 3);
      in_i := !in_i + 4;
      out_i := !out_i + 4;
      i := !i + 4
    done;
    while !i < n do
      f !in_i !out_i;
      incr in_i;
      incr out_i;
      incr i
    done
  ) else (
    let dims = Array.length shape in
    let coord = Array.make dims 0 in
    let in_i = ref in_offset in
    let out_i = ref out_offset in
    for _ = 0 to n - 1 do
      f !in_i !out_i;
      let d = ref (dims - 1) in
      while !d >= 0 do
        coord.(!d) <- coord.(!d) + 1;
        in_i := !in_i + in_strides.(!d);
        out_i := !out_i + out_strides.(!d);
        if coord.(!d) < shape.(!d) then d := -1
        else (
          in_i := !in_i - (coord.(!d) * in_strides.(!d));
          out_i := !out_i - (coord.(!d) * out_strides.(!d));
          coord.(!d) <- 0;
          decr d)
      done
    done)

let cast_float64_float32 src dst src_view dst_view =
  cast_with_views src_view dst_view (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float32_u.of_float (Array.unsafe_get src src_lin)))

let cast_float64_int8 (src : float# array) (dst : int8# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int8_u.of_int (Float_u.to_int (Array.unsafe_get src src_lin))))

let cast_float64_int16 (src : float# array) (dst : int16# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int16_u.of_int (Float_u.to_int (Array.unsafe_get src src_lin))))

let cast_float64_int32 (src : float# array) (dst : int32# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int32_u.of_int32
           (Int32.of_int (Float_u.to_int (Array.unsafe_get src src_lin)))))

let cast_float64_int64 (src : float# array) (dst : int64# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int64_u.of_int64
           (Int64.of_int (Float_u.to_int (Array.unsafe_get src src_lin)))))

let cast_float64_bool (src : float# array) (dst : bool array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float_u.to_float (Array.unsafe_get src src_lin) <> 0.0))

(* --- Float32 --- *)

let cast_float32_float64 (src : float32# array) (dst : float# array) src_view
    dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float32_u.to_float (Array.unsafe_get src src_lin)))

let cast_float32_int8 (src : float32# array) (dst : int8# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int8_u.of_int (Float32_u.to_int (Array.unsafe_get src src_lin))))

let cast_float32_int16 (src : float32# array) (dst : int16# array) src_view
    dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int16_u.of_int (Float32_u.to_int (Array.unsafe_get src src_lin))))

let cast_float32_int32 (src : float32# array) (dst : int32# array) src_view
    dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int32_u.of_int32
           (Int32.of_int (Float32_u.to_int (Array.unsafe_get src src_lin)))))

let cast_float32_int64 (src : float32# array) (dst : int64# array) src_view
    dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int64_u.of_int64
           (Int64.of_int (Float32_u.to_int (Array.unsafe_get src src_lin)))))

let cast_float32_bool (src : float32# array) (dst : bool array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float_u.to_float (Float32_u.to_float (Array.unsafe_get src src_lin))
        <> 0.0))

(* --- Int8 --- *)

let cast_int8_float64 (src : int8# array) (dst : float# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float_u.of_int (Int8_u.to_int (Array.unsafe_get src src_lin))))

let cast_int8_float32 (src : int8# array) (dst : float32# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float32_u.of_int (Int8_u.to_int (Array.unsafe_get src src_lin))))

let cast_int8_int16 (src : int8# array) (dst : int16# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int16_u.of_int (Int8_u.to_int (Array.unsafe_get src src_lin))))

let cast_int8_int32 (src : int8# array) (dst : int32# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int32_u.of_int32
           (Int32.of_int (Int8_u.to_int (Array.unsafe_get src src_lin)))))

let cast_int8_int64 (src : int8# array) (dst : int64# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int64_u.of_int64
           (Int64.of_int (Int8_u.to_int (Array.unsafe_get src src_lin)))))

let cast_int8_bool (src : int8# array) (dst : bool array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int8_u.to_int (Array.unsafe_get src src_lin) <> 0))

(* --- Int16 --- *)

let cast_int16_float64 (src : int16# array) (dst : float# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float_u.of_int (Int16_u.to_int (Array.unsafe_get src src_lin))))

let cast_int16_float32 (src : int16# array) (dst : float32# array) src_view
    dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float32_u.of_int (Int16_u.to_int (Array.unsafe_get src src_lin))))

let cast_int16_int8 (src : int16# array) (dst : int8# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int8_u.of_int (Int16_u.to_int (Array.unsafe_get src src_lin))))

let cast_int16_int32 (src : int16# array) (dst : int32# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int32_u.of_int32
           (Int32.of_int (Int16_u.to_int (Array.unsafe_get src src_lin)))))

let cast_int16_int64 (src : int16# array) (dst : int64# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int64_u.of_int64
           (Int64.of_int (Int16_u.to_int (Array.unsafe_get src src_lin)))))

let cast_int16_bool (src : int16# array) (dst : bool array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int16_u.to_int (Array.unsafe_get src src_lin) <> 0))

(* --- Int32 --- *)

let cast_int32_float64 (src : int32# array) (dst : float# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float_u.of_int
           (Int32.to_int (Int32_u.to_int32 (Array.unsafe_get src src_lin)))))

let cast_int32_float32 (src : int32# array) (dst : float32# array) src_view
    dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float32_u.of_int
           (Int32.to_int (Int32_u.to_int32 (Array.unsafe_get src src_lin)))))

let cast_int32_int8 (src : int32# array) (dst : int8# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int8_u.of_int
           (Int32.to_int (Int32_u.to_int32 (Array.unsafe_get src src_lin)))))

let cast_int32_int16 (src : int32# array) (dst : int16# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int16_u.of_int
           (Int32.to_int (Int32_u.to_int32 (Array.unsafe_get src src_lin)))))

let cast_int32_int64 (src : int32# array) (dst : int64# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int64_u.of_int64
           (Int64.of_int32 (Int32_u.to_int32 (Array.unsafe_get src src_lin)))))

let cast_int32_bool (src : int32# array) (dst : bool array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int32_u.to_int32 (Array.unsafe_get src src_lin) <> 0l))

(* --- Int64 --- *)

let cast_int64_float64 (src : int64# array) (dst : float# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float_u.of_int
           (Int64.to_int (Int64_u.to_int64 (Array.unsafe_get src src_lin)))))

let cast_int64_float32 (src : int64# array) (dst : float32# array) src_view
    dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float32_u.of_int
           (Int64.to_int (Int64_u.to_int64 (Array.unsafe_get src src_lin)))))

let cast_int64_int8 (src : int64# array) (dst : int8# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int8_u.of_int
           (Int64.to_int (Int64_u.to_int64 (Array.unsafe_get src src_lin)))))

let cast_int64_int16 (src : int64# array) (dst : int16# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int16_u.of_int
           (Int64.to_int (Int64_u.to_int64 (Array.unsafe_get src src_lin)))))

let cast_int64_int32 (src : int64# array) (dst : int32# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int32_u.of_int32
           (Int64.to_int32 (Int64_u.to_int64 (Array.unsafe_get src src_lin)))))

let cast_int64_bool (src : int64# array) (dst : bool array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int64_u.to_int64 (Array.unsafe_get src src_lin) <> 0L))

(* --- Bool --- *)

let cast_bool_float64 (src : bool array) (dst : float# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float_u.of_float (if Array.unsafe_get src src_lin then 1.0 else 0.0)))

let cast_bool_float32 (src : bool array) (dst : float32# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Float32_u.of_int (if Array.unsafe_get src src_lin then 1 else 0)))

let cast_bool_int8 (src : bool array) (dst : int8# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int8_u.of_int (if Array.unsafe_get src src_lin then 1 else 0)))

let cast_bool_int16 (src : bool array) (dst : int16# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int16_u.of_int (if Array.unsafe_get src src_lin then 1 else 0)))

let cast_bool_int32 (src : bool array) (dst : int32# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int32_u.of_int32 (if Array.unsafe_get src src_lin then 1l else 0l)))

let cast_bool_int64 (src : bool array) (dst : int64# array) src_view dst_view =
  cast_with_views src_view dst_view
    (fun src_lin dst_lin ->
      Array.unsafe_set dst dst_lin
        (Int64_u.of_int64 (if Array.unsafe_get src src_lin then 1L else 0L)))
