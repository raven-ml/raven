(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Import

(* --- Float64 --- *)

let cast_float64_float32 (src : float# array) (dst : float32# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float32_u.of_float (Array.unsafe_get src src_lin))
  done

let cast_float64_int8 (src : float# array) (dst : int8# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int8_u.of_int (Float_u.to_int (Array.unsafe_get src src_lin)))
  done

let cast_float64_int16 (src : float# array) (dst : int16# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int16_u.of_int (Float_u.to_int (Array.unsafe_get src src_lin)))
  done

let cast_float64_int32 (src : float# array) (dst : int32# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int32_u.of_int32
          (Int32.of_int (Float_u.to_int (Array.unsafe_get src src_lin))))
  done

let cast_float64_int64 (src : float# array) (dst : int64# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int64_u.of_int64
          (Int64.of_int (Float_u.to_int (Array.unsafe_get src src_lin))))
  done

let cast_float64_bool (src : float# array) (dst : bool array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float_u.to_float (Array.unsafe_get src src_lin) <> 0.0)
  done

(* --- Float32 --- *)

let cast_float32_float64 (src : float32# array) (dst : float# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float32_u.to_float (Array.unsafe_get src src_lin))
  done

let cast_float32_int8 (src : float32# array) (dst : int8# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int8_u.of_int (Float32_u.to_int (Array.unsafe_get src src_lin)))
  done

let cast_float32_int16 (src : float32# array) (dst : int16# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int16_u.of_int (Float32_u.to_int (Array.unsafe_get src src_lin)))
  done

let cast_float32_int32 (src : float32# array) (dst : int32# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int32_u.of_int32
          (Int32.of_int (Float32_u.to_int (Array.unsafe_get src src_lin))))
  done

let cast_float32_int64 (src : float32# array) (dst : int64# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int64_u.of_int64
          (Int64.of_int (Float32_u.to_int (Array.unsafe_get src src_lin))))
  done

let cast_float32_bool (src : float32# array) (dst : bool array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float_u.to_float (Float32_u.to_float (Array.unsafe_get src src_lin))
      <> 0.0)
  done

(* --- Int8 --- *)

let cast_int8_float64 (src : int8# array) (dst : float# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float_u.of_int (Int8_u.to_int (Array.unsafe_get src src_lin)))
  done

let cast_int8_float32 (src : int8# array) (dst : float32# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float32_u.of_int (Int8_u.to_int (Array.unsafe_get src src_lin)))
  done

let cast_int8_int16 (src : int8# array) (dst : int16# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int16_u.of_int (Int8_u.to_int (Array.unsafe_get src src_lin)))
  done

let cast_int8_int32 (src : int8# array) (dst : int32# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int32_u.of_int32
          (Int32.of_int (Int8_u.to_int (Array.unsafe_get src src_lin))))
  done

let cast_int8_int64 (src : int8# array) (dst : int64# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int64_u.of_int64
          (Int64.of_int (Int8_u.to_int (Array.unsafe_get src src_lin))))
  done

let cast_int8_bool (src : int8# array) (dst : bool array) n in_shape in_offset
    in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int8_u.to_int (Array.unsafe_get src src_lin) <> 0)
  done

(* --- Int16 --- *)

let cast_int16_float64 (src : int16# array) (dst : float# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float_u.of_int (Int16_u.to_int (Array.unsafe_get src src_lin)))
  done

let cast_int16_float32 (src : int16# array) (dst : float32# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float32_u.of_int (Int16_u.to_int (Array.unsafe_get src src_lin)))
  done

let cast_int16_int8 (src : int16# array) (dst : int8# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int8_u.of_int (Int16_u.to_int (Array.unsafe_get src src_lin)))
  done

let cast_int16_int32 (src : int16# array) (dst : int32# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int32_u.of_int32
          (Int32.of_int (Int16_u.to_int (Array.unsafe_get src src_lin))))
  done

let cast_int16_int64 (src : int16# array) (dst : int64# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int64_u.of_int64
          (Int64.of_int (Int16_u.to_int (Array.unsafe_get src src_lin))))
  done

let cast_int16_bool (src : int16# array) (dst : bool array) n in_shape in_offset
    in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int16_u.to_int (Array.unsafe_get src src_lin) <> 0)
  done

(* --- Int32 --- *)

let cast_int32_float64 (src : int32# array) (dst : float# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float_u.of_int
          (Int32.to_int (Int32_u.to_int32 (Array.unsafe_get src src_lin))))
  done

let cast_int32_float32 (src : int32# array) (dst : float32# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float32_u.of_int
          (Int32.to_int (Int32_u.to_int32 (Array.unsafe_get src src_lin))))
  done

let cast_int32_int8 (src : int32# array) (dst : int8# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int8_u.of_int
          (Int32.to_int (Int32_u.to_int32 (Array.unsafe_get src src_lin))))
  done

let cast_int32_int16 (src : int32# array) (dst : int16# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int16_u.of_int
          (Int32.to_int (Int32_u.to_int32 (Array.unsafe_get src src_lin))))
  done

let cast_int32_int64 (src : int32# array) (dst : int64# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int64_u.of_int64
          (Int64.of_int32 (Int32_u.to_int32 (Array.unsafe_get src src_lin))))
  done

let cast_int32_bool (src : int32# array) (dst : bool array) n in_shape in_offset
    in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int32_u.to_int32 (Array.unsafe_get src src_lin) <> 0l)
  done

(* --- Int64 --- *)

let cast_int64_float64 (src : int64# array) (dst : float# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float_u.of_int
          (Int64.to_int (Int64_u.to_int64 (Array.unsafe_get src src_lin))))
  done

let cast_int64_float32 (src : int64# array) (dst : float32# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float32_u.of_int
          (Int64.to_int (Int64_u.to_int64 (Array.unsafe_get src src_lin))))
  done

let cast_int64_int8 (src : int64# array) (dst : int8# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int8_u.of_int
          (Int64.to_int (Int64_u.to_int64 (Array.unsafe_get src src_lin))))
  done

let cast_int64_int16 (src : int64# array) (dst : int16# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int16_u.of_int
          (Int64.to_int (Int64_u.to_int64 (Array.unsafe_get src src_lin))))
  done

let cast_int64_int32 (src : int64# array) (dst : int32# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int32_u.of_int32
          (Int64.to_int32 (Int64_u.to_int64 (Array.unsafe_get src src_lin))))
  done

let cast_int64_bool (src : int64# array) (dst : bool array) n in_shape in_offset
    in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int64_u.to_int64 (Array.unsafe_get src src_lin) <> 0L)
  done

(* --- Bool --- *)

let cast_bool_float64 (src : bool array) (dst : float# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float_u.of_float (if Array.unsafe_get src src_lin then 1.0 else 0.0))
  done

let cast_bool_float32 (src : bool array) (dst : float32# array) n in_shape
    in_offset in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Float32_u.of_int (if Array.unsafe_get src src_lin then 1 else 0))
  done

let cast_bool_int8 (src : bool array) (dst : int8# array) n in_shape in_offset
    in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int8_u.of_int (if Array.unsafe_get src src_lin then 1 else 0))
  done

let cast_bool_int16 (src : bool array) (dst : int16# array) n in_shape in_offset
    in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int16_u.of_int (if Array.unsafe_get src src_lin then 1 else 0))
  done

let cast_bool_int32 (src : bool array) (dst : int32# array) n in_shape in_offset
    in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int32_u.of_int32 (if Array.unsafe_get src src_lin then 1l else 0l))
  done

let cast_bool_int64 (src : bool array) (dst : int64# array) n in_shape in_offset
    in_strides out_offset out_strides =
  let md_index = Array.make (Array.length in_shape) 0 in
  for k = 0 to n - 1 do
    Shape.unravel_index_into k in_shape md_index;
    let src_lin = in_offset + Shape.ravel_index md_index in_strides in
    let dst_lin = out_offset + Shape.ravel_index md_index out_strides in
    Array.unsafe_set dst dst_lin
      (Int64_u.of_int64 (if Array.unsafe_get src src_lin then 1L else 0L))
  done