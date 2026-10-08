(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Scalar = struct
  type t =
    | Float16
    | Float32
    | Float64
    | BFloat16
    | Float8_e4m3
    | Float8_e5m2
    | Float8_e4m3fnuz
    | Float8_e5m2fnuz
    | Int4
    | UInt4
    | Int8
    | UInt8
    | Int16
    | UInt16
    | Int32
    | UInt32
    | Int64
    | UInt64
    | Complex64
    | Complex128
    | Bool
    | Bit

  let of_bigarray_kind : type a b. (a, b) Bigarray.kind -> t option = function
    | Bigarray.Float16 -> Some Float16
    | Bigarray.Float32 -> Some Float32
    | Bigarray.Float64 -> Some Float64
    | Bigarray.Int8_signed -> Some Int8
    | Bigarray.Int8_unsigned | Bigarray.Char -> Some UInt8
    | Bigarray.Int16_signed -> Some Int16
    | Bigarray.Int16_unsigned -> Some UInt16
    | Bigarray.Int32 -> Some Int32
    | Bigarray.Int64 -> Some Int64
    | Bigarray.Complex32 -> Some Complex64
    | Bigarray.Complex64 -> Some Complex128
    | Bigarray.Int | Bigarray.Nativeint -> None

  let bitsize = function
    | Float8_e4m3 | Float8_e5m2 | Float8_e4m3fnuz | Float8_e5m2fnuz | Int8
    | UInt8 | Bool ->
        8
    | Int4 | UInt4 -> 4
    | Bit -> 1
    | Float16 | BFloat16 | Int16 | UInt16 -> 16
    | Float32 | Int32 | UInt32 -> 32
    | Float64 | Int64 | UInt64 | Complex64 -> 64
    | Complex128 -> 128

  let to_string = function
    | Float16 -> "float16"
    | Float32 -> "float32"
    | Float64 -> "float64"
    | BFloat16 -> "bfloat16"
    | Float8_e4m3 -> "float8_e4m3"
    | Float8_e5m2 -> "float8_e5m2"
    | Float8_e4m3fnuz -> "float8_e4m3fnuz"
    | Float8_e5m2fnuz -> "float8_e5m2fnuz"
    | Int4 -> "int4"
    | UInt4 -> "uint4"
    | Int8 -> "int8"
    | UInt8 -> "uint8"
    | Int16 -> "int16"
    | UInt16 -> "uint16"
    | Int32 -> "int32"
    | UInt32 -> "uint32"
    | Int64 -> "int64"
    | UInt64 -> "uint64"
    | Complex64 -> "complex64"
    | Complex128 -> "complex128"
    | Bool -> "bool"
    | Bit -> "bit"

  let equal (a : t) b = a = b
end
