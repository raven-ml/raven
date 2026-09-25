(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Element types *)

type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt
type bfloat16_elt = |
type float8_e4m3_elt = |
type float8_e5m2_elt = |
type int4_elt = |
type uint4_elt = |
type int8_elt = Bigarray.int8_signed_elt
type uint8_elt = Bigarray.int8_unsigned_elt
type int16_elt = Bigarray.int16_signed_elt
type uint16_elt = Bigarray.int16_unsigned_elt
type int32_elt = Bigarray.int32_elt
type uint32_elt = |
type int64_elt = Bigarray.int64_elt
type uint64_elt = |
type complex32_elt = Bigarray.complex32_elt
type complex64_elt = Bigarray.complex64_elt
type bool_elt = |

(* Dtypes. The constructor order is pinned by [caml_nx_buffer_kind] in
   nx.buffer's C and JavaScript stubs; keep the three in sync. *)

type ('a, 'b) t =
  | Float16 : (float, float16_elt) t
  | Float32 : (float, float32_elt) t
  | Float64 : (float, float64_elt) t
  | BFloat16 : (float, bfloat16_elt) t
  | Float8_e4m3 : (float, float8_e4m3_elt) t
  | Float8_e5m2 : (float, float8_e5m2_elt) t
  | Int4 : (int, int4_elt) t
  | UInt4 : (int, uint4_elt) t
  | Int8 : (int, int8_elt) t
  | UInt8 : (int, uint8_elt) t
  | Int16 : (int, int16_elt) t
  | UInt16 : (int, uint16_elt) t
  | Int32 : (int32, int32_elt) t
  | UInt32 : (int32, uint32_elt) t
  | Int64 : (int64, int64_elt) t
  | UInt64 : (int64, uint64_elt) t
  | Complex64 : (Complex.t, complex32_elt) t
  | Complex128 : (Complex.t, complex64_elt) t
  | Bool : (bool, bool_elt) t

type ('a, 'b) dtype = ('a, 'b) t

let float16 = Float16
let float32 = Float32
let float64 = Float64
let bfloat16 = BFloat16
let float8_e4m3 = Float8_e4m3
let float8_e5m2 = Float8_e5m2
let int4 = Int4
let uint4 = UInt4
let int8 = Int8
let uint8 = UInt8
let int16 = Int16
let uint16 = UInt16
let int32 = Int32
let uint32 = UInt32
let int64 = Int64
let uint64 = UInt64
let complex64 = Complex64
let complex128 = Complex128
let bool = Bool

(* Scalar descriptors *)

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

  let of_dtype : type a b. (a, b) dtype -> t = function
    | Float16 -> Float16
    | Float32 -> Float32
    | Float64 -> Float64
    | BFloat16 -> BFloat16
    | Float8_e4m3 -> Float8_e4m3
    | Float8_e5m2 -> Float8_e5m2
    | Int4 -> Int4
    | UInt4 -> UInt4
    | Int8 -> Int8
    | UInt8 -> UInt8
    | Int16 -> Int16
    | UInt16 -> UInt16
    | Int32 -> Int32
    | UInt32 -> UInt32
    | Int64 -> Int64
    | UInt64 -> UInt64
    | Complex64 -> Complex64
    | Complex128 -> Complex128
    | Bool -> Bool

  let bitsize = function
    | Float8_e4m3 | Float8_e5m2 | Float8_e4m3fnuz | Float8_e5m2fnuz | Int8
    | UInt8 | Bool ->
        8
    | Int4 | UInt4 -> 4
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

  let equal (a : t) b = a = b
end

(* Properties *)

let to_string dt = Scalar.to_string (Scalar.of_dtype dt)
let pp ppf dt = Format.pp_print_string ppf (to_string dt)
let itemsize dt = (Scalar.bitsize (Scalar.of_dtype dt) + 7) / 8

let is_float (type a b) (dt : (a, b) t) =
  match dt with
  | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2 -> true
  | _ -> false

let is_complex (type a b) (dt : (a, b) t) =
  match dt with Complex64 | Complex128 -> true | _ -> false

let is_int (type a b) (dt : (a, b) t) =
  match dt with
  | Int4 | UInt4 | Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64
  | UInt64 ->
      true
  | _ -> false

let is_uint (type a b) (dt : (a, b) t) =
  match dt with UInt4 | UInt8 | UInt16 | UInt32 | UInt64 -> true | _ -> false

(* Constants *)

let zero : type a b. (a, b) t -> a = function
  | Float16 -> 0.0
  | Float32 -> 0.0
  | Float64 -> 0.0
  | BFloat16 -> 0.0
  | Float8_e4m3 -> 0.0
  | Float8_e5m2 -> 0.0
  | Int4 -> 0
  | UInt4 -> 0
  | Int8 -> 0
  | UInt8 -> 0
  | Int16 -> 0
  | UInt16 -> 0
  | Int32 -> 0l
  | UInt32 -> 0l
  | Int64 -> 0L
  | UInt64 -> 0L
  | Complex64 -> Complex.zero
  | Complex128 -> Complex.zero
  | Bool -> false

let one : type a b. (a, b) t -> a = function
  | Float16 -> 1.0
  | Float32 -> 1.0
  | Float64 -> 1.0
  | BFloat16 -> 1.0
  | Float8_e4m3 -> 1.0
  | Float8_e5m2 -> 1.0
  | Int4 -> 1
  | UInt4 -> 1
  | Int8 -> 1
  | UInt8 -> 1
  | Int16 -> 1
  | UInt16 -> 1
  | Int32 -> 1l
  | UInt32 -> 1l
  | Int64 -> 1L
  | UInt64 -> 1L
  | Complex64 -> Complex.one
  | Complex128 -> Complex.one
  | Bool -> true

let two : type a b. (a, b) t -> a = function
  | Float16 -> 2.0
  | Float32 -> 2.0
  | Float64 -> 2.0
  | BFloat16 -> 2.0
  | Float8_e4m3 -> 2.0
  | Float8_e5m2 -> 2.0
  | Int4 -> 2
  | UInt4 -> 2
  | Int8 -> 2
  | UInt8 -> 2
  | Int16 -> 2
  | UInt16 -> 2
  | Int32 -> 2l
  | UInt32 -> 2l
  | Int64 -> 2L
  | UInt64 -> 2L
  | Complex64 -> Complex.{ re = 2.0; im = 0.0 }
  | Complex128 -> Complex.{ re = 2.0; im = 0.0 }
  | Bool -> true

let minus_one : type a b. (a, b) t -> a = function
  | Float16 -> -1.0
  | Float32 -> -1.0
  | Float64 -> -1.0
  | BFloat16 -> -1.0
  | Float8_e4m3 -> -1.0
  | Float8_e5m2 -> -1.0
  | Int4 -> -1
  | UInt4 -> 15
  | Int8 -> -1
  | UInt8 -> 255
  | Int16 -> -1
  | UInt16 -> 65535
  | Int32 -> -1l
  | UInt32 -> -1l
  | Int64 -> -1L
  | UInt64 -> -1L
  | Complex64 -> Complex.{ re = -1.0; im = 0.0 }
  | Complex128 -> Complex.{ re = -1.0; im = 0.0 }
  | Bool -> true

let min_value : type a b. (a, b) t -> a = function
  | Float16 -> Float.neg_infinity
  | Float32 -> Float.neg_infinity
  | Float64 -> Float.neg_infinity
  | BFloat16 -> Float.neg_infinity
  | Float8_e4m3 -> -448.0
  | Float8_e5m2 -> Float.neg_infinity
  | Int4 -> -8
  | UInt4 -> 0
  | Int8 -> -128
  | UInt8 -> 0
  | Int16 -> -32768
  | UInt16 -> 0
  | Int32 -> Int32.min_int
  | UInt32 -> 0l
  | Int64 -> Int64.min_int
  | UInt64 -> 0L
  | Complex64 | Complex128 ->
      invalid_arg "Nx_dtype.min_value: complex numbers are not ordered"
  | Bool -> false

let max_value : type a b. (a, b) t -> a = function
  | Float16 -> Float.infinity
  | Float32 -> Float.infinity
  | Float64 -> Float.infinity
  | BFloat16 -> Float.infinity
  | Float8_e4m3 -> 448.0
  | Float8_e5m2 -> Float.infinity
  | Int4 -> 7
  | UInt4 -> 15
  | Int8 -> 127
  | UInt8 -> 255
  | Int16 -> 32767
  | UInt16 -> 65535
  | Int32 -> Int32.max_int
  | UInt32 -> -1l
  | Int64 -> Int64.max_int
  | UInt64 -> -1L
  | Complex64 | Complex128 ->
      invalid_arg "Nx_dtype.max_value: complex numbers are not ordered"
  | Bool -> true

(* Converting *)

let of_float : type a b. (a, b) t -> float -> a =
 fun dt v ->
  let clamp lo hi v = if v <= lo then lo else if v >= hi then hi else v in
  match dt with
  | Float16 -> v
  | Float32 -> v
  | Float64 -> v
  | BFloat16 -> v
  | Float8_e4m3 -> v
  | Float8_e5m2 -> v
  | Int4 -> int_of_float v
  | UInt4 -> int_of_float (clamp 0. 15. v)
  | Int8 -> int_of_float v
  | UInt8 -> int_of_float (clamp 0. 255. v)
  | Int16 -> int_of_float v
  | UInt16 -> int_of_float (clamp 0. 65535. v)
  | Int32 -> Int32.of_float v
  | UInt32 -> Int64.to_int32 (Int64.of_float (clamp 0. 4294967295. v))
  | Int64 -> Int64.of_float v
  | UInt64 ->
      if v <= 0.0 then 0L
      else if v >= 18446744073709551615.0 then -1L
      else if v <= Int64.to_float Int64.max_int then Int64.of_float v
      else Int64.of_float (v -. 18446744073709551616.0)
  | Complex64 -> Complex.{ re = v; im = 0. }
  | Complex128 -> Complex.{ re = v; im = 0. }
  | Bool -> v <> 0.0

let of_bigarray_kind : type a b. (a, b) Bigarray.kind -> (a, b) t = function
  | Bigarray.Float16 -> Float16
  | Bigarray.Float32 -> Float32
  | Bigarray.Float64 -> Float64
  | Bigarray.Int8_signed -> Int8
  | Bigarray.Int8_unsigned -> UInt8
  | Bigarray.Int16_signed -> Int16
  | Bigarray.Int16_unsigned -> UInt16
  | Bigarray.Int32 -> Int32
  | Bigarray.Int64 -> Int64
  | Bigarray.Complex32 -> Complex64
  | Bigarray.Complex64 -> Complex128
  | Bigarray.Char | Bigarray.Int | Bigarray.Nativeint ->
      invalid_arg "Nx_dtype.of_bigarray_kind: no dtype for this kind"

let to_bigarray_kind : type a b. (a, b) t -> (a, b) Bigarray.kind option =
  function
  | Float16 -> Some Bigarray.Float16
  | Float32 -> Some Bigarray.Float32
  | Float64 -> Some Bigarray.Float64
  | Int8 -> Some Bigarray.Int8_signed
  | UInt8 -> Some Bigarray.Int8_unsigned
  | Int16 -> Some Bigarray.Int16_signed
  | UInt16 -> Some Bigarray.Int16_unsigned
  | Int32 -> Some Bigarray.Int32
  | Int64 -> Some Bigarray.Int64
  | Complex64 -> Some Bigarray.Complex32
  | Complex128 -> Some Bigarray.Complex64
  | BFloat16 | Float8_e4m3 | Float8_e5m2 | Int4 | UInt4 | UInt32 | UInt64 | Bool
    ->
      None

(* Equality *)

let equal a b = Scalar.equal (Scalar.of_dtype a) (Scalar.of_dtype b)

let equal_witness : type a b c d.
    (a, b) t -> (c, d) t -> ((a, b) t, (c, d) t) Type.eq option =
 fun a b ->
  match (a, b) with
  | Float16, Float16 -> Some Type.Equal
  | Float32, Float32 -> Some Type.Equal
  | Float64, Float64 -> Some Type.Equal
  | BFloat16, BFloat16 -> Some Type.Equal
  | Float8_e4m3, Float8_e4m3 -> Some Type.Equal
  | Float8_e5m2, Float8_e5m2 -> Some Type.Equal
  | Int4, Int4 -> Some Type.Equal
  | UInt4, UInt4 -> Some Type.Equal
  | Int8, Int8 -> Some Type.Equal
  | UInt8, UInt8 -> Some Type.Equal
  | Int16, Int16 -> Some Type.Equal
  | UInt16, UInt16 -> Some Type.Equal
  | Int32, Int32 -> Some Type.Equal
  | UInt32, UInt32 -> Some Type.Equal
  | Int64, Int64 -> Some Type.Equal
  | UInt64, UInt64 -> Some Type.Equal
  | Complex64, Complex64 -> Some Type.Equal
  | Complex128, Complex128 -> Some Type.Equal
  | Bool, Bool -> Some Type.Equal
  | _ -> None
