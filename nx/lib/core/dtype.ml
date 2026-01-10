(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Element Types ───── *)

type float16_elt = Nx_buffer.float16_elt
type float32_elt = Nx_buffer.float32_elt
type float64_elt = Nx_buffer.float64_elt
type bfloat16_elt = Nx_buffer.bfloat16_elt
type float8_e4m3_elt = Nx_buffer.float8_e4m3_elt
type float8_e5m2_elt = Nx_buffer.float8_e5m2_elt
type int4_elt = Nx_buffer.int4_signed_elt
type uint4_elt = Nx_buffer.int4_unsigned_elt
type int8_elt = Nx_buffer.int8_signed_elt
type uint8_elt = Nx_buffer.int8_unsigned_elt
type int16_elt = Nx_buffer.int16_signed_elt
type uint16_elt = Nx_buffer.int16_unsigned_elt
type int32_elt = Nx_buffer.int32_elt
type uint32_elt = Nx_buffer.uint32_elt
type int64_elt = Nx_buffer.int64_elt
type uint64_elt = Nx_buffer.uint64_elt
type complex32_elt = Nx_buffer.complex32_elt
type complex64_elt = Nx_buffer.complex64_elt
type bool_elt = Nx_buffer.bool_elt

(* ───── Dtype GADT ───── *)

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

(* ───── Constructor Shortcuts ───── *)

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

(* ───── String Conversion ───── *)

let to_string : type a b. (a, b) t -> string = function
  | Float16 -> "float16"
  | Float32 -> "float32"
  | Float64 -> "float64"
  | BFloat16 -> "bfloat16"
  | Float8_e4m3 -> "float8_e4m3"
  | Float8_e5m2 -> "float8_e5m2"
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

let pp fmt dtype = Format.fprintf fmt "%s" (to_string dtype)

(* ───── Properties ───── *)

let itemsize : type a b. (a, b) t -> int = function
  | Float16 -> 2
  | Float32 -> 4
  | Float64 -> 8
  | BFloat16 -> 2
  | Float8_e4m3 -> 1
  | Float8_e5m2 -> 1
  | Int4 -> 1 (* stored as 1 byte; packing is caller's responsibility *)
  | UInt4 -> 1 (* stored as 1 byte; packing is caller's responsibility *)
  | Int8 -> 1
  | UInt8 -> 1
  | Int16 -> 2
  | UInt16 -> 2
  | Int32 -> 4
  | UInt32 -> 4
  | Int64 -> 8
  | UInt64 -> 8
  | Complex64 -> 8
  | Complex128 -> 16
  | Bool -> 1

let bits : type a b. (a, b) t -> int = function
  | Float16 -> 16
  | Float32 -> 32
  | Float64 -> 64
  | BFloat16 -> 16
  | Float8_e4m3 -> 8
  | Float8_e5m2 -> 8
  | Int4 -> 4
  | UInt4 -> 4
  | Int8 -> 8
  | UInt8 -> 8
  | Int16 -> 16
  | UInt16 -> 16
  | Int32 -> 32
  | UInt32 -> 32
  | Int64 -> 64
  | UInt64 -> 64
  | Complex64 -> 64
  | Complex128 -> 128
  | Bool -> 8

(* ───── Type Predicates ───── *)

let is_float (type a b) (dt : (a, b) t) : bool =
  match dt with
  | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2 -> true
  | _ -> false

let is_complex (type a b) (dt : (a, b) t) : bool =
  match dt with Complex64 | Complex128 -> true | _ -> false

let is_int (type a b) (dt : (a, b) t) : bool =
  match dt with
  | Int4 | UInt4 | Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64
  | UInt64 ->
      true
  | _ -> false

let is_uint (type a b) (dt : (a, b) t) : bool =
  match dt with UInt4 | UInt8 | UInt16 | UInt32 | UInt64 -> true | _ -> false

(* ───── Constants ───── *)

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

let minus_one : type a b. (a, b) t -> a = function
  | Float16 -> -1.0
  | Float32 -> -1.0
  | Float64 -> -1.0
  | BFloat16 -> -1.0
  | Float8_e4m3 -> -1.0
  | Float8_e5m2 -> -1.0
  | Int4 -> -1
  | UInt4 -> 15 (* all bits set *)
  | Int8 -> -1
  | UInt8 -> 255 (* all bits set *)
  | Int16 -> -1
  | UInt16 -> 65535 (* all bits set *)
  | Int32 -> -1l
  | UInt32 -> Int32.lognot 0l
  | Int64 -> -1L
  | UInt64 -> Int64.lognot 0L
  | Complex64 -> Complex.{ re = -1.0; im = 0.0 }
  | Complex128 -> Complex.{ re = -1.0; im = 0.0 }
  | Bool -> true (* all bits set *)

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
  | Bool -> true (* saturates to max *)

(* ───── Bounds ───── *)

let min_value : type a b. (a, b) t -> a = function
  | Float16 -> Float.neg_infinity
  | Float32 -> Float.neg_infinity
  | Float64 -> Float.neg_infinity
  | BFloat16 -> Float.neg_infinity
  | Float8_e4m3 -> Float.neg_infinity
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
  | Complex64 ->
      Error.invalid ~op:"Dtype.min_value" ~what:"complex64"
        ~reason:"complex numbers are not ordered" ()
  | Complex128 ->
      Error.invalid ~op:"Dtype.min_value" ~what:"complex128"
        ~reason:"complex numbers are not ordered" ()
  | Bool -> false

let max_value : type a b. (a, b) t -> a = function
  | Float16 -> Float.infinity
  | Float32 -> Float.infinity
  | Float64 -> Float.infinity
  | BFloat16 -> Float.infinity
  | Float8_e4m3 -> Float.infinity
  | Float8_e5m2 -> Float.infinity
  | Int4 -> 7
  | UInt4 -> 15
  | Int8 -> 127
  | UInt8 -> 255
  | Int16 -> 32767
  | UInt16 -> 65535
  | Int32 -> Int32.max_int
  | UInt32 -> Int32.lognot 0l
  | Int64 -> Int64.max_int
  | UInt64 -> Int64.lognot 0L
  | Complex64 ->
      Error.invalid ~op:"Dtype.max_value" ~what:"complex64"
        ~reason:"complex numbers are not ordered" ()
  | Complex128 ->
      Error.invalid ~op:"Dtype.max_value" ~what:"complex128"
        ~reason:"complex numbers are not ordered" ()
  | Bool -> true

(* ───── Conversion ───── *)

let of_float (type a b) (dtype : (a, b) t) (v : float) : a =
  match dtype with
  | Float16 -> v
  | Float32 -> v
  | Float64 -> v
  | BFloat16 -> v
  | Float8_e4m3 -> v
  | Float8_e5m2 -> v
  | Int4 -> int_of_float v
  | UInt4 ->
      int_of_float (if v <= 0.0 then 0.0 else if v >= 15.0 then 15.0 else v)
  | Int8 -> int_of_float v
  | UInt8 ->
      int_of_float (if v <= 0.0 then 0.0 else if v >= 255.0 then 255.0 else v)
  | Int16 -> int_of_float v
  | UInt16 ->
      int_of_float
        (if v <= 0.0 then 0.0 else if v >= 65535.0 then 65535.0 else v)
  | Int32 -> Int32.of_float v
  | UInt32 ->
      Int64.to_int32
        (Int64.of_float
           (if v <= 0.0 then 0.0
            else if v >= 4294967295.0 then 4294967295.0
            else v))
  | Int64 -> Int64.of_float v
  | UInt64 ->
      let max_u64 = 18446744073709551615.0 in
      let max_i64 = Int64.to_float Int64.max_int in
      if v <= 0.0 then 0L
      else if v >= max_u64 then Int64.lognot 0L
      else if v <= max_i64 then Int64.of_float v
      else Int64.of_float (v -. 18446744073709551616.0)
  | Complex64 -> Complex.{ re = v; im = 0. }
  | Complex128 -> Complex.{ re = v; im = 0. }
  | Bool -> v <> 0.0

(* ───── Buffer/Bigarray Conversions ───── *)

let of_buffer_kind : type a b. (a, b) Nx_buffer.kind -> (a, b) t = function
  | Nx_buffer.Float16 -> Float16
  | Nx_buffer.Float32 -> Float32
  | Nx_buffer.Float64 -> Float64
  | Nx_buffer.Bfloat16 -> BFloat16
  | Nx_buffer.Float8_e4m3 -> Float8_e4m3
  | Nx_buffer.Float8_e5m2 -> Float8_e5m2
  | Nx_buffer.Int4_signed -> Int4
  | Nx_buffer.Int4_unsigned -> UInt4
  | Nx_buffer.Int8_signed -> Int8
  | Nx_buffer.Int8_unsigned -> UInt8
  | Nx_buffer.Int16_signed -> Int16
  | Nx_buffer.Int16_unsigned -> UInt16
  | Nx_buffer.Int32 -> Int32
  | Nx_buffer.Uint32 -> UInt32
  | Nx_buffer.Int64 -> Int64
  | Nx_buffer.Uint64 -> UInt64
  | Nx_buffer.Complex32 -> Complex64
  | Nx_buffer.Complex64 -> Complex128
  | Nx_buffer.Bool -> Bool
  | _ ->
      Error.invalid ~op:"Dtype.of_buffer_kind" ~what:"buffer kind"
        ~reason:"unsupported kind" ()

let to_buffer_kind : type a b. (a, b) t -> (a, b) Nx_buffer.kind = function
  | Float16 -> Nx_buffer.Float16
  | Float32 -> Nx_buffer.Float32
  | Float64 -> Nx_buffer.Float64
  | BFloat16 -> Nx_buffer.Bfloat16
  | Float8_e4m3 -> Nx_buffer.Float8_e4m3
  | Float8_e5m2 -> Nx_buffer.Float8_e5m2
  | Int4 -> Nx_buffer.Int4_signed
  | UInt4 -> Nx_buffer.Int4_unsigned
  | Int8 -> Nx_buffer.Int8_signed
  | UInt8 -> Nx_buffer.Int8_unsigned
  | Int16 -> Nx_buffer.Int16_signed
  | UInt16 -> Nx_buffer.Int16_unsigned
  | Int32 -> Nx_buffer.Int32
  | UInt32 -> Nx_buffer.Uint32
  | Int64 -> Nx_buffer.Int64
  | UInt64 -> Nx_buffer.Uint64
  | Complex64 -> Nx_buffer.Complex32
  | Complex128 -> Nx_buffer.Complex64
  | Bool -> Nx_buffer.Bool

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
  | _ ->
      Error.invalid ~op:"Dtype.of_bigarray_kind" ~what:"bigarray kind"
        ~reason:"unsupported kind" ()

let to_bigarray_kind : type a b. (a, b) t -> (a, b) Bigarray.kind = function
  | Float16 -> Bigarray.Float16
  | Float32 -> Bigarray.Float32
  | Float64 -> Bigarray.Float64
  | Int8 -> Bigarray.Int8_signed
  | UInt8 -> Bigarray.Int8_unsigned
  | Int16 -> Bigarray.Int16_signed
  | UInt16 -> Bigarray.Int16_unsigned
  | Int32 -> Bigarray.Int32
  | Int64 -> Bigarray.Int64
  | Complex64 -> Bigarray.Complex32
  | Complex128 -> Bigarray.Complex64
  | BFloat16 | Float8_e4m3 | Float8_e5m2 | Int4 | UInt4 | UInt32 | UInt64 | Bool
    ->
      Error.invalid ~op:"Dtype.to_bigarray_kind" ~what:"dtype"
        ~reason:"extended type not supported by standard Bigarray" ()

(* ───── Equality ───── *)

let equal (type a b c d) (dt1 : (a, b) t) (dt2 : (c, d) t) : bool =
  match (dt1, dt2) with
  | Float16, Float16 -> true
  | Float32, Float32 -> true
  | Float64, Float64 -> true
  | BFloat16, BFloat16 -> true
  | Float8_e4m3, Float8_e4m3 -> true
  | Float8_e5m2, Float8_e5m2 -> true
  | Int4, Int4 -> true
  | UInt4, UInt4 -> true
  | Int8, Int8 -> true
  | UInt8, UInt8 -> true
  | Int16, Int16 -> true
  | UInt16, UInt16 -> true
  | Int32, Int32 -> true
  | UInt32, UInt32 -> true
  | Int64, Int64 -> true
  | UInt64, UInt64 -> true
  | Complex64, Complex64 -> true
  | Complex128, Complex128 -> true
  | Bool, Bool -> true
  | _ -> false

let equal_witness (type a b c d) (dt1 : (a, b) t) (dt2 : (c, d) t) :
    ((a, b) t, (c, d) t) Type.eq option =
  match (dt1, dt2) with
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

(* ───── Packed ───── *)

type packed = Pack : ('a, 'b) t -> packed

let pack (type a b) (dt : (a, b) t) : packed = Pack dt

module Packed = struct
  type t = packed

  let all : t list =
    [
      Pack Float16;
      Pack Float32;
      Pack Float64;
      Pack BFloat16;
      Pack Float8_e4m3;
      Pack Float8_e5m2;
      Pack Int4;
      Pack UInt4;
      Pack Int8;
      Pack UInt8;
      Pack Int16;
      Pack UInt16;
      Pack Int32;
      Pack UInt32;
      Pack Int64;
      Pack UInt64;
      Pack Complex64;
      Pack Complex128;
      Pack Bool;
    ]

  let to_string (Pack dt) = to_string dt
  let pp fmt t = Format.fprintf fmt "%s" (to_string t)

  let of_string (s : string) : t option =
    List.find_map
      (fun packed ->
        if String.equal (to_string packed) s then Some packed else None)
      all

  let equal (Pack dt1) (Pack dt2) : bool = equal dt1 dt2

  let tag : t -> int = function
    | Pack Float16 -> 0
    | Pack Float32 -> 1
    | Pack Float64 -> 2
    | Pack BFloat16 -> 3
    | Pack Float8_e4m3 -> 4
    | Pack Float8_e5m2 -> 5
    | Pack Int4 -> 6
    | Pack UInt4 -> 7
    | Pack Int8 -> 8
    | Pack UInt8 -> 9
    | Pack Int16 -> 10
    | Pack UInt16 -> 11
    | Pack Int32 -> 12
    | Pack UInt32 -> 13
    | Pack Int64 -> 14
    | Pack UInt64 -> 15
    | Pack Complex64 -> 16
    | Pack Complex128 -> 17
    | Pack Bool -> 18

  let compare a b = Int.compare (tag a) (tag b)
  let hash t = tag t
end

(* ───── Narrow Integer Wrapping ───── *)

let wrap_uint4 x = x land 0xF
let wrap_uint8 x = x land 0xFF
let wrap_uint16 x = x land 0xFFFF

let wrap_int4 x =
  let masked = x land 0xF in
  if masked land 0x8 <> 0 then masked lor lnot 0xF else masked

let wrap_int8 x =
  let masked = x land 0xFF in
  if masked land 0x80 <> 0 then masked lor lnot 0xFF else masked

let wrap_int16 x =
  let masked = x land 0xFFFF in
  if masked land 0x8000 <> 0 then masked lor lnot 0xFFFF else masked

(* ───── Arithmetic Operations ───── *)

let add (type a b) (dt : (a, b) t) (x : a) (y : a) : a =
  match dt with
  | Float16 -> x +. y
  | Float32 -> x +. y
  | Float64 -> x +. y
  | BFloat16 -> x +. y
  | Float8_e4m3 -> x +. y
  | Float8_e5m2 -> x +. y
  | Int4 -> wrap_int4 (x + y)
  | UInt4 -> wrap_uint4 (x + y)
  | Int8 -> wrap_int8 (x + y)
  | UInt8 -> wrap_uint8 (x + y)
  | Int16 -> wrap_int16 (x + y)
  | UInt16 -> wrap_uint16 (x + y)
  | Int32 -> Int32.add x y
  | UInt32 -> Int32.add x y
  | Int64 -> Int64.add x y
  | UInt64 -> Int64.add x y
  | Complex64 -> Complex.add x y
  | Complex128 -> Complex.add x y
  | Bool -> x || y

let sub (type a b) (dt : (a, b) t) (x : a) (y : a) : a =
  match dt with
  | Float16 -> x -. y
  | Float32 -> x -. y
  | Float64 -> x -. y
  | BFloat16 -> x -. y
  | Float8_e4m3 -> x -. y
  | Float8_e5m2 -> x -. y
  | Int4 -> wrap_int4 (x - y)
  | UInt4 -> wrap_uint4 (x - y)
  | Int8 -> wrap_int8 (x - y)
  | UInt8 -> wrap_uint8 (x - y)
  | Int16 -> wrap_int16 (x - y)
  | UInt16 -> wrap_uint16 (x - y)
  | Int32 -> Int32.sub x y
  | UInt32 -> Int32.sub x y
  | Int64 -> Int64.sub x y
  | UInt64 -> Int64.sub x y
  | Complex64 -> Complex.sub x y
  | Complex128 -> Complex.sub x y
  | Bool -> Error.invalid ~op:"Dtype.sub" ~what:"bool" ~reason:"undefined" ()

let mul (type a b) (dt : (a, b) t) (x : a) (y : a) : a =
  match dt with
  | Float16 -> x *. y
  | Float32 -> x *. y
  | Float64 -> x *. y
  | BFloat16 -> x *. y
  | Float8_e4m3 -> x *. y
  | Float8_e5m2 -> x *. y
  | Int4 -> wrap_int4 (x * y)
  | UInt4 -> wrap_uint4 (x * y)
  | Int8 -> wrap_int8 (x * y)
  | UInt8 -> wrap_uint8 (x * y)
  | Int16 -> wrap_int16 (x * y)
  | UInt16 -> wrap_uint16 (x * y)
  | Int32 -> Int32.mul x y
  | UInt32 -> Int32.mul x y
  | Int64 -> Int64.mul x y
  | UInt64 -> Int64.mul x y
  | Complex64 -> Complex.mul x y
  | Complex128 -> Complex.mul x y
  | Bool -> x && y

let uint64_compare a b =
  Int64.compare (Int64.logxor a Int64.min_int) (Int64.logxor b Int64.min_int)

let uint32_div x y =
  let ux = Int64.logand (Int64.of_int32 x) 0xFFFFFFFFL in
  let uy = Int64.logand (Int64.of_int32 y) 0xFFFFFFFFL in
  if uy = 0L then raise Division_by_zero;
  Int32.of_int (Int64.to_int (Int64.div ux uy))

let uint64_div x y =
  if y = 0L then raise Division_by_zero;
  let open Int64 in
  let rec loop i rem quot =
    if i < 0 then quot
    else
      let bit = logand (shift_right_logical x i) 1L in
      let rem' = logor (shift_left rem 1) bit in
      if uint64_compare rem' y >= 0 then
        loop (i - 1) (sub rem' y) (logor quot (shift_left 1L i))
      else loop (i - 1) rem' quot
  in
  loop 63 0L 0L

let div (type a b) (dt : (a, b) t) (x : a) (y : a) : a =
  match dt with
  | Float16 -> x /. y
  | Float32 -> x /. y
  | Float64 -> x /. y
  | BFloat16 -> x /. y
  | Float8_e4m3 -> x /. y
  | Float8_e5m2 -> x /. y
  | Int4 -> wrap_int4 (x / y)
  | UInt4 -> wrap_uint4 (x / y)
  | Int8 -> wrap_int8 (x / y)
  | UInt8 -> wrap_uint8 (x / y)
  | Int16 -> wrap_int16 (x / y)
  | UInt16 -> wrap_uint16 (x / y)
  | Int32 -> Int32.div x y
  | UInt32 -> uint32_div x y
  | Int64 -> Int64.div x y
  | UInt64 -> uint64_div x y
  | Complex64 -> Complex.div x y
  | Complex128 -> Complex.div x y
  | Bool -> Error.invalid ~op:"Dtype.div" ~what:"bool" ~reason:"undefined" ()
