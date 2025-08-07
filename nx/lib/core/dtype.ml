(* Runtime representations of scalar types. *)

module BA = Bigarray_ext

type float16_elt = BA.float16_elt
type float32_elt = BA.float32_elt
type float64_elt = BA.float64_elt
type int8_elt = BA.int8_signed_elt
type uint8_elt = BA.int8_unsigned_elt
type int16_elt = BA.int16_signed_elt
type uint16_elt = BA.int16_unsigned_elt
type int32_elt = BA.int32_elt
type int64_elt = BA.int64_elt
type int_elt = BA.int_elt
type nativeint_elt = BA.nativeint_elt
type complex32_elt = BA.complex32_elt
type complex64_elt = BA.complex64_elt

(* Extended types from Bigarray_ext *)
type bfloat16_elt = BA.bfloat16_elt
type bool_elt = BA.bool_elt
type int4_elt = BA.int4_signed_elt
type uint4_elt = BA.int4_unsigned_elt
type float8_e4m3_elt = BA.float8_e4m3_elt
type float8_e5m2_elt = BA.float8_e5m2_elt
type complex16_elt = BA.complex16_elt
type qint8_elt = BA.qint8_elt
type quint8_elt = BA.quint8_elt

type ('a, 'b) t =
  | Float16 : (float, float16_elt) t
  | Float32 : (float, float32_elt) t
  | Float64 : (float, float64_elt) t
  | Int8 : (int, int8_elt) t
  | UInt8 : (int, uint8_elt) t
  | Int16 : (int, int16_elt) t
  | UInt16 : (int, uint16_elt) t
  | Int32 : (int32, int32_elt) t
  | Int64 : (int64, int64_elt) t
  | Int : (int, int_elt) t
  | NativeInt : (nativeint, nativeint_elt) t
  | Complex32 : (Complex.t, complex32_elt) t
  | Complex64 : (Complex.t, complex64_elt) t
  (* Extended types *)
  | BFloat16 : (float, bfloat16_elt) t
  | Bool : (bool, bool_elt) t
  | Int4 : (int, int4_elt) t
  | UInt4 : (int, uint4_elt) t
  | Float8_e4m3 : (float, float8_e4m3_elt) t
  | Float8_e5m2 : (float, float8_e5m2_elt) t
  | Complex16 : (Complex.t, complex16_elt) t
  | QInt8 : (int, qint8_elt) t
  | QUInt8 : (int, quint8_elt) t
(* The type parameter ['a] is the OCaml representation and ['b] is the
   corresponding Bigarray element kind (layout). *)

(* Constructor shortcuts *)
let float16 = Float16
let float32 = Float32
let float64 = Float64
let int8 = Int8
let uint8 = UInt8
let int16 = Int16
let uint16 = UInt16
let int32 = Int32
let int64 = Int64
let int = Int
let nativeint = NativeInt
let complex32 = Complex32
let complex64 = Complex64

(* Extended types *)
let bfloat16 = BFloat16
let bool = Bool
let int4 = Int4
let uint4 = UInt4
let float8_e4m3 = Float8_e4m3
let float8_e5m2 = Float8_e5m2
let complex16 = Complex16
let qint8 = QInt8
let quint8 = QUInt8

(* Printable name of the dtype. *)
let to_string : type a b. (a, b) t -> string = function
  | Float16 -> "float16"
  | Float32 -> "float32"
  | Float64 -> "float64"
  | Int8 -> "int8"
  | UInt8 -> "uint8"
  | Int16 -> "int16"
  | UInt16 -> "uint16"
  | Int32 -> "int32"
  | Int64 -> "int64"
  | Int -> "int"
  | NativeInt -> "nativeint"
  | Complex32 -> "complex32"
  | Complex64 -> "complex64"
  | BFloat16 -> "bfloat16"
  | Bool -> "bool"
  | Int4 -> "int4"
  | UInt4 -> "uint4"
  | Float8_e4m3 -> "float8_e4m3"
  | Float8_e5m2 -> "float8_e5m2"
  | Complex16 -> "complex16"
  | QInt8 -> "qint8"
  | QUInt8 -> "quint8"

let pp fmt dtype = Format.fprintf fmt "%s" (to_string dtype)

(* Additive identity for a given dtype. *)
let zero : type a b. (a, b) t -> a =
 fun dtype ->
  match dtype with
  | Float16 -> 0.0
  | Float32 -> 0.0
  | Float64 -> 0.0
  | Int8 -> 0
  | UInt8 -> 0
  | Int16 -> 0
  | UInt16 -> 0
  | Int32 -> 0l
  | Int64 -> 0L
  | Int -> 0
  | NativeInt -> 0n
  | Complex32 -> Complex.zero
  | Complex64 -> Complex.zero
  | BFloat16 -> 0.0
  | Bool -> false
  | Int4 -> 0
  | UInt4 -> 0
  | Float8_e4m3 -> 0.0
  | Float8_e5m2 -> 0.0
  | Complex16 -> Complex.zero
  | QInt8 -> 0
  | QUInt8 -> 0

(* Multiplicative identity for a given dtype. *)
let one : type a b. (a, b) t -> a =
 fun dtype ->
  match dtype with
  | Float16 -> 1.0
  | Float32 -> 1.0
  | Float64 -> 1.0
  | Int8 -> 1
  | UInt8 -> 1
  | Int16 -> 1
  | UInt16 -> 1
  | Int32 -> 1l
  | Int64 -> 1L
  | Int -> 1
  | NativeInt -> 1n
  | Complex32 -> Complex.one
  | Complex64 -> Complex.one
  | BFloat16 -> 1.0
  | Bool -> true
  | Int4 -> 1
  | UInt4 -> 1
  | Float8_e4m3 -> 1.0
  | Float8_e5m2 -> 1.0
  | Complex16 -> Complex.one
  | QInt8 -> 1
  | QUInt8 -> 1

let minus_one : type a b. (a, b) t -> a =
 fun dtype ->
  match dtype with
  | Float16 -> -1.0
  | Float32 -> -1.0
  | Float64 -> -1.0
  | Int8 -> -1
  | UInt8 ->
      (* Interpreting -1 as all bits set for uint8 *)
      255
  | Int16 -> -1
  | UInt16 -> -1
  | Int32 -> -1l
  | Int64 -> -1L
  | Int -> -1
  | NativeInt -> -1n
  | Complex32 -> Complex.{ re = -1.0; im = 0.0 }
  | Complex64 -> Complex.{ re = -1.0; im = 0.0 }
  | BFloat16 -> -1.0
  | Bool -> true (* -1 for bool is true (all bits set) *)
  | Int4 -> -1
  | UInt4 -> 15 (* All bits set for uint4 *)
  | Float8_e4m3 -> -1.0
  | Float8_e5m2 -> -1.0
  | Complex16 -> Complex.{ re = -1.0; im = 0.0 }
  | QInt8 -> -1
  | QUInt8 -> 255

let two : type a b. (a, b) t -> a =
 fun dtype ->
  match dtype with
  | Float16 -> 2.0
  | Float32 -> 2.0
  | Float64 -> 2.0
  | Int8 -> 2
  | UInt8 -> 2
  | Int16 -> 2
  | UInt16 -> 2
  | Int32 -> 2l
  | Int64 -> 2L
  | Int -> 2
  | NativeInt -> 2n
  | Complex32 -> Complex.{ re = 2.0; im = 0.0 }
  | Complex64 -> Complex.{ re = 2.0; im = 0.0 }
  | BFloat16 -> 2.0
  | Bool -> true (* Can't represent 2 in bool *)
  | Int4 -> 2
  | UInt4 -> 2
  | Float8_e4m3 -> 2.0
  | Float8_e5m2 -> 2.0
  | Complex16 -> Complex.{ re = 2.0; im = 0.0 }
  | QInt8 -> 2
  | QUInt8 -> 2

(* Size in bytes of one element of the dtype. *)
let itemsize : type a b. (a, b) t -> int = function
  | Float16 -> 2
  | Float32 -> 4
  | Float64 -> 8
  | Int8 -> 1
  | UInt8 -> 1
  | Int16 -> 2
  | UInt16 -> 2
  | Int32 -> 4
  | Int64 -> 8
  | Int -> Sys.int_size / 8
  | NativeInt -> Nativeint.size / 8
  | Complex32 -> 8
  | Complex64 -> 16
  | BFloat16 -> 2
  | Bool -> 1
  | Int4 -> 1 (* 2 values packed per byte *)
  | UInt4 -> 1 (* 2 values packed per byte *)
  | Float8_e4m3 -> 1
  | Float8_e5m2 -> 1
  | Complex16 -> 4 (* 2 x bfloat16 *)
  | QInt8 -> 1
  | QUInt8 -> 1

(* Inverse of [to_bigarray_ext_kind]. *)
let of_bigarray_ext_kind : type a b. (a, b) BA.kind -> (a, b) t = function
  | BA.Float16 -> Float16
  | BA.Float32 -> Float32
  | BA.Float64 -> Float64
  | BA.Int8_signed -> Int8
  | BA.Int8_unsigned -> UInt8
  | BA.Int16_signed -> Int16
  | BA.Int16_unsigned -> UInt16
  | BA.Int32 -> Int32
  | BA.Int64 -> Int64
  | BA.Int -> Int
  | BA.Nativeint -> NativeInt
  | BA.Complex32 -> Complex32
  | BA.Complex64 -> Complex64
  (* Extended types *)
  | BA.Bfloat16 -> BFloat16
  | BA.Bool -> Bool
  | BA.Int4_signed -> Int4
  | BA.Int4_unsigned -> UInt4
  | BA.Float8_e4m3 -> Float8_e4m3
  | BA.Float8_e5m2 -> Float8_e5m2
  | BA.Complex16 -> Complex16
  | BA.Qint8 -> QInt8
  | BA.Quint8 -> QUInt8
  | _ ->
      Error.invalid ~op:"of_bigarray_kind" ~what:"bigarray kind"
        ~reason:"unsupported kind" ()

(* Map a dtype to the corresponding standard Bigarray kind. Only works for types
   supported by standard Bigarray. *)
let to_bigarray_kind : type a b. (a, b) t -> (a, b) Bigarray.kind =
 fun dtype ->
  match dtype with
  | Float16 -> Bigarray.Float16
  | Float32 -> Bigarray.Float32
  | Float64 -> Bigarray.Float64
  | Int8 -> Bigarray.Int8_signed
  | Int16 -> Bigarray.Int16_signed
  | UInt8 -> Bigarray.Int8_unsigned
  | UInt16 -> Bigarray.Int16_unsigned
  | Int32 -> Bigarray.Int32
  | Int64 -> Bigarray.Int64
  | Int -> Bigarray.Int
  | NativeInt -> Bigarray.Nativeint
  | Complex32 -> Bigarray.Complex32
  | Complex64 -> Bigarray.Complex64
  | BFloat16 | Bool | Int4 | UInt4 | Float8_e4m3 | Float8_e5m2 | Complex16
  | QInt8 | QUInt8 ->
      Error.invalid ~op:"to_bigarray_kind" ~what:"dtype"
        ~reason:"extended type not supported by standard Bigarray" ()

(* Map a dtype to the corresponding Bigarray_ext kind. Works for all types
   including extended ones. *)
let to_bigarray_ext_kind : type a b. (a, b) t -> (a, b) Bigarray_ext.kind =
 fun dtype ->
  match dtype with
  | Float16 -> Bigarray_ext.Float16
  | Float32 -> Bigarray_ext.Float32
  | Float64 -> Bigarray_ext.Float64
  | Int8 -> Bigarray_ext.Int8_signed
  | Int16 -> Bigarray_ext.Int16_signed
  | UInt8 -> Bigarray_ext.Int8_unsigned
  | UInt16 -> Bigarray_ext.Int16_unsigned
  | Int32 -> Bigarray_ext.Int32
  | Int64 -> Bigarray_ext.Int64
  | Int -> Bigarray_ext.Int
  | NativeInt -> Bigarray_ext.Nativeint
  | Complex32 -> Bigarray_ext.Complex32
  | Complex64 -> Bigarray_ext.Complex64
  | BFloat16 -> Bigarray_ext.Bfloat16
  | Bool -> Bigarray_ext.Bool
  | Int4 -> Bigarray_ext.Int4_signed
  | UInt4 -> Bigarray_ext.Int4_unsigned
  | Float8_e4m3 -> Bigarray_ext.Float8_e4m3
  | Float8_e5m2 -> Bigarray_ext.Float8_e5m2
  | Complex16 -> Bigarray_ext.Complex16
  | QInt8 -> Bigarray_ext.Qint8
  | QUInt8 -> Bigarray_ext.Quint8

(* Inverse of [to_bigarray_kind]. Only handles standard Bigarray kinds. *)
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
  | Bigarray.Int -> Int
  | Bigarray.Nativeint -> NativeInt
  | Bigarray.Complex32 -> Complex32
  | Bigarray.Complex64 -> Complex64
  | _ ->
      Error.invalid ~op:"of_bigarray_kind" ~what:"bigarray kind"
        ~reason:"unsupported kind" ()

(* Shallow equality on constructors. Useful for runtime checks. *)
let equal (type a b c d) (dt1 : (a, b) t) (dt2 : (c, d) t) : bool =
  match (dt1, dt2) with
  | Float16, Float16 -> true
  | Float32, Float32 -> true
  | Float64, Float64 -> true
  | Int8, Int8 -> true
  | UInt8, UInt8 -> true
  | Int16, Int16 -> true
  | UInt16, UInt16 -> true
  | Int32, Int32 -> true
  | Int64, Int64 -> true
  | Int, Int -> true
  | NativeInt, NativeInt -> true
  | Complex32, Complex32 -> true
  | Complex64, Complex64 -> true
  | BFloat16, BFloat16 -> true
  | Bool, Bool -> true
  | Int4, Int4 -> true
  | UInt4, UInt4 -> true
  | Float8_e4m3, Float8_e4m3 -> true
  | Float8_e5m2, Float8_e5m2 -> true
  | Complex16, Complex16 -> true
  | QInt8, QInt8 -> true
  | QUInt8, QUInt8 -> true
  | _ -> false

let equal_witness : type a b c d.
    (a, b) t -> (c, d) t -> ((a, b) t, (c, d) t) Type.eq option =
 fun dt1 dt2 ->
  match (dt1, dt2) with
  | Float16, Float16 -> Some Equal
  | Float32, Float32 -> Some Equal
  | Float64, Float64 -> Some Equal
  | Int8, Int8 -> Some Equal
  | UInt8, UInt8 -> Some Equal
  | Int16, Int16 -> Some Equal
  | UInt16, UInt16 -> Some Equal
  | Int32, Int32 -> Some Equal
  | Int64, Int64 -> Some Equal
  | Int, Int -> Some Equal
  | NativeInt, NativeInt -> Some Equal
  | Complex32, Complex32 -> Some Equal
  | Complex64, Complex64 -> Some Equal
  | BFloat16, BFloat16 -> Some Equal
  | Bool, Bool -> Some Equal
  | Int4, Int4 -> Some Equal
  | UInt4, UInt4 -> Some Equal
  | Float8_e4m3, Float8_e4m3 -> Some Equal
  | Float8_e5m2, Float8_e5m2 -> Some Equal
  | Complex16, Complex16 -> Some Equal
  | QInt8, QInt8 -> Some Equal
  | QUInt8, QUInt8 -> Some Equal
  | _ -> None

let is_float (type a b) (dt : (a, b) t) : bool =
  match dt with
  | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2 -> true
  | _ -> false

let is_complex (type a b) (dt : (a, b) t) : bool =
  match dt with Complex32 | Complex64 | Complex16 -> true | _ -> false

let is_int (type a b) (dt : (a, b) t) : bool =
  match dt with
  | Int8 | UInt8 | Int16 | UInt16 | Int32 | Int64 | Int | NativeInt | Int4
  | UInt4 | QInt8 | QUInt8 ->
      true
  | _ -> false

let is_uint (type a b) (dt : (a, b) t) : bool =
  match dt with UInt8 | UInt16 | UInt4 | QUInt8 -> true | _ -> false

(* Minimum value for each dtype (identity for max reduction) *)
let min_value : type a b. (a, b) t -> a =
 fun dtype ->
  match dtype with
  | Float16 -> Float.neg_infinity
  | Float32 -> Float.neg_infinity
  | Float64 -> Float.neg_infinity
  | Int8 -> -128
  | UInt8 -> 0
  | Int16 -> -32768
  | UInt16 -> 0
  | Int32 -> Int32.min_int
  | Int64 -> Int64.min_int
  | Int -> Int.min_int
  | NativeInt -> Nativeint.min_int
  | Complex32 -> Complex.{ re = Float.neg_infinity; im = Float.neg_infinity }
  | Complex64 -> Complex.{ re = Float.neg_infinity; im = Float.neg_infinity }
  | BFloat16 -> Float.neg_infinity
  | Bool -> false
  | Int4 -> -8 (* 4-bit signed: -8 to 7 *)
  | UInt4 -> 0
  | Float8_e4m3 -> Float.neg_infinity
  | Float8_e5m2 -> Float.neg_infinity
  | Complex16 -> Complex.{ re = Float.neg_infinity; im = Float.neg_infinity }
  | QInt8 -> -128
  | QUInt8 -> 0

(* Maximum value for each dtype (identity for min reduction) *)
let max_value : type a b. (a, b) t -> a =
 fun dtype ->
  match dtype with
  | Float16 -> Float.infinity
  | Float32 -> Float.infinity
  | Float64 -> Float.infinity
  | Int8 -> 127
  | UInt8 -> 255
  | Int16 -> 32767
  | UInt16 -> 65535
  | Int32 -> Int32.max_int
  | Int64 -> Int64.max_int
  | Int -> Int.max_int
  | NativeInt -> Nativeint.max_int
  | Complex32 -> Complex.{ re = Float.infinity; im = Float.infinity }
  | Complex64 -> Complex.{ re = Float.infinity; im = Float.infinity }
  | BFloat16 -> Float.infinity
  | Bool -> true
  | Int4 -> 7 (* 4-bit signed: -8 to 7 *)
  | UInt4 -> 15 (* 4-bit unsigned: 0 to 15 *)
  | Float8_e4m3 -> Float.infinity
  | Float8_e5m2 -> Float.infinity
  | Complex16 -> Complex.{ re = Float.infinity; im = Float.infinity }
  | QInt8 -> 127
  | QUInt8 -> 255

(* Helper function to convert a float to the OCaml representation ('a) of a
   given Dtype. *)
let of_float (type a b) (dtype : (a, b) t) (v_float : float) : a =
  match dtype with
  | Float16 -> v_float
  | Float32 -> v_float
  | Float64 -> v_float
  | Int8 -> int_of_float v_float
  | UInt8 -> int_of_float v_float
  | Int16 -> int_of_float v_float
  | UInt16 -> int_of_float v_float
  | Int32 -> Int32.of_float v_float
  | Int64 -> Int64.of_float v_float
  | Int -> int_of_float v_float
  | NativeInt -> Nativeint.of_float v_float
  | Complex32 -> Complex.{ re = v_float; im = 0. }
  | Complex64 -> Complex.{ re = v_float; im = 0. }
  | BFloat16 -> v_float
  | Bool -> v_float <> 0.0
  | Int4 -> int_of_float v_float
  | UInt4 -> int_of_float v_float
  | Float8_e4m3 -> v_float
  | Float8_e5m2 -> v_float
  | Complex16 -> Complex.{ re = v_float; im = 0. }
  | QInt8 -> int_of_float v_float
  | QUInt8 -> int_of_float v_float

(* Packed type that hides the type parameters *)
type packed = Pack : ('a, 'b) t -> packed

(* Constructor for packing dtypes *)
let pack (type a b) (dtype : (a, b) t) : packed = Pack dtype

(* List of all available dtypes *)
let all_dtypes : packed list =
  [
    Pack Float16;
    Pack Float32;
    Pack Float64;
    Pack Int8;
    Pack UInt8;
    Pack Int16;
    Pack UInt16;
    Pack Int32;
    Pack Int64;
    Pack Int;
    Pack NativeInt;
    Pack Complex32;
    Pack Complex64;
    Pack BFloat16;
    Pack Bool;
    Pack Int4;
    Pack UInt4;
    Pack Float8_e4m3;
    Pack Float8_e5m2;
    Pack Complex16;
    Pack QInt8;
    Pack QUInt8;
  ]

(* Find a dtype by string name *)
let of_string (s : string) : packed option =
  let rec find = function
    | [] -> None
    | Pack dtype :: rest ->
        if String.equal (to_string dtype) s then Some (Pack dtype)
        else find rest
  in
  find all_dtypes

(* Equality for packed dtypes *)
let equal_packed (Pack dt1) (Pack dt2) : bool = equal dt1 dt2

(* Pretty printer for packed dtypes *)
let pp_packed fmt (Pack dtype) = pp fmt dtype

(* Convert packed dtype to string *)
let packed_to_string (Pack dtype) = to_string dtype

(* *)

let add (type a b) (dt : (a, b) t) (x : a) (y : a) : a =
  match dt with
  | Float16 -> x +. y
  | Float32 -> x +. y
  | Float64 -> x +. y
  | Int8 -> x + y
  | UInt8 -> x + y
  | Int16 -> x + y
  | UInt16 -> x + y
  | Int32 -> Int32.add x y
  | Int64 -> Int64.add x y
  | Int -> x + y
  | NativeInt -> Nativeint.add x y
  | Complex32 -> Complex.add x y
  | Complex64 -> Complex.add x y
  | BFloat16 -> x +. y
  | Bool -> x || y (* Logical OR for bool addition *)
  | Int4 -> x + y
  | UInt4 -> x + y
  | Float8_e4m3 -> x +. y
  | Float8_e5m2 -> x +. y
  | Complex16 -> Complex.add x y
  | QInt8 -> x + y
  | QUInt8 -> x + y

let sub (type a b) (dt : (a, b) t) (x : a) (y : a) : a =
  match dt with
  | Float16 -> x -. y
  | Float32 -> x -. y
  | Float64 -> x -. y
  | Int8 -> x - y
  | UInt8 -> x - y
  | Int16 -> x - y
  | UInt16 -> x - y
  | Int32 -> Int32.sub x y
  | Int64 -> Int64.sub x y
  | Int -> x - y
  | NativeInt -> Nativeint.sub x y
  | Complex32 -> Complex.sub x y
  | Complex64 -> Complex.sub x y
  | BFloat16 -> x -. y
  | Bool -> x && not y (* Logical AND NOT for bool subtraction *)
  | Int4 -> x - y
  | UInt4 -> x - y
  | Float8_e4m3 -> x -. y
  | Float8_e5m2 -> x -. y
  | Complex16 -> Complex.sub x y
  | QInt8 -> x - y
  | QUInt8 -> x - y

let mul (type a b) (dt : (a, b) t) (x : a) (y : a) : a =
  match dt with
  | Float16 -> x *. y
  | Float32 -> x *. y
  | Float64 -> x *. y
  | Int8 -> x * y
  | UInt8 -> x * y
  | Int16 -> x * y
  | UInt16 -> x * y
  | Int32 -> Int32.mul x y
  | Int64 -> Int64.mul x y
  | Int -> x * y
  | NativeInt -> Nativeint.mul x y
  | Complex32 -> Complex.mul x y
  | Complex64 -> Complex.mul x y
  | BFloat16 -> x *. y
  | Bool -> x && y (* Logical AND for bool multiplication *)
  | Int4 -> x * y
  | UInt4 -> x * y
  | Float8_e4m3 -> x *. y
  | Float8_e5m2 -> x *. y
  | Complex16 -> Complex.mul x y
  | QInt8 -> x * y
  | QUInt8 -> x * y

let div (type a b) (dt : (a, b) t) (x : a) (y : a) : a =
  match dt with
  | Float16 -> x /. y
  | Float32 -> x /. y
  | Float64 -> x /. y
  | Int8 -> x / y
  | UInt8 -> x / y
  | Int16 -> x / y
  | UInt16 -> x / y
  | Int32 -> Int32.div x y
  | Int64 -> Int64.div x y
  | Int -> x / y
  | NativeInt -> Nativeint.div x y
  | Complex32 -> Complex.div x y
  | Complex64 -> Complex.div x y
  | BFloat16 -> x /. y
  | Bool -> x (* Bool division just returns x *)
  | Int4 -> x / y
  | UInt4 -> x / y
  | Float8_e4m3 -> x /. y
  | Float8_e5m2 -> x /. y
  | Complex16 -> Complex.div x y
  | QInt8 -> x / y
  | QUInt8 -> x / y
