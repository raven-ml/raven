(* Runtime representations of scalar types. *)

type float16_elt = Bigarray.float16_elt
type float32_elt = Bigarray.float32_elt
type float64_elt = Bigarray.float64_elt
type int8_elt = Bigarray.int8_signed_elt
type uint8_elt = Bigarray.int8_unsigned_elt
type int16_elt = Bigarray.int16_signed_elt
type uint16_elt = Bigarray.int16_unsigned_elt
type int32_elt = Bigarray.int32_elt
type int64_elt = Bigarray.int64_elt
type int_elt = Bigarray.int_elt
type nativeint_elt = Bigarray.nativeint_elt
type complex32_elt = Bigarray.complex32_elt
type complex64_elt = Bigarray.complex64_elt

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
(* The type parameter ['a] is the OCaml representation and ['b] is the
   corresponding Bigarray element kind (layout). *)

type float16_t = (float, float16_elt) t
type float32_t = (float, float32_elt) t
type float64_t = (float, float64_elt) t
type int8_t = (int, int8_elt) t
type uint8_t = (int, uint8_elt) t
type int16_t = (int, int16_elt) t
type uint16_t = (int, uint16_elt) t
type int32_t = (int32, int32_elt) t
type int64_t = (int64, int64_elt) t
type std_int_t = (int, int_elt) t
type std_nativeint_t = (nativeint, nativeint_elt) t
type complex32_t = (Complex.t, complex32_elt) t
type complex64_t = (Complex.t, complex64_elt) t
type 'b float_t = (float, 'b) t
type 'b int_based_t = (int, 'b) t

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

(* Create a power of 2 for integer shift operations *)
let power_of_two : type a b. (a, b) t -> int -> a =
 fun dtype shift_val ->
  if shift_val < 0 then
    invalid_arg "power_of_two: shift_val must be non-negative";
  match dtype with
  | Int8 | UInt8 | Int16 | UInt16 | Int | NativeInt -> (
      let power = 1 lsl shift_val in
      match dtype with
      | Int8 -> power
      | UInt8 -> power land 0xFF
      | Int16 -> power
      | UInt16 -> power land 0xFFFF
      | Int -> power
      | NativeInt -> Nativeint.shift_left Nativeint.one shift_val
      | _ -> failwith "Unreachable")
  | Int32 -> Int32.shift_left Int32.one shift_val
  | Int64 -> Int64.shift_left Int64.one shift_val
  | _ ->
      failwith
        ("power_of_two: unsupported dtype: " ^ to_string dtype
       ^ ". Expected integer type.")

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

(* Map a dtype to the corresponding Bigarray kind. *)
let kind_of_dtype : type a b. (a, b) t -> (a, b) Bigarray.kind =
 fun dtype ->
  match dtype with
  | Float16 -> Bigarray.Float16
  | Float32 -> Bigarray.Float32
  | Float64 -> Bigarray.Float64
  | Int8 -> Bigarray.Int8_signed
  | UInt8 -> Bigarray.Int8_unsigned
  | Int16 -> Bigarray.Int16_signed
  | UInt16 -> Bigarray.Int16_unsigned
  | Int32 -> Bigarray.Int32
  | Int64 -> Bigarray.Int64
  | Int -> Bigarray.Int
  | NativeInt -> Bigarray.Nativeint
  | Complex32 -> Bigarray.Complex32
  | Complex64 -> Bigarray.Complex64

(* Inverse of [kind_of_dtype]. Raises when the kind is not supported. *)
let dtype_of_kind : type a b. (a, b) Bigarray.kind -> (a, b) t = function
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
  | _ -> failwith "dtype_of_kind: Unsupported Bigarray kind"

(* Shallow equality on constructors. Useful for runtime checks. *)
let eq (type a b c d) (dt1 : (a, b) t) (dt2 : (c, d) t) : bool =
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
  | _ -> false

type (_, _) eq = Refl : ('x, 'x) eq

let eq_gadt : type a b c d.
    (a, b) t -> (c, d) t -> ((a, b) t, (c, d) t) eq option =
 fun dt1 dt2 ->
  match (dt1, dt2) with
  | Float16, Float16 -> Some Refl
  | Float32, Float32 -> Some Refl
  | Float64, Float64 -> Some Refl
  | Int8, Int8 -> Some Refl
  | UInt8, UInt8 -> Some Refl
  | Int16, Int16 -> Some Refl
  | UInt16, UInt16 -> Some Refl
  | Int32, Int32 -> Some Refl
  | Int64, Int64 -> Some Refl
  | Int, Int -> Some Refl
  | NativeInt, NativeInt -> Some Refl
  | Complex32, Complex32 -> Some Refl
  | Complex64, Complex64 -> Some Refl
  | _ -> None

let is_float (type a b) (dt : (a, b) t) : bool =
  match dt with Float16 | Float32 | Float64 -> true | _ -> false

let is_complex (type a b) (dt : (a, b) t) : bool =
  match dt with Complex32 | Complex64 -> true | _ -> false

let is_int (type a b) (dt : (a, b) t) : bool =
  match dt with
  | Int8 | UInt8 | Int16 | UInt16 | Int32 | Int64 | Int | NativeInt -> true
  | _ -> false

let is_uint (type a b) (dt : (a, b) t) : bool =
  match dt with UInt8 | UInt16 -> true | _ -> false

(* Minimum value for each dtype (identity for max reduction) *)
let min_val : type a b. (a, b) t -> a =
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

(* Helper function to convert a float to the OCaml representation ('a) of a
   given Dtype. *)
let float_to_dtype (type a b) (dtype : (a, b) t) (v_float : float) : a =
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
