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
let to_bigarray_kind : type a b. (a, b) t -> (a, b) Bigarray.kind =
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
