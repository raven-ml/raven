(* Runtime representations of scalar types. *)

type float16_elt = Nx_buffer.float16_elt
type float32_elt = Nx_buffer.float32_elt
type float64_elt = Nx_buffer.float64_elt
type int8_elt = Nx_buffer.int8_signed_elt
type uint8_elt = Nx_buffer.int8_unsigned_elt
type int16_elt = Nx_buffer.int16_signed_elt
type uint16_elt = Nx_buffer.int16_unsigned_elt
type int32_elt = Nx_buffer.int32_elt
type int64_elt = Nx_buffer.int64_elt
type complex32_elt = Nx_buffer.complex32_elt
type complex64_elt = Nx_buffer.complex64_elt

(* Extended types from Nx_buffer *)
type uint32_elt = Nx_buffer.uint32_elt
type uint64_elt = Nx_buffer.uint64_elt
type bfloat16_elt = Nx_buffer.bfloat16_elt
type bool_elt = Nx_buffer.bool_elt
type int4_elt = Nx_buffer.int4_signed_elt
type uint4_elt = Nx_buffer.int4_unsigned_elt
type float8_e4m3_elt = Nx_buffer.float8_e4m3_elt
type float8_e5m2_elt = Nx_buffer.float8_e5m2_elt

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
  | UInt32 : (int32, uint32_elt) t
  | UInt64 : (int64, uint64_elt) t
  | Complex64 : (Complex.t, complex32_elt) t
  | Complex128 : (Complex.t, complex64_elt) t
  (* Extended types *)
  | BFloat16 : (float, bfloat16_elt) t
  | Bool : (bool, bool_elt) t
  | Int4 : (int, int4_elt) t
  | UInt4 : (int, uint4_elt) t
  | Float8_e4m3 : (float, float8_e4m3_elt) t
  | Float8_e5m2 : (float, float8_e5m2_elt) t
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
let uint32 = UInt32
let uint64 = UInt64
let complex64 = Complex64
let complex128 = Complex128

(* Extended types *)
let bfloat16 = BFloat16
let bool = Bool
let int4 = Int4
let uint4 = UInt4
let float8_e4m3 = Float8_e4m3
let float8_e5m2 = Float8_e5m2

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
  | UInt32 -> "uint32"
  | UInt64 -> "uint64"
  | Complex64 -> "complex64"
  | Complex128 -> "complex128"
  | BFloat16 -> "bfloat16"
  | Bool -> "bool"
  | Int4 -> "int4"
  | UInt4 -> "uint4"
  | Float8_e4m3 -> "float8_e4m3"
  | Float8_e5m2 -> "float8_e5m2"

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
  | UInt32 -> 0l
  | UInt64 -> 0L
  | Complex64 -> Complex.zero
  | Complex128 -> Complex.zero
  | BFloat16 -> 0.0
  | Bool -> false
  | Int4 -> 0
  | UInt4 -> 0
  | Float8_e4m3 -> 0.0
  | Float8_e5m2 -> 0.0

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
  | UInt32 -> 1l
  | UInt64 -> 1L
  | Complex64 -> Complex.one
  | Complex128 -> Complex.one
  | BFloat16 -> 1.0
  | Bool -> true
  | Int4 -> 1
  | UInt4 -> 1
  | Float8_e4m3 -> 1.0
  | Float8_e5m2 -> 1.0

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
  | UInt32 -> Int32.lognot 0l
  | UInt64 -> Int64.lognot 0L
  | Complex64 -> Complex.{ re = -1.0; im = 0.0 }
  | Complex128 -> Complex.{ re = -1.0; im = 0.0 }
  | BFloat16 -> -1.0
  | Bool -> true (* -1 for bool is true (all bits set) *)
  | Int4 -> -1
  | UInt4 -> 15 (* All bits set for uint4 *)
  | Float8_e4m3 -> -1.0
  | Float8_e5m2 -> -1.0

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
  | UInt32 -> 2l
  | UInt64 -> 2L
  | Complex64 -> Complex.{ re = 2.0; im = 0.0 }
  | Complex128 -> Complex.{ re = 2.0; im = 0.0 }
  | BFloat16 -> 2.0
  | Bool -> true (* Can't represent 2 in bool *)
  | Int4 -> 2
  | UInt4 -> 2
  | Float8_e4m3 -> 2.0
  | Float8_e5m2 -> 2.0

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
  | UInt32 -> 4
  | UInt64 -> 8
  | Complex64 -> 8
  | Complex128 -> 16
  | BFloat16 -> 2
  | Bool -> 1
  | Int4 -> 1 (* 2 values packed per byte *)
  | UInt4 -> 1 (* 2 values packed per byte *)
  | Float8_e4m3 -> 1
  | Float8_e5m2 -> 1

(* Inverse of [to_buffer_kind]. *)
let of_buffer_kind : type a b. (a, b) Nx_buffer.kind -> (a, b) t = function
  | Nx_buffer.Float16 -> Float16
  | Nx_buffer.Float32 -> Float32
  | Nx_buffer.Float64 -> Float64
  | Nx_buffer.Int8_signed -> Int8
  | Nx_buffer.Int8_unsigned -> UInt8
  | Nx_buffer.Int16_signed -> Int16
  | Nx_buffer.Int16_unsigned -> UInt16
  | Nx_buffer.Int32 -> Int32
  | Nx_buffer.Int64 -> Int64
  | Nx_buffer.Uint32 -> UInt32
  | Nx_buffer.Uint64 -> UInt64
  | Nx_buffer.Complex32 -> Complex64
  | Nx_buffer.Complex64 -> Complex128
  (* Extended types *)
  | Nx_buffer.Bfloat16 -> BFloat16
  | Nx_buffer.Bool -> Bool
  | Nx_buffer.Int4_signed -> Int4
  | Nx_buffer.Int4_unsigned -> UInt4
  | Nx_buffer.Float8_e4m3 -> Float8_e4m3
  | Nx_buffer.Float8_e5m2 -> Float8_e5m2
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
  | Complex64 -> Bigarray.Complex32
  | Complex128 -> Bigarray.Complex64
  | BFloat16 | Bool | Int4 | UInt4 | Float8_e4m3 | Float8_e5m2 | UInt32 | UInt64
    ->
      Error.invalid ~op:"to_bigarray_kind" ~what:"dtype"
        ~reason:"extended type not supported by standard Bigarray" ()

(* Map a dtype to the corresponding Nx_buffer kind. Works for all types
   including extended ones. *)
let to_buffer_kind : type a b. (a, b) t -> (a, b) Nx_buffer.kind =
 fun dtype ->
  match dtype with
  | Float16 -> Nx_buffer.Float16
  | Float32 -> Nx_buffer.Float32
  | Float64 -> Nx_buffer.Float64
  | Int8 -> Nx_buffer.Int8_signed
  | Int16 -> Nx_buffer.Int16_signed
  | UInt8 -> Nx_buffer.Int8_unsigned
  | UInt16 -> Nx_buffer.Int16_unsigned
  | Int32 -> Nx_buffer.Int32
  | Int64 -> Nx_buffer.Int64
  | UInt32 -> Nx_buffer.Uint32
  | UInt64 -> Nx_buffer.Uint64
  | Complex64 -> Nx_buffer.Complex32
  | Complex128 -> Nx_buffer.Complex64
  | BFloat16 -> Nx_buffer.Bfloat16
  | Bool -> Nx_buffer.Bool
  | Int4 -> Nx_buffer.Int4_signed
  | UInt4 -> Nx_buffer.Int4_unsigned
  | Float8_e4m3 -> Nx_buffer.Float8_e4m3
  | Float8_e5m2 -> Nx_buffer.Float8_e5m2

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
  | Bigarray.Complex32 -> Complex64
  | Bigarray.Complex64 -> Complex128
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
  | UInt32, UInt32 -> true
  | UInt64, UInt64 -> true
  | Complex64, Complex64 -> true
  | Complex128, Complex128 -> true
  | BFloat16, BFloat16 -> true
  | Bool, Bool -> true
  | Int4, Int4 -> true
  | UInt4, UInt4 -> true
  | Float8_e4m3, Float8_e4m3 -> true
  | Float8_e5m2, Float8_e5m2 -> true
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
  | UInt32, UInt32 -> Some Equal
  | UInt64, UInt64 -> Some Equal
  | Complex64, Complex64 -> Some Equal
  | Complex128, Complex128 -> Some Equal
  | BFloat16, BFloat16 -> Some Equal
  | Bool, Bool -> Some Equal
  | Int4, Int4 -> Some Equal
  | UInt4, UInt4 -> Some Equal
  | Float8_e4m3, Float8_e4m3 -> Some Equal
  | Float8_e5m2, Float8_e5m2 -> Some Equal
  | _ -> None

let is_float (type a b) (dt : (a, b) t) : bool =
  match dt with
  | Float16 | Float32 | Float64 | BFloat16 | Float8_e4m3 | Float8_e5m2 -> true
  | _ -> false

let is_complex (type a b) (dt : (a, b) t) : bool =
  match dt with Complex64 | Complex128 -> true | _ -> false

let is_int (type a b) (dt : (a, b) t) : bool =
  match dt with
  | Int8 | UInt8 | Int16 | UInt16 | Int32 | Int64 | UInt32 | UInt64 | Int4
  | UInt4 ->
      true
  | _ -> false

let is_uint (type a b) (dt : (a, b) t) : bool =
  match dt with UInt8 | UInt16 | UInt32 | UInt64 | UInt4 -> true | _ -> false

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
  | UInt32 -> 0l
  | UInt64 -> 0L
  | Complex64 -> Complex.{ re = Float.neg_infinity; im = Float.neg_infinity }
  | Complex128 -> Complex.{ re = Float.neg_infinity; im = Float.neg_infinity }
  | BFloat16 -> Float.neg_infinity
  | Bool -> false
  | Int4 -> -8 (* 4-bit signed: -8 to 7 *)
  | UInt4 -> 0
  | Float8_e4m3 -> Float.neg_infinity
  | Float8_e5m2 -> Float.neg_infinity

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
  | UInt32 -> Int32.lognot 0l
  | UInt64 -> Int64.lognot 0L
  | Complex64 -> Complex.{ re = Float.infinity; im = Float.infinity }
  | Complex128 -> Complex.{ re = Float.infinity; im = Float.infinity }
  | BFloat16 -> Float.infinity
  | Bool -> true
  | Int4 -> 7 (* 4-bit signed: -8 to 7 *)
  | UInt4 -> 15 (* 4-bit unsigned: 0 to 15 *)
  | Float8_e4m3 -> Float.infinity
  | Float8_e5m2 -> Float.infinity

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
  | UInt32 ->
      let max_u32 = 4294967295.0 in
      let clamped =
        if v_float <= 0.0 then 0.0
        else if v_float >= max_u32 then max_u32
        else v_float
      in
      Int32.of_int (int_of_float clamped)
  | UInt64 ->
      let max_u64 = 18446744073709551615.0 in
      let max_i64 = Int64.to_float Int64.max_int in
      if v_float <= 0.0 then 0L
      else if v_float >= max_u64 then Int64.lognot 0L
      else if v_float <= max_i64 then Int64.of_float v_float
      else Int64.of_float (v_float -. 18446744073709551616.0)
  | Complex64 -> Complex.{ re = v_float; im = 0. }
  | Complex128 -> Complex.{ re = v_float; im = 0. }
  | BFloat16 -> v_float
  | Bool -> v_float <> 0.0
  | Int4 -> int_of_float v_float
  | UInt4 -> int_of_float v_float
  | Float8_e4m3 -> v_float
  | Float8_e5m2 -> v_float

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
    Pack Complex64;
    Pack UInt32;
    Pack UInt64;
    Pack Complex128;
    Pack BFloat16;
    Pack Bool;
    Pack Int4;
    Pack UInt4;
    Pack Float8_e4m3;
    Pack Float8_e5m2;
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
  | UInt32 -> Int32.add x y
  | UInt64 -> Int64.add x y
  | Complex64 -> Complex.add x y
  | Complex128 -> Complex.add x y
  | BFloat16 -> x +. y
  | Bool -> x || y (* Logical OR for bool addition *)
  | Int4 -> x + y
  | UInt4 -> x + y
  | Float8_e4m3 -> x +. y
  | Float8_e5m2 -> x +. y

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
  | UInt32 -> Int32.sub x y
  | UInt64 -> Int64.sub x y
  | Complex64 -> Complex.sub x y
  | Complex128 -> Complex.sub x y
  | BFloat16 -> x -. y
  | Bool -> x && not y (* Logical AND NOT for bool subtraction *)
  | Int4 -> x - y
  | UInt4 -> x - y
  | Float8_e4m3 -> x -. y
  | Float8_e5m2 -> x -. y

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
  | UInt32 -> Int32.mul x y
  | UInt64 -> Int64.mul x y
  | Complex64 -> Complex.mul x y
  | Complex128 -> Complex.mul x y
  | BFloat16 -> x *. y
  | Bool -> x && y (* Logical AND for bool multiplication *)
  | Int4 -> x * y
  | UInt4 -> x * y
  | Float8_e4m3 -> x *. y
  | Float8_e5m2 -> x *. y

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
  | Int8 -> x / y
  | UInt8 -> x / y
  | Int16 -> x / y
  | UInt16 -> x / y
  | Int32 -> Int32.div x y
  | Int64 -> Int64.div x y
  | UInt32 -> uint32_div x y
  | UInt64 -> uint64_div x y
  | Complex64 -> Complex.div x y
  | Complex128 -> Complex.div x y
  | BFloat16 -> x /. y
  | Bool -> x (* Bool division just returns x *)
  | Int4 -> x / y
  | UInt4 -> x / y
  | Float8_e4m3 -> x /. y
  | Float8_e5m2 -> x /. y
