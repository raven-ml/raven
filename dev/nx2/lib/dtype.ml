(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type float64_elt = Bigarray.float64_elt
type float32_elt = Bigarray.float32_elt
type float16_elt = Bigarray.float16_elt
type bfloat16_elt = |
type float8_e4m3_elt = |
type float8_e5m2_elt = |
type float4_e2m1_elt = |
type int64_elt = Bigarray.int64_elt
type uint64_elt = |
type int32_elt = Bigarray.int32_elt
type uint32_elt = |
type int16_signed_elt = Bigarray.int16_signed_elt
type int16_unsigned_elt = Bigarray.int16_unsigned_elt
type int8_signed_elt = Bigarray.int8_signed_elt
type int8_unsigned_elt = Bigarray.int8_unsigned_elt
type int4_elt = |
type uint4_elt = |
type complex64_elt = Bigarray.complex64_elt
type complex32_elt = Bigarray.complex32_elt
type bool_elt = |
type bit_elt = |

(* The constructors' order is the codes' order, NX_DTYPES in nx_dtype.h: C reads
   a dtype as the immediate integer OCaml makes of it. *)

type ('v, 's) t =
  | Float64 : (float, float64_elt) t
  | Float32 : (float, float32_elt) t
  | Float16 : (float, float16_elt) t
  | Bfloat16 : (float, bfloat16_elt) t
  | Float8_e4m3 : (float, float8_e4m3_elt) t
  | Float8_e5m2 : (float, float8_e5m2_elt) t
  | Float4_e2m1 : (float, float4_e2m1_elt) t
  | Int64 : (int64, int64_elt) t
  | Uint64 : (int64, uint64_elt) t
  | Int32 : (int32, int32_elt) t
  | Uint32 : (int32, uint32_elt) t
  | Int16 : (int, int16_signed_elt) t
  | Uint16 : (int, int16_unsigned_elt) t
  | Int8 : (int, int8_signed_elt) t
  | Uint8 : (int, int8_unsigned_elt) t
  | Int4 : (int, int4_elt) t
  | Uint4 : (int, uint4_elt) t
  | Complex128 : (Complex.t, complex64_elt) t
  | Complex64 : (Complex.t, complex32_elt) t
  | Bool : (bool, bool_elt) t
  | Bit : (bool, bit_elt) t

type any = Any : ('v, 's) t -> any

let all =
  [
    Any Float64;
    Any Float32;
    Any Float16;
    Any Bfloat16;
    Any Float8_e4m3;
    Any Float8_e5m2;
    Any Float4_e2m1;
    Any Int64;
    Any Uint64;
    Any Int32;
    Any Uint32;
    Any Int16;
    Any Uint16;
    Any Int8;
    Any Uint8;
    Any Int4;
    Any Uint4;
    Any Complex128;
    Any Complex64;
    Any Bool;
    Any Bit;
  ]

let code : type v s. (v, s) t -> int = function
  | Float64 -> 0
  | Float32 -> 1
  | Float16 -> 2
  | Bfloat16 -> 3
  | Float8_e4m3 -> 4
  | Float8_e5m2 -> 5
  | Float4_e2m1 -> 6
  | Int64 -> 7
  | Uint64 -> 8
  | Int32 -> 9
  | Uint32 -> 10
  | Int16 -> 11
  | Uint16 -> 12
  | Int8 -> 13
  | Uint8 -> 14
  | Int4 -> 15
  | Uint4 -> 16
  | Complex128 -> 17
  | Complex64 -> 18
  | Bool -> 19
  | Bit -> 20

(* Facts *)

type float_format = {
  exponent_bits : int;
  mantissa_bits : int;
  infinities : bool;
  nans : bool;
  epsilon : float;
  min_normal : float;
  max_finite : float;
}

type row = { name : string; bits : int; format : float_format option }

let ieee e m =
  let bias = (1 lsl (e - 1)) - 1 in
  let max_finite = Float.ldexp (2. -. Float.ldexp 1. (-m)) bias in
  Some
    {
      exponent_bits = e;
      mantissa_bits = m;
      infinities = true;
      nans = true;
      epsilon = Float.ldexp 1. (-m);
      min_normal = Float.ldexp 1. (1 - bias);
      max_finite;
    }

(* The OCP minifloats e4m3 and e2m1 spend the top exponent on finite values:
   e4m3 keeps one NaN code per sign, e2m1 none. *)
let ocp e m ~nans ~max_finite =
  let bias = (1 lsl (e - 1)) - 1 in
  Some
    {
      exponent_bits = e;
      mantissa_bits = m;
      infinities = false;
      nans;
      epsilon = Float.ldexp 1. (-m);
      min_normal = Float.ldexp 1. (1 - bias);
      max_finite;
    }

let row name bits format = { name; bits; format }

let rows =
  [|
    row "float64" 64 (ieee 11 52);
    row "float32" 32 (ieee 8 23);
    row "float16" 16 (ieee 5 10);
    row "bfloat16" 16 (ieee 8 7);
    row "float8_e4m3" 8 (ocp 4 3 ~nans:true ~max_finite:448.);
    row "float8_e5m2" 8 (ieee 5 2);
    row "float4_e2m1" 4 (ocp 2 1 ~nans:false ~max_finite:6.);
    row "int64" 64 None;
    row "uint64" 64 None;
    row "int32" 32 None;
    row "uint32" 32 None;
    row "int16" 16 None;
    row "uint16" 16 None;
    row "int8" 8 None;
    row "uint8" 8 None;
    row "int4" 4 None;
    row "uint4" 4 None;
    row "complex128" 128 None;
    row "complex64" 64 None;
    row "bool" 8 None;
    row "bit" 1 None;
  |]

let bits dt = (Array.unsafe_get rows (code dt)).bits
let name dt = (Array.unsafe_get rows (code dt)).name
let pp ppf dt = Format.pp_print_string ppf (name dt)
let of_name s = List.find_opt (fun (Any dt) -> String.equal (name dt) s) all

let float_format (type s) (dt : (float, s) t) =
  match rows.(code dt).format with Some f -> f | None -> assert false

let bytes dt n =
  if n < 0 then invalid_arg (Printf.sprintf "Dtype.bytes: %d elements" n);
  let b = bits dt in
  if b >= 8 then begin
    let w = b / 8 in
    if n > max_int / w then
      invalid_arg
        (Printf.sprintf "Dtype.bytes: %d elements of %s overflow" n (name dt));
    n * w
  end
  else
    let k = 8 / b in
    (n / k) + if n mod k = 0 then 0 else 1

(* Kinds *)

type 'v kind =
  | Float : float kind
  | Complex : Complex.t kind
  | Signed : 'v kind
  | Unsigned : 'v kind
  | Boolean : bool kind

let kind : type v s. (v, s) t -> v kind = function
  | Float64 -> Float
  | Float32 -> Float
  | Float16 -> Float
  | Bfloat16 -> Float
  | Float8_e4m3 -> Float
  | Float8_e5m2 -> Float
  | Float4_e2m1 -> Float
  | Int64 | Int32 | Int16 | Int8 | Int4 -> Signed
  | Uint64 | Uint32 | Uint16 | Uint8 | Uint4 -> Unsigned
  | Complex128 -> Complex
  | Complex64 -> Complex
  | Bool -> Boolean
  | Bit -> Boolean

let is (type k v s) (k : k kind) (dt : (v, s) t) =
  match (k, kind dt) with
  | Float, Float -> true
  | Complex, Complex -> true
  | Signed, Signed -> true
  | Unsigned, Unsigned -> true
  | Boolean, Boolean -> true
  | (Float | Complex | Signed | Unsigned | Boolean), _ -> false

(* Equality *)

let equal dt dt' = code dt = code dt'

let equal_witness : type v s w r.
    (v, s) t -> (w, r) t -> ((v, s) t, (w, r) t) Type.eq option =
 fun dt dt' ->
  match (dt, dt') with
  | Float64, Float64 -> Some Equal
  | Float32, Float32 -> Some Equal
  | Float16, Float16 -> Some Equal
  | Bfloat16, Bfloat16 -> Some Equal
  | Float8_e4m3, Float8_e4m3 -> Some Equal
  | Float8_e5m2, Float8_e5m2 -> Some Equal
  | Float4_e2m1, Float4_e2m1 -> Some Equal
  | Int64, Int64 -> Some Equal
  | Uint64, Uint64 -> Some Equal
  | Int32, Int32 -> Some Equal
  | Uint32, Uint32 -> Some Equal
  | Int16, Int16 -> Some Equal
  | Uint16, Uint16 -> Some Equal
  | Int8, Int8 -> Some Equal
  | Uint8, Uint8 -> Some Equal
  | Int4, Int4 -> Some Equal
  | Uint4, Uint4 -> Some Equal
  | Complex128, Complex128 -> Some Equal
  | Complex64, Complex64 -> Some Equal
  | Bool, Bool -> Some Equal
  | Bit, Bit -> Some Equal
  | _ -> None

(* Values *)

let zero : type v s. (v, s) t -> v = function
  | Float64 -> 0.
  | Float32 -> 0.
  | Float16 -> 0.
  | Bfloat16 -> 0.
  | Float8_e4m3 -> 0.
  | Float8_e5m2 -> 0.
  | Float4_e2m1 -> 0.
  | Int64 -> 0L
  | Uint64 -> 0L
  | Int32 -> 0l
  | Uint32 -> 0l
  | Int16 -> 0
  | Uint16 -> 0
  | Int8 -> 0
  | Uint8 -> 0
  | Int4 -> 0
  | Uint4 -> 0
  | Complex128 -> Complex.zero
  | Complex64 -> Complex.zero
  | Bool -> false
  | Bit -> false

let one : type v s. (v, s) t -> v = function
  | Float64 -> 1.
  | Float32 -> 1.
  | Float16 -> 1.
  | Bfloat16 -> 1.
  | Float8_e4m3 -> 1.
  | Float8_e5m2 -> 1.
  | Float4_e2m1 -> 1.
  | Int64 -> 1L
  | Uint64 -> 1L
  | Int32 -> 1l
  | Uint32 -> 1l
  | Int16 -> 1
  | Uint16 -> 1
  | Int8 -> 1
  | Uint8 -> 1
  | Int4 -> 1
  | Uint4 -> 1
  | Complex128 -> Complex.one
  | Complex64 -> Complex.one
  | Bool -> true
  | Bit -> true

let unordered fn dt =
  invalid_arg
    (Printf.sprintf "Dtype.%s: %s values are not ordered" fn (name dt))

let min_value : type v s. (v, s) t -> v = function
  | Float64 -> Float.neg_infinity
  | Float32 -> Float.neg_infinity
  | Float16 -> Float.neg_infinity
  | Bfloat16 -> Float.neg_infinity
  | Float8_e4m3 -> -448.
  | Float8_e5m2 -> Float.neg_infinity
  | Float4_e2m1 -> -6.
  | Int64 -> Int64.min_int
  | Uint64 -> 0L
  | Int32 -> Int32.min_int
  | Uint32 -> 0l
  | Int16 -> -0x8000
  | Uint16 -> 0
  | Int8 -> -0x80
  | Uint8 -> 0
  | Int4 -> -8
  | Uint4 -> 0
  | Complex128 -> unordered "min_value" Complex128
  | Complex64 -> unordered "min_value" Complex64
  | Bool -> false
  | Bit -> false

let max_value : type v s. (v, s) t -> v = function
  | Float64 -> Float.infinity
  | Float32 -> Float.infinity
  | Float16 -> Float.infinity
  | Bfloat16 -> Float.infinity
  | Float8_e4m3 -> 448.
  | Float8_e5m2 -> Float.infinity
  | Float4_e2m1 -> 6.
  | Int64 -> Int64.max_int
  | Uint64 -> -1L
  | Int32 -> Int32.max_int
  | Uint32 -> -1l
  | Int16 -> 0x7FFF
  | Uint16 -> 0xFFFF
  | Int8 -> 0x7F
  | Uint8 -> 0xFF
  | Int4 -> 7
  | Uint4 -> 15
  | Complex128 -> unordered "max_value" Complex128
  | Complex64 -> unordered "max_value" Complex64
  | Bool -> true
  | Bit -> true

(* Conversions. Both stores go through nx_dtype.h, the one source of the rule:
   [round] is a float's value once stored in a float dtype, [store] the bits a
   store writes into an integer dtype, sign-extended for signed ones. *)

external round : (int[@untagged]) -> (float[@unboxed]) -> (float[@unboxed])
  = "nx_dtype_round_byte" "nx_dtype_round"
[@@noalloc]

external store : (int[@untagged]) -> (float[@unboxed]) -> (int64[@unboxed])
  = "nx_dtype_store_byte" "nx_dtype_store"
[@@noalloc]

let small dt x = Int64.to_int (store (code dt) x)
let word dt x = Int64.to_int32 (store (code dt) x)
let complex32 x = { Complex.re = round (code Float32) x; im = 0. }

let of_float : type v s. (v, s) t -> float -> v =
 fun dt x ->
  match dt with
  | Float64 -> x
  | Float32 -> round (code dt) x
  | Float16 -> round (code dt) x
  | Bfloat16 -> round (code dt) x
  | Float8_e4m3 -> round (code dt) x
  | Float8_e5m2 -> round (code dt) x
  | Float4_e2m1 -> round (code dt) x
  | Int64 -> store (code dt) x
  | Uint64 -> store (code dt) x
  | Int32 -> word dt x
  | Uint32 -> word dt x
  | Int16 -> small dt x
  | Uint16 -> small dt x
  | Int8 -> small dt x
  | Uint8 -> small dt x
  | Int4 -> small dt x
  | Uint4 -> small dt x
  | Complex128 -> { Complex.re = x; im = 0. }
  | Complex64 -> complex32 x
  | Bool -> x <> 0.
  | Bit -> x <> 0.

(* Printing *)

(* The shortest decimal that a store into [dt] reads back as [x], so a value
   prints as briefly as its format allows: a float32 [0.1] prints as [0.1],
   though the double it holds has 17 digits. *)
let float_text code x =
  if Float.is_nan x then "nan"
  else if Float.is_integer x && Float.abs x < 1e16 then Printf.sprintf "%.0f" x
  else
    let rec shortest p =
      let s = Printf.sprintf "%.*g" p x in
      if p >= 17 || Float.equal (round code (float_of_string s)) x then s
      else shortest (p + 1)
    in
    shortest 1

let pp_float dt ppf x = Format.pp_print_string ppf (float_text (code dt) x)

let pp_complex dt ppf (z : Complex.t) =
  let part = float_text (code dt) in
  let im = part z.im in
  let sign = if String.length im > 0 && im.[0] = '-' then "" else "+" in
  Format.fprintf ppf "%s%s%si" (part z.re) sign im

let pp_value : type v s. (v, s) t -> Format.formatter -> v -> unit =
 fun dt ppf v ->
  match dt with
  | Float64 -> pp_float dt ppf v
  | Float32 -> pp_float dt ppf v
  | Float16 -> pp_float dt ppf v
  | Bfloat16 -> pp_float dt ppf v
  | Float8_e4m3 -> pp_float dt ppf v
  | Float8_e5m2 -> pp_float dt ppf v
  | Float4_e2m1 -> pp_float dt ppf v
  | Int64 -> Format.fprintf ppf "%Ld" v
  | Uint64 -> Format.fprintf ppf "%Lu" v
  | Int32 -> Format.fprintf ppf "%ld" v
  | Uint32 -> Format.fprintf ppf "%lu" v
  | Int16 -> Format.pp_print_int ppf v
  | Uint16 -> Format.pp_print_int ppf v
  | Int8 -> Format.pp_print_int ppf v
  | Uint8 -> Format.pp_print_int ppf v
  | Int4 -> Format.pp_print_int ppf v
  | Uint4 -> Format.pp_print_int ppf v
  | Complex128 -> pp_complex Float64 ppf v
  | Complex64 -> pp_complex Float32 ppf v
  | Bool -> Format.pp_print_bool ppf v
  | Bit -> Format.pp_print_bool ppf v
