(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

type scalar =
  | Void
  | Bool
  | Int8
  | Int16
  | Int32
  | Int64
  | Uint8
  | Uint16
  | Uint32
  | Uint64
  | Float16
  | Bfloat16
  | Float32
  | Float64
  | Fp8e4m3
  | Fp8e5m2
  | Index

type addr_space = Global | Local | Reg
type t = { scalar : scalar; count : int }
type ptr = { base : t; addrspace : addr_space; v : int; size : int }
type any = T of t | P of ptr

(* Value dtype accessors *)

let scalar dt = dt.scalar
let count dt = dt.count

(* Pointer dtype accessors *)

let base p = p.base
let addrspace p = p.addrspace
let ptr_size p = p.size
let ptr_v p = p.v

(* Any dtype accessors *)

let any_scalar = function T dt -> dt.scalar | P p -> p.base.scalar
let any_count = function T dt -> dt.count | P p -> p.base.count
let vcount = function T dt -> dt.count | P p -> p.v
let any_to_val = function T dt -> dt | P p -> p.base
let any_is_ptr = function T _ -> false | P _ -> true

(* Coercions *)

let to_any dt = T dt
let ptr_to_any p = P p
let ptr_base_to_any p = T p.base

(* Constructors *)

let of_scalar s = { scalar = s; count = 1 }
let void = of_scalar Void
let bool = of_scalar Bool
let int8 = of_scalar Int8
let int16 = of_scalar Int16
let int32 = of_scalar Int32
let int64 = of_scalar Int64
let uint8 = of_scalar Uint8
let uint16 = of_scalar Uint16
let uint32 = of_scalar Uint32
let uint64 = of_scalar Uint64
let float16 = of_scalar Float16
let bfloat16 = of_scalar Bfloat16
let float32 = of_scalar Float32
let float64 = of_scalar Float64
let fp8e4m3 = of_scalar Fp8e4m3
let fp8e5m2 = of_scalar Fp8e5m2
let index = of_scalar Index
let default_float = float32
let default_int = int32

(* Pointer dtype constructors *)

let err_ptr_vcount n =
  Printf.sprintf "pointer vcount must be >= 1, got %d" n

let ptr_of base ~addrspace ~size =
  { base; addrspace; v = 1; size }

let ptr_of_v base ~addrspace ~size ~v =
  if v < 1 then invalid_arg (err_ptr_vcount v);
  { base; addrspace; v; size }

(* Pointer dtype transformers *)

let ptr_with_v p n =
  if n < 1 then invalid_arg (err_ptr_vcount n);
  { p with v = n }

let ptr_with_scalar p = if p.v = 1 then p else { p with v = 1 }
let ptr_with_size p n = { p with size = n }
let ptr_with_base p dt = { p with base = dt }

(* Predicates *)

let is_float t =
  match t.scalar with
  | Float16 | Bfloat16 | Float32 | Float64 | Fp8e4m3 | Fp8e5m2 -> true
  | _ -> false

let is_fp8 t = match t.scalar with Fp8e4m3 | Fp8e5m2 -> true | _ -> false

let is_int t =
  match t.scalar with
  | Int8 | Int16 | Int32 | Int64 | Uint8 | Uint16 | Uint32 | Uint64 | Index ->
      true
  | _ -> false

let is_unsigned t =
  match t.scalar with Uint8 | Uint16 | Uint32 | Uint64 -> true | _ -> false

let is_bool t = t.scalar = Bool

(* Properties *)

let scalar_bitsize = function
  | Void -> 0
  | Bool -> 1
  | Int8 | Uint8 | Fp8e4m3 | Fp8e5m2 -> 8
  | Int16 | Uint16 | Float16 | Bfloat16 -> 16
  | Int32 | Uint32 | Float32 -> 32
  | Int64 | Uint64 | Float64 -> 64
  | Index -> 800 (* sentinel: symbolic, not a machine type *)

let bitsize t = scalar_bitsize t.scalar * t.count
let itemsize t = (bitsize t + 7) / 8

let scalar_priority = function
  | Void | Index -> -1
  | Bool -> 0
  | Int8 -> 1 | Uint8 -> 2
  | Int16 -> 3 | Uint16 -> 4
  | Int32 -> 5 | Uint32 -> 6
  | Int64 -> 7 | Uint64 -> 8
  | Fp8e4m3 -> 9 | Fp8e5m2 -> 10
  | Float16 -> 11 | Bfloat16 -> 12
  | Float32 -> 13 | Float64 -> 14

let priority t = scalar_priority t.scalar

(* Operations *)

let scalar_of t = { t with count = 1 }

let any_scalar_of = function
  | T dt -> T (scalar_of dt)
  | P p -> P { p with base = scalar_of p.base; v = 1 }

let vec t n =
  if t.count <> 1 then
    invalid_arg (Printf.sprintf "can't vectorize type with count %d" t.count);
  if n < 0 then invalid_arg (Printf.sprintf "vector size must be >= 0, got %d" n);
  if n = 0 && t.scalar <> Index then
    invalid_arg "only index dtype can use zero-length vectors";
  if n = 1 || t.scalar = Void then t else { t with count = n }

type bound =
  [ `Bool of bool | `SInt of int64 | `UInt of int64 | `Float of float ]

let err_void_bounds = "void has no numeric bounds"

let min t =
  let b = scalar_bitsize t.scalar in
  match t.scalar with
  | Bool -> `Bool false
  | Uint8 | Uint16 | Uint32 | Uint64 -> `UInt 0L
  | Index -> `SInt Int64.min_int
  | Int8 | Int16 | Int32 | Int64 ->
      if b >= 64 then `SInt Int64.min_int
      else `SInt Int64.(neg (shift_left 1L (b - 1)))
  | Float16 | Bfloat16 | Float32 | Float64 | Fp8e4m3 | Fp8e5m2 ->
      `Float neg_infinity
  | Void -> invalid_arg err_void_bounds

let max t =
  let b = scalar_bitsize t.scalar in
  match t.scalar with
  | Bool -> `Bool true
  | Uint8 | Uint16 | Uint32 | Uint64 ->
      if b >= 64 then `UInt Int64.minus_one
      else `UInt Int64.(sub (shift_left 1L b) 1L)
  | Index -> `SInt Int64.max_int
  | Int8 | Int16 | Int32 | Int64 ->
      if b >= 64 then `SInt Int64.max_int
      else `SInt Int64.(sub (shift_left 1L (b - 1)) 1L)
  | Float16 | Bfloat16 | Float32 | Float64 | Fp8e4m3 | Fp8e5m2 ->
      `Float infinity
  | Void -> invalid_arg err_void_bounds

(* Type promotion lattice (JAX JEP-9407). Promotion is total: any pair of
   numeric types has a common supertype, at the cost of some lossy edges. *)

let promo_lattice =
  [ Bool, [ Int8; Uint8 ];
    Int8, [ Int16 ];       Int16, [ Int32 ];
    Int32, [ Int64 ];      Int64, [ Uint64 ];
    Uint8, [ Int16; Uint16 ];
    Uint16, [ Int32; Uint32 ];
    Uint32, [ Int64; Uint64 ];
    Uint64, [ Fp8e4m3; Fp8e5m2 ];
    Fp8e4m3, [ Float16; Bfloat16 ];
    Fp8e5m2, [ Float16; Bfloat16 ];
    Float16, [ Float32 ];  Bfloat16, [ Float32 ];
    Float32, [ Float64 ] ]

module Scalar_set = Set.Make (struct
  type t = scalar
  let compare = Stdlib.compare
end)

let ancestor_cache : (scalar, Scalar_set.t) Hashtbl.t = Hashtbl.create 16

let rec scalar_ancestors s =
  match Hashtbl.find_opt ancestor_cache s with
  | Some set -> set
  | None ->
      let parents = Option.value ~default:[] (List.assoc_opt s promo_lattice) in
      let set =
        List.fold_left
          (fun acc p -> Scalar_set.union acc (scalar_ancestors p))
          (Scalar_set.singleton s) parents
      in
      Hashtbl.add ancestor_cache s set;
      set

let scalar_compare a b =
  let c = Int.compare (scalar_priority a) (scalar_priority b) in
  if c <> 0 then c
  else
    let c = Int.compare (scalar_bitsize a) (scalar_bitsize b) in
    if c <> 0 then c else Stdlib.compare a b

let min_by_priority scalars =
  Scalar_set.fold
    (fun s best ->
      match best with
      | None -> Some s
      | Some b when scalar_compare s b < 0 -> Some s
      | _ -> best)
    scalars None

(* Find the least upper bound of a list of dtypes in the promotion lattice.
   Computes the ancestor set of each scalar (all types it can promote to),
   intersects them, and picks the smallest element by priority/bitsize.
   This mirrors NumPy/JAX-style type promotion. *)
let least_upper_dtype dts =
  if List.exists (fun d -> d.scalar = Index) dts then
    invalid_arg "Index does not participate in dtype promotion";
  match dts with
  | [] -> invalid_arg "least_upper_dtype requires at least one dtype"
  | [ d ] -> scalar_of d
  | first :: rest ->
      let intersection =
        List.fold_left
          (fun acc d -> Scalar_set.inter acc (scalar_ancestors d.scalar))
          (scalar_ancestors first.scalar) rest
      in
      match min_by_priority intersection with
      | Some s -> of_scalar s
      | None -> invalid_arg "least_upper_dtype: no common type in promotion lattice"

let least_upper_float dt =
  if is_float dt then scalar_of dt
  else least_upper_dtype [ scalar_of dt; float32 ]

let can_lossless_cast dt0 dt1 =
  let s0 = dt0.scalar and s1 = dt1.scalar in
  s0 = s1 || s0 = Bool ||
  match s1 with
  | Index ->
      List.mem s0 [ Uint8; Uint16; Uint32; Uint64; Int8; Int16; Int32; Int64 ]
  | Float64 ->
      List.mem s0
        [ Float32; Float16; Bfloat16; Fp8e4m3; Fp8e5m2;
          Uint32; Uint16; Uint8; Int32; Int16; Int8 ]
  | Float32 ->
      List.mem s0
        [ Float16; Bfloat16; Fp8e4m3; Fp8e5m2; Uint16; Uint8; Int16; Int8 ]
  | Float16 -> List.mem s0 [ Fp8e4m3; Fp8e5m2; Uint8; Int8 ]
  | Uint64 -> List.mem s0 [ Uint32; Uint16; Uint8 ]
  | Uint32 -> List.mem s0 [ Uint16; Uint8 ]
  | Uint16 -> s0 = Uint8
  | Int64 -> List.mem s0 [ Uint32; Uint16; Uint8; Int32; Int16; Int8 ]
  | Int32 -> List.mem s0 [ Uint16; Uint8; Int16; Int8 ]
  | Int16 -> List.mem s0 [ Uint8; Int8 ]
  | _ -> false

let sum_acc_dtype dt =
  if dt.scalar = Index then invalid_arg "sum_acc_dtype does not accept index dtype";
  let dt = scalar_of dt in
  if is_unsigned dt then least_upper_dtype [ dt; uint32 ]
  else if is_int dt || is_bool dt then least_upper_dtype [ dt; int32 ]
  else least_upper_dtype [ dt; float32 ]

let finfo dt =
  match dt.scalar with
  | Float16 -> 5, 10   | Bfloat16 -> 8, 7
  | Float32 -> 8, 23   | Float64 -> 11, 52
  | Fp8e5m2 -> 5, 2    | Fp8e4m3 -> 4, 3
  | _ -> invalid_arg "finfo expects a floating-point dtype"

(* Comparison *)

let equal a b = a.scalar = b.scalar && a.count = b.count

let compare a b =
  let c = scalar_compare a.scalar b.scalar in
  if c <> 0 then c else Int.compare a.count b.count

let ptr_equal a b =
  equal a.base b.base && a.addrspace = b.addrspace
  && a.v = b.v && a.size = b.size

let ptr_compare a b =
  let ( |? ) c f = if c <> 0 then c else f () in
  scalar_compare a.base.scalar b.base.scalar |? fun () ->
  Int.compare a.base.count b.base.count |? fun () ->
  Stdlib.compare a.addrspace b.addrspace |? fun () ->
  Int.compare a.v b.v |? fun () -> Int.compare a.size b.size

let any_equal a b =
  match a, b with
  | T a, T b -> equal a b
  | P a, P b -> ptr_equal a b
  | _ -> false

let any_compare a b =
  match a, b with
  | T a, T b -> compare a b
  | P a, P b -> ptr_compare a b
  | T _, P _ -> -1
  | P _, T _ -> 1

(* Formatting *)

let scalar_to_string = function
  | Void -> "void"   | Bool -> "bool"   | Index -> "index"
  | Int8 -> "i8"     | Int16 -> "i16"   | Int32 -> "i32"   | Int64 -> "i64"
  | Uint8 -> "u8"    | Uint16 -> "u16"  | Uint32 -> "u32"  | Uint64 -> "u64"
  | Float16 -> "f16" | Bfloat16 -> "bf16"
  | Float32 -> "f32" | Float64 -> "f64"
  | Fp8e4m3 -> "fp8e4m3" | Fp8e5m2 -> "fp8e5m2"

let to_string t =
  let s = scalar_to_string t.scalar in
  if t.count = 1 then s else Printf.sprintf "%s×%d" s t.count

let addr_space_to_string = function
  | Global -> "global" | Local -> "local" | Reg -> "reg"

let pp_addr_space fmt a = Format.pp_print_string fmt (addr_space_to_string a)

let ptr_to_string p =
  let vec = if p.v = 1 then "" else Printf.sprintf ".vec(%d)" p.v in
  Printf.sprintf "%s*%s [%s]" (to_string p.base) vec
    (addr_space_to_string p.addrspace)

let pp_scalar fmt s = Format.pp_print_string fmt (scalar_to_string s)
let pp fmt t = Format.pp_print_string fmt (to_string t)
let pp_ptr fmt p = Format.pp_print_string fmt (ptr_to_string p)

let pp_any fmt = function
  | T dt -> pp fmt dt
  | P p -> pp_ptr fmt p

(* C type names *)

let scalar_cname = function
  | Void -> "void"          | Bool -> "bool"           | Index -> "index"
  | Int8 -> "signed char"   | Int16 -> "short"
  | Int32 -> "int"          | Int64 -> "long"
  | Uint8 -> "unsigned char"  | Uint16 -> "unsigned short"
  | Uint32 -> "unsigned int"  | Uint64 -> "unsigned long"
  | Float16 -> "half"       | Bfloat16 -> "__bf16"
  | Float32 -> "float"      | Float64 -> "double"
  | Fp8e4m3 -> "float8_e4m3"  | Fp8e5m2 -> "float8_e5m2"

(* FP conversion *)

let float_to_fp16 x =
  if Float.is_nan x then Float.nan
  else if Float.is_infinite x then x
  else if x = 0.0 then x
  else
    let bits = Int64.bits_of_float x in
    let sign = Int64.logand (Int64.shift_right_logical bits 63) 1L in
    let exp =
      Int64.to_int (Int64.logand (Int64.shift_right_logical bits 52) 0x7FFL)
    in
    let mant = Int64.logand bits 0xFFFFFFFFFFFFFL in
    let unbiased = exp - 1023 in
    if unbiased > 15 then
      if sign = 1L then Float.neg Float.infinity else Float.infinity
    else if unbiased < -24 then if sign = 1L then -0.0 else 0.0
    else
      let fp16_sign = Int64.shift_left sign 15 in
      let fp16_bits =
        if unbiased < -14 then begin
          let shift = -14 - unbiased in
          let full_mant = Int64.logor mant 0x10000000000000L in
          let total_shift = 42 + shift in
          let shifted = Int64.shift_right_logical full_mant total_shift in
          let round_bit =
            Int64.to_int
              (Int64.logand
                 (Int64.shift_right_logical full_mant (total_shift - 1))
                 1L)
          in
          let sticky =
            let mask = Int64.sub (Int64.shift_left 1L (total_shift - 1)) 1L in
            if Int64.logand full_mant mask <> 0L then 1 else 0
          in
          let rounded =
            if round_bit = 1 && (sticky = 1 || Int64.logand shifted 1L <> 0L)
            then Int64.add shifted 1L
            else shifted
          in
          Int64.logor fp16_sign rounded
        end
        else begin
          let biased16 = unbiased + 15 in
          let shifted_mant = Int64.shift_right_logical mant 42 in
          let round_bit =
            Int64.to_int (Int64.logand (Int64.shift_right_logical mant 41) 1L)
          in
          let sticky =
            if Int64.logand mant 0x1FFFFFFFFFFL <> 0L then 1 else 0
          in
          let rounded =
            if
              round_bit = 1 && (sticky = 1 || Int64.logand shifted_mant 1L <> 0L)
            then Int64.add shifted_mant 1L
            else shifted_mant
          in
          let final_exp, final_mant =
            if rounded > 0x3FFL then (biased16 + 1, 0L) else (biased16, rounded)
          in
          if final_exp > 30 then Int64.logor fp16_sign 0x7C00L
          else
            Int64.logor fp16_sign
              (Int64.logor (Int64.of_int (final_exp lsl 10)) final_mant)
        end
      in
      let fp16_exp =
        Int64.to_int
          (Int64.logand (Int64.shift_right_logical fp16_bits 10) 0x1FL)
      in
      let fp16_mant = Int64.logand fp16_bits 0x3FFL in
      let f =
        if fp16_exp = 0x1F then
          if fp16_mant = 0L then Float.infinity else Float.nan
        else if fp16_exp = 0 then Float.ldexp (Int64.to_float fp16_mant) (-24)
        else
          Float.ldexp
            (Int64.to_float (Int64.logor fp16_mant 0x400L))
            (fp16_exp - 25)
      in
      if sign = 1L then Float.neg f else f

let float_to_bf16 x =
  if not (Float.is_finite x) then x
  else
    let u = Int32.bits_of_float x in
    let u =
      Int32.logand
        (Int32.add u
           (Int32.add 0x7FFFl
              (Int32.logand (Int32.shift_right_logical u 16) 1l)))
        0xFFFF_0000l
    in
    Int32.float_of_bits u

type fp8_params = {
  exp_bias : int; sig_bits : int; mantissa_mask : int;
  mindenorm_o2 : int64; overflow_threshold : int64;
  maxnorm : int; minnorm : int64;
}

let fp8e4m3_params =
  { exp_bias = 7; sig_bits = 4; mantissa_mask = 0x7;
    mindenorm_o2 = 0x3F50000000000000L; overflow_threshold = 0x407D000000000000L;
    maxnorm = 0x7E; minnorm = 0x3F90000000000000L }

let fp8e5m2_params =
  { exp_bias = 15; sig_bits = 3; mantissa_mask = 0x3;
    mindenorm_o2 = 0x3EE0000000000000L;
    overflow_threshold = Int64.sub 0x40EE000000000000L 1L;
    maxnorm = 0x7B; minnorm = 0x3F10000000000000L }

let float_to_fp8 scalar x =
  match scalar with
  | Fp8e4m3 when not (Float.is_finite x) ->
      if Float.copy_sign 1.0 x > 0.0 then 0x7f else 0xff
  | Fp8e5m2 when Float.is_infinite x ->
      if Float.copy_sign 1.0 x > 0.0 then 0x7c else 0xfc
  | Fp8e4m3 | Fp8e5m2 ->
      let p = match scalar with
        | Fp8e4m3 -> fp8e4m3_params | _ -> fp8e5m2_params
      in
      let xbits = Int64.bits_of_float x in
      let half_ulp = Int64.shift_left 1L (53 - p.sig_bits - 1) in
      let sign =
        Int64.to_int (Int64.logand (Int64.shift_right_logical xbits 63) 1L)
        lsl 7
      in
      let raw_exp =
        Int64.to_int (Int64.logand (Int64.shift_right_logical xbits 52) 0x7FFL)
      in
      let exp = raw_exp - 1023 + p.exp_bias in
      let mantissa =
        Int64.to_int
          (Int64.logand
             (Int64.shift_right_logical xbits (53 - p.sig_bits))
             (Int64.of_int p.mantissa_mask))
      in
      let absx = Int64.logand xbits 0x7FFFFFFFFFFFFFFFL in
      let res =
        if Int64.compare absx p.mindenorm_o2 <= 0 then 0
        else if Int64.compare absx 0x7FF0000000000000L > 0 then
          if scalar = Fp8e4m3 then 0x7F else 0x7E lor mantissa
        else if Int64.compare absx p.overflow_threshold > 0 then p.maxnorm
        else if Int64.compare absx p.minnorm >= 0 then begin
          let base = (exp lsl (p.sig_bits - 1)) lor mantissa in
          let round_mask = Int64.sub (Int64.shift_left half_ulp 1) 1L in
          let round_bits = Int64.logand xbits round_mask in
          if Int64.compare round_bits half_ulp > 0
             || (round_bits = half_ulp && mantissa land 1 <> 0)
          then base + 1 else base
        end
        else begin
          let shift = 1 - exp in
          let mant_with_implicit = mantissa lor (1 lsl (p.sig_bits - 1)) in
          let base = mant_with_implicit asr shift in
          let round_bits =
            Int64.logand
              (Int64.logor xbits (Int64.shift_left 1L 52))
              (Int64.sub (Int64.shift_left half_ulp (shift + 1)) 1L)
          in
          let threshold = Int64.shift_left half_ulp shift in
          if Int64.compare round_bits threshold > 0
             || (round_bits = threshold && base land 1 <> 0)
          then base + 1 else base
        end
      in
      res lor sign
  | _ -> invalid_arg "float_to_fp8: expected Fp8e4m3 or Fp8e5m2"

let fp8_to_float scalar x =
  match scalar with
  | Fp8e4m3 | Fp8e5m2 ->
      let ur = x lsl 8 in
      let ur =
        if scalar = Fp8e5m2 && ur land 0x7FFF > 0x7C00 then 0x7FFF
        else if scalar = Fp8e4m3 then begin
          let sign = ur land 0x8000 in
          let exponent = ((ur land 0x7800) asr 1) + 0x2000 in
          let mantissa_init = (ur land 0x0700) asr 1 in
          let absx = x land 0x7F in
          if absx = 0x7F then 0x7FFF
          else if exponent = 0x2000 then begin
            if mantissa_init <> 0 then begin
              let rec normalize m e =
                if m land 0x0400 <> 0 then (m, e)
                else normalize (m lsl 1) (e - 0x0400)
              in
              let m, e = normalize (mantissa_init lsl 1) exponent in
              sign lor e lor (m land 0x03FF)
            end
            else sign
          end
          else sign lor exponent lor mantissa_init
        end
        else ur
      in
      let fp16_sign = (ur asr 15) land 1 in
      let fp16_exp = (ur asr 10) land 0x1F in
      let fp16_mant = ur land 0x3FF in
      let f =
        if fp16_exp = 0x1F then
          if fp16_mant = 0 then Float.infinity else Float.nan
        else if fp16_exp = 0 then Float.ldexp (Float.of_int fp16_mant) (-24)
        else Float.ldexp (Float.of_int (fp16_mant + 1024)) (fp16_exp - 25)
      in
      if fp16_sign = 1 then Float.neg f else f
  | _ -> invalid_arg "fp8_to_float: expected Fp8e4m3 or Fp8e5m2"

let truncate_float t x =
  match t.scalar with
  | Float64 -> x
  | Float32 -> Int32.float_of_bits (Int32.bits_of_float x)
  | Float16 -> float_to_fp16 x
  | Bfloat16 -> float_to_bf16 x
  | Fp8e4m3 | Fp8e5m2 -> fp8_to_float t.scalar (float_to_fp8 t.scalar x)
  | _ -> invalid_arg "truncate_float: expected a floating-point dtype"

let truncate_int t x =
  let b = scalar_bitsize t.scalar in
  match t.scalar with
  | Bool -> if x <> 0 then 1 else 0
  | Uint8 | Uint16 | Uint32 | Uint64 ->
      if b >= Sys.int_size then x else x land ((1 lsl b) - 1)
  | Int8 | Int16 | Int32 | Int64 | Index ->
      if b >= Sys.int_size then x
      else
        let mask = (1 lsl b) - 1 in
        let unsigned = x land mask in
        if unsigned land (1 lsl (b - 1)) <> 0 then unsigned lor lnot mask
        else unsigned
  | _ -> invalid_arg "truncate_int: expected an integer or bool dtype"
