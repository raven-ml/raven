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

type ptr = {
  base : t;
  addrspace : addr_space;
  v : int;
  size : int;
  image : (int * int) option;
}

(* ───── Constructors ───── *)

let void = { scalar = Void; count = 1 }
let bool = { scalar = Bool; count = 1 }
let int8 = { scalar = Int8; count = 1 }
let int16 = { scalar = Int16; count = 1 }
let int32 = { scalar = Int32; count = 1 }
let int64 = { scalar = Int64; count = 1 }
let uint8 = { scalar = Uint8; count = 1 }
let uint16 = { scalar = Uint16; count = 1 }
let uint32 = { scalar = Uint32; count = 1 }
let uint64 = { scalar = Uint64; count = 1 }
let float16 = { scalar = Float16; count = 1 }
let bfloat16 = { scalar = Bfloat16; count = 1 }
let float32 = { scalar = Float32; count = 1 }
let float64 = { scalar = Float64; count = 1 }
let fp8e4m3 = { scalar = Fp8e4m3; count = 1 }
let fp8e5m2 = { scalar = Fp8e5m2; count = 1 }
let index = { scalar = Index; count = 1 }
let default_float = float32
let default_int = int32

(* ───── Predicates ───── *)

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

(* ───── Properties ───── *)

let scalar_bitsize = function
  | Void -> 0
  | Bool -> 1
  | Int8 | Uint8 | Fp8e4m3 | Fp8e5m2 -> 8
  | Int16 | Uint16 | Float16 | Bfloat16 -> 16
  | Int32 | Uint32 | Float32 -> 32
  | Int64 | Uint64 | Float64 -> 64
  (* Sentinel value: Index is symbolic, not a fixed-width machine type. 800 bits
     ensures it won't accidentally match any real type's bitsize. *)
  | Index -> 800

let bitsize t = scalar_bitsize t.scalar * t.count
let itemsize t = (bitsize t + 7) / 8

(* Priority determines tie-breaking in least_upper_dtype when multiple types
   have the same promotion distance. The actual promotion result is determined
   by the promo_lattice: both Float16 and Bfloat16 promote to Float32 (not to
   each other), so mixed f16/bf16 operations become Float32. *)
let scalar_priority = function
  | Void | Index -> -1
  | Bool -> 0
  | Int8 -> 1
  | Uint8 -> 2
  | Int16 -> 3
  | Uint16 -> 4
  | Int32 -> 5
  | Uint32 -> 6
  | Int64 -> 7
  | Uint64 -> 8
  | Fp8e4m3 -> 9
  | Fp8e5m2 -> 10
  | Float16 -> 11
  | Bfloat16 -> 12
  | Float32 -> 13
  | Float64 -> 14

let priority t = scalar_priority t.scalar

(* ───── Operations ───── *)

let scalar_of t = { t with count = 1 }

let vec t n =
  if t.count <> 1 then
    invalid_arg (Printf.sprintf "can't vectorize type with count %d" t.count);
  if n < 0 then
    invalid_arg (Printf.sprintf "vector size must be >= 0, got %d" n);
  (* index.vec(0) represents empty shape vectors for scalar tensors. *)
  if n = 0 && t.scalar <> Index then
    invalid_arg "only index dtype can use zero-length vectors";
  if n = 1 || t.scalar = Void then t else { t with count = n }

type bound =
  [ `Bool of bool | `SInt of int64 | `UInt of int64 | `Float of float ]

let min t =
  let b = scalar_bitsize t.scalar in
  match t.scalar with
  | Bool -> `Bool false
  | Uint8 | Uint16 | Uint32 | Uint64 -> `UInt 0L
  (* Index uses 800 bits as a non-machine sentinel; we approximate with Int64
     bounds since backends lower Index to int32/int64. *)
  | Index -> `SInt Int64.min_int
  | Int8 | Int16 | Int32 | Int64 ->
      if b = 64 then `SInt Int64.min_int
      else `SInt Int64.(neg (shift_left 1L (b - 1)))
  | Float16 | Bfloat16 | Float32 | Float64 | Fp8e4m3 | Fp8e5m2 ->
      `Float neg_infinity
  | Void -> invalid_arg "void has no numeric bounds"

let max t =
  let b = scalar_bitsize t.scalar in
  match t.scalar with
  | Bool -> `Bool true
  | Uint8 | Uint16 | Uint32 | Uint64 ->
      if b = 64 then `UInt Int64.minus_one
      else `UInt Int64.(sub (shift_left 1L b) 1L)
  (* See min for rationale on Int64 approximation. *)
  | Index -> `SInt Int64.max_int
  | Int8 | Int16 | Int32 | Int64 ->
      if b = 64 then `SInt Int64.max_int
      else `SInt Int64.(sub (shift_left 1L (b - 1)) 1L)
  | Float16 | Bfloat16 | Float32 | Float64 | Fp8e4m3 | Fp8e5m2 ->
      `Float infinity
  | Void -> invalid_arg "void has no numeric bounds"

(* ───── Type Promotion ───── *)

(* Type promotion lattice based on JAX JEP-9407. See:
   https://jax.readthedocs.io/en/latest/jep/9407-type-promotion.html

   Promotion is deterministic and total: any pair of numeric types has a common
   supertype. This requires some lossy edges (e.g., Int64 -> Uint64 loses
   negative values, Uint64 -> Fp8 is extremely lossy). The alternative would be
   failing with type errors; we prioritize ergonomics over precision guarantees.

   Index is excluded: it's an IR-level concept for loop counters and memory
   addressing, lowered to int32/int64 by backends. *)
let promo_lattice =
  [
    (Bool, [ Int8; Uint8 ]);
    (Int8, [ Int16 ]);
    (Int16, [ Int32 ]);
    (Int32, [ Int64 ]);
    (Int64, [ Uint64 ]);
    (Uint8, [ Int16; Uint16 ]);
    (Uint16, [ Int32; Uint32 ]);
    (Uint32, [ Int64; Uint64 ]);
    (Uint64, [ Fp8e4m3; Fp8e5m2 ]);
    (Fp8e4m3, [ Float16; Bfloat16 ]);
    (Fp8e5m2, [ Float16; Bfloat16 ]);
    (Float16, [ Float32 ]);
    (Bfloat16, [ Float32 ]);
    (Float32, [ Float64 ]);
  ]

module Scalar_set = Set.Make (struct
  type t = scalar

  let compare = Stdlib.compare
end)

(* Write-once memo table, populated lazily. Not thread-safe; assumes
   single-domain init. *)
let parent_cache : (scalar, Scalar_set.t) Hashtbl.t = Hashtbl.create 16

let rec scalar_ancestors scalar =
  match Hashtbl.find_opt parent_cache scalar with
  | Some cached -> cached
  | None ->
      let parents =
        List.assoc_opt scalar promo_lattice |> Option.value ~default:[]
      in
      let ancestors =
        List.fold_left
          (fun acc p -> Scalar_set.union acc (scalar_ancestors p))
          (Scalar_set.singleton scalar)
          parents
      in
      Hashtbl.add parent_cache scalar ancestors;
      ancestors

let scalar_compare a b =
  match Int.compare (scalar_priority a) (scalar_priority b) with
  | 0 -> (
      match Int.compare (scalar_bitsize a) (scalar_bitsize b) with
      | 0 -> Stdlib.compare a b
      | c -> c)
  | c -> c

let min_by_priority scalars =
  Scalar_set.fold
    (fun s acc ->
      match acc with
      | None -> Some s
      | Some best when scalar_compare s best < 0 -> Some s
      | _ -> acc)
    scalars None

let least_upper_dtype dts =
  if List.exists (fun d -> d.scalar = Index) dts then
    invalid_arg "Index does not participate in dtype promotion";
  match dts with
  | [] -> invalid_arg "least_upper_dtype requires at least one dtype"
  | [ d ] -> scalar_of d
  | first :: rest -> (
      let intersection =
        List.fold_left
          (fun acc d -> Scalar_set.inter acc (scalar_ancestors d.scalar))
          (scalar_ancestors first.scalar)
          rest
      in
      match min_by_priority intersection with
      | Some scalar -> { scalar; count = 1 }
      | None ->
          (* Lattice is connected; unreachable for valid promotable scalars. *)
          invalid_arg "least_upper_dtype: no common type in promotion lattice")

let least_upper_float dt =
  (* Keep promotion deterministic for scalar math ops: non-float inputs promote
     through float32, which is the smallest broadly supported IEEE float in
     backends. *)
  if is_float dt then scalar_of dt
  else least_upper_dtype [ scalar_of dt; float32 ]

(* Unlike promotion (which finds a common supertype), this checks whether every
   value of dt0 is exactly representable in dt1. Bool casts losslessly to
   anything. *)
let can_lossless_cast dt0 dt1 =
  let s0 = dt0.scalar and s1 = dt1.scalar in
  if s0 = s1 || s0 = Bool then true
  else
    match s1 with
    | Index ->
        List.mem s0 [ Uint8; Uint16; Uint32; Uint64; Int8; Int16; Int32; Int64 ]
    | Float64 ->
        List.mem s0
          [
            Float32;
            Float16;
            Bfloat16;
            Fp8e4m3;
            Fp8e5m2;
            Uint32;
            Uint16;
            Uint8;
            Int32;
            Int16;
            Int8;
          ]
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
  let dt = scalar_of dt in
  if dt.scalar = Index then
    invalid_arg "sum_acc_dtype does not accept index dtype";
  (* Accumulators widen narrow integer domains to 32-bit to reduce overflow
     while keeping hardware-friendly integer arithmetic. Floats keep their lane
     type. *)
  if is_unsigned dt then least_upper_dtype [ dt; uint32 ]
  else if is_int dt || is_bool dt then least_upper_dtype [ dt; int32 ]
  else least_upper_dtype [ dt; float32 ]

(* Returns (exponent_bits, mantissa_bits) for IEEE 754 float types. *)
let finfo dt =
  match (scalar_of dt).scalar with
  | Float16 -> (5, 10)
  | Bfloat16 -> (8, 7)
  | Float32 -> (8, 23)
  | Float64 -> (11, 52)
  | Fp8e5m2 -> (5, 2)
  | Fp8e4m3 -> (4, 3)
  | _ -> invalid_arg "finfo expects a floating-point dtype"

(* ───── Comparison ───── *)

let equal a b = a.scalar = b.scalar && a.count = b.count

let compare a b =
  match scalar_compare a.scalar b.scalar with
  | 0 -> Int.compare a.count b.count
  | c -> c

(* ───── Pretty Printing ───── *)

let scalar_to_string = function
  | Void -> "void"
  | Bool -> "bool"
  | Int8 -> "i8"
  | Int16 -> "i16"
  | Int32 -> "i32"
  | Int64 -> "i64"
  | Uint8 -> "u8"
  | Uint16 -> "u16"
  | Uint32 -> "u32"
  | Uint64 -> "u64"
  | Float16 -> "f16"
  | Bfloat16 -> "bf16"
  | Float32 -> "f32"
  | Float64 -> "f64"
  | Fp8e4m3 -> "fp8e4m3"
  | Fp8e5m2 -> "fp8e5m2"
  | Index -> "index"

let to_string t =
  let s = scalar_to_string t.scalar in
  if t.count = 1 then s else Printf.sprintf "%s×%d" s t.count

let pp_scalar fmt s = Format.pp_print_string fmt (scalar_to_string s)
let pp fmt t = Format.pp_print_string fmt (to_string t)

let addr_space_to_string = function
  | Global -> "global"
  | Local -> "local"
  | Reg -> "reg"

let pp_addr_space fmt a = Format.pp_print_string fmt (addr_space_to_string a)

(* ───── Pointer Operations ───── *)

module Ptr = struct
  let create t ?(size = -1) ?(addrspace = Global) ?(v = 1) ?image () =
    if v < 1 then
      invalid_arg (Printf.sprintf "pointer vcount must be >= 1, got %d" v);
    if Option.is_some image && addrspace <> Global then
      invalid_arg "image pointers must be in global address space";
    { base = t; addrspace; v; size; image }

  let vec p n =
    if n < 1 then
      invalid_arg (Printf.sprintf "pointer vcount must be >= 1, got %d" n);
    { p with v = n }

  let scalar p = if p.v = 1 then p else { p with v = 1 }
  let vcount p = p.v

  let equal a b =
    equal a.base b.base && a.addrspace = b.addrspace && a.v = b.v
    && a.size = b.size && a.image = b.image

  let compare a b =
    let ( |? ) c f = if c <> 0 then c else f () in
    scalar_compare a.base.scalar b.base.scalar |? fun () ->
    Int.compare a.base.count b.base.count |? fun () ->
    Stdlib.compare a.addrspace b.addrspace |? fun () ->
    Int.compare a.v b.v |? fun () ->
    Int.compare a.size b.size |? fun () -> Stdlib.compare a.image b.image

  let to_string p =
    let base = to_string p.base in
    let vec_suffix = if p.v = 1 then "" else Printf.sprintf ".vec(%d)" p.v in
    Printf.sprintf "%s*%s [%s]" base vec_suffix
      (addr_space_to_string p.addrspace)

  let pp fmt p = Format.pp_print_string fmt (to_string p)
end

(* ───── C Type Names ───── *)

(* C-language type names used by codegen renderers as fallback when no
   device-specific type_map override exists. *)
let scalar_cname = function
  | Void -> "void"
  | Bool -> "bool"
  | Int8 -> "signed char"
  | Int16 -> "short"
  | Int32 -> "int"
  | Int64 -> "long"
  | Uint8 -> "unsigned char"
  | Uint16 -> "unsigned short"
  | Uint32 -> "unsigned int"
  | Uint64 -> "unsigned long"
  | Float16 -> "half"
  | Bfloat16 -> "__bf16"
  | Float32 -> "float"
  | Float64 -> "double"
  | Fp8e4m3 -> "float8_e4m3"
  | Fp8e5m2 -> "float8_e5m2"
  | Index -> "index"

(* ───── FP Conversion ───── *)

(* Converts float64 to fp16 precision by extracting IEEE 754 binary64 fields,
   re-biasing the exponent for the fp16 range, and applying
   round-to-nearest-even. Returns a float64 that holds the exact
   fp16-representable value. *)
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
          (* Denormal range: no implicit leading 1 in the stored mantissa. *)
          let shift = -14 - unbiased in
          let full_mant = Int64.logor mant 0x10000000000000L in
          let total_shift = 42 + shift in
          let shifted = Int64.shift_right_logical full_mant total_shift in
          (* Round-to-nearest-even. *)
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
          (* Normal range. *)
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
      (* Decode fp16 bits back to float64. *)
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
  (* Rounds to bfloat16 precision by zeroing the lower 16 bits of the float32
     representation with round-to-nearest-even. Int32.bits_of_float converts
     float64 -> float32 bits, giving us the right starting point. *)
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

let float_to_fp8 scalar x =
  match scalar with
  | Fp8e4m3 when not (Float.is_finite x) ->
      if Float.copy_sign 1.0 x > 0.0 then 0x7f else 0xff
  | Fp8e5m2 when Float.is_infinite x ->
      if Float.copy_sign 1.0 x > 0.0 then 0x7c else 0xfc
  | Fp8e4m3 | Fp8e5m2 ->
      let ( exp_bias,
            sig_bits,
            mantissa_mask,
            mindenorm_o2,
            overflow_threshold,
            maxnorm,
            minnorm ) =
        match scalar with
        | Fp8e4m3 ->
            ( 7,
              4,
              0x7,
              0x3F50000000000000L,
              0x407D000000000000L,
              0x7E,
              0x3F90000000000000L )
        | Fp8e5m2 ->
            ( 15,
              3,
              0x3,
              0x3EE0000000000000L,
              Int64.sub 0x40EE000000000000L 1L,
              0x7B,
              0x3F10000000000000L )
        | _ -> assert false (* guarded by outer match *)
      in
      let xbits = Int64.bits_of_float x in
      let fp8_dp_half_ulp = Int64.shift_left 1L (53 - sig_bits - 1) in
      let sign =
        Int64.to_int (Int64.logand (Int64.shift_right_logical xbits 63) 1L)
        lsl 7
      in
      let raw_exp =
        Int64.to_int (Int64.logand (Int64.shift_right_logical xbits 52) 0x7FFL)
      in
      let exp = raw_exp - 1023 + exp_bias in
      let mantissa =
        Int64.to_int
          (Int64.logand
             (Int64.shift_right_logical xbits (53 - sig_bits))
             (Int64.of_int mantissa_mask))
      in
      let absx = Int64.logand xbits 0x7FFFFFFFFFFFFFFFL in
      let res =
        if Int64.compare absx mindenorm_o2 <= 0 then 0
        else if Int64.compare absx 0x7FF0000000000000L > 0 then
          if scalar = Fp8e4m3 then 0x7F else 0x7E lor mantissa
        else if Int64.compare absx overflow_threshold > 0 then maxnorm
        else if Int64.compare absx minnorm >= 0 then begin
          let base = (exp lsl (sig_bits - 1)) lor mantissa in
          let round_mask = Int64.sub (Int64.shift_left fp8_dp_half_ulp 1) 1L in
          let round_bits = Int64.logand xbits round_mask in
          if
            Int64.compare round_bits fp8_dp_half_ulp > 0
            || (round_bits = fp8_dp_half_ulp && mantissa land 1 <> 0)
          then base + 1
          else base
        end
        else begin
          let shift = 1 - exp in
          let mant_with_implicit = mantissa lor (1 lsl (sig_bits - 1)) in
          let base = mant_with_implicit asr shift in
          let round_bits =
            Int64.logand
              (Int64.logor xbits (Int64.shift_left 1L 52))
              (Int64.sub (Int64.shift_left fp8_dp_half_ulp (shift + 1)) 1L)
          in
          let threshold = Int64.shift_left fp8_dp_half_ulp shift in
          if
            Int64.compare round_bits threshold > 0
            || (round_bits = threshold && base land 1 <> 0)
          then base + 1
          else base
        end
      in
      res lor sign
  | _ -> invalid_arg "float_to_fp8: expected Fp8e4m3 or Fp8e5m2"

(* Converts fp8 byte to float by first expanding to fp16 bit layout (shift left
   8), normalizing exponent/mantissa for fp8e4m3's narrower exponent, then
   decoding the resulting fp16 bits to float64. *)
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
      (* Decode fp16 bits: sign(1) exp(5) mant(10). *)
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
