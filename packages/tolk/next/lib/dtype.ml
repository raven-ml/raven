(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Constants *)

type value = [ `Bool of bool | `Int of Bigint.t | `Float of float ]
type const = [ value | `Invalid ]

let nan = Int64.float_of_bits 0x7FF8_0000_0000_0000L

let equal_const (c0 : [< const ]) (c1 : [< const ]) =
  match ((c0 :> const), (c1 :> const)) with
  | `Bool b0, `Bool b1 -> Bool.equal b0 b1
  | `Int n0, `Int n1 -> Bigint.equal n0 n1
  | `Float x0, `Float x1 ->
      Int64.equal (Int64.bits_of_float x0) (Int64.bits_of_float x1)
  | `Invalid, `Invalid -> true
  | _ -> false

(* Hashtbl.hash would hash every NaN alike, and -0.0 like 0.0. *)
let hash_const (c : [< const ]) =
  match (c :> const) with
  | `Float x -> Hashtbl.hash (Int64.bits_of_float x)
  | c -> Hashtbl.hash c

(* The fewest significant digits that read back as [x], a finite float, and
   where their decimal point goes: [x] reads as 0.[digits] times 10 to the
   [point]. Just above a power of two the floats below are closer than those
   above, so digits rounded to nearest can miss [x] where their neighbour does
   not. *)
let shortest_digits x =
  let rec shortest n =
    let s = strf "%.*e" (n - 1) (Float.abs x) in
    let e = String.index s 'e' in
    let mantissa =
      Bigint.of_string
        (String.concat "" (String.split_on_char '.' (String.sub s 0 e)))
    in
    let exponent =
      int_of_string (String.sub s (e + 1) (String.length s - e - 1)) - n + 1
    in
    let reads m =
      Float.equal
        (float_of_string (strf "%se%d" (Bigint.to_string m) exponent))
        (Float.abs x)
    in
    match
      List.find_opt reads
        [ mantissa; Bigint.succ mantissa; Bigint.pred mantissa ]
    with
    | Some m -> (Bigint.to_string m, exponent)
    | None -> shortest (n + 1)
  in
  let digits, exponent = shortest 1 in
  (digits, exponent + String.length digits)

let float_repr x =
  if Float.is_nan x then "nan"
  else if not (Float.is_finite x) then
    if Float.sign_bit x then "-inf" else "inf"
  else if Float.is_integer x && Float.abs x < 1e16 then strf "%.0f.0" x
  else
    let sign = if Float.sign_bit x then "-" else "" in
    let digits, point = shortest_digits x in
    let n = String.length digits in
    if point > 16 || point < -3 then
      let exponent = point - 1 and rest = String.sub digits 1 (n - 1) in
      strf "%s%c%s%se%+03d" sign digits.[0]
        (if rest = "" then "" else ".")
        rest exponent
    else if point <= 0 then
      strf "%s0.%s%s" sign (String.make (-point) '0') digits
    else
      strf "%s%s.%s" sign
        (String.sub digits 0 point)
        (String.sub digits point (n - point))

let pp_const ppf (c : [< const ]) =
  match c with
  | `Bool b -> Format.pp_print_string ppf (if b then "True" else "False")
  | `Int n -> Bigint.pp_print ppf n
  | `Float x -> Format.pp_print_string ppf (float_repr x)
  | `Invalid -> Format.pp_print_string ppf "Invalid"

(* Address spaces *)

type addr_space = Global | Local | Reg | Alu

let addr_space_name = function
  | Global -> "GLOBAL"
  | Local -> "LOCAL"
  | Reg -> "REG"
  | Alu -> "ALU"

let pp_addr_space ppf space =
  Format.fprintf ppf "AddrSpace.%s" (addr_space_name space)

let addr_space_of_string s =
  match
    List.find_opt
      (fun space -> addr_space_name space = s)
      [ Global; Local; Reg; Alu ]
  with
  | Some space -> Ok space
  | None -> Error (strf "%S is not an address space" s)

(* Data types *)

type t =
  | Void
  | Weak_int
  | Bool
  | Int8
  | Uint8
  | Int16
  | Uint16
  | Int32
  | Uint32
  | Int64
  | Uint64
  | Weak_float
  | Fp8e4m3
  | Fp8e5m2
  | Fp8e4m3fnuz
  | Fp8e5m2fnuz
  | Float16
  | Bfloat16
  | Float32
  | Float64

type info = { priority : int; bitsize : int; name : string; fmt : char option }

let info = function
  | Void -> { priority = -1; bitsize = 0; name = "void"; fmt = None }
  | Weak_int -> { priority = 0; bitsize = 800; name = "weakint"; fmt = None }
  | Bool -> { priority = 0; bitsize = 1; name = "bool"; fmt = Some '?' }
  | Int8 -> { priority = 1; bitsize = 8; name = "signed char"; fmt = Some 'b' }
  | Uint8 ->
      { priority = 2; bitsize = 8; name = "unsigned char"; fmt = Some 'B' }
  | Int16 -> { priority = 3; bitsize = 16; name = "short"; fmt = Some 'h' }
  | Uint16 ->
      { priority = 4; bitsize = 16; name = "unsigned short"; fmt = Some 'H' }
  | Int32 -> { priority = 5; bitsize = 32; name = "int"; fmt = Some 'i' }
  | Uint32 ->
      { priority = 6; bitsize = 32; name = "unsigned int"; fmt = Some 'I' }
  | Int64 -> { priority = 7; bitsize = 64; name = "long"; fmt = Some 'q' }
  | Uint64 ->
      { priority = 8; bitsize = 64; name = "unsigned long"; fmt = Some 'Q' }
  | Weak_float ->
      { priority = 9; bitsize = 800; name = "weakfloat"; fmt = None }
  | Fp8e4m3 -> { priority = 10; bitsize = 8; name = "float8_e4m3"; fmt = None }
  | Fp8e5m2 -> { priority = 11; bitsize = 8; name = "float8_e5m2"; fmt = None }
  | Fp8e4m3fnuz ->
      { priority = 10; bitsize = 8; name = "float8_e4m3fnuz"; fmt = None }
  | Fp8e5m2fnuz ->
      { priority = 11; bitsize = 8; name = "float8_e5m2fnuz"; fmt = None }
  | Float16 -> { priority = 12; bitsize = 16; name = "half"; fmt = Some 'e' }
  | Bfloat16 -> { priority = 13; bitsize = 16; name = "__bf16"; fmt = None }
  | Float32 -> { priority = 14; bitsize = 32; name = "float"; fmt = Some 'f' }
  | Float64 -> { priority = 15; bitsize = 64; name = "double"; fmt = Some 'd' }

let priority dt = (info dt).priority
let bitsize dt = (info dt).bitsize
let itemsize dt = (bitsize dt + 7) / 8
let name dt = (info dt).name
let fmt dt = (info dt).fmt
let equal (d0 : t) d1 = d0 = d1

let compare d0 d1 =
  let i0 = info d0 and i1 = info d1 in
  match Int.compare i0.priority i1.priority with
  | 0 -> (
      match Int.compare i0.bitsize i1.bitsize with
      | 0 -> String.compare i0.name i1.name
      | c -> c)
  | c -> c

let hash (dt : t) = Hashtbl.hash dt

(* The names of each data type, the one it prints as first. *)
let names =
  [
    (Void, [ "void" ]);
    (Weak_int, [ "weakint" ]);
    (Bool, [ "bool" ]);
    (Int8, [ "char"; "int8" ]);
    (Uint8, [ "uchar"; "uint8" ]);
    (Int16, [ "short"; "int16" ]);
    (Uint16, [ "ushort"; "uint16" ]);
    (Int32, [ "int"; "int32" ]);
    (Uint32, [ "uint"; "uint32" ]);
    (Int64, [ "long"; "int64" ]);
    (Uint64, [ "ulong"; "uint64" ]);
    (Weak_float, [ "weakfloat" ]);
    (Fp8e4m3, [ "fp8e4m3" ]);
    (Fp8e5m2, [ "fp8e5m2" ]);
    (Fp8e4m3fnuz, [ "fp8e4m3fnuz" ]);
    (Fp8e5m2fnuz, [ "fp8e5m2fnuz" ]);
    (Float16, [ "half"; "float16" ]);
    (Bfloat16, [ "bfloat16" ]);
    (Float32, [ "float"; "float32" ]);
    (Float64, [ "double"; "float64" ]);
  ]

let pp ppf dt = Format.fprintf ppf "dtypes.%s" (List.hd (List.assoc dt names))

(* Predicates and groups *)

let fp8_ocp = [ Fp8e4m3; Fp8e5m2 ]
let fp8_fnuz = [ Fp8e4m3fnuz; Fp8e5m2fnuz ]
let fp8s = fp8_ocp @ fp8_fnuz
let floats = fp8s @ [ Float16; Bfloat16; Float32; Float64 ]
let uints = [ Uint8; Uint16; Uint32; Uint64 ]
let sints = [ Int8; Int16; Int32; Int64 ]
let ints = uints @ sints
let weaks = [ Weak_int; Weak_float ]
let all = floats @ ints @ [ Bool ]

let is_float = function
  | Weak_float | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz | Float16
  | Bfloat16 | Float32 | Float64 ->
      true
  | _ -> false

let is_int = function
  | Weak_int | Int8 | Uint8 | Int16 | Uint16 | Int32 | Uint32 | Int64 | Uint64
    ->
      true
  | _ -> false

let is_unsigned = function
  | Uint8 | Uint16 | Uint32 | Uint64 -> true
  | _ -> false

let is_bool dt = dt = Bool

let finfo dt =
  match dt with
  | Float16 -> (5, 10)
  | Bfloat16 -> (8, 7)
  | Float32 -> (8, 23)
  | Float64 -> (11, 52)
  | Fp8e4m3 | Fp8e4m3fnuz -> (4, 3)
  | Fp8e5m2 | Fp8e5m2fnuz -> (5, 2)
  | _ -> invalid_arg (Format.asprintf "%a is not a float of known width" pp dt)

(* Conversions *)

(* A value as an integer. A float has none: rounding it is the caller's. *)
let integer : value -> Bigint.t = function
  | `Bool b -> Bigint.of_int (Bool.to_int b)
  | `Int n -> n
  | `Float x ->
      invalid_arg (strf "%s is a float, not an integer" (float_repr x))

(* Arithmetic on values *)

module Value = struct
  type t = value

  let of_int n = `Int (Bigint.of_int n)

  let to_float : t -> float = function
    | `Bool b -> if b then 1. else 0.
    | `Int n -> Bigint.to_float n
    | `Float x -> x

  let to_z : t -> Bigint.t = function
    | `Float x when not (Float.is_finite x) ->
        invalid_arg (strf "%s has no integer value" (float_repr x))
    | `Float x -> Bigint.of_float x
    | v -> integer v

  let to_int v =
    let n = to_z v in
    if Bigint.fits_int n then Bigint.to_int n
    else invalid_arg (strf "%s does not fit an int" (Bigint.to_string n))

  let to_bool : t -> bool = function
    | `Bool b -> b
    | `Int n -> not (Bigint.equal n Bigint.zero)
    | `Float x -> x <> 0.

  (* A value as a number: a [`Bool] counts as [0] or [1]. *)
  let number : t -> [ `Int of Bigint.t | `Float of float ] = function
    | `Bool b -> `Int (Bigint.of_int (Bool.to_int b))
    | (`Int _ | `Float _) as v -> v

  (* The order of the integer [n] and the float [x], exactly, or [None] if [x]
     is NaN. *)
  let compare_int_float n x =
    if Float.is_nan x then None
    else if not (Float.is_finite x) then Some (if x > 0. then -1 else 1)
    else
      let floor = Float.floor x in
      match Bigint.compare n (Bigint.of_float floor) with
      | 0 -> Some (if x > floor then -1 else 0)
      | c -> Some c

  (* The order of two values, or [None] if either is NaN. *)
  let order v0 v1 =
    match (number v0, number v1) with
    | `Int n0, `Int n1 -> Some (Bigint.compare n0 n1)
    | `Int n, `Float x -> compare_int_float n x
    | `Float x, `Int n -> Option.map Int.neg (compare_int_float n x)
    | `Float x0, `Float x1 ->
        if Float.is_nan x0 || Float.is_nan x1 then None
        else Some (Float.compare x0 x1)

  let compare v0 v1 =
    match order v0 v1 with
    | Some c -> c
    | None -> (
        let is_nan = function `Float x -> Float.is_nan x | _ -> false in
        match (is_nan v0, is_nan v1) with
        | true, true -> 0
        | true, false -> -1
        | false, _ -> 1)

  let arith int_op float_op v0 v1 =
    match (number v0, number v1) with
    | `Int n0, `Int n1 -> `Int (int_op n0 n1)
    | v0, v1 -> `Float (float_op (to_float (v0 :> t)) (to_float (v1 :> t)))

  (* Floats divide as CPython's float_divmod does: through fmod, whose remainder
     is exact, then moved to the divisor's sign. *)
  let float_divmod x y =
    if y = 0. then raise Division_by_zero;
    let m = Float.rem x y in
    let d = (x -. m) /. y in
    let d, m =
      if m = 0. then (d, Float.copy_sign 0. y)
      else if not (Bool.equal (y < 0.) (m < 0.)) then (d -. 1., m +. y)
      else (d, m)
    in
    let q =
      if d = 0. then Float.copy_sign 0. (x /. y)
      else
        let f = Float.floor d in
        if d -. f > 0.5 then f +. 1. else f
    in
    (q, m)

  let ( = ) v0 v1 = order v0 v1 = Some 0
  let ( <> ) v0 v1 = not (v0 = v1)
  let ( < ) v0 v1 = match order v0 v1 with Some c -> c < 0 | None -> false
  let ( <= ) v0 v1 = match order v0 v1 with Some c -> c <= 0 | None -> false
  let ( > ) v0 v1 = v1 < v0
  let ( >= ) v0 v1 = v1 <= v0
  let min v0 v1 = if v1 < v0 then v1 else v0
  let max v0 v1 = if v1 > v0 then v1 else v0

  let ( ~- ) v =
    match number v with
    | `Int n -> `Int (Bigint.neg n)
    | `Float x -> `Float (-.x)

  let ( + ) = arith Bigint.add ( +. )
  let ( - ) = arith Bigint.sub ( -. )
  let ( * ) = arith Bigint.mul ( *. )
  let ( // ) = arith Bigint.fdiv (fun x y -> fst (float_divmod x y))

  let ( % ) =
    arith
      (fun n0 n1 -> Bigint.sub n0 (Bigint.mul n1 (Bigint.fdiv n0 n1)))
      (fun x y -> snd (float_divmod x y))
end

let int_min dt =
  if is_unsigned dt then Bigint.zero
  else Bigint.neg (Bigint.shift_left Bigint.one (bitsize dt - 1))

let int_max dt =
  Bigint.add
    (Bigint.pred (Bigint.shift_left Bigint.one (bitsize dt)))
    (int_min dt)

(* Narrow floats *)

(* What follows a format's greatest finite value: an infinity then NaNs, NaNs
   only, or nothing, the one NaN taking negative zero's code. *)
type specials = Ieee | Nan_only | Fnuz

(* A float format narrower than float32: [bits] wide with [mant] explicit
   mantissa bits and exponent bias [bias]. [top] is the magnitude code of its
   greatest finite value and [nan] the code of its positive NaN. *)
type float_format = {
  bits : int;
  mant : int;
  bias : int;
  top : int;
  nan : int;
  specials : specials;
}

let float_format dt =
  let f bits mant bias top nan specials =
    { bits; mant; bias; top; nan; specials }
  in
  match dt with
  | Float16 -> f 16 10 15 0x7BFF 0x7E00 Ieee
  | Bfloat16 -> f 16 7 127 0x7F7F 0x7FC0 Ieee
  | Fp8e4m3 -> f 8 3 7 0x7E 0x7F Nan_only
  | Fp8e5m2 -> f 8 2 15 0x7B 0x7F Ieee
  | Fp8e4m3fnuz -> f 8 3 8 0x7F 0x80 Fnuz
  | Fp8e5m2fnuz -> f 8 2 16 0x7F 0x80 Fnuz
  | dt -> invalid_arg (Format.asprintf "%a is not a narrow float" pp dt)

(* A NaN's bits: the double NaN of sign [negative] whose leading [mant] mantissa
   bits, quiet bit first, are [payload], and back. The bits move by hand, since
   the hardware's conversions between floats and doubles set the quiet bit of a
   signalling NaN. *)
let nan_of_payload ~mant ~negative payload =
  let bits =
    Int64.(
      logor 0x7FF0_0000_0000_0000L (shift_left (of_int payload) (52 - mant)))
  in
  Int64.float_of_bits
    (if negative then Int64.logor Int64.min_int bits else bits)

(* A payload whose leading bits are all zero is the quiet NaN's. *)
let nan_payload ~mant x =
  let p =
    Int64.(
      to_int
        (shift_right_logical
           (logand (bits_of_float x) 0xF_FFFF_FFFF_FFFFL)
           (52 - mant)))
  in
  if p = 0 then 1 lsl (mant - 1) else p

let decode_format f code =
  let negative = code lsr (f.bits - 1) = 1 in
  let sign = if negative then -1. else 1. in
  let q = code land ((1 lsl (f.bits - 1)) - 1) in
  if f.specials = Fnuz && code = f.nan then nan
  else if f.specials = Ieee && q = f.top + 1 then
    Float.copy_sign Float.infinity sign
  else if f.specials = Ieee && q > f.top then
    nan_of_payload ~mant:f.mant ~negative (q land ((1 lsl f.mant) - 1))
  else if q > f.top then Float.copy_sign nan sign
  else
    let exp = q lsr f.mant and m = q land ((1 lsl f.mant) - 1) in
    let v =
      if exp = 0 then Float.ldexp (Float.of_int m) (1 - f.bias - f.mant)
      else
        Float.ldexp (Float.of_int (m lor (1 lsl f.mant))) (exp - f.bias - f.mant)
    in
    Float.copy_sign v sign

(* [x] rounded to nearest, ties to even, to [mant] mantissa bits, subnormal
   below 2^[emin], with no upper exponent bound. The scaling by a power of two,
   the floor and the difference are exact. *)
let nearest ~mant ~emin x =
  if x = 0. || not (Float.is_finite x) then x
  else
    let _, e = Float.frexp x in
    let q = Int.max (e - 1) emin - mant in
    let s = Float.ldexp x (-q) in
    let fl = Float.floor s in
    let d = s -. fl in
    let r =
      if d > 0.5 || (d = 0.5 && Float.rem fl 2. <> 0.) then fl +. 1. else fl
    in
    Float.copy_sign (Float.ldexp r q) x

(* Past the greatest finite value, 16-bit formats give their infinity and 8-bit
   ones saturate to it. An infinity stays one where the format has infinities,
   and is its NaN where it has none. A NaN keeps the leading bits of its payload
   where the format has room for them. *)
let encode_format f x =
  let sign = if Float.sign_bit x then 1 lsl (f.bits - 1) else 0 in
  if Float.is_nan x && f.specials = Ieee then
    sign lor (f.top + 1) lor nan_payload ~mant:f.mant x
  else if Float.is_nan x || (f.specials <> Ieee && not (Float.is_finite x)) then
    if f.specials = Fnuz then f.nan else sign lor f.nan
  else
    let emin = 1 - f.bias in
    let a = Float.abs (nearest ~mant:f.mant ~emin x) in
    let q =
      if a > decode_format f f.top then
        if f.bits = 8 && Float.is_finite x then f.top else f.top + 1
      else if a = 0. then 0
      else
        (* Subnormals share the least normal exponent, whose field is 1: the
           field and the mantissa's leading bit add up alike for both. *)
        let _, e = Float.frexp a in
        let exp = Int.max (e - 1) emin in
        ((exp + f.bias - 1) lsl f.mant)
        + Float.to_int (Float.ldexp a (f.mant - exp))
    in
    if f.specials = Fnuz && q = 0 then 0 else sign lor q

let decode dt code = decode_format (float_format dt) code
let encode dt x = encode_format (float_format dt) x
let round dt x = decode dt (encode dt x)

(* Casts *)

let storage_fmt dt =
  match dt with
  | Bfloat16 -> Some 'H'
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> Some 'B'
  | dt -> fmt dt

(* An integer as a double rounded to odd: towards zero, with the last bit set if
   bits were dropped. A float of fewer bits rounded to nearest from it is the
   integer rounded to nearest once. An integer past every double is the greatest
   double, which every narrower float overflows as it would. *)
let float_of_integer z =
  let a = Bigint.abs z in
  let shift = Bigint.numbits a - 53 in
  if shift <= 0 then Bigint.to_float z
  else
    let q = Bigint.shift_right a shift in
    let q =
      if Bigint.equal (Bigint.shift_left q shift) a then q
      else Bigint.logor q Bigint.one
    in
    let x = Float.ldexp (Bigint.to_float q) shift in
    let x = if Float.is_finite x then x else Float.max_float in
    if Bigint.sign z < 0 then -.x else x

(* [v] as a float to round to [dt]: an integer reaches a float narrower than a
   double once, so exactly or rounded to odd. *)
let float_value dt (v : value) =
  match (dt, v) with
  | (Float64 | Weak_float), _ -> Value.to_float v
  | _, `Int z -> float_of_integer z
  | _ -> Value.to_float v

let to_storage_scalar dt (v : value) : value =
  match dt with
  | Float16 -> `Float (round Float16 (float_value dt v))
  | Bfloat16 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      `Int (Bigint.of_int (encode dt (float_value dt v)))
  | _ -> v

let from_storage_scalar dt (s : value) : value =
  match dt with
  | Bfloat16 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> (
      match s with
      | `Int n ->
          `Float (decode dt (Bigint.to_int (Bigint.extract n 0 (bitsize dt))))
      | `Bool _ | `Float _ ->
          invalid_arg
            (Format.asprintf "%a is not a storage of %a" pp_const s pp dt))
  | _ -> s

(* A conversion to [dt]'s precision. It quiets a signalling NaN and keeps its
   payload, as IEEE conversions do, and an 8-bit float gives its canonical NaN
   of the same sign, as nx's encoders do. The weak float and the doubles keep
   every other value. *)
let truncate_float dt x =
  let x =
    if Float.is_nan x then
      Int64.(float_of_bits (logor (bits_of_float x) 0x0008_0000_0000_0000L))
    else x
  in
  match dt with
  | (Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz) when Float.is_nan x ->
      let f = float_format dt in
      let sign = if Float.sign_bit x && f.specials <> Fnuz then 0x80 else 0 in
      decode_format f (sign lor f.nan)
  | Float16 | Bfloat16 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      round dt x
  | Float32 -> Int32.float_of_bits (Int32.bits_of_float x)
  | _ -> x

let truncate dt (v : value) : value =
  match dt with
  | Void -> invalid_arg "void has no value"
  | Weak_int | Weak_float -> v
  | Bool -> `Bool (Value.to_bool v)
  | dt when is_float dt -> `Float (truncate_float dt (float_value dt v))
  | dt when is_unsigned dt -> `Int (Bigint.extract (integer v) 0 (bitsize dt))
  | dt -> `Int (Bigint.signed_extract (integer v) 0 (bitsize dt))

(* The integer data type whose storage stores [dt]'s. *)
let storage_int dt =
  match dt with
  | Bfloat16 -> Uint16
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> Uint8
  | dt -> dt

(* A float32's bits and back, a NaN's moved by hand to keep a signalling one. *)
let float32_bits x =
  if Float.is_nan x then
    (if Float.sign_bit x then 0x8000_0000 else 0)
    lor 0x7F80_0000 lor nan_payload ~mant:23 x
  else Int32.to_int (Int32.bits_of_float x) land 0xFFFF_FFFF

let float32_of_bits b =
  if b land 0x7F80_0000 = 0x7F80_0000 && b land 0x7F_FFFF <> 0 then
    nan_of_payload ~mant:23 ~negative:(b lsr 31 = 1) (b land 0x7F_FFFF)
  else Int32.float_of_bits (Int32.of_int b)

(* The bits that store [s], a value of [dt]'s storage format. *)
let pack dt (s : value) =
  match dt with
  | Bool -> if Value.to_bool s then Bigint.one else Bigint.zero
  | Float16 -> Bigint.of_int (encode Float16 (Value.to_float s))
  | Float32 -> Bigint.of_int (float32_bits (Value.to_float s))
  | Float64 ->
      Bigint.extract
        (Bigint.of_int64 (Int64.bits_of_float (Value.to_float s)))
        0 64
  | Void | Weak_int | Weak_float ->
      invalid_arg (Format.asprintf "%a has no storage" pp dt)
  | dt ->
      let dt = storage_int dt and n = integer s in
      if Bigint.lt n (int_min dt) || Bigint.gt n (int_max dt) then
        invalid_arg
          (Format.asprintf "%a is out of the range of %a" Bigint.pp_print n pp
             dt);
      Bigint.extract n 0 (bitsize dt)

(* The value of [dt]'s storage format that [bits] store. *)
let unpack dt bits : value =
  match dt with
  | Bool -> `Bool (not (Bigint.equal bits Bigint.zero))
  | Float16 -> `Float (decode Float16 (Bigint.to_int bits))
  | Float32 -> `Float (float32_of_bits (Bigint.to_int bits))
  | Float64 ->
      `Float
        (Int64.float_of_bits
           (Bigint.to_int64 (Bigint.signed_extract bits 0 64)))
  | dt ->
      let dt = storage_int dt in
      `Int
        (if is_unsigned dt then bits
         else Bigint.signed_extract bits 0 (bitsize dt))

let bitcast d0 d1 v =
  if itemsize d0 <> itemsize d1 then
    invalid_arg (Format.asprintf "%a and %a differ in size" pp d0 pp d1);
  from_storage_scalar d1 (unpack d1 (pack d0 (to_storage_scalar d0 v)))

(* Bounds and constants *)

let float_max dt =
  match dt with
  | Fp8e4m3 | Fp8e4m3fnuz | Fp8e5m2fnuz -> decode dt (float_format dt).top
  | _ -> Float.infinity

let min dt : value =
  if is_int dt then `Int (int_min dt)
  else if is_float dt then `Float (-.float_max dt)
  else `Bool false

let max dt : value =
  if is_int dt then `Int (int_max dt)
  else if is_float dt then `Float (float_max dt)
  else `Bool true

(* A NaN constant is the NaN [dt] stores for it, a signalling one included: a
   bitcast of the constant reveals its bits, which a conversion would quiet. *)
let const dt (c : [< const ]) : const =
  match c with
  | `Invalid -> `Invalid
  | `Float x when Float.is_nan x && is_float dt -> (
      match dt with
      | Float32 -> `Float (float32_of_bits (float32_bits x))
      | Float64 | Weak_float -> `Float x
      | dt -> `Float (round dt x))
  | #value as v ->
      if is_float dt then `Float (truncate_float dt (float_value dt v))
      else if is_bool dt then `Bool (Value.to_bool v)
      else `Int (Value.to_z v)

(* Names and defaults *)

let of_name s =
  List.find_map (fun (dt, ns) -> if List.mem s ns then Some dt else None) names

(* The data type a setting names, of the kind [is_kind] accepts. *)
let setting_dtype setting ~kind is_kind =
  let value = Helpers.Context_var.value setting in
  match of_name (String.lowercase_ascii value) with
  | Some dt when is_kind dt -> dt
  | _ ->
      invalid_arg
        (strf "%s=%s is not %s" (Helpers.Context_var.key setting) value kind)

let default_float () =
  setting_dtype Helpers.default_float ~kind:"a float of known width" (fun dt ->
      List.mem dt floats)

let default_int () =
  setting_dtype Helpers.default_int ~kind:"an integer of known width" (fun dt ->
      List.mem dt ints)

let of_string s =
  match String.lowercase_ascii s with
  | "default_float" -> Ok (default_float ())
  | "default_int" -> Ok (default_int ())
  | name -> (
      match of_name name with
      | Some dt -> Ok dt
      | None -> Error (strf "%S is not a data type" s))

let strong dt =
  match dt with
  | Weak_int -> default_int ()
  | Weak_float -> default_float ()
  | dt -> dt

let commit_int ?default_int:first lo hi =
  if
    Bigint.equal lo hi
    && (Bigint.lt lo (int_min Int64) || Bigint.gt lo (int_max Uint64))
  then invalid_arg (strf "%s does not fit any integer" (Bigint.to_string lo));
  let first = match first with Some dt -> dt | None -> default_int () in
  if not (List.mem first ints) then
    invalid_arg (Format.asprintf "%a is not an integer of known width" pp first);
  let holds dt = Bigint.leq (int_min dt) lo && Bigint.leq hi (int_max dt) in
  Option.value
    (List.find_opt holds [ first; Int32; Int64; Uint64 ])
    ~default:Int64

let weak dt =
  if is_float dt then Weak_float else if is_int dt then Weak_int else dt

(* Literals *)

let of_const (c : [< const ]) =
  match c with
  | `Bool _ | `Invalid -> Bool
  | `Int _ -> Weak_int
  | `Float _ -> Weak_float

let of_consts (cs : [< const ] list) =
  let greatest d0 d1 = if compare d1 d0 > 0 then d1 else d0 in
  match List.map of_const cs with
  | [] -> strong Weak_float
  | dt :: dts -> (
      match List.fold_left greatest dt dts with
      | Weak_int ->
          let integral (c : [< const ]) =
            match c with
            | (`Int _ | `Bool _) as v -> Some (integer v)
            | `Float _ | `Invalid -> None
          in
          let ns = List.filter_map integral cs in
          let lo = List.fold_left Bigint.min (List.hd ns) ns in
          let hi = List.fold_left Bigint.max (List.hd ns) ns in
          commit_int lo hi
      | dt -> strong dt)

(* Promotion *)

let promo_lattice = function
  | Bool -> [ Weak_int ]
  | Weak_int -> [ Int8; Uint8 ]
  | Int8 -> [ Int16 ]
  | Int16 -> [ Int32 ]
  | Int32 -> [ Int64 ]
  | Int64 -> [ Weak_float ]
  | Uint8 -> [ Int16; Uint16 ]
  | Uint16 -> [ Int32; Uint32 ]
  | Uint32 -> [ Int64; Uint64 ]
  | Uint64 -> [ Weak_float ]
  | Weak_float -> fp8s
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> [ Float16; Bfloat16 ]
  | Float16 | Bfloat16 -> [ Float32 ]
  | Float32 -> [ Float64 ]
  | Float64 | Void -> []

(* Each data type with every data type it promotes to, itself included. *)
let recursive_parents =
  let rec parents dt = dt :: List.concat_map parents (promo_lattice dt) in
  List.map (fun dt -> (dt, List.sort_uniq compare (parents dt))) (weaks @ all)

let parents dt =
  match List.assq_opt dt recursive_parents with
  | Some ps -> ps
  | None -> invalid_arg "void does not promote"

let least_upper = function
  | [] -> invalid_arg "no data type to promote"
  | dt :: dts ->
      let common p = List.for_all (fun dt -> List.memq p (parents dt)) dts in
      (* [parents dt] is sorted, and the top of the lattice is common. *)
      List.find common (parents dt)

let least_upper_float dt =
  if dt = Weak_int then Weak_float
  else if is_float dt then dt
  else least_upper [ dt; default_float () ]

let can_lossless_cast d0 d1 =
  d0 = d1 || d0 = Bool
  ||
  match d1 with
  | Weak_int -> List.mem d0 ints
  | Float64 ->
      List.mem d0
        ([ Float32; Float16; Bfloat16 ]
        @ fp8s
        @ [ Uint32; Uint16; Uint8; Int32; Int16; Int8 ])
  | Float32 ->
      List.mem d0 ([ Float16; Bfloat16 ] @ fp8s @ [ Uint16; Uint8; Int16; Int8 ])
  | Float16 -> List.mem d0 (fp8s @ [ Uint8; Int8 ])
  | Uint64 -> List.mem d0 [ Uint32; Uint16; Uint8 ]
  | Uint32 -> List.mem d0 [ Uint16; Uint8 ]
  | Uint16 -> List.mem d0 [ Uint8 ]
  | Int64 -> List.mem d0 [ Uint32; Uint16; Uint8; Int32; Int16; Int8 ]
  | Int32 -> List.mem d0 [ Uint16; Uint8; Int16; Int8 ]
  | Int16 -> List.mem d0 [ Uint8; Int8 ]
  | _ -> false

let sum_acc dt =
  if is_unsigned dt then least_upper [ dt; Uint32 ]
  else if is_int dt || dt = Bool then least_upper [ dt; Int32 ]
  else
    let value = Helpers.getenv_string "SUM_DTYPE" "float32" in
    match of_string value with
    | Ok acc -> least_upper [ dt; acc ]
    | Error _ -> invalid_arg (strf "SUM_DTYPE=%s is not a data type" value)
