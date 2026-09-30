(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* Constants *)

type value = [ `Bool of bool | `Int of Z.t | `Float of float ]
type const = [ value | `Invalid ]

(* The bits of the NaN every constant NaN becomes, which a bitcast reveals. *)
let nan = Int64.float_of_bits 0x7FF8_0000_0000_0000L

let equal_const (c0 : [< const ]) (c1 : [< const ]) =
  match ((c0 :> const), (c1 :> const)) with
  | `Bool b0, `Bool b1 -> Bool.equal b0 b1
  | `Int n0, `Int n1 -> Z.equal n0 n1
  | `Float x0, `Float x1 ->
      (Float.is_nan x0 && Float.is_nan x1)
      || Int64.equal (Int64.bits_of_float x0) (Int64.bits_of_float x1)
  | `Invalid, `Invalid -> true
  | _ -> false

(* Hashtbl.hash hashes every NaN alike, and -0.0 like 0.0. *)
let hash_const (c : [< const ]) = Hashtbl.hash c

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
      Z.of_string
        (String.concat "" (String.split_on_char '.' (String.sub s 0 e)))
    in
    let exponent =
      int_of_string (String.sub s (e + 1) (String.length s - e - 1)) - n + 1
    in
    let reads m =
      Float.equal
        (float_of_string (strf "%se%d" (Z.to_string m) exponent))
        (Float.abs x)
    in
    match
      List.find_opt reads [ mantissa; Z.succ mantissa; Z.pred mantissa ]
    with
    | Some m -> (Z.to_string m, exponent)
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
  | `Int n -> Z.pp_print ppf n
  | `Float x -> Format.pp_print_string ppf (float_repr x)
  | `Invalid -> Format.pp_print_string ppf "Invalid"

(* Address spaces *)

type addr_space = Global | Local | Reg | Alu

let pp_addr_space ppf space =
  Format.pp_print_string ppf
    (match space with
    | Global -> "AddrSpace.GLOBAL"
    | Local -> "AddrSpace.LOCAL"
    | Reg -> "AddrSpace.REG"
    | Alu -> "AddrSpace.ALU")

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
let equal (dt0 : t) dt1 = dt0 = dt1

let compare dt0 dt1 =
  let i0 = info dt0 and i1 = info dt1 in
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
let is_float dt = List.mem dt floats || dt = Weak_float
let is_int dt = List.mem dt ints || dt = Weak_int
let is_unsigned dt = List.mem dt uints
let is_bool dt = dt = Bool

let finfo dt =
  match dt with
  | Float16 -> (5, 10)
  | Bfloat16 -> (8, 7)
  | Float32 -> (8, 23)
  | Float64 -> (11, 52)
  | Fp8e4m3 | Fp8e4m3fnuz -> (4, 3)
  | Fp8e5m2 | Fp8e5m2fnuz -> (5, 2)
  | _ ->
      invalid_arg
        (Format.asprintf "Dtype.finfo: %a is not a float of known width" pp dt)

(* Conversions *)

(* A value as a float, rounded to nearest: an integer beyond the doubles is an
   infinity. *)
let to_float : value -> float = function
  | `Bool b -> if b then 1. else 0.
  | `Float x -> x
  | `Int n -> Z.to_float n

(* A value as an integer. A float has none: rounding it is the caller's. *)
let to_int : value -> Z.t = function
  | `Bool b -> Z.of_int (Bool.to_int b)
  | `Int n -> n
  | `Float x ->
      invalid_arg (strf "Dtype: %s is a float, not an integer" (float_repr x))

let is_nonzero : value -> bool = function
  | `Bool b -> b
  | `Int n -> not (Z.equal n Z.zero)
  | `Float x -> x <> 0.

let int_min dt =
  if is_unsigned dt then Z.zero else Z.neg (Z.shift_left Z.one (bitsize dt - 1))

let int_max dt = Z.add (Z.pred (Z.shift_left Z.one (bitsize dt))) (int_min dt)

(* Narrow floats *)

(* A float format narrower than float32: [bits] wide with [mant] explicit
   mantissa bits and exponent bias [bias]. [top] is the magnitude code of its
   greatest finite value; the codes past it are its infinity, if it has
   [infinities], and its NaNs. [nan] is its positive NaN. The [fnuz] formats
   have no negative zero, and [nan] is their one NaN. *)
type float_format = {
  bits : int;
  mant : int;
  bias : int;
  top : int;
  nan : int;
  infinities : bool;
  fnuz : bool;
}

let float_format dt =
  match dt with
  | Float16 ->
      {
        bits = 16;
        mant = 10;
        bias = 15;
        top = 0x7BFF;
        nan = 0x7E00;
        infinities = true;
        fnuz = false;
      }
  | Bfloat16 ->
      {
        bits = 16;
        mant = 7;
        bias = 127;
        top = 0x7F7F;
        nan = 0x7FC0;
        infinities = true;
        fnuz = false;
      }
  | Fp8e4m3 ->
      {
        bits = 8;
        mant = 3;
        bias = 7;
        top = 0x7E;
        nan = 0x7F;
        infinities = false;
        fnuz = false;
      }
  | Fp8e5m2 ->
      {
        bits = 8;
        mant = 2;
        bias = 15;
        top = 0x7B;
        nan = 0x7F;
        infinities = true;
        fnuz = false;
      }
  | Fp8e4m3fnuz ->
      {
        bits = 8;
        mant = 3;
        bias = 8;
        top = 0x7F;
        nan = 0x80;
        infinities = false;
        fnuz = true;
      }
  | Fp8e5m2fnuz ->
      {
        bits = 8;
        mant = 2;
        bias = 16;
        top = 0x7F;
        nan = 0x80;
        infinities = false;
        fnuz = true;
      }
  | dt -> invalid_arg (Format.asprintf "Dtype: %a is not a narrow float" pp dt)

let decode f code =
  let sign = if code lsr (f.bits - 1) = 1 then -1. else 1. in
  let q = code land ((1 lsl (f.bits - 1)) - 1) in
  if f.fnuz && code = f.nan then nan
  else if q > f.top then
    Float.copy_sign
      (if f.infinities && q = f.top + 1 then Float.infinity else nan)
      sign
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
   and is its NaN where it has none. *)
let encode f x =
  let sign = if Float.sign_bit x then 1 lsl (f.bits - 1) else 0 in
  if Float.is_nan x || not (f.infinities || Float.is_finite x) then
    if f.fnuz then f.nan else sign lor f.nan
  else
    let emin = 1 - f.bias in
    let a = Float.abs (nearest ~mant:f.mant ~emin x) in
    let q =
      if a > decode f f.top then
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
    if f.fnuz && q = 0 then 0 else sign lor q

let round dt x =
  let f = float_format dt in
  decode f (encode f x)

let storage_fmt dt =
  match dt with
  | Bfloat16 -> Some 'H'
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> Some 'B'
  | dt -> fmt dt

let to_storage_scalar dt (v : value) : value =
  match dt with
  | Float16 -> `Float (round Float16 (to_float v))
  | Bfloat16 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      `Int (Z.of_int (encode (float_format dt) (to_float v)))
  | _ -> v

let from_storage_scalar dt (s : value) : value =
  match dt with
  | Bfloat16 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> (
      let f = float_format dt in
      match s with
      | `Int n -> `Float (decode f (Z.to_int (Z.extract n 0 f.bits)))
      | `Bool _ | `Float _ ->
          invalid_arg
            (Format.asprintf "Dtype: %a is not a storage of %a" pp_const s pp dt)
      )
  | _ -> s

(* A float of [dt]'s precision. The weak float and the doubles keep all. *)
let truncate_float dt x =
  match dt with
  | Float16 | Bfloat16 | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      round dt x
  | Float32 -> Int32.float_of_bits (Int32.bits_of_float x)
  | _ -> x

let truncate dt (v : value) : value =
  match dt with
  | Void -> invalid_arg "Dtype.truncate: void has no value"
  | Weak_int | Weak_float -> v
  | Bool -> `Bool (is_nonzero v)
  | dt when is_float dt -> `Float (truncate_float dt (to_float v))
  | dt when is_unsigned dt -> `Int (Z.extract (to_int v) 0 (bitsize dt))
  | dt -> `Int (Z.signed_extract (to_int v) 0 (bitsize dt))

(* The integer data type whose storage stores [dt]'s. *)
let storage_int dt =
  match dt with
  | Bfloat16 -> Uint16
  | Fp8e4m3 | Fp8e5m2 | Fp8e4m3fnuz | Fp8e5m2fnuz -> Uint8
  | dt -> dt

(* The bits that store [s], a value of [dt]'s storage format. *)
let pack dt (s : value) =
  match dt with
  | Bool -> if is_nonzero s then Z.one else Z.zero
  | Float16 -> Z.of_int (encode (float_format Float16) (to_float s))
  | Float32 -> Z.extract (Z.of_int32 (Int32.bits_of_float (to_float s))) 0 32
  | Float64 -> Z.extract (Z.of_int64 (Int64.bits_of_float (to_float s))) 0 64
  | Void | Weak_int | Weak_float ->
      invalid_arg (Format.asprintf "Dtype.bitcast: %a has no storage" pp dt)
  | dt ->
      let dt = storage_int dt and n = to_int s in
      if Z.lt n (int_min dt) || Z.gt n (int_max dt) then
        invalid_arg
          (Format.asprintf "Dtype.bitcast: %a is out of the range of %a"
             Z.pp_print n pp dt);
      Z.extract n 0 (bitsize dt)

(* The value of [dt]'s storage format that [bits] store. *)
let unpack dt bits : value =
  match dt with
  | Bool -> `Bool (not (Z.equal bits Z.zero))
  | Float16 -> `Float (decode (float_format Float16) (Z.to_int bits))
  | Float32 ->
      `Float (Int32.float_of_bits (Z.to_int32 (Z.signed_extract bits 0 32)))
  | Float64 ->
      `Float (Int64.float_of_bits (Z.to_int64 (Z.signed_extract bits 0 64)))
  | dt ->
      let dt = storage_int dt in
      `Int
        (if is_unsigned dt then bits else Z.signed_extract bits 0 (bitsize dt))

let bitcast dt0 dt1 v =
  if itemsize dt0 <> itemsize dt1 then
    invalid_arg
      (Format.asprintf "Dtype.bitcast: %a and %a differ in size" pp dt0 pp dt1);
  from_storage_scalar dt1 (unpack dt1 (pack dt0 (to_storage_scalar dt0 v)))

(* Bounds and constants *)

let float_max dt =
  match dt with
  | Fp8e4m3 | Fp8e4m3fnuz | Fp8e5m2fnuz ->
      let f = float_format dt in
      decode f f.top
  | _ -> Float.infinity

let min dt : value =
  if is_int dt then `Int (int_min dt)
  else if is_float dt then `Float (-.float_max dt)
  else `Bool false

let max dt : value =
  if is_int dt then `Int (int_max dt)
  else if is_float dt then `Float (float_max dt)
  else `Bool true

let const dt (c : [< const ]) : const =
  match c with
  | `Invalid -> `Invalid
  | #value as v -> (
      let v =
        match v with `Float x when Float.is_nan x -> `Float nan | v -> v
      in
      if is_float dt then `Float (truncate_float dt (to_float v))
      else if is_bool dt then `Bool (is_nonzero v)
      else
        match v with
        | `Float x when not (Float.is_finite x) ->
            invalid_arg
              (Format.asprintf "Dtype.const: %s is not a %a" (float_repr x) pp
                 dt)
        | `Float x -> `Int (Z.of_float x)
        | v -> `Int (to_int v))

(* Names and defaults *)

let of_name s =
  List.find_map (fun (dt, ns) -> if List.mem s ns then Some dt else None) names

(* The data type a setting names, of the kind [is_kind] accepts. *)
let setting_dtype key value ~is_kind ~kind =
  match of_name (String.lowercase_ascii value) with
  | Some dt when is_kind dt -> dt
  | _ -> invalid_arg (strf "Dtype: %s=%s is not %s" key value kind)

let default_float () =
  let setting = Helpers.default_float in
  setting_dtype
    (Helpers.Context_var.key setting)
    (Helpers.Context_var.value setting)
    ~is_kind:(fun dt -> List.mem dt floats)
    ~kind:"a float of known width"

let default_int () =
  let setting = Helpers.default_int in
  setting_dtype
    (Helpers.Context_var.key setting)
    (Helpers.Context_var.value setting)
    ~is_kind:(fun dt -> List.mem dt ints)
    ~kind:"an integer of known width"

let of_string s =
  let name = String.lowercase_ascii s in
  let prefix = "dtypes." in
  let name =
    if String.starts_with ~prefix name then
      String.sub name (String.length prefix)
        (String.length name - String.length prefix)
    else name
  in
  match name with
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
  if Z.equal lo hi && (Z.lt lo (int_min Int64) || Z.gt lo (int_max Uint64)) then
    invalid_arg
      (strf "Dtype.commit_int: %s does not fit any integer" (Z.to_string lo));
  let first = match first with Some dt -> dt | None -> default_int () in
  let holds dt = Z.leq (int_min dt) lo && Z.leq hi (int_max dt) in
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
  let greatest dt0 dt1 = if compare dt1 dt0 > 0 then dt1 else dt0 in
  match List.map of_const cs with
  | [] -> strong Weak_float
  | dt :: dts -> (
      match List.fold_left greatest dt dts with
      | Weak_int ->
          let integer (c : [< const ]) =
            match c with
            | `Int n -> Some n
            | `Bool b -> Some (Z.of_int (Bool.to_int b))
            | `Float _ | `Invalid -> None
          in
          let ns = List.filter_map integer cs in
          let lo = List.fold_left Z.min (List.hd ns) ns in
          let hi = List.fold_left Z.max (List.hd ns) ns in
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
  | Float64 -> []
  | Void -> invalid_arg "Dtype: void does not promote"

(* Each data type with every data type it promotes to, itself included. *)
let recursive_parents =
  let rec parents dt = dt :: List.concat_map parents (promo_lattice dt) in
  List.map (fun dt -> (dt, List.sort_uniq compare (parents dt))) (weaks @ all)

let parents dt =
  match List.assq_opt dt recursive_parents with
  | Some ps -> ps
  | None -> invalid_arg "Dtype.least_upper: void does not promote"

let least_upper = function
  | [] -> invalid_arg "Dtype.least_upper: no data type"
  | dt :: dts ->
      let common p = List.for_all (fun dt -> List.memq p (parents dt)) dts in
      (* [parents dt] is sorted, and the top of the lattice is common. *)
      List.find common (parents dt)

let least_upper_float dt =
  if dt = Weak_int then Weak_float
  else if is_float dt then dt
  else least_upper [ dt; default_float () ]

let can_lossless_cast dt0 dt1 =
  dt0 = dt1 || dt0 = Bool
  ||
  match dt1 with
  | Weak_int -> List.mem dt0 ints
  | Float64 ->
      List.mem dt0
        ([ Float32; Float16; Bfloat16 ]
        @ fp8s
        @ [ Uint32; Uint16; Uint8; Int32; Int16; Int8 ])
  | Float32 ->
      List.mem dt0
        ([ Float16; Bfloat16 ] @ fp8s @ [ Uint16; Uint8; Int16; Int8 ])
  | Float16 -> List.mem dt0 (fp8s @ [ Uint8; Int8 ])
  | Uint64 -> List.mem dt0 [ Uint32; Uint16; Uint8 ]
  | Uint32 -> List.mem dt0 [ Uint16; Uint8 ]
  | Uint16 -> List.mem dt0 [ Uint8 ]
  | Int64 -> List.mem dt0 [ Uint32; Uint16; Uint8; Int32; Int16; Int8 ]
  | Int32 -> List.mem dt0 [ Uint16; Uint8; Int16; Int8 ]
  | Int16 -> List.mem dt0 [ Uint8; Int8 ]
  | _ -> false

let sum_acc dt =
  if is_unsigned dt then least_upper [ dt; Uint32 ]
  else if is_int dt || dt = Bool then least_upper [ dt; Int32 ]
  else
    let value = Helpers.getenv_string "SUM_DTYPE" "float32" in
    match of_string value with
    | Ok acc -> least_upper [ dt; acc ]
    | Error _ ->
        invalid_arg (strf "Dtype: SUM_DTYPE=%s is not a data type" value)
