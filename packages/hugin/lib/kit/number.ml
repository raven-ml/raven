(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type notation = Decimal.notation = Plain | Exponent | Si | Percent
type precision = Decimal.precision = Decimals of int | Significant of int

type t = {
  notation : notation;
  precision : precision;
  trim : bool;
  group : bool;
}

let v ?(trim = false) ?(group = false) notation precision =
  (match precision with
  | Decimals d when d < 0 -> invalid_arg "Number.v: negative decimals"
  | Significant s when s < 1 ->
      invalid_arg "Number.v: fewer than 1 significant digit"
  | Decimals _ | Significant _ -> ());
  { notation; precision; trim; group }

let to_string ?(locale = Locale.default) f x =
  if Float.is_nan x then "NaN"
  else if x = Float.infinity then "\u{221E}"
  else if x = Float.neg_infinity then Locale.minus locale ^ "\u{221E}"
  else
    Decimal.write locale ~group:f.group ~trim:f.trim f.notation f.precision
      (Decimal.of_float x)

(* The format of the float dtype [dt], read from its limits: [emin] is the
   exponent of its least normal value. The float8 dtypes store a value past
   their largest finite one as that value. *)
let float_format dt ~saturates : Decimal.format option =
  Some
    {
      precision = Nx_dtype.precision dt;
      emin = snd (Float.frexp (Nx_dtype.min_normal dt)) - 1;
      max = Nx_dtype.max_finite dt;
      saturates;
    }

let format (type a b) (dtype : (a, b) Nx.dtype) : Decimal.format option =
  match dtype with
  | Float64 -> Some Decimal.binary64
  | Float32 -> float_format Float32 ~saturates:false
  | Float16 -> float_format Float16 ~saturates:false
  | BFloat16 -> float_format BFloat16 ~saturates:false
  | Float8_e4m3 -> float_format Float8_e4m3 ~saturates:true
  | Float8_e5m2 -> float_format Float8_e5m2 ~saturates:true
  | Int4 | UInt4 | Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64
  | UInt64 ->
      None
  | Complex64 | Complex128 | Bool | Bit ->
      invalid_arg "Number.decimals: not a real dtype"

let decimals dtype x =
  match format dtype with
  | None -> 0
  | Some f ->
      if (not (Float.is_finite x)) || Float.is_integer x then 0
      else
        let y = Decimal.round_to_format f x in
        if not (Float.is_finite y) then 0
        else
          let d = Decimal.of_float x in
          let rec loop n =
            if Decimal.rounds_to f (Decimal.round Nearest (-n) d) y then n
            else loop (n + 1)
          in
          loop 0

let notation_name = function
  | Plain -> "plain"
  | Exponent -> "exponent"
  | Si -> "si"
  | Percent -> "percent"

let pp_notation ppf n = Format.pp_print_string ppf (notation_name n)

let equal f f' =
  f.notation = f'.notation && f.precision = f'.precision
  && Bool.equal f.trim f'.trim
  && Bool.equal f.group f'.group

let pp ppf f =
  let n, unit =
    match f.precision with
    | Decimals n -> (n, "decimals")
    | Significant n -> (n, "significant")
  in
  Format.fprintf ppf "%s %d %s%s%s" (notation_name f.notation) n unit
    (if f.trim then " trim" else "")
    (if f.group then " group" else "")
