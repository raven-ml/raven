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

let format (type a b) (dtype : (a, b) Nx.dtype) : Decimal.format option =
  match dtype with
  | Float64 -> Some Decimal.binary64
  | Float32 ->
      Some
        { precision = 24; emin = -126; max = 0x1.fffffep127; saturates = false }
  | Float16 ->
      Some { precision = 11; emin = -14; max = 65504.; saturates = false }
  | BFloat16 ->
      Some { precision = 8; emin = -126; max = 0x1.fep127; saturates = false }
  | Float8_e4m3 ->
      Some { precision = 4; emin = -6; max = 448.; saturates = true }
  | Float8_e5m2 ->
      Some { precision = 3; emin = -14; max = 57344.; saturates = true }
  | Int4 | UInt4 | Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 | Int64
  | UInt64 ->
      None
  | Complex64 | Complex128 | Bool ->
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
