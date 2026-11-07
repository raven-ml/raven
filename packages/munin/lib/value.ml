(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = [ `Bool of bool | `Int of int | `Float of float | `String of string ]

let pp_float ppf x =
  if Float.is_nan x then Format.pp_print_string ppf "nan"
  else if x = Float.infinity then Format.pp_print_string ppf "inf"
  else if x = Float.neg_infinity then Format.pp_print_string ppf "-inf"
  else Format.pp_print_string ppf (Munin_float_text.Float_print.exact_float x)

(* As a JSON string escapes, without the quotes. *)
let pp_string ppf s =
  String.iter
    (function
      | '\\' -> Format.pp_print_string ppf "\\\\"
      | '\n' -> Format.pp_print_string ppf "\\n"
      | '\r' -> Format.pp_print_string ppf "\\r"
      | '\t' -> Format.pp_print_string ppf "\\t"
      | c when Char.code c < 0x20 || c = '\x7F' ->
          Format.fprintf ppf "\\u%04x" (Char.code c)
      | c -> Format.pp_print_char ppf c)
    s

let pp ppf = function
  | `Bool b -> Format.pp_print_bool ppf b
  | `Int n -> Format.pp_print_int ppf n
  | `Float x -> pp_float ppf x
  | `String s -> pp_string ppf s

let to_float = function
  | `Float f -> Some f
  | `Int n -> Some (Float.of_int n)
  | _ -> None

let to_int = function `Int n -> Some n | _ -> None
let to_string = function `String s -> Some s | _ -> None
let to_bool = function `Bool b -> Some b | _ -> None
