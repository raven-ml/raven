(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = [ `Bool of bool | `Int of int | `Float of float | `String of string ]

let pp ppf = function
  | `Bool b -> Format.pp_print_bool ppf b
  | `Int n -> Format.pp_print_int ppf n
  | `Float f -> Format.fprintf ppf "%g" f
  | `String s -> Format.fprintf ppf "%S" s

let to_float = function
  | `Float f -> Some f
  | `Int n -> Some (Float.of_int n)
  | _ -> None

let to_int = function
  | `Int n -> Some n
  | `Float f when Float.is_integer f -> Some (Float.to_int f)
  | _ -> None

let to_string = function `String s -> Some s | _ -> None
let to_bool = function `Bool b -> Some b | _ -> None

(* A float JSON has no number for is the object [{"float": "NaN"}], or
   ["Infinity"] or ["-Infinity"]: a shape no other value takes, so that a
   [`String] is always a string. *)
let to_json : t -> Jsont.json = function
  | `Bool b -> Jsont.Json.bool b
  | `Int n -> Jsont.Json.int n
  | `Float f when Float.is_finite f -> Jsont.Json.number f
  | `Float f -> Json_utils.json_obj [ ("float", Json_utils.float_json f) ]
  | `String s -> Jsont.Json.string s

let of_json : Jsont.json -> t option = function
  | Jsont.Bool (b, _) -> Some (`Bool b)
  | Jsont.Number (f, _) ->
      if Float.is_integer f && Float.abs f < 4503599627370496. then
        Some (`Int (Float.to_int f))
      else Some (`Float f)
  | Jsont.String (s, _) -> Some (`String s)
  | Jsont.Object _ as json -> (
      match Json_utils.json_assoc json with
      | [ ("float", (Jsont.String _ as x)) ] ->
          Json_utils.json_float x |> Option.map (fun f -> `Float f)
      | _ -> None)
  | _ -> None

let assoc_of_json json =
  let rec go acc = function
    | [] -> Some (List.rev acc)
    | (k, v) :: rest -> (
        match of_json v with Some v -> go ((k, v) :: acc) rest | None -> None)
  in
  match json with
  | Jsont.Object _ -> go [] (Json_utils.json_assoc json)
  | _ -> None
