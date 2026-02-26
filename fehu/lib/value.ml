(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t =
  | Null
  | Bool of bool
  | Int of int
  | Float of float
  | String of string
  | Int_array of int array
  | Float_array of float array
  | Bool_array of bool array
  | List of t list
  | Dict of (string * t) list

(* Equality *)

let rec equal a b =
  match (a, b) with
  | Null, Null -> true
  | Bool a, Bool b -> Bool.equal a b
  | Int a, Int b -> Int.equal a b
  | Float a, Float b -> Float.equal a b
  | String a, String b -> String.equal a b
  | Int_array a, Int_array b -> a = b
  | Float_array a, Float_array b -> a = b
  | Bool_array a, Bool_array b -> a = b
  | List a, List b -> equal_list a b
  | Dict a, Dict b -> equal_dict a b
  | ( ( Null | Bool _ | Int _ | Float _ | String _ | Int_array _ | Float_array _
      | Bool_array _ | List _ | Dict _ ),
      _ ) ->
      false

and equal_list a b =
  match (a, b) with
  | [], [] -> true
  | x :: xs, y :: ys -> equal x y && equal_list xs ys
  | _ -> false

and equal_dict a b =
  match (a, b) with
  | [], [] -> true
  | (ka, va) :: rest_a, (kb, vb) :: rest_b ->
      String.equal ka kb && equal va vb && equal_dict rest_a rest_b
  | _ -> false

(* Formatting *)

let pp_array pp_elt ppf a =
  Format.fprintf ppf "[|";
  for i = 0 to Array.length a - 1 do
    if i > 0 then Format.fprintf ppf "; ";
    pp_elt ppf a.(i)
  done;
  Format.fprintf ppf "|]"

let rec pp ppf = function
  | Null -> Format.fprintf ppf "null"
  | Bool b -> Format.fprintf ppf "%b" b
  | Int i -> Format.fprintf ppf "%d" i
  | Float f -> Format.fprintf ppf "%g" f
  | String s -> Format.fprintf ppf "%S" s
  | Int_array a -> pp_array (fun ppf v -> Format.fprintf ppf "%d" v) ppf a
  | Float_array a -> pp_array (fun ppf v -> Format.fprintf ppf "%g" v) ppf a
  | Bool_array a -> pp_array (fun ppf v -> Format.fprintf ppf "%b" v) ppf a
  | List items ->
      Format.fprintf ppf "[";
      List.iteri
        (fun i v ->
          if i > 0 then Format.fprintf ppf "; ";
          pp ppf v)
        items;
      Format.fprintf ppf "]"
  | Dict fields ->
      Format.fprintf ppf "{";
      List.iteri
        (fun i (k, v) ->
          if i > 0 then Format.fprintf ppf "; ";
          Format.fprintf ppf "%s: %a" k pp v)
        fields;
      Format.fprintf ppf "}"

let to_string v = Format.asprintf "%a" pp v
