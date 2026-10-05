(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let strf = Printf.sprintf

(* A constant's magnitude and uncertainty are exact dimensionless units, so they
   round once, through [Unit.round], from the published decimal. An exact
   constant has no uncertainty. *)
type t = {
  name : string;
  text : string;
  negative : bool;
  magnitude : Unit.t;
  uncertainty : Unit.t option;
  unit : Unit.t;
}

let is_digit c = c >= '0' && c <= '9'
let is_zero digits = String.for_all (fun c -> c = '0') digits

(* [below_2_62 digits] is [true] iff [digits], read as one integer, is below
   2^62. *)
let below_2_62 digits =
  let rec go acc i =
    i = String.length digits
    ||
    let d = Char.code digits.[i] - Char.code '0' in
    acc <= (max_int - d) / 10 && go ((acc * 10) + d) (i + 1)
  in
  go 0 0

(* [parse text] reads [["-"] digits ["." digits] ["(" digits ")"] ["e" ["-"]
   digits]] as its sign, its magnitude's decimal text and its uncertainty's, if
   any: ["-6.67430(15)e-11"] is [(true, "6.67430e-11", Some "0.00015e-11")]. The
   uncertainty's digits take the places of the magnitude's last ones, and zeros
   fill the places before them. *)
let parse text =
  let fail why = invalid_arg (strf "Constant.v: %S %s" text why) in
  let invalid () = fail "is not a constant's value" in
  let len = String.length text in
  let i = ref 0 in
  let eat c =
    !i < len
    && text.[!i] = c
    &&
    (incr i;
     true)
  in
  let digits () =
    let start = !i in
    while !i < len && is_digit text.[!i] do
      incr i
    done;
    if !i = start then invalid ();
    String.sub text start (!i - start)
  in
  let negative = eat '-' in
  let whole = digits () in
  let frac = if eat '.' then digits () else "" in
  let uncertainty =
    if not (eat '(') then None
    else
      let u = digits () in
      if not (eat ')') then invalid ();
      Some u
  in
  let exponent =
    if not (eat 'e') then ""
    else
      let neg = eat '-' in
      strf "e%s%s" (if neg then "-" else "") (digits ())
  in
  if !i <> len then invalid ();
  if is_zero (whole ^ frac) then fail "is zero";
  if not (below_2_62 (whole ^ frac)) then fail "has a mantissa of 2^62 or more";
  if Option.fold ~none:false ~some:is_zero uncertainty then
    fail "has a zero uncertainty; an exact value has no parentheses";
  if not (Option.fold ~none:true ~some:below_2_62 uncertainty) then
    fail "has an uncertainty of 2^62 or more";
  let point = if frac = "" then "" else "." ^ frac in
  let magnitude = whole ^ point ^ exponent in
  let uncertainty =
    Option.map
      (fun u ->
        let places = String.length frac in
        let u = String.make (max 0 (places + 1 - String.length u)) '0' ^ u in
        let cut = String.length u - places in
        let point = if places = 0 then "" else "." ^ String.sub u cut places in
        String.sub u 0 cut ^ point ^ exponent)
      uncertainty
  in
  (negative, magnitude, uncertainty)

let v ~name text unit =
  let negative, magnitude, uncertainty = parse text in
  let magnitude = Unit.decimal_named "Constant.v" magnitude in
  let uncertainty = Option.map (Unit.decimal_named "Constant.v") uncertainty in
  { name; text; negative; magnitude; uncertainty; unit }

let name k = k.name
let unit k = k.unit

(* [fail fn k why] raises the error of [fn] on [k] for the reason [why]. *)
let fail fn k why = invalid_arg (strf "%s: %s: %s" fn k.name why)

(* [no_factor fn k d] raises for [d], a dtype of booleans. *)
let no_factor fn k d =
  fail fn k (strf "%s holds no factor" (Nx_dtype.to_string d))

(* [round fn k subject d u] is the magnitude [u] of [k] rounded once to [d]. An
   error names [fn], [k] and [subject], the text that writes [u]. *)
let round (type a b) fn k subject (d : (a, b) Nx.dtype) u : a =
  let dt = Nx_dtype.to_string d in
  let refuse why = fail fn k (strf "%s %s" subject why) in
  match Unit.round d u with
  | Ok x -> x
  | Error Exact.Zero -> refuse (strf "is 0 in %s" dt)
  | Error Exact.Subnormal -> refuse (strf "is subnormal in %s" dt)
  | Error Exact.Overflow -> refuse (strf "overflows %s" dt)
  | Error Exact.Not_integer -> refuse "is not an integer"
  | Error Exact.Out_of_range ->
      refuse (strf "has a magnitude %s does not hold" dt)
  | Error Exact.Too_wide ->
      refuse
        (strf "needs a natural wider than %d bits to evaluate" Exact.budget)
  | Error Exact.Boolean -> no_factor fn k d

(* [negation fn k d] is negation in [d], for the negative [k]. It raises for a
   dtype that holds no negative value; [quantity] takes it before rounding, so
   that this refusal comes first. *)
let negation (type a b) fn k (d : (a, b) Nx.dtype) : a -> a =
  match d with
  | Float16 -> Float.neg
  | Float32 -> Float.neg
  | Float64 -> Float.neg
  | BFloat16 -> Float.neg
  | Float8_e4m3 -> Float.neg
  | Float8_e5m2 -> Float.neg
  | Complex64 -> fun x -> { x with re = Float.neg x.re }
  | Complex128 -> fun x -> { x with re = Float.neg x.re }
  | Int4 -> Int.neg
  | Int8 -> Int.neg
  | Int16 -> Int.neg
  | Int32 -> Int32.neg
  | Int64 -> Int64.neg
  | UInt4 | UInt8 | UInt16 | UInt32 | UInt64 ->
      fail fn k
        (strf "%s is negative, which %s does not hold" k.text
           (Nx_dtype.to_string d))
  | Bool | Bit -> no_factor fn k d

let quantity d k =
  let fn = "Constant.quantity" in
  let sign = if k.negative then negation fn k d else Fun.id in
  let x = round fn k k.text d k.magnitude in
  Quantity.v k.unit (Nx.scalar d (sign x))

let uncertainty (type a b) (d : (a, b) Nx.dtype) k =
  let fn = "Constant.uncertainty" in
  match (k.uncertainty, d) with
  | Some u, _ ->
      let subject = "the uncertainty of " ^ k.text in
      Quantity.v k.unit (Nx.scalar d (round fn k subject d u))
  | None, (Bool | Bit) -> no_factor fn k d
  | None, _ -> Quantity.v k.unit (Nx.zeros d [||])

let pp ppf k =
  if Unit.equal k.unit Unit.one then
    Format.fprintf ppf "@[<h>%s = %s@]" k.name k.text
  else Format.fprintf ppf "@[<h>%s = %s %a@]" k.name k.text Unit.pp k.unit
