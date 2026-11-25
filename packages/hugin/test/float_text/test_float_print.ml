(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The text hugin writes floats as: C's %g and %f where the C library rounds
   once, and the shortest digits that read back. The laws read the text back by
   hand, so that no C library takes part, and take the digits it should hold
   from Float_text, whose own suite checks them against an exact reference. *)

open Windtrap
module F = Hugin_float_text.Float_text
module P = Hugin_float_text.Float_print

(* [decimal s] is the sign and the decimal of the finite number [s] writes, as
   [(d, e)] without leading or trailing zeros, [("0", 0)] for zero. *)
let decimal s =
  let neg = String.length s > 0 && s.[0] = '-' in
  let s = if neg then String.sub s 1 (String.length s - 1) else s in
  let m, x =
    match String.index_opt s 'e' with
    | None -> (s, 0)
    | Some i ->
        let t = String.sub s (i + 1) (String.length s - i - 1) in
        (String.sub s 0 i, int_of_string t)
  in
  let frac =
    match String.index_opt m '.' with
    | None -> 0
    | Some i -> String.length m - i - 1
  in
  let d = String.concat "" (String.split_on_char '.' m) in
  let n = String.length d in
  let first = ref 0 and last = ref (n - 1) in
  while !first < n && d.[!first] = '0' do
    incr first
  done;
  while !last >= !first && d.[!last] = '0' do
    decr last
  done;
  if !first > !last then (neg, ("0", 0))
  else
    (neg, (String.sub d !first (!last - !first + 1), x - frac + (n - 1 - !last)))

let signed_decimal = pair bool (pair string int)

(* [exponent s] is the text after [e] in [s], if any. *)
let exponent s =
  match String.index_opt s 'e' with
  | None -> None
  | Some i -> Some (String.sub s (i + 1) (String.length s - i - 1))

(* [flaw ~decimals s] is what breaks the positional form in the digits of [s],
   its text after any sign and before any exponent, or [None]: an integer part
   of digits that is [0] or starts with a nonzero digit, then a point and digits
   only if a fraction follows. With [~decimals:n] the fraction has exactly [n]
   digits, as %f writes it; without, it ends in a nonzero digit, as %g writes
   it, and before an exponent the integer part is one nonzero digit. *)
let flaw ?decimals s =
  let s =
    if String.length s > 0 && s.[0] = '-' then
      String.sub s 1 (String.length s - 1)
    else s
  in
  let m, exp =
    match String.index_opt s 'e' with
    | None -> (s, false)
    | Some i -> (String.sub s 0 i, true)
  in
  let int, frac =
    match String.index_opt m '.' with
    | None -> (m, None)
    | Some i ->
        let f = String.sub m (i + 1) (String.length m - i - 1) in
        (String.sub m 0 i, Some f)
  in
  let digits t = t <> "" && String.for_all (fun c -> c >= '0' && c <= '9') t in
  if not (digits int) then Some "integer part not digits"
  else if int <> "0" && int.[0] = '0' then Some "leading zero"
  else if exp && (String.length int <> 1 || int = "0") then
    Some "not one nonzero digit before the exponent"
  else
    match (frac, decimals) with
    | Some f, _ when not (digits f) -> Some "point without digits after it"
    | Some f, Some n when String.length f <> n -> Some "not that many decimals"
    | None, Some n when n > 0 -> Some "no decimals"
    | Some f, None when f.[String.length f - 1] = '0' ->
        Some "trailing zero after the point"
    | _ -> None

(* C's %g layout at precision [p]: an exponent of a sign and at least two digits
   when the leading digit's exponent is below -4 or at least [p], the positional
   form otherwise. *)
let check_layout s p (d, e) =
  let lead = String.length d - 1 + e in
  let want =
    if d <> "0" && (lead < -4 || lead >= p) then
      Some (Printf.sprintf "%c%02d" (if lead < 0 then '-' else '+') (abs lead))
    else None
  in
  equal ~msg:s (option string) want (exponent s);
  equal ~msg:s (option string) None (flaw s)

(* Inputs *)

let finite =
  let open Gen in
  let magnitude =
    frequency
      [
        (3, float);
        ( 2,
          let+ d = int_range 1 99_999 and+ j = int_range (-12) 25 in
          float_of_string (Printf.sprintf "%de%d" d j) );
        (1, map Int64.float_of_bits (int64_range 1L 0xF_FFFF_FFFF_FFFFL));
      ]
  in
  with_pp
    (fun ppf x -> Format.fprintf ppf "%h" x)
    (let+ x = magnitude and+ neg = bool in
     if neg then -.x else Float.abs x)

let classes x =
  let x = Float.abs x in
  cover "subnormal" (x > 0. && x < 0x1p-1022);
  cover "below 1e-4" (x > 0. && x < 1e-4);
  cover "from 1e-4 to 1e15" (x >= 1e-4 && x < 1e15);
  cover "from 1e15" (x >= 1e15)

(* Expected texts are those of C99 7.19.6.1's %g and %f, as glibc's printf
   writes them, and the .mli's examples. *)

(* g *)

let test_g =
  prop "is x as %.*g writes it, with precision significant digits" ~count:2000
    Gen.(pair finite (int_range 1 20))
    (fun (x, p) ->
      classes x;
      let s = P.g ~precision:p x in
      let want = F.significant p x in
      equal ~msg:s signed_decimal (Float.sign_bit x, want) (decimal s);
      check_layout s p want)

(* 135.5625 is the .mli's example; 68405 at four digits lies halfway between
   6.840e+04 and 6.841e+04. *)
let test_g_rounding =
  cases "rounds once, to nearest, ties to even"
    ~id:(fun (name, _, _, _) -> name)
    [
      ("135.5625", None, 135.5625, "135.562");
      ("68405 at 4", Some 4, 68405., "6.84e+04");
      ("1234567", None, 1234567., "1.23457e+06");
      ("0.1 at 17", Some 17, 0.1, "0.10000000000000001");
    ]
    (fun (_, precision, x, want) -> equal string want (P.g ?precision x))

let test_g_layout =
  cases
    "is positional for an exponent from -4 to below precision, otherwise with \
     two exponent digits or more"
    ~id:(fun (name, _, _) -> name)
    [
      ("1e-4", 1e-4, "0.0001");
      ("1.5e-5", 1.5e-5, "1.5e-05");
      ("123456", 123456., "123456");
      ("1e16", 1e16, "1e+16");
      ("1e100", 1e100, "1e+100");
    ]
    (fun (_, x, want) -> equal string want (P.g x))

let test_g_floor =
  cases "takes a precision below 1 as 1" ~id:string_of_int [ 0; -3 ]
    (fun precision -> equal string "2" (P.g ~precision 2.5))

let test_g_special =
  cases
    "writes a NaN as nan whatever its sign, the infinities as inf and -inf, \
     and keeps the sign of zero"
    ~id:(fun (name, _, _) -> name)
    [
      ("0", 0., "0");
      ("-0", -0., "-0");
      ("nan", Float.nan, "nan");
      ("-nan", -.Float.nan, "nan");
      ("infinity", Float.infinity, "inf");
      ("neg_infinity", Float.neg_infinity, "-inf");
    ]
    (fun (_, x, want) -> equal string want (P.g x))

(* pp *)

let test_pp =
  prop "formats g x" Gen.any_float (fun x ->
      equal string (P.g x) (Format.asprintf "%a" P.pp x))

(* f *)

let test_f =
  prop "is x as %.*f writes it, with decimals digits after the point"
    ~count:2000
    Gen.(pair finite (int_range 0 30))
    (fun (x, n) ->
      classes x;
      let s = P.f n x in
      equal ~msg:s signed_decimal (Float.sign_bit x, F.decimals n x) (decimal s);
      equal ~msg:s (option string) None (flaw ~decimals:n s))

(* 0.125 at two is the .mli's example. *)
let test_f_rounding =
  cases "rounds once, to nearest, ties to even"
    ~id:(fun (name, _, _, _) -> name)
    [
      ("0.125 at 2", 2, 0.125, "0.12");
      ("0.5 at 0", 0, 0.5, "0");
      ("1.5 at 0", 0, 1.5, "2");
      ("2.25 at 1", 1, 2.25, "2.2");
    ]
    (fun (_, n, x, want) -> equal string want (P.f n x))

let test_f_digits =
  cases "writes decimals digits after the point, trailing zeros included"
    ~id:(fun (name, _, _, _) -> name)
    [
      ("0.1 at 6", 6, 0.1, "0.100000");
      ("1e21 at 2", 2, 1e21, "1000000000000000000000.00");
    ]
    (fun (_, n, x, want) -> equal string want (P.f n x))

let test_f_floor =
  test "takes negative decimals as 0" (fun () ->
      equal string "4" (P.f (-1) 3.7))

let test_f_special =
  cases
    "writes a NaN as nan whatever its sign, the infinities as inf and -inf, \
     and keeps the sign of zero"
    ~id:(fun (name, _, _) -> name)
    [
      ("-0", -0., "-0.00");
      ("-0.001", -0.001, "-0.00");
      ("nan", Float.nan, "nan");
      ("-nan", -.Float.nan, "nan");
      ("infinity", Float.infinity, "inf");
      ("neg_infinity", Float.neg_infinity, "-inf");
    ]
    (fun (_, x, want) -> equal string want (P.f 2 x))

(* exact *)

let test_exact =
  prop
    "is the shortest digits as g writes them at the larger of 15 and their \
     number"
    ~count:2000 finite (fun x ->
      classes x;
      let s = P.exact x in
      let ((d, _) as want) = F.shortest F.binary64 x in
      equal ~msg:s signed_decimal (Float.sign_bit x, want) (decimal s);
      check_layout s (max 15 (String.length d)) want)

(* The .mli's examples 0.1, 0.30000000000000004 and 5e-324; 2^-24 at the bottom
   of its binade, max_float and 2^53 + 2. *)
let test_exact_digits =
  cases "writes the fewest digits that read back"
    ~id:(fun (name, _, _) -> name)
    [
      ("0.1", 0.1, "0.1");
      ("0.1 + 0.2", 0.1 +. 0.2, "0.30000000000000004");
      ("2^53 + 2", 0x1p53 +. 2., "9007199254740994");
      ("2^-24", 0x1p-24, "5.960464477539063e-08");
      ("5e-324", 5e-324, "5e-324");
      ("max_float", max_float, "1.7976931348623157e+308");
    ]
    (fun (_, x, want) -> equal string want (P.exact x))

(* 1e+15 is the .mli's example. *)
let test_exact_layout =
  cases
    "is positional below an exponent of 15, or of its number of digits when it \
     has more"
    ~id:(fun (name, _, _) -> name)
    [
      ("123456789012345", 123456789012345., "123456789012345");
      ("1e15", 1e15, "1e+15");
    ]
    (fun (_, x, want) -> equal string want (P.exact x))

let test_exact_special =
  cases
    "writes a NaN as nan whatever its sign, the infinities as inf and -inf, \
     and keeps the sign of zero"
    ~id:(fun (name, _, _) -> name)
    [
      ("-0", -0., "-0");
      ("nan", Float.nan, "nan");
      ("-nan", -.Float.nan, "nan");
      ("infinity", Float.infinity, "inf");
      ("neg_infinity", Float.neg_infinity, "-inf");
    ]
    (fun (_, x, want) -> equal string want (P.exact x))

(* pp_exact *)

let test_pp_exact =
  prop "formats exact x" Gen.any_float (fun x ->
      equal string (P.exact x) (Format.asprintf "%a" P.pp_exact x))

(* The form check *)

let test_flaw =
  cases "form check names what breaks the positional form"
    ~id:(fun (s, decimals, _) ->
      match decimals with None -> s | Some n -> Printf.sprintf "%s at %d" s n)
    [
      (".5", None, "integer part not digits");
      ("0012", None, "leading zero");
      ("15.", None, "point without digits after it");
      ("1.50", None, "trailing zero after the point");
      ("12e+05", None, "not one nonzero digit before the exponent");
      ("0.1", Some 2, "not that many decimals");
      ("1", Some 1, "no decimals");
    ]
    (fun (s, decimals, want) ->
      equal (option string) (Some want) (flaw ?decimals s))

let () =
  exit
    (run "hugin float print"
       [
         group "g"
           [
             test_g;
             test_g_rounding;
             test_g_layout;
             test_g_floor;
             test_g_special;
           ];
         group "pp" [ test_pp ];
         group "f"
           [
             test_f;
             test_f_rounding;
             test_f_digits;
             test_f_floor;
             test_f_special;
           ];
         group "exact"
           [
             test_exact;
             test_exact_digits;
             test_exact_layout;
             test_exact_special;
           ];
         group "pp_exact" [ test_pp_exact ];
         test_flaw;
       ])
