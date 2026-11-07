(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The text munin writes floats as: C's %g and %f where the C library rounds
   once, and the shortest digits that read back. *)

open Windtrap
module F = Munin_float_text.Float_text
module P = Munin_float_text.Float_print

(* [decimal s] is the sign and the decimal of the finite number [s] writes, as
   [(d, e)] without leading or trailing zeros, [("0", 0)] for zero, read by
   hand so that no C library takes part. *)
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
    ( neg,
      (String.sub d !first (!last - !first + 1), x - frac + (n - 1 - !last)) )

(* [exponent s] is the text after [e] in [s], if any. *)
let exponent s =
  match String.index_opt s 'e' with
  | None -> None
  | Some i -> Some (String.sub s (i + 1) (String.length s - i - 1))

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

(* Laws *)

let signed_decimal = pair bool (pair string int)

(* [flaw ~decimals s] is what breaks the positional form in the digits of
   [s], its text after any sign and before any exponent, or [None]: an integer
   part of digits that is [0] or starts with a nonzero digit, then a point and
   digits only if a fraction follows. With [~decimals:n] the fraction has
   exactly [n] digits, as %f writes it; without, it ends in a nonzero digit,
   as %g writes it, and before an exponent the integer part is one nonzero
   digit. *)
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

(* C's %g layout at precision [p]: an exponent of a sign and at least two
   digits when the leading digit's exponent is below -4 or at least [p], the
   positional form otherwise. *)
let check_layout s p (d, e) =
  let lead = String.length d - 1 + e in
  let want =
    if d <> "0" && (lead < -4 || lead >= p) then
      Some (Printf.sprintf "%c%02d" (if lead < 0 then '-' else '+') (abs lead))
    else None
  in
  equal ~msg:s (option string) want (exponent s);
  equal ~msg:s (option string) None (flaw s)

let test_g =
  prop "g writes significant digits as %g does" ~count:2000
    Gen.(pair finite (int_range 1 20))
    (fun (x, p) ->
      classes x;
      let s = P.g ~precision:p x in
      let want = F.significant p x in
      equal ~msg:s signed_decimal (Float.sign_bit x, want) (decimal s);
      check_layout s p want)

let test_f =
  prop "f writes decimals as %f does" ~count:2000
    Gen.(pair finite (int_range 0 30))
    (fun (x, n) ->
      classes x;
      let s = P.f n x in
      equal ~msg:s signed_decimal
        (Float.sign_bit x, F.decimals n x)
        (decimal s);
      equal ~msg:s (option string) None (flaw ~decimals:n s))

let test_exact =
  prop "exact writes the shortest digits as %g does" ~count:2000 finite
    (fun x ->
      classes x;
      let s = P.exact x in
      let ((d, _) as want) = F.shortest F.binary64 x in
      equal ~msg:s signed_decimal (Float.sign_bit x, want) (decimal s);
      check_layout s (max 15 (String.length d)) want)

(* Stated cases *)

let written =
  [
    ("g 135.5625", P.g 135.5625, "135.562");
    ("g 1e16", P.g 1e16, "1e+16");
    ("g 1.5e-5", P.g 1.5e-5, "1.5e-05");
    ("g 1e-4", P.g 1e-4, "0.0001");
    ("g 123456", P.g 123456., "123456");
    ("g 1234567", P.g 1234567., "1.23457e+06");
    ("g 1e100", P.g 1e100, "1e+100");
    ("g 4 68405", P.g ~precision:4 68405., "6.84e+04");
    ("g 0 2.5", P.g ~precision:0 2.5, "2");
    ("g 17 0.1", P.g ~precision:17 0.1, "0.10000000000000001");
    ("g 0", P.g 0., "0");
    ("g -0", P.g (-0.), "-0");
    ("g -1.5", P.g (-1.5), "-1.5");
    ("g nan", P.g Float.nan, "nan");
    ("g -nan", P.g (-.Float.nan), "nan");
    ("g inf", P.g Float.infinity, "inf");
    ("g -inf", P.g Float.neg_infinity, "-inf");
    ("f 2 0.125", P.f 2 0.125, "0.12");
    ("f 0 0.5", P.f 0 0.5, "0");
    ("f 0 1.5", P.f 0 1.5, "2");
    ("f 1 2.25", P.f 1 2.25, "2.2");
    ("f 6 0.1", P.f 6 0.1, "0.100000");
    ("f 2 -0.001", P.f 2 (-0.001), "-0.00");
    ("f -1 3.7", P.f (-1) 3.7, "4");
    ("f 2 1e21", P.f 2 1e21, "1000000000000000000000.00");
    ("f 2 nan", P.f 2 Float.nan, "nan");
    ("exact 0.1", P.exact 0.1, "0.1");
    ("exact 0.1 + 0.2", P.exact (0.1 +. 0.2), "0.30000000000000004");
    ("exact 1e15", P.exact 1e15, "1e+15");
    ("exact 123456789012345", P.exact 123456789012345., "123456789012345");
    ("exact 2^53 + 2", P.exact (0x1p53 +. 2.), "9007199254740994");
    ("exact 2^-24", P.exact 0x1p-24, "5.960464477539063e-08");
    ("exact 5e-324", P.exact 5e-324, "5e-324");
    ("exact max_float", P.exact max_float, "1.7976931348623157e+308");
    ("exact -0", P.exact (-0.), "-0");
    ("exact -inf", P.exact Float.neg_infinity, "-inf");
    ("exact_float 1", P.exact_float 1., "1.0");
    ("exact_float -0", P.exact_float (-0.), "-0.0");
    ("exact_float 0.1", P.exact_float 0.1, "0.1");
    ("exact_float 1e15", P.exact_float 1e15, "1e+15");
    ( "exact_float 2^53 + 2",
      P.exact_float (0x1p53 +. 2.),
      "9007199254740994.0" );
  ]

let test_written =
  cases "written" ~id:(fun (name, _, _) -> name) written (fun (_, got, want) ->
      equal string want got)

(* The form check itself, on texts that are not in the form. *)
let malformed =
  [
    (".5", None, "integer part not digits");
    ("0012", None, "leading zero");
    ("15.", None, "point without digits after it");
    ("1.50", None, "trailing zero after the point");
    ("12e+05", None, "not one nonzero digit before the exponent");
    ("0.1", Some 2, "not that many decimals");
    ("1", Some 1, "no decimals");
  ]

let test_malformed =
  cases "form check" ~id:(fun (s, _, _) -> s) malformed
    (fun (s, decimals, want) ->
      equal (option string) (Some want) (flaw ?decimals s))

let () =
  exit
    (run "munin float print"
       [
         group "laws" [ test_g; test_f; test_exact ];
         group "cases" [ test_written; test_malformed ];
       ])
