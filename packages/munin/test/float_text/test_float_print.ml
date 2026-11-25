(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The text munin writes floats as: C's %g and %f where the C library rounds
   once, and the shortest digits that read back. A text is read back to its sign
   and decimal by [Exact.decimal], by hand, so that no C library takes part. *)

open Windtrap
module F = Munin_float_text.Float_text
module P = Munin_float_text.Float_print

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

let signed_decimal = pair bool (pair string int)

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

let test_g =
  prop "writes the significant digits as %g does" ~count:2000
    Gen.(pair finite (int_range 1 20))
    (fun (x, p) ->
      classes x;
      let s = P.g ~precision:p x in
      let want = F.significant p x in
      equal ~msg:s signed_decimal (Float.sign_bit x, want) (Exact.decimal s);
      check_layout s p want)

let test_f =
  prop "writes the decimals as %f does" ~count:2000
    Gen.(pair finite (int_range 0 30))
    (fun (x, n) ->
      classes x;
      let s = P.f n x in
      equal ~msg:s signed_decimal
        (Float.sign_bit x, F.decimals n x)
        (Exact.decimal s);
      equal ~msg:s (option string) None (flaw ~decimals:n s))

let test_exact =
  prop "writes the shortest digits as %g does" ~count:2000 finite (fun x ->
      classes x;
      let s = P.exact x in
      let ((d, _) as want) = F.shortest F.binary64 x in
      equal ~msg:s signed_decimal (Float.sign_bit x, want) (Exact.decimal s);
      check_layout s (max 15 (String.length d)) want)

(* Stated cases: the .mli's examples, C's text for NaN, the infinities and the
   signed zero, and the layout's edges: the exponent from -4 to below the
   precision is positional, and an exponent takes three digits past 99. *)

let test_g_rows =
  cases "is the stated text"
    ~id:(fun (p, x, _) -> Printf.sprintf "%%.%dg %h" p x)
    [
      (6, 135.5625, "135.562");
      (6, 1e16, "1e+16");
      (6, 1.5e-5, "1.5e-05");
      (6, 1e-4, "0.0001");
      (6, 123456., "123456");
      (6, 1234567., "1.23457e+06");
      (6, 1e100, "1e+100");
      (4, 68405., "6.84e+04");
      (17, 0.1, "0.10000000000000001");
      (6, 0., "0");
      (6, -0., "-0");
      (6, -1.5, "-1.5");
      (6, Float.nan, "nan");
      (6, -.Float.nan, "nan");
      (6, Float.infinity, "inf");
      (6, Float.neg_infinity, "-inf");
    ]
    (fun (precision, x, want) -> equal string want (P.g ~precision x))

let test_g_default =
  test "takes 6 digits by default" (fun () ->
      equal string "1.23457e+06" (P.g 1234567.))

let test_f_rows =
  cases "is the stated text"
    ~id:(fun (n, x, _) -> Printf.sprintf "%%.%df %h" n x)
    [
      (2, 0.125, "0.12");
      (0, 0.5, "0");
      (0, 1.5, "2");
      (1, 2.25, "2.2");
      (6, 0.1, "0.100000");
      (2, -0.001, "-0.00");
      (2, 1e21, "1000000000000000000000.00");
      (2, Float.nan, "nan");
      (2, -.Float.nan, "nan");
      (2, Float.infinity, "inf");
      (2, Float.neg_infinity, "-inf");
    ]
    (fun (n, x, want) -> equal string want (P.f n x))

let test_exact_rows =
  cases "is the stated text"
    ~id:(fun (x, _) -> Printf.sprintf "%h" x)
    [
      (0.1, "0.1");
      (0.1 +. 0.2, "0.30000000000000004");
      (1e15, "1e+15");
      (123456789012345., "123456789012345");
      (0x1p53 +. 2., "9007199254740994");
      (0x1p-24, "5.960464477539063e-08");
      (5e-324, "5e-324");
      (max_float, "1.7976931348623157e+308");
      (-0., "-0");
      (Float.nan, "nan");
      (Float.infinity, "inf");
      (Float.neg_infinity, "-inf");
    ]
    (fun (x, want) -> equal string want (P.exact x))

let test_exact_float_rows =
  cases "is the stated text"
    ~id:(fun (x, _) -> Printf.sprintf "%h" x)
    [
      (1., "1.0");
      (-0., "-0.0");
      (0.1, "0.1");
      (1e15, "1e+15");
      (0x1p53 +. 2., "9007199254740994.0");
    ]
    (fun (x, want) -> equal string want (P.exact_float x))

let () =
  exit
    (run "munin float print"
       [
         group "g" [ test_g; test_g_rows; test_g_default ];
         group "f" [ test_f; test_f_rows ];
         group "exact" [ test_exact; test_exact_rows ];
         group "exact_float" [ test_exact_float_rows ];
       ])
