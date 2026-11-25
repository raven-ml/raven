(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Decimal text read as floats. Whether a float is the one a decimal reads as is
   decided by [Exact.reads], in integers, so that no C library takes part. *)

open Windtrap
module F = Munin_float_text.Float_text
module P = Munin_float_text.Float_print
module R = Munin_float_text.Float_read

(* Inputs *)

(* Decimals in JSON's syntax, of up to 40 digits and exponents past either end
   of the floats. *)
let gen_decimal =
  let open Gen in
  let digits = string_of ~size:(int_range 1 20) (char_range '0' '9') in
  map
    (fun ((neg, (int, frac)), e) ->
      (if neg then "-" else "")
      ^ (if int.[0] = '0' && String.length int > 1 then "1" ^ int else int)
      ^ (if frac = "" then "" else "." ^ frac)
      ^ match e with None -> "" | Some e -> "e" ^ string_of_int e)
    (pair
       (pair bool
          (pair digits (string_of ~size:(int_range 0 20) (char_range '0' '9'))))
       (option (int_range (-340) 330)))

(* A decimal of at most 18 significant digits, with an exponent that keeps most
   of them within 10^-22 to 10^22 of an integer. *)
let gen_short_decimal =
  let open Gen in
  map
    (fun ((neg, m), (e, point)) ->
      let d = string_of_int m in
      let point = Int.min point (String.length d) in
      let int = String.sub d 0 (String.length d - point)
      and frac = String.sub d (String.length d - point) point in
      (if neg then "-" else "")
      ^ (if int = "" then "0" else int)
      ^ (if frac = "" then "" else "." ^ frac)
      ^ if e = 0 then "" else "e" ^ string_of_int e)
    (pair
       (pair bool
          (one_of
             [
               int_range 1 (1 lsl 53);
               int_range (1 lsl 53) 999_999_999_999_999_999;
             ]))
       (pair (int_range (-30) 30) (int_range 0 20)))

(* A midpoint between two floats [M 2^k] and [(M + 1) 2^k], [M] from 2^52 to
   2^53, as [D 10^-j] with [D] of at most 18 digits, or [D] moved by one unit
   below or above it, and the float it reads as: the even one of the two at the
   midpoint, else the nearer. [padded] adds three zeros to [D], taking the text
   past 18 digits. *)
let gen_midpoint =
  let open Gen in
  map
    (fun (((m, k), off), padded) ->
      let mid = (2 * m) + 1 in
      let d, j =
        match k with
        | -1 -> (mid * 25, 2)
        | 0 -> (mid * 5, 1)
        | k -> (mid lsl (k - 1), 0)
      in
      let text =
        if padded then Printf.sprintf "%d000e-%d" (d + off) (j + 3)
        else Printf.sprintf "%de-%d" (d + off) j
      in
      let lo = Float.ldexp (Float.of_int m) k
      and hi = Float.ldexp (Float.of_int (m + 1)) k in
      let expected =
        if off < 0 then lo
        else if off > 0 then hi
        else if m mod 2 = 0 then lo
        else hi
      in
      (text, off, padded, expected))
    (pair
       (pair
          (pair (int_range (1 lsl 52) ((1 lsl 53) - 1)) (int_range (-1) 6))
          (int_range (-1) 1))
       bool)

(* Floats spread over the whole range, and as often near 1, where a store's
   numbers are. *)
let gen_float =
  let open Gen in
  frequency
    [
      (1, float);
      ( 1,
        let+ d = int_range 1 999_999
        and+ j = int_range (-30) 30
        and+ neg = bool in
        let x = float_of_string (Printf.sprintf "%de%d" d j) in
        if neg then -.x else x );
    ]

(* Text a reader must not read into: digits, points, exponents and signs. *)
let gen_beside =
  Gen.(
    string_of ~size:(int_range 0 3)
      (of_list (String.to_seq "0123456789.eE-+" |> List.of_seq)))

(* of_string *)

(* [nearest_law s] checks that [s] reads as the float nearest it, with its
   sign. *)
let nearest_law s =
  let neg, ((d, e) as de) = Exact.decimal s in
  let short = String.length d <= 18 && abs e <= 22 in
  cover "at most 2^53 times a power of ten within 10^22"
    (short && int_of_string d <= 1 lsl 53);
  cover "past 2^53, at most 18 digits within 10^22"
    (short && int_of_string d > 1 lsl 53);
  cover "more digits or a larger exponent" (not short);
  let x = require_some (R.of_string s) in
  satisfies ~claim:"has the decimal's sign" float_exact
    (fun x -> Float.sign_bit x = neg)
    x;
  if Float.is_finite x then
    satisfies ~claim:"is the float the decimal reads as" float_exact
      (fun x -> Exact.reads F.binary64 x de)
      x
  else
    satisfies ~claim:"is an infinity of a decimal past the greatest float"
      float_exact
      (fun _ -> Exact.overflows de)
      x

let test_nearest =
  prop "a decimal reads as the float nearest it" ~count:300
    (Gen.one_of [ gen_decimal; gen_short_decimal ])
    nearest_law

let test_midpoint =
  prop "a midpoint reads as the even float, a unit off it as the nearer"
    ~count:200 gen_midpoint (fun (text, off, padded, expected) ->
      cover "a midpoint" (off = 0);
      cover "a midpoint past 18 digits" (off = 0 && padded);
      equal float_exact expected (require_some (R.of_string text)))

let test_libc =
  test "a decimal the C library reads wrongly somewhere reads exactly"
    (fun () ->
      (* mingw-w64's strtod reads it one float above its nearest. *)
      equal float_exact
        (Float.of_string "0x1.fffffffffffffp-793")
        (require_some (R.of_string "3.839223843572815e-239")))

(* IEEE 754's rounding to nearest, ties to even, at the ends of the floats and
   of their exact integers. The greatest float is 1.7976931348623157e308 and the
   overflow threshold 1.797693134862315807937...e308. *)
let boundary =
  [
    ("0.1", 0.1);
    ("9007199254740993", 0x1p53);
    ("9007199254740995", 0x1p53 +. 4.);
    ("1.7976931348623157e308", max_float);
    ("1.7976931348623158e308", max_float);
    ("1.7976931348623159e308", Float.infinity);
    ("-1.7976931348623159e308", Float.neg_infinity);
    ("1e-400", 0.);
    ("-1e-400", -0.);
    ("-0", -0.);
    ("0e999999999", 0.);
  ]

let test_boundary =
  cases "a decimal at a boundary reads as IEEE 754 rounds it" ~id:fst boundary
    (fun (text, expected) ->
      equal float_exact expected (require_some (R.of_string text)))

let test_half_subnormal =
  test "half the least subnormal reads as zero, a unit above it as 5e-324"
    (fun () ->
      (* 2^-1075 = 5^1075 10^-1075, the midpoint between 0 and 2^-1074; zero is
         even. *)
      let half = Exact.pow5 1075 in
      let above = Exact.pow5 1075 ^ "1" in
      equal ~msg:"half" float_exact 0.
        (require_some (R.of_string (half ^ "e-1075")));
      equal ~msg:"above" float_exact 5e-324
        (require_some (R.of_string (above ^ "e-1076"))))

let test_infinity =
  cases "a decimal beyond the greatest float reads as an infinity" ~id:fst
    [
      ("688706.565804865188962e303", Float.infinity);
      ("-688706.565804865188962e303", Float.neg_infinity);
    ]
    (fun (text, expected) ->
      equal ~msg:"read" (option float_exact) (Some expected) (R.read text);
      equal ~msg:"of_string" (option float_exact) (Some expected)
        (R.of_string text))

let not_numbers =
  [
    "";
    "-";
    "+1";
    "01";
    ".5";
    "1.";
    "1e";
    "1e+";
    " 1";
    "1 ";
    "--1";
    "1.5.2";
    "1e5.0";
    "nan";
    "inf";
    "Infinity";
    "0x10";
    "1_000";
  ]

let test_not_numbers =
  cases "text that is not one number in JSON's syntax is None"
    ~id:(Printf.sprintf "%S") not_numbers (fun s ->
      equal (option float_exact) None (R.of_string s))

let test_round_trip =
  prop "a float's exact text reads back as it" gen_float (fun x ->
      cover "within 10^-22 to 10^22"
        (Float.abs x >= 1e-22 && Float.abs x <= 1e22);
      Law.round_trip float_exact string P.exact
        (fun s -> require_some (R.of_string s))
        x)

(* number_end *)

(* Rows of [(s, i, stop, expected)], from the sentence of [number_end]. *)
let ends =
  [
    ("0123", 0, 4, 1);
    ("-0.5e-3", 0, 7, 7);
    ("1E+5", 0, 4, 4);
    ("12,3", 0, 4, 2);
    ("12", 0, 1, 1);
    ("1.5e3", 0, 3, 3);
    ("x12", 1, 3, 3);
    ("", 0, 0, -1);
    ("-", 0, 1, -1);
    ("abc", 0, 3, -1);
    ("1.", 0, 2, -1);
    ("1e", 0, 2, -1);
    ("1.5e+", 0, 5, -1);
  ]

let test_ends =
  cases "number_end is the index after the longest number"
    ~id:(fun (s, i, stop, _) -> Printf.sprintf "%S %d %d" s i stop)
    ends
    (fun (s, i, stop, expected) -> equal int expected (R.number_end s i stop))

(* of_substring *)

let test_substring =
  prop "a substring reads as its copy"
    Gen.(triple gen_beside gen_decimal gen_beside)
    (fun (before, text, after) ->
      let digit s i =
        i >= 0 && i < String.length s && s.[i] >= '0' && s.[i] <= '9'
      in
      cover "a digit on both sides"
        (digit before (String.length before - 1) && digit after 0);
      let s = before ^ text ^ after in
      equal (option float_exact) (R.of_string text)
        (R.of_substring s (String.length before) (String.length text)))

let test_substring_bounds =
  cases "of_substring refuses what names no substring"
    ~id:(fun (pos, len) -> Printf.sprintf "pos %d len %d of \"123\"" pos len)
    [ (-1, 1); (0, -1); (2, 2); (4, 0) ]
    (fun (pos, len) ->
      raises_match Exn.invalid_arg (fun () -> R.of_substring "123" pos len))

let () =
  exit
    (run "munin float read"
       [
         group "of_string"
           [
             test_nearest;
             test_midpoint;
             test_libc;
             test_boundary;
             test_half_subnormal;
             test_infinity;
             test_not_numbers;
             test_round_trip;
           ];
         group "number_end" [ test_ends ];
         group "of_substring" [ test_substring; test_substring_bounds ];
       ])
