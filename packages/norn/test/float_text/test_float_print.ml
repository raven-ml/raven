(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The text norn writes floats as: C's %g and %f where the C library rounds
   once. *)

open Windtrap
module F = Norn_float_text.Float_text
module P = Norn_float_text.Float_print

let sign x = if Float.sign_bit x then "-" else ""

(* [c_g p x] is the finite [x] as C11 7.21.6.1p8 lays out [%.*g] of its [p]
   significant digits, [F.significant]: "if P > X >= -4, the conversion is with
   style f ... and precision P - (X + 1)", "otherwise ... with style e", whose
   exponent "always contains at least two digits"; then "any trailing zeros are
   removed from the fractional portion of the result" and "the decimal-point
   character is removed if there is no fractional portion remaining". *)
let c_g p x =
  let p = max 1 p in
  let d, e = F.significant p x in
  let n = String.length d in
  let big_x = n - 1 + e in
  let body =
    if d = "0" then "0"
    else if p > big_x && big_x >= -4 then
      if e >= 0 then d ^ String.make e '0'
      else if big_x >= 0 then
        String.sub d 0 (big_x + 1)
        ^ "."
        ^ String.sub d (big_x + 1) (n - big_x - 1)
      else "0." ^ String.make (-big_x - 1) '0' ^ d
    else
      let fraction = if n > 1 then "." ^ String.sub d 1 (n - 1) else "" in
      Printf.sprintf "%c%se%c%02d" d.[0] fraction
        (if big_x < 0 then '-' else '+')
        (abs big_x)
  in
  sign x ^ body

(* [c_f n x] is the finite [x] as C11 7.21.6.1p8 lays out [%.*f] of its
   [F.decimals n]: "the number of digits after the decimal-point character is
   equal to the precision", and with precision zero "no decimal-point character
   appears". *)
let c_f n x =
  let n = max 0 n in
  let d, e = F.decimals n x in
  (* The value times [10^n], an integer of at least [n + 1] digits. *)
  let scaled = if d = "0" then "0" else d ^ String.make (e + n) '0' in
  let scaled =
    String.make (max 0 (n + 1 - String.length scaled)) '0' ^ scaled
  in
  let int = String.length scaled - n in
  sign x ^ String.sub scaled 0 int
  ^ if n = 0 then "" else "." ^ String.sub scaled int n

(* Finite floats of either sign, from every exponent, from short decimals and
   from the subnormals. *)
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

let g =
  group "g"
    [
      prop "lays out the significant digits as C's %g" ~count:2000
        Gen.(pair finite (int_range (-1) 20))
        (fun (x, p) ->
          classes x;
          equal string (c_g p x) (P.g p x));
      cases "writes the stated texts"
        ~id:(fun (p, x, _) -> Printf.sprintf "%d %h" p x)
        [
          (6, 135.5625, "135.562");
          (6, 1e16, "1e+16");
          (6, 1.5e-5, "1.5e-05");
          (6, 1e-4, "0.0001");
          (6, 123456., "123456");
          (6, 1234567., "1.23457e+06");
          (6, 1e100, "1e+100");
          (4, 68405., "6.84e+04");
          (0, 2.5, "2");
          (17, 0.1, "0.10000000000000001");
          (6, 0., "0");
          (6, -0., "-0");
          (6, -1.5, "-1.5");
          (6, Float.nan, "nan");
          (6, -.Float.nan, "nan");
          (6, Float.infinity, "inf");
          (6, Float.neg_infinity, "-inf");
        ]
        (fun (p, x, text) -> equal string text (P.g p x));
    ]

let f =
  group "f"
    [
      prop "lays out the decimals as C's %f" ~count:2000
        Gen.(pair finite (int_range (-1) 30))
        (fun (x, n) ->
          classes x;
          equal string (c_f n x) (P.f n x));
      cases "writes the stated texts"
        ~id:(fun (n, x, _) -> Printf.sprintf "%d %h" n x)
        [
          (2, 0.125, "0.12");
          (0, 0.5, "0");
          (0, 1.5, "2");
          (1, 2.25, "2.2");
          (6, 0.1, "0.100000");
          (2, -0.001, "-0.00");
          (-1, 3.7, "4");
          (2, 1e21, "1000000000000000000000.00");
          (2, Float.nan, "nan");
        ]
        (fun (n, x, text) -> equal string text (P.f n x));
    ]

let () = exit (run "norn float print" [ g; f ])
