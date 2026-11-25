(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* C99 Annex F's special values of the float kinds, F.10, with the signs of
   their zero results, each exact, any NaN for a NaN (prog.mli's Results); and
   beside them values that are none, as results whose exact value rounds to an
   infinity or a zero. A row of [Both] widths holds the f64 result, exact or
   correctly rounded, whose rounding to binary32 is the f32 one; a row of one
   width holds that width's operands and result. A kind of two operands takes
   them as prog.mli orders them: atan2 y x, pow x y. *)

type width = Both | F64 | F32

let inf = Float.infinity
let ninf = Float.neg_infinity
let nan = Float.nan
let pi = Float.pi

(* Rows of kind [k] at widths [w], of one operand and of two. *)
let table w k = List.map (fun (args, r) -> (w, k, args, r))
let one ?(w = Both) k rs = table w k (List.map (fun (x, r) -> ([| x |], r)) rs)

let two ?(w = Both) k rs =
  table w k (List.map (fun (x, y, r) -> ([| x; y |], r)) rs)

(* Rows of [(width, kind, operands, result)], the kind named as nx_kinds.h names
   it: Annex F's special values. *)
let rows =
  let pio2 = 0x1.921fb54442d18p0 and pio4 = 0x1.921fb54442d18p-1 in
  let tpio4 = 0x1.2d97c7f3321d2p1 in
  let exp_like k =
    one k [ (0., 1.); (-0., 1.); (ninf, 0.); (inf, inf); (nan, nan) ]
  in
  let log_like k =
    one k
      [
        (0., ninf);
        (-0., ninf);
        (1., 0.);
        (-1., nan);
        (ninf, nan);
        (inf, inf);
        (nan, nan);
      ]
  in
  let odd_periodic k =
    one k [ (0., 0.); (-0., -0.); (inf, nan); (ninf, nan); (nan, nan) ]
  in
  List.concat
    [
      exp_like "exp";
      exp_like "exp2";
      one "expm1" [ (0., 0.); (-0., -0.); (ninf, -1.); (inf, inf); (nan, nan) ];
      log_like "log";
      log_like "log2";
      one "log1p"
        [
          (0., 0.);
          (-0., -0.);
          (-1., ninf);
          (-2., nan);
          (ninf, nan);
          (inf, inf);
          (nan, nan);
        ];
      odd_periodic "sin";
      odd_periodic "tan";
      one "cos" [ (0., 1.); (-0., 1.); (inf, nan); (ninf, nan); (nan, nan) ];
      one "asin" [ (0., 0.); (-0., -0.); (2., nan); (-2., nan); (nan, nan) ];
      one "acos" [ (1., 0.); (2., nan); (-2., nan); (nan, nan) ];
      one "atan"
        [ (0., 0.); (-0., -0.); (inf, pio2); (ninf, -.pio2); (nan, nan) ];
      one "sinh" [ (0., 0.); (-0., -0.); (inf, inf); (ninf, ninf); (nan, nan) ];
      one "cosh" [ (0., 1.); (-0., 1.); (inf, inf); (ninf, inf); (nan, nan) ];
      one "tanh" [ (0., 0.); (-0., -0.); (inf, 1.); (ninf, -1.); (nan, nan) ];
      one "erf" [ (0., 0.); (-0., -0.); (inf, 1.); (ninf, -1.); (nan, nan) ];
      (* atan2 y x *)
      two "atan2"
        [
          (0., -0., pi);
          (-0., -0., -.pi);
          (0., 0., 0.);
          (-0., 0., -0.);
          (0., -1., pi);
          (-0., -1., -.pi);
          (0., 1., 0.);
          (-0., 1., -0.);
          (-1., 0., -.pio2);
          (-1., -0., -.pio2);
          (1., 0., pio2);
          (1., -0., pio2);
          (1., ninf, pi);
          (-1., ninf, -.pi);
          (1., inf, 0.);
          (-1., inf, -0.);
          (inf, 1., pio2);
          (ninf, 1., -.pio2);
          (inf, ninf, tpio4);
          (ninf, ninf, -.tpio4);
          (inf, inf, pio4);
          (ninf, inf, -.pio4);
          (nan, 1., nan);
          (1., nan, nan);
        ];
      two "pow"
        [
          (-0., -3., ninf);
          (0., -3., inf);
          (0., ninf, inf);
          (-0., ninf, inf);
          (0., -2., inf);
          (-0., -2., inf);
          (-0., -0.5, inf);
          (-0., 3., -0.);
          (0., 3., 0.);
          (-0., 2., 0.);
          (-0., 0.5, 0.);
          (-1., inf, 1.);
          (-1., ninf, 1.);
          (1., nan, 1.);
          (1., ninf, 1.);
          (nan, 0., 1.);
          (nan, -0., 1.);
          (inf, 0., 1.);
          (-2., 0.5, nan);
          (0.5, ninf, inf);
          (-0.5, ninf, inf);
          (2., ninf, 0.);
          (-2., ninf, 0.);
          (0.5, inf, 0.);
          (2., inf, inf);
          (ninf, -3., -0.);
          (ninf, -2., 0.);
          (ninf, -0.5, 0.);
          (ninf, 3., ninf);
          (ninf, 2., inf);
          (ninf, 0.5, inf);
          (inf, -1., 0.);
          (inf, 0.5, inf);
          (nan, 1., nan);
          (2., nan, nan);
        ];
      (* Odd integers end where the format's integers grow even. *)
      two ~w:F64 "pow"
        [
          (-0., 0x1.fffffffffffffp52, -0.);
          (-0., 0x1p53, 0.);
          (ninf, -0x1.fffffffffffffp52, -0.);
          (ninf, -0x1p53, 0.);
          (ninf, 0x1.fffffffffffffp52, ninf);
          (ninf, 0x1p53, inf);
        ];
      two ~w:F32 "pow"
        [
          (-0., 0x1.fffffep23, -0.);
          (-0., 0x1p24, 0.);
          (ninf, -0x1.fffffep23, -0.);
          (ninf, -0x1p24, 0.);
          (ninf, 0x1.fffffep23, ninf);
          (ninf, 0x1p24, inf);
        ];
    ]

(* Rows as {!rows}'s whose results are no special value of Annex F: an exact
   value that rounds to an infinity or a zero, a subnormal operand of atan2's
   that rounds the result to a zero or to pi, and -1 to the powers either side
   of where the format's integers grow even. Each is the correctly rounded
   result, which a kind gives within its bound in ranks; a zero result's sign is
   exact. *)
let rounded =
  List.concat
    [
      one "exp" [ (710., inf); (-1000., 0.) ];
      one ~w:F32 "exp" [ (89., inf); (-110., 0.) ];
      one "exp2" [ (1024., inf); (-1100., 0.); (-2000., 0.) ];
      one ~w:F32 "exp2" [ (128., inf); (-151., 0.) ];
      one "expm1" [ (710., inf) ];
      one ~w:F32 "expm1" [ (89., inf) ];
      one "sinh" [ (711., inf); (-711., ninf) ];
      one ~w:F32 "sinh" [ (90., inf); (-90., ninf) ];
      one "cosh" [ (711., inf); (-711., inf) ];
      one ~w:F32 "cosh" [ (90., inf); (-90., inf) ];
      two ~w:F64 "atan2" [ (-0x1p-1074, 1e300, -0.); (0x1p-1074, -1e300, pi) ];
      two ~w:F32 "atan2"
        [ (-0x1p-149, 0x1p127, -0.); (0x1p-149, -0x1p127, Exact.round_f32 pi) ];
      two "pow"
        [
          (-0x1p-60, 3., -0x1p-180);
          (-0x1p-600, 3., -0.);
          (2., 1024., inf);
          (-2., 1025., ninf);
          (2., -1100., 0.);
          (-2., -1101., -0.);
          (0.5, 1100., 0.);
          (-0.5, 1101., -0.);
        ];
      two ~w:F64 "pow" [ (-1., 0x1.fffffffffffffp52, -1.); (-1., 0x1p53, 1.) ];
      two ~w:F32 "pow"
        [
          (-1., 0x1.fffffep23, -1.);
          (-1., 0x1p24, 1.);
          (2., 128., inf);
          (-2., 129., ninf);
          (2., -151., 0.);
          (-2., -151., -0.);
        ];
    ]

(* A row's name: its widths, its kind and its operands. *)
let id (w, k, args, _) =
  Printf.sprintf "%s %s %s"
    (match w with Both -> "f64 and f32" | F64 -> "f64" | F32 -> "f32")
    k
    (String.concat " " (Array.to_list (Array.map (Printf.sprintf "%h") args)))
