(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The float kinds, lowered and compiled on the host. An exact kind gives nx's
   eager bits, any NaN for a NaN. A transcendental kind is within its host bound
   of a reference: at float32, OCaml's float64 functions rounded once; at
   float64, nx's eager kernels, within the sum of both bounds. Each case is one
   compiled map over a few hundred values of float32 or float64, drawn to reach
   the kind's special values and the edges of its ranges. *)

open Windtrap
module D = Nx_array.Dtype
module K = Nx_kernel.Prog
module P = Nx.Prim

let n = 256

(* Values *)

let f32 v = Int32.float_of_bits (Int32.bits_of_float v)
let round wide v = if wide then v else f32 v

let nans wide =
  if wide then
    List.map Int64.float_of_bits
      [
        0x7FF8000000000000L;
        0xFFF8000000000000L;
        0x7FF8000000000123L;
        0xFFFC0000DEADBEEFL;
        0x7FFFFFFFFFFFFFFFL;
      ]
  else
    List.map Int32.float_of_bits
      [ 0x7FC00000l; 0xFFC00000l; 0x7FC00123l; 0xFFE12345l; 0x7FFFFFFFl ]

(* The least subnormal, the greatest subnormal, the least normal and the
   greatest float. *)
let edges wide =
  if wide then
    List.map Int64.float_of_bits
      [ 1L; 0x000FFFFFFFFFFFFFL; 0x0010000000000000L; 0x7FEFFFFFFFFFFFFFL ]
  else
    List.map Int32.float_of_bits [ 1l; 0x007FFFFFl; 0x00800000l; 0x7F7FFFFFl ]

let specials wide =
  let signed = List.concat_map (fun v -> [ v; -.v ]) in
  nans wide @ signed ([ 0.; Float.infinity; 1.; 0.5; 2. ] @ edges wide)

let is_nan_neg v = Float.is_nan v && Int64.bits_of_float v < 0L
let is_neg_zero v = v = 0. && 1. /. v < 0.

let is_sub wide v =
  v <> 0. && Float.abs v < if wide then Float.min_float else 0x1p-126

(* [step wide v d] is the float [d] ranks of bits from [v]. *)
let step wide v d =
  if wide then
    Int64.float_of_bits (Int64.add (Int64.bits_of_float v) (Int64.of_int d))
  else Int32.float_of_bits (Int32.add (Int32.bits_of_float v) (Int32.of_int d))

let any_bits wide =
  if wide then Gen.map Int64.float_of_bits Gen.int64
  else Gen.map Int32.float_of_bits Gen.int32

(* A value of the dtype [wide] says: a special value, one of [points] or a
   neighbour, any bits, or a value of [range]. *)
let value wide points (lo, hi) =
  let points = List.map (round wide) points in
  Gen.frequency
    [
      (2, Gen.of_list (specials wide));
      ( 4,
        if points = [] then Gen.of_list (specials wide)
        else
          Gen.bind (Gen.of_list points) (fun p ->
              Gen.map (step wide p) (Gen.int_range (-2) 2)) );
      (2, any_bits wide);
      (3, Gen.map (round wide) (Gen.float_range lo hi));
    ]

(* Kinds *)

(* A kind's operands: per operand, the points and the range its values come
   from; its special tuples; the cases it must reach, on one element of the
   dtype [wide]; and, for a transcendental kind, its float64 function and its
   bound in ulps at an element of the dtype [wide]. *)
type kind = {
  name : string;
  node : K.node;
  points : (bool -> float list) list;
  range : bool -> float * float;
  tuples : bool -> float array list;
  covers : (string * (bool -> float array -> bool)) list;
  within : ((float array -> float) * (bool -> float array -> float)) option;
}

let arity k = List.length k.points

let kind ?(tuples = fun _ -> []) ?(covers = []) ?within ~range name node points
    =
  { name; node; points; range; tuples; covers; within }

(* [f1 f b] and [f2 f b]: the float64 function [f] of one or two operands, and
   the bound [b] at each dtype. *)
let f1 f b = Some ((fun e -> f e.(0)), fun w _ -> if w then snd b else fst b)

let f2 f b =
  Some ((fun e -> f e.(0) e.(1)), fun w _ -> if w then snd b else fst b)

let fixed r _ = r
let by32 r32 r64 wide = if wide then r64 else r32
let on f = fun wide (e : float array) -> f wide e.(0)
let both a b wide = a wide @ b wide

(* Each kind's points: the ends of its ranges, where its algorithm switches and
   where its result saturates, at both dtypes. *)

let rounding _ =
  [
    0.5; 1.5; 2.5; 0x1p22 +. 0.5; 0x1p23; 0x1p52; 4503599627370495.5; 0.49999997;
  ]

let exp_points _ =
  [
    89.;
    -104.;
    86.;
    88.72284;
    -87.33655;
    -103.27893;
    -103.97208;
    710.;
    -746.;
    708.;
    709.782712893384;
    -708.3964185322641;
    -744.4400719213812;
    -745.1332191019411;
    0.34657359;
  ]

let exp2_points _ =
  [
    128.;
    129.;
    -126.;
    -149.;
    -150.;
    -151.;
    1024.;
    1025.;
    -1022.;
    -1074.;
    -1075.;
    -1076.;
    0.5;
    -0.5;
    3.;
  ]

let expm1_points _ =
  [
    -26.;
    -40.;
    0x1p-25;
    0x1p-54;
    0x1p-30;
    88.72284;
    709.782712893384;
    89.;
    710.;
    0.34657359;
    44.;
    89.;
    45.;
  ]

let log_points _ =
  [
    1.;
    2.;
    0.5;
    0.70710677;
    1.4142135;
    0.70710678118654752;
    1.4142135623730951;
    3.;
    10.;
    0x1p-126;
    0x1p-1022;
    1e-40;
    1e-310;
  ]

let log1p_points w = [ -1.; -0.5; 0x1p-25; 0x1p-54; -0.29289322 ] @ log_points w

let trig_points _ =
  [
    0x1p15;
    0x1p28;
    0x1p-12;
    0x1p-27;
    0.78539816;
    1.5707963267948966;
    3.141592653589793;
    1e22;
    6381956970095103. *. 0x1p797;
    1e30;
    1e300;
    105414350.;
    2.1904577e+38;
  ]

let asin_points _ = [ 1.; -1.; 0.5; -0.5; 0.99999994; 0.9999999999999999 ]

let atan_points _ =
  [ 1.; -1.; 0.4375; 0.6875; 1.1875; 2.4375; 1e10; 1e300; 0x1p-960 ]

let hyp_points _ =
  [
    90.;
    712.;
    89.41599;
    710.4758600739439;
    22.;
    0.34657359;
    44.;
    45.;
    0x1p-30;
    6.;
  ]

let tanh_points _ = [ 0.625; 9.5; 19.5; 0x1p-30; 0.3 ]
let erf_points _ = [ 1.; 2.5; 4.; 5.95; 0x1p-100; 1.75; 6.; 0.5; 4.25 ]
let wide_range = by32 (-1e3, 1e3) (-1e3, 1e3)
let none _ = []

let unary =
  let u = kind ~range:wide_range in
  [
    u "neg" (Op1 (Unary Neg, D.Any D.Float32, 0)) [ none ];
    u "abs" (Op1 (Unary Abs, D.Any D.Float32, 0)) [ none ];
    u "sign" (Op1 (Unary Sign, D.Any D.Float32, 0)) [ none ];
    u "recip" (Op1 (Unary Recip, D.Any D.Float32, 0)) [ none ];
    u "sqrt" (Op1 (Unary Sqrt, D.Any D.Float32, 0)) [ log_points ];
    u "floor" (Op1 (Unary Floor, D.Any D.Float32, 0)) [ rounding ];
    u "ceil" (Op1 (Unary Ceil, D.Any D.Float32, 0)) [ rounding ];
    u "round"
      (Op1 (Unary Round, D.Any D.Float32, 0))
      [ rounding ]
      ~covers:
        [ ("a half", on (fun _ x -> Float.abs (x -. Float.trunc x) = 0.5)) ];
    u "trunc" (Op1 (Unary Trunc, D.Any D.Float32, 0)) [ rounding ];
    kind "exp"
      (Op1 (Unary Exp, D.Any D.Float32, 0))
      ?within:(f1 Float.exp (2., 2.))
      [ exp_points ]
      ~range:(by32 (-120., 120.) (-800., 800.))
      ~covers:
        [
          ("overflow", on (fun w x -> x > if w then 709.79 else 88.73));
          ("underflow to zero", on (fun w x -> x < if w then -745.2 else -104.));
          ( "a subnormal result",
            on (fun w x ->
                if w then x < -708.4 && x > -745. else x < -87.4 && x > -103.)
          );
          ( "the near range",
            on (fun w x -> Float.abs x <= if w then 708. else 86.) );
        ];
    kind "exp2"
      (Op1 (Unary Exp2, D.Any D.Float32, 0))
      ?within:(f1 Float.exp2 (4., 4.))
      [ exp2_points ]
      ~range:(by32 (-160., 160.) (-1100., 1100.))
      ~covers:
        [
          ("an integer", on (fun _ x -> Float.is_integer x));
          ("overflow", on (fun w x -> x >= if w then 1024. else 128.));
          ( "a subnormal result",
            on (fun w x ->
                (x < if w then -1022. else -126.)
                && x > if w then -1075. else -150.) );
        ];
    kind "expm1"
      (Op1 (Unary Expm1, D.Any D.Float32, 0))
      ?within:(f1 Float.expm1 (21., 21.))
      [ expm1_points ]
      ~range:(by32 (-40., 100.) (-50., 720.))
      ~covers:
        [
          ( "tiny",
            on (fun w x ->
                x <> 0. && Float.abs x < if w then 0x1p-54 else 0x1p-25) );
          ("past the cap", on (fun w x -> x > if w then 89. else 45.));
          ("saturated below", on (fun w x -> x < if w then -40. else -26.));
        ];
    kind "log"
      (Op1 (Unary Log, D.Any D.Float32, 0))
      ?within:(f1 Float.log (1., 1.))
      [ log_points ]
      ~range:(fixed (-2., 1e4))
      ~covers:
        [
          ("negative", on (fun _ x -> x < 0.));
          ("a subnormal", on is_sub);
          ("a negative subnormal", on (fun w x -> x < 0. && is_sub w x));
        ];
    kind "log2"
      (Op1 (Unary Log2, D.Any D.Float32, 0))
      ?within:(f1 Float.log2 (4., 4.))
      [ log_points ]
      ~range:(fixed (-2., 1e4))
      ~covers:
        [
          ("a power of two", on (fun _ x -> x > 0. && fst (Float.frexp x) = 0.5));
        ];
    kind "log1p"
      (Op1 (Unary Log1p, D.Any D.Float32, 0))
      ?within:(f1 Float.log1p (12., 12.))
      [ log1p_points ]
      ~range:(fixed (-1.5, 10.))
      ~covers:
        [
          ("-1", on (fun _ x -> x = -1.)); ("below -1", on (fun _ x -> x < -1.));
        ];
  ]
  @ List.map
      (fun (name, k, within) ->
        kind name ?within
          (Op1 (Unary k, D.Any D.Float32, 0))
          [ trig_points ]
          ~range:(by32 (-1e5, 1e5) (-1e9, 1e9))
          ~covers:
            [
              ( "the big reduction",
                on (fun w x ->
                    (Float.abs x >= if w then 0x1p28 else 0x1p15)
                    && Float.is_finite x) );
              ( "huge",
                on (fun _ x ->
                    Float.abs x > 1e30 && Float.abs x < Float.infinity) );
              ( "tiny",
                on (fun w x ->
                    x <> 0. && Float.abs x < if w then 0x1p-27 else 0x1p-12) );
            ])
      [
        ("sin", K.Sin, f1 Float.sin (2., 2.));
        ("cos", K.Cos, f1 Float.cos (2., 2.));
        ("tan", K.Tan, f1 Float.tan (19., 19.));
      ]
  @ List.map
      (fun (name, k, within) ->
        kind name ?within
          (Op1 (Unary k, D.Any D.Float32, 0))
          [ asin_points ]
          ~range:(fixed (-1.1, 1.1))
          ~covers:
            [
              ("±1", on (fun _ x -> Float.abs x = 1.));
              ("past 1", on (fun _ x -> Float.abs x > 1.));
              ( "above 1/2",
                on (fun _ x -> Float.abs x > 0.5 && Float.abs x < 1.) );
            ])
      [
        ("asin", K.Asin, f1 Float.asin (5., 9.));
        ("acos", K.Acos, f1 Float.acos (5., 9.));
      ]
  @ [
      kind "atan"
        (Op1 (Unary Atan, D.Any D.Float32, 0))
        ?within:(f1 Float.atan (2., 7.))
        [ atan_points ]
        ~range:(fixed (-10., 10.))
        ~covers:
          [
            ( "above 1",
              on (fun _ x -> Float.abs x > 1. && Float.abs x < Float.infinity)
            );
          ];
    ]
  @ List.map
      (fun (name, k, within) ->
        kind name ?within
          (Op1 (Unary k, D.Any D.Float32, 0))
          [ hyp_points ]
          ~range:(by32 (-100., 100.) (-720., 720.))
          ~covers:
            [
              ("overflow", on (fun w x -> Float.abs x > if w then 711. else 90.));
              ( "past the cap",
                on (fun w x -> Float.abs x > if w then 45. else 23.) );
            ])
      [
        ("sinh", K.Sinh, f1 Float.sinh (21., 21.));
        ("cosh", K.Cosh, f1 Float.cosh (10., 10.));
      ]
  @ [
      kind "tanh"
        (Op1 (Unary Tanh, D.Any D.Float32, 0))
        ?within:(f1 Float.tanh (18., 18.))
        [ tanh_points ]
        ~range:(fixed (-25., 25.))
        ~covers:
          [
            ("small", on (fun _ x -> x <> 0. && Float.abs x < 0.625));
            ("saturated", on (fun w x -> Float.abs x > if w then 19.5 else 9.5));
          ];
      kind "erf"
        (Op1 (Unary Erf, D.Any D.Float32, 0))
        ?within:(f1 Float.erf (2., 3.))
        [ erf_points ]
        ~range:(fixed (-7., 7.))
        ~covers:
          [
            ("below 1", on (fun _ x -> Float.abs x < 1.));
            ( "the far range",
              on (fun _ x -> Float.abs x > 2.5 && Float.abs x < 5.95) );
            ( "rounds to 1",
              on (fun _ x -> Float.abs x > 6. && Float.abs x < Float.infinity)
            );
          ];
    ]

let pow_tuples _ =
  let pairs =
    [
      (1., Float.nan);
      (Float.nan, 0.);
      (-1., Float.infinity);
      (-1., Float.neg_infinity);
      (-2., 0.5);
      (-2., 3.);
      (-2., 2.);
      (-0., -3.);
      (-0., -2.);
      (0., -1.);
      (Float.neg_infinity, 3.);
      (Float.neg_infinity, -3.);
      (-8., 1. /. 3.);
      (0.5, Float.infinity);
      (2., Float.neg_infinity);
      (-3., 0x1p53);
      (-3., 9007199254740991.);
      (-1., 16777215.);
      (2., 1024.);
      (2., -1075.);
      (10., 308.);
      (10., -320.);
      (1.0000001, 1e9);
      (-1.5, 1e9 +. 1.);
    ]
  in
  List.map (fun (x, y) -> [| x; y |]) pairs

let atan2_tuples _ =
  let z = [ 0.; -0.; Float.infinity; Float.neg_infinity; 1.; -1.; 1e-310 ] in
  List.concat_map (fun y -> List.map (fun x -> [| y; x |]) z) z

let mod_tuples wide =
  let e = edges wide in
  List.concat_map
    (fun x ->
      List.map (fun y -> [| x; y |]) (e @ [ 3.; -0.75; 0.1; Float.infinity ]))
    (e @ [ 7.; -7.5; 1e30; 0. ])

(* pow's bound grows with the exponent of its result. *)
let pow_bound _ e =
  let p = Float.abs (e.(1) *. Float.log2 (Float.abs e.(0))) in
  9. +. (6. *. if Float.is_nan p then 0. else p)

let binary =
  let b ?tuples ?covers ?within ?(range = wide_range) name k pts =
    kind ?tuples ?covers ?within ~range name (Op2 (Binary k, 0, 1)) [ pts; pts ]
  in
  [
    b "add" K.Add none;
    b "sub" K.Sub none;
    b "mul" K.Mul none;
    b "fdiv" K.Fdiv none;
    b "maximum" K.Maximum none
      ~covers:
        [
          ( "zeros of both signs",
            fun _ e ->
              e.(0) = 0. && e.(1) = 0. && is_neg_zero e.(0) <> is_neg_zero e.(1)
          );
        ];
    b "minimum" K.Minimum none
      ~covers:
        [
          ( "zeros of both signs",
            fun _ e ->
              e.(0) = 0. && e.(1) = 0. && is_neg_zero e.(0) <> is_neg_zero e.(1)
          );
        ];
    b "mod" K.Mod none ~tuples:mod_tuples
      ~covers:
        [
          ( "an exponent gap past 1000",
            fun _ e ->
              Float.abs e.(0) > 1e300 && Float.abs e.(1) < 1e-300 && e.(1) <> 0.
          );
          ( "an exact multiple",
            fun _ e ->
              e.(1) <> 0.
              && Float.is_finite e.(0)
              && Float.rem e.(0) e.(1) = 0.
              && e.(0) <> 0. );
          ("a subnormal divisor", fun w e -> is_sub w e.(1));
        ];
    b "pow" K.Pow
      (both log_points exp2_points)
      ?within:(Some ((fun e -> Float.pow e.(0) e.(1)), pow_bound))
      ~tuples:pow_tuples
      ~range:(fixed (-20., 20.))
      ~covers:
        [
          ( "a negative base, odd power",
            fun _ e ->
              e.(0) < 0. && Float.is_integer e.(1) && Float.rem e.(1) 2. <> 0.
          );
          ( "a negative base, fraction",
            fun _ e ->
              e.(0) < 0.
              && Float.is_finite e.(0)
              && (not (Float.is_integer e.(1)))
              && Float.is_finite e.(1) );
          ( "-1 to an infinity",
            fun _ e -> e.(0) = -1. && Float.abs e.(1) = Float.infinity );
          ("1 to a NaN", fun _ e -> e.(0) = 1. && Float.is_nan e.(1));
          ("a zero base, negative power", fun _ e -> e.(0) = 0. && e.(1) < 0.);
        ];
    b "atan2" K.Atan2 atan_points ~tuples:atan2_tuples
      ?within:(f2 Float.atan2 (6., 8.))
      ~covers:
        [
          ("two zeros", fun _ e -> e.(0) = 0. && e.(1) = 0.);
          ( "two infinities",
            fun _ e ->
              Float.abs e.(0) = Float.infinity
              && Float.abs e.(1) = Float.infinity );
          ("a negative x", fun _ e -> e.(1) < 0. && Float.is_finite e.(1));
        ];
  ]

let compare =
  List.map
    (fun (name, k) ->
      kind name
        (Op2 (Compare k, 0, 1))
        [ none; none ]
        ~range:(fixed (-2., 2.))
        ~tuples:(fun _ -> [ [| 0.; -0. |]; [| -0.; 0. |]; [| 1.; 1. |] ])
        ~covers:[ ("equal operands", fun _ e -> e.(0) = e.(1)) ])
    [
      ("equal", K.Equal);
      ("not_equal", K.Not_equal);
      ("less", K.Less);
      ("less_equal", K.Less_equal);
    ]

let fma =
  let cancel w =
    if w then [ 1. +. 0x1p-30; -.(1. +. 0x1p-29) ]
    else [ 1. +. 0x1p-12; -.(1. +. 0x1p-11) ]
  in
  kind "fma"
    (Op3 (Fma, 0, 1, 2))
    [ cancel; cancel; cancel ]
    ~range:(fixed (-10., 10.))
    ~covers:
      [
        ( "a NaN third",
          fun _ e ->
            Float.is_nan e.(2) && not (Float.is_nan e.(0) || Float.is_nan e.(1))
        );
      ]

(* Cases *)

(* A case: the dtype, and per element its operands. *)
let case k =
  let open Gen in
  let* wide = bool in
  let free = List.map (fun p -> value wide (p wide) (k.range wide)) k.points in
  let tuple =
    List.fold_right
      (fun g acc ->
        let+ v = g and+ vs = acc in
        v :: vs)
      free (constant [])
  in
  let tuple = map Array.of_list tuple in
  let tuples = List.map (Array.map (round wide)) (k.tuples wide) in
  let element =
    if tuples = [] then tuple else frequency [ (4, tuple); (1, of_list tuples) ]
  in
  map (fun es -> (wide, es)) (array ~size:(constant n) element)

let hex (type s) (dt : (float, s) D.t) v =
  let b = K.bits dt v in
  String.concat ""
    (List.rev
       (List.init (String.length b) (fun i ->
            Printf.sprintf "%02x" (Char.code b.[i]))))

let operands (type s) (dt : (float, s) D.t) es i =
  Nx.create dt [| n |] (Array.map (fun e -> e.(i)) es)

(* Ulps: the distance in ordered-bit ranks, -0 and +0 one apart, of two floats
   of the dtype [wide]. *)
let ulps wide a b =
  let mag v =
    if wide then Int64.logand (Int64.bits_of_float v) Int64.max_int
    else Int64.of_int32 (Int32.logand (Int32.bits_of_float v) Int32.max_int)
  in
  let ma = mag a and mb = mag b in
  if Float.sign_bit a = Float.sign_bit b then
    Int64.to_float (Int64.abs (Int64.sub ma mb))
  else Int64.to_float (Int64.add (Int64.add ma mb) 1L)

let special v = v = 0. || Float.is_nan v || Float.abs v = Float.infinity

(* Where a kind's value is exact: at operands each a zero, an infinity or a NaN,
   a reference of zero, infinity, ±1 or NaN; and pow's F.9.4.4 values. *)
let exact name e r =
  (Array.for_all special e && (special r || Float.abs r = 1.))
  || name = "pow"
     && (e.(0) = 1.
        || e.(1) = 0.
        || (e.(0) = -1. && Float.abs e.(1) = Float.infinity))

(* Whether [got] meets the reference [r] for operands [e] within [tol] ulps: a
   NaN where [r] is NaN, and nowhere else. *)
let meets wide name tol e r got =
  if Float.is_nan r || Float.is_nan got then Float.is_nan r = Float.is_nan got
  else if exact name e r then ulps wide r got = 0.
  else ulps wide r got <= tol

let check k (wide, es) =
  let run (type s) (dt : (float, s) D.t) =
    let xs = Array.init (arity k) (operands dt es) in
    let ins = Iarray.of_array (Array.make (arity k) (D.Any dt)) in
    let node : K.node =
      match k.node with Op1 (u, _, j) -> Op1 (u, D.Any dt, j) | other -> other
    in
    let prog = K.of_node ~ins node in
    let layout = Nx_array.Layout.contiguous [| n |] in
    let loads = Iarray.of_array (Array.map (fun x -> P.Plain x) xs) in
    let args e = String.concat ", " (Array.to_list (Array.map (hex dt) e)) in
    let show i y = Printf.sprintf "%s(%s) = %s" k.name (args es.(i)) y in
    let map out =
      fst
        (P.eval ~by:"t"
           (P.Map { layout; prog; outs = [ out ]; loads; at = None }))
    in
    match k.node with
    | Op2 (Compare _, _, _) ->
        let op () = map D.Bool in
        let eager = Nx.to_array (op ()) in
        let lowered = Lowered.run op in
        let show r = Array.mapi (fun i b -> show i (string_of_bool b)) r in
        equal (array string) (show eager) (show lowered)
    | _ -> (
        let op () = map dt in
        let lowered = Lowered.run op in
        let bits r =
          Array.mapi
            (fun i y -> show i (if Float.is_nan y then "nan" else hex dt y))
            r
        in
        match k.within with
        | None ->
            equal (array string) (bits (Nx.to_array (op ()))) (bits lowered)
        | Some (f, bound) ->
            (* float64 against eager, nx's bound of 2 ulps added *)
            let refs, slack =
              if wide then (Nx.to_array (op ()), 2.)
              else (Array.map (fun e -> f32 (f e)) es, 0.)
            in
            let bad = ref [] in
            Array.iteri
              (fun i r ->
                let tol = bound wide es.(i) +. slack in
                let got = lowered.(i) in
                if not (meets wide k.name tol es.(i) r got) then
                  bad :=
                    Printf.sprintf "%s, reference %s, %g ulps over %g"
                      (show i (hex dt got))
                      (hex dt r) (ulps wide r got) tol
                    :: !bad)
              refs;
            equal (list string) [] (List.rev !bad))
  in
  let cover_some label f = cover label (Array.exists (f wide) es) in
  cover "float32" (not wide);
  cover "float64" wide;
  cover_some "a NaN" (fun _ e -> Array.exists Float.is_nan e);
  cover_some "a negative NaN" (fun _ e -> Array.exists is_nan_neg e);
  cover_some "-0" (fun _ e -> Array.exists is_neg_zero e);
  cover_some "an infinity" (fun _ e ->
      Array.exists (fun v -> Float.abs v = Float.infinity) e);
  cover_some "a subnormal" (fun w e -> Array.exists (is_sub w) e);
  List.iter (fun (label, f) -> cover_some label f) k.covers;
  if wide then run D.Float64 else run D.Float32

let prop_of k = prop ~count:20 k.name (case k) (check k)

(* float64 operands and their correctly rounded results, from mpmath at 200
   bits: the drawn float64 cases meet eager's results within the sum of both
   bounds, which cannot see a bound of one. *)
let pinned name k bound pairs =
  test name (fun () ->
      let xs =
        Array.of_list (List.map (fun (x, _) -> Float.of_string x) pairs)
      in
      let n = Array.length xs in
      let x = Nx.create D.Float64 [| n |] xs in
      let ins = Iarray.of_array [| D.Any D.Float64 |] in
      let prog = K.of_node ~ins (Op1 (Unary k, D.Any D.Float64, 0)) in
      let layout = Nx_array.Layout.contiguous [| n |] in
      let loads = Iarray.of_array [| P.Plain x |] in
      let op () =
        fst
          (P.eval ~by:"t"
             (P.Map { layout; prog; outs = [ D.Float64 ]; loads; at = None }))
      in
      let got = Lowered.run op in
      let far =
        List.concat
          (List.mapi
             (fun i (x, r) ->
               let d = ulps true (Float.of_string r) got.(i) in
               if d <= bound then []
               else [ Printf.sprintf "%s: %h, %g ulps from %s" x got.(i) d r ])
             pairs)
      in
      equal (list string) [] far)

(* Where tolk's log2 times ln 2 is two ulps off. *)
let log_near_one =
  pinned "log within one ulp near 1" K.Log 1.
    [
      ("0x1.481e0a20c684p+0", "0x1.fc4d0ea863fa9p-3");
      ("0x1.203eace7848f4p+0", "0x1.e5eb7824a1d0cp-4");
      ("0x1.936dc3c250523p-1", "-0x1.e816c6150210cp-3");
      ("0x1.67372bc172382p-1", "-0x1.6ae862c1635bdp-2");
      ("0x1.410308102a104p+0", "0x1.cf76ac1e049dfp-3");
    ]

let () =
  exit
    (run "Lower_real"
       [
         group "unary" (List.map prop_of unary @ [ log_near_one ]);
         group "binary" (List.map prop_of binary);
         group "compare" (List.map prop_of compare);
         group "fma" [ prop_of fma ];
       ])
