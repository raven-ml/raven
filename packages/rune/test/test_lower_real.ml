(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The float kinds at float32 and float64, as maps of one kind compiled by
   Rune.jit, which must compile them whole: no call is evaluated. An exact kind
   gives the bits of nx's eager kernel, any NaN for a NaN, over operands drawn
   to reach its special values and the edges of its range. A transcendental kind
   is within its host bound in ulps of the correctly rounded result (rune.mli,
   Numerics) at the points of golden/lower_real, which gen/lower_real.py writes
   from mpmath; where every operand is a zero, an infinity or a NaN, it gives
   Annex F's zero, infinity, ±1 or NaN, stated below. *)

open Windtrap
module D = Nx_array.Dtype
module K = Nx_kernel.Prog
module P = Nx.Prim
module T = Nx.Ptree

let n = 256

(* Compiling *)

(* [counted f] is [f ()], whose compiled calls must compile whole: a declined
   lowering, evaluated by nx, fails. *)
let counted f =
  let before = Rune.evaluations () in
  let y = f () in
  equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - before);
  y

(* The program of [node] over operands of [dt]. *)
let program (type s) (dt : (float, s) D.t) arity (node : K.node) =
  let ins = Iarray.of_array (Array.make arity (D.Any dt)) in
  let node : K.node =
    match node with Op1 (u, _, j) -> Op1 (u, D.Any dt, j) | other -> other
  in
  K.of_node ~ins node

(* [staged out prog] maps [prog] over operands of one shape into [out]: called
   on operands, it gives nx's eager result and the result of the map staged in
   Rune.jit. One staged map compiles once per shape. *)
let staged out prog =
  let map xs =
    let layout =
      Nx_array.Layout.contiguous (Iarray.of_array (Nx.shape (List.hd xs)))
    in
    let loads = Iarray.of_list (List.map (fun x -> P.Plain x) xs) in
    fst
      (P.eval ~by:"t"
         (P.Map { layout; prog; outs = [ out ]; loads; at = None }))
  in
  let compiled = Rune.jit T.(list tensor @-> returns tensor) map in
  fun xs -> (map xs, counted (fun () -> compiled xs))

let show v = if Float.is_nan v then "nan" else Printf.sprintf "%h" v

(* Ulps: the distance in ordered-bit ranks, -0 and +0 one apart, of two floats
   of float64 if [wide], else of float32. *)
let ulps wide a b =
  let mag v =
    if wide then Int64.logand (Int64.bits_of_float v) Int64.max_int
    else Int64.of_int32 (Int32.logand (Int32.bits_of_float v) Int32.max_int)
  in
  let ma = mag a and mb = mag b in
  if Float.sign_bit a = Float.sign_bit b then
    Int64.to_float (Int64.abs (Int64.sub ma mb))
  else Int64.to_float (Int64.add (Int64.add ma mb) 1L)

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
        0x7FF0000000000001L;
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

(* Two NaNs, the zeros, the infinities and the least subnormal first, which
   every example pairs. *)
let specials wide =
  let signed = List.concat_map (fun v -> [ v; -.v ]) in
  let nans = nans wide in
  List.filteri (fun i _ -> i < 2) nans
  @ signed ([ 0.; Float.infinity ] @ edges wide @ [ 1.; 0.5; 2. ])
  @ List.filteri (fun i _ -> i >= 2) nans

let is_nan_neg v = Float.is_nan v && Float.sign_bit v
let is_neg_zero v = v = 0. && Float.sign_bit v

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

(* A value of float64 if [wide], else of float32: a special value, one of
   [points] or a neighbour, any bits, or a value of [range]. *)
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

(* Exact kinds *)

(* A kind's operands: per operand, the points its values come from; the range of
   its uniform values; its special tuples; and the cases it must reach, on one
   element of float64 if the first argument holds, else float32. *)
type kind = {
  name : string;
  node : K.node;
  points : (bool -> float list) list;
  range : float * float;
  tuples : bool -> float array list;
  covers : (string * (bool -> float array -> bool)) list;
}

let arity k = List.length k.points

let kind ?(tuples = fun _ -> []) ?(covers = []) ?(range = (-1e3, 1e3)) name node
    points =
  { name; node; points; range; tuples; covers }

let on f wide (e : float array) = f wide e.(0)
let none _ = []

let rounding _ =
  [
    0.5; 1.5; 2.5; 0x1p22 +. 0.5; 0x1p23; 0x1p52; 4503599627370495.5; 0.49999997;
  ]

let roots _ = [ 2.; 0.5; 3.; 0x1p-126; 0x1p-1022; 1e-40; 1e-310 ]

let unary =
  let u ?covers name k pts =
    kind ?covers name (Op1 (Unary k, D.Any D.Float32, 0)) [ pts ]
  in
  [
    u "neg" K.Neg none;
    u "abs" K.Abs none;
    u "sign" K.Sign none;
    u "recip" K.Recip none;
    u "sqrt" K.Sqrt roots;
    u "floor" K.Floor rounding;
    u "ceil" K.Ceil rounding;
    u "round" K.Round rounding
      ~covers:
        [ ("a half", on (fun _ x -> Float.abs (x -. Float.trunc x) = 0.5)) ];
    u "trunc" K.Trunc rounding;
  ]

let zeros_apart _ e =
  e.(0) = 0. && e.(1) = 0. && is_neg_zero e.(0) <> is_neg_zero e.(1)

let mod_tuples wide =
  let e = edges wide in
  List.concat_map
    (fun x ->
      List.map (fun y -> [| x; y |]) (e @ [ 3.; -0.75; 0.1; Float.infinity ]))
    (e @ [ 7.; -7.5; 1e30; 0. ])

let binary =
  let b ?tuples ?covers name k =
    kind ?tuples ?covers name (Op2 (Binary k, 0, 1)) [ none; none ]
  in
  [
    b "add" K.Add;
    b "sub" K.Sub;
    b "mul" K.Mul;
    b "fdiv" K.Fdiv;
    b "maximum" K.Maximum ~covers:[ ("zeros of both signs", zeros_apart) ];
    b "minimum" K.Minimum ~covers:[ ("zeros of both signs", zeros_apart) ];
    b "mod" K.Mod ~tuples:mod_tuples
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
  ]

let compare =
  List.map
    (fun (name, k) ->
      kind name
        (Op2 (Compare k, 0, 1))
        [ none; none ] ~range:(-2., 2.)
        ~tuples:(fun _ -> [ [| 0.; -0. |]; [| -0.; 0. |]; [| 1.; 1. |] ])
        ~covers:[ ("equal operands", fun _ e -> e.(0) = e.(1)) ])
    [
      ("equal", K.Equal);
      ("not_equal", K.Not_equal);
      ("less", K.Less);
      ("less_equal", K.Less_equal);
    ]

(* Products that cancel to a few ulps of either dtype, a subnormal product and a
   NaN third operand. *)
let fma =
  let cancel w =
    if w then [ 1. +. 0x1p-30; -.(1. +. 0x1p-29) ]
    else [ 1. +. 0x1p-12; -.(1. +. 0x1p-11) ]
  in
  let tuples w =
    let a = List.hd (cancel w) in
    let tiny = List.hd (edges w) in
    [
      [| a; a; -.(a *. a) |];
      [| a; -.a; round w (a *. a) |];
      [| tiny; 0.5; 0. |];
      [| 1.; 1.; Float.nan |];
      [| 1.; 1.; -0. |];
    ]
  in
  kind "fma"
    (Op3 (Fma, 0, 1, 2))
    [ cancel; cancel; cancel ] ~range:(-10., 10.) ~tuples
    ~covers:
      [
        ( "a NaN third",
          fun _ e ->
            Float.is_nan e.(2) && not (Float.is_nan e.(0) || Float.is_nan e.(1))
        );
      ]

(* A case: whether it is float64, and per element its operands. *)
let case k =
  let open Gen in
  let* wide = bool in
  let free = List.map (fun p -> value wide (p wide) k.range) k.points in
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

(* A case that reaches every cover: the kind's tuples, then every tuple of the
   first [b] specials and points, [b] the most whose tuples fit in the rest of
   [n]. *)
let example k wide =
  let tuples = List.map (Array.map (round wide)) (k.tuples wide) in
  let pool =
    Array.of_list
      (specials wide
      @ List.concat_map (fun p -> List.map (round wide) (p wide)) k.points)
  in
  let a = arity k in
  let room = n - List.length tuples in
  let rec base b =
    if b > 1 && Float.of_int b ** Float.of_int a > Float.of_int room then
      base (b - 1)
    else b
  in
  let b = min (Array.length pool) (base n) in
  let t = Array.of_list tuples in
  let element i =
    if i < Array.length t then t.(i)
    else
      let i = i - Array.length t in
      Array.init a (fun j ->
          pool.(i / int_of_float (Float.of_int b ** Float.of_int j) mod b))
  in
  (wide, Array.init n element)

let operands (type s) (dt : (float, s) D.t) es =
  List.init
    (Array.length es.(0))
    (fun j ->
      Nx.create dt [| Array.length es |] (Array.map (fun e -> e.(j)) es))

(* [same dt k es] compiles [k] over the operands [es] of [dt] and compares its
   elements with eager's, shown as [k(operands) = result]. *)
let same (type s) (dt : (float, s) D.t) k =
  let prog = program dt (arity k) k.node in
  let at es i y =
    let args = String.concat ", " (Array.to_list (Array.map show es.(i))) in
    Printf.sprintf "%s(%s) = %s" k.name args y
  in
  let both run show es =
    let eager, compiled = run (operands dt es) in
    let shown y = Array.mapi (fun i v -> at es i (show v)) (Nx.to_array y) in
    equal (array string) (shown eager) (shown compiled)
  in
  match k.node with
  | Op2 (Compare _, _, _) -> both (staged D.Bool prog) string_of_bool
  | _ -> both (staged dt prog) show

let law k =
  let single = same D.Float32 k and double = same D.Float64 k in
  prop ~count:20 k.name (case k)
    ~examples:[ example k false; example k true ]
    (fun (wide, es) ->
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
      if wide then double es else single es)

(* Transcendental kinds *)

(* A value rounded to float32 and to float64. *)
let both x = (x, x)
let pi = (0x1.921fb6p+1, 0x1.921fb54442d18p+1)
let half_pi = (0x1.921fb6p+0, 0x1.921fb54442d18p+0)
let quarter_pi = (0x1.921fb6p-1, 0x1.921fb54442d18p-1)
let three_quarter_pi = (0x1.2d97c8p+1, 0x1.2d97c7f3321d2p+1)
let neg (a, b) = (-.a, -.b)
let inf = Float.infinity
let nan = Float.nan

(* Annex F's values (F.9) where an operand is a zero, an infinity or a NaN, as
   each kind's rule: at operands of float64 if [wide], else float32, the value,
   or [None] where Annex F gives none and the goldens hold the kind. A value
   where every operand is one of these and the value is a zero, an infinity, ±1
   or a NaN is exact; another, such as acos 0, is held to the kind's bound. *)
let pick wide (a, b) = if wide then b else a
let special v = v = 0. || Float.is_nan v || Float.abs v = Float.infinity

let annex1 name wide e =
  let x = e.(0) in
  let at (z, nz, i, ni) =
    if Float.is_nan x then Some nan
    else if x = 0. then Some (pick wide (if Float.sign_bit x then nz else z))
    else if x = inf then Some (pick wide i)
    else if x = -.inf then Some (pick wide ni)
    else None
  in
  let z = both 0. and nz = both (-0.) and i = both inf and ni = both (-.inf) in
  let one = both 1. and none = both (-1.) and nn = both nan in
  at
    (match name with
    | "exp" | "exp2" -> (one, one, i, z)
    | "expm1" -> (z, nz, i, none)
    | "log" | "log2" -> (ni, ni, i, nn)
    | "log1p" -> (z, nz, i, nn)
    | "sin" | "tan" | "asin" -> (z, nz, nn, nn)
    | "cos" -> (one, one, nn, nn)
    | "acos" -> (half_pi, half_pi, nn, nn)
    | "atan" -> (z, nz, half_pi, neg half_pi)
    | "sinh" -> (z, nz, i, ni)
    | "cosh" -> (one, one, i, i)
    | "tanh" | "erf" -> (z, nz, one, none)
    | _ -> invalid_arg name)

let odd y = Float.is_integer y && Float.rem y 2. <> 0.

(* pow x y, F.9.4.4: 1 at y = ±0 and at x = 1, NaN included, then the zeros' and
   infinities' rules. *)
let annex_pow _ e =
  let x = e.(0) and y = e.(1) in
  if y = 0. || x = 1. then Some 1.
  else if Float.is_nan x || Float.is_nan y then Some nan
  else if Float.abs y = inf then
    let a = Float.abs x in
    if a = 1. then Some 1. else Some (if a < 1. = (y > 0.) then 0. else inf)
  else if x = 0. then
    if y < 0. then Some (if odd y then Float.copy_sign inf x else inf)
    else Some (if odd y then x else 0.)
  else if Float.abs x = inf then
    if y > 0. then Some (if odd y then x else inf)
    else Some (if odd y then Float.copy_sign 0. x else 0.)
  else None

(* atan2 y x, F.9.1.4. *)
let annex_atan2 wide e =
  let y = e.(0) and x = e.(1) in
  let signed v = Some (pick wide (if Float.sign_bit y then neg v else v)) in
  if Float.is_nan x || Float.is_nan y then Some nan
  else if y = 0. then if Float.sign_bit x then signed pi else Some y
  else if Float.abs y = inf then
    if x = inf then signed quarter_pi
    else if x = -.inf then signed three_quarter_pi
    else signed half_pi
  else if x = 0. then signed half_pi
  else if x = inf then Some (Float.copy_sign 0. y)
  else if x = -.inf then signed pi
  else None

(* The rows of golden/lower_real/<name>.golden at float64 if [wide], else at
   float32: operands and the correctly rounded value. *)
let golden name wide =
  let ic = open_in ("golden/lower_real/" ^ name ^ ".golden") in
  let rows = ref [] in
  (try
     while true do
       match String.split_on_char '\t' (input_line ic) with
       | d :: cols when d = if wide then "f64" else "f32" ->
           let vs = Array.of_list (List.map Float.of_string cols) in
           let a = Array.length vs - 1 in
           rows := (Array.sub vs 0 a, vs.(a)) :: !rows
       | _ -> ()
     done
   with End_of_file -> close_in ic);
  greater ~msg:(name ^ "'s rows") int ~than:0 (List.length !rows);
  List.rev !rows

(* Where a kind's value is exact: Annex F's zero, infinity, ±1 or NaN at
   operands each a zero, an infinity or a NaN; and pow's 1 at [y = ±0], [x = 1]
   and [x = -1], [y] infinite. *)
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

(* [accurate name node bound annex] holds [node] within [bound] ulps, at float32
   and float64, at its golden points and where [annex] gives a value: at the
   dtype's specials, every pair of them for a binary kind, and pow's golden
   bases at y = ±0. *)
let accurate name node (b32, b64) annex =
  let at (type s) (dt : (float, s) D.t) wide tol =
    let table = golden name wide in
    let sp = specials wide in
    let tuples =
      match Array.length (fst (List.hd table)) with
      | 1 -> List.map (fun x -> [| x |]) sp
      | _ ->
          List.concat_map (fun a -> List.map (fun b -> [| a; b |]) sp) sp
          @ List.concat_map
              (fun (e, _) -> [ [| e.(0); 0. |]; [| e.(0); -0. |] ])
              (if name = "pow" then table else [])
    in
    let stated =
      List.filter_map
        (fun e -> Option.map (fun r -> (e, r)) (annex wide e))
        tuples
    in
    let rows = table @ stated in
    let es = Array.of_list (List.map fst rows) in
    let run = staged dt (program dt (Array.length es.(0)) node) in
    let got = Nx.to_array (snd (run (operands dt es))) in
    List.concat
      (List.mapi
         (fun i (e, r) ->
           if meets wide name tol e r got.(i) then []
           else
             let args = String.concat ", " (Array.to_list (Array.map show e)) in
             [
               Printf.sprintf "%s(%s) = %s at %s, reference %s, %g ulps over %g"
                 name args
                 (show got.(i))
                 (if wide then "float64" else "float32")
                 (show r)
                 (ulps wide r got.(i))
                 tol;
             ])
         rows)
  in
  test
    (Printf.sprintf "%s within %d ulps at float32, %d at float64" name b32 b64)
    (fun () ->
      equal (list string) []
        (at D.Float32 false (Float.of_int b32)
        @ at D.Float64 true (Float.of_int b64)))

(* Each kind's host bound at float32 and float64 (rune.mli, Numerics). *)
let transcendental =
  List.map
    (fun (name, k, bound) ->
      accurate name (Op1 (Unary k, D.Any D.Float32, 0)) bound (annex1 name))
    [
      ("exp", K.Exp, (2, 9));
      ("exp2", K.Exp2, (4, 4));
      ("expm1", K.Expm1, (23, 23));
      ("log", K.Log, (1, 1));
      ("log2", K.Log2, (4, 4));
      ("log1p", K.Log1p, (5, 5));
      ("sin", K.Sin, (2, 4));
      ("cos", K.Cos, (2, 9));
      ("tan", K.Tan, (19, 19));
      ("asin", K.Asin, (5, 9));
      ("acos", K.Acos, (5, 9));
      ("atan", K.Atan, (2, 7));
      ("sinh", K.Sinh, (23, 23));
      ("cosh", K.Cosh, (10, 10));
      ("tanh", K.Tanh, (18, 18));
      ("erf", K.Erf, (2, 3));
    ]
  @ [
      accurate "pow" (Op2 (Binary Pow, 0, 1)) (2, 2) annex_pow;
      accurate "atan2" (Op2 (Binary Atan2, 0, 1)) (6, 8) annex_atan2;
    ]

(* Bounded arguments

   An argument whose bounds lie below the switch-over to Payne and Hanek is
   reduced by Cody and Waite's alone. The program bounds its operand within
   [-29.9, 29.9], selecting 0 outside, and cos and tan hold their bounds at
   their golden points inside. *)
let bounded (name, kind, (b32, b64)) =
  let at (type s) (dt : (float, s) D.t) wide tol =
    let limit = 29.9 in
    let rows =
      List.filter (fun (e, _) -> Float.abs e.(0) < limit) (golden name wide)
    in
    let es = Array.of_list (List.map fst rows) in
    let prog =
      let open K.E in
      let x = input 0 dt and c = const dt limit and zero = const dt 0. in
      let inside = where (x < c) (where (neg c < x) x zero) zero in
      K.of_exprs ~ins:[| D.Any dt |] [| any (op1 (Unary kind) dt inside) |]
    in
    let got = Nx.to_array (snd (staged dt prog (operands dt es))) in
    List.concat
      (List.mapi
         (fun i (e, r) ->
           if meets wide name tol e r got.(i) then []
           else
             [
               Printf.sprintf "%s(%s) = %s at %s, reference %s" name
                 (show e.(0))
                 (show got.(i))
                 (D.name dt) (show r);
             ])
         rows)
  in
  test (Printf.sprintf "%s of an argument bounded below the switch-over" name)
    (fun () ->
      equal (list string) []
        (at D.Float32 false (Float.of_int b32)
        @ at D.Float64 true (Float.of_int b64)))

let () =
  exit
    (run "Lower_real"
       [
         group "exact" (List.map law (unary @ binary @ compare @ [ fma ]));
         group "transcendental" transcendental;
         group "bounded arguments"
           (List.map bounded
              [ ("cos", K.Cos, (2, 9)); ("tan", K.Tan, (19, 19)) ]);
       ])
