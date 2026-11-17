(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The float kinds at float32 and float64, as maps of one kind compiled by
   Rune.jit, which must compile them whole: no call is evaluated. An exact kind
   gives the bits of nx's eager kernel, any NaN for a NaN, over operands drawn
   to reach its special values and the edges of its range. So does each
   transcendental kind, compiled to nx's own sequence (Kinds), at every narrow
   float's code and at a constant the compiler may fold too. *)

open Windtrap
module Kinds = Rune_support.Kinds
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

(* Kinds of nx's sequence

   Each gives eager's bits over drawn operands (Kinds), at every narrow
   float's code and of a constant inside the compiled function. *)
let ported k =
  let name = Kinds.name k in
  [
    prop ~count:12
      ~examples:[ Kinds.example k ]
      (name ^ " gives eager's bits")
      (Kinds.gen k)
      (Kinds.law (module Nx.Host) ~f64:true k);
    test (name ^ " of every narrow float gives eager's bits") (fun () ->
        Kinds.narrow (module Nx.Host) k);
    test (name ^ " of a constant gives eager's bits") (fun () ->
        Kinds.folded (module Nx.Host) ~f64:true k);
  ]

let kinds =
  Kinds.
    [
      sin; cos; tan; log; log2; log1p; exp; exp2; expm1; sinh; cosh; tanh; asin;
      acos; atan; atan2; erf; pow; mod_;
    ]

(* exp and log of an operand the program clamps by selections on float
   comparisons, whose bounds tolk's range analysis reads: the compiled kind
   still gives eager's bits. *)
let clamped = Kinds.[ clamped_exp; clamped_log ]

let clamped_law k =
  prop ~count:12
    ~examples:[ Kinds.example k ]
    (Kinds.name k ^ " gives eager's bits")
    (Kinds.gen k)
    (Kinds.law (module Nx.Host) ~f64:true k)

(* Near forms

   Lower_real.sin_near and cos_near, nx_sin_near and nx_cos_near, give the
   full sine's and cosine's bits wherever |x| is below the switch to Payne and
   Hanek's reduction, 2^15 or 2^28, or is not finite, at float32 and float64.
   Each dtype's block draws a uniform
   code below the switch of either sign (3 in 6), a multiple of pi/2 below it
   stepped up to 2 ranks (2 in 6), or one of 7 special codes (1 in 6): the
   zeros, the infinities, a NaN, the least subnormal and the float just below
   the switch. With at least 7/8 of draws fresh, each special is drawn at
   1/48 a draw or more, and a block of 4096 misses one with probability
   below 1e-37. *)

let near_specials wide =
  let below = if wide then 0x41AFFFFFFFFFFFFFL else 0x46FFFFFFL in
  let sign = if wide then Int64.min_int else 0x80000000L in
  let inf = if wide then 0x7FF0000000000000L else 0x7F800000L in
  let nan = if wide then 0x7FF8000000000000L else 0x7FC00000L in
  [ 0L; sign; inf; Int64.logor inf sign; nan; 1L; below ]

let near_switch wide = if wide then 0x1p28 else 0x1p15

let near_block wide =
  let open Gen in
  let top = if wide then 0x41B0000000000000L else 0x47000000L in
  let sign = if wide then Int64.min_int else 0x80000000L in
  let code v =
    if wide then Int64.bits_of_float v
    else Int64.logand (Int64.of_int32 (Int32.bits_of_float v)) 0xFFFFFFFFL
  in
  let signed g =
    let+ c = g and+ neg = bool in
    if neg then Int64.logor c sign else c
  in
  let last = int_of_float (near_switch wide /. Float.pi *. 2.) - 1 in
  let multiple =
    let+ k = int_range 1 last and+ d = int_range (-2) 2 in
    Int64.add (code (Float.of_int k *. Float.pi /. 2.)) (Int64.of_int d)
  in
  array ~size:(constant 4096)
    (frequency
       [
         (3, signed (int64_range 0L (Int64.pred top)));
         (2, signed multiple);
         (1, of_list (near_specials wide));
       ])

let near_value wide c =
  if wide then Int64.float_of_bits c else Int32.float_of_bits (Int64.to_int32 c)

(* [f] of the codes [cs] compiled on the host, as codes. *)
let near_run wide f cs =
  let n = Array.length cs in
  if wide then
    Array.map Int64.bits_of_float
      (Lowered.map f (Nx.bitcast Nx.float64 (Nx.create Nx.int64 [| n |] cs)))
  else
    let x =
      Nx.bitcast Nx.float32
        (Nx.create Nx.int32 [| n |] (Array.map Int64.to_int32 cs))
    in
    let code v = Int64.of_int32 (Int32.bits_of_float v) in
    Array.map (fun v -> Int64.logand (code v) 0xFFFFFFFFL) (Lowered.map f x)

let near_misses (name, near, full) wide cs =
  let near = near_run wide near cs in
  let full = near_run wide (Rune.Lower_real.unary full) cs in
  let nan c = Float.is_nan (near_value wide c) in
  let out = ref [] in
  Array.iteri
    (fun i c ->
      let a = near.(i) and b = full.(i) in
      if (not (Int64.equal a b || (nan a && nan b))) && List.length !out < 8
      then
        out :=
          Printf.sprintf "%s_near(%h) = %#Lx, %s = %#Lx" name
            (near_value wide c) a name b
          :: !out)
    cs;
  List.rev !out

let near_covers wide cs =
  let any p = Array.exists (fun c -> p (near_value wide c)) cs in
  let dt = if wide then "float64: " else "float32: " in
  let switch = near_switch wide in
  let q x = Float.round (x /. (Float.pi /. 2.)) in
  let off x = Float.abs (x -. (q x *. Float.pi /. 2.)) in
  cover (dt ^ "beside a multiple of pi/2")
    (any (fun x -> q x <> 0. && off x <= 0x1p-20 *. Float.abs x));
  cover
    (dt ^ "within 2^-12 of the switch, below it")
    (any (fun x ->
         Float.abs x < switch && Float.abs x >= switch *. (1. -. 0x1p-12)));
  cover (dt ^ "a subnormal")
    (any (fun x ->
         x <> 0. && Float.abs x < if wide then Float.min_float else 0x1p-126));
  cover (dt ^ "a NaN") (any Float.is_nan);
  cover (dt ^ "an infinity") (any (fun x -> Float.abs x = Float.infinity))

let near_law ((name, _, _) as k) =
  prop ~count:12
    (name ^ "_near gives " ^ name ^ "'s bits below the switch")
    (Gen.with_pp
       (fun ppf _ ->
         Format.pp_print_string ppf "4096 float32 and float64 codes")
       Gen.(pair (near_block false) (near_block true)))
    (fun (c32, c64) ->
      near_covers false c32;
      near_covers true c64;
      equal (list string) [] (near_misses k false c32 @ near_misses k true c64))

let nears =
  [
    ("sin", Rune.Lower_real.sin_near, K.Sin);
    ("cos", Rune.Lower_real.cos_near, K.Cos);
  ]

let () =
  exit
    (run "Lower_real"
       [
         group "exact" (List.map law (unary @ binary @ compare @ [ fma ]));
         group "kinds" (List.concat_map ported kinds);
         group "clamped operands" (List.map clamped_law clamped);
         group "near forms" (List.map near_law nears);
       ])
