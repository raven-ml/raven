(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Each row's edges: the points where a derivative is one-sided, infinite,
   undefined or chosen by convention. Expected values are the rule's formula
   evaluated in IEEE arithmetic, or the convention named in the test. *)

open Windtrap
module Op = Nx.Op

let vec a = Nx.create Nx.float64 [| Array.length a |] a
let unary k x = Op.eval (Unary (k, x))
let binary k a b = Op.eval (Binary (k, a, b))
let tangent f x v = Nx.to_array (snd (Rune.jvp' f x v))
let gradient f x = Nx.to_array (Rune.grad' (fun x -> Nx.sum (f x)) x)
let floats = array float_exact

(* Zeros of either sign: a claim of no effect says nothing of a zero's sign. *)
let zeros n = (array (float 1e-300), Array.make n 0.)

(* [at f x v expected] checks the tangent of [f] at the scalar [x] along [v]. *)
let at ?(v = 1.) f x expected =
  equal floats [| expected |] (tangent f (vec [| x |]) (vec [| v |]))

(* The conventions at ties, one line each, so that changing one is changing its
   line: the shares of the first and second operand of [maximum] and [minimum]
   at a tie, the share of each of [n] tied elements of a reduced extremum, and
   whether a running extremum takes the first of equal elements. *)
let binary_tie = (0., 1.)
let reduce_tie n = 1. /. float_of_int n
let running_takes_first = true

(* Unary *)

(* Near 1, [1 - x²] cancels: its references round once, from exact terms. In
   float64, [x² = p + e] exactly and [1 - p] is exact by Sterbenz; in float32,
   [(1 - x) (1 + x)] is exact in float64 and rounds once to float32. *)
let one_minus_square x =
  let p = x *. x in
  1. -. p -. Float.fma x x (-.p)

let to_float32 r = Int32.float_of_bits (Int32.bits_of_float r)

let ulps64 a b =
  Int64.to_int (Int64.abs (Int64.sub (Int64.bits_of_float a) (Int64.bits_of_float b)))

let ulps32 a b =
  abs (Int32.to_int (Int32.bits_of_float a) - Int32.to_int (Int32.bits_of_float b))

(* An elementwise function at every float dtype. *)
type elementwise = { f : 'b. (float, 'b) Nx.t -> (float, 'b) Nx.t }

(* [within_ulps n e ~reference64 ~reference32 xs] checks the gradient of [e.f]
   at [xs], in float64 and in float32, against the reference of each dtype to
   [n] ulps. *)
let within_ulps n e ~reference64 ~reference32 xs =
  let grad (type b) (dt : (float, b) Nx.dtype) xs =
    Nx.to_array
      (Rune.grad'
         (fun x -> Nx.sum (e.f x))
         (Nx.create dt [| Array.length xs |] xs))
  in
  let check name ulps reference xs got =
    Array.iteri
      (fun i x ->
        let expected = reference x in
        at_most int
          ~msg:(Printf.sprintf "%s at %g: %h against %h" name x got.(i) expected)
          ~than:n (ulps got.(i) expected))
      xs
  in
  check "float64" ulps64 reference64 xs (grad Nx.float64 xs);
  let xs32 = Array.map to_float32 xs in
  check "float32" ulps32 reference32 xs32 (grad Nx.float32 xs32)

let point_cases k points =
  cases
    ~name:(fun (name, _, _, _) -> name)
    "edges" points
    (fun (_, x, v, expected) -> at ~v (unary k) x expected)

let unary_edges (k : Nx_backend.unary) =
  let inf = Float.infinity and nan = Float.nan in
  let row (name, x, expected) = (name, x, 1., expected) in
  match k with
  | Neg ->
      [
        point_cases k
          [
            ("a +0 tangent turns into -0", 2., 0., -0.);
            ("a -0 tangent turns into +0", 2., -0., 0.);
          ];
      ]
  | Recip ->
      [
        point_cases k
          (List.map row
             [
               ("at +0 the tangent is -inf", 0., Float.neg_infinity);
               ("at -0 the tangent is -inf", -0., Float.neg_infinity);
               ("at +inf the tangent is -0", inf, -0.);
               ("at NaN the tangent is NaN", nan, nan);
               ( "at a subnormal the tangent overflows to -inf",
                 5e-324,
                 Float.neg_infinity );
             ]);
      ]
  | Sqrt ->
      [
        point_cases k
          (List.map row
             [
               ("at +0 the tangent is +inf", 0., inf);
               ( "at -0 the tangent is -inf, the formula's at sqrt (-0) = -0",
                 -0.,
                 Float.neg_infinity );
               ("below zero the tangent is NaN", -1., nan);
               ("at +inf the tangent is 0", inf, 0.);
             ]);
      ]
  | Exp ->
      [
        point_cases k
          (List.map row
             [
               ("at -inf the tangent is 0", Float.neg_infinity, 0.);
               ("past the float range the tangent is +inf", 710., inf);
               ("a subnormal tangent is not flushed", -740., Float.exp (-740.));
             ]);
      ]
  | Log ->
      [
        point_cases k
          (List.map row
             [
               ("at +0 the tangent is +inf", 0., inf);
               ("at -0 the tangent is -inf", -0., Float.neg_infinity);
               ("below zero the tangent is v / x beside a NaN value", -2., -0.5);
               ("at +inf the tangent is 0", inf, 0.);
             ]);
      ]
  | Sin | Cos ->
      [
        point_cases k
          (List.map row [ ("at +inf the tangent is NaN", inf, nan) ]);
      ]
  | Asin | Acos ->
      let s = match k with Asin -> 1. | _ -> -1. in
      [
        point_cases k
          (List.map row
             [
               ("at 1 the tangent is infinite", 1., s *. inf);
               ("at -1 the tangent is infinite", -1., s *. inf);
               ("past 1 the tangent is NaN", 1.5, nan);
             ]);
        test "near ±1 the gradient is ±1 / sqrt (1 - x²), to 4 ulps" (fun () ->
            let e =
              match k with
              | Asin -> { f = Nx.asin }
              | _ -> { f = Nx.acos }
            in
            within_ulps 4 e
              ~reference64:(fun x -> s /. Float.sqrt (one_minus_square x))
              ~reference32:(fun x ->
                to_float32 (s /. Float.sqrt ((1. -. x) *. (1. +. x))))
              [| 0.9; 0.999; 0.999999; -0.999999 |]);
      ]
  | Atan ->
      [ point_cases k (List.map row [ ("at +inf the tangent is 0", inf, 0.) ]) ]
  | Tanh ->
      [
        point_cases k
          (List.map row
             [
               ("at +inf the tangent is 0", inf, 0.);
               ("where tanh rounds to 1 the tangent is 0", 20., 0.);
             ]);
      ]
  | Erf ->
      [ point_cases k (List.map row [ ("far out the tangent is 0", 30., 0.) ]) ]
  | Abs ->
      [
        test "at +0 and -0 the tangent is a zero" (fun () ->
            List.iter
              (fun x ->
                equal ~msg:(Printf.sprintf "at %g" x) (float 1e-300) 0.
                  (tangent (unary Abs) (vec [| x |]) (vec [| 1. |])).(0))
              [ 0.; -0. ]);
        point_cases k
          (List.map row
             [
               ("at +inf the tangent is v", inf, 1.);
               ("at -inf the tangent is -v", Float.neg_infinity, -1.);
               ("at NaN the tangent is NaN", nan, nan);
             ]);
      ]
  | Sign | Cosh | Sinh | Tan | Trunc | Ceil | Floor | Round -> []

(* Binary *)

let along_a k b x = binary k x (vec [| b |])
let along_b k a x = binary k (vec [| a |]) x

let binary_edges (k : Nx_backend.binary) =
  let inf = Float.infinity and nan = Float.nan in
  match k with
  | Add | Sub ->
      [
        test "a single tangent keeps -0: the constant adds no zero" (fun () ->
            at ~v:(-0.) (along_a k 3.) 1. (-0.);
            at ~v:(-0.) (along_b Add 3.) 1. (-0.));
      ]
  | Mul ->
      [
        test "a constant's infinite coefficient never meets a zero" (fun () ->
            at (along_a Mul 2.) inf 2.;
            at (along_b Mul 2.) inf 2.);
        test "an infinite constant is the coefficient" (fun () ->
            at (along_a Mul inf) 1. inf);
      ]
  | Fdiv ->
      [
        test "a numerator at +inf over a constant has the tangent v / b"
          (fun () -> at (along_a Fdiv 2.) inf 0.5);
        test "over a zero the numerator's tangent is +inf" (fun () ->
            at (along_a Fdiv 0.) 1. inf);
        test "a zero denominator's tangent is -inf" (fun () ->
            at (along_b Fdiv 1.) 0. Float.neg_infinity);
      ]
  | Pow ->
      let base b x = binary Pow x (vec [| b |])
      and exponent a x = binary Pow (vec [| a |]) x in
      [
        cases
          ~name:(fun (name, _, _, _) -> name)
          "at a zero base"
          [
            ("x ** 2 has tangent 0", 2., 0., 0.);
            ("x ** 3 has tangent 0", 3., 0., 0.);
            ("x ** 0.5 has tangent +inf", 0.5, 0., inf);
            ("x ** 1 has tangent 1", 1., 0., 1.);
          ]
          (fun (_, b, x, expected) ->
            at (base b) x expected;
            equal floats [| expected |] (gradient (base b) (vec [| x |])));
        test "x ** 0 has tangent 0 at a zero base, as everywhere" (fun () ->
            at (base 0.) 0. 0.;
            equal floats [| 0. |] (gradient (base 0.) (vec [| 0. |])));
        test "0 ** b has tangent 0 along b > 0" (fun () ->
            at (exponent 0.) 2. 0.;
            equal floats [| 0. |] (gradient (exponent 0.) (vec [| 2. |])));
        test
          "a negative base with an integer exponent has a finite tangent along \
           the base" (fun () -> at (base 3.) (-2.) 12.);
        test "a negative base has a NaN tangent along the exponent" (fun () ->
            at (exponent (-2.)) 3. nan);
        test "a negative base with a fractional exponent has NaN tangents"
          (fun () ->
            at (base (1. /. 3.)) (-8.) nan;
            at (exponent (-8.)) (1. /. 3.) nan);
        test "a base of 1 has tangent 0 along the exponent" (fun () ->
            at (exponent 1.) 2.5 0.);
        test "the mixed second derivative at a zero exponent is 1 / a"
          (fun () ->
            let pair = Nx.Ptree.(pair tensor tensor) in
            let f (a, b) = binary Pow a b in
            let one = vec [| 1. |] and zero = vec [| 0. |] in
            let along_a p =
              snd (Rune.jvp pair Nx.Ptree.tensor f p (one, zero))
            in
            let _, ab =
              Rune.jvp pair Nx.Ptree.tensor along_a
                (vec [| 0.1 |], zero)
                (zero, one)
            in
            equal (array (float 1e-9)) [| 10. |] (Nx.to_array ab));
      ]
  | Maximum | Minimum ->
      let first, second = binary_tie in
      let tie name a b =
        test name (fun () ->
            at (along_a k b) a first;
            at (along_b k a) b second)
      in
      [
        tie "at a tie each operand takes its share of the derivative" 1. 1.;
        tie "-0 and +0 are a tie" (-0.) 0.;
        tie "+0 and -0 are a tie" 0. (-0.);
        test "a NaN operand's tangent is the result's" (fun () ->
            at (along_a k 1.) nan 1.;
            at (along_b k nan) 1. 0.;
            at (along_b k 1.) nan 1.;
            at (along_a k nan) 1. 0.);
      ]
  | Atan2 ->
      [
        test "at the origin the tangents are NaN" (fun () ->
            at (along_a Atan2 0.) 0. nan;
            at (along_b Atan2 0.) 0. nan);
      ]
  | Mod ->
      [
        test "at a multiple the tangent is the one from above" (fun () ->
            at (along_a Mod 2.) 4. 1.;
            at (along_b Mod 4.) 2. (-2.));
        test "a negative quotient truncates toward zero" (fun () ->
            at (along_b Mod (-3.5)) 2. 1.);
      ]
  | Idiv | And | Or | Xor ->
      let f x = binary k x (vec [| 3.; 3. |]) and x = vec [| 5.; -2. |] in
      [
        test "on floats, its tangent and gradient are eager's verdict"
          (fun () ->
            match f x with
            | exception e ->
                raises ~msg:"the tangent" e (fun () ->
                    Rune.jvp' f x (vec [| 1.; 1. |]));
                raises ~msg:"the gradient" e (fun () ->
                    Rune.grad' (fun x -> Nx.sum (f x)) x)
            | y ->
                let y', dy = Rune.jvp' f x (vec [| 1.; 1. |]) in
                equal ~msg:"the primal" (Reference.exact ()) y y';
                equal ~msg:"the tangent" floats [| 0.; 0. |] (Nx.to_array dy);
                equal ~msg:"the gradient" floats [| 0.; 0. |] (gradient f x));
      ]

(* Selection, reductions *)

let where_edges =
  let positive x = Op.eval (Compare (Less, Nx.zeros_like x, x)) in
  [
    test "a non-finite tangent of the unselected branch does not leak"
      (fun () ->
        at
          (fun x -> Op.eval (Where (positive x, unary Sqrt x, Nx.zeros_like x)))
          0. 0.;
        at (fun x -> Op.eval (Where (positive x, unary Recip x, x))) 0. 1.);
  ]

let reduce k axes x = Op.eval (Reduce (k, axes, x))

let reduce_edges (k : Nx_backend.reduce) =
  match k with
  | Sum ->
      [
        test "over no axis the tangent is the tangent" (fun () ->
            equal
              (array (float 1e-300))
              [| 3.; -2. |]
              (tangent (reduce Sum [||]) (vec [| 1.; 2. |]) (vec [| 3.; -2. |])));
        test "over an empty axis the tangent is zeros" (fun () ->
            let x = Nx.zeros Nx.float64 [| 2; 0 |] in
            equal floats [| 0.; 0. |]
              (Nx.to_array (snd (Rune.jvp' (reduce Sum [| 1 |]) x x))));
      ]
  | Prod ->
      let input =
        Nx.create Nx.float64 [| 3; 3 |] [| 2.; 3.; 4.; 2.; 0.; 4.; 0.; 0.; 4. |]
      in
      [
        test "one zero leaves the product of the others, two leave zero"
          (fun () ->
            equal floats
              [| 12.; 8.; 6.; 0.; 8.; 0.; 0.; 0.; 0. |]
              (gradient (reduce Prod [| 1 |]) input);
            equal floats [| 34. |]
              (Nx.to_array
                 (snd
                    (Rune.jvp'
                       (fun x -> Nx.sum (reduce Prod [| 1 |] x))
                       input (Nx.ones_like input)))));
        test "the second derivative at one zero is the polynomial's" (fun () ->
            equal floats [| 2.5; 0.5; 2. |]
              (Nx.to_array
                 (Rune.grad'
                    (fun x -> Nx.sum (Rune.grad' (reduce Prod [| 0 |]) x))
                    (vec [| 0.; 2.; 0.5 |]))));
        test "the second derivative at two zeros is the polynomial's" (fun () ->
            equal floats [| 0.; 0.01; 0.01 |]
              (Nx.to_array
                 (Rune.grad'
                    (fun x -> Nx.sum (Rune.grad' (reduce Prod [| 0 |]) x))
                    (vec [| 0.01; 0.; 0. |]))));
        test "the tangent survives a product that underflows" (fun () ->
            equal floats [| 5.33e-9; 4.94e-324 |]
              (Nx.to_array
                 (Rune.grad' (reduce Prod [| 0 |])
                    (vec [| 4.94e-324; 5.33e-9 |]))));
      ]
  | Max | Min ->
      let input =
        Nx.create Nx.float64 [| 2; 3 |] [| 2.; 2.; 0.; -1.; -1.; 3. |]
      in
      let v = Nx.create Nx.float64 [| 2; 3 |] [| 1.; 3.; 5.; 2.; 4.; 6. |] in
      let share = reduce_tie 2 in
      let expected, along =
        match k with
        | Max -> ([| share; share; 0.; 0.; 0.; 1. |], (share *. 4.) +. 6.)
        | _ -> ([| 0.; 0.; 1.; share; share; 0. |], 5. +. (share *. 6.))
      in
      [
        test "tied elements share the derivative" (fun () ->
            equal floats expected (gradient (reduce k [| 1 |]) input);
            equal floats [| along |]
              (Nx.to_array
                 (snd
                    (Rune.jvp' (fun x -> Nx.sum (reduce k [| 1 |] x)) input v))));
        test "a float16 tie count above 65,504 does not overflow" (fun () ->
            let n = 65536 in
            let x = Nx.ones Nx.float16 [| n |] in
            equal ~msg:"the gradient"
              (array (float 1e-9))
              (Array.make n (reduce_tie n))
              (Nx.to_array (Rune.grad' (reduce k [| 0 |]) x));
            equal ~msg:"the tangent along ones" (array float_exact) [| 1. |]
              (Nx.to_array
                 (Nx.reshape [| 1 |] (snd (Rune.jvp' (reduce k [| 0 |]) x x)))));
        test "a NaN extremum has a NaN tangent" (fun () ->
            equal floats [| Float.nan |]
              (Nx.to_array
                 (Nx.reshape [| 1 |]
                    (snd
                       (Rune.jvp' (reduce k [| 0 |])
                          (vec [| 1.; Float.nan |])
                          (vec [| 1.; 1. |]))))));
      ]

(* Sorting *)

let sort_law =
  let draw =
    let open Gen in
    let* n = int_range 1 7 in
    let+ x =
      array ~size:(constant n) (of_list [ -1.; 0.; -0.; 1.; 1.; Float.nan ])
    and+ v = array ~size:(constant n) (float_range (-2.) 2.)
    and+ descending = bool in
    (x, v, descending)
  in
  prop
    "the tangent is the tangent gathered by the primal's argsort, bit for bit"
    (Gen.with_pp
       (fun ppf (x, v, d) ->
         Format.fprintf ppf "x %a@ v %a@ descending %b" Nx.pp (vec x) Nx.pp
           (vec v) d)
       draw)
    (fun (x, v, descending) ->
      cover "a tie"
        (Array.length x
        <> Array.length
             (Array.of_list (List.sort_uniq compare (Array.to_list x))));
      cover "a NaN" (Array.exists Float.is_nan x);
      let x = vec x and v = vec v in
      let indices = Op.eval (Argsort { descending; axis = 0; x }) in
      equal (Reference.exact ())
        (Op.eval (Gather (0, indices, v)))
        (snd
           (Rune.jvp' (fun x -> Op.eval (Sort { descending; axis = 0; x })) x v)))

(* Assembly, indexing, conversions *)

let assembly_edges : Row.t -> test list = function
  | Pad ->
      [
        test "the tangent's fill is zero" (fun () ->
            equal floats [| 0.; 1.; 2.; 0. |]
              (tangent
                 (fun x -> Op.eval (Pad ([| (1, 1) |], 5., x)))
                 (vec [| 3.; 4. |])
                 (vec [| 1.; 2. |])));
      ]
  | Cat ->
      [
        test "a piece with no tangent contributes zeros" (fun () ->
            equal floats [| 0.; 0.; 1.; 2. |]
              (tangent
                 (fun x -> Op.eval (Cat (0, [ vec [| 7.; 8. |]; x ])))
                 (vec [| 3.; 4. |])
                 (vec [| 1.; 2. |])));
      ]
  | Gather ->
      [
        test "an index outside the axis reads a zero tangent" (fun () ->
            let indices = Nx.create Nx.int64 [| 4 |] [| -1L; 3L; 1L; 1L |] in
            equal floats [| 0.; 0.; 20.; 20. |]
              (tangent
                 (fun x -> Op.eval (Gather (0, indices, x)))
                 (vec [| 1.; 2.; 3. |])
                 (vec [| 10.; 20.; 30. |])));
      ]
  | Scatter (`Set | `Add) ->
      let indices = Nx.create Nx.int64 [| 2 |] [| 1L; 1L |] in
      let scatter mode updates =
        Op.eval
          (Scatter
             {
               mode;
               unique = false;
               axis = 0;
               indices;
               updates;
               into = Nx.zeros Nx.float64 [| 3 |];
             })
      in
      [
        test "under Set the last of duplicate updates wins its tangent"
          (fun () ->
            equal floats [| 0.; 2.; 0. |]
              (tangent (scatter `Set) (vec [| 5.; 6. |]) (vec [| 1.; 2. |])));
        test "under Add duplicate updates add their tangents" (fun () ->
            equal floats [| 0.; 3.; 0. |]
              (tangent (scatter `Add) (vec [| 5.; 6. |]) (vec [| 1.; 2. |])));
      ]
  | Scatter ((`Max | `Min) as mode) ->
      (* The tangent at [into] and [updates], laid end to end in one vector,
         along the element tangents [10; 20; 30] and the update tangents [1; 2;
         ...]. *)
      let at into indices updates =
        let n = Array.length into and k = Array.length updates in
        let f x =
          Op.eval
            (Scatter
               {
                 mode;
                 unique = false;
                 axis = 0;
                 indices =
                   Nx.create Nx.int64 [| k |] (Array.map Int64.of_int indices);
                 updates = Nx.slice [ R (n, n + k) ] x;
                 into = Nx.slice [ R (0, n) ] x;
               })
        in
        tangent f
          (vec (Array.append into updates))
          (vec
             (Array.append
                (Array.init n (fun i -> float_of_int (10 * (i + 1))))
                (Array.init k (fun i -> float_of_int (i + 1)))))
      in
      let best = match mode with `Max -> 7. | `Min -> -7. in
      let zero, other_zero =
        match mode with `Max -> (0., -0.) | `Min -> (-0., 0.)
      in
      let nan = Float.nan in
      [
        test "a tie with the element gives the element's tangent" (fun () ->
            equal floats [| 10.; 20. |] (at [| best; 0. |] [| 0 |] [| best |]));
        test "of tied updates the first gives its tangent" (fun () ->
            (* At a tie a scatter by extremes gives the tangent of the element,
               then of the first update in index order. *)
            equal floats [| 10.; 1. |]
              (at [| 0.; 0. |] [| 1; 1 |] [| best; best |]));
        test
          "the zero that wins, +0 under Max and -0 under Min, gives its tangent"
          (fun () ->
            equal floats [| 1. |] (at [| other_zero |] [| 0 |] [| zero |]);
            equal floats [| 10. |] (at [| zero |] [| 0 |] [| other_zero |]));
        test "a NaN element keeps its tangent ahead of NaN updates" (fun () ->
            equal floats [| 10. |] (at [| nan |] [| 0; 0 |] [| nan; 3. |]));
        test "a number element takes the first NaN update's tangent" (fun () ->
            equal floats [| 2. |] (at [| 1. |] [| 0; 0; 0 |] [| 2.; nan; nan |]));
        test "a dropped or losing update contributes nothing" (fun () ->
            equal floats [| 10.; 20.; 30. |]
              (at [| best; best; best |] [| 0; 3 |] [| 0.; best |]));
        test
          "reduce_segments and the reduction agree in value and differ in \
           tangent at a tie" (fun () ->
            let x = vec [| best; best; 0. |] and v = vec [| 1.; 2.; 4. |] in
            let segments x =
              Nx.reduce_segments mode ~segments:1 (Nx.zeros Nx.int64 [| 3 |]) x
            in
            let reduced x =
              Nx.reshape [| 1 |]
                (match mode with `Max -> Nx.max x | `Min -> Nx.min x)
            in
            let y, dy = Rune.jvp' segments x v
            and y', dy' = Rune.jvp' reduced x v in
            equal floats (Nx.to_array y') (Nx.to_array y);
            equal floats [| 1. |] (Nx.to_array dy);
            equal floats [| reduce_tie 2 *. 3. |] (Nx.to_array dy'));
      ]
  | Cast ->
      let z re im = { Complex.re; im } in
      [
        test "a cast to a real dtype keeps the real part of the tangent"
          (fun () ->
            equal floats [| 3. |]
              (Nx.to_array
                 (snd
                    (Rune.jvp'
                       (fun x -> Op.eval (Convert (Cast, Nx.float64, x)))
                       (Nx.create Nx.complex128 [| 1 |] [| z 1. 2. |])
                       (Nx.create Nx.complex128 [| 1 |] [| z 3. 4. |])))));
        test "a cast to a complex dtype has a zero imaginary tangent" (fun () ->
            equal
              (array (pair float_exact float_exact))
              [| (3., 0.) |]
              (Array.map
                 (fun c -> (c.Complex.re, c.im))
                 (Nx.to_array
                    (snd
                       (Rune.jvp'
                          (fun x -> Op.eval (Convert (Cast, Nx.complex128, x)))
                          (vec [| 1. |]) (vec [| 3. |]))))));
        test "a tangent past float16's range overflows as its value does"
          (fun () ->
            equal floats [| Float.infinity |]
              (Nx.to_array
                 (Nx.cast Nx.float64
                    (snd
                       (Rune.jvp'
                          (fun x -> Op.eval (Convert (Cast, Nx.float16, x)))
                          (vec [| 1. |]) (vec [| 1e5 |]))))));
        test "a cast to an integer or boolean dtype has no tangent" (fun () ->
            let check (type c d) (dtype : (c, d) Nx.dtype) =
              let y, dy =
                Rune.jvp'
                  (fun x -> Op.eval (Convert (Cast, dtype, x)))
                  (vec [| 1.5; -2. |])
                  (vec [| 1.; 1. |])
              in
              equal (Reference.exact ()) (Nx.zeros_like y) dy
            in
            check Nx.int32;
            check Nx.int64;
            check Nx.bool);
      ]
  | _ -> []

let reshape_edges =
  [
    test "a tangent laid out unlike its primal reshapes as the primal does"
      (fun () ->
        (* An upper Cholesky factor's tangent is a transposed view. *)
        let a = Nx.create Nx.float64 [| 2; 2 |] [| 4.; 0.; 1.; 3. |] in
        let v = Nx.create Nx.float64 [| 2; 2 |] [| 1.; 0.; 0.5; 2. |] in
        let f x =
          Op.eval
            (Move (Op.eval (Cholesky { upper = true; x }), Reshape [| 4 |]))
        in
        let factor x = Op.eval (Cholesky { upper = true; x }) in
        equal (Reference.exact ())
          (Nx.reshape [| 4 |] (Nx.contiguous (snd (Rune.jvp' factor a v))))
          (snd (Rune.jvp' f a v)));
  ]

let read_edges =
  [
    test "a read of a dual reads its primal" (fun () ->
        let y, dy =
          Rune.jvp'
            (fun x -> Nx.mul_s x (Nx.item [ 0 ] x))
            (vec [| 3. |]) (vec [| 1. |])
        in
        equal floats [| 9. |] (Nx.to_array y);
        equal floats [| 3. |] (Nx.to_array dy));
  ]

(* Linear algebra *)

let mat n a = Nx.create Nx.float64 [| n; n |] a

let linalg_edges : Row.t -> test list = function
  | Cholesky ->
      let a = mat 2 [| 4.; 9.; 1.; 3. |] in
      [
        test "a tangent above the diagonal has no effect, in both modes"
          (fun () ->
            List.iter
              (fun upper ->
                let w, z = zeros 4 in
                equal
                  ~msg:(if upper then "upper" else "lower")
                  w z
                  (tangent
                     (fun x -> Op.eval (Cholesky { upper; x }))
                     a
                     (mat 2 [| 0.; 1.; 0.; 0. |])))
              [ false; true ]);
        test "an imaginary tangent on the diagonal has no effect" (fun () ->
            let a = Nx.cast Nx.complex128 a in
            let v =
              Nx.create Nx.complex128 [| 2; 2 |]
                Complex.
                  [| { re = 0.; im = 1. }; zero; zero; { re = 0.; im = 1. } |]
            in
            equal (Reference.exact ()) (Nx.zeros_like v)
              (snd
                 (Rune.jvp'
                    (fun x -> Op.eval (Cholesky { upper = false; x }))
                    a v)));
      ]
  | Qr ->
      [
        test "a wide matrix's tangent agrees with a central difference"
          (fun () ->
            (* The second block of R depends on the first block's Q. *)
            let x =
              Nx.create Nx.float64 [| 2; 4 |]
                [| 2.; 0.5; 1.; -0.3; 0.4; 1.5; 0.7; 2. |]
            in
            let v =
              Nx.create Nx.float64 [| 2; 4 |]
                [| 0.3; -1.; 0.6; 1.2; -0.4; 0.9; -1.1; 0.5 |]
            in
            let f x =
              let q, r = Op.eval (Qr { reduced = true; x }) in
              Nx.concatenate ~axis:(-1) [ q; r ]
            in
            equal
              (Reference.close ~rel:1e-8 ~floor:1e-8 ())
              (Reference.central Nx.Ptree.tensor Nx.Ptree.tensor ~eps:1e-6 f x v)
              [ Reference.complexes (snd (Rune.jvp' f x v)) ]);
        test "a complete factorisation of a tall matrix has no tangent"
          (fun () ->
            let x =
              Nx.create Nx.float64 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 1.; 1. |]
            in
            raises
              (Invalid_argument
                 "Rune.jvp': the tangent of a complete QR factorisation of a \
                  tall matrix has no definition") (fun () ->
                Rune.jvp'
                  (fun x -> fst (Op.eval (Qr { reduced = false; x })))
                  x x));
      ]
  | Solve_triangular ->
      let a = mat 2 [| 2.; 9.; 0.5; 3. |] and b = vec [| 1.; -2. |] in
      let solve ~unit_diag x =
        Op.eval
          (Solve_triangular
             { upper = false; transpose = false; unit_diag; a = x; b })
      in
      [
        test "a tangent in the triangle the solve does not read has no effect"
          (fun () ->
            let w, z = zeros 2 in
            equal w z
              (tangent (solve ~unit_diag:false) a (mat 2 [| 0.; 1.; 0.; 0. |])));
        test "under unit_diag a tangent on the diagonal has no effect"
          (fun () ->
            let w, z = zeros 2 in
            equal w z
              (tangent (solve ~unit_diag:true) a (mat 2 [| 1.; 0.; 0.; 1. |])));
      ]
  | _ -> []

let of_row (r : Row.t) =
  match r with
  | Unary k -> unary_edges k
  | Binary k -> binary_edges k
  | Where -> where_edges
  | Reduce k -> reduce_edges k
  | Sort -> [ sort_law ]
  | Pad | Cat | Gather | Scatter _ | Cast -> assembly_edges r
  | Cholesky | Qr | Solve_triangular -> linalg_edges r
  | Read -> read_edges
  | Move Reshape -> reshape_edges
  | _ -> []
