(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Piecewise: one series value for interpolants and fits. The trusted side
   is scipy's splines, the samples an interpolant must pass through, and the
   laws of calculus checked against rune's derivative of evaluation. *)

open Windtrap
open Jera
module G = Golden_piecewise

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()
let close ?(rel = 1e-13) () = Oracle.tensor ~rel ~abs:1e-14 ()
let knots = vec G.knots
let samples = vec G.samples
let samples2 = Nx.create f64 [| Array.length G.knots; 2 |] G.samples2
let points = vec G.points
let raises_with sub f = raises_match (Exn.invalid_arg ~substring:sub) f

(* Goldens *)

let clamped = `Clamped (scalar 0.5, scalar (-2.))

let golden_tests =
  let spline name ends values derivative integral =
    let p = Piecewise.cubic ends knots samples in
    [
      test (name ^ " is scipy's spline") (fun () ->
          equal (close ()) (vec values) (Piecewise.eval p points));
      test (name ^ "'s derivative is scipy's") (fun () ->
          equal (close ()) (vec derivative)
            (Piecewise.eval (Piecewise.derivative p) points));
      test (name ^ "'s integral is scipy's") (fun () ->
          equal (close ()) (vec integral)
            (Piecewise.eval (Piecewise.integral p) points));
    ]
  in
  List.concat
    [
      spline "natural" `Natural G.cubic_natural G.cubic_natural_derivative
        G.cubic_natural_integral;
      spline "not-a-knot" `Not_a_knot G.cubic_not_a_knot
        G.cubic_not_a_knot_derivative G.cubic_not_a_knot_integral;
      spline "clamped" clamped G.cubic_clamped G.cubic_clamped_derivative
        G.cubic_clamped_integral;
      [
        cases
          ~name:(fun (n, _, _) -> Printf.sprintf "%d knots" n)
          "not-a-knot through few knots is scipy's"
          [
            (2, G.not_a_knot2_points, G.not_a_knot2);
            (3, G.not_a_knot3_points, G.not_a_knot3);
            (4, G.not_a_knot4_points, G.not_a_knot4);
          ]
          (fun (n, pts, values) ->
            let p =
              Piecewise.cubic `Not_a_knot
                (Nx.slice [ Nx.R (0, n) ] knots)
                (Nx.slice [ Nx.R (0, n) ] samples)
            in
            equal (close ()) (vec values) (Piecewise.eval p (vec pts)));
        cases
          ~name:(fun (n, _, _) -> n)
          "samples with a value axis give scipy's splines"
          [
            ("natural", `Natural, G.cubic_natural2);
            ("not-a-knot", `Not_a_knot, G.cubic_not_a_knot2);
            ( "clamped",
              `Clamped (vec [| 0.5; 0.5 |], vec [| -2.; -2. |]),
              G.cubic_clamped2 );
          ]
          (fun (_, ends, values) ->
            let p = Piecewise.cubic ends knots samples2 in
            equal (close ())
              (Nx.create f64 [| Array.length G.points; 2 |] values)
              (Piecewise.eval p points));
        test "hermite is scipy's" (fun () ->
            let p =
              Piecewise.hermite knots ~values:samples ~slopes:(vec G.slopes)
            in
            equal (close ()) (vec G.hermite) (Piecewise.eval p points));
        test "linear is numpy's interp" (fun () ->
            equal (close ()) (vec G.linear)
              (Piecewise.eval (Piecewise.linear knots samples) points));
      ];
    ]

(* Laws *)

(* Random series: 1 to 4 pieces of degree 0 to 8 over increasing breaks in [-2,
   2], coefficients in [-1, 1], and points in the domain. *)
let series =
  Gen.(
    let* pieces = int_range 1 4 in
    let* degree = int_range 0 8 in
    let* widths = array ~size:(constant pieces) (float_range 0.1 1.) in
    let* start = float_range (-2.) 0. in
    let* c =
      array ~size:(constant (pieces * (degree + 1))) (float_range (-1.) 1.)
    in
    let* t = array ~size:(constant 5) (float_range 0. 1.) in
    let breaks = Array.make (pieces + 1) start in
    Array.iteri (fun i w -> breaks.(i + 1) <- breaks.(i) +. w) widths;
    let last = breaks.(pieces) in
    let point t = Float.min last (start +. (t *. (last -. start))) in
    constant
      (breaks, Nx.create f64 [| pieces; degree + 1 |] c, Array.map point t))
  |> Gen.with_pp (fun ppf (b, c, x) ->
      let floats =
        Format.pp_print_array ~pp_sep:Format.pp_print_space
          Format.pp_print_float
      in
      Format.fprintf ppf "@[breaks [%a]@ coefficients %a@ points [%a]@]" floats
        b Nx.pp c floats x)

let of_series (breaks, c, _) =
  Piecewise.v Nx.Ptree.tensor ~breaks:(vec breaks) c

(* (degree + 1)² eps Σ |c_k|, scaled by the largest piece's derivative
   factor. *)
let bound (breaks, c, _) ~order =
  let degree = Nx.dim 1 c - 1 in
  let sum =
    Array.fold_left (fun acc x -> acc +. Float.abs x) 0. (Nx.to_array c)
  in
  let narrow = ref infinity in
  Array.iteri
    (fun i b -> if i > 0 then narrow := Float.min !narrow (b -. breaks.(i - 1)))
    breaks;
  let scale = (2. /. !narrow) ** Float.of_int order in
  Float.of_int ((degree + 2) * (degree + 2))
  *. epsilon_float *. (sum +. 1.) *. Float.max scale 1. *. 16.

let law_tests =
  [
    prop "eval of derivative is the derivative of eval" series
      (fun ((_, _, pts) as s) ->
        let p = of_series s in
        let x = vec pts in
        let along = snd (Rune.jvp' (Piecewise.eval p) x (Nx.ones_like x)) in
        equal
          (Oracle.tensor ~abs:(bound s ~order:1) ())
          along
          (Piecewise.eval (Piecewise.derivative p) x));
    prop "the integral vanishes at the first break" series
      (fun ((breaks, _, _) as s) ->
        let p = Piecewise.integral (of_series s) in
        equal
          (Oracle.tensor ~abs:(bound s ~order:0) ())
          (vec [| 0. |])
          (Piecewise.eval p (vec [| breaks.(0) |])));
    prop "the integral's derivative is the series" series
      (fun ((_, _, pts) as s) ->
        let p = of_series s in
        let x = vec pts in
        let i = Piecewise.integral p in
        let along = snd (Rune.jvp' (Piecewise.eval i) x (Nx.ones_like x)) in
        equal
          (Oracle.tensor ~abs:(bound s ~order:0) ())
          (Piecewise.eval p x) along);
    prop "the integral is continuous at the breaks" series
      (fun ((breaks, _, _) as s) ->
        let i =
          Piecewise.extend `Polynomial (Piecewise.integral (of_series s))
        in
        let c = Piecewise.coefficients i in
        let n = Array.length breaks - 1 in
        (* Piece k − 1 at u = 1 against piece k at u = −1. *)
        let left =
          Piecewise.eval_at i
            (Nx.arange Nx.int64 0 (n - 1) 1)
            (Nx.ones f64 [| n - 1 |])
        in
        let right =
          Piecewise.eval_at i (Nx.arange Nx.int64 1 n 1)
            (Nx.full f64 [| n - 1 |] (-1.))
        in
        ignore c;
        equal (Oracle.tensor ~abs:(bound s ~order:0) ()) left right);
    prop "interpolants pass through their samples"
      Gen.(
        let* n = int_range 2 9 in
        let* gaps = array ~size:(constant (n - 1)) (float_range 0.05 1.) in
        let* y = array ~size:(constant n) (float_range (-3.) 3.) in
        let x = Array.make n 0. in
        Array.iteri (fun i g -> x.(i + 1) <- x.(i) +. g) gaps;
        constant (x, y))
      (fun (x, y) ->
        let x = vec x and y = vec y in
        List.iter
          (fun (name, p) ->
            equal ~msg:name
              (Oracle.tensor ~rel:1e-12 ~abs:1e-12 ())
              y (Piecewise.eval p x))
          [
            ("linear", Piecewise.linear x y);
            ("natural", Piecewise.cubic `Natural x y);
            ("not-a-knot", Piecewise.cubic `Not_a_knot x y);
            ("clamped", Piecewise.cubic (`Clamped (scalar 1., scalar 0.)) x y);
            ("steffen", Piecewise.steffen x y);
            ("hermite", Piecewise.hermite x ~values:y ~slopes:(Nx.zeros_like y));
          ]);
    prop "steffen of monotone samples is monotone"
      Gen.(
        let* n = int_range 2 9 in
        let* gaps = array ~size:(constant (n - 1)) (float_range 0.05 1.) in
        let* rises = array ~size:(constant (n - 1)) (float_range 0. 2.) in
        let x = Array.make n 0. and y = Array.make n 0. in
        Array.iteri
          (fun i g ->
            x.(i + 1) <- x.(i) +. g;
            y.(i + 1) <- y.(i) +. rises.(i))
          gaps;
        constant (x, y))
      (fun (x, y) ->
        let p = Piecewise.steffen (vec x) (vec y) in
        let last = x.(Array.length x - 1) in
        let fine = Nx.minimum (Nx.linspace f64 x.(0) last 400) (scalar last) in
        let v = Oracle.floats (Piecewise.eval p fine) in
        let tiny = 1e-13 *. (1. +. Float.abs y.(Array.length y - 1)) in
        for i = 1 to Array.length v - 1 do
          at_least
            ~msg:(Printf.sprintf "point %d" i)
            float_exact
            ~than:(v.(i - 1) -. tiny)
            v.(i)
        done);
  ]

(* Fits *)

let fit_tests =
  [
    test "a fit of exp converges to rounding" (fun () ->
        let p =
          Piecewise.chebyshev Nx.Ptree.tensor ~degree:16 ~pieces:2 Nx.exp
            (scalar 0.) (scalar 2.)
        in
        let x = Nx.linspace f64 0. 2. 37 in
        equal (Oracle.tensor ~rel:4e-15 ()) (Nx.exp x) (Piecewise.eval p x));
    test "a fit of degree d reproduces a polynomial of degree d" (fun () ->
        let f x = Nx.add_s (Nx.mul x (Nx.sub_s (Nx.mul x x) 2.)) 0.5 in
        let p =
          Piecewise.chebyshev Nx.Ptree.tensor ~degree:3 ~pieces:3 f
            (scalar (-1.)) (scalar 2.)
        in
        let x = Nx.linspace f64 (-1.) 2. 11 in
        equal (close ~rel:1e-14 ()) (f x) (Piecewise.eval p x));
    test "a fit of a structure fits each leaf" (fun () ->
        let s = Nx.Ptree.(pair tensor tensor) in
        let p =
          Piecewise.chebyshev s ~degree:20 ~pieces:1
            (fun x -> (Nx.sin x, Nx.cos x))
            (scalar 0.) (scalar 1.)
        in
        let x = vec [| 0.; 0.25; 1. |] in
        equal
          (Oracle.structure ~rel:1e-14 ~abs:1e-15 s)
          (Nx.sin x, Nx.cos x)
          (Piecewise.eval p x));
    test "degree 0 is the midpoint's value" (fun () ->
        let p =
          Piecewise.chebyshev Nx.Ptree.tensor ~degree:0 ~pieces:2 Nx.exp
            (scalar 0.) (scalar 2.)
        in
        equal (close ())
          (vec [| Float.exp 0.5; Float.exp 1.5 |])
          (Piecewise.eval p (vec [| 0.2; 1.9 |])));
    test "a fit rejects a negative degree" (fun () ->
        raises_with "degree = -1 is negative" (fun () ->
            Piecewise.chebyshev Nx.Ptree.tensor ~degree:(-1) ~pieces:1 Nx.exp
              (scalar 0.) (scalar 1.)));
    test "a fit rejects a function of another shape" (fun () ->
        raises_with "shape [2] for points of shape [2,4]" (fun () ->
            Piecewise.chebyshev Nx.Ptree.tensor ~degree:3 ~pieces:2
              (fun x -> Nx.sum ~axes:[ 1 ] x)
              (scalar 0.) (scalar 1.)));
  ]

(* Domain *)

let domain_tests =
  let p = Piecewise.linear (vec [| 0.; 1.; 2. |]) (vec [| 0.; 2.; 3. |]) in
  [
    test "the last break is in the last piece" (fun () ->
        equal (exact ()) (vec [| 3. |]) (Piecewise.eval p (vec [| 2. |])));
    test "an interior break is in the piece that ends there" (fun () ->
        let step =
          Piecewise.v Nx.Ptree.tensor
            ~breaks:(vec [| 0.; 1.; 2. |])
            (Nx.create f64 [| 2; 1 |] [| 5.; 7. |])
        in
        equal (exact ())
          (vec [| 5.; 5.; 7. |])
          (Piecewise.eval step (vec [| 0.; 1.; 2. |])));
    test "NaN is NaN, even for a constant" (fun () ->
        let step =
          Piecewise.v Nx.Ptree.tensor
            ~breaks:(vec [| 0.; 1. |])
            (Nx.create f64 [| 1; 1 |] [| 5. |])
        in
        equal (exact ())
          (vec [| nan; 5. |])
          (Piecewise.eval step (vec [| nan; 0.5 |])));
    test "a point outside raises" (fun () ->
        raises_with "the point at [1] is 2.5, outside the domain [0, 2]"
          (fun () -> Piecewise.eval p (vec [| 1.; 2.5 |])));
    test "an infinite point raises" (fun () ->
        raises_with "outside the domain" (fun () ->
            Piecewise.eval p (vec [| neg_infinity |])));
    test "hold takes the nearest end" (fun () ->
        equal (exact ())
          (vec [| 0.; 3. |])
          (Piecewise.eval (Piecewise.extend `Hold p) (vec [| -4.; 9. |])));
    test "polynomial continues the end pieces" (fun () ->
        equal (exact ())
          (vec [| -2.; 4. |])
          (Piecewise.eval (Piecewise.extend `Polynomial p) (vec [| -1.; 3. |])));
    test "zero points give zero values" (fun () ->
        equal (exact ()) (vec [||]) (Piecewise.eval p (vec [||])));
    test "points keep their shape and values add theirs" (fun () ->
        let q =
          Piecewise.linear
            (vec [| 0.; 1. |])
            (Nx.create f64 [| 2; 3 |] [| 0.; 1.; 2.; 1.; 1.; 1. |])
        in
        equal (array int) [| 2; 2; 3 |]
          (Nx.shape (Piecewise.eval q (Nx.zeros f64 [| 2; 2 |]))));
    test "eval_at is eval at its piece's coordinate" (fun () ->
        let s = Piecewise.cubic `Natural knots samples in
        equal (close ())
          (Piecewise.eval s (vec [| -0.7; 2.45 |]))
          (Piecewise.eval_at s
             (Nx.create Nx.int64 [| 2 |] [| 0L; 5L |])
             (vec [| 0.; 0. |])));
    test "eval_at rejects an index that is not a piece" (fun () ->
        raises_with "the index at [0] is 2, not one of the 2 pieces" (fun () ->
            Piecewise.eval_at p
              (Nx.create Nx.int64 [| 1 |] [| 2L |])
              (vec [| 0. |])));
  ]

(* Transformations *)

let at_points y =
  Nx.sum (Nx.sin (Piecewise.eval (Piecewise.cubic `Not_a_knot knots y) points))

let transformation_tests =
  [
    test "grad in the samples is the finite difference" (fun () ->
        let v = Nx.cos (Nx.mul_s samples 3.) in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 at_points samples v))
          (scalar (Oracle.dot (Rune.grad' at_points samples) v)));
    test "grad in the knots is the finite difference" (fun () ->
        let f x =
          Nx.sum
            (Piecewise.eval
               (Piecewise.cubic `Natural x samples)
               (vec [| 0.2; 1.0 |]))
        in
        let v = Nx.sin knots in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 f knots v))
          (scalar (Oracle.dot (Rune.grad' f knots) v)));
    test "compiled equals eager, the spline solve included" (fun () ->
        equal (close ()) (at_points samples) (Rune.jit' at_points samples));
    test "a series is a compiled function's argument" (fun () ->
        let p = Piecewise.cubic `Natural knots samples in
        let s = Piecewise.ptree Nx.Ptree.tensor in
        let f =
          Rune.jit Nx.Ptree.(s @-> tensor @-> returns tensor) Piecewise.eval
        in
        equal (close ()) (Piecewise.eval p points) (f p points));
    test "vmap fits each lane's samples" (fun () ->
        let ys = Nx.stack [ samples; Nx.cos samples; Nx.neg samples ] in
        let one y = Piecewise.eval (Piecewise.cubic `Natural knots y) points in
        equal (close ())
          (Nx.stack (List.init 3 (fun i -> one (Nx.get [ i ] ys))))
          (Rune.vmap' one ys));
    test "float32 evaluates within float32's rounding" (fun () ->
        let c32 = Nx.cast Nx.float32 in
        let p = Piecewise.cubic `Natural (c32 knots) (c32 samples) in
        equal
          (Oracle.tensor ~rel:1e-5 ~abs:1e-6 ())
          (c32 (vec G.cubic_natural))
          (Piecewise.eval p (c32 points)));
  ]

let error_tests =
  [
    test "knots that are not increasing raise" (fun () ->
        raises_with "the knots are not strictly increasing at [2]: 1 after 1"
          (fun () ->
            Piecewise.linear (vec [| 0.; 1.; 1. |]) (vec [| 0.; 1.; 2. |])));
    test "one knot raises" (fun () ->
        raises_with "at least two elements" (fun () ->
            Piecewise.linear (vec [| 0. |]) (vec [| 1. |])));
    test "samples of another length raise" (fun () ->
        raises_with "the samples have shape [3] for 2 knots" (fun () ->
            Piecewise.steffen (vec [| 0.; 1. |]) (vec [| 0.; 1.; 2. |])));
    test "clamped slopes of another shape raise" (fun () ->
        raises_with "the clamped slopes have shapes [2] and []" (fun () ->
            Piecewise.cubic
              (`Clamped (vec [| 1.; 1. |], scalar 0.))
              knots samples));
    test "v rejects a leaf with another number of pieces" (fun () ->
        raises_with "shape [3,2]" (fun () ->
            Piecewise.v Nx.Ptree.tensor
              ~breaks:(vec [| 0.; 1.; 2. |])
              (Nx.zeros f64 [| 3; 2 |])));
    test "v rejects a leaf that is not a float" (fun () ->
        raises_with "a int32 leaf" (fun () ->
            Piecewise.v Nx.Ptree.tensor
              ~breaks:(vec [| 0.; 1. |])
              (Nx.zeros Nx.int32 [| 1; 2 |])));
  ]

let structure_tests =
  [
    test "a series visits its extension, breaks and coefficients" (fun () ->
        let p = Piecewise.linear (vec [| 0.; 1. |]) (vec [| 0.; 1. |]) in
        let visits = Nx.Ptree.visits (Piecewise.ptree Nx.Ptree.tensor) p in
        equal (list string)
          [
            "extension: case \"bounded\"";
            "breaks: a leaf";
            "coefficients: a leaf";
          ]
          (List.map (Format.asprintf "%a" Nx.Ptree.pp_visit) visits));
  ]

let () =
  exit
    (run "Jera.Piecewise"
       [
         group "goldens" golden_tests;
         group "laws" law_tests;
         group "fits" fit_tests;
         group "domain" domain_tests;
         group "transformations" transformation_tests;
         group "errors" error_tests;
         group "structure" structure_tests;
       ])
