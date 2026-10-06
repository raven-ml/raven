(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Grid: tensor products of piecewise series. The trusted side is scipy's
   regular-grid interpolant and its splines applied along each axis in turn, the
   values a grid interpolant passes through, and rune's derivative of
   evaluation. *)

open Windtrap
open Jera
module G = Golden_grid

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let close ?(rel = 1e-13) () = Oracle.tensor ~rel ~abs:1e-14 ()
let axes = [ vec G.axis0; vec G.axis1 ]

let n0 = Array.length G.axis0
and n1 = Array.length G.axis1

let values = Nx.create f64 [| n0; n1 |] G.values
let points = Nx.create f64 [| Array.length G.points / 2; 2 |] G.points
let raises_with sub f = raises_match (Exn.invalid_arg ~substring:sub) f

let golden_tests =
  [
    test "linear is scipy's regular-grid interpolant" (fun () ->
        equal (close ()) (vec G.linear)
          (Grid.eval (Grid.linear ~axes values) points));
    cases
      ~name:(fun (n, _, _, _, _) -> n)
      "cubic is scipy's spline along each axis"
      [
        ( "natural",
          `Natural,
          G.cubic_natural,
          G.cubic_natural_d0,
          G.cubic_natural_i0 );
        ( "not-a-knot",
          `Not_a_knot,
          G.cubic_not_a_knot,
          G.cubic_not_a_knot_d0,
          G.cubic_not_a_knot_i0 );
      ]
      (fun (_, ends, v, d0, i0) ->
        let g = Grid.cubic ends ~axes values in
        equal ~msg:"values" (close ()) (vec v) (Grid.eval g points);
        equal ~msg:"derivative along 0" (close ()) (vec d0)
          (Grid.eval (Grid.derivative ~axis:0 g) points);
        equal ~msg:"integral along 0" (close ()) (vec i0)
          (Grid.eval (Grid.integral ~axis:0 g) points));
  ]

(* The grid's own points, of shape [n0; n1; 2]. *)
let nodes =
  let g0 =
    Nx.broadcast_to [| n0; n1 |] (Nx.reshape [| n0; 1 |] (vec G.axis0))
  in
  let g1 =
    Nx.broadcast_to [| n0; n1 |] (Nx.reshape [| 1; n1 |] (vec G.axis1))
  in
  Nx.stack ~axis:(-1) [ g0; g1 ]

let law_tests =
  [
    test "every interpolant passes through its values" (fun () ->
        List.iter
          (fun (name, g) ->
            equal ~msg:name
              (Oracle.tensor ~rel:1e-13 ~abs:1e-14 ())
              values (Grid.eval g nodes))
          [
            ("linear", Grid.linear ~axes values);
            ("natural", Grid.cubic `Natural ~axes values);
            ("not-a-knot", Grid.cubic `Not_a_knot ~axes values);
          ]);
    test "linear reproduces a multilinear function on three axes" (fun () ->
        let a =
          [ vec [| 0.; 1.; 3. |]; vec [| -1.; 2. |]; vec [| 0.; 0.5; 1.; 4. |] ]
        in
        let f x y z = 1. +. (2. *. x) -. (y *. z) +. (0.5 *. x *. y *. z) in
        let v =
          Nx.init f64 [| 3; 2; 4 |] (fun i ->
              f
                (Nx.item [ i.(0) ] (List.nth a 0))
                (Nx.item [ i.(1) ] (List.nth a 1))
                (Nx.item [ i.(2) ] (List.nth a 2)))
        in
        let p = [| 0.3; -0.5; 2.5; 2.9; 1.9; 0.1 |] in
        let pts = Nx.create f64 [| 2; 3 |] p in
        equal (close ())
          (vec [| f p.(0) p.(1) p.(2); f p.(3) p.(4) p.(5) |])
          (Grid.eval (Grid.linear ~axes:a v) pts));
    test "eval of a derivative is the derivative of eval" (fun () ->
        let g = Grid.cubic `Not_a_knot ~axes values in
        let x = Nx.slice [ Nx.R (1, 4) ] points in
        let dx = Nx.create f64 [| 3; 2 |] [| 0.; 1.; 0.; 1.; 0.; 1. |] in
        equal (close ~rel:1e-12 ())
          (Grid.eval (Grid.derivative ~axis:1 g) x)
          (snd (Rune.jvp' (Grid.eval g) x dx)));
    test "a fit converges on a smooth function" (fun () ->
        (* e^x cos y at points whose last axis is (x, y). *)
        let f p =
          let coordinate k =
            Nx.slice [ Nx.I k ] (Nx.moveaxis (Nx.ndim p - 1) 0 p)
          in
          Nx.mul (Nx.exp (coordinate 0)) (Nx.cos (coordinate 1))
        in
        let g =
          Grid.chebyshev ~degree:14 ~pieces:2 f
            ~lo:(vec [| 0.; -1. |])
            ~hi:(vec [| 1.; 1. |])
        in
        let x = Nx.create f64 [| 3; 2 |] [| 0.; -1.; 0.37; 0.2; 1.; 1. |] in
        equal (Oracle.tensor ~rel:1e-13 ~abs:1e-14 ()) (f x) (Grid.eval g x));
    test "a fit's values keep their own axes" (fun () ->
        (* (x + y, x y) at each point: a value of shape [2]. *)
        let f p =
          let c k = Nx.slice [ Nx.I k ] (Nx.moveaxis (Nx.ndim p - 1) 0 p) in
          Nx.stack ~axis:(-1) [ Nx.add (c 0) (c 1); Nx.mul (c 0) (c 1) ]
        in
        let g =
          Grid.chebyshev ~degree:2 ~pieces:1 f
            ~lo:(vec [| 0.; -1. |])
            ~hi:(vec [| 1.; 1. |])
        in
        let x = Nx.create f64 [| 2; 2 |] [| 0.25; 0.5; 1.; -1. |] in
        equal
          (Oracle.tensor ~rel:1e-14 ~abs:1e-15 ())
          (Nx.create f64 [| 2; 2 |] [| 0.75; 0.125; 0.; -1. |])
          (Grid.eval g x));
    test "points keep their leading shape" (fun () ->
        let g = Grid.linear ~axes values in
        equal (array int) [| 2; 3 |]
          (Nx.shape (Grid.eval g (Nx.full f64 [| 2; 3; 2 |] 0.5))));
  ]

let domain_tests =
  let g = Grid.linear ~axes values in
  [
    test "a point outside raises" (fun () ->
        raises_with "outside the domain" (fun () ->
            Grid.eval g (Nx.create f64 [| 1; 2 |] [| 2.5; 0. |])));
    test "a NaN coordinate is NaN" (fun () ->
        equal (Oracle.tensor ()) (vec [| nan |])
          (Grid.eval g (Nx.create f64 [| 1; 2 |] [| 0.5; nan |])));
    test "points of another dimension raise" (fun () ->
        raises_with "points of shape [4,3] for a grid of 2 axes" (fun () ->
            Grid.eval g (Nx.zeros f64 [| 4; 3 |])));
    test "an axis that does not increase raises" (fun () ->
        raises_with "the knots of axis 1 are not strictly increasing" (fun () ->
            Grid.linear ~axes:[ vec G.axis0; vec [| 0.; 2.; 1.; 3. |] ] values));
    test "values that do not match the axes raise" (fun () ->
        raises_with "axis 0 has 5 knots and the values 4" (fun () ->
            Grid.linear ~axes (Nx.zeros f64 [| 4; 4 |])));
    test "a derivative along a missing axis raises" (fun () ->
        raises_with "axis = 2 is not in [0, 2)" (fun () ->
            Grid.derivative ~axis:2 g));
  ]

let structure_tests =
  [
    test "a grid visits its axes' breaks and its coefficients" (fun () ->
        let g = Grid.linear ~axes values in
        equal (list string)
          [
            "breaks: length 2";
            "breaks.0: a leaf";
            "breaks.1: a leaf";
            "coefficients: a leaf";
          ]
          (List.map
             (Format.asprintf "%a" Nx.Ptree.pp_visit)
             (Nx.Ptree.visits (Grid.ptree f64) g)));
  ]

let transformation_tests =
  let loss v = Nx.sum (Grid.eval (Grid.cubic `Natural ~axes v) points) in
  [
    test "grad in the values is the finite difference" (fun () ->
        let dv = Nx.sin values in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 loss values dv))
          (Nx.scalar f64 (Oracle.dot (Rune.grad' loss values) dv)));
    test "compiled equals eager" (fun () ->
        equal (close ()) (loss values) (Rune.jit' loss values));
    test "a grid is a compiled function's argument" (fun () ->
        let g = Grid.linear ~axes values in
        let f =
          Rune.jit
            Nx.Ptree.(Grid.ptree f64 @-> tensor @-> returns tensor)
            Grid.eval
        in
        equal (close ()) (Grid.eval g points) (f g points));
  ]

let () =
  exit
    (run "Jera.Grid"
       [
         group "goldens" golden_tests;
         group "laws" law_tests;
         group "domain" domain_tests;
         group "transformations" transformation_tests;
         group "structure" structure_tests;
       ])
