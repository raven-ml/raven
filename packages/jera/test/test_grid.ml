(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Grid: tensor products of piecewise series. The trusted side is
   Piecewise's interpolants applied along each axis in turn, functions a grid
   interpolant reproduces, and rune's derivative of evaluation. *)

open Windtrap
open Jera

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()
let close () = Oracle.tensor ~rel:1e-13 ~abs:1e-14 ()
let raises_in fn f = raises_match (Exn.invalid_arg ~sub:fn) f

(* A 5 × 4 grid of values of e^−x sin(y + 1) + x y / 5, and points in it. *)
let axes = [ vec [| 0.; 0.3; 0.9; 1.4; 2. |]; vec [| -1.; -0.2; 0.5; 1.5 |] ]

let values =
  Nx.init f64 [| 5; 4 |] (fun i ->
      let x = Nx.item [ i.(0) ] (List.nth axes 0)
      and y = Nx.item [ i.(1) ] (List.nth axes 1) in
      (Float.exp (-.x) *. Float.sin (y +. 1.)) +. (0.2 *. x *. y))

let points =
  Nx.create f64 [| 6; 2 |]
    [| 0.; -1.; 0.1; 0.; 0.9; 0.5; 1.7; 1.2; 2.; 1.5; 1.2; -0.7 |]

(* The grid's own points, of shape [5; 4; 2]. *)
let nodes =
  let along k shape =
    Nx.broadcast_to [| 5; 4 |] (Nx.reshape shape (List.nth axes k))
  in
  Nx.stack ~axis:(-1) [ along 0 [| 5; 1 |]; along 1 [| 1; 4 |] ]

(* Each point's coordinate [k], of points of shape [q @ [d]]. *)
let coord k p = Nx.get [ k ] (Nx.moveaxis (Nx.ndim p - 1) 0 p)

(* The interpolants, as Grid's and as Piecewise's along one axis. *)
let interpolants =
  [
    ("linear", (Grid.linear, Piecewise.linear));
    ("natural", (Grid.cubic `Natural, Piecewise.cubic `Natural));
    ("not-a-knot", (Grid.cubic `Not_a_knot, Piecewise.cubic `Not_a_knot));
  ]

(* Piecewise's interpolant along axis 1 of each row of [v], then along axis 0
   through the rows' values at each point. *)
let along_each_axis interp a0 a1 v p =
  let rows = Piecewise.eval (interp a1 (Nx.transpose v)) (coord 1 p) in
  List.init (Nx.dim 0 p) (fun j ->
      Piecewise.eval (interp a0 (Nx.get [ j ] rows)) (Nx.get [ j; 0 ] p))
  |> Nx.stack

(* Two axes of 2 to 6 knots with gaps in [0.1, 1], values in [−2, 2], and six
   points in the box. *)
let grid =
  Gen.(
    let knots =
      let* n = frequency [ (1, constant 2); (3, int_range 3 6) ] in
      let+ gaps = array ~size:(constant (n - 1)) (float_range 0.1 1.) in
      let x = Array.make n 0. in
      Array.iteri (fun i g -> x.(i + 1) <- x.(i) +. g) gaps;
      x
    in
    let* a0, a1 = pair knots knots in
    let n0 = Array.length a0 and n1 = Array.length a1 in
    let* v = array ~size:(constant (n0 * n1)) (float_range (-2.) 2.) in
    let+ p =
      array ~size:(constant 6)
        (pair (float_range 0. a0.(n0 - 1)) (float_range 0. a1.(n1 - 1)))
    in
    (a0, a1, v, p))
  |> Gen.with_pp (fun ppf (a0, a1, v, p) ->
      let floats =
        Format.pp_print_array ~pp_sep:Format.pp_print_space
          Format.pp_print_float
      in
      Format.fprintf ppf "@[axes [%a] [%a]@ values [%a]@ %d points@]" floats a0
        floats a1 floats v (Array.length p))

let tensors (a0, a1, v, p) =
  let flat =
    Array.concat (Array.to_list (Array.map (fun (x, y) -> [| x; y |]) p))
  in
  ( vec a0,
    vec a1,
    Nx.create f64 [| Array.length a0; Array.length a1 |] v,
    Nx.create f64 [| Array.length p; 2 |] flat )

(* The degrees of [g] along its two axes, from its coefficients' shape [[P₀; P₁;
   D₀ + 1; D₁ + 1]]. *)
let degrees g =
  match
    Nx.Ptree.fold (Grid.ptree f64) (fun _ t acc -> Nx.shape t :: acc) g []
  with
  | shape :: _ -> (shape.(2) - 1, shape.(3) - 1)
  | [] -> fail "a grid without coefficients"

(* Interpolants *)

let interpolant_tests =
  [
    prop "linear and cubic are Piecewise's along each axis in turn"
      Gen.(pair (of_list interpolants) grid)
      (fun ((name, (on_grid, on_axis)), ((a0, a1, _, _) as g)) ->
        cover name true;
        cover "an axis of two knots" (Array.length a0 = 2 || Array.length a1 = 2);
        (* Each axis's spline system has condition below 30 for gaps in [0.1,
           1], and the two orders round apart by that times the knots' count of
           ε of the values, up to 2. *)
        let n = Array.length a0 + Array.length a1 in
        let a0, a1, v', p = tensors g in
        let bound = 1e3 *. float_of_int n *. epsilon_float *. 2. in
        equal
          (Oracle.tensor ~abs:bound ())
          (along_each_axis on_axis a0 a1 v' p)
          (Grid.eval (on_grid ~axes:[ a0; a1 ] v') p));
    cases "a grid of one axis is Piecewise's series" ~id:fst interpolants
      (fun (_, (on_grid, on_axis)) ->
        let x = vec [| 0.; 0.4; 1.; 2.5 |] and y = vec [| 1.; -1.; 0.5; 2. |] in
        let t = Nx.linspace f64 0. 2.5 11 in
        equal (close ())
          (Piecewise.eval (on_axis x y) t)
          (Grid.eval (on_grid ~axes:[ x ] y) (Nx.reshape [| 11; 1 |] t)));
    cases "every interpolant passes through its values" ~id:fst interpolants
      (fun (_, (on_grid, _)) ->
        equal (close ()) values (Grid.eval (on_grid ~axes values) nodes));
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
        equal (close ())
          (vec [| f p.(0) p.(1) p.(2); f p.(3) p.(4) p.(5) |])
          (Grid.eval (Grid.linear ~axes:a v) (Nx.create f64 [| 2; 3 |] p)));
    test "no axis raises" (fun () ->
        raises_in "Jera.Grid.linear" (fun () -> Grid.linear ~axes:[] values));
    test "an axis that does not increase raises" (fun () ->
        raises_in "Jera.Grid.linear" (fun () ->
            Grid.linear
              ~axes:[ List.nth axes 0; vec [| 0.; 2.; 1.; 3. |] ]
              values));
    test "values that do not match the axes raise" (fun () ->
        raises_in "Jera.Grid.linear" (fun () ->
            Grid.linear ~axes (Nx.zeros f64 [| 4; 4 |])));
  ]

let cubic_tests =
  let loss v = Nx.sum (Grid.eval (Grid.cubic `Natural ~axes v) points) in
  [
    test "grad in the values is the central difference" (fun () ->
        (* eps 1e-6: truncation O(eps²), rounding ε/eps ≈ 2e-10. *)
        let dv = Nx.sin values in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 loss values dv))
          (scalar (Oracle.dot (Rune.grad' loss values) dv)));
    test "grad in an axis's knots is the central difference" (fun () ->
        let along y =
          Nx.sum
            (Grid.eval
               (Grid.cubic `Natural ~axes:[ List.nth axes 0; y ] values)
               points)
        in
        let y = List.nth axes 1 in
        (* The interior knots move, so the points stay in the domain. *)
        let dy = vec [| 0.; 1.; -1.; 0. |] in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 along y dy))
          (scalar (Oracle.dot (Rune.grad' along y) dy)));
    test "compiled equals eager" (fun () ->
        Compiled.here ();
        equal (close ()) (loss values) (Rune.jit' loss values));
  ]

(* Fits *)

let chebyshev ?(degree = 3) ?(pieces = 2) ?(lo = vec [| 0.; -1. |]) f =
  Grid.chebyshev ~degree ~pieces f ~lo ~hi:(vec [| 1.; 1. |])

let chebyshev_tests =
  let sum p = Nx.sum ~axes:[ -1 ] p in
  [
    test "converges on a smooth function" (fun () ->
        (* e^x cos y by degree 14 on pieces of width 1/2: truncation far below
           rounding. *)
        let f p = Nx.mul (Nx.exp (coord 0 p)) (Nx.cos (coord 1 p)) in
        let g = chebyshev ~degree:14 f in
        let x = Nx.create f64 [| 3; 2 |] [| 0.; -1.; 0.37; 0.2; 1.; 1. |] in
        equal (close ()) (f x) (Grid.eval g x));
    test "a fit's values keep their own axes" (fun () ->
        (* (x + y, x y) at each point: a value of shape [2]. *)
        let f p =
          Nx.stack ~axis:(-1)
            [ Nx.add (coord 0 p) (coord 1 p); Nx.mul (coord 0 p) (coord 1 p) ]
        in
        let g = chebyshev ~degree:2 ~pieces:1 f in
        let x = Nx.create f64 [| 2; 2 |] [| 0.25; 0.5; 1.; -1. |] in
        equal
          (Oracle.tensor ~rel:1e-14 ~abs:1e-15 ())
          (Nx.create f64 [| 2; 2 |] [| 0.75; 0.125; 0.; -1. |])
          (Grid.eval g x));
    test "a negative degree raises" (fun () ->
        raises_in "Jera.Grid.chebyshev" (fun () -> chebyshev ~degree:(-1) sum));
    test "no piece raises" (fun () ->
        raises_in "Jera.Grid.chebyshev" (fun () -> chebyshev ~pieces:0 sum));
    cases "corners not of one shape [d] with d ≥ 1 raise" ~id:fst
      [
        ("[3] and [2]", (vec [| 0.; 0.; 0. |], vec [| 1.; 1. |]));
        ("[0]", (vec [||], vec [||]));
      ]
      (fun (_, (lo, hi)) ->
        raises_in "Jera.Grid.chebyshev" (fun () ->
            Grid.chebyshev ~degree:3 ~pieces:2 sum ~lo ~hi));
    test "a result without the points' shape raises" (fun () ->
        raises_in "Jera.Grid.chebyshev" (fun () -> chebyshev Nx.sum));
  ]

(* Evaluation *)

let eval_tests =
  let g = Grid.linear ~axes values in
  [
    test "a point outside raises" (fun () ->
        raises_in "Jera.Grid.eval" (fun () ->
            Grid.eval g (Nx.create f64 [| 1; 2 |] [| 2.5; 0. |])));
    test "a NaN coordinate is NaN" (fun () ->
        equal (exact ()) (vec [| nan |])
          (Grid.eval g (Nx.create f64 [| 1; 2 |] [| 0.5; nan |])));
    test "a break lies in the piece that ends there, the first in the first"
      (fun () ->
        (* Along axis 0 the values rise by 2 on [0, 1] and 1 on [1, 2]. *)
        let x = vec [| 0.; 1.; 2. |] in
        let v =
          Nx.broadcast_to [| 3; 2 |]
            (Nx.reshape [| 3; 1 |] (vec [| 0.; 2.; 3. |]))
        in
        let slope =
          Grid.derivative ~axis:0 (Grid.linear ~axes:[ x; vec [| 0.; 1. |] ] v)
        in
        let at = Nx.create f64 [| 3; 2 |] [| 0.; 0.5; 1.; 0.5; 2.; 0.5 |] in
        equal (exact ()) (vec [| 2.; 2.; 1. |]) (Grid.eval slope at));
    test "points of another dimension raise" (fun () ->
        raises_in "Jera.Grid.eval" (fun () ->
            Grid.eval g (Nx.zeros f64 [| 4; 3 |])));
    test "zero points give zero values" (fun () ->
        equal (exact ()) (vec [||]) (Grid.eval g (Nx.zeros f64 [| 0; 2 |])));
    test "points keep their leading shape" (fun () ->
        (* Every point is the first node, whose value is 0. *)
        let p = Nx.broadcast_to [| 2; 3; 2 |] (vec [| 0.; -1. |]) in
        equal (exact ()) (Nx.zeros f64 [| 2; 3 |]) (Grid.eval g p));
  ]

(* Calculus *)

let along_axis = Gen.(pair (of_list [ 0; 1 ]) grid)

let derivative_tests =
  let g = Grid.cubic `Not_a_knot ~axes values in
  [
    prop "eval of the derivative is the derivative of eval" along_axis
      (fun (axis, g') ->
        cover "axis 0" (axis = 0);
        cover "axis 1" (axis = 1);
        let a0, a1, v, p = tensors g' in
        let g = Grid.cubic `Not_a_knot ~axes:[ a0; a1 ] v in
        let dp =
          Nx.broadcast_to (Nx.shape p)
            (if axis = 0 then vec [| 1.; 0. |] else vec [| 0.; 1. |])
        in
        (* Values up to 2 over gaps down to 0.1 give a cubic's derivative
           coefficients up to about 360, and each side rounds in about 16 terms
           of them: 16 · 360 ε ≈ 1.3e-12, below 1e-11. *)
        equal
          (Oracle.tensor ~abs:1e-11 ())
          (snd (Rune.jvp' (Grid.eval g) p dp))
          (Grid.eval (Grid.derivative ~axis g) p));
    test "has one degree less along its axis only" (fun () ->
        equal (pair int int) (2, 3) (degrees (Grid.derivative ~axis:0 g)));
    test "an axis that is not the grid's raises" (fun () ->
        raises_in "Jera.Grid.derivative" (fun () -> Grid.derivative ~axis:2 g));
  ]

let integral_tests =
  let g = Grid.cubic `Natural ~axes values in
  [
    test "is zero at its axis's first break" (fun () ->
        let at axis p = Grid.eval (Grid.integral ~axis g) p in
        let first0 = Nx.create f64 [| 2; 2 |] [| 0.; -0.5; 0.; 1. |] in
        let first1 = Nx.create f64 [| 2; 2 |] [| 0.5; -1.; 1.7; -1. |] in
        (* The constant cancels the first piece's terms: a few ε of values up to
           2. *)
        let zero = Oracle.tensor ~abs:1e-15 () in
        equal ~msg:"axis 0" zero (Nx.zeros f64 [| 2 |]) (at 0 first0);
        equal ~msg:"axis 1" zero (Nx.zeros f64 [| 2 |]) (at 1 first1));
    prop "its derivative along the axis is the grid" along_axis
      (fun (axis, g') ->
        cover "axis 0" (axis = 0);
        cover "axis 1" (axis = 1);
        let a0, a1, v, p = tensors g' in
        let g = Grid.cubic `Natural ~axes:[ a0; a1 ] v in
        let back = Grid.derivative ~axis (Grid.integral ~axis g) in
        (* The integral's and the derivative's coefficient maps are inverse;
           their rounding is a few ε of coefficients up to about 40. *)
        equal (Oracle.tensor ~abs:1e-12 ()) (Grid.eval g p) (Grid.eval back p));
    test "has one degree more along its axis only" (fun () ->
        equal (pair int int) (3, 4) (degrees (Grid.integral ~axis:1 g)));
    test "an axis that is not the grid's raises" (fun () ->
        raises_in "Jera.Grid.integral" (fun () -> Grid.integral ~axis:(-1) g));
  ]

(* Access *)

let ptree_tests =
  [
    test "visits the axes' breaks and the coefficients" (fun () ->
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
    test "a grid is a compiled function's argument" (fun () ->
        Compiled.here ();
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
         group "Interpolants" (interpolant_tests @ cubic_tests);
         group "chebyshev" chebyshev_tests;
         group "eval" eval_tests;
         group "derivative" derivative_tests;
         group "integral" integral_tests;
         group "ptree" ptree_tests;
       ])
