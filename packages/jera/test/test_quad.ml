(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Quad's rules, formulas and solves. The trusted side is mpmath's
   Gauss–Legendre rules, scipy's Gauss–Kronrod tables, the rule's sum in OCaml
   floats, closed-form integrals and their derivatives, and central
   differences. *)

open Windtrap
open Jera

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()

(* The golden is mpmath's 50-digit value rounded once, and the rule's
   double-word arithmetic rounds once: two ulps allow both. *)
let two_ulps () = Oracle.tensor ~rel:4e-16 ~abs:1e-16 ()
let unit_range x = Quad.Range.v (Nx.zeros_like x) (Nx.ones_like x)

(* Each lane's coordinate [k] of points [x] of shape [... @ [d]]. *)
let coord k x = Nx.get [ k ] (Nx.moveaxis (Nx.ndim x - 1) 0 x)
let product x = Nx.prod ~axes:[ Nx.ndim x - 1 ] x
let sum_coords x = Nx.sum ~axes:[ Nx.ndim x - 1 ] x

let unit_box lanes d =
  Quad.Box.v
    (Nx.zeros f64 (Array.append lanes [| d |]))
    (Nx.ones f64 (Array.append lanes [| d |]))

(* [f] with the shapes of the points it receives added to [seen]. *)
let recording seen f x =
  seen := Nx.shape x :: !seen;
  f x

(* ∫₀¹ e^(θx) dx = (e^θ − 1) / θ, its θ-derivative, and the derivative of its
   square, ∫ e^(θ(x + y)) over [0, 1]². *)
let exp_integral t = (Float.exp t -. 1.) /. t
let exp_integral' t = ((t *. Float.exp t) -. Float.exp t +. 1.) /. (t *. t)
let squared' t = 2. *. exp_integral t *. exp_integral' t

(* Rules *)

(* A polynomial of degree [d] with coefficients in [-1, 1] and a range inside
   [-3, 3]. *)
let polynomial d =
  Gen.(
    triple
      (array ~size:(constant (d + 1)) (float_range (-1.) 1.))
      (float_range (-3.) 3.) (float_range (-3.) 3.))

let horner c x = Array.fold_right (fun ci acc -> ci +. (x *. acc)) c 0.

let antiderivative c x =
  let a = Array.mapi (fun k ck -> ck /. float_of_int (k + 1)) c in
  x *. horner a x

(* The sum of the terms' magnitudes, which bounds the integral's rounding. *)
let magnitude c a b =
  let m = Float.max (Float.abs a) (Float.abs b) in
  Array.fold_left
    (fun (acc, p) ck -> (acc +. (Float.abs ck *. p), p *. m))
    (0., m) c
  |> fst

(* A rule exact to the polynomial's degree integrates it to its rounding: 64 ε
   of its magnitude, where 300 draws at n = 20, 50 and 100 measured at most 9.3
   ε. *)
let integrates_exactly rule (c, a, b) =
  let integral =
    Quad.fixed rule
      (fun x -> Nx.map_item (horner c) x)
      (Quad.Range.v (scalar a) (scalar b))
  in
  let truth = antiderivative c b -. antiderivative c a in
  let bound = 64. *. epsilon_float *. (magnitude c a b +. 1.) in
  equal (float_rel ~rel:0. ~abs:bound) truth (Nx.item [] integral)

let gauss_tests =
  [
    cases "is mpmath's Gauss–Legendre rule within two ulps"
      ~id:(fun (n, _, _) -> Printf.sprintf "%d points" n)
      Golden_quad.
        [
          (1, gauss1_x, gauss1_w);
          (2, gauss2_x, gauss2_w);
          (3, gauss3_x, gauss3_w);
          (4, gauss4_x, gauss4_w);
          (5, gauss5_x, gauss5_w);
          (8, gauss8_x, gauss8_w);
          (10, gauss10_x, gauss10_w);
          (16, gauss16_x, gauss16_w);
          (20, gauss20_x, gauss20_w);
          (32, gauss32_x, gauss32_w);
          (64, gauss64_x, gauss64_w);
          (100, gauss100_x, gauss100_w);
        ]
      (fun (n, gx, gw) ->
        let x, w = Quad.Rule.nodes (Quad.Rule.gauss n) f64 in
        equal ~msg:"nodes" (two_ulps ()) (vec gx) x;
        equal ~msg:"weights" (two_ulps ()) (vec gw) w);
    prop "integrates polynomials of degree 2n − 1"
      Gen.(
        let* n =
          frequency [ (1, constant 1); (2, int_range 2 99); (1, constant 100) ]
        in
        let+ p = polynomial ((2 * n) - 1) in
        (n, p))
      (fun (n, p) ->
        cover "one point" (n = 1);
        cover "a hundred points" (n = 100);
        integrates_exactly (Quad.Rule.gauss n) p);
    test "n below 1 raises" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.Rule.gauss") (fun () ->
            Quad.Rule.gauss 0));
  ]

let kronrod_tests =
  [
    cases "is QUADPACK's rule as scipy tabulates it"
      ~id:(fun (n, _, _) -> Printf.sprintf "%d points" ((2 * n) + 1))
      Golden_quad.
        [ (7, kronrod7_x, kronrod7_w); (10, kronrod10_x, kronrod10_w) ]
      (fun (n, gx, gw) ->
        let x, w = Quad.Rule.nodes (Quad.Rule.kronrod n) f64 in
        equal ~msg:"nodes" (exact ()) (vec gx) x;
        equal ~msg:"weights" (exact ()) (vec gw) w);
    cases "holds gauss n's nodes at its odd positions"
      ~id:(Printf.sprintf "n = %d") [ 7; 10 ] (fun n ->
        let k, _ = Quad.Rule.nodes (Quad.Rule.kronrod n) f64 in
        let g, _ = Quad.Rule.nodes (Quad.Rule.gauss n) f64 in
        equal (two_ulps ()) g (Nx.slice [ Nx.Rs (1, (2 * n) + 1, 2) ] k));
    prop "integrates polynomials of degree 3n + 1"
      Gen.(
        let* n = of_list [ 7; 10 ] in
        let+ p = polynomial ((3 * n) + 1) in
        (n, p))
      (fun (n, p) ->
        cover "n = 7" (n = 7);
        cover "n = 10" (n = 10);
        integrates_exactly (Quad.Rule.kronrod n) p);
    cases "n other than 7 or 10 raises" ~id:(Printf.sprintf "n = %d")
      [ 6; 8; 9; 11 ] (fun n ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.Rule.kronrod") (fun () ->
            Quad.Rule.kronrod n));
  ]

let nodes_tests =
  [
    test "in another dtype are the float64 nodes rounded once" (fun () ->
        let rounded : type b. (float, b) Nx.dtype -> unit =
         fun dtype ->
          List.iter
            (fun r ->
              let x, w = Quad.Rule.nodes r f64 in
              let x', w' = Quad.Rule.nodes r dtype in
              equal ~msg:"nodes" (exact ()) (Nx.cast dtype x) x';
              equal ~msg:"weights" (exact ()) (Nx.cast dtype w) w')
            [
              Quad.Rule.gauss 7;
              Quad.Rule.gauss 100;
              (Quad.Rule.kronrod 10 :> [ `Formula ] Quad.Rule.t);
            ]
        in
        rounded Nx.float32;
        rounded Nx.float16;
        rounded Nx.bfloat16);
  ]

let nodes_at_tests =
  [
    test "at the host are the nodes" (fun () ->
        let rule = Quad.Rule.kronrod 7 in
        let x, w = Quad.Rule.nodes rule f64 in
        let xa, wa = Quad.Rule.nodes_at Nx.Host.on rule f64 in
        equal ~msg:"nodes" (exact ()) x xa;
        equal ~msg:"weights" (exact ()) w wa);
  ]

(* Ranges and boxes *)

let range_tests =
  let backwards integral = Nx.neg (integral (vec [| 0. |]) (vec [| 1. |])) in
  [
    test "the ends broadcast together" (fun () ->
        let i =
          Quad.fixed (Quad.Rule.gauss 3) Nx.ones_like
            (Quad.Range.v (scalar 0.) (vec [| 1.; 2.; 3. |]))
        in
        equal (two_ulps ()) (vec [| 1.; 2.; 3. |]) i);
    test "fixed over [1, 0] is minus fixed over [0, 1]" (fun () ->
        let integral a b =
          Quad.fixed (Quad.Rule.gauss 6) Nx.exp (Quad.Range.v a b)
        in
        equal (exact ()) (backwards integral)
          (integral (vec [| 1. |]) (vec [| 0. |])));
    test "adaptive over [1, 0] is minus adaptive over [0, 1]" (fun () ->
        (* Each meets rel 1e-10, so the two differ by at most twice that. *)
        let integral a b =
          Solution.get
            (Quad.adaptive (Quad.Rule.kronrod 7) ~tol:(Tol.rel 1e-10) ~budget:50
               Nx.sqrt (Quad.Range.v a b))
        in
        equal
          (Oracle.tensor ~rel:2e-10 ())
          (backwards integral)
          (integral (vec [| 1. |]) (vec [| 0. |])));
    test "tanh_sinh over [1, 0] is minus tanh_sinh over [0, 1]" (fun () ->
        (* Each meets rel 1e-12, so the two differ by at most twice that. *)
        let integral a b =
          Solution.get
            (Quad.tanh_sinh ~tol:(Tol.rel 1e-12) Nx.sqrt (Quad.Range.v a b))
        in
        equal
          (Oracle.tensor ~rel:2e-12 ())
          (backwards integral)
          (integral (vec [| 1. |]) (vec [| 0. |])));
    test "ends that do not broadcast raise" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.Range.v") (fun () ->
            Quad.Range.v (Nx.zeros f64 [| 2 |]) (Nx.ones f64 [| 3 |])));
  ]

let from_tests =
  [
    test "integrates e^-(x - a) to 1 at unit scale" (fun () ->
        (* 40 points resolve e^-x on the mapped half-line to 1e-10. *)
        let a = vec [| 0.; 2.5; -1. |] in
        let i =
          Quad.fixed (Quad.Rule.gauss 40)
            (fun x -> Nx.exp (Nx.neg (Nx.sub x a)))
            (Quad.Range.from a)
        in
        equal (Oracle.tensor ~rel:1e-10 ()) (Nx.ones_like a) i);
  ]

let line_tests =
  [
    test "integrates e^-(x - c)² to √π at unit scale" (fun () ->
        let c = vec [| 0.; 0.5 |] in
        let i =
          Quad.fixed (Quad.Rule.gauss 100)
            (fun x -> Nx.exp (Nx.neg (Nx.square (Nx.sub x c))))
            (Quad.Range.line c)
        in
        equal
          (Oracle.tensor ~rel:1e-10 ())
          (Nx.full f64 [| 2 |] (Float.sqrt Float.pi))
          i);
  ]

(* ∫ (1 + x + y) over a box, exact in the degree-7 rule's one box. *)
let plane box =
  Solution.get
    (Quad.cubature ~tol:(Tol.rel 1e-12) ~budget:4
       (fun x -> Nx.add_s (sum_coords x) 1.)
       box)

let box_tests =
  [
    test "corners broadcast to lanes @ [d]" (fun () ->
        (* One unit square shared by two lanes: 2 in each. *)
        let hi = Nx.ones f64 [| 2; 2 |] in
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (vec [| 2.; 2. |])
          (plane (Quad.Box.v (vec [| 0.; 0. |]) hi)));
    test "an axis with hi < lo flips the integral's sign" (fun () ->
        (* Over [0, −1] × [0, 1]: −(1 − 1/2 + 1/2) = −1. *)
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (scalar (-1.))
          (plane (Quad.Box.v (vec [| 0.; 0. |]) (vec [| -1.; 1. |]))));
    test "corners that do not broadcast raise" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.Box.v") (fun () ->
            Quad.Box.v (Nx.zeros f64 [| 2 |]) (Nx.ones f64 [| 3 |])));
    test "scalar corners raise" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.Box.v") (fun () ->
            Quad.Box.v (scalar 0.) (scalar 1.)));
  ]

(* Formulas *)

let fixed_tests =
  let theta = vec [| -1.; 0.5; 2. |] in
  (* ∫₀¹ e^(θ x) dx by 15 Kronrod points. *)
  let integral theta =
    Quad.fixed (Quad.Rule.kronrod 7)
      (fun x -> Nx.exp (Nx.mul x theta))
      (unit_range theta)
  in
  [
    prop
      "is the rule's sum (b − a)/2 Σ w f(mid + u (b − a)/2)"
      (* n points of Gauss, or 0 for Kronrod's 15, a coerced formula. A range of
         subnormal width once failed a purely relative bound. *)
      ~examples:[ (0, 0., -3.6e-310); (3, 0., -3.6e-310) ]
      Gen.(
        triple
          (frequency [ (2, int_range 1 20); (1, constant 0) ])
          (float_range (-3.) 3.) (float_range (-3.) 3.))
      (fun (n, a, b) ->
        let rule =
          if n = 0 then (Quad.Rule.kronrod 7 :> [ `Formula ] Quad.Rule.t)
          else Quad.Rule.gauss n
        in
        let u, w = Quad.Rule.nodes rule f64 in
        cover "a Kronrod rule" (Nx.numel u = 15);
        let mid = (a +. b) /. 2. and half = (b -. a) /. 2. in
        let terms =
          Array.map2
            (fun u w -> half *. w *. Float.exp (mid +. (half *. u)))
            (Nx.to_array u) (Nx.to_array w)
        in
        (* exp to an ulp or so, and at most 20 terms of one sign summed in
           another order: 32 ε of the sum, or subnormals' spacing for a range of
           nearly zero width. *)
        let sum = Array.fold_left ( +. ) 0. terms in
        equal
          (float_rel ~rel:(32. *. epsilon_float) ~abs:Float.min_float)
          sum
          (Nx.item []
             (Quad.fixed rule Nx.exp (Quad.Range.v (scalar a) (scalar b)))));
    test "f receives points of shape [m] @ shape" (fun () ->
        let seen = ref [] in
        ignore
          (Quad.fixed (Quad.Rule.gauss 5) (recording seen Nx.exp)
             (unit_range (Nx.zeros f64 [| 2; 3 |])));
        equal (list (array int)) [ [| 5; 2; 3 |] ] !seen);
    test "a zero-size range gives a zero-size integral" (fun () ->
        let i =
          Quad.fixed (Quad.Rule.gauss 3) Nx.exp
            (Quad.Range.v (vec [||]) (vec [||]))
        in
        equal (exact ()) (vec [||]) i);
    test "a NaN end gives a NaN integral in its element only" (fun () ->
        let i =
          Quad.fixed (Quad.Rule.gauss 3) Nx.ones_like
            (Quad.Range.v (vec [| 0.; nan |]) (vec [| 1.; 1. |]))
        in
        equal (two_ulps ()) (vec [| 1.; nan |]) i);
    test "computes in the range's dtype" (fun () ->
        (* e − 1 by 8 Gauss points, whose error is far below rounding: the
           nodes, the points, exp and 8 products and sums each round, 16 ε of
           the dtype in all. *)
        let in_dtype : type b. (float, b) Nx.dtype -> unit =
         fun dtype ->
          let i =
            Quad.fixed (Quad.Rule.gauss 8) Nx.exp
              (Quad.Range.v (Nx.scalar dtype 0.) (Nx.scalar dtype 1.))
          in
          let eps = (Nx.Dtype.float_format dtype).epsilon in
          equal ~msg:(Nx.Dtype.name dtype)
            (Oracle.tensor ~rel:(16. *. eps) ())
            (Nx.scalar dtype (Float.exp 1. -. 1.))
            i
        in
        in_dtype Nx.float32;
        in_dtype Nx.float16;
        in_dtype Nx.bfloat16);
    test "grad in a parameter is the central difference" (fun () ->
        (* eps 1e-6: truncation O(eps²), rounding ε/eps ≈ 2e-10. *)
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Oracle.central ~eps:1e-6 integral theta (Nx.ones_like theta))
          (Rune.grad' (fun t -> Nx.sum (integral t)) theta));
    test "grad in an end is the central difference" (fun () ->
        let upper b =
          Nx.sum
            (Quad.fixed (Quad.Rule.gauss 4) Nx.sin
               (Quad.Range.v (Nx.zeros_like b) b))
        in
        let b = vec [| 0.3; 1.7 |] and v = vec [| 1.; -2. |] in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 upper b v))
          (scalar (Oracle.dot (Rune.grad' upper b) v)));
    test "compiled equals eager" (fun () ->
        Compiled.here ();
        (* A compiled sum of 15 terms may associate otherwise: 15 ε. *)
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (integral theta) (Rune.jit' integral theta));
    test "vmap is each element's integral" (fun () ->
        equal
          (Oracle.tensor ~rel:1e-15 ())
          (integral theta)
          (Rune.vmap' integral theta));
    test "an integrand that changes the shape raises" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.fixed") (fun () ->
            Quad.fixed (Quad.Rule.gauss 3)
              (fun x -> Nx.sum ~axes:[ 1 ] x)
              (Quad.Range.v (vec [| 0.; 0. |]) (vec [| 1.; 1. |]))));
  ]

let cumulative_tests =
  let r = Quad.Rule.gauss 5 in
  [
    test "row i integrates from the first knot to knot i" (fun () ->
        (* 5 points on intervals up to 0.75 integrate cos to 1e-13. *)
        let knots = vec [| 0.; 0.5; 1.25; 2. |] in
        let c = Quad.cumulative r Nx.cos knots in
        equal (Oracle.tensor ~rel:1e-12 ~abs:1e-15 ()) (Nx.sin knots) c);
    test "row 0 is zero" (fun () ->
        let c = Quad.cumulative r Nx.exp (vec [| 1.; 2. |]) in
        equal (exact ()) (scalar 0.) (Nx.get [ 0 ] c));
    test "one knot is a zero integral" (fun () ->
        equal (exact ()) (vec [| 0. |])
          (Quad.cumulative r Nx.exp (vec [| 1. |])));
    test "lanes behind the knots are integrals of their own" (fun () ->
        let knots = Nx.create f64 [| 3; 2 |] [| 0.; 1.; 1.; 2.; 2.; 4. |] in
        let c = Quad.cumulative (Quad.Rule.gauss 12) Nx.cos knots in
        equal
          (Oracle.tensor ~rel:1e-14 ~abs:1e-15 ())
          (Nx.sub (Nx.sin knots) (Nx.sin (Nx.get [ 0 ] knots)))
          c);
    test "f receives points of shape [m; n − 1] @ shape" (fun () ->
        let seen = ref [] in
        ignore
          (Quad.cumulative r (recording seen Nx.exp) (Nx.zeros f64 [| 4; 2 |]));
        equal (list (array int)) [ [| 5; 3; 2 |] ] !seen);
    cases "knots without a knot or an axis raise" ~id:fst
      [ ("no knot", vec [||]); ("no axis", scalar 1.) ]
      (fun (_, knots) ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.cumulative") (fun () ->
            Quad.cumulative r Nx.exp knots));
    test "an integrand that changes the shape raises" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.cumulative") (fun () ->
            Quad.cumulative r
              (fun x -> Nx.sum ~axes:[ 1 ] x)
              (Nx.zeros f64 [| 2; 2 |])));
  ]

(* Solves *)

let k7 = Quad.Rule.kronrod 7

let adaptive ?(budget = 200) ?(tol = Tol.v ~rel:1e-10 ~abs:1e-12) f range =
  Quad.adaptive k7 ~tol ~budget f range

(* The tolerance the adaptive solves are asked for. On a smooth piece the
   estimate |K − G| exceeds the Kronrod sum's own error. *)
let at_tol () = Oracle.tensor ~rel:1e-10 ~abs:1e-12 ()

(* A Gaussian peak of width w at 0.3 on [0, 1]. *)
let peak w x = Nx.exp (Nx.neg (Nx.square (Nx.div (Nx.sub_s x 0.3) w)))

(* 0 below 0.3 and 1 above: no piece of positive width resolves the jump. *)
let jump x = Nx.cast f64 (Nx.greater_s x 0.3)

(* A lane's points have shape [m; k] @ lanes for some k. *)
let pieces m lanes =
  satisfies ~claim:"[m; k] @ lanes" (array int) (fun s ->
      Array.length s = 2 + Array.length lanes
      && s.(0) = m
      && Array.sub s 2 (Array.length lanes) = lanes)

let adaptive_tests =
  let theta = vec [| -2.; 0.5; 3. |] in
  let solve t =
    Solution.get (adaptive (fun x -> Nx.exp (Nx.mul x t)) (unit_range t))
  in
  [
    test "a smooth integral converges" (fun () ->
        equal (at_tol ()) (Nx.map_item exp_integral theta) (solve theta));
    cases "an endpoint singularity converges by bisection" ~id:fst
      [ ("√x", (Nx.sqrt, 2. /. 3.)); ("log x", (Nx.log, -1.)) ]
      (fun (_, (f, truth)) ->
        let s = adaptive ~budget:400 f (unit_range (vec [| 0. |])) in
        equal (at_tol ()) (vec [| truth |]) (Solution.get s));
    test "each element is its integral solved alone" (fun () ->
        let widths = [| 1.; 1e-1; 1e-2 |] in
        let solve w = adaptive (peak w) (unit_range w) in
        let all = solve (vec widths) in
        let alone read =
          Nx.concatenate ~axis:0
            (List.map
               (fun w -> read (solve (vec [| w |])))
               (Array.to_list widths))
        in
        equal ~msg:"best" (exact ()) (alone Solution.best) (Solution.best all);
        equal ~msg:"evaluations" (exact ())
          (alone Solution.evaluations)
          (Solution.evaluations all));
    test "a half-line converges" (fun () ->
        let a = vec [| 0.; 1. |] in
        let s =
          adaptive (fun x -> Nx.exp (Nx.neg (Nx.sub x a))) (Quad.Range.from a)
        in
        equal (at_tol ()) (Nx.ones_like a) (Solution.get s));
    test "a range of zero width integrates to 0" (fun () ->
        let s = adaptive Nx.exp (Quad.Range.v (vec [| 1. |]) (vec [| 1. |])) in
        equal (exact ()) (vec [| 0. |]) (Solution.get s));
    test "f receives points of shape [15; k] @ shape" (fun () ->
        let seen = ref [] in
        ignore
          (adaptive (recording seen Nx.sqrt)
             (unit_range (Nx.zeros f64 [| 2; 3 |])));
        List.iter (pieces 15 [| 2; 3 |]) !seen);
    test "budget pieces end an element Budget_spent" (fun () ->
        let s = adaptive ~budget:2 Nx.log (unit_range (vec [| 0. |])) in
        equal Oracle.status_t Budget_spent (Oracle.status s));
    test "evaluations count the search's pieces and the answer's 32" (fun () ->
        (* Budget 3: 2·3 − 1 pieces searched, then one chunk of 32 pieces, 15
           points each. *)
        let s = adaptive ~budget:3 Nx.log (unit_range (vec [| 0. |])) in
        equal Oracle.status_t Budget_spent (Oracle.status s);
        equal (exact ())
          (Nx.create Nx.int32 [| 1 |] [| Int32.of_int (15 * (5 + 32)) |])
          (Solution.evaluations s));
    test "a pole at a node is a non-finite sum" (fun () ->
        (* The 15-point rule's middle node on [0, 1] is 0.5, where 1 / (x − 0.5)
           is infinite. *)
        let s =
          adaptive
            (fun x -> Nx.recip (Nx.sub_s x 0.5))
            (unit_range (vec [| 0. |]))
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    test "a jump no piece resolves ends Stalled" (fun () ->
        (* Floats near 0.3 are 2^−54 apart, so the piece holding the jump holds
           none inside at level 54: 55 pieces, which evaluate 15 (2 · 55 − 1 +
           416) points with the answer's 13 chunks. *)
        let s =
          adaptive ~budget:400 ~tol:(Tol.rel 1e-30) jump
            (unit_range (vec [| 0. |]))
        in
        equal Oracle.status_t Stalled (Oracle.status s);
        equal (exact ())
          (Nx.create Nx.int32 [| 1 |] [| 7875l |])
          (Solution.evaluations s));
    test "a piece at level 62 ends Stalled" (fun () ->
        (* The worst piece of x^(−½) is [0, 2^−L], whose floats never run out:
           only the level stops it, at 63 pieces, which evaluate 15 (2 · 63 − 1
           + 96) points with the answer's 3 chunks. *)
        let s =
          adaptive ~budget:80 ~tol:(Tol.rel 1e-30) Nx.rsqrt
            (unit_range (vec [| 0. |]))
        in
        equal Oracle.status_t Stalled (Oracle.status s);
        equal (exact ())
          (Nx.create Nx.int32 [| 1 |] [| 3315l |])
          (Solution.evaluations s));
    test "grad in a parameter is the exact integral's" (fun () ->
        (* The derivative's integrand x e^(θx), on the partition chosen for
           e^(θx): within 10 tol. *)
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.map_item exp_integral' theta)
          (Rune.grad' (fun t -> Nx.sum (solve t)) theta));
    test "grad in an end is the integrand there" (fun () ->
        let b = vec [| 0.7; 2. |] in
        let integral b =
          Nx.sum
            (Solution.get (adaptive Nx.cos (Quad.Range.v (Nx.zeros_like b) b)))
        in
        equal (Oracle.tensor ~rel:1e-9 ()) (Nx.cos b) (Rune.grad' integral b));
    test "an element that did not converge has a zero derivative" (fun () ->
        (* The pole of p / (x − 1/4) is a node of the piece [0, ½]. *)
        let solve p =
          adaptive (fun x -> Nx.div p (Nx.sub_s x 0.25)) (unit_range p)
        in
        let p = vec [| 1. |] in
        equal Oracle.status_t Not_finite (Oracle.status (solve p));
        equal (exact ()) (vec [| 0. |])
          (Rune.grad' (fun p -> Nx.sum (Solution.best (solve p))) p));
    test "compiled equals eager" (fun () ->
        Compiled.here ();
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (solve theta) (Rune.jit' solve theta));
    test "compiled grad equals eager grad" (fun () ->
        Compiled.here ();
        let g = Rune.grad' (fun t -> Nx.sum (solve t)) in
        equal (Oracle.tensor ~rel:1e-13 ()) (g theta) (Rune.jit' g theta));
    test "vmap is each lane's solve" (fun () ->
        let theta = Nx.create f64 [| 2; 2 |] [| -2.; 0.5; 3.; 1. |] in
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (solve theta) (Rune.vmap' solve theta));
    test "a budget below 1 raises" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.adaptive") (fun () ->
            adaptive ~budget:0 Nx.exp (unit_range (vec [| 0. |]))));
    test "an integrand that changes the shape raises" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.adaptive") (fun () ->
            adaptive
              (fun x -> Nx.sum ~axes:[ 1 ] x)
              (unit_range (vec [| 0.; 0. |]))));
  ]

let tanh_sinh_tests =
  let ts = Quad.tanh_sinh ~tol:(Tol.v ~rel:1e-12 ~abs:1e-14) in
  (* The tolerance the solves are asked for: the last two levels' difference
     exceeds the final level's error, which falls quadratically. *)
  let ts_tol () = Oracle.tensor ~rel:1e-12 ~abs:1e-14 () in
  (* ∫₀¹ x^(a−1) log x dx = −1 / a². *)
  let log_moment a =
    Solution.get
      (ts
         (fun x -> Nx.mul (Nx.pow x (Nx.sub_s a 1.)) (Nx.log x))
         (unit_range a))
  in
  let a = vec [| 0.5; 1.; 2.5 |] in
  [
    test "endpoint singularities at 0 converge" (fun () ->
        let a = vec [| 0.3; 0.5; 1.; 2.5 |] in
        equal (ts_tol ())
          (Nx.map_item (fun a -> -1. /. (a *. a)) a)
          (log_moment a));
    test "1 / √(1 − x²) on [−1, 1] is π to the ends' resolution" (fun () ->
        (* The integrand computes each end's distance from x, so it misses the
           mass within an ulp of each end, 2√(2 · 2^−53) ≈ 3e-8. *)
        let r = Quad.Range.v (vec [| -1. |]) (vec [| 1. |]) in
        let f x = Nx.rsqrt (Nx.mul (Nx.add_s x 1.) (Nx.rsub_s 1. x)) in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (vec [| Float.pi |])
          (Solution.best (ts f r)));
    test "exp-sinh integrates a half-line" (fun () ->
        (* Lane 0 integrates e^-x, lane 1 1 / (1 + x²). *)
        let f x =
          let lane i = Nx.slice [ Nx.A; Nx.I i ] x in
          Nx.stack ~axis:1
            [
              Nx.exp (Nx.neg (lane 0));
              Nx.recip (Nx.add_s (Nx.square (lane 1)) 1.);
            ]
        in
        let s = ts f (Quad.Range.from (vec [| 0.; 0. |])) in
        equal (ts_tol ()) (vec [| 1.; Float.pi /. 2. |]) (Solution.get s));
    test "an end singularity converges on a half-line" (fun () ->
        (* ∫₀^∞ x^(−½) e^(−x) dx = Γ(½) = √π. *)
        let s =
          ts
            (fun x -> Nx.mul (Nx.rsqrt x) (Nx.exp (Nx.neg x)))
            (Quad.Range.from (vec [| 0. |]))
        in
        equal (ts_tol ()) (vec [| Float.sqrt Float.pi |]) (Solution.get s));
    test "sinh-sinh integrates the line" (fun () ->
        let c = vec [| 0.; 0.5 |] in
        let s =
          ts
            (fun x -> Nx.exp (Nx.neg (Nx.square (Nx.sub x c))))
            (Quad.Range.line c)
        in
        equal (ts_tol ())
          (Nx.full f64 [| 2 |] (Float.sqrt Float.pi))
          (Solution.get s));
    test "f receives points of shape [64] @ shape" (fun () ->
        let seen = ref [] in
        ignore
          (ts (recording seen Nx.sqrt) (unit_range (Nx.zeros f64 [| 2; 3 |])));
        List.iter (equal (array int) [| 64; 2; 3 |]) !seen);
    test "a divergent integral ends Stalled" (fun () ->
        let s = ts Nx.recip (unit_range (vec [| 0. |])) in
        equal Oracle.status_t Stalled (Oracle.status s));
    test "a pole at a node is a non-finite sum" (fun () ->
        (* t = 0 is the range's midpoint. *)
        let s =
          ts (fun x -> Nx.recip (Nx.sub_s x 0.5)) (unit_range (vec [| 0. |]))
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    test "evaluations count the search's chunks and the whole schedule"
      (fun () ->
        (* Lane 0 diverges and searches every chunk of 64 nodes, so it evaluates
           the schedule twice; lane 1 converges sooner and adds its search's
           chunks to one pass. *)
        let f x =
          let lane i = Nx.slice [ Nx.A; Nx.I i ] x in
          Nx.stack ~axis:1 [ Nx.recip (lane 0); Nx.sqrt (lane 1) ]
        in
        let s = ts f (unit_range (vec [| 0.; 0. |])) in
        equal (exact ())
          (Nx.create Nx.bool [| 2 |] [| false; true |])
          (Solution.ok s);
        let n = Nx.to_array (Solution.evaluations s) in
        let schedule = Int32.div n.(0) 2l in
        equal ~msg:"lane 0, in chunks" int32 0l (Int32.rem schedule 64l);
        equal ~msg:"lane 1's search, in chunks" int32 0l
          (Int32.rem (Int32.sub n.(1) schedule) 64l);
        less int32 ~than:n.(0) n.(1));
    test "grad in a is 2 / a³" (fun () ->
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.map_item (fun a -> 2. /. (a *. a *. a)) a)
          (Rune.grad' (fun a -> Nx.sum (log_moment a)) a));
    test "grad in an end is the integrand there" (fun () ->
        (* d/db ∫₀ᵇ x^(−½) dx = b^(−½). *)
        let b = vec [| 0.5; 2. |] in
        let integral b =
          Nx.sum (Solution.get (ts Nx.rsqrt (Quad.Range.v (Nx.zeros_like b) b)))
        in
        equal (Oracle.tensor ~rel:1e-9 ()) (Nx.rsqrt b) (Rune.grad' integral b));
    test "an element that did not converge has a zero derivative" (fun () ->
        (* x^(p − 1) on [0, 1]: integrable for p = 1, not for p = −0.5. *)
        let p = vec [| 1.; -0.5 |] in
        let solve p = ts (fun x -> Nx.pow x (Nx.sub_s p 1.)) (unit_range p) in
        equal (exact ())
          (Nx.create Nx.bool [| 2 |] [| true; false |])
          (Solution.ok (solve p));
        let g = Rune.grad' (fun p -> Nx.sum (Solution.best (solve p))) p in
        equal (exact ()) (scalar 0.) (Nx.get [ 1 ] g));
    test "compiled equals eager" (fun () ->
        Compiled.here ();
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (log_moment a) (Rune.jit' log_moment a));
    test "compiled grad equals eager grad" (fun () ->
        Compiled.here ();
        let g = Rune.grad' (fun a -> Nx.sum (log_moment a)) in
        equal (Oracle.tensor ~rel:1e-12 ()) (g a) (Rune.jit' g a));
    test "float32 converges to float32's tolerance" (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 0.5; 2. |] in
        let s =
          Quad.tanh_sinh ~tol:(Tol.rel 1e-5)
            (fun x -> Nx.mul (Nx.pow x (Nx.sub_s a 1.)) (Nx.log x))
            (unit_range a)
        in
        equal
          (Oracle.tensor ~rel:1e-5 ())
          (Nx.create Nx.float32 [| 2 |] [| -4.; -0.25 |])
          (Solution.get s));
    test "an integrand that changes the shape raises" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.tanh_sinh") (fun () ->
            ts (fun x -> Nx.sum ~axes:[ 1 ] x) (unit_range (vec [| 0.; 0. |]))));
  ]

let cube ?(budget = 2000) f box =
  Quad.cubature ~tol:(Tol.v ~rel:1e-9 ~abs:1e-13) ~budget f box

(* exp(−50 |x − 0.4|²) on [0, 1]², and its integral. *)
let peak2 x = Nx.exp (Nx.mul_s (sum_coords (Nx.square (Nx.sub_s x 0.4))) (-50.))

let peak2_integral =
  let one =
    Float.sqrt (Float.pi /. 50.)
    /. 2.
    *. (Float.erf (0.6 *. Float.sqrt 50.) +. Float.erf (0.4 *. Float.sqrt 50.))
  in
  one *. one

(* A monomial Π x_k^(e_k) of total degree at most 7 in d dimensions, d from 2 to
   10, over a box with corners in [−2, 2]. *)
let monomial =
  Gen.(
    let* d =
      frequency [ (1, constant 2); (1, int_range 3 9); (1, constant 10) ]
    in
    let* total = frequency [ (1, constant 7); (1, int_range 0 6) ] in
    let* axes = array ~size:(constant total) (int_range 0 (d - 1)) in
    let+ lo = array ~size:(constant d) (float_range (-2.) 2.)
    and+ hi = array ~size:(constant d) (float_range (-2.) 2.) in
    let e = Array.make d 0 in
    Array.iter (fun k -> e.(k) <- e.(k) + 1) axes;
    (e, lo, hi))

let cubature_tests =
  let theta = vec [| 0.5; -1. |] in
  let integral t =
    let f x = Nx.exp (Nx.mul (sum_coords x) t) in
    Solution.get (cube f (unit_box [| 2 |] 2))
  in
  [
    prop "integrates polynomials of total degree 7 in one box" monomial
      (fun (e, lo, hi) ->
        let d = Array.length e in
        cover "d = 2" (d = 2);
        cover "d = 10" (d = 10);
        cover "total degree 7" (Array.fold_left ( + ) 0 e = 7);
        cover "a flipped axis" (Array.exists2 (fun l h -> h < l) lo hi);
        let f x =
          List.init d (fun k -> Nx.pow_s (coord k x) (Float.of_int e.(k)))
          |> List.fold_left Nx.mul (Nx.ones_like (coord 0 x))
        in
        let s =
          Quad.cubature ~tol:(Tol.rel 1e-9) ~budget:1 f
            (Quad.Box.v (vec lo) (vec hi))
        in
        let along k =
          let p = Float.of_int (e.(k) + 1) in
          ((hi.(k) ** p) -. (lo.(k) ** p)) /. p
        in
        let width k = Float.abs (hi.(k) -. lo.(k)) in
        let product g = List.fold_left ( *. ) 1. (List.init d g) in
        (* At most 1245 points, weights whose magnitudes sum to under 10 and
           values up to 2^7: the sum rounds by 1245 · 10 · 2^7 ε of the
           volume. *)
        let bound = 1245. *. 10. *. 128. *. epsilon_float *. product width in
        equal
          (float_rel ~rel:0. ~abs:(bound +. Float.min_float))
          (product along)
          (Nx.item [] (Solution.best s)));
    test "a smooth integral converges in three dimensions" (fun () ->
        (* ∫ e^(x + y + z) over [0, 1]³ = (e − 1)³, at the solve's tol. *)
        let s = cube (fun x -> Nx.exp (sum_coords x)) (unit_box [||] 3) in
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (scalar ((Float.exp 1. -. 1.) ** 3.))
          (Solution.get s));
    test "each lane integrates its own box" (fun () ->
        (* ∫ x y over [0, 1] × [0, 2] = 1 and over [−1, 1] × [0, 1] = 0. *)
        let lo = Nx.create f64 [| 2; 2 |] [| 0.; 0.; -1.; 0. |] in
        let hi = Nx.create f64 [| 2; 2 |] [| 1.; 2.; 1.; 1. |] in
        let s = cube product (Quad.Box.v lo hi) in
        equal
          (Oracle.tensor ~rel:1e-12 ~abs:1e-14 ())
          (vec [| 1.; 0. |])
          (Solution.get s));
    test "a peak converges by bisection" (fun () ->
        let s =
          Quad.cubature ~tol:(Tol.rel 1e-7) ~budget:2000 peak2 (unit_box [||] 2)
        in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (scalar peak2_integral) (Solution.get s));
    test "f receives points of shape [17; k] @ lanes @ [2]" (fun () ->
        let seen = ref [] in
        ignore (cube ~budget:4 (recording seen product) (unit_box [| 3 |] 2));
        List.iter (pieces 17 [| 3; 2 |]) !seen);
    test "budget boxes end a lane Budget_spent" (fun () ->
        let s = cube ~budget:2 peak2 (unit_box [||] 2) in
        equal Oracle.status_t Budget_spent (Oracle.status s));
    test "evaluations count the search's boxes and the answer's 32" (fun () ->
        (* Budget 2 in two dimensions: 2·2 − 1 boxes searched, then one chunk of
           32 boxes, 2² + 2·2² + 2·2 + 1 = 17 points each. *)
        let s = cube ~budget:2 peak2 (unit_box [| 1 |] 2) in
        equal (exact ())
          (Nx.create Nx.int32 [| 1 |] [| Int32.of_int (17 * (3 + 32)) |])
          (Solution.evaluations s));
    test "a pole at a point is a non-finite sum" (fun () ->
        (* The box's center has x = 0.5. *)
        let s =
          cube (fun x -> Nx.recip (Nx.sub_s (coord 0 x) 0.5)) (unit_box [||] 2)
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    test "a box at level 62 ends Stalled" (fun () ->
        (* The worst box of x^(−½) is [0, 2^−L] × [0, 1], split along x, whose
           floats never run out: only the level stops it, within a budget of
           about four boxes per level. *)
        let s =
          Quad.cubature ~tol:(Tol.rel 1e-30) ~budget:300
            (fun x -> Nx.rsqrt (coord 0 x))
            (unit_box [||] 2)
        in
        equal Oracle.status_t Stalled (Oracle.status s));
    test "grad in a parameter is the exact integral's" (fun () ->
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.map_item squared' theta)
          (Rune.grad' (fun t -> Nx.sum (integral t)) theta));
    test "grad in a corner is the integral over that face" (fun () ->
        (* ∂/∂h of ∫ e^(x + y) over [0, h] × [0, 1] is e^h (e − 1). *)
        let integral h =
          let box = Quad.Box.v (vec [| 0.; 0. |]) (Nx.stack [ h; scalar 1. ]) in
          Solution.get (cube (fun x -> Nx.exp (sum_coords x)) box)
        in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (scalar (Float.exp 0.7 *. (Float.exp 1. -. 1.)))
          (Rune.grad' integral (scalar 0.7)));
    test "a lane that did not converge has a zero derivative" (fun () ->
        (* The pole of t / (x − 1/4) is the center of the box [0, ½] × [0,
           1]. *)
        let solve t =
          cube (fun x -> Nx.div t (Nx.sub_s (coord 0 x) 0.25)) (unit_box [||] 2)
        in
        let t = scalar 1. in
        equal Oracle.status_t Not_finite (Oracle.status (solve t));
        equal (exact ()) (scalar 0.)
          (Rune.grad' (fun t -> Solution.best (solve t)) t));
    test "compiled equals eager" (fun () ->
        Compiled.here ();
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (integral theta) (Rune.jit' integral theta));
    test "compiled grad equals eager grad" (fun () ->
        Compiled.here ();
        let g = Rune.grad' (fun t -> Nx.sum (integral t)) in
        equal (Oracle.tensor ~rel:1e-12 ()) (g theta) (Rune.jit' g theta));
    cases "d outside [2, 10] raises" ~id:(Printf.sprintf "d = %d") [ 1; 11 ]
      (fun d ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.cubature") (fun () ->
            cube product (unit_box [||] d)));
    test "a budget below 1 raises" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.cubature") (fun () ->
            cube ~budget:0 product (unit_box [||] 2)));
    test "an integrand that keeps the points' last axis raises" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.cubature") (fun () ->
            cube Fun.id (unit_box [||] 2)));
  ]

let key = Nx.Rng.key 2026

(* ∏ (1 + (x_i − 1/2) / 2), whose integral over [0, 1]^d is 1. *)
let tilted x = product (Nx.add_s (Nx.div_s (Nx.sub_s x 0.5) 2.) 1.)

let qmc_tests =
  let theta = vec [| 0.5; -1. |] in
  let integral t =
    let f x = Nx.exp (Nx.mul (sum_coords x) t) in
    Solution.get
      (Quad.qmc key ~tol:(Tol.rel 1e-5) ~budget:256 f (unit_box [| 2 |] 2))
  in
  [
    test "each dimension's prefix of 2^k points is balanced" (fun () ->
        (* Lane l integrates coordinate dims.(l) over [0, 1]^1111. 1024 points
           put one point in each 1/1024 of [0, 1] along any axis, so each mean
           is within 2^-11 of 1/2. *)
        let dims = [| 0; 1; 2; 7; 100; 555; 1110 |] in
        let lanes = Array.length dims in
        let index =
          Nx.create Nx.int64 [| 1; 1; lanes; 1 |] (Array.map Int64.of_int dims)
        in
        let f x =
          let index = Nx.broadcast_to [| 64; 16; lanes; 1 |] index in
          Nx.squeeze ~axes:[ 3 ] (Nx.take_along_axis ~axis:3 index x)
        in
        let s =
          Quad.qmc key ~tol:(Tol.abs 1e-300) ~budget:16 f
            (unit_box [| lanes |] 1111)
        in
        equal
          (Oracle.tensor ~abs:(Float.ldexp 1. (-11)) ())
          (Nx.full f64 [| lanes |] 0.5)
          (Solution.best s));
    cases "the integral of 1 over a box is its signed volume"
      ~id:(fun (name, _, _, _) -> name)
      [
        ("[0, 2]", [| 0. |], [| 2. |], 2.);
        ("[0, 2] × [−1, ½]", [| 0.; -1. |], [| 2.; 0.5 |], 3.);
        ("[0, 1] × [1, 0]", [| 0.; 1. |], [| 1.; 0. |], -1.);
      ]
      (fun (_, lo, hi, volume) ->
        let one x = Nx.ones_like (coord 0 x) in
        let s =
          Quad.qmc key ~tol:(Tol.abs 1e-12) ~budget:4 one
            (Quad.Box.v (vec lo) (vec hi))
        in
        equal (Oracle.tensor ~rel:1e-15 ()) (scalar volume) (Solution.get s));
    test "the error scales with the box's volume" (fun () ->
        (* e^(x/2 + y/2) over [0, 2]² sees the values of e^(x + y) over [0, 1]²
           at the same points, times the volume 4. *)
        let error f box =
          Solution.error (Quad.qmc key ~tol:(Tol.abs 1e-300) ~budget:2 f box)
        in
        let big = Quad.Box.v (vec [| 0.; 0. |]) (vec [| 2.; 2. |]) in
        equal (exact ())
          (Nx.mul_s
             (error (fun x -> Nx.exp (sum_coords x)) (unit_box [||] 2))
             4.)
          (error (fun x -> Nx.exp (Nx.div_s (sum_coords x) 2.)) big));
    test "a smooth integral converges" (fun () ->
        let s =
          Quad.qmc key ~tol:(Tol.abs 1e-5) ~budget:256 tilted (unit_box [||] 5)
        in
        (* The test is statistical: five standard errors. *)
        equal (Oracle.tensor ~abs:5e-5 ()) (scalar 1.) (Solution.get s));
    test "a converged lane stops at a power of two chunks" (fun () ->
        let budget = 300 in
        let s =
          Quad.qmc key ~tol:(Tol.abs 1e-5) ~budget tilted (unit_box [||] 5)
        in
        equal Oracle.status_t Converged (Oracle.status s);
        (* Evaluations are the chunks searched and the answer's [budget], 1024
           points each. *)
        let n = Int32.to_int (Nx.item [] (Solution.evaluations s)) in
        satisfies ~claim:"a power of two" int
          (fun k -> k > 0 && k land (k - 1) = 0)
          ((n / 1024) - budget));
    test "budget chunks end a lane Budget_spent" (fun () ->
        let s =
          Quad.qmc key ~tol:(Tol.abs 1e-300) ~budget:3 product (unit_box [||] 2)
        in
        equal Oracle.status_t Budget_spent (Oracle.status s));
    test "evaluations count the search's chunks and the answer's budget"
      (fun () ->
        (* Budget 3: three chunks of 1024 points searched, then three again. *)
        let s =
          Quad.qmc key ~tol:(Tol.abs 1e-300) ~budget:3 product
            (unit_box [| 1 |] 2)
        in
        equal (exact ())
          (Nx.create Nx.int32 [| 1 |] [| Int32.of_int (1024 * (3 + 3)) |])
          (Solution.evaluations s));
    test "a NaN of f ends a lane Not_finite" (fun () ->
        let s =
          Quad.qmc key ~tol:(Tol.rel 1e-3) ~budget:4
            (fun x -> Nx.log (Nx.sub_s (coord 0 x) 0.5))
            (unit_box [||] 2)
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    test "an estimate averaged over keys is the integral" (fun () ->
        (* 64 points of ∫ e^(x + y) over [0, 1]²: each key's estimate is
           unbiased, so 200 keys' mean is within a few standard errors. *)
        let f x = Nx.exp (sum_coords x) in
        let estimates =
          List.init 200 (fun i ->
              Nx.item []
                (Solution.best
                   (Quad.qmc (Nx.Rng.key i) ~tol:(Tol.abs 1e-300) ~budget:1 f
                      (unit_box [||] 2))))
        in
        let mean = List.fold_left ( +. ) 0. estimates /. 200. in
        let sd =
          Float.sqrt
            (List.fold_left (fun a e -> a +. ((e -. mean) ** 2.)) 0. estimates
            /. 199.)
        in
        less (float 1.)
          ~than:(5. *. sd /. Float.sqrt 200.)
          (Float.abs (mean -. ((Float.exp 1. -. 1.) ** 2.))));
    test "grad in a parameter estimates the integral's" (fun () ->
        (* The derivative is an estimate too: 1e-3 is a few of its standard
           errors. *)
        equal
          (Oracle.tensor ~rel:1e-3 ())
          (Nx.map_item squared' theta)
          (Rune.grad' (fun t -> Nx.sum (integral t)) theta));
    test "grad in a corner estimates the integral's" (fun () ->
        (* ∂/∂h of ∫ e^(x + y) over [0, h] × [0, 1] is e^h (e − 1). *)
        let integral h =
          let box = Quad.Box.v (vec [| 0.; 0. |]) (Nx.stack [ h; scalar 1. ]) in
          Solution.get
            (Quad.qmc key ~tol:(Tol.rel 1e-5) ~budget:256
               (fun x -> Nx.exp (sum_coords x))
               box)
        in
        equal
          (Oracle.tensor ~rel:1e-3 ())
          (scalar (Float.exp 0.7 *. (Float.exp 1. -. 1.)))
          (Rune.grad' integral (scalar 0.7)));
    test "a lane that did not converge has a zero derivative" (fun () ->
        let solve t =
          Quad.qmc key ~tol:(Tol.abs 1e-300) ~budget:1
            (fun x -> Nx.exp (Nx.mul (sum_coords x) t))
            (unit_box [||] 2)
        in
        let t = scalar 0.5 in
        equal Oracle.status_t Budget_spent (Oracle.status (solve t));
        equal (exact ()) (scalar 0.)
          (Rune.grad' (fun t -> Solution.best (solve t)) t));
    test "compiled equals eager with the key an argument" (fun () ->
        Compiled.here ();
        let f x = Nx.exp (sum_coords x) in
        let integral k =
          Solution.best
            (Quad.qmc k ~tol:(Tol.rel 1e-5) ~budget:64 f (unit_box [||] 3))
        in
        let compiled =
          Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> returns tensor) integral
        in
        equal (Oracle.tensor ~rel:1e-12 ()) (integral key) (compiled key));
    test "d above 1111 raises" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.qmc") (fun () ->
            Quad.qmc key ~tol:(Tol.rel 1e-3) ~budget:1 product
              (unit_box [||] 1112)));
    test "a budget below 1 raises" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.qmc") (fun () ->
            Quad.qmc key ~tol:(Tol.rel 1e-3) ~budget:0 product (unit_box [||] 2)));
    test "an integrand that keeps the points' last axis raises" (fun () ->
        raises_match (Exn.invalid_arg ~sub:"Jera.Quad.qmc") (fun () ->
            Quad.qmc key ~tol:(Tol.rel 1e-3) ~budget:1 Fun.id (unit_box [||] 2)));
  ]

let () =
  exit
    (run "Jera.Quad"
       [
         group "Rule.gauss" gauss_tests;
         group "Rule.kronrod" kronrod_tests;
         group "Rule.nodes" nodes_tests;
         group "Rule.nodes_at" nodes_at_tests;
         group "Range.v" range_tests;
         group "Range.from" from_tests;
         group "Range.line" line_tests;
         group "Box.v" box_tests;
         group "fixed" fixed_tests;
         group "cumulative" cumulative_tests;
         group "adaptive" adaptive_tests;
         group "tanh_sinh" tanh_sinh_tests;
         group "cubature" cubature_tests;
         group "qmc" qmc_tests;
       ])
