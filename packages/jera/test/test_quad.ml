(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Quad's rules and formulas. The trusted side is mpmath's Gauss–Legendre
   rules, scipy's Gauss–Kronrod tables, closed-form integrals of polynomials and
   the integrals of e⁻ˣ and e⁻ˣ², and finite differences of the rule's own
   sum. *)

open Windtrap
open Jera

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let scalar x = Nx.scalar f64 x
let exact () = Oracle.tensor ()
let ulps () = Oracle.tensor ~rel:4e-16 ~abs:1e-16 ()

(* Rules *)

let gauss_goldens =
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

let kronrod_goldens =
  Golden_quad.[ (7, kronrod7_x, kronrod7_w); (10, kronrod10_x, kronrod10_w) ]

let nodes_match (x, w) (gx, gw) =
  equal ~msg:"nodes" (ulps ()) (vec gx) x;
  equal ~msg:"weights" (ulps ()) (vec gw) w

let rule_tests =
  [
    cases
      ~name:(fun (n, _, _) -> Printf.sprintf "%d points" n)
      "gauss n is mpmath's rule to the last bit or so" gauss_goldens
      (fun (n, gx, gw) ->
        nodes_match (Quad.Rule.nodes (Quad.Rule.gauss n) f64) (gx, gw));
    cases
      ~name:(fun (n, _, _) -> Printf.sprintf "%d points" ((2 * n) + 1))
      "kronrod n is QUADPACK's rule as scipy tabulates it" kronrod_goldens
      (fun (n, gx, gw) ->
        let x, w = Quad.Rule.nodes (Quad.Rule.kronrod n) f64 in
        equal ~msg:"nodes" (exact ()) (vec gx) x;
        equal ~msg:"weights" (exact ()) (vec gw) w);
    test "nodes in float32 are the float64 nodes rounded once" (fun () ->
        let x64, _ = Quad.Rule.nodes (Quad.Rule.gauss 7) f64 in
        let x32, _ = Quad.Rule.nodes (Quad.Rule.gauss 7) Nx.float32 in
        equal (exact ()) (Nx.cast Nx.float32 x64) x32);
    test "gauss rejects no point" (fun () ->
        raises_match (Exn.invalid_arg ~substring:"n = 0 is below 1") (fun () ->
            Quad.Rule.gauss 0));
    test "a Kronrod rule serves as a formula" (fun () ->
        let rules =
          [
            (Quad.Rule.kronrod 7 :> [ `Formula ] Quad.Rule.t); Quad.Rule.gauss 8;
          ]
        in
        List.iter
          (fun r ->
            equal
              (Oracle.tensor ~rel:1e-14 ())
              (Nx.scalar f64 (Float.exp 1. -. 1.))
              (Quad.fixed r Nx.exp (Quad.Range.v (scalar 0.) (scalar 1.))))
          rules);
    test "kronrod rejects a size it has no table for" (fun () ->
        raises_match (Exn.invalid_arg ~substring:"n = 8 is neither 7 nor 10")
          (fun () -> Quad.Rule.kronrod 8));
  ]

(* Exactness *)

(* A polynomial of degree [d] with coefficients in [-1, 1] and a range inside
   [-3, 3], and its integral from the antiderivative, in OCaml floats, with the
   sum of the terms' magnitudes that bounds its rounding. *)
let polynomial d =
  Gen.(
    triple
      (array ~size:(constant (d + 1)) (float_range (-1.) 1.))
      (float_range (-3.) 3.) (float_range (-3.) 3.))

let horner c x = Array.fold_right (fun ci acc -> ci +. (x *. acc)) c 0.

let antiderivative c x =
  let a = Array.mapi (fun k ck -> ck /. float_of_int (k + 1)) c in
  x *. horner a x

let magnitude c a b =
  let m = Float.max (Float.abs a) (Float.abs b) in
  Array.fold_left
    (fun (acc, p) ck -> (acc +. (Float.abs ck *. p), p *. m))
    (0., m) c
  |> fst

let exact_for rule d =
  prop (Printf.sprintf "degree %d" d) (polynomial d) (fun (c, a, b) ->
      let integral =
        Quad.fixed rule
          (fun x -> Nx.map_item (horner c) x)
          (Quad.Range.v (scalar a) (scalar b))
      in
      let truth = antiderivative c b -. antiderivative c a in
      let bound = 64. *. epsilon_float *. (magnitude c a b +. 1.) in
      equal (Windtrap.float_rel ~rel:0. ~abs:bound) truth (Nx.item [] integral))

let exactness_tests =
  [
    group "gauss n integrates polynomials of degree 2n - 1"
      (List.map
         (fun n -> exact_for (Quad.Rule.gauss n) ((2 * n) - 1))
         [ 1; 2; 5; 10; 20 ]);
    group "kronrod n integrates polynomials of degree 3n + 1"
      (List.map
         (fun n -> exact_for (Quad.Rule.kronrod n) ((3 * n) + 1))
         [ 7; 10 ]);
    test "gauss n misses degree 2n" (fun () ->
        (* ∫₋₁¹ x⁴ = 2/5, where two-point Gauss gives 2/9. *)
        let i =
          Quad.fixed (Quad.Rule.gauss 2)
            (fun x -> Nx.pow_s x 4.)
            (Quad.Range.v (scalar (-1.)) (scalar 1.))
        in
        equal (Oracle.tensor ~rel:1e-15 ()) (scalar (2. /. 9.)) i);
  ]

(* Ranges *)

let range_tests =
  [
    test "from a integrates e^-(x - a) to 1" (fun () ->
        let a = vec [| 0.; 2.5; -1. |] in
        let i =
          Quad.fixed (Quad.Rule.gauss 40)
            (fun x -> Nx.exp (Nx.neg (Nx.sub x a)))
            (Quad.Range.from a)
        in
        equal (Oracle.tensor ~rel:1e-10 ()) (Nx.ones_like a) i);
    test "line c integrates e^-(x - c)² to √π" (fun () ->
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
    test "a reversed range negates the integral" (fun () ->
        let f x = Nx.exp x in
        let r = Quad.Rule.gauss 6 in
        equal (exact ())
          (Nx.neg (Quad.fixed r f (Quad.Range.v (scalar 0.) (scalar 1.))))
          (Quad.fixed r f (Quad.Range.v (scalar 1.) (scalar 0.))));
    test "the ends broadcast together" (fun () ->
        let i =
          Quad.fixed (Quad.Rule.gauss 3) Nx.ones_like
            (Quad.Range.v (scalar 0.) (vec [| 1.; 2.; 3. |]))
        in
        equal (ulps ()) (vec [| 1.; 2.; 3. |]) i);
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
        equal (ulps ()) (vec [| 1.; nan |]) i);
    test "float32 integrates within float32's rounding" (fun () ->
        let i =
          Quad.fixed (Quad.Rule.gauss 8) Nx.exp
            (Quad.Range.v (Nx.scalar Nx.float32 0.) (Nx.scalar Nx.float32 1.))
        in
        equal
          (Oracle.tensor ~rel:1e-6 ())
          (Nx.scalar Nx.float32 (Float.exp 1. -. 1.))
          i);
  ]

(* Cumulative *)

let cumulative_tests =
  let r = Quad.Rule.gauss 5 in
  [
    test "row i integrates from the first knot to knot i" (fun () ->
        let knots = vec [| 0.; 0.5; 1.25; 2. |] in
        let c = Quad.cumulative r Nx.cos knots in
        equal (Oracle.tensor ~rel:1e-12 ~abs:1e-15 ()) (Nx.sin knots) c);
    test "the first row is zero" (fun () ->
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
    test "no knot raises" (fun () ->
        raises_match (Exn.invalid_arg ~substring:"at least one knot") (fun () ->
            Quad.cumulative r Nx.exp (vec [||])));
  ]

(* Transformations *)

(* ∫₀¹ e^(θ x) dx, whose θ-derivative the rule's sum states. *)
let integral theta =
  Quad.fixed (Quad.Rule.kronrod 7)
    (fun x -> Nx.exp (Nx.mul x theta))
    (Quad.Range.v (Nx.zeros_like theta) (Nx.ones_like theta))

let total theta = Nx.sum (integral theta)

let upper b =
  Nx.sum
    (Quad.fixed (Quad.Rule.gauss 4) Nx.sin (Quad.Range.v (Nx.zeros_like b) b))

let transformation_tests =
  let theta = vec [| -1.; 0.5; 2. |] in
  [
    test "grad in a parameter is the finite difference of the sum" (fun () ->
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Oracle.central ~eps:1e-6 integral theta (Nx.ones_like theta))
          (Rune.grad' total theta));
    test "grad in an end is the finite difference of the sum" (fun () ->
        let b = vec [| 0.3; 1.7 |] in
        let v = vec [| 1.; -2. |] in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 upper b v))
          (scalar (Oracle.dot (Rune.grad' upper b) v)));
    test "compiled equals eager" (fun () ->
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (integral theta) (Rune.jit' integral theta));
    test "vmap is each element's integral" (fun () ->
        equal
          (Oracle.tensor ~rel:1e-15 ())
          (integral theta)
          (Rune.vmap' integral theta));
    test "an integrand that changes the shape raises" (fun () ->
        raises_match
          (Exn.invalid_arg
             ~substring:"returned shape [3] for points of shape [3,2]")
          (fun () ->
            Quad.fixed (Quad.Rule.gauss 3)
              (fun x -> Nx.sum ~axes:[ 1 ] x)
              (Quad.Range.v (vec [| 0.; 0. |]) (vec [| 1.; 1. |]))));
  ]

(* Adaptive *)

let k7 = Quad.Rule.kronrod 7
let unit_range x = Quad.Range.v (Nx.zeros_like x) (Nx.ones_like x)

let adaptive ?(budget = 200) ?(tol = Tol.v ~rel:1e-10 ~abs:1e-12) f range =
  Quad.adaptive k7 ~tol ~budget f range

(* ∫₀¹ e^(θx) dx = (e^θ − 1) / θ, and its θ-derivative. *)
let exp_integral t = (Float.exp t -. 1.) /. t
let exp_integral' t = ((t *. Float.exp t) -. Float.exp t +. 1.) /. (t *. t)

let adaptive_tests =
  [
    test "a smooth integral converges" (fun () ->
        let theta = vec [| -2.; 0.5; 3. |] in
        let s =
          adaptive (fun x -> Nx.exp (Nx.mul x theta)) (unit_range theta)
        in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Nx.map_item exp_integral theta)
          (Solution.get s));
    test "an endpoint singularity converges by bisection" (fun () ->
        (* ∫₀¹ √x = 2/3 and ∫₀¹ log x = −1. *)
        let a = vec [| 0. |] in
        let r = Quad.Range.v a (Nx.ones_like a) in
        equal ~msg:"sqrt"
          (Oracle.tensor ~rel:1e-9 ())
          (vec [| 2. /. 3. |])
          (Solution.get (adaptive Nx.sqrt r));
        equal ~msg:"log"
          (Oracle.tensor ~rel:1e-9 ())
          (vec [| -1. |])
          (Solution.get (adaptive ~budget:400 Nx.log r)));
    test "each element refines on its own" (fun () ->
        (* Gaussian peaks of widths 1, 10⁻¹ and 10⁻² at 0.3 on [0, 1]. A peak
           narrower than the first rule's nodes is invisible to it, as to any
           adaptive rule. *)
        let w = vec [| 1.; 1e-1; 1e-2 |] in
        let f x = Nx.exp (Nx.neg (Nx.square (Nx.div (Nx.sub_s x 0.3) w))) in
        let s = adaptive f (unit_range w) in
        let truth w =
          w *. Float.sqrt Float.pi /. 2.
          *. (Float.erf (0.7 /. w) +. Float.erf (0.3 /. w))
        in
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.map_item truth w) (Solution.get s);
        let n = Nx.to_array (Solution.evaluations s) in
        less int32 ~than:n.(2) n.(0));
    test "a half-line converges" (fun () ->
        let a = vec [| 0.; 1. |] in
        let s =
          adaptive (fun x -> Nx.exp (Nx.neg (Nx.sub x a))) (Quad.Range.from a)
        in
        equal (Oracle.tensor ~rel:1e-9 ()) (Nx.ones_like a) (Solution.get s));
    test "the budget ends a solve" (fun () ->
        let s = adaptive ~budget:2 Nx.log (unit_range (vec [| 0. |])) in
        equal (Oracle.tensor ()) (Nx.ones Nx.bool [| 1 |])
          (Solution.is Budget_spent s);
        let report = Format.asprintf "%a" Solution.pp s in
        contains ~sub:"rule kronrod 7, tol rel 1e-10 abs 1e-12, budget 2" report;
        contains ~sub:"a 0, b 1, estimate" report;
        contains ~sub:"2 of 2 pieces, 45 evaluations." report;
        contains ~sub:"Raise the budget or loosen tol" report);
    test "a pole is not finite or stalls" (fun () ->
        let s =
          adaptive
            (fun x -> Nx.recip (Nx.sub_s x 0.5))
            (unit_range (vec [| 0. |]))
        in
        equal (Oracle.tensor ()) (Nx.zeros Nx.bool [| 1 |]) (Solution.ok s));
    test "grad in a parameter is the exact integral's" (fun () ->
        let theta = vec [| -2.; 0.5; 3. |] in
        let integral t =
          Nx.sum
            (Solution.get
               (adaptive (fun x -> Nx.exp (Nx.mul x t)) (unit_range t)))
        in
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.map_item exp_integral' theta)
          (Rune.grad' integral theta));
    test "grad in an end is the integrand there" (fun () ->
        let b = vec [| 0.7; 2. |] in
        let integral b =
          Nx.sum
            (Solution.get (adaptive Nx.cos (Quad.Range.v (Nx.zeros_like b) b)))
        in
        equal (Oracle.tensor ~rel:1e-9 ()) (Nx.cos b) (Rune.grad' integral b));
    test "an element that did not converge has a zero derivative" (fun () ->
        let p = vec [| 1.; -0.5 |] in
        (* x^p on [0, 1]: integrable for p = 1, not for p = −1.5. *)
        let integral p =
          Nx.sum
            (Solution.best
               (adaptive ~budget:20
                  (fun x -> Nx.pow x (Nx.sub_s (Nx.mul_s p 1.) 1.))
                  (unit_range p)))
        in
        let g = Rune.grad' integral p in
        equal (Oracle.tensor ()) (scalar 0.) (Nx.get [ 1 ] g));
    test "compiled equals eager" (fun () ->
        let theta = vec [| -2.; 0.5; 3. |] in
        let f t =
          Solution.get (adaptive (fun x -> Nx.exp (Nx.mul x t)) (unit_range t))
        in
        equal (Oracle.tensor ~rel:1e-14 ()) (f theta) (Rune.jit' f theta));
    xfail
      ~reason:
        "tolk: Divandmod.fold divide_by_gcd raises on option is None compiling \
         the reverse of the chunked answer"
    @@ test "compiled grad equals eager grad" (fun () ->
        let theta = vec [| -2.; 0.5; 3. |] in
        let g =
          Rune.grad' (fun t ->
              Nx.sum
                (Solution.get
                   (adaptive (fun x -> Nx.exp (Nx.mul x t)) (unit_range t))))
        in
        equal (Oracle.tensor ~rel:1e-13 ()) (g theta) (Rune.jit' g theta));
    test "vmap is each lane's solve" (fun () ->
        let theta = Nx.create f64 [| 2; 2 |] [| -2.; 0.5; 3.; 1. |] in
        let f t =
          Solution.get (adaptive (fun x -> Nx.exp (Nx.mul x t)) (unit_range t))
        in
        equal (Oracle.tensor ~rel:1e-14 ()) (f theta) (Rune.vmap' f theta));
    test "adaptive rejects a budget below 1" (fun () ->
        raises_match (Exn.invalid_arg ~substring:"budget = 0 is below 1")
          (fun () ->
            Quad.adaptive k7 ~tol:(Tol.rel 1e-6) ~budget:0 Nx.exp
              (unit_range (vec [| 0. |]))));
  ]

(* Double-exponential *)

let ts = Quad.tanh_sinh ~tol:(Tol.v ~rel:1e-12 ~abs:1e-14)

(* ∫₀¹ x^(a−1) log x dx = −1 / a². *)
let log_moment a =
  ts
    (fun x -> Nx.mul (Nx.pow x (Nx.sub_s a 1.)) (Nx.log x))
    (Quad.Range.v (Nx.zeros_like a) (Nx.ones_like a))

let de_tests =
  let close = Oracle.tensor ~rel:1e-11 () in
  [
    test "endpoint singularities at 0 converge" (fun () ->
        let a = vec [| 0.3; 0.5; 1.; 2.5 |] in
        equal close
          (Nx.map_item (fun a -> -1. /. (a *. a)) a)
          (Solution.get (log_moment a)));
    test "1 / √(1 − x²) on [−1, 1] is π" (fun () ->
        (* The integrand computes the distance to each end from x itself. *)
        let r = Quad.Range.v (vec [| -1. |]) (vec [| 1. |]) in
        let f x = Nx.rsqrt (Nx.mul (Nx.add_s x 1.) (Nx.rsub_s 1. x)) in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (vec [| Float.pi |])
          (Solution.best (ts f r)));
    test "exp-sinh integrates a half-line" (fun () ->
        let a = vec [| 0.; 0. |] in
        (* Lane 0 integrates e^-x, lane 1 1 / (1 + x²). *)
        let f x =
          let lane i = Nx.slice [ Nx.A; Nx.I i ] x in
          Nx.stack ~axis:1
            [
              Nx.exp (Nx.neg (lane 0));
              Nx.recip (Nx.add_s (Nx.square (lane 1)) 1.);
            ]
        in
        let s = ts f (Quad.Range.from a) in
        equal close (vec [| 1.; Float.pi /. 2. |]) (Solution.get s));
    test "sinh-sinh integrates the line" (fun () ->
        let c = vec [| 0.; 0.5 |] in
        let s =
          ts
            (fun x -> Nx.exp (Nx.neg (Nx.square (Nx.sub x c))))
            (Quad.Range.line c)
        in
        equal close (Nx.full f64 [| 2 |] (Float.sqrt Float.pi)) (Solution.get s));
    test "grad in a is 2 / a³" (fun () ->
        let a = vec [| 0.5; 1.; 2.5 |] in
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.map_item (fun a -> 2. /. (a *. a *. a)) a)
          (Rune.grad' (fun a -> Nx.sum (Solution.get (log_moment a))) a));
    test "a divergent integral does not converge" (fun () ->
        let s = ts Nx.recip (Quad.Range.v (vec [| 0. |]) (vec [| 1. |])) in
        equal (Oracle.tensor ()) (Nx.zeros Nx.bool [| 1 |]) (Solution.ok s));
    test "compiled equals eager" (fun () ->
        let a = vec [| 0.5; 1.; 2.5 |] in
        let f a = Solution.get (log_moment a) in
        equal (Oracle.tensor ~rel:1e-13 ()) (f a) (Rune.jit' f a));
    test "compiled grad equals eager grad" (fun () ->
        let a = vec [| 0.5; 1.; 2.5 |] in
        let g = Rune.grad' (fun a -> Nx.sum (Solution.get (log_moment a))) in
        equal (Oracle.tensor ~rel:1e-12 ()) (g a) (Rune.jit' g a));
    test "float32 converges to float32's tolerance" (fun () ->
        let a = Nx.create Nx.float32 [| 2 |] [| 0.5; 2. |] in
        let s =
          Quad.tanh_sinh ~tol:(Tol.rel 1e-5)
            (fun x -> Nx.mul (Nx.pow x (Nx.sub_s a 1.)) (Nx.log x))
            (Quad.Range.v (Nx.zeros_like a) (Nx.ones_like a))
        in
        equal
          (Oracle.tensor ~rel:1e-5 ())
          (Nx.create Nx.float32 [| 2 |] [| -4.; -0.25 |])
          (Solution.get s));
  ]

(* Cubature *)

(* Each lane's coordinate [k] of points [x] of shape [... @ [d]]. *)
let coord k x = Nx.get [ k ] (Nx.moveaxis (Nx.ndim x - 1) 0 x)
let product x = Nx.prod ~axes:[ Nx.ndim x - 1 ] x

let unit_box lanes d =
  Quad.Box.v
    (Nx.zeros f64 (Array.append lanes [| d |]))
    (Nx.ones f64 (Array.append lanes [| d |]))

let cube ?(budget = 2000) f box =
  Quad.cubature ~tol:(Tol.v ~rel:1e-9 ~abs:1e-13) ~budget f box

let cubature_tests =
  [
    test "a degree-7 monomial is exact in one box" (fun () ->
        (* ∫ x⁴ y³ over [0, 1]² = 1/20. *)
        let f x = Nx.mul (Nx.pow_s (coord 0 x) 4.) (Nx.pow_s (coord 1 x) 3.) in
        let s = cube ~budget:1 f (unit_box [||] 2) in
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (Nx.scalar f64 0.05) (Solution.best s));
    test "a smooth integral converges in three dimensions" (fun () ->
        (* ∫ e^(x + y + z) over [0, 1]³ = (e − 1)³. *)
        let s =
          cube
            (fun x -> Nx.exp (Nx.sum ~axes:[ Nx.ndim x - 1 ] x))
            (unit_box [||] 3)
        in
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.scalar f64 ((Float.exp 1. -. 1.) ** 3.))
          (Solution.get s));
    test "each lane integrates its own box" (fun () ->
        let lo = Nx.create f64 [| 2; 2 |] [| 0.; 0.; -1.; 0. |] in
        let hi = Nx.create f64 [| 2; 2 |] [| 1.; 2.; 1.; 1. |] in
        (* ∫ x y over [0, 1] × [0, 2] = 1 and over [−1, 1] × [0, 1] = 0. *)
        let s = cube (fun x -> product x) (Quad.Box.v lo hi) in
        equal
          (Oracle.tensor ~rel:1e-12 ~abs:1e-14 ())
          (vec [| 1.; 0. |])
          (Solution.get s));
    test "a peak refines the partition" (fun () ->
        let f x =
          Nx.exp
            (Nx.mul_s
               (Nx.sum ~axes:[ Nx.ndim x - 1 ] (Nx.square (Nx.sub_s x 0.4)))
               (-50.))
        in
        let s =
          Quad.cubature ~tol:(Tol.rel 1e-7) ~budget:2000 f (unit_box [||] 2)
        in
        let one =
          Float.sqrt (Float.pi /. 50.)
          /. 2.
          *. (Float.erf (0.6 *. Float.sqrt 50.)
             +. Float.erf (0.4 *. Float.sqrt 50.))
        in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (Nx.scalar f64 (one *. one))
          (Solution.get s));
    test "grad in a parameter is the exact integral's" (fun () ->
        (* ∫ e^(θ(x + y)) over [0, 1]² = ((e^θ − 1) / θ)². *)
        let theta = vec [| 0.5; -1. |] in
        let integral t =
          let f x = Nx.exp (Nx.mul (Nx.sum ~axes:[ Nx.ndim x - 1 ] x) t) in
          Nx.sum (Solution.get (cube f (unit_box [| 2 |] 2)))
        in
        let d t = 2. *. exp_integral t *. exp_integral' t in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.map_item d theta)
          (Rune.grad' integral theta));
    test "compiled equals eager" (fun () ->
        let theta = vec [| 0.5; -1. |] in
        let integral t =
          Solution.get
            (cube
               (fun x -> Nx.exp (Nx.mul (Nx.sum ~axes:[ Nx.ndim x - 1 ] x) t))
               (unit_box [| 2 |] 2))
        in
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (integral theta) (Rune.jit' integral theta));
    xfail
      ~reason:
        "tolk: Divandmod.fold divide_by_gcd raises on option is None compiling \
         the reverse of the chunked answer"
    @@ test "compiled grad equals eager grad" (fun () ->
        let theta = vec [| 0.5; -1. |] in
        let g =
          Rune.grad' (fun t ->
              Nx.sum
                (Solution.get
                   (cube
                      (fun x ->
                        Nx.exp (Nx.mul (Nx.sum ~axes:[ Nx.ndim x - 1 ] x) t))
                      (unit_box [| 2 |] 2))))
        in
        equal (Oracle.tensor ~rel:1e-12 ()) (g theta) (Rune.jit' g theta));
    test "one dimension raises" (fun () ->
        raises_match (Exn.invalid_arg ~substring:"d = 1 is not in [2, 10]")
          (fun () -> cube (fun x -> product x) (unit_box [||] 1)));
    test "eleven dimensions raise" (fun () ->
        raises_match (Exn.invalid_arg ~substring:"d = 11 is not in [2, 10]")
          (fun () -> cube (fun x -> product x) (unit_box [||] 11)));
  ]

(* Quasi-Monte Carlo *)

let key = Nx.Rng.key 2026

let qmc_tests =
  [
    cases
      ~name:(fun j -> Printf.sprintf "dimension %d" j)
      "each dimension's prefix of 2^k points is balanced"
      [ 0; 1; 2; 7; 100; 555; 1110 ]
      (fun j ->
        (* 1024 points put one point in each 1/1024 of [0, 1] along any axis, so
           the mean of x_j is within 2^-11 of 1/2. *)
        let s =
          Quad.qmc key ~tol:(Tol.abs 1e-300) ~budget:16 (coord j)
            (unit_box [||] 1111)
        in
        equal
          (Oracle.tensor ~abs:(Float.ldexp 1. (-11)) ())
          (Nx.scalar f64 0.5) (Solution.best s));
    test "a smooth integral converges" (fun () ->
        (* ∫ Π (1 + (x_i − 1/2) / 2) over [0, 1]⁵ = 1. *)
        let f x = product (Nx.add_s (Nx.div_s (Nx.sub_s x 0.5) 2.) 1.) in
        let s =
          Quad.qmc key ~tol:(Tol.abs 1e-5) ~budget:1024 f (unit_box [||] 5)
        in
        (* The test is statistical: five standard errors. *)
        equal (Oracle.tensor ~abs:5e-5 ()) (Nx.scalar f64 1.) (Solution.get s));
    test "an estimate averaged over keys is the integral" (fun () ->
        (* 64 points of ∫ e^(x + y) over [0, 1]²: each key's estimate is
           unbiased, so 200 keys' mean is within a few standard errors. *)
        let f x = Nx.exp (Nx.sum ~axes:[ Nx.ndim x - 1 ] x) in
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
        less (Windtrap.float 1.)
          ~than:(5. *. sd /. Float.sqrt 200.)
          (Float.abs (mean -. ((Float.exp 1. -. 1.) ** 2.))));
    test "grad in a parameter estimates the integral's" (fun () ->
        let theta = vec [| 0.5; -1. |] in
        let integral t =
          let f x = Nx.exp (Nx.mul (Nx.sum ~axes:[ Nx.ndim x - 1 ] x) t) in
          Nx.sum
            (Solution.get
               (Quad.qmc key ~tol:(Tol.rel 1e-5) ~budget:1024 f
                  (unit_box [| 2 |] 2)))
        in
        let d t = 2. *. exp_integral t *. exp_integral' t in
        equal
          (Oracle.tensor ~rel:1e-3 ())
          (Nx.map_item d theta)
          (Rune.grad' integral theta));
    test "compiled equals eager with the key an argument" (fun () ->
        let f x = Nx.exp (Nx.sum ~axes:[ Nx.ndim x - 1 ] x) in
        let integral k =
          Solution.best
            (Quad.qmc k ~tol:(Tol.rel 1e-5) ~budget:64 f (unit_box [||] 3))
        in
        let compiled =
          Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> returns tensor) integral
        in
        equal (Oracle.tensor ~rel:1e-12 ()) (integral key) (compiled key));
    test "dimensions past the table raise" (fun () ->
        raises_match (Exn.invalid_arg ~substring:"d = 1112 is above 1111")
          (fun () ->
            Quad.qmc key ~tol:(Tol.rel 1e-3) ~budget:1 product
              (unit_box [||] 1112)));
  ]

let () =
  exit
    (run "Jera.Quad"
       [
         group "qmc" qmc_tests;
         group "cubature" cubature_tests;
         group "double-exponential" de_tests;
         group "adaptive" adaptive_tests;
         group "rules" rule_tests;
         group "exactness" exactness_tests;
         group "ranges" range_tests;
         group "cumulative" cumulative_tests;
         group "transformations" transformation_tests;
       ])
