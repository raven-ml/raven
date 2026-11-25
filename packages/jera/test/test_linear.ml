(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Linear. The trusted side is Nx.solve on the matrix the test builds
   itself, the bounds the solvers state, computed in OCaml floats, the
   closed-form derivative of a solution, d u = −A⁻¹ (d A) u, and lanes solved
   one at a time. *)

open Windtrap
open Jera

let f64 = Nx.float64
let scalar x = Nx.scalar f64 x
let vec a = Nx.create f64 [| Array.length a |] a
let one = Nx.Ptree.tensor
let product a u = Nx.matmul a u
let norm x = Float.sqrt (Oracle.dot x x)
let raises_from fn f = raises_match (Exn.invalid_arg ~sub:fn) f

(* Machine epsilon and the least positive subnormal of float32, and float64's
   least positive subnormal. *)
let eps32 = Float.ldexp 1. (-23)
let eta32 = Float.ldexp 1. (-149)
let eta = Float.ldexp 1. (-1074)

(* A count of unknowns: none and one each in a fifth of the draws. *)
let unknowns =
  Gen.(frequency [ (1, constant 0); (1, constant 1); (3, int_range 2 9) ])

let size_covers r =
  cover "an empty system" (Nx.dim 0 r = 0);
  cover "one unknown" (Nx.dim 0 r = 1)

let pp_system ppf (a, r) = Format.fprintf ppf "a = %a@ r = %a" Nx.pp a Nx.pp r

(* A system of [n] unknowns, diagonally dominant so well conditioned: a matrix
   of entries in [-1, 1] plus [n + 1] on its diagonal, and a right-hand side. *)
let system =
  let open Gen in
  (let* n = unknowns in
   let+ a = list ~size:(constant (n * n)) (float_range (-1.) 1.)
   and+ r = list ~size:(constant n) (float_range (-10.) 10.) in
   let a = Nx.create f64 [| n; n |] (Array.of_list a) in
   let a = Nx.add a (Nx.mul_s (Nx.eye f64 n) (Float.of_int (n + 1))) in
   (a, Nx.create f64 [| n |] (Array.of_list r)))
  |> with_pp pp_system

(* A symmetric positive-definite system of [n] unknowns: [Bᵀ B + I] for [B] of
   entries in [-1, 1], and a right-hand side. *)
let spd =
  let open Gen in
  (let* n = unknowns in
   let+ b = list ~size:(constant (n * n)) (float_range (-1.) 1.)
   and+ r = list ~size:(constant n) (float_range (-10.) 10.) in
   let b = Nx.create f64 [| n; n |] (Array.of_list b) in
   let a = Nx.add (Nx.matmul (Nx.transpose b) b) (Nx.eye f64 n) in
   (a, Nx.create f64 [| n |] (Array.of_list r)))
  |> with_pp pp_system

let dense a r = Linear.solve one Linear.dense (product a) r
let banded width a r = Linear.solve one (Linear.banded ~width) (product a) r

let cg a r =
  Linear.solve one
    (Linear.cg ~rel:1e-12 ~budget:100 ~precondition:Fun.id)
    (product a) r

let gmres ~restart a r =
  Linear.solve one
    (Linear.gmres ~restart ~rel:1e-12 ~budget:100 ~precondition:Fun.id)
    (product a) r

(* Nx.solve on the matrix itself. For [system]'s matrices, of condition below
   about [2n + 1], both solutions meet a backward error of [16 n ε], so they
   agree to about [cond · 16 n ε], below 1e-12 for [n ≤ 9]. *)
let solution a r = if Nx.dim 0 r = 0 then r else Nx.solve a r
let near_solution = Oracle.tensor ~rel:1e-12 ~abs:1e-12 ()

(* The bound that Linear.dense states on the norm of its residual, in floats,
   [16 n ε (‖A‖ ‖u‖ + ‖r‖) + 16 n η (1 + ‖A‖)]. *)
let dense_bound ~eps ~eta a u r =
  let n = Float.of_int (Nx.dim 0 r) and norm_a = norm a in
  (16. *. n *. eps *. ((norm_a *. norm u) +. norm r))
  +. (16. *. n *. eta *. (1. +. norm_a))

(* [‖a u − r‖] of an answer, its residual computed in its dtype. *)
let residual a u r = norm (Nx.sub (product a u) r)

(* The bound of {!Linear.cg} and {!Linear.gmres}: [rel ‖r‖ + n η]. *)
let krylov_bound ~rel ~eta r =
  (rel *. norm r) +. (Float.of_int (Nx.dim 0 r) *. eta)

(* [A + θ I] and its product, for a θ that a derivative tracks. *)
let shifted a theta u = Nx.add (product a u) (Nx.mul theta u)

(* The derivative of [Σ u] in the shift θ of [A + θ I], at θ = 0, through the
   solver [s]. *)
let grad_in_shift s a r =
  Rune.grad'
    (fun theta ->
      Nx.sum (Solution.get (Linear.solve one s (shifted a theta) r)))
    (scalar 0.)

(* A system of condition below 3 for the derivative and transformation tests,
   and its symmetric positive-definite companion [aᵀ a + I], of condition below
   50. *)
let fixed_a = Nx.create f64 [| 3; 3 |] [| 4.; 1.; 0.; 1.; 5.; 2.; 0.; 1.; 6. |]
let fixed_spd = Nx.add (Nx.matmul (Nx.transpose fixed_a) fixed_a) (Nx.eye f64 3)
let fixed_r = vec [| 1.; 2.; 3. |]

(* Dense *)

let dense_tests =
  [
    prop "u is Nx.solve's on the matrix" system (fun (a, r) ->
        size_covers r;
        equal near_solution (solution a r) (Solution.get (dense a r)));
    prop "its residual meets dense's bound, in float64 and float32"
      Gen.(pair system bool)
      (fun ((a, r), single) ->
        cover "float32" single;
        if single then begin
          let a = Nx.cast Nx.float32 a and r = Nx.cast Nx.float32 r in
          let u = Solution.get (dense a r) in
          at_most float_exact
            ~than:(dense_bound ~eps:eps32 ~eta:eta32 a u r)
            (residual a u r)
        end
        else
          let u = Solution.get (dense a r) in
          at_most float_exact
            ~than:(dense_bound ~eps:epsilon_float ~eta a u r)
            (residual a u r));
    prop "its error is the residual's magnitude" system (fun (a, r) ->
        let s = dense a r in
        let u = Solution.get s in
        equal (Oracle.tensor ())
          (Nx.abs (Nx.sub (product a u) r))
          (Solution.error s));
    test "a subnormal solution of a large operator converges" (fun () ->
        (* u = 0.6 η rounds to η, so the residual 10 η − 6 η = 4 η is the least
           a float u gives, which the bound's η term must admit. *)
        let s =
          Linear.solve one Linear.dense
            (fun u -> Nx.mul_s u 10.)
            (vec [| 6. *. eta |])
        in
        equal (Oracle.tensor ()) (vec [| eta |]) (Solution.get s));
    test "it applies a once per unknown and once to check" (fun () ->
        let a = Nx.add (Nx.eye f64 5) (Nx.full f64 [| 5; 5 |] 0.1) in
        let s = dense a (Nx.ones f64 [| 5 |]) in
        equal int32 6l (Nx.item [] (Solution.evaluations s)));
    prop "grad in r is A⁻ᵀ 1" system (fun (a, r) ->
        let expected =
          if Nx.dim 0 r = 0 then r
          else Nx.solve (Nx.transpose a) (Nx.ones_like r)
        in
        equal near_solution expected
          (Rune.grad' (fun r -> Nx.sum (Solution.get (dense a r))) r));
    prop "grad in the operator is −1ᵀ A⁻¹ A⁻¹ r" system (fun (a, r) ->
        let expected =
          if Nx.dim 0 r = 0 then scalar 0.
          else Nx.neg (Nx.sum (Nx.solve a (Nx.solve a r)))
        in
        equal near_solution expected (grad_in_shift Linear.dense a r));
    prop "jvp in the operator is −A⁻¹ A⁻¹ r" system (fun (a, r) ->
        let expected =
          if Nx.dim 0 r = 0 then r else Nx.neg (Nx.solve a (Nx.solve a r))
        in
        let _, t =
          Rune.jvp'
            (fun theta ->
              Solution.get (Linear.solve one Linear.dense (shifted a theta) r))
            (scalar 0.) (scalar 1.)
        in
        equal near_solution expected t);
    test "compiled equals eager on 96 unknowns" (fun () ->
        Compiled.here ();
        (* A matrix of condition below 3: both answers meet the backward error
           16 n ε, so they agree to about 1e-13. *)
        let n = 96 in
        let m =
          Nx.add
            (Nx.mul_s (Nx.eye f64 n) (Float.of_int n))
            (Nx.sin
               (Nx.reshape [| n; n |]
                  (Nx.arange_f f64 0. (Float.of_int (n * n)) 1.)))
        in
        let f r = Solution.get (dense m r) in
        let r = Nx.linspace f64 (-1.) 1. n in
        equal (Oracle.tensor ~rel:1e-12 ~abs:1e-15 ()) (f r) (Rune.jit' f r));
    test "lanes of a vmap solve their own systems" (fun () ->
        let rs = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; -1.; 0.; 5. |] in
        let one_lane i = Solution.get (dense fixed_a (Nx.get [ i ] rs)) in
        equal (Oracle.tensor ())
          (Nx.stack [ one_lane 0; one_lane 1 ])
          (Rune.vmap' (fun r -> Solution.get (dense fixed_a r)) rs));
  ]

(* Banded *)

(* A system of [n] unknowns whose matrix has entries in [-1, 1] within [w] of
   the diagonal, plus [2w + 2] on it, so strictly diagonally dominant; [w] may
   exceed [n]. *)
let band =
  let open Gen in
  (let* n = frequency [ (1, constant 0); (4, int_range 1 12) ] in
   let* w = int_range 0 3 in
   let+ a = list ~size:(constant (n * n)) (float_range (-1.) 1.)
   and+ r = list ~size:(constant n) (float_range (-10.) 10.) in
   let a = Nx.create f64 [| n; n |] (Array.of_list a) in
   let i = Nx.reshape [| n; 1 |] (Nx.arange Nx.int32 0 n 1)
   and j = Nx.reshape [| 1; n |] (Nx.arange Nx.int32 0 n 1) in
   let inside = Nx.less_equal_s (Nx.abs (Nx.sub i j)) (Int32.of_int w) in
   let a = Nx.where inside a (Nx.zeros_like a) in
   let a = Nx.add a (Nx.mul_s (Nx.eye f64 n) (Float.of_int ((2 * w) + 2))) in
   (w, (a, Nx.create f64 [| n |] (Array.of_list r))))
  |> with_pp (fun ppf (w, ar) -> Format.fprintf ppf "w = %d@ %a" w pp_system ar)

let banded_tests =
  [
    prop "u is Nx.solve's on the matrix" band (fun (w, (a, r)) ->
        cover "an empty system" (Nx.dim 0 r = 0);
        cover "a band wider than the system" (2 * w >= Nx.dim 0 r);
        cover "a band narrower than the system" ((2 * w) + 1 < Nx.dim 0 r);
        equal near_solution (solution a r) (Solution.get (banded w a r)));
    test "rows are interchanged where the diagonal is zero" (fun () ->
        (* A tridiagonal matrix with a zero diagonal, non-singular. Its rows
           read x₂ = 1, x₁ + 2 x₃ = 2, 3 x₂ + x₄ = 3 and 2 x₃ = 4. *)
        let a =
          Nx.create f64 [| 4; 4 |]
            [| 0.; 1.; 0.; 0.; 1.; 0.; 2.; 0.; 0.; 3.; 0.; 1.; 0.; 0.; 2.; 0. |]
        in
        equal
          (Oracle.tensor ~rel:1e-15 ~abs:1e-15 ())
          (vec [| -2.; 1.; 2.; 0. |])
          (Solution.get (banded 1 a (vec [| 1.; 2.; 3.; 4. |]))));
    test "it applies a 2 width + 1 times and once to check" (fun () ->
        let a =
          Nx.add
            (Nx.mul_s (Nx.eye f64 20) 4.)
            (Nx.diag ~k:1 (Nx.ones f64 [| 19 |]))
        in
        let s = banded 1 a (Nx.ones f64 [| 20 |]) in
        equal int32 4l (Nx.item [] (Solution.evaluations s)));
    test "a band too narrow for a misses the check" (fun () ->
        (* A pentadiagonal matrix probed as a tridiagonal one. *)
        let n = 8 in
        let a =
          Nx.add
            (Nx.mul_s (Nx.eye f64 n) 6.)
            (Nx.add
               (Nx.diag ~k:2 (Nx.ones f64 [| n - 2 |]))
               (Nx.diag ~k:(-2) (Nx.ones f64 [| n - 2 |])))
        in
        equal Oracle.status_t Stalled
          (Oracle.status (banded 1 a (Nx.ones f64 [| n |]))));
    test "a singular band ends the lane Stalled" (fun () ->
        let a =
          Nx.create f64 [| 3; 3 |] [| 1.; 1.; 0.; 1.; 1.; 0.; 0.; 0.; 1. |]
        in
        equal Oracle.status_t Stalled
          (Oracle.status (banded 1 a (vec [| 1.; 2.; 3. |]))));
    test "grad is dense's" (fun () ->
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (grad_in_shift Linear.dense fixed_a fixed_r)
          (grad_in_shift (Linear.banded ~width:1) fixed_a fixed_r));
    test "compiled equals eager" (fun () ->
        Compiled.here ();
        (* Both answers meet the backward error 16 n ε of a system of condition
           below 3, so they agree to about 1e-14. *)
        let f r = Solution.get (banded 1 fixed_a r) in
        equal (Oracle.tensor ~rel:1e-13 ()) (f fixed_r) (Rune.jit' f fixed_r));
    test "lanes of a vmap solve their own systems" (fun () ->
        let first = vec [| 1.; 0.; 0. |] in
        equal (Oracle.tensor ())
          (Nx.stack
             [
               Solution.get (banded 1 fixed_a first);
               Solution.get (banded 1 fixed_a fixed_r);
             ])
          (Rune.vmap'
             (fun r -> Solution.get (banded 1 fixed_a r))
             (Nx.stack [ first; fixed_r ])));
    test "a negative width raises" (fun () ->
        raises_from "Jera.Linear.banded" (fun () -> Linear.banded ~width:(-1)));
  ]

(* Conjugate gradients *)

let cg_tests =
  [
    prop "its residual meets rel ‖r‖ + n η" spd (fun (a, r) ->
        size_covers r;
        let u = Solution.get (cg a r) in
        at_most float_exact
          ~than:(krylov_bound ~rel:1e-12 ~eta r)
          (residual a u r));
    test "in float32 its residual meets float32's bound" (fun () ->
        let a = Nx.cast Nx.float32 fixed_spd
        and r = Nx.cast Nx.float32 fixed_r in
        let u =
          Solution.get
            (Linear.solve one
               (Linear.cg ~rel:1e-5 ~budget:20 ~precondition:Fun.id)
               (product a) r)
        in
        at_most float_exact
          ~than:(krylov_bound ~rel:1e-5 ~eta:eta32 r)
          (residual a u r));
    test "an exact preconditioner takes one step" (fun () ->
        (* A diagonal a, preconditioned by its inverse: one step and the
           check. *)
        let d = vec [| 1.; 10.; 100.; 1000. |] in
        let s =
          Linear.solve one
            (Linear.cg ~rel:1e-12 ~budget:10 ~precondition:(fun v -> Nx.div v d))
            (Nx.mul d) (Nx.ones f64 [| 4 |])
        in
        equal
          (Oracle.tensor ~rel:1e-15 ())
          (Nx.div (Nx.ones f64 [| 4 |]) d)
          (Solution.get s);
        equal int32 2l (Nx.item [] (Solution.evaluations s)));
    test "r = 0 gives u = 0 with the check alone" (fun () ->
        let s = cg (Nx.eye f64 3) (Nx.zeros f64 [| 3 |]) in
        equal (Oracle.tensor ()) (Nx.zeros f64 [| 3 |]) (Solution.get s);
        equal int32 1l (Nx.item [] (Solution.evaluations s)));
    cases ~id:fst "a direction of non-positive curvature ends the lane Stalled"
      [
        ("−I", Nx.neg (Nx.eye f64 3));
        (* pᵀ a p = 0 for every p in the plane the rotation turns. *)
        ( "a rotation",
          Nx.create f64 [| 3; 3 |] [| 0.; 1.; 0.; -1.; 0.; 0.; 0.; 0.; 1. |] );
      ]
      (fun (_, a) ->
        equal Oracle.status_t Stalled
          (Oracle.status (cg a (vec [| 1.; 2.; 0. |]))));
    test "a spent budget ends the lane Budget_spent" (fun () ->
        let a = Nx.diag (vec [| 1.; 2.; 3.; 4.; 5. |]) in
        let s =
          Linear.solve one
            (Linear.cg ~rel:1e-12 ~budget:2 ~precondition:Fun.id)
            (product a) (Nx.ones f64 [| 5 |])
        in
        equal Oracle.status_t Budget_spent (Oracle.status s);
        equal int32 3l (Nx.item [] (Solution.evaluations s)));
    test "grad is dense's" (fun () ->
        (* Both derivatives solve a system of condition below 50 to rel 1e-13 or
           better, so they agree to about 1e-11. *)
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (grad_in_shift Linear.dense fixed_spd fixed_r)
          (grad_in_shift
             (Linear.cg ~rel:1e-13 ~budget:50 ~precondition:Fun.id)
             fixed_spd fixed_r));
    test "compiled equals eager" (fun () ->
        Compiled.here ();
        (* Both answers meet rel 1e-12 of a system of condition below 50, so
           they agree to about 1e-10. *)
        let f r = Solution.get (cg fixed_spd r) in
        equal (Oracle.tensor ~rel:1e-10 ()) (f fixed_r) (Rune.jit' f fixed_r));
    test "lanes of a vmap stop on their own" (fun () ->
        (* The first lane's r is zero and needs no step. *)
        let zero = Nx.zeros f64 [| 3 |] in
        equal (Oracle.tensor ())
          (Nx.stack [ zero; Solution.get (cg fixed_spd fixed_r) ])
          (Rune.vmap'
             (fun r -> Solution.get (cg fixed_spd r))
             (Nx.stack [ zero; fixed_r ])));
    cases ~id:fst "raises on rel outside (0, 1) or a budget below 1"
      [
        ("rel 0", (0., 10));
        ("rel 1", (1., 10));
        ("rel nan", (nan, 10));
        ("budget 0", (1e-6, 0));
      ]
      (fun (_, (rel, budget)) ->
        raises_from "Jera.Linear.cg" (fun () ->
            Linear.cg ~rel ~budget ~precondition:Fun.id));
  ]

(* GMRES *)

let gmres_tests =
  [
    prop "its residual meets rel ‖r‖ + n η" system (fun (a, r) ->
        size_covers r;
        cover "more unknowns than a cycle's steps" (Nx.dim 0 r > 4);
        let u = Solution.get (gmres ~restart:4 a r) in
        at_most float_exact
          ~than:(krylov_bound ~rel:1e-12 ~eta r)
          (residual a u r));
    test "in float32 its residual meets float32's bound" (fun () ->
        let a = Nx.cast Nx.float32 fixed_a and r = Nx.cast Nx.float32 fixed_r in
        let u =
          Solution.get
            (Linear.solve one
               (Linear.gmres ~restart:3 ~rel:1e-5 ~budget:30
                  ~precondition:Fun.id)
               (product a) r)
        in
        at_most float_exact
          ~than:(krylov_bound ~rel:1e-5 ~eta:eta32 r)
          (residual a u r));
    test "a cycle as long as the system solves it" (fun () ->
        (* The Krylov space of n steps holds the solution: the first residual's
           product, one cycle of n products, and the residual's. *)
        let a =
          Nx.create f64 [| 4; 4 |]
            [|
              4.; 1.; 0.; 2.; -1.; 5.; 1.; 0.; 0.; 2.; 6.; 1.; 1.; 0.; -1.; 3.;
            |]
        in
        let r = vec [| 1.; 2.; 3.; 4. |] in
        let s = gmres ~restart:4 a r in
        equal (Oracle.tensor ~rel:1e-12 ()) (Nx.solve a r) (Solution.get s);
        equal int32 6l (Nx.item [] (Solution.evaluations s)));
    test "it solves a rotation, which is indefinite" (fun () ->
        let a = Nx.create f64 [| 2; 2 |] [| 0.; 1.; -1.; 0. |] in
        let r = vec [| 1.; 2. |] in
        equal
          (Oracle.tensor ~rel:1e-14 ~abs:1e-15 ())
          (Nx.solve a r)
          (Solution.get (gmres ~restart:2 a r)));
    test "an exact preconditioner takes one step" (fun () ->
        let a = Nx.create f64 [| 2; 2 |] [| 3.; 1.; 1.; 2. |] in
        let r = vec [| 1.; 1. |] in
        let s =
          Linear.solve one
            (Linear.gmres ~restart:1 ~rel:1e-12 ~budget:1
               ~precondition:(Nx.matmul (Nx.inv a)))
            (product a) r
        in
        equal (Oracle.tensor ~rel:1e-14 ()) (Nx.solve a r) (Solution.get s);
        equal int32 3l (Nx.item [] (Solution.evaluations s)));
    test "r = 0 gives u = 0 with the check alone" (fun () ->
        let s = gmres ~restart:3 (Nx.eye f64 3) (Nx.zeros f64 [| 3 |]) in
        equal (Oracle.tensor ()) (Nx.zeros f64 [| 3 |]) (Solution.get s);
        equal int32 1l (Nx.item [] (Solution.evaluations s)));
    test "an affine a at r = 0 does not converge" (fun () ->
        (* a 0 = 1 is the first residual's product: the cycles cannot meet the
           bound and the budget ends the lane. *)
        let s =
          Linear.solve one
            (Linear.gmres ~restart:3 ~rel:1e-12 ~budget:30 ~precondition:Fun.id)
            (fun u -> Nx.add_s (Nx.mul_s u 2.) 1.)
            (Nx.zeros f64 [| 3 |])
        in
        equal Oracle.status_t Budget_spent (Oracle.status s));
    test "a spent budget ends the lane Budget_spent" (fun () ->
        let a = Nx.diag (vec [| 1.; 2.; 3.; 4.; 5. |]) in
        let s =
          Linear.solve one
            (Linear.gmres ~restart:1 ~rel:1e-12 ~budget:2 ~precondition:Fun.id)
            (product a) (Nx.ones f64 [| 5 |])
        in
        equal Oracle.status_t Budget_spent (Oracle.status s);
        equal int32 5l (Nx.item [] (Solution.evaluations s)));
    test "grad is dense's" (fun () ->
        (* Both derivatives solve a system of condition below 3 to rel 1e-13 or
           better. *)
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (grad_in_shift Linear.dense fixed_a fixed_r)
          (grad_in_shift
             (Linear.gmres ~restart:3 ~rel:1e-13 ~budget:30 ~precondition:Fun.id)
             fixed_a fixed_r));
    test "compiled equals eager" (fun () ->
        Compiled.here ();
        (* Both answers meet rel 1e-12 of a system of condition below 3. *)
        let f r = Solution.get (gmres ~restart:2 fixed_a r) in
        equal (Oracle.tensor ~rel:1e-11 ()) (f fixed_r) (Rune.jit' f fixed_r));
    test "lanes of a vmap stop on their own" (fun () ->
        let zero = Nx.zeros f64 [| 3 |] in
        equal (Oracle.tensor ())
          (Nx.stack [ zero; Solution.get (gmres ~restart:2 fixed_a fixed_r) ])
          (Rune.vmap'
             (fun r -> Solution.get (gmres ~restart:2 fixed_a r))
             (Nx.stack [ zero; fixed_r ])));
    cases ~id:fst
      "raises on a restart below 1, a budget below it, or rel outside (0, 1)"
      [
        ("restart 0", (0, 10, 1e-6));
        ("budget 3, restart 4", (4, 3, 1e-6));
        ("rel 1", (4, 10, 1.));
      ]
      (fun (_, (restart, budget, rel)) ->
        raises_from "Jera.Linear.gmres" (fun () ->
            Linear.gmres ~restart ~rel ~budget ~precondition:Fun.id));
  ]

(* Solve *)

let solvers =
  [
    ("dense", Linear.dense);
    ("banded", Linear.banded ~width:1);
    ("cg", Linear.cg ~rel:1e-12 ~budget:10 ~precondition:Fun.id);
    ("gmres", Linear.gmres ~restart:2 ~rel:1e-12 ~budget:10 ~precondition:Fun.id);
  ]

let solve_tests =
  [
    test "a structure's float tensors are one vector" (fun () ->
        (* The system [[2, 1], [1, 3]] (p, q) = (3, 5) over a pair, solved by
           (0.8, 1.4); its error has the pair's structure. *)
        let a (p, q) =
          ( Nx.add (Nx.mul_s p 2.) (Nx.reshape [| 1 |] q),
            Nx.add (Nx.reshape [||] p) (Nx.mul_s q 3.) )
        in
        let pair = Nx.Ptree.(pair tensor tensor) in
        let s = Linear.solve pair Linear.dense a (vec [| 3. |], scalar 5.) in
        equal
          (Oracle.structure ~rel:1e-14 pair)
          (vec [| 0.8 |], scalar 1.4)
          (Solution.get s);
        let ap, aq = a (Solution.get s) in
        equal (Oracle.structure pair)
          (Nx.abs (Nx.sub_s ap 3.), Nx.abs (Nx.sub_s aq 5.))
          (Solution.error s));
    test "a tensor of another dtype is carried from r" (fun () ->
        let a (u, k) = (Nx.mul_s u 2., k) in
        let u, k =
          Solution.get
            (Linear.solve
               Nx.Ptree.(pair tensor tensor)
               Linear.dense a
               (vec [| 4.; 6. |], Nx.scalar Nx.int32 7l))
        in
        equal (Oracle.tensor ()) (vec [| 2.; 3. |]) u;
        equal (Oracle.tensor ()) (Nx.scalar Nx.int32 7l) k);
    test "a singular system ends the lane Stalled" (fun () ->
        let a = Nx.create f64 [| 2; 2 |] [| 1.; 2.; 2.; 4. |] in
        equal Oracle.status_t Stalled
          (Oracle.status (dense a (vec [| 1.; 1. |]))));
    test "an affine a misses the check and ends the lane Stalled" (fun () ->
        let s =
          Linear.solve one Linear.dense
            (fun u -> Nx.add_s (Nx.mul_s u 2.) 1.)
            (vec [| 1.; 2.; 3. |])
        in
        equal Oracle.status_t Stalled (Oracle.status s));
    cases ~id:fst "a system of no unknowns applies a once, to check" solvers
      (fun (_, solver) ->
        let s = Linear.solve one solver (fun u -> Nx.mul_s u 2.) (vec [||]) in
        equal Oracle.status_t Converged (Oracle.status s);
        equal int32 1l (Nx.item [] (Solution.evaluations s)));
    cases ~id:fst "a non-finite r ends the lane Not_finite" solvers
      (fun (_, solver) ->
        let s =
          Linear.solve one solver (product fixed_spd) (vec [| 1.; nan; 2. |])
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    cases ~id:fst "a non-finite product of a ends the lane Not_finite" solvers
      (fun (_, solver) ->
        let s =
          Linear.solve one solver (fun u -> Nx.mul_s u infinity) fixed_r
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    test "a derivative whose solve misses its bound raises" (fun () ->
        (* cg on a non-symmetric a: r = (1, 0), an eigenvector, converges in one
           step, but reverse mode solves aᵀ v = 1, which two steps of cg do
           not. *)
        let a = Nx.create f64 [| 2; 2 |] [| 1.; 1.; 0.; 2. |] in
        let solve r =
          Linear.solve one
            (Linear.cg ~rel:1e-12 ~budget:2 ~precondition:Fun.id)
            (product a) r
        in
        let r = vec [| 1.; 0. |] in
        equal Oracle.status_t Converged (Oracle.status (solve r));
        raises_from "Jera.Linear.solve" (fun () ->
            Rune.grad' (fun r -> Nx.sum (Solution.get (solve r))) r));
    test "float tensors of two dtypes raise" (fun () ->
        raises_from "Jera.Linear.solve" (fun () ->
            Linear.solve
              Nx.Ptree.(pair tensor tensor)
              Linear.dense Fun.id
              (vec [| 1. |], Nx.ones Nx.float32 [| 1 |])));
    test "an a of another shape raises" (fun () ->
        raises_from "Jera.Linear.solve" (fun () ->
            Linear.solve one Linear.dense
              (fun u -> Nx.concatenate ~axis:0 [ u; u ])
              (vec [| 1. |])));
    test "a precondition of another shape raises" (fun () ->
        raises_from "Jera.Linear.solve" (fun () ->
            Linear.solve one
              (Linear.cg ~rel:1e-12 ~budget:10 ~precondition:(fun u ->
                   Nx.concatenate ~axis:0 [ u; u ]))
              (product fixed_spd) fixed_r));
    test "an r that holds no tensor raises" (fun () ->
        raises_from "Jera.Linear.solve" (fun () ->
            Linear.solve Nx.Ptree.unit Linear.dense Fun.id ()));
  ]

let () =
  exit
    (run "Jera.Linear"
       [
         group "dense" dense_tests;
         group "banded" banded_tests;
         group "cg" cg_tests;
         group "gmres" gmres_tests;
         group "solve" solve_tests;
       ])
