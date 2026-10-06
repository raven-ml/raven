(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Linear. The trusted side is Nx.solve on the matrix the test builds
   itself, the closed-form derivative of a solution, d u = −A⁻¹ (d A) u, and
   lanes solved one at a time. *)

open Windtrap
open Jera

let f64 = Nx.float64
let scalar x = Nx.scalar f64 x
let vec a = Nx.create f64 [| Array.length a |] a
let one = Nx.Ptree.tensor
let failure_with sub f = raises_match (Exn.failure ~substring:sub) f
let invalid_with sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let is st s = Nx.item [] (Solution.is st s)

(* A system of [n] unknowns, diagonally dominant so well conditioned: a matrix
   of entries in [-1, 1] plus [n + 1] on its diagonal, and a right-hand side. *)
let system =
  let open Gen in
  (let* n = int_range 0 9 in
   let+ a = list ~size:(constant (n * n)) (float_range (-1.) 1.)
   and+ r = list ~size:(constant n) (float_range (-10.) 10.) in
   let a = Nx.create f64 [| n; n |] (Array.of_list a) in
   let a = Nx.add a (Nx.mul_s (Nx.eye f64 n) (Float.of_int (n + 1))) in
   (a, Nx.create f64 [| n |] (Array.of_list r)))
  |> with_pp (fun ppf (a, r) ->
      Format.fprintf ppf "a = %a@ r = %a" Nx.pp a Nx.pp r)

let product a u = Nx.matmul a u
let dense a r = Linear.solve one Linear.dense (product a) r

(* Dense *)

let dense_tests =
  [
    prop "the solution is Nx.solve's on the matrix" system (fun (a, r) ->
        cover "an empty system" (Nx.dim 0 r = 0);
        cover "one unknown" (Nx.dim 0 r = 1);
        let expected = if Nx.dim 0 r = 0 then r else Nx.solve a r in
        equal
          (Oracle.tensor ~rel:1e-12 ~abs:1e-12 ())
          expected
          (Solution.get (dense a r)));
    prop "its error is the residual's magnitude" system (fun (a, r) ->
        let s = dense a r in
        let u = Solution.get s in
        equal
          (Oracle.tensor ~abs:1e-30 ())
          (Nx.abs (Nx.sub (product a u) r))
          (Solution.error s));
    test "it applies a once per unknown and once to check" (fun () ->
        let a = Nx.add (Nx.eye f64 5) (Nx.full f64 [| 5; 5 |] 0.1) in
        let s = dense a (Nx.ones f64 [| 5 |]) in
        equal int32 6l (Nx.item [] (Solution.evaluations s)));
    test "r = 0 gives u = 0" (fun () ->
        let a = Nx.add (Nx.eye f64 3) (Nx.full f64 [| 3; 3 |] 0.5) in
        equal (Oracle.tensor ()) (Nx.zeros f64 [| 3 |])
          (Solution.get (dense a (Nx.zeros f64 [| 3 |]))));
    test "a structure's float tensors are one vector" (fun () ->
        (* The system [[2, 1], [1, 3]] (p, q) = (3, 5) over a pair. *)
        let a (p, q) =
          ( Nx.add (Nx.mul_s p 2.) (Nx.reshape [| 1 |] q),
            Nx.add (Nx.reshape [||] p) (Nx.mul_s q 3.) )
        in
        let p, q =
          Solution.get
            (Linear.solve
               Nx.Ptree.(pair tensor tensor)
               Linear.dense a
               (vec [| 3. |], scalar 5.))
        in
        equal (Oracle.tensor ~rel:1e-14 ()) (vec [| 0.8 |]) p;
        equal (Oracle.tensor ~rel:1e-14 ()) (scalar 1.4) q);
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
        let s = dense a (vec [| 1.; 1. |]) in
        equal bool true (is Stalled s);
        failure_with "Jera.Linear.solve" (fun () -> Solution.get s));
    test "an affine a misses the check and ends the lane Stalled" (fun () ->
        let s =
          Linear.solve one Linear.dense
            (fun u -> Nx.add_s (Nx.mul_s u 2.) 1.)
            (vec [| 1.; 2.; 3. |])
        in
        equal bool true (is Stalled s);
        failure_with "The residual |a u - r| exceeds the solver's bound"
          (fun () -> Solution.get s));
    test "a non-finite r ends the lane Not_finite" (fun () ->
        let s = dense (Nx.eye f64 2) (vec [| 1.; Float.nan |]) in
        equal bool true (is Not_finite s));
    test "float tensors of two dtypes raise" (fun () ->
        invalid_with "Jera.Linear.solve: the float tensors of r" (fun () ->
            Linear.solve
              Nx.Ptree.(pair tensor tensor)
              Linear.dense Fun.id
              (vec [| 1. |], Nx.ones Nx.float32 [| 1 |])));
    test "a of another shape raises" (fun () ->
        invalid_with "Jera.Linear.solve: a returned" (fun () ->
            Linear.solve one Linear.dense
              (fun u -> Nx.concatenate ~axis:0 [ u; u ])
              (vec [| 1. |])));
  ]

(* Banded *)

(* A system of [n] unknowns whose matrix has entries in [-1, 1] within [w] of
   the diagonal, plus [2w + 2] on it, so strictly diagonally dominant; [w] may
   exceed [n]. *)
let band =
  let open Gen in
  (let* n = int_range 0 12 in
   let* w = int_range 0 3 in
   let+ a = list ~size:(constant (n * n)) (float_range (-1.) 1.)
   and+ r = list ~size:(constant n) (float_range (-10.) 10.) in
   let a = Nx.create f64 [| n; n |] (Array.of_list a) in
   let i = Nx.reshape [| n; 1 |] (Nx.arange Nx.int32 0 n 1)
   and j = Nx.reshape [| 1; n |] (Nx.arange Nx.int32 0 n 1) in
   let inside = Nx.less_equal_s (Nx.abs (Nx.sub i j)) (Int32.of_int w) in
   let a = Nx.where inside a (Nx.zeros_like a) in
   let a = Nx.add a (Nx.mul_s (Nx.eye f64 n) (Float.of_int ((2 * w) + 2))) in
   (w, a, Nx.create f64 [| n |] (Array.of_list r)))
  |> with_pp (fun ppf (w, a, r) ->
      Format.fprintf ppf "w = %d@ a = %a@ r = %a" w Nx.pp a Nx.pp r)

let banded width a r = Linear.solve one (Linear.banded ~width) (product a) r

let banded_tests =
  [
    prop "the solution is Nx.solve's on the matrix" band (fun (w, a, r) ->
        cover "an empty system" (Nx.dim 0 r = 0);
        cover "a band wider than the system" (2 * w >= Nx.dim 0 r);
        cover "a band narrower than the system" ((2 * w) + 1 < Nx.dim 0 r);
        let expected = if Nx.dim 0 r = 0 then r else Nx.solve a r in
        equal
          (Oracle.tensor ~rel:1e-12 ~abs:1e-12 ())
          expected
          (Solution.get (banded w a r)));
    test "rows are interchanged where the diagonal is zero" (fun () ->
        (* A tridiagonal matrix with a zero diagonal, non-singular. *)
        let a =
          Nx.create f64 [| 4; 4 |]
            [| 0.; 1.; 0.; 0.; 1.; 0.; 2.; 0.; 0.; 3.; 0.; 1.; 0.; 0.; 2.; 0. |]
        in
        (* x₂ = 1, 2 x₃ = 4, x₁ + 2 x₃ = 2 and 3 x₂ + x₄ = 3. *)
        let r = vec [| 1.; 2.; 3.; 4. |] in
        equal
          (Oracle.tensor ~rel:1e-15 ~abs:1e-15 ())
          (vec [| -2.; 1.; 2.; 0. |])
          (Solution.get (banded 1 a r)));
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
        let s = banded 1 a (Nx.ones f64 [| n |]) in
        equal bool true (is Stalled s);
        failure_with "The residual |a u - r| exceeds" (fun () -> Solution.get s));
    test "a singular band ends the lane Stalled" (fun () ->
        let a =
          Nx.create f64 [| 3; 3 |] [| 1.; 1.; 0.; 1.; 1.; 0.; 0.; 0.; 1. |]
        in
        equal bool true (is Stalled (banded 1 a (vec [| 1.; 2.; 3. |]))));
    test "a negative width raises" (fun () ->
        invalid_with "Jera.Linear.banded: width = -1 is negative" (fun () ->
            Linear.banded ~width:(-1)));
  ]

(* Conjugate gradients *)

(* A symmetric positive-definite system of [n] unknowns: [Bᵀ B + I] for [B] of
   entries in [-1, 1], and a right-hand side. *)
let spd =
  let open Gen in
  (let* n = int_range 0 9 in
   let+ b = list ~size:(constant (n * n)) (float_range (-1.) 1.)
   and+ r = list ~size:(constant n) (float_range (-10.) 10.) in
   let b = Nx.create f64 [| n; n |] (Array.of_list b) in
   let a = Nx.add (Nx.matmul (Nx.transpose b) b) (Nx.eye f64 n) in
   (a, Nx.create f64 [| n |] (Array.of_list r)))
  |> with_pp (fun ppf (a, r) ->
      Format.fprintf ppf "a = %a@ r = %a" Nx.pp a Nx.pp r)

let cg ?(rel = 1e-12) ?(budget = 100) ?(precondition = Fun.id) a r =
  Linear.solve one (Linear.cg ~rel ~budget ~precondition) (product a) r

let cg_tests =
  [
    prop "its residual meets rel, so u is Nx.solve's to cond (a) rel" spd
      (fun (a, r) ->
        cover "an empty system" (Nx.dim 0 r = 0);
        let s = cg a r in
        let u = Solution.get s in
        let norm x = Nx.item [] (Nx.norm x) in
        at_most float_exact
          ~than:(1e-12 *. norm r)
          (norm (Nx.sub (product a u) r));
        if Nx.dim 0 r > 0 then
          equal (Oracle.tensor ~rel:1e-9 ~abs:1e-12 ()) (Nx.solve a r) u);
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
    test "a direction of non-positive curvature ends the lane Stalled"
      (fun () ->
        let s = cg (Nx.neg (Nx.eye f64 3)) (vec [| 1.; 2.; 3. |]) in
        equal bool true (is Stalled s);
        failure_with "curvature" (fun () -> Solution.get s));
    test "a spent budget ends the lane Budget_spent" (fun () ->
        let a = Nx.diag (vec [| 1.; 2.; 3.; 4.; 5. |]) in
        let s = cg ~budget:2 a (Nx.ones f64 [| 5 |]) in
        equal bool true (is Budget_spent s);
        equal int32 3l (Nx.item [] (Solution.evaluations s)));
    test "a non-finite r ends the lane Not_finite" (fun () ->
        equal bool true
          (is Not_finite (cg (Nx.eye f64 2) (vec [| Float.infinity; 1. |]))));
    test "a tolerance outside (0, 1) raises" (fun () ->
        invalid_with "Jera.Linear.cg: rel = 0 is not in (0, 1)" (fun () ->
            Linear.cg ~rel:0. ~budget:10 ~precondition:Fun.id));
    test "a budget below 1 raises" (fun () ->
        invalid_with "Jera.Linear.cg: budget = 0 is below 1" (fun () ->
            Linear.cg ~rel:1e-6 ~budget:0 ~precondition:Fun.id));
  ]

(* GMRES *)

let gmres ?(restart = 10) ?(rel = 1e-12) ?(budget = 100)
    ?(precondition = Fun.id) a r =
  Linear.solve one
    (Linear.gmres ~restart ~rel ~budget ~precondition)
    (product a) r

let gmres_tests =
  [
    prop "its residual meets rel, so u is Nx.solve's to cond (a) rel" system
      (fun (a, r) ->
        cover "an empty system" (Nx.dim 0 r = 0);
        cover "more unknowns than a cycle's steps" (Nx.dim 0 r > 4);
        let u = Solution.get (gmres ~restart:4 a r) in
        let norm x = Nx.item [] (Nx.norm x) in
        at_most float_exact
          ~than:(1e-12 *. norm r)
          (norm (Nx.sub (product a u) r));
        if Nx.dim 0 r > 0 then
          equal (Oracle.tensor ~rel:1e-9 ~abs:1e-12 ()) (Nx.solve a r) u);
    test "a cycle as long as the system solves it" (fun () ->
        (* The Krylov space of n steps holds the solution: one cycle of n
           products, and the residual's. *)
        let a =
          Nx.create f64 [| 4; 4 |]
            [|
              4.; 1.; 0.; 2.; -1.; 5.; 1.; 0.; 0.; 2.; 6.; 1.; 1.; 0.; -1.; 3.;
            |]
        in
        let s = gmres ~restart:4 a (vec [| 1.; 2.; 3.; 4. |]) in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Nx.solve a (vec [| 1.; 2.; 3.; 4. |]))
          (Solution.get s);
        equal int32 5l (Nx.item [] (Solution.evaluations s)));
    test "it solves an indefinite system cg cannot" (fun () ->
        (* A rotation: pᵀ a p = 0 for every p. *)
        let a = Nx.create f64 [| 2; 2 |] [| 0.; 1.; -1.; 0. |] in
        let r = vec [| 1.; 2. |] in
        equal bool true (is Stalled (cg a r));
        equal
          (Oracle.tensor ~rel:1e-14 ~abs:1e-15 ())
          (Nx.solve a r)
          (Solution.get (gmres ~restart:2 a r)));
    test "an exact preconditioner takes one step" (fun () ->
        let a = Nx.create f64 [| 2; 2 |] [| 3.; 1.; 1.; 2. |] in
        let s =
          gmres ~restart:1 ~budget:1
            ~precondition:(Nx.matmul (Nx.inv a))
            a
            (vec [| 1.; 1. |])
        in
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (Nx.solve a (vec [| 1.; 1. |]))
          (Solution.get s);
        equal int32 2l (Nx.item [] (Solution.evaluations s)));
    test "r = 0 gives u = 0 with no product" (fun () ->
        let s = gmres (Nx.eye f64 3) (Nx.zeros f64 [| 3 |]) in
        equal (Oracle.tensor ()) (Nx.zeros f64 [| 3 |]) (Solution.get s);
        equal int32 0l (Nx.item [] (Solution.evaluations s)));
    test "a spent budget ends the lane Budget_spent" (fun () ->
        let a = Nx.diag (vec [| 1.; 2.; 3.; 4.; 5. |]) in
        let s = gmres ~restart:1 ~budget:2 a (Nx.ones f64 [| 5 |]) in
        equal bool true (is Budget_spent s);
        equal int32 4l (Nx.item [] (Solution.evaluations s)));
    test "a non-finite r ends the lane Not_finite" (fun () ->
        equal bool true
          (is Not_finite (gmres (Nx.eye f64 2) (vec [| Float.nan; 1. |]))));
    test "a restart below 1 raises" (fun () ->
        invalid_with "Jera.Linear.gmres: restart = 0 is below 1" (fun () ->
            Linear.gmres ~restart:0 ~rel:1e-6 ~budget:10 ~precondition:Fun.id));
    test "a budget below the restart raises" (fun () ->
        invalid_with "Jera.Linear.gmres: budget = 3 is below restart = 4"
          (fun () ->
            Linear.gmres ~restart:4 ~rel:1e-6 ~budget:3 ~precondition:Fun.id));
    test "a tolerance outside (0, 1) raises" (fun () ->
        invalid_with "Jera.Linear.gmres: rel = 1 is not in (0, 1)" (fun () ->
            Linear.gmres ~restart:4 ~rel:1. ~budget:10 ~precondition:Fun.id));
  ]

(* Derivatives and transformations *)

(* [A + θ I] and its product, for a θ that a derivative tracks. *)
let shifted a theta u = Nx.add (product a u) (Nx.mul theta u)

let transformation_tests =
  let a = Nx.create f64 [| 3; 3 |] [| 4.; 1.; 0.; 1.; 5.; 2.; 0.; 1.; 6. |] in
  let r = vec [| 1.; 2.; 3. |] in
  [
    test "grad in the operator is −1ᵀ A⁻¹ A⁻¹ r" (fun () ->
        let g =
          Rune.grad'
            (fun theta ->
              Nx.sum
                (Solution.get
                   (Linear.solve one Linear.dense (shifted a theta) r)))
            (scalar 0.)
        in
        let u = Nx.solve a r in
        equal (Oracle.tensor ~rel:1e-12 ()) (Nx.neg (Nx.sum (Nx.solve a u))) g);
    test "grad in r is A⁻ᵀ 1" (fun () ->
        let g = Rune.grad' (fun r -> Nx.sum (Solution.get (dense a r))) r in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Nx.solve (Nx.transpose a) (Nx.ones f64 [| 3 |]))
          g);
    test "jvp in the operator is −A⁻¹ u" (fun () ->
        let _, t =
          Rune.jvp'
            (fun theta ->
              Solution.get (Linear.solve one Linear.dense (shifted a theta) r))
            (scalar 0.) (scalar 1.)
        in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Nx.neg (Nx.solve a (Nx.solve a r)))
          t);
    test "grad through cg is dense's" (fun () ->
        let spd = Nx.add (Nx.matmul (Nx.transpose a) a) (Nx.eye f64 3) in
        let g s =
          Rune.grad'
            (fun theta ->
              Nx.sum (Solution.get (Linear.solve one s (shifted spd theta) r)))
            (scalar 0.)
        in
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (g Linear.dense)
          (g (Linear.cg ~rel:1e-13 ~budget:50 ~precondition:Fun.id)));
    test "grad through banded is dense's" (fun () ->
        let g s =
          Rune.grad'
            (fun theta ->
              Nx.sum (Solution.get (Linear.solve one s (shifted a theta) r)))
            (scalar 0.)
        in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (g Linear.dense)
          (g (Linear.banded ~width:1)));
    test "compiled banded equals eager" (fun () ->
        let f r = Solution.get (banded 1 a r) in
        equal (Oracle.tensor ~rel:1e-14 ()) (f r) (Rune.jit' f r));
    test "lanes of banded solve their own systems" (fun () ->
        let rs = Nx.stack [ vec [| 1.; 0.; 0. |]; r ] in
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (Nx.stack
             [
               Solution.get (banded 1 a (vec [| 1.; 0.; 0. |]));
               Solution.get (banded 1 a r);
             ])
          (Rune.vmap' (fun r -> Solution.get (banded 1 a r)) rs));
    test "grad through gmres is dense's" (fun () ->
        let g s =
          Rune.grad'
            (fun theta ->
              Nx.sum (Solution.get (Linear.solve one s (shifted a theta) r)))
            (scalar 0.)
        in
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (g Linear.dense)
          (g
             (Linear.gmres ~restart:3 ~rel:1e-13 ~budget:30 ~precondition:Fun.id)));
    test "compiled gmres equals eager" (fun () ->
        let f r = Solution.get (gmres ~restart:2 a r) in
        equal (Oracle.tensor ~rel:1e-12 ()) (f r) (Rune.jit' f r));
    test "lanes of gmres stop on their own" (fun () ->
        let rs = Nx.stack [ Nx.zeros f64 [| 3 |]; r ] in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Nx.stack
             [ Nx.zeros f64 [| 3 |]; Solution.get (gmres ~restart:2 a r) ])
          (Rune.vmap' (fun r -> Solution.get (gmres ~restart:2 a r)) rs));
    test "compiled cg equals eager" (fun () ->
        let spd = Nx.add (Nx.matmul (Nx.transpose a) a) (Nx.eye f64 3) in
        let f r = Solution.get (cg spd r) in
        equal (Oracle.tensor ~rel:1e-12 ()) (f r) (Rune.jit' f r));
    test "lanes of cg stop on their own" (fun () ->
        (* The first lane's r is zero and needs no step. *)
        let spd = Nx.add (Nx.matmul (Nx.transpose a) a) (Nx.eye f64 3) in
        let rs = Nx.stack [ Nx.zeros f64 [| 3 |]; r ] in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Nx.stack [ Nx.zeros f64 [| 3 |]; Solution.get (cg spd r) ])
          (Rune.vmap' (fun r -> Solution.get (cg spd r)) rs));
    test "compiled dense equals eager on 96 unknowns" (fun () ->
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
    test "compiled equals eager to rounding" (fun () ->
        let f r = Solution.get (dense a r) in
        equal (Oracle.tensor ~rel:1e-14 ()) (f r) (Rune.jit' f r));
    test "lanes solve their own systems" (fun () ->
        let rs = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; -1.; 0.; 5. |] in
        let one_lane i = Solution.get (dense a (Nx.slice [ Nx.I i ] rs)) in
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (Nx.stack [ one_lane 0; one_lane 1 ])
          (Rune.vmap' (fun r -> Solution.get (dense a r)) rs));
  ]

let () =
  exit
    (run "Jera.Linear"
       [
         group "dense" dense_tests;
         group "banded" banded_tests;
         group "cg" cg_tests;
         group "gmres" gmres_tests;
         group "transformations" transformation_tests;
       ])
