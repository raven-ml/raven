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
    test "compiled equals eager to rounding" (fun () ->
        let f r = Solution.get (dense a r) in
        equal (Oracle.tensor ~rel:1e-14 ()) (f r) (Rune.jit' f r));
    test "lanes solve their own systems" (fun () ->
        let rs = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; -1.; 0.; 5. |] in
        let one_lane i = Solution.get (dense a (Nx.get [ i ] rs)) in
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
         group "cg" cg_tests;
         group "transformations" transformation_tests;
       ])
