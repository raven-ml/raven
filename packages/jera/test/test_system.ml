(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.System. The trusted side is systems built around a known zero, the
   implicit derivative written with Nx.solve on the Jacobian the test builds
   itself, central differences, and lanes solved one at a time. *)

open Windtrap
open Jera

let f64 = Nx.float64
let scalar x = Nx.scalar f64 x
let vec a = Nx.create f64 [| Array.length a |] a
let one = Nx.Ptree.tensor
let tight = Tol.v ~rel:1e-12 ~abs:1e-12
let failure_with sub f = raises_match (Exn.failure ~substring:sub) f
let invalid_with sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let is st s = Nx.item [] (Solution.is st s)

(* A system [a x + sin x = b] of [n] unknowns whose zero is [z]: [a] has entries
   in [-1, 1] plus [n + 2] on its diagonal, so the Jacobian [a + diag (cos x)]
   is diagonally dominant everywhere. *)
let system =
  let open Gen in
  (let* n = int_range 0 6 in
   let+ a = list ~size:(constant (n * n)) (float_range (-1.) 1.)
   and+ z = list ~size:(constant n) (float_range (-2.) 2.) in
   let a = Nx.create f64 [| n; n |] (Array.of_list a) in
   let a = Nx.add a (Nx.mul_s (Nx.eye f64 n) (Float.of_int (n + 2))) in
   (a, Nx.create f64 [| n |] (Array.of_list z)))
  |> with_pp (fun ppf (a, z) ->
      Format.fprintf ppf "a = %a@ z = %a" Nx.pp a Nx.pp z)

let field a b x = Nx.sub (Nx.add (Nx.matmul a x) (Nx.sin x)) b
let rhs a z = Nx.add (Nx.matmul a z) (Nx.sin z)
let jacobian a x = Nx.add a (Nx.diag (Nx.cos x))
let derivative f x dx = snd (Rune.jvp' f x dx)

let newton ?(linear = Linear.dense) ?(budget = 50) f guess =
  System.solve one
    (System.newton ~derivative:(derivative f))
    ~linear ~tol:tight ~budget f guess

let broyden ?(budget = 100) f guess =
  System.solve one System.broyden ~linear:Linear.dense ~tol:tight ~budget f
    guess

let anderson ?(budget = 200) ~memory f guess =
  System.solve one (System.anderson ~memory) ~linear:Linear.dense ~tol:tight
    ~budget f guess

let near = Oracle.tensor ~rel:1e-9 ~abs:1e-10 ()

(* Newton *)

let newton_tests =
  [
    prop "its zero is the one the system is built around" system (fun (a, z) ->
        cover "an empty system" (Nx.dim 0 z = 0);
        cover "several unknowns" (Nx.dim 0 z > 2);
        let f = field a (rhs a z) in
        equal near z (Solution.get (newton f (Nx.zeros_like z))));
    prop "with gmres it is Newton–Krylov, to the same zero" system
      (fun (a, z) ->
        let f = field a (rhs a z) in
        let linear =
          Linear.gmres ~restart:6 ~rel:1e-13 ~budget:60 ~precondition:Fun.id
        in
        equal near z (Solution.get (newton ~linear f (Nx.zeros_like z))));
    test "a linear system takes one step, then the zero step" (fun () ->
        (* f at the guess, at x + δ in the line search, then δ = 0. *)
        let a = Nx.create f64 [| 2; 2 |] [| 3.; 1.; 1.; 2. |] in
        let b = vec [| 1.; 2. |] in
        let s = newton (fun x -> Nx.sub (Nx.matmul a x) b) (vec [| 0.; 0. |]) in
        equal (Oracle.tensor ~rel:1e-14 ()) (Nx.solve a b) (Solution.get s);
        equal int32 2l (Nx.item [] (Solution.evaluations s)));
    test "x² + 1 = 0 never converges by shrinking its steps" (fun () ->
        let s = newton (fun x -> Nx.add_s (Nx.square x) 1.) (vec [| 0.5 |]) in
        equal bool false (Nx.item [] (Solution.ok s));
        failure_with "Jera.System.solve" (fun () -> Solution.get s));
    test "a guess where f is not finite ends Not_finite" (fun () ->
        let s = newton (fun x -> Nx.sub_s (Nx.sqrt x) 1.) (vec [| -1. |]) in
        equal bool true (is Not_finite s));
    test "a budget of one iteration ends Budget_spent" (fun () ->
        let a = jacobian (Nx.eye f64 2) (vec [| 0.; 0. |]) in
        let f = field a (vec [| 1.; 2. |]) in
        equal bool true
          (is Budget_spent (newton ~budget:1 f (vec [| 0.; 0. |]))));
    test "a structure's float tensors are one vector, others carried" (fun () ->
        (* p + q² = 3 and p q = 1 near (1, 1.4), with a counter carried. *)
        let s = Nx.Ptree.(pair (pair tensor tensor) tensor) in
        let f ((p, q), k) =
          ( ( Nx.sub_s (Nx.add p (Nx.square (Nx.reshape [| 1 |] q))) 3.,
              Nx.sub_s (Nx.mul (Nx.reshape [||] p) q) 1. ),
            k )
        in
        let derivative x dx = snd (Rune.jvp s s f x dx) in
        let (p, q), k =
          Solution.get
            (System.solve s
               (System.newton ~derivative)
               ~linear:Linear.dense ~tol:tight ~budget:50 f
               ((vec [| 1. |], scalar 1.4), Nx.scalar Nx.int32 7l))
        in
        equal
          (Oracle.tensor ~abs:1e-12 ())
          (vec [| 0. |])
          (Nx.sub_s (Nx.add p (Nx.square (Nx.reshape [| 1 |] q))) 3.);
        equal
          (Oracle.tensor ~abs:1e-12 ())
          (scalar 0.)
          (Nx.sub_s (Nx.mul (Nx.reshape [||] p) q) 1.);
        equal int32 7l (Nx.item [] k));
    test "a budget below 1 raises" (fun () ->
        invalid_with "Jera.System.solve: budget = 0 is below 1" (fun () ->
            newton ~budget:0 Fun.id (vec [| 1. |])));
    test "f of another shape raises" (fun () ->
        invalid_with "Jera.System.solve: f returned a value of another"
          (fun () ->
            newton (fun x -> Nx.concatenate ~axis:0 [ x; x ]) (vec [| 1. |])));
  ]

(* Broyden *)

let broyden_tests =
  [
    prop "its zero is the one the system is built around" system (fun (a, z) ->
        let f = field a (rhs a z) in
        equal near z (Solution.get (broyden f (Nx.zeros_like z))));
    test "its derivative is rune's, whatever the differences" (fun () ->
        let a = jacobian (Nx.eye f64 3) (Nx.zeros f64 [| 3 |]) in
        let solve b =
          Solution.get (broyden (field a b) (Nx.zeros f64 [| 3 |]))
        in
        let b = vec [| 1.; -2.; 0.5 |] in
        let x = solve b in
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.solve (Nx.transpose (jacobian a x)) (Nx.ones f64 [| 3 |]))
          (Rune.grad' (fun b -> Nx.sum (solve b)) b));
  ]

(* Anderson *)

(* The fixed point [z] of [g x = sin x / 2 + c], a contraction by 1/2. *)
let fixed z =
  let c = Nx.sub z (Nx.mul_s (Nx.sin z) 0.5) in
  fun x -> Nx.sub (Nx.add (Nx.mul_s (Nx.sin x) 0.5) c) x

let anderson_tests =
  [
    prop "its zero is the fixed point" system (fun (_, z) ->
        List.iter
          (fun memory ->
            equal near z
              (Solution.get (anderson ~memory (fixed z) (Nx.zeros_like z))))
          [ 0; 1; 3 ]);
    test "mixing takes fewer evaluations than Picard iteration" (fun () ->
        let z = vec [| 1.; -0.5; 2.; 0.3 |] in
        let count memory =
          Nx.item []
            (Solution.evaluations
               (anderson ~memory (fixed z) (Nx.zeros_like z)))
        in
        less int32 ~than:(count 0) (count 3));
    test "an equilibrium's derivative is central differences'" (fun () ->
        (* x = tanh (w x + θ), a contraction for a small w. *)
        let w =
          Nx.create f64 [| 3; 3 |]
            [| 0.2; -0.1; 0.3; 0.1; 0.1; -0.2; -0.3; 0.2; 0.1 |]
        in
        let solve theta =
          Solution.get
            (anderson ~memory:3
               (fun x -> Nx.sub (Nx.tanh (Nx.add (Nx.matmul w x) theta)) x)
               (Nx.zeros f64 [| 3 |]))
        in
        let theta = vec [| 0.5; -1.; 0.25 |] and v = vec [| 1.; 2.; -1. |] in
        equal
          (Oracle.tensor ~rel:1e-6 ())
          (Oracle.central ~eps:1e-5 (fun t -> Nx.sum (solve t)) theta v)
          (Nx.sum (Nx.mul (Rune.grad' (fun t -> Nx.sum (solve t)) theta) v)));
    test "a negative memory raises" (fun () ->
        invalid_with "Jera.System.anderson: memory = -1 is negative" (fun () ->
            System.anderson ~memory:(-1)));
  ]

(* Laws *)

let law_tests =
  [
    prop "grad is the implicit derivative J⁻ᵀ 1 (law 1)" system (fun (a, z) ->
        let b = rhs a z in
        let solve b = Solution.get (newton (field a b) (Nx.zeros_like z)) in
        let expected =
          if Nx.dim 0 z = 0 then z
          else Nx.solve (Nx.transpose (jacobian a z)) (Nx.ones_like z)
        in
        equal
          (Oracle.tensor ~rel:1e-8 ~abs:1e-12 ())
          expected
          (Rune.grad' (fun b -> Nx.sum (solve b)) b));
    prop "jvp is the implicit tangent J⁻¹ v (law 1)" system (fun (a, z) ->
        let b = rhs a z in
        let solve b = Solution.get (newton (field a b) (Nx.zeros_like z)) in
        let v = Nx.cos (Nx.arange_f f64 0. (Float.of_int (Nx.dim 0 z)) 1.) in
        let expected =
          if Nx.dim 0 z = 0 then z else Nx.solve (jacobian a z) v
        in
        equal
          (Oracle.tensor ~rel:1e-8 ~abs:1e-12 ())
          expected
          (snd (Rune.jvp' solve b v)));
    prop "a derivative wrong by a factor changes the speed only (law 2)" system
      (fun (a, z) ->
        let f = field a (rhs a z) in
        (* Steps 2 times too short contract by 1/2; 1.25 times too long, by 1/4
           with a line search that may shorten them; 2 times too long never
           contract, and must not converge early. *)
        let converged k =
          let derivative x dx = Nx.mul_s (derivative f x dx) k in
          let s =
            System.solve one
              (System.newton ~derivative)
              ~linear:Linear.dense ~tol:tight ~budget:60 f (Nx.zeros_like z)
          in
          let ok = Nx.item [] (Solution.ok s) in
          if ok then equal near z (Solution.get s);
          ok
        in
        let ok = List.map converged [ 2.; 0.8; 0.5 ] in
        cover "a wrong derivative converged" (List.exists Fun.id ok));
    test "a lane that did not converge has a zero derivative (law 5)" (fun () ->
        let g =
          Rune.grad'
            (fun theta ->
              Nx.sum
                (Solution.best
                   (newton
                      (fun x -> Nx.add (Nx.square x) theta)
                      (vec [| 0.5 |]))))
            (scalar 1.)
        in
        equal (Oracle.tensor ()) (scalar 0.) g);
    test "compiled equals eager (law 3)" (fun () ->
        let a = jacobian (Nx.eye f64 3) (Nx.zeros f64 [| 3 |]) in
        List.iter
          (fun solve ->
            let f b = Solution.get (solve (field a b) (Nx.zeros f64 [| 3 |])) in
            let b = vec [| 1.; -2.; 0.5 |] in
            equal (Oracle.tensor ~rel:1e-14 ()) (f b) (Rune.jit' f b))
          [
            newton ?linear:None ?budget:None;
            broyden ?budget:None;
            anderson ?budget:None ~memory:2;
          ]);
    test "vmap is each lane's solve (law 3)" (fun () ->
        let a = jacobian (Nx.eye f64 2) (Nx.zeros f64 [| 2 |]) in
        let solve b =
          Solution.get (newton (field a b) (Nx.zeros f64 [| 2 |]))
        in
        let bs = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 0.; 0.; -3.; 0.5 |] in
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (Nx.stack (List.init 3 (fun i -> solve (Nx.get [ i ] bs))))
          (Rune.vmap' solve bs));
  ]

let () =
  exit
    (run "Jera.System"
       [
         group "newton" newton_tests;
         group "broyden" broyden_tests;
         group "anderson" anderson_tests;
         group "laws" law_tests;
       ])
