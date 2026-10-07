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

(* [system] with exactly five unknowns. *)
let five =
  let open Gen in
  (let+ a = list ~size:(constant 25) (float_range (-1.) 1.)
   and+ z = list ~size:(constant 5) (float_range (-2.) 2.) in
   let a = Nx.create f64 [| 5; 5 |] (Array.of_list a) in
   ( Nx.add a (Nx.mul_s (Nx.eye f64 5) 7.),
     Nx.create f64 [| 5 |] (Array.of_list z) ))
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
    prop "from a good seed, a zero reached to rounding in two steps converges"
      system (fun (a, z) ->
        (* Seeded within 1e-6 of the zero, quadratic convergence reaches f's
           evaluation level before a third point exists to measure two
           contractions; the next step cannot move the estimate. *)
        cover "several unknowns" (Nx.dim 0 z > 2);
        let f = field a (rhs a z) in
        let seed = Nx.add z (Nx.mul_s (Nx.cos (Nx.mul_s z 7.)) 1e-6) in
        let s =
          System.solve one
            (System.newton ~derivative:(derivative f))
            ~linear:Linear.dense
            ~tol:(Tol.v ~rel:1e-14 ~abs:1e-14)
            ~budget:20 f seed
        in
        equal near z (Solution.get s));
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
    prop
      "steps twice too long, which the line search halves onto the zero, never \
       converge early (law 2)"
      five (fun (a, z) ->
        (* The undamped map x + 2δ does not contract, whatever the damped steps
           do: a solve that converges is at the zero to its tolerance. *)
        let f = field a (rhs a z) in
        let s =
          System.solve one
            (System.newton ~derivative:(fun x dx ->
                 Nx.mul_s (derivative f x dx) 0.5))
            ~linear:Linear.dense
            ~tol:(Tol.v ~rel:1e-12 ~abs:1e-14)
            ~budget:60 f (Nx.zeros_like z)
        in
        if Nx.item [] (Solution.ok s) then
          equal (Oracle.tensor ~rel:1e-10 ~abs:1e-12 ()) z (Solution.get s));
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
    test "compiled newton with a banded solver equals eager" (fun () ->
        (* Bratu's problem on 16 nodes, h = 1/17, below its critical λ. The
           band's probe and factors are taken inside the search's loop. *)
        let second u =
          let n = Nx.dim 0 u in
          let left = Nx.pad [| (1, 0) |] 0. (Nx.slice [ Nx.R (0, n - 1) ] u)
          and right = Nx.pad [| (0, 1) |] 0. (Nx.slice [ Nx.R (1, n) ] u) in
          Nx.sub (Nx.mul_s u 2.) (Nx.add left right)
        in
        let solve lambda =
          let f u =
            Nx.sub (second u) (Nx.mul_s (Nx.mul lambda (Nx.exp u)) (1. /. 289.))
          in
          Solution.get
            (System.solve one
               (System.newton ~derivative:(derivative f))
               ~linear:(Linear.banded ~width:1) ~tol:tight ~budget:30 f
               (Nx.zeros_like lambda))
        in
        let lambda = Nx.linspace f64 1. 1.5 16 in
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (solve lambda) (Rune.jit' solve lambda));
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

(* Lanes *)

(* [l] lanes of systems [a x + sin x = b] of [k] unknowns each, built around
   zeros [z], with the same diagonally dominant [a] as [system]. *)
let lane_systems =
  let open Gen in
  (let* l = int_range 0 5 in
   let* k = int_range 1 3 in
   let+ a = list ~size:(constant (l * k * k)) (float_range (-1.) 1.)
   and+ z = list ~size:(constant (l * k)) (float_range (-2.) 2.) in
   let a = Nx.create f64 [| l; k; k |] (Array.of_list a) in
   let a = Nx.add a (Nx.mul_s (Nx.eye f64 k) (Float.of_int (k + 2))) in
   (a, Nx.create f64 [| l; k |] (Array.of_list z)))
  |> with_pp (fun ppf (a, z) ->
      Format.fprintf ppf "a = %a@ z = %a" Nx.pp a Nx.pp z)

let apply a x =
  Nx.squeeze ~axes:[ -1 ] (Nx.matmul a (Nx.unsqueeze ~axes:[ -1 ] x))

let lane_field a b x = Nx.sub (Nx.add (apply a x) (Nx.sin x)) b

let lane_jacobian a x =
  Nx.add a
    (Nx.mul (Nx.eye f64 (Nx.dim (-1) x)) (Nx.unsqueeze ~axes:[ -1 ] (Nx.cos x)))

let in_lanes ?(budget = 50) a b guess =
  System.lanes ~tol:tight ~budget ~jacobian:(lane_jacobian a) (lane_field a b)
    guess

let lane_tests =
  [
    prop "each lane's zero is the one its system is built around" lane_systems
      (fun (a, z) ->
        cover "no lane" (Nx.dim 0 z = 0);
        cover "several lanes of several unknowns"
          (Nx.dim 0 z > 1 && Nx.dim 1 z > 1);
        let b = apply a z |> Nx.add (Nx.sin z) in
        equal near z (Solution.get (in_lanes a b (Nx.zeros_like z))));
    prop "lanes are vmap of solve per lane, bit for bit (law 3)" lane_systems
      (fun (a, z) ->
        let b = Nx.add (apply a z) (Nx.sin z) in
        let one (a, (b, x0)) =
          let f x = Nx.sub (Nx.add (Nx.matmul a x) (Nx.sin x)) b in
          let jac x = Nx.add a (Nx.mul (Nx.eye f64 (Nx.dim 0 x)) (Nx.cos x)) in
          Solution.get
            (System.solve one
               (System.newton ~derivative:(fun x dx -> Nx.matmul (jac x) dx))
               ~linear:Linear.dense ~tol:tight ~budget:50 f x0)
        in
        if Nx.dim 0 z > 0 then
          equal (Oracle.tensor ())
            (Rune.vmap
               Nx.Ptree.(pair tensor (pair tensor tensor) @-> returns tensor)
               one
               (a, (b, Nx.zeros_like z)))
            (Solution.get (in_lanes a b (Nx.zeros_like z))));
    test "leading axes are lanes, whatever their number" (fun () ->
        let a =
          Nx.broadcast_to [| 2; 3; 2; 2 |]
            (Nx.create f64 [| 2; 2 |] [| 4.; 1.; 1.; 3. |])
        in
        let z = Nx.reshape [| 2; 3; 2 |] (Nx.linspace f64 (-1.) 1. 12) in
        let b = Nx.add (apply a z) (Nx.sin z) in
        equal near z (Solution.get (in_lanes a b (Nx.zeros_like z))));
    test "a lane with no zero fails alone, with a zero derivative" (fun () ->
        (* Lane 1 is x² + 1 = 0 in its first unknown. *)
        let f c x =
          let first = Nx.slice [ Nx.A; Nx.R (0, 1) ] x in
          let lift = Nx.reshape [| 2; 1 |] c in
          Nx.concatenate ~axis:1
            [ Nx.add (Nx.square first) lift; Nx.slice [ Nx.A; Nx.R (1, 2) ] x ]
        in
        let jacobian x =
          let first = Nx.mul_s (Nx.slice [ Nx.A; Nx.I 0 ] x) 2. in
          let zero = Nx.zeros_like first and one = Nx.ones_like first in
          Nx.reshape [| 2; 2; 2 |] (Nx.stack ~axis:1 [ first; zero; zero; one ])
        in
        let solve c =
          System.lanes ~tol:tight ~budget:30 ~jacobian (f c)
            (Nx.create f64 [| 2; 2 |] [| 0.5; 0.5; 0.5; 0.5 |])
        in
        let c = vec [| -4.; 1. |] in
        let s = solve c in
        equal (array bool) [| true; false |] (Nx.to_array (Solution.ok s));
        equal
          (Oracle.tensor ~rel:1e-10 ())
          (vec [| 2.; 0. |])
          (Nx.get [ 0 ] (Solution.best s));
        (* d x / d c = −1 / (2x) in lane 0, zero in the failed lane. *)
        equal
          (Oracle.tensor ~rel:1e-9 ~abs:0. ())
          (vec [| -0.25; 0. |])
          (Rune.grad'
             (fun c ->
               Nx.sum (Nx.slice [ Nx.A; Nx.I 0 ] (Solution.best (solve c))))
             c));
    prop "grad in b is each lane's implicit derivative J⁻ᵀ 1 (law 1)"
      lane_systems (fun (a, z) ->
        let b = Nx.add (apply a z) (Nx.sin z) in
        let expected =
          if Nx.dim 0 z = 0 then z
          else
            Nx.squeeze ~axes:[ -1 ]
              (Nx.solve
                 (Nx.transpose ~axes:[ 0; 2; 1 ] (lane_jacobian a z))
                 (Nx.ones f64 [| Nx.dim 0 z; Nx.dim 1 z; 1 |]))
        in
        equal
          (Oracle.tensor ~rel:1e-8 ~abs:1e-12 ())
          expected
          (Rune.grad'
             (fun b -> Nx.sum (Solution.get (in_lanes a b (Nx.zeros_like z))))
             b));
    test "a field that mixes lanes raises in the derivative" (fun () ->
        let mixed b x =
          Nx.sub (Nx.add x (Nx.mul_s (Nx.flip ~axes:[ 0 ] x) 0.1)) b
        in
        let jacobian x =
          Nx.broadcast_to
            (Array.append (Nx.shape x) [| 1 |])
            (Nx.ones f64 [| 1 |])
        in
        invalid_with "is not per lane" (fun () ->
            Rune.grad'
              (fun b ->
                (* A cotangent that differs between lanes, which a mixing
                   derivative's blocks cannot solve. *)
                Nx.sum
                  (Nx.mul
                     (Nx.create f64 [| 2; 1 |] [| 1.; 2. |])
                     (Solution.best
                        (System.lanes ~tol:tight ~budget:30 ~jacobian (mixed b)
                           (Nx.zeros f64 [| 2; 1 |])))))
              (Nx.create f64 [| 2; 1 |] [| 1.; 2. |])));
    prop "seeded lanes reaching rounding in two steps converge" lane_systems
      (fun (a, z) ->
        let b = Nx.add (apply a z) (Nx.sin z) in
        let seed = Nx.add z (Nx.mul_s (Nx.cos (Nx.mul_s z 7.)) 1e-6) in
        equal near z
          (Solution.get
             (System.lanes
                ~tol:(Tol.v ~rel:1e-14 ~abs:1e-14)
                ~budget:20 ~jacobian:(lane_jacobian a) (lane_field a b) seed)));
    test "a lane whose step at its zero moves one ulp converges" (fun () ->
        (* The seed's step lands two ulps from the zero, the next moves the
           estimate one ulp, and the third cannot move it. The contraction over
           that one-ulp step is a ratio of rounding errors, 1 here. *)
        let a = Nx.create f64 [| 1; 1; 1 |] [| 2.074449998531568 |] in
        let z = Nx.create f64 [| 1; 1 |] [| 0.6661511920166958 |] in
        let b = Nx.add (apply a z) (Nx.sin z) in
        let seed = Nx.add z (Nx.mul_s (Nx.cos (Nx.mul_s z 7.)) 1e-6) in
        equal near z
          (Solution.get
             (System.lanes
                ~tol:(Tol.v ~rel:1e-14 ~abs:1e-14)
                ~budget:20 ~jacobian:(lane_jacobian a) (lane_field a b) seed)));
    test "compiled lanes equal eager" (fun () ->
        let a =
          Nx.broadcast_to [| 3; 2; 2 |]
            (Nx.create f64 [| 2; 2 |] [| 4.; 1.; 1.; 3. |])
        in
        let f b = Solution.get (in_lanes a b (Nx.zeros f64 [| 3; 2 |])) in
        let b = Nx.reshape [| 3; 2 |] (Nx.linspace f64 (-2.) 2. 6) in
        equal (Oracle.tensor ()) (f b) (Rune.jit' f b));
    test "invalid arguments raise" (fun () ->
        invalid_with "Jera.System.lanes: the guess is a scalar" (fun () ->
            System.lanes ~tol:tight ~budget:5 ~jacobian:Fun.id Fun.id
              (scalar 1.));
        invalid_with "Jera.System.lanes: jacobian returned shape [2,1]"
          (fun () ->
            System.lanes ~tol:tight ~budget:5 ~jacobian:Fun.id Fun.id
              (Nx.zeros f64 [| 2; 1 |])));
  ]

let () =
  exit
    (run "Jera.System"
       [
         group "newton" newton_tests;
         group "broyden" broyden_tests;
         group "anderson" anderson_tests;
         group "lanes" lane_tests;
         group "laws" law_tests;
       ])
