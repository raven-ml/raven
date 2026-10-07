(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Root, and the Tol and Solution values every solve returns through. The
   trusted side is closed-form zeros (√a, Kepler's equation at e = 0), the
   implicit function theorem's derivative written by hand, the evaluation bound
   the methods state, and lanes computed one at a time. *)

open Windtrap
open Jera

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let tight = Tol.v ~rel:1e-13 ~abs:1e-15
let raises_with sub f = raises_match (Exn.invalid_arg ~substring:sub) f
let failure_with sub f = raises_match (Exn.failure ~substring:sub) f
let statuses s = Oracle.floats (Nx.cast f64 (Nx.cast Nx.int32 (Solution.ok s)))

let positive =
  Gen.(
    map
      (fun l -> vec (Array.of_list l))
      (list ~size:(int_range 1 6) (float_range 0.01 100.)))
  |> Gen.with_pp Nx.pp

(* Tol *)

let tol_tests =
  [
    test "v rejects a negative tolerance" (fun () ->
        raises_with "rel = -1 is not finite and non-negative" (fun () ->
            Tol.v ~rel:(-1.) ~abs:0.));
    test "v rejects two zeros" (fun () ->
        raises_with "rel and abs are both 0" (fun () -> Tol.v ~rel:0. ~abs:0.));
    test "v rejects a NaN" (fun () ->
        raises_with "abs = nan" (fun () -> Tol.v ~rel:1e-3 ~abs:nan));
    test "ulps rejects zero" (fun () ->
        raises_with "k = 0 is not finite and positive" (fun () -> Tol.ulps 0.));
    test "a tolerance formats its parts" (fun () ->
        equal string "rel 1e-06 abs 1e-10"
          (Format.asprintf "%a" Tol.pp (Tol.v ~rel:1e-6 ~abs:1e-10)));
  ]

(* Bracket *)

let sqrt_bracket a =
  Root.bracket ~tol:tight
    (fun x -> Nx.sub (Nx.mul x x) a)
    ~lo:(Nx.zeros_like a) ~hi:(Nx.add_s a 1.)

let bracket_tests =
  [
    prop "the zero of x² − a is √a" positive (fun a ->
        let s = sqrt_bracket a in
        equal (Oracle.tensor ~rel:2e-13 ()) (Nx.sqrt a) (Solution.get s));
    prop "an element ends within 2b + 2 evaluations" positive (fun a ->
        (* A flat and a steep function, whose interpolation steps help least. *)
        let flat =
          Root.bracket ~tol:(Tol.ulps 1.)
            (fun x -> Nx.pow_s (Nx.sub x a) 9.)
            ~lo:(Nx.neg a) ~hi:(Nx.mul_s a 3.)
        in
        let steep =
          Root.bracket ~tol:(Tol.ulps 1.)
            (fun x -> Nx.tanh (Nx.mul_s (Nx.sub x a) 1e6))
            ~lo:(Nx.neg a) ~hi:(Nx.mul_s a 3.)
        in
        List.iter
          (fun s ->
            Array.iter
              (fun n -> at_most int32 ~than:130l n)
              (Nx.to_array (Solution.evaluations s)))
          [ flat; steep ]);
    test "ends of one sign are not bracketed" (fun () ->
        let s =
          Root.bracket ~tol:tight
            (fun x -> Nx.add_s (Nx.square x) 1.)
            ~lo:(vec [| -1. |]) ~hi:(vec [| 2. |])
        in
        equal (Oracle.tensor ()) (Nx.ones Nx.bool [| 1 |])
          (Solution.is Not_bracketed s));
    test "a pole stalls" (fun () ->
        let s =
          Root.bracket ~tol:tight
            (fun x -> Nx.recip (Nx.sub_s x 0.3))
            ~lo:(vec [| -1. |]) ~hi:(vec [| 2. |])
        in
        equal (Oracle.tensor ()) (Nx.ones Nx.bool [| 1 |])
          (Solution.is Stalled s));
    test "a jump converges at the jump" (fun () ->
        let s =
          Root.bracket ~tol:tight Nx.sign ~lo:(vec [| -1. |]) ~hi:(vec [| 2. |])
        in
        equal (Oracle.tensor ~abs:1e-300 ()) (vec [| 0. |]) (Solution.get s));
    test "a NaN value is not finite" (fun () ->
        let s =
          Root.bracket ~tol:tight
            (fun x -> Nx.sub (Nx.sqrt x) (Nx.full_like x 0.5))
            ~lo:(vec [| -1. |]) ~hi:(vec [| 2. |])
        in
        equal (Oracle.tensor ()) (Nx.ones Nx.bool [| 1 |])
          (Solution.is Not_finite s));
    test "a zero at an end converges there" (fun () ->
        let s =
          Root.bracket ~tol:tight
            (fun x -> Nx.sub_s x 2.)
            ~lo:(vec [| 2. |]) ~hi:(vec [| 5. |])
        in
        equal (Oracle.tensor ()) (vec [| 2. |]) (Solution.get s));
    test "the ends come in either order" (fun () ->
        equal
          (Oracle.tensor ~rel:2e-13 ())
          (vec [| Float.sqrt 2. |])
          (Solution.get
             (Root.bracket ~tol:tight
                (fun x -> Nx.sub_s (Nx.square x) 2.)
                ~lo:(vec [| 3. |]) ~hi:(vec [| 0. |]))));
    test "float32 converges to float32's tolerance" (fun () ->
        let a = Nx.create Nx.float32 [| 3 |] [| 2.; 3.; 10. |] in
        let s =
          Root.bracket ~tol:(Tol.ulps 4.)
            (fun x -> Nx.sub (Nx.mul x x) a)
            ~lo:(Nx.zeros_like a) ~hi:a
        in
        equal (Oracle.tensor ~rel:1e-6 ()) (Nx.sqrt a) (Solution.get s));
    test "a zero-size problem is an empty answer" (fun () ->
        equal (Oracle.tensor ()) (vec [||])
          (Solution.get (sqrt_bracket (vec [||]))));
    test "changing one element changes no other" (fun () ->
        let a = vec [| 2.; 5.; 7. |] and b = vec [| 2.; 500.; 7. |] in
        let sa = sqrt_bracket a and sb = sqrt_bracket b in
        let pick i t =
          Nx.slice [ Nx.L [ 0; 2 ] ] t |> fun t ->
          ignore i;
          t
        in
        equal ~msg:"values" (Oracle.tensor ())
          (pick 0 (Solution.best sa))
          (pick 0 (Solution.best sb));
        equal ~msg:"evaluations" (Oracle.tensor ())
          (pick 0 (Solution.evaluations sa))
          (pick 0 (Solution.evaluations sb)));
  ]

(* Derivatives *)

let derivative_tests =
  [
    prop "grad of √a by bracket is 1 / 2√a" positive (fun a ->
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.div (Nx.full_like a 0.5) (Nx.sqrt a))
          (Rune.grad' (fun a -> Nx.sum (Solution.get (sqrt_bracket a))) a));
    prop "jvp of √a by bracket is 1 / 2√a" positive (fun a ->
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.div (Nx.full_like a 0.5) (Nx.sqrt a))
          (snd
             (Rune.jvp'
                (fun a -> Solution.get (sqrt_bracket a))
                a (Nx.ones_like a))));
    test "an element that did not converge has a zero derivative" (fun () ->
        (* x² − a on [0, a + 1] brackets for a > 0 and not for a < 0. *)
        let a = vec [| 4.; -1.; 9. |] in
        let g =
          Rune.grad' (fun a -> Nx.sum (Solution.best (sqrt_bracket a))) a
        in
        equal (Oracle.tensor ~rel:1e-9 ()) (vec [| 0.25; 0.; 1. /. 6. |]) g);
    test "a function that mixes elements raises in the derivative" (fun () ->
        let a = vec [| 4.; 9. |] in
        (* Each element also reads the sum of all, slightly. *)
        let f a x =
          Nx.add (Nx.sub (Nx.pow_s x 3.) a) (Nx.mul_s (Nx.sum x) 1e-3)
        in
        let solve a =
          Solution.best
            (Root.bracket ~tol:tight (f a) ~lo:(Nx.zeros_like a) ~hi:a)
        in
        raises_with "is not elementwise" (fun () ->
            Rune.grad' (fun a -> Nx.sum (solve a)) a));
    test "compiled equals eager" (fun () ->
        let a = vec [| 0.5; 2.; 30. |] in
        let f a = Solution.get (sqrt_bracket a) in
        equal (Oracle.tensor ()) (f a) (Rune.jit' f a));
    test "compiled grad equals eager grad" (fun () ->
        let a = vec [| 0.5; 2.; 30. |] in
        let g = Rune.grad' (fun a -> Nx.sum (Solution.get (sqrt_bracket a))) in
        equal (Oracle.tensor ~rel:1e-14 ()) (g a) (Rune.jit' g a));
    test "vmap is each lane's solve" (fun () ->
        let a = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let f a = Solution.get (sqrt_bracket a) in
        equal (Oracle.tensor ()) (f a) (Rune.vmap' f a));
  ]

(* Newton *)

(* E − e sin E = M. *)
let kepler ?(budget = 30) ?(factor = 1.) ~e m =
  Root.newton
    ~tol:(Tol.v ~rel:1e-13 ~abs:1e-15)
    ~budget
    ~slope:(fun x -> Nx.mul_s (Nx.rsub_s 1. (Nx.mul e (Nx.cos x))) factor)
    (fun x -> Nx.sub (Nx.sub x (Nx.mul e (Nx.sin x))) m)
    m

let newton_tests =
  let m = vec [| 0.; 0.3; 1.; 2.5; 3.1 |] in
  let e = Nx.full f64 [| 5 |] 0.6 in
  [
    prop "from a good seed, a zero reached to rounding converges" positive
      (fun a ->
        (* √a seeded within 1e-7: Newton reaches the floats' resolution in two
           steps. *)
        let seed =
          Nx.mul (Nx.sqrt a) (Nx.add_s (Nx.mul_s (Nx.cos a) 1e-7) 1.)
        in
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (Nx.sqrt a)
          (Solution.get
             (Root.newton ~tol:(Tol.ulps 4.) ~budget:20
                ~slope:(fun x -> Nx.mul_s x 2.)
                (fun x -> Nx.sub (Nx.square x) a)
                seed)));
    test "a slope 1000 times too steep never understates the error" (fun () ->
        (* Steps 1000 times too short contract by 0.999 and fall within an ulp
           of √2 while the estimate is still about 1000 ulps from it. *)
        let z = Float.sqrt 2. in
        let s =
          Root.newton ~tol:tight ~budget:20000
            ~slope:(fun x -> Nx.mul_s x 2000.)
            (fun x -> Nx.sub_s (Nx.square x) 2.)
            (vec [| z +. 1e-8 |])
        in
        let x = Nx.item [ 0 ] (Solution.get s) in
        at_least float_exact
          ~than:(Float.abs (x -. z))
          (Nx.item [ 0 ] (Solution.error s)));
    test "a solve satisfies Kepler's equation" (fun () ->
        let x = Solution.get (kepler ~e m) in
        equal (Oracle.tensor ~abs:1e-14 ()) m (Nx.sub x (Nx.mul e (Nx.sin x))));
    test "a slope wrong by a constant factor converges to the same zero"
      (fun () ->
        let x = Solution.get (kepler ~e m) in
        let y = Solution.get (kepler ~budget:200 ~factor:3. ~e m) in
        equal (Oracle.tensor ~rel:1e-12 ~abs:1e-14 ()) x y);
    test "a slope wrong by a constant factor takes more steps" (fun () ->
        let n = Solution.evaluations (kepler ~e m)
        and n3 = Solution.evaluations (kepler ~budget:200 ~factor:3. ~e m) in
        Array.iteri
          (fun i k -> at_least int32 ~than:(Nx.item [ i ] n) k)
          (Nx.to_array n3));
    test "grad is the implicit derivative 1 / (1 − e cos E)" (fun () ->
        let x = Solution.get (kepler ~e m) in
        equal
          (Oracle.tensor ~rel:1e-10 ())
          (Nx.recip (Nx.rsub_s 1. (Nx.mul e (Nx.cos x))))
          (Rune.grad' (fun m -> Nx.sum (Solution.get (kepler ~e m))) m));
    test "grad in a captured parameter is the implicit derivative" (fun () ->
        let x = Solution.get (kepler ~e m) in
        equal
          (Oracle.tensor ~rel:1e-10 ~abs:1e-15 ())
          (Nx.div (Nx.sin x) (Nx.rsub_s 1. (Nx.mul e (Nx.cos x))))
          (Rune.grad' (fun e -> Nx.sum (Solution.get (kepler ~e m))) e));
    test "the budget ends a solve" (fun () ->
        let s =
          Root.newton ~tol:tight ~budget:2
            ~slope:(fun x -> Nx.mul_s x 2.)
            (fun x -> Nx.sub_s (Nx.square x) 2.)
            (vec [| 10. |])
        in
        equal (Oracle.tensor ()) (Nx.ones Nx.bool [| 1 |])
          (Solution.is Budget_spent s));
    test "a zero slope stalls" (fun () ->
        let s =
          Root.newton ~tol:tight ~budget:10
            ~slope:(fun x -> Nx.mul_s x 2.)
            (fun x -> Nx.add_s (Nx.square x) 1.)
            (vec [| 0. |])
        in
        equal (Oracle.tensor ()) (Nx.ones Nx.bool [| 1 |])
          (Solution.is Stalled s));
    test "a NaN value is not finite" (fun () ->
        let s =
          Root.newton ~tol:tight ~budget:10 ~slope:Nx.recip Nx.log
            (vec [| -1. |])
        in
        equal (Oracle.tensor ()) (Nx.ones Nx.bool [| 1 |])
          (Solution.is Not_finite s));
    test "a step at the zero is zero, which is the zero" (fun () ->
        let s =
          Root.newton ~tol:tight ~budget:3 ~slope:Nx.ones_like
            (fun x -> Nx.sub_s x 2.)
            (vec [| 2. |])
        in
        equal (Oracle.tensor ()) (vec [| 2. |]) (Solution.get s));
    test "newton rejects a budget below 1" (fun () ->
        raises_with "budget = 0 is below 1" (fun () ->
            Root.newton ~tol:tight ~budget:0 ~slope:Nx.ones_like Fun.id
              (vec [| 1. |])));
    test "compiled equals eager" (fun () ->
        let f m = Solution.get (kepler ~e m) in
        equal (Oracle.tensor ()) (f m) (Rune.jit' f m));
  ]

(* Solution *)

let solution_tests =
  let mixed () = sqrt_bracket (vec [| 4.; -1.; 9. |]) in
  [
    test "get raises the first failing lane's report" (fun () ->
        failure_with
          "Jera.Root.bracket: lane [1]: the ends do not bracket a zero"
          (fun () -> Solution.get (mixed ())));
    test "the report prints the lane's data" (fun () ->
        failure_with "lo 0, hi 0, f lo 1, f hi 1, estimate 0" (fun () ->
            Solution.get (mixed ())));
    test "the report counts the other lanes that converged" (fun () ->
        failure_with "2 other elements of this problem converged." (fun () ->
            Solution.get (mixed ())));
    test "a compiled get raises when the call returns" (fun () ->
        failure_with "lane [1]" (fun () ->
            Rune.jit'
              (fun a -> Solution.get (sqrt_bracket a))
              (vec [| 4.; -1.; 9. |])));
    test "ok marks the converged lanes" (fun () ->
        equal
          (array (Windtrap.float 0.5))
          [| 1.; 0.; 1. |]
          (statuses (mixed ())));
    test "best holds every lane" (fun () ->
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (vec [| 2.; 0.; 3. |])
          (Solution.best (mixed ())));
    test "an answer is a compiled function's result" (fun () ->
        let f =
          Rune.jit
            Nx.Ptree.(tensor @-> returns (Solution.ptree tensor))
            sqrt_bracket
        in
        let a = vec [| 4.; -1. |] in
        equal (Oracle.tensor ())
          (Solution.best (sqrt_bracket a))
          (Solution.best (f a));
        failure_with "lane [1]" (fun () -> Solution.get (f a)));
    test "pp counts the lanes and reports the first failing one" (fun () ->
        equal text
          "Jera.Root.bracket: 2 converged, 1 not bracketed\n\
           Jera.Root.bracket: lane [1]: the ends do not bracket a zero: f has \
           one sign at both.\n\
          \  tol rel 1e-13 abs 1e-15\n\
          \  lo 0, hi 0, f lo 1, f hi 1, estimate 0\n\
          \  2 evaluations.\n\
          \  Widen [lo, hi] until f changes sign between them.\n\
          \  2 other elements of this problem converged."
          (Format.asprintf "%a" Solution.pp (mixed ())));
  ]

let () =
  exit
    (run "Jera.Root"
       [
         group "tol" tol_tests;
         group "bracket" bracket_tests;
         group "derivatives" derivative_tests;
         group "newton" newton_tests;
         group "solution" solution_tests;
       ])
