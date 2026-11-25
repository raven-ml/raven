(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Root, and the Tol and Solution values every solve returns through. The
   trusted side is closed-form zeros (√a, Kepler's equation at a known anomaly),
   the implicit function theorem's derivative written by hand, the bounds the
   methods state, and lanes computed one at a time. *)

open Windtrap
open Jera

let f64 = Nx.float64
let vec a = Nx.create f64 [| Array.length a |] a
let tight = Tol.v ~rel:1e-13 ~abs:1e-15
let raises_from fn f = raises_match (Exn.invalid_arg ~sub:fn) f
let failure_with sub f = raises_match (Exn.failure ~sub) f
let printed tol = Format.asprintf "%a" Tol.pp tol

(* A vector of one to six positive numbers. *)
let positive =
  Gen.(
    map
      (fun l -> vec (Array.of_list l))
      (list ~size:(int_range 1 6) (float_range 0.01 100.)))
  |> Gen.with_pp Nx.pp

(* √a as the zero of x² − a in [0, a + 1]. *)
let sqrt_bracket a =
  Root.bracket ~tol:tight
    (fun x -> Nx.sub (Nx.mul x x) a)
    ~lo:(Nx.zeros_like a) ~hi:(Nx.add_s a 1.)

(* Three lanes of x² − a, the middle one with no zero in [0, a + 1]. *)
let mixed () = sqrt_bracket (vec [| 4.; -1.; 9. |])

(* Kepler's equation E − e sin E = M, from E = M, with a slope [factor] times
   its derivative. *)
let kepler ?(budget = 30) ?(factor = 1.) ~e m =
  Root.newton
    ~tol:(Tol.v ~rel:1e-13 ~abs:1e-15)
    ~budget
    ~slope:(fun x -> Nx.mul_s (Nx.rsub_s 1. (Nx.mul e (Nx.cos x))) factor)
    (fun x -> Nx.sub (Nx.sub x (Nx.mul e (Nx.sin x))) m)
    m

let anomalies = vec [| 0.; 0.3; 1.; 2.5; 3.1 |]
let eccentricity = Nx.full f64 [| 5 |] 0.6

(* Tol *)

let tol_tests =
  [
    cases ~id:fst "v raises on a negative, non-finite or doubly zero tolerance"
      [
        ("rel -1", (-1., 0.));
        ("abs -1", (0., -1.));
        ("rel nan", (nan, 1.));
        ("abs inf", (1e-3, infinity));
        ("both 0", (0., 0.));
        ("both -0", (-0., -0.));
      ]
      (fun (_, (rel, abs)) ->
        raises_from "Jera.Tol" (fun () -> Tol.v ~rel ~abs));
    test "rel and abs raise as v does" (fun () ->
        raises_from "Jera.Tol" (fun () -> Tol.rel 0.);
        raises_from "Jera.Tol" (fun () -> Tol.rel (-1.));
        raises_from "Jera.Tol" (fun () -> Tol.abs nan));
    cases ~id:fst "ulps raises unless k is finite and positive"
      [ ("0", 0.); ("-1", -1.); ("nan", nan); ("inf", infinity) ]
      (fun (_, k) -> raises_from "Jera.Tol.ulps" (fun () -> Tol.ulps k));
    cases ~id:fst "pp writes each number as its shortest decimal"
      [
        ("rel 1e-06 abs 1e-10", Tol.v ~rel:1e-6 ~abs:1e-10);
        ("ulps 4", Tol.ulps 4.);
        ("rel 0 abs 1000000", Tol.abs 1e6);
        (* The C library's [%g] writes 1e+06, and 1e+006 under mingw-w64. *)
        ("rel 4.58e-72 abs 1000000", Tol.v ~rel:4.58e-72 ~abs:1e6);
        (* −0 is not negative. *)
        ("rel -0 abs 0.001", Tol.v ~rel:(-0.) ~abs:1e-3);
      ]
      (fun (expected, tol) -> equal string expected (printed tol));
  ]

(* Solution *)

let solution_tests =
  [
    test "get raises the report of the first failing lane in C order" (fun () ->
        (* x² − a fails where a < 0: at [0, 1] and [1, 0]. *)
        let a = Nx.create f64 [| 2; 2 |] [| 4.; -1.; -1.; 9. |] in
        failure_with "Jera.Root.bracket: lane [0, 1]:" (fun () ->
            Solution.get (sqrt_bracket a)));
    test "a compiled get raises when the call returns" (fun () ->
        Compiled.here ();
        failure_with "lane [1]" (fun () ->
            Rune.jit'
              (fun a -> Solution.get (sqrt_bracket a))
              (vec [| 4.; -1.; 9. |])));
    test "best holds every lane" (fun () ->
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (vec [| 2.; 0.; 3. |])
          (Solution.best (mixed ())));
    test "a failed lane's best is its last finite estimate" (fun () ->
        (* The slope 1e-300 sends the step from 0 to −∞, where f is −∞. *)
        let s =
          Root.newton ~tol:tight ~budget:5
            ~slope:(fun x -> Nx.full_like x 1e-300)
            (fun x -> Nx.add_s x 1e300)
            (vec [| 0. |])
        in
        equal Oracle.status_t Not_finite (Oracle.status s);
        equal (Oracle.tensor ()) (vec [| 0. |]) (Solution.best s));
    test
      "a step that overflows where f is finite ends Stalled, at a finite best"
      (fun () ->
        (* The slope 1e-310 sends the step from 0 to ∞, where tanh is 1. *)
        let s =
          Root.newton ~tol:tight ~budget:5
            ~slope:(fun x -> Nx.full_like x 1e-310)
            (fun x -> Nx.sub_s (Nx.tanh x) 0.5)
            (vec [| 0. |])
        in
        equal Oracle.status_t Stalled (Oracle.status s);
        equal (Oracle.tensor ()) (vec [| 0. |]) (Solution.best s));
    test "ok marks the converged lanes" (fun () ->
        equal (Oracle.tensor ())
          (Nx.create Nx.bool [| 3 |] [| true; false; true |])
          (Solution.ok (mixed ())));
    test "is and evaluations have the lanes' shape" (fun () ->
        let bracket = sqrt_bracket (Nx.ones f64 [| 2; 3 |]) in
        let system =
          System.solve Nx.Ptree.tensor System.broyden ~linear:Linear.dense
            ~tol:tight ~budget:10
            (fun x -> Nx.sub_s x 1.)
            (vec [| 0.; 0. |])
        in
        let shapes s =
          (Nx.shape (Solution.is Converged s), Nx.shape (Solution.evaluations s))
        in
        let both = pair (array int) (array int) in
        equal ~msg:"bracket" both ([| 2; 3 |], [| 2; 3 |]) (shapes bracket);
        equal ~msg:"System.solve" both ([||], [||]) (shapes system));
    test "a compiled function returns the eager answer" (fun () ->
        Compiled.here ();
        let f =
          Rune.jit
            Nx.Ptree.(tensor @-> returns (Solution.ptree tensor))
            sqrt_bracket
        in
        let a = vec [| 4.; -1. |] in
        let eager = sqrt_bracket a and compiled = f a in
        let read s =
          ( Solution.best s,
            (Solution.error s, (Solution.ok s, Solution.evaluations s)) )
        in
        let answer =
          Oracle.structure
            Nx.Ptree.(pair tensor (pair tensor (pair tensor tensor)))
        in
        equal answer (read eager) (read compiled));
    test "pp counts the lanes in each status, then get's report" (fun () ->
        (* √2 from 10 in two of Newton's steps, beside 2 from its zero. *)
        let a = vec [| 2.; 4. |] in
        let s =
          Root.newton ~tol:tight ~budget:2
            ~slope:(fun x -> Nx.mul_s x 2.)
            (fun x -> Nx.sub (Nx.square x) a)
            (vec [| 10.; 2. |])
        in
        expect (Format.asprintf "%a" Solution.pp s)
        @@ __POS_OF__
             {|
          Jera.Root.newton: 1 converged, 1 budget spent
          Jera.Root.newton: lane [0]: the budget is spent.
            tol rel 1e-13 abs 1e-15, budget 2
            start 10, estimate 2.7460784313725486, error 4.530188679245284
            2 of 2 iterations, 3 evaluations.
            Raise the budget, start nearer the zero, or bracket it with Root.bracket.
            1 other element of this problem converged.
          |});
    test "pp of an answer that converged everywhere counts its lanes" (fun () ->
        expect
          (Format.asprintf "%a" Solution.pp (sqrt_bracket (vec [| 4.; 9. |])))
        @@ __POS_OF__ {| Jera.Root.bracket: 2 converged |});
  ]

(* Root *)

let root_tests =
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
    test "a zero where f' is 0 has no finite derivative" (fun () ->
        (* x³ − a at a = 0 on [−1, 1], whose first interpolation lands on the
           zero 0 exactly: its implicit derivative is 1 / 3x². *)
        let g =
          Rune.grad'
            (fun a ->
              Nx.sum
                (Solution.get
                   (Root.bracket ~tol:tight
                      (fun x -> Nx.sub (Nx.pow_s x 3.) a)
                      ~lo:(Nx.sub_s a 1.) ~hi:(Nx.add_s a 1.))))
            (vec [| 0. |])
        in
        satisfies ~claim:"is not finite" float_exact
          (fun g -> not (Float.is_finite g))
          (Nx.item [ 0 ] g));
    test "compiled grad equals eager grad" (fun () ->
        Compiled.here ();
        let a = vec [| 0.5; 2.; 30. |] in
        let g = Rune.grad' (fun a -> Nx.sum (Solution.get (sqrt_bracket a))) in
        equal (Oracle.tensor ~rel:1e-14 ()) (g a) (Rune.jit' g a));
    test "vmap is each lane's solve" (fun () ->
        let a = Nx.create f64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
        let f a = Solution.get (sqrt_bracket a) in
        equal (Oracle.tensor ()) (f a) (Rune.vmap' f a));
  ]

(* Bracket *)

(* A float dtype and its width in bits. *)
type dtype = Dtype : (float, 'b) Nx.dtype -> dtype

let dtypes =
  Nx.
    [
      Dtype float64;
      Dtype float32;
      Dtype float16;
      Dtype bfloat16;
      Dtype float8_e4m3fn;
      Dtype float8_e5m2;
      Dtype float4_e2m1fn;
    ]

let pp_dtype ppf (Dtype d) = Format.pp_print_string ppf (Nx.Dtype.name d)

(* The evaluations of a flat and a steep function's brackets, whose
   interpolation steps help least, in dtype [d]. *)
let flat_and_steep (Dtype d) a =
  let a = Nx.cast d a in
  let solve f =
    Root.bracket ~tol:(Tol.ulps 1.) f ~lo:(Nx.neg a) ~hi:(Nx.mul_s a 3.)
  in
  let flat = solve (fun x -> Nx.pow_s (Nx.sub x a) 9.)
  and steep = solve (fun x -> Nx.tanh (Nx.mul_s (Nx.sub x a) 1e6)) in
  ( Nx.Dtype.bits d,
    Array.append
      (Nx.to_array (Solution.evaluations flat))
      (Nx.to_array (Solution.evaluations steep)) )

let bracket_tests =
  [
    prop "the zero of x² − a is √a" positive (fun a ->
        (* Within the bracket's half-width, under rel 1e-13, of the zero, and
           Nx.sqrt within half an ulp of it. *)
        equal
          (Oracle.tensor ~rel:2e-13 ())
          (Nx.sqrt a)
          (Solution.get (sqrt_bracket a)));
    prop "the zero lies within twice the error of the estimate" positive
      (fun a ->
        (* The estimate is an end of the final bracket, which holds the zero of
           x² − a as the floats compute it: within an ulp of √a, and Nx.sqrt
           within half of one more. *)
        let s = sqrt_bracket a in
        let x = Oracle.floats (Solution.best s)
        and e = Oracle.floats (Solution.error s)
        and z = Oracle.floats (Nx.sqrt a) in
        Array.iteri
          (fun i x ->
            at_most float_exact
              ~than:((2. *. e.(i)) +. (2. *. epsilon_float *. z.(i)))
              (Float.abs (x -. z.(i))))
          x);
    prop "an element ends within 2b + 2 evaluations for a b-bit dtype"
      ~examples:(List.map (fun d -> (vec [| 2. |], d)) dtypes)
      Gen.(pair positive (with_pp pp_dtype (of_list dtypes)))
      (fun (a, d) ->
        let bits, counts = flat_and_steep d a in
        Array.iter
          (fun n -> at_most int32 ~than:(Int32.of_int ((2 * bits) + 2)) n)
          counts);
    test "a tolerance finer than the floats ends Stalled" (fun () ->
        (* The bracket closes on two adjacent floats around √2, half an ulp
           apart from its midpoint, far above 1e-300. *)
        let s =
          Root.bracket ~tol:(Tol.abs 1e-300)
            (fun x -> Nx.sub_s (Nx.square x) 2.)
            ~lo:(vec [| 0. |]) ~hi:(vec [| 2. |])
        in
        equal Oracle.status_t Stalled (Oracle.status s));
    test "ends of one sign are not bracketed" (fun () ->
        let s =
          Root.bracket ~tol:tight
            (fun x -> Nx.add_s (Nx.square x) 1.)
            ~lo:(vec [| -1. |]) ~hi:(vec [| 2. |])
        in
        equal Oracle.status_t Not_bracketed (Oracle.status s));
    test "a pole stalls" (fun () ->
        let s =
          Root.bracket ~tol:tight
            (fun x -> Nx.recip (Nx.sub_s x 0.3))
            ~lo:(vec [| -1. |]) ~hi:(vec [| 2. |])
        in
        equal Oracle.status_t Stalled (Oracle.status s));
    test "a jump converges at the jump" (fun () ->
        let s =
          Root.bracket ~tol:tight Nx.sign ~lo:(vec [| -1. |]) ~hi:(vec [| 2. |])
        in
        equal (Oracle.tensor ~abs:1e-300 ()) (vec [| 0. |]) (Solution.get s));
    test "a value at an end that is not finite ends Not_finite" (fun () ->
        let s =
          Root.bracket ~tol:tight
            (fun x -> Nx.sub_s (Nx.sqrt x) 0.5)
            ~lo:(vec [| -1. |]) ~hi:(vec [| 2. |])
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    test "a value inside that is not finite ends Not_finite" (fun () ->
        (* x, finite at −2 and 3 and NaN in (−1, 1), where its zero lies. *)
        let s =
          Root.bracket ~tol:tight
            (fun x ->
              Nx.add x (Nx.mul_s (Nx.log (Nx.sub_s (Nx.square x) 1.)) 0.))
            ~lo:(vec [| -2. |]) ~hi:(vec [| 3. |])
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    cases ~id:fst "a zero at an end converges there, with no error"
      [ ("lo", (2., 5.)); ("hi", (-1., 2.)) ]
      (fun (_, (lo, hi)) ->
        let s =
          Root.bracket ~tol:tight
            (fun x -> Nx.sub_s x 2.)
            ~lo:(vec [| lo |]) ~hi:(vec [| hi |])
        in
        equal (Oracle.tensor ()) (vec [| 2. |]) (Solution.get s);
        equal (Oracle.tensor ()) (vec [| 0. |]) (Solution.error s));
    test "the ends broadcast together" (fun () ->
        equal
          (Oracle.tensor ~rel:2e-13 ())
          (Nx.full f64 [| 3 |] (Float.sqrt 2.))
          (Solution.get
             (Root.bracket ~tol:tight
                (fun x -> Nx.sub_s (Nx.square x) 2.)
                ~lo:(Nx.scalar f64 0.)
                ~hi:(vec [| 2.; 3.; 4. |]))));
    test "the ends come in either order" (fun () ->
        equal
          (Oracle.tensor ~rel:2e-13 ())
          (vec [| Float.sqrt 2. |])
          (Solution.get
             (Root.bracket ~tol:tight
                (fun x -> Nx.sub_s (Nx.square x) 2.)
                ~lo:(vec [| 3. |]) ~hi:(vec [| 0. |]))));
    test "Tol.ulps meets float32's epsilon" (fun () ->
        (* Within 4 ulps of float32, and float32's √a within half of one
           more. *)
        let a = Nx.create Nx.float32 [| 3 |] [| 2.; 3.; 10. |] in
        let s =
          Root.bracket ~tol:(Tol.ulps 4.)
            (fun x -> Nx.sub (Nx.mul x x) a)
            ~lo:(Nx.zeros_like a) ~hi:a
        in
        equal
          (Oracle.tensor ~rel:(4.5 *. Float.ldexp 1. (-23)) ())
          (Nx.sqrt a) (Solution.get s));
    test "a zero-size problem is an empty answer" (fun () ->
        equal (Oracle.tensor ()) (vec [||])
          (Solution.get (sqrt_bracket (vec [||]))));
    test "an element's answer and evaluations depend on its data alone"
      (fun () ->
        let a = vec [| 2.; 5.; 7. |] and b = vec [| 2.; 500.; 7. |] in
        let sa = sqrt_bracket a and sb = sqrt_bracket b in
        let others s =
          let at t = Nx.slice [ Nx.L [ 0; 2 ] ] t in
          (at (Solution.best s), at (Solution.evaluations s))
        in
        equal
          (Oracle.structure Nx.Ptree.(pair tensor tensor))
          (others sa) (others sb));
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
        raises_from "Jera.Root.bracket" (fun () ->
            Rune.grad' (fun a -> Nx.sum (solve a)) a));
    test "compiled equals eager, bit for bit" (fun () ->
        Compiled.here ();
        let a = vec [| 0.5; 2.; 30. |] in
        let f a = Solution.get (sqrt_bracket a) in
        equal (Oracle.tensor ()) (f a) (Rune.jit' f a));
    test "ends that do not broadcast raise" (fun () ->
        raises_from "Jera.Root.bracket" (fun () ->
            Root.bracket ~tol:tight Fun.id
              ~lo:(vec [| 0.; 1. |])
              ~hi:(vec [| 1.; 2.; 3. |])));
  ]

(* Newton *)

(* [positive], a factor by which a slope is too steep, and a tolerance in
   ulps. *)
let slowed =
  Gen.(
    with_pp
      (fun ppf (a, (factor, k)) ->
        Format.fprintf ppf "a = %a@ factor = %g@ k = %g" Nx.pp a factor k)
      (pair positive (pair (float_range 1. 12.) (float_range 0.25 8.))))

(* [positive] and a tolerance [abs + rel |y|]. *)
let scaled_tolerances =
  Gen.(
    with_pp
      (fun ppf (a, (rel, abs)) ->
        Format.fprintf ppf "a = %a@ rel = %g@ abs = %g" Nx.pp a rel abs)
      (pair positive
         (pair (float_range 1e-14 1e-6) (of_list [ 0.; 1e-300; 1e-12; 1e-10 ]))))

let newton_tests =
  [
    prop "from a good seed, every element converges within tol" positive
      (fun a ->
        (* √a seeded within 1e-7: Newton's steps shrink quadratically, so the
           answer [x + δ] is within [tol] once [|δ| / (1 − q)] meets it, two
           steps from the seed. *)
        let seed =
          Nx.mul (Nx.sqrt a) (Nx.add_s (Nx.mul_s (Nx.cos a) 1e-7) 1.)
        in
        let s =
          Root.newton ~tol:(Tol.ulps 4.) ~budget:20
            ~slope:(fun x -> Nx.mul_s x 2.)
            (fun x -> Nx.sub (Nx.square x) a)
            seed
        in
        equal (Oracle.tensor ()) (Nx.ones Nx.bool (Nx.shape a)) (Solution.ok s);
        (* [Nx.sqrt] rounds the zero by half an ulp at most. *)
        let x = Solution.best s in
        at_most float_exact ~than:1.
          (Oracle.ulps ~k:4.5 ~y:x (Nx.abs (Nx.sub x (Nx.sqrt a)))));
    prop "a converged element's error meets tol in ulps at its value"
      ~examples:[ (vec [| 2. |], (1., 8.)); (vec [| 2.; 50. |], (12., 0.25)) ]
      slowed
      (fun (a, (factor, k)) ->
        (* A slope [factor] times too steep contracts by [1 − 1 / factor]: the
           estimate reaches the floats' resolution about [factor] steps from its
           zero, which a tolerance of [k] ulps may not admit. *)
        let s =
          Root.newton ~tol:(Tol.ulps k) ~budget:400
            ~slope:(fun x -> Nx.mul_s x (2. *. factor))
            (fun x -> Nx.sub (Nx.square x) a)
            (Nx.mul_s (Nx.sqrt a) (1. +. 1e-6))
        in
        let ok = Nx.to_array (Solution.ok s) in
        cover "an element converges" (Array.mem true ok);
        cover "an element stalls"
          (Array.mem true (Nx.to_array (Solution.is Stalled s)));
        let x = Solution.best s and e = Solution.error s in
        Array.iteri
          (fun i ok ->
            if ok then
              at_most float_exact ~than:1.
                (Oracle.ulps ~k ~y:(Nx.get [ i ] x) (Nx.get [ i ] e)))
          ok);
    prop "a converged element's error meets abs + rel |y| at its value"
      ~examples:[ (vec [| 2. |], (1e-8, 0.)); (vec [| 2. |], (1e-14, 1e-10)) ]
      scaled_tolerances
      (fun (a, (rel, abs)) ->
        let s =
          Root.newton ~tol:(Tol.v ~rel ~abs) ~budget:20
            ~slope:(fun x -> Nx.mul_s x 2.)
            (fun x -> Nx.sub (Nx.square x) a)
            (Nx.mul_s (Nx.sqrt a) 1.001)
        in
        let ok = Nx.to_array (Solution.ok s) in
        let y = Oracle.floats (Solution.best s) in
        cover "an element converges" (Array.mem true ok);
        cover "abs dominates" (Array.exists (fun y -> abs > rel *. y) y);
        cover "rel dominates" (Array.exists (fun y -> abs < rel *. y) y);
        let x = Solution.best s and e = Solution.error s in
        Array.iteri
          (fun i ok ->
            if ok then
              at_most float_exact ~than:1.
                (Oracle.scaled ~rel ~abs ~y:(Nx.get [ i ] x) (Nx.get [ i ] e)))
          ok);
    test "a slope 1000 times too steep never understates the error" (fun () ->
        (* Steps 1000 times too short contract by 0.999 and fall within an ulp
           of √2 while the estimate is still about 1000 ulps from it, past
           [tight]: the element stalls. *)
        let z = Float.sqrt 2. in
        let s =
          Root.newton ~tol:tight ~budget:20000
            ~slope:(fun x -> Nx.mul_s x 2000.)
            (fun x -> Nx.sub_s (Nx.square x) 2.)
            (vec [| z +. 1e-8 |])
        in
        equal Oracle.status_t Stalled (Oracle.status s);
        let x = Nx.item [ 0 ] (Solution.best s) in
        at_least float_exact
          ~than:(Float.abs (x -. z))
          (Nx.item [ 0 ] (Solution.error s)));
    test "a solve satisfies Kepler's equation" (fun () ->
        let x = Solution.get (kepler ~e:eccentricity anomalies) in
        equal
          (Oracle.tensor ~abs:1e-14 ())
          anomalies
          (Nx.sub x (Nx.mul eccentricity (Nx.sin x))));
    test "a slope too steep by a constant factor converges to the same zero"
      (fun () ->
        let x = Solution.get (kepler ~e:eccentricity anomalies) in
        let y =
          Solution.get (kepler ~budget:200 ~factor:3. ~e:eccentricity anomalies)
        in
        equal (Oracle.tensor ~rel:1e-12 ~abs:1e-14 ()) x y);
    test "a slope too steep by a constant factor takes more steps" (fun () ->
        let n = Solution.evaluations (kepler ~e:eccentricity anomalies)
        and n3 =
          Solution.evaluations
            (kepler ~budget:200 ~factor:3. ~e:eccentricity anomalies)
        in
        Array.iteri
          (fun i k -> at_least int32 ~than:(Nx.item [ i ] n) k)
          (Nx.to_array n3));
    test "grad is the implicit derivative 1 / (1 − e cos E)" (fun () ->
        let x = Solution.get (kepler ~e:eccentricity anomalies) in
        equal
          (Oracle.tensor ~rel:1e-10 ())
          (Nx.recip (Nx.rsub_s 1. (Nx.mul eccentricity (Nx.cos x))))
          (Rune.grad'
             (fun m -> Nx.sum (Solution.get (kepler ~e:eccentricity m)))
             anomalies));
    test "grad in a captured parameter is the implicit derivative" (fun () ->
        let x = Solution.get (kepler ~e:eccentricity anomalies) in
        equal
          (Oracle.tensor ~rel:1e-10 ~abs:1e-15 ())
          (Nx.div (Nx.sin x) (Nx.rsub_s 1. (Nx.mul eccentricity (Nx.cos x))))
          (Rune.grad'
             (fun e -> Nx.sum (Solution.get (kepler ~e anomalies)))
             eccentricity));
    test "a linear f takes one step, then the zero step" (fun () ->
        (* f at 3 and at 2; the step from 2 is 0, where f is not evaluated. *)
        let s =
          Root.newton ~tol:tight ~budget:3 ~slope:Nx.ones_like
            (fun x -> Nx.sub_s x 2.)
            (vec [| 3. |])
        in
        equal (Oracle.tensor ()) (vec [| 2. |]) (Solution.get s);
        equal int32 2l (Nx.item [] (Nx.reshape [||] (Solution.evaluations s))));
    test "a budget of 1 converges on a zero step, at the zero" (fun () ->
        let s =
          Root.newton ~tol:tight ~budget:1 ~slope:Nx.ones_like
            (fun x -> Nx.sub_s x 2.)
            (vec [| 2. |])
        in
        equal (Oracle.tensor ()) (vec [| 2. |]) (Solution.get s));
    cases ~id:fst "a spent budget ends Budget_spent"
      [ ("budget 1 from 3", (1, 3.)); ("budget 2 from 10", (2, 10.)) ]
      (fun (_, (budget, x0)) ->
        let s =
          Root.newton ~tol:tight ~budget
            ~slope:(fun x -> Nx.mul_s x 2.)
            (fun x -> Nx.sub_s (Nx.square x) 2.)
            (vec [| x0 |])
        in
        equal Oracle.status_t Budget_spent (Oracle.status s));
    cases ~id:fst "a zero or non-finite slope stalls"
      [ ("0", 0.); ("nan", nan); ("inf", infinity) ]
      (fun (_, slope) ->
        let s =
          Root.newton ~tol:tight ~budget:10
            ~slope:(fun x -> Nx.full_like x slope)
            (fun x -> Nx.add_s (Nx.square x) 1.)
            (vec [| 0.5 |])
        in
        equal Oracle.status_t Stalled (Oracle.status s));
    cases ~id:fst "a value that is not finite ends Not_finite"
      (* log x with the slope 0.1: from −1 at the guess, from 10 at the first
         iterate, −13. *)
      [ ("at the guess", -1.); ("at an iterate", 10.) ]
      (fun (_, x0) ->
        let s =
          Root.newton ~tol:tight ~budget:10
            ~slope:(fun x -> Nx.full_like x 0.1)
            Nx.log (vec [| x0 |])
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    test "an element's answer and evaluations depend on its data alone"
      (fun () ->
        let changed = vec [| 0.; 2.; 1.; 2.5; 3.1 |] in
        let others s =
          let at t = Nx.slice [ Nx.L [ 0; 2; 3; 4 ] ] t in
          (at (Solution.best s), at (Solution.evaluations s))
        in
        equal
          (Oracle.structure Nx.Ptree.(pair tensor tensor))
          (others (kepler ~e:eccentricity anomalies))
          (others (kepler ~e:eccentricity changed)));
    test "a function that mixes elements raises in the derivative" (fun () ->
        let solve m =
          Solution.best
            (Root.newton ~tol:tight ~budget:30 ~slope:Nx.ones_like
               (fun x -> Nx.add (Nx.sub x m) (Nx.mul_s (Nx.sum x) 1e-3))
               m)
        in
        (* A cotangent that differs between elements, which a mixing
           derivative's diagonal cannot solve. *)
        let weights = vec [| 1.; 2. |] in
        raises_from "Jera.Root.newton" (fun () ->
            Rune.grad'
              (fun m -> Nx.sum (Nx.mul weights (solve m)))
              (vec [| 1.; 2. |])));
    test "compiled equals eager, bit for bit" (fun () ->
        Compiled.here ();
        let f m = Solution.get (kepler ~e:eccentricity m) in
        equal (Oracle.tensor ()) (f anomalies) (Rune.jit' f anomalies));
    test "a budget below 1 raises" (fun () ->
        raises_from "Jera.Root.newton" (fun () ->
            Root.newton ~tol:tight ~budget:0 ~slope:Nx.ones_like Fun.id
              (vec [| 1. |])));
  ]

let () =
  exit
    (run "Jera.Root"
       [
         group "Tol" tol_tests;
         group "Solution" solution_tests;
         group "Root" root_tests;
         group "bracket" bracket_tests;
         group "newton" newton_tests;
       ])
