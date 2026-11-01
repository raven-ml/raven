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
let failure_with sub f = raises_match (Exn.failure ~sub) f
let invalid_with sub f = raises_match (Exn.invalid_arg ~sub) f
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
    test "the zero of s atan (x / s) at s = 2⁻⁵⁴⁰ is found" (fun () ->
        (* The full step from [3 s] overshoots to [−12.5 s], which the line
           search must shorten; the squares of these residuals are below the
           floats' range. *)
        let s = Float.ldexp 1. (-540) in
        let f x = Nx.mul_s (Nx.atan (Nx.div_s x s)) s in
        let st =
          System.solve one
            (System.newton ~derivative:(derivative f))
            ~linear:Linear.dense
            ~tol:(Tol.v ~rel:1e-12 ~abs:(1e-12 *. s))
            ~budget:50 f
            (vec [| 3. *. s; 0. |])
        in
        equal bool true (is Converged st);
        equal
          (Oracle.tensor ~abs:(1e-12 *. s) ())
          (Nx.zeros f64 [| 2 |]) (Solution.get st));
    prop "a system scaled by a power of two solves to its zero scaled, bit for bit"
      system (fun (a, z) ->
        (* [atan (x − z) + a (x − z) / 10] from [z + 3]: Newton's full steps on
           the arctangent overshoot, and the line search shortens them.
           [s f (x / s)] from [s (z + 3)] evaluates [s] times what [f] does,
           exactly, at [s = 2⁻⁵⁴⁰], whose squares are below the floats' range:
           every decision of the search is the unscaled one's. A zero below
           10⁻¹⁰⁰ is 0, so that the scaled one is not subnormal. *)
        let s = Float.ldexp 1. (-540) in
        let z = Nx.where (Nx.less_s (Nx.abs z) 1e-100) (Nx.zeros_like z) z in
        let f x =
          let d = Nx.sub x z in
          Nx.add (Nx.atan d) (Nx.mul_s (Nx.matmul a d) 0.1)
        in
        let scaled x = Nx.mul_s (f (Nx.div_s x s)) s in
        let solve f tol x0 =
          System.solve one
            (System.newton ~derivative:(derivative f))
            ~linear:Linear.dense ~tol ~budget:50 f x0
        in
        let x0 = Nx.add_s z 3. in
        let plain = solve f tight x0
        and small =
          solve scaled (Tol.v ~rel:1e-12 ~abs:(1e-12 *. s)) (Nx.mul_s x0 s)
        in
        cover "a lane converges" (is Converged plain);
        equal (Oracle.tensor ())
          (Nx.mul_s (Solution.best plain) s)
          (Solution.best small);
        equal bool (is Converged plain) (is Converged small));
    test "a linear system takes one step, then the zero step" (fun () ->
        (* f at the guess, at x + δ in the line search, then δ = 0. *)
        let a = Nx.create f64 [| 2; 2 |] [| 3.; 1.; 1.; 2. |] in
        let b = vec [| 1.; 2. |] in
        let s = newton (fun x -> Nx.sub (Nx.matmul a x) b) (vec [| 0.; 0. |]) in
        equal (Oracle.tensor ~rel:1e-14 ()) (Nx.solve a b) (Solution.get s);
        equal int32 2l (Nx.item [] (Solution.evaluations s)));
    test "a zero whose unknown at 0 is another equation's is met in ulps"
      (fun () ->
        (* (x₁² − 2, x₀) from (1, 1): the step to x₀ = 0 is solved from the
           equation x₀ = 0, the second, while the first is never 0 in the
           floats; x₀'s zero step is exact all the same. *)
        let f v =
          Nx.stack [ Nx.sub_s (Nx.square (Nx.get [ 1 ] v)) 2.; Nx.get [ 0 ] v ]
        in
        let s =
          System.solve one
            (System.newton ~derivative:(derivative f))
            ~linear:Linear.dense ~tol:(Tol.ulps 4.) ~budget:50 f
            (vec [| 1.; 1. |])
        in
        equal bool true (is Converged s);
        let x = Solution.best s in
        equal (Oracle.tensor ()) (scalar 0.) (Nx.get [ 0 ] x);
        at_most float_exact ~than:(1. +. (1. /. 4.))
          (Oracle.ulps ~k:4. ~y:x
             (Nx.abs (Nx.sub x (vec [| 0.; Float.sqrt 2. |])))));
    test "a zero with a component at 0 is met in ulps" (fun () ->
        (* (x₀, x₁² − 2) from (1, 1): Newton's last step at least halves the
           distance, so the answer is within 4 ulps of (0, √2), its first
           component exactly 0. *)
        let f v =
          Nx.stack [ Nx.get [ 0 ] v; Nx.sub_s (Nx.square (Nx.get [ 1 ] v)) 2. ]
        in
        let s =
          System.solve one
            (System.newton ~derivative:(derivative f))
            ~linear:Linear.dense ~tol:(Tol.ulps 4.) ~budget:50 f
            (vec [| 1.; 1. |])
        in
        equal bool true (is Converged s);
        let x = Solution.best s in
        equal (Oracle.tensor ()) (scalar 0.) (Nx.get [ 0 ] x);
        at_most float_exact ~than:1.
          (Oracle.ulps ~k:4. ~y:x
             (Nx.abs (Nx.sub x (vec [| 0.; Float.sqrt 2. |])))));
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
    test "the estimate of a tiny system's Jacobian learns from its steps"
      (fun () ->
        (* [x / 2 − 3 s atan (x / s) / 2 = s b] at [s = 2⁻⁵⁴⁰]: the
           differences' probes, √ε away, saturate the arctangent, so [B]
           starts as [1/2] while the slope at the zero is negative. Only the
           rank-one updates, from steps whose squares are below the floats'
           range, turn it. *)
        let s = Float.ldexp 1. (-540) and z = vec [| 1.; -0.5 |] in
        let g u = Nx.sub (Nx.mul_s u 0.5) (Nx.mul_s (Nx.atan u) 1.5) in
        let b = g z in
        let f x = Nx.sub (Nx.mul_s (g (Nx.div_s x s)) s) (Nx.mul_s b s) in
        let st =
          System.solve one System.broyden ~linear:Linear.dense
            ~tol:(Tol.v ~rel:1e-12 ~abs:(1e-12 *. s))
            ~budget:100 f
            (Nx.mul_s (vec [| 0.9; -0.4 |]) s)
        in
        equal near z (Nx.div_s (Solution.get st) s));
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

(* The bound on the answer of a map that shrinks the distance by at most [top]
   per step, under a tolerance of [k] ulps, in units of it: the answer is
   within [top / (1 − top)] of [tol], and the map's rounding of an ulp per
   step adds [1 / (k (1 − top))]. *)
let within ~top k = (top +. (1. /. k)) /. (1. -. top)

(* The measure [Tol.ulps k] takes of the distance from [s]'s answer to [z]. *)
let distance k s z =
  let x = Solution.best s in
  Oracle.ulps ~k ~y:x (Nx.abs (Nx.sub x z))

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
    test "a lane whose computed zero is 120 ulps off is within its bound"
      (fun () ->
        (* f's rounding in the large component, coupled into the small one,
           leaves the small one's computed zero about 120 ulps from √a₀: under
           a few ulps that is the evaluation error the error does not see. The
           map's rate near the zero is about 0.56, above 1/2, so under 1024
           ulps, where that rounding is small, an answer is within
           [0.56 / 0.44] of [tol] to rounding, not within [tol]. *)
        let a =
          vec
            [|
              Float.of_string "0x1.07217b2039223p-3";
              Float.of_string "0x1.4cfba9b2565bep+6";
            |]
        and factor = Float.of_string "0x1.a9d9b3b586687p+0" in
        let m = Nx.create f64 [| 2; 2 |] [| 1.; 0.5; -0.5; 1. |] in
        let root = Nx.sqrt a in
        let f x =
          Nx.div_s
            (Nx.div
               (Nx.reshape [| 2 |]
                  (Nx.matmul m
                     (Nx.reshape [| 2; 1 |] (Nx.sub a (Nx.square x)))))
               (Nx.mul_s root 2.))
            factor
        in
        let k = 1024. in
        let s =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol:(Tol.ulps k) ~budget:400 f
            (Nx.mul_s root (1. +. 1e-6))
        in
        if is Converged s then
          at_most float_exact ~than:(within ~top:0.56 k) (distance k s root));
    test "anderson with memory reaches the zero of a map that does not contract"
      (fun () ->
        (* 2 x + sin x = b: the Picard map x + f x expands by up to 3, and
           mixing two steps still reaches its zero, which Newton in OCaml
           floats gives. *)
        let b = [| 1.; -2.; 0.5 |] in
        let zero b =
          let x = ref 0. in
          for _ = 1 to 60 do
            x := !x -. (((2. *. !x) +. sin !x -. b) /. (2. +. cos !x))
          done;
          !x
        in
        let a = Nx.mul_s (Nx.eye f64 3) 2. in
        let s = anderson ~memory:2 (field a (vec b)) (Nx.zeros f64 [| 3 |]) in
        equal bool true (is Converged s);
        equal near (vec (Array.map zero b)) (Solution.best s));
    test "a small slow component beside a large fast one converges in ulps"
      (fun () ->
        (* z + c ⊙ (x − z) with rates 0.027 and 0.94 and zeros 1.8 and 0.0028:
           measured in [tol]'s scaled norm, the readings see the slow small
           component settle at 0.94, and the answer is within
           [(0.94 + 1 / 64) / 0.06] of [tol]; a Euclidean secant, dominated by
           the large component, never settles it. *)
        let h = Float.of_string in
        let c = vec [| h "0x1.bd7651a93286dp-6"; h "0x1.e0ce22dde1eafp-1" |]
        and z = vec [| h "-0x1.cc26fb884a3cep+0"; h "-0x1.7529c8bd30666p-9" |]
        and x0 = vec [| h "-0x1.b78f6b62e373dp+0"; h "-0x1.752bf6ef80993p-9" |] in
        let s =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol:(Tol.ulps 64.) ~budget:20000
            (fun x -> Nx.mul (Nx.sub_s c 1.) (Nx.sub x z))
            x0
        in
        equal bool true (is Converged s);
        at_most float_exact
          ~than:(within ~top:(Nx.item [ 1 ] c) 64.)
          (distance 64. s z));
    test "a slow map toward 0 converges in ulps at 0 alone" (fun () ->
        (* x − x / 100 from a thousand subnormal spacings: the steps round to 0
           at fifty spacings from 0, where [f] underflows too. That point is
           no zero a tolerance in ulps admits; one in absolute terms admits
           it. *)
        let solve tol =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol ~budget:2000
            (fun x -> Nx.mul_s x (-0.01))
            (vec [| 1000. *. Float.ldexp 1. (-1074) |])
        in
        let s = solve (Tol.ulps 4.) in
        if is Converged s then
          equal (Oracle.tensor ()) (vec [| 0. |]) (Solution.best s);
        equal bool true (is Converged (solve (Tol.abs 1e-300))));
    test "Picard on (z − x) / 2 from 0 reaches its small component" (fun () ->
        (* The rate is 1/2: each step halves the distance, so the error
           [2 |δ|] meets the tolerance once the steps fall to an eighth of it,
           which the floats resolve in both components, and the answer is
           within it. *)
        let z = vec [| 1.; 1e-3 |] in
        List.iter
          (fun k ->
            let s =
              System.solve one
                (System.anderson ~memory:0)
                ~linear:Linear.dense ~tol:(Tol.ulps k) ~budget:400
                (fun x -> Nx.div_s (Nx.sub z x) 2.)
                (vec [| 1.; 0. |])
            in
            equal bool true (is Converged s);
            at_most float_exact ~than:(within ~top:0.5 k) (distance k s z))
          [ 16.; 256. ]);
    test "a negative memory raises" (fun () ->
        invalid_with "Jera.System.anderson: memory = -1 is negative" (fun () ->
            System.anderson ~memory:(-1)));
  ]

(* Laws *)

(* [system], a factor by which a step is too short, a tolerance in ulps, and a
   method: Newton's with a derivative [factor] times too steep, Broyden's, or
   Anderson's without memory on the fixed point of [z] with its residual
   [factor] times too small. *)
let limited =
  let open Gen in
  (let+ az = system
   and+ factor = float_range 1. 8.
   and+ k = float_range 0.25 8.
   and+ m = one_of (List.map constant [ "newton"; "broyden"; "anderson" ]) in
   (az, (factor, (k, m))))
  |> with_pp (fun ppf ((a, z), (factor, (k, m))) ->
      Format.fprintf ppf "a = %a@ z = %a@ factor = %g@ k = %g@ %s" Nx.pp a
        Nx.pp z factor k m)

(* The magnitude of a zero's component: in [0.5, 2], 0, below the least normal
   float, where the floats resolve it only to the subnormals' spacing and an
   answer within a tolerance in ulps is the zero itself, or tiny and normal. *)
let magnitude =
  let normal = Gen.float_range 0.5 2. in
  Gen.one_of
    [
      normal;
      normal;
      normal;
      Gen.constant 0.;
      Gen.map (fun u -> Float.ldexp u (-1040)) (Gen.float_range 0.5 2.);
      Gen.map (fun u -> Float.ldexp u (-960)) (Gen.float_range 0.5 2.);
    ]

(* Covers for the kinds of zero [magnitude] draws, in a vector [z]. *)
let kinds z =
  let any p = Array.exists p (Oracle.floats z) in
  cover "a zero at 0" (any (fun v -> v = 0.));
  cover "a subnormal zero"
    (any (fun v -> v <> 0. && Float.abs v < Float.min_float));
  cover "a tiny normal zero"
    (any (fun v -> Float.abs v >= Float.min_float && Float.abs v < 1e-280))

(* The spacing of the subnormals, float64's absolute resolution. *)
let spacing = Float.ldexp 1. (-1074)

(* [z + d r]: within [r] of a zero [z] of normal magnitude, within a thousand
   subnormal spacings of a smaller one. *)
let start_at z d r =
  if Float.abs z < Float.min_float then z +. (d *. 1000. *. spacing)
  else z +. (d *. r)

(* A linear map [z + c ⊙ (x − z)] of up to three components whose largest rate
   is [top], the others anywhere below it, down to 0, and a tolerance: [k]
   ulps, or [Tol.rel (k ε)] for [k] up to [10⁹·⁶]. A zero's components differ
   in magnitude by up to [10³], and one of rate above 1/2 starts within 300
   ulps of its zero, or five times a looser tolerance, the others within 1,
   so that a fast large one can collapse while the error lies in a slow small
   one; a zero below the normal floats is started as {!start_at} says. *)
let linear_maps =
  let open Gen in
  (let* n = int_range 1 3 in
   let* top = one_of (List.map constant [ 0.5; 0.9; 0.99 ]) in
   let+ rest = list ~size:(constant (n - 1)) (float_range 0. top)
   and+ zs = list ~size:(constant n) magnitude
   and+ decades = list ~size:(constant n) (one_of [ constant 1.; constant 1e-3 ])
   and+ signs = list ~size:(constant n) bool
   and+ offsets = list ~size:(constant n) (float_range (-1.) 1.)
   and+ rel, k =
     one_of
       [
         map (fun k -> (false, k)) (float_range 4. 64.);
         map (fun e -> (true, 10. ** e)) (float_range 5.6 9.6);
       ]
   in
   let c = Array.of_list (top :: rest) in
   let z =
     Array.of_list
       (List.map2
          (fun (z, d) s -> if s then z *. d else -.(z *. d))
          (List.combine zs decades) signs)
   in
   let x0 =
     Array.mapi
       (fun i d ->
         let reach =
           if c.(i) > 0.5 then
             Float.max 300. (5. *. k) *. epsilon_float *. Float.abs z.(i)
           else 1.
         in
         start_at z.(i) d reach)
       (Array.of_list offsets)
   in
   (top, (vec c, (vec z, (vec x0, (rel, k))))))
  |> with_pp (fun ppf (top, (c, (z, (x0, (rel, k))))) ->
      Format.fprintf ppf "top = %g@ c = %a@ z = %a@ x0 = %a@ %s %g" top Nx.pp
        c Nx.pp z Nx.pp x0 (if rel then "rel" else "ulps") k)

(* [g x = z + h (x − z)], continuous, contracting by [far] beyond [w] ulps of
   [z] and by 0.99 within, from [z + d0], and a tolerance of [k] ulps. *)
let rising =
  let open Gen in
  (let+ z = float_range 1. 2.
   and+ w = float_range 100. 1000.
   and+ far, d0 = one_of [ constant (1e-8, 1e-6); constant (1e-3, 1e-3) ]
   and+ k = float_range 4. 64. in
   (z, (w, (far, (d0, k)))))
  |> with_pp (fun ppf (z, (w, (far, (d0, k)))) ->
      Format.fprintf ppf "z = %.17g@ w = %g@ far = %g@ d0 = %g@ k = %g" z w far
        d0 k)

let rising_residual ~far ~w z v =
  let zt = scalar z in
  let t = Nx.sub v zt in
  let b = w *. epsilon_float *. Float.abs z in
  let inside = Nx.mul_s t 0.99 in
  let outside =
    Nx.mul (Nx.sign t)
      (Nx.add_s (Nx.mul_s (Nx.sub_s (Nx.abs t) b) far) (0.99 *. b))
  in
  Nx.sub (Nx.add zt (Nx.where (Nx.less_equal_s (Nx.abs t) b) inside outside)) v

(* [d ⊙ t + t³ / 10] with [t = x − c] and its equations in the order [order],
   so that equation [i] need not be unknown [i]'s, curvatures [d] in [e^±3]
   and [c] of {!magnitude}s, from within 10⁻³ of [c] in each component, as
   {!start_at} says, and a tolerance of [k] ulps. *)
let cubics =
  let open Gen in
  (let* n = int_range 2 6 in
   let+ d = list ~size:(constant n) (float_range (-3.) 3.)
   and+ c = list ~size:(constant n) magnitude
   and+ signs = list ~size:(constant n) bool
   and+ offsets = list ~size:(constant n) (float_range (-1.) 1.)
   and+ keys = list ~size:(constant n) (float_range 0. 1.)
   and+ k = float_range 4. 64. in
   let c =
     Array.of_list (List.map2 (fun c s -> if s then c else -.c) c signs)
   in
   let x0 =
     Array.mapi (fun i o -> start_at c.(i) o 1e-3) (Array.of_list offsets)
   in
   let order =
     List.mapi (fun i key -> (key, i)) keys
     |> List.sort compare |> List.map snd |> List.map Int64.of_int
     |> Array.of_list
   in
   ( vec (Array.of_list (List.map exp d)),
     (vec c, (vec x0, (Nx.create Nx.int64 [| n |] order, k))) ))
  |> with_pp (fun ppf (d, (c, (x0, (order, k)))) ->
      Format.fprintf ppf "d = %a@ c = %a@ x0 = %a@ order = %a@ k = %g" Nx.pp
        d Nx.pp c Nx.pp x0 Nx.pp order k)

let law_tests =
  [
    prop "a linear map's answer is within c / (1 − c) of tol, c its top rate"
      linear_maps (fun (top, (c, (z, (x0, (rel, k))))) ->
        (* If the step from [x] shrinks the distance by [L], the answer
           [x + δ] is within [L |δ| / (1 − L)] and [|δ|] within [tol]: [L] is
           at most [top] here, and at most 1/2 brings the answer within
           [tol], to rounding. *)
        cover "rates that differ" (Nx.item [] (Nx.min c) < top /. 2.);
        cover "a relative tolerance" rel;
        kinds z;
        let tol = if rel then Tol.rel (k *. epsilon_float) else Tol.ulps k in
        let s =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol ~budget:2000
            (fun x -> Nx.mul (Nx.sub_s c 1.) (Nx.sub x z))
            x0
        in
        cover "a lane converges" (is Converged s);
        if is Converged s then
          at_most float_exact ~than:(within ~top k) (distance k s z));
    prop "a map whose rate rises near its zero is within 99 tol" rising
      (fun (z, (w, (far, (d0, k)))) ->
        (* The rate is at most 0.99 everywhere, so a converged answer is within
           [0.99 / 0.01] of [tol] to rounding, whatever rate the readings
           saw. *)
        let s =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol:(Tol.ulps k) ~budget:2000
            (rising_residual ~far ~w z)
            (vec [| z +. d0 |])
        in
        cover "a lane converges" (is Converged s);
        if is Converged s then
          at_most float_exact ~than:(within ~top:0.99 k)
            (distance k s (vec [| z |])));
    prop "newton and broyden near a simple zero converge within tol" cubics
      (fun (d, (c, (x0, (order, k)))) ->
        (* Within 10⁻³ of [c] the cubic moves the Jacobian by at most
           [3 · 10⁻⁷] of itself, and Broyden's estimate, from differences
           exact to [√ε] for a nearly linear [f], keeps to that: either map
           shrinks the distance by far less than 1/2 at every step, and the
           answer is within [tol] to its rounding, an ulp over [k]. *)
        kinds c;
        cover "equations out of order"
          (Oracle.floats order <> Array.init (Nx.dim 0 order) Float.of_int);
        let f x =
          let t = Nx.sub x c in
          Nx.take order
            (Nx.add (Nx.mul d t) (Nx.mul_s (Nx.mul t (Nx.square t)) 0.1))
        in
        List.iter
          (fun (name, m) ->
            let s =
              System.solve one m ~linear:Linear.dense ~tol:(Tol.ulps k)
                ~budget:200 f x0
            in
            cover (name ^ " converges") (is Converged s);
            if is Converged s then
              at_most float_exact ~msg:name ~than:(1. +. (1. /. k))
                (distance k s c))
          [
            ("newton", System.newton ~derivative:(derivative f));
            ("broyden", System.broyden);
          ]);
    prop "a converged solve's error meets tol at its estimate" limited
      (fun ((a, z), (factor, (k, m))) ->
        (* Seeded near the zero, the methods reach the floats' resolution,
           some short of a tolerance of [k] ulps. *)
        let tol = Tol.ulps k and f = field a (rhs a z) in
        let seed = Nx.add z (Nx.mul_s (Nx.cos (Nx.mul_s z 7.)) 1e-6) in
        let s =
          match m with
          | "newton" ->
              let derivative x dx = Nx.mul_s (derivative f x dx) factor in
              System.solve one
                (System.newton ~derivative)
                ~linear:Linear.dense ~tol ~budget:300 f seed
          | "broyden" ->
              System.solve one System.broyden ~linear:Linear.dense ~tol
                ~budget:100 f seed
          | _ ->
              System.solve one
                (System.anderson ~memory:0)
                ~linear:Linear.dense ~tol ~budget:300
                (fun x -> Nx.div_s (fixed z x) factor)
                seed
        in
        cover (m ^ " converges") (is Converged s);
        (* Broyden's differences estimate the Jacobian well enough that its
           steps contract fast to the end. *)
        if m <> "broyden" then cover (m ^ " stalls") (is Stalled s);
        (* The solve adds pairwise, [Oracle.ulps] in order: they differ by
           rounding. *)
        if is Converged s then
          at_most float_exact ~than:(1. +. 1e-12)
            (Oracle.ulps ~k ~y:(Solution.best s) (Solution.error s)));
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

(* Lanes whose zero has an unknown below the normal floats beside one at 1:
   the steps reach it in the subnormals, whose products round by an absolute
   2⁻¹⁰⁷⁵ rather than relatively. *)
let subnormal_zeros =
  List.map
    (fun small ->
      ( Nx.create f64 [| 1; 2; 2 |] [| 4.; 1.; 0.; 4. |],
        Nx.create f64 [| 1; 2 |] [| 1.; small |] ))
    [ 7.5e-311; 8.471802945955e-311; 8.6e-311 ]

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
    prop "seeded lanes reaching rounding in two steps converge"
      ~examples:subnormal_zeros lane_systems (fun (a, z) ->
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
    test "a lane whose residual is rounding converges" (fun () ->
        (* The seed's step lands at the zero to f's evaluation level, 3e-17,
           while the unknown at 0 still has ulps of 1e-33: no step changes f,
           and a short enough one meets the sufficient decrease by rounding. *)
        let a =
          Nx.create f64 [| 1; 2; 2 |]
            [|
              4.; -0.0025442380518094533; 0.370970824797177; 3.3353786945226602;
            |]
        in
        let z = Nx.create f64 [| 1; 2 |] [| -0.6718716184867024; 0. |] in
        let b = Nx.add (apply a z) (Nx.sin z) in
        let seed = Nx.add z (Nx.mul_s (Nx.cos (Nx.mul_s z 7.)) 1e-6) in
        equal near z
          (Solution.get
             (System.lanes
                ~tol:(Tol.v ~rel:1e-14 ~abs:1e-14)
                ~budget:20 ~jacobian:(lane_jacobian a) (lane_field a b) seed)));
    test "a lane whose zero has an unknown at 0 converges" (fun () ->
        (* The seed's step reaches f's evaluation level, 4e-16. The next moves
           the first unknown by 1.6 ulps and the one at 0 by 6e-20, whose ulps
           are 1e-35, and no trial lowers the merit: within tol, the lane is at
           its zero. *)
        let a =
          Nx.create f64 [| 1; 3; 3 |]
            [|
              5.;
              0.;
              0.;
              0.43536040733974823;
              5.;
              6.500960975843154e-10;
              0.;
              0.;
              5.;
            |]
        in
        let z =
          Nx.create f64 [| 1; 3 |]
            [| 0.21418297743675652; -1.564054759905721; 0. |]
        in
        let b = Nx.add (apply a z) (Nx.sin z) in
        let seed = Nx.add z (Nx.mul_s (Nx.cos (Nx.mul_s z 7.)) 1e-6) in
        equal near z
          (Solution.get
             (System.lanes
                ~tol:(Tol.v ~rel:1e-14 ~abs:1e-14)
                ~budget:20 ~jacobian:(lane_jacobian a) (lane_field a b) seed)));
    test "a lane far from its zero whose line search fails stalls" (fun () ->
        (* f x = x from 2, with the slope 2 above 1.5, 1000 down to 0.9995 and
           -999 below. The map contracts by 0.001 over the step from 2, which
           leaves 1 the bound 0.001 / 0.999, twice the tolerance of 5 · 10⁻⁴. The
           step from 0.999 is +0.001, uphill, and no trial lowers the merit;
           the map moves over it by its own length, q = 1, far above its
           rounding, which drops the 0.001 held. *)
        let jacobian x =
          Nx.reshape [| 1; 1; 1 |]
            (Nx.where (Nx.greater_s x 1.5) (Nx.full_like x 2.)
               (Nx.where (Nx.greater_s x 0.9995) (Nx.full_like x 1000.)
                  (Nx.full_like x (-999.))))
        in
        let s =
          System.lanes ~tol:(Tol.v ~rel:0. ~abs:5e-4) ~budget:20 ~jacobian
            Fun.id
            (Nx.create f64 [| 1; 1 |] [| 2. |])
        in
        equal (array bool) [| false |] (Nx.to_array (Solution.ok s)));
    test "a lane of tiny unknowns far from its zero stalls" (fun () ->
        (* The lane above in its first unknown, scaled by [s], with a second
           unknown at its zero, and the tolerance halved again: its root mean
           square counts the second's zero error, so the bound at 1 s is 2.8
           times it. Every step of the first is a
           multiple of [s], and the square of the one from 1 s, 10⁻³ s, is
           below the floats' range: the search must still measure the map's
           contraction over it. *)
        let s = Float.ldexp 1. (-530) in
        let jacobian x =
          let u = Nx.div_s (Nx.slice [ Nx.A; Nx.I 0 ] x) s in
          let slope =
            Nx.where (Nx.greater_s u 1.5) (Nx.full_like u 2.)
              (Nx.where (Nx.greater_s u 0.9995) (Nx.full_like u 1000.)
                 (Nx.full_like u (-999.)))
          in
          Nx.reshape [| 1; 2; 2 |]
            (Nx.stack ~axis:(-1)
               [ slope; Nx.zeros_like u; Nx.zeros_like u; Nx.ones_like u ])
        in
        let st =
          System.lanes
            ~tol:(Tol.v ~rel:0. ~abs:(2.5e-4 *. s))
            ~budget:20 ~jacobian Fun.id
            (Nx.create f64 [| 1; 2 |] [| 2. *. s; 0. |])
        in
        equal (array bool) [| false |] (Nx.to_array (Solution.ok st)));
    prop "lanes of unknowns near the underflow reach their zero" lane_systems
      (fun (a, z) ->
        (* [lane_field] scaled by a power of two [s]: its evaluations are the
           unscaled ones times [s], exactly, and its zero is [s z]. Every step
           is below 10⁻¹⁶², whose square is below the floats' range. *)
        cover "several unknowns" (Nx.dim 1 z > 1);
        let s = Float.ldexp 1. (-540) in
        let b = Nx.add (apply a z) (Nx.sin z) in
        let f x = Nx.mul_s (lane_field a b (Nx.div_s x s)) s
        and jacobian x = lane_jacobian a (Nx.div_s x s) in
        let st =
          System.lanes
            ~tol:(Tol.v ~rel:1e-12 ~abs:(1e-12 *. s))
            ~budget:50 ~jacobian f (Nx.zeros_like z)
        in
        equal near z (Nx.div_s (Solution.get st) s));
    test "compiled lanes equal eager" (fun () ->
        let a =
          Nx.broadcast_to [| 3; 2; 2 |]
            (Nx.create f64 [| 2; 2 |] [| 4.; 1.; 1.; 3. |])
        in
        let f b = Solution.get (in_lanes a b (Nx.zeros f64 [| 3; 2 |])) in
        let b = Nx.reshape [| 3; 2 |] (Nx.linspace f64 (-2.) 2. 6) in
        (* The compiled solve sums in an order of its own, so the zeros agree
           to rounding. *)
        equal (Oracle.tensor ~rel:1e-14 ()) (f b) (Rune.jit' f b));
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
