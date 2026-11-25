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
let raises_from fn f = raises_match (Exn.invalid_arg ~sub:fn) f
let is st s = Nx.item [] (Solution.is st s)

(* That [x] is within [abs + rel |x|] of the zero [z], in the tolerance's root
   mean square. A solve whose last step at least halved the distance to the zero
   is within its tolerance there: Newton's and Broyden's near a simple zero, and
   Anderson's on a map contracting by 1/2. [f]'s evaluation error moves the
   computed zero by a few ε, far below these tolerances. *)
let within ~rel ~abs z x =
  at_most float_exact ~than:1.
    (Oracle.scaled ~rel ~abs ~y:x (Nx.abs (Nx.sub x z)))

let within_tight = within ~rel:1e-12 ~abs:1e-12

(* A system [a x + sin x = b] of [n] unknowns whose zero is [z]: [a] has entries
   in [-1, 1] plus [n + 2] on its diagonal, so the Jacobian [a + diag (cos x)]
   is diagonally dominant everywhere. *)
let system_of n =
  let open Gen in
  (let+ a = list ~size:(constant (n * n)) (float_range (-1.) 1.)
   and+ z = list ~size:(constant n) (float_range (-2.) 2.) in
   let a = Nx.create f64 [| n; n |] (Array.of_list a) in
   let a = Nx.add a (Nx.mul_s (Nx.eye f64 n) (Float.of_int (n + 2))) in
   (a, Nx.create f64 [| n |] (Array.of_list z)))
  |> with_pp (fun ppf (a, z) ->
      Format.fprintf ppf "a = %a@ z = %a" Nx.pp a Nx.pp z)

(* [system_of] none to six unknowns, none and one each a fifth of the time. *)
let system =
  Gen.(
    bind
      (frequency [ (1, constant 0); (1, constant 1); (3, int_range 2 6) ])
      system_of)

let field a b x = Nx.sub (Nx.add (Nx.matmul a x) (Nx.sin x)) b
let rhs a z = Nx.add (Nx.matmul a z) (Nx.sin z)
let jacobian a x = Nx.add a (Nx.diag (Nx.cos x))
let derivative f x dx = snd (Rune.jvp' f x dx)

(* [z] moved by up to 1e-6 in each component, a seed near the zero. *)
let seed z = Nx.add z (Nx.mul_s (Nx.cos (Nx.mul_s z 7.)) 1e-6)

let newton ?(linear = Linear.dense) f guess =
  System.solve one
    (System.newton ~derivative:(derivative f))
    ~linear ~tol:tight ~budget:50 f guess

let broyden f guess =
  System.solve one System.broyden ~linear:Linear.dense ~tol:tight ~budget:100 f
    guess

let anderson ~memory f guess =
  System.solve one (System.anderson ~memory) ~linear:Linear.dense ~tol:tight
    ~budget:200 f guess

(* The fixed point [z] of [g x = sin x / 2 + c], a contraction by 1/2, as the
   residual [g x − x]. *)
let fixed z =
  let c = Nx.sub z (Nx.mul_s (Nx.sin z) 0.5) in
  fun x -> Nx.sub (Nx.add (Nx.mul_s (Nx.sin x) 0.5) c) x

(* The bound on the answer of a map that shrinks the distance by at most [top]
   per step, under a tolerance of [k] ulps, in units of it: the answer is within
   [top / (1 − top)] of [tol], and the map's rounding of an ulp per step adds [1
   / (k (1 − top))]. *)
let bound ~top k = (top +. (1. /. k)) /. (1. -. top)

(* The measure [Tol.ulps k] takes of the distance from [s]'s answer to [z]. *)
let distance k s z =
  let x = Solution.best s in
  Oracle.ulps ~k ~y:x (Nx.abs (Nx.sub x z))

(* A system, a factor by which a step is too short, and a tolerance in ulps, for
   the laws of a converged error. *)
let seeded =
  Gen.(
    with_pp
      (fun ppf ((a, z), (factor, k)) ->
        Format.fprintf ppf "a = %a@ z = %a@ factor = %g@ k = %g" Nx.pp a Nx.pp z
          factor k)
      (pair system (pair (float_range 1. 8.) (float_range 0.25 8.))))

(* Two unknowns about a zero of normal magnitude, a fast-converging example for
   each of the [seeded] laws, and a slow one whose steps fall below the floats'
   resolution short of a quarter ulp. *)
let two = Nx.create f64 [| 2; 2 |] [| 4.; 1.; 0.5; 4. |]
let fast = ((two, vec [| 0.7; -1.3 |]), (1., 8.))
let slow = ((two, vec [| 0.7; -1.3 |]), (8., 0.25))

(* That a converged solve's error meets [Tol.ulps k] at its estimate. The solve
   adds pairwise, [Oracle.ulps] in order: they differ by rounding. *)
let error_meets k s =
  if is Converged s then
    at_most float_exact ~than:(1. +. 1e-12)
      (Oracle.ulps ~k ~y:(Solution.best s) (Solution.error s))

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

(* [d ⊙ t + t³ / 10] with [t = x − c] and its equations in the order [order], so
   that equation [i] need not be unknown [i]'s, curvatures [d] in [e^±3] and [c]
   of {!magnitude}s, from within 10⁻³ of [c] in each component, as {!start_at}
   says, and a tolerance of [k] ulps. *)
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
      Format.fprintf ppf "d = %a@ c = %a@ x0 = %a@ order = %a@ k = %g" Nx.pp d
        Nx.pp c Nx.pp x0 Nx.pp order k)

let cubic d c order x =
  let t = Nx.sub x c in
  Nx.take order (Nx.add (Nx.mul d t) (Nx.mul_s (Nx.mul t (Nx.square t)) 0.1))

(* Near a simple zero, the claim of the [cubics] law for method [m]. Within 10⁻³
   of [c] the cubic moves the Jacobian by at most [3 · 10⁻⁷] of itself, and
   Broyden's estimate, from differences exact to [√ε] for a nearly linear [f],
   keeps to that: either map shrinks the distance by far less than 1/2 at every
   step, so the answer is within [tol] to its rounding, an ulp over [k], and a
   zero whose components are all normal and nonzero converges. *)
let near_simple_zero m (d, (c, (x0, (order, k)))) =
  kinds c;
  cover "equations out of order"
    (Oracle.floats order <> Array.init (Nx.dim 0 order) Float.of_int);
  let normal =
    Array.for_all (fun v -> Float.abs v >= Float.min_float) (Oracle.floats c)
  in
  cover "every component normal" normal;
  let f = cubic d c order in
  let s =
    System.solve one (m f) ~linear:Linear.dense ~tol:(Tol.ulps k) ~budget:200 f
      x0
  in
  if normal then equal Oracle.status_t Converged (Oracle.status s);
  if is Converged s then
    at_most float_exact ~than:(1. +. (1. /. k)) (distance k s c)

(* Five unknowns at a derivative exactly half too shallow: the class of
   fa71726a2b's counterexample, where a contraction read over the last two
   undamped steps alone converged 1e-12 from the zero. *)
let five =
  let a =
    Nx.add
      (Nx.mul_s (Nx.eye f64 5) 7.)
      (Nx.sin (Nx.reshape [| 5; 5 |] (Nx.arange_f f64 0. 25. 1.)))
  in
  (a, vec [| 1.3; -0.4; 1.9; -1.7; 0.2 |])

(* Newton *)

let newton_tests =
  [
    prop "its zero is the one the system is built around" system (fun (a, z) ->
        cover "an empty system" (Nx.dim 0 z = 0);
        cover "several unknowns" (Nx.dim 0 z > 2);
        within_tight z
          (Solution.get (newton (field a (rhs a z)) (Nx.zeros_like z))));
    prop "with gmres it is Newton–Krylov, to the same zero" system
      (fun (a, z) ->
        let linear =
          Linear.gmres ~restart:6 ~rel:1e-13 ~budget:60 ~precondition:Fun.id
        in
        within_tight z
          (Solution.get (newton ~linear (field a (rhs a z)) (Nx.zeros_like z))));
    prop "a seed two steps from rounding converges" system (fun (a, z) ->
        (* Seeded within 1e-6 of the zero, quadratic convergence reaches f's
           evaluation level before a third point exists to measure two
           contractions; the next step cannot move the estimate. *)
        cover "several unknowns" (Nx.dim 0 z > 2);
        let f = field a (rhs a z) in
        let s =
          System.solve one
            (System.newton ~derivative:(derivative f))
            ~linear:Linear.dense
            ~tol:(Tol.v ~rel:1e-14 ~abs:1e-14)
            ~budget:20 f (seed z)
        in
        within ~rel:1e-14 ~abs:1e-14 z (Solution.get s));
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
        equal Oracle.status_t Converged (Oracle.status st);
        equal
          (Oracle.tensor ~abs:(1e-12 *. s) ())
          (Nx.zeros f64 [| 2 |]) (Solution.get st));
    prop "a system scaled by 2⁻⁵⁴⁰ solves to its zero scaled, bit for bit"
      system (fun (a, z) ->
        (* [atan (x − z) + a (x − z) / 10] from [z + 3]: Newton's full steps on
           the arctangent overshoot, and the line search shortens them. [s f (x
           / s)] from [s (z + 3)] evaluates [s] times what [f] does, exactly, at
           [s = 2⁻⁵⁴⁰], whose squares are below the floats' range: every
           decision of the search is the unscaled one's. A zero below 10⁻¹⁰⁰ is
           0, so that the scaled one is not subnormal. *)
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
        equal Oracle.status_t (Oracle.status plain) (Oracle.status small));
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
        equal Oracle.status_t Converged (Oracle.status s);
        let x = Solution.best s in
        equal (Oracle.tensor ()) (scalar 0.) (Nx.get [ 0 ] x);
        at_most float_exact
          ~than:(1. +. (1. /. 4.))
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
        equal Oracle.status_t Converged (Oracle.status s);
        let x = Solution.best s in
        equal (Oracle.tensor ()) (scalar 0.) (Nx.get [ 0 ] x);
        at_most float_exact ~than:1.
          (Oracle.ulps ~k:4. ~y:x
             (Nx.abs (Nx.sub x (vec [| 0.; Float.sqrt 2. |])))));
    prop "near a simple zero a converged answer is within tol" cubics
      (near_simple_zero (fun f -> System.newton ~derivative:(derivative f)));
    prop "a converged solve's error meets tol at its estimate" ~count:40
      ~examples:[ fast; slow ] seeded (fun ((a, z), (factor, k)) ->
        (* A derivative [factor] times too steep contracts by [1 − 1 / factor]:
           seeded near the zero, the search reaches the floats' resolution, some
           short of a tolerance of [k] ulps. *)
        let f = field a (rhs a z) in
        let derivative x dx = Nx.mul_s (derivative f x dx) factor in
        let s =
          System.solve one
            (System.newton ~derivative)
            ~linear:Linear.dense ~tol:(Tol.ulps k) ~budget:300 f (seed z)
        in
        cover "converges" (is Converged s);
        cover "stalls" (is Stalled s);
        error_meets k s);
    prop "a derivative too steep by 2 converges to the zero" system
      (fun (a, z) ->
        (* Steps half as long contract by 1/2: from 10⁻³ away, about 30 steps
           reach [tight], long before the floats' resolution, even at a zero of
           subnormal magnitude, where a first step from 0 could not move. *)
        let f = field a (rhs a z) in
        let s =
          System.solve one
            (System.newton ~derivative:(fun x dx ->
                 Nx.mul_s (derivative f x dx) 2.))
            ~linear:Linear.dense ~tol:tight ~budget:60 f (Nx.add_s z 1e-3)
        in
        equal Oracle.status_t Converged (Oracle.status s);
        within_tight z (Solution.best s));
    prop "a derivative too shallow, by up to 2, converges only at the zero"
      ~examples:[ (five, 0.5) ]
      Gen.(
        frequency
          [
            (1, pair (system_of 5) (constant ~pp:Format.pp_print_float 0.5));
            (3, pair system (float_range 0.5 1.));
          ])
      (fun ((a, z), shallow) ->
        (* Steps [1 / shallow] times too long: the undamped map x + δ / shallow
           contracts by [1 / shallow − 1], and not at all at 1/2, whatever the
           line search's damped steps do. A solve that converges is at the zero
           to its tolerance; [rel 1e-12, abs 1e-14] once found a solve that
           converged 1e-12 away, when [q] read only the last two steps. *)
        (* Five unknowns at exactly half are a quarter of the draws. *)
        cover "five unknowns at a derivative exactly half"
          (Nx.dim 0 z = 5 && shallow = 0.5);
        let f = field a (rhs a z) in
        let s =
          System.solve one
            (System.newton ~derivative:(fun x dx ->
                 Nx.mul_s (derivative f x dx) shallow))
            ~linear:Linear.dense
            ~tol:(Tol.v ~rel:1e-12 ~abs:1e-14)
            ~budget:60 f (Nx.zeros_like z)
        in
        cover "converges" (is Converged s);
        if is Converged s then
          equal (Oracle.tensor ~rel:1e-10 ~abs:1e-12 ()) z (Solution.get s));
    test "a step that overflows where f is finite keeps a finite best"
      (fun () ->
        (* A derivative of 1e-310 sends the full step from 0 to ∞, where tanh is
           1; no shortened step lowers |f| short of the floats' range. *)
        let f x = Nx.sub_s (Nx.tanh x) 0.5 in
        let s =
          System.solve one
            (System.newton ~derivative:(fun _ dx -> Nx.mul_s dx 1e-310))
            ~linear:Linear.dense ~tol:tight ~budget:5 f (vec [| 0. |])
        in
        satisfies ~claim:"is finite" float_exact Float.is_finite
          (Nx.item [ 0 ] (Solution.best s)));
    test "x² + 1 = 0 stalls at the minimum of |f|" (fun () ->
        let s = newton (fun x -> Nx.add_s (Nx.square x) 1.) (vec [| 0.5 |]) in
        equal Oracle.status_t Stalled (Oracle.status s));
    test "compiled is within tol of the zero" (fun () ->
        Compiled.here ();
        let a = jacobian (Nx.eye f64 3) (Nx.zeros f64 [| 3 |]) in
        let z = vec [| 0.5; -1.; 0.2 |] in
        let f b = Solution.get (newton (field a b) (Nx.zeros f64 [| 3 |])) in
        within_tight z (Rune.jit' f (rhs a z)));
    test "compiled with a banded solver agrees with eager to tol" (fun () ->
        Compiled.here ();
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
        (* Both answers are within [tight] of the zero. *)
        equal
          (Oracle.tensor ~rel:2e-12 ~abs:2e-12 ())
          (solve lambda) (Rune.jit' solve lambda));
  ]

(* Broyden *)

let broyden_tests =
  [
    prop "its zero is the one the system is built around" system (fun (a, z) ->
        within_tight z
          (Solution.get (broyden (field a (rhs a z)) (Nx.zeros_like z))));
    test
      "it finds the zero of a tiny system whose differences point the wrong way"
      (fun () ->
        (* [x / 2 − 3 s atan (x / s) / 2 = s b] at [s = 2⁻⁵⁴⁰]: the differences'
           probes, √ε away, saturate the arctangent, so [B] starts as [1/2]
           while the slope at the zero is negative. Only the rank-one updates,
           from steps whose squares are below the floats' range, turn it. *)
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
        within_tight z (Nx.div_s (Solution.get st) s));
    prop "near a simple zero a converged answer is within tol" cubics
      (near_simple_zero (fun _ -> System.broyden));
    prop "a converged solve's error meets tol at its estimate" ~count:40
      ~examples:[ fast ] seeded (fun ((a, z), (_, k)) ->
        let f = field a (rhs a z) in
        let s =
          System.solve one System.broyden ~linear:Linear.dense ~tol:(Tol.ulps k)
            ~budget:100 f (seed z)
        in
        cover "converges" (is Converged s);
        error_meets k s);
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
    test "compiled is within tol of the zero" (fun () ->
        Compiled.here ();
        let a = jacobian (Nx.eye f64 3) (Nx.zeros f64 [| 3 |]) in
        let z = vec [| 0.5; -1.; 0.2 |] in
        let f b = Solution.get (broyden (field a b) (Nx.zeros f64 [| 3 |])) in
        within_tight z (Rune.jit' f (rhs a z)));
  ]

(* Anderson *)

(* A linear map [z + c ⊙ (x − z)] of up to three components whose largest rate
   is [top], the others anywhere below it, down to 0, and a tolerance: [k] ulps,
   or [Tol.rel (k ε)] for [k] up to [10⁹·⁶]. A zero's components differ in
   magnitude by up to [10³], and one of rate above 1/2 starts within 300 ulps of
   its zero, or five times a looser tolerance, the others within 1, so that a
   fast large one can collapse while the error lies in a slow small one; a zero
   below the normal floats is started as {!start_at} says. *)
let linear_maps =
  let open Gen in
  (let* n = int_range 1 3 in
   let* top = one_of (List.map constant [ 0.5; 0.9; 0.99 ]) in
   let+ rest = list ~size:(constant (n - 1)) (float_range 0. top)
   and+ zs = list ~size:(constant n) magnitude
   and+ decades =
     list ~size:(constant n) (one_of [ constant 1.; constant 1e-3 ])
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
      Format.fprintf ppf "top = %g@ c = %a@ z = %a@ x0 = %a@ %s %g" top Nx.pp c
        Nx.pp z Nx.pp x0
        (if rel then "rel" else "ulps")
        k)

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

let anderson_tests =
  [
    prop "its zero is the fixed point"
      Gen.(pair system (int_range 0 3))
      (fun ((_, z), memory) ->
        cover "Picard" (memory = 0);
        cover "a memory at least the unknowns" (memory >= Nx.dim 0 z);
        within_tight z
          (Solution.get (anderson ~memory (fixed z) (Nx.zeros_like z))));
    prop "a linear map's answer is within c / (1 − c) of tol, c its top rate"
      linear_maps (fun (top, (c, (z, (x0, (rel, k))))) ->
        (* If the step from [x] shrinks the distance by [L], the answer [x + δ]
           is within [L |δ| / (1 − L)] and [|δ|] within [tol]: [L] is at most
           [top] here, and at most 1/2 brings the answer within [tol], to
           rounding. *)
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
          at_most float_exact ~than:(bound ~top k) (distance k s z));
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
          at_most float_exact ~than:(bound ~top:0.99 k)
            (distance k s (vec [| z |])));
    prop "a converged solve's error meets tol at its estimate" ~count:40
      ~examples:[ fast; slow ] seeded (fun ((_, z), (factor, k)) ->
        (* The fixed point's residual [factor] times too small contracts by [1 −
           1 / (2 factor)]. *)
        let s =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol:(Tol.ulps k) ~budget:300
            (fun x -> Nx.div_s (fixed z x) factor)
            (seed z)
        in
        cover "converges" (is Converged s);
        cover "stalls" (is Stalled s);
        error_meets k s);
    test "a lane whose computed zero is 120 ulps off is within its bound"
      (fun () ->
        (* f's rounding in the large component, coupled into the small one,
           leaves the small one's computed zero about 120 ulps from √a₀: under a
           few ulps that is the evaluation error the error does not see. The
           map's rate near the zero is about 0.56, above 1/2, so under 1024
           ulps, where that rounding is small, the lane converges within [0.56 /
           0.44] of [tol] to rounding. *)
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
                  (Nx.matmul m (Nx.reshape [| 2; 1 |] (Nx.sub a (Nx.square x)))))
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
        equal Oracle.status_t Converged (Oracle.status s);
        at_most float_exact ~than:(bound ~top:0.56 k) (distance k s root));
    test "a memory reaches the zero of a map that does not contract" (fun () ->
        (* 2 x + sin x = b: the Picard map x + f x expands by up to 3, and
           mixing two steps still reaches its zero, which Newton in OCaml floats
           gives. *)
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
        equal Oracle.status_t Converged (Oracle.status s);
        within_tight (vec (Array.map zero b)) (Solution.best s));
    test "a small slow component beside a large fast one converges in ulps"
      (fun () ->
        (* z + c ⊙ (x − z) with rates 0.027 and 0.94 and zeros 1.8 and 0.0028:
           measured in [tol]'s scaled norm, the readings see the slow small
           component settle at 0.94, and the answer is within [(0.94 + 1 / 64) /
           0.06] of [tol]; a Euclidean secant, dominated by the large component,
           never settles it. *)
        let h = Float.of_string in
        let c = vec [| h "0x1.bd7651a93286dp-6"; h "0x1.e0ce22dde1eafp-1" |]
        and z = vec [| h "-0x1.cc26fb884a3cep+0"; h "-0x1.7529c8bd30666p-9" |]
        and x0 =
          vec [| h "-0x1.b78f6b62e373dp+0"; h "-0x1.752bf6ef80993p-9" |]
        in
        let s =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol:(Tol.ulps 64.) ~budget:20000
            (fun x -> Nx.mul (Nx.sub_s c 1.) (Nx.sub x z))
            x0
        in
        equal Oracle.status_t Converged (Oracle.status s);
        at_most float_exact
          ~than:(bound ~top:(Nx.item [ 1 ] c) 64.)
          (distance 64. s z));
    test "a component converging fast hides a slow one's error" (fun () ->
        (* (−x₀, (z − x₁) / 100) from (1, z + 300 ulps): the readings see the
           first component's collapse while the error lies in the second, and
           the answer converges about 50 times past [Tol.ulps 4]. *)
        let z = 1. in
        let s =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol:(Tol.ulps 4.) ~budget:2000
            (fun x ->
              Nx.stack
                [
                  Nx.neg (Nx.get [ 0 ] x);
                  Nx.div_s (Nx.rsub_s z (Nx.get [ 1 ] x)) 100.;
                ])
            (vec [| 1.; z +. (300. *. epsilon_float) |])
        in
        equal Oracle.status_t Converged (Oracle.status s);
        expect (Printf.sprintf "%.0f" (distance 4. s (vec [| 0.; z |])))
        @@ __POS_OF__ {| 52 |});
    test "a slow map toward 0 converges in ulps at 0 alone" (fun () ->
        (* x − x / 100 from a thousand subnormal spacings: the steps round to 0
           at fifty spacings from 0, where [f] underflows too. That point is no
           zero a tolerance in ulps admits; one in absolute terms admits it. *)
        let solve tol =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol ~budget:2000
            (fun x -> Nx.mul_s x (-0.01))
            (vec [| 1000. *. spacing |])
        in
        let s = solve (Tol.ulps 4.) in
        satisfies ~claim:"Stalled, or Converged at 0" float_exact
          (fun x -> (not (is Converged s)) || x = 0.)
          (Nx.item [ 0 ] (Solution.best s));
        equal Oracle.status_t Converged (Oracle.status (solve (Tol.abs 1e-300))));
    cases ~id:(Printf.sprintf "ulps %g")
      "Picard on (z − x) / 2 from 0 reaches its small component" [ 16.; 256. ]
      (fun k ->
        (* The rate is 1/2: each step halves the distance, so the error [2 |δ|]
           meets the tolerance once the steps fall to an eighth of it, which the
           floats resolve in both components, and the answer is within it. *)
        let z = vec [| 1.; 1e-3 |] in
        let s =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol:(Tol.ulps k) ~budget:400
            (fun x -> Nx.div_s (Nx.sub z x) 2.)
            (vec [| 1.; 0. |])
        in
        equal Oracle.status_t Converged (Oracle.status s);
        at_most float_exact ~than:(bound ~top:0.5 k) (distance k s z));
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
    test "compiled without memory equals eager, bit for bit" (fun () ->
        Compiled.here ();
        let z = vec [| 0.5; -1.; 0.2 |] in
        let f z =
          Solution.get (anderson ~memory:0 (fixed z) (Nx.zeros_like z))
        in
        equal (Oracle.tensor ()) (f z) (Rune.jit' f z));
    test "compiled with memory is within tol of the zero" (fun () ->
        Compiled.here ();
        let z = vec [| 0.5; -1.; 0.2 |] in
        let f z =
          Solution.get (anderson ~memory:2 (fixed z) (Nx.zeros_like z))
        in
        within_tight z (Rune.jit' f z));
    test "a negative memory raises" (fun () ->
        raises_from "Jera.System.anderson" (fun () ->
            System.anderson ~memory:(-1)));
  ]

(* Solve *)

let solve_tests =
  [
    test "a structure's float tensors are one vector, others carried" (fun () ->
        (* p + q² = 3 and p q = 1 near (1, 1.4), with a counter carried. *)
        let s = Nx.Ptree.(pair (pair tensor tensor) tensor) in
        let f ((p, q), k) =
          ( ( Nx.sub_s (Nx.add p (Nx.square (Nx.reshape [| 1 |] q))) 3.,
              Nx.sub_s (Nx.mul (Nx.reshape [||] p) q) 1. ),
            k )
        in
        let derivative x dx = snd (Rune.jvp s s f x dx) in
        let x =
          Solution.get
            (System.solve s
               (System.newton ~derivative)
               ~linear:Linear.dense ~tol:tight ~budget:50 f
               ((vec [| 1. |], scalar 1.4), Nx.scalar Nx.int32 7l))
        in
        equal
          (Oracle.structure ~abs:1e-12 s)
          ((vec [| 0. |], scalar 0.), Nx.scalar Nx.int32 7l)
          (f x));
    test "a guess where f is not finite ends Not_finite" (fun () ->
        let s = newton (fun x -> Nx.sub_s (Nx.sqrt x) 1.) (vec [| -1. |]) in
        equal Oracle.status_t Not_finite (Oracle.status s));
    test "an estimate where f is not finite ends Not_finite" (fun () ->
        (* Picard on log x − x from 0.5 steps to log 0.5 < 0. *)
        let s =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol:tight ~budget:10
            (fun x -> Nx.sub (Nx.log x) x)
            (vec [| 0.5 |])
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    test
      "a point that overflows where f is finite ends Stalled, at a finite best"
      (fun () ->
        (* Picard on the constant 1e308 from 1e308 steps to ∞, where f is still
           1e308. *)
        let s =
          System.solve one
            (System.anderson ~memory:0)
            ~linear:Linear.dense ~tol:tight ~budget:10
            (fun x -> Nx.full_like x 1e308)
            (vec [| 1e308 |])
        in
        equal Oracle.status_t Stalled (Oracle.status s);
        equal (Oracle.tensor ()) (vec [| 1e308 |]) (Solution.best s));
    test "a budget of one iteration ends Budget_spent" (fun () ->
        let a = jacobian (Nx.eye f64 2) (vec [| 0.; 0. |]) in
        let f = field a (vec [| 1.; 2. |]) in
        let s =
          System.solve one
            (System.newton ~derivative:(derivative f))
            ~linear:Linear.dense ~tol:tight ~budget:1 f
            (vec [| 0.; 0. |])
        in
        equal Oracle.status_t Budget_spent (Oracle.status s));
    test "a system of no unknowns converges after one evaluation" (fun () ->
        let s =
          System.solve one System.broyden ~linear:Linear.dense ~tol:tight
            ~budget:5 Fun.id (vec [||])
        in
        equal Oracle.status_t Converged (Oracle.status s);
        equal int32 1l (Nx.item [] (Solution.evaluations s)));
    prop "grad is the implicit derivative J⁻ᵀ 1" system (fun (a, z) ->
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
    prop "jvp is the implicit tangent J⁻¹ v" system (fun (a, z) ->
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
    test "a lane that did not converge has a zero derivative" (fun () ->
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
    test "a derivative at a singular J raises" (fun () ->
        (* (x − θ)² from its zero θ: Broyden's differences step off it and
           converge at once, but J = 0 at the zero. *)
        let solve theta =
          Solution.get
            (broyden (fun x -> Nx.square (Nx.sub x theta)) (vec [| 1. |]))
        in
        raises_from "Jera.System.solve" (fun () ->
            Rune.grad' (fun t -> Nx.sum (solve t)) (vec [| 1. |])));
    test "vmap is each lane's solve" (fun () ->
        let a = jacobian (Nx.eye f64 2) (Nx.zeros f64 [| 2 |]) in
        let solve b =
          Solution.get (newton (field a b) (Nx.zeros f64 [| 2 |]))
        in
        let bs = Nx.create f64 [| 3; 2 |] [| 1.; 2.; 0.; 0.; -3.; 0.5 |] in
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (Nx.stack (List.init 3 (fun i -> solve (Nx.get [ i ] bs))))
          (Rune.vmap' solve bs));
    test "a budget below 1 raises" (fun () ->
        raises_from "Jera.System.solve" (fun () ->
            System.solve one System.broyden ~linear:Linear.dense ~tol:tight
              ~budget:0 Fun.id (vec [| 1. |])));
    test "float tensors of two dtypes raise" (fun () ->
        raises_from "Jera.System.solve" (fun () ->
            System.solve
              Nx.Ptree.(pair tensor tensor)
              System.broyden ~linear:Linear.dense ~tol:tight ~budget:5 Fun.id
              (vec [| 1. |], Nx.ones Nx.float32 [| 1 |])));
    test "an f of another shape raises" (fun () ->
        raises_from "Jera.System.solve" (fun () ->
            newton (fun x -> Nx.concatenate ~axis:0 [ x; x ]) (vec [| 1. |])));
    test "a guess that holds no tensor raises" (fun () ->
        raises_from "Jera.System.solve" (fun () ->
            System.solve Nx.Ptree.unit System.broyden ~linear:Linear.dense
              ~tol:tight ~budget:5 Fun.id ()));
    test "a precondition of another shape raises" (fun () ->
        let linear =
          Linear.cg ~rel:1e-12 ~budget:10 ~precondition:(fun u ->
              Nx.concatenate ~axis:0 [ u; u ])
        in
        raises_from "Jera.System.solve" (fun () ->
            newton ~linear (fun x -> Nx.sub_s x 1.) (vec [| 0.; 0. |])));
    test "a failed linear solve ends the lane Stalled" (fun () ->
        (* cg on J = −I, which is not positive-definite. *)
        let linear = Linear.cg ~rel:1e-12 ~budget:10 ~precondition:Fun.id in
        let s = newton ~linear (fun x -> Nx.rsub_s 1. x) (vec [| 0.; 0. |]) in
        equal Oracle.status_t Stalled (Oracle.status s));
  ]

(* Lanes *)

(* [l] lanes of systems [a x + sin x = b] of [k] unknowns each, built around
   zeros [z], with the same diagonally dominant [a] as [system]; no lane in a
   fifth of the draws. *)
let lane_systems =
  let open Gen in
  (let* l = frequency [ (1, constant 0); (4, int_range 1 5) ] in
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

let in_lanes a b guess =
  System.lanes ~tol:tight ~budget:50 ~jacobian:(lane_jacobian a)
    (lane_field a b) guess

(* That each lane of [x], of shape [lanes; k], is within [tol] of [z]'s. *)
let lanes_within ~rel ~abs z x =
  for i = 0 to Nx.dim 0 z - 1 do
    within ~rel ~abs (Nx.get [ i ] z) (Nx.get [ i ] x)
  done

(* Lanes whose zero has an unknown below the normal floats beside one at 1: the
   steps reach it in the subnormals, whose products round by an absolute 2⁻¹⁰⁷⁵
   rather than relatively; a lane whose step at its zero moves one ulp; a lane
   whose residual is rounding; and a lane whose zero has an unknown at 0 that
   its last step moves by far less than tol. *)
let rounding_lanes =
  List.map
    (fun small ->
      ( Nx.create f64 [| 1; 2; 2 |] [| 4.; 1.; 0.; 4. |],
        Nx.create f64 [| 1; 2 |] [| 1.; small |] ))
    [ 7.5e-311; 8.471802945955e-311; 8.6e-311 ]
  @ [
      ( Nx.create f64 [| 1; 1; 1 |] [| 2.074449998531568 |],
        Nx.create f64 [| 1; 1 |] [| 0.6661511920166958 |] );
      ( Nx.create f64 [| 1; 2; 2 |]
          [|
            4.; -0.0025442380518094533; 0.370970824797177; 3.3353786945226602;
          |],
        Nx.create f64 [| 1; 2 |] [| -0.6718716184867024; 0. |] );
      ( Nx.create f64 [| 1; 3; 3 |]
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
          |],
        Nx.create f64 [| 1; 3 |]
          [| 0.21418297743675652; -1.564054759905721; 0. |] );
    ]

(* f x = x from 2 in lane 0, with the slope 2 above 1.5, 1000 down to 0.9995 and
   −999 below, scaled by [s]. *)
let steep_slope s u =
  let u = Nx.div_s u s in
  Nx.where (Nx.greater_s u 1.5) (Nx.full_like u 2.)
    (Nx.where (Nx.greater_s u 0.9995) (Nx.full_like u 1000.)
       (Nx.full_like u (-999.)))

let lane_tests =
  [
    prop "each lane's zero is the one its system is built around" lane_systems
      (fun (a, z) ->
        cover "no lane" (Nx.dim 0 z = 0);
        cover "several lanes of several unknowns"
          (Nx.dim 0 z > 1 && Nx.dim 1 z > 1);
        let b = Nx.add (apply a z) (Nx.sin z) in
        lanes_within ~rel:1e-12 ~abs:1e-12 z
          (Solution.get (in_lanes a b (Nx.zeros_like z))));
    prop
      "each lane's answer, status and evaluations are its system solved alone"
      lane_systems (fun (a, z) ->
        (* Newton's search with a direct solve per lane, bit for bit. *)
        let b = Nx.add (apply a z) (Nx.sin z) in
        let alone i =
          let a = Nx.get [ i ] a and b = Nx.get [ i ] b in
          let f x = Nx.sub (Nx.add (Nx.matmul a x) (Nx.sin x)) b in
          let jac x = Nx.add a (Nx.mul (Nx.eye f64 (Nx.dim 0 x)) (Nx.cos x)) in
          System.solve one
            (System.newton ~derivative:(fun x dx -> Nx.matmul (jac x) dx))
            ~linear:Linear.dense ~tol:tight ~budget:50 f
            (Nx.zeros_like (Nx.get [ i ] z))
        in
        let read s =
          (Solution.best s, (Solution.is Converged s, Solution.evaluations s))
        in
        let lanes = List.init (Nx.dim 0 z) (fun i -> read (alone i)) in
        if lanes <> [] then
          equal
            (Oracle.structure Nx.Ptree.(pair tensor (pair tensor tensor)))
            ( Nx.stack (List.map fst lanes),
              ( Nx.stack (List.map (fun (_, (c, _)) -> c) lanes),
                Nx.stack (List.map (fun (_, (_, n)) -> n) lanes) ) )
            (read (in_lanes a b (Nx.zeros_like z))));
    prop "seeded lanes two steps from rounding converge"
      ~examples:rounding_lanes lane_systems (fun (a, z) ->
        let b = Nx.add (apply a z) (Nx.sin z) in
        lanes_within ~rel:1e-14 ~abs:1e-14 z
          (Solution.get
             (System.lanes
                ~tol:(Tol.v ~rel:1e-14 ~abs:1e-14)
                ~budget:20 ~jacobian:(lane_jacobian a) (lane_field a b) (seed z))));
    test "leading axes are lanes, whatever their number" (fun () ->
        let a =
          Nx.broadcast_to [| 2; 3; 2; 2 |]
            (Nx.create f64 [| 2; 2 |] [| 4.; 1.; 1.; 3. |])
        in
        let z = Nx.reshape [| 2; 3; 2 |] (Nx.linspace f64 (-1.) 1. 12) in
        let b = Nx.add (apply a z) (Nx.sin z) in
        lanes_within ~rel:1e-12 ~abs:1e-12
          (Nx.reshape [| 6; 2 |] z)
          (Nx.reshape [| 6; 2 |]
             (Solution.get (in_lanes a b (Nx.zeros_like z)))));
    test "a lane with no zero fails alone, with a zero derivative" (fun () ->
        (* Lane 1 is x² + 1 = 0 in its first unknown; in lane 0, x² = 4 gives d
           x / d c = −1 / (2x). *)
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
        equal (Oracle.tensor ())
          (Nx.create Nx.bool [| 2 |] [| true; false |])
          (Solution.ok s);
        equal
          (Oracle.tensor ~rel:1e-9 ~abs:0. ())
          (vec [| -0.25; 0. |])
          (Rune.grad'
             (fun c ->
               Nx.sum (Nx.slice [ Nx.A; Nx.I 0 ] (Solution.best (solve c))))
             c));
    prop "grad in b is each lane's implicit derivative J⁻ᵀ 1" lane_systems
      (fun (a, z) ->
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
        raises_from "Jera.System.lanes" (fun () ->
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
    test "a lane far from its zero whose line search fails stalls" (fun () ->
        (* The map contracts by 0.001 over the step from 2, which leaves 1 the
           bound 0.001 / 0.999, twice the tolerance of 5 · 10⁻⁴. The step from
           0.999 is +0.001, uphill, and no trial lowers the merit; the map moves
           over it by its own length, q = 1, far above its rounding, which drops
           the 0.001 held. *)
        let s =
          System.lanes ~tol:(Tol.v ~rel:0. ~abs:5e-4) ~budget:20
            ~jacobian:(fun x -> Nx.reshape [| 1; 1; 1 |] (steep_slope 1. x))
            Fun.id
            (Nx.create f64 [| 1; 1 |] [| 2. |])
        in
        equal Oracle.status_t Stalled (Oracle.status s));
    test "a lane of tiny unknowns far from its zero stalls" (fun () ->
        (* The lane above in its first unknown, scaled by [s], with a second
           unknown at its zero, and the tolerance halved again: its root mean
           square counts the second's zero error, so the bound at 1 s is 2.8
           times it. Every step of the first is a multiple of [s], and the
           square of the one from 1 s, 10⁻³ s, is below the floats' range: the
           search must still measure the map's contraction over it. *)
        let s = Float.ldexp 1. (-530) in
        let jacobian x =
          let slope = steep_slope s (Nx.slice [ Nx.A; Nx.I 0 ] x) in
          let zero = Nx.zeros_like slope in
          Nx.reshape [| 1; 2; 2 |]
            (Nx.stack ~axis:(-1) [ slope; zero; zero; Nx.ones_like slope ])
        in
        let st =
          System.lanes
            ~tol:(Tol.v ~rel:0. ~abs:(2.5e-4 *. s))
            ~budget:20 ~jacobian Fun.id
            (Nx.create f64 [| 1; 2 |] [| 2. *. s; 0. |])
        in
        equal Oracle.status_t Stalled (Oracle.status st));
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
        lanes_within ~rel:1e-12 ~abs:1e-12 z (Nx.div_s (Solution.get st) s));
    test "compiled lanes are within tol of their zeros" (fun () ->
        Compiled.here ();
        let a =
          Nx.broadcast_to [| 3; 2; 2 |]
            (Nx.create f64 [| 2; 2 |] [| 4.; 1.; 1.; 3. |])
        in
        let z = Nx.reshape [| 3; 2 |] (Nx.linspace f64 (-1.) 1. 6) in
        let f b = Solution.get (in_lanes a b (Nx.zeros f64 [| 3; 2 |])) in
        lanes_within ~rel:1e-12 ~abs:1e-12 z
          (Rune.jit' f (Nx.add (apply a z) (Nx.sin z))));
    test "lanes of no unknowns converge after one evaluation" (fun () ->
        let s =
          System.lanes ~tol:tight ~budget:5
            ~jacobian:(fun _ -> Nx.zeros f64 [| 3; 0; 0 |])
            Fun.id
            (Nx.zeros f64 [| 3; 0 |])
        in
        equal (Oracle.tensor ()) (Nx.ones Nx.bool [| 3 |]) (Solution.ok s);
        equal (Oracle.tensor ()) (Nx.ones Nx.int32 [| 3 |])
          (Solution.evaluations s));
    test "a scalar guess raises" (fun () ->
        raises_from "Jera.System.lanes" (fun () ->
            System.lanes ~tol:tight ~budget:5 ~jacobian:Fun.id Fun.id
              (scalar 1.)));
    test "a jacobian of another shape raises" (fun () ->
        raises_from "Jera.System.lanes" (fun () ->
            System.lanes ~tol:tight ~budget:5 ~jacobian:Fun.id Fun.id
              (Nx.zeros f64 [| 2; 1 |])));
    test "an f of another shape raises" (fun () ->
        raises_from "Jera.System.lanes" (fun () ->
            System.lanes ~tol:tight ~budget:5
              ~jacobian:(fun _ -> Nx.ones f64 [| 2; 1; 1 |])
              (fun x -> Nx.concatenate ~axis:0 [ x; x ])
              (Nx.zeros f64 [| 2; 1 |])));
    test "a budget below 1 raises" (fun () ->
        raises_from "Jera.System.lanes" (fun () ->
            System.lanes ~tol:tight ~budget:0
              ~jacobian:(fun _ -> Nx.ones f64 [| 2; 1; 1 |])
              Fun.id
              (Nx.zeros f64 [| 2; 1 |])));
  ]

let () =
  exit
    (run "Jera.System"
       [
         group "newton" newton_tests;
         group "broyden" broyden_tests;
         group "anderson" anderson_tests;
         group "solve" solve_tests;
         group "lanes" lane_tests;
       ])
