(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Ode's methods, marches and solves. The trusted side is a forced pendulum
   integrated in OCaml floats by a fine classical Runge–Kutta march, closed-form
   solutions, the methods' stability polynomials on y' = λy, and finite
   differences. *)

open Windtrap
open Jera

let f64 = Nx.float64
let scalar x = Nx.scalar f64 x
let vec a = Nx.create f64 [| Array.length a |] a
let one = Nx.Ptree.tensor
let pair = Nx.Ptree.(pair tensor tensor)
let formula m = (m :> ([ `Formula ], _, _, _) Ode.t)
let embedded m = (m :> ([ `Embedded ], _, _, _) Ode.t)
let raises_in fn f = raises_match (Exn.invalid_arg ~sub:fn) f

(* A forced pendulum with a time-varying stiffness: every elementary
   differential of order five is nonzero, so a method that misses one order
   condition shows a lower order. Its Jacobian, [[0, 1], [−cos q + 0.3 cos 2t,
   0]], has maximum norm at most 1.3. *)
let forced t (q, p) =
  ( p,
    Nx.add
      (Nx.neg (Nx.sin q))
      (Nx.mul (Nx.mul_s (Nx.cos (Nx.mul_s t 2.)) 0.3) q) )

let forced_lipschitz = 1.3

let forced_host t (q, p) =
  (p, -.Float.sin q +. (0.3 *. Float.cos (2. *. t) *. q))

(* The state at [t] of [field] from [y0], by 2¹⁴ classical steps in OCaml
   floats: its error, about h⁴ ≈ 10⁻¹⁵, is below every bound compared with
   it. *)
let integrate field y0 t =
  let n = 1 lsl 14 in
  let h = t /. Float.of_int n in
  let ( +^ ) (a, b) (c, d) = (a +. c, b +. d)
  and ( *^ ) s (a, b) = (s *. a, s *. b) in
  let rec go i y =
    if i = n then y
    else
      let t = Float.of_int i *. h in
      let k1 = field t y in
      let k2 = field (t +. (h /. 2.)) (y +^ (h /. 2. *^ k1)) in
      let k3 = field (t +. (h /. 2.)) (y +^ (h /. 2. *^ k2)) in
      let k4 = field (t +. h) (y +^ (h *^ k3)) in
      go (i + 1) (y +^ (h /. 6. *^ (k1 +^ (2. *^ k2) +^ (2. *^ k3) +^ k4)))
  in
  go 0 y0

(* The forced pendulum's state at [t] from (1, 0). *)
let reference t = integrate forced_host (1., 0.) t

(* The global error that steps whose local errors sum to [estimates] can make
   over [span] on a field of Lipschitz constant [lipschitz]: e^(L T) times the
   sum (Hairer, Nørsett and Wanner, I, §II.3). It assumes each step's error is
   below its estimate, which holds where the estimate, of the embedded formula's
   order, dominates the step's error, of one order more. *)
let propagated ~lipschitz ~span estimates =
  Float.exp (lipschitz *. span) *. estimates

let decay k _ y = Nx.neg (Nx.mul k y)
let jvp f t y dy = snd (Rune.jvp' (f t) y dy)
let kvaerno ?mass f = Ode.kvaerno5 ?mass ~linear:Linear.dense (jvp f)

(* y' = k t^(k−1), whose solution from 0 is t^k. *)
let power k t y =
  Nx.add (Nx.zeros_like y)
    (Nx.mul_s (Nx.pow_s t (Float.of_int (k - 1))) (Float.of_int k))

(* Methods *)

let march_error m n =
  let q, p =
    Ode.march pair m ~steps:n forced
      ~at:(vec [| 0.; 2. |])
      (scalar 1., scalar 0.)
  in
  let rq, rp = reference 2. in
  Float.max
    (Float.abs (Nx.item [ 1 ] q -. rq))
    (Float.abs (Nx.item [ 1 ] p -. rp))

(* One step of [m] on y' = λy from 1: the stability polynomial at z = λh. *)
let growth m z =
  let y =
    Ode.march one m ~steps:1
      (fun _ y -> Nx.mul_s y z)
      ~at:(vec [| 0.; 1. |])
      (scalar 1.)
  in
  Nx.item [ 1 ] y

let taylor k z =
  let rec go i term acc =
    if i > k then acc
    else go (i + 1) (term *. z /. Float.of_int (i + 1)) (acc +. term)
  in
  go 0 1. 0.

(* Upwind differences of u_t + u_x = 0 on a periodic grid of unit cells, whose
   Euler steps keep the total variation for sizes up to 1. *)
let upwind _ u =
  let n = Nx.dim 0 u in
  let left =
    Nx.concatenate ~axis:0
      [ Nx.slice [ Nx.R (n - 1, n) ] u; Nx.slice [ Nx.R (0, n - 1) ] u ]
  in
  Nx.neg (Nx.sub u left)

let variation u =
  let n = Array.length u in
  let tv = ref 0. in
  for j = 0 to n - 1 do
    tv := !tv +. Float.abs (u.(j) -. u.((j + n - 1) mod n))
  done;
  !tv

let square_wave = Array.init 16 (fun j -> if j / 4 mod 2 = 0 then 1. else -1.)

let methods =
  [
    cases "each method reaches its order on a forced pendulum"
      ~id:(fun (n, _, _, _) -> n)
      [
        ("euler", Ode.euler, 1, 256);
        ("ssprk3", Ode.ssprk3, 3, 32);
        ("rk4", Ode.rk4, 4, 16);
        ("bs3", formula Ode.bs3, 3, 32);
        ("tsit5", formula Ode.tsit5, 5, 32);
        ("dopri5", formula Ode.dopri5, 5, 16);
      ]
      (fun (_, m, p, n) ->
        (* Between n and 2n steps the error's slope is p + O(h); at these n the
           next order's term can lift it by up to one. *)
        let order = Oracle.slope (march_error m n) (march_error m (2 * n)) in
        at_least (float 1e-9) ~than:(Float.of_int p -. 0.25) order;
        at_most (float 1e-9) ~than:(Float.of_int p +. 1.25) order);
    cases "a step on y' = λy is the method's stability polynomial"
      ~id:(fun (n, _, _, _) -> n)
      [
        ("euler", Ode.euler, 1, 0.);
        ("ssprk3", Ode.ssprk3, 3, 0.);
        ("rk4", Ode.rk4, 4, 0.);
        ("bs3", formula Ode.bs3, 3, 0.);
        (* bᵀ A⁵ 1 of Dormand and Prince's tableau, in rationals. *)
        ("dopri5", formula Ode.dopri5, 5, 1. /. 600.);
      ]
      (fun (_, m, order, sixth) ->
        (* A step is a polynomial in z of degree at most 6, so 7 points decide
           it. Its terms are at most |z|^k / k!, so it rounds within a few ulps
           of e^|z|. *)
        List.iter
          (fun z ->
            equal
              ~msg:(Printf.sprintf "z = %g" z)
              (float (1e-14 *. Float.exp (Float.abs z)))
              (taylor order z +. (sixth *. (z ** 6.)))
              (growth m z))
          [ -3.; -2.5; -1.; -0.3; 0.4; 0.7; 1. ]);
    prop "an ssprk3 step at Euler's limit keeps Euler's total-variation bound"
      ~examples:[ square_wave ]
      Gen.(array ~size:(constant 16) (float_range (-1.) 1.))
      (fun u0 ->
        let jumps = Array.mapi (fun j a -> a -. u0.((j + 15) mod 16)) u0 in
        cover "a jump above 1" (Array.exists (fun d -> Float.abs d > 1.) jumps);
        let u =
          Ode.march one Ode.ssprk3 ~steps:1 upwind
            ~at:(vec [| 0.; 1.; 2.; 3.; 4. |])
            (vec u0)
        in
        (* A step is a convex combination of shifts, rounded a few times per
           cell. *)
        let bound = variation u0 *. (1. +. (64. *. epsilon_float)) in
        for i = 1 to 4 do
          at_most float_exact ~than:bound
            (variation (Oracle.floats (Nx.get [ i ] u)))
        done);
    cases "a step and its estimate are exact on fields of time to their orders"
      ~id:(fun (n, _, _, _) -> n)
      [
        ("bs3", embedded Ode.bs3, 3, 2);
        ("tsit5", embedded Ode.tsit5, 5, 4);
        ("dopri5", embedded Ode.dopri5, 5, 4);
        ("kvaerno5", kvaerno (power 5), 5, 4);
      ]
      (fun (_, m, p, q) ->
        (* The step of order p integrates y' = p t^(p−1) exactly, so y(1) is 1
           whatever the steps, and the embedded formula of order q integrates y'
           = q t^(q−1) exactly, so every estimate is zero. This checks the
           quadrature conditions alone. *)
        let solve f =
          Ode.solve one m
            ~tol:(Tol.v ~rel:1e-6 ~abs:1e-9)
            ~budget:400 f ~t0:(scalar 0.) ~t1:(scalar 1.) (scalar 0.)
        in
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (scalar 1.)
          (Solution.get (solve (power p)));
        at_most float_exact ~than:1e-14
          (Nx.item [] (Solution.error (solve (power q)))));
  ]

(* Tableaus *)

let heun_a = [| [||]; [| 1. |] |]
let heun_b = [| 0.5; 0.5 |]
let heun_c = [| 0.; 1. |]

let tableau =
  [
    test "a tableau of rk4's coefficients marches as rk4" (fun () ->
        let m =
          Ode.tableau
            ~a:[| [||]; [| 0.5 |]; [| 0.; 0.5 |]; [| 0.; 0.; 1. |] |]
            ~b:[| 1. /. 6.; 1. /. 3.; 1. /. 3.; 1. /. 6. |]
            ~c:[| 0.; 0.5; 0.5; 1. |]
        in
        let run m =
          Ode.march pair m ~steps:3 forced
            ~at:(vec [| 0.; 0.4; 1. |])
            (scalar 1., scalar 0.)
        in
        equal (Oracle.structure pair) (run Ode.rk4) (run m));
    test "Heun's tableau takes Heun's step" (fun () ->
        (* On y' = y from 1, a step of 1/2 is 1 + h + h²/2 = 1.625, exact. *)
        equal (Oracle.tensor ())
          (vec [| 1.; 1.625 |])
          (Ode.march one
             (Ode.tableau ~a:heun_a ~b:heun_b ~c:heun_c)
             ~steps:1
             (fun _ y -> y)
             ~at:(vec [| 0.; 0.5 |])
             (scalar 1.)));
    cases "raises unless it is explicit, finite and its b sums to 1"
      ~id:(fun (n, _, _, _) -> n)
      [
        ("no stage", [||], [||], [||]);
        ("c is short", heun_a, heun_b, [| 0. |]);
        ("a has one row for two stages", [| [||] |], heun_b, heun_c);
        ("row 1 of a has two elements", [| [||]; [| 1.; 0. |] |], heun_b, heun_c);
        ("a NaN", heun_a, heun_b, [| 0.; Float.nan |]);
        ("an infinity", [| [||]; [| Float.infinity |] |], heun_b, heun_c);
        ("b sums to 1.1", heun_a, [| 0.5; 0.6 |], heun_c);
      ]
      (fun (_, a, b, c) ->
        raises_in "Jera.Ode.tableau" (fun () -> Ode.tableau ~a ~b ~c));
  ]

(* kvaerno5 *)

(* Prothero and Robinson's field y' = λ (y − sin t) + cos t, whose solution from
   0 is sin t whatever λ: for λ = −10⁶ an explicit method needs a step below
   2·10⁻⁶ on [0, 10]. Its one-sided Lipschitz constant is λ < 0, so the steps'
   errors do not grow and the global error is at most their sum (Hairer, Nørsett
   and Wanner, I, §I.10). *)
let prothero lambda t y =
  Nx.add (Nx.mul (Nx.sub y (Nx.sin t)) lambda) (Nx.cos t)

let stiff_tol = Tol.v ~rel:1e-8 ~abs:1e-10

(* The index-1 DAE y₁' = y₂ − y₁, 0 = y₂ − cos t: from y₁(0) = 1/2, y₁ = (cos t
   + sin t) / 2 and y₂ = cos t. *)
let dae t v =
  Nx.stack
    [
      Nx.sub (Nx.get [ 1 ] v) (Nx.get [ 0 ] v);
      Nx.sub (Nx.get [ 1 ] v) (Nx.cos t);
    ]

let dae_mass v = Nx.mul v (vec [| 1.; 0. |])

(* Robertson's chemical kinetics to t = 40, the rates [k] as the argument. *)
let robertson k =
  let field _ y =
    let y1 = Nx.get [ 0 ] y and y2 = Nx.get [ 1 ] y and y3 = Nx.get [ 2 ] y in
    let k1 = Nx.get [ 0 ] k and k2 = Nx.get [ 1 ] k and k3 = Nx.get [ 2 ] k in
    let r1 = Nx.mul k1 y1
    and r2 = Nx.mul k2 (Nx.square y2)
    and r3 = Nx.mul k3 (Nx.mul y2 y3) in
    Nx.stack [ Nx.add (Nx.neg r1) r3; Nx.sub (Nx.sub r1 r2) r3; r2 ]
  in
  Solution.get
    (Ode.solve one (kvaerno field)
       ~tol:(Tol.v ~rel:1e-6 ~abs:1e-10)
       ~budget:400 field ~t0:(scalar 0.) ~t1:(scalar 40.)
       (vec [| 1.; 0.; 0. |]))

let kvaerno5 =
  [
    test "follows a stiff field's slow solution in thousands of evaluations"
      (fun () ->
        let f = prothero (scalar (-1e6)) in
        let s =
          Ode.solve one (kvaerno f) ~tol:stiff_tol ~budget:500 f ~t0:(scalar 0.)
            ~t1:(scalar 10.) (vec [| 0. |])
        in
        at_most float_exact
          ~than:(Nx.item [ 0 ] (Solution.error s))
          (Float.abs (Nx.item [ 0 ] (Solution.get s) -. Float.sin 10.));
        (* An explicit method's stability needs 5·10⁶ steps here. *)
        at_most int32 ~than:5000l (Nx.item [] (Solution.evaluations s)));
    test "solves an index-1 DAE: y₁' = y₂ − y₁, 0 = y₂ − cos t" (fun () ->
        (* y₁' = y₂ − y₁ contracts at rate 1 and the constraint fixes y₂, so the
           global error is at most the estimates' sum. *)
        let s =
          Ode.solve one
            (kvaerno ~mass:dae_mass dae)
            ~tol:stiff_tol ~budget:500 dae ~t0:(scalar 0.) ~t1:(scalar 2.)
            (vec [| 0.5; 1. |])
        in
        let error = Nx.item [] (Nx.sum (Solution.error s)) in
        equal
          (Oracle.tensor ~abs:error ())
          (vec [| (Float.cos 2. +. Float.sin 2.) /. 2.; Float.cos 2. |])
          (Solution.get s));
    test "a mass scales the field: 2 y' = −2 y is y' = −y" (fun () ->
        let f _ y = Nx.mul_s y (-2.) in
        let s =
          Ode.solve one
            (kvaerno ~mass:(fun v -> Nx.mul_s v 2.) f)
            ~tol:stiff_tol ~budget:500 f ~t0:(scalar 0.) ~t1:(scalar 3.)
            (vec [| 1. |])
        in
        (* y' = −y contracts, so the global error is at most the estimates'
           sum. *)
        at_most float_exact
          ~than:(Nx.item [ 0 ] (Solution.error s))
          (Float.abs (Nx.item [ 0 ] (Solution.get s) -. Float.exp (-3.))));
    cases "its error is within e^(L T) of its estimate on a smooth field"
      ~id:(Printf.sprintf "tol %g") [ 1e-6; 1e-9 ] (fun tl ->
        let m =
          Ode.kvaerno5 ~linear:Linear.dense (fun t y dy ->
              snd (Rune.jvp pair pair (forced t) y dy))
        in
        let s =
          Ode.solve pair m ~tol:(Tol.v ~rel:tl ~abs:tl) ~budget:5000 forced
            ~t0:(scalar 0.) ~t1:(scalar 3.)
            (scalar 1., scalar 0.)
        in
        let (q, p), (eq, ep) = (Solution.get s, Solution.error s) in
        let rq, rp = reference 3. in
        let bound =
          propagated ~lipschitz:forced_lipschitz ~span:3.
            (Nx.item [] eq +. Nx.item [] ep)
        in
        at_most float_exact ~than:bound (Float.abs (Nx.item [] q -. rq));
        at_most float_exact ~than:bound (Float.abs (Nx.item [] p -. rp)));
    test "grad in λ agrees with central differences" (fun () ->
        (* From y₀ = 1 the solution keeps a stiff transient e^(λt). A perturbed
           λ may take other steps, so the difference carries the answers' error,
           about 10⁻¹⁰, over 2·10⁻⁴. *)
        let final lambda =
          let f = prothero lambda in
          Nx.sum
            (Solution.get
               (Ode.solve one (kvaerno f) ~tol:stiff_tol ~budget:500 f
                  ~t0:(scalar 0.) ~t1:(scalar 0.5) (vec [| 1. |])))
        in
        let lambda = scalar (-20.) in
        equal
          (Oracle.tensor ~rel:1e-5 ())
          (Oracle.central ~eps:1e-4 final lambda (scalar 1.))
          (Rune.grad' final lambda));
    (* The march, its stages' Newton iterations and their solves are loops in
       loops: a schedule that runs each stage's loops again in the next one's
       runs for hours, and the limit fails it. *)
    test "the gradient of a stiff solve compiles" (fun () ->
        Compiled.here ();
        (* Robertson's rates sum to zero, so every step keeps y₁ + y₂ + y₃,
           whatever steps the solve takes, and its gradient is zero. The
           components' gradients in k₁ are about 4, rounded over some hundred
           steps. *)
        let g = Rune.grad' (fun k -> Nx.sum (robertson k)) in
        let k = vec [| 0.04; 3e7; 1e4 |] in
        let calls = Rune.evaluations () in
        let compiled = Rune.jit' g k in
        equal ~msg:"no call evaluated" int 0 (Rune.evaluations () - calls);
        equal (Oracle.tensor ~abs:1e-12 ()) (Nx.zeros f64 [| 3 |]) compiled);
    test "compiled, it is within its error of the slow solution" (fun () ->
        Compiled.here ();
        let solve lambda =
          let f = prothero lambda in
          Ode.solve one (kvaerno f) ~tol:stiff_tol ~budget:500 f ~t0:(scalar 0.)
            ~t1:(scalar 1.) (vec [| 0. |])
        in
        let s =
          Rune.jit
            Nx.Ptree.(tensor @-> returns (Solution.ptree tensor))
            solve (scalar (-1e4))
        in
        at_most float_exact
          ~than:(Nx.item [ 0 ] (Solution.error s))
          (Float.abs (Nx.item [ 0 ] (Solution.get s) -. Float.sin 1.)));
  ]

(* Marches *)

let interval_pp ppf (a, b, steps) = Format.fprintf ppf "(%h, %h, %d)" a b steps

(* An interval of a few ulps, where a step's size rounds below the times'
   spacing, or a wide one, a count of steps, either way. *)
let narrow_or_wide =
  let open Gen in
  let narrow =
    let+ a = float_range (-1e3) 1e3 and+ k = int_range 1 16 in
    let b = ref a in
    for _ = 1 to k do
      b := Float.succ !b
    done;
    (a, !b)
  in
  let wide =
    let+ a = float_range (-10.) 10. and+ b = float_range (-10.) 10. in
    (a, if b = a then b +. 1. else b)
  in
  with_pp interval_pp
    (let+ a, b = frequency [ (1, narrow); (1, wide) ]
     and+ steps = int_range 1 16
     and+ backward = bool in
     if backward then (b, a, steps) else (a, b, steps))

(* An interval whose ends have opposite signs, in one or two steps: there the
   last step's start plus its size rounds apart from its end in about a quarter
   of draws. *)
let across_zero =
  let open Gen in
  with_pp interval_pp
    (let+ a = float_range (-10.) (-1e-3)
     and+ b = float_range 1e-3 10.
     and+ steps = int_range 1 2
     and+ backward = bool in
     if backward then (b, a, steps) else (a, b, steps))

(* An interval, a count of three or more steps, and a step k with steps before
   and after it of size h. *)
let interior =
  let open Gen in
  with_pp
    (fun ppf (a, b, steps, k) ->
      Format.fprintf ppf "(%h, %h, %d, %d)" a b steps k)
    (let* steps = int_range 3 8 in
     let+ a = float_range (-10.) 10.
     and+ b = float_range (-10.) 10.
     and+ k = int_range 1 (steps - 2) in
     (a, (if b = a then b +. 1. else b), steps, k))

(* The last step's start as the march places it: a + (steps − 1) h, or b where
   that rounds past it. *)
let last_start (a, b, steps) =
  let h = (b -. a) /. Float.of_int steps in
  let x = a +. (Float.of_int (steps - 1) *. h) in
  if (b > a && x > b) || (b < a && x < b) then b else x

(* The position at t = 1 and t = 2, as functions of the initial position, of a
   stiffness the field captures, and of the end time. *)
let positions q0 =
  let q, _ =
    Ode.march pair Ode.tsit5 ~steps:6 forced
      ~at:(vec [| 0.; 1.; 2. |])
      (q0, Nx.zeros_like q0)
  in
  Nx.sum (Nx.slice [ Nx.R (1, 3) ] q)

let stiffness k =
  let f _ (q, p) = (p, Nx.neg (Nx.mul k (Nx.sin q))) in
  let q, _ =
    Ode.march pair Ode.dopri5 ~steps:10 f
      ~at:(vec [| 0.; 1.5 |])
      (scalar 1., scalar 0.)
  in
  Nx.sum q

let ending t1 =
  let at = Nx.concatenate ~axis:0 [ vec [| 0. |]; Nx.reshape [| 1 |] t1 ] in
  let q, _ =
    Ode.march pair Ode.rk4 ~steps:12 forced ~at (scalar 1., scalar 0.)
  in
  Nx.sum q

(* A march's derivative is the composition's, so central differences at eps =
   10⁻⁶ meet it to their truncation, eps² f''' ≈ 10⁻¹², and rounding, ε |f| /
   eps ≈ 2·10⁻¹⁰. *)
let finite_difference f x = Oracle.central ~eps:1e-6 f x (Nx.ones_like x)
let to_differences = Oracle.tensor ~rel:1e-9 ()

let march =
  [
    test "stacks the state at each time, the start first" (fun () ->
        let y0 = vec [| 1.; 2. |] in
        let y =
          Ode.march one Ode.rk4 ~steps:2
            (fun _ y -> Nx.neg y)
            ~at:(vec [| 0.; 1.; 3. |])
            y0
        in
        equal (array int) [| 3; 2 |] (Nx.shape y);
        equal (Oracle.tensor ()) y0 (Nx.get [ 0 ] y));
    test "one time is the start alone" (fun () ->
        let y =
          Ode.march one Ode.tsit5 ~steps:2
            (fun _ y -> y)
            ~at:(vec [| 1. |]) (scalar 2.)
        in
        equal (Oracle.tensor ()) (vec [| 2. |]) y);
    test "decreasing times march back" (fun () ->
        (* rk4 at h = 1/64 errs by about h⁴ / 120 ≈ 5·10⁻¹⁰. *)
        let y =
          Ode.march one Ode.rk4 ~steps:64
            (fun _ y -> y)
            ~at:(vec [| 1.; 0. |])
            (scalar (Float.exp 1.))
        in
        equal (Oracle.tensor ~rel:1e-9 ()) (scalar 1.) (Nx.get [ 1 ] y));
    test "the field reads the stage times" (fun () ->
        (* y' = 3t² from 0 is t³, which rk4 integrates exactly. *)
        let y =
          Ode.march one Ode.rk4 ~steps:3
            (fun t _ -> Nx.mul_s (Nx.square t) 3.)
            ~at:(vec [| 0.; 0.5; 2. |])
            (scalar 0.)
        in
        equal (Oracle.tensor ~rel:1e-15 ()) (vec [| 0.; 0.125; 8. |]) y);
    test "times and leaves may have different dtypes" (fun () ->
        let y =
          Ode.march one Ode.rk4 ~steps:64
            (fun _ y -> Nx.neg y)
            ~at:(vec [| 0.; 1. |])
            (Nx.scalar Nx.float32 1.)
        in
        (* 64 steps of four stages each round about 2⁸ times in float32. *)
        equal
          (Oracle.tensor ~rel:1e-5 ())
          (Nx.create Nx.float32 [| 2 |] [| 1.; Float.exp (-1.) |])
          y);
    prop "no step starts past its interval's end"
      ~examples:
        [
          (0., 0.3, 10);
          (0.9, 0.3, 7);
          (0., 1.5e-323, 5);
          (1., 1. +. Float.epsilon, 4);
        ]
      narrow_or_wide
      (fun (a, b, steps) ->
        let unit = Float.succ (Float.abs a) -. Float.abs a in
        cover "steps outnumber the span's units"
          (Float.abs (b -. a) < Float.of_int steps *. unit);
        cover "backward" (b < a);
        (* Euler reads the field at each step's start only: 1 past the end, 0
           before it. *)
        let past t =
          Nx.cast f64 (if b > a then Nx.greater_s t b else Nx.less_s t b)
        in
        let y =
          Ode.march one Ode.euler ~steps
            (fun t _ -> past t)
            ~at:(vec [| a; b |])
            (scalar 0.)
        in
        equal (Oracle.tensor ()) (vec [| 0.; 0. |]) y);
    prop "the last step ends at its interval's end"
      ~examples:
        [
          (0., 0.3, 10);
          (0.9, 0.3, 7);
          (0.2, 0.7, 3);
          (-7.074863082399303, 4.437342817489506, 1);
        ]
      across_zero
      (fun (a, b, steps) ->
        let t = last_start (a, b, steps) in
        cover "the last start plus its size rounds apart from the end"
          (t +. (b -. t) <> b);
        (* rk4's last stage reads the field at its step's end: 1 there and 0
           elsewhere, so only a last step ending at b moves y. *)
        let y =
          Ode.march one Ode.rk4 ~steps
            (fun t _ -> Nx.cast f64 (Nx.equal_s t b))
            ~at:(vec [| a; b |])
            (scalar 0.)
        in
        less float_exact ~than:(Float.abs (Nx.item [ 1 ] y)) 0.);
    prop "every other step ends where the next starts"
      ~examples:[ (-8.84, 0.149, 4, 2) ]
      interior
      (fun (a, b, steps, k) ->
        let h = (b -. a) /. Float.of_int steps in
        let start j = a +. Sys.opaque_identity (Float.of_int j *. h) in
        cover "the start plus h rounds apart from the next start"
          (start (k - 1) +. h <> start k);
        (* The field is 1 at step k's start alone, which step k − 1's last stage
           (c = 1) and step k's first (c = 0) read: each adds h / 6. *)
        let y =
          Ode.march one Ode.rk4 ~steps
            (fun t _ -> Nx.cast f64 (Nx.equal_s t (start k)))
            ~at:(vec [| a; b |])
            (scalar 0.)
        in
        equal
          (float_rel ~rel:1e-15 ~abs:0.)
          (2. *. (h *. (1. /. 6.)))
          (Nx.item [ 1 ] y));
    test "a leaf that is not a float is carried unchanged" (fun () ->
        let s = Nx.Ptree.(pair tensor tensor) in
        let _, n =
          Ode.march s Ode.tsit5 ~steps:4
            (fun _ (y, n) -> (Nx.neg y, Nx.mul_s n 7l))
            ~at:(vec [| 0.; 1.; 2. |])
            (scalar 1., Nx.scalar Nx.int32 5l)
        in
        equal (Oracle.tensor ()) (Nx.full Nx.int32 [| 3 |] 5l) n);
    test "grad in the initial state is the finite difference" (fun () ->
        let q0 = scalar 0.7 in
        equal to_differences
          (finite_difference positions q0)
          (Rune.grad' positions q0));
    test "grad in a captured parameter is the finite difference" (fun () ->
        let k = scalar 1.3 in
        equal to_differences
          (finite_difference stiffness k)
          (Rune.grad' stiffness k));
    test "grad in the end time is the finite difference" (fun () ->
        let t1 = scalar 1.9 in
        equal to_differences
          (finite_difference ending t1)
          (Rune.grad' ending t1));
    test "jvp agrees with grad" (fun () ->
        let k = scalar 0.8 in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Rune.grad' stiffness k)
          (snd (Rune.jvp' stiffness k (scalar 1.))));
    test "compiled equals eager" (fun () ->
        Compiled.here ();
        (* A march takes the same steps compiled; sin rounds within ulps. *)
        let q0 = scalar 0.7 in
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (positions q0) (Rune.jit' positions q0));
    test "compiled grad equals eager grad" (fun () ->
        Compiled.here ();
        let k = scalar 1.3 in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Rune.grad' stiffness k)
          (Rune.jit' (Rune.grad' stiffness) k));
    test "vmap is each lane's march" (fun () ->
        let q0 = vec [| 0.7; -0.2; 1.4 |] in
        equal
          (Oracle.tensor ~rel:1e-15 ())
          (Nx.stack (List.init 3 (fun i -> positions (Nx.get [ i ] q0))))
          (Rune.vmap' positions q0));
    test "raises if steps < 1" (fun () ->
        raises_in "Jera.Ode.march" (fun () ->
            Ode.march one Ode.rk4 ~steps:0
              (fun _ y -> y)
              ~at:(vec [| 0.; 1. |])
              (scalar 1.)));
    cases "raises if at is not a non-empty 1-D tensor"
      ~id:(Format.asprintf "%a" Nx.pp_shape)
      [ [| 0 |]; [||]; [| 2; 2 |] ]
      (fun shape ->
        raises_in "Jera.Ode.march" (fun () ->
            Ode.march one Ode.rk4 ~steps:1
              (fun _ y -> y)
              ~at:(Nx.zeros f64 shape) (scalar 1.)));
    test "raises if at is not strictly monotone" (fun () ->
        raises_in "Jera.Ode.march" (fun () ->
            Ode.march one Ode.rk4 ~steps:1
              (fun _ y -> y)
              ~at:(vec [| 0.; 1.; 1. |])
              (scalar 1.)));
    test "raises if the field returns another shape" (fun () ->
        raises_in "Jera.Ode.march" (fun () ->
            Ode.march one Ode.rk4 ~steps:1
              (fun _ y -> Nx.stack [ y; y ])
              ~at:(vec [| 0.; 1. |])
              (scalar 1.)));
    test "raises if the field returns another structure" (fun () ->
        let s = Nx.Ptree.(pair tensor tensor) in
        raises_in "Jera.Ode.march" (fun () ->
            Ode.march s Ode.rk4 ~steps:1
              (fun _ (q, p) -> (q, Nx.stack [ p; p ]))
              ~at:(vec [| 0.; 1. |])
              (scalar 1., scalar 0.)));
  ]

(* Solves *)

(* y' = cos(t) y + sin(3t): smooth, non-autonomous. *)
let forced2 t y = Nx.add (Nx.mul (Nx.cos t) y) (Nx.sin (Nx.mul_s t 3.))

(* A solve by tsit5 of a one-tensor state. *)
let solve1 f ~t0 ~t1 y0 =
  Ode.solve one Ode.tsit5
    ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
    ~budget:1000 f ~t0:(scalar t0) ~t1:(scalar t1) y0

(* Each accepted step of a forward path [p] of [pair] with error [e], as its
   estimate and its two states: successive pieces' errors differ by one step's
   estimate, and the breaks hold the steps' states. *)
let accepted_steps p e =
  let breaks = Oracle.floats (Piecewise.breaks p) in
  let host v t =
    let q, p = Piecewise.eval v (scalar t) in
    [| Nx.item [] q; Nx.item [] p |]
  in
  let rec go i before acc =
    if i + 1 >= Array.length breaks || breaks.(i + 1) <= breaks.(i) then
      List.rev acc
    else
      let a = breaks.(i) and b = breaks.(i + 1) in
      let through = host e ((a +. b) /. 2.) in
      let step = (Array.map2 ( -. ) through before, host p a, host p b) in
      go (i + 1) through (step :: acc)
  in
  go 0 [| 0.; 0. |] []

(* [e] against the tolerance at the larger of the states [v] and [w]: the root
   mean square over the components of e / (abs + rel max (|v|, |w|)). *)
let measure ~rel ~abs (e, v, w) =
  let n = Array.length e in
  let sum = ref 0. in
  for i = 0 to n - 1 do
    let s = abs +. (rel *. Float.max (Float.abs v.(i)) (Float.abs w.(i))) in
    sum := !sum +. ((e.(i) /. s) ** 2.)
  done;
  Float.sqrt (!sum /. Float.of_int n)

let solve =
  [
    cases "each embedded method's error is within e^(L T) of its estimate"
      ~id:(fun (n, _, _) -> n)
      [
        ("bs3", embedded Ode.bs3, 1e-7);
        ("tsit5", embedded Ode.tsit5, 1e-9);
        ("dopri5", embedded Ode.dopri5, 1e-9);
      ]
      (fun (_, m, tol) ->
        let s =
          Ode.solve pair m ~tol:(Tol.v ~rel:tol ~abs:tol) ~budget:5000 forced
            ~t0:(scalar 0.) ~t1:(scalar 2.)
            (scalar 1., scalar 0.)
        in
        let (q, p), (eq, ep) = (Solution.get s, Solution.error s) in
        let rq, rp = reference 2. in
        let bound =
          propagated ~lipschitz:forced_lipschitz ~span:2.
            (Nx.item [] eq +. Nx.item [] ep)
        in
        at_most float_exact ~than:bound (Float.abs (Nx.item [] q -. rq));
        at_most float_exact ~than:bound (Float.abs (Nx.item [] p -. rp)));
    cases "every accepted step's estimate meets tol at its larger state"
      ~id:(fun (n, _, _, _) -> n)
      [
        ("bs3", embedded Ode.bs3, 1e-4, 1e-6);
        ("tsit5", embedded Ode.tsit5, 1e-6, 1e-8);
        ("dopri5", embedded Ode.dopri5, 1e-9, 1e-10);
      ]
      (fun (_, m, rel, abs) ->
        let s =
          Ode.path pair m ~tol:(Tol.v ~rel ~abs) ~budget:400 forced
            ~t0:(scalar 0.) ~t1:(scalar 2.)
            (scalar 1., scalar 0.)
        in
        (* The answer takes each step again from its fraction of the span, so
           its estimate and states differ from the search's by a few ulps. *)
        List.iter
          (fun step ->
            at_most float_exact ~than:(1. +. 1e-9) (measure ~rel ~abs step))
          (accepted_steps (Solution.get s) (Solution.error s)));
    test "a tighter tolerance gives a smaller error" (fun () ->
        let err tol =
          let q, _ =
            Solution.get
              (Ode.solve pair Ode.tsit5 ~tol:(Tol.v ~rel:tol ~abs:tol)
                 ~budget:5000 forced ~t0:(scalar 0.) ~t1:(scalar 2.)
                 (scalar 1., scalar 0.))
          in
          Float.abs (Nx.item [] q -. fst (reference 2.))
        in
        let e4 = err 1e-4 and e7 = err 1e-7 and e10 = err 1e-10 in
        less float_exact ~than:e4 e7;
        less float_exact ~than:e7 e10);
    test "decreasing times solve backward" (fun () ->
        let s =
          solve1 (decay (scalar 1.)) ~t0:1. ~t1:0. (scalar (Float.exp (-1.)))
        in
        let bound =
          propagated ~lipschitz:1. ~span:1. (Nx.item [] (Solution.error s))
        in
        at_most float_exact ~than:bound
          (Float.abs (Nx.item [] (Solution.get s) -. 1.)));
    test "at t1 = t0 it is y0, converged with a zero error" (fun () ->
        let s = solve1 (decay (scalar 1.)) ~t0:0.7 ~t1:0.7 (scalar 2.) in
        equal (Oracle.tensor ()) (scalar 2.) (Solution.get s);
        equal (Oracle.tensor ()) (scalar 0.) (Solution.error s));
    test "the last step's stages at its end read the field at t1 itself"
      (fun () ->
        (* The field is 1 at t1 alone: tsit5's stages at c = 1 read it, so the
           last step's error estimate is not zero. t0 + (t1 − t0) is one ulp
           below t1. *)
        let t0 = -7.074863082399303 and t1 = 4.437342817489506 in
        let s =
          Ode.solve one Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:100
            (fun t y ->
              Nx.add (Nx.zeros_like y) (Nx.cast f64 (Nx.equal_s t t1)))
            ~t0:(scalar t0) ~t1:(scalar t1) (scalar 1.)
        in
        less float_exact ~than:(Nx.item [] (Solution.error s)) 0.;
        (* The answer retakes the steps: y leaves 1 only through its stages at c
           = 1, tsit5's sixth weighing 2.3. *)
        less float_exact
          ~than:(Float.abs (Nx.item [] (Solution.get s) -. 1.))
          0.);
    test "an attempt with a non-finite stage is rejected" (fun () ->
        (* y' = −y is NaN where y < 0. At a loose tol the steps grow until a
           stage overshoots below zero, and those attempts are rejected: the
           solve converges on e^(−t), at more evaluations than the plain
           field's. y' = −y contracts, so the global error is at most the
           estimates' sum (Hairer, Nørsett and Wanner, I, §I.10). *)
        let solve f =
          Ode.solve one Ode.tsit5
            ~tol:(Tol.v ~rel:1e-2 ~abs:1e-12)
            ~budget:400 f ~t0:(scalar 0.) ~t1:(scalar 20.) (scalar 1.)
        in
        let s =
          solve (fun _ y ->
              Nx.where (Nx.less_s y 0.) (Nx.full_like y Float.nan) (Nx.neg y))
        in
        at_most float_exact
          ~than:(Nx.item [] (Solution.error s))
          (Float.abs (Nx.item [] (Solution.get s) -. Float.exp (-20.)));
        greater int32
          ~than:(Nx.item [] (Solution.evaluations (solve (decay (scalar 1.)))))
          (Nx.item [] (Solution.evaluations s)));
    test "a field that stays non-finite ends the lane Not_finite" (fun () ->
        (* Every attempt past t = 0.5 has a NaN stage and is rejected, until the
           step falls below the time's resolution. *)
        let f t y =
          Nx.where (Nx.greater_s t 0.5) (Nx.full_like y Float.nan) (Nx.neg y)
        in
        equal Oracle.status_t Not_finite
          (Oracle.status (solve1 f ~t0:0. ~t1:1. (scalar 1.))));
    cases "a field non-finite past t0 ends the lane Not_finite wherever t0 is"
      ~id:(Printf.sprintf "t0 = %g") [ 0.; 1. ] (fun t0 ->
        (* Near t = 0 the floats' spacing is subnormal; the steps stop at the
           resolution of the interval's larger end all the same. *)
        let f t y =
          Nx.where (Nx.greater_s t t0) (Nx.full_like y Float.nan) (Nx.neg y)
        in
        let s =
          Ode.solve one Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:100 f
            ~t0:(scalar t0)
            ~t1:(scalar (t0 +. 1.))
            (scalar 1.)
        in
        equal Oracle.status_t Not_finite (Oracle.status s));
    test "a blow-up ends its lane Stalled" (fun () ->
        (* y' = y² from 1 is 1 / (1 − t): the steps shrink toward t = 1 until
           they fall below the time's resolution, the values still finite. *)
        let s =
          Ode.solve one Ode.tsit5 ~tol:(Tol.rel 1e-3) ~budget:400
            (fun _ y -> Nx.square y)
            ~t0:(scalar 0.) ~t1:(scalar 2.) (scalar 1.)
        in
        equal Oracle.status_t Stalled (Oracle.status s));
    test "budget attempted steps end the lane Budget_spent" (fun () ->
        let s =
          Ode.solve one Ode.tsit5 ~tol:(Tol.rel 1e-10) ~budget:3 forced2
            ~t0:(scalar 0.) ~t1:(scalar 10.) (scalar 1.)
        in
        equal Oracle.status_t Budget_spent (Oracle.status s);
        (* The report holds the entry point, the settings and the budget used,
           as Solution.get states. *)
        let report =
          match Solution.get s with
          | _ -> failf "the solve converged"
          | exception Failure m -> m
        in
        List.iter
          (fun sub -> contains ~sub report)
          [ "Jera.Ode.solve"; "tsit5"; "budget 3"; "3 of 3" ]);
    cases
      "evaluations count the first step's two, each attempt's and the answer's"
      ~id:(fun (n, _, _) -> n)
      [
        ("bs3", embedded Ode.bs3, 3);
        ("tsit5", embedded Ode.tsit5, 6);
        ("dopri5", embedded Ode.dopri5, 6);
      ]
      (fun (_, m, per_step) ->
        (* y' = 1 is exact for every method, so each attempt is accepted and the
           three spend the budget short of t1. *)
        let s =
          Ode.solve one m
            ~tol:(Tol.v ~rel:1e-6 ~abs:1e-9)
            ~budget:3
            (fun _ y -> Nx.ones_like y)
            ~t0:(scalar 0.) ~t1:(scalar 10.) (scalar 0.)
        in
        equal Oracle.status_t Budget_spent (Oracle.status s);
        equal int32
          (Int32.of_int (2 + (3 * per_step) + 1 + (3 * per_step)))
          (Nx.item [] (Solution.evaluations s)));
    test "a leaf that is not a float is carried unchanged" (fun () ->
        let s = Nx.Ptree.(pair tensor tensor) in
        let _, n =
          Solution.get
            (Ode.solve s Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:100
               (fun _ (y, n) -> (Nx.neg y, n))
               ~t0:(scalar 0.) ~t1:(scalar 1.)
               (scalar 1., Nx.scalar Nx.int32 5l))
        in
        equal (Oracle.tensor ()) (Nx.scalar Nx.int32 5l) n);
    test "times and leaves may have different dtypes" (fun () ->
        let s =
          Ode.solve one Ode.tsit5 ~tol:(Tol.rel 1e-5) ~budget:200
            (fun _ y -> Nx.neg y)
            ~t0:(scalar 0.) ~t1:(scalar 1.) (Nx.scalar Nx.float32 1.)
        in
        (* Some ten steps of float32 stages round about 2⁶ times, 4·10⁻⁶, beside
           an error near tol. *)
        equal
          (Oracle.tensor ~rel:2e-5 ())
          (Nx.scalar Nx.float32 (Float.exp (-1.)))
          (Solution.get s));
    test "raises if t0 or t1 is not a scalar" (fun () ->
        raises_in "Jera.Ode.solve" (fun () ->
            Ode.solve one Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:10
              (decay (scalar 1.))
              ~t0:(vec [| 0. |]) ~t1:(scalar 1.) (scalar 1.)));
    test "raises if budget < 1" (fun () ->
        raises_in "Jera.Ode.solve" (fun () ->
            Ode.solve one Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:0
              (decay (scalar 1.))
              ~t0:(scalar 0.) ~t1:(scalar 1.) (scalar 1.)));
    test "raises as march does for a field of another shape" (fun () ->
        raises_in "Jera.Ode.solve" (fun () ->
            Ode.solve one Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:10
              (fun _ y -> Nx.stack [ y; y ])
              ~t0:(scalar 0.) ~t1:(scalar 1.) (scalar 1.)));
    test "grad in the initial state is e^(−kT)" (fun () ->
        let g =
          Rune.grad'
            (fun y0 ->
              Solution.get (solve1 (decay (scalar 0.8)) ~t0:0. ~t1:1.5 y0))
            (scalar 2.)
        in
        (* The derivative is the accepted steps', whose error is the solve's,
           near 10⁻¹⁰. *)
        equal (Oracle.tensor ~rel:1e-8 ()) (scalar (Float.exp (-1.2))) g);
    test "grad in a captured rate is −T y0 e^(−kT)" (fun () ->
        let g =
          Rune.grad'
            (fun k ->
              Solution.get (solve1 (decay k) ~t0:0. ~t1:1.5 (scalar 2.)))
            (scalar 0.8)
        in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (scalar (-1.5 *. 2. *. Float.exp (-1.2)))
          g);
    test "grad in the end time is the field there" (fun () ->
        let f t1 =
          Solution.get
            (Ode.solve one Ode.tsit5
               ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
               ~budget:1000
               (decay (scalar 0.8))
               ~t0:(scalar 0.) ~t1 (scalar 2.))
        in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (scalar (-0.8 *. 2. *. Float.exp (-1.2)))
          (Rune.grad' f (scalar 1.5)));
    test "jvp agrees with grad" (fun () ->
        let f k = Solution.get (solve1 (decay k) ~t0:0. ~t1:1.5 (scalar 2.)) in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Rune.grad' f (scalar 0.8))
          (snd (Rune.jvp' f (scalar 0.8) (scalar 1.))));
    test "compiled grad is −T y0 e^(−kT)" (fun () ->
        Compiled.here ();
        (* Compiled, the controller's powers may round apart from eager's and
           take other steps; either way the derivative is the accepted steps',
           within their error, as eagerly. *)
        let f k = Solution.get (solve1 (decay k) ~t0:0. ~t1:1.5 (scalar 2.)) in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (scalar (-1.5 *. 2. *. Float.exp (-1.2)))
          (Rune.jit' (Rune.grad' f) (scalar 0.8)));
    test "compiled, a polynomial field's answer is within its error" (fun () ->
        Compiled.here ();
        (* A Duffing oscillator, whose Jacobian has maximum norm at most 1 + 0.9
           q² ≤ 1.73 for |q| ≤ 0.9. *)
        let duffing _ (q, p) =
          (p, Nx.sub (Nx.neg q) (Nx.mul_s (Nx.mul q (Nx.square q)) 0.3))
        in
        let f y0 =
          Ode.solve pair Ode.tsit5
            ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
            ~budget:1000 duffing ~t0:(scalar 0.) ~t1:(scalar 2.)
            (y0, Nx.zeros_like y0)
        in
        let s =
          Rune.jit
            Nx.Ptree.(tensor @-> returns (Solution.ptree (pair tensor tensor)))
            f (scalar 0.9)
        in
        let (q, p), (eq, ep) = (Solution.get s, Solution.error s) in
        let rq, rp =
          integrate
            (fun _ (q, p) -> (p, -.q -. (0.3 *. q *. q *. q)))
            (0.9, 0.) 2.
        in
        let bound =
          propagated ~lipschitz:1.73 ~span:2. (Nx.item [] eq +. Nx.item [] ep)
        in
        at_most float_exact ~than:bound (Float.abs (Nx.item [] q -. rq));
        at_most float_exact ~than:bound (Float.abs (Nx.item [] p -. rp)));
    test "a compiled solve of a transcendental field is within its error"
      (fun () ->
        Compiled.here ();
        (* Compiled tanh is within ulps of eager's, which changes the steps:
           from 0.5 on the host, eager takes 230 evaluations and compiled 236. L
           = 1 bounds the field's Jacobian in the maximum norm, as sech² q ≤ 1.
           Rounding, a few ulps in each of about 40 steps, adds under 10⁻¹², a
           ten-thousandth of the bound. *)
        let soft _ (q, p) = (p, Nx.neg (Nx.tanh q)) in
        let f y0 =
          Ode.solve pair Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:1000 soft
            ~t0:(scalar 0.) ~t1:(scalar 2.)
            (y0, Nx.zeros_like y0)
        in
        let s =
          Rune.jit
            Nx.Ptree.(tensor @-> returns (Solution.ptree (pair tensor tensor)))
            f (scalar 0.5)
        in
        let (q, p), (eq, ep) = (Solution.get s, Solution.error s) in
        let rq, rp =
          integrate (fun _ (q, p) -> (p, -.Float.tanh q)) (0.5, 0.) 2.
        in
        let bound =
          propagated ~lipschitz:1. ~span:2. (Nx.item [] eq +. Nx.item [] ep)
        in
        at_most float_exact ~than:bound (Float.abs (Nx.item [] q -. rq));
        at_most float_exact ~than:bound (Float.abs (Nx.item [] p -. rp)));
    test "a lane that did not converge has a zero derivative" (fun () ->
        (* y' = y² from 0.2 is fine to t = 2; from 1 it blows up at t = 1 and
           spends the budget. *)
        let f y0 =
          Solution.best
            (Ode.solve one Ode.tsit5
               ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
               ~budget:300
               (fun _ y -> Nx.square y)
               ~t0:(scalar 0.) ~t1:(scalar 2.) y0)
        in
        let g =
          Rune.grad' (fun y0 -> Nx.sum (Rune.vmap' f y0)) (vec [| 0.2; 1. |])
        in
        equal
          (Oracle.tensor ~rel:1e-6 ())
          (vec [| 1. /. ((1. -. 0.4) ** 2.); 0. |])
          g);
    test "vmap gives each lane its own steps" (fun () ->
        let f k = Solution.get (solve1 (decay k) ~t0:0. ~t1:1.5 (scalar 2.)) in
        let ks = vec [| 0.1; 5.; 0.8 |] in
        equal (Oracle.tensor ())
          (Nx.stack (List.init 3 (fun i -> f (Nx.get [ i ] ks))))
          (Rune.vmap' f ks));
  ]

(* Samples *)

let sample1 f ~at y0 =
  Ode.sample one Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:100 f ~at y0

let sample =
  [
    test "lands on each time" (fun () ->
        (* y' = max(t − 1, 0) has a kink at t = 1, about which each step's
           polynomial is exact: y = 0 to 1 and (t − 1)² / 2 after. A step across
           the kink would err by its size squared. *)
        let kink t y =
          Nx.add (Nx.zeros_like y) (Nx.maximum (Nx.sub_s t 1.) (scalar 0.))
        in
        let y =
          Solution.get (sample1 kink ~at:(vec [| 0.; 1.; 2.5 |]) (scalar 0.))
        in
        equal (Oracle.tensor ~abs:1e-14 ()) (vec [| 0.; 0.; 1.125 |]) y);
    test "one time is the start alone, converged" (fun () ->
        let s = sample1 (decay (scalar 1.)) ~at:(vec [| 0.5 |]) (scalar 2.) in
        equal (Oracle.tensor ()) (vec [| 2. |]) (Solution.get s));
    test "its error is zero at the first time and never decreases" (fun () ->
        let s =
          sample1 forced2 ~at:(vec [| 0.; 0.5; 1.; 2.; 3. |]) (scalar 1.)
        in
        let e = Oracle.floats (Solution.error s) in
        equal float_exact 0. e.(0);
        satisfies ~claim:"non-decreasing" (array float_exact)
          (fun e -> Array.for_all2 ( <= ) (Array.sub e 0 4) (Array.sub e 1 4))
          e);
    test "a long interval after many short ones converges" (fun () ->
        (* Short intervals grow the step without bound unless the span caps it;
           the long one then needs rejections to shrink it. *)
        let at =
          Nx.concatenate ~axis:0 [ Nx.linspace f64 0. 1. 400; vec [| 30. |] ]
        in
        let s =
          Ode.sample one Ode.tsit5
            ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
            ~budget:2000
            (fun t y -> Nx.add (Nx.mul_s y (-0.1)) (Nx.sin t))
            ~at (scalar 1.)
        in
        equal Oracle.status_t Converged (Oracle.status s));
    test "times out of order end the lane Stalled, the report naming the first"
      (fun () ->
        let s =
          sample1 (decay (scalar 1.)) ~at:(vec [| 0.; 1.; 0.5 |]) (scalar 1.)
        in
        equal Oracle.status_t Stalled (Oracle.status s);
        raises_match (Exn.failure ~sub:"[2]") (fun () -> Solution.get s));
    test "times out of order in one lane leave the others converged" (fun () ->
        let at =
          Nx.create f64 [| 3; 3 |] [| 0.; 1.; 2.; 0.; 1.; 0.5; 0.; 0.5; 1. |]
        in
        let ok =
          Rune.vmap
            Nx.Ptree.(tensor @-> returns tensor)
            (fun at ->
              Solution.ok (sample1 (decay (scalar 1.)) ~at (scalar 1.)))
            at
        in
        equal (Oracle.tensor ())
          (Nx.create Nx.bool [| 3 |] [| true; false; true |])
          ok);
    test "raises if at is not a non-empty 1-D tensor" (fun () ->
        raises_in "Jera.Ode.sample" (fun () ->
            sample1 (decay (scalar 1.)) ~at:(Nx.zeros f64 [| 0 |]) (scalar 1.)));
  ]

(* Paths *)

let path1 ?(tol = Tol.v ~rel:1e-10 ~abs:1e-12) f ~t0 ~t1 y0 =
  Ode.path one Ode.tsit5 ~tol ~budget:200 f ~t0:(scalar t0) ~t1:(scalar t1) y0

let between = Nx.linspace f64 0. 2. 101

let exponential k t0 y0 t =
  Nx.mul_s (Nx.exp (Nx.mul_s (Nx.sub_s t t0) (-.k))) y0

let path =
  [
    test "at a step's end it is the step's state" (fun () ->
        let p = Solution.get (path1 forced2 ~t0:0. ~t1:2. (scalar 1.)) in
        let y = Solution.get (solve1 forced2 ~t0:0. ~t1:2. (scalar 1.)) in
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (vec [| 1.; Nx.item [] y |])
          (Piecewise.eval p (vec [| 0.; 2. |])));
    cases "its domain runs to t1 itself, either way"
      ~id:(fun (t0, t1) -> Printf.sprintf "%h to %h" t0 t1)
      [
        (-7.074863082399303, 4.437342817489506);
        (4.437342817489506, -7.074863082399303);
      ]
      (fun (t0, t1) ->
        (* t0 + (t1 − t0) is one ulp off t1 here. *)
        let f = decay (scalar 0.1) in
        let p = Solution.get (path1 f ~t0 ~t1 (scalar 1.)) in
        let y = Solution.get (solve1 f ~t0 ~t1 (scalar 1.)) in
        equal (Oracle.tensor ~rel:1e-14 ()) y (Piecewise.eval p (scalar t1)));
    cases "between steps it reproduces a solution of its extension's order"
      ~id:(fun (n, _, _) -> n)
      [
        ("bs3", embedded Ode.bs3, 3);
        ("tsit5", embedded Ode.tsit5, 4);
        ("dopri5", embedded Ode.dopri5, 4);
        ("kvaerno5", kvaerno (power 3), 3);
      ]
      (fun (_, m, q) ->
        (* An extension of order q meets the quadrature conditions Σ b_i(θ)
           c_i^(k−1) = θ^k / k for k ≤ q, so on y' = q t^(q−1) it is t^q at
           every θ. *)
        let p =
          Solution.get
            (Ode.path one m
               ~tol:(Tol.v ~rel:1e-4 ~abs:1e-8)
               ~budget:400 (power q) ~t0:(scalar 0.) ~t1:(scalar 2.) (scalar 0.))
        in
        (* Exact up to the rounding of some hundred steps' sums, of values up to
           2^q. *)
        equal
          (Oracle.tensor ~rel:1e-13 ~abs:1e-14 ())
          (Nx.pow_s between (Float.of_int q))
          (Piecewise.eval p between));
    cases "between steps a path of y' = −3y is within its extension's error"
      ~id:(fun (n, _, _, _) -> n)
      [
        ("bs3", embedded Ode.bs3, 1e-8, 1e-6);
        ("tsit5", embedded Ode.tsit5, 1e-10, 1e-8);
        ("dopri5", embedded Ode.dopri5, 1e-10, 1e-8);
      ]
      (fun (_, m, tol, within) ->
        (* On y' = λy a step has |λh| ≈ (tol / C)^(1/(r+1)) for an estimate of
           order r, and an extension of order q errs by about |λh|^(q+1): for
           tsit5 and dopri5 (r = q = 4) that is tol times a constant of the
           method, for bs3 (r = 2, q = 3) below tol^(4/3). The windows leave a
           factor of 100 for the constants. *)
        let p =
          Solution.get
            (Ode.path one m ~tol:(Tol.v ~rel:tol ~abs:tol) ~budget:2000
               (decay (scalar 3.))
               ~t0:(scalar 0.) ~t1:(scalar 2.) (scalar 1.))
        in
        equal
          (Oracle.tensor ~abs:within ())
          (exponential 3. 0. 1. between)
          (Piecewise.eval p between));
    test "backward, its pieces are in increasing time" (fun () ->
        let y2 = Float.exp (-2.) in
        let p =
          Solution.get (path1 (decay (scalar 1.)) ~t0:2. ~t1:0. (scalar y2))
        in
        let breaks = Oracle.floats (Piecewise.breaks p) in
        let n = Array.length breaks in
        satisfies ~claim:"non-decreasing" (array float_exact)
          (fun b ->
            Array.for_all2 ( <= )
              (Array.sub b 0 (n - 1))
              (Array.sub b 1 (n - 1)))
          breaks;
        (* Backward, y' = −y grows the steps' errors by e² over the span, about
           10⁻⁹ at tol 10⁻¹⁰, and the extension adds about tol between steps. *)
        equal
          (Oracle.tensor ~abs:1e-9 ())
          (exponential 1. 2. y2 between)
          (Piecewise.eval p between));
    test "past the last step the pieces are empty at its end" (fun () ->
        let p =
          Solution.get (path1 (decay (scalar 1.)) ~t0:0. ~t1:2. (scalar 1.))
        in
        let breaks = Piecewise.breaks p in
        let n = Nx.dim 0 breaks in
        equal (Oracle.tensor ()) (scalar 2.) (Nx.get [ n - 1 ] breaks);
        equal (Oracle.tensor ()) (scalar 2.) (Nx.get [ n - 2 ] breaks));
    test "it is a series of budget pieces" (fun () ->
        let p =
          Solution.get
            (Ode.path one Ode.tsit5 ~tol:(Tol.rel 1e-6) ~budget:17
               (decay (scalar 1.))
               ~t0:(scalar 0.) ~t1:(scalar 2.) (scalar 1.))
        in
        equal int 18 (Nx.dim 0 (Piecewise.breaks p)));
    test "over no time it ends its lane Stalled and says why" (fun () ->
        let s = path1 (decay (scalar 1.)) ~t0:1. ~t1:1. (scalar 1.) in
        equal Oracle.status_t Stalled (Oracle.status s);
        raises_match (Exn.failure ~sub:"Jera.Ode.path") (fun () ->
            Solution.get s));
    test "its error is the accumulated estimate, as sample's" (fun () ->
        let tol = Tol.v ~rel:1e-6 ~abs:1e-8 in
        let p = path1 ~tol (decay (scalar 1.)) ~t0:0. ~t1:2. (scalar 1.) in
        let s =
          Ode.sample one Ode.tsit5 ~tol ~budget:200
            (decay (scalar 1.))
            ~at:(vec [| 0.; 2. |])
            (scalar 1.)
        in
        (* The answer retakes the search's steps to the bit, so its estimates
           are the search's, summed in the same order. *)
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (Nx.get [ 1 ] (Solution.error s))
          (Piecewise.eval (Solution.error p) (scalar 2.)));
    test "a path of kvaerno5 follows a stiff field's slow solution" (fun () ->
        let f = prothero (scalar (-1e3)) in
        let p =
          Solution.get
            (Ode.path one (kvaerno f) ~tol:stiff_tol ~budget:300 f
               ~t0:(scalar 0.) ~t1:(scalar 2.) (vec [| 0. |]))
        in
        let ts = vec [| 0.3; 1.1; 1.9 |] in
        (* Between steps the cubic Hermite extension errs by at most h⁴ max |y⁗|
           / 384, under 3·10⁻⁷ for the steps below 0.1 that tol 10⁻⁸ sets for a
           fifth-order method on sin t. *)
        equal
          (Oracle.tensor ~abs:1e-6 ())
          (Nx.sin ts)
          (Nx.reshape [| 3 |] (Piecewise.eval p ts)));
    test "a path of a DAE passes through its algebraic component" (fun () ->
        let p =
          Solution.get
            (Ode.path one
               (kvaerno ~mass:dae_mass dae)
               ~tol:stiff_tol ~budget:300 dae ~t0:(scalar 0.) ~t1:(scalar 2.)
               (vec [| 0.5; 1. |]))
        in
        let ts = vec [| 0.3; 1.1; 1.9 |] in
        (* Each stage solves 0 = y₂ − cos t, so the extension interpolates cos
           at five fractions of a step and errs by at most h⁵ / 5!, under 10⁻⁷
           for steps below 0.1. *)
        equal
          (Oracle.tensor ~abs:1e-6 ())
          (Nx.cos ts)
          (Nx.slice [ Nx.A; Nx.I 1 ] (Piecewise.eval p ts)));
    test "grad in the initial state is e^(−kt)" (fun () ->
        let g =
          Rune.grad'
            (fun y0 ->
              Piecewise.eval
                (Solution.get (path1 (decay (scalar 0.8)) ~t0:0. ~t1:2. y0))
                (scalar 1.3))
            (scalar 2.)
        in
        equal (Oracle.tensor ~rel:1e-8 ()) (scalar (Float.exp (-0.8 *. 1.3))) g);
    test "grad in a captured rate is −t y0 e^(−kt)" (fun () ->
        let g =
          Rune.grad'
            (fun k ->
              Piecewise.eval
                (Solution.get (path1 (decay k) ~t0:0. ~t1:2. (scalar 2.)))
                (scalar 1.3))
            (scalar 0.8)
        in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (scalar (-1.3 *. 2. *. Float.exp (-0.8 *. 1.3)))
          g);
    test "grad in the end time moves the breaks" (fun () ->
        (* y(t) = e^(−t) at a fixed time does not depend on t1. *)
        let g =
          Rune.grad'
            (fun t1 ->
              Piecewise.eval
                (Solution.get
                   (Ode.path one Ode.tsit5
                      ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
                      ~budget:200
                      (decay (scalar 1.))
                      ~t0:(scalar 0.) ~t1 (scalar 1.)))
                (scalar 0.7))
            (scalar 2.)
        in
        equal (Oracle.tensor ~abs:1e-8 ()) (scalar 0.) g);
    test "compiled, its values are within its error" (fun () ->
        Compiled.here ();
        (* y' = −0.8 y contracts, so its global error is at most the estimates'
           sum, near 10⁻¹⁰, and between steps the extension's error, below 10⁻⁸
           at this tol (see above), adds to it. *)
        let p =
          Rune.jit
            Nx.Ptree.(
              tensor @-> returns (Solution.ptree (Piecewise.ptree tensor f64)))
            (fun k -> path1 (decay k) ~t0:0. ~t1:2. (scalar 1.))
            (scalar 0.8)
        in
        equal
          (Oracle.tensor ~abs:1e-8 ())
          (exponential 0.8 0. 1. between)
          (Piecewise.eval (Solution.get p) between));
    test "raises at once if a leaf of the state is not a float tensor"
      (fun () ->
        let calls = ref 0 in
        let field _ (y, n) =
          incr calls;
          (Nx.neg y, n)
        in
        raises_in "Jera.Ode.path" (fun () ->
            Ode.path
              Nx.Ptree.(pair tensor tensor)
              Ode.tsit5 ~tol:(Tol.rel 1e-6) ~budget:20 field ~t0:(scalar 0.)
              ~t1:(scalar 1.)
              (scalar 1., Nx.scalar Nx.int32 5l));
        equal int 0 !calls);
  ]

(* Events *)

let gravity = 9.81
let fall _ (_, p) = (p, Nx.full_like p (-.gravity))

(* The first time a ball dropped from rest at [q0] reaches [level]. *)
let reaches q0 level = Float.sqrt (2. *. (q0 -. level) /. gravity)

(* A ball dropped at [q0] at t = 0, toward [t1], watched by [event]. *)
let drop ~t1 event q0 =
  Ode.event pair Ode.tsit5
    ~tol:(Tol.v ~rel:1e-12 ~abs:1e-12)
    ~budget:200 fall ~event ~t0:(scalar 0.) ~t1:(scalar t1)
    (q0, Nx.zeros_like q0)

let height _ (q, _) = q

(* An event of sign −1 before t = 1, 0 on [1, 2], where the steps end on its
   zeros, and [after] past 2. *)
let plateau ~after =
  let event t _ =
    Nx.add
      (Nx.minimum (Nx.sub_s t 1.) (scalar 0.))
      (Nx.mul_s (Nx.maximum (Nx.sub_s t 2.) (scalar 0.)) after)
    |> Nx.sign
  in
  Ode.event one Ode.tsit5
    ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
    ~budget:200
    (decay (scalar 1.))
    ~event ~t0:(scalar 0.) ~t1:(scalar 5.) (scalar 1.)

let index i = Nx.scalar Nx.int32 i

let event =
  (* A falling ball's height is quadratic in t, which every step integrates
     exactly, so its crossing is found to the time's resolution. *)
  let close = Oracle.tensor ~rel:1e-13 () in
  [
    test "a ball reaches the ground at sqrt(2 h / g)" (fun () ->
        let t, _, i = Solution.get (drop ~t1:5. height (scalar 10.)) in
        equal close (scalar (reaches 10. 0.)) t;
        equal (Oracle.tensor ()) (index 0l) i);
    test "the crossing's time does not loosen at a large epoch" (fun () ->
        (* At t0 = 10^6 a tolerance relative to the time would accept a bracket
           of width 1; the crossing's error is the state's. *)
        let t, _, _ =
          Solution.get
            (Ode.event pair Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:200 fall
               ~event:height ~t0:(scalar 1e6)
               ~t1:(scalar (1e6 +. 5.))
               (scalar 10., scalar 0.))
        in
        equal (Oracle.tensor ~abs:1e-6 ()) (scalar (1e6 +. reaches 10. 0.)) t);
    test "the time's error is about one unit in the last place" (fun () ->
        let s = drop ~t1:5. height (scalar 10.) in
        let t, _, _ = Solution.get s and e, _, i = Solution.error s in
        let t = Nx.item [] t in
        at_most float_exact ~than:(2. *. (Float.succ t -. t)) (Nx.item [] e);
        equal (Oracle.tensor ()) (index 0l) i);
    test "the returned state has the component's new sign" (fun () ->
        let _, (q, _), _ = Solution.get (drop ~t1:5. height (scalar 10.)) in
        less float_exact ~than:0. (Nx.item [] q));
    test "the earliest of several events wins, with its index" (fun () ->
        let t, _, i =
          Solution.get
            (drop ~t1:5.
               (fun _ (q, _) -> Nx.stack [ Nx.sub_s q 5.; Nx.sub_s q 8. ])
               (scalar 10.))
        in
        equal close (scalar (reaches 10. 8.)) t;
        equal (Oracle.tensor ()) (index 1l) i);
    test "the index is flat over the event's shape" (fun () ->
        (* Of a 2 × 2 event, component (1, 0) crosses first. *)
        let levels = Nx.create f64 [| 2; 2 |] [| 2.; 3.; 9.; 4. |] in
        let _, _, i =
          Solution.get
            (drop ~t1:5. (fun _ (q, _) -> Nx.sub q levels) (scalar 10.))
        in
        equal (Oracle.tensor ()) (index 2l) i);
    test "without a crossing the solve ends at t1 with index -1" (fun () ->
        let t, (q, _), i = Solution.get (drop ~t1:1. height (scalar 10.)) in
        equal (Oracle.tensor ()) (scalar 1.) t;
        equal close (scalar (10. -. (gravity /. 2.))) q;
        equal (Oracle.tensor ()) (index (-1l)) i);
    test "a backward solve without a crossing ends at t1 with index -1"
      (fun () ->
        let t, _, i =
          Solution.get
            (Ode.event one Ode.tsit5
               ~tol:(Tol.v ~rel:1e-11 ~abs:1e-13)
               ~budget:200
               (decay (scalar 1.))
               ~event:(fun _ y -> Nx.sub_s y 2.)
               ~t0:(scalar 2.) ~t1:(scalar 0.)
               (scalar (Float.exp (-2.))))
        in
        equal (Oracle.tensor ()) (scalar 0.) t;
        equal (Oracle.tensor ()) (index (-1l)) i);
    test "a component through an exact zero at a step's end crosses" (fun () ->
        (* It takes its new sign at t = 2, past its zeros. *)
        let t, _, i = Solution.get (plateau ~after:1.) in
        equal (Oracle.tensor ()) (index 0l) i;
        equal (Oracle.tensor ~rel:1e-15 ()) (scalar 2.) t);
    test "a component that touches zero without changing sign does not cross"
      (fun () ->
        let t, _, i = Solution.get (plateau ~after:(-1.)) in
        equal (Oracle.tensor ()) (index (-1l)) i;
        equal (Oracle.tensor ()) (scalar 5.) t);
    test "at t1 = t0 the answer is (t0, y0, -1), converged" (fun () ->
        let t, (q, _), i = Solution.get (drop ~t1:0. height (scalar 10.)) in
        equal (Oracle.tensor ()) (scalar 0.) t;
        equal (Oracle.tensor ()) (scalar 10.) q;
        equal (Oracle.tensor ()) (index (-1l)) i);
    test "a ball bounces by restarts with the crossing component at zero"
      (fun () ->
        (* Each bounce keeps 0.9 of the speed, so the flights after the first
           fall last 2 (0.9^k) v / g = 2 (0.9^k) t*, v = g t* the speed at the
           first impact. *)
        let rec bounces t (q, p) n acc =
          if n = 0 then List.rev acc
          else
            let t', (_, p'), _ =
              Solution.get
                (Ode.event pair Ode.tsit5
                   ~tol:(Tol.v ~rel:1e-12 ~abs:1e-12)
                   ~budget:200 fall ~event:height ~t0:t ~t1:(scalar 20.) (q, p))
            in
            bounces t'
              (Nx.zeros_like q, Nx.mul_s p' (-0.9))
              (n - 1) (Nx.item [] t' :: acc)
        in
        (* Every flight is a quadratic the steps integrate exactly, so the times
           are exact to the rounding of a few crossings. *)
        let first = reaches 10. 0. in
        let flight k = 2. *. Float.pow 0.9 (Float.of_int k) *. first in
        let expected =
          [ first; first +. flight 1; first +. flight 1 +. flight 2 ]
        in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (vec (Array.of_list expected))
          (vec
             (Array.of_list (bounces (scalar 0.) (scalar 10., scalar 0.) 3 []))));
    test "a zero at t0 is not a crossing" (fun () ->
        let _, _, i =
          Solution.get
            (drop ~t1:1. (fun _ (q, _) -> Nx.sub_s q 10.) (scalar 10.))
        in
        equal (Oracle.tensor ()) (index (-1l)) i);
    test "a backward solve finds the crossing behind it" (fun () ->
        (* y = e^(−t) solved from t = 2 back to 0 crosses 1/2 at ln 2. *)
        let t, _, _ =
          Solution.get
            (Ode.event one Ode.tsit5
               ~tol:(Tol.v ~rel:1e-11 ~abs:1e-13)
               ~budget:200
               (decay (scalar 1.))
               ~event:(fun _ y -> Nx.sub_s y 0.5)
               ~t0:(scalar 2.) ~t1:(scalar 0.)
               (scalar (Float.exp (-2.))))
        in
        equal (Oracle.tensor ~rel:1e-8 ()) (scalar (Float.log 2.)) t);
    test "an event on a DAE's algebraic component: y₂ = cos t crosses at π/2"
      (fun () ->
        let t, v, i =
          Solution.get
            (Ode.event one
               (kvaerno ~mass:dae_mass dae)
               ~tol:stiff_tol ~budget:500 dae
               ~event:(fun _ v -> Nx.get [ 1 ] v)
               ~t0:(scalar 0.) ~t1:(scalar 3.)
               (vec [| 0.5; 1. |]))
        in
        equal int32 0l (Nx.item [] i);
        (* The crossing and the state there are the extension's, whose error
           [tol] does not control: the stage values are of stage order 2, and
           the differential component misses by some 10⁻⁵ at these steps. *)
        equal (Oracle.tensor ~abs:1e-6 ()) (scalar (Float.pi /. 2.)) t;
        equal
          (Oracle.tensor ~abs:1e-4 ())
          (scalar
             ((Float.cos (Float.pi /. 2.) +. Float.sin (Float.pi /. 2.)) /. 2.))
          (Nx.get [ 0 ] v));
    test "grad of the time in the height is 1 / (g t)" (fun () ->
        let g =
          Rune.grad'
            (fun q0 ->
              let t, _, _ = Solution.get (drop ~t1:5. height q0) in
              t)
            (scalar 10.)
        in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (scalar (1. /. (gravity *. reaches 10. 0.)))
          g);
    test "grad of the time in a level the event reads is −1 / (g t)" (fun () ->
        (* q0 − g t² / 2 = level at the crossing. *)
        let g =
          Rune.grad'
            (fun level ->
              let t, _, _ =
                Solution.get
                  (drop ~t1:5. (fun _ (q, _) -> Nx.sub q level) (scalar 10.))
              in
              t)
            (scalar 3.)
        in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (scalar (-1. /. (gravity *. reaches 10. 3.)))
          g);
    test "grad of the time in t0 is 1 for a field that ignores time" (fun () ->
        let g =
          Rune.grad'
            (fun t0 ->
              let t, _, _ =
                Solution.get
                  (Ode.event pair Ode.tsit5
                     ~tol:(Tol.v ~rel:1e-12 ~abs:1e-12)
                     ~budget:200 fall ~event:height ~t0 ~t1:(Nx.add_s t0 5.)
                     (scalar 10., scalar 0.))
              in
              t)
            (scalar 0.5)
        in
        equal (Oracle.tensor ~rel:1e-8 ()) (scalar 1.) g);
    test "grad of the state at the crossing is the field there" (fun () ->
        (* At the ground the velocity is −g t*, so d v / d q0 is −1 / t*. *)
        let g =
          Rune.grad'
            (fun q0 ->
              let _, (_, p), _ = Solution.get (drop ~t1:5. height q0) in
              p)
            (scalar 10.)
        in
        equal (Oracle.tensor ~rel:1e-8 ()) (scalar (-1. /. reaches 10. 0.)) g);
    test "without a crossing the derivative is solve's" (fun () ->
        let by_event q0 =
          let _, (q, _), _ = Solution.get (drop ~t1:1. height q0) in
          q
        and by_solve q0 =
          fst
            (Solution.get
               (Ode.solve pair Ode.tsit5
                  ~tol:(Tol.v ~rel:1e-12 ~abs:1e-12)
                  ~budget:200 fall ~t0:(scalar 0.) ~t1:(scalar 1.)
                  (q0, Nx.zeros_like q0)))
        in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Rune.grad' by_solve (scalar 10.))
          (Rune.grad' by_event (scalar 10.)));
    test "lanes with and without a crossing are independent" (fun () ->
        let f q0 =
          let t, _, _ = Solution.get (drop ~t1:1.5 height q0) in
          t
        in
        equal close
          (vec [| reaches 5. 0.; 1.5 |])
          (Rune.vmap' f (vec [| 5.; 50. |])));
    test "compiled equals eager within the crossing's tolerance" (fun () ->
        Compiled.here ();
        (* The steps' sizes round differently compiled, so the crossing's search
           takes other points; both times are within the tolerance, rel 1e-12,
           of the crossing. *)
        let f q0 =
          let t, _, _ = Solution.get (drop ~t1:5. height q0) in
          t
        in
        equal
          (Oracle.tensor ~rel:5e-12 ())
          (f (scalar 10.))
          (Rune.jit' f (scalar 10.)));
    test "a crossing whose search fails ends the lane and names it" (fun () ->
        (* The second component is not finite near its crossing at t = 1. *)
        let event t (q, _) =
          let near = Nx.less_s (Nx.abs (Nx.sub_s t 1.)) 1e-3 in
          Nx.stack
            [ q; Nx.where near (Nx.full_like t Float.nan) (Nx.sub_s t 1.) ]
        in
        let s = drop ~t1:5. event (scalar 10.) in
        equal Oracle.status_t Stalled (Oracle.status s);
        raises_match (Exn.failure ~sub:"[1]") (fun () -> Solution.get s));
    test "a compiled function returns the answer as a triple" (fun () ->
        Compiled.here ();
        let answer =
          Solution.ptree
            (Nx.Ptree.v (fun c (t, y, i) ->
                 let open Nx.Ptree.Walk in
                 let t = index c 0 tensor t in
                 let y = index c 1 (structure pair) y in
                 let i = index c 2 tensor i in
                 (t, y, i)))
        in
        let t, _, i =
          Solution.get
            (Rune.jit
               Nx.Ptree.(tensor @-> returns answer)
               (drop ~t1:5. height) (scalar 10.))
        in
        equal (Oracle.tensor ~rel:5e-12 ()) (scalar (reaches 10. 0.)) t;
        equal (Oracle.tensor ()) (index 0l) i);
    test "an event of no component raises before any step" (fun () ->
        let calls = ref 0 in
        let counted t y =
          incr calls;
          fall t y
        in
        raises_in "Jera.Ode.event" (fun () ->
            Ode.event pair Ode.tsit5 ~tol:(Tol.rel 1e-6) ~budget:200 counted
              ~event:(fun _ _ -> Nx.zeros f64 [| 0 |])
              ~t0:(scalar 0.) ~t1:(scalar 5.)
              (scalar 10., scalar 0.));
        equal int 0 !calls);
  ]

(* Delays *)

(* y' = −a y(t − 1) from y = c before 0: on [0, 1] y = c (1 − a t), and with a =
   1, y(2) = −c / 2 and y(3) = −c / 6, by the method of steps. Between
   breakpoints, where the steps land, the solution is a polynomial of degree at
   most 3, so a method of order 3 or more is exact to rounding whatever tol. *)
let lagged _ _ d = Nx.neg (Nx.squeeze ~axes:[ 0 ] d)
let times = vec [| 0.; 1.; 2.; 3. |]
let exact = Oracle.tensor ~rel:1e-12 ~abs:1e-14 ()
let constant c _ = c

let delay1 ?(lags = vec [| 1. |]) ?(pieces = 16) f ~history ~at y0 =
  Ode.delay one Ode.tsit5 ~tol:(Tol.rel 1e-6) ~budget:400 ~pieces f ~lags
    ~history ~at y0

(* The [needed] count a delay's report names after "pieces to about". *)
let needed report =
  let marker = "pieces to about " in
  let n = String.length marker in
  let rec find i =
    if i + n > String.length report then failf "no count in: %s" report
    else if String.sub report i n = marker then
      Scanf.sscanf
        (String.sub report (i + n) (String.length report - i - n))
        "%d" Fun.id
    else find (i + 1)
  in
  find 0

let delay =
  [
    cases "y' = −y(t − 1) is exact to rounding for every method"
      ~id:(fun (n, _) -> n)
      [ ("bs3", Ode.bs3); ("tsit5", Ode.tsit5); ("dopri5", Ode.dopri5) ]
      (fun (_, m) ->
        let s =
          Ode.delay one m
            ~tol:(Tol.v ~rel:1e-3 ~abs:1e-6)
            ~budget:400 ~pieces:16 lagged ~lags:(vec [| 1. |])
            ~history:(constant (scalar 1.))
            ~at:times (scalar 1.)
        in
        equal exact (vec [| 1.; 0.; -0.5; -1. /. 6. |]) (Solution.get s));
    test "y' = e y(t − 1) from e^t keeps e^t between breakpoints" (fun () ->
        (* λ = e e^(−λ) holds at λ = 1, so e^t solves it for all t: its delayed
           states come from the steps' extensions. *)
        let e = Float.exp 1. in
        let s =
          Ode.delay one Ode.tsit5
            ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
            ~budget:400 ~pieces:64
            (fun _ _ d -> Nx.mul_s (Nx.squeeze ~axes:[ 0 ] d) e)
            ~lags:(vec [| 1. |]) ~history:Nx.exp
            ~at:(vec [| 0.; 0.5; 1.7; 3. |])
            (scalar 1.)
        in
        (* The delayed term is e-Lipschitz, so the error at t is at most e^(e t)
           times the estimates' sum before it. *)
        let ts = [| 0.; 0.5; 1.7; 3. |] in
        let y = Oracle.floats (Solution.get s)
        and e_t = Oracle.floats (Solution.error s) in
        Array.iteri
          (fun i t ->
            at_most float_exact
              ~than:(propagated ~lipschitz:e ~span:t e_t.(i))
              (Float.abs (y.(i) -. Float.exp t)))
          ts);
    test "two lags stack on a leading axis" (fun () ->
        let s =
          delay1
            ~lags:(vec [| 2.; 1. |])
            (fun _ _ d -> Nx.neg (Nx.get [ 1 ] d))
            ~history:(constant (scalar 1.))
            ~at:times (scalar 1.)
        in
        equal exact (vec [| 1.; 0.; -0.5; -1. /. 6. |]) (Solution.get s));
    test "y0 may differ from the history at the first time" (fun () ->
        (* y' = −y(t − 1) from y = 0 before 0 and y(0) = 1: y = 1 on [0, 1], 2 −
           t on [1, 2], and y(3) = −1/2. *)
        let s =
          delay1 lagged ~history:(constant (scalar 0.)) ~at:times (scalar 1.)
        in
        equal exact (vec [| 1.; 1.; 0.; -0.5 |]) (Solution.get s));
    test "history gives the state at one time, stacked per lag" (fun () ->
        (* y = (e^t, 2 e^t) solves y' = (e/2) y(t − 1) + (√e/2) y(t − 1/2): the
           history must hold each lag's own time. *)
        let e = Float.exp 1. in
        let scale = vec [| 1.; 2. |] in
        let solution t = Nx.mul (Nx.exp t) scale in
        let f _ _ d =
          Nx.add
            (Nx.mul_s (Nx.get [ 0 ] d) (e /. 2.))
            (Nx.mul_s (Nx.get [ 1 ] d) (Float.sqrt e /. 2.))
        in
        let at = vec [| 0.; 0.8; 2. |] in
        let s =
          Ode.delay one Ode.tsit5
            ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
            ~budget:400 ~pieces:64 f
            ~lags:(vec [| 1.; 0.5 |])
            ~history:solution ~at
            (solution (scalar 0.))
        in
        (* The delayed terms are (e + √e) / 2 ≈ 2.2-Lipschitz, so the error at t
           is at most e^(2.2 t) times the estimates' sum before it. *)
        let y = Solution.get s and errors = Solution.error s in
        List.iteri
          (fun i t ->
            let bound =
              propagated ~lipschitz:2.2 ~span:t
                (Nx.item [] (Nx.sum (Nx.get [ i ] errors)))
            in
            equal
              (Oracle.tensor ~abs:bound ())
              (solution (scalar t))
              (Nx.get [ i ] y))
          [ 0.; 0.8; 2. ]);
    test "grad in the history reaches the delayed states" (fun () ->
        (* y(3) = −c / 6. *)
        let g =
          Rune.grad'
            (fun c ->
              Nx.get [ 3 ]
                (Solution.get
                   (delay1 lagged ~history:(constant c) ~at:times
                      (Nx.reshape [||] c))))
            (scalar 1.)
        in
        equal exact (scalar (-1. /. 6.)) g);
    test "grad in the lag moves the breakpoints" (fun () ->
        (* With τ in (1, 2], y(2) = (1 − τ) − ((1 + τ)(2 − τ) − (4 − τ²)/2),
           whose derivative in τ is τ − 2. *)
        let g =
          Rune.grad'
            (fun lags ->
              Nx.get [ 1 ]
                (Solution.get
                   (delay1 ~lags lagged
                      ~history:(constant (scalar 1.))
                      ~at:(vec [| 0.; 2. |])
                      (scalar 1.))))
            (vec [| 1.2 |])
        in
        equal exact (vec [| -0.8 |]) g);
    test "grad in a captured rate reaches the field" (fun () ->
        (* y' = −a y(t − 1) from 1: y(2) = 1 − 2a + a² / 2, whose derivative at
           a = 1 is −1. *)
        let g =
          Rune.grad'
            (fun a ->
              Nx.get [ 1 ]
                (Solution.get
                   (delay1
                      (fun _ _ d ->
                        Nx.neg (Nx.mul a (Nx.squeeze ~axes:[ 0 ] d)))
                      ~history:(constant (scalar 1.))
                      ~at:(vec [| 0.; 2. |])
                      (scalar 1.))))
            (scalar 1.)
        in
        equal exact (scalar (-1.)) g);
    test "one time is the start alone, converged" (fun () ->
        equal (Oracle.tensor ()) (vec [| 1. |])
          (Solution.get
             (delay1 lagged
                ~history:(constant (scalar 1.))
                ~at:(vec [| 0. |]) (scalar 1.))));
    test "times that do not increase end the lane Stalled" (fun () ->
        let s =
          delay1 lagged
            ~history:(constant (scalar 1.))
            ~at:(vec [| 0.; 2.; 1. |])
            (scalar 1.)
        in
        equal Oracle.status_t Stalled (Oracle.status s));
    cases "a lag that is not finite and positive ends the lane Stalled"
      ~id:(Printf.sprintf "%g") [ 0.; -1.; Float.infinity; Float.nan ]
      (fun lag ->
        let s =
          delay1 ~lags:(vec [| lag |]) lagged
            ~history:(constant (scalar 1.))
            ~at:times (scalar 1.)
        in
        equal Oracle.status_t Stalled (Oracle.status s));
    test "a lag past the pieces kept ends the lane with the count it needs"
      (fun () ->
        (* A forcing keeps the steps below the lag, so one piece cannot hold the
           state a lag back; the count the report names holds it. *)
        let f t _ d =
          Nx.add (Nx.neg (Nx.squeeze ~axes:[ 0 ] d)) (Nx.sin (Nx.mul_s t 5.))
        in
        let solve pieces =
          delay1 ~pieces f ~history:(constant (scalar 1.)) ~at:times (scalar 1.)
        in
        let s = solve 1 in
        equal Oracle.status_t Stalled (Oracle.status s);
        let count =
          match Solution.get s with
          | _ -> failf "the solve converged"
          | exception Failure report -> needed report
        in
        greater ~than:1 int count;
        equal Oracle.status_t Converged (Oracle.status (solve count)));
    test "compiled is exact to rounding" (fun () ->
        Compiled.here ();
        let f c =
          Solution.get
            (delay1 lagged ~history:(constant c) ~at:times (Nx.reshape [||] c))
        in
        equal exact (f (scalar 1.)) (Rune.jit' f (scalar 1.)));
    test "raises if history returns another shape than the state" (fun () ->
        raises_in "Jera.Ode.delay" (fun () ->
            delay1 lagged
              ~history:(constant (vec [| 1. |]))
              ~at:times (scalar 1.)));
    test "raises if pieces < 1" (fun () ->
        raises_in "Jera.Ode.delay" (fun () ->
            delay1 ~pieces:0 lagged
              ~history:(constant (scalar 1.))
              ~at:times (scalar 1.)));
    test "raises if lags is not a non-empty 1-D tensor" (fun () ->
        raises_in "Jera.Ode.delay" (fun () ->
            delay1 ~lags:(vec [||]) lagged
              ~history:(constant (scalar 1.))
              ~at:times (scalar 1.)));
  ]

let () =
  exit
    (run "Jera.Ode"
       [
         group "methods" methods;
         group "tableau" tableau;
         group "kvaerno5" kvaerno5;
         group "march" march;
         group "solve" solve;
         group "sample" sample;
         group "path" path;
         group "event" event;
         group "delay" delay;
       ])
