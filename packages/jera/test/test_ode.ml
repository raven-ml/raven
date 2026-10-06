(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Ode's methods and marches. The trusted side is a nonlinear
   non-autonomous system integrated in OCaml floats by a fine classical
   Runge–Kutta march, the methods' stability polynomials on y' = λy, and finite
   differences of the march itself. *)

open Windtrap
open Jera

let f64 = Nx.float64
let scalar x = Nx.scalar f64 x
let vec a = Nx.create f64 [| Array.length a |] a
let one = Nx.Ptree.tensor
let pair = Nx.Ptree.(pair tensor tensor)

(* A forced pendulum with a time-varying stiffness: every elementary
   differential of order five is nonzero, so a method that misses one order
   condition shows a lower order. *)
let forced t (q, p) =
  ( p,
    Nx.add
      (Nx.neg (Nx.sin q))
      (Nx.mul (Nx.mul_s (Nx.cos (Nx.mul_s t 2.)) 0.3) q) )

let forced_host t (q, p) =
  (p, -.Float.sin q +. (0.3 *. Float.cos (2. *. t) *. q))

(* The state at [t] from (1, 0), by 2¹⁴ classical steps in OCaml floats. *)
let reference t =
  let n = 1 lsl 14 in
  let h = t /. Float.of_int n in
  let ( +^ ) (a, b) (c, d) = (a +. c, b +. d)
  and ( *^ ) s (a, b) = (s *. a, s *. b) in
  let rec go i y =
    if i = n then y
    else
      let t = Float.of_int i *. h in
      let k1 = forced_host t y in
      let k2 = forced_host (t +. (h /. 2.)) (y +^ (h /. 2. *^ k1)) in
      let k3 = forced_host (t +. (h /. 2.)) (y +^ (h /. 2. *^ k2)) in
      let k4 = forced_host (t +. h) (y +^ (h *^ k3)) in
      go (i + 1) (y +^ (h /. 6. *^ (k1 +^ (2. *^ k2) +^ (2. *^ k3) +^ k4)))
  in
  go 0 (1., 0.)

let error m n =
  let q, p =
    Ode.march pair m ~steps:n forced
      ~at:(vec [| 0.; 2. |])
      (scalar 1., scalar 0.)
  in
  let rq, rp = reference 2. in
  Float.max
    (Float.abs (Nx.item [ 1 ] q -. rq))
    (Float.abs (Nx.item [ 1 ] p -. rp))

let methods =
  let formula m = (m :> ([ `Formula ], _, _) Ode.t) in
  [
    ("euler", Ode.euler, 1, 256);
    ("ssprk3", Ode.ssprk3, 3, 32);
    ("rk4", Ode.rk4, 4, 16);
    ("bs3", formula Ode.bs3, 3, 32);
    ("tsit5", formula Ode.tsit5, 5, 32);
    ("dopri5", formula Ode.dopri5, 5, 16);
  ]

let name (n, _, _, _) = n

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

let order_tests =
  [
    cases ~name "each method reaches its order on a forced pendulum" methods
      (fun (_, m, p, n) ->
        let order = Oracle.slope (error m n) (error m (2 * n)) in
        at_least (float 1e-9) ~than:(Float.of_int p -. 0.25) order;
        at_most (float 1e-9) ~than:(Float.of_int p +. 1.25) order);
    cases
      ~name:(fun (n, _, _) -> n)
      "each method's step on y' = λy is its stability polynomial"
      [
        ("euler", (fun z -> growth Ode.euler z), taylor 1);
        ("ssprk3", (fun z -> growth Ode.ssprk3 z), taylor 3);
        ("rk4", (fun z -> growth Ode.rk4 z), taylor 4);
        ("bs3", (fun z -> growth Ode.bs3 z), taylor 3);
        ( "dopri5",
          (fun z -> growth Ode.dopri5 z),
          fun z -> taylor 5 z +. ((z ** 6.) /. 600.) );
      ]
      (fun (_, step, poly) ->
        List.iter
          (fun z ->
            equal
              ~msg:(Printf.sprintf "z = %g" z)
              (float_rel ~rel:1e-14 ~abs:0.)
              (poly z) (step z))
          [ -2.5; -1.; -0.3; 0.7 ]);
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
  ]

let march_tests =
  [
    test "a march stacks the state at each time, the start first" (fun () ->
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
    test "a leaf that is not a float is carried unchanged" (fun () ->
        let s = Nx.Ptree.(pair tensor tensor) in
        let _, n =
          Ode.march s Ode.tsit5 ~steps:4
            (fun _ (y, n) -> (Nx.neg y, Nx.mul_s n 7l))
            ~at:(vec [| 0.; 1.; 2. |])
            (scalar 1., Nx.scalar Nx.int32 5l)
        in
        equal (Oracle.tensor ()) (Nx.full Nx.int32 [| 3 |] 5l) n);
  ]

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

let transformation_tests =
  let fd ~eps f x = Oracle.central ~eps f x (Nx.ones_like x) in
  [
    test "grad in the initial state is the finite difference" (fun () ->
        let q0 = scalar 0.7 in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (fd ~eps:1e-6 positions q0)
          (Rune.grad' positions q0));
    test "grad in a captured parameter is the finite difference" (fun () ->
        let k = scalar 1.3 in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (fd ~eps:1e-6 stiffness k) (Rune.grad' stiffness k));
    test "grad in the end time is the finite difference" (fun () ->
        let t1 = scalar 1.9 in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (fd ~eps:1e-6 ending t1) (Rune.grad' ending t1));
    test "jvp agrees with grad" (fun () ->
        let k = scalar 0.8 in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Rune.grad' stiffness k)
          (snd (Rune.jvp' stiffness k (scalar 1.))));
    test "compiled equals eager" (fun () ->
        let q0 = scalar 0.7 in
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (positions q0) (Rune.jit' positions q0));
    test "compiled grad equals eager grad" (fun () ->
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
  ]

let error_tests =
  let raises_with sub f = raises_match (Exn.invalid_arg ~substring:sub) f in
  let tableau ?(a = [| [||]; [| 1. |] |]) ?(b = [| 0.5; 0.5 |])
      ?(c = [| 0.; 1. |]) () =
    ignore (Ode.tableau ~a ~b ~c)
  in
  [
    test "tableau accepts Heun's method" (fun () -> tableau ());
    test "tableau rejects an empty b" (fun () ->
        raises_with "b is empty" (fun () -> tableau ~a:[||] ~b:[||] ~c:[||] ()));
    test "tableau rejects a short c" (fun () ->
        raises_with "c has 1 elements, b 2" (fun () -> tableau ~c:[| 0. |] ()));
    test "tableau rejects a row of the wrong length" (fun () ->
        raises_with "row 1 of a has 2 elements" (fun () ->
            tableau ~a:[| [||]; [| 1.; 0. |] |] ()));
    test "tableau rejects weights that do not sum to one" (fun () ->
        raises_with "b does not sum to 1" (fun () ->
            tableau ~b:[| 0.5; 0.6 |] ()));
    test "tableau rejects a NaN" (fun () ->
        raises_with "not finite" (fun () -> tableau ~c:[| 0.; nan |] ()));
    test "march rejects zero steps" (fun () ->
        raises_with "steps = 0" (fun () ->
            Ode.march one Ode.rk4 ~steps:0
              (fun _ y -> y)
              ~at:(vec [| 0.; 1. |])
              (scalar 1.)));
    test "march rejects times that are not monotone" (fun () ->
        raises_with "not strictly monotone at [2]: 1 after 1" (fun () ->
            Ode.march one Ode.rk4 ~steps:1
              (fun _ y -> y)
              ~at:(vec [| 0.; 1.; 1. |])
              (scalar 1.)));
    test "march rejects a field of another shape" (fun () ->
        raises_with "the field returned a value of another structure" (fun () ->
            Ode.march one Ode.rk4 ~steps:1
              (fun _ y -> Nx.stack [ y; y ])
              ~at:(vec [| 0.; 1. |])
              (scalar 1.)));
  ]

(* Solves *)

let decay k _ y = Nx.neg (Nx.mul k y)

let solve ?(m = (Ode.tsit5 :> ([ `Embedded ], _, _) Ode.t))
    ?(tol = Tol.v ~rel:1e-10 ~abs:1e-12) ?(budget = 1000) f ~t0 ~t1 y0 =
  Ode.solve one m ~tol ~budget f ~t0:(scalar t0) ~t1:(scalar t1) y0

(* y' = cos(t) y + sin(3t): smooth, non-autonomous. *)
let forced2 t y = Nx.add (Nx.mul (Nx.cos t) y) (Nx.sin (Nx.mul_s t 3.))

let solve_tests =
  let embedded m = (m :> ([ `Embedded ], _, _) Ode.t) in
  [
    cases
      ~name:(fun (n, _, _) -> n)
      "each embedded method meets its tolerance on a forced pendulum"
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
        let q, p = Solution.get s in
        let rq, rp = reference 2. in
        equal
          (Oracle.structure ~abs:(100. *. tol) pair)
          (scalar rq, scalar rp)
          (q, p));
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
    test "sample lands on each time" (fun () ->
        let at = vec [| 0.; 0.5; 1.25; 2. |] in
        let y =
          Solution.get
            (Ode.sample one Ode.dopri5
               ~tol:(Tol.v ~rel:1e-11 ~abs:1e-13)
               ~budget:1000
               (decay (scalar 0.7))
               ~at (scalar 2.))
        in
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.mul_s (Nx.exp (Nx.mul_s at (-0.7))) 2.)
          y);
    test "decreasing times solve backward" (fun () ->
        let y =
          Solution.get
            (solve (decay (scalar 1.)) ~t0:1. ~t1:0. (scalar (Float.exp (-1.))))
        in
        equal (Oracle.tensor ~rel:1e-9 ()) (scalar 1.) y);
    test "the budget ends a solve and the report says where" (fun () ->
        let s = solve ~budget:3 forced2 ~t0:0. ~t1:10. (scalar 1.) in
        equal (Oracle.tensor ()) (Nx.scalar Nx.bool true)
          (Solution.is Budget_spent s);
        raises_match
          (Exn.failure ~substring:"Jera.Ode.solve: the budget is spent")
          (fun () -> Solution.get s));
    test "a blow-up does not converge" (fun () ->
        (* y' = y² from 1 is 1 / (1 − t), infinite at t = 1. *)
        let s = solve (fun _ y -> Nx.square y) ~t0:0. ~t1:2. (scalar 1.) in
        equal (Oracle.tensor ()) (Nx.scalar Nx.bool false) (Solution.ok s));
    test "a leaf that is not a float is carried unchanged" (fun () ->
        let s2 = Nx.Ptree.(pair tensor tensor) in
        let _, n =
          Solution.get
            (Ode.solve s2 Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:100
               (fun _ (y, n) -> (Nx.neg y, n))
               ~t0:(scalar 0.) ~t1:(scalar 1.)
               (scalar 1., Nx.scalar Nx.int32 5l))
        in
        equal (Oracle.tensor ()) (Nx.scalar Nx.int32 5l) n);
  ]

let solve_derivative_tests =
  let close = Oracle.tensor ~rel:1e-7 () in
  [
    test "grad in the initial state is e^(−kT)" (fun () ->
        let g =
          Rune.grad'
            (fun y0 ->
              Solution.get (solve (decay (scalar 0.8)) ~t0:0. ~t1:1.5 y0))
            (scalar 2.)
        in
        equal close (scalar (Float.exp (-1.2))) g);
    test "grad in a captured rate is −T y0 e^(−kT)" (fun () ->
        let g =
          Rune.grad'
            (fun k -> Solution.get (solve (decay k) ~t0:0. ~t1:1.5 (scalar 2.)))
            (scalar 0.8)
        in
        equal close (scalar (-1.5 *. 2. *. Float.exp (-1.2))) g);
    test "grad in the end time is the field there" (fun () ->
        let f t1 =
          Solution.get
            (Ode.solve one Ode.tsit5
               ~tol:(Tol.v ~rel:1e-10 ~abs:1e-12)
               ~budget:1000
               (decay (scalar 0.8))
               ~t0:(scalar 0.) ~t1 (scalar 2.))
        in
        equal close
          (scalar (-0.8 *. 2. *. Float.exp (-1.2)))
          (Rune.grad' f (scalar 1.5)));
    test "compiled grad equals eager grad" (fun () ->
        let f k = Solution.get (solve (decay k) ~t0:0. ~t1:1.5 (scalar 2.)) in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Rune.grad' f (scalar 0.8))
          (Rune.jit' (Rune.grad' f) (scalar 0.8)));
    test "jvp agrees with grad" (fun () ->
        let f k = Solution.get (solve (decay k) ~t0:0. ~t1:1.5 (scalar 2.)) in
        equal
          (Oracle.tensor ~rel:1e-12 ())
          (Rune.grad' f (scalar 0.8))
          (snd (Rune.jvp' f (scalar 0.8) (scalar 1.))));
    test "a lane that did not converge has a zero derivative" (fun () ->
        (* y' = y² from 0.2 is fine to t = 2; from 1 it blows up. *)
        let f y0 =
          Solution.best (solve (fun _ y -> Nx.square y) ~t0:0. ~t1:2. y0)
        in
        let g =
          Rune.grad' (fun y0 -> Nx.sum (Rune.vmap' f y0)) (vec [| 0.2; 1. |])
        in
        equal
          (Oracle.tensor ~rel:1e-6 ())
          (vec [| 1. /. ((1. -. 0.4) ** 2.); 0. |])
          g);
    test "compiled equals eager to rounding for a polynomial field" (fun () ->
        (* A Duffing oscillator: the steps' sizes come from powers, which a
           compiled call computes within a few ulps of eager's. *)
        let duffing _ (q, p) =
          (p, Nx.sub (Nx.neg q) (Nx.mul_s (Nx.mul q (Nx.square q)) 0.3))
        in
        let f y0 =
          Solution.get
            (Ode.solve pair Ode.tsit5
               ~tol:(Tol.v ~rel:1e-8 ~abs:1e-10)
               ~budget:1000 duffing ~t0:(scalar 0.) ~t1:(scalar 2.)
               (y0, Nx.zeros_like y0))
        in
        let q0 = scalar 0.9 in
        equal
          (Oracle.structure ~rel:1e-13 pair)
          (f q0)
          (Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) f q0));
    test "compiled equals eager within the tolerance for a transcendental field"
      (fun () ->
        let f y0 =
          Solution.get
            (Ode.solve pair Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:1000 forced
               ~t0:(scalar 0.) ~t1:(scalar 2.)
               (y0, Nx.zeros_like y0))
        in
        let q0 = scalar 0.9 in
        equal
          (Oracle.structure ~rel:1e-8 pair)
          (f q0)
          (Rune.jit Nx.Ptree.(tensor @-> returns (pair tensor tensor)) f q0));
    test "vmap gives each lane its own steps" (fun () ->
        let f k = Solution.get (solve (decay k) ~t0:0. ~t1:1.5 (scalar 2.)) in
        let ks = vec [| 0.1; 5.; 0.8 |] in
        equal (Oracle.tensor ())
          (Nx.stack (List.init 3 (fun i -> f (Nx.get [ i ] ks))))
          (Rune.vmap' f ks));
  ]

let () =
  exit
    (run "Jera.Ode"
       [
         group "solve" solve_tests;
         group "solve derivatives" solve_derivative_tests;
         group "order" order_tests;
         group "march" march_tests;
         group "transformations" transformation_tests;
         group "errors" error_tests;
       ])
