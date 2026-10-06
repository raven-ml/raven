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
          (fun () -> Solution.get s);
        let report = Format.asprintf "%a" Solution.pp s in
        contains ~sub:"method tsit5, tol rel 1e-10 abs 1e-12, budget 3" report;
        contains ~sub:"3 of 3 attempted steps" report);
    test "a long interval after many short ones still shrinks its steps"
      (fun () ->
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
        equal (Oracle.tensor ()) (Nx.scalar Nx.bool true) (Solution.ok s));
    test "times out of order end their lane, and the report says where"
      (fun () ->
        let s =
          Ode.sample one Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:100
            (decay (scalar 1.))
            ~at:(vec [| 0.; 1.; 0.5 |])
            (scalar 1.)
        in
        equal (Oracle.tensor ()) (Nx.scalar Nx.bool true)
          (Solution.is Stalled s);
        raises_match
          (Exn.failure ~substring:"The times are not strictly monotone at [2].")
          (fun () -> Solution.get s));
    test "times out of order in one lane leave the others converged" (fun () ->
        let at =
          Nx.create f64 [| 3; 3 |] [| 0.; 1.; 2.; 0.; 1.; 0.5; 0.; 0.5; 1. |]
        in
        let ok =
          Rune.vmap
            Nx.Ptree.(tensor @-> returns tensor)
            (fun at ->
              Solution.ok
                (Ode.sample one Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:100
                   (decay (scalar 1.))
                   ~at (scalar 1.)))
            at
        in
        equal (Oracle.tensor ())
          (Nx.create Nx.bool [| 3 |] [| true; false; true |])
          ok);
    test "a solve over no time is its start" (fun () ->
        let s = solve (decay (scalar 1.)) ~t0:0.7 ~t1:0.7 (scalar 2.) in
        equal (Oracle.tensor ()) (scalar 2.) (Solution.get s);
        equal (Oracle.tensor ()) (scalar 0.) (Solution.error s));
    test "solve rejects a time that is not a scalar" (fun () ->
        raises_match
          (Exn.invalid_arg
             ~substring:"Jera.Ode.solve: t0 and t1 must be scalars") (fun () ->
            Ode.solve one Ode.tsit5 ~tol:(Tol.rel 1e-8) ~budget:10
              (decay (scalar 1.))
              ~t0:(vec [| 0. |]) ~t1:(scalar 1.) (scalar 1.)));
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

(* Paths *)

let path ?(m = (Ode.tsit5 :> ([ `Embedded ], _, _) Ode.t))
    ?(tol = Tol.v ~rel:1e-10 ~abs:1e-12) ?(budget = 200) f ~t0 ~t1 y0 =
  Ode.path one m ~tol ~budget f ~t0:(scalar t0) ~t1:(scalar t1) y0

let path_tests =
  let embedded m = (m :> ([ `Embedded ], _, _) Ode.t) in
  let between = Nx.linspace f64 0. 2. 101 in
  let exact k t0 y0 t = Nx.mul_s (Nx.exp (Nx.mul_s (Nx.sub_s t t0) (-.k))) y0 in
  [
    test "a path ends at the solve's state" (fun () ->
        let p = Solution.get (path forced2 ~t0:0. ~t1:2. (scalar 1.)) in
        let y = Solution.get (solve forced2 ~t0:0. ~t1:2. (scalar 1.)) in
        equal
          (Oracle.tensor ~rel:1e-14 ())
          (vec [| 1.; Nx.item [] y |])
          (Piecewise.eval p (vec [| 0.; 2. |])));
    cases
      ~name:(fun (n, _, _, _) -> n)
      "between steps a path carries its extension's error"
      [
        ("bs3", embedded Ode.bs3, 1e-8, 1e-6);
        ("tsit5", embedded Ode.tsit5, 1e-10, 1e-8);
        ("dopri5", embedded Ode.dopri5, 1e-10, 1e-8);
      ]
      (fun (_, m, tol, within) ->
        let p =
          Solution.get
            (path ~m ~tol:(Tol.v ~rel:tol ~abs:tol) ~budget:2000
               (decay (scalar 3.))
               ~t0:0. ~t1:2. (scalar 1.))
        in
        equal
          (Oracle.tensor ~abs:within ())
          (exact 3. 0. 1. between) (Piecewise.eval p between));
    test "a path backward in time holds its pieces in increasing time"
      (fun () ->
        let y2 = Float.exp (-2.) in
        let p =
          Solution.get (path (decay (scalar 1.)) ~t0:2. ~t1:0. (scalar y2))
        in
        equal
          (Oracle.tensor ~abs:1e-9 ())
          (exact 1. 2. y2 between) (Piecewise.eval p between));
    test "past the last step the pieces are empty at its end" (fun () ->
        let p =
          Solution.get (path (decay (scalar 1.)) ~t0:0. ~t1:2. (scalar 1.))
        in
        let breaks = Piecewise.breaks p in
        let n = Nx.dim 0 breaks in
        equal (Oracle.tensor ()) (scalar 2.) (Nx.get [ n - 1 ] breaks);
        equal (Oracle.tensor ()) (scalar 2.) (Nx.get [ n - 2 ] breaks));
    test "a path over no time ends its lane Stalled" (fun () ->
        let s = path (decay (scalar 1.)) ~t0:1. ~t1:1. (scalar 1.) in
        equal (Oracle.tensor ()) (Nx.scalar Nx.bool true)
          (Solution.is Stalled s));
    test "its error is the accumulated estimate, as sample's" (fun () ->
        let tol = Tol.v ~rel:1e-6 ~abs:1e-8 in
        let p = path ~tol (decay (scalar 1.)) ~t0:0. ~t1:2. (scalar 1.) in
        let s =
          Ode.sample one Ode.tsit5 ~tol ~budget:200
            (decay (scalar 1.))
            ~at:(vec [| 0.; 2. |])
            (scalar 1.)
        in
        (* The answer's steps are the search's, recomputed from their fractions
           of the span: equal up to the rounding of each step. *)
        equal
          (Oracle.tensor ~rel:1e-9 ())
          (Nx.get [ 1 ] (Solution.error s))
          (Piecewise.eval (Solution.error p) (scalar 2.)));
    test "grad in a captured rate is −t y0 e^(−kt)" (fun () ->
        let g =
          Rune.grad'
            (fun k ->
              Piecewise.eval
                (Solution.get (path (decay k) ~t0:0. ~t1:2. (scalar 2.)))
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
    test "compiled equals eager to rounding for a polynomial field" (fun () ->
        let f k =
          Piecewise.eval
            (Solution.get (path (decay k) ~t0:0. ~t1:2. (scalar 1.)))
            between
        in
        equal
          (Oracle.tensor ~rel:1e-13 ())
          (f (scalar 0.8))
          (Rune.jit' f (scalar 0.8)));
  ]

(* Events *)

let gravity = 9.81
let fall _ (_, p) = (p, Nx.full_like p (-.gravity))

(* The first time a ball dropped from rest at [q0] reaches [level]. *)
let reaches q0 level = Float.sqrt (2. *. (q0 -. level) /. gravity)

let drop ?(t1 = 5.) ?(tol = Tol.v ~rel:1e-12 ~abs:1e-12) event q0 =
  Ode.event pair Ode.tsit5 ~tol ~budget:200 fall ~event ~t0:(scalar 0.)
    ~t1:(scalar t1)
    (q0, Nx.zeros_like q0)

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

let event_tests =
  let close = Oracle.tensor ~rel:1e-10 () in
  let index i = Nx.scalar Nx.int32 i in
  [
    test "a ball reaches the ground at sqrt(2 h / g)" (fun () ->
        let t, _, i = Solution.get (drop (fun _ (q, _) -> q) (scalar 10.)) in
        equal close (scalar (reaches 10. 0.)) t;
        equal (Oracle.tensor ()) (index 0l) i);
    test "the returned state has the component's new sign" (fun () ->
        let _, (q, _), _ =
          Solution.get (drop (fun _ (q, _) -> q) (scalar 10.))
        in
        at_most float_exact ~than:0. (Nx.item [] q));
    test "the earliest of several events wins, with its index" (fun () ->
        let t, _, i =
          Solution.get
            (drop
               (fun _ (q, _) -> Nx.stack [ Nx.sub_s q 5.; Nx.sub_s q 8. ])
               (scalar 10.))
        in
        equal close (scalar (reaches 10. 8.)) t;
        equal (Oracle.tensor ()) (index 1l) i);
    test "without a crossing the solve ends at t1 with index -1" (fun () ->
        let t, (q, _), i =
          Solution.get (drop ~t1:1. (fun _ (q, _) -> q) (scalar 10.))
        in
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
        equal (Oracle.tensor ~rel:1e-7 ()) (scalar 2.) t);
    test "a component that touches zero without changing sign does not cross"
      (fun () ->
        let t, _, i = Solution.get (plateau ~after:(-1.)) in
        equal (Oracle.tensor ()) (index (-1l)) i;
        equal (Oracle.tensor ()) (scalar 5.) t);
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
    test "grad of the time in the height is 1 / (g t)" (fun () ->
        let g =
          Rune.grad'
            (fun q0 ->
              let t, _, _ = Solution.get (drop (fun _ (q, _) -> q) q0) in
              t)
            (scalar 10.)
        in
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (scalar (1. /. (gravity *. reaches 10. 0.)))
          g);
    test "grad of the state at the crossing is the field there" (fun () ->
        (* At the ground the velocity is −g t*, so d v / d q0 is −1 / t*. *)
        let g =
          Rune.grad'
            (fun q0 ->
              let _, (_, p), _ = Solution.get (drop (fun _ (q, _) -> q) q0) in
              p)
            (scalar 10.)
        in
        equal (Oracle.tensor ~rel:1e-8 ()) (scalar (-1. /. reaches 10. 0.)) g);
    test "lanes with and without a crossing are independent" (fun () ->
        let f q0 =
          let t, _, _ = Solution.get (drop ~t1:1.5 (fun _ (q, _) -> q) q0) in
          t
        in
        equal close
          (vec [| reaches 5. 0.; 1.5 |])
          (Rune.vmap' f (vec [| 5.; 50. |])));
    test "compiled equals eager within the crossing's tolerance" (fun () ->
        (* The steps' sizes round differently compiled, so the crossing's search
           takes other points; both times are within the tolerance, rel 1e-12,
           of the crossing. *)
        let f q0 =
          let t, _, _ = Solution.get (drop (fun _ (q, _) -> q) q0) in
          t
        in
        equal
          (Oracle.tensor ~rel:5e-12 ())
          (f (scalar 10.))
          (Rune.jit' f (scalar 10.)));
    test "an event of no component raises" (fun () ->
        raises_match
          (Exn.invalid_arg
             ~substring:"Jera.Ode.event: the event has no component") (fun () ->
            drop (fun _ _ -> Nx.zeros f64 [| 0 |]) (scalar 10.)));
  ]

(* Delays *)

(* y' = −y(t − τ) from y = c before 0: on [0, τ] y = c (1 − t); with τ = 1, y(2)
   = −c / 2 and y(3) = −c / 6, by the method of steps. *)
let lagged _ _ d = Nx.neg (Nx.squeeze ~axes:[ 0 ] d)

let delayed ?(tol = Tol.v ~rel:1e-12 ~abs:1e-12) ?(pieces = 64)
    ?(lags = vec [| 1. |]) ?(c = scalar 1.) ?(f = lagged) at =
  Ode.delay one Ode.tsit5 ~tol ~budget:400 ~pieces f ~lags
    ~history:(fun _ -> c)
    ~at (Nx.reshape [||] c)

let delay_tests =
  let close = Oracle.tensor ~rel:1e-10 ~abs:1e-12 () in
  let times = vec [| 0.; 1.; 2.; 3. |] in
  [
    test "y' = −y(t − 1) follows the method of steps" (fun () ->
        equal close
          (vec [| 1.; 0.; -0.5; -1. /. 6. |])
          (Solution.get (delayed times)));
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
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.exp (vec [| 0.; 0.5; 1.7; 3. |]))
          (Solution.get s));
    test "two lags stack on a leading axis" (fun () ->
        let f _ _ d = Nx.neg (Nx.get [ 1 ] d) in
        equal close
          (vec [| 1.; 0.; -0.5; -1. /. 6. |])
          (Solution.get (delayed ~lags:(vec [| 2.; 1. |]) ~f times)));
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
        equal
          (Oracle.tensor ~rel:1e-8 ())
          (Nx.stack (List.map (fun t -> solution (scalar t)) [ 0.; 0.8; 2. ]))
          (Solution.get s));
    test "grad in the history reaches the delayed states" (fun () ->
        let g =
          Rune.grad'
            (fun c -> Nx.get [ 3 ] (Solution.get (delayed ~c times)))
            (scalar 1.)
        in
        equal close (scalar (-1. /. 6.)) g);
    test "grad in the lag moves the breakpoints" (fun () ->
        (* With τ in (1, 2], y(2) = (1 − τ) − ((1 + τ)(2 − τ) − (4 − τ²)/2),
           whose derivative in τ is τ − 2. *)
        let g =
          Rune.grad'
            (fun lags ->
              Nx.get [ 1 ] (Solution.get (delayed ~lags (vec [| 0.; 2. |]))))
            (vec [| 1.2 |])
        in
        equal close (vec [| -0.8 |]) g);
    test "a history of another shape than the state raises" (fun () ->
        raises_match
          (Exn.invalid_arg
             ~substring:"Jera.Ode.delay: history returned a value of another")
          (fun () ->
            Ode.delay one Ode.tsit5 ~tol:(Tol.rel 1e-6) ~budget:10 ~pieces:4
              lagged ~lags:(vec [| 1. |])
              ~history:(fun _ -> vec [| 1. |])
              ~at:times (scalar 1.)));
    test "a lag that is not positive ends the lane Stalled" (fun () ->
        let s = delayed ~lags:(vec [| 0. |]) times in
        equal (Oracle.tensor ()) (Nx.scalar Nx.bool true)
          (Solution.is Stalled s);
        raises_match (Exn.failure ~substring:"Every lag must be positive.")
          (fun () -> Solution.get s));
    test "a lag past the pieces kept ends the lane with the count it needs"
      (fun () ->
        (* A forcing keeps the steps below the lag, so one piece cannot hold the
           state a lag back; the count the report names holds it. *)
        let f t _ d =
          Nx.add (Nx.neg (Nx.squeeze ~axes:[ 0 ] d)) (Nx.sin (Nx.mul_s t 5.))
        in
        let s = delayed ~pieces:1 ~f times in
        equal (Oracle.tensor ()) (Nx.scalar Nx.bool true)
          (Solution.is Stalled s);
        let report =
          match Solution.get s with
          | _ -> failf "the solve converged"
          | exception Failure m -> m
        in
        contains ~sub:"reaches back past the last pieces = 1 steps" report;
        let marker = "raise pieces to about " in
        let needed =
          let rec find i =
            if i + String.length marker > String.length report then
              failf "no count in: %s" report
            else if String.sub report i (String.length marker) = marker then
              i + String.length marker
            else find (i + 1)
          in
          let i = find 0 in
          Scanf.sscanf
            (String.sub report i (String.length report - i))
            "%d" Fun.id
        in
        greater ~than:1 int needed;
        equal (Oracle.tensor ()) (Nx.scalar Nx.bool true)
          (Solution.ok (delayed ~pieces:needed ~f times)));
    test "compiled equals eager to rounding" (fun () ->
        let f c = Solution.get (delayed ~c times) in
        equal
          (Oracle.tensor ~rel:1e-13 ~abs:1e-15 ())
          (f (scalar 1.))
          (Rune.jit' f (scalar 1.)));
  ]

let () =
  exit
    (run "Jera.Ode"
       [
         group "solve" solve_tests;
         group "solve derivatives" solve_derivative_tests;
         group "path" path_tests;
         group "event" event_tests;
         group "delay" delay_tests;
         group "order" order_tests;
         group "march" march_tests;
         group "transformations" transformation_tests;
         group "errors" error_tests;
       ])
