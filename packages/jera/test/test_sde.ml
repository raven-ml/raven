(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jera.Sde: Brownian paths and stochastic marches. The trusted side is the
   Gaussian law of increments and areas, Chen's relation, and linear SDEs whose
   solutions are known in their calculus: geometric Brownian motion in closed
   form, and the Ornstein–Uhlenbeck process by its exact transition over the
   path's finest intervals. Strong errors are means over many independent
   components of one path. *)

open Windtrap
open Jera

let f64 = Nx.float64
let scalar x = Nx.scalar f64 x
let vec a = Nx.create f64 [| Array.length a |] a
let one = Nx.Ptree.tensor
let raises_with sub f = raises_match (Exn.invalid_arg ~substring:sub) f

let path ?(depth = 10) ?(n = 4000) seed =
  Sde.Brownian.v (Nx.Rng.key seed) f64 ~shape:[| n |] ~t0:0. ~t1:1. ~depth

let mean x = Nx.item [] (Nx.mean x)
let variance x = Nx.item [] (Nx.var x)

(* Brownian paths *)

let times =
  Gen.(
    let+ a = float_range 0. 1.
    and+ b = float_range 0. 1.
    and+ c = float_range 0. 1. in
    let l = List.sort compare [ a; b; c ] in
    (List.nth l 0, List.nth l 1, List.nth l 2))
  |> Gen.with_pp (fun ppf (s, t, u) -> Format.fprintf ppf "(%g, %g, %g)" s t u)

let brownian_tests =
  let w = path ~n:8 3 in
  [
    prop "increments compose by Chen's relation" times (fun (s, t, u) ->
        let inc a b = Sde.Brownian.increment w (scalar a) (scalar b) in
        let w1, h1 = inc s t and w2, h2 = inc t u and w12, h12 = inc s u in
        let tol = Oracle.tensor ~abs:1e-12 () in
        equal ~msg:"W" tol w12 (Nx.add w1 w2);
        assume (u -. s > 1e-9);
        let h1' = t -. s and h2' = u -. t and h = u -. s in
        let chen =
          Nx.div_s
            (Nx.add
               (Nx.add (Nx.mul_s h1 h1') (Nx.mul_s h2 h2'))
               (Nx.div_s (Nx.sub (Nx.mul_s w1 h2') (Nx.mul_s w2 h1')) 2.))
            h
        in
        equal ~msg:"H" (Oracle.tensor ~abs:(1e-12 /. h) ()) h12 chen);
    test "a query is a pure function of the key and the times" (fun () ->
        let inc w = Sde.Brownian.increment w (scalar 0.2) (scalar 0.7) in
        equal
          (Oracle.structure Nx.Ptree.(pair tensor tensor))
          (inc (path ~n:8 3))
          (inc (path ~n:8 3)));
    test "an empty interval has no increment and no area" (fun () ->
        let dw, h = Sde.Brownian.increment w (scalar 0.3) (scalar 0.3) in
        equal (Oracle.tensor ()) (Nx.zeros f64 [| 8 |]) dw;
        equal (Oracle.tensor ()) (Nx.zeros f64 [| 8 |]) h);
    cases
      ~name:(fun (s, t) -> Printf.sprintf "[%g, %g]" s t)
      "increments and areas have their Gaussian law"
      [ (0., 1.); (0.25, 0.75); (0.1, 0.43); (0.6001, 0.6123) ]
      (fun (s, t) ->
        (* 4000 components: a mean's standard error is σ / 63, a variance's
           about σ² / 45; five of either is the bound. *)
        let dw, h = Sde.Brownian.increment (path 11) (scalar s) (scalar t) in
        let l = t -. s in
        less ~msg:"W mean" (float 1.)
          ~than:(5. *. Float.sqrt l /. 63.)
          (Float.abs (mean dw));
        less ~msg:"W variance" (float 1.)
          ~than:(5. *. l /. 45.)
          (Float.abs (variance dw -. l));
        less ~msg:"H mean" (float 1.)
          ~than:(5. *. Float.sqrt (l /. 12.) /. 63.)
          (Float.abs (mean h));
        less ~msg:"H variance" (float 1.)
          ~than:(5. *. l /. 12. /. 45.)
          (Float.abs (variance h -. (l /. 12.)));
        less ~msg:"W H covariance" (float 1.)
          ~than:(5. *. l /. Float.sqrt 12. /. 63.)
          (Float.abs (mean (Nx.mul dw h))));
    test "increments over disjoint intervals are uncorrelated" (fun () ->
        let w = path 12 in
        let a, _ = Sde.Brownian.increment w (scalar 0.1) (scalar 0.4) in
        let b, _ = Sde.Brownian.increment w (scalar 0.4) (scalar 0.9) in
        less (float 1.)
          ~than:(5. *. Float.sqrt (0.3 *. 0.5) /. 63.)
          (Float.abs (mean (Nx.mul a b))));
    test "a time outside the path raises" (fun () ->
        raises_with "t = 1.5 is outside [0, 1]" (fun () ->
            Sde.Brownian.increment w (scalar 0.) (scalar 1.5)));
    test "v rejects an empty interval" (fun () ->
        raises_with "t1 = 1 is not above t0 = 1" (fun () ->
            Sde.Brownian.v (Nx.Rng.key 0) f64 ~shape:[||] ~t0:1. ~t1:1. ~depth:3));
    test "v rejects a depth past 30" (fun () ->
        raises_with "depth = 31 is not in [0, 30]" (fun () ->
            Sde.Brownian.v (Nx.Rng.key 0) f64 ~shape:[||] ~t0:0. ~t1:1.
              ~depth:31));
  ]

(* Strong orders *)

let mu = 0.5
and sigma = 0.8

let x0 n = Nx.full f64 [| n |] 1.

(* The mean, over the components, of the error at t = 1 of [m] in [steps] steps
   against [exact]. *)
let strong m ~drift ~diffusion ~exact w steps =
  let n = 4000 in
  let y =
    Sde.march one m ~steps ~drift ~diffusion w ~at:(vec [| 0.; 1. |]) (x0 n)
  in
  mean (Nx.abs (Nx.sub (Nx.get [ 1 ] y) exact))

(* The least-squares slope of log₂ error against log₂ steps, negated. *)
let order errors =
  let k = Float.of_int (List.length errors) in
  let xs = List.mapi (fun i _ -> Float.of_int i) errors
  and ys = List.map Float.log2 errors in
  let mx = List.fold_left ( +. ) 0. xs /. k
  and my = List.fold_left ( +. ) 0. ys /. k in
  let sxy =
    List.fold_left2 (fun acc x y -> acc +. ((x -. mx) *. (y -. my))) 0. xs ys
  in
  let sxx =
    List.fold_left (fun acc x -> acc +. ((x -. mx) *. (x -. mx))) 0. xs
  in
  -.sxy /. sxx

let gbm_drift _ x = Nx.mul_s x mu
let gbm_diffusion _ x dw = Nx.mul (Nx.mul_s x sigma) dw

(* X(1) of the Itô and Stratonovich geometric Brownian motions from 1. *)
let gbm_exact ~ito w =
  let dw, _ = Sde.Brownian.increment w (scalar 0.) (scalar 1.) in
  let drift = if ito then mu -. (sigma *. sigma /. 2.) else mu in
  Nx.exp (Nx.add_s (Nx.mul_s dw sigma) drift)

let ou_drift _ x = Nx.neg x
let ou_diffusion _ _ dw = Nx.mul_s dw sigma

(* X(1) of dX = −X dt + σ dW from 1, by the exact transition over each of the
   path's finest intervals δ: e^(−δ) X + σ (ΔW − δ (ΔW / 2 + H)), whose error is
   O(δ^(5/2)) per interval. *)
let ou_exact w ~depth =
  let k = 1 lsl depth in
  let d = 1. /. Float.of_int k in
  let x = ref (x0 4000) in
  for i = 0 to k - 1 do
    let dw, h =
      Sde.Brownian.increment w
        (scalar (Float.of_int i *. d))
        (scalar (Float.of_int (i + 1) *. d))
    in
    let noise =
      Nx.mul_s (Nx.sub dw (Nx.mul_s (Nx.add (Nx.div_s dw 2.) h) d)) sigma
    in
    x := Nx.add (Nx.mul_s !x (Float.exp (-.d))) noise
  done;
  !x

let steps = [ 4; 8; 16; 32 ]

let order_tests =
  [
    slow "euler_maruyama has strong order 1/2 on Itô's geometric motion"
      (fun () ->
        let w = path 21 in
        let exact = gbm_exact ~ito:true w in
        let e =
          List.map
            (strong Sde.euler_maruyama ~drift:gbm_drift ~diffusion:gbm_diffusion
               ~exact w)
            steps
        in
        at_least (float 1e-9) ~than:0.35 (order e);
        at_most (float 1e-9) ~than:0.75 (order e));
    slow "milstein has strong order 1 on Itô's geometric motion" (fun () ->
        let w = path 22 in
        let exact = gbm_exact ~ito:true w in
        let e =
          List.map
            (strong Sde.milstein ~drift:gbm_drift ~diffusion:gbm_diffusion
               ~exact w)
            steps
        in
        at_least (float 1e-9) ~than:0.85 (order e));
    slow "sra1 has strong order 3/2 on additive noise" (fun () ->
        let w = path ~depth:9 23 in
        let exact = ou_exact w ~depth:9 in
        let e =
          List.map
            (strong Sde.sra1 ~drift:ou_drift ~diffusion:ou_diffusion ~exact w)
            steps
        in
        at_least (float 1e-9) ~than:1.3 (order e));
    slow "reversible_heun has strong order 1 on additive noise" (fun () ->
        let w = path ~depth:9 24 in
        let exact = ou_exact w ~depth:9 in
        let e =
          List.map
            (strong Sde.reversible_heun ~drift:ou_drift ~diffusion:ou_diffusion
               ~exact w)
            steps
        in
        at_least (float 1e-9) ~than:0.85 (order e));
    slow "reversible_heun converges on Stratonovich's geometric motion"
      (fun () ->
        let w = path 25 in
        let exact = gbm_exact ~ito:false w in
        let e =
          List.map
            (strong Sde.reversible_heun ~drift:gbm_drift
               ~diffusion:gbm_diffusion ~exact w)
            steps
        in
        at_least (float 1e-9) ~than:0.4 (order e));
  ]

(* Marches *)

let march_tests =
  let w = path ~n:3 31 in
  let run ?(steps = 4) ?(m = Sde.euler_maruyama) at x =
    Sde.march one m ~steps ~drift:gbm_drift ~diffusion:gbm_diffusion w ~at x
  in
  [
    test "a march stacks the state at each time, the start first" (fun () ->
        let y = run (vec [| 0.; 0.5; 1. |]) (x0 3) in
        equal (array int) [| 3; 3 |] (Nx.shape y);
        equal (Oracle.tensor ()) (x0 3) (Nx.get [ 0 ] y));
    test "marches with different steps see one path" (fun () ->
        (* With zero drift and a constant diffusion every method's state is x0 +
           σ W, whatever its steps. *)
        let at = vec [| 0.; 0.3; 1. |] in
        let walk steps =
          Sde.march one Sde.euler_maruyama ~steps
            ~drift:(fun _ x -> Nx.zeros_like x)
            ~diffusion:ou_diffusion w ~at (x0 3)
        in
        equal (Oracle.tensor ~abs:1e-14 ()) (walk 1) (walk 7));
    test "grad in the initial state is the finite difference" (fun () ->
        let f x =
          Nx.sum (Nx.get [ 2 ] (run ~m:Sde.milstein (vec [| 0.; 0.5; 1. |]) x))
        in
        let x = vec [| 1.; 0.5; 2. |] in
        let v = vec [| 1.; -1.; 0.5 |] in
        equal
          (Oracle.tensor ~rel:1e-7 ())
          (Nx.reshape [||] (Oracle.central ~eps:1e-6 f x v))
          (scalar (Oracle.dot (Rune.grad' f x) v)));
    test "a compiled march with the path an argument equals eager" (fun () ->
        let f w x =
          Sde.march one Sde.milstein ~steps:4 ~drift:gbm_drift
            ~diffusion:gbm_diffusion w
            ~at:(vec [| 0.; 0.5; 1. |])
            x
        in
        let compiled =
          Rune.jit
            Nx.Ptree.(Sde.Brownian.ptree f64 @-> tensor @-> returns tensor)
            f
        in
        let x = vec [| 1.; 0.5; 2. |] in
        equal (Oracle.tensor ~rel:1e-12 ()) (f w x) (compiled w x);
        (* Another key replays the same program on another path. *)
        let w' = path ~n:3 32 in
        equal (Oracle.tensor ~rel:1e-12 ()) (f w' x) (compiled w' x));
    test "times that are not increasing raise" (fun () ->
        raises_with "the times of at are not strictly increasing" (fun () ->
            run (vec [| 1.; 0.5 |]) (x0 3)));
    test "times outside the path raise" (fun () ->
        raises_with "a time of at = 2 is outside [0, 1]" (fun () ->
            run (vec [| 0.; 2. |]) (x0 3)));
    test "a drift of another shape raises" (fun () ->
        raises_with "the drift returned a value of another structure" (fun () ->
            Sde.march one Sde.sra1 ~steps:1
              ~drift:(fun _ x -> Nx.sum x)
              ~diffusion:ou_diffusion w
              ~at:(vec [| 0.; 1. |])
              (x0 3)));
  ]

let () =
  exit
    (run "Jera.Sde"
       [
         group "brownian" brownian_tests;
         group "order" order_tests;
         group "march" march_tests;
       ])
