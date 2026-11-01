(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module G = Norn.Gaussian
module D = Norn.Dist

type 'a pos = { a : 'a; b : 'a }

module Pos = struct
  type ('a, _) t = 'a pos

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    { a; b }
end

let pos : (Nx.host Nx.float64_t pos, Nx.host) Nx.Ptree.t =
  Nx.Ptree.instantiate (module Pos)

let t = Nx.Ptree.tensor

let pf u ?history ?elbo_draws ~steps lp k start =
  (G.pathfinder u ?history ?elbo_draws ~steps lp k start).gaussian

let f64 = Nx.scalar Nx.float64
let floats x = Nx.to_array x
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs

(* A position of 3 elements per chain: [a] of 1, [b] of 2. *)
let of_rows m =
  let c = (Nx.shape m).(0) in
  {
    a = Nx.slice [ Nx.R (0, c); Nx.R (0, 1) ] m;
    b = Nx.slice [ Nx.R (0, c); Nx.R (1, 3) ] m;
  }

let rows p = Nx.concatenate ~axis:1 [ p.a; p.b ]

(* A correlated Gaussian of three elements. *)
let loc = vec [| 1.; -2.; 0.5 |]

let tril =
  Nx.create Nx.float64 [| 3; 3 |] [| 1.; 0.; 0.; 0.8; 0.6; 0.; -0.5; 0.3; 0.4 |]

let cov = Nx.matmul tril (Nx.transpose tril)
let gaussian p = D.factors (D.mvn ~loc ~scale_tril:tril) (rows p)

let starts =
  of_rows
    (Nx.create Nx.float64 [| 4; 3 |]
       [| 3.; 3.; 3.; -4.; 1.; 2.; 0.; 0.; 0.; 1.5; -1.; -2. |])

(* [chain u g i] is chain [i]'s Gaussian of the Gaussians [g] stacked on the
   chain axis. *)
let chain u g i = Nx.Ptree.map (G.ptree u) (fun _ x -> Nx.get [ i ] x) g

(* [parts g] is [g]'s mean, scale, directions and variances, by its walk. *)
let parts u g =
  let all =
    Nx.Ptree.fold (G.ptree u)
      (fun p x acc -> (Nx.Ptree.Path.to_string p, Nx.cast Nx.float64 x) :: acc)
      g []
  in
  List.rev all

(* The dense covariance of one Gaussian over [pos]: [S (I + Σ (v - 1) u uᵀ)
   S]. *)
let dense g =
  let p = parts pos g in
  let get prefix =
    Nx.concatenate ~axis:(-1)
      [ List.assoc (prefix ^ ".a") p; List.assoc (prefix ^ ".b") p ]
  in
  let s = get "scale" and dirs = get "directions" in
  let v = List.assoc "variances" p in
  let k = (Nx.shape v).(0) in
  let inner =
    Nx.add (Nx.eye Nx.float64 3)
      (Nx.matmul (Nx.transpose dirs)
         (Nx.mul (Nx.reshape [| k; 1 |] (Nx.sub_s v 1.)) dirs))
  in
  Nx.mul (Nx.mul (Nx.reshape [| 3; 1 |] s) inner) (Nx.reshape [| 1; 3 |] s)

(* The dense covariance of one Gaussian over a three-element tensor. *)
let dense3 g =
  let p = parts t g in
  let s = List.assoc "scale" p and dirs = List.assoc "directions" p in
  let v = List.assoc "variances" p in
  let k = (Nx.shape v).(0) in
  let inner =
    Nx.add (Nx.eye Nx.float64 3)
      (Nx.matmul (Nx.transpose dirs)
         (Nx.mul (Nx.reshape [| k; 1 |] (Nx.sub_s v 1.)) dirs))
  in
  Nx.mul (Nx.mul (Nx.reshape [| 3; 1 |] s) inner) (Nx.reshape [| 1; 3 |] s)

let largest x = Nx.item [] (Nx.max (Nx.abs x))

(* [bfgs lp start ~steps] is, for each iterate [θ_l], [l >= 1], of L-BFGS's
   first [steps] on [-lp] from [start], a chain of three elements, the mean and
   covariance of its Gaussian by the dense recursion, every pair taken. *)
let bfgs lp start ~steps =
  let xs =
    Jera.Minimize.iterates t
      (Jera.Minimize.lbfgs ~memory:6 ~linear:Jera.Linear.dense)
      ~steps
      (fun x -> Nx.neg (Nx.reshape [||] (lp x)))
      start
  in
  let theta l = Nx.reshape [| 3 |] (Nx.get [ l ] xs) in
  let grad x =
    Nx.reshape [| 3 |]
      (Rune.grad t (fun x -> Nx.sum (lp x)) (Nx.reshape [| 1; 3 |] x))
  in
  let dot a b = Nx.item [] (Nx.sum (Nx.mul a b)) in
  let outer u v =
    Nx.matmul (Nx.reshape [| 3; 1 |] u) (Nx.reshape [| 1; 3 |] v)
  in
  let update alpha (s, z) =
    let a = dot (Nx.mul z alpha) z and b = dot s z in
    let c = dot (Nx.div s alpha) s in
    (* [1 / α' = a / (b α) + z² / b - a s² / (b c α²)] *)
    Nx.recip
      (Nx.sub
         (Nx.add
            (Nx.recip (Nx.mul_s alpha (b /. a)))
            (Nx.div_s (Nx.square z) b))
         (Nx.div
            (Nx.mul_s (Nx.square s) a)
            (Nx.mul_s (Nx.square alpha) (b *. c))))
  in
  let pair l =
    ( Nx.sub (theta l) (theta (l - 1)),
      Nx.sub (grad (theta (l - 1))) (grad (theta l)) )
  in
  let pairs = List.init (steps - 1) (fun l -> pair (l + 1)) in
  List.mapi
    (fun l _ ->
      let taken = List.filteri (fun i _ -> i <= l) pairs in
      let alpha = List.fold_left update (Nx.ones Nx.float64 [| 3 |]) taken in
      let h =
        List.fold_left
          (fun h (s, z) ->
            let rho = 1. /. dot s z in
            let v = Nx.sub (Nx.eye Nx.float64 3) (Nx.mul_s (outer s z) rho) in
            Nx.add
              (Nx.matmul v (Nx.matmul h (Nx.transpose v)))
              (Nx.mul_s (outer s s) rho))
          (Nx.diag alpha) taken
      in
      let at = theta (l + 1) in
      (Nx.add at (Nx.matmul h (grad at)), h))
    pairs

(* [bound_near u lp f ~chains] checks each chain's [f.elbo], from 100 draws,
   against the mean of 200000 log importance ratios of its Gaussian, whose
   expectation is the bound: within 6 standard errors of the difference, a false
   alarm of 2e-9 per chain. *)
let bound_near u lp (f : (_, _, _) G.fit) ~chains =
  for i = 0 to chains - 1 do
    let n = 200000 in
    let w =
      G.importance u lp (Nx.Rng.key (100 + i)) ~n (chain u f.gaussian i)
    in
    let lw = w.log_weights in
    let m = Nx.item [] (Nx.mean lw) in
    let var = Nx.item [] (Nx.mean (Nx.square (Nx.sub_s lw m))) in
    let se = Float.sqrt ((var /. 100.) +. (var /. float_of_int n)) in
    less
      ~msg:(Printf.sprintf "chain %d" i)
      float_exact
      ~than:((6. *. se) +. 1e-12)
      (Float.abs (Nx.item [ i ] f.elbo -. m))
  done

let khats u lp k ~sets ~n g =
  let w = G.importance u lp k ~n:(sets * n) g in
  floats (Norn.Diag.psis (Nx.reshape [| n; sets |] w.log_weights)).khat

let median xs =
  let xs = Array.copy xs in
  Array.sort Float.compare xs;
  xs.(Array.length xs / 2)

let flat x = floats (Nx.reshape [| -1 |] x)

let gaussians =
  group "gaussian targets"
    [
      prop ~count:25 "a one-element Gaussian is recovered to rounding"
        Gen.(
          triple (float_range (-5.) 5.) (float_range 0.1 10.)
            (float_range 0.5 10.))
        (fun (m, s, away) ->
          (* After one step the secant pair gives [α = s / z = σ²] exactly, and
             the compact form adds nothing to a diagonal that meets the secant
             equation, so every later iterate's Gaussian is exact. *)
          let lp x =
            D.factors
              (D.normal ~loc:(f64 m) ~scale:(f64 s))
              (Nx.reshape [| -1 |] x)
          in
          let start = Nx.create Nx.float64 [| 1; 1 |] [| m +. away |] in
          let g = pf t ~steps:10 lp (Nx.Rng.key 1) start in
          let g = chain t g 0 in
          let close = float_rel ~rel:1e-8 ~abs:1e-8 in
          equal close m (Nx.item [ 0 ] (G.mean t g));
          equal close (s *. s) (Nx.item [ 0 ] (G.variance t g)));
      test "each path's Gaussian is near the target" (fun () ->
          (* The chosen iterate's covariance is L-BFGS's estimate, exact on no
             path. Of 10000 paths from these starts with 100 draws per bound,
             none had a mean or a covariance more than 0.1 from the target's: a
             path does with probability below 3e-4 (95% upper bound), so more
             than 4 of 8 do with probability below 1e-15. *)
          let g =
            pf pos ~elbo_draws:100 ~steps:30 gaussian (Nx.Rng.key 1)
              (Nx.Ptree.map pos (fun _ x -> Nx.tile [| 2; 1 |] x) starts)
          in
          let far i =
            let g = chain pos g i in
            let m = G.mean pos g in
            let off x = Array.exists (fun x -> Float.abs x > 0.1) x in
            off (floats (Nx.sub (Nx.concatenate ~axis:0 [ m.a; m.b ]) loc))
            || off (flat (Nx.sub (dense g) cov))
          in
          at_most int ~than:4
            (List.length (List.filter far (List.init 8 Fun.id))));
      test "a fit is the dense BFGS update at one of its iterates" (fun () ->
          (* The fit is the Gaussian of an iterate [θ_l], [l >= 1]: its
             covariance is BFGS's updates [H ← V H Vᵀ + ρ s sᵀ], [V = I - ρ s
             zᵀ], [ρ = 1 / sᵀz], over the pairs to [θ_l], oldest first, from
             [diag α_l], [α_l] Gilbert and Lemaréchal's update of [α_(l-1)] at
             pair [l] from the identity; its mean is [θ_l + H ∇lp (θ_l)]. With
             [steps = 2] it is [θ_1]'s, with [steps = 3] [θ_1]'s or [θ_2]'s. *)
          let lp x = D.factors (D.mvn ~loc ~scale_tril:tril) x in
          let start = Nx.create Nx.float64 [| 1; 3 |] [| 3.; 3.; 3. |] in
          List.iter
            (fun steps ->
              let f = G.pathfinder t ~steps lp (Nx.Rng.key 1) start in
              let g = chain t f.gaussian 0 in
              let off (mean, h) =
                Float.max
                  (largest (Nx.sub mean (G.mean t g)))
                  (largest (Nx.sub h (dense3 g)))
              in
              let nearest =
                List.fold_left Float.min Float.infinity
                  (List.map off (bfgs lp start ~steps))
              in
              less
                ~msg:(Printf.sprintf "steps = %d" steps)
                float_exact ~than:1e-10 nearest)
            [ 2; 3 ]);
      test "a fit's bound is its Gaussian's" (fun () ->
          (* The chosen iterate's estimates are the largest of many, so the
             fit's bound is estimated anew. *)
          let f = G.pathfinder pos ~steps:30 gaussian (Nx.Rng.key 3) starts in
          bound_near pos gaussian f ~chains:4);
      test "one step is a failed fit, centred at the start" (fun () ->
          let lp x = D.factors (D.mvn ~loc ~scale_tril:tril) x in
          let start = Nx.create Nx.float64 [| 1; 3 |] [| 3.; 3.; 3. |] in
          let f = G.pathfinder t ~steps:1 lp (Nx.Rng.key 1) start in
          let g = chain t f.gaussian 0 in
          equal (array float_exact) [| Float.neg_infinity |] (floats f.elbo);
          equal (array float_exact) [| 3.; 3.; 3. |] (floats (G.mean t g));
          equal (array float_exact) [| 1.; 1.; 1. |] (floats (G.variance t g)));
      test "each path's k̂ is below 0.5" (fun () ->
          (* Over 1000 sets of 1000 draws of each path's Gaussian, at most 2 k̂
             in 1000 reached 0.5: the median of 21 does with probability below
             1e-20. *)
          let g =
            pf pos ~elbo_draws:100 ~steps:30 gaussian (Nx.Rng.key 2) starts
          in
          for i = 0 to 3 do
            less float_exact ~than:0.5
              (median
                 (khats pos gaussian
                    (Nx.Rng.key (10 + i))
                    ~sets:21 ~n:1000 (chain pos g i)))
          done);
    ]

let tails =
  group "heavy tails"
    [
      test "a Cauchy target's k̂ is above 0.7" (fun () ->
          (* A Gaussian's draws weighted for five independent Cauchy elements
             have ratios with no finite moment. Over 1000 sets of 4000 draws 12%
             of the k̂ fall below 0.7; the median of 41 does with probability
             below 1e-9. *)
          let lp x =
            Nx.sum ~axes:[ 1 ]
              (D.factors (D.cauchy ~loc:(f64 0.) ~scale:(f64 1.)) x)
          in
          let start =
            Nx.create Nx.float64 [| 1; 5 |] [| 3.; -1.; 2.; 0.5; -2. |]
          in
          let g = pf t ~elbo_draws:100 ~steps:30 lp (Nx.Rng.key 1) start in
          greater float_exact ~than:0.7
            (median (khats t lp (Nx.Rng.key 3) ~sets:41 ~n:4000 (chain t g 0))));
    ]

let multipath =
  group "multi-path"
    [
      test "importance weighs each mode by its mass" (fun () ->
          (* 3/4 N(-4, 1) + 1/4 N(4, 1), one path to each mode: each path's
             Gaussian is its mode's component, so its draws' ratios are the
             component's weight, and the smoothed weights put 1/4 of the mass
             above 0. *)
          let lp x =
            let x = Nx.reshape [| -1 |] x in
            let one m = D.factors (D.normal ~loc:(f64 m) ~scale:(f64 1.)) x in
            Nx.logsumexp ~axes:[ 0 ]
              (Nx.stack ~axis:0
                 [
                   Nx.add_s (one (-4.)) (Float.log 0.75);
                   Nx.add_s (one 4.) (Float.log 0.25);
                 ])
          in
          let start = Nx.create Nx.float64 [| 2; 1 |] [| -3.; 3. |] in
          let g = pf t ~steps:20 lp (Nx.Rng.key 1) start in
          let w = G.importance t lp (Nx.Rng.key 2) ~n:2000 g in
          let s = Norn.Diag.psis w.log_weights in
          let p = Nx.exp (Nx.sub s.log_weights (Nx.logsumexp s.log_weights)) in
          let x = Nx.reshape [| -1 |] w.values in
          let above =
            Nx.sum (Nx.where (Nx.greater_s x 0.) p (Nx.zeros_like p))
          in
          equal (float 1e-3) 0.25 (Nx.item [] above);
          let draws =
            Norn.Weighted.resample t (Nx.Rng.key 3) ~n:100
              { w with log_weights = s.log_weights }
          in
          let count =
            Array.fold_left
              (fun n x -> if x > 0. then n + 1 else n)
              0 (flat draws)
          in
          (* Systematic resampling copies a share of 1/4 [floor] or [ceil] of 25
             times. *)
          at_least int ~than:24 count;
          at_most int ~than:26 count);
    ]

let keys =
  group "keys"
    [
      test "the Gaussians depend on the key alone" (fun () ->
          let run k = pf pos ~steps:30 gaussian k starts in
          let a = parts pos (run (Nx.Rng.key 4))
          and b = parts pos (run (Nx.Rng.key 4)) in
          List.iter2
            (fun (name, x) (_, y) ->
              equal ~msg:name (array float_exact) (flat x) (flat y))
            a b);
      test "a path does not depend on the other chains" (fun () ->
          let k = Nx.Rng.key 5 in
          let all = pf pos ~steps:30 gaussian k starts in
          let first =
            Nx.Ptree.map pos (fun _ x -> Nx.slice [ Nx.R (0, 1) ] x) starts
          in
          let alone = pf pos ~steps:30 gaussian k first in
          List.iter2
            (fun (name, x) (_, y) ->
              equal ~msg:name
                (array (float_rel ~rel:1e-12 ~abs:1e-12))
                (flat x) (flat y))
            (parts pos (chain pos all 0))
            (parts pos (chain pos alone 0)));
    ]

let errors =
  group "errors"
    [
      test "counts below 1 are refused" (fun () ->
          let run ?history ?elbo_draws steps () =
            ignore
              (pf pos ?history ?elbo_draws ~steps gaussian (Nx.Rng.key 1) starts)
          in
          raises
            (Invalid_argument
               "Norn.Gaussian.pathfinder: steps = 0 is not positive") (run 0);
          raises
            (Invalid_argument
               "Norn.Gaussian.pathfinder: history = 0 is not positive")
            (run ~history:0 5);
          raises
            (Invalid_argument
               "Norn.Gaussian.pathfinder: elbo_draws = 0 is not positive")
            (run ~elbo_draws:0 5));
      test "a start where the density is not finite is refused" (fun () ->
          let lp p =
            Nx.where
              (Nx.greater_s (Nx.reshape [| -1 |] p.a) 2.)
              (Nx.full Nx.float64 [| 4 |] Float.neg_infinity)
              (gaussian p)
          in
          raises
            (Invalid_argument
               "Norn.Gaussian.pathfinder: the density or its gradient is not \
                finite at chain 0 of the start") (fun () ->
              pf pos ~steps:5 lp (Nx.Rng.key 1) starts));
    ]

let () =
  exit
    (run "Norn.Gaussian.pathfinder"
       [ gaussians; tails; multipath; keys; errors ])
