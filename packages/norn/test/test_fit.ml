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
let f64 = Nx.scalar Nx.float64
let floats x = Nx.to_array x
let flat x = floats (Nx.reshape [| -1 |] x)
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let largest xs = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. xs

(* A position of 3 elements per chain: [a] of 1, [b] of 2. *)
let of_rows m =
  let c = (Nx.shape m).(0) in
  {
    a = Nx.slice [ Nx.R (0, c); Nx.R (0, 1) ] m;
    b = Nx.slice [ Nx.R (0, c); Nx.R (1, 3) ] m;
  }

let rows p = Nx.concatenate ~axis:1 [ p.a; p.b ]
let point p = Nx.concatenate ~axis:0 [ p.a; p.b ]

(* A correlated Gaussian of three elements, the target of most tests. *)
let loc = vec [| 1.; -2.; 0.5 |]

let tril =
  Nx.create Nx.float64 [| 3; 3 |] [| 1.; 0.; 0.; 0.8; 0.6; 0.; -0.5; 0.3; 0.4 |]

let cov = Nx.matmul tril (Nx.transpose tril)
let gaussian p = D.factors (D.mvn ~loc ~scale_tril:tril) (rows p)
let target x = D.factors (D.mvn ~loc ~scale_tril:tril) x

let starts =
  of_rows
    (Nx.create Nx.float64 [| 4; 3 |]
       [| 3.; 3.; 3.; -4.; 1.; 2.; 0.; 0.; 0.; 1.5; -1.; -2. |])

(* [first n p] is the first [n] chains of [p]. *)
let first n p = Nx.Ptree.map pos (fun _ x -> Nx.slice [ Nx.R (0, n) ] x) p
let tiled n p = Nx.Ptree.map pos (fun _ x -> Nx.tile [| n; 1 |] x) p

(* [chain u g i] is chain [i]'s Gaussian of the Gaussians [g] stacked on the
   chain axis. *)
let chain u g i = Nx.Ptree.map (G.ptree u) (fun _ x -> Nx.get [ i ] x) g

(* [parts u g] is [g]'s mean, scale, directions and variances, by path. *)
let parts u g =
  List.rev
    (Nx.Ptree.fold (G.ptree u)
       (fun p x acc -> (Nx.Ptree.Path.to_string p, Nx.cast Nx.float64 x) :: acc)
       g [])

let same_parts ?(close = float_exact) u g g' =
  List.iter2
    (fun (name, x) (_, y) -> equal ~msg:name (array close) (flat x) (flat y))
    (parts u g) (parts u g')

(* [covariance u g] is the dense covariance of one Gaussian, [S (I + Σ (v - 1) u
   uᵀ) S] with its leaves' elements taken together in walk order. *)
let covariance u g =
  let p = parts u g in
  let joined field =
    Nx.concatenate ~axis:(-1)
      (List.filter_map
         (fun (path, x) ->
           if path = field || String.starts_with ~prefix:(field ^ ".") path then
             Some x
           else None)
         p)
  in
  let s = joined "scale" and dirs = joined "directions" in
  let v = List.assoc "variances" p in
  let d = (Nx.shape s).(0) and k = (Nx.shape v).(0) in
  let inner =
    Nx.add (Nx.eye Nx.float64 d)
      (Nx.matmul (Nx.transpose dirs)
         (Nx.mul (Nx.reshape [| k; 1 |] (Nx.sub_s v 1.)) dirs))
  in
  Nx.mul (Nx.mul (Nx.reshape [| d; 1 |] s) inner) (Nx.reshape [| 1; d |] s)

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

(* [khat_median u lp k ~sets ~n g] is the median k̂ of [sets] sets of [n]
   importance draws of [g]. *)
let khat_median u lp k ~sets ~n g =
  let w = G.importance u lp k ~n:(sets * n) g in
  let khats =
    floats (Norn.Diag.psis (Nx.reshape [| n; sets |] w.log_weights)).khat
  in
  Array.sort Float.compare khats;
  khats.(sets / 2)

(* A failed fit is centred at its start with unit covariance, and its bound is
   [-inf]. *)
let failed (f : (_, _, _) G.fit) i ~at =
  let g = chain t f.gaussian i in
  equal float_exact ~msg:"the bound" Float.neg_infinity (Nx.item [ i ] f.elbo);
  equal (array float_exact) ~msg:"the mean" at (floats (G.mean t g));
  equal (array float_exact) ~msg:"the variance"
    (Array.make (Array.length at) 1.)
    (floats (G.variance t g))

(* Refusals shared by both fits. *)

let not_finite_rows =
  [
    ( "a density of -inf",
      fun x ->
        Nx.where
          (Nx.greater_s (Nx.slice [ Nx.A; Nx.I 0 ] x) 2.)
          (Nx.full Nx.float64 [| (Nx.shape x).(0) |] Float.neg_infinity)
          (target x) );
    (* [-sqrt |x|]'s gradient is infinite at 0, the first chain's first
       element. *)
    ( "a gradient of inf",
      fun x ->
        Nx.add (target x)
          (Nx.neg
             (Nx.sqrt (Nx.abs (Nx.slice [ Nx.A; Nx.I 0 ] (Nx.sub_s x 3.))))) );
  ]

let not_per_chain_rows =
  [
    ("one value for all chains", fun x -> Nx.sum (target x));
    ( "centred on the chains' mean",
      fun x -> target (Nx.add_s (Nx.sub x (Nx.mean ~axes:[ 0 ] x)) 1.) );
    ( "normalised by the chains' largest",
      fun x ->
        let l = target x in
        Nx.sub l (Nx.max l) );
    ( "centred on the chains' median",
      fun x -> target (Nx.add_s (Nx.sub x (Nx.quantile ~axis:0 [| 0.5 |] x)) 1.)
    );
  ]

(* Two dtypes: a float32 tensor beside a float64 one. *)
let mixed_dtypes = Nx.Ptree.(pair tensor tensor)

let mixed_lp (a, b) =
  Nx.add
    (Nx.sum ~axes:[ 1 ] (Nx.square (Nx.cast Nx.float64 a)))
    (Nx.sum ~axes:[ 1 ] (Nx.square b))

let mixed_start = (Nx.ones Nx.float32 [| 2; 1 |], Nx.ones Nx.float64 [| 2; 2 |])

(* A position of an integer tensor alone. *)
let ints = Nx.ones Nx.int32 [| 2; 3 |]
let no_floats x = Nx.zeros Nx.float64 [| (Nx.shape x).(0) |]
let start3 = Nx.create Nx.float64 [| 1; 3 |] [| 3.; 3.; 3. |]

(* Pathfinder *)

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

let pf u ?history ?elbo_draws ~steps lp k start =
  (G.pathfinder u ?history ?elbo_draws ~steps lp k start).gaussian

let pathfinder_tests =
  group "pathfinder"
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
          let g = chain t (pf t ~steps:10 lp (Nx.Rng.key 1) start) 0 in
          let close = float_rel ~rel:1e-8 ~abs:1e-8 in
          equal close m (Nx.item [ 0 ] (G.mean t g));
          equal close (s *. s) (Nx.item [ 0 ] (G.variance t g)));
      test "a fit is the dense BFGS update at one of its iterates" (fun () ->
          (* The fit is the Gaussian of an iterate [θ_l], [l >= 1]: its
             covariance is BFGS's updates [H ← V H Vᵀ + ρ s sᵀ], [V = I - ρ s
             zᵀ], [ρ = 1 / sᵀz], over the pairs to [θ_l], oldest first, from
             [diag α_l], [α_l] Gilbert and Lemaréchal's update of [α_(l-1)] at
             pair [l] from the identity; its mean is [θ_l + H ∇lp (θ_l)]. With
             [steps = 2] it is [θ_1]'s, with [steps = 3] [θ_1]'s or [θ_2]'s. *)
          List.iter
            (fun steps ->
              let g = chain t (pf t ~steps target (Nx.Rng.key 1) start3) 0 in
              let off (mean, h) =
                Float.max
                  (largest (flat (Nx.sub mean (G.mean t g))))
                  (largest (flat (Nx.sub h (covariance t g))))
              in
              let nearest =
                List.fold_left Float.min Float.infinity
                  (List.map off (bfgs target start3 ~steps))
              in
              less
                ~msg:(Printf.sprintf "steps = %d" steps)
                float_exact ~than:1e-10 nearest)
            [ 2; 3 ]);
      test "each path's Gaussian is near the target" (fun () ->
          (* The chosen iterate's covariance is L-BFGS's estimate, exact on no
             path. Of 10000 paths from these starts with 100 draws per bound,
             none had a mean or a covariance more than 0.1 from the target's: a
             path does with probability below 3e-4 (95% upper bound), so more
             than 4 of 8 do with probability below 1e-15. *)
          let g =
            pf pos ~elbo_draws:100 ~steps:30 gaussian (Nx.Rng.key 1)
              (tiled 2 starts)
          in
          let far i =
            let g = chain pos g i in
            largest (floats (Nx.sub (point (G.mean pos g)) loc)) > 0.1
            || largest (flat (Nx.sub (covariance pos g) cov)) > 0.1
          in
          at_most int ~than:4
            (List.length (List.filter far (List.init 8 Fun.id))));
      test "a fit's bound is its Gaussian's, from draws of its own" (fun () ->
          (* The chosen iterate's estimates are the largest of many, so the
             fit's bound is estimated anew. *)
          let f = G.pathfinder pos ~steps:30 gaussian (Nx.Rng.key 3) starts in
          bound_near pos gaussian f ~chains:4);
      test "the defaults are a history of 6 and 5 draws per bound" (fun () ->
          (* Ten steps make nine pairs, more than a history of 6 holds. *)
          let run ?history ?elbo_draws () =
            G.pathfinder pos ?history ?elbo_draws ~steps:10 gaussian
              (Nx.Rng.key 4) starts
          in
          let f = run () and f' = run ~history:6 ~elbo_draws:5 () in
          same_parts pos f'.gaussian f.gaussian;
          equal (array float_exact) ~msg:"the bounds" (floats f'.elbo)
            (floats f.elbo));
      test "one step is a failed fit, centred at the start" (fun () ->
          failed
            (G.pathfinder t ~steps:1 target (Nx.Rng.key 1) start3)
            0 ~at:[| 3.; 3.; 3. |]);
      test "a path that does not move is a failed fit" (fun () ->
          (* At the target's mode the gradient is 0, and L-BFGS stays. *)
          let start =
            Nx.concatenate ~axis:0 [ Nx.reshape [| 1; 3 |] loc; start3 ]
          in
          let f = G.pathfinder t ~steps:10 target (Nx.Rng.key 1) start in
          failed f 0 ~at:(floats loc);
          greater float_exact ~msg:"the moving chain's bound"
            ~than:Float.neg_infinity (Nx.item [ 1 ] f.elbo));
      test "each path's k̂ is below 0.5" (fun () ->
          (* Over 1000 sets of 1000 draws of each path's Gaussian, at most 2 k̂
             in 1000 reached 0.5: the median of 21 does with probability below
             1e-20. *)
          let g =
            pf pos ~elbo_draws:100 ~steps:30 gaussian (Nx.Rng.key 2) starts
          in
          for i = 0 to 3 do
            less float_exact ~than:0.5
              (khat_median pos gaussian
                 (Nx.Rng.key (10 + i))
                 ~sets:21 ~n:1000 (chain pos g i))
          done);
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
            (khat_median t lp (Nx.Rng.key 3) ~sets:41 ~n:4000 (chain t g 0)));
      test "importance weighs each mode by its mass" (fun () ->
          (* 3/4 N(-4, 1) + 1/4 N(4, 1), one path to each mode: each path's
             Gaussian is its mode's component, so a draw's ratio is its
             component's weight, but for the other component's tail, and the
             smoothed weights put 1/4 of the mass above 0. Over keys 1001 to
             1400 none put it more than 1e-3 away (the largest 9.0e-4): a key
             does with probability below 0.75% (95% upper bound). *)
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
      test "the Gaussians depend on the key alone" (fun () ->
          let run () = pf pos ~steps:30 gaussian (Nx.Rng.key 4) starts in
          same_parts pos (run ()) (run ()));
      test "a path does not depend on the other chains" (fun () ->
          let k = Nx.Rng.key 5 in
          same_parts
            ~close:(float_rel ~rel:1e-12 ~abs:1e-12)
            pos
            (chain pos (pf pos ~steps:30 gaussian k starts) 0)
            (chain pos (pf pos ~steps:30 gaussian k (first 1 starts)) 0));
      cases "a count below 1 is refused" ~id:fst
        [
          ("steps", fun () -> pf pos ~steps:0 gaussian (Nx.Rng.key 1) starts);
          ( "history",
            fun () -> pf pos ~history:0 ~steps:5 gaussian (Nx.Rng.key 1) starts
          );
          ( "elbo_draws",
            fun () ->
              pf pos ~elbo_draws:0 ~steps:5 gaussian (Nx.Rng.key 1) starts );
        ]
        (fun (_, run) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.pathfinder") run);
      cases "a start where the density or its gradient is not finite is refused"
        ~id:fst not_finite_rows (fun (_, lp) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.pathfinder")
            (fun () -> pf t ~steps:5 lp (Nx.Rng.key 1) (rows starts)));
      cases
        "a density is refused that does not return one log density per chain"
        ~id:fst not_per_chain_rows (fun (_, lp) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.pathfinder")
            (fun () -> pf t ~steps:5 lp (Nx.Rng.key 1) (rows starts)));
      test "a start with no float element is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.pathfinder")
            (fun () -> G.pathfinder t ~steps:5 no_floats (Nx.Rng.key 1) ints));
      test "a start of float tensors of two dtypes is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.pathfinder")
            (fun () ->
              G.pathfinder mixed_dtypes ~steps:5 mixed_lp (Nx.Rng.key 1)
                mixed_start));
    ]

(* VI *)

let vi u ?rank ?draws ~steps lp k start =
  (G.vi u ?rank ?draws ~steps lp k start).gaussian

(* A two-element Gaussian of correlation 0.95. *)
let rho = 0.95

let correlated x =
  D.factors
    (D.mvn
       ~loc:(Nx.zeros Nx.float64 [| 2 |])
       ~scale_tril:
         (Nx.create Nx.float64 [| 2; 2 |]
            [| 1.; 0.; rho; Float.sqrt (1. -. (rho *. rho)) |]))
    x

let start2 = Nx.create Nx.float64 [| 1; 2 |] [| 1.; -1. |]

(* [far g ~chains off] is the number of [g]'s first [chains] chains whose mean
   is more than 0.1 from the target's or whose Gaussian is [off]. *)
let far g ~chains off =
  List.length
    (List.filter
       (fun i ->
         let g = chain pos g i in
         largest (floats (Nx.sub (point (G.mean pos g)) loc)) > 0.1 || off g)
       (List.init chains Fun.id))

let vi_tests =
  group "vi"
    [
      test "a full-rank fit is the target" (fun () ->
          (* The bound's maximum is the target itself; the last step keeps its
             gradient noise. Of 2000 chains from these starts, 4 had a mean more
             than 0.1 or a covariance more than 0.15 from the target's: a chain
             does with probability below 5e-3 (95% upper bound), so more than 5
             of 10 do with probability below 3e-12. *)
          let g =
            vi pos ~rank:3 ~draws:100 ~steps:2000 gaussian (Nx.Rng.key 1)
              (tiled 5 (first 2 starts))
          in
          at_most int ~than:5
            (far g ~chains:10 (fun g ->
                 largest (flat (Nx.sub (covariance pos g) cov)) > 0.15)));
      test "a mean-field fit has the conditionals' variances" (fun () ->
          (* For a Gaussian of precision [Λ] the mean-field maximum has the
             target's mean and the variances [1 / Λ_ii] of its conditionals. Of
             2000 chains from these starts, 25 had a mean more than 0.1 off or a
             variance more than 30% off: a chain does with probability below
             0.018 (95% upper bound), so more than 5 of 10 do with probability
             below 1e-8. *)
          let g =
            vi pos ~draws:100 ~steps:2000 gaussian (Nx.Rng.key 1)
              (tiled 5 (first 2 starts))
          in
          let conditional = Nx.recip (Nx.diagonal (Nx.inv cov)) in
          at_most int ~than:5
            (far g ~chains:10 (fun g ->
                 largest
                   (floats
                      (Nx.div
                         (Nx.sub (point (G.variance pos g)) conditional)
                         conditional))
                 > 0.3)));
      cases "a fit's bound is its Gaussian's, from draws of its own"
        ~id:(Printf.sprintf "rank %d") [ 0; 3 ] (fun rank ->
          let f =
            G.vi pos ~rank ~steps:200 gaussian (Nx.Rng.key 3) (first 2 starts)
          in
          bound_near pos gaussian f ~chains:2);
      test "the defaults are mean field and one draw per gradient" (fun () ->
          let run ?rank ?draws () =
            G.vi pos ?rank ?draws ~steps:10 gaussian (Nx.Rng.key 4) starts
          in
          let f = run () and f' = run ~rank:0 ~draws:1 () in
          same_parts pos f'.gaussian f.gaussian;
          equal (array float_exact) ~msg:"the bounds" (floats f'.elbo)
            (floats f.elbo));
      test "a rank above the elements is the full rank" (fun () ->
          let run rank =
            vi pos ~rank ~steps:20 gaussian (Nx.Rng.key 4) (first 2 starts)
          in
          same_parts pos (run 3) (run 5));
      test "one step from a unit target's mean keeps its unit covariance"
        (fun () ->
          (* At the optimum Ong et al.'s gradients of [B] and [D] vanish draw by
             draw: a rank-0 fit's first step leaves [D = I] exactly, a rank-2
             fit's [B Bᵀ + D² = I] to rounding. *)
          let lp x =
            Nx.sum ~axes:[ 1 ]
              (D.factors (D.normal ~loc:(f64 0.) ~scale:(f64 1.)) x)
          in
          let zero = Nx.zeros Nx.float64 [| 1; 2 |] in
          let g = chain t (vi t ~steps:1 lp (Nx.Rng.key 1) zero) 0 in
          equal (array float_exact) [| 1.; 1. |] (floats (G.variance t g));
          let g = chain t (vi t ~rank:2 ~steps:1 lp (Nx.Rng.key 1) zero) 0 in
          equal
            (array (float 1e-12))
            [| 1.; 0.; 0.; 1. |]
            (flat (covariance t g)));
      test "a full-rank fit of a correlated target has k̂ below 0.5" (fun () ->
          (* Over 1000 sets of 1000 draws every k̂ was below 0.22. *)
          let g =
            vi t ~rank:2 ~draws:100 ~steps:2000 correlated (Nx.Rng.key 1) start2
          in
          less float_exact ~than:0.5
            (khat_median t correlated (Nx.Rng.key 2) ~sets:21 ~n:1000
               (chain t g 0)));
      test "a mean-field fit of a correlated target has k̂ above 0.7" (fun () ->
          (* Along the correlation the fit's variance is [1 - ρ²] against the
             target's [1 + ρ], so the ratios' tail has shape [k = ρ = 0.95]
             (Yao, Vehtari, Simpson and Gelman 2018). Over 1000 sets of 4000
             draws 12% of the k̂ fell below 0.7; the median of 41 does with
             probability below 1e-8. *)
          let g =
            vi t ~draws:100 ~steps:2000 correlated (Nx.Rng.key 1) start2
          in
          greater float_exact ~than:0.7
            (khat_median t correlated (Nx.Rng.key 77) ~sets:41 ~n:4000
               (chain t g 0)));
      test "a density is read at finite positions alone" (fun () ->
          (* [-cosh x] is steep far from 0: the search's largest step sizes
             throw the draws past the floats' range, where the density and its
             gradient are not finite. A model's density raises there, as this
             one does. *)
          let lp x =
            Nx.check Nx.Ptree.unit
              (Nx.all (Nx.isfinite x))
              ()
              (fun _ () -> Invalid_argument "read outside the reals");
            Nx.neg (Nx.sum ~axes:[ 1 ] (Nx.add (Nx.exp x) (Nx.exp (Nx.neg x))))
          in
          let g =
            vi t ~steps:200 lp (Nx.Rng.key 1)
              (Nx.create Nx.float64 [| 1; 1 |] [| 3. |])
          in
          satisfies ~claim:"finite" float_exact Float.is_finite
            (Nx.item [ 0; 0 ] (G.mean t g)));
      test "a chain whose bound is not finite stays where it was" (fun () ->
          (* The density is finite at the start alone: every draw's log density
             is [-inf], and the gradient is the entropy's, which would widen the
             Gaussian at every step. The fit stays at the start and says it
             failed. *)
          let start = Nx.create Nx.float64 [| 1; 2 |] [| 0.5; -0.5 |] in
          let lp x =
            Nx.where
              (Nx.all ~axes:[ 1 ]
                 (Nx.equal x (Nx.broadcast_to (Nx.shape x) start)))
              (Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.square x)) (-0.5))
              (Nx.full Nx.float64 [| (Nx.shape x).(0) |] Float.neg_infinity)
          in
          failed
            (G.vi t ~steps:50 lp (Nx.Rng.key 1) start)
            0 ~at:[| 0.5; -0.5 |]);
      test "the fit depends on the key alone" (fun () ->
          let run () =
            vi pos ~rank:3 ~steps:100 gaussian (Nx.Rng.key 3) (first 2 starts)
          in
          same_parts pos (run ()) (run ()));
      test "a fit does not depend on the other chains" (fun () ->
          let k = Nx.Rng.key 5 in
          same_parts
            ~close:(float_rel ~rel:1e-12 ~abs:1e-12)
            pos
            (chain pos
               (vi pos ~rank:2 ~steps:100 gaussian k (first 2 starts))
               0)
            (chain pos
               (vi pos ~rank:2 ~steps:100 gaussian k (first 1 starts))
               0));
      cases "a count below 1 or a negative rank is refused" ~id:fst
        [
          ("steps", fun () -> vi pos ~steps:0 gaussian (Nx.Rng.key 1) starts);
          ( "draws",
            fun () -> vi pos ~draws:0 ~steps:5 gaussian (Nx.Rng.key 1) starts );
          ( "rank",
            fun () -> vi pos ~rank:(-1) ~steps:5 gaussian (Nx.Rng.key 1) starts
          );
        ]
        (fun (_, run) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.vi") run);
      cases "a start where the density or its gradient is not finite is refused"
        ~id:fst not_finite_rows (fun (_, lp) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.vi") (fun () ->
              vi t ~steps:5 lp (Nx.Rng.key 1) (rows starts)));
      cases
        "a density is refused that does not return one log density per chain"
        ~id:fst not_per_chain_rows (fun (_, lp) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.vi") (fun () ->
              vi t ~steps:5 lp (Nx.Rng.key 1) (rows starts)));
      test "a start with no float element is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.vi") (fun () ->
              G.vi t ~steps:5 no_floats (Nx.Rng.key 1) ints));
      test "a start of float tensors of two dtypes is refused" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.vi") (fun () ->
              G.vi mixed_dtypes ~steps:5 mixed_lp (Nx.Rng.key 1) mixed_start));
    ]

let () = exit (run "Norn.Gaussian fits" [ pathfinder_tests; vi_tests ])
