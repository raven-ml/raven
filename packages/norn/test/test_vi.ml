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

let vi u ?rank ?draws ~steps lp k start =
  (G.vi u ?rank ?draws ~steps lp k start).gaussian

let floats x = Nx.to_array x
let flat x = floats (Nx.reshape [| -1 |] x)
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs

(* A position of 3 elements per chain: [a] of 1, [b] of 2. *)
let of_rows m =
  let c = (Nx.shape m).(0) in
  {
    a = Nx.slice [ Nx.R (0, c); Nx.R (0, 1) ] m;
    b = Nx.slice [ Nx.R (0, c); Nx.R (1, 3) ] m;
  }

let rows p = Nx.concatenate ~axis:1 [ p.a; p.b ]
let loc = vec [| 1.; -2.; 0.5 |]

let tril =
  Nx.create Nx.float64 [| 3; 3 |] [| 1.; 0.; 0.; 0.8; 0.6; 0.; -0.5; 0.3; 0.4 |]

let cov = Nx.matmul tril (Nx.transpose tril)
let gaussian p = D.factors (D.mvn ~loc ~scale_tril:tril) (rows p)

let starts =
  of_rows (Nx.create Nx.float64 [| 2; 3 |] [| 3.; 3.; 3.; -4.; 1.; 2. |])

let chain u g i = Nx.Ptree.map (G.ptree u) (fun _ x -> Nx.get [ i ] x) g

let parts u g =
  List.rev
    (Nx.Ptree.fold (G.ptree u)
       (fun p x acc -> (Nx.Ptree.Path.to_string p, Nx.cast Nx.float64 x) :: acc)
       g [])

let joined g prefix =
  let p = parts pos g in
  Nx.concatenate ~axis:(-1)
    [ List.assoc (prefix ^ ".a") p; List.assoc (prefix ^ ".b") p ]

(* The dense covariance of one Gaussian over [pos]. *)
let dense g =
  let s = joined g "scale" and dirs = joined g "directions" in
  let v = List.assoc "variances" (parts pos g) in
  let k = (Nx.shape v).(0) in
  let inner =
    Nx.add (Nx.eye Nx.float64 3)
      (Nx.matmul (Nx.transpose dirs)
         (Nx.mul (Nx.reshape [| k; 1 |] (Nx.sub_s v 1.)) dirs))
  in
  Nx.mul (Nx.mul (Nx.reshape [| 3; 1 |] s) inner) (Nx.reshape [| 1; 3 |] s)

(* The dense covariance of one Gaussian over a two-element tensor. *)
let dense2 g =
  let p = parts t g in
  let s = List.assoc "scale" p and dirs = List.assoc "directions" p in
  let v = List.assoc "variances" p in
  let k = (Nx.shape v).(0) in
  let inner =
    Nx.add (Nx.eye Nx.float64 2)
      (Nx.matmul (Nx.transpose dirs)
         (Nx.mul (Nx.reshape [| k; 1 |] (Nx.sub_s v 1.)) dirs))
  in
  Nx.mul (Nx.mul (Nx.reshape [| 2; 1 |] s) inner) (Nx.reshape [| 1; 2 |] s)

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
let tiled n p = Nx.Ptree.map pos (fun _ x -> Nx.tile [| n; 1 |] x) p
let point p = Nx.concatenate ~axis:0 [ p.a; p.b ]
let largest x = Array.fold_left (fun m x -> Float.max m (Float.abs x)) 0. x

(* [far g ~mean ~spread] is the number of [g]'s chains whose mean is more than
   [mean] from the target's or whose [spread g i] exceeds [spread]. *)
let far g ~chains ~mean ~spread:(spread, measure) =
  List.length
    (List.filter
       (fun i ->
         let g = chain pos g i in
         largest (floats (Nx.sub (point (G.mean pos g)) loc)) > mean
         || measure g > spread)
       (List.init chains Fun.id))

let fits =
  group "gaussian targets"
    [
      test "a full-rank fit is the target" (fun () ->
          (* The bound's maximum is the target itself; the last step keeps its
             gradient noise. Of 2000 chains from these starts, 4 had a mean more
             than 0.1 or a covariance more than 0.15 from the target's: a chain
             does with probability below 5e-3 (95% upper bound), so more than 5
             of 10 do with probability below 3e-12. *)
          let g =
            vi pos ~rank:3 ~draws:100 ~steps:2000 gaussian (Nx.Rng.key 1)
              (tiled 5 starts)
          in
          let covariance g = largest (flat (Nx.sub (dense g) cov)) in
          at_most int ~than:5
            (far g ~chains:10 ~mean:0.1 ~spread:(0.15, covariance)));
      test "a mean-field fit has the conditionals' variances" (fun () ->
          (* For a Gaussian of precision [Λ] the mean-field maximum has the
             target's mean and the variances [1 / Λ_ii] of its conditionals. Of
             2000 chains from these starts, 25 had a mean more than 0.1 off or a
             variance more than 30% off: a chain does with probability below
             0.018 (95% upper bound), so more than 5 of 10 do with probability
             below 1e-8. *)
          let g =
            vi pos ~draws:100 ~steps:2000 gaussian (Nx.Rng.key 1)
              (tiled 5 starts)
          in
          let conditional = Nx.recip (Nx.diagonal (Nx.inv cov)) in
          let variance g =
            largest
              (floats
                 (Nx.div
                    (Nx.sub (point (G.variance pos g)) conditional)
                    conditional))
          in
          at_most int ~than:5
            (far g ~chains:10 ~mean:0.1 ~spread:(0.3, variance)));
      test "a fit's bound is its Gaussian's" (fun () ->
          List.iter
            (fun rank ->
              let f =
                G.vi pos ~rank ~steps:200 gaussian (Nx.Rng.key 3) starts
              in
              bound_near pos gaussian f ~chains:2)
            [ 0; 3 ]);
      test "one step from a unit target's mean keeps its unit covariance"
        (fun () ->
          (* At the optimum Ong et al.'s gradients of [B] and [D] vanish draw by
             draw: a rank-0 fit's first step leaves [D = I] exactly, a rank-2
             fit's [B Bᵀ + D² = I] to rounding. *)
          let lp x =
            Nx.sum ~axes:[ 1 ]
              (D.factors
                 (D.normal ~loc:(Nx.scalar Nx.float64 0.)
                    ~scale:(Nx.scalar Nx.float64 1.))
                 x)
          in
          let zero = Nx.zeros Nx.float64 [| 1; 2 |] in
          let g = chain t (vi t ~steps:1 lp (Nx.Rng.key 1) zero) 0 in
          equal (array float_exact) [| 1.; 1. |] (floats (G.variance t g));
          let g = chain t (vi t ~rank:2 ~steps:1 lp (Nx.Rng.key 1) zero) 0 in
          equal (array (float 1e-12)) [| 1.; 0.; 0.; 1. |] (flat (dense2 g)));
    ]

let khat =
  group "k-hat"
    [
      test "a full-rank fit of a correlated target has k̂ below 0.5" (fun () ->
          (* Over 1000 sets of 1000 draws every k̂ was below 0.22. *)
          let g =
            vi t ~rank:2 ~draws:100 ~steps:2000 correlated (Nx.Rng.key 1) start2
          in
          less float_exact ~than:0.5
            (median
               (khats t correlated (Nx.Rng.key 2) ~sets:21 ~n:1000 (chain t g 0))));
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
            (median
               (khats t correlated (Nx.Rng.key 77) ~sets:41 ~n:4000
                  (chain t g 0))));
    ]

let reals =
  group "reals"
    [
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
          let m = Nx.item [ 0; 0 ] (G.mean t g) in
          satisfies ~claim:"finite" float_exact Float.is_finite m);
    ]

let holds =
  group "holds"
    [
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
          let f = G.vi t ~steps:50 lp (Nx.Rng.key 1) start in
          let g = chain t f.gaussian 0 in
          equal (array float_exact) [| 0.5; -0.5 |] (floats (G.mean t g));
          equal (array float_exact) [| 1.; 1. |] (floats (G.variance t g));
          equal (array float_exact) [| Float.neg_infinity |] (floats f.elbo));
    ]

let keys =
  group "keys"
    [
      test "the fit depends on the key alone" (fun () ->
          let run k = parts pos (vi pos ~rank:3 ~steps:100 gaussian k starts) in
          List.iter2
            (fun (name, x) (_, y) ->
              equal ~msg:name (array float_exact) (flat x) (flat y))
            (run (Nx.Rng.key 3))
            (run (Nx.Rng.key 3)));
      test "a fit does not depend on the other chains" (fun () ->
          let k = Nx.Rng.key 5 in
          let all = vi pos ~rank:2 ~steps:100 gaussian k starts in
          let first =
            Nx.Ptree.map pos (fun _ x -> Nx.slice [ Nx.R (0, 1) ] x) starts
          in
          let alone = vi pos ~rank:2 ~steps:100 gaussian k first in
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
          raises
            (Invalid_argument "Norn.Gaussian.vi: steps = 0 is not positive")
            (fun () -> vi pos ~steps:0 gaussian (Nx.Rng.key 1) starts);
          raises
            (Invalid_argument "Norn.Gaussian.vi: draws = 0 is not positive")
            (fun () -> vi pos ~draws:0 ~steps:5 gaussian (Nx.Rng.key 1) starts));
      test "a start where the density is not finite is refused" (fun () ->
          let lp p =
            Nx.where
              (Nx.greater_s (Nx.reshape [| -1 |] p.a) 2.)
              (Nx.full Nx.float64 [| 2 |] Float.neg_infinity)
              (gaussian p)
          in
          raises
            (Invalid_argument
               "Norn.Gaussian.vi: the density or its gradient is not finite at \
                chain 0 of the start") (fun () ->
              vi pos ~steps:5 lp (Nx.Rng.key 1) starts));
    ]

let () =
  exit (run "Norn.Gaussian.vi" [ fits; khat; reals; holds; keys; errors ])
