(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module G = Norn.Gaussian

type 'a pos = { a : 'a; b : 'a }

module Pos = struct
  type ('a, _) t = 'a pos

  let walk c { a; b } =
    let open Nx.Ptree.Walk in
    let a = field c "a" leaf a in
    let b = field c "b" leaf b in
    { a; b }
end

let pos : (Nx.host Nx.float64_t pos, Nx.host) Nx.Ptree.t = Nx.Ptree.instantiate (module Pos)
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let floats x = Nx.to_array x
let close = float_rel ~rel:1e-10 ~abs:1e-12

(* A position of 4 elements: [a] of 1, [b] of 3. *)
let of_vec v =
  { a = Nx.slice [ Nx.R (0, 1) ] v; b = Nx.slice [ Nx.R (1, 4) ] v }

let to_vec p = Nx.concatenate ~axis:0 [ p.a; p.b ]
let mean = of_vec (vec [| 0.5; -1.; 2.; 0.25 |])
let scale = of_vec (vec [| 2.; 0.5; 1.; 3. |])

(* Two orthonormal directions in 4 elements, and their variances. *)
let u1 = vec [| 0.5; 0.5; 0.5; 0.5 |]
let u2 = vec [| 0.5; -0.5; 0.5; -0.5 |]

let stacked us =
  let m = Nx.stack ~axis:0 us in
  {
    a = Nx.slice [ Nx.R (0, List.length us); Nx.R (0, 1) ] m;
    b = Nx.slice [ Nx.R (0, List.length us); Nx.R (1, 4) ] m;
  }

let variances = vec [| 4.; 0.25 |]
let g = G.low_rank pos ~mean ~scale ~directions:(stacked [ u1; u2 ]) ~variances

(* The dense covariance [S (I + Σ (v - 1) u uᵀ) S]. *)
let dense =
  let s = Nx.diag (to_vec scale) in
  let inner =
    List.fold_left2
      (fun acc u v ->
        Nx.add acc
          (Nx.mul_s
             (Nx.matmul
                (Nx.unsqueeze ~axes:[ 1 ] u)
                (Nx.unsqueeze ~axes:[ 0 ] u))
             (v -. 1.)))
      (Nx.eye Nx.float64 4) [ u1; u2 ] [ 4.; 0.25 ]
  in
  Nx.matmul s (Nx.matmul inner s)

let gaussian_log_density x =
  let d = Nx.sub x (to_vec mean) in
  let _, logdet = Nx.slogdet Nx.float64 dense in
  let q = Nx.item [] (Nx.vdot d (Nx.solve dense d)) in
  (-0.5 *. q) -. (0.5 *. Nx.item [] logdet) -. (2. *. Float.log (2. *. Float.pi))

let points =
  Gen.(
    with_pp
      (fun ppf xs ->
        Format.fprintf ppf "[%s]"
          (String.concat "; " (Array.to_list (Array.map string_of_float xs))))
      (array ~size:(constant 4) (float_range (-5.) 5.)))

let laws =
  group "laws"
    [
      prop "the log density is the dense Gaussian's" points (fun x ->
          let x = vec x in
          let row = Nx.unsqueeze ~axes:[ 0 ] x in
          let lp =
            G.log_density pos g
              {
                a = Nx.slice [ Nx.R (0, 1); Nx.R (0, 1) ] row;
                b = Nx.slice [ Nx.R (0, 1); Nx.R (1, 4) ] row;
              }
          in
          equal close (gaussian_log_density x) (Nx.item [ 0 ] lp));
      test "the variance is the dense covariance's diagonal" (fun () ->
          equal (array close)
            (floats (Nx.diagonal dense))
            (floats (to_vec (G.variance pos g))));
      test "directions are orthonormalised" (fun () ->
          let g' =
            G.low_rank pos ~mean ~scale
              ~directions:
                (stacked [ Nx.mul_s u1 3.; Nx.add u2 (Nx.mul_s u1 0.7) ])
              ~variances
          in
          equal (array close)
            (floats (to_vec (G.variance pos g)))
            (floats (to_vec (G.variance pos g'))));
      test "a precision's Gaussian has its inverse as covariance" (fun () ->
          let p = of_vec (vec [| 1.; 4.; 0.5; 2. |]) in
          let w = stacked [ vec [| 1.; 0.; 1.; 0. |] ] in
          let l = vec [| 3. |] in
          let g = G.of_precision pos Nx.float64 ~low_rank:(w, l) ~mean p in
          let precision =
            Nx.add
              (Nx.diag (to_vec p))
              (Nx.mul_s
                 (Nx.matmul
                    (Nx.reshape [| 4; 1 |] (vec [| 1.; 0.; 1.; 0. |]))
                    (Nx.reshape [| 1; 4 |] (vec [| 1.; 0.; 1.; 0. |])))
                 3.)
          in
          let cov = Nx.inv precision in
          equal (array close)
            (floats (Nx.diagonal cov))
            (floats (to_vec (G.variance pos g))));
      slow
      @@ test "draws have the Gaussian's moments" (fun () ->
          let n = 200000 in
          let x = G.sample pos (Nx.Rng.key 4) ~n g in
          let x = Nx.concatenate ~axis:1 [ x.a; x.b ] in
          let m = Nx.mean ~axes:[ 0 ] x in
          (* Each mean is within six standard errors, a false alarm of 2e-9. *)
          let se = Nx.sqrt (Nx.div_s (Nx.diagonal dense) (float_of_int n)) in
          Array.iteri
            (fun i d ->
              less (float 1e-12) ~than:(6. *. (floats se).(i)) (Float.abs d))
            (floats (Nx.sub m (to_vec mean)));
          let c =
            Nx.div_s
              (Nx.matmul (Nx.transpose (Nx.sub x m)) (Nx.sub x m))
              (float_of_int n)
          in
          equal
            (array (float 0.1))
            (floats (Nx.reshape [| 16 |] dense))
            (floats (Nx.reshape [| 16 |] c)));
    ]

let errors =
  group "errors"
    [
      test "a mean of no tensor is refused" (fun () ->
          raises
            (Invalid_argument "Norn.Gaussian.diagonal: the mean holds no tensor")
            (fun () -> G.diagonal Nx.Ptree.unit Nx.float64 ~mean:() ~scale:());
          raises
            (Invalid_argument
               "Norn.Gaussian.of_precision: the mean holds no tensor")
            (fun () -> G.of_precision Nx.Ptree.unit Nx.float64 ~mean:() ()));
      test "a precision that is not positive names its path" (fun () ->
          raises
            (Invalid_argument
               "Norn.Gaussian.of_precision: b at [1] is -1, not in (0, inf)")
            (fun () ->
              G.of_precision pos Nx.float64 ~mean
                (of_vec (vec [| 1.; 1.; -1.; 1. |]))));
      test "a variance that is not positive raises" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"variances at [1] is 0") (fun () ->
              G.low_rank pos ~mean ~scale
                ~directions:(stacked [ u1; u2 ])
                ~variances:(vec [| 1.; 0. |])));
      test "a diagonal Gaussian has rank 0" (fun () ->
          equal string "gaussian(dimension 4, rank 0)"
            (Format.asprintf "%a" (G.pp pos)
               (G.diagonal pos Nx.float64 ~mean ~scale)));
    ]

(* Importance *)

let rows p = Nx.concatenate ~axis:1 [ p.a; p.b ]
let log_weights (w : (_, _, _) Norn.Weighted.t) = floats w.log_weights

(* Two diagonal Gaussians stacked on a leading axis of two. *)
let pair_of g g' =
  Nx.Ptree.map2 (G.ptree pos) (fun _ x y -> Nx.stack ~axis:0 [ x; y ]) g g'

let wide =
  G.diagonal pos Nx.float64 ~mean ~scale:(of_vec (vec [| 4.; 2.; 3.; 6. |]))

let narrow =
  G.diagonal pos Nx.float64 ~mean ~scale:(of_vec (vec [| 1.; 1.; 1.; 1. |]))

(* [khats w ~sets] is the [k̂] of each of [sets] equal parts of the draws [w]:
   their draws are independent, so each part is a set of its own. *)
let khats (w : (_, _, _) Norn.Weighted.t) ~sets =
  let n = (Nx.shape w.log_weights).(0) / sets in
  floats (Norn.Diag.psis (Nx.reshape [| n; sets |] w.log_weights)).khat

let median xs =
  let xs = Array.copy xs in
  Array.sort Float.compare xs;
  xs.(Array.length xs / 2)

let std_normal x =
  Nx.sum ~axes:[ 1 ]
    (Norn.Dist.factors
       (Norn.Dist.normal ~loc:(Nx.scalar Nx.float64 0.)
          ~scale:(Nx.scalar Nx.float64 1.))
       x)

let one_d m s =
  G.diagonal Nx.Ptree.tensor Nx.float64 ~mean:(vec [| m |]) ~scale:(vec [| s |])

let importance =
  group "importance"
    [
      prop "a Gaussian weighted for its own density has log weights of zero"
        Gen.(pair (int_range 0 1000) (int_range 1 50))
        (fun (key, n) ->
          let w =
            G.importance pos (G.log_density pos g) (Nx.Rng.key key) ~n g
          in
          equal (array (float 1e-12)) (Array.make n 0.) (log_weights w));
      test "stacked Gaussians each draw n, weighted by the one that drew them"
        (fun () ->
          let n = 6 in
          let w =
            G.importance pos (G.log_density pos narrow) (Nx.Rng.key 3) ~n
              (pair_of narrow wide)
          in
          let x = rows w.values in
          let theirs =
            Nx.sub
              (G.log_density pos narrow w.values)
              (G.log_density pos wide w.values)
          in
          let expected =
            Array.init (2 * n) (fun i ->
                if i < n then 0. else (floats theirs).(i))
          in
          equal (array int) [| 2 * n; 4 |] (Nx.shape x);
          equal (array (float 1e-12)) expected (log_weights w));
      test "weighted means of a wider proposal match the target's" (fun () ->
          (* A proposal wider than the target in every direction bounds the
             ratios, so k̂ is below 0.5 and the smoothed weights' means are
             normal about the target's with the summary's standard error. Six
             errors give a false alarm of 2e-9 per element. *)
          let w =
            G.importance pos (G.log_density pos g) (Nx.Rng.key 5) ~n:4000 wide
          in
          less float_exact ~than:0.5
            (Nx.item [] (Norn.Diag.psis w.log_weights).khat);
          let s = Norn.Summary.of_weighted pos w in
          let column name = floats (List.assoc name (Norn.Summary.columns s)) in
          let m = column "mean" and se = column "mcse" in
          Array.iteri
            (fun i exact ->
              less float_exact ~than:(6. *. se.(i)) (Float.abs (m.(i) -. exact)))
            (floats (to_vec mean)));
      test "a proposal narrower than the target has k̂ above 0.7" (fun () ->
          (* N(0, 0.3²) for N(0, 1): the ratios' tail has shape [k = 1 - 0.3²] =
             0.91. Over 1000 sets of 4000 draws, 20% of the k̂ fall below 0.7;
             the median of 40 does with probability below 1e-5. *)
          let w =
            G.importance Nx.Ptree.tensor std_normal (Nx.Rng.key 7)
              ~n:(40 * 4000) (one_d 0. 0.3)
          in
          greater float_exact ~than:0.7 (median (khats w ~sets:40)));
      test "the draws depend on the key alone" (fun () ->
          let draw k =
            rows (G.importance pos (G.log_density pos wide) k ~n:5 g).values
          in
          let a = draw (Nx.Rng.key 11) in
          equal (array float_exact)
            (floats (Nx.reshape [| 20 |] a))
            (floats (Nx.reshape [| 20 |] (draw (Nx.Rng.key 11))));
          not_equal (array float_exact)
            (floats (Nx.reshape [| 20 |] a))
            (floats (Nx.reshape [| 20 |] (draw (Nx.Rng.key 12)))));
      test "compiled, it is the eager draws" (fun () ->
          let lp = G.log_density pos wide in
          let f k = G.importance pos lp k ~n:5 g in
          let wp = Norn.Weighted.ptree pos in
          let compiled =
            Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> returns wp) f (Nx.Rng.key 13)
          in
          let eager = f (Nx.Rng.key 13) in
          equal
            (array (float_rel ~rel:1e-12 ~abs:1e-12))
            (floats (Nx.reshape [| 20 |] (rows eager.values)))
            (floats (Nx.reshape [| 20 |] (rows compiled.values)));
          equal
            (array (float_rel ~rel:1e-12 ~abs:1e-12))
            (log_weights eager) (log_weights compiled));
      test "n below 1 is refused" (fun () ->
          raises
            (Invalid_argument "Norn.Gaussian.importance: n = 0 is not positive")
            (fun () ->
              G.importance pos (G.log_density pos g) (Nx.Rng.key 1) ~n:0 g));
      test "a density of the wrong shape is refused" (fun () ->
          raises
            (Invalid_argument
               "Norn.Gaussian.importance: the density returned shape [] for 3 \
                draws; a density returns one log density per draw, shape [3]")
            (fun () ->
              G.importance pos
                (fun x -> Nx.sum (G.log_density pos g x))
                (Nx.Rng.key 1) ~n:3 g));
    ]

let () = exit (run "Norn.Gaussian" [ laws; errors; importance ])
