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

let pos : (Nx.host Nx.float64_t pos, Nx.host) Nx.Ptree.t =
  Nx.Ptree.instantiate (module Pos)

let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs
let floats x = Nx.to_array x
let close = float_rel ~rel:1e-10 ~abs:1e-12

(* A position of 4 elements, [a] of 1 and [b] of 3, from a vector, or from rows
   [[n; 4]]. *)
let of_vec v =
  { a = Nx.slice [ Nx.R (0, 1) ] v; b = Nx.slice [ Nx.R (1, 4) ] v }

let of_rows m =
  { a = Nx.slice [ Nx.A; Nx.R (0, 1) ] m; b = Nx.slice [ Nx.A; Nx.R (1, 4) ] m }

let to_vec p = Nx.concatenate ~axis:0 [ p.a; p.b ]
let rows p = Nx.concatenate ~axis:1 [ p.a; p.b ]
let mean () = of_vec (vec [| 0.5; -1.; 2.; 0.25 |])
let scale () = of_vec (vec [| 2.; 0.5; 1.; 3. |])

(* Orthonormal directions in 4 elements, with variances along them. *)
let directions =
  [
    ([| 0.5; 0.5; 0.5; 0.5 |], 4.);
    ([| 0.5; -0.5; 0.5; -0.5 |], 0.25);
    ([| 0.5; 0.5; -0.5; -0.5 |], 2.);
    ([| 0.5; -0.5; -0.5; 0.5 |], 0.5);
  ]

(* [gaussian k] is the Gaussian of [mean] and [scale] with the first [k]
   directions, and [dense k] its covariance [S (I + Σ (v - 1) u uᵀ) S]. *)
let gaussian k =
  let us = List.filteri (fun i _ -> i < k) directions in
  if k = 0 then G.diagonal pos Nx.float64 ~mean:(mean ()) ~scale:(scale ())
  else
    G.low_rank pos ~mean:(mean ()) ~scale:(scale ())
      ~directions:
        (of_rows (Nx.stack ~axis:0 (List.map (fun (u, _) -> vec u) us)))
      ~variances:(vec (Array.of_list (List.map snd us)))

let dense k =
  let s = Nx.diag (to_vec (scale ())) in
  let inner =
    List.fold_left
      (fun acc (u, v) ->
        let u = vec u in
        Nx.add acc
          (Nx.mul_s
             (Nx.matmul
                (Nx.unsqueeze ~axes:[ 1 ] u)
                (Nx.unsqueeze ~axes:[ 0 ] u))
             (v -. 1.)))
      (Nx.eye Nx.float64 4)
      (List.filteri (fun i _ -> i < k) directions)
  in
  Nx.matmul s (Nx.matmul inner s)

(* [normal_log_density ~precision x] is the log density at [x] of the normal of
   mean [mean] and precision matrix [precision]. *)
let normal_log_density ~precision x =
  let d = Nx.sub x (to_vec (mean ())) in
  let _, logdet = Nx.slogdet Nx.float64 precision in
  let q = Nx.item [] (Nx.vdot d (Nx.matmul precision d)) in
  (-0.5 *. q) +. (0.5 *. Nx.item [] logdet) -. (2. *. Float.log (2. *. Float.pi))

(* Rows of 4 elements, from none to 3. *)
let points =
  Gen.(
    let* n = int_range 0 3 in
    array ~size:(constant (4 * n)) (float_range (-5.) 5.))
  |> Gen.with_pp (fun ppf xs ->
      Format.fprintf ppf "[%s]"
        (String.concat "; " (Array.to_list (Array.map string_of_float xs))))

(* [check_rows lp g xs] checks that [g]'s log density is [lp] at each row of
   [xs]. *)
let check_rows lp g xs =
  let n = Array.length xs / 4 in
  cover "no row" (n = 0);
  let m = Nx.create Nx.float64 [| n; 4 |] xs in
  let expected = Array.init n (fun i -> lp (Nx.slice [ Nx.I i ] m)) in
  equal (array close) expected (floats (G.log_density pos g (of_rows m)))

(* A precision of diagonal [1, 4, 0.5, 2] plus 3 w wᵀ, w = [1, 0, 1, 0]. *)
let p () = of_vec (vec [| 1.; 4.; 0.5; 2. |])
let w () = vec [| 1.; 0.; 1.; 0. |]

let precision () =
  Nx.add
    (Nx.diag (to_vec (p ())))
    (Nx.mul_s
       (Nx.matmul (Nx.reshape [| 4; 1 |] (w ())) (Nx.reshape [| 1; 4 |] (w ())))
       3.)

let of_precision () =
  G.of_precision pos Nx.float64
    ~low_rank:(of_rows (Nx.reshape [| 1; 4 |] (w ())), vec [| 3. |])
    ~mean:(mean ()) (p ())

let covariance =
  group "covariance"
    [
      prop "gives the log density of the dense Gaussian, row by row"
        Gen.(pair (int_range 0 2) points)
        (fun (i, xs) ->
          let k = [| 0; 2; 4 |].(i) in
          cover "diagonal" (k = 0);
          cover "every direction" (k = 4);
          let precision = Nx.inv (dense k) in
          check_rows (normal_log_density ~precision) (gaussian k) xs);
      prop "gives of_precision's Gaussian the log density of its precision"
        points (fun xs ->
          check_rows
            (normal_log_density ~precision:(precision ()))
            (of_precision ()) xs);
      cases "has variance the dense covariance's diagonal" ~id:fst
        [ ("diagonal", 0); ("rank 2", 2); ("rank 4", 4); ("of_precision", -1) ]
        (fun (_, k) ->
          let g, cov =
            if k < 0 then (of_precision (), Nx.inv (precision ()))
            else (gaussian k, dense k)
          in
          equal (array close)
            (floats (Nx.diagonal cov))
            (floats (to_vec (G.variance pos g))));
    ]

let diagonal =
  test "diagonal raises when the mean holds no tensor" (fun () ->
      raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.diagonal") (fun () ->
          G.diagonal Nx.Ptree.unit Nx.float64 ~mean:() ~scale:()))

let low_rank =
  let low_rank directions variances =
    G.low_rank pos ~mean:(mean ()) ~scale:(scale ())
      ~directions:(of_rows directions) ~variances
  in
  let two =
    Nx.stack ~axis:0
      [ vec (fst (List.hd directions)); vec (fst (List.nth directions 1)) ]
  in
  group "low_rank"
    [
      test "orthonormalises its directions" (fun () ->
          let u1 = Nx.slice [ Nx.I 0 ] two and u2 = Nx.slice [ Nx.I 1 ] two in
          let skewed =
            Nx.stack ~axis:0 [ Nx.mul_s u1 3.; Nx.add u2 (Nx.mul_s u1 0.7) ]
          in
          equal (array close)
            (floats (to_vec (G.variance pos (gaussian 2))))
            (floats
               (to_vec (G.variance pos (low_rank skewed (vec [| 4.; 0.25 |]))))));
      cases "raises for variances not of shape [k], or not positive and finite"
        ~id:fst
        [
          ("[3]", [| 4.; 0.25; 1. |]);
          ("0", [| 4.; 0. |]);
          ("inf", [| 4.; Float.infinity |]);
        ]
        (fun (_, v) ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.low_rank")
            (fun () -> low_rank two (vec v)));
    ]

let of_precision_refusals =
  let refuse p l =
    G.of_precision pos Nx.float64
      ~low_rank:(of_rows (Nx.reshape [| 1; 4 |] (w ())), vec [| l |])
      ~mean:(mean ())
      (of_vec (vec p))
  in
  group "of_precision"
    [
      test "raises naming the path of a precision not positive" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"b at [1]") (fun () ->
              refuse [| 1.; 1.; -1.; 1. |] 3.));
      cases "raises for a low-rank weight negative or not finite"
        ~id:string_of_float [ -1.; Float.nan; Float.infinity ] (fun l ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.of_precision")
            (fun () -> refuse [| 1.; 1.; 1.; 1. |] l));
      test "accepts a low-rank weight of zero" (fun () ->
          let g = refuse [| 1.; 4.; 0.5; 2. |] 0. in
          equal (array close) [| 1.; 0.25; 2.; 0.5 |]
            (floats (to_vec (G.variance pos g))));
      test "raises when the mean holds no tensor" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.of_precision")
            (fun () -> G.of_precision Nx.Ptree.unit Nx.float64 ~mean:() ()));
    ]

let sample =
  group "sample"
    [
      test "draws the Gaussian's mean and covariance" (fun () ->
          let n = 200_000 and fn = 200_000. in
          let x = rows (G.sample pos (Nx.Rng.key 4) ~n (gaussian 2)) in
          let cov = dense 2 in
          let m = Nx.mean ~axes:[ 0 ] x in
          let c =
            Nx.div_s (Nx.matmul (Nx.transpose (Nx.sub x m)) (Nx.sub x m)) fn
          in
          (* Six standard errors, a false failure at 2e-9 an entry: the mean's
             sqrt (σ_ii / n), a covariance's sqrt ((σ_ii σ_jj + σ_ij²) / n). *)
          let s i j = Nx.item [ i; j ] cov in
          Array.iteri
            (fun i d ->
              less float_exact
                ~than:(6. *. Float.sqrt (s i i /. fn))
                (Float.abs d))
            (floats (Nx.sub m (to_vec (G.mean pos (gaussian 2)))));
          for i = 0 to 3 do
            for j = 0 to 3 do
              let se =
                Float.sqrt (((s i i *. s j j) +. (s i j *. s i j)) /. fn)
              in
              less float_exact ~than:(6. *. se)
                (Float.abs (Nx.item [ i; j ] c -. s i j))
            done
          done);
      test "draws none for n = 0" (fun () ->
          let x = G.sample pos (Nx.Rng.key 1) ~n:0 (gaussian 2) in
          equal (array int) [| 0; 3 |] (Nx.shape x.b));
      test "raises for n below 0" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.sample") (fun () ->
              G.sample pos (Nx.Rng.key 1) ~n:(-1) (gaussian 2)));
    ]

(* [stacked g g'] is the two Gaussians on a leading axis of two. *)
let stacked g g' =
  Nx.Ptree.map2 (G.ptree pos) (fun _ x y -> Nx.stack ~axis:0 [ x; y ]) g g'

let narrow () =
  G.diagonal pos Nx.float64 ~mean:(mean ())
    ~scale:(of_vec (vec [| 1.; 1.; 1.; 1. |]))

let wide () =
  G.diagonal pos Nx.float64 ~mean:(mean ())
    ~scale:(of_vec (vec [| 4.; 2.; 3.; 6. |]))

let stacked_gaussians =
  group "stacked Gaussians"
    [
      test "mean of stacked Gaussians is their means stacked" (fun () ->
          let m = G.mean pos (stacked (narrow ()) (wide ())) in
          equal (array float_exact)
            (floats (Nx.stack ~axis:0 [ to_vec (mean ()); to_vec (mean ()) ]))
            (floats (rows m)));
      cases "sample, log_density, variance and pp raise for stacked Gaussians"
        ~id:fst
        [
          ("sample", fun g -> ignore (G.sample pos (Nx.Rng.key 1) ~n:2 g));
          ( "log_density",
            fun g ->
              ignore
                (G.log_density pos g (of_rows (Nx.zeros Nx.float64 [| 2; 4 |])))
          );
          ("variance", fun g -> ignore (G.variance pos g));
          ("pp", fun g -> ignore (Format.asprintf "%a" (G.pp pos) g));
        ]
        (fun (name, f) ->
          raises_match
            (Exn.invalid_arg ~sub:("Norn.Gaussian." ^ name))
            (fun () -> f (stacked (narrow ()) (wide ()))));
    ]

let std_normal x =
  Nx.sum ~axes:[ 1 ]
    (Norn.Dist.factors
       (Norn.Dist.normal ~loc:(Nx.scalar Nx.float64 0.)
          ~scale:(Nx.scalar Nx.float64 1.))
       x)

let importance =
  group "importance"
    [
      prop "weighs a Gaussian's draws for its own density by zero"
        Gen.(pair (int_range 0 1000) (int_range 1 50))
        (fun (key, n) ->
          let g = gaussian 2 in
          let w =
            G.importance pos (G.log_density pos g) (Nx.Rng.key key) ~n g
          in
          equal (array (float 1e-12)) (Array.make n 0.) (floats w.log_weights));
      test "weighs stacked Gaussians' draws by the one that drew them"
        (fun () ->
          let n = 6 in
          let narrow = narrow () and wide = wide () in
          let w =
            G.importance pos (G.log_density pos narrow) (Nx.Rng.key 3) ~n
              (stacked narrow wide)
          in
          let theirs =
            floats
              (Nx.sub
                 (G.log_density pos narrow w.values)
                 (G.log_density pos wide w.values))
          in
          let expected =
            Array.init (2 * n) (fun i -> if i < n then 0. else theirs.(i))
          in
          equal (array int) [| 2 * n; 4 |] (Nx.shape (rows w.values));
          equal (array (float 1e-12)) expected (floats w.log_weights));
      test "draws rows i n to i n + n - 1 from stacked Gaussian i" (fun () ->
          (* Each block's sample variance against its Gaussian's, within six
             standard errors σ² sqrt (2 / (n - 1)). *)
          let n = 2000 in
          let w =
            G.importance pos
              (fun x -> std_normal (rows x))
              (Nx.Rng.key 6) ~n
              (stacked (narrow ()) (wide ()))
          in
          let x = rows w.values in
          List.iteri
            (fun i scale ->
              let block = Nx.slice [ Nx.R (i * n, (i + 1) * n) ] x in
              let var = floats (Nx.var ~axes:[ 0 ] ~ddof:1 block) in
              Array.iteri
                (fun j s ->
                  let v = s *. s in
                  less float_exact
                    ~than:(6. *. v *. Float.sqrt (2. /. float_of_int (n - 1)))
                    (Float.abs (var.(j) -. v)))
                scale)
            [ [| 1.; 1.; 1.; 1. |]; [| 4.; 2.; 3.; 6. |] ]);
      test "gives a wider proposal the target's means" (fun () ->
          (* A proposal wider than the target in every direction bounds the
             ratios, so k̂ is below 0.5 and the smoothed weights' means are
             normal about the target's with the summary's standard error. Six
             errors give a false alarm of 2e-9 per element. *)
          let g = gaussian 2 in
          let w =
            G.importance pos (G.log_density pos g) (Nx.Rng.key 5) ~n:4000
              (wide ())
          in
          less float_exact ~than:0.5
            (Nx.item [] (Norn.Diag.psis w.log_weights).khat);
          let s = Norn.Summary.of_weighted pos w in
          let column name = floats (List.assoc name (Norn.Summary.columns s)) in
          let m = column "mean" and se = column "mcse" in
          Array.iteri
            (fun i exact ->
              less float_exact ~than:(6. *. se.(i)) (Float.abs (m.(i) -. exact)))
            (floats (to_vec (mean ()))));
      test "gives a proposal narrower than the target k̂ above 0.7" (fun () ->
          (* N(0, 0.3²) for N(0, 1): the ratios' tail has shape [k = 1 - 0.3²] =
             0.91. Over 1000 sets of 4000 draws, 20% of the k̂ fall below 0.7;
             the median of 40 does with probability below 1e-5. *)
          let one_d =
            G.diagonal Nx.Ptree.tensor Nx.float64 ~mean:(vec [| 0. |])
              ~scale:(vec [| 0.3 |])
          in
          let w =
            G.importance Nx.Ptree.tensor std_normal (Nx.Rng.key 7)
              ~n:(40 * 4000) one_d
          in
          let khats =
            floats
              (Norn.Diag.psis (Nx.reshape [| 4000; 40 |] w.log_weights)).khat
          in
          Array.sort Float.compare khats;
          greater float_exact ~than:0.7 khats.(20));
      test "traces under jit, drawing what it draws eagerly" (fun () ->
          Compiled.here ();
          let wide = wide () and g = gaussian 2 in
          let f k = G.importance pos (G.log_density pos wide) k ~n:5 g in
          let wp = Norn.Weighted.ptree pos in
          let compiled =
            Rune.jit Nx.Ptree.(Nx.Rng.ptree @-> returns wp) f (Nx.Rng.key 13)
          in
          let eager = f (Nx.Rng.key 13) in
          let agree = array (float_rel ~rel:1e-12 ~abs:1e-12) in
          equal agree
            (floats (Nx.reshape [| 20 |] (rows eager.values)))
            (floats (Nx.reshape [| 20 |] (rows compiled.values)));
          equal agree (floats eager.log_weights) (floats compiled.log_weights));
      test "raises for n below 1" (fun () ->
          let g = gaussian 2 in
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.importance")
            (fun () ->
              G.importance pos (G.log_density pos g) (Nx.Rng.key 1) ~n:0 g));
      test "raises for a density of other than one value per draw" (fun () ->
          let g = gaussian 2 in
          raises_match (Exn.invalid_arg ~sub:"Norn.Gaussian.importance")
            (fun () ->
              G.importance pos
                (fun x -> Nx.sum (G.log_density pos g x))
                (Nx.Rng.key 1) ~n:3 g));
    ]

let ptree =
  test "ptree walks mean, scale, directions, then variances" (fun () ->
      let paths =
        List.filter_map
          (function
            | Nx.Ptree.Leaf p -> Some (Nx.Ptree.Path.to_string p)
            | Report _ -> None)
          (Nx.Ptree.visits (G.ptree pos) (gaussian 2))
      in
      equal (list string)
        [
          "mean.a";
          "mean.b";
          "scale.a";
          "scale.b";
          "directions.a";
          "directions.b";
          "variances";
        ]
        paths)

let pp =
  test "pp gives the dimension and rank" (fun () ->
      let text k = Format.asprintf "%a" (G.pp pos) (gaussian k) in
      expect (text 0 ^ "\n" ^ text 2)
      @@ __POS_OF__
           {|
             gaussian(dimension 4, rank 0)
             gaussian(dimension 4, rank 2)
             |})

let () =
  exit
    (run "Norn.Gaussian"
       [
         covariance;
         stacked_gaussians;
         diagonal;
         low_rank;
         of_precision_refusals;
         sample;
         importance;
         ptree;
         pp;
       ])
