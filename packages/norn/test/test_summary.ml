(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Norn.Summary
module Draws = Norn.Draws

type 'a p = { mu : 'a; theta : 'a }

module P = struct
  type ('a, _) t = 'a p

  let walk c { mu; theta } =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf mu in
    let theta = field c "theta" leaf theta in
    { mu; theta }
end

let p : (Nx.host Nx.float64_t p, Nx.host) Nx.Ptree.t = Nx.Ptree.instantiate (module P)

(* 4 chains of 200 seeded normal draws: [mu] mixes, [theta.(0)] is shifted by
   chain, [theta.(1)] is constant. *)
let draws =
  let k = Nx.Rng.key 3 in
  let z = Nx.Rng.normal ~key:k Nx.float64 [| 4; 200; 2 |] in
  let shift = Nx.reshape [| 4; 1 |] (Nx.arange_f Nx.float64 0. 4. 1.) in
  let theta0 = Nx.add (Nx.slice [ Nx.A; Nx.A; Nx.I 1 ] z) (Nx.mul_s shift 3.) in
  let theta1 = Nx.full Nx.float64 [| 4; 200 |] 2. in
  Draws.v p
    {
      mu = Nx.slice [ Nx.A; Nx.A; Nx.I 0 ] z;
      theta = Nx.stack ~axis:2 [ theta0; theta1 ];
    }

let stats ~diverging ~saturated ~energy =
  let z = Nx.zeros Nx.float64 [| 4; 200 |] in
  Draws.v
    (Norn.Stats.ptree Nx.float64)
    Norn.Stats.
      {
        lp = z;
        acceptance = z;
        step_size = z;
        n_steps = Nx.zeros Nx.int32 [| 4; 200 |];
        diverging;
        saturated;
        energy;
      }

let column s name = Nx.to_array (List.assoc name (S.columns s))

let rows =
  group "rows"
    [
      test "a row per element, labelled by path and index" (fun () ->
          equal (array string)
            [| "mu"; "theta[0]"; "theta[1]" |]
            (S.labels (S.v p draws)));
      test "a tensor at the root is labelled by its index" (fun () ->
          let d =
            Draws.v Nx.Ptree.tensor (Nx.zeros Nx.float64 [| 2; 5; 2; 1 |])
          in
          equal (array string) [| "[0,0]"; "[1,0]" |]
            (S.labels (S.v Nx.Ptree.tensor d)));
      test "columns are the moments, quantiles and diagnostics" (fun () ->
          equal (list string)
            [
              "mean";
              "sd";
              "q5";
              "median";
              "q95";
              "mcse";
              "ess_bulk";
              "ess_tail";
              "rhat";
            ]
            (List.map fst (S.columns (S.v p draws))));
      test "the mean and sd are of the pooled draws" (fun () ->
          let x = Nx.create Nx.float64 [| 1; 4 |] [| 1.; 2.; 3.; 6. |] in
          let s = S.v Nx.Ptree.tensor (Draws.v Nx.Ptree.tensor x) in
          equal (array (float 1e-12)) [| 3. |] (column s "mean");
          equal
            (array (float 1e-12))
            [| Float.sqrt (14. /. 3.) |]
            (column s "sd");
          equal (array (float 1e-12)) [| 2.5 |] (column s "median"));
      test "the diagnostics are Diag's" (fun () ->
          let s = S.v p draws in
          let r = Norn.Diag.rhat p draws in
          equal (float 1e-12) (Nx.item [] r.mu) (column s "rhat").(0));
      test "chains of fewer than 4 draws are refused" (fun () ->
          raises
            (Invalid_argument
               "Norn.Summary.v: chains of 3 draws; a summary needs at least 4")
            (fun () ->
              S.v Nx.Ptree.tensor
                (Draws.v Nx.Ptree.tensor (Nx.zeros Nx.float64 [| 2; 3 |]))));
      test "an element with draws that are not finite is a finding" (fun () ->
          let x =
            Nx.create Nx.float64 [| 1; 4 |]
              [| 1.; Float.nan; 3.; Float.infinity |]
          in
          let s = S.v Nx.Ptree.tensor (Draws.v Nx.Ptree.tensor x) in
          equal bool true
            (List.exists
               (function S.Not_finite { count = 2; _ } -> true | _ -> false)
               (S.findings s)));
      test "concat puts rows after rows" (fun () ->
          let s = S.v p draws in
          equal int 6 (Array.length (S.labels (S.concat [ s; s ]))));
    ]

let has_finding s pred = List.exists pred (S.findings s)

let findings =
  group "findings"
    [
      test "chains that disagree have a high R-hat" (fun () ->
          let s = S.v p draws in
          equal bool true
            (has_finding s (function
              | S.Rhat_high { index = [| 0 |]; _ } -> true
              | _ -> false)));
      test "a constant element is a finding of its own" (fun () ->
          let s = S.v p draws in
          equal bool true
            (has_finding s (function
              | S.Constant { index = [| 1 |]; _ } -> true
              | _ -> false)));
      test "mixed draws raise no finding" (fun () ->
          let s =
            S.v Nx.Ptree.tensor
              (Draws.v Nx.Ptree.tensor (draws :> Nx.host Nx.float64_t p).mu)
          in
          equal int 0 (List.length (S.findings s)));
      test "divergences gather where their draws stand apart" (fun () ->
          let mu = (draws :> Nx.host Nx.float64_t p).mu in
          (* The transitions whose draw of mu is above 1.5 diverged. *)
          let diverging = Nx.greater_s mu 1.5 in
          let st =
            stats ~diverging
              ~saturated:(Nx.zeros Nx.bool [| 4; 200 |])
              ~energy:(Nx.Rng.normal ~key:(Nx.Rng.key 4) Nx.float64 [| 4; 200 |])
          in
          let s = S.v Nx.Ptree.tensor ~stats:st (Draws.v Nx.Ptree.tensor mu) in
          let count =
            Array.fold_left
              (fun n b -> if b then n + 1 else n)
              0 (Nx.to_array diverging)
          in
          match
            List.find
              (function S.Divergent _ -> true | _ -> false)
              (S.findings s)
          with
          | S.Divergent { count = c; total; regions } ->
              equal int count c;
              equal int 800 total;
              equal int 1 (List.length regions)
          | _ -> fail "no divergence finding");
      test "a chain whose energy barely moves has a low E-BFMI" (fun () ->
          (* Energy that follows a slow random walk on chain 2. *)
          let walk =
            Nx.cumsum ~axis:1
              (Nx.Rng.normal ~key:(Nx.Rng.key 5) Nx.float64 [| 4; 200 |])
          in
          let white = Nx.Rng.normal ~key:(Nx.Rng.key 6) Nx.float64 [| 4; 200 |] in
          let chain2 =
            Nx.reshape [| 4; 1 |]
              (Nx.create Nx.bool [| 4 |] [| false; false; true; false |])
          in
          let energy = Nx.where chain2 walk white in
          let zb = Nx.zeros Nx.bool [| 4; 200 |] in
          let st = stats ~diverging:zb ~saturated:zb ~energy in
          let mu = (draws :> Nx.host Nx.float64_t p).mu in
          let s = S.v Nx.Ptree.tensor ~stats:st (Draws.v Nx.Ptree.tensor mu) in
          equal (list int) [ 2 ]
            (List.filter_map
               (function S.Ebfmi_low { chain; _ } -> Some chain | _ -> None)
               (S.findings s)));
      test "saturated transitions are counted" (fun () ->
          let zb = Nx.zeros Nx.bool [| 4; 200 |] in
          let saturated =
            Nx.set [ Nx.I 0; Nx.R (0, 7) ] (Nx.scalar Nx.bool true) zb
          in
          let st =
            stats ~diverging:zb ~saturated
              ~energy:(Nx.Rng.normal ~key:(Nx.Rng.key 4) Nx.float64 [| 4; 200 |])
          in
          let mu = (draws :> Nx.host Nx.float64_t p).mu in
          let s = S.v Nx.Ptree.tensor ~stats:st (Draws.v Nx.Ptree.tensor mu) in
          equal bool true
            (has_finding s (function
              | S.Saturated { count = 7; total = 800 } -> true
              | _ -> false)));
    ]

let printing =
  test "a summary prints as a table, then its findings" (fun () ->
      let x =
        Nx.create Nx.float64 [| 2; 4; 2 |]
          [| 1.; 5.; 2.; 5.; 3.; 5.; 4.; 5.; 2.; 5.; 3.; 5.; 4.; 5.; 5.; 5. |]
      in
      let s = S.v Nx.Ptree.tensor (Draws.v Nx.Ptree.tensor x) in
      expect (Format.asprintf "%a" S.pp s)
      @@ __POS_OF__
           {|
             mean    sd    q5  median   q95   mcse  ess_bulk  ess_tail  rhat
        [0]     3  1.31  1.35       3  4.65  0.487         7         7  1.89
        [1]     5     0     5       5     5    nan       nan       nan   nan
        R-hat of [0] is 1.889, above 1.01
        effective sample size of [0] is 7 in the bulk and 7 in the tails, too few
        [1] is constant: it has no R-hat nor effective sample size
        |})

(* Weighted draws *)

let weighted values log_weights = Norn.Weighted.{ values; log_weights }
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs

let pareto_k s =
  List.filter_map
    (function S.Pareto_k { khat; _ } -> Some khat | _ -> None)
    (S.findings s)

let nan_or_close =
  let close = float_rel ~rel:1e-9 ~abs:1e-12 in
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%h" x)
    ~equal:(fun a b ->
      (Float.is_nan a && Float.is_nan b) || Testable.equal close a b)

let weighted_gen =
  Gen.(
    let* n = int_range 1 60 in
    let* xs = array ~size:(constant n) (float_range (-5.) 5.) in
    let+ lw = array ~size:(constant n) (float_range (-4.) 4.) in
    (xs, lw))

let of_weighted =
  group "weighted"
    [
      test "columns are the moments, quantiles, error and effective size"
        (fun () ->
          let w = weighted (vec [| 1. |]) (vec [| 0. |]) in
          let s = S.of_weighted Nx.Ptree.tensor w in
          equal (list string)
            [ "mean"; "sd"; "q5"; "median"; "q95"; "mcse"; "ess" ]
            (List.map fst (S.columns s)));
      test "the statistics weigh each draw by its normalised weight" (fun () ->
          (* Weights 1/4, 1/4, 1/2 and 0. *)
          let w =
            weighted (vec [| 1.; 2.; 3.; 6. |])
              (vec [| 0.; 0.; Float.log 2.; Float.neg_infinity |])
          in
          let s = S.of_weighted Nx.Ptree.tensor w in
          let one name = (column s name).(0) in
          equal (float 1e-12) 2.25 (one "mean");
          equal (float 1e-12) (Float.sqrt 0.6875) (one "sd");
          equal (float 1e-12) 1. (one "q5");
          (* Draws 1, 2 and 3 stand at 1/8, 3/8 and 3/4. *)
          equal (float 1e-12) (7. /. 3.) (one "median");
          equal (float 1e-12) 3. (one "q95");
          equal (float 1e-12) (Float.sqrt 0.2421875) (one "mcse");
          equal (float 1e-12) (0.6875 /. 0.2421875) (one "ess"));
      prop "equal weights give the draws' mean and an effective size of n"
        weighted_gen (fun (xs, _) ->
          let n = Array.length xs in
          let s =
            S.of_weighted Nx.Ptree.tensor
              (weighted (vec xs) (Nx.zeros Nx.float64 [| n |]))
          in
          let mean = Array.fold_left ( +. ) 0. xs /. float_of_int n in
          equal (float 1e-9) mean (column s "mean").(0);
          let ess = (column s "ess").(0) in
          (* A constant element has no spread to measure an error by. *)
          if Array.for_all (fun x -> x = xs.(0)) xs then
            satisfies ~claim:"nan" float_exact Float.is_nan ess
          else equal (float_rel ~rel:1e-9 ~abs:0.) (float_of_int n) ess);
      prop "a constant added to the log weights changes nothing" weighted_gen
        (fun (xs, lw) ->
          let s = S.of_weighted Nx.Ptree.tensor (weighted (vec xs) (vec lw)) in
          let s' =
            S.of_weighted Nx.Ptree.tensor
              (weighted (vec xs) (vec (Array.map (fun l -> l +. 300.) lw)))
          in
          List.iter2
            (fun (name, a) (_, b) ->
              equal ~msg:name (array nan_or_close) (Nx.to_array a)
                (Nx.to_array b))
            (S.columns s) (S.columns s'));
      test "heavy-tailed weights are a finding with their raw k-hat"
        (fun () ->
          (* Weights U^(-1.3): no mean, so k-hat is far above 0.7. *)
          let u = Nx.Rng.uniform ~key:(Nx.Rng.key 8) Nx.float64 [| 2000 |] in
          let lw = Nx.mul_s (Nx.log u) (-1.3) in
          let s = S.of_weighted Nx.Ptree.tensor (weighted u lw) in
          match pareto_k s with
          | [ k ] ->
              greater float_exact ~than:0.7 k;
              equal float_exact (Nx.item [] (Norn.Diag.psis lw).khat) k
          | ks -> failf "Pareto_k findings: %d" (List.length ks));
      test "raw weights are summarised as smoothed" (fun () ->
          let k = Nx.Rng.key 10 in
          let x = Nx.Rng.normal ~key:(Nx.Rng.fold_in k 0) Nx.float64 [| 1000 |] in
          let u = Nx.Rng.uniform ~key:(Nx.Rng.fold_in k 1) Nx.float64 [| 1000 |] in
          let lw = Nx.mul_s (Nx.log u) (-0.8) in
          let s = S.of_weighted Nx.Ptree.tensor (weighted x lw) in
          (* The statistics by hand, from Diag.psis's weights. *)
          let sw = Nx.to_array (Norn.Diag.psis lw).log_weights in
          let xs = Nx.to_array x in
          let m = Array.fold_left Float.max Float.neg_infinity sw in
          let w = Array.map (fun l -> Float.exp (l -. m)) sw in
          let total = Array.fold_left ( +. ) 0. w in
          let w = Array.map (fun w -> w /. total) w in
          let sum f =
            let s = ref 0. in
            Array.iteri (fun i x -> s := !s +. f w.(i) x) xs;
            !s
          in
          let mean = sum (fun w x -> w *. x) in
          let var = sum (fun w x -> w *. ((x -. mean) ** 2.)) in
          let err = sum (fun w x -> w *. w *. ((x -. mean) ** 2.)) in
          let close = float_rel ~rel:1e-9 ~abs:1e-12 in
          let one name = (column s name).(0) in
          equal close mean (one "mean");
          equal close (Float.sqrt var) (one "sd");
          equal close (Float.sqrt err) (one "mcse");
          equal close (var /. err) (one "ess"));
      test "equal weights are no finding" (fun () ->
          let x = Nx.Rng.normal ~key:(Nx.Rng.key 9) Nx.float64 [| 500 |] in
          let w = weighted x (Nx.zeros Nx.float64 [| 500 |]) in
          let s = S.of_weighted Nx.Ptree.tensor w in
          equal int 0 (List.length (S.findings s)));
      test "weights too few to judge are a finding" (fun () ->
          let x = Nx.Rng.normal ~key:(Nx.Rng.key 9) Nx.float64 [| 24 |] in
          let s = S.of_weighted Nx.Ptree.tensor (weighted x x) in
          match pareto_k s with
          | [ k ] -> satisfies ~claim:"nan" float_exact Float.is_nan k
          | ks -> failf "Pareto_k findings: %d" (List.length ks));
      test "a tensor with another number of draws is refused" (fun () ->
          let d =
            {
              mu = Nx.zeros Nx.float64 [| 4 |];
              theta = Nx.zeros Nx.float64 [| 3; 2 |];
            }
          in
          let w = weighted d (Nx.zeros Nx.float64 [| 4 |]) in
          raises
            (Invalid_argument
               "Norn.Summary.of_weighted: theta: 3 draws for 4 weights")
            (fun () -> S.of_weighted p w));
      test "log weights of more than one axis are refused" (fun () ->
          raises
            (Invalid_argument
               "Norn.Summary.of_weighted: log weights of shape [4; 1], not [n]")
            (fun () ->
              S.of_weighted Nx.Ptree.tensor
                (weighted
                   (Nx.zeros Nx.float64 [| 4 |])
                   (Nx.zeros Nx.float64 [| 4; 1 |]))));
      test "chains and weighted draws do not concatenate" (fun () ->
          let w = weighted (vec [| 1. |]) (vec [| 0. |]) in
          let w = S.of_weighted Nx.Ptree.tensor w in
          raises_match (Exn.invalid_arg ~sub:"different columns") (fun () ->
              S.concat [ S.v p draws; w ]));
    ]

let weighted_printing =
  test "a weighted summary prints its own columns" (fun () ->
      let w =
        weighted (vec [| 1.; 2.; 3.; 6. |])
          (vec [| 0.; 0.; Float.log 2.; Float.neg_infinity |])
      in
      expect (Format.asprintf "%a" S.pp (S.of_weighted Nx.Ptree.tensor w))
      @@ __POS_OF__
           {|
               mean     sd  q5  median  q95   mcse  ess
        value  2.25  0.829   1    2.33    3  0.492    3
        the weights have no Pareto k-hat: estimates from them are not judged
        |})

let () =
  exit
    (run "Norn.Summary"
       [
         rows;
         findings;
         of_weighted;
         group "pp" [ printing; weighted_printing ];
       ])
