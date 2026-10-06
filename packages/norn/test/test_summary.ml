(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module S = Norn.Summary
module Draws = Norn.Draws

type 'a p = { mu : 'a; theta : 'a }

module P = struct
  type 'a t = 'a p

  let walk c { mu; theta } =
    let open Nx.Ptree.Walk in
    let mu = field c "mu" leaf mu in
    let theta = field c "theta" leaf theta in
    { mu; theta }
end

let p : Nx.float64_t p Nx.Ptree.t = Nx.Ptree.instantiate (module P)

(* 4 chains of 200 seeded normal draws: [mu] mixes, [theta.(0)] is shifted by
   chain, [theta.(1)] is constant. *)
let draws =
  let k = Nx.Rng.key 3 in
  let z = Nx.Rng.normal k Nx.float64 [| 4; 200; 2 |] in
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
              (Draws.v Nx.Ptree.tensor (draws :> Nx.float64_t p).mu)
          in
          equal int 0 (List.length (S.findings s)));
      test "divergences gather where their draws stand apart" (fun () ->
          let mu = (draws :> Nx.float64_t p).mu in
          (* The transitions whose draw of mu is above 1.5 diverged. *)
          let diverging = Nx.greater_s mu 1.5 in
          let st =
            stats ~diverging
              ~saturated:(Nx.zeros Nx.bool [| 4; 200 |])
              ~energy:(Nx.Rng.normal (Nx.Rng.key 4) Nx.float64 [| 4; 200 |])
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
              (Nx.Rng.normal (Nx.Rng.key 5) Nx.float64 [| 4; 200 |])
          in
          let white = Nx.Rng.normal (Nx.Rng.key 6) Nx.float64 [| 4; 200 |] in
          let chain2 =
            Nx.reshape [| 4; 1 |]
              (Nx.create Nx.bool [| 4 |] [| false; false; true; false |])
          in
          let energy = Nx.where chain2 walk white in
          let zb = Nx.zeros Nx.bool [| 4; 200 |] in
          let st = stats ~diverging:zb ~saturated:zb ~energy in
          let mu = (draws :> Nx.float64_t p).mu in
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
              ~energy:(Nx.Rng.normal (Nx.Rng.key 4) Nx.float64 [| 4; 200 |])
          in
          let mu = (draws :> Nx.float64_t p).mu in
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

let () = exit (run "Norn.Summary" [ rows; findings; group "pp" [ printing ] ])
