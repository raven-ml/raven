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

let p : (Nx.host Nx.float64_t p, Nx.host) Nx.Ptree.t =
  Nx.Ptree.instantiate (module P)

let t = Nx.Ptree.tensor
let chains c n xs = Draws.v t (Nx.create Nx.float64 [| c; n |] xs)

let nan_or_close =
  let close = float_rel ~rel:1e-9 ~abs:1e-12 in
  Testable.make
    ~pp:(fun ppf x -> Format.fprintf ppf "%h" x)
    ~equal:(fun a b ->
      (Float.is_nan a && Float.is_nan b) || Testable.equal close a b)

let column s name = Nx.to_array (List.assoc name (S.columns s))

(* 4 chains of 200 seeded normal draws: [mu] mixes, [theta.(0)] is shifted by
   chain, [theta.(1)] is constant. *)
let draws () =
  let z = Nx.Rng.normal ~key:(Nx.Rng.key 3) Nx.float64 [| 4; 200; 2 |] in
  let shift = Nx.reshape [| 4; 1 |] (Nx.arange_f Nx.float64 0. 4. 1.) in
  let theta0 = Nx.add (Nx.slice [ Nx.A; Nx.A; Nx.I 1 ] z) (Nx.mul_s shift 3.) in
  let theta1 = Nx.full Nx.float64 [| 4; 200 |] 2. in
  Draws.v p
    {
      mu = Nx.slice [ Nx.A; Nx.A; Nx.I 0 ] z;
      theta = Nx.stack ~axis:2 [ theta0; theta1 ];
    }

let mu () = Draws.v t (draws () :> Nx.host Nx.float64_t p).mu

(* [named s] is each finding of [s] as its constructor, the element it names and
   its integers, as ["Not_finite theta[0] 2"]. *)
let named s =
  let element path index =
    let at =
      if index = [||] then ""
      else
        "["
        ^ String.concat "," (Array.to_list (Array.map string_of_int index))
        ^ "]"
    in
    match Nx.Ptree.Path.to_string path with
    | "" when at = "" -> "value"
    | name -> name ^ at
  in
  List.map
    (function
      | S.Rhat_high { path; index; _ } -> "Rhat_high " ^ element path index
      | Ess_low { path; index; _ } -> "Ess_low " ^ element path index
      | Not_finite { path; index; count } ->
          Printf.sprintf "Not_finite %s %d" (element path index) count
      | Constant { path; index } -> "Constant " ^ element path index
      | Divergent { count; total; regions } ->
          Printf.sprintf "Divergent %d/%d at [%s]" count total
            (String.concat "; "
               (List.map (fun (r : S.region) -> element r.path r.index) regions))
      | Saturated { count; total } ->
          Printf.sprintf "Saturated %d/%d" count total
      | Ebfmi_low { chain; _ } -> Printf.sprintf "Ebfmi_low %d" chain
      | Pareto_k _ -> "Pareto_k")
    (S.findings s)

(* [transitions ~diverging ~saturated ~energy] is the statistics of 4 chains of
   200 transitions, the other fields zero. *)
let transitions ?(diverging = Nx.zeros Nx.bool [| 4; 200 |])
    ?(saturated = Nx.zeros Nx.bool [| 4; 200 |])
    ?(energy = Nx.Rng.normal ~key:(Nx.Rng.key 4) Nx.float64 [| 4; 200 |]) () =
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

(* Draws on a grid of 1/8, so that a scaled draw and the spread of differing
   ones stay normal, and a power [j] of two whose square, beside such draws,
   underflows or overflows for some. *)
let scaled_gen =
  Gen.(
    pair
      (array ~size:(constant 8)
         (map (fun k -> float_of_int k /. 8.) (int_range (-64) 64)))
      (int_range (-1000) 1000))

(* A statistic of draws times [2^j] is [2^j] times the draws' statistic, exactly
   for ordinary draws, so 1e-12 of it is far below what an intermediate that
   overflows or underflows would lose. *)
let scaled = float_rel ~rel:1e-12 ~abs:0.

let cover_scaled j =
  cover "a scale whose square underflows" (j < -540);
  cover "a scale whose square overflows" (j > 520)

let statistics = [ "mean"; "sd"; "q5"; "median"; "q95"; "mcse" ]

let labels =
  group "labels"
    [
      test "label a row by path and index" (fun () ->
          let d =
            Draws.v p
              {
                mu = Nx.zeros Nx.float64 [| 2; 5 |];
                theta = Nx.zeros Nx.float64 [| 2; 5; 2; 2 |];
              }
          in
          equal (array string)
            [| "mu"; "theta[0,0]"; "theta[0,1]"; "theta[1,0]"; "theta[1,1]" |]
            (S.labels (S.v p d)));
      cases "label a root tensor by its index alone, a root scalar value"
        ~id:fst
        [
          ("tensor", ([| 2; 5; 2; 1 |], [| "[0,0]"; "[1,0]" |]));
          ("scalar", ([| 2; 5 |], [| "value" |]));
        ]
        (fun (_, (shape, expected)) ->
          let d = Draws.v t (Nx.zeros Nx.float64 shape) in
          equal (array string) expected (S.labels (S.v t d)));
    ]

let v =
  group "v"
    [
      test "has the moments, quantiles and diagnostics as columns" (fun () ->
          equal (list string)
            (statistics @ [ "ess_bulk"; "ess_tail"; "rhat" ])
            (List.map fst (S.columns (S.v p (draws ())))));
      test "computes the moments and quantiles of the pooled draws" (fun () ->
          (* Linear between order statistics: q5 at 0.15 of the way from 1 to 2,
             the median halfway from 2 to 3, q95 at 0.85 from 3 to 6. *)
          let s = S.v t (chains 1 4 [| 1.; 6.; 3.; 2. |]) in
          equal
            (list (pair string (array (float 1e-12))))
            [
              ("mean", [| 3. |]);
              ("sd", [| Float.sqrt (14. /. 3.) |]);
              ("q5", [| 1.15 |]);
              ("median", [| 2.5 |]);
              ("q95", [| 5.55 |]);
            ]
            (List.map
               (fun name -> (name, column s name))
               [ "mean"; "sd"; "q5"; "median"; "q95" ]));
      test "has Diag's diagnostics" (fun () ->
          let d = draws () in
          let s = S.v p d in
          let diag f =
            let r = f p d in
            Array.append (Nx.to_array r.mu) (Nx.to_array r.theta)
          in
          equal
            (list (pair string (array nan_or_close)))
            [
              ("mcse", diag Norn.Diag.mcse_mean);
              ("ess_bulk", diag Norn.Diag.ess_bulk);
              ("ess_tail", diag Norn.Diag.ess_tail);
              ("rhat", diag Norn.Diag.rhat);
            ]
            (List.map
               (fun name -> (name, column s name))
               [ "mcse"; "ess_bulk"; "ess_tail"; "rhat" ]));
      test "fills rhat with the nested R-hat of its superchains" (fun () ->
          let d = mu () in
          let s = S.v t ~superchains:2 d in
          equal (array float_exact)
            (Nx.to_array (Norn.Diag.nested_rhat t ~superchains:2 d))
            (column s "rhat"));
      test "computes float32 draws at float64" (fun () ->
          let x = Nx.cast Nx.float32 (mu () :> Nx.host Nx.float64_t) in
          let columns s =
            List.map (fun (name, c) -> (name, Nx.to_array c)) (S.columns s)
          in
          equal
            (list (pair string (array float_exact)))
            (columns (S.v t (Draws.v t (Nx.cast Nx.float64 x))))
            (columns (S.v t (Draws.v t x))));
      prop "keeps the statistics of draws scaled toward the float range's ends"
        scaled_gen (fun (xs, j) ->
          cover_scaled j;
          let unit = S.v t (chains 2 4 xs) in
          let s =
            S.v t (chains 2 4 (Array.map (fun x -> Float.ldexp x j) xs))
          in
          (* The sd of the draws as written, one degree of freedom less. *)
          let sd =
            let n = float_of_int (Array.length xs) in
            let m = Array.fold_left ( +. ) 0. xs /. n in
            let ss =
              Array.fold_left (fun s x -> s +. ((x -. m) *. (x -. m))) 0. xs
            in
            Float.sqrt (ss /. (n -. 1.))
          in
          equal ~msg:"sd" scaled (Float.ldexp sd j) (column s "sd").(0);
          List.iter
            (fun name ->
              equal ~msg:name scaled
                (Float.ldexp (column unit name).(0) j)
                (column s name).(0))
            statistics);
      test "keeps the statistics of draws a float's range apart finite"
        (fun () ->
          let s =
            S.v t
              (chains 2 4
                 [|
                   -1e308; 1e308; -1e308; 1e308; 1e308; -1e308; 1e308; -1e308;
                 |])
          in
          let one name = (column s name).(0) in
          let close = float_rel ~rel:1e-15 ~abs:0. in
          (* Eight draws of ±d have the mean 0, the variance 8d² / 7, and the
             median halfway between the fourth and fifth, -d and d. *)
          equal ~msg:"sd" close (1e308 *. Float.sqrt (8. /. 7.)) (one "sd");
          equal ~msg:"mean" float_exact 0. (one "mean");
          equal ~msg:"median" float_exact 0. (one "median");
          equal ~msg:"q5" float_exact (-1e308) (one "q5");
          let s =
            S.v t
              (chains 2 4
                 [|
                   1e308; 15e307; 1e308; 15e307; 1e308; 15e307; 1e308; 15e307;
                 |])
          in
          equal ~msg:"mean of 1e308 and 1.5e308" close 125e306
            (column s "mean").(0));
      cases
        "raises for fewer than 4 draws, or superchains below 2 or not dividing \
         the chains"
        ~id:fst
        [
          ("3 draws", (3, None));
          ("1 superchain", (4, Some 1));
          ("3 superchains", (4, Some 3));
        ]
        (fun (_, (n, superchains)) ->
          let d = Draws.v t (Nx.zeros Nx.float64 [| 4; n |]) in
          raises_match (Exn.invalid_arg ~sub:"Norn.Summary.v") (fun () ->
              S.v t ?superchains d));
    ]

let findings =
  group "findings"
    [
      test "are each element's, in row order" (fun () ->
          equal (list string)
            [ "Rhat_high theta[0]"; "Ess_low theta[0]"; "Constant theta[1]" ]
            (named (S.v p (draws ()))));
      test "count an element's draws that are not finite" (fun () ->
          let d = chains 1 4 [| 1.; Float.nan; 3.; Float.infinity |] in
          equal (list string) [ "Not_finite value 2" ] (named (S.v t d)));
      test "find effective sizes below 100 per chain, or 400 with superchains"
        (fun () ->
          (* 8 AR(0.6) chains of 200 at key 3: effective sizes between 400 and
             800, and no high R-hat. *)
          let z =
            Nx.to_array
              (Nx.Rng.normal ~key:(Nx.Rng.key 3) Nx.float64 [| 8; 200 |])
          in
          for k = 1 to 1599 do
            if k mod 200 <> 0 then z.(k) <- (0.6 *. z.(k - 1)) +. (0.8 *. z.(k))
          done;
          let d = chains 8 200 z in
          let between x = x > 400. && x < 800. in
          satisfies ~claim:"between 400 and 800" float_exact between
            (Nx.item [] (Norn.Diag.ess_bulk t d));
          satisfies ~claim:"between 400 and 800" float_exact between
            (Nx.item [] (Norn.Diag.ess_tail t d));
          equal (list string) [ "Ess_low value" ] (named (S.v t d));
          equal (list string) [] (named (S.v t ~superchains:2 d)));
      test "are none for mixed draws" (fun () ->
          equal (list string) [] (named (S.v t (mu ()))));
      test "find where divergences gather, after the elements'" (fun () ->
          let d = draws () in
          (* The transitions whose draw of mu is above 1.5 diverged. *)
          let diverging = Nx.greater_s (d :> Nx.host Nx.float64_t p).mu 1.5 in
          let count =
            Array.fold_left
              (fun n b -> if b then n + 1 else n)
              0 (Nx.to_array diverging)
          in
          equal (list string)
            [
              "Rhat_high theta[0]";
              "Ess_low theta[0]";
              "Constant theta[1]";
              Printf.sprintf "Divergent %d/800 at [mu]" count;
            ]
            (named (S.v p ~stats:(transitions ~diverging ()) d)));
      test "count saturated transitions" (fun () ->
          let saturated =
            Nx.set
              [ Nx.I 0; Nx.R (0, 7) ]
              (Nx.scalar Nx.bool true)
              (Nx.zeros Nx.bool [| 4; 200 |])
          in
          equal (list string) [ "Saturated 7/800" ]
            (named (S.v t ~stats:(transitions ~saturated ()) (mu ()))));
      test "find a chain whose energy barely moves" (fun () ->
          (* Energy that follows a slow random walk on chain 2. *)
          let walk =
            Nx.cumsum ~axis:1
              (Nx.Rng.normal ~key:(Nx.Rng.key 5) Nx.float64 [| 4; 200 |])
          in
          let white =
            Nx.Rng.normal ~key:(Nx.Rng.key 6) Nx.float64 [| 4; 200 |]
          in
          let chain2 =
            Nx.reshape [| 4; 1 |]
              (Nx.create Nx.bool [| 4 |] [| false; false; true; false |])
          in
          let energy = Nx.where chain2 walk white in
          equal (list string) [ "Ebfmi_low 2" ]
            (named (S.v t ~stats:(transitions ~energy ()) (mu ()))));
    ]

(* Weighted draws *)

let weighted values log_weights = Norn.Weighted.{ values; log_weights }
let vec xs = Nx.create Nx.float64 [| Array.length xs |] xs

let pareto_k s =
  List.filter_map
    (function
      | S.Pareto_k { khat; threshold } -> Some (khat, threshold) | _ -> None)
    (S.findings s)

let weighted_gen =
  Gen.(
    let* n = int_range 1 60 in
    let* xs = array ~size:(constant n) (float_range (-5.) 5.) in
    let+ lw = array ~size:(constant n) (float_range (-4.) 4.) in
    (xs, lw))

let of_weighted =
  group "of_weighted"
    [
      test "has the moments, quantiles, error and effective size as columns"
        (fun () ->
          let s = S.of_weighted t (weighted (vec [| 1. |]) (vec [| 0. |])) in
          equal (list string) (statistics @ [ "ess" ])
            (List.map fst (S.columns s)));
      test "weighs each draw by its normalised weight" (fun () ->
          (* Weights 0, 1/4, 1/4 and 1/2: draws 1, 2 and 3 stand at 1/8, 3/8 and
             3/4, q5 below the first is the first draw, and the draw of no
             weight, first and far, is read by no statistic. *)
          let w =
            weighted
              (vec [| 1e300; 1.; 2.; 3. |])
              (vec [| Float.neg_infinity; 0.; 0.; Float.log 2. |])
          in
          let s = S.of_weighted t w in
          equal
            (list (pair string (array (float 1e-12))))
            [
              ("mean", [| 2.25 |]);
              ("sd", [| Float.sqrt 0.6875 |]);
              ("q5", [| 1. |]);
              ("median", [| 7. /. 3. |]);
              ("q95", [| 3. |]);
              ("mcse", [| Float.sqrt 0.2421875 |]);
              ("ess", [| 0.6875 /. 0.2421875 |]);
            ]
            (List.map (fun (name, c) -> (name, Nx.to_array c)) (S.columns s)));
      test "keeps the statistics of draws far from 1 at their scale" (fun () ->
          (* Two draws [2d] apart, of weight 1/2 each, have the standard
             deviation [d], the error [d / sqrt 2] and the effective size 2.
             [d²] is below the least float for the first and above the largest
             for the second. *)
          let spread d xs =
            let s =
              S.of_weighted t (weighted (vec xs) (Nx.zeros Nx.float64 [| 2 |]))
            in
            let one name = (column s name).(0) in
            let close = float_rel ~rel:1e-12 ~abs:0. in
            let middle = (xs.(0) /. 2.) +. (xs.(1) /. 2.) in
            equal ~msg:"sd" close d (one "sd");
            equal ~msg:"mcse" close (d /. Float.sqrt 2.) (one "mcse");
            equal ~msg:"ess" close 2. (one "ess");
            equal ~msg:"mean" close middle (one "mean");
            equal ~msg:"median" close middle (one "median")
          in
          spread 1e-200 [| 1e-200; 3e-200 |];
          spread 1e200 [| -1e200; 1e200 |];
          spread 1e308 [| -1e308; 1e308 |]);
      prop "keeps the statistics of draws scaled toward the float range's ends"
        Gen.(pair scaled_gen (array ~size:(constant 8) (float_range (-4.) 4.)))
        (fun ((xs, j), lw) ->
          cover_scaled j;
          let summary xs = S.of_weighted t (weighted (vec xs) (vec lw)) in
          let unit = summary xs in
          let s = summary (Array.map (fun x -> Float.ldexp x j) xs) in
          List.iter
            (fun name ->
              equal ~msg:name scaled
                (Float.ldexp (column unit name).(0) j)
                (column s name).(0))
            statistics;
          equal ~msg:"ess" scaled (column unit "ess").(0) (column s "ess").(0));
      prop "gives equal weights the draws' mean and an effective size of n"
        ~examples:[ ([| 1.5; 1.5; 1.5 |], [| 0.; 0.; 0. |]) ]
        weighted_gen
        (fun (xs, _) ->
          let n = Array.length xs in
          let constant = Array.for_all (fun x -> x = xs.(0)) xs in
          cover "constant draws" constant;
          let s =
            S.of_weighted t (weighted (vec xs) (Nx.zeros Nx.float64 [| n |]))
          in
          let mean = Array.fold_left ( +. ) 0. xs /. float_of_int n in
          equal (float 1e-9) mean (column s "mean").(0);
          (* Constant draws have no spread to measure an error by. *)
          let ess = if constant then Float.nan else float_of_int n in
          equal nan_or_close ess (column s "ess").(0));
      prop "ignores a constant added to the log weights" weighted_gen
        (fun (xs, lw) ->
          (* On a grid of [2^-20] the sums with 300 are exact: a sum that rounds
             moves a weight just above the tail's threshold by much more than
             its rounding. *)
          let lw =
            Array.map (fun l -> Float.round (l *. 0x1p20) /. 0x1p20) lw
          in
          let s = S.of_weighted t (weighted (vec xs) (vec lw)) in
          let s' =
            S.of_weighted t
              (weighted (vec xs) (vec (Array.map (fun l -> l +. 300.) lw)))
          in
          List.iter2
            (fun (name, a) (_, b) ->
              equal ~msg:name (array nan_or_close) (Nx.to_array a)
                (Nx.to_array b))
            (S.columns s) (S.columns s'));
      test "judges heavy-tailed weights by their raw k-hat" (fun () ->
          (* Weights U^(-1.3): no mean, so k-hat is far above 0.7. 2000 draws
             have the threshold min (1 - 1 / log10 2000, 0.7). *)
          let u = Nx.Rng.uniform ~key:(Nx.Rng.key 8) Nx.float64 [| 2000 |] in
          let lw = Nx.mul_s (Nx.log u) (-1.3) in
          let s = S.of_weighted t (weighted u lw) in
          equal
            (list (pair float_exact (float 1e-15)))
            [
              ( Nx.item [] (Norn.Diag.psis lw).khat,
                1. -. (1. /. Float.log10 2000.) );
            ]
            (pareto_k s));
      test "judges 100 draws by a threshold of 0.5" (fun () ->
          (* Weights U^(-0.7) at key 1 have a k-hat between 0.5 and 0.7. *)
          let u = Nx.Rng.uniform ~key:(Nx.Rng.key 1) Nx.float64 [| 100 |] in
          let lw = Nx.mul_s (Nx.log u) (-0.7) in
          let khat = Nx.item [] (Norn.Diag.psis lw).khat in
          satisfies ~claim:"between 0.5 and 0.7" float_exact
            (fun k -> k > 0.5 && k < 0.7)
            khat;
          equal
            (list (pair float_exact (float 1e-15)))
            [ (khat, 0.5) ]
            (pareto_k (S.of_weighted t (weighted u lw))));
      test "summarises raw weights as smoothed" (fun () ->
          let k = Nx.Rng.key 10 in
          let x =
            Nx.Rng.normal ~key:(Nx.Rng.fold_in k 0) Nx.float64 [| 1000 |]
          in
          let u =
            Nx.Rng.uniform ~key:(Nx.Rng.fold_in k 1) Nx.float64 [| 1000 |]
          in
          let lw = Nx.mul_s (Nx.log u) (-0.8) in
          let s = S.of_weighted t (weighted x lw) in
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
      test "finds nothing in equal weights of 500 draws" (fun () ->
          let x = Nx.Rng.normal ~key:(Nx.Rng.key 9) Nx.float64 [| 500 |] in
          let s =
            S.of_weighted t (weighted x (Nx.zeros Nx.float64 [| 500 |]))
          in
          equal (list string) [] (named s));
      test "finds the elements' problems in row order, then the weights'"
        (fun () ->
          (* 10 draws, too few to fit a tail: k-hat is NaN. *)
          let theta0 =
            Array.init 10 (fun i -> if i = 3 then Float.nan else float_of_int i)
          in
          let values =
            {
              mu = Nx.arange_f Nx.float64 0. 10. 1.;
              theta =
                Nx.stack ~axis:1 [ vec theta0; Nx.full Nx.float64 [| 10 |] 2. ];
            }
          in
          let s =
            S.of_weighted p (weighted values (Nx.zeros Nx.float64 [| 10 |]))
          in
          equal (list string)
            [ "Not_finite theta[0] 1"; "Constant theta[1]"; "Pareto_k" ]
            (named s);
          satisfies ~claim:"nan" float_exact Float.is_nan
            (fst (List.hd (pareto_k s))));
      test "raises naming a tensor of another number of draws" (fun () ->
          let d =
            {
              mu = Nx.zeros Nx.float64 [| 4 |];
              theta = Nx.zeros Nx.float64 [| 3; 2 |];
            }
          in
          raises_match (Exn.invalid_arg ~sub:"theta") (fun () ->
              S.of_weighted p (weighted d (Nx.zeros Nx.float64 [| 4 |]))));
      test "raises for log weights of more than one axis" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Summary.of_weighted")
            (fun () ->
              S.of_weighted t
                (weighted
                   (Nx.zeros Nx.float64 [| 4 |])
                   (Nx.zeros Nx.float64 [| 4; 1 |]))));
    ]

let concat =
  group "concat"
    [
      test "puts rows, columns and findings in order" (fun () ->
          let s = S.v p (draws ()) in
          let s' = S.v t (chains 1 4 [| 1.; Float.nan; 3.; 4. |]) in
          let both = S.concat [ s; s' ] in
          equal (array string)
            (Array.append (S.labels s) (S.labels s'))
            (S.labels both);
          equal (array nan_or_close)
            (Array.append (column s "mean") (column s' "mean"))
            (column both "mean");
          equal (list string) (named s @ named s') (named both));
      test "raises for summaries of chains and of weighted draws" (fun () ->
          let w = S.of_weighted t (weighted (vec [| 1. |]) (vec [| 0. |])) in
          raises_match (Exn.invalid_arg ~sub:"Norn.Summary.concat") (fun () ->
              S.concat [ S.v p (draws ()); w ]));
    ]

(* The texts below are baselines of the formatting, written once from the
   code. *)

let pp =
  group "pp"
    [
      test "prints a table, then the findings" (fun () ->
          let x =
            Nx.create Nx.float64 [| 2; 4; 2 |]
              [|
                1.; 5.; 2.; 5.; 3.; 5.; 4.; 5.; 2.; 5.; 3.; 5.; 4.; 5.; 5.; 5.;
              |]
          in
          expect (Format.asprintf "%a" S.pp (S.v t (Draws.v t x)))
          @@ __POS_OF__
               {|
                 mean    sd    q5  median   q95   mcse  ess_bulk  ess_tail  rhat
            [0]     3  1.31  1.35       3  4.65  0.487         7         7  1.89
            [1]     5     0     5       5     5    nan       nan       nan   nan
            R-hat of [0] is 1.889, above 1.01
            effective sample size of [0] is 7 in the bulk and 7 in the tails, too few
            [1] is constant: it has no R-hat nor effective sample size
            |});
      test "prints a weighted summary's own columns" (fun () ->
          let w =
            weighted
              (vec [| 1.; 2.; 3.; 6. |])
              (vec [| 0.; 0.; Float.log 2.; Float.neg_infinity |])
          in
          expect (Format.asprintf "%a" S.pp (S.of_weighted t w))
          @@ __POS_OF__
               {|
                   mean     sd  q5  median  q95   mcse  ess
            value  2.25  0.829   1    2.33    3  0.492    3
            the weights have no Pareto k-hat: estimates from them are not judged
            |});
      (* Cells print three significant digits, with an exponent of two digits
         past five, the same on every system: the C library's [%.3g] writes
         2e+006 under mingw-w64. *)
      test "prints exponents with two digits" (fun () ->
          let w = weighted (vec [| 1e6; 3e6 |]) (vec [| 0.; 0. |]) in
          expect (Format.asprintf "%a" S.pp (S.of_weighted t w))
          @@ __POS_OF__
               {|
                    mean     sd     q5  median    q95      mcse  ess
            value  2e+06  1e+06  1e+06   2e+06  3e+06  7.07e+05    2
            the weights have no Pareto k-hat: estimates from them are not judged
            |});
      (* A finding's numbers round once, ties to even, the same on every system:
         msvcrt's [%.2f] writes 1.125 as 1.13. *)
      test "rounds a finding's numbers ties to even" (fun () ->
          let root = Nx.Ptree.Path.root in
          let region shift = { S.path = root; index = [||]; shift } in
          let findings =
            [
              S.Rhat_high { path = root; index = [||]; rhat = 1.0625 };
              S.Pareto_k { khat = 1.125; threshold = 0.7 };
              S.Ebfmi_low { chain = 0; ebfmi = 0.125 };
              S.Divergent
                {
                  count = 1;
                  total = 2;
                  regions = [ region 0.25; region (-0.25) ];
                };
            ]
          in
          expect
            (String.concat "\n"
               (List.map (Format.asprintf "%a" S.pp_finding) findings))
          @@ __POS_OF__
               {|
            R-hat of value is 1.062, above 1.01
            Pareto k-hat of the weights is 1.12, above 0.70: estimates from them are unreliable
            E-BFMI of chain 0 is 0.12, below 0.3
            1 of 2 transitions diverged; they gather at value (+0.2 sd), value (-0.2 sd)
            |});
    ]

let () =
  exit (run "Norn.Summary" [ labels; v; findings; of_weighted; concat; pp ])
