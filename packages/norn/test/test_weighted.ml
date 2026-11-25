(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module W = Norn.Weighted

let t = Nx.Ptree.tensor

(* Draw [i] has the value [i]. *)
let weighted lw =
  let n = Array.length lw in
  W.
    {
      values = Nx.arange_f Nx.float64 0. (float_of_int n) 1.;
      log_weights = Nx.create Nx.float64 [| n |] lw;
    }

(* Log weights in [-10, 2], some of them [-inf]: a draw of no weight. *)
let log_weights =
  Gen.array ~size:(Gen.int_range 1 40)
    (Gen.frequency
       [ (4, Gen.float_range (-10.) 2.); (1, Gen.constant Float.neg_infinity) ])
  |> Gen.such_that (Array.exists Float.is_finite)

let normalised lw =
  let w = Array.map Float.exp lw in
  let s = Array.fold_left ( +. ) 0. w in
  Array.map (fun w -> w /. s) w

let ptree =
  test "ptree walks values then log_weights" (fun () ->
      let paths =
        List.filter_map
          (function
            | Nx.Ptree.Leaf p -> Some (Nx.Ptree.Path.to_string p)
            | Report _ -> None)
          (Nx.Ptree.visits (W.ptree t) (weighted [| 0. |]))
      in
      equal (list string) [ "values"; "log_weights" ] paths)

let ess =
  group "ess"
    [
      (* The weights are unnormalised: a constant up to ±1000 added to the log
         weights leaves the size alone, though their exponentials overflow. *)
      prop "is Kish's (Σ w)² / Σ w², from 1 to n"
        ~examples:[ ([| 0. |], 0.); (Array.make 7 3., 1000.) ]
        Gen.(pair log_weights (float_range (-1000.) 1000.))
        (fun (lw, c) ->
          cover "a draw of no weight" (Array.mem Float.neg_infinity lw);
          let w = Array.map Float.exp lw in
          let s = Array.fold_left ( +. ) 0. w in
          let s2 = Array.fold_left (fun a w -> a +. (w *. w)) 0. w in
          let ess =
            Nx.item [] (W.ess (weighted (Array.map (fun l -> l +. c) lw)))
          in
          let n = float_of_int (Array.length lw) in
          equal (float_rel ~rel:1e-9 ~abs:0.) (s *. s /. s2) ess;
          (* The bounds within rounding. *)
          at_least float_exact ~than:(1. -. 1e-9) ess;
          at_most float_exact ~than:(n +. 1e-9) ess);
      cases "is NaN when no weight is positive" ~id:fst
        [ ("every weight zero", [| Float.neg_infinity |]); ("no draw", [||]) ]
        (fun (_, lw) ->
          satisfies ~claim:"nan" float_exact Float.is_nan
            (Nx.item [] (W.ess (weighted lw))));
    ]

(* [copies lw x] is how many times each draw of [weighted lw] is in [x]. *)
let copies lw x =
  let c = Array.make (Array.length lw) 0 in
  Array.iter (fun v -> c.(int_of_float v) <- c.(int_of_float v) + 1) x;
  c

let resample =
  group "resample"
    [
      prop "copies a draw floor (n w) or ceil (n w) times"
        Gen.(triple log_weights (int_range 1 60) int)
        (fun (lw, n, seed) ->
          let x = W.resample t (Nx.Rng.key seed) ~n (weighted lw) in
          let c = copies lw (Nx.to_array x) in
          Array.iteri
            (fun i w ->
              let e = float_of_int n *. w in
              satisfies ~claim:"floor (n w) <= copies <= ceil (n w)" int
                (fun c ->
                  float_of_int c >= Float.floor e -. 1e-9
                  && float_of_int c <= Float.ceil e +. 1e-9)
                c.(i))
            (normalised lw));
      test "copies a draw n w times on average" (fun () ->
          (* One copy of weights 0.3 and 0.7 at each of 1000 keys: the first's
             count is binomial (1000, 0.3), of sd 14.5, and six sds bound a
             false failure at 2e-9. *)
          let lw = [| Float.log 0.3; Float.log 0.7 |] in
          let firsts = ref 0 in
          for k = 0 to 999 do
            let x = W.resample t (Nx.Rng.key k) ~n:1 (weighted lw) in
            if Nx.item [ 0 ] x = 0. then incr firsts
          done;
          less float_exact ~than:87. (Float.abs (float_of_int !firsts -. 300.)));
      prop "never copies a draw of no weight, in float32 too" ~count:5 Gen.int
        (fun seed ->
          (* 3000 equal float32 weights sum, in float32, to about 0.99997, below
             the last points of 100000 at every key: those points fall past the
             cumulative weights, beside a last draw of no weight. *)
          let lw = Array.append (Array.make 3000 0.) [| Float.neg_infinity |] in
          let w =
            W.
              {
                values = Nx.arange_f Nx.float32 0. 3001. 1.;
                log_weights = Nx.create Nx.float32 [| 3001 |] lw;
              }
          in
          let x = W.resample t (Nx.Rng.key seed) ~n:100_000 w in
          equal (float 0.5) 2999. (Nx.item [] (Nx.max x)));
      test "raises when no weight is positive" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Weighted.resample")
            (fun () ->
              W.resample t (Nx.Rng.key 1) ~n:3
                (weighted [| Float.neg_infinity; Float.neg_infinity |])));
      test "raises for n below 1" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Norn.Weighted.resample")
            (fun () -> W.resample t (Nx.Rng.key 1) ~n:0 (weighted [| 0. |])));
    ]

let () = exit (run "Norn.Weighted" [ ptree; ess; resample ])
