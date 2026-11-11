(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tessera

let f64 = Nx.float64

(* Rows of [d] values around a centre per class, [n] of them, labels cycling
   through [classes]. *)
let labelled ~seed ~n ~d ~classes ~spread =
  let key = Nx.Rng.key seed in
  let y = Nx.mod_s (Nx.arange Nx.int64 0 n 1) (Int64.of_int classes) in
  let centres = Nx.Rng.normal ~key f64 [| classes; d |] in
  let noise = Nx.Rng.normal ~key:(Nx.Rng.fold_in key 1) f64 [| n; d |] in
  (Nx.add (Nx.take ~axis:0 y centres) (Nx.mul_s noise spread), y)

let problems =
  Gen.(
    map
      (fun ((n, d), (classes, seed)) -> (max n classes, d, classes, seed))
      (pair (pair (int_range 2 40) (int_range 1 6)) (pair (int_range 2 5) int)))

(* The probe's objective, whose gradient its minimum zeroes. *)
let objective ~c x y (w, b) =
  let logits = Nx.add (Nx.matmul x w) b in
  let k = Nx.dim 1 logits in
  let onehot = Nx.cast f64 (Nx.one_hot ~num_classes:k y) in
  Nx.add
    (Nx.sub
       (Nx.sum (Nx.logsumexp ~axes:[ 1 ] logits))
       (Nx.sum (Nx.mul onehot logits)))
    (Nx.mul_s (Nx.sum (Nx.square w)) (1. /. (2. *. c)))

let minimum =
  prop "a probe is its objective's minimum" ~count:30 problems
    (fun (n, d, classes, seed) ->
      let x, y = labelled ~seed ~n ~d ~classes ~spread:1. in
      let c = 0.5 in
      let p = Eval.fit ~c x y in
      let gw, gb =
        Rune.grad Nx.Ptree.(pair tensor tensor) (objective ~c x y) (p.w, p.b)
      in
      let norm g = Float.sqrt (Nx.item [] (Nx.sum (Nx.square g))) in
      (* At the minimum the gradient is rounding beside its terms' scale. *)
      let scale = Float.of_int n in
      at_most float_exact ~than:(1e-6 *. scale) (norm gw);
      at_most float_exact ~than:(1e-6 *. scale) (norm gb);
      at_most float_exact ~than:1e-9 (Float.abs (Nx.item [] (Nx.sum p.b))))

let separable =
  test "a probe reads separated classes" (fun () ->
      let x, y = labelled ~seed:3 ~n:60 ~d:4 ~classes:3 ~spread:0.01 in
      let p = Eval.fit ~c:10. x y in
      let s = Eval.score ~predicted:(Eval.predict p x) y in
      equal float_exact 1. s.accuracy;
      equal float_exact 1. s.f1)

let fit_refusals =
  test "a probe refuses classes it has no row of" (fun () ->
      let x = Nx.zeros f64 [| 3; 2 |] in
      let y = Nx.create Nx.int64 [| 3 |] [| 0L; 2L; 0L |] in
      raises (Invalid_argument "Tessera.Eval.fit: class 1 has no row")
        (fun () -> Eval.fit ~c:1. x y);
      let y = Nx.create Nx.int64 [| 3 |] [| 0L; -1L; 0L |] in
      raises (Invalid_argument "Tessera.Eval.fit: label -1 is negative")
        (fun () -> Eval.fit ~c:1. x y))

let scores =
  test "scores of a confusion" (fun () ->
      (* Labels 0 0 0 1 1 2, predicted 0 0 1 1 2 2: class 0 has t 2, m 1; class
         1 t 1, f 1, m 1; class 2 t 1, f 1. *)
      let y = Nx.create Nx.int64 [| 6 |] [| 0L; 0L; 0L; 1L; 1L; 2L |] in
      let predicted = Nx.create Nx.int64 [| 6 |] [| 0L; 0L; 1L; 1L; 2L; 2L |] in
      let s = Eval.score ~predicted y in
      equal (float 1e-15) (4. /. 6.) s.accuracy;
      let f1 = ((3. *. 0.8) +. (2. *. 0.5) +. (1. *. (2. /. 3.))) /. 6. in
      equal (float 1e-15) f1 s.f1)

let rankme =
  test "RankMe of equal and of single directions" (fun () ->
      (* Singular values all 2: shares 1/4, entropy log 4 up to the guards. *)
      let eye = Nx.mul_s (Nx.eye f64 4) 2. in
      let z = Nx.concatenate ~axis:0 [ eye; Nx.zeros f64 [| 3; 4 |] ] in
      equal (float 1e-6) 1. (Eval.rankme z);
      equal (float 1e-6) 1. (Eval.rankme (Nx.matrix_transpose z));
      (* One direction: one share near 1, the others 0. *)
      let col = Nx.reshape [| 6; 1 |] (Nx.arange f64 1 7 1) in
      let rank1 = Nx.matmul col (Nx.ones f64 [| 1; 5 |]) in
      less float_exact ~than:1e-5 (Eval.rankme rank1);
      (* Scale does not change the shares. *)
      let z = Nx.Rng.normal ~key:(Nx.Rng.key 2) f64 [| 9; 5 |] in
      equal (float 1e-6) (Eval.rankme z) (Eval.rankme (Nx.mul_s z 1e3)))

let () =
  exit
    (run "tessera.eval"
       [
         group "probe" [ minimum; separable; fit_refusals; scores ];
         group "rankme" [ rankme ];
       ])
