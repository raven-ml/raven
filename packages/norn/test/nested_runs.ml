(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Runs of nested sampling on the evidence models, shared by its suites. *)

open Evidence_models

let t = Nx.Ptree.tensor
let ez = Norn.Evidence.ptree t

let stop_name = function
  | Norn.Evidence.Converged -> "converged"
  | Remaining r -> Printf.sprintf "Remaining %g" r
  | Temperature b -> Printf.sprintf "Temperature %g" b

let lnz z = Nx.item [] (Norn.Evidence.log_evidence z)
let err z = Nx.item [] (Norn.Evidence.error z)

(* Every run here converges within 40 steps; the dead buffer's size sets the
   cost of the errors. *)
let budget = 60

(* [nest ?budget m live k] is a run on [m] from [live] prior draws: the run's
   key is [Nx.Rng.fold_in k 0] and the draws' [Nx.Rng.fold_in k 1]. *)
let nest ?(budget = budget) m live k =
  Norn.Nested.run t ~budget ~prior:m.prior ~likelihood:m.likelihood
    (Nx.Rng.fold_in k 0)
    (m.draw (Nx.Rng.fold_in k 1) live)

(* The first and second moments of a point's first element. *)
let first_moments x =
  let x1 = Nx.slice [ Nx.I 0 ] x in
  Nx.stack ~axis:0 [ x1; Nx.square x1 ]

(* [compiled m live] is one compiled run on [m] from [live] prior draws, with
   the expectation of [first_moments] under it. Eager, a run of these models
   takes five to twenty times as long. Every test that runs the conjugate model
   at 50 live points makes this same program, so once one has, the others read
   it from tolk's cache. *)
let compiled m live =
  Rune.jit
    Nx.Ptree.(Nx.Rng.ptree @-> returns (pair ez (pair tensor tensor)))
    (fun k ->
      let z = nest m live k in
      (z, Norn.Evidence.expectation t first_moments z))

(* [runs ?live m n] is [n] runs of [compiled m live], the [i]-th from row [i] of
   [Nx.Rng.split_batch ~n (Nx.Rng.key 70)]. A call per run: mapped over all the
   keys at once, every lane waits for the slowest walker of every slice move,
   and 200 runs of the banana take twice the processor time. *)
let runs ~live m n =
  let one = compiled m live in
  let keys = Nx.Rng.split_batch ~n (Nx.Rng.key 70) in
  let row i = Nx.Ptree.map Nx.Rng.ptree (fun _ k -> Nx.slice [ I i ] k) in
  List.init n (fun i -> one (row i keys))
