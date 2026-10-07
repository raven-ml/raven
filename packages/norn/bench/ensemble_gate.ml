(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The ensemble sampler's chains on the gate's targets, for ensemble_gate.py,
   which compares their evaluations per effective draw with zeus's and emcee's.
   Each target writes every tenth draw, [steps / 10; walkers; d], to
   <dir>/<target>.npy and a line "<target> <rows>" to stdout: every row the
   density was called on, warmup included. *)

let t = Nx.Ptree.tensor

(* Each target runs long enough for every coordinate's autocorrelation time to
   be under a fiftieth of the steps, emcee's condition for its estimate. *)

(* An AR(1) correlation of 0.9 in 10 dimensions. *)
let correlated d =
  let s =
    Nx.init Nx.float64 [| d; d |] (fun i ->
        0.9 ** float_of_int (abs (i.(0) - i.(1))))
  in
  let p = Nx.inv s in
  fun x -> Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.mul (Nx.matmul x p) x)) (-0.5)

(* Goodman and Weare's Rosenbrock density. *)
let rosenbrock x =
  let a = Nx.slice [ Nx.A; Nx.I 0 ] x and b = Nx.slice [ Nx.A; Nx.I 1 ] x in
  let r = Nx.sub b (Nx.square a) and s = Nx.rsub_s 1. a in
  Nx.div_s (Nx.add (Nx.mul_s (Nx.square r) 100.) (Nx.square s)) (-20.)

let isotropic x = Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.square x)) (-0.5)

let targets =
  [
    ("correlated10", 10, 10000, correlated 10);
    ("rosenbrock2", 2, 40000, rosenbrock);
    ("isotropic50", 50, 40000, isotropic);
  ]

let walkers d = max 32 (4 * d)
let burn = 500

(* Every [thin]-th draw is written, sampled in chunks of [chunk]: thinning
   divides the autocorrelation time as it divides the steps. *)
let thin = 10
let chunk = 1000

let () =
  let dir = Sys.argv.(1) in
  List.iter
    (fun (name, d, steps, lp) ->
      let rows = ref 0 in
      let counted x =
        rows := !rows + (Nx.shape x).(0);
        lp x
      in
      let w = walkers d in
      let start = Nx.Rng.normal (Nx.Rng.key 1) Nx.float64 [| w; d |] in
      let s = Norn.Ensemble.init t counted start in
      let s = Norn.Ensemble.warmup t counted (Nx.Rng.key 2) ~steps:burn s in
      let rec chunks s n draws =
        if n = 0 then List.rev draws
        else
          let s, d, _ =
            Norn.Ensemble.sample t counted (Nx.Rng.key 3) ~draws:chunk s
          in
          chunks s (n - chunk)
            ((Norn.Draws.thin t ~every:thin d :> Nx.float64_t) :: draws)
      in
      let draws = chunks s steps [] in
      let x = Nx.moveaxis 0 1 (Nx.concatenate ~axis:1 draws) in
      Nx_io.save_npy ~overwrite:true (Filename.concat dir (name ^ ".npy")) x;
      Printf.printf "%s %d\n%!" name !rows)
    targets
