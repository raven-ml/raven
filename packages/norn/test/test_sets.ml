(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Samplers off the host: positions on a set of two memory devices, which
   compute on the host, with the chains on both devices or split between
   them, and keys that lie on one device alone. A sampler's state and draws
   lie where its start does. *)

open Windtrap

let t = Nx.Ptree.tensor

module Pair =
  (val Nx.devices
         (List.init 2 (fun k ->
              Result.get_ok (Rig.memory_device (Printf.sprintf "norn-s%d" k)))))

let first = Nx.Placement.device Pair.v (List.hd (Nx.rigs Pair.v))
let placements = [ ("whole", Pair.on); ("split", Pair.split ~axis:0) ]
let pp p = Format.asprintf "%a" Nx.Placement.pp p
let at x = pp (Nx.placement x)

(* A standard normal in 2 dimensions, chain by chain. *)
let normal x = Nx.mul_s (Nx.sum ~axes:[ 1 ] (Nx.square x)) (-0.5)

let start p chains =
  Nx.place p (Nx.Rng.normal ~key:(Nx.Rng.key 1) Nx.float64 [| chains; 2 |])

let key = Nx.Rng.place first (Nx.Rng.key 2)

(* [lies p name x] checks that [x] lies at [p]. *)
let lies p name x = equal ~msg:name string (pp p) (at x)

let nuts =
  List.map
    (fun (where, p) ->
      test ("NUTS keeps the chains " ^ where) (fun () ->
          let s = Norn.Nuts.init t normal (start p 4) in
          let s = Norn.Nuts.warmup t normal key ~steps:10 s in
          let s, d, stats = Norn.Nuts.sample t normal key ~draws:5 s in
          lies p "position" s.position;
          lies p "draws" (d :> (float, Nx.float64_elt, Pair.d) Nx.t);
          lies p "acceptance" (stats :> (Nx.float64_elt, Pair.d) Norn.Stats.t).acceptance;
          (* A diagnostic reduces the chains: it lies whole. *)
          lies (Nx.Placement.uncut p) "rhat" (Norn.Diag.rhat t d)))
    placements

let whole =
  [
    test "HMC keeps whole chains whole" (fun () ->
        let s = Norn.Hmc.init t normal (start Pair.on 4) in
        let s = Norn.Hmc.warmup t normal key ~steps:10 s in
        let s, d, _ = Norn.Hmc.sample t normal key ~draws:5 s in
        lies Pair.on "position" s.position;
        lies Pair.on "draws" (d :> (float, Nx.float64_elt, Pair.d) Nx.t));
    test "ensemble sampling keeps whole walkers whole" (fun () ->
        let s = Norn.Ensemble.init t normal (start Pair.on 8) in
        let s = Norn.Ensemble.warmup t normal key ~steps:5 s in
        let s, d, _ = Norn.Ensemble.sample t normal key ~draws:5 s in
        lies Pair.on "position" s.position;
        lies Pair.on "draws" (d :> (float, Nx.float64_elt, Pair.d) Nx.t));
  ]

let cut by why =
  Exn.invalid_arg
    ~substring:(by ^ ": the position is cut across devices; " ^ why)

let refusals =
  let split = start (Pair.split ~axis:0) 8 in
  [
    test "HMC refuses chains cut across devices" (fun () ->
        raises_match
          (cut "Norn.Hmc.init" "its chains share one step size and geometry")
          (fun () -> Norn.Hmc.init t normal split));
    test "ensemble sampling refuses walkers cut across devices" (fun () ->
        raises_match
          (cut "Norn.Ensemble.init" "a walker moves along others")
          (fun () -> Norn.Ensemble.init t normal split));
    test "nested sampling refuses live points cut across devices" (fun () ->
        raises_match
          (cut "Norn.Nested.init"
             "a step replaces the lowest points by moves from others")
          (fun () ->
            Norn.Nested.init t ~budget:4 ~prior:normal ~likelihood:normal split));
    test "SMC refuses particles cut across devices" (fun () ->
        raises_match
          (cut "Norn.Smc.init" "a step resamples the particles")
          (fun () -> Norn.Smc.init t ~prior:normal ~likelihood:normal split));
  ]

(* Draws made with a key on one device lie where their operands do. *)
let draws =
  [
    test "nested sampling's evidence lies with its live points" (fun () ->
        let z =
          Norn.Nested.run t ~budget:4 ~prior:normal ~likelihood:normal key
            (start Pair.on 16)
        in
        lies Pair.on "log_evidence" (Norn.Evidence.log_evidence z);
        lies Pair.on "error" (Norn.Evidence.error z));
    test "SMC's evidence lies with its particles" (fun () ->
        let z =
          Norn.Smc.run t ~budget:4 ~prior:normal ~likelihood:normal key
            (start Pair.on 16)
        in
        lies Pair.on "log_evidence" (Norn.Evidence.log_evidence z));
    test "a Gaussian's draws lie with its mean" (fun () ->
        let mean = Nx.place Pair.on (Nx.zeros Nx.float64 [| 2 |]) in
        let g = Norn.Gaussian.diagonal t Nx.float64 ~mean ~scale:(Nx.ones_like mean) in
        lies Pair.on "sample" (Norn.Gaussian.sample t key ~n:3 g));
    test "resampled draws lie with their values" (fun () ->
        let values = start Pair.on 4 in
        let w =
          { Norn.Weighted.values; log_weights = Nx.zeros_like (normal values) }
        in
        lies Pair.on "resample" (Norn.Weighted.resample t key ~n:4 w));
    test "simulated draws lie with the draws" (fun () ->
        let d =
          Norn.Draws.v t
            (Nx.place Pair.on (Nx.zeros Nx.float64 [| 2; 3 |]))
        in
        let s =
          Norn.Draws.simulate t t
            (fun k x -> Nx.add x (Nx.Rng.normal_at (Nx.beside x) ~key:k Nx.float64 [||]))
            key d
        in
        lies Pair.on "simulate" (s :> (float, Nx.float64_elt, Pair.d) Nx.t));
  ]

let () =
  exit
    (run "Norn on a device set"
       [
         group "NUTS" nuts;
         group "whole chains" whole;
         group "refusals" refusals;
         group "draws" draws;
       ])
