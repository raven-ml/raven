(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rune2's costs per operation over one-element float32 values, where the
   overhead of tracing and evaluating dominates kernel time. Each row's time
   divided by its operations is the cost per operation. *)

module Rune = Rune2

let f32 = Nx.float32
let ops = 100

let chain x =
  let t = ref x in
  for _ = 1 to ops do
    t := Nx.sin !t
  done;
  !t

let one () = Nx.copy (Nx.Rng.uniform ~key:(Nx.Rng.key 0) f32 [| 1 |])

(* Trace: a step of [ops] sines traced over no row, which evaluates nothing. *)
let trace_rows () =
  let init = one () and none = Nx.zeros f32 [| 0; 1 |] in
  let step c _ = (chain c, ()) in
  let t = Nx.Ptree.tensor in
  [
    Thumper.bench "chain-100" (fun () ->
        Rune.scan t t Nx.Ptree.unit ~f:step ~init none);
  ]

(* Loops: a step of [ops] sines over [rows] rows, against the OCaml fold of the
   same step. *)
let rows = 1000

let loop_rows () =
  let init = one () and xs = Nx.copy (Nx.zeros f32 [| rows; 1 |]) in
  let step c _ = (chain c, ()) in
  let t = Nx.Ptree.tensor in
  [
    Thumper.bench "scan-eager-100x1000" (fun () ->
        Rune.scan t t Nx.Ptree.unit ~f:step ~init xs);
    Thumper.bench "fold-eager-100x1000" (fun () ->
        let c = ref init in
        for i = 0 to rows - 1 do
          c := fst (step !c (Nx.get [ i ] xs))
        done;
        !c);
  ]

(* Declined: [ops] adds of constants in a traced step, which reach no tracer and
   compute at the call, against the same adds eagerly. *)
let adds a b =
  for _ = 1 to ops do
    ignore (Sys.opaque_identity (Nx.add a b))
  done

let declined_rows () =
  let a = one () and b = one () and xs = Nx.copy (Nx.zeros f32 [| 1; 1 |]) in
  let t = Nx.Ptree.tensor in
  [
    Thumper.bench "add-1" (fun () -> adds a b);
    Thumper.bench "add-1-under-scan" (fun () ->
        Rune.scan t t Nx.Ptree.unit
          ~f:(fun c _ ->
            adds a b;
            (c, ()))
          ~init:a xs);
  ]

let () =
  exit
    (Thumper.run "rune2"
       [
         Thumper.group "Trace" (trace_rows ());
         Thumper.group "Loops" (loop_rows ());
         Thumper.group "Declined" (declined_rows ());
       ])
