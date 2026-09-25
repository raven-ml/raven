(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let p = Nx.Ptree.tensor
let lr = Vega.lr 1e-3
let shapes = [ ("256", [| 256; 256 |]); ("1024", [| 1024; 1024 |]) ]

let make_step_bench name init step (label, shape) =
  let params = Nx.rand Nx.Float32 shape in
  let grads = Nx.rand Nx.Float32 shape in
  let state = ref (init p params) in
  Thumper.bench (Printf.sprintf "%s/%s" name label) (fun () ->
      let params, st = step !state ~params ~grads in
      state := st;
      params)

let optimizer_benches name init step =
  List.map (make_step_bench name init step) shapes

let build_benchmarks () =
  [
    Thumper.group "SGD"
      (optimizer_benches "SGD" Vega.sgd_init (fun st -> Vega.sgd_step p ~lr st));
    Thumper.group "SGD+Momentum"
      (optimizer_benches "SGD+Momentum" Vega.sgd_init (fun st ->
           Vega.sgd_step p ~lr ~momentum:0.9 st));
    Thumper.group "Adam"
      (optimizer_benches "Adam" Vega.adam_init (fun st ->
           Vega.adam_step p ~lr st));
    Thumper.group "AdamW"
      (optimizer_benches "AdamW" Vega.adamw_init (fun st ->
           Vega.adamw_step p ~lr st));
    Thumper.group "RMSprop"
      (optimizer_benches "RMSprop" Vega.rmsprop_init (fun st ->
           Vega.rmsprop_step p ~lr st));
    Thumper.group "Adagrad"
      (optimizer_benches "Adagrad" Vega.adagrad_init (fun st ->
           Vega.adagrad_step p ~lr st));
    Thumper.group "Lion"
      (optimizer_benches "Lion" Vega.lion_init (fun st ->
           Vega.lion_step p ~lr st));
    Thumper.group "RAdam"
      (optimizer_benches "RAdam" Vega.radam_init (fun st ->
           Vega.radam_step p ~lr st));
    Thumper.group "LAMB"
      (optimizer_benches "LAMB" Vega.lamb_init (fun st ->
           Vega.lamb_step p ~lr st));
    Thumper.group "LARS"
      (optimizer_benches "LARS" Vega.lars_init (fun st ->
           Vega.lars_step p ~lr st));
    Thumper.group "Adan"
      (optimizer_benches "Adan" Vega.adan_init (fun st ->
           Vega.adan_step p ~lr st));
    Thumper.group "Adafactor"
      (optimizer_benches "Adafactor"
         (fun p x -> Vega.adafactor_init p x)
         (fun st -> Vega.adafactor_step p ~lr st));
  ]

let () =
  let benchmarks = build_benchmarks () in
  Thumper.run "vega" benchmarks
