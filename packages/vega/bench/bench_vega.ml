(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let lr = Vega.Schedule.constant 1e-3
let shapes = [ ("256", [| 256; 256 |]); ("1024", [| 1024; 1024 |]) ]

let make_step_bench name tx (label, shape) =
  let param = Nx.rand Nx.Float32 shape in
  let grad = Nx.rand Nx.Float32 shape in
  let state = Vega.init tx param in
  let state = ref state in
  Thumper.bench (Printf.sprintf "%s/%s" name label) (fun () ->
      let new_param, new_state = Vega.step !state ~grad ~param in
      state := new_state;
      new_param)

let optimizer_benches name tx = List.map (make_step_bench name tx) shapes

let build_benchmarks () =
  [
    Thumper.group "SGD" (optimizer_benches "SGD" (Vega.sgd lr));
    Thumper.group "SGD+Momentum"
      (optimizer_benches "SGD+Momentum" (Vega.sgd ~momentum:0.9 lr));
    Thumper.group "Adam" (optimizer_benches "Adam" (Vega.adam lr));
    Thumper.group "AdamW" (optimizer_benches "AdamW" (Vega.adamw lr));
    Thumper.group "RMSprop" (optimizer_benches "RMSprop" (Vega.rmsprop lr));
    Thumper.group "Adagrad" (optimizer_benches "Adagrad" (Vega.adagrad lr));
    Thumper.group "Lion" (optimizer_benches "Lion" (Vega.lion lr));
    Thumper.group "RAdam" (optimizer_benches "RAdam" (Vega.radam lr));
    Thumper.group "LAMB" (optimizer_benches "LAMB" (Vega.lamb lr));
    Thumper.group "LARS" (optimizer_benches "LARS" (Vega.lars lr));
    Thumper.group "Adan" (optimizer_benches "Adan" (Vega.adan lr));
    Thumper.group "Adafactor"
      (optimizer_benches "Adafactor" (Vega.adafactor ()));
  ]

let () =
  let benchmarks = build_benchmarks () in
  Thumper.run "vega" benchmarks
