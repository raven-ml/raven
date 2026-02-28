(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module Data = Kaun.Data
module Layer = Kaun.Layer
module Train = Kaun.Train
module Loss = Kaun.Loss

let test_make_init () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let model =
    Layer.sequential
      [
        Layer.linear ~in_features:2 ~out_features:4 ();
        Layer.tanh ();
        Layer.linear ~in_features:4 ~out_features:1 ();
      ]
  in
  let optimizer = Kaun.Optim.adam ~lr:(Kaun.Optim.Schedule.constant 0.01) () in
  let trainer = Train.make ~model ~optimizer in
  let st = Train.init trainer ~dtype:Nx.float32 in
  let vars = Train.vars st in
  let param_count = Kaun.Ptree.count_parameters (Layer.params vars) in
  equal ~msg:"has parameters" bool true (param_count > 0)

let test_step () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let model =
    Layer.sequential
      [
        Layer.linear ~in_features:2 ~out_features:4 ();
        Layer.tanh ();
        Layer.linear ~in_features:4 ~out_features:1 ();
      ]
  in
  let optimizer = Kaun.Optim.adam ~lr:(Kaun.Optim.Schedule.constant 0.01) () in
  let trainer = Train.make ~model ~optimizer in
  let st = Train.init trainer ~dtype:Nx.float32 in
  let x =
    Nx.create Nx.float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]
  in
  let y = Nx.create Nx.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in
  let loss_val, st' =
    Train.step trainer st ~training:true
      ~loss:(fun pred -> Loss.binary_cross_entropy pred y)
      x
  in
  let loss_f = Nx.item [] loss_val in
  equal ~msg:"loss is finite" bool true (Float.is_finite loss_f);
  let vars0 = Train.vars st in
  let vars1 = Train.vars st' in
  equal ~msg:"params changed" bool false
    (Layer.params vars0 == Layer.params vars1)

let test_fit () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let model =
    Layer.sequential
      [
        Layer.linear ~in_features:2 ~out_features:4 ();
        Layer.tanh ();
        Layer.linear ~in_features:4 ~out_features:1 ();
      ]
  in
  let optimizer = Kaun.Optim.adam ~lr:(Kaun.Optim.Schedule.constant 0.01) () in
  let trainer = Train.make ~model ~optimizer in
  let st = Train.init trainer ~dtype:Nx.float32 in
  let x =
    Nx.create Nx.float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]
  in
  let y = Nx.create Nx.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in
  let st' =
    Train.fit trainer st
      (Data.repeat 100 (x, fun pred -> Loss.binary_cross_entropy pred y))
  in
  let pred = Train.predict trainer st' x |> Nx.sigmoid in
  let p0 = Nx.item [ 0; 0 ] pred in
  let p1 = Nx.item [ 1; 0 ] pred in
  let p2 = Nx.item [ 2; 0 ] pred in
  let p3 = Nx.item [ 3; 0 ] pred in
  equal ~msg:"[0,0] -> ~0" bool true (p0 < 0.4);
  equal ~msg:"[0,1] -> ~1" bool true (p1 > 0.6);
  equal ~msg:"[1,0] -> ~1" bool true (p2 > 0.6);
  equal ~msg:"[1,1] -> ~0" bool true (p3 < 0.4)

let test_predict () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let model =
    Layer.sequential
      [ Layer.linear ~in_features:3 ~out_features:2 (); Layer.relu () ]
  in
  let optimizer = Kaun.Optim.sgd ~lr:(Kaun.Optim.Schedule.constant 0.01) () in
  let trainer = Train.make ~model ~optimizer in
  let st = Train.init trainer ~dtype:Nx.float32 in
  let x = Nx.ones Nx.float32 [| 2; 3 |] in
  let y = Train.predict trainer st x in
  equal ~msg:"predict shape" (list int) [ 2; 2 ] (Array.to_list (Nx.shape y))

let test_fit_with_reporting () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let model =
    Layer.sequential [ Layer.linear ~in_features:2 ~out_features:1 () ]
  in
  let optimizer = Kaun.Optim.sgd ~lr:(Kaun.Optim.Schedule.constant 0.01) () in
  let trainer = Train.make ~model ~optimizer in
  let st = Train.init trainer ~dtype:Nx.float32 in
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let y = Nx.create Nx.float32 [| 2; 1 |] [| 1.; 0. |] in
  let report_count = ref 0 in
  let _st' =
    Train.fit trainer st
      ~report:(fun ~step ~loss:_ _st ->
        if step mod 3 = 0 then incr report_count)
      (Data.repeat 10 (x, fun pred -> Loss.binary_cross_entropy pred y))
  in
  equal ~msg:"report called 3 times (steps 3,6,9)" int 3 !report_count

let test_fit_early_stop () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let model =
    Layer.sequential [ Layer.linear ~in_features:2 ~out_features:1 () ]
  in
  let optimizer = Kaun.Optim.sgd ~lr:(Kaun.Optim.Schedule.constant 0.01) () in
  let trainer = Train.make ~model ~optimizer in
  let st = Train.init trainer ~dtype:Nx.float32 in
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let y = Nx.create Nx.float32 [| 2; 1 |] [| 1.; 0. |] in
  let last_step = ref 0 in
  let _st' =
    Train.fit trainer st
      ~report:(fun ~step ~loss:_ _st ->
        last_step := step;
        if step >= 15 then raise_notrace Train.Early_stop)
      (Data.repeat 100 (x, fun pred -> Loss.binary_cross_entropy pred y))
  in
  equal ~msg:"stopped at step 15" int 15 !last_step

let test_batch_norm_state_threading () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let model =
    Layer.sequential
      [
        Layer.linear ~in_features:2 ~out_features:4 ();
        Layer.batch_norm ~num_features:4 ();
        Layer.relu ();
        Layer.linear ~in_features:4 ~out_features:1 ();
      ]
  in
  let optimizer = Kaun.Optim.adam ~lr:(Kaun.Optim.Schedule.constant 0.01) () in
  let trainer = Train.make ~model ~optimizer in
  let st0 = Train.init trainer ~dtype:Nx.float32 in
  let x =
    Nx.create Nx.float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]
  in
  let y = Nx.create Nx.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in
  let _, st1 =
    Train.step trainer st0 ~training:true
      ~loss:(fun pred -> Loss.binary_cross_entropy pred y)
      x
  in
  let state0 = Layer.state (Train.vars st0) in
  let state1 = Layer.state (Train.vars st1) in
  equal ~msg:"batch_norm state changed after step" bool false (state0 == state1)

let () =
  run "Kaun.Train"
    [
      group "make/init" [ test "make and init" test_make_init ];
      group "step" [ test "single step" test_step ];
      group "fit"
        [
          test "xor convergence" test_fit;
          test "reporting" test_fit_with_reporting;
          test "early stop" test_fit_early_stop;
        ];
      group "predict" [ test "shapes" test_predict ];
      group "stateful"
        [ test "batch_norm state threading" test_batch_norm_state_threading ];
    ]
